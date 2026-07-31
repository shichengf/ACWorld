"""Sealed, privacy-safe World projection for one after-sales Agent turn.

The projection is generated from the same ``TrustedAfterSalesContext`` and
``AfterSalesPlanner`` used for real commits.  It is therefore a view of the
authoritative CommerceWorld state, not a second lifecycle simulator.  The
projection contains only operation availability, actor-visible stable
references, and bounded business choices.  It deliberately omits money,
payment and packing rows, policy service allowlists, evidence facts, causal
record digests, and every benchmark or scorer field.

The canonical digest is an integrity seal, not an authorization substitute.
Platform and World must still validate a compiled action against current state
before committing it.  A consumer should also reject the projection whenever
its ``source_context_digest`` or ``issued_at_tick`` no longer matches World.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, cast

from protocol.after_sales import (
    AfterSalesAuthorityError,
    AfterSalesIdempotencyConflict,
    AfterSalesTransitionError,
    DisputeCase,
    DisputeEvidence,
    ExchangeCase,
    RefundCase,
    ReturnAuthorization,
    ReturnRequest,
    validate_after_sales_order_binding,
)
from protocol.evidence_records import EvidenceRecord, validate_evidence_record
from protocol.errors import SchemaError
from world.after_sales_core import (
    AfterSalesCoreTransitionError,
    AfterSalesIntentError,
    AfterSalesPolicyRevision,
    DisputeResponseRecord,
    derive_after_sales_authority,
    derive_after_sales_binding,
    normalize_after_sales_intent,
    validate_after_sales_policy,
)
from world.after_sales_persistence import AfterSalesTableName, canonical_digest
from world.after_sales_service import (
    AfterSalesPlanner,
    AfterSalesServiceError,
    TrustedAfterSalesContext,
    TrustedReplacementContext,
    load_trusted_after_sales_evidence,
)
from world.errors import (
    AfterSalesReferenceRejected,
    IdempotencyConflict,
    WorldError,
    WriteNotAuthorized,
)
from world.payment_fulfillment import (
    validate_packing_record,
    validate_payment_state_record,
)
from world.types import Order


AFTER_SALES_AUTHORITY_PROJECTION_SCHEMA = "cwe.world-after-sales-authority-projection.v1"
AFTER_SALES_OPERATION_AUTHORITY_SCHEMA = "cwe.world-after-sales-operation-authority.v1"


class AfterSalesProjectionError(ValueError):
    """World cannot issue or verify an exact after-sales projection."""


@dataclass(frozen=True, slots=True)
class AfterSalesOperationAuthority:
    """One currently legal compact operation and its World-bound choices."""

    operation: str
    request_id: str | None = None
    authorization_id: str | None = None
    case_id: str | None = None
    dispute_id: str | None = None
    max_qty: int | None = None
    allowed_conditions: tuple[str, ...] = ()
    replacement_sku_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    allowed_positions: tuple[str, ...] = ()
    schema_id: str = AFTER_SALES_OPERATION_AUTHORITY_SCHEMA


@dataclass(frozen=True, slots=True)
class AfterSalesAuthorityProjection:
    """World-issued authority view for one actor and one order snapshot."""

    projection_id: str
    projection_digest: str
    actor_id: str
    order_id: str
    counterparty_id: str
    issued_at_tick: int
    source_context_digest: str
    operations: tuple[AfterSalesOperationAuthority, ...]
    issuer_id: str = "world"
    schema_id: str = AFTER_SALES_AUTHORITY_PROJECTION_SCHEMA


_OPERATIONS = frozenset(
    {
        "cancel_paid_order",
        "request_return",
        "authorize_return",
        "deny_return",
        "receive_return",
        "open_refund_case",
        "approve_refund",
        "deny_refund",
        "request_exchange",
        "authorize_exchange",
        "deny_exchange",
        "complete_exchange",
        "open_dispute",
        "submit_dispute_evidence",
        "respond_to_dispute",
        "request_ledger_reconciliation",
    }
)

_ROLE_OPERATIONS = {
    "buyer": frozenset(
        {
            "cancel_paid_order",
            "request_return",
            "open_refund_case",
            "request_exchange",
            "open_dispute",
            "submit_dispute_evidence",
            "respond_to_dispute",
            "request_ledger_reconciliation",
        }
    ),
    "merchant": frozenset(
        {
            "cancel_paid_order",
            "authorize_return",
            "deny_return",
            "receive_return",
            "approve_refund",
            "deny_refund",
            "authorize_exchange",
            "deny_exchange",
            "complete_exchange",
            "open_dispute",
            "submit_dispute_evidence",
            "respond_to_dispute",
            "request_ledger_reconciliation",
        }
    ),
}

_CORRELATION_BY_OPERATION: dict[str, frozenset[str]] = {
    "cancel_paid_order": frozenset(),
    "request_return": frozenset(),
    "authorize_return": frozenset({"request_id"}),
    "deny_return": frozenset({"request_id"}),
    "receive_return": frozenset({"request_id", "authorization_id"}),
    "open_refund_case": frozenset(),
    "approve_refund": frozenset({"case_id"}),
    "deny_refund": frozenset({"case_id"}),
    "request_exchange": frozenset(),
    "authorize_exchange": frozenset({"case_id"}),
    "deny_exchange": frozenset({"case_id"}),
    "complete_exchange": frozenset({"case_id"}),
    "open_dispute": frozenset(),
    "submit_dispute_evidence": frozenset({"dispute_id"}),
    "respond_to_dispute": frozenset({"dispute_id"}),
    "request_ledger_reconciliation": frozenset(),
}

_FORBIDDEN_WIRE_KEYS = frozenset(
    {
        "amount",
        "approved_amount",
        "authorized_amount",
        "binding",
        "binding_digest",
        "buyer_id",
        "captured_amount",
        "currency",
        "evidence_digests",
        "facts",
        "floor_price",
        "gross_amount",
        "ledger_sources",
        "merchant_id",
        "net_amount",
        "packing",
        "payment",
        "policy_digest",
        "previous_digest",
        "price",
        "record_digest",
        "refund_amount",
        "refunded_amount",
        "return_authorizer_ids",
        "return_receiver_ids",
        "refund_decider_ids",
        "exchange_authorizer_ids",
        "adjudicator_ids",
        "ledger_reconciler_ids",
        "shipment_digest",
    }
)


def issue_after_sales_authority_projection(
    context: TrustedAfterSalesContext,
    *,
    original_actor: str,
    replacement_options: Sequence[TrustedReplacementContext] = (),
    evidence_records: Sequence[EvidenceRecord] = (),
    planner: AfterSalesPlanner | None = None,
) -> AfterSalesAuthorityProjection:
    """Issue a sealed projection from one exact World context.

    ``replacement_options`` and ``evidence_records`` must already be selected
    by owner-aware World queries.  This function validates their order and
    actor scope, then probes them through the real planner before exposing an
    identifier as a choice.
    """

    order, policy, authority = _validate_context(context, original_actor)
    replacements = _validate_replacements(replacement_options, order=order)
    evidence = _validate_evidence(
        evidence_records,
        order=order,
        actor_id=original_actor,
    )
    probe = planner or AfterSalesPlanner()
    operations = _operation_authorities(
        context,
        actor_id=original_actor,
        policy=policy,
        authority=authority,
        replacements=replacements,
        evidence=evidence,
        planner=probe,
    )
    source_digest = _authority_source_digest(context, replacements, evidence)
    projection_id = (
        "after-sales-authority:"
        + canonical_digest(
            {
                "actor_id": original_actor,
                "order_id": str(order.order_id),
                "issued_at_tick": context.logical_tick,
                "source_context_digest": source_digest,
            }
        )[:32]
    )
    unsigned = AfterSalesAuthorityProjection(
        projection_id=projection_id,
        projection_digest="",
        actor_id=original_actor,
        order_id=str(order.order_id),
        counterparty_id=_counterparty(order, original_actor),
        issued_at_tick=context.logical_tick,
        source_context_digest=source_digest,
        operations=operations,
    )
    sealed = replace(
        unsigned,
        projection_digest=canonical_digest(_projection_contract(unsigned, include_digest=False)),
    )
    validate_after_sales_authority_projection(sealed)
    return sealed


def validate_after_sales_authority_projection(
    projection: AfterSalesAuthorityProjection,
) -> None:
    """Validate schema, canonical ordering, role partition, and integrity seal."""

    if not isinstance(projection, AfterSalesAuthorityProjection):
        raise AfterSalesProjectionError("after-sales projection has the wrong type")
    if projection.schema_id != AFTER_SALES_AUTHORITY_PROJECTION_SCHEMA:
        raise AfterSalesProjectionError("unsupported after-sales projection schema")
    if projection.issuer_id != "world":
        raise AfterSalesProjectionError("after-sales projection issuer must be World")
    for field in (
        "projection_id",
        "projection_digest",
        "actor_id",
        "order_id",
        "counterparty_id",
        "source_context_digest",
    ):
        _text(getattr(projection, field), field)
    _digest(projection.projection_digest, "projection_digest")
    _digest(projection.source_context_digest, "source_context_digest")
    if (
        isinstance(projection.issued_at_tick, bool)
        or not isinstance(projection.issued_at_tick, int)
        or projection.issued_at_tick < 0
    ):
        raise AfterSalesProjectionError("issued_at_tick must be non-negative")
    role = projection.actor_id.split(":", 1)[0]
    if role not in _ROLE_OPERATIONS:
        raise AfterSalesProjectionError("projection actor must be buyer or merchant")
    if projection.counterparty_id == projection.actor_id:
        raise AfterSalesProjectionError("projection counterparty must be distinct")
    expected_order = tuple(sorted(projection.operations, key=lambda row: row.operation))
    if projection.operations != expected_order:
        raise AfterSalesProjectionError("operation authorities are not canonical")
    names = [row.operation for row in projection.operations]
    if len(names) != len(set(names)):
        raise AfterSalesProjectionError("operation authority is duplicated")
    for operation in projection.operations:
        _validate_operation(operation, role=role)
    expected = canonical_digest(_projection_contract(projection, include_digest=False))
    if projection.projection_digest != expected:
        raise AfterSalesProjectionError("after-sales projection seal is invalid")
    _assert_privacy_safe(_projection_to_wire_unchecked(projection))


def after_sales_authority_projection_to_wire(
    projection: AfterSalesAuthorityProjection,
) -> dict[str, Any]:
    """Encode only a valid, sealed projection for a transport boundary."""

    validate_after_sales_authority_projection(projection)
    return _projection_to_wire_unchecked(projection)


def after_sales_authority_projection_from_wire(
    value: Mapping[str, Any],
) -> AfterSalesAuthorityProjection:
    """Strictly decode a projection crossing an in-process or HTTP boundary."""

    if not isinstance(value, Mapping):
        raise AfterSalesProjectionError("after-sales projection wire value must be an object")
    required = {
        "schema_id",
        "issuer_id",
        "projection_id",
        "projection_digest",
        "actor_id",
        "order_id",
        "counterparty_id",
        "issued_at_tick",
        "source_context_digest",
        "operations",
    }
    if set(value) != required:
        raise AfterSalesProjectionError("after-sales projection wire fields are not exact")
    raw_operations = value["operations"]
    if not isinstance(raw_operations, (list, tuple)):
        raise AfterSalesProjectionError("projection operations must be an array")
    operations = tuple(_operation_from_wire(row) for row in raw_operations)
    projection = AfterSalesAuthorityProjection(
        schema_id=cast(str, value["schema_id"]),
        issuer_id=cast(str, value["issuer_id"]),
        projection_id=cast(str, value["projection_id"]),
        projection_digest=cast(str, value["projection_digest"]),
        actor_id=cast(str, value["actor_id"]),
        order_id=cast(str, value["order_id"]),
        counterparty_id=cast(str, value["counterparty_id"]),
        issued_at_tick=cast(int, value["issued_at_tick"]),
        source_context_digest=cast(str, value["source_context_digest"]),
        operations=operations,
    )
    validate_after_sales_authority_projection(projection)
    return projection


def after_sales_projection_is_current(
    projection: AfterSalesAuthorityProjection,
    context: TrustedAfterSalesContext,
    *,
    replacement_options: Sequence[TrustedReplacementContext] = (),
    evidence_records: Sequence[EvidenceRecord] = (),
) -> bool:
    """Return whether a sealed projection still names exact World authority.

    Candidate replacement inventory and evidence versions are part of that
    authority.  Callers must therefore supply the current candidate rows too.
    Omitting rows that were present at issuance correctly makes the projection
    stale, even if the order itself has not changed.
    """

    validate_after_sales_authority_projection(projection)
    if projection.actor_id not in {
        str(context.order.buyer_id),
        str(context.order.merchant_id),
    }:
        return False
    replacements = _validate_replacements(replacement_options, order=context.order)
    evidence = _validate_evidence(
        evidence_records,
        order=context.order,
        actor_id=projection.actor_id,
    )
    return (
        projection.order_id == str(context.order.order_id)
        and projection.issued_at_tick == context.logical_tick
        and projection.source_context_digest
        == _authority_source_digest(context, replacements, evidence)
    )


def _validate_context(
    context: TrustedAfterSalesContext,
    actor_id: str,
) -> tuple[Order, AfterSalesPolicyRevision, Any]:
    if not isinstance(context, TrustedAfterSalesContext):
        raise AfterSalesProjectionError("projection requires a trusted World context")
    order = context.order
    if actor_id not in {str(order.buyer_id), str(order.merchant_id)}:
        raise AfterSalesProjectionError("projection actor is not an order party")
    try:
        validate_payment_state_record(context.payment)
        if context.packing is not None:
            validate_packing_record(context.packing)
        context.tables.state_digest()
    except (ValueError, WorldError, SchemaError) as exc:
        raise AfterSalesProjectionError("World context failed validation") from exc
    if (
        context.payment.order_id != str(order.order_id)
        or context.payment.owner_id != str(order.buyer_id)
        or context.payment.merchant_id != str(order.merchant_id)
        or context.payment.sku_id != str(order.sku_id)
    ):
        raise AfterSalesProjectionError("payment context belongs to another order")
    binding = context.tables.binding_for_order(str(order.order_id), caller="world")
    if binding is None:
        policy = context.tables.latest_policy(str(order.merchant_id), caller="world")
        if policy is None:
            raise AfterSalesProjectionError("merchant has no after-sales policy")
        try:
            binding = derive_after_sales_binding(
                order=order,
                receipt=context.charge_receipt,
                shipment=context.shipment,
                policy=policy,
                payment=context.payment,
            )
        except (ValueError, WorldError, SchemaError) as exc:
            raise AfterSalesProjectionError("World binding derivation failed") from exc
    else:
        validate_after_sales_order_binding(binding)
        policies = [
            cast(AfterSalesPolicyRevision, row)
            for _, row in context.tables.internal_all("after_sales_policies")
            if row.merchant_id == binding.merchant_id
            and row.revision == binding.policy_revision
            and row.policy_digest == binding.policy_digest
        ]
        if len(policies) != 1:
            raise AfterSalesProjectionError("binding policy is missing or ambiguous")
        policy = policies[0]
    validate_after_sales_policy(policy)
    return order, policy, derive_after_sales_authority(binding, policy)


def _validate_replacements(
    values: Sequence[TrustedReplacementContext],
    *,
    order: Order,
) -> tuple[TrustedReplacementContext, ...]:
    rows = tuple(sorted(values, key=lambda row: row.sku_id))
    if len({row.sku_id for row in rows}) != len(rows):
        raise AfterSalesProjectionError("replacement authority is duplicated")
    for row in rows:
        if not isinstance(row, TrustedReplacementContext):
            raise AfterSalesProjectionError("replacement authority has the wrong type")
        if row.merchant_id != str(order.merchant_id):
            raise AfterSalesProjectionError("replacement belongs to another merchant")
    return rows


def _validate_evidence(
    values: Sequence[EvidenceRecord],
    *,
    order: Order,
    actor_id: str,
) -> tuple[EvidenceRecord, ...]:
    rows = tuple(sorted(values, key=lambda row: row.record_id))
    if len({row.record_id for row in rows}) != len(rows):
        raise AfterSalesProjectionError("evidence authority is duplicated")
    for row in rows:
        try:
            validate_evidence_record(row)
        except SchemaError as exc:
            raise AfterSalesProjectionError("evidence authority is malformed") from exc
        if row.subject_id != str(order.order_id):
            raise AfterSalesProjectionError("evidence belongs to another order")
        if actor_id not in {row.owner_id, *row.read_acl}:
            raise AfterSalesProjectionError("evidence is not visible to the actor")
    return rows


def _operation_authorities(
    context: TrustedAfterSalesContext,
    *,
    actor_id: str,
    policy: AfterSalesPolicyRevision,
    authority: Any,
    replacements: Sequence[TrustedReplacementContext],
    evidence: Sequence[EvidenceRecord],
    planner: AfterSalesPlanner,
) -> tuple[AfterSalesOperationAuthority, ...]:
    order_id = str(context.order.order_id)
    role = actor_id.split(":", 1)[0]
    output: list[AfterSalesOperationAuthority] = []

    def legal(
        operation: str,
        fields: Mapping[str, Any],
        *,
        replacement: TrustedReplacementContext | None = None,
        cited: tuple[EvidenceRecord, ...] = (),
        suffix: str = "",
    ) -> bool:
        candidate = replace(
            context,
            replacement=replacement,
            evidence_records=cited,
        )
        return _planner_accepts(
            planner,
            candidate,
            {"op": operation, "order_id": order_id, **fields},
            actor_id=actor_id,
            probe_key=(
                f"projection-probe:{context.logical_tick}:{order_id}:"
                f"{actor_id}:{operation}:{suffix}"
            ),
        )

    def trusted_evidence(operation: str, row: EvidenceRecord) -> bool:
        return _evidence_is_trusted_for_operation(
            row,
            operation=operation,
            context=context,
            policy=policy,
            actor_id=actor_id,
        )

    if legal("cancel_paid_order", {"reason": "authority probe"}):
        output.append(AfterSalesOperationAuthority("cancel_paid_order"))

    if role == "buyer" and legal(
        "request_return",
        {"requested_qty": 1, "reason": "authority probe", "evidence_ids": ()},
    ):
        usable = tuple(
            row.record_id
            for row in evidence
            if trusted_evidence("request_return", row)
            and legal(
                "request_return",
                {
                    "requested_qty": 1,
                    "reason": "authority probe",
                    "evidence_ids": (row.record_id,),
                },
                cited=(row,),
                suffix=row.record_id,
            )
        )
        output.append(
            AfterSalesOperationAuthority(
                "request_return",
                max_qty=context.payment.qty,
                evidence_ids=usable,
            )
        )

    request = _single_row(context, "return_requests")
    authorization = _single_row(context, "return_authorizations")
    receipt = _single_row(context, "return_receipts")
    if role == "merchant" and isinstance(request, ReturnRequest) and authorization is None:
        for operation in ("authorize_return", "deny_return"):
            if legal(
                operation,
                {"request_id": request.request_id, "reason": "authority probe"},
            ):
                output.append(
                    AfterSalesOperationAuthority(operation, request_id=request.request_id)
                )
    if (
        role == "merchant"
        and isinstance(request, ReturnRequest)
        and isinstance(authorization, ReturnAuthorization)
        and authorization.outcome == "authorized"
        and receipt is None
    ):
        conditions = tuple(
            condition
            for condition in policy.allowed_return_conditions
            if legal(
                "receive_return",
                {
                    "request_id": request.request_id,
                    "authorization_id": authorization.authorization_id,
                    "received_qty": 1,
                    "condition": condition,
                },
                suffix=condition,
            )
        )
        if conditions:
            output.append(
                AfterSalesOperationAuthority(
                    "receive_return",
                    request_id=request.request_id,
                    authorization_id=authorization.authorization_id,
                    max_qty=authorization.authorized_qty,
                    allowed_conditions=conditions,
                )
            )

    refund_case = _single_row(context, "refund_cases")
    refund_decision = _single_row(context, "refund_decisions")
    if (
        role == "buyer"
        and refund_case is None
        and legal("open_refund_case", {"reason": "authority probe"})
    ):
        output.append(AfterSalesOperationAuthority("open_refund_case"))
    if role == "merchant" and isinstance(refund_case, RefundCase) and refund_decision is None:
        for operation in ("approve_refund", "deny_refund"):
            if legal(
                operation,
                {"case_id": refund_case.case_id, "reason": "authority probe"},
            ):
                output.append(AfterSalesOperationAuthority(operation, case_id=refund_case.case_id))

    exchange_rows = cast(list[ExchangeCase], _rows(context, "exchange_cases"))
    exchange = max(exchange_rows, key=lambda row: row.version) if exchange_rows else None
    if role == "buyer" and exchange is None:
        replacement_ids = tuple(
            row.sku_id
            for row in replacements
            if legal(
                "request_exchange",
                {"replacement_sku_id": row.sku_id, "reason": "authority probe"},
                replacement=row,
                suffix=row.sku_id,
            )
        )
        if replacement_ids:
            output.append(
                AfterSalesOperationAuthority(
                    "request_exchange", replacement_sku_ids=replacement_ids
                )
            )
    elif role == "merchant" and exchange is not None:
        replacement = next(
            (row for row in replacements if row.sku_id == exchange.replacement_sku_id),
            None,
        )
        if replacement is not None:
            operations = (
                ("authorize_exchange", "deny_exchange")
                if exchange.state == "requested"
                else ("complete_exchange",)
                if exchange.state == "authorized"
                else ()
            )
            for operation in operations:
                if legal(
                    operation,
                    {"case_id": exchange.case_id, "reason": "authority probe"},
                    replacement=replacement,
                ):
                    output.append(AfterSalesOperationAuthority(operation, case_id=exchange.case_id))

    dispute_rows = cast(list[DisputeCase], _rows(context, "dispute_cases"))
    dispute = max(dispute_rows, key=lambda row: row.version) if dispute_rows else None
    if dispute is None and legal("open_dispute", {"reason": "authority probe"}):
        output.append(AfterSalesOperationAuthority("open_dispute"))
    elif dispute is not None and dispute.state in {"open", "under_review"}:
        submitted = {
            row.evidence_id
            for row in cast(list[DisputeEvidence], _rows(context, "dispute_evidence"))
        }
        usable_evidence = tuple(
            row.record_id
            for row in evidence
            if row.record_id not in submitted
            and trusted_evidence("submit_dispute_evidence", row)
            and legal(
                "submit_dispute_evidence",
                {"dispute_id": dispute.dispute_id, "evidence_id": row.record_id},
                cited=(row,),
                suffix=row.record_id,
            )
        )
        if usable_evidence:
            output.append(
                AfterSalesOperationAuthority(
                    "submit_dispute_evidence",
                    dispute_id=dispute.dispute_id,
                    evidence_ids=usable_evidence,
                )
            )
        responses = cast(list[DisputeResponseRecord], _rows(context, "dispute_responses"))
        if actor_id == dispute.against_id and not any(
            row.actor_id == actor_id for row in responses
        ):
            allowed_positions = tuple(
                position
                for position in ("accept", "contest")
                if legal(
                    "respond_to_dispute",
                    {
                        "dispute_id": dispute.dispute_id,
                        "evidence_ids": (),
                        "position": position,
                    },
                    suffix=position,
                )
            )
            response_evidence = tuple(
                row.record_id
                for row in evidence
                if allowed_positions
                and trusted_evidence("respond_to_dispute", row)
                and legal(
                    "respond_to_dispute",
                    {
                        "dispute_id": dispute.dispute_id,
                        "evidence_ids": (row.record_id,),
                        "position": allowed_positions[-1],
                    },
                    cited=(row,),
                    suffix=row.record_id,
                )
            )
            if allowed_positions:
                output.append(
                    AfterSalesOperationAuthority(
                        "respond_to_dispute",
                        dispute_id=dispute.dispute_id,
                        evidence_ids=response_evidence,
                        allowed_positions=allowed_positions,
                    )
                )

    if legal("request_ledger_reconciliation", {"reason": "authority probe"}):
        output.append(AfterSalesOperationAuthority("request_ledger_reconciliation"))
    return tuple(sorted(output, key=lambda row: row.operation))


def _planner_accepts(
    planner: AfterSalesPlanner,
    context: TrustedAfterSalesContext,
    intent: Mapping[str, Any],
    *,
    actor_id: str,
    probe_key: str,
) -> bool:
    try:
        normalized = normalize_after_sales_intent(intent)
        planner.plan(
            context,
            normalized,
            original_actor=actor_id,
            idempotency_key=probe_key,
        )
        return True
    except (
        AfterSalesAuthorityError,
        AfterSalesCoreTransitionError,
        AfterSalesIdempotencyConflict,
        AfterSalesReferenceRejected,
        AfterSalesTransitionError,
        IdempotencyConflict,
        WriteNotAuthorized,
        AfterSalesServiceError,
    ):
        return False
    except AfterSalesIntentError as exc:
        raise AfterSalesProjectionError("projection probe authored an invalid intent") from exc
    except (SchemaError, WorldError, ValueError) as exc:
        raise AfterSalesProjectionError("World projection probe failed") from exc


def _rows(
    context: TrustedAfterSalesContext,
    table: AfterSalesTableName,
) -> list[Any]:
    order_id = str(context.order.order_id)
    return [
        row
        for _, row in context.tables.internal_all(table)
        if getattr(row, "binding", None) is not None and row.binding.order_id == order_id
    ]


def _single_row(
    context: TrustedAfterSalesContext,
    table: AfterSalesTableName,
) -> Any | None:
    rows = _rows(context, table)
    if len(rows) > 1:
        raise AfterSalesProjectionError(f"World {table} history is ambiguous")
    return rows[0] if rows else None


def _counterparty(order: Order, actor_id: str) -> str:
    if actor_id == str(order.buyer_id):
        return str(order.merchant_id)
    if actor_id == str(order.merchant_id):
        return str(order.buyer_id)
    raise AfterSalesProjectionError("projection actor is not an order party")


def _authority_source_digest(
    context: TrustedAfterSalesContext,
    replacements: Sequence[TrustedReplacementContext],
    evidence: Sequence[EvidenceRecord],
) -> str:
    """Bind the projection to every World row used to advertise choices."""

    return canonical_digest(
        {
            "context_digest": context.context_digest,
            "replacement_options": [
                {
                    "sku_id": row.sku_id,
                    "merchant_id": row.merchant_id,
                    "available_qty": row.available_qty,
                    "listing_digest": row.listing_digest,
                    "inventory_digest": row.inventory_digest,
                    "projected_order_digest": row.projected_order_digest,
                }
                for row in replacements
            ],
            "evidence_records": [
                {
                    "record_id": row.record_id,
                    "record_digest": row.record_digest,
                }
                for row in evidence
            ],
        }
    )


def _evidence_is_trusted_for_operation(
    record: EvidenceRecord,
    *,
    operation: str,
    context: TrustedAfterSalesContext,
    policy: AfterSalesPolicyRevision,
    actor_id: str,
) -> bool:
    """Apply the same World evidence trust policy used by real execution."""

    order_id = str(context.order.order_id)
    if operation == "request_return":
        intent: Mapping[str, Any] = {
            "op": operation,
            "order_id": order_id,
            "requested_qty": 1,
            "reason": "authority probe",
            "evidence_ids": (record.record_id,),
        }
    elif operation == "submit_dispute_evidence":
        intent = {
            "op": operation,
            "order_id": order_id,
            "dispute_id": "authority-probe:dispute",
            "evidence_id": record.record_id,
        }
    elif operation == "respond_to_dispute":
        intent = {
            "op": operation,
            "order_id": order_id,
            "dispute_id": "authority-probe:dispute",
            "evidence_ids": (record.record_id,),
            "position": "contest",
        }
    else:
        raise AfterSalesProjectionError("unsupported evidence-bearing projection probe")
    try:
        loaded = load_trusted_after_sales_evidence(
            intent,
            order=context.order,
            policy=policy,
            logical_tick=context.logical_tick,
            original_actor=actor_id,
            lookup=lambda record_id: record if record_id == record.record_id else None,
        )
    except (AfterSalesReferenceRejected, AfterSalesServiceError):
        return False
    return loaded == (record,)


def _validate_operation(
    value: AfterSalesOperationAuthority,
    *,
    role: str,
) -> None:
    if not isinstance(value, AfterSalesOperationAuthority):
        raise AfterSalesProjectionError("operation authority has the wrong type")
    if value.schema_id != AFTER_SALES_OPERATION_AUTHORITY_SCHEMA:
        raise AfterSalesProjectionError("unsupported operation authority schema")
    if value.operation not in _OPERATIONS or value.operation not in _ROLE_OPERATIONS[role]:
        raise AfterSalesProjectionError("operation violates actor role partition")
    correlations = {
        field
        for field in ("request_id", "authorization_id", "case_id", "dispute_id")
        if getattr(value, field) is not None
    }
    if correlations != _CORRELATION_BY_OPERATION[value.operation]:
        raise AfterSalesProjectionError("operation correlation fields are not exact")
    for field in correlations:
        _text(getattr(value, field), field)
    if value.max_qty is not None and (
        isinstance(value.max_qty, bool) or not isinstance(value.max_qty, int) or value.max_qty <= 0
    ):
        raise AfterSalesProjectionError("operation max_qty must be positive")
    if value.operation in {"request_return", "receive_return"}:
        if value.max_qty is None:
            raise AfterSalesProjectionError("quantity operation lacks max_qty")
    elif value.max_qty is not None:
        raise AfterSalesProjectionError("operation has an unexpected max_qty")
    for field in (
        "allowed_conditions",
        "replacement_sku_ids",
        "evidence_ids",
        "allowed_positions",
    ):
        values = getattr(value, field)
        if values != tuple(sorted(set(values))) or any(
            not isinstance(item, str) or not item for item in values
        ):
            raise AfterSalesProjectionError(f"{field} is not canonical")
    if value.operation == "receive_return":
        if not value.allowed_conditions:
            raise AfterSalesProjectionError("receive_return lacks conditions")
    elif value.allowed_conditions:
        raise AfterSalesProjectionError("operation has unexpected conditions")
    if value.operation == "request_exchange":
        if not value.replacement_sku_ids:
            raise AfterSalesProjectionError("request_exchange lacks replacements")
    elif value.replacement_sku_ids:
        raise AfterSalesProjectionError("operation has unexpected replacements")
    if (
        value.operation
        not in {
            "request_return",
            "submit_dispute_evidence",
            "respond_to_dispute",
        }
        and value.evidence_ids
    ):
        raise AfterSalesProjectionError("operation has unexpected evidence choices")
    if value.operation == "respond_to_dispute":
        if not value.allowed_positions or not set(value.allowed_positions).issubset(
            {"accept", "contest"}
        ):
            raise AfterSalesProjectionError("dispute response positions are invalid")
    elif value.allowed_positions:
        raise AfterSalesProjectionError("operation has unexpected positions")


def _projection_contract(
    projection: AfterSalesAuthorityProjection,
    *,
    include_digest: bool,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_id": projection.schema_id,
        "issuer_id": projection.issuer_id,
        "projection_id": projection.projection_id,
        "actor_id": projection.actor_id,
        "order_id": projection.order_id,
        "counterparty_id": projection.counterparty_id,
        "issued_at_tick": projection.issued_at_tick,
        "source_context_digest": projection.source_context_digest,
        "operations": [_operation_to_wire(row) for row in projection.operations],
    }
    if include_digest:
        value["projection_digest"] = projection.projection_digest
    return value


def _projection_to_wire_unchecked(
    projection: AfterSalesAuthorityProjection,
) -> dict[str, Any]:
    return _projection_contract(projection, include_digest=True)


def _operation_to_wire(value: AfterSalesOperationAuthority) -> dict[str, Any]:
    return {
        "schema_id": value.schema_id,
        "operation": value.operation,
        "request_id": value.request_id,
        "authorization_id": value.authorization_id,
        "case_id": value.case_id,
        "dispute_id": value.dispute_id,
        "max_qty": value.max_qty,
        "allowed_conditions": list(value.allowed_conditions),
        "replacement_sku_ids": list(value.replacement_sku_ids),
        "evidence_ids": list(value.evidence_ids),
        "allowed_positions": list(value.allowed_positions),
    }


def _operation_from_wire(value: Any) -> AfterSalesOperationAuthority:
    if not isinstance(value, Mapping):
        raise AfterSalesProjectionError("operation authority wire value is invalid")
    required = {
        "schema_id",
        "operation",
        "request_id",
        "authorization_id",
        "case_id",
        "dispute_id",
        "max_qty",
        "allowed_conditions",
        "replacement_sku_ids",
        "evidence_ids",
        "allowed_positions",
    }
    if set(value) != required:
        raise AfterSalesProjectionError("operation authority wire fields are not exact")
    arrays: dict[str, tuple[str, ...]] = {}
    for field in (
        "allowed_conditions",
        "replacement_sku_ids",
        "evidence_ids",
        "allowed_positions",
    ):
        row = value[field]
        if not isinstance(row, (list, tuple)):
            raise AfterSalesProjectionError(f"{field} must be an array")
        arrays[field] = tuple(cast(Sequence[str], row))
    return AfterSalesOperationAuthority(
        schema_id=cast(str, value["schema_id"]),
        operation=cast(str, value["operation"]),
        request_id=cast(str | None, value["request_id"]),
        authorization_id=cast(str | None, value["authorization_id"]),
        case_id=cast(str | None, value["case_id"]),
        dispute_id=cast(str | None, value["dispute_id"]),
        max_qty=cast(int | None, value["max_qty"]),
        **arrays,
    )


def _assert_privacy_safe(value: Mapping[str, Any]) -> None:
    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            forbidden = set(item).intersection(_FORBIDDEN_WIRE_KEYS)
            if forbidden:
                raise AfterSalesProjectionError(
                    "after-sales projection exposes private/internal fields: "
                    + ", ".join(sorted(forbidden))
                )
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    walk(value)
    # This round trip also rejects dataclasses or other accidental non-JSON
    # state before the projection is allowed near a transport adapter.
    try:
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AfterSalesProjectionError("projection is not strict JSON") from exc


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AfterSalesProjectionError(f"{field} must be non-empty text")
    return value


def _digest(value: Any, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise AfterSalesProjectionError(f"{field} must be lowercase SHA-256")
    return text


__all__ = [
    "AFTER_SALES_AUTHORITY_PROJECTION_SCHEMA",
    "AfterSalesAuthorityProjection",
    "AfterSalesOperationAuthority",
    "AfterSalesProjectionError",
    "after_sales_authority_projection_from_wire",
    "after_sales_authority_projection_to_wire",
    "after_sales_projection_is_current",
    "issue_after_sales_authority_projection",
    "validate_after_sales_authority_projection",
]
