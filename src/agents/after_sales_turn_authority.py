"""World-grounded, high-level Agent authority for after-sales turns.

The language model must not manufacture CommerceWorld protocol lineage.  This
module takes an authenticated inbound envelope and one or more exact World
after-sales contexts.  It exposes only the business choices that the existing
``AfterSalesPlanner`` accepts in those contexts, then deterministically binds
orders, return requests, authorizations, cases, disputes, counterparties, and
service destinations when a model choice is compiled.

There is intentionally no benchmark, scenario, scorer, Runtime, or provider
dependency here.  The planner is used only as a pure legality probe.  It does
not commit a World transition.  Platform and World still validate every
compiled action at execution time, which closes the usual time-of-check to
time-of-use window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, cast

from agents.business_decision import BusinessIntentSpec
from agents.decision_errors import (
    ModelBusinessDecisionError,
    PlatformContractError,
    SemanticBoundaryError,
)

from protocol.after_sales import (
    AfterSalesAuthority,
    AfterSalesAuthorityError,
    AfterSalesIdempotencyConflict,
    AfterSalesTransitionError,
    DisputeCase,
    DisputeEvidence,
    ExchangeCase,
    RefundCase,
    ReturnAuthorization,
    ReturnReceipt,
    ReturnRequest,
    Ruling,
    replay_exchange_cases,
    replay_refund_records,
    replay_return_records,
    validate_after_sales_order_binding,
    validate_after_sales_record,
)
from protocol.envelope import Envelope, validate as validate_envelope
from protocol.errors import SchemaError
from protocol.evidence_records import EvidenceRecord, validate_evidence_record
from world.after_sales_core import (
    AfterSalesCoreTransitionError,
    AfterSalesIntentError,
    AfterSalesPolicyRevision,
    DisputeResponseRecord,
    LedgerReconciliationRequest,
    LedgerReconciliationResult,
    PaidCancellationRecord,
    derive_after_sales_authority,
    derive_after_sales_binding,
    normalize_after_sales_intent,
    validate_after_sales_policy,
    validate_core_after_sales_record,
)
from world.after_sales_authority_projection import (
    AfterSalesAuthorityProjection,
    AfterSalesOperationAuthority,
    AfterSalesProjectionError,
    validate_after_sales_authority_projection,
)
from world.after_sales_persistence import (
    AfterSalesTableName,
    AfterSalesTables,
    AfterSalesWrite,
    after_sales_result_references,
    canonical_digest,
    record_digest,
    validate_after_sales_operation,
)
from world.after_sales_service import (
    AfterSalesPlanner,
    AfterSalesServiceError,
    TrustedAfterSalesContext,
    TrustedReplacementContext,
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


AFTER_SALES_TURN_AUTHORITY_SCHEMA = "cwe.after-sales-turn-authority.v1"


class AfterSalesTurnError(SemanticBoundaryError):
    """Base error at the high-level after-sales Agent boundary."""


class FrameworkAuthorityError(PlatformContractError, AfterSalesTurnError):
    """Authenticated or World-derived turn authority is absent or inconsistent."""


class ModelChoiceError(ModelBusinessDecisionError, AfterSalesTurnError):
    """A model decision is malformed or chooses outside the advertised surface."""


@dataclass(frozen=True, slots=True)
class WorldAfterSalesOrderView:
    """Exact World context plus actor-visible replacement and evidence choices.

    ``context`` is the same immutable input consumed by ``AfterSalesPlanner``.
    The optional choices are also World-authored records.  They are supplied as
    a collection because an Agent turn may need to let the model choose among
    several replacements or evidence rows before World loads the exact intent.
    """

    context: TrustedAfterSalesContext
    replacement_options: tuple[TrustedReplacementContext, ...] = ()
    evidence_records: tuple[EvidenceRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class CompiledAfterSalesAction:
    """A model business choice with all wire authority bound by the Agent."""

    operation: str
    payload: Mapping[str, Any]
    source_msg_id: str
    authority_digest: str


@dataclass(frozen=True, slots=True)
class _OrderAuthority:
    view: WorldAfterSalesOrderView
    order: Order
    policy: AfterSalesPolicyRevision
    authority: Any

    @property
    def order_id(self) -> str:
        return str(self.order.order_id)


@dataclass(frozen=True, slots=True)
class _ProjectedOrderAuthority:
    """Agent-private handle to one sealed World business projection."""

    projection: AfterSalesAuthorityProjection

    @property
    def order_id(self) -> str:
        return self.projection.order_id

    @property
    def counterparty_id(self) -> str:
        return self.projection.counterparty_id


@dataclass(frozen=True, slots=True)
class _Choice:
    operation: str
    order: _OrderAuthority | _ProjectedOrderAuthority
    request_id: str | None = None
    authorization_id: str | None = None
    case_id: str | None = None
    dispute_id: str | None = None
    decisions: tuple[str, ...] = ()
    max_qty: int | None = None
    conditions: tuple[str, ...] = ()
    replacement_skus: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    positions: tuple[str, ...] = ()


_DESCRIPTION = {
    "cancel_paid_order": "Cancel one eligible paid order before dispatch.",
    "request_return": "Request a policy-eligible return for an owned order.",
    "decide_return": "Approve or deny the current return request.",
    "receive_return": "Record receipt and condition of an authorized return.",
    "open_refund_case": "Open a refund case from the current return or ruling.",
    "decide_refund": "Approve or deny the current refund case.",
    "request_exchange": "Request an eligible in-stock replacement.",
    "decide_exchange": "Approve or deny the current exchange request.",
    "complete_exchange": "Complete the currently authorized exchange.",
    "open_dispute": "Open a dispute with the other party to an order.",
    "submit_dispute_evidence": "Submit one World-authorized evidence record.",
    "respond_to_dispute": "Respond to the current dispute and cite visible evidence.",
    "request_ledger_reconciliation": "Ask Platform accounting to reconcile an order ledger.",
}

# Registry operations owned by the complete after-sales phase.  This is the
# domain's single operation vocabulary; it contains no destination or action
# kind and is translated to routes only by ``RouteRegistry``.
AFTER_SALES_ROUTE_OPERATIONS = frozenset(
    {
        "cancel_paid_order",
        "request_order_return",
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
_WIRE_AFTER_SALES_OPERATIONS = AFTER_SALES_ROUTE_OPERATIONS.difference({"request_order_return"}) | {
    "request_return"
}

_CORRELATION_FIELDS = frozenset(
    {
        "request_id",
        "authorization_id",
        "case_id",
        "dispute_id",
        "decision_id",
        "receipt_id",
        "ruling_id",
    }
)
_ACK_REFERENCE_FIELDS = frozenset(
    {
        *_CORRELATION_FIELDS,
        "cancellation_id",
        "evidence_id",
        "response_id",
        "source_id",
        "result_id",
    }
)

_RECIPIENT_FIELDS = frozenset({"recipient_id", "counterparty_id"})
_FORBIDDEN_PROVIDER_FIELDS = frozenset(
    {
        *_CORRELATION_FIELDS,
        *_RECIPIENT_FIELDS,
        "actor_id",
        "buyer_id",
        "merchant_id",
        "owner_id",
        "sender_id",
        "destination",
        "idempotency_key",
        "logical_tick",
        "policy_digest",
        "binding_digest",
        "record_digest",
    }
)

_TEXT = {"type": "string", "minLength": 1, "maxLength": 2000}
_POSITIVE_INTEGER = {"type": "integer", "minimum": 1}


class AfterSalesTurnAuthority:
    """Immutable business-decision surface for one authenticated turn."""

    def __init__(
        self,
        *,
        inbound: Envelope,
        actor_id: str,
        choices: Mapping[str, Sequence[_Choice]],
        specs: Sequence[BusinessIntentSpec],
        context_digests: Sequence[str],
        closed_route_operations: Sequence[str] = (),
    ) -> None:
        self.inbound = inbound
        self.actor_id = actor_id
        self._choices = {name: tuple(rows) for name, rows in sorted(choices.items())}
        self.specs = tuple(specs)
        self._closed_route_operations = frozenset(closed_route_operations)
        if not self._closed_route_operations.issubset(AFTER_SALES_ROUTE_OPERATIONS):
            raise FrameworkAuthorityError("after-sales route closure contains another domain")
        self.authority_digest = canonical_digest(
            {
                "schema_id": AFTER_SALES_TURN_AUTHORITY_SCHEMA,
                "actor_id": actor_id,
                "inbound": {
                    "msg_id": inbound.msg_id,
                    "from": inbound.from_,
                    "to": inbound.to,
                    "in_reply_to": inbound.in_reply_to,
                    "idempotency_key": inbound.idempotency_key,
                    "action_kind": inbound.action["kind"],
                    "payload": inbound.action["payload"],
                },
                "contexts": list(context_digests),
                "business_intents": [spec.to_public_dict() for spec in self.specs],
            }
        )

    def decision_schema(self) -> tuple[BusinessIntentSpec, ...]:
        return self.specs

    def routed_operations(self) -> frozenset[str]:
        """Return every registry operation this exact authority may compile.

        Model intents such as ``decide_return`` branch into approve/deny wire
        operations only after validating the selected decision.  Advertising
        this closed union lets ``AgentPhaseContract`` bind all possible routes
        without pretending that model intent names are route identities.
        """

        routed: set[str] = set()
        for operation, choices in self._choices.items():
            if operation == "request_return":
                routed.add("request_order_return")
            elif operation == "decide_return":
                decisions = {item for row in choices for item in row.decisions}
                if "approve" in decisions:
                    routed.add("authorize_return")
                if "deny" in decisions:
                    routed.add("deny_return")
            elif operation == "decide_refund":
                decisions = {item for row in choices for item in row.decisions}
                if "approve" in decisions:
                    routed.add("approve_refund")
                if "deny" in decisions:
                    routed.add("deny_refund")
            elif operation == "decide_exchange":
                decisions = {item for row in choices for item in row.decisions}
                if "approve" in decisions:
                    routed.add("authorize_exchange")
                if "deny" in decisions:
                    routed.add("deny_exchange")
            else:
                routed.add(operation)
        return frozenset(routed).union(self._closed_route_operations)

    def compile(self, operation: str, arguments: Mapping[str, Any]) -> CompiledAfterSalesAction:
        """Compile one advertised high-level choice into a compact VCP action."""

        if operation not in self._choices:
            raise ModelChoiceError("after-sales intent was not advertised this turn")
        if not isinstance(arguments, Mapping):
            raise ModelChoiceError("after-sales tool arguments must be an object")
        args = dict(arguments)
        if set(args).intersection(_FORBIDDEN_PROVIDER_FIELDS):
            raise ModelChoiceError("model supplied a framework-owned after-sales authority field")
        choices = self._choices[operation]
        choice, args = _select_choice(choices, args)
        payload = self._compile_payload(operation, choice, args)
        compiled_operation = payload.pop("op", None)
        if not isinstance(compiled_operation, str):
            raise FrameworkAuthorityError("compiled after-sales intent lost its operation")
        if compiled_operation == "request_return":
            compiled_operation = "request_order_return"
        return CompiledAfterSalesAction(
            operation=compiled_operation,
            payload=payload,
            source_msg_id=self.inbound.msg_id,
            authority_digest=self.authority_digest,
        )

    def _compile_payload(
        self, operation: str, choice: _Choice, args: dict[str, Any]
    ) -> dict[str, Any]:
        order_id = choice.order.order_id
        reason_operations = {
            "cancel_paid_order",
            "request_return",
            "decide_return",
            "open_refund_case",
            "decide_refund",
            "request_exchange",
            "decide_exchange",
            "complete_exchange",
            "open_dispute",
            "request_ledger_reconciliation",
        }
        reason = (
            _required_text(args.pop("reason", None), "reason")
            if operation in reason_operations
            else None
        )
        intent: dict[str, Any] = {"op": operation, "order_id": order_id}
        if reason is not None:
            intent["reason"] = reason

        if operation == "request_return":
            qty = _positive_int(args.pop("requested_qty", None), "requested_qty")
            if choice.max_qty is None or qty > choice.max_qty:
                raise ModelChoiceError("requested return quantity exceeds World authority")
            evidence = _evidence_selection(args.pop("evidence_ids", ()), choice.evidence_ids)
            intent.update(requested_qty=qty, evidence_ids=evidence)
        elif operation == "decide_return":
            decision = _enum(args.pop("decision", None), choice.decisions, "decision")
            intent["op"] = "authorize_return" if decision == "approve" else "deny_return"
            intent["request_id"] = _bound(choice.request_id, "request_id")
        elif operation == "receive_return":
            qty = _positive_int(args.pop("received_qty", None), "received_qty")
            if choice.max_qty is None or qty > choice.max_qty:
                raise ModelChoiceError("received quantity exceeds World authority")
            intent.update(
                request_id=_bound(choice.request_id, "request_id"),
                authorization_id=_bound(choice.authorization_id, "authorization_id"),
                received_qty=qty,
                condition=_enum(args.pop("condition", None), choice.conditions, "condition"),
            )
        elif operation == "decide_refund":
            decision = _enum(args.pop("decision", None), choice.decisions, "decision")
            intent["op"] = "approve_refund" if decision == "approve" else "deny_refund"
            intent["case_id"] = _bound(choice.case_id, "case_id")
        elif operation == "request_exchange":
            intent["replacement_sku_id"] = _enum(
                args.pop("replacement_sku_id", None),
                choice.replacement_skus,
                "replacement_sku_id",
            )
        elif operation == "decide_exchange":
            decision = _enum(args.pop("decision", None), choice.decisions, "decision")
            intent["op"] = "authorize_exchange" if decision == "approve" else "deny_exchange"
            intent["case_id"] = _bound(choice.case_id, "case_id")
        elif operation == "complete_exchange":
            intent["case_id"] = _bound(choice.case_id, "case_id")
        elif operation == "submit_dispute_evidence":
            intent.update(
                dispute_id=_bound(choice.dispute_id, "dispute_id"),
                evidence_id=_enum(
                    args.pop("evidence_id", None), choice.evidence_ids, "evidence_id"
                ),
            )
        elif operation == "respond_to_dispute":
            intent.update(
                dispute_id=_bound(choice.dispute_id, "dispute_id"),
                evidence_ids=_evidence_selection(args.pop("evidence_ids", ()), choice.evidence_ids),
                position=_enum(
                    args.pop("position", None),
                    choice.positions or ("accept", "contest"),
                    "position",
                ),
            )
        _require_no_arguments(args)
        try:
            return dict(normalize_after_sales_intent(intent))
        except AfterSalesIntentError as exc:
            raise FrameworkAuthorityError(
                "Agent compiled an invalid compact after-sales intent"
            ) from exc


def build_after_sales_turn_authority(
    *,
    inbound: Envelope,
    actor_id: str,
    order_views: Sequence[WorldAfterSalesOrderView],
    planner: AfterSalesPlanner | None = None,
) -> AfterSalesTurnAuthority:
    """Build one turn surface from authenticated input and World state."""

    _validate_inbound(inbound, actor_id)
    if not order_views:
        raise FrameworkAuthorityError("after-sales turn has no World order context")
    actor_role = actor_id.split(":", 1)[0]
    if actor_role not in {"buyer", "merchant"}:
        raise FrameworkAuthorityError("after-sales actor must be buyer or merchant")

    orders: dict[str, _OrderAuthority] = {}
    context_digests: list[str] = []
    for view in order_views:
        row = _validate_order_view(view, actor_id)
        if row.order_id in orders:
            raise FrameworkAuthorityError("duplicate World order context")
        orders[row.order_id] = row
        context_digests.append(_view_authority_digest(view))

    selected = _select_inbound_orders(inbound, actor_id, orders)
    if not selected:
        raise FrameworkAuthorityError("inbound has no actor-visible order authority")
    _validate_inbound_lineage(inbound, actor_id, selected)

    probe = planner or AfterSalesPlanner()
    choices: dict[str, list[_Choice]] = {}
    for row in selected:
        for choice in _legal_choices(inbound, actor_id, row, probe):
            choices.setdefault(choice.operation, []).append(choice)

    specs = [_decision_spec(name, rows) for name, rows in sorted(choices.items())]
    return AfterSalesTurnAuthority(
        inbound=inbound,
        actor_id=actor_id,
        choices=choices,
        specs=specs,
        context_digests=sorted(context_digests),
    )


def build_projected_after_sales_turn_authority(
    *,
    inbound: Envelope,
    actor_id: str,
    projections: Sequence[AfterSalesAuthorityProjection],
    allowed_route_operations: Sequence[str],
) -> AfterSalesTurnAuthority:
    """Build the complete Agent phase from sealed actor-scoped World views.

    ``allowed_route_operations`` is the current public phase's intersection
    with :data:`AFTER_SALES_ROUTE_OPERATIONS`.  It both filters model choices
    and closes every declared after-sales route, including inactive siblings
    of a high-level approve/deny decision.  Consequently an absent legal
    choice can never revive a generic protocol-shaped function.
    """

    _validate_inbound(inbound, actor_id)
    allowed = frozenset(allowed_route_operations)
    if not allowed or not allowed.issubset(AFTER_SALES_ROUTE_OPERATIONS):
        raise FrameworkAuthorityError("after-sales phase has no exact registered route closure")
    if not projections:
        raise FrameworkAuthorityError("after-sales phase has no World projection")

    orders: dict[str, _ProjectedOrderAuthority] = {}
    for projection in projections:
        try:
            validate_after_sales_authority_projection(projection)
        except AfterSalesProjectionError as exc:
            raise FrameworkAuthorityError("after-sales World projection failed validation") from exc
        if projection.actor_id != actor_id:
            raise FrameworkAuthorityError("after-sales World projection crossed actor authority")
        if projection.order_id in orders:
            raise FrameworkAuthorityError("after-sales World projection is duplicated")
        orders[projection.order_id] = _ProjectedOrderAuthority(projection)

    selected = _select_projected_inbound_orders(inbound, actor_id, orders)
    _validate_projected_inbound_lineage(inbound, selected)
    choices: dict[str, list[_Choice]] = {}
    for order in selected:
        for choice in _projected_legal_choices(
            inbound,
            order,
            allowed_route_operations=allowed,
        ):
            choices.setdefault(choice.operation, []).append(choice)
    specs = [_decision_spec(name, rows) for name, rows in sorted(choices.items())]
    return AfterSalesTurnAuthority(
        inbound=inbound,
        actor_id=actor_id,
        choices=choices,
        specs=specs,
        context_digests=sorted(row.projection.projection_digest for row in selected),
        closed_route_operations=sorted(allowed),
    )


def _select_projected_inbound_orders(
    inbound: Envelope,
    actor_id: str,
    orders: Mapping[str, _ProjectedOrderAuthority],
) -> tuple[_ProjectedOrderAuthority, ...]:
    payload = cast(Mapping[str, Any], inbound.action["payload"])
    ledger_cursor_advanced = (
        inbound.action["kind"] == "platform.after_sales_updated"
        and payload.get("operation") == "request_ledger_reconciliation"
    )
    selected_ids: set[str] = set()
    order_id = payload.get("order_id")
    if order_id is not None and not ledger_cursor_advanced:
        if not isinstance(order_id, str) or not order_id:
            raise FrameworkAuthorityError("inbound order_id is malformed")
        selected_ids.add(order_id)
    raw_order_ids = payload.get("order_ids")
    if raw_order_ids is not None and not ledger_cursor_advanced:
        if (
            not isinstance(raw_order_ids, (list, tuple))
            or not raw_order_ids
            or any(not isinstance(value, str) or not value for value in raw_order_ids)
            or len(set(raw_order_ids)) != len(raw_order_ids)
        ):
            raise FrameworkAuthorityError("inbound order_ids are malformed")
        requested_ids = set(cast(Sequence[str], raw_order_ids))
        # A plural task scope supersedes its legacy singular display id.  The
        # Agent-owned projection set remains the actual finite turn authority
        # (one current ledger order for the sequential reconciliation lane).
        selected_ids = requested_ids.intersection(orders)
        if not selected_ids:
            raise FrameworkAuthorityError("inbound references an unauthorized order")
    for field in _RECIPIENT_FIELDS:
        if field in payload and payload[field] != actor_id:
            raise FrameworkAuthorityError("inbound names a different recipient")

    # A Platform acknowledgement is already bound by its exact order/result
    # lineage below.  Its references describe the just-committed row and need
    # not appear in the actor's *next* operation projection (for example, the
    # last submitted dispute evidence).  Peer messages, by contrast, must bind
    # every carried correlation against current World authority.
    correlations = (
        {}
        if inbound.from_.startswith("platform:")
        else {field: payload[field] for field in _CORRELATION_FIELDS if field in payload}
    )
    if any(not isinstance(value, str) or not value for value in correlations.values()):
        raise FrameworkAuthorityError("inbound correlation identity is malformed")
    correlated: set[str] | None = None
    for field, value in correlations.items():
        matched = {
            row.order_id
            for row in orders.values()
            if any(
                getattr(operation, field, None) == value for operation in row.projection.operations
            )
        }
        if len(matched) != 1:
            raise FrameworkAuthorityError(f"inbound {field} has no unique World projection")
        correlated = matched if correlated is None else correlated.intersection(matched)
        if not correlated:
            raise FrameworkAuthorityError("inbound correlations cross after-sales orders")
    if correlated is not None:
        if selected_ids and selected_ids != correlated:
            raise FrameworkAuthorityError("inbound order and case lineage disagree")
        selected_ids = correlated

    if selected_ids:
        missing = selected_ids.difference(orders)
        if missing:
            raise FrameworkAuthorityError("inbound references an unauthorized order")
        candidates = tuple(orders[key] for key in sorted(selected_ids))
    else:
        candidates = tuple(orders[key] for key in sorted(orders))
    if inbound.from_.startswith("platform:"):
        return candidates
    if any(inbound.from_ != row.counterparty_id for row in candidates):
        raise FrameworkAuthorityError("inbound sender is not the bound counterparty")
    return candidates


def _validate_projected_inbound_lineage(
    inbound: Envelope,
    orders: Sequence[_ProjectedOrderAuthority],
) -> None:
    if inbound.action["kind"] != "platform.after_sales_updated":
        return
    payload = cast(Mapping[str, Any], inbound.action["payload"])
    required = {
        "operation",
        "order_id",
        "disposition",
        "result_table",
        "result_key",
        "result_digest",
        "references",
    }
    if set(payload) != required:
        raise FrameworkAuthorityError("Platform after-sales acknowledgement is malformed")
    operation = payload.get("operation")
    if operation != "request_ledger_reconciliation" and payload.get("order_id") not in {
        row.order_id for row in orders
    }:
        raise FrameworkAuthorityError("Platform acknowledgement names another order")
    if operation not in _WIRE_AFTER_SALES_OPERATIONS:
        raise FrameworkAuthorityError("Platform acknowledgement operation is invalid")
    if payload.get("disposition") not in {"committed", "idempotent"}:
        raise FrameworkAuthorityError("Platform acknowledgement disposition is invalid")
    if payload.get("result_table") not in _ALL_RESULT_TABLES:
        raise FrameworkAuthorityError("Platform acknowledgement result table is invalid")
    for field in ("result_key", "result_digest"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise FrameworkAuthorityError("Platform acknowledgement identity is malformed")
    digest = cast(str, payload["result_digest"])
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise FrameworkAuthorityError("Platform acknowledgement digest is malformed")
    references = payload.get("references")
    if not isinstance(references, Mapping) or any(
        key not in _ACK_REFERENCE_FIELDS or not isinstance(value, str) or not value
        for key, value in references.items()
    ):
        raise FrameworkAuthorityError("Platform acknowledgement references are malformed")


def _projected_legal_choices(
    inbound: Envelope,
    order: _ProjectedOrderAuthority,
    *,
    allowed_route_operations: frozenset[str],
) -> tuple[_Choice, ...]:
    rows = {row.operation: row for row in order.projection.operations}
    output: list[_Choice] = []

    def allowed(operation: str) -> AfterSalesOperationAuthority | None:
        route_operation = "request_order_return" if operation == "request_return" else operation
        return rows.get(operation) if route_operation in allowed_route_operations else None

    for operation in (
        "cancel_paid_order",
        "request_return",
        "receive_return",
        "open_refund_case",
        "request_exchange",
        "complete_exchange",
        "open_dispute",
        "submit_dispute_evidence",
        "respond_to_dispute",
        "request_ledger_reconciliation",
    ):
        row = allowed(operation)
        if row is None:
            continue
        output.append(
            _Choice(
                operation,
                order,
                request_id=row.request_id,
                authorization_id=row.authorization_id,
                case_id=row.case_id,
                dispute_id=row.dispute_id,
                max_qty=row.max_qty,
                conditions=row.allowed_conditions,
                replacement_skus=row.replacement_sku_ids,
                evidence_ids=row.evidence_ids,
                positions=row.allowed_positions,
            )
        )

    for intent, branches, correlation in (
        (
            "decide_return",
            (("approve", "authorize_return"), ("deny", "deny_return")),
            "request_id",
        ),
        (
            "decide_refund",
            (("approve", "approve_refund"), ("deny", "deny_refund")),
            "case_id",
        ),
        (
            "decide_exchange",
            (("approve", "authorize_exchange"), ("deny", "deny_exchange")),
            "case_id",
        ),
    ):
        selected = [
            (decision, row)
            for decision, operation in branches
            if (row := allowed(operation)) is not None
        ]
        if not selected:
            continue
        correlations = {getattr(row, correlation) for _, row in selected}
        if len(correlations) != 1 or None in correlations:
            raise FrameworkAuthorityError(
                "after-sales decision branches have inconsistent correlation"
            )
        output.append(
            _Choice(
                intent,
                order,
                request_id=(
                    cast(str, next(iter(correlations))) if correlation == "request_id" else None
                ),
                case_id=(cast(str, next(iter(correlations))) if correlation == "case_id" else None),
                decisions=tuple(decision for decision, _ in selected),
            )
        )

    return tuple(output)


def _validate_inbound(inbound: Envelope, actor_id: str) -> None:
    try:
        validate_envelope(inbound)
    except SchemaError as exc:
        raise FrameworkAuthorityError("after-sales inbound envelope is malformed") from exc
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise FrameworkAuthorityError("after-sales actor identity is missing")
    if inbound.to != actor_id:
        raise FrameworkAuthorityError("after-sales inbound targets another actor")
    if inbound.from_ == actor_id:
        raise FrameworkAuthorityError("after-sales inbound cannot be self-authored")
    if not isinstance(inbound.action.get("payload"), Mapping):
        raise FrameworkAuthorityError("after-sales inbound payload is not an object")


def _view_authority_digest(view: WorldAfterSalesOrderView) -> str:
    """Bind optional model choices as tightly as the underlying World context."""

    return canonical_digest(
        {
            "context_digest": view.context.context_digest,
            "replacement_options": [
                {
                    "sku_id": row.sku_id,
                    "merchant_id": row.merchant_id,
                    "available_qty": row.available_qty,
                    "listing_digest": row.listing_digest,
                    "inventory_digest": row.inventory_digest,
                    "projected_order_digest": row.projected_order_digest,
                }
                for row in sorted(view.replacement_options, key=lambda item: item.sku_id)
            ],
            "evidence_records": [
                {
                    "record_id": row.record_id,
                    "record_digest": row.record_digest,
                }
                for row in sorted(view.evidence_records, key=lambda item: item.record_id)
            ],
        }
    )


def _validate_order_view(view: WorldAfterSalesOrderView, actor_id: str) -> _OrderAuthority:
    if not isinstance(view, WorldAfterSalesOrderView):
        raise FrameworkAuthorityError("after-sales order view has the wrong type")
    context = view.context
    if not isinstance(context, TrustedAfterSalesContext):
        raise FrameworkAuthorityError("after-sales context has the wrong type")
    order = context.order
    if not isinstance(order, Order):
        raise FrameworkAuthorityError("World after-sales context has no typed order")
    parties = {str(order.buyer_id), str(order.merchant_id)}
    if actor_id not in parties:
        raise FrameworkAuthorityError("actor cannot read the supplied World order")
    try:
        validate_payment_state_record(context.payment)
        if context.packing is not None:
            validate_packing_record(context.packing)
        context.tables.state_digest()
    except (ValueError, WorldError, SchemaError) as exc:
        raise FrameworkAuthorityError("World after-sales context failed validation") from exc
    if (
        context.payment.order_id != str(order.order_id)
        or context.payment.owner_id != str(order.buyer_id)
        or context.payment.merchant_id != str(order.merchant_id)
        or context.payment.sku_id != str(order.sku_id)
    ):
        raise FrameworkAuthorityError("payment state is bound to another order")
    if context.packing is not None and context.packing.order_id != str(order.order_id):
        raise FrameworkAuthorityError("packing state is bound to another order")
    if context.shipment is not None and str(context.shipment.order_id) != str(order.order_id):
        raise FrameworkAuthorityError("shipment state is bound to another order")

    binding = context.tables.binding_for_order(str(order.order_id), caller="world")
    if binding is None:
        policy = context.tables.latest_policy(str(order.merchant_id), caller="world")
        if policy is None:
            raise FrameworkAuthorityError("merchant has no World after-sales policy")
        try:
            binding = derive_after_sales_binding(
                order=order,
                receipt=context.charge_receipt,
                shipment=context.shipment,
                policy=policy,
                payment=context.payment,
            )
        except (ValueError, WorldError, SchemaError) as exc:
            raise FrameworkAuthorityError("World could not derive after-sales binding") from exc
    else:
        try:
            validate_after_sales_order_binding(binding)
        except (ValueError, SchemaError) as exc:
            raise FrameworkAuthorityError("persisted after-sales binding is invalid") from exc
        policy = _policy_for_binding(context.tables, binding)
    try:
        validate_after_sales_policy(policy)
        authority = derive_after_sales_authority(binding, policy)
        _validate_history(context.tables, binding, authority.protocol_authority)
    except (ValueError, WorldError, SchemaError) as exc:
        raise FrameworkAuthorityError("persisted after-sales history is inconsistent") from exc

    replacement_ids: set[str] = set()
    for replacement in view.replacement_options:
        if replacement.sku_id in replacement_ids:
            raise FrameworkAuthorityError("duplicate replacement authority")
        replacement_ids.add(replacement.sku_id)
        if replacement.merchant_id != str(order.merchant_id):
            raise FrameworkAuthorityError("replacement belongs to another merchant")
    evidence_ids: set[str] = set()
    for evidence in view.evidence_records:
        try:
            validate_evidence_record(evidence)
        except SchemaError as exc:
            raise FrameworkAuthorityError("World evidence record is malformed") from exc
        if evidence.record_id in evidence_ids:
            raise FrameworkAuthorityError("duplicate evidence authority")
        evidence_ids.add(evidence.record_id)
        if evidence.subject_id != str(order.order_id):
            raise FrameworkAuthorityError("evidence belongs to another order")
        if actor_id not in {evidence.owner_id, *evidence.read_acl}:
            raise FrameworkAuthorityError("evidence is not visible to the actor")
    return _OrderAuthority(view, order, policy, authority)


def _validate_history(
    tables: AfterSalesTables,
    binding: Any,
    protocol_authority: AfterSalesAuthority,
) -> None:
    def rows(table: str) -> list[Any]:
        return [
            row
            for _, row in tables.internal_all(cast(AfterSalesTableName, table))
            if getattr(row, "binding", None) is not None
            and row.binding.order_id == binding.order_id
        ]

    for table in (
        "paid_cancellations",
        "return_requests",
        "return_authorizations",
        "return_receipts",
        "refund_cases",
        "refund_decisions",
        "exchange_cases",
        "dispute_cases",
        "dispute_evidence",
        "dispute_responses",
        "after_sales_rulings",
        "ledger_reconciliation_requests",
        "ledger_reconciliation_results",
    ):
        for row in rows(table):
            if row.binding.binding_digest != binding.binding_digest:
                raise FrameworkAuthorityError("order has cross-binding after-sales rows")
            if isinstance(
                row,
                (
                    PaidCancellationRecord,
                    DisputeResponseRecord,
                    LedgerReconciliationRequest,
                    LedgerReconciliationResult,
                ),
            ):
                validate_core_after_sales_record(row)
            else:
                validate_after_sales_record(row)
    cancellations = rows("paid_cancellations")
    if len(cancellations) > 1:
        raise FrameworkAuthorityError("order has duplicate paid cancellations")
    return_records = [
        *rows("return_requests"),
        *rows("return_authorizations"),
        *rows("return_receipts"),
    ]
    if return_records:
        return_records.sort(
            key=lambda row: (
                row.logical_tick,
                {ReturnRequest: 0, ReturnAuthorization: 1, ReturnReceipt: 2}[type(row)],
            )
        )
        replay_return_records(return_records, authority=protocol_authority)
    refund_records = [*rows("refund_cases"), *rows("refund_decisions")]
    if refund_records:
        refund_records.sort(
            key=lambda row: (row.logical_tick, 0 if isinstance(row, RefundCase) else 1)
        )
        replay_refund_records(refund_records, authority=protocol_authority)
    exchange_rows = cast(list[ExchangeCase], rows("exchange_cases"))
    if exchange_rows:
        exchange_rows.sort(key=lambda row: row.version)
        replay_exchange_cases(exchange_rows, authority=protocol_authority)
    _validate_dispute_history(
        cast(list[DisputeCase], rows("dispute_cases")),
        cast(list[DisputeEvidence], rows("dispute_evidence")),
        cast(list[DisputeResponseRecord], rows("dispute_responses")),
        cast(list[Ruling], rows("after_sales_rulings")),
        binding_digest=binding.binding_digest,
    )


def _validate_dispute_history(
    cases: list[DisputeCase],
    evidence: list[DisputeEvidence],
    responses: list[DisputeResponseRecord],
    rulings: list[Ruling],
    *,
    binding_digest: str,
) -> None:
    if not any((cases, evidence, responses, rulings)):
        return
    if not cases:
        raise FrameworkAuthorityError("dispute child exists without a case")
    ids = {row.dispute_id for row in cases}
    if len(ids) != 1:
        raise FrameworkAuthorityError("order has multiple dispute identities")
    [dispute_id] = ids
    cases.sort(key=lambda row: row.version)
    if [row.version for row in cases] != list(range(1, len(cases) + 1)):
        raise FrameworkAuthorityError("dispute versions are not contiguous")
    if cases[0].previous_digest is not None or cases[0].state != "open":
        raise FrameworkAuthorityError("dispute genesis is invalid")
    immutable = (
        cases[0].filed_by_id,
        cases[0].against_id,
        cases[0].binding.binding_digest,
    )
    for previous, current in zip(cases, cases[1:]):
        if current.previous_digest != previous.record_digest:
            raise FrameworkAuthorityError("dispute predecessor digest is stale")
        if (
            current.filed_by_id,
            current.against_id,
            current.binding.binding_digest,
        ) != immutable:
            raise FrameworkAuthorityError("dispute immutable identity changed")
        if previous.state in {"ruled", "withdrawn"}:
            raise FrameworkAuthorityError("terminal dispute has a later version")
    current_digests = {row.record_digest for row in cases}
    for row in (*evidence, *responses):
        if (
            row.dispute_id != dispute_id
            or row.dispute_digest not in current_digests
            or row.binding.binding_digest != binding_digest
        ):
            raise FrameworkAuthorityError("dispute child has cross-case lineage")
    if len(rulings) > 1:
        raise FrameworkAuthorityError("dispute has multiple rulings")
    if rulings:
        ruling = rulings[0]
        if (
            ruling.dispute_id != dispute_id
            or ruling.dispute_digest not in current_digests
            or cases[-1].state != "ruled"
            or cases[-1].ruling_digest != ruling.record_digest
        ):
            raise FrameworkAuthorityError("dispute ruling lineage is inconsistent")


def _policy_for_binding(tables: AfterSalesTables, binding: Any) -> AfterSalesPolicyRevision:
    matches = [
        cast(AfterSalesPolicyRevision, row)
        for _, row in tables.internal_all("after_sales_policies")
        if row.merchant_id == binding.merchant_id
        and row.revision == binding.policy_revision
        and row.policy_digest == binding.policy_digest
    ]
    if len(matches) != 1:
        raise FrameworkAuthorityError("binding policy revision is missing or ambiguous")
    return matches[0]


def _select_inbound_orders(
    inbound: Envelope,
    actor_id: str,
    orders: Mapping[str, _OrderAuthority],
) -> tuple[_OrderAuthority, ...]:
    payload = cast(Mapping[str, Any], inbound.action["payload"])
    selected_ids: set[str] = set()
    if "order_id" in payload:
        order_id = payload["order_id"]
        if not isinstance(order_id, str) or not order_id:
            raise FrameworkAuthorityError("inbound order_id is malformed")
        selected_ids.add(order_id)
    if "order_ids" in payload:
        values = payload["order_ids"]
        if (
            not isinstance(values, (list, tuple))
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise FrameworkAuthorityError("inbound order_ids are malformed")
        selected_ids.update(values)
    for field in _RECIPIENT_FIELDS:
        if field in payload and payload[field] != actor_id:
            raise FrameworkAuthorityError("inbound names a different recipient")

    correlated = _orders_for_correlations(payload, orders)
    if correlated:
        if selected_ids and selected_ids != correlated:
            raise FrameworkAuthorityError("inbound order and case lineage disagree")
        selected_ids = correlated
    if selected_ids:
        missing = selected_ids.difference(orders)
        if missing:
            raise FrameworkAuthorityError("inbound references an unauthorized order")
        candidates = [orders[order_id] for order_id in sorted(selected_ids)]
    else:
        candidates = list(orders.values())

    if inbound.from_.startswith("platform:"):
        return tuple(candidates)
    peer_matches = [
        row for row in candidates if inbound.from_ == _counterparty(row.order, actor_id)
    ]
    if not peer_matches or len(peer_matches) != len(candidates):
        raise FrameworkAuthorityError("inbound sender is not the bound counterparty")
    return tuple(peer_matches)


def _orders_for_correlations(
    payload: Mapping[str, Any], orders: Mapping[str, _OrderAuthority]
) -> set[str]:
    references = {field: payload[field] for field in _CORRELATION_FIELDS if field in payload}
    if not references:
        return set()
    if any(not isinstance(value, str) or not value for value in references.values()):
        raise FrameworkAuthorityError("inbound correlation identity is malformed")
    common: set[str] | None = None
    for field, value in references.items():
        matched: set[str] = set()
        for order_id, row in orders.items():
            context = row.view.context
            for table in _REFERENCE_TABLES[field]:
                for _, record in context.tables.internal_all(table):
                    if getattr(record, "binding", None) is None:
                        continue
                    if record.binding.order_id != order_id:
                        continue
                    if getattr(record, field, None) == value:
                        matched.add(order_id)
        if not matched:
            raise FrameworkAuthorityError(f"inbound {field} does not exist in World")
        if len(matched) != 1:
            raise FrameworkAuthorityError(f"inbound {field} is ambiguous")
        common = matched if common is None else common.intersection(matched)
        if not common:
            raise FrameworkAuthorityError("inbound correlations cross orders or cases")
    assert common is not None
    return common


_REFERENCE_TABLES: dict[str, tuple[AfterSalesTableName, ...]] = {
    "request_id": (
        "return_requests",
        "return_authorizations",
        "return_receipts",
        "ledger_reconciliation_requests",
        "ledger_reconciliation_results",
    ),
    "authorization_id": ("return_authorizations", "return_receipts"),
    "case_id": ("refund_cases", "refund_decisions", "exchange_cases"),
    "dispute_id": (
        "dispute_cases",
        "dispute_evidence",
        "dispute_responses",
        "after_sales_rulings",
    ),
    "decision_id": ("refund_decisions",),
    "receipt_id": ("return_receipts",),
    "ruling_id": ("after_sales_rulings",),
}


def _validate_inbound_lineage(
    inbound: Envelope, actor_id: str, orders: Sequence[_OrderAuthority]
) -> None:
    kind = inbound.action["kind"]
    payload = cast(Mapping[str, Any], inbound.action["payload"])
    if kind != "platform.after_sales_updated":
        return
    required = {
        "operation",
        "order_id",
        "disposition",
        "result_table",
        "result_key",
        "result_digest",
        "references",
    }
    if set(payload) != required:
        raise FrameworkAuthorityError("Platform after-sales acknowledgement is malformed")
    order_id = payload["order_id"]
    row = next((item for item in orders if item.order_id == order_id), None)
    if row is None:
        raise FrameworkAuthorityError("Platform acknowledgement names another order")
    if payload["disposition"] not in {"committed", "idempotent"}:
        raise FrameworkAuthorityError("Platform acknowledgement disposition is invalid")
    table = payload["result_table"]
    if table not in _ALL_RESULT_TABLES:
        raise FrameworkAuthorityError("Platform acknowledgement result table is invalid")
    key = payload["result_key"]
    digest = payload["result_digest"]
    if not all(isinstance(value, str) and value for value in (key, digest, payload["operation"])):
        raise FrameworkAuthorityError("Platform acknowledgement identity is malformed")
    persisted = row.view.context.tables.read(table, key, caller="world")
    if persisted is None or record_digest(persisted) != digest:
        raise FrameworkAuthorityError("Platform acknowledgement is stale or not persisted")
    matches = [
        operation
        for operation in row.view.context.tables.operations
        if operation.actor_id == actor_id
        and operation.operation == payload["operation"]
        and operation.order_id == order_id
        and operation.result_table == table
        and operation.result_key == key
        and operation.result_digest == digest
    ]
    if len(matches) != 1:
        raise FrameworkAuthorityError("Platform acknowledgement has no exact World operation")
    operation = matches[0]
    validate_after_sales_operation(operation)
    write = AfterSalesWrite(table, key, persisted)
    if payload["references"] != after_sales_result_references(write, operation):
        raise FrameworkAuthorityError("Platform acknowledgement references are inconsistent")


_ALL_RESULT_TABLES: frozenset[AfterSalesTableName] = frozenset(
    {
        "paid_cancellations",
        "return_requests",
        "return_authorizations",
        "return_receipts",
        "refund_cases",
        "refund_decisions",
        "exchange_cases",
        "dispute_cases",
        "dispute_evidence",
        "dispute_responses",
        "after_sales_rulings",
        "ledger_reconciliation_requests",
        "ledger_reconciliation_results",
    }
)


def _legal_choices(
    inbound: Envelope,
    actor_id: str,
    row: _OrderAuthority,
    planner: AfterSalesPlanner,
) -> tuple[_Choice, ...]:
    role = actor_id.split(":", 1)[0]
    context = row.view.context
    order_id = row.order_id
    output: list[_Choice] = []

    def legal(
        operation: str,
        fields: Mapping[str, Any],
        *,
        replacement: TrustedReplacementContext | None = None,
        evidence: tuple[EvidenceRecord, ...] = (),
        suffix: str = "",
    ) -> bool:
        candidate = replace(
            context,
            replacement=replacement,
            evidence_records=evidence,
        )
        intent = {"op": operation, "order_id": order_id, **fields}
        return _planner_accepts(
            planner,
            candidate,
            intent,
            actor_id=actor_id,
            probe_key=f"turn-probe:{inbound.msg_id}:{operation}:{suffix}",
        )

    if legal("cancel_paid_order", {"reason": "model-selected reason"}):
        output.append(_Choice("cancel_paid_order", row))

    evidence_ids = tuple(sorted(record.record_id for record in row.view.evidence_records))
    if role == "buyer" and legal(
        "request_return",
        {"requested_qty": 1, "reason": "model-selected reason", "evidence_ids": ()},
    ):
        return_evidence_ids = tuple(
            record.record_id
            for record in row.view.evidence_records
            if legal(
                "request_return",
                {
                    "requested_qty": 1,
                    "reason": "model-selected reason",
                    "evidence_ids": (record.record_id,),
                },
                evidence=(record,),
                suffix=f"evidence:{record.record_id}",
            )
        )
        output.append(
            _Choice(
                "request_return",
                row,
                max_qty=row.view.context.payment.qty,
                evidence_ids=return_evidence_ids,
            )
        )

    request = _single_row(context.tables, "return_requests", row)
    authorization = _single_row(context.tables, "return_authorizations", row)
    receipt = _single_row(context.tables, "return_receipts", row)
    if role == "merchant" and isinstance(request, ReturnRequest) and authorization is None:
        decisions = tuple(
            decision
            for decision, operation in (("approve", "authorize_return"), ("deny", "deny_return"))
            if legal(
                operation,
                {"request_id": request.request_id, "reason": "model-selected reason"},
                suffix=decision,
            )
        )
        if decisions:
            output.append(
                _Choice("decide_return", row, request_id=request.request_id, decisions=decisions)
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
            for condition in row.policy.allowed_return_conditions
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
                _Choice(
                    "receive_return",
                    row,
                    request_id=request.request_id,
                    authorization_id=authorization.authorization_id,
                    max_qty=authorization.authorized_qty,
                    conditions=conditions,
                )
            )

    refund_case = _single_row(context.tables, "refund_cases", row)
    refund_decision = _single_row(context.tables, "refund_decisions", row)
    if (
        role == "buyer"
        and refund_case is None
        and legal("open_refund_case", {"reason": "model-selected reason"})
    ):
        output.append(_Choice("open_refund_case", row))
    if role == "merchant" and isinstance(refund_case, RefundCase) and refund_decision is None:
        decisions = tuple(
            decision
            for decision, operation in (("approve", "approve_refund"), ("deny", "deny_refund"))
            if legal(
                operation,
                {"case_id": refund_case.case_id, "reason": "model-selected reason"},
                suffix=decision,
            )
        )
        if decisions:
            output.append(
                _Choice("decide_refund", row, case_id=refund_case.case_id, decisions=decisions)
            )

    exchanges = _rows(context.tables, "exchange_cases", row)
    exchange = (
        max(cast(list[ExchangeCase], exchanges), key=lambda item: item.version)
        if exchanges
        else None
    )
    if role == "buyer" and exchange is None:
        replacements = tuple(
            option.sku_id
            for option in row.view.replacement_options
            if legal(
                "request_exchange",
                {"replacement_sku_id": option.sku_id, "reason": "model-selected reason"},
                replacement=option,
                suffix=option.sku_id,
            )
        )
        if replacements:
            output.append(_Choice("request_exchange", row, replacement_skus=replacements))
    elif role == "merchant" and isinstance(exchange, ExchangeCase):
        replacement = next(
            (
                option
                for option in row.view.replacement_options
                if option.sku_id == exchange.replacement_sku_id
            ),
            None,
        )
        if exchange.state == "requested" and replacement is not None:
            decisions = tuple(
                decision
                for decision, operation in (
                    ("approve", "authorize_exchange"),
                    ("deny", "deny_exchange"),
                )
                if legal(
                    operation,
                    {"case_id": exchange.case_id, "reason": "model-selected reason"},
                    replacement=replacement,
                    suffix=decision,
                )
            )
            if decisions:
                output.append(
                    _Choice("decide_exchange", row, case_id=exchange.case_id, decisions=decisions)
                )
        elif (
            exchange.state == "authorized"
            and replacement is not None
            and legal(
                "complete_exchange",
                {"case_id": exchange.case_id, "reason": "model-selected reason"},
                replacement=replacement,
            )
        ):
            output.append(_Choice("complete_exchange", row, case_id=exchange.case_id))

    dispute_cases = cast(list[DisputeCase], _rows(context.tables, "dispute_cases", row))
    dispute = max(dispute_cases, key=lambda item: item.version) if dispute_cases else None
    if dispute is None and legal("open_dispute", {"reason": "model-selected reason"}):
        output.append(_Choice("open_dispute", row))
    elif dispute is not None and dispute.state in {"open", "under_review"}:
        available = tuple(
            record
            for record in row.view.evidence_records
            if record.record_id
            not in {
                item.evidence_id
                for item in cast(
                    list[DisputeEvidence], _rows(context.tables, "dispute_evidence", row)
                )
            }
        )
        legal_evidence = tuple(
            record.record_id
            for record in available
            if legal(
                "submit_dispute_evidence",
                {"dispute_id": dispute.dispute_id, "evidence_id": record.record_id},
                evidence=(record,),
                suffix=record.record_id,
            )
        )
        if legal_evidence:
            output.append(
                _Choice(
                    "submit_dispute_evidence",
                    row,
                    dispute_id=dispute.dispute_id,
                    evidence_ids=legal_evidence,
                )
            )
        response_rows = cast(
            list[DisputeResponseRecord], _rows(context.tables, "dispute_responses", row)
        )
        if actor_id == dispute.against_id and not any(
            item.actor_id == actor_id for item in response_rows
        ):
            positions = tuple(
                position
                for position in ("accept", "contest")
                if legal(
                    "respond_to_dispute",
                    {
                        "dispute_id": dispute.dispute_id,
                        "evidence_ids": (),
                        "position": position,
                    },
                    evidence=(),
                    suffix=f"response-position:{position}",
                )
            )
            selectable = tuple(
                record
                for record in row.view.evidence_records
                if record.record_id in evidence_ids
                and "contest" in positions
                and legal(
                    "respond_to_dispute",
                    {
                        "dispute_id": dispute.dispute_id,
                        "evidence_ids": (record.record_id,),
                        "position": "contest",
                    },
                    evidence=(record,),
                    suffix=f"response-evidence:{record.record_id}",
                )
            )
            if positions:
                output.append(
                    _Choice(
                        "respond_to_dispute",
                        row,
                        dispute_id=dispute.dispute_id,
                        evidence_ids=tuple(record.record_id for record in selectable),
                        positions=positions,
                    )
                )

    if legal("request_ledger_reconciliation", {"reason": "model-selected reason"}):
        output.append(_Choice("request_ledger_reconciliation", row))

    return tuple(output)


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
        AfterSalesCoreTransitionError,
        AfterSalesAuthorityError,
        AfterSalesIdempotencyConflict,
        AfterSalesReferenceRejected,
        AfterSalesTransitionError,
        IdempotencyConflict,
        WriteNotAuthorized,
        AfterSalesServiceError,
    ):
        return False
    except AfterSalesIntentError as exc:
        raise FrameworkAuthorityError("Agent legality probe authored an invalid intent") from exc
    except (SchemaError, WorldError, ValueError) as exc:
        raise FrameworkAuthorityError("World legality probe failed unexpectedly") from exc


def _decision_spec(
    operation: str,
    choices: Sequence[_Choice],
) -> BusinessIntentSpec:
    properties: dict[str, Any] = {}
    required: list[str] = []
    order_ids = tuple(sorted({choice.order.order_id for choice in choices}))
    if len(order_ids) > 1:
        properties["order_id"] = {"type": "string", "enum": list(order_ids)}
        required.append("order_id")
    if operation in {
        "cancel_paid_order",
        "request_return",
        "decide_return",
        "open_refund_case",
        "decide_refund",
        "request_exchange",
        "decide_exchange",
        "complete_exchange",
        "open_dispute",
        "request_ledger_reconciliation",
    }:
        properties["reason"] = _json_copy(_TEXT)
        required.append("reason")
    if operation == "request_return":
        properties["requested_qty"] = {
            **_json_copy(_POSITIVE_INTEGER),
            "maximum": max(
                cast(int, choice.max_qty) for choice in choices if choice.max_qty is not None
            ),
        }
        evidence = sorted({item for choice in choices for item in choice.evidence_ids})
        if evidence:
            properties["evidence_ids"] = _enum_array(evidence)
        required.append("requested_qty")
    elif operation in {"decide_return", "decide_refund", "decide_exchange"}:
        decisions = sorted({item for choice in choices for item in choice.decisions})
        properties["decision"] = {"type": "string", "enum": decisions}
        required.append("decision")
    elif operation == "receive_return":
        properties["received_qty"] = {
            **_json_copy(_POSITIVE_INTEGER),
            "maximum": max(
                cast(int, choice.max_qty) for choice in choices if choice.max_qty is not None
            ),
        }
        properties["condition"] = {
            "type": "string",
            "enum": sorted({item for choice in choices for item in choice.conditions}),
        }
        required.extend(("received_qty", "condition"))
    elif operation == "request_exchange":
        properties["replacement_sku_id"] = {
            "type": "string",
            "enum": sorted({item for choice in choices for item in choice.replacement_skus}),
        }
        required.append("replacement_sku_id")
    elif operation == "submit_dispute_evidence":
        properties["evidence_id"] = {
            "type": "string",
            "enum": sorted({item for choice in choices for item in choice.evidence_ids}),
        }
        required.append("evidence_id")
    elif operation == "respond_to_dispute":
        properties["position"] = {
            "type": "string",
            "enum": sorted(
                {item for choice in choices for item in (choice.positions or ("accept", "contest"))}
            ),
        }
        evidence = sorted({item for choice in choices for item in choice.evidence_ids})
        if evidence:
            properties["evidence_ids"] = _enum_array(evidence)
        required.append("position")
    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    forbidden = set(properties).intersection(_FORBIDDEN_PROVIDER_FIELDS)
    if forbidden:
        raise FrameworkAuthorityError("business schema exposes protocol authority")
    return BusinessIntentSpec(
        intent=operation,
        description=_DESCRIPTION[operation],
        parameters=schema,
        category="act",
        source_name=operation,
    )


def _select_choice(
    choices: Sequence[_Choice], args: dict[str, Any]
) -> tuple[_Choice, dict[str, Any]]:
    order_ids = {choice.order.order_id for choice in choices}
    if len(order_ids) > 1:
        selected = args.pop("order_id", None)
        if selected not in order_ids:
            raise ModelChoiceError("order_id must be one advertised World choice")
    else:
        if "order_id" in args:
            raise ModelChoiceError("single-order authority binds order_id automatically")
        selected = next(iter(order_ids))
    matching = [choice for choice in choices if choice.order.order_id == selected]
    if len(matching) != 1:
        raise FrameworkAuthorityError("after-sales tool has ambiguous order authority")
    return matching[0], args


def _rows(
    tables: AfterSalesTables,
    table: AfterSalesTableName,
    row: _OrderAuthority,
) -> list[Any]:
    return [
        value
        for _, value in tables.internal_all(table)
        if getattr(value, "binding", None) is not None and value.binding.order_id == row.order_id
    ]


def _single_row(
    tables: AfterSalesTables,
    table: AfterSalesTableName,
    row: _OrderAuthority,
) -> Any | None:
    values = _rows(tables, table, row)
    if len(values) > 1:
        raise FrameworkAuthorityError(f"World {table} history is ambiguous")
    return values[0] if values else None


def _counterparty(order: Order, actor_id: str) -> str:
    if actor_id == str(order.buyer_id):
        return str(order.merchant_id)
    if actor_id == str(order.merchant_id):
        return str(order.buyer_id)
    raise FrameworkAuthorityError("actor is not an order party")


def _bound(value: str | None, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrameworkAuthorityError(f"Agent lacks bound {field}")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ModelChoiceError(f"{field} must be non-empty text")
    return value.strip()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelChoiceError(f"{field} must be a positive integer")
    return value


def _enum(value: Any, allowed: Sequence[str], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ModelChoiceError(f"{field} must be one advertised World choice")
    return value


def _evidence_selection(value: Any, allowed: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ModelChoiceError("evidence_ids must be an array")
    if any(not isinstance(item, str) for item in value):
        raise ModelChoiceError("evidence_ids must contain strings")
    selected = tuple(sorted(set(value)))
    if len(selected) != len(value) or not set(selected).issubset(allowed):
        raise ModelChoiceError("evidence_ids must be unique advertised World choices")
    return selected


def _require_no_arguments(args: Mapping[str, Any]) -> None:
    if args:
        raise ModelChoiceError("after-sales tool has unexpected arguments")


def _enum_array(values: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "enum": list(values)},
        "uniqueItems": True,
    }


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


__all__ = [
    "AFTER_SALES_ROUTE_OPERATIONS",
    "AFTER_SALES_TURN_AUTHORITY_SCHEMA",
    "AfterSalesTurnAuthority",
    "CompiledAfterSalesAction",
    "FrameworkAuthorityError",
    "ModelChoiceError",
    "WorldAfterSalesOrderView",
    "build_projected_after_sales_turn_authority",
    "build_after_sales_turn_authority",
]
