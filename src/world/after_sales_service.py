"""Atomic orchestration contract for CommerceWorld after-sales commands.

The pure rules in :mod:`world.after_sales_core` derive trusted records.  This
module joins those rules to a World transaction without owning a second copy of
commerce state.  The World port supplies authoritative orders, receipts,
shipments, inventory projections, logical time, and persisted after-sales rows.
The planner returns one compare-and-swap commit containing all records and all
commerce effects.  The port must apply that commit together with the order,
inventory, ledger, logical-time, authority-operation, and commit-journal writes
inside one lock or one SQL transaction.

Rejected commands produce no commit.  Exact retries return the prior operation
without advancing World time.  This module has no dependency on benchmark task
ids, scenarios, scorers, agent channels, or transport implementations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast

from protocol.after_sales import (
    AfterSalesOrderBinding,
    DisputeCase,
    DisputeEvidence,
    ExchangeCase,
    RefundCase,
    RefundChain,
    ReturnAuthorization,
    ReturnChain,
    ReturnReceipt,
    ReturnRequest,
    Ruling,
)
from protocol.evidence_records import (
    EvidenceRecord,
    evidence_record_to_dict,
    validate_evidence_record,
)
from world.after_sales_core import (
    AfterSalesCoreAuthority,
    AfterSalesCoreTransitionError,
    AfterSalesPolicyRevision,
    CoreDisputeBundle,
    DisputeResponseRecord,
    LedgerReconciliationRequest,
    LedgerReconciliationSource,
    PaidCancellationRecord,
    after_sales_intent_fingerprint,
    derive_after_sales_authority,
    derive_after_sales_binding,
    derive_dispute_evidence,
    derive_dispute_open,
    derive_dispute_response,
    derive_dispute_ruling,
    derive_exchange_case,
    derive_ledger_reconciliation_request,
    derive_ledger_reconciliation_result,
    derive_paid_cancellation,
    derive_refund_case,
    derive_refund_decision,
    derive_return_authorization,
    derive_return_receipt,
    derive_return_request,
    normalize_after_sales_intent,
    referenced_after_sales_evidence_ids,
    validate_after_sales_policy_transition,
    validate_paid_cancellation_transition,
)
from world.after_sales_persistence import (
    AfterSalesCommit,
    AfterSalesOperationRecord,
    AfterSalesTableName,
    AfterSalesTables,
    AfterSalesWrite,
    CommerceEffect,
    build_after_sales_commit,
    build_after_sales_operation,
    canonical_digest,
    record_digest,
    record_key,
    record_to_wire,
)
from world.errors import AfterSalesReferenceRejected, IdempotencyConflict
from world.payment_fulfillment import (
    PackingRecord,
    PaymentStateRecord,
    packing_record_to_dict,
    payment_state_to_dict,
)
from world.types import Order, Receipt, Shipment


CommandDisposition: TypeAlias = Literal["committed", "idempotent"]


class AfterSalesServiceError(ValueError):
    """The World context cannot safely execute an after-sales command."""


class AfterSalesContextConflict(AfterSalesServiceError):
    """The authoritative context changed before an atomic commit."""


_TRUSTED_EVIDENCE_KINDS: dict[str, tuple[tuple[str, ...], str]] = {
    "carrier_scan": (("carrier:",), "carrier_api"),
    "inspection_report": (
        ("inspector:", "platform:inspection"),
        "authenticated_inspection",
    ),
}


def _reject_actor_reference(
    original_actor: str | None,
    *,
    reason_code: str,
    internal_message: str,
) -> None:
    """Reject actor-selected state while preserving trusted-service failures.

    Only Buyer and Merchant intents cross this public after-sales path.  A
    missing or unusable reference from any trusted service remains an internal
    context failure and therefore cannot be downgraded to a model outcome.
    """

    role = (
        original_actor.split(":", 1)[0]
        if isinstance(original_actor, str) and original_actor
        else ""
    )
    if role in {"buyer", "merchant"}:
        raise AfterSalesReferenceRejected(reason_code)
    raise AfterSalesServiceError(internal_message)


def load_trusted_after_sales_evidence(
    intent: Mapping[str, Any] | None,
    *,
    order: Order,
    policy: AfterSalesPolicyRevision,
    logical_tick: int,
    original_actor: str | None,
    lookup: Callable[[str], EvidenceRecord | None],
) -> tuple[EvidenceRecord, ...]:
    """Resolve and validate exact persisted evidence cited by an actor intent.

    This is a reusable CommerceWorld trust policy.  It is intentionally
    independent of benchmark task identifiers and never accepts actor supplied
    facts, issuers, owners, access lists, or digests.
    """

    if intent is None:
        return ()
    if intent.get("op") not in {
        "request_return",
        "submit_dispute_evidence",
        "respond_to_dispute",
    }:
        return ()
    normalized = normalize_after_sales_intent(intent)
    references = referenced_after_sales_evidence_ids(normalized)
    if references and (not isinstance(original_actor, str) or not original_actor):
        raise AfterSalesServiceError(
            "authenticated actor is required for evidence-bearing intent"
        )
    operation = cast(str, normalized["op"])
    rows: list[EvidenceRecord] = []
    for record_id in references:
        record = lookup(record_id)
        if record is None:
            _reject_actor_reference(
                original_actor,
                reason_code="after_sales_evidence_not_found",
                internal_message=f"referenced evidence does not exist: {record_id}",
            )
        validate_evidence_record(record)
        if record.record_id != record_id:
            raise AfterSalesServiceError("evidence lookup returned the wrong record")
        if record.issued_at_tick > logical_tick:
            _reject_actor_reference(
                original_actor,
                reason_code="after_sales_evidence_stale",
                internal_message="evidence record is from the future",
            )
        if record.subject_id != str(order.order_id):
            _reject_actor_reference(
                original_actor,
                reason_code="after_sales_evidence_identity_mismatch",
                internal_message="evidence is not bound to this order",
            )
        order_parties = {str(order.buyer_id), str(order.merchant_id)}
        if record.owner_id not in order_parties:
            _reject_actor_reference(
                original_actor,
                reason_code="after_sales_evidence_identity_mismatch",
                internal_message="evidence owner is not an order party",
            )
        if original_actor not in {record.owner_id, *record.read_acl}:
            _reject_actor_reference(
                original_actor,
                reason_code="after_sales_evidence_not_authorized",
                internal_message=(
                    "authenticated actor cannot cite this evidence record"
                ),
            )
        required_acl = {
            str(order.buyer_id),
            str(order.merchant_id),
            "platform:after-sales",
        }
        if operation in {"submit_dispute_evidence", "respond_to_dispute"}:
            required_acl.update(policy.adjudicator_ids)
        missing_acl = required_acl.difference(record.read_acl)
        if missing_acl:
            _reject_actor_reference(
                original_actor,
                reason_code="after_sales_evidence_not_usable",
                internal_message=(
                    "evidence ACL omits required order parties or services"
                ),
            )
        trust_rule = _TRUSTED_EVIDENCE_KINDS.get(record.kind)
        if trust_rule is None:
            _reject_actor_reference(
                original_actor,
                reason_code="after_sales_evidence_not_usable",
                internal_message=(
                    f"unsupported after-sales evidence kind: {record.kind}"
                ),
            )
        issuer_prefixes, verification_method = trust_rule
        if not any(record.issuer_id.startswith(prefix) for prefix in issuer_prefixes):
            _reject_actor_reference(
                original_actor,
                reason_code="after_sales_evidence_not_usable",
                internal_message=(
                    "evidence issuer is not trusted for this evidence kind"
                ),
            )
        if record.trust.get("verification_method") != verification_method:
            _reject_actor_reference(
                original_actor,
                reason_code="after_sales_evidence_not_usable",
                internal_message=(
                    "evidence verification method does not match its kind"
                ),
            )
        rows.append(record)
    return tuple(rows)


def _single_evidence_record(context: "TrustedAfterSalesContext") -> EvidenceRecord:
    if len(context.evidence_records) != 1:
        raise AfterSalesServiceError(
            "submit_dispute_evidence requires exactly one trusted evidence record"
        )
    return context.evidence_records[0]


@dataclass(frozen=True, slots=True)
class TrustedReplacementContext:
    """World-derived replacement listing, inventory, and order projection."""

    sku_id: str
    merchant_id: str
    available_qty: int
    listing_digest: str
    inventory_digest: str
    projected_order_digest: str

    def __post_init__(self) -> None:
        for label in ("sku_id", "merchant_id"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise AfterSalesServiceError(f"replacement {label} is invalid")
        if (
            isinstance(self.available_qty, bool)
            or not isinstance(self.available_qty, int)
            or self.available_qty < 0
        ):
            raise AfterSalesServiceError("replacement available_qty is invalid")
        for label in (
            "listing_digest",
            "inventory_digest",
            "projected_order_digest",
        ):
            value = getattr(self, label)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise AfterSalesServiceError(f"replacement {label} is invalid")


@dataclass(frozen=True, slots=True)
class TrustedAfterSalesContext:
    """World-owned rows required to derive and atomically validate a command."""

    logical_tick: int
    order: Order
    payment: PaymentStateRecord
    charge_receipt: Receipt | None
    packing: PackingRecord | None
    shipment: Shipment | None
    settled_at_tick: int
    tables: AfterSalesTables
    replacement: TrustedReplacementContext | None = None
    evidence_records: tuple[EvidenceRecord, ...] = ()
    ledger_sources: tuple[LedgerReconciliationSource, ...] = ()

    @property
    def context_digest(self) -> str:
        """CAS digest over every trusted input consumed by the planner."""

        latest_policy = self.tables.latest_policy(
            str(self.order.merchant_id), caller="world"
        )
        return canonical_digest(
            {
                "logical_tick": self.logical_tick,
                "order": {
                    "order_id": str(self.order.order_id),
                    "buyer_id": str(self.order.buyer_id),
                    "merchant_id": str(self.order.merchant_id),
                    "sku_id": str(self.order.sku_id),
                    "qty": self.order.qty,
                    "price": str(self.order.agreed_price.amount),
                    "currency": self.order.agreed_price.currency,
                    "state": self.order.state.value,
                    "request_order": self.order.request_order,
                },
                "payment": payment_state_to_dict(self.payment),
                "charge_receipt": None
                if self.charge_receipt is None
                else {
                    "txn_id": str(self.charge_receipt.txn_id),
                    "order_id": str(self.charge_receipt.order_id),
                    "buyer_id": str(self.charge_receipt.buyer_id),
                    "merchant_id": str(self.charge_receipt.merchant_id),
                    "sku_id": str(self.charge_receipt.sku_id),
                    "qty": self.charge_receipt.qty,
                    "price": str(self.charge_receipt.price.amount),
                    "currency": self.charge_receipt.price.currency,
                    "idempotency_key": self.charge_receipt.idempotency_key,
                    "effect": self.charge_receipt.effect,
                },
                "packing": None
                if self.packing is None
                else packing_record_to_dict(self.packing),
                "shipment": None
                if self.shipment is None
                else {
                    "shipment_id": str(self.shipment.shipment_id),
                    "order_id": str(self.shipment.order_id),
                    "status": self.shipment.status.value,
                    "version": self.shipment.version,
                },
                "settled_at_tick": self.settled_at_tick,
                "latest_policy": None
                if latest_policy is None
                else record_to_wire(latest_policy),
                "after_sales_state": self.tables.state_digest(),
                "replacement": None
                if self.replacement is None
                else {
                    "sku_id": self.replacement.sku_id,
                    "merchant_id": self.replacement.merchant_id,
                    "available_qty": self.replacement.available_qty,
                    "listing_digest": self.replacement.listing_digest,
                    "inventory_digest": self.replacement.inventory_digest,
                    "projected_order_digest": self.replacement.projected_order_digest,
                },
                "evidence_records": [
                    evidence_record_to_dict(record)
                    for record in self.evidence_records
                ],
                "ledger_sources": [
                    record_to_wire(source) for source in self.ledger_sources
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class AfterSalesCommandResult:
    disposition: CommandDisposition
    operation: AfterSalesOperationRecord
    commit: AfterSalesCommit | None


class AfterSalesWorldPort(Protocol):
    """Required World integration surface.

    ``commit_after_sales`` must compare both ``expected_logical_tick`` and
    ``context_digest`` before writing.  It must then apply all record writes,
    commerce effects, the operation journal, logical time, and transaction
    commit journal atomically.  On any error it must leave every table and the
    clock unchanged.
    """

    def load_after_sales_context(
        self,
        order_id: str,
        *,
        intent: Mapping[str, Any] | None = None,
        original_actor: str | None = None,
        evidence_digests: Mapping[str, str] | None = None,
    ) -> TrustedAfterSalesContext:
        ...

    def commit_after_sales(self, commit: AfterSalesCommit) -> None:
        ...


@dataclass(frozen=True, slots=True)
class PolicyPublicationPlan:
    """Trusted policy append staged for the enclosing World transaction."""

    write: AfterSalesWrite
    expected_logical_tick: int
    logical_tick: int
    previous_policy_digest: str | None


class AfterSalesPlanner:
    """Derive replayable commits from compact intents and trusted World rows."""

    def plan(
        self,
        context: TrustedAfterSalesContext,
        intent: Mapping[str, Any],
        *,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesCommandResult:
        normalized = normalize_after_sales_intent(intent)
        operation = cast(str, normalized["op"])
        order_id = cast(str, normalized["order_id"])
        if order_id != str(context.order.order_id):
            raise AfterSalesServiceError("intent order does not match World context")
        policy, binding, authority, prefix_writes = self._authority_context(context)
        fingerprint = after_sales_intent_fingerprint(
            normalized,
            binding=binding,
            policy=policy,
            original_actor=original_actor,
            evidence_records=context.evidence_records,
        )
        replay = context.tables.operation_for_retry(
            actor_id=original_actor,
            operation=operation,
            idempotency_key=idempotency_key,
        )
        if replay is not None:
            if replay.request_fingerprint != fingerprint:
                raise IdempotencyConflict(
                    "after-sales idempotency key was reused with different intent"
                )
            return AfterSalesCommandResult("idempotent", replay, None)

        tick = _next_tick(context.logical_tick)
        writes, effects, result_table, result_key = self._derive_operation(
            context,
            normalized,
            policy=policy,
            binding=binding,
            authority=authority,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            server_tick=tick,
        )
        all_writes = (*prefix_writes, *writes)
        result = next(
            row.value
            for row in all_writes
            if row.table == result_table and row.key == result_key
        )
        operation_row = build_after_sales_operation(
            operation=operation,
            order_id=binding.order_id,
            binding_digest=binding.binding_digest,
            actor_id=original_actor,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint,
            logical_tick=tick,
            result_table=result_table,
            result_key=result_key,
            result_digest=record_digest(result),
            effects=effects,
        )
        commit = build_after_sales_commit(
            expected_logical_tick=context.logical_tick,
            logical_tick=tick,
            context_digest=context.context_digest,
            writes=all_writes,
            commerce_effects=effects,
            operation=operation_row,
        )
        return AfterSalesCommandResult("committed", operation_row, commit)

    def plan_reconciliation_result(
        self,
        context: TrustedAfterSalesContext,
        *,
        request_id: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesCommandResult:
        """Derive a platform accounting result from typed World ledger sources."""

        policy, binding, authority, prefix_writes = self._authority_context(context)
        if prefix_writes:
            raise AfterSalesServiceError(
                "ledger reconciliation requires a persisted order binding"
            )
        request = _one_by_id(
            context.tables,
            "ledger_reconciliation_requests",
            request_id,
            LedgerReconciliationRequest,
        )
        operation = "complete_ledger_reconciliation"
        replay = context.tables.operation_for_retry(
            actor_id=original_actor,
            operation=operation,
            idempotency_key=idempotency_key,
        )
        tick = _next_tick(context.logical_tick)
        result = derive_ledger_reconciliation_result(
            request,
            ledger_sources=context.ledger_sources,
            authority=authority,
            original_actor=original_actor,
            server_tick=tick,
            idempotency_key=idempotency_key,
        )
        fingerprint = result.request_fingerprint
        if replay is not None:
            if replay.request_fingerprint != fingerprint:
                raise IdempotencyConflict(
                    "ledger reconciliation retry changed authoritative sources"
                )
            return AfterSalesCommandResult("idempotent", replay, None)
        table: AfterSalesTableName = "ledger_reconciliation_results"
        key = record_key(table, result)
        source_writes = tuple(
            AfterSalesWrite(
                "ledger_reconciliation_sources",
                record_key("ledger_reconciliation_sources", source),
                source,
            )
            for source in context.ledger_sources
        )
        write = AfterSalesWrite(table, key, result)
        operation_row = build_after_sales_operation(
            operation=operation,
            order_id=binding.order_id,
            binding_digest=binding.binding_digest,
            actor_id=original_actor,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint,
            logical_tick=tick,
            result_table=table,
            result_key=key,
            result_digest=result.record_digest,
            effects=(),
        )
        commit = build_after_sales_commit(
            expected_logical_tick=context.logical_tick,
            logical_tick=tick,
            context_digest=context.context_digest,
            writes=(*source_writes, write),
            commerce_effects=(),
            operation=operation_row,
        )
        return AfterSalesCommandResult("committed", operation_row, commit)

    def _authority_context(
        self, context: TrustedAfterSalesContext
    ) -> tuple[
        AfterSalesPolicyRevision,
        AfterSalesOrderBinding,
        AfterSalesCoreAuthority,
        tuple[AfterSalesWrite, ...],
    ]:
        existing = context.tables.binding_for_order(
            str(context.order.order_id), caller="world"
        )
        if existing is None:
            policy = context.tables.latest_policy(
                str(context.order.merchant_id), caller="world"
            )
            if policy is None:
                raise AfterSalesServiceError(
                    "merchant has no published after-sales policy"
                )
            binding = derive_after_sales_binding(
                order=context.order,
                receipt=context.charge_receipt,
                shipment=context.shipment,
                policy=policy,
                payment=context.payment,
            )
            table: AfterSalesTableName = "after_sales_bindings"
            prefix = (AfterSalesWrite(table, record_key(table, binding), binding),)
        else:
            binding = existing
            policy = _policy_for_binding(context.tables, binding)
            prefix = ()
        return policy, binding, derive_after_sales_authority(binding, policy), prefix

    def _derive_operation(
        self,
        context: TrustedAfterSalesContext,
        intent: Mapping[str, Any],
        *,
        policy: AfterSalesPolicyRevision,
        binding: AfterSalesOrderBinding,
        authority: AfterSalesCoreAuthority,
        original_actor: str,
        idempotency_key: str,
        server_tick: int,
    ) -> tuple[
        tuple[AfterSalesWrite, ...],
        tuple[CommerceEffect, ...],
        AfterSalesTableName,
        str,
    ]:
        operation = cast(str, intent["op"])
        if operation == "cancel_paid_order":
            return self._cancel_paid(
                context,
                intent,
                policy,
                binding,
                authority,
                original_actor,
                idempotency_key,
                server_tick,
            )
        if operation == "request_return":
            if _rows_for_binding(context.tables, "return_requests", binding):
                raise AfterSalesCoreTransitionError(
                    "order already has a persisted return request"
                )
            record = derive_return_request(
                intent,
                binding=binding,
                policy=policy,
                authority=authority,
                evidence_records=context.evidence_records,
                original_actor=original_actor,
                server_tick=server_tick,
                settled_at_tick=context.settled_at_tick,
                idempotency_key=idempotency_key,
            )
            return _single("return_requests", record)
        if operation in {"authorize_return", "deny_return"}:
            chain = _return_chain(context.tables, binding)
            record = derive_return_authorization(
                intent,
                chain=chain,
                policy=policy,
                authority=authority,
                original_actor=original_actor,
                server_tick=server_tick,
                idempotency_key=idempotency_key,
            )
            return _single("return_authorizations", record)
        if operation == "receive_return":
            chain = _return_chain(context.tables, binding)
            record = derive_return_receipt(
                intent,
                chain=chain,
                policy=policy,
                authority=authority,
                original_actor=original_actor,
                server_tick=server_tick,
                idempotency_key=idempotency_key,
            )
            effect = CommerceEffect(
                kind="mark_returned",
                order_id=binding.order_id,
                sku_id=binding.sku_id,
                qty=record.received_qty,
                amount=0,
                currency=binding.currency,
                inventory_release_qty=0,
            )
            return _single("return_receipts", record, effects=(effect,))
        if operation == "open_refund_case":
            if _rows_for_binding(context.tables, "refund_cases", binding):
                raise AfterSalesCoreTransitionError(
                    "order already has a persisted refund case"
                )
            source = _refund_source(context.tables, binding)
            record = derive_refund_case(
                intent,
                binding=binding,
                policy=policy,
                authority=authority,
                source=source,
                original_actor=original_actor,
                server_tick=server_tick,
                idempotency_key=idempotency_key,
            )
            return _single("refund_cases", record)
        if operation in {"approve_refund", "deny_refund"}:
            chain = _refund_chain(context.tables, binding)
            record = derive_refund_decision(
                intent,
                chain=chain,
                policy=policy,
                authority=authority,
                original_actor=original_actor,
                server_tick=server_tick,
                idempotency_key=idempotency_key,
            )
            effects: tuple[CommerceEffect, ...] = ()
            if record.outcome == "approved" and record.approved_amount > 0:
                effects = (
                    CommerceEffect(
                        kind="issue_refund",
                        order_id=binding.order_id,
                        sku_id=binding.sku_id,
                        qty=binding.qty,
                        amount=record.approved_amount,
                        currency=binding.currency,
                        inventory_release_qty=_refund_inventory_release_qty(
                            context.tables, binding, chain.case
                        ),
                        payment_before_digest=context.payment.record_digest,
                        financial_effect="refund",
                    ),
                )
            return _single("refund_decisions", record, effects=effects)
        if operation in {
            "request_exchange",
            "authorize_exchange",
            "deny_exchange",
            "complete_exchange",
        }:
            return self._exchange(
                context,
                intent,
                policy,
                binding,
                authority,
                original_actor,
                idempotency_key,
                server_tick,
            )
        if operation == "open_dispute":
            if _rows_for_binding(context.tables, "dispute_cases", binding):
                raise AfterSalesCoreTransitionError(
                    "order already has a persisted dispute"
                )
            bundle = derive_dispute_open(
                intent,
                binding=binding,
                policy=policy,
                authority=authority,
                original_actor=original_actor,
                server_tick=server_tick,
                idempotency_key=idempotency_key,
            )
            return _single("dispute_cases", bundle.case)
        if operation == "submit_dispute_evidence":
            before = _dispute_bundle(context.tables, binding, intent)
            trusted_intent = {
                **intent,
                "dispute_id": before.case.dispute_id,
            }
            after = derive_dispute_evidence(
                trusted_intent,
                bundle=before,
                evidence_record=_single_evidence_record(context),
                policy=policy,
                authority=authority,
                original_actor=original_actor,
                server_tick=server_tick,
                idempotency_key=idempotency_key,
            )
            evidence = _new_evidence(before, after)
            return _compound(
                primary_table="dispute_evidence",
                primary=evidence,
                secondary_table="dispute_cases",
                secondary=after.case,
            )
        if operation == "respond_to_dispute":
            before = _dispute_bundle(context.tables, binding, intent)
            trusted_intent = {
                **intent,
                "dispute_id": before.case.dispute_id,
            }
            after = derive_dispute_response(
                trusted_intent,
                bundle=before,
                evidence_records=context.evidence_records,
                policy=policy,
                authority=authority,
                original_actor=original_actor,
                server_tick=server_tick,
                idempotency_key=idempotency_key,
            )
            response = _new_response(before, after)
            return _compound(
                primary_table="dispute_responses",
                primary=response,
                secondary_table="dispute_cases",
                secondary=after.case,
            )
        if operation in {"rule_for_filer", "rule_for_respondent", "rule_split"}:
            before = _dispute_bundle(context.tables, binding, intent)
            trusted_intent = {
                **intent,
                "dispute_id": before.case.dispute_id,
            }
            after = derive_dispute_ruling(
                trusted_intent,
                bundle=before,
                policy=policy,
                authority=authority,
                original_actor=original_actor,
                server_tick=server_tick,
                idempotency_key=idempotency_key,
            )
            if after.ruling is None:
                raise AfterSalesServiceError("ruling derivation returned no ruling")
            return _compound(
                primary_table="after_sales_rulings",
                primary=after.ruling,
                secondary_table="dispute_cases",
                secondary=after.case,
            )
        if operation == "request_ledger_reconciliation":
            record = derive_ledger_reconciliation_request(
                intent,
                binding=binding,
                policy=policy,
                authority=authority,
                original_actor=original_actor,
                server_tick=server_tick,
                idempotency_key=idempotency_key,
            )
            return _single("ledger_reconciliation_requests", record)
        raise AfterSalesServiceError(f"unsupported after-sales operation: {operation}")

    def _cancel_paid(
        self,
        context: TrustedAfterSalesContext,
        intent: Mapping[str, Any],
        policy: AfterSalesPolicyRevision,
        binding: AfterSalesOrderBinding,
        authority: AfterSalesCoreAuthority,
        original_actor: str,
        idempotency_key: str,
        tick: int,
    ) -> tuple[
        tuple[AfterSalesWrite, ...],
        tuple[CommerceEffect, ...],
        AfterSalesTableName,
        str,
    ]:
        record = derive_paid_cancellation(
            intent,
            binding=binding,
            policy=policy,
            authority=authority,
            original_actor=original_actor,
            server_tick=tick,
            idempotency_key=idempotency_key,
            order_state=context.order.state,
            dispatched=context.shipment is not None,
            payment=context.payment,
            packing=context.packing,
        )
        existing_rows = cast(
            list[PaidCancellationRecord],
            _rows_for_binding(context.tables, "paid_cancellations", binding),
        )
        if len(existing_rows) > 1:
            raise AfterSalesServiceError("order has multiple paid cancellations")
        validate_paid_cancellation_transition(
            existing_rows[0] if existing_rows else None,
            record,
            authority=authority,
            server_tick=tick,
        )
        effect = CommerceEffect(
            kind="cancel_paid_order",
            order_id=binding.order_id,
            sku_id=binding.sku_id,
            qty=record.inventory_release_qty,
            amount=record.refund_amount,
            currency=binding.currency,
            inventory_release_qty=record.inventory_release_qty,
            payment_before_digest=record.payment_digest,
            packing_before_digest=record.packing_digest,
            fulfillment_stage=record.fulfillment_stage,
            financial_effect=record.financial_effect,
        )
        return _single("paid_cancellations", record, effects=(effect,))

    def _exchange(
        self,
        context: TrustedAfterSalesContext,
        intent: Mapping[str, Any],
        policy: AfterSalesPolicyRevision,
        binding: AfterSalesOrderBinding,
        authority: AfterSalesCoreAuthority,
        original_actor: str,
        idempotency_key: str,
        tick: int,
    ) -> tuple[
        tuple[AfterSalesWrite, ...],
        tuple[CommerceEffect, ...],
        AfterSalesTableName,
        str,
    ]:
        chain = _return_chain(context.tables, binding)
        previous = _latest_exchange(context.tables, binding)
        operation = cast(str, intent["op"])
        trusted_intent = dict(intent)
        if operation != "request_exchange":
            if previous is None:
                raise AfterSalesCoreTransitionError("exchange case does not exist")
            referenced = resolve_current_versioned_record(
                context.tables,
                "exchange_cases",
                _rows_for_binding(context.tables, "exchange_cases", binding),
                reference=cast(str, intent["case_id"]),
                identity_field="case_id",
            )
            if referenced != previous:
                raise AfterSalesCoreTransitionError(
                    "exchange reference is not the current case"
                )
            trusted_intent["case_id"] = previous.case_id
        requested_sku = (
            cast(str, intent["replacement_sku_id"])
            if operation == "request_exchange"
            else None if previous is None else previous.replacement_sku_id
        )
        replacement = context.replacement
        if replacement is None or replacement.sku_id != requested_sku:
            raise AfterSalesServiceError(
                "World did not provide the requested replacement listing context"
            )
        if replacement.merchant_id != binding.merchant_id:
            raise AfterSalesServiceError(
                "replacement listing is not owned by the order merchant"
            )
        if replacement.available_qty < binding.qty:
            raise AfterSalesCoreTransitionError(
                "replacement inventory cannot satisfy the bound quantity"
            )
        completion_digest = None
        if operation == "complete_exchange":
            completion_digest = replacement.projected_order_digest
        record = derive_exchange_case(
            trusted_intent,
            return_chain=chain,
            previous=previous,
            policy=policy,
            authority=authority,
            original_actor=original_actor,
            server_tick=tick,
            idempotency_key=idempotency_key,
            completion_order_digest=completion_digest,
        )
        effects: tuple[CommerceEffect, ...] = ()
        if record.state == "completed":
            effects = (
                CommerceEffect(
                    kind="complete_exchange",
                    order_id=binding.order_id,
                    sku_id=binding.sku_id,
                    qty=binding.qty,
                    amount=0,
                    currency=binding.currency,
                    inventory_release_qty=binding.qty,
                    replacement_sku_id=record.replacement_sku_id,
                ),
            )
        return _single("exchange_cases", record, effects=effects)


class AfterSalesOrchestrator:
    """Execute planner output through the World atomic commit port."""

    def __init__(
        self, port: AfterSalesWorldPort, *, planner: AfterSalesPlanner | None = None
    ) -> None:
        self._port = port
        self._planner = planner or AfterSalesPlanner()

    def execute(
        self,
        intent: Mapping[str, Any],
        *,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesCommandResult:
        normalized = normalize_after_sales_intent(intent)
        context = self._port.load_after_sales_context(
            cast(str, normalized["order_id"]),
            intent=normalized,
            original_actor=original_actor,
        )
        result = self._planner.plan(
            context,
            normalized,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )
        if result.commit is not None:
            self._port.commit_after_sales(result.commit)
        return result

    def complete_reconciliation(
        self,
        *,
        order_id: str,
        request_id: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesCommandResult:
        context = self._port.load_after_sales_context(
            order_id,
            intent=None,
            original_actor=original_actor,
        )
        result = self._planner.plan_reconciliation_result(
            context,
            request_id=request_id,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )
        if result.commit is not None:
            self._port.commit_after_sales(result.commit)
        return result


def stage_policy_publication(
    tables: AfterSalesTables,
    policy: AfterSalesPolicyRevision,
    *,
    original_actor: str,
    server_tick: int,
    trusted_publisher_ids: Sequence[str],
) -> PolicyPublicationPlan:
    """Validate a policy revision and return its World transaction write.

    The enclosing World must append the write, advance logical time, add an
    authority-operation row, and journal the commit atomically.  Agent actions
    cannot call this function directly; only the trusted policy service can.
    """

    current = tables.latest_policy(policy.merchant_id, caller="world")
    disposition = validate_after_sales_policy_transition(
        current,
        policy,
        original_actor=original_actor,
        server_tick=server_tick,
        trusted_publisher_ids=trusted_publisher_ids,
    )
    if disposition == "idempotent":
        if current is None:
            raise AssertionError("idempotent policy transition has no current row")
        row = current
    else:
        row = policy
    table: AfterSalesTableName = "after_sales_policies"
    return PolicyPublicationPlan(
        write=AfterSalesWrite(table, record_key(table, row), row),
        expected_logical_tick=server_tick - 1,
        logical_tick=server_tick,
        previous_policy_digest=None if current is None else current.policy_digest,
    )


def _single(
    table: AfterSalesTableName,
    record: Any,
    *,
    effects: tuple[CommerceEffect, ...] = (),
) -> tuple[
    tuple[AfterSalesWrite, ...],
    tuple[CommerceEffect, ...],
    AfterSalesTableName,
    str,
]:
    key = record_key(table, record)
    return (AfterSalesWrite(table, key, record),), effects, table, key


def _compound(
    *,
    primary_table: AfterSalesTableName,
    primary: Any,
    secondary_table: AfterSalesTableName,
    secondary: Any,
    effects: tuple[CommerceEffect, ...] = (),
) -> tuple[
    tuple[AfterSalesWrite, ...],
    tuple[CommerceEffect, ...],
    AfterSalesTableName,
    str,
]:
    primary_key = record_key(primary_table, primary)
    secondary_key = record_key(secondary_table, secondary)
    return (
        (
            AfterSalesWrite(primary_table, primary_key, primary),
            AfterSalesWrite(secondary_table, secondary_key, secondary),
        ),
        effects,
        primary_table,
        primary_key,
    )


def _policy_for_binding(
    tables: AfterSalesTables, binding: AfterSalesOrderBinding
) -> AfterSalesPolicyRevision:
    for _, row in tables.internal_all("after_sales_policies"):
        policy = cast(AfterSalesPolicyRevision, row)
        if (
            policy.merchant_id == binding.merchant_id
            and policy.revision == binding.policy_revision
            and policy.policy_digest == binding.policy_digest
        ):
            return policy
    raise AfterSalesServiceError("persisted binding policy revision is missing")


def _rows_for_binding(
    tables: AfterSalesTables,
    table: AfterSalesTableName,
    binding: AfterSalesOrderBinding,
) -> list[Any]:
    return [
        row
        for _, row in tables.internal_all(table)
        if cast(Any, row).binding.binding_digest == binding.binding_digest
    ]


def _return_chain(
    tables: AfterSalesTables, binding: AfterSalesOrderBinding
) -> ReturnChain:
    requests = cast(
        list[ReturnRequest], _rows_for_binding(tables, "return_requests", binding)
    )
    if len(requests) != 1:
        raise AfterSalesCoreTransitionError(
            "operation requires exactly one persisted return request"
        )
    request = requests[0]
    authorizations = [
        cast(ReturnAuthorization, row)
        for row in _rows_for_binding(tables, "return_authorizations", binding)
        if cast(ReturnAuthorization, row).request_id == request.request_id
    ]
    receipts = [
        cast(ReturnReceipt, row)
        for row in _rows_for_binding(tables, "return_receipts", binding)
        if cast(ReturnReceipt, row).request_id == request.request_id
    ]
    if len(authorizations) > 1 or len(receipts) > 1:
        raise AfterSalesServiceError("persisted return chain is ambiguous")
    return ReturnChain(
        request,
        authorizations[0] if authorizations else None,
        receipts[0] if receipts else None,
    )


def _refund_chain(
    tables: AfterSalesTables, binding: AfterSalesOrderBinding
) -> RefundChain:
    cases = cast(list[RefundCase], _rows_for_binding(tables, "refund_cases", binding))
    if len(cases) != 1:
        raise AfterSalesCoreTransitionError(
            "operation requires exactly one persisted refund case"
        )
    decisions = [
        cast(Any, row)
        for row in _rows_for_binding(tables, "refund_decisions", binding)
        if cast(Any, row).case_id == cases[0].case_id
    ]
    if len(decisions) > 1:
        raise AfterSalesServiceError("persisted refund chain is ambiguous")
    return RefundChain(cases[0], decisions[0] if decisions else None)


def _latest_exchange(
    tables: AfterSalesTables, binding: AfterSalesOrderBinding
) -> ExchangeCase | None:
    rows = cast(
        list[ExchangeCase], _rows_for_binding(tables, "exchange_cases", binding)
    )
    if not rows:
        return None
    case_ids = {row.case_id for row in rows}
    if len(case_ids) != 1:
        raise AfterSalesServiceError("order has multiple exchange cases")
    versions = sorted(row.version for row in rows)
    if versions != list(range(1, len(rows) + 1)):
        raise AfterSalesServiceError("exchange version history is not contiguous")
    return max(rows, key=lambda row: row.version)


def _dispute_bundle(
    tables: AfterSalesTables,
    binding: AfterSalesOrderBinding,
    intent: Mapping[str, Any],
) -> CoreDisputeBundle:
    all_cases = [
        cast(DisputeCase, row)
        for row in _rows_for_binding(tables, "dispute_cases", binding)
    ]
    if not all_cases:
        raise AfterSalesCoreTransitionError("dispute does not exist")
    current = cast(
        DisputeCase,
        resolve_current_versioned_record(
            tables,
            "dispute_cases",
            all_cases,
            reference=cast(str, intent["dispute_id"]),
            identity_field="dispute_id",
        ),
    )
    dispute_id = current.dispute_id
    cases = [row for row in all_cases if row.dispute_id == dispute_id]
    versions = sorted(row.version for row in cases)
    if versions != list(range(1, len(cases) + 1)):
        raise AfterSalesServiceError("dispute version history is not contiguous")
    evidence = tuple(
        cast(DisputeEvidence, row)
        for row in _rows_for_binding(tables, "dispute_evidence", binding)
        if cast(DisputeEvidence, row).dispute_id == dispute_id
    )
    responses = tuple(
        cast(DisputeResponseRecord, row)
        for row in _rows_for_binding(tables, "dispute_responses", binding)
        if cast(DisputeResponseRecord, row).dispute_id == dispute_id
    )
    rulings = tuple(
        cast(Ruling, row)
        for row in _rows_for_binding(tables, "after_sales_rulings", binding)
        if cast(Ruling, row).dispute_id == dispute_id
    )
    if len(rulings) > 1:
        raise AfterSalesServiceError("dispute has multiple rulings")
    return CoreDisputeBundle(
        current,
        tuple(sorted(evidence, key=lambda row: (row.logical_tick, row.evidence_id))),
        tuple(sorted(responses, key=lambda row: (row.logical_tick, row.response_id))),
        rulings[0] if rulings else None,
    )


def resolve_current_versioned_record(
    tables: AfterSalesTables,
    table: AfterSalesTableName,
    rows: Sequence[Any],
    *,
    reference: str,
    identity_field: str,
) -> Any:
    """Resolve a logical id or the exact physical key of the current row.

    No suffix is parsed or stripped.  A physical reference is accepted only
    when it equals a key that World actually persists, and stale physical keys
    are rejected after comparing against the current version for that stable
    identity.
    """

    matches = [
        row
        for row in rows
        if reference == getattr(row, identity_field)
        or reference == record_key(table, row)
    ]
    if not matches:
        raise AfterSalesCoreTransitionError(
            f"{identity_field} does not reference a persisted {table} row"
        )
    stable_ids = {getattr(row, identity_field) for row in matches}
    if len(stable_ids) != 1:
        raise AfterSalesServiceError(
            f"{identity_field} reference is ambiguous in World state"
        )
    [stable_id] = stable_ids
    history = [row for row in rows if getattr(row, identity_field) == stable_id]
    current = max(history, key=lambda row: row.version)
    if reference not in {stable_id, record_key(table, current)}:
        raise AfterSalesCoreTransitionError(
            f"{identity_field} references a stale World version"
        )
    # Force the table codec to validate the exact current physical identity.
    persisted = tables.read(table, record_key(table, current), caller="world")
    if persisted != current:
        raise AfterSalesServiceError("current versioned record is not persisted")
    return current


def _refund_source(
    tables: AfterSalesTables, binding: AfterSalesOrderBinding
) -> ReturnChain | CoreDisputeBundle | PaidCancellationRecord:
    disputes = cast(
        list[DisputeCase], _rows_for_binding(tables, "dispute_cases", binding)
    )
    if disputes:
        current = max(disputes, key=lambda row: row.version)
        if current.state == "ruled":
            return _dispute_bundle(
                tables,
                binding,
                {"dispute_id": current.dispute_id},
            )
    returns = cast(
        list[ReturnReceipt], _rows_for_binding(tables, "return_receipts", binding)
    )
    if returns:
        return _return_chain(tables, binding)
    raise AfterSalesCoreTransitionError(
        "refund case requires a received return or ruled dispute"
    )


def _refund_inventory_release_qty(
    tables: AfterSalesTables,
    binding: AfterSalesOrderBinding,
    case: RefundCase,
) -> int:
    if case.source_kind != "return_receipt":
        return 0
    chain = _return_chain(tables, binding)
    if chain.receipt is None or chain.receipt.record_digest != case.causal_digest:
        raise AfterSalesServiceError("refund case return receipt is not current")
    return chain.receipt.received_qty


def _new_evidence(
    before: CoreDisputeBundle, after: CoreDisputeBundle
) -> DisputeEvidence:
    old = {row.record_digest for row in before.evidence}
    rows = [row for row in after.evidence if row.record_digest not in old]
    if len(rows) != 1:
        raise AfterSalesServiceError("evidence command did not append exactly one row")
    return rows[0]


def _new_response(
    before: CoreDisputeBundle, after: CoreDisputeBundle
) -> DisputeResponseRecord:
    old = {row.record_digest for row in before.responses}
    rows = [row for row in after.responses if row.record_digest not in old]
    if len(rows) != 1:
        raise AfterSalesServiceError("response command did not append exactly one row")
    return rows[0]


def _one_by_id(
    tables: AfterSalesTables,
    table: AfterSalesTableName,
    key: str,
    expected_type: type[Any],
) -> Any:
    row = tables.read(table, key, caller="world")
    if row is None or not isinstance(row, expected_type):
        raise AfterSalesServiceError(f"{table} row {key!r} does not exist")
    return row


def _next_tick(current: int) -> int:
    if isinstance(current, bool) or not isinstance(current, int) or current < 0:
        raise AfterSalesServiceError("World logical time is invalid")
    return current + 1


__all__ = [
    "AfterSalesCommandResult",
    "AfterSalesContextConflict",
    "AfterSalesOrchestrator",
    "AfterSalesPlanner",
    "AfterSalesServiceError",
    "AfterSalesWorldPort",
    "PolicyPublicationPlan",
    "TrustedAfterSalesContext",
    "TrustedReplacementContext",
    "load_trusted_after_sales_evidence",
    "stage_policy_publication",
]
