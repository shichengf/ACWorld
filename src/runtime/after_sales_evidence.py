"""Exact Runtime, Platform, and World evidence for after-sales commands.

This verifier is a reusable CommerceWorld authority contract.  It proves that
an authenticated compact actor request was accepted by the stateless
after-sales Platform service, caused the exact actor-scoped World operation,
and produced canonical typed rows in the replayed final snapshot.  It is
independent of benchmark task identifiers and scoring rubrics.

The episode evidence manifest validates hashes and deterministic replay before
this contract is called.  This module adds the semantic join that replay alone
cannot establish: request, decision, response, intent fingerprint, authority
operation, atomic World commit, typed outcome, and causal payment or packing
rows must all agree.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from protocol.after_sales import AfterSalesOrderBinding, DisputeEvidence, Ruling
from protocol.evidence_records import (
    EvidenceRecord,
    coerce_evidence_record,
    evidence_record_to_dict,
)
from runtime.exact_join import (
    DEFAULT_OPERATION_EVIDENCE_REGISTRY,
    ExactJoinContext,
    ExactJoinError,
    LinkedPlatformExchange,
    OperationEvidenceContract,
)
from runtime.world_state_replay import apply_commit_writes
from world.after_sales_core import (
    AFTER_SALES_INTENT_SCHEMA,
    DisputeResponseRecord,
    LedgerReconciliationRequest,
    LedgerReconciliationResult,
    LedgerReconciliationSource,
    after_sales_intent_fingerprint,
    after_sales_policy_from_dict,
    authoritative_receipt_digest,
    derive_ledger_reconciliation_source,
    normalize_after_sales_intent,
    referenced_after_sales_evidence_ids,
)
from world.after_sales_persistence import (
    AfterSalesWrite,
    physical_after_sales_record_key,
    record_digest,
    record_from_wire,
    record_key,
)
from world.payment_fulfillment import (
    packing_record_from_dict,
    payment_state_from_dict,
    validate_packing_transition,
    validate_payment_transition,
)
from world.types import AgentId, Money, OrderId, Receipt, SkuId, TxnId


AFTER_SALES_EVIDENCE_CONTRACT = "commerceworld.after-sales.v1"
AFTER_SALES_ENDPOINT = "platform:after-sales"
AFTER_SALES_RESPONSE_KIND = "platform.after_sales_updated"
AFTER_SALES_READ_RESPONSE_KIND = "platform.after_sales_snapshot"

ACTION_TO_OPERATION: Mapping[str, str] = {
    "commerce.cancel_paid_order": "cancel_paid_order",
    "commerce.request_return": "request_return",
    "commerce.authorize_return": "authorize_return",
    "commerce.deny_return": "deny_return",
    "commerce.receive_return": "receive_return",
    "commerce.open_refund_case": "open_refund_case",
    "commerce.approve_refund": "approve_refund",
    "commerce.deny_refund": "deny_refund",
    "commerce.request_exchange": "request_exchange",
    "commerce.authorize_exchange": "authorize_exchange",
    "commerce.deny_exchange": "deny_exchange",
    "commerce.complete_exchange": "complete_exchange",
    "commerce.open_dispute": "open_dispute",
    "commerce.submit_dispute_evidence": "submit_dispute_evidence",
    "commerce.respond_to_dispute": "respond_to_dispute",
    "commerce.request_ledger_reconciliation": "request_ledger_reconciliation",
}

ACTION_TO_READ_RESOURCE: Mapping[str, str] = {
    "commerce.read_payment_history": "payment_history",
    "commerce.read_ledger_history": "ledger_history",
    "commerce.read_packing_history": "packing_history",
    "commerce.read_after_sales_history": "after_sales_history",
    "commerce.read_after_sales_policy": "policy",
}

_REQUIRED_COMMIT_INVARIANTS = frozenset(
    {
        "after-sales-context-cas",
        "principal-provenance-preserved",
        "single-clock-advance",
    }
)

_REFERENCE_FIELDS_BY_TABLE: Mapping[str, tuple[str, ...]] = {
    "paid_cancellations": ("cancellation_id",),
    "return_requests": ("request_id",),
    "return_authorizations": ("authorization_id", "request_id"),
    "return_receipts": ("receipt_id", "request_id", "authorization_id"),
    "refund_cases": ("case_id",),
    "refund_decisions": ("decision_id", "case_id"),
    "exchange_cases": ("case_id",),
    "dispute_cases": ("dispute_id",),
    "dispute_evidence": ("evidence_id", "dispute_id"),
    "dispute_responses": ("response_id", "dispute_id"),
    "after_sales_rulings": ("ruling_id", "dispute_id"),
    "ledger_reconciliation_requests": ("request_id",),
    "ledger_reconciliation_results": ("result_id", "request_id"),
}


@dataclass(frozen=True, slots=True)
class VerifiedAfterSalesRequest:
    """One accepted request joined to its exact World authority effect."""

    exchange: LinkedPlatformExchange
    response: dict[str, Any]
    operation: str
    intent: dict[str, Any]
    request_fingerprint: str
    authority_operation: dict[str, Any]
    result_record: dict[str, Any]
    commit: dict[str, Any]
    evidence_records: tuple[EvidenceRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifiedRejectedAfterSalesRequest:
    """One rejected request proven not to have its claimed World effect."""

    exchange: LinkedPlatformExchange
    operation: str | None
    reason_code: str


@dataclass(frozen=True, slots=True)
class VerifiedAfterSalesRead:
    """One actor-scoped Platform read matched to an exact World projection."""

    exchange: LinkedPlatformExchange
    resource: str
    actor_id: str
    order_id: str | None
    merchant_id: str | None
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class VerifiedAfterSalesServiceOperation:
    """One trusted non-agent operation joined through typed causal records."""

    operation: str
    actor_id: str
    authority_operation: dict[str, Any]
    result_record: dict[str, Any]
    commit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedAfterSalesEvidence:
    """Complete selected after-sales authority graph."""

    requests: tuple[VerifiedAfterSalesRequest, ...]
    service_operations: tuple[VerifiedAfterSalesServiceOperation, ...] = ()
    rejected_requests: tuple[VerifiedRejectedAfterSalesRequest, ...] = ()
    reads: tuple[VerifiedAfterSalesRead, ...] = ()

    @property
    def operations(self) -> tuple[str, ...]:
        ordered = sorted(
            (
                *((_commit_sequence(row.commit), row.operation) for row in self.requests),
                *((_commit_sequence(row.commit), row.operation) for row in self.service_operations),
            )
        )
        return tuple(operation for _, operation in ordered)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_requests)

    def requests_for_actor(self, actor_id: str) -> tuple[VerifiedAfterSalesRequest, ...]:
        return tuple(
            row for row in self.requests if row.exchange.decision.get("actor_id") == actor_id
        )


def verify_after_sales_evidence_contract(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedAfterSalesEvidence:
    """Verify every after-sales Platform exchange selected by ``options``."""

    allowed_options = {
        "expected_order_id",
        "expected_order_ids",
        "allowed_order_ids",
        "expected_actor_id",
        "expected_operations",
        "expected_actor_operations",
        "expected_service_operations",
        "expected_service_order_ids",
        "expected_read_resources",
        "allow_rejected",
    }
    unknown_options = sorted(set(options) - allowed_options)
    if unknown_options:
        raise ExactJoinError("unknown after-sales evidence options: " + ", ".join(unknown_options))
    expected_order = _optional_text(options.get("expected_order_id"), "expected_order_id")
    expected_order_ids = _text_sequence(options.get("expected_order_ids", ()), "expected_order_ids")
    allowed_order_ids = _text_sequence(options.get("allowed_order_ids", ()), "allowed_order_ids")
    if sum(bool(value) for value in (expected_order, expected_order_ids, allowed_order_ids)) > 1:
        raise ExactJoinError(
            "expected_order_id, expected_order_ids, and allowed_order_ids are mutually exclusive"
        )
    if len(set(expected_order_ids)) != len(expected_order_ids):
        raise ExactJoinError("expected_order_ids must be unique")
    if len(set(allowed_order_ids)) != len(allowed_order_ids):
        raise ExactJoinError("allowed_order_ids must be unique")
    expected_orders = (expected_order,) if expected_order is not None else expected_order_ids
    expected_order_set = set(expected_orders)
    selected_order_set = expected_order_set or set(allowed_order_ids)
    expected_actor = _optional_text(options.get("expected_actor_id"), "expected_actor_id")
    expected_operations = _text_sequence(
        options.get("expected_operations", ()), "expected_operations"
    )
    expected_actor_operations = _text_sequence(
        options.get("expected_actor_operations", ()),
        "expected_actor_operations",
    )
    expected_service_operations = _text_sequence(
        options.get("expected_service_operations", ()),
        "expected_service_operations",
    )
    expected_service_order_ids = _text_sequence(
        options.get("expected_service_order_ids", ()),
        "expected_service_order_ids",
    )
    expected_read_resources = _text_sequence(
        options.get("expected_read_resources", ()),
        "expected_read_resources",
    )
    if any(
        resource not in set(ACTION_TO_READ_RESOURCE.values())
        for resource in expected_read_resources
    ):
        raise ExactJoinError("expected_read_resources contains an unknown resource")
    if expected_service_order_ids and (
        len(expected_service_order_ids) != len(expected_service_operations)
    ):
        raise ExactJoinError(
            "expected_service_order_ids must align with expected_service_operations"
        )
    if selected_order_set and not set(expected_service_order_ids).issubset(selected_order_set):
        raise ExactJoinError("service operation references an unexpected order")
    if not expected_service_order_ids:
        if len(expected_orders) == 1:
            expected_service_order_ids = expected_orders * len(expected_service_operations)
        elif len(expected_orders) == len(expected_service_operations):
            expected_service_order_ids = expected_orders
        elif expected_service_operations:
            expected_service_order_ids = ("",) * len(expected_service_operations)
    allow_rejected = options.get("allow_rejected", True)
    if not isinstance(allow_rejected, bool):
        raise ExactJoinError("allow_rejected must be boolean")

    exchanges = tuple(
        row
        for row in context.exchanges
        if row.decision.get("platform_endpoint") == AFTER_SALES_ENDPOINT
    )
    if not exchanges:
        raise ExactJoinError("after-sales evidence has no Platform exchange")
    positioned_commits = _positioned_after_sales_commits(context, exchanges)

    final_records = _after_sales_records(context.final_tables)
    bindings = {
        _binding_order_id(row): row
        for row in final_records.values()
        if row["table"] == "after_sales_bindings"
    }
    policies = _policies(context.final_tables)
    final_authority = _authority_operations(context.final_tables)
    evidence_records = _evidence_records(context.final_tables)
    _validate_payment_and_packing_histories(context.final_tables)

    accepted: list[VerifiedAfterSalesRequest] = []
    services: list[VerifiedAfterSalesServiceOperation] = []
    rejected: list[VerifiedRejectedAfterSalesRequest] = []
    reads: list[VerifiedAfterSalesRead] = []
    claimed_commits: set[str] = set()

    for exchange in exchanges:
        decision = exchange.decision
        actor_id = _text(decision.get("actor_id"), "Platform actor_id")
        action_kind = _text(decision.get("action_kind"), "Platform action_kind")
        read_resource = ACTION_TO_READ_RESOURCE.get(action_kind)
        if read_resource is not None:
            read_order_id, merchant_id = _read_identity(exchange, resource=read_resource)
            if (
                selected_order_set
                and read_order_id is not None
                and read_order_id not in selected_order_set
            ):
                raise ExactJoinError("after-sales read references another order")
            if decision.get("decision") == "rejected":
                if exchange.responses:
                    raise ExactJoinError("rejected after-sales read emitted a response")
                if not allow_rejected:
                    raise ExactJoinError("selected after-sales flow contains a rejected read")
                rejected.append(
                    VerifiedRejectedAfterSalesRequest(
                        exchange=exchange,
                        operation=None,
                        reason_code=_text(decision.get("reason_code"), "reason_code"),
                    )
                )
                continue
            if decision.get("decision") != "accepted":
                raise ExactJoinError("after-sales Platform decision is invalid")
            reads.append(
                _verify_read_response(
                    exchange,
                    tables=_state_before_request(
                        context,
                        positioned_commits=positioned_commits,
                        request_position=exchange.request_position,
                    ),
                    resource=read_resource,
                    actor_id=actor_id,
                    order_id=read_order_id,
                    merchant_id=merchant_id,
                )
            )
            continue
        operation = ACTION_TO_OPERATION.get(action_kind)
        if operation is None:
            raise ExactJoinError(f"unsupported after-sales action kind {action_kind!r}")
        intent = _intent_from_exchange(exchange, operation)
        order_id = _text(intent.get("order_id"), "order_id")
        if selected_order_set and order_id not in selected_order_set:
            raise ExactJoinError("after-sales request references another order")

        if decision.get("decision") == "rejected":
            if exchange.responses:
                raise ExactJoinError("rejected after-sales request emitted a response")
            if not allow_rejected:
                raise ExactJoinError("selected after-sales flow contains a rejection")
            if any(
                row.get("authority_action") == "world.apply_after_sales_intent"
                and row.get("operation") == operation
                and row.get("subject_id") == order_id
                and row.get("actor_id") == actor_id
                and row.get("idempotency_key") == exchange.request.get("idempotency_key")
                for row in context.world_commits
            ):
                raise ExactJoinError("rejected after-sales request has a World effect")
            rejected.append(
                VerifiedRejectedAfterSalesRequest(
                    exchange=exchange,
                    operation=operation,
                    reason_code=_text(decision.get("reason_code"), "reason_code"),
                )
            )
            continue
        if decision.get("decision") != "accepted":
            raise ExactJoinError("after-sales Platform decision is invalid")

        binding_wrapper = bindings.get(order_id)
        if binding_wrapper is None:
            raise ExactJoinError("after-sales order has no typed World binding")
        binding = binding_wrapper["typed"]
        policy = policies.get(binding.merchant_id)
        if policy is None or policy.policy_digest != binding.policy_digest:
            raise ExactJoinError("after-sales binding and policy do not agree")
        try:
            referenced_evidence = _referenced_evidence_records(
                intent,
                evidence_records=evidence_records,
                binding=binding,
                actor_id=actor_id,
            )
            fingerprint = after_sales_intent_fingerprint(
                intent,
                binding=binding,
                policy=policy,
                original_actor=actor_id,
                evidence_records=referenced_evidence,
            )
        except Exception as exc:
            raise ExactJoinError("after-sales request is not a valid compact intent") from exc

        key = _text(exchange.request.get("idempotency_key"), "idempotency_key")
        commit = _unique_commit(
            context,
            operation=operation,
            order_id=order_id,
            actor_id=actor_id,
            idempotency_key=key,
            request_fingerprint=fingerprint,
        )
        commit_id = _text(commit.get("commit_id"), "commit_id")
        if commit_id in claimed_commits:
            raise ExactJoinError("two after-sales requests claimed one World commit")
        claimed_commits.add(commit_id)

        authority, result = _verify_commit(
            commit,
            final_records=final_records,
            final_authority=final_authority,
            operation=operation,
            order_id=order_id,
            actor_id=actor_id,
            idempotency_key=key,
            request_fingerprint=fingerprint,
            scope="apply_after_sales_intent",
        )
        _verify_result_evidence_binding(
            operation=operation,
            intent=intent,
            result=result,
            evidence_records=referenced_evidence,
        )
        response = _verify_response(
            exchange,
            operation=operation,
            order_id=order_id,
            result=result,
        )
        accepted.append(
            VerifiedAfterSalesRequest(
                exchange=exchange,
                response=response,
                operation=operation,
                intent=intent,
                request_fingerprint=fingerprint,
                authority_operation=authority,
                result_record=result,
                commit=commit,
                evidence_records=referenced_evidence,
            )
        )

    if not expected_service_operations:
        discovered_services = sorted(
            (
                _commit_sequence(row),
                _text(row.get("operation"), "service operation"),
                _text(row.get("subject_id"), "service order_id"),
            )
            for row in context.world_commits
            if row.get("commit_kind") == "transaction"
            and row.get("actor_id") in {"platform:adjudicator", "platform:accounting"}
            and row.get("operation")
            in {
                "rule_for_filer",
                "rule_for_respondent",
                "rule_split",
                "complete_ledger_reconciliation",
            }
            and (not selected_order_set or row.get("subject_id") in selected_order_set)
        )
        expected_service_operations = tuple(row[1] for row in discovered_services)
        expected_service_order_ids = tuple(row[2] for row in discovered_services)

    for operation, service_order_id in zip(
        expected_service_operations,
        expected_service_order_ids,
        strict=True,
    ):
        service = _verify_service_operation(
            context,
            operation=operation,
            expected_order_id=service_order_id or None,
            final_records=final_records,
            final_authority=final_authority,
            claimed_commits=claimed_commits,
        )
        claimed_commits.add(_text(service.commit.get("commit_id"), "commit_id"))
        services.append(service)

    accepted.sort(key=lambda row: _commit_sequence(row.commit))
    services.sort(key=lambda row: _commit_sequence(row.commit))
    complete = sorted(
        (
            *((_commit_sequence(row.commit), row.operation) for row in accepted),
            *((_commit_sequence(row.commit), row.operation) for row in services),
        )
    )
    observed = tuple(operation for _, operation in complete)
    if expected_operations and observed != expected_operations:
        raise ExactJoinError(
            "after-sales operation sequence changed: "
            f"expected {expected_operations!r}, observed {observed!r}"
        )
    if expected_actor is not None:
        actor_observed = tuple(
            row.operation
            for row in accepted
            if row.exchange.decision.get("actor_id") == expected_actor
        )
        actor_reads = tuple(
            row.resource for row in reads if row.actor_id == expected_actor
        )
        actor_rejected = tuple(
            row for row in rejected if row.exchange.decision.get("actor_id") == expected_actor
        )
        if not actor_observed and not actor_reads and not actor_rejected:
            raise ExactJoinError("selected actor has no after-sales request")
        if expected_actor_operations and actor_observed != expected_actor_operations:
            raise ExactJoinError(
                "selected actor operation sequence changed: "
                f"expected {expected_actor_operations!r}, observed {actor_observed!r}"
            )
        if expected_read_resources and actor_reads != expected_read_resources:
            raise ExactJoinError(
                "selected actor read sequence changed: "
                f"expected {expected_read_resources!r}, observed {actor_reads!r}"
            )
    if expected_order_set:
        observed_orders = {_text(row.intent.get("order_id"), "order_id") for row in accepted}
        observed_orders.update(
            _text(
                row.exchange.request.get("action", {}).get("payload", {}).get("order_id"),
                "rejected order_id",
            )
            for row in rejected
        )
        observed_orders.update(
            row.order_id for row in reads if row.order_id is not None
        )
        if observed_orders != expected_order_set:
            raise ExactJoinError(
                "after-sales request order coverage changed: "
                f"expected {sorted(expected_order_set)!r}, "
                f"observed {sorted(observed_orders)!r}"
            )
    unclaimed = [
        row
        for row in context.world_commits
        if row.get("authority_action")
        in {
            "world.apply_after_sales_intent",
            "world.complete_ledger_reconciliation",
        }
        and row.get("commit_id") not in claimed_commits
    ]
    if unclaimed:
        raise ExactJoinError("after-sales evidence contains an unclaimed World operation")
    return VerifiedAfterSalesEvidence(
        requests=tuple(accepted),
        service_operations=tuple(services),
        rejected_requests=tuple(rejected),
        reads=tuple(reads),
    )


def _positioned_after_sales_commits(
    context: ExactJoinContext,
    exchanges: tuple[LinkedPlatformExchange, ...],
) -> tuple[tuple[int, dict[str, Any]], ...]:
    """Bind every after-sales World commit to its triggering request position."""

    request_positions = tuple(exchange.request_position for exchange in exchanges)
    if any(
        isinstance(position, bool) or not isinstance(position, int) or position < 0
        for position in request_positions
    ) or len(set(request_positions)) != len(request_positions):
        raise ExactJoinError("after-sales Platform request positions are invalid")

    relevant = tuple(
        commit
        for commit in context.world_commits
        if commit.get("commit_kind") == "transaction"
        and commit.get("authority_action")
        in {
            "world.apply_after_sales_intent",
            "world.complete_ledger_reconciliation",
        }
    )
    commit_ids = tuple(_text(commit.get("commit_id"), "commit_id") for commit in relevant)
    if len(commit_ids) != len(set(commit_ids)):
        raise ExactJoinError("after-sales World commit ids are not unique")
    journal_sequences = tuple(_commit_sequence(commit) for commit in relevant)
    if journal_sequences != tuple(sorted(journal_sequences)) or len(
        journal_sequences
    ) != len(set(journal_sequences)):
        raise ExactJoinError("after-sales World commit journal order changed")

    positioned: list[tuple[int, dict[str, Any]]] = []
    for commit in relevant:
        matches = [
            exchange
            for exchange in exchanges
            if _after_sales_commit_was_triggered_by(commit, exchange)
        ]
        if len(matches) != 1:
            raise ExactJoinError(
                "after-sales World commit has no unique triggering Platform request"
            )
        positioned.append((matches[0].request_position, commit))
    positioned.sort(key=lambda row: (row[0], _commit_sequence(row[1])))
    if tuple(_commit_sequence(commit) for _, commit in positioned) != journal_sequences:
        raise ExactJoinError("after-sales commit order crosses Platform request order")
    return tuple(positioned)


def _after_sales_commit_was_triggered_by(
    commit: Mapping[str, Any],
    exchange: LinkedPlatformExchange,
) -> bool:
    decision = exchange.decision
    if decision.get("decision") != "accepted":
        return False
    action_kind = decision.get("action_kind")
    operation = ACTION_TO_OPERATION.get(str(action_kind))
    if operation is None:
        return False
    action = exchange.request.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if not isinstance(payload, Mapping):
        return False
    order_id = payload.get("order_id")
    raw_key = exchange.request.get("idempotency_key")
    actor_id = decision.get("actor_id")
    if not all(isinstance(value, str) and value for value in (order_id, raw_key, actor_id)):
        return False
    if commit.get("subject_id") != order_id:
        return False

    if (
        commit.get("authority_action") == "world.apply_after_sales_intent"
        and commit.get("operation") == operation
        and commit.get("actor_id") == actor_id
        and commit.get("idempotency_key") == raw_key
    ):
        return True
    if operation == "respond_to_dispute":
        return bool(
            commit.get("authority_action") == "world.apply_after_sales_intent"
            and commit.get("operation")
            in {"rule_for_filer", "rule_for_respondent", "rule_split"}
            and commit.get("actor_id") == "platform:adjudicator"
            and commit.get("idempotency_key") == f"{raw_key}:adjudicator"
        )
    if operation == "request_ledger_reconciliation":
        return bool(
            commit.get("authority_action") == "world.complete_ledger_reconciliation"
            and commit.get("operation") == "complete_ledger_reconciliation"
            and commit.get("actor_id") == "platform:accounting"
            and commit.get("idempotency_key") == f"{raw_key}:accounting"
        )
    return False


def _state_before_request(
    context: ExactJoinContext,
    *,
    positioned_commits: tuple[tuple[int, dict[str, Any]], ...],
    request_position: int,
) -> dict[str, Any]:
    """Replay exactly the after-sales commit prefix visible to one read."""

    state = copy.deepcopy(context.initial_tables)
    for position, commit in positioned_commits:
        if position >= request_position:
            break
        apply_commit_writes(state, commit, allowed_tables=frozenset(state))
    return state


def _intent_from_exchange(exchange: LinkedPlatformExchange, operation: str) -> dict[str, Any]:
    action = exchange.request.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if not isinstance(payload, Mapping) or "op" in payload:
        raise ExactJoinError("agent after-sales payload must omit the World operation field")
    try:
        return dict(normalize_after_sales_intent({"op": operation, **dict(payload)}))
    except Exception as exc:
        raise ExactJoinError("agent after-sales payload is not exact") from exc


def _read_identity(
    exchange: LinkedPlatformExchange, *, resource: str
) -> tuple[str | None, str | None]:
    action = exchange.request.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ExactJoinError("after-sales read payload is missing")
    if resource == "policy":
        if set(payload) != {"merchant_id"}:
            raise ExactJoinError("policy read payload is not exact")
        return None, _text(payload.get("merchant_id"), "merchant_id")
    if set(payload) != {"order_id"}:
        raise ExactJoinError("history read payload is not exact")
    return _text(payload.get("order_id"), "order_id"), None


def _verify_read_response(
    exchange: LinkedPlatformExchange,
    *,
    tables: Mapping[str, Any],
    resource: str,
    actor_id: str,
    order_id: str | None,
    merchant_id: str | None,
) -> VerifiedAfterSalesRead:
    if len(exchange.responses) != 1:
        raise ExactJoinError("accepted after-sales read needs one response")
    response = exchange.responses[0]
    request = exchange.request
    if (
        response.get("from") != AFTER_SALES_ENDPOINT
        or response.get("to") != actor_id
        or response.get("in_reply_to") != request.get("msg_id")
        or response.get("idempotency_key") != request.get("idempotency_key")
    ):
        raise ExactJoinError("after-sales read response identity changed")
    action = response.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if (
        not isinstance(action, Mapping)
        or action.get("kind") != AFTER_SALES_READ_RESPONSE_KIND
        or not isinstance(payload, Mapping)
        or set(payload) != {"resource", "records"}
        or payload.get("resource") != resource
    ):
        raise ExactJoinError("after-sales read response schema changed")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not all(
        isinstance(row, Mapping) for row in raw_records
    ):
        raise ExactJoinError("after-sales read records are invalid")
    records = tuple(json.loads(_canonical_text(dict(row))) for row in raw_records)
    expected = _expected_read_records(
        tables,
        resource=resource,
        actor_id=actor_id,
        order_id=order_id,
        merchant_id=merchant_id,
    )
    if records != expected:
        raise ExactJoinError("after-sales read differs from the caller-scoped World projection")
    return VerifiedAfterSalesRead(
        exchange=exchange,
        resource=resource,
        actor_id=actor_id,
        order_id=order_id,
        merchant_id=merchant_id,
        records=records,
    )


def _expected_read_records(
    tables: Mapping[str, Any],
    *,
    resource: str,
    actor_id: str,
    order_id: str | None,
    merchant_id: str | None,
) -> tuple[dict[str, Any], ...]:
    if resource == "policy":
        raw = tables.get("after_sales_policies", [])
        if not isinstance(raw, list) or merchant_id is None:
            raise ExactJoinError("World policy table is invalid")
        matches = [
            dict(row)
            for row in raw
            if isinstance(row, Mapping) and row.get("merchant_id") == merchant_id
        ]
        if not matches:
            return ()
        latest = max(matches, key=lambda row: int(row.get("revision", 0)))
        public_fields = (
            "schema_id",
            "policy_id",
            "merchant_id",
            "revision",
            "effective_tick",
            "return_window_ticks",
            "max_refund_bps",
            "split_refund_bps",
            "owner_paid_cancel_allowed",
            "merchant_paid_cancel_allowed",
            "allowed_return_conditions",
            "policy_digest",
        )
        if any(field not in latest for field in public_fields):
            raise ExactJoinError("World policy row is incomplete")
        return (json.loads(_canonical_text({field: latest[field] for field in public_fields})),)

    if order_id is None:
        raise ExactJoinError("history read has no order id")
    _require_order_party(tables, order_id=order_id, actor_id=actor_id)
    table_name = {
        "payment_history": "payment_states",
        "ledger_history": "ledger",
        "packing_history": "packing_records",
    }.get(resource)
    if table_name is not None:
        raw = tables.get(table_name, [])
        if not isinstance(raw, list):
            raise ExactJoinError(f"World {table_name} table is invalid")
        rows = [
            json.loads(_canonical_text(dict(row)))
            for row in raw
            if isinstance(row, Mapping) and row.get("order_id") == order_id
        ]
        if resource in {"payment_history", "packing_history"}:
            rows.sort(key=lambda row: int(row.get("version", -1)))
        else:
            rows.sort(key=lambda row: str(row.get("txn_id", "")))
        return tuple(rows)
    if resource == "after_sales_history":
        raw = tables.get("after_sales_records", [])
        if not isinstance(raw, list):
            raise ExactJoinError("World after_sales_records table is invalid")
        rows: list[dict[str, Any]] = []
        validated = _after_sales_records({"after_sales_records": raw})
        for wrapper in sorted(
            validated.values(), key=lambda row: str(row["physical_key"])
        ):
            value = wrapper.get("value")
            binding = value.get("binding") if isinstance(value, Mapping) else None
            row_order_id = (
                binding.get("order_id")
                if isinstance(binding, Mapping)
                else value.get("order_id")
                if isinstance(value, Mapping)
                else None
            )
            if row_order_id == order_id:
                rows.append(
                    {
                        "table": wrapper["table"],
                        "key": wrapper["key"],
                        "value": wrapper["value"],
                    }
                )
        return tuple(rows)
    raise ExactJoinError(f"unsupported after-sales read resource {resource!r}")


def _require_order_party(tables: Mapping[str, Any], *, order_id: str, actor_id: str) -> None:
    raw = tables.get("orders", [])
    if not isinstance(raw, list):
        raise ExactJoinError("World orders table is invalid")
    matches = [row for row in raw if isinstance(row, Mapping) and row.get("order_id") == order_id]
    if len(matches) != 1:
        raise ExactJoinError("after-sales read order is not unique")
    if actor_id not in {matches[0].get("buyer_id"), matches[0].get("merchant_id")}:
        raise ExactJoinError("after-sales read actor is not an order party")


def _evidence_records(tables: Mapping[str, Any]) -> dict[str, EvidenceRecord]:
    raw = tables.get("evidence_records", [])
    if not isinstance(raw, list):
        raise ExactJoinError("World evidence_records must be an array")
    versions: dict[str, dict[int, EvidenceRecord]] = {}
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ExactJoinError(f"evidence_records[{index}] is not an object")
        try:
            record = coerce_evidence_record(value)
            canonical = evidence_record_to_dict(record)
        except Exception as exc:
            raise ExactJoinError(f"evidence_records[{index}] failed its strict codec") from exc
        if dict(canonical) != dict(value):
            raise ExactJoinError(f"evidence_records[{index}] is not canonical")
        by_version = versions.setdefault(record.record_id, {})
        if record.version in by_version:
            raise ExactJoinError("World evidence record version is duplicated")
        by_version[record.version] = record
    latest: dict[str, EvidenceRecord] = {}
    for record_id, history in versions.items():
        ordered = sorted(history)
        if ordered != list(range(1, ordered[-1] + 1)):
            raise ExactJoinError("World evidence record version chain has a gap")
        latest[record_id] = history[ordered[-1]]
    return latest


def _referenced_evidence_records(
    intent: Mapping[str, Any],
    *,
    evidence_records: Mapping[str, EvidenceRecord],
    binding: AfterSalesOrderBinding,
    actor_id: str,
) -> tuple[EvidenceRecord, ...]:
    try:
        record_ids = referenced_after_sales_evidence_ids(intent)
    except Exception as exc:
        raise ExactJoinError("after-sales evidence references are invalid") from exc
    selected: list[EvidenceRecord] = []
    required_acl = {binding.owner_id, binding.merchant_id, AFTER_SALES_ENDPOINT}
    operation = _text(intent.get("op"), "after-sales evidence operation")
    for record_id in record_ids:
        record = evidence_records.get(record_id)
        if record is None:
            raise ExactJoinError("after-sales request cites missing World evidence")
        if record.subject_id != binding.order_id:
            raise ExactJoinError("after-sales evidence belongs to another order")
        if record.owner_id not in {binding.owner_id, binding.merchant_id}:
            raise ExactJoinError("after-sales evidence is not owned by an order party")
        if operation == "request_return" and record.owner_id != binding.owner_id:
            raise ExactJoinError("return evidence is not owned by the order owner")
        if operation in {
            "submit_dispute_evidence",
            "respond_to_dispute",
        } and record.owner_id != actor_id:
            raise ExactJoinError("dispute evidence is not owned by the submitting actor")
        if not required_acl.issubset(set(record.read_acl)):
            raise ExactJoinError("after-sales evidence omits required order-party ACL")
        if actor_id != record.owner_id and actor_id not in record.read_acl:
            raise ExactJoinError("after-sales actor cannot read cited evidence")
        if record.trust.get("verified") is not True:
            raise ExactJoinError("after-sales evidence is not marked verified")
        method = record.trust.get("verification_method")
        if not isinstance(method, str) or not method.strip():
            raise ExactJoinError("after-sales evidence has no verification method")
        selected.append(record)
    return tuple(selected)


def _verify_result_evidence_binding(
    *,
    operation: str,
    intent: Mapping[str, Any],
    result: Mapping[str, Any],
    evidence_records: tuple[EvidenceRecord, ...],
) -> None:
    typed = result["typed"]
    if operation == "submit_dispute_evidence":
        if len(evidence_records) != 1 or not isinstance(typed, DisputeEvidence):
            raise ExactJoinError("dispute submission lacks one typed evidence result")
        [record] = evidence_records
        if (
            typed.evidence_id != record.record_id
            or typed.evidence_kind != record.kind
            or dict(typed.facts) != dict(record.facts)
        ):
            raise ExactJoinError("dispute result differs from cited World evidence")
    elif operation == "respond_to_dispute":
        if not isinstance(typed, DisputeResponseRecord):
            raise ExactJoinError("dispute response has the wrong typed result")
        expected_facts = {
            "position": intent["position"],
            "evidence": [
                {
                    "record_id": record.record_id,
                    "record_digest": record.record_digest,
                    "kind": record.kind,
                    "issuer_id": record.issuer_id,
                }
                for record in evidence_records
            ],
        }
        if _deep_json_value(typed.facts, label="dispute response facts") != expected_facts:
            raise ExactJoinError("dispute response differs from cited World evidence")


def _after_sales_records(tables: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = tables.get("after_sales_records", [])
    if not isinstance(raw, list):
        raise ExactJoinError("World after_sales_records must be an array")
    rows: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != {"table", "key", "value"}:
            raise ExactJoinError(f"after_sales_records[{index}] is not a typed wrapper")
        table = item.get("table")
        key = item.get("key")
        value = item.get("value")
        if not isinstance(table, str) or table == "after_sales_policies":
            raise ExactJoinError("generic after-sales row has invalid domain table")
        if not isinstance(key, str) or not isinstance(value, Mapping):
            raise ExactJoinError("generic after-sales row has invalid identity")
        try:
            typed = record_from_wire(table, value)  # type: ignore[arg-type]
            canonical_key = record_key(table, typed)  # type: ignore[arg-type]
            physical = physical_after_sales_record_key(
                AfterSalesWrite(table, canonical_key, typed)  # type: ignore[arg-type]
            )
        except Exception as exc:
            raise ExactJoinError("generic after-sales row failed its strict codec") from exc
        if canonical_key != key or physical in rows:
            raise ExactJoinError("generic after-sales row key is non-canonical or duplicated")
        rows[physical] = {
            "table": table,
            "key": key,
            "value": json.loads(_canonical_text(value)),
            "typed": typed,
            "physical_key": physical,
            "record_digest": record_digest(typed),
        }
    return rows


def _binding_order_id(wrapper: Mapping[str, Any]) -> str:
    typed = wrapper["typed"]
    return _text(getattr(typed, "order_id", None), "binding.order_id")


def _policies(tables: Mapping[str, Any]) -> dict[str, Any]:
    raw = tables.get("after_sales_policies", [])
    if not isinstance(raw, list):
        raise ExactJoinError("World after_sales_policies must be an array")
    latest: dict[str, Any] = {}
    seen: set[tuple[str, int]] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ExactJoinError(f"after_sales_policies[{index}] is invalid")
        try:
            policy = after_sales_policy_from_dict(item)
        except Exception as exc:
            raise ExactJoinError("after-sales policy failed its strict codec") from exc
        identity = (policy.merchant_id, policy.revision)
        if identity in seen:
            raise ExactJoinError("after-sales policy revision is duplicated")
        previous = latest.get(policy.merchant_id)
        if previous is not None and (
            policy.revision != previous.revision + 1
            or policy.previous_digest != previous.policy_digest
        ):
            raise ExactJoinError("after-sales policy chain is not contiguous")
        latest[policy.merchant_id] = policy
        seen.add(identity)
    return latest


def _authority_operations(tables: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = tables.get("authority_operations", [])
    if not isinstance(raw, list):
        raise ExactJoinError("World authority_operations must be an array")
    rows: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ExactJoinError(f"authority_operations[{index}] is invalid")
        row = json.loads(_canonical_text(item))
        key = _text(row.get("operation_key"), "operation_key")
        if key in rows:
            raise ExactJoinError("World authority operation is duplicated")
        rows[key] = row
    return rows


def _unique_commit(
    context: ExactJoinContext,
    *,
    operation: str,
    order_id: str,
    actor_id: str,
    idempotency_key: str,
    request_fingerprint: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in context.world_commits
        if row.get("commit_kind") == "transaction"
        and row.get("authority_action") == "world.apply_after_sales_intent"
        and row.get("operation") == operation
        and row.get("subject_id") == order_id
        and row.get("actor_id") == actor_id
        and row.get("idempotency_key") == idempotency_key
        and row.get("request_fingerprint") == request_fingerprint
    ]
    if len(rows) != 1:
        raise ExactJoinError("after-sales request has no unique World commit")
    return rows[0]


def _verify_service_operation(
    context: ExactJoinContext,
    *,
    operation: str,
    expected_order_id: str | None,
    final_records: Mapping[str, Mapping[str, Any]],
    final_authority: Mapping[str, Mapping[str, Any]],
    policies: Mapping[str, Any] | None = None,
    claimed_commits: set[str],
) -> VerifiedAfterSalesServiceOperation:
    if operation in {"rule_for_filer", "rule_for_respondent", "rule_split"}:
        actor_id = "platform:adjudicator"
        authority_action = "world.apply_after_sales_intent"
        scope = "apply_after_sales_intent"
    elif operation == "complete_ledger_reconciliation":
        actor_id = "platform:accounting"
        authority_action = "world.complete_ledger_reconciliation"
        scope = "complete_ledger_reconciliation"
    else:
        raise ExactJoinError(f"unsupported after-sales service operation {operation!r}")
    matches = [
        row
        for row in context.world_commits
        if row.get("commit_kind") == "transaction"
        and row.get("authority_action") == authority_action
        and row.get("operation") == operation
        and row.get("actor_id") == actor_id
        and (expected_order_id is None or row.get("subject_id") == expected_order_id)
        and row.get("commit_id") not in claimed_commits
    ]
    if len(matches) != 1:
        raise ExactJoinError(f"{operation} has no unique trusted service commit")
    commit = matches[0]
    order_id = _text(commit.get("subject_id"), "service operation order_id")
    key = _text(commit.get("idempotency_key"), "service operation idempotency_key")
    fingerprint = _text(commit.get("request_fingerprint"), "service operation fingerprint")
    authority, result = _verify_commit(
        commit,
        final_records=final_records,
        final_authority=final_authority,
        operation=operation,
        order_id=order_id,
        actor_id=actor_id,
        idempotency_key=key,
        request_fingerprint=fingerprint,
        scope=scope,
    )
    typed = result["typed"]
    if operation.startswith("rule_"):
        if not isinstance(typed, Ruling) or typed.actor_id != actor_id:
            raise ExactJoinError("service ruling result has the wrong typed authority")
        binding = typed.binding
        if policies is None:
            policies = _policies(context.final_tables)
        policy = policies.get(binding.merchant_id)
        if policy is None or policy.policy_digest != binding.policy_digest:
            raise ExactJoinError("service ruling policy binding changed")
        intent = {
            "op": operation,
            "order_id": binding.order_id,
            "dispute_id": typed.dispute_id,
            "rationale": typed.rationale,
        }
        expected_fingerprint = after_sales_intent_fingerprint(
            intent,
            binding=binding,
            policy=policy,
            original_actor=actor_id,
        )
        if fingerprint != expected_fingerprint:
            raise ExactJoinError("service ruling fingerprint changed")
        dispute_versions = [
            row["typed"]
            for row in final_records.values()
            if row["table"] == "dispute_cases"
            and getattr(row["typed"], "dispute_id", None) == typed.dispute_id
        ]
        if not any(
            getattr(row, "record_digest", None) == typed.dispute_digest for row in dispute_versions
        ):
            raise ExactJoinError("service ruling lacks its causal dispute version")
    else:
        if not isinstance(typed, LedgerReconciliationResult):
            raise ExactJoinError("accounting result has the wrong typed schema")
        if typed.actor_id != actor_id or typed.request_fingerprint != fingerprint:
            raise ExactJoinError("accounting result authority changed")
        requests = [
            row["typed"]
            for row in final_records.values()
            if row["table"] == "ledger_reconciliation_requests"
            and isinstance(row["typed"], LedgerReconciliationRequest)
            and row["typed"].request_id == typed.request_id
        ]
        if len(requests) != 1 or requests[0].record_digest != typed.request_digest:
            raise ExactJoinError("accounting result lacks its causal request")
        source_wrappers = [
            row
            for row in final_records.values()
            if row["table"] == "ledger_reconciliation_sources"
            and isinstance(row["typed"], LedgerReconciliationSource)
            and row["typed"].txn_id in set(typed.source_txn_ids)
        ]
        sources = [row["typed"] for row in source_wrappers]
        if len(sources) != len(typed.source_txn_ids) or {row.txn_id for row in sources} != set(
            typed.source_txn_ids
        ):
            raise ExactJoinError("accounting result lacks exact typed ledger sources")
        _verify_ledger_reconciliation_sources(
            context,
            result=typed,
            request=requests[0],
            source_wrappers=source_wrappers,
            commit=commit,
        )
    return VerifiedAfterSalesServiceOperation(
        operation=operation,
        actor_id=actor_id,
        authority_operation=authority,
        result_record=result,
        commit=commit,
    )


def _verify_ledger_reconciliation_sources(
    context: ExactJoinContext,
    *,
    result: LedgerReconciliationResult,
    request: LedgerReconciliationRequest,
    source_wrappers: list[Mapping[str, Any]],
    commit: Mapping[str, Any],
) -> None:
    """Bind a reconciliation result to exact World-owned ledger effects."""

    ledger = _ledger_receipts(context.final_tables)
    relevant = {
        txn_id: receipt
        for txn_id, receipt in ledger.items()
        if str(receipt.order_id) == result.binding.order_id
    }
    if set(result.source_txn_ids) != set(relevant):
        raise ExactJoinError("accounting result does not cover the exact final order ledger")

    sources = {
        row["typed"].txn_id: row["typed"]
        for row in source_wrappers
        if isinstance(row.get("typed"), LedgerReconciliationSource)
    }
    if set(sources) != set(relevant):
        raise ExactJoinError("typed ledger sources do not cover the exact order ledger")

    entries: list[dict[str, Any]] = []
    gross = 0
    refunded = 0
    for txn_id in sorted(relevant):
        receipt = relevant[txn_id]
        source = sources[txn_id]
        try:
            expected = derive_ledger_reconciliation_source(
                receipt,
                binding=result.binding,
                effect=receipt.effect,
                server_tick=source.logical_tick,
            )
        except Exception as exc:
            raise ExactJoinError("ledger Receipt cannot produce its claimed typed source") from exc
        if source != expected:
            raise ExactJoinError("typed ledger source differs from the authoritative Receipt")
        if source.receipt_digest != authoritative_receipt_digest(receipt):
            raise ExactJoinError("typed ledger source Receipt digest changed")
        if source.logical_tick > result.logical_tick:
            raise ExactJoinError("typed ledger source is from the future")
        if source.effect == "charge":
            gross += source.amount
        elif source.effect == "refund":
            refunded += source.amount
        else:  # Defensive even though the strict source codec rejects this.
            raise ExactJoinError("typed ledger source has an unknown effect")
        entries.append(
            {
                "txn_id": source.txn_id,
                "effect": source.effect,
                "qty": source.qty,
                "amount": source.amount,
                "currency": source.currency,
                "source_digest": source.source_digest,
            }
        )

    expected_source_digest = _sha256_json(entries)
    expected_result_fingerprint = _sha256_json(
        {
            "schema_id": AFTER_SALES_INTENT_SCHEMA,
            "op": "complete_ledger_reconciliation",
            "actor_id": result.actor_id,
            "request_digest": request.record_digest,
            "source": entries,
        }
    )
    expected_totals = {
        "gross_amount": gross,
        "refund_amount": refunded,
        "net_amount": gross - refunded,
        "source_digest": expected_source_digest,
        "request_fingerprint": expected_result_fingerprint,
    }
    if any(getattr(result, name) != value for name, value in expected_totals.items()):
        raise ExactJoinError("accounting result totals or source binding changed")

    writes = commit.get("table_writes")
    if not isinstance(writes, list):
        raise ExactJoinError("accounting commit has no table writes")
    expected_wrappers = {
        row["physical_key"]: {
            "table": row["table"],
            "key": row["key"],
            "value": row["value"],
        }
        for row in source_wrappers
    }
    source_writes = {
        row.get("key"): row.get("after")
        for row in writes
        if isinstance(row, Mapping)
        and row.get("table") == "after_sales_records"
        and isinstance(row.get("after"), Mapping)
        and row["after"].get("table") == "ledger_reconciliation_sources"
        and row.get("op") == "create"
        and row.get("before") is None
    }
    if source_writes != expected_wrappers:
        raise ExactJoinError("accounting commit did not atomically create its exact typed sources")


def _ledger_receipts(tables: Mapping[str, Any]) -> dict[str, Receipt]:
    raw = tables.get("ledger", [])
    if not isinstance(raw, list):
        raise ExactJoinError("World ledger must be an array")
    expected_fields = {
        "txn_id",
        "ts",
        "order_id",
        "buyer_id",
        "merchant_id",
        "sku_id",
        "qty",
        "price",
        "idempotency_key",
        "effect",
    }
    rows: dict[str, Receipt] = {}
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise ExactJoinError(f"ledger[{index}] is not a canonical Receipt")
        price = value.get("price")
        if not isinstance(price, Mapping) or set(price) != {"amount", "currency"}:
            raise ExactJoinError(f"ledger[{index}] has an invalid Money value")
        amount = price.get("amount")
        if not isinstance(amount, str):
            raise ExactJoinError(f"ledger[{index}] price amount must be exact text")
        try:
            decimal_amount = Decimal(amount)
        except (InvalidOperation, ValueError) as exc:
            raise ExactJoinError(f"ledger[{index}] price amount is invalid") from exc
        if not decimal_amount.is_finite() or decimal_amount < 0:
            raise ExactJoinError(f"ledger[{index}] price amount is invalid")
        qty = value.get("qty")
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            raise ExactJoinError(f"ledger[{index}] quantity is invalid")
        effect = value.get("effect")
        if effect not in {"charge", "refund"}:
            raise ExactJoinError(f"ledger[{index}] effect is invalid")
        txn_id = _text(value.get("txn_id"), f"ledger[{index}].txn_id")
        if txn_id in rows:
            raise ExactJoinError("World ledger transaction id is duplicated")
        try:
            receipt = Receipt(
                txn_id=TxnId(txn_id),
                ts=_text(value.get("ts"), f"ledger[{index}].ts"),
                order_id=OrderId(_text(value.get("order_id"), f"ledger[{index}].order_id")),
                buyer_id=AgentId(_text(value.get("buyer_id"), f"ledger[{index}].buyer_id")),
                merchant_id=AgentId(
                    _text(value.get("merchant_id"), f"ledger[{index}].merchant_id")
                ),
                sku_id=SkuId(_text(value.get("sku_id"), f"ledger[{index}].sku_id")),
                qty=qty,
                price=Money(
                    decimal_amount,
                    _text(price.get("currency"), f"ledger[{index}].currency"),
                ),
                idempotency_key=_text(
                    value.get("idempotency_key"),
                    f"ledger[{index}].idempotency_key",
                ),
                effect=effect,
            )
            authoritative_receipt_digest(receipt)
        except Exception as exc:
            raise ExactJoinError(f"ledger[{index}] failed its strict codec") from exc
        rows[txn_id] = receipt
    return rows


def _verify_commit(
    commit: Mapping[str, Any],
    *,
    final_records: Mapping[str, Mapping[str, Any]],
    final_authority: Mapping[str, Mapping[str, Any]],
    operation: str,
    order_id: str,
    actor_id: str,
    idempotency_key: str,
    request_fingerprint: str,
    scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    invariants = commit.get("invariants_held")
    if not isinstance(invariants, list) or not _REQUIRED_COMMIT_INVARIANTS.issubset(
        set(invariants)
    ):
        raise ExactJoinError("after-sales commit is missing authority invariants")
    raw_writes = commit.get("table_writes")
    if not isinstance(raw_writes, list) or not raw_writes:
        raise ExactJoinError("after-sales commit has no table writes")
    writes = [json.loads(_canonical_text(row)) for row in raw_writes if isinstance(row, Mapping)]
    if len(writes) != len(raw_writes):
        raise ExactJoinError("after-sales commit contains an invalid table write")

    clocks = [row for row in writes if row.get("table") == "logical_time"]
    if len(clocks) != 1 or (
        clocks[0].get("op") != "update"
        or isinstance(clocks[0].get("before"), bool)
        or not isinstance(clocks[0].get("before"), int)
        or clocks[0].get("after") != clocks[0].get("before") + 1
    ):
        raise ExactJoinError("after-sales commit has an invalid World clock advance")

    authority_writes = [
        row
        for row in writes
        if row.get("table") == "authority_operations"
        and row.get("op") == "create"
        and row.get("before") is None
        and isinstance(row.get("after"), Mapping)
    ]
    if len(authority_writes) != 1:
        raise ExactJoinError("after-sales commit has no unique authority operation")
    authority = authority_writes[0]["after"]
    expected = {
        "scope": scope,
        "actor_id": actor_id,
        "idempotency_key": idempotency_key,
        "request_fingerprint": request_fingerprint,
        "outcome_table": "after_sales_records",
    }
    if any(authority.get(name) != value for name, value in expected.items()):
        raise ExactJoinError("after-sales authority operation changed")
    operation_key = _text(authority.get("operation_key"), "operation_key")
    if final_authority.get(operation_key) != authority:
        raise ExactJoinError("after-sales authority operation is absent from final World")

    outcome_key = _text(authority.get("outcome_key"), "outcome_key")
    result = final_records.get(outcome_key)
    if result is None:
        raise ExactJoinError("after-sales authority outcome is absent from final World")
    outcome_writes = [
        row
        for row in writes
        if row.get("table") == "after_sales_records"
        and row.get("key") == outcome_key
        and row.get("op") == "create"
        and row.get("before") is None
    ]
    expected_wrapper = {
        "table": result["table"],
        "key": result["key"],
        "value": result["value"],
    }
    if len(outcome_writes) != 1 or outcome_writes[0].get("after") != expected_wrapper:
        raise ExactJoinError("after-sales World outcome write changed")
    if _binding_order_id_from_record(result) != order_id:
        raise ExactJoinError("after-sales result belongs to another order")

    for row in writes:
        if row.get("table") == "payment_states" and isinstance(row.get("after"), Mapping):
            try:
                payment_state_from_dict(row["after"])
            except Exception as exc:
                raise ExactJoinError("commit payment state is not canonical") from exc
        if row.get("table") == "packing_records" and isinstance(row.get("after"), Mapping):
            try:
                packing_record_from_dict(row["after"])
            except Exception as exc:
                raise ExactJoinError("commit packing record is not canonical") from exc
    if commit.get("operation") != operation or commit.get("subject_id") != order_id:
        raise ExactJoinError("after-sales commit identity changed")
    return authority, dict(result)


def _binding_order_id_from_record(wrapper: Mapping[str, Any]) -> str:
    typed = wrapper["typed"]
    binding = getattr(typed, "binding", None)
    order_id = getattr(binding, "order_id", None)
    if order_id is None:
        order_id = getattr(typed, "order_id", None)
    return _text(order_id, "after-sales result order_id")


def _verify_response(
    exchange: LinkedPlatformExchange,
    *,
    operation: str,
    order_id: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        row
        for row in exchange.responses
        if (row.get("action") or {}).get("kind") == AFTER_SALES_RESPONSE_KIND
    ]
    if len(rows) != 1 or len(exchange.responses) != 1:
        raise ExactJoinError("accepted after-sales request needs one safe response")
    response = rows[0]
    request = exchange.request
    if (
        response.get("from") != AFTER_SALES_ENDPOINT
        or response.get("to") != request.get("from")
        or response.get("in_reply_to") != request.get("msg_id")
        or response.get("idempotency_key") != request.get("idempotency_key")
    ):
        raise ExactJoinError("after-sales response envelope identity changed")
    action = response.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ExactJoinError("after-sales response payload is missing")
    expected = {
        "operation": operation,
        "order_id": order_id,
        "result_table": result["table"],
        "result_key": result["key"],
        "result_digest": result["record_digest"],
        "references": _stable_result_references(result),
    }
    if set(payload) != {*expected, "disposition"} or any(
        payload.get(name) != value for name, value in expected.items()
    ):
        raise ExactJoinError("after-sales response changed the World result reference")
    if payload.get("disposition") not in {"committed", "idempotent"}:
        raise ExactJoinError("after-sales response disposition is invalid")
    return json.loads(_canonical_text(response))


def _stable_result_references(result: Mapping[str, Any]) -> dict[str, str]:
    table = _text(result.get("table"), "result table")
    value = result.get("value")
    if not isinstance(value, Mapping):
        raise ExactJoinError("after-sales result value is missing")
    try:
        fields = _REFERENCE_FIELDS_BY_TABLE[table]
    except KeyError as exc:
        raise ExactJoinError(
            f"after-sales result table {table!r} has no reference contract"
        ) from exc
    references: dict[str, str] = {}
    for field in fields:
        references[field] = _text(value.get(field), f"result reference {field}")
    return references


def _validate_payment_and_packing_histories(tables: Mapping[str, Any]) -> None:
    raw_payments = tables.get("payment_states", [])
    raw_packings = tables.get("packing_records", [])
    if not isinstance(raw_payments, list) or not isinstance(raw_packings, list):
        raise ExactJoinError("payment and packing World tables must be arrays")
    payments: dict[str, list[Any]] = {}
    packings: dict[str, list[Any]] = {}
    try:
        for row in raw_payments:
            if not isinstance(row, Mapping):
                raise TypeError("payment row is not an object")
            typed = payment_state_from_dict(row)
            payments.setdefault(typed.order_id, []).append(typed)
        for row in raw_packings:
            if not isinstance(row, Mapping):
                raise TypeError("packing row is not an object")
            typed = packing_record_from_dict(row)
            packings.setdefault(typed.order_id, []).append(typed)
        for history in payments.values():
            history.sort(key=lambda row: row.version)
            previous = None
            for row in history:
                validate_payment_transition(previous, row, server_tick=row.logical_tick)
                previous = row
        for history in packings.values():
            history.sort(key=lambda row: row.version)
            previous = None
            for row in history:
                validate_packing_transition(previous, row, server_tick=row.logical_tick)
                previous = row
    except Exception as exc:
        raise ExactJoinError("payment or packing causal history is invalid") from exc


def _text_sequence(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ExactJoinError(f"{label} must be an array")
    return tuple(_text(item, label) for item in value)


def _commit_sequence(commit: Mapping[str, Any]) -> int:
    value = commit.get("sequence")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExactJoinError("after-sales World commit sequence is invalid")
    return value


def _optional_text(value: Any, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _deep_json_value(value: Any, *, label: str) -> Any:
    """Thaw immutable protocol values without weakening exact comparison.

    World codecs freeze nested arrays as tuples and objects as immutable
    mappings.  Evidence contracts compare their JSON meaning, but still reject
    non-JSON values, non-text object keys, and every field-level difference.
    """

    if isinstance(value, Mapping):
        thawed: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExactJoinError(f"{label} contains a non-text object key")
            thawed[key] = _deep_json_value(item, label=label)
        return thawed
    if isinstance(value, (tuple, list)):
        return [_deep_json_value(item, label=label) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ExactJoinError(f"{label} contains a non-JSON value")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExactJoinError(f"{label} must be non-empty text")
    return value


def _canonical_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_text(value).encode("utf-8")).hexdigest()


DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        AFTER_SALES_EVIDENCE_CONTRACT,
        verify_after_sales_evidence_contract,
    )
)


__all__ = [
    "ACTION_TO_OPERATION",
    "ACTION_TO_READ_RESOURCE",
    "AFTER_SALES_ENDPOINT",
    "AFTER_SALES_EVIDENCE_CONTRACT",
    "AFTER_SALES_READ_RESPONSE_KIND",
    "AFTER_SALES_RESPONSE_KIND",
    "VerifiedAfterSalesEvidence",
    "VerifiedAfterSalesRead",
    "VerifiedAfterSalesRequest",
    "VerifiedAfterSalesServiceOperation",
    "VerifiedRejectedAfterSalesRequest",
    "verify_after_sales_evidence_contract",
]
