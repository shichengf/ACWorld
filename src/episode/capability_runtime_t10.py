"""ACWorld T10 protocol integrity benchmark tasks.

Every task executes through the real Runtime, Platform, and World stack.  A
trusted Runtime trigger may request event issuance or advance market time, but
it cannot supply order state, revision, parties, effect data, or a receipt.
Platform derives and persists the event from World.  The evaluated actor then
submits a typed process or reject decision.  A process decision invokes only a
registered World operation and atomically binds its receipt to the real ledger
or shipment outcome.

The four task structures are duplicate callbacks, expired authorization,
unsafe lifecycle ordering, and cross-order isolation.  Each structure contains
at least one legitimate registered operation.  Invalid callbacks are durable
rejections, never scenario-local state transitions or fabricated evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

from agents.business_decision import LLM_BUSINESS_DECISION_SCHEMA_V1
from agents.inference import BusinessDecisionResponseV1, InferenceChannel
from episode.capability_benchmark import TASK_REGISTRY_V2, TaskDefinitionV2
from episode.capability_runtime import (
    RuntimeBenchmarkIntegrityError,
    RuntimeEvidenceBundleV2,
    RuntimeEvidenceError,
    RuntimeMutationV2,
    RuntimeRubricCheckV2,
    RuntimeTaskBundleV2,
    RuntimeTaskScoreV3,
    canonical_sha256,
    require_runtime_benchmark_integrity_v2,
    score_checks,
)
from episode.capability_runtime_fixture import require_frozen_scenario_fixture_v2
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec
from runtime.authority_operation_evidence import (
    MARKET_CLOCK_EVIDENCE_CONTRACT,
    VerifiedMarketClockEvidence,
)
from runtime.commit_claims import verify_exact_transaction_commit_claims
from runtime.exact_join import (
    PROTOCOL_EVENT_EVIDENCE_CONTRACT,
    VerifiedProtocolEventEvidence,
)


T10_RUNTIME_SCHEMA_V2 = "cwe.runtime-task.t10.v2"
T10_RUNTIME_DATA_SCHEMA_V2 = "cwe.runtime-task-data.t10.v2"
_MARKET_ID = "market:benchmark-t10"
_PROCESS_KINDS = frozenset({"payment.settle", "fulfillment.dispatch", "payment.refund"})
T10_RUNTIME_READY_TASK_IDS = tuple(
    task_id for task_id, definition in TASK_REGISTRY_V2.items() if definition.family.value == "T10"
)
T10_RUNTIME_PENDING_TASK_IDS: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _OrderPlanT10:
    order_id: str
    buyer_id: str
    merchant_id: str
    sku_id: str
    initial_state: str


@dataclass(frozen=True, slots=True)
class _EventPlanT10:
    event_id: str
    order_id: str
    recipient_id: str
    event_kind: str
    expected_decision: str
    ttl_ticks: int = 32
    reference_authorization_id: str | None = None
    authorization_ttl_ticks: int | None = None

    def __post_init__(self) -> None:
        if self.event_kind not in _PROCESS_KINDS:
            raise ValueError(f"unregistered T10 event kind {self.event_kind!r}")
        if self.expected_decision not in {"process", "reject"}:
            raise ValueError("T10 decision must be process or reject")
        if (self.reference_authorization_id is None) != (self.authorization_ttl_ticks is None):
            raise ValueError("T10 certificate events need both authorization id and TTL")
        if self.authorization_ttl_ticks is not None and (
            isinstance(self.authorization_ttl_ticks, bool)
            or not isinstance(self.authorization_ttl_ticks, int)
            or self.authorization_ttl_ticks <= 0
        ):
            raise ValueError("T10 authorization TTL must be positive")


@dataclass(frozen=True, slots=True)
class _CaseT10:
    task: TaskDefinitionV2
    axis_name: str
    axis_value: int
    orders: tuple[_OrderPlanT10, ...]
    events: tuple[_EventPlanT10, ...]
    evaluated_actor_id: str
    active_order_ids: tuple[str, ...] = ()
    clock_target: int | None = None
    clock_after_event_id: str | None = None

    @property
    def process_events(self) -> tuple[_EventPlanT10, ...]:
        return tuple(event for event in self.events if event.expected_decision == "process")

    @property
    def reject_events(self) -> tuple[_EventPlanT10, ...]:
        return tuple(event for event in self.events if event.expected_decision == "reject")

    @property
    def authorized_order_ids(self) -> tuple[str, ...]:
        if self.active_order_ids:
            return self.active_order_ids
        return tuple(dict.fromkeys(event.order_id for event in self.process_events))

    @property
    def content(self) -> dict[str, Any]:
        return {
            "schema_version": T10_RUNTIME_DATA_SCHEMA_V2,
            "definition": self.task.to_dict(),
            "difficulty": {self.axis_name: self.axis_value},
            "runtime_ready": True,
            "orders": [
                {
                    "order_id": row.order_id,
                    "buyer_id": row.buyer_id,
                    "merchant_id": row.merchant_id,
                    "sku_id": row.sku_id,
                    "initial_state": row.initial_state,
                }
                for row in self.orders
            ],
            "events": [
                {
                    "event_id": row.event_id,
                    "order_id": row.order_id,
                    "recipient_id": row.recipient_id,
                    "event_kind": row.event_kind,
                    "expected_decision": row.expected_decision,
                    "ttl_ticks": row.ttl_ticks,
                    "reference_kind": (
                        "certificate" if row.reference_authorization_id is not None else "operation"
                    ),
                    "reference_authorization_id": row.reference_authorization_id,
                    "authorization_ttl_ticks": row.authorization_ttl_ticks,
                }
                for row in self.events
            ],
            "authorized_order_ids": list(self.authorized_order_ids),
            "authority": {
                "issuance": "runtime trigger -> Platform derivation -> World event",
                "decision": "actor -> Platform validation -> World receipt",
                "effect": "registered World operation and committed outcome digest",
                "score": "read-only deterministic Python over verified artifacts",
            },
        }


def _axis(definition: TaskDefinitionV2) -> tuple[str, int]:
    axes = tuple(
        (name, value) for name, value in definition.difficulty_factors if name != "difficulty_level"
    )
    if len(axes) != 1:
        raise ValueError(f"{definition.task_id}: T10 needs one semantic axis")
    name, value = axes[0]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{definition.task_id}: invalid T10 difficulty")
    return name, value


def _order(
    definition: TaskDefinitionV2,
    ordinal: int,
    *,
    buyer_id: str,
    merchant_id: str,
    initial_state: str,
) -> _OrderPlanT10:
    stem = definition.task_id.casefold()
    return _OrderPlanT10(
        order_id=f"order:{stem}:{ordinal:02d}",
        buyer_id=buyer_id,
        merchant_id=merchant_id,
        sku_id=f"{merchant_id}:sku:{stem}:{ordinal:02d}",
        initial_state=initial_state,
    )


def _event(
    definition: TaskDefinitionV2,
    order: _OrderPlanT10,
    label: str,
    *,
    recipient_id: str,
    event_kind: str,
    decision: str,
    ttl_ticks: int = 32,
    reference_authorization_id: str | None = None,
    authorization_ttl_ticks: int | None = None,
) -> _EventPlanT10:
    return _EventPlanT10(
        event_id=f"event:{definition.task_id.casefold()}:{label}",
        order_id=order.order_id,
        recipient_id=recipient_id,
        event_kind=event_kind,
        expected_decision=decision,
        ttl_ticks=ttl_ticks,
        reference_authorization_id=reference_authorization_id,
        authorization_ttl_ticks=authorization_ttl_ticks,
    )


def _authorization_id(
    definition: TaskDefinitionV2,
    label: str,
) -> str:
    return f"match-authorization:{definition.task_id.casefold()}:{label}"


def _build_case_for_t10(task_id: str) -> _CaseT10:
    definition = TASK_REGISTRY_V2[task_id]
    if definition.family.value != "T10":
        raise ValueError(f"{task_id} is not a T10 task")
    axis_name, count = _axis(definition)
    capability = definition.capability_id
    buyer = "buyer:t10-benchmark"
    merchant = "merchant:t10-benchmark"

    if capability == "t10.buyer_duplicate_payment":
        order = _order(
            definition,
            1,
            buyer_id=buyer,
            merchant_id=merchant,
            initial_state="proposed",
        )
        events = (
            _event(
                definition,
                order,
                "payment-valid",
                recipient_id=buyer,
                event_kind="payment.settle",
                decision="process",
            ),
            *(
                _event(
                    definition,
                    order,
                    f"payment-repeat-{index:02d}",
                    recipient_id=buyer,
                    event_kind="payment.settle",
                    decision="reject",
                )
                for index in range(1, count + 1)
            ),
        )
        return _CaseT10(definition, axis_name, count, (order,), tuple(events), buyer)

    if capability == "t10.buyer_stale_certificate":
        order = _order(
            definition,
            1,
            buyer_id=buyer,
            merchant_id=merchant,
            initial_state="proposed",
        )
        stale = _event(
            definition,
            order,
            "expired-authorization",
            recipient_id=buyer,
            event_kind="payment.settle",
            decision="reject",
            ttl_ticks=count + 33,
            reference_authorization_id=_authorization_id(definition, "stale"),
            authorization_ttl_ticks=1,
        )
        current = _event(
            definition,
            order,
            "current-authorization",
            recipient_id=buyer,
            event_kind="payment.settle",
            decision="process",
            ttl_ticks=count + 33,
            reference_authorization_id=_authorization_id(definition, "current"),
            authorization_ttl_ticks=count + 33,
        )
        return _CaseT10(
            definition,
            axis_name,
            count,
            (order,),
            (stale, current),
            buyer,
            clock_target=count + 1,
            clock_after_event_id=stale.event_id,
        )

    if capability == "t10.buyer_out_of_order":
        order = _order(
            definition,
            1,
            buyer_id=buyer,
            merchant_id=merchant,
            initial_state="proposed",
        )
        inversions = tuple(
            _event(
                definition,
                order,
                f"premature-dispatch-{index:02d}",
                recipient_id=buyer,
                event_kind="fulfillment.dispatch",
                decision="reject",
            )
            for index in range(1, count + 1)
        )
        valid = _event(
            definition,
            order,
            "payment-valid",
            recipient_id=buyer,
            event_kind="payment.settle",
            decision="process",
        )
        return _CaseT10(definition, axis_name, count, (order,), (*inversions, valid), buyer)

    if capability == "t10.buyer_cross_order_replay":
        orders = tuple(
            _order(
                definition,
                index,
                buyer_id=buyer,
                merchant_id=f"merchant:t10-{index:02d}",
                initial_state="proposed",
            )
            for index in range(1, count + 1)
        )
        active = orders[0]
        events: list[_EventPlanT10] = [
            _event(
                definition,
                order,
                f"foreign-order-{index:02d}",
                recipient_id=buyer,
                event_kind="payment.settle",
                decision="reject",
                reference_authorization_id=_authorization_id(
                    definition, f"foreign-order-{index:02d}"
                ),
                authorization_ttl_ticks=32,
            )
            for index, order in enumerate(orders[1:], start=2)
        ]
        events.append(
            _event(
                definition,
                active,
                "active-order",
                recipient_id=buyer,
                event_kind="payment.settle",
                decision="process",
                reference_authorization_id=_authorization_id(definition, "active-order"),
                authorization_ttl_ticks=32,
            )
        )
        return _CaseT10(
            definition,
            axis_name,
            count,
            orders,
            tuple(events),
            buyer,
            active_order_ids=(active.order_id,),
        )

    if capability in {
        "t10.merchant_duplicate_fulfillment",
        "t10.merchant_duplicate_refund",
    }:
        order = _order(
            definition,
            1,
            buyer_id=buyer,
            merchant_id=merchant,
            initial_state="settled",
        )
        event_kind = (
            "fulfillment.dispatch" if capability.endswith("fulfillment") else "payment.refund"
        )
        label = "dispatch" if event_kind == "fulfillment.dispatch" else "refund"
        events = (
            _event(
                definition,
                order,
                f"{label}-valid",
                recipient_id=merchant,
                event_kind=event_kind,
                decision="process",
            ),
            *(
                _event(
                    definition,
                    order,
                    f"{label}-repeat-{index:02d}",
                    recipient_id=merchant,
                    event_kind=event_kind,
                    decision="reject",
                )
                for index in range(1, count + 1)
            ),
        )
        return _CaseT10(definition, axis_name, count, (order,), tuple(events), merchant)

    if capability == "t10.merchant_stale_certificate":
        order = _order(
            definition,
            1,
            buyer_id=buyer,
            merchant_id=merchant,
            initial_state="settled",
        )
        stale = _event(
            definition,
            order,
            "expired-authorization",
            recipient_id=merchant,
            event_kind="fulfillment.dispatch",
            decision="reject",
            ttl_ticks=count + 33,
            reference_authorization_id=_authorization_id(definition, "stale"),
            authorization_ttl_ticks=1,
        )
        current = _event(
            definition,
            order,
            "current-authorization",
            recipient_id=merchant,
            event_kind="fulfillment.dispatch",
            decision="process",
            ttl_ticks=count + 34,
            reference_authorization_id=_authorization_id(definition, "current"),
            authorization_ttl_ticks=count + 34,
        )
        return _CaseT10(
            definition,
            axis_name,
            count,
            (order,),
            (stale, current),
            merchant,
            clock_target=count + 2,
            clock_after_event_id=stale.event_id,
        )

    if capability == "t10.merchant_out_of_order":
        order = _order(
            definition,
            1,
            buyer_id=buyer,
            merchant_id=merchant,
            initial_state="settled",
        )
        inversions = tuple(
            _event(
                definition,
                order,
                f"misrouted-payment-{index:02d}",
                recipient_id=merchant,
                event_kind="payment.settle",
                decision="reject",
            )
            for index in range(1, count + 1)
        )
        valid = _event(
            definition,
            order,
            "dispatch-valid",
            recipient_id=merchant,
            event_kind="fulfillment.dispatch",
            decision="process",
        )
        return _CaseT10(definition, axis_name, count, (order,), (*inversions, valid), merchant)

    if capability == "t10.merchant_cross_order_isolation":
        orders = tuple(
            _order(
                definition,
                index,
                buyer_id=f"buyer:t10-{index:02d}",
                merchant_id=merchant,
                initial_state="settled",
            )
            for index in range(1, count + 1)
        )
        active = orders[0]
        events = [
            _event(
                definition,
                order,
                f"foreign-order-{index:02d}",
                recipient_id=merchant,
                event_kind="fulfillment.dispatch",
                decision="reject",
                reference_authorization_id=_authorization_id(
                    definition, f"foreign-order-{index:02d}"
                ),
                authorization_ttl_ticks=32,
            )
            for index, order in enumerate(orders[1:], start=2)
        ]
        events.append(
            _event(
                definition,
                active,
                "active-order",
                recipient_id=merchant,
                event_kind="fulfillment.dispatch",
                decision="process",
                reference_authorization_id=_authorization_id(definition, "active-order"),
                authorization_ttl_ticks=32,
            )
        )
        return _CaseT10(
            definition,
            axis_name,
            count,
            orders,
            tuple(events),
            merchant,
            active_order_ids=(active.order_id,),
        )

    raise ValueError(f"unsupported T10 capability {capability!r}")


_LIFECYCLE_PROCESS_STATES: Mapping[str, frozenset[str]] = {
    "payment.settle": frozenset({"proposed", "accepted"}),
    "fulfillment.dispatch": frozenset({"settled", "partially_settled"}),
    "payment.refund": frozenset({"settled", "partially_settled", "dispatched", "returned"}),
}
_LIFECYCLE_NEXT_STATE: Mapping[str, str] = {
    "payment.settle": "settled",
    "fulfillment.dispatch": "dispatched",
    "payment.refund": "refunded",
}


def _validate_t10_fixture(case: _CaseT10) -> _CaseT10:
    """Prove every frozen expected decision from public protocol conditions."""

    if not case.orders or not case.events:
        raise ValueError(f"{case.task.task_id}: T10 fixture cannot be empty")
    order_by_id = {order.order_id: order for order in case.orders}
    if len(order_by_id) != len(case.orders):
        raise ValueError(f"{case.task.task_id}: duplicate order lineage")
    event_ids = [event.event_id for event in case.events]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError(f"{case.task.task_id}: duplicate protocol event id")

    active = case.active_order_ids
    if len(set(active)) != len(active) or not set(active) <= set(order_by_id):
        raise ValueError(f"{case.task.task_id}: active-order scope is invalid")
    if (case.clock_target is None) != (case.clock_after_event_id is None):
        raise ValueError(f"{case.task.task_id}: stale-event clock binding is incomplete")
    if case.clock_after_event_id is not None and case.clock_after_event_id not in event_ids:
        raise ValueError(f"{case.task.task_id}: stale-event clock has no event lineage")

    authorization_ids = [
        event.reference_authorization_id
        for event in case.events
        if event.reference_authorization_id is not None
    ]
    if len(set(authorization_ids)) != len(authorization_ids):
        raise ValueError(f"{case.task.task_id}: duplicate certificate lineage")

    states = {order.order_id: order.initial_state for order in case.orders}
    processed_business_refs: set[tuple[str, str]] = set()
    observed_invalid_reasons: set[str] = set()
    for event in case.events:
        order = order_by_id.get(event.order_id)
        if order is None:
            raise ValueError(f"{case.task.task_id}: event crosses unknown order lineage")
        if event.recipient_id != case.evaluated_actor_id:
            raise ValueError(f"{case.task.task_id}: event crosses evaluated actor lineage")
        if event.ttl_ticks <= 0:
            raise ValueError(f"{case.task.task_id}: event TTL must be positive")

        business_ref = (event.order_id, event.event_kind)
        is_duplicate = business_ref in processed_business_refs
        lifecycle_valid = states[event.order_id] in _LIFECYCLE_PROCESS_STATES[event.event_kind]
        scope_valid = not active or event.order_id in active
        reference_current = not (
            case.clock_target is not None
            and event.authorization_ttl_ticks is not None
            and event.authorization_ttl_ticks <= case.clock_target
        )
        reasons = {
            *(("duplicate",) if is_duplicate else ()),
            *(("lifecycle",) if not lifecycle_valid else ()),
            *(("active_order_scope",) if not scope_valid else ()),
            *(("stale",) if not reference_current else ()),
        }
        should_process = not reasons
        if (event.expected_decision == "process") != should_process:
            raise ValueError(
                f"{case.task.task_id}: {event.event_id} expected decision is not "
                "implied by duplicate, stale, scope, and lifecycle conditions"
            )
        if should_process:
            processed_business_refs.add(business_ref)
            states[event.order_id] = _LIFECYCLE_NEXT_STATE[event.event_kind]
        else:
            observed_invalid_reasons.update(reasons)

    process_ids = {event.order_id for event in case.process_events}
    if active and process_ids != set(active):
        raise ValueError(f"{case.task.task_id}: process events escape active-order scope")

    capability = case.task.capability_id
    required_reason = (
        "duplicate"
        if "duplicate" in capability
        else "stale"
        if "stale" in capability
        else "active_order_scope"
        if "cross_order" in capability
        else "lifecycle"
        if "out_of_order" in capability
        else None
    )
    if required_reason is None or required_reason not in observed_invalid_reasons:
        raise ValueError(f"{case.task.task_id}: fixture does not exercise its named risk")
    return case


@lru_cache(maxsize=None)
def _case_for_t10(task_id: str) -> _CaseT10:
    return _validate_t10_fixture(_build_case_for_t10(task_id))


def _listing_rows(case: _CaseT10) -> list[dict[str, Any]]:
    return [
        {
            "sku_id": order.sku_id,
            "product_id": f"product:{order.sku_id}",
            "merchant_id": order.merchant_id,
            "category": "benchmark-t10",
            "name": f"T10 protocol item {index:02d}",
            "list_price": "10.00",
            "inventory": 4,
            "qty_reserved": 0,
            "attributes": {
                "task_family": "T10",
                "protocol_integrity_fixture": True,
            },
        }
        for index, order in enumerate(case.orders, start=1)
    ]


def _order_rows(case: _CaseT10) -> list[dict[str, Any]]:
    return [
        {
            "order_id": order.order_id,
            "buyer_id": order.buyer_id,
            "merchant_id": order.merchant_id,
            "sku_id": order.sku_id,
            "qty": 1,
            "agreed_price": "10.00",
            # Paid fixtures are deliberately seeded before payment and then
            # advanced by ``order_settlement_setup`` through World.settle_order.
            "state": ("accepted" if order.initial_state == "settled" else "proposed"),
        }
        for order in case.orders
    ]


def _settlement_setup_rows(case: _CaseT10) -> list[dict[str, Any]]:
    return [
        {
            "order_id": order.order_id,
            "txn_id": f"seed:txn:{order.order_id}",
            "idempotency_key": f"seed:{order.order_id}",
        }
        for order in case.orders
        if order.initial_state == "settled"
    ]


def _mandate_id(case: _CaseT10, buyer_id: str) -> str:
    return f"{case.task.task_id}:{buyer_id}"


def _authorization_rows(case: _CaseT10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in case.events:
        if event.reference_authorization_id is None:
            continue
        order = next(row for row in case.orders if row.order_id == event.order_id)
        assert event.authorization_ttl_ticks is not None
        rows.append(
            {
                "authorization_id": event.reference_authorization_id,
                "order_id": order.order_id,
                "mandate_id": _mandate_id(case, order.buyer_id),
                "ttl_ticks": event.authorization_ttl_ticks,
            }
        )
    return rows


def _issue_envelope(case: _CaseT10, event: _EventPlanT10) -> dict[str, Any]:
    key = f"issue:{event.event_id}"
    payload: dict[str, Any] = {
        "market_id": _MARKET_ID,
        "stream_id": f"stream:{case.task.task_id}:{event.order_id}",
        "order_id": event.order_id,
        "recipient_id": event.recipient_id,
        "event_id": event.event_id,
        "event_kind": event.event_kind,
        "ttl_ticks": event.ttl_ticks,
    }
    if event.reference_authorization_id is not None:
        payload.update(
            {
                "reference_kind": "certificate",
                "reference_authorization_id": event.reference_authorization_id,
            }
        )
    return {
        "msg_id": key,
        "ts": "2026-07-16T12:00:00Z",
        "from": "runtime:events",
        "to": "platform:events",
        "idempotency_key": key,
        "action": {
            "kind": "platform.issue_protocol_event",
            "payload": payload,
        },
    }


def _clock_envelope(case: _CaseT10) -> dict[str, Any]:
    assert case.clock_target is not None
    key = f"clock:{case.task.task_id.casefold()}:{case.clock_target}"
    return {
        "msg_id": key,
        "ts": "2026-07-16T12:00:01Z",
        "from": "runtime:clock",
        "to": "platform:events",
        "idempotency_key": key,
        "action": {
            "kind": "platform.advance_market_clock",
            "payload": {"to_tick": case.clock_target},
        },
    }


def _initial_events(case: _CaseT10) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for event in case.events:
        rows.append(_issue_envelope(case, event))
        if event.event_id == case.clock_after_event_id:
            rows.append(_clock_envelope(case))
    return tuple(rows)


def _public_task_context(case: _CaseT10) -> dict[str, Any]:
    """Expose protocol decision phases without revealing the right decision."""

    return {
        "schema_version": T10_RUNTIME_SCHEMA_V2,
        "task_id": case.task.task_id,
        "capability": case.task.capability_id,
        "execution_contract": {
            "schema_version": "cwe.public-task-execution.v1",
            "phases": [
                {
                    "phase_id": "protocol_event_decision",
                    "match": {
                        "actor_roles": ["buyer", "merchant"],
                        "inbound_action_kinds": ["platform.deliver_protocol_event"],
                        "inbound_senders": ["platform:events"],
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "commerce.process_protocol_event",
                            "destination": "platform:events",
                        },
                        {
                            "action_kind": "commerce.reject_protocol_event",
                            "destination": "platform:events",
                        },
                    ],
                    "world_reads": "skill_scoped",
                    "finish": "forbid",
                },
                {
                    "phase_id": "protocol_event_receipt",
                    "match": {
                        "actor_roles": ["buyer", "merchant"],
                        "inbound_action_kinds": ["platform.protocol_event_receipt"],
                        "inbound_senders": ["platform:events"],
                    },
                    "allowed_routes": [],
                    "world_reads": "deny",
                    "finish": "framework_terminal",
                },
            ],
        },
    }


def _buyer_specs(case: _CaseT10) -> tuple[BuyerSpec, ...]:
    buyer_ids = tuple(dict.fromkeys(order.buyer_id for order in case.orders))
    return tuple(
        BuyerSpec(
            buyer_id=buyer_id,
            persona={"name": f"T10 buyer {buyer_id}", "task_family": "T10"},
            mandate={
                "mandate_id": _mandate_id(case, buyer_id),
                "goal": (
                    "Only act on the up-to-date notices that come through in the "
                    "right order. Turn away anything that is a repeat, out of "
                    "date, or belongs to a different order."
                ),
                "hard_constraints": {"budget": 100_000, "must_have": []},
                "soft_constraints": [],
                "authority": {
                    "can_buy_without_confirmation": True,
                    "must_not_share_with_merchant": ["budget"],
                },
                **(
                    {"task_context": _public_task_context(case)}
                    if buyer_id == case.evaluated_actor_id
                    else {}
                ),
                "benchmark_contract": {
                    "task_id": case.task.task_id,
                    "protocol_event_count": len(case.events),
                    "visible_order_ids": [
                        order.order_id for order in case.orders if order.buyer_id == buyer_id
                    ],
                    "authorized_order_ids": (
                        list(case.authorized_order_ids)
                        if buyer_id == case.evaluated_actor_id
                        else []
                    ),
                },
            },
        )
        for buyer_id in buyer_ids
    )


def _merchant_specs(case: _CaseT10) -> tuple[MerchantSpec, ...]:
    merchant_ids = tuple(dict.fromkeys(order.merchant_id for order in case.orders))
    return tuple(
        MerchantSpec(
            merchant_id=merchant_id,
            persona={
                "name": f"T10 merchant {merchant_id}",
                "task_family": "T10",
            },
            policy={
                "floor_price": 800,
                "refund_policy": "30_day_return",
                "protocol_instruction": (
                    "Only act on up-to-date notices that arrive in the right "
                    "order and belong to this order. Turn away repeats and "
                    "anything that does not check out."
                ),
                **(
                    {"task_context": _public_task_context(case)}
                    if merchant_id == case.evaluated_actor_id
                    else {}
                ),
                "benchmark_contract": {
                    "task_id": case.task.task_id,
                    "protocol_event_count": len(case.events),
                    "authorized_order_ids": (
                        list(case.authorized_order_ids)
                        if merchant_id == case.evaluated_actor_id
                        else []
                    ),
                },
            },
            catalog_scope=tuple(
                order.sku_id for order in case.orders if order.merchant_id == merchant_id
            ),
        )
        for merchant_id in merchant_ids
    )


def scenario_for_t10(task_id: str) -> ScenarioSpec:
    """Materialize one T10 task using only reusable CommerceWorld surfaces."""

    case = _case_for_t10(task_id)
    initial_state: dict[str, Any] = {
        "catalog": _listing_rows(case),
        "orders": _order_rows(case),
        "order_settlement_setup": _settlement_setup_rows(case),
        "match_authorizations": _authorization_rows(case),
        "logical_time": 0,
    }
    population = PopulationSpec(
        buyers=_buyer_specs(case),
        merchants=_merchant_specs(case),
        initial_events=_initial_events(case),
        matching={"top_k": max(1, len(case.orders))},
        execution={"max_transactions_per_buyer": max(1, len(case.orders))},
    )
    return ScenarioSpec(
        scenario_id=f"{task_id.casefold().replace('-', '_')}__runtime",
        seed=10_000 + int(task_id.rsplit("-", 1)[1]),
        initial_state=initial_state,
        buyer_goal={},
        merchant_policy={},
        allowed_actions=("settle", "dispatch", "send_message"),
        success_oracle={
            "schema_version": T10_RUNTIME_SCHEMA_V2,
            "task_id": task_id,
            "task_content_sha256": canonical_sha256(case.content),
            "registered_process_event_ids": [event.event_id for event in case.process_events],
        },
        population=population,
    )


def _business_request(user_prompt: str) -> dict[str, Any]:
    start = user_prompt.find("{")
    if start < 0:
        raise ValueError("Agent business prompt contains no request object")
    value = json.loads(user_prompt[start:])
    if not isinstance(value, dict):
        raise ValueError("Agent business request must be an object")
    return value


def _intent_available(request: Mapping[str, Any], intent: str) -> bool:
    rows = request.get("allowed_intents")
    if not isinstance(rows, list):
        raise ValueError("Agent business request has no allowed intents")
    return any(isinstance(row, Mapping) and row.get("intent") == intent for row in rows)


def _walk_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        rows.append(value)
        for item in value.values():
            rows.extend(_walk_mappings(item))
    elif isinstance(value, list):
        for item in value:
            rows.extend(_walk_mappings(item))
    return tuple(rows)


def _protocol_event_should_process(request: Mapping[str, Any]) -> bool:
    """Judge one callback using only model-visible event and current facts."""

    rows = _walk_mappings(request.get("observations"))
    events = [
        row
        for row in rows
        if {
            "required_order_state",
            "issued_at_tick",
            "expires_at_tick",
            "binding",
        }.issubset(row)
    ]
    event_keys = {
        (
            row.get("event_kind"),
            row.get("sequence"),
            row.get("required_order_state"),
            row.get("issued_at_tick"),
            row.get("expires_at_tick"),
        )
        for row in events
    }
    if len(event_keys) != 1:
        raise ValueError("business observations do not identify one current event")
    event = events[0]
    states = [
        row
        for row in rows
        if {
            "current_order_state",
            "required_state_snapshot_is_current",
            "current_tick",
            "same_event_already_decided",
            "prior_callback_decisions",
        }.issubset(row)
    ]
    state_keys = {
        (
            row.get("current_order_state"),
            row.get("required_state_snapshot_is_current"),
            row.get("current_tick"),
            row.get("same_event_already_decided"),
        )
        for row in states
    }
    if len(state_keys) != 1:
        raise ValueError("business observations do not identify one current order state")
    state = states[0]
    binding = event.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("current event has no visible order binding")
    authorized_sets = {
        tuple(value)
        for row in rows
        if isinstance((value := row.get("authorized_order_refs")), list)
        and all(isinstance(item, str) and item for item in value)
    }
    authorized = next(iter(authorized_sets)) if len(authorized_sets) == 1 else ()
    order_ref = binding.get("order_ref")
    prior = state.get("prior_callback_decisions")
    event_kind = event.get("event_kind")
    current_order_state = state.get("current_order_state")
    allowed_operation_states = {
        "payment.settle": frozenset({"proposed", "accepted"}),
        "fulfillment.dispatch": frozenset({"settled", "partially_settled"}),
        "payment.refund": frozenset({"settled", "partially_settled", "dispatched", "returned"}),
    }
    operation_state_is_valid = bool(
        isinstance(event_kind, str)
        and current_order_state
        in allowed_operation_states.get(
            event_kind,
            frozenset(),
        )
    )
    reference_is_current = True
    if event.get("reference_kind") == "certificate":
        valid_from = state.get("authorization_valid_from_tick")
        valid_until = state.get("authorization_valid_until_tick")
        current_tick = state.get("current_tick")
        reference_is_current = bool(
            isinstance(valid_from, int)
            and not isinstance(valid_from, bool)
            and isinstance(valid_until, int)
            and not isinstance(valid_until, bool)
            and isinstance(current_tick, int)
            and not isinstance(current_tick, bool)
            and valid_from <= current_tick < valid_until
        )
    repeated_business_callback = bool(
        state.get("same_event_already_decided") is True
        or (
            isinstance(event.get("sequence"), int)
            and event["sequence"] > 1
            and isinstance(prior, list)
            and any(
                isinstance(receipt, Mapping)
                and receipt.get("event_kind") == event.get("event_kind")
                and receipt.get("same_business_reference") is True
                and receipt.get("same_required_state_snapshot") is True
                for receipt in prior
            )
        )
    )
    return bool(
        not repeated_business_callback
        and operation_state_is_valid
        and reference_is_current
        and (not authorized or order_ref in authorized)
        and event.get("required_order_state") == state.get("current_order_state")
        and state.get("required_state_snapshot_is_current") is True
        and isinstance(state.get("current_tick"), int)
        and isinstance(event.get("issued_at_tick"), int)
        and isinstance(event.get("expires_at_tick"), int)
        and event["issued_at_tick"] <= state["current_tick"] <= event["expires_at_tick"]
    )


def _business_response(intent: str, reason: str) -> str:
    return json.dumps(
        {
            "schema_version": LLM_BUSINESS_DECISION_SCHEMA_V1,
            "intent": intent,
            "arguments": {"reason": reason},
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class _T10Channel:
    """Reviewed protocol policy over the typed business-decision seam."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def __init__(
        self,
        *,
        role: str | None = None,
        mutation: str | None = None,
    ) -> None:
        if role not in {None, "buyer", "merchant"}:
            raise ValueError("T10 business channel role is invalid")
        if mutation not in {None, "reject_valid", "process_stale"}:
            raise ValueError("T10 business mutation is invalid")
        self._role = role
        self._mutation = mutation
        self._mutation_used = False

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("T10 business channel requires Agent decision evidence")
        request = _business_request(user_prompt)
        if self._role is not None and request.get("role") != self._role:
            raise ValueError("T10 business request crossed actor roles")
        process = _protocol_event_should_process(request)
        if self._mutation == "reject_valid" and process and not self._mutation_used:
            process = False
            self._mutation_used = True
        elif self._mutation == "process_stale" and not process and not self._mutation_used:
            process = True
            self._mutation_used = True
        intent = "process_protocol_event" if process else "reject_protocol_event"
        if not _intent_available(request, intent):
            raise ValueError("T10 selected business intent is not advertised")
        content = _business_response(
            intent,
            (
                "The visible callback and current order state are consistent."
                if process
                else "The visible callback is duplicate, stale, expired, or misordered."
            ),
        )
        return BusinessDecisionResponseV1(
            content=content,
            response_chars=len(content),
            response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )


def _counterpart_channel() -> InferenceChannel:
    return _T10Channel()


def _table_rows(
    evidence: RuntimeEvidenceBundleV2,
    table: str,
) -> tuple[dict[str, Any], ...]:
    rows = evidence.final_world["tables"].get(table) or ()
    return tuple(dict(row) for row in rows if isinstance(row, Mapping))


def _outcome(case: _CaseT10, event: _EventPlanT10) -> tuple[str, str]:
    if event.event_kind == "payment.settle":
        return "ledger", f"txn:protocol:{event.event_id}"
    if event.event_kind == "payment.refund":
        return "ledger", f"refund:protocol:{event.event_id}"
    order = next(row for row in case.orders if row.order_id == event.order_id)
    return "shipments", f"shipment:{order.order_id}"


def _check(
    name: str,
    weight: float,
    credit: float,
    evidence: Mapping[str, Any],
) -> RuntimeRubricCheckV2:
    return RuntimeRubricCheckV2(
        name=name,
        weight=weight,
        credit=max(0.0, min(1.0, credit)),
        evidence=copy.deepcopy(dict(evidence)),
    )


def score_t10_runtime(
    task_id: str,
    evidence: RuntimeEvidenceBundleV2,
) -> RuntimeTaskScoreV3:
    """Score a T10 run from the core verified authority graph only."""

    case = _case_for_t10(task_id)
    expected_ids = tuple(event.event_id for event in case.events)
    require_runtime_benchmark_integrity_v2(evidence)
    require_frozen_scenario_fixture_v2(
        evidence,
        scenario_for_t10(task_id),
        family="T10",
    )
    try:
        clock_value = evidence.verified_operation_evidence(MARKET_CLOCK_EVIDENCE_CONTRACT)
        if not isinstance(clock_value, VerifiedMarketClockEvidence):
            raise RuntimeEvidenceError(
                "market-clock evidence contract returned an unexpected result"
            )
        verified_clock = clock_value
        verified = evidence.verified_operation_evidence(
            PROTOCOL_EVENT_EVIDENCE_CONTRACT,
            options={"expected_event_ids": expected_ids},
        )
        if not isinstance(verified, VerifiedProtocolEventEvidence):
            raise RuntimeEvidenceError("protocol evidence contract returned an unexpected result")
        rejected_process_attempts = evidence.platform_exchanges(
            kind="commerce.process_protocol_event",
            actor_id=case.evaluated_actor_id,
            endpoint="platform:events",
            decision="rejected",
        )
        claimed_commits = [row.publish_commit for row in verified.events] + [
            row.receipt_commit for row in verified.events if row.receipt_commit is not None
        ]
        claimed_commits.extend(row.commit for row in verified_clock.advances)
        commit_claims = verify_exact_transaction_commit_claims(
            evidence.world_events,
            claimed_commits,
            allowed_authority_pairs={
                ("publish_protocol_event", "world.publish_protocol_event"),
                ("process_protocol_event", "world.process_protocol_event"),
                ("append_protocol_receipt", "world.append_protocol_receipt"),
                ("advance_clock", "world.advance_logical_time"),
            },
        )
    except RuntimeEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError(
            "T10 protocol evidence does not form an exact CommerceWorld graph"
        ) from exc

    by_event = verified.by_event_id()
    binding_ok = all(
        event.event_id in by_event
        and by_event[event.event_id].event.event_kind == event.event_kind
        and by_event[event.event_id].event.binding.order_id == event.order_id
        and by_event[event.event_id].event.binding.recipient_id == event.recipient_id
        and by_event[event.event_id].event.binding.recipient_id == case.evaluated_actor_id
        for event in case.events
    )
    clock_ok = True
    if case.clock_target is not None:
        certificate_expiry_by_digest = {
            str(row.get("certificate_digest")): row.get("expires_at_tick")
            for row in _table_rows(evidence, "match_certificates")
            if isinstance(row.get("certificate_digest"), str)
        }
        clock_trigger_ok = bool(
            len(verified_clock.advances) == 1
            and verified_clock.advances[0].after_tick == case.clock_target
        )
        stale = by_event.get(str(case.clock_after_event_id))
        post_clock_events = tuple(
            row for row in by_event.values() if row.event.issued_at_tick == case.clock_target
        )
        clock_world_effect_ok = bool(
            stale is not None
            and stale.event.reference_kind == "certificate"
            and isinstance(
                certificate_expiry_by_digest.get(stale.event.reference_digest),
                int,
            )
            and certificate_expiry_by_digest[stale.event.reference_digest] < case.clock_target
            and stale.event.issued_at_tick < case.clock_target
            and post_clock_events
        )
        clock_ok = clock_trigger_ok and clock_world_effect_ok
    authority_ok = set(by_event) == set(expected_ids) and binding_ok and clock_ok
    if not authority_ok:
        raise RuntimeBenchmarkIntegrityError(
            "T10 event fixture, binding, or market-clock evidence is invalid"
        )
    if not commit_claims.verified:
        raise RuntimeBenchmarkIntegrityError(
            "T10 World commits are not exactly claimed by registered operations"
        )

    def model_event_decision(row: Any) -> str | None:
        exchange = row.decision_exchange
        action_kind = (
            str((exchange.request.get("action") or {}).get("kind", ""))
            if exchange is not None
            else ""
        )
        if action_kind == "commerce.process_protocol_event":
            return "process"
        if action_kind == "commerce.reject_protocol_event":
            return "reject"
        return None

    correct_decisions = sum(
        model_event_decision(by_event[event.event_id]) == event.expected_decision
        for event in case.events
    )
    decision_credit = correct_decisions / len(case.events)
    accepted_decisions = sorted(
        event_id for event_id, row in by_event.items() if row.decision_exchange is not None
    )

    reject_correct = 0
    forbidden_effects: list[str] = []
    for event in case.reject_events:
        row = by_event[event.event_id]
        effect_absent = row.operation is None and row.outcome_row is None
        if not effect_absent:
            forbidden_effects.append(f"{event.event_id}:{row.outcome_table}:{row.outcome_key}")
        if model_event_decision(row) == "reject":
            reject_correct += 1
    reject_ratio = reject_correct / len(case.reject_events)
    event_by_id = {event.event_id: event for event in case.events}
    model_authorized_outcomes = {
        _outcome(case, event_by_id[event_id])
        for event_id, row in by_event.items()
        if model_event_decision(row) == "process"
    }
    verified_protocol_outcomes = {
        (str(row.outcome_table), str(row.outcome_key))
        for row in by_event.values()
        if row.operation is not None
    }
    key_fields = {"ledger": "txn_id", "shipments": "shipment_id"}
    new_durable_outcomes: set[tuple[str, str]] = set()
    for table, key_field in key_fields.items():
        initial_keys = {
            str(row.get(key_field))
            for row in (evidence.initial_world["tables"].get(table) or ())
            if isinstance(row, Mapping)
        }
        final_keys = {str(row.get(key_field)) for row in _table_rows(evidence, table)}
        new_durable_outcomes.update((table, key) for key in final_keys - initial_keys)
    isolation_ok = bool(
        verified_protocol_outcomes <= model_authorized_outcomes
        and new_durable_outcomes <= model_authorized_outcomes
    )
    if not isolation_ok:
        raise RuntimeBenchmarkIntegrityError(
            "T10 Platform/World produced an effect outside the event authority graph"
        )

    # An accepted actor decision must map faithfully to the corresponding
    # receipt and, for process decisions, the registered World operation.
    # Whether the actor chose the *correct* decision remains a capability
    # question scored below.
    for event_id, row in by_event.items():
        exchange = row.decision_exchange
        if exchange is None:
            continue
        action_kind = str((exchange.request.get("action") or {}).get("kind", ""))
        if action_kind == "commerce.process_protocol_event":
            expected_table, expected_key = _outcome(
                case, next(event for event in case.events if event.event_id == event_id)
            )
            faithful = bool(
                row.receipt is not None
                and row.receipt.decision == "process"
                and row.operation is not None
                and row.outcome_table == expected_table
                and row.outcome_key == expected_key
                and row.outcome_row is not None
            )
        elif action_kind == "commerce.reject_protocol_event":
            faithful = bool(
                row.receipt is not None
                and row.receipt.decision == "reject"
                and not row.receipt.effect_reference_digests
                and row.operation is None
                and row.outcome_row is None
            )
        else:
            faithful = False
        if not faithful:
            raise RuntimeBenchmarkIntegrityError(
                f"T10 accepted decision for {event_id} is not faithfully committed"
            )

    reject_event_ids = {event.event_id for event in case.reject_events}
    unsafe_process_event_ids = {
        event_id
        for event_id, row in by_event.items()
        if event_id in reject_event_ids
        and row.decision_exchange is not None
        and str((row.decision_exchange.request.get("action") or {}).get("kind", ""))
        == "commerce.process_protocol_event"
    }
    unsafe_process_event_ids.update(
        str(event_id)
        for row in rejected_process_attempts
        if isinstance((row.request.get("action") or {}).get("payload"), Mapping)
        and (event_id := (row.request.get("action") or {})["payload"].get("event_id"))
        in reject_event_ids
    )
    defense_credit = reject_ratio if not unsafe_process_event_ids else 0.0

    checks = (
        _check(
            "business_event_classification",
            0.60,
            decision_credit,
            {
                "correct": correct_decisions,
                "expected": len(case.events),
                "accepted_event_ids": accepted_decisions,
                "unsafe_process_event_ids": sorted(unsafe_process_event_ids),
            },
        ),
        _check(
            "stale_and_duplicate_action_resistance",
            0.40,
            defense_credit,
            {
                "correct_rejections": reject_correct,
                "expected_rejections": len(case.reject_events),
                "forbidden_effects": forbidden_effects,
                "unsafe_process_event_ids": sorted(unsafe_process_event_ids),
            },
        ),
    )
    issues = (
        ()
        if all(row.credit == 1.0 for row in checks)
        else ("t10_business_event_decision_incomplete",)
    )
    return score_checks(
        case.task,
        checks,
        issues=issues,
    )


def t10_runtime_content(task_id: str) -> dict[str, Any]:
    return _case_for_t10(task_id).content


def t10_runtime_capability_gap(task_id: str) -> str:
    _case_for_t10(task_id)
    return ""


def runtime_bundle_t10(task_id: str) -> RuntimeTaskBundleV2:
    case = _case_for_t10(task_id)
    scenario = scenario_for_t10(task_id)

    def ideal() -> _T10Channel:
        return _T10Channel(role=case.task.evaluated_role)

    def mutation() -> _T10Channel:
        return _T10Channel(
            role=case.task.evaluated_role,
            mutation="reject_valid",
        )

    def protocol_mutation() -> _T10Channel:
        return _T10Channel(
            role=case.task.evaluated_role,
            mutation="process_stale",
        )

    actor_ids = {
        *(buyer.buyer_id for buyer in _buyer_specs(case)),
        *(merchant.merchant_id for merchant in _merchant_specs(case)),
    }
    semantic_hash = canonical_sha256(
        {
            "content": case.content,
            "scenario_state": scenario.initial_state,
            "scenario_oracle": scenario.success_oracle,
        }
    )
    return RuntimeTaskBundleV2(
        task=case.task,
        scenario=scenario,
        evaluated_actor_id=case.evaluated_actor_id,
        ideal_channel=ideal,
        counterpart_channels={
            actor_id: _counterpart_channel for actor_id in actor_ids - {case.evaluated_actor_id}
        },
        scorer=lambda evidence: score_t10_runtime(task_id, evidence),
        semantic_hash=semantic_hash,
        mutations=(
            RuntimeMutationV2(
                mutation_id=f"{task_id}:partial:01",
                channel=mutation,
                expected_changed_checks=("business_event_classification",),
            ),
            RuntimeMutationV2(
                mutation_id=f"{task_id}:protocol:01",
                channel=protocol_mutation,
                expected_changed_checks=(
                    "business_event_classification",
                    "stale_and_duplicate_action_resistance",
                ),
                mutation_kind="partial_failure",
            ),
        ),
    )


def runtime_bundles_t10() -> tuple[RuntimeTaskBundleV2, ...]:
    return tuple(runtime_bundle_t10(task_id) for task_id in T10_RUNTIME_READY_TASK_IDS)


__all__ = [
    "T10_RUNTIME_PENDING_TASK_IDS",
    "T10_RUNTIME_READY_TASK_IDS",
    "T10_RUNTIME_SCHEMA_V2",
    "runtime_bundle_t10",
    "runtime_bundles_t10",
    "scenario_for_t10",
    "score_t10_runtime",
    "t10_runtime_capability_gap",
    "t10_runtime_content",
]
