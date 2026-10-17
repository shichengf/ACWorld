"""World-authoritative domain rules for CommerceWorld after-sales.

The passive records in :mod:`protocol.after_sales` are replay-stable values.
This module is the authority boundary that derives those values from trusted
World rows and authenticated actors.  Agents submit only compact intents.
They cannot submit owners, merchants, policy revisions, logical time, money,
outcomes, causal digests, record versions, or sealed records.

This file intentionally contains no scenario, scorer, transport, or storage
adapter.  A later integration step can persist these immutable records in both
World backends and expose the same methods over in-process and HTTP VCP paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, TypedDict, cast

from protocol.after_sales import (
    AfterSalesAuthority,
    AfterSalesAuthorityError,
    AfterSalesOrderBinding,
    DisputeBundle,
    DisputeCase,
    DisputeEvidence,
    ExchangeCase,
    RefundCase,
    RefundChain,
    RefundDecision,
    ReturnAuthorization,
    ReturnChain,
    ReturnReceipt,
    ReturnRequest,
    Ruling,
    after_sales_binding_from_json,
    after_sales_binding_to_dict,
    apply_dispute_record,
    apply_exchange_case,
    apply_refund_record,
    apply_return_record,
    build_after_sales_authority,
    build_after_sales_order_binding,
    build_dispute_case,
    build_dispute_evidence,
    build_exchange_case,
    build_refund_case,
    build_refund_decision,
    build_return_authorization,
    build_return_receipt,
    build_return_request,
    build_ruling,
    validate_after_sales_order_binding,
    validate_after_sales_record,
)
from protocol.evidence_records import EvidenceRecord, validate_evidence_record
from world.errors import IdempotencyConflict, WorldError, WriteNotAuthorized
from world.payment_fulfillment import (
    PackingRecord,
    PaymentStateRecord,
    payment_state_to_dict,
    validate_packing_record,
    validate_payment_state_record,
)
from world.types import Order, OrderState, Receipt, Shipment


AFTER_SALES_POLICY_SCHEMA = "cwe.world-after-sales-policy.v1"
PAID_CANCELLATION_SCHEMA = "cwe.world-paid-cancellation.v1"
DISPUTE_RESPONSE_SCHEMA = "cwe.world-dispute-response.v1"
LEDGER_RECONCILIATION_REQUEST_SCHEMA = (
    "cwe.world-ledger-reconciliation-request.v1"
)
LEDGER_RECONCILIATION_SOURCE_SCHEMA = (
    "cwe.world-ledger-reconciliation-source.v1"
)
LEDGER_RECONCILIATION_RESULT_SCHEMA = (
    "cwe.world-ledger-reconciliation-result.v1"
)
AFTER_SALES_INTENT_SCHEMA = "cwe.world-after-sales-intent.v1"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | tuple["JsonValue", ...]
AppendDisposition: TypeAlias = Literal["append", "idempotent"]
PaymentState: TypeAlias = Literal["authorized", "captured"]
FinancialEffect: TypeAlias = Literal["void", "refund"]
LedgerEffect: TypeAlias = Literal["charge", "refund"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


class AfterSalesCoreError(WorldError):
    """Base error for World-authoritative after-sales rules."""


class AfterSalesIntentError(AfterSalesCoreError):
    """An actor intent is malformed or tries to self-report authority."""


class AfterSalesPolicyError(AfterSalesCoreError):
    """A policy revision is malformed or has invalid lineage."""


class AfterSalesCoreTransitionError(AfterSalesCoreError):
    """A trusted record is invalid for the current World state."""


@dataclass(frozen=True, slots=True)
class AfterSalesPolicyRevision:
    """Immutable merchant policy revision published by a trusted service."""

    policy_id: str
    merchant_id: str
    revision: int
    previous_digest: str | None
    effective_tick: int
    published_by_id: str
    return_window_ticks: int
    max_refund_bps: int
    split_refund_bps: int
    owner_paid_cancel_allowed: bool
    merchant_paid_cancel_allowed: bool
    allowed_return_conditions: tuple[str, ...]
    return_authorizer_ids: tuple[str, ...]
    return_receiver_ids: tuple[str, ...]
    refund_decider_ids: tuple[str, ...]
    exchange_authorizer_ids: tuple[str, ...]
    adjudicator_ids: tuple[str, ...]
    evidence_service_ids: tuple[str, ...]
    ledger_requester_ids: tuple[str, ...]
    ledger_reconciler_ids: tuple[str, ...]
    policy_digest: str = ""
    schema_id: str = AFTER_SALES_POLICY_SCHEMA


@dataclass(frozen=True, slots=True)
class AfterSalesCoreAuthority:
    """Trusted permissions derived from a persisted binding and policy."""

    binding_digest: str
    policy_digest: str
    owner_id: str
    merchant_id: str
    paid_cancel_actor_ids: tuple[str, ...]
    ledger_requester_ids: tuple[str, ...]
    ledger_reconciler_ids: tuple[str, ...]
    protocol_authority: AfterSalesAuthority


@dataclass(frozen=True, slots=True)
class PaidCancellationRecord:
    """Committed commercial effect of cancelling a paid, undispatched order."""

    cancellation_id: str
    binding: AfterSalesOrderBinding
    policy_digest: str
    actor_id: str
    reason: str
    payment_state: PaymentState
    payment_digest: str
    fulfillment_stage: Literal["authorized", "created", "packed"]
    packing_digest: str | None
    financial_effect: FinancialEffect
    refund_amount: int
    inventory_release_qty: int
    logical_tick: int
    idempotency_key: str
    request_fingerprint: str
    record_digest: str = ""
    schema_id: str = PAID_CANCELLATION_SCHEMA


@dataclass(frozen=True, slots=True)
class DisputeResponseRecord:
    """An opposing party's immutable response to the current dispute version."""

    response_id: str
    dispute_id: str
    dispute_digest: str
    binding: AfterSalesOrderBinding
    facts: Mapping[str, JsonValue]
    actor_id: str
    logical_tick: int
    idempotency_key: str
    request_fingerprint: str
    record_digest: str = ""
    schema_id: str = DISPUTE_RESPONSE_SCHEMA


@dataclass(frozen=True, slots=True)
class LedgerReconciliationRequest:
    """A party's request for a World-derived per-order ledger explanation."""

    request_id: str
    binding: AfterSalesOrderBinding
    actor_id: str
    reason: str
    logical_tick: int
    idempotency_key: str
    request_fingerprint: str
    record_digest: str = ""
    schema_id: str = LEDGER_RECONCILIATION_REQUEST_SCHEMA


@dataclass(frozen=True, slots=True)
class LedgerReconciliationSource:
    """Typed World-derived ledger effect used by reconciliation after restart."""

    source_id: str
    binding: AfterSalesOrderBinding
    txn_id: str
    effect: LedgerEffect
    qty: int
    amount: int
    currency: str
    receipt_digest: str
    logical_tick: int
    source_digest: str = ""
    schema_id: str = LEDGER_RECONCILIATION_SOURCE_SCHEMA


@dataclass(frozen=True, slots=True)
class LedgerReconciliationResult:
    """World-accounting result computed only from authoritative ledger rows."""

    result_id: str
    request_id: str
    request_digest: str
    binding: AfterSalesOrderBinding
    source_txn_ids: tuple[str, ...]
    source_digest: str
    gross_amount: int
    refund_amount: int
    net_amount: int
    currency: str
    actor_id: str
    logical_tick: int
    idempotency_key: str
    request_fingerprint: str
    record_digest: str = ""
    schema_id: str = LEDGER_RECONCILIATION_RESULT_SCHEMA


@dataclass(frozen=True, slots=True)
class CoreDisputeBundle:
    """Materialized dispute state, including explicit opposing responses."""

    case: DisputeCase
    evidence: tuple[DisputeEvidence, ...] = ()
    responses: tuple[DisputeResponseRecord, ...] = ()
    ruling: Ruling | None = None


class AfterSalesIntent(TypedDict, total=False):
    """Normalized compact intent. Operation-specific fields are exact."""

    op: str
    order_id: str
    reason: str
    requested_qty: int
    request_id: str
    authorization_id: str
    received_qty: int
    condition: str
    case_id: str
    replacement_sku_id: str
    dispute_id: str
    evidence_id: str
    evidence_ids: tuple[str, ...]
    position: Literal["accept", "contest"]
    rationale: str


_INTENT_FIELDS: dict[str, frozenset[str]] = {
    "cancel_paid_order": frozenset({"op", "order_id", "reason"}),
    "request_return": frozenset(
        {"op", "order_id", "requested_qty", "reason", "evidence_ids"}
    ),
    "authorize_return": frozenset(
        {"op", "order_id", "request_id", "reason"}
    ),
    "deny_return": frozenset({"op", "order_id", "request_id", "reason"}),
    "receive_return": frozenset(
        {
            "op",
            "order_id",
            "request_id",
            "authorization_id",
            "received_qty",
            "condition",
        }
    ),
    "open_refund_case": frozenset({"op", "order_id", "reason"}),
    "approve_refund": frozenset({"op", "order_id", "case_id", "reason"}),
    "deny_refund": frozenset({"op", "order_id", "case_id", "reason"}),
    "request_exchange": frozenset(
        {"op", "order_id", "replacement_sku_id", "reason"}
    ),
    "authorize_exchange": frozenset(
        {"op", "order_id", "case_id", "reason"}
    ),
    "deny_exchange": frozenset({"op", "order_id", "case_id", "reason"}),
    "complete_exchange": frozenset({"op", "order_id", "case_id", "reason"}),
    "open_dispute": frozenset({"op", "order_id", "reason"}),
    "submit_dispute_evidence": frozenset(
        {"op", "order_id", "dispute_id", "evidence_id"}
    ),
    "respond_to_dispute": frozenset(
        {"op", "order_id", "dispute_id", "evidence_ids", "position"}
    ),
    "rule_for_filer": frozenset(
        {"op", "order_id", "dispute_id", "rationale"}
    ),
    "rule_for_respondent": frozenset(
        {"op", "order_id", "dispute_id", "rationale"}
    ),
    "rule_split": frozenset(
        {"op", "order_id", "dispute_id", "rationale"}
    ),
    "request_ledger_reconciliation": frozenset(
        {"op", "order_id", "reason"}
    ),
}

_AGENT_FORBIDDEN_FIELDS = frozenset(
    {
        "owner",
        "owner_id",
        "merchant",
        "merchant_id",
        "actor",
        "actor_id",
        "original_actor",
        "logical_tick",
        "server_tick",
        "outcome",
        "state",
        "money",
        "amount",
        "requested_amount",
        "authorized_amount",
        "approved_amount",
        "refund_amount",
        "gross_amount",
        "net_amount",
        "currency",
        "payment_state",
        "financial_effect",
        "inventory_release_qty",
        "policy_revision",
        "policy_digest",
        "binding",
        "binding_digest",
        "record_digest",
        "request_fingerprint",
        "idempotency_key",
        "version",
        "previous_digest",
        "causal_digest",
        "completion_order_digest",
        "source_digest",
    }
)


def build_after_sales_policy_revision(
    *,
    policy_id: str,
    merchant_id: str,
    revision: int,
    previous_digest: str | None,
    effective_tick: int,
    published_by_id: str,
    return_window_ticks: int,
    max_refund_bps: int = 10_000,
    split_refund_bps: int = 5_000,
    owner_paid_cancel_allowed: bool = True,
    merchant_paid_cancel_allowed: bool = True,
    allowed_return_conditions: Iterable[str] = ("new", "opened", "damaged"),
    return_authorizer_ids: Iterable[str],
    return_receiver_ids: Iterable[str],
    refund_decider_ids: Iterable[str],
    exchange_authorizer_ids: Iterable[str],
    adjudicator_ids: Iterable[str],
    evidence_service_ids: Iterable[str] = (),
    ledger_requester_ids: Iterable[str] = (),
    ledger_reconciler_ids: Iterable[str],
) -> AfterSalesPolicyRevision:
    """Seal a trusted policy revision using canonical ordering and digesting."""

    candidate = AfterSalesPolicyRevision(
        policy_id=_text(policy_id, "policy_id"),
        merchant_id=_text(merchant_id, "merchant_id"),
        revision=_positive_int(revision, "revision"),
        previous_digest=previous_digest,
        effective_tick=_nonnegative_int(effective_tick, "effective_tick"),
        published_by_id=_text(published_by_id, "published_by_id"),
        return_window_ticks=_nonnegative_int(
            return_window_ticks, "return_window_ticks"
        ),
        max_refund_bps=_basis_points(max_refund_bps, "max_refund_bps"),
        split_refund_bps=_basis_points(split_refund_bps, "split_refund_bps"),
        owner_paid_cancel_allowed=_boolean(
            owner_paid_cancel_allowed, "owner_paid_cancel_allowed"
        ),
        merchant_paid_cancel_allowed=_boolean(
            merchant_paid_cancel_allowed, "merchant_paid_cancel_allowed"
        ),
        allowed_return_conditions=_canonical_texts(
            allowed_return_conditions, "allowed_return_conditions"
        ),
        return_authorizer_ids=_canonical_texts(
            return_authorizer_ids, "return_authorizer_ids"
        ),
        return_receiver_ids=_canonical_texts(
            return_receiver_ids, "return_receiver_ids"
        ),
        refund_decider_ids=_canonical_texts(
            refund_decider_ids, "refund_decider_ids"
        ),
        exchange_authorizer_ids=_canonical_texts(
            exchange_authorizer_ids, "exchange_authorizer_ids"
        ),
        adjudicator_ids=_canonical_texts(adjudicator_ids, "adjudicator_ids"),
        evidence_service_ids=_canonical_texts(
            evidence_service_ids, "evidence_service_ids"
        ),
        ledger_requester_ids=_canonical_texts(
            ledger_requester_ids, "ledger_requester_ids"
        ),
        ledger_reconciler_ids=_canonical_texts(
            ledger_reconciler_ids, "ledger_reconciler_ids"
        ),
    )
    sealed = _replace_policy_digest(candidate, _digest(_policy_contract(candidate)))
    validate_after_sales_policy(sealed)
    return sealed


def validate_after_sales_policy(policy: AfterSalesPolicyRevision) -> None:
    if not isinstance(policy, AfterSalesPolicyRevision):
        raise AfterSalesPolicyError("policy must be AfterSalesPolicyRevision")
    if policy.schema_id != AFTER_SALES_POLICY_SCHEMA:
        raise AfterSalesPolicyError("unsupported after-sales policy schema")
    _text(policy.policy_id, "policy_id")
    _text(policy.merchant_id, "merchant_id")
    _positive_int(policy.revision, "revision")
    _nonnegative_int(policy.effective_tick, "effective_tick")
    _text(policy.published_by_id, "published_by_id")
    _nonnegative_int(policy.return_window_ticks, "return_window_ticks")
    _basis_points(policy.max_refund_bps, "max_refund_bps")
    _basis_points(policy.split_refund_bps, "split_refund_bps")
    _boolean(policy.owner_paid_cancel_allowed, "owner_paid_cancel_allowed")
    _boolean(policy.merchant_paid_cancel_allowed, "merchant_paid_cancel_allowed")
    if policy.revision == 1:
        if policy.previous_digest is not None:
            raise AfterSalesPolicyError("first policy revision cannot have predecessor")
    else:
        _require_digest(policy.previous_digest, "previous_digest")
    for name in (
        "allowed_return_conditions",
        "return_authorizer_ids",
        "return_receiver_ids",
        "refund_decider_ids",
        "exchange_authorizer_ids",
        "adjudicator_ids",
        "evidence_service_ids",
        "ledger_requester_ids",
        "ledger_reconciler_ids",
    ):
        values = getattr(policy, name)
        if values != tuple(sorted(set(values))):
            raise AfterSalesPolicyError(f"{name} must be unique and sorted")
        for value in values:
            _text(value, name)
    if not policy.allowed_return_conditions:
        raise AfterSalesPolicyError("policy requires at least one return condition")
    if not policy.ledger_reconciler_ids:
        raise AfterSalesPolicyError("policy requires a ledger reconciler")
    _require_digest(policy.policy_digest, "policy_digest")
    if policy.policy_digest != _digest(_policy_contract(policy)):
        raise AfterSalesPolicyError("after-sales policy digest mismatch")


def after_sales_policy_to_dict(policy: AfterSalesPolicyRevision) -> dict[str, Any]:
    validate_after_sales_policy(policy)
    return {**_policy_contract(policy), "policy_digest": policy.policy_digest}


def after_sales_policy_from_dict(value: Mapping[str, Any]) -> AfterSalesPolicyRevision:
    expected = {
        "schema_id",
        "policy_id",
        "merchant_id",
        "revision",
        "previous_digest",
        "effective_tick",
        "published_by_id",
        "return_window_ticks",
        "max_refund_bps",
        "split_refund_bps",
        "owner_paid_cancel_allowed",
        "merchant_paid_cancel_allowed",
        "allowed_return_conditions",
        "return_authorizer_ids",
        "return_receiver_ids",
        "refund_decider_ids",
        "exchange_authorizer_ids",
        "adjudicator_ids",
        "evidence_service_ids",
        "ledger_requester_ids",
        "ledger_reconciler_ids",
        "policy_digest",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise AfterSalesPolicyError("after-sales policy payload fields are not exact")
    tuple_fields = {
        "allowed_return_conditions",
        "return_authorizer_ids",
        "return_receiver_ids",
        "refund_decider_ids",
        "exchange_authorizer_ids",
        "adjudicator_ids",
        "evidence_service_ids",
        "ledger_requester_ids",
        "ledger_reconciler_ids",
    }
    payload = dict(value)
    for field in tuple_fields:
        item = payload[field]
        if not isinstance(item, (list, tuple)):
            raise AfterSalesPolicyError(f"{field} must be an array")
        payload[field] = tuple(item)
    policy = AfterSalesPolicyRevision(**cast(Any, payload))
    validate_after_sales_policy(policy)
    return policy


def validate_after_sales_policy_transition(
    current: AfterSalesPolicyRevision | None,
    candidate: AfterSalesPolicyRevision,
    *,
    original_actor: str,
    server_tick: int,
    trusted_publisher_ids: Iterable[str],
) -> AppendDisposition:
    """Validate publisher authority, World time, and contiguous policy lineage."""

    validate_after_sales_policy(candidate)
    actor = _text(original_actor, "original_actor")
    trusted = frozenset(_canonical_texts(trusted_publisher_ids, "trusted publishers"))
    if actor not in trusted or candidate.published_by_id != actor:
        raise WriteNotAuthorized("after-sales policy publisher is not trusted")
    if candidate.effective_tick != _nonnegative_int(server_tick, "server_tick"):
        raise AfterSalesPolicyError("policy effective_tick must equal World time")
    if current is None:
        if candidate.revision != 1 or candidate.previous_digest is not None:
            raise AfterSalesPolicyError("policy history must start at revision 1")
        return "append"
    validate_after_sales_policy(current)
    if candidate.policy_digest == current.policy_digest:
        return "idempotent"
    if (
        candidate.policy_id != current.policy_id
        or candidate.merchant_id != current.merchant_id
    ):
        raise AfterSalesPolicyError("policy immutable identity changed")
    if (
        candidate.revision != current.revision + 1
        or candidate.previous_digest != current.policy_digest
    ):
        raise AfterSalesPolicyError("policy revision or predecessor is not contiguous")
    if candidate.effective_tick <= current.effective_tick:
        raise AfterSalesPolicyError("policy effective time must increase")
    return "append"


def authoritative_order_digest(order: Order) -> str:
    """Digest the complete authoritative order identity and current state."""

    if not isinstance(order, Order):
        raise AfterSalesCoreError("order must be a World Order")
    return _digest(
        {
            "order_id": str(order.order_id),
            "buyer_id": str(order.buyer_id),
            "merchant_id": str(order.merchant_id),
            "sku_id": str(order.sku_id),
            "qty": order.qty,
            "agreed_price": {
                "amount": _decimal_string(order.agreed_price.amount),
                "currency": order.agreed_price.currency,
            },
            "state": order.state.value,
            "request_order": order.request_order,
        }
    )


def authoritative_receipt_digest(receipt: Receipt) -> str:
    """Digest a persisted ledger receipt without inferring its economic effect."""

    if not isinstance(receipt, Receipt):
        raise AfterSalesCoreError("receipt must be a World Receipt")
    return _digest(
        {
            "txn_id": str(receipt.txn_id),
            "ts": receipt.ts,
            "order_id": str(receipt.order_id),
            "buyer_id": str(receipt.buyer_id),
            "merchant_id": str(receipt.merchant_id),
            "sku_id": str(receipt.sku_id),
            "qty": receipt.qty,
            "price": {
                "amount": _decimal_string(receipt.price.amount),
                "currency": receipt.price.currency,
            },
            "idempotency_key": receipt.idempotency_key,
            "effect": receipt.effect,
        }
    )


def authoritative_shipment_digest(shipment: Shipment) -> str:
    """Digest the complete shipment and its append-only status history."""

    if not isinstance(shipment, Shipment):
        raise AfterSalesCoreError("shipment must be a World Shipment")
    return _digest(
        {
            "shipment_id": str(shipment.shipment_id),
            "order_id": str(shipment.order_id),
            "buyer_id": str(shipment.buyer_id),
            "merchant_id": str(shipment.merchant_id),
            "original_sku_id": str(shipment.original_sku_id),
            "status": shipment.status.value,
            "status_history": [
                {
                    "event_id": event.event_id,
                    "status": event.status.value,
                    "logical_time": event.logical_time,
                    "idempotency_key": event.idempotency_key,
                    "shipment_version": event.shipment_version,
                }
                for event in shipment.status_history
            ],
            "resolution": (
                shipment.resolution.value if shipment.resolution is not None else None
            ),
            "replacement_sku_id": (
                str(shipment.replacement_sku_id)
                if shipment.replacement_sku_id is not None
                else None
            ),
            "version": shipment.version,
            "resolution_idempotency_key": shipment.resolution_idempotency_key,
            "resolved_by": (
                str(shipment.resolved_by) if shipment.resolved_by is not None else None
            ),
            "resolution_version": shipment.resolution_version,
            "resolution_history_length": shipment.resolution_history_length,
        }
    )


def derive_after_sales_binding(
    *,
    order: Order,
    receipt: Receipt | None,
    shipment: Shipment | None,
    policy: AfterSalesPolicyRevision,
    payment: PaymentStateRecord | None = None,
) -> AfterSalesOrderBinding:
    """Derive owner, merchant, amount, policy, and causal digests from World rows."""

    validate_after_sales_policy(policy)
    if not isinstance(order, Order):
        raise AfterSalesCoreError("binding requires authoritative Order")
    if str(order.merchant_id) != policy.merchant_id:
        raise AfterSalesCoreError("policy merchant does not own order")
    payment_contract: dict[str, Any] | None = None
    if payment is not None:
        validate_payment_state_record(payment)
        if (
            payment.order_id != str(order.order_id)
            or payment.owner_id != str(order.buyer_id)
            or payment.merchant_id != str(order.merchant_id)
            or payment.sku_id != str(order.sku_id)
            or payment.qty != order.qty
            or payment.currency != order.agreed_price.currency
        ):
            raise AfterSalesCoreError("payment identity does not match order")
        if payment.state not in {
            "authorized",
            "captured",
            "partially_refunded",
        }:
            raise AfterSalesCoreError("after-sales binding requires active payment")
        amount = payment.amount
        qty = payment.qty
        currency = payment.currency
        payment_contract = payment_state_to_dict(payment)
        if payment.state in {"captured", "partially_refunded"}:
            if receipt is None:
                raise AfterSalesCoreError("captured payment requires ledger receipt")
            if (
                payment.capture_receipt_digest
                != authoritative_receipt_digest(receipt)
            ):
                raise AfterSalesCoreError("payment receipt digest mismatch")
        elif receipt is not None:
            raise AfterSalesCoreError("authorized payment cannot have capture receipt")
    else:
        if not isinstance(receipt, Receipt):
            raise AfterSalesCoreError(
                "legacy binding requires authoritative settlement Receipt"
            )
        amount = _money_minor_units(receipt.price.amount, receipt.qty)
        qty = receipt.qty
        currency = receipt.price.currency
    if receipt is not None:
        identities = (
            str(receipt.order_id) == str(order.order_id),
            str(receipt.buyer_id) == str(order.buyer_id),
            str(receipt.merchant_id) == str(order.merchant_id),
            str(receipt.sku_id) == str(order.sku_id),
        )
        if not all(identities):
            raise AfterSalesCoreError("receipt identity does not match order")
        if receipt.qty <= 0 or receipt.qty > order.qty:
            raise AfterSalesCoreError("receipt quantity is invalid for order")
        if receipt.price.currency != order.agreed_price.currency:
            raise AfterSalesCoreError("receipt currency does not match order")
    shipment_digest: str | None = None
    shipment_id: str | None = None
    if shipment is not None:
        if (
            str(shipment.order_id) != str(order.order_id)
            or str(shipment.buyer_id) != str(order.buyer_id)
            or str(shipment.merchant_id) != str(order.merchant_id)
            or str(shipment.original_sku_id) != str(order.sku_id)
        ):
            raise AfterSalesCoreError("shipment identity does not match order")
        shipment_id = str(shipment.shipment_id)
        shipment_digest = authoritative_shipment_digest(shipment)
    return build_after_sales_order_binding(
        order_id=str(order.order_id),
        item_id=f"{order.order_id}:line:{order.sku_id}",
        sku_id=str(order.sku_id),
        shipment_id=shipment_id,
        owner_id=str(order.buyer_id),
        merchant_id=str(order.merchant_id),
        qty=qty,
        amount=amount,
        currency=currency,
        policy_revision=policy.revision,
        policy_digest=policy.policy_digest,
        order_digest=_digest(
            {
                "order_digest": authoritative_order_digest(order),
                "payment": payment_contract,
            }
        ),
        shipment_digest=shipment_digest,
    )


def derive_after_sales_authority(
    binding: AfterSalesOrderBinding,
    policy: AfterSalesPolicyRevision,
) -> AfterSalesCoreAuthority:
    """Derive actor permissions from persisted binding and policy only."""

    validate_after_sales_order_binding(binding)
    validate_after_sales_policy(policy)
    if (
        binding.policy_revision != policy.revision
        or binding.policy_digest != policy.policy_digest
        or binding.merchant_id != policy.merchant_id
    ):
        raise AfterSalesAuthorityError("binding is not governed by supplied policy")
    evidence_ids = _canonical_texts(
        tuple(
            sorted(
                {binding.owner_id, binding.merchant_id, *policy.evidence_service_ids}
            )
        ),
        "evidence submitters",
    )
    protocol_authority = build_after_sales_authority(
        binding,
        return_authorizer_ids=policy.return_authorizer_ids,
        return_receiver_ids=policy.return_receiver_ids,
        refund_decider_ids=policy.refund_decider_ids,
        exchange_authorizer_ids=policy.exchange_authorizer_ids,
        adjudicator_ids=policy.adjudicator_ids,
        evidence_submitter_ids=evidence_ids,
    )
    paid_cancel: list[str] = []
    if policy.owner_paid_cancel_allowed:
        paid_cancel.append(binding.owner_id)
    if policy.merchant_paid_cancel_allowed:
        paid_cancel.append(binding.merchant_id)
    return AfterSalesCoreAuthority(
        binding_digest=binding.binding_digest,
        policy_digest=policy.policy_digest,
        owner_id=binding.owner_id,
        merchant_id=binding.merchant_id,
        paid_cancel_actor_ids=tuple(sorted(set(paid_cancel))),
        ledger_requester_ids=_canonical_texts(
            tuple(sorted({binding.merchant_id, *policy.ledger_requester_ids})),
            "ledger requesters",
        ),
        ledger_reconciler_ids=policy.ledger_reconciler_ids,
        protocol_authority=protocol_authority,
    )


def after_sales_owner(binding: AfterSalesOrderBinding) -> str:
    validate_after_sales_order_binding(binding)
    return binding.owner_id


def after_sales_merchant(binding: AfterSalesOrderBinding) -> str:
    validate_after_sales_order_binding(binding)
    return binding.merchant_id


def normalize_after_sales_intent(value: Mapping[str, Any]) -> AfterSalesIntent:
    """Validate an exact compact intent and reject all self-reported authority."""

    if not isinstance(value, Mapping):
        raise AfterSalesIntentError("after-sales intent must be an object")
    if any(not isinstance(key, str) or not key for key in value):
        raise AfterSalesIntentError("after-sales intent field names must be strings")
    forbidden = sorted(set(value).intersection(_AGENT_FORBIDDEN_FIELDS))
    if forbidden:
        raise AfterSalesIntentError(
            "after-sales authority fields are World-owned: " + ", ".join(forbidden)
        )
    op = value.get("op")
    if not isinstance(op, str) or op not in _INTENT_FIELDS:
        raise AfterSalesIntentError("unsupported after-sales operation")
    expected = _INTENT_FIELDS[op]
    if set(value) != expected:
        raise AfterSalesIntentError(
            f"{op} intent fields must be exactly: " + ", ".join(sorted(expected))
        )
    normalized: dict[str, Any] = {"op": op}
    for key in expected - {"op"}:
        item = value[key]
        if key in {"requested_qty", "received_qty"}:
            normalized[key] = _positive_int(item, key)
        elif key == "evidence_ids":
            if not isinstance(item, (list, tuple)):
                raise AfterSalesIntentError("evidence_ids must be a string array")
            if any(not isinstance(value, str) or not value.strip() for value in item):
                raise AfterSalesIntentError(
                    "evidence_ids must contain non-empty strings"
                )
            values = tuple(item)
            if values != tuple(sorted(set(values))):
                raise AfterSalesIntentError(
                    "evidence_ids must be sorted and unique"
                )
            normalized[key] = values
        elif key == "position":
            if item not in {"accept", "contest"}:
                raise AfterSalesIntentError(
                    "position must be either accept or contest"
                )
            normalized[key] = item
        else:
            normalized[key] = _text(item, key)
    _canonical_json(_thaw_json(normalized))
    return cast(AfterSalesIntent, normalized)


def after_sales_intent_fingerprint(
    intent: Mapping[str, Any],
    *,
    binding: AfterSalesOrderBinding,
    policy: AfterSalesPolicyRevision,
    original_actor: str,
    evidence_records: Iterable[EvidenceRecord] = (),
) -> str:
    """Bind canonical compact intent to authenticated actor and trusted context."""

    normalized = normalize_after_sales_intent(intent)
    _validate_authority_context(binding, policy)
    actor = _text(original_actor, "original_actor")
    _require_order(normalized, binding)
    evidence = tuple(evidence_records)
    referenced_ids = referenced_after_sales_evidence_ids(normalized)
    if tuple(record.record_id for record in evidence) != referenced_ids:
        raise AfterSalesIntentError(
            "trusted evidence rows do not match the intent references"
        )
    for record in evidence:
        validate_evidence_record(record)
    return _digest(
        {
            "schema_id": AFTER_SALES_INTENT_SCHEMA,
            "actor_id": actor,
            "binding_digest": binding.binding_digest,
            "policy_digest": policy.policy_digest,
            "intent": _thaw_json(normalized),
            "evidence": [
                {
                    "record_id": record.record_id,
                    "record_digest": record.record_digest,
                }
                for record in evidence
            ],
        }
    )


def referenced_after_sales_evidence_ids(
    intent: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return canonical evidence references from one normalized intent."""

    normalized = normalize_after_sales_intent(intent)
    if "evidence_id" in normalized:
        return (cast(str, normalized["evidence_id"]),)
    return cast(tuple[str, ...], normalized.get("evidence_ids", ()))


def derive_paid_cancellation(
    intent: Mapping[str, Any],
    *,
    binding: AfterSalesOrderBinding,
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
    order_state: OrderState,
    dispatched: bool,
    payment: PaymentStateRecord,
    packing: PackingRecord | None,
) -> PaidCancellationRecord:
    normalized = _intent(intent, "cancel_paid_order", binding)
    _validate_core_authority(binding, policy, authority)
    actor = _text(original_actor, "original_actor")
    if actor not in authority.paid_cancel_actor_ids:
        raise WriteNotAuthorized("actor cannot cancel this paid order")
    validate_payment_state_record(payment)
    if payment.order_id != binding.order_id:
        raise AfterSalesCoreTransitionError("cancellation payment binding mismatch")
    payment_state = cast(PaymentState, payment.state)
    if payment_state == "authorized":
        if order_state not in {OrderState.PROPOSED, OrderState.ACCEPTED}:
            raise AfterSalesCoreTransitionError(
                "authorized cancellation requires proposed or accepted order"
            )
        if packing is not None:
            raise AfterSalesCoreTransitionError(
                "authorized cancellation cannot have packing state"
            )
        fulfillment_stage: Literal["authorized", "created", "packed"] = "authorized"
        packing_digest = None
    elif payment_state == "captured":
        if order_state not in {OrderState.SETTLED, OrderState.PARTIALLY_SETTLED}:
            raise AfterSalesCoreTransitionError(
                "captured cancellation requires settled order"
            )
        if packing is None:
            raise AfterSalesCoreTransitionError(
                "captured cancellation requires pre-dispatch packing state"
            )
        validate_packing_record(packing)
        if (
            packing.order_id != binding.order_id
            or packing.payment_digest != payment.record_digest
        ):
            raise AfterSalesCoreTransitionError("cancellation packing binding mismatch")
        if packing.state not in {"created", "packed"}:
            raise AfterSalesCoreTransitionError(
                "packing state no longer permits cancellation"
            )
        fulfillment_stage = cast(Any, packing.state)
        packing_digest = packing.record_digest
    else:
        raise AfterSalesCoreTransitionError("payment is already resolved")
    if dispatched:
        raise AfterSalesCoreTransitionError("paid cancellation cannot follow dispatch")
    tick = _nonnegative_int(server_tick, "server_tick")
    key = _text(idempotency_key, "idempotency_key")
    fingerprint = after_sales_intent_fingerprint(
        normalized, binding=binding, policy=policy, original_actor=actor
    )
    financial_effect: FinancialEffect = (
        "void" if payment_state == "authorized" else "refund"
    )
    unsigned = PaidCancellationRecord(
        cancellation_id=_derived_id("paid-cancel", binding, actor, key),
        binding=binding,
        policy_digest=policy.policy_digest,
        actor_id=actor,
        reason=normalized["reason"],
        payment_state=payment_state,
        payment_digest=payment.record_digest,
        fulfillment_stage=fulfillment_stage,
        packing_digest=packing_digest,
        financial_effect=financial_effect,
        refund_amount=binding.amount if financial_effect == "refund" else 0,
        inventory_release_qty=binding.qty,
        logical_tick=tick,
        idempotency_key=key,
        request_fingerprint=fingerprint,
    )
    return cast(PaidCancellationRecord, _seal_core_record(unsigned))


def validate_paid_cancellation_transition(
    existing: PaidCancellationRecord | None,
    candidate: PaidCancellationRecord,
    *,
    authority: AfterSalesCoreAuthority,
    server_tick: int,
) -> AppendDisposition:
    validate_core_after_sales_record(candidate)
    if candidate.binding.binding_digest != authority.binding_digest:
        raise AfterSalesCoreTransitionError("paid cancellation binding mismatch")
    if candidate.actor_id not in authority.paid_cancel_actor_ids:
        raise WriteNotAuthorized("actor cannot cancel this paid order")
    if candidate.logical_tick != server_tick:
        raise AfterSalesCoreTransitionError("cancellation tick is not World time")
    if existing is None:
        return "append"
    validate_core_after_sales_record(existing)
    if existing.record_digest == candidate.record_digest:
        return "idempotent"
    if (
        existing.cancellation_id == candidate.cancellation_id
        or (
            existing.actor_id == candidate.actor_id
            and existing.idempotency_key == candidate.idempotency_key
        )
    ):
        raise IdempotencyConflict("conflicting paid cancellation retry")
    raise AfterSalesCoreTransitionError("paid order is already cancelled")


def derive_return_request(
    intent: Mapping[str, Any],
    *,
    binding: AfterSalesOrderBinding,
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
    evidence_records: Iterable[EvidenceRecord],
    original_actor: str,
    server_tick: int,
    settled_at_tick: int,
    idempotency_key: str,
) -> ReturnRequest:
    normalized = _intent(intent, "request_return", binding)
    _validate_core_authority(binding, policy, authority)
    tick = _nonnegative_int(server_tick, "server_tick")
    settled = _nonnegative_int(settled_at_tick, "settled_at_tick")
    if tick > settled + policy.return_window_ticks:
        raise AfterSalesCoreTransitionError("return window is closed")
    actor = _text(original_actor, "original_actor")
    if actor != authority.owner_id:
        raise WriteNotAuthorized("only the order owner may request return")
    qty = normalized["requested_qty"]
    evidence = tuple(evidence_records)
    if tuple(record.record_id for record in evidence) != cast(
        tuple[str, ...], normalized["evidence_ids"]
    ):
        raise AfterSalesCoreTransitionError(
            "trusted return evidence does not match intent references"
        )
    for record in evidence:
        validate_evidence_record(record)
        if record.subject_id != binding.order_id:
            raise AfterSalesCoreTransitionError(
                "return evidence is not bound to this order"
            )
        if record.owner_id != actor:
            raise WriteNotAuthorized(
                "return evidence must be owned by the order owner"
            )
    record = build_return_request(
        binding,
        request_id=_derived_id("return-request", binding, actor, idempotency_key),
        requested_qty=qty,
        requested_amount=_prorated_amount(binding, qty, policy.max_refund_bps),
        reason=normalized["reason"],
        actor_id=actor,
        logical_tick=tick,
        idempotency_key=_text(idempotency_key, "idempotency_key"),
        evidence_ids=tuple(item.record_id for item in evidence),
        evidence_digests=tuple(item.record_digest for item in evidence),
    )
    apply_return_record(
        None, record, authority=authority.protocol_authority, server_tick=tick
    )
    return record


def derive_return_authorization(
    intent: Mapping[str, Any],
    *,
    chain: ReturnChain,
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
) -> ReturnAuthorization:
    op = _operation(intent, {"authorize_return", "deny_return"})
    normalized = _intent(intent, op, chain.request.binding)
    _validate_core_authority(chain.request.binding, policy, authority)
    if normalized["request_id"] != chain.request.request_id:
        raise AfterSalesCoreTransitionError("return request reference mismatch")
    actor = _text(original_actor, "original_actor")
    if actor not in authority.protocol_authority.return_authorizer_ids:
        raise WriteNotAuthorized("actor cannot authorize returns")
    approved = op == "authorize_return"
    record = build_return_authorization(
        chain.request,
        authorization_id=_derived_id(
            "return-authorization",
            chain.request.binding,
            original_actor,
            idempotency_key,
        ),
        outcome="authorized" if approved else "denied",
        authorized_qty=chain.request.requested_qty if approved else 0,
        authorized_amount=chain.request.requested_amount if approved else 0,
        reason=normalized["reason"],
        actor_id=actor,
        logical_tick=_nonnegative_int(server_tick, "server_tick"),
        idempotency_key=_text(idempotency_key, "idempotency_key"),
    )
    apply_return_record(
        chain,
        record,
        authority=authority.protocol_authority,
        server_tick=server_tick,
    )
    return record


def derive_return_receipt(
    intent: Mapping[str, Any],
    *,
    chain: ReturnChain,
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
) -> ReturnReceipt:
    if chain.authorization is None:
        raise AfterSalesCoreTransitionError("return has no authorization")
    normalized = _intent(intent, "receive_return", chain.request.binding)
    _validate_core_authority(chain.request.binding, policy, authority)
    if (
        normalized["request_id"] != chain.request.request_id
        or normalized["authorization_id"] != chain.authorization.authorization_id
    ):
        raise AfterSalesCoreTransitionError("return receipt reference mismatch")
    condition = normalized["condition"]
    if condition not in policy.allowed_return_conditions:
        raise AfterSalesCoreTransitionError("return condition is not policy-allowed")
    actor = _text(original_actor, "original_actor")
    if actor not in authority.protocol_authority.return_receiver_ids:
        raise WriteNotAuthorized("actor cannot receive returns")
    record = build_return_receipt(
        chain.authorization,
        receipt_id=_derived_id(
            "return-receipt",
            chain.request.binding,
            original_actor,
            idempotency_key,
        ),
        received_qty=normalized["received_qty"],
        condition=condition,
        actor_id=actor,
        logical_tick=_nonnegative_int(server_tick, "server_tick"),
        idempotency_key=_text(idempotency_key, "idempotency_key"),
    )
    apply_return_record(
        chain,
        record,
        authority=authority.protocol_authority,
        server_tick=server_tick,
    )
    return record


RefundSource: TypeAlias = ReturnChain | CoreDisputeBundle | PaidCancellationRecord


def derive_refund_case(
    intent: Mapping[str, Any],
    *,
    binding: AfterSalesOrderBinding,
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
    source: RefundSource,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
) -> RefundCase:
    normalized = _intent(intent, "open_refund_case", binding)
    _validate_core_authority(binding, policy, authority)
    source_kind, causal_digest, requested_amount = _refund_source(source, binding)
    actor = _text(original_actor, "original_actor")
    if actor not in {
        authority.owner_id,
        authority.merchant_id,
        *authority.protocol_authority.refund_decider_ids,
    }:
        raise WriteNotAuthorized("actor cannot open refund case")
    tick = _nonnegative_int(server_tick, "server_tick")
    record = build_refund_case(
        binding,
        case_id=_derived_id("refund-case", binding, actor, idempotency_key),
        source_kind=source_kind,
        causal_digest=causal_digest,
        requested_amount=min(
            requested_amount,
            _prorated_amount(binding, binding.qty, policy.max_refund_bps),
        ),
        reason=normalized["reason"],
        actor_id=actor,
        logical_tick=tick,
        idempotency_key=_text(idempotency_key, "idempotency_key"),
    )
    apply_refund_record(
        None, record, authority=authority.protocol_authority, server_tick=tick
    )
    return record


def derive_refund_decision(
    intent: Mapping[str, Any],
    *,
    chain: RefundChain,
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
) -> RefundDecision:
    op = _operation(intent, {"approve_refund", "deny_refund"})
    normalized = _intent(intent, op, chain.case.binding)
    _validate_core_authority(chain.case.binding, policy, authority)
    if normalized["case_id"] != chain.case.case_id:
        raise AfterSalesCoreTransitionError("refund case reference mismatch")
    actor = _text(original_actor, "original_actor")
    if actor not in authority.protocol_authority.refund_decider_ids:
        raise WriteNotAuthorized("actor cannot decide refunds")
    approved = op == "approve_refund"
    record = build_refund_decision(
        chain.case,
        decision_id=_derived_id(
            "refund-decision",
            chain.case.binding,
            original_actor,
            idempotency_key,
        ),
        outcome="approved" if approved else "denied",
        approved_amount=chain.case.requested_amount if approved else 0,
        reason=normalized["reason"],
        actor_id=actor,
        logical_tick=_nonnegative_int(server_tick, "server_tick"),
        idempotency_key=_text(idempotency_key, "idempotency_key"),
    )
    apply_refund_record(
        chain,
        record,
        authority=authority.protocol_authority,
        server_tick=server_tick,
    )
    return record


def derive_exchange_case(
    intent: Mapping[str, Any],
    *,
    return_chain: ReturnChain,
    previous: ExchangeCase | None,
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
    completion_order_digest: str | None = None,
) -> ExchangeCase:
    if return_chain.receipt is None:
        raise AfterSalesCoreTransitionError("exchange requires received return")
    binding = return_chain.request.binding
    op = _operation(
        intent,
        {
            "request_exchange",
            "authorize_exchange",
            "deny_exchange",
            "complete_exchange",
        },
    )
    normalized = _intent(intent, op, binding)
    _validate_core_authority(binding, policy, authority)
    actor = _text(original_actor, "original_actor")
    tick = _nonnegative_int(server_tick, "server_tick")
    key = _text(idempotency_key, "idempotency_key")
    if previous is None:
        if op != "request_exchange":
            raise AfterSalesCoreTransitionError("exchange must start with request")
        if actor != authority.owner_id:
            raise WriteNotAuthorized("only the order owner may request exchange")
        case_id = _derived_id("exchange-case", binding, actor, key)
        replacement_sku = normalized["replacement_sku_id"]
        state = "requested"
        version = 1
        previous_digest = None
        completion_digest = None
    else:
        validate_after_sales_record(previous)
        if op == "request_exchange":
            raise AfterSalesCoreTransitionError("exchange request already exists")
        if normalized["case_id"] != previous.case_id:
            raise AfterSalesCoreTransitionError("exchange case reference mismatch")
        if actor not in authority.protocol_authority.exchange_authorizer_ids:
            raise WriteNotAuthorized("actor cannot decide or complete exchange")
        case_id = previous.case_id
        replacement_sku = previous.replacement_sku_id
        state = {
            "authorize_exchange": "authorized",
            "deny_exchange": "denied",
            "complete_exchange": "completed",
        }[op]
        version = previous.version + 1
        previous_digest = previous.record_digest
        if state == "completed":
            _require_digest(completion_order_digest, "completion_order_digest")
            completion_digest = completion_order_digest
        else:
            if completion_order_digest is not None:
                raise AfterSalesCoreTransitionError(
                    "completion digest is allowed only for completion"
                )
            completion_digest = None
    candidate = build_exchange_case(
        binding,
        case_id=case_id,
        return_receipt_digest=return_chain.receipt.record_digest,
        replacement_item_id=f"replacement:{case_id}",
        replacement_sku_id=replacement_sku,
        state=cast(Any, state),
        version=version,
        previous_digest=previous_digest,
        completion_order_digest=completion_digest,
        reason=normalized["reason"],
        actor_id=actor,
        logical_tick=tick,
        idempotency_key=key,
    )
    apply_exchange_case(
        previous,
        candidate,
        authority=authority.protocol_authority,
        server_tick=tick,
    )
    return candidate


def derive_dispute_open(
    intent: Mapping[str, Any],
    *,
    binding: AfterSalesOrderBinding,
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
) -> CoreDisputeBundle:
    normalized = _intent(intent, "open_dispute", binding)
    _validate_core_authority(binding, policy, authority)
    actor = _text(original_actor, "original_actor")
    if actor not in {binding.owner_id, binding.merchant_id}:
        raise WriteNotAuthorized("only an order party may open dispute")
    against = binding.merchant_id if actor == binding.owner_id else binding.owner_id
    case = build_dispute_case(
        binding,
        dispute_id=_derived_id("dispute", binding, actor, idempotency_key),
        filed_by_id=actor,
        against_id=against,
        reason=normalized["reason"],
        state="open",
        version=1,
        previous_digest=None,
        evidence_digests=(),
        ruling_digest=None,
        actor_id=actor,
        logical_tick=_nonnegative_int(server_tick, "server_tick"),
        idempotency_key=_text(idempotency_key, "idempotency_key"),
    )
    _, bundle = apply_dispute_record(
        None,
        case,
        authority=authority.protocol_authority,
        server_tick=server_tick,
    )
    return CoreDisputeBundle(bundle.case)


def derive_dispute_evidence(
    intent: Mapping[str, Any],
    *,
    bundle: CoreDisputeBundle,
    evidence_record: EvidenceRecord,
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
) -> CoreDisputeBundle:
    binding = bundle.case.binding
    normalized = _intent(intent, "submit_dispute_evidence", binding)
    _validate_core_authority(binding, policy, authority)
    if normalized["dispute_id"] != bundle.case.dispute_id:
        raise AfterSalesCoreTransitionError("dispute reference mismatch")
    actor = _text(original_actor, "original_actor")
    if actor not in authority.protocol_authority.evidence_submitter_ids:
        raise WriteNotAuthorized("actor cannot submit dispute evidence")
    validate_evidence_record(evidence_record)
    if normalized["evidence_id"] != evidence_record.record_id:
        raise AfterSalesCoreTransitionError("trusted evidence reference mismatch")
    if evidence_record.subject_id != binding.order_id:
        raise AfterSalesCoreTransitionError("evidence is not bound to this order")
    if evidence_record.owner_id != actor:
        raise WriteNotAuthorized(
            "dispute evidence must be owned by the submitting actor"
        )
    evidence = build_dispute_evidence(
        bundle.case,
        evidence_id=evidence_record.record_id,
        evidence_kind=evidence_record.kind,
        facts=cast(Mapping[str, Any], evidence_record.facts),
        actor_id=actor,
        logical_tick=_nonnegative_int(server_tick, "server_tick"),
        idempotency_key=_text(idempotency_key, "idempotency_key"),
        source_version=evidence_record.version,
        source_digest=evidence_record.record_digest,
    )
    passive = DisputeBundle(bundle.case, bundle.evidence, bundle.ruling)
    _, updated = apply_dispute_record(
        passive,
        evidence,
        authority=authority.protocol_authority,
        server_tick=server_tick,
    )
    return CoreDisputeBundle(
        updated.case, updated.evidence, bundle.responses, updated.ruling
    )


def derive_dispute_response(
    intent: Mapping[str, Any],
    *,
    bundle: CoreDisputeBundle,
    evidence_records: Iterable[EvidenceRecord],
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
) -> CoreDisputeBundle:
    binding = bundle.case.binding
    normalized = _intent(intent, "respond_to_dispute", binding)
    _validate_core_authority(binding, policy, authority)
    if normalized["dispute_id"] != bundle.case.dispute_id:
        raise AfterSalesCoreTransitionError("dispute reference mismatch")
    actor = _text(original_actor, "original_actor")
    if actor != bundle.case.against_id:
        raise WriteNotAuthorized("only the opposing party may file dispute response")
    if bundle.case.state not in {"open", "under_review"}:
        raise AfterSalesCoreTransitionError("dispute does not accept responses")
    tick = _nonnegative_int(server_tick, "server_tick")
    if tick < bundle.case.logical_tick:
        raise AfterSalesCoreTransitionError("dispute response time regressed")
    key = _text(idempotency_key, "idempotency_key")
    evidence = tuple(evidence_records)
    referenced_ids = cast(tuple[str, ...], normalized["evidence_ids"])
    if tuple(record.record_id for record in evidence) != referenced_ids:
        raise AfterSalesCoreTransitionError(
            "trusted response evidence does not match intent references"
        )
    for record in evidence:
        validate_evidence_record(record)
        if record.subject_id != binding.order_id:
            raise AfterSalesCoreTransitionError(
                "response evidence is not bound to this order"
            )
        if record.owner_id != actor:
            raise WriteNotAuthorized(
                "dispute response evidence must be owned by the responding actor"
            )
    fingerprint = after_sales_intent_fingerprint(
        normalized,
        binding=binding,
        policy=policy,
        original_actor=actor,
        evidence_records=evidence,
    )
    response = cast(
        DisputeResponseRecord,
        _seal_core_record(
            DisputeResponseRecord(
                response_id=_derived_id("dispute-response", binding, actor, key),
                dispute_id=bundle.case.dispute_id,
                dispute_digest=bundle.case.record_digest,
                binding=binding,
                facts=_freeze_object(
                    {
                        "position": normalized["position"],
                        "evidence": [
                            {
                                "record_id": record.record_id,
                                "record_digest": record.record_digest,
                                "kind": record.kind,
                                "issuer_id": record.issuer_id,
                            }
                            for record in evidence
                        ],
                    },
                    "response facts",
                ),
                actor_id=actor,
                logical_tick=tick,
                idempotency_key=key,
                request_fingerprint=fingerprint,
            )
        ),
    )
    validate_dispute_response_transition(
        bundle,
        response,
        authority=authority,
        server_tick=tick,
    )
    next_case = build_dispute_case(
        binding,
        dispute_id=bundle.case.dispute_id,
        filed_by_id=bundle.case.filed_by_id,
        against_id=bundle.case.against_id,
        reason=bundle.case.reason,
        state="under_review",
        version=bundle.case.version + 1,
        previous_digest=bundle.case.record_digest,
        evidence_digests=bundle.case.evidence_digests,
        ruling_digest=None,
        actor_id=actor,
        logical_tick=tick,
        idempotency_key=key,
    )
    return CoreDisputeBundle(
        next_case, bundle.evidence, (*bundle.responses, response), bundle.ruling
    )


def validate_dispute_response_transition(
    bundle: CoreDisputeBundle,
    candidate: DisputeResponseRecord,
    *,
    authority: AfterSalesCoreAuthority,
    server_tick: int,
) -> AppendDisposition:
    """Validate a response against the exact current dispute version."""

    validate_after_sales_record(bundle.case)
    validate_core_after_sales_record(candidate)
    if candidate.binding.binding_digest != authority.binding_digest:
        raise AfterSalesCoreTransitionError("dispute response binding mismatch")
    if bundle.case.state not in {"open", "under_review"}:
        raise AfterSalesCoreTransitionError("dispute does not accept responses")
    if candidate.actor_id != bundle.case.against_id:
        raise WriteNotAuthorized("only the opposing party may file dispute response")
    if (
        candidate.dispute_id != bundle.case.dispute_id
        or candidate.dispute_digest != bundle.case.record_digest
    ):
        raise AfterSalesCoreTransitionError("dispute response causal mismatch")
    if candidate.logical_tick != server_tick or server_tick < bundle.case.logical_tick:
        raise AfterSalesCoreTransitionError("dispute response tick is not World time")
    for existing in bundle.responses:
        if existing.record_digest == candidate.record_digest:
            return "idempotent"
        if (
            existing.response_id == candidate.response_id
            or (
                existing.actor_id == candidate.actor_id
                and existing.idempotency_key == candidate.idempotency_key
            )
        ):
            raise IdempotencyConflict("conflicting dispute response retry")
    return "append"


def derive_dispute_ruling(
    intent: Mapping[str, Any],
    *,
    bundle: CoreDisputeBundle,
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
) -> CoreDisputeBundle:
    binding = bundle.case.binding
    op = _operation(
        intent, {"rule_for_filer", "rule_for_respondent", "rule_split"}
    )
    normalized = _intent(intent, op, binding)
    _validate_core_authority(binding, policy, authority)
    if normalized["dispute_id"] != bundle.case.dispute_id:
        raise AfterSalesCoreTransitionError("dispute reference mismatch")
    actor = _text(original_actor, "original_actor")
    if actor not in authority.protocol_authority.adjudicator_ids:
        raise WriteNotAuthorized("only configured adjudicator may rule")
    if op == "rule_for_filer":
        winner = bundle.case.filed_by_id
        outcome = "claim_upheld"
        refund_amount = binding.amount if winner == binding.owner_id else 0
    elif op == "rule_for_respondent":
        winner = bundle.case.against_id
        outcome = "claim_denied"
        refund_amount = 0
    else:
        winner = binding.owner_id
        outcome = "split"
        refund_amount = _prorated_amount(binding, binding.qty, policy.split_refund_bps)
    ruling = build_ruling(
        bundle.case,
        ruling_id=_derived_id(
            "dispute-ruling", binding, original_actor, idempotency_key
        ),
        winner_id=winner,
        outcome=cast(Any, outcome),
        refund_amount=refund_amount,
        rationale=normalized["rationale"],
        actor_id=actor,
        logical_tick=_nonnegative_int(server_tick, "server_tick"),
        idempotency_key=_text(idempotency_key, "idempotency_key"),
    )
    passive = DisputeBundle(bundle.case, bundle.evidence, bundle.ruling)
    _, updated = apply_dispute_record(
        passive,
        ruling,
        authority=authority.protocol_authority,
        server_tick=server_tick,
    )
    return CoreDisputeBundle(
        updated.case, bundle.evidence, bundle.responses, updated.ruling
    )


def derive_ledger_reconciliation_request(
    intent: Mapping[str, Any],
    *,
    binding: AfterSalesOrderBinding,
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
) -> LedgerReconciliationRequest:
    normalized = _intent(intent, "request_ledger_reconciliation", binding)
    _validate_core_authority(binding, policy, authority)
    actor = _text(original_actor, "original_actor")
    if actor not in authority.ledger_requester_ids:
        raise WriteNotAuthorized("actor cannot request ledger reconciliation")
    key = _text(idempotency_key, "idempotency_key")
    fingerprint = after_sales_intent_fingerprint(
        normalized, binding=binding, policy=policy, original_actor=actor
    )
    return cast(
        LedgerReconciliationRequest,
        _seal_core_record(
            LedgerReconciliationRequest(
                request_id=_derived_id("ledger-request", binding, actor, key),
                binding=binding,
                actor_id=actor,
                reason=normalized["reason"],
                logical_tick=_nonnegative_int(server_tick, "server_tick"),
                idempotency_key=key,
                request_fingerprint=fingerprint,
            )
        ),
    )


def derive_ledger_reconciliation_source(
    receipt: Receipt,
    *,
    binding: AfterSalesOrderBinding,
    effect: LedgerEffect,
    server_tick: int,
) -> LedgerReconciliationSource:
    """Seal a typed ledger effect inside World, never from an actor intent.

    ``Receipt.effect`` is authoritative World state.  The caller passes that
    value explicitly so this helper can assert the typed ledger row and the
    requested projection agree.  No transaction-id prefix is interpreted as
    financial authority.
    """

    validate_after_sales_order_binding(binding)
    if not isinstance(receipt, Receipt):
        raise AfterSalesCoreTransitionError("ledger source must be a Receipt")
    if effect not in {"charge", "refund"}:
        raise AfterSalesCoreTransitionError("unsupported trusted ledger effect")
    if receipt.effect != effect:
        raise AfterSalesCoreTransitionError("ledger effect does not match receipt")
    if (
        str(receipt.order_id) != binding.order_id
        or str(receipt.buyer_id) != binding.owner_id
        or str(receipt.merchant_id) != binding.merchant_id
        or str(receipt.sku_id) != binding.sku_id
        or receipt.price.currency != binding.currency
    ):
        raise AfterSalesCoreTransitionError("ledger row identity mismatch")
    qty = _positive_int(receipt.qty, "ledger qty")
    amount = _money_minor_units(receipt.price.amount, qty)
    tick = _nonnegative_int(server_tick, "server_tick")
    receipt_digest = authoritative_receipt_digest(receipt)
    unsigned = LedgerReconciliationSource(
        source_id=f"ledger-source:{receipt_digest[:32]}",
        binding=binding,
        txn_id=str(receipt.txn_id),
        effect=effect,
        qty=qty,
        amount=amount,
        currency=receipt.price.currency,
        receipt_digest=receipt_digest,
        logical_tick=tick,
    )
    sealed = _replace_ledger_source_digest(
        unsigned, _digest(_ledger_source_contract(unsigned))
    )
    validate_ledger_reconciliation_source(sealed)
    return sealed


def validate_ledger_reconciliation_source(
    source: LedgerReconciliationSource,
) -> None:
    if not isinstance(source, LedgerReconciliationSource):
        raise AfterSalesCoreError("ledger source must be LedgerReconciliationSource")
    if source.schema_id != LEDGER_RECONCILIATION_SOURCE_SCHEMA:
        raise AfterSalesCoreError("unsupported ledger source schema")
    validate_after_sales_order_binding(source.binding)
    _text(source.source_id, "source_id")
    _text(source.txn_id, "txn_id")
    if source.effect not in {"charge", "refund"}:
        raise AfterSalesCoreError("unsupported ledger effect")
    _positive_int(source.qty, "qty")
    _nonnegative_int(source.amount, "amount")
    if source.currency != source.binding.currency:
        raise AfterSalesCoreError("ledger source currency mismatch")
    if _CURRENCY_RE.fullmatch(source.currency) is None:
        raise AfterSalesCoreError("invalid ledger source currency")
    _require_digest(source.receipt_digest, "receipt_digest")
    _nonnegative_int(source.logical_tick, "logical_tick")
    _require_digest(source.source_digest, "source_digest")
    if source.source_digest != _digest(_ledger_source_contract(source)):
        raise AfterSalesCoreError("ledger source digest mismatch")


def derive_ledger_reconciliation_result(
    request: LedgerReconciliationRequest,
    *,
    ledger_sources: Iterable[LedgerReconciliationSource],
    authority: AfterSalesCoreAuthority,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
) -> LedgerReconciliationResult:
    validate_core_after_sales_record(request)
    actor = _text(original_actor, "original_actor")
    if actor not in authority.ledger_reconciler_ids:
        raise WriteNotAuthorized("only platform accounting may reconcile ledger")
    if request.binding.binding_digest != authority.binding_digest:
        raise AfterSalesCoreTransitionError("ledger request binding mismatch")
    rows = tuple(ledger_sources)
    entries: list[dict[str, Any]] = []
    gross = 0
    refunds = 0
    seen: set[str] = set()
    for source in rows:
        validate_ledger_reconciliation_source(source)
        if source.binding.binding_digest != request.binding.binding_digest:
            raise AfterSalesCoreTransitionError("ledger source binding mismatch")
        if source.logical_tick > server_tick:
            raise AfterSalesCoreTransitionError("ledger source is from the future")
        txn_id = source.txn_id
        if txn_id in seen:
            raise AfterSalesCoreTransitionError("duplicate ledger transaction")
        seen.add(txn_id)
        if source.effect == "refund":
            refunds += source.amount
        else:
            gross += source.amount
        entries.append(
            {
                "txn_id": txn_id,
                "effect": source.effect,
                "qty": source.qty,
                "amount": source.amount,
                "currency": source.currency,
                "source_digest": source.source_digest,
            }
        )
    entries.sort(key=lambda row: cast(str, row["txn_id"]))
    if gross <= 0:
        raise AfterSalesCoreTransitionError("reconciliation has no charge receipt")
    if refunds > gross:
        raise AfterSalesCoreTransitionError("ledger refunds exceed charges")
    key = _text(idempotency_key, "idempotency_key")
    fingerprint = _digest(
        {
            "schema_id": AFTER_SALES_INTENT_SCHEMA,
            "op": "complete_ledger_reconciliation",
            "actor_id": actor,
            "request_digest": request.record_digest,
            "source": entries,
        }
    )
    unsigned = LedgerReconciliationResult(
        result_id=_derived_id("ledger-result", request.binding, actor, key),
        request_id=request.request_id,
        request_digest=request.record_digest,
        binding=request.binding,
        source_txn_ids=tuple(cast(str, row["txn_id"]) for row in entries),
        source_digest=_digest(entries),
        gross_amount=gross,
        refund_amount=refunds,
        net_amount=gross - refunds,
        currency=request.binding.currency,
        actor_id=actor,
        logical_tick=_nonnegative_int(server_tick, "server_tick"),
        idempotency_key=key,
        request_fingerprint=fingerprint,
    )
    result = cast(LedgerReconciliationResult, _seal_core_record(unsigned))
    validate_ledger_reconciliation_transition(
        request,
        None,
        result,
        authority=authority,
        server_tick=server_tick,
    )
    return result


def validate_ledger_reconciliation_transition(
    request: LedgerReconciliationRequest,
    existing: LedgerReconciliationResult | None,
    candidate: LedgerReconciliationResult,
    *,
    authority: AfterSalesCoreAuthority,
    server_tick: int,
) -> AppendDisposition:
    """Bind a derived result to its request, accounting actor, and World tick."""

    validate_core_after_sales_record(request)
    validate_core_after_sales_record(candidate)
    if (
        request.binding.binding_digest != authority.binding_digest
        or candidate.binding.binding_digest != authority.binding_digest
    ):
        raise AfterSalesCoreTransitionError("ledger reconciliation binding mismatch")
    if candidate.actor_id not in authority.ledger_reconciler_ids:
        raise WriteNotAuthorized("only platform accounting may reconcile ledger")
    if (
        candidate.request_id != request.request_id
        or candidate.request_digest != request.record_digest
    ):
        raise AfterSalesCoreTransitionError("ledger result causal mismatch")
    if candidate.logical_tick != server_tick or server_tick < request.logical_tick:
        raise AfterSalesCoreTransitionError("ledger result tick is not World time")
    if existing is None:
        return "append"
    validate_core_after_sales_record(existing)
    if existing.record_digest == candidate.record_digest:
        return "idempotent"
    if (
        existing.result_id == candidate.result_id
        or (
            existing.actor_id == candidate.actor_id
            and existing.idempotency_key == candidate.idempotency_key
        )
    ):
        raise IdempotencyConflict("conflicting ledger reconciliation retry")
    raise AfterSalesCoreTransitionError("ledger request already has a result")


CoreRecord: TypeAlias = (
    PaidCancellationRecord
    | DisputeResponseRecord
    | LedgerReconciliationRequest
    | LedgerReconciliationResult
)


def validate_core_after_sales_record(record: CoreRecord) -> None:
    if not isinstance(
        record,
        (
            PaidCancellationRecord,
            DisputeResponseRecord,
            LedgerReconciliationRequest,
            LedgerReconciliationResult,
        ),
    ):
        raise AfterSalesCoreError("unsupported core after-sales record")
    validate_after_sales_order_binding(record.binding)
    _text(record.actor_id, "actor_id")
    _nonnegative_int(record.logical_tick, "logical_tick")
    _text(record.idempotency_key, "idempotency_key")
    _require_digest(record.request_fingerprint, "request_fingerprint")
    if isinstance(record, PaidCancellationRecord):
        if record.schema_id != PAID_CANCELLATION_SCHEMA:
            raise AfterSalesCoreError("unsupported paid cancellation schema")
        _text(record.cancellation_id, "cancellation_id")
        _text(record.reason, "reason")
        _require_digest(record.policy_digest, "policy_digest")
        if record.policy_digest != record.binding.policy_digest:
            raise AfterSalesCoreError("cancellation policy binding mismatch")
        if record.payment_state not in {"authorized", "captured"}:
            raise AfterSalesCoreError("unsupported payment state")
        _require_digest(record.payment_digest, "payment_digest")
        if record.payment_state == "authorized":
            if record.fulfillment_stage != "authorized" or record.packing_digest is not None:
                raise AfterSalesCoreError("authorized cancellation packing mismatch")
        else:
            if record.fulfillment_stage not in {"created", "packed"}:
                raise AfterSalesCoreError("captured cancellation packing stage mismatch")
            _require_digest(record.packing_digest, "packing_digest")
        expected_effect = "void" if record.payment_state == "authorized" else "refund"
        if record.financial_effect != expected_effect:
            raise AfterSalesCoreError("cancellation financial effect mismatch")
        expected_refund = record.binding.amount if expected_effect == "refund" else 0
        if record.refund_amount != expected_refund:
            raise AfterSalesCoreError("cancellation refund is not World-derived")
        if record.inventory_release_qty != record.binding.qty:
            raise AfterSalesCoreError("cancellation inventory release mismatch")
    elif isinstance(record, DisputeResponseRecord):
        if record.schema_id != DISPUTE_RESPONSE_SCHEMA:
            raise AfterSalesCoreError("unsupported dispute response schema")
        _text(record.response_id, "response_id")
        _text(record.dispute_id, "dispute_id")
        _require_digest(record.dispute_digest, "dispute_digest")
        _freeze_object(record.facts, "facts")
    elif isinstance(record, LedgerReconciliationRequest):
        if record.schema_id != LEDGER_RECONCILIATION_REQUEST_SCHEMA:
            raise AfterSalesCoreError("unsupported reconciliation request schema")
        _text(record.request_id, "request_id")
        _text(record.reason, "reason")
    else:
        if record.schema_id != LEDGER_RECONCILIATION_RESULT_SCHEMA:
            raise AfterSalesCoreError("unsupported reconciliation result schema")
        _text(record.result_id, "result_id")
        _text(record.request_id, "request_id")
        _require_digest(record.request_digest, "request_digest")
        _require_digest(record.source_digest, "source_digest")
        if record.source_txn_ids != tuple(sorted(set(record.source_txn_ids))):
            raise AfterSalesCoreError("source transaction ids must be unique and sorted")
        if not record.source_txn_ids:
            raise AfterSalesCoreError("reconciliation result requires ledger sources")
        for txn_id in record.source_txn_ids:
            _text(txn_id, "source_txn_id")
        for name in ("gross_amount", "refund_amount", "net_amount"):
            _nonnegative_int(getattr(record, name), name)
        if record.net_amount != record.gross_amount - record.refund_amount:
            raise AfterSalesCoreError("ledger net amount mismatch")
        if record.refund_amount > record.gross_amount:
            raise AfterSalesCoreError("ledger refund exceeds gross amount")
        if record.currency != record.binding.currency:
            raise AfterSalesCoreError("ledger result currency mismatch")
        if _CURRENCY_RE.fullmatch(record.currency) is None:
            raise AfterSalesCoreError("invalid reconciliation currency")
    _require_digest(record.record_digest, "record_digest")
    if record.record_digest != _digest(_core_record_contract(record)):
        raise AfterSalesCoreError("core after-sales record digest mismatch")


def core_after_sales_record_to_dict(record: CoreRecord) -> dict[str, Any]:
    validate_core_after_sales_record(record)
    return {**_core_record_contract(record), "record_digest": record.record_digest}


def core_after_sales_record_from_dict(value: Mapping[str, Any]) -> CoreRecord:
    if not isinstance(value, Mapping):
        raise AfterSalesCoreError("core after-sales payload must be an object")
    schema = value.get("schema_id")
    types: dict[str, type[CoreRecord]] = {
        PAID_CANCELLATION_SCHEMA: PaidCancellationRecord,
        DISPUTE_RESPONSE_SCHEMA: DisputeResponseRecord,
        LEDGER_RECONCILIATION_REQUEST_SCHEMA: LedgerReconciliationRequest,
        LEDGER_RECONCILIATION_RESULT_SCHEMA: LedgerReconciliationResult,
    }
    row_type = types.get(cast(Any, schema))
    if row_type is None:
        raise AfterSalesCoreError("unsupported core after-sales payload schema")
    binding_value = value.get("binding")
    if not isinstance(binding_value, Mapping):
        raise AfterSalesCoreError("core after-sales payload has no binding")
    payload = dict(value)
    payload["binding"] = after_sales_binding_from_json(_canonical_json(binding_value))
    if row_type is DisputeResponseRecord:
        payload["facts"] = _freeze_object(payload.get("facts"), "facts")
    elif row_type is LedgerReconciliationResult:
        source_ids = payload.get("source_txn_ids")
        if not isinstance(source_ids, (list, tuple)):
            raise AfterSalesCoreError("source_txn_ids must be an array")
        payload["source_txn_ids"] = tuple(source_ids)
    try:
        record = row_type(**cast(Any, payload))
    except TypeError as exc:
        raise AfterSalesCoreError(
            "core after-sales payload fields are not exact"
        ) from exc
    validate_core_after_sales_record(record)
    return record


def ledger_reconciliation_source_to_dict(
    source: LedgerReconciliationSource,
) -> dict[str, Any]:
    validate_ledger_reconciliation_source(source)
    return {**_ledger_source_contract(source), "source_digest": source.source_digest}


def ledger_reconciliation_source_from_dict(
    value: Mapping[str, Any],
) -> LedgerReconciliationSource:
    if not isinstance(value, Mapping):
        raise AfterSalesCoreError("ledger source payload must be an object")
    binding_value = value.get("binding")
    if not isinstance(binding_value, Mapping):
        raise AfterSalesCoreError("ledger source payload has no binding")
    payload = dict(value)
    payload["binding"] = after_sales_binding_from_json(_canonical_json(binding_value))
    try:
        source = LedgerReconciliationSource(**cast(Any, payload))
    except TypeError as exc:
        raise AfterSalesCoreError("ledger source payload fields are not exact") from exc
    validate_ledger_reconciliation_source(source)
    return source


def _seal_core_record(record: CoreRecord) -> CoreRecord:
    unsigned = _replace_core_record_digest(record, "")
    digest = _digest(_core_record_contract(unsigned))
    sealed = _replace_core_record_digest(unsigned, digest)
    validate_core_after_sales_record(sealed)
    return sealed


def _core_record_contract(record: CoreRecord) -> dict[str, Any]:
    common = {
        "schema_id": record.schema_id,
        "binding": after_sales_binding_to_dict(record.binding),
        "actor_id": record.actor_id,
        "logical_tick": record.logical_tick,
        "idempotency_key": record.idempotency_key,
        "request_fingerprint": record.request_fingerprint,
    }
    if isinstance(record, PaidCancellationRecord):
        return {
            **common,
            "cancellation_id": record.cancellation_id,
            "policy_digest": record.policy_digest,
            "reason": record.reason,
            "payment_state": record.payment_state,
            "payment_digest": record.payment_digest,
            "fulfillment_stage": record.fulfillment_stage,
            "packing_digest": record.packing_digest,
            "financial_effect": record.financial_effect,
            "refund_amount": record.refund_amount,
            "inventory_release_qty": record.inventory_release_qty,
        }
    if isinstance(record, DisputeResponseRecord):
        return {
            **common,
            "response_id": record.response_id,
            "dispute_id": record.dispute_id,
            "dispute_digest": record.dispute_digest,
            "facts": _thaw_json(record.facts),
        }
    if isinstance(record, LedgerReconciliationRequest):
        return {
            **common,
            "request_id": record.request_id,
            "reason": record.reason,
        }
    return {
        **common,
        "result_id": record.result_id,
        "request_id": record.request_id,
        "request_digest": record.request_digest,
        "source_txn_ids": list(record.source_txn_ids),
        "source_digest": record.source_digest,
        "gross_amount": record.gross_amount,
        "refund_amount": record.refund_amount,
        "net_amount": record.net_amount,
        "currency": record.currency,
    }


def _replace_core_record_digest(record: CoreRecord, digest: str) -> CoreRecord:
    values = dict(record.__dict__) if hasattr(record, "__dict__") else {
        field: getattr(record, field)
        for field in record.__dataclass_fields__
    }
    values["record_digest"] = digest
    return type(record)(**values)


def _ledger_source_contract(source: LedgerReconciliationSource) -> dict[str, Any]:
    return {
        "schema_id": source.schema_id,
        "source_id": source.source_id,
        "binding": after_sales_binding_to_dict(source.binding),
        "txn_id": source.txn_id,
        "effect": source.effect,
        "qty": source.qty,
        "amount": source.amount,
        "currency": source.currency,
        "receipt_digest": source.receipt_digest,
        "logical_tick": source.logical_tick,
    }


def _replace_ledger_source_digest(
    source: LedgerReconciliationSource, digest: str
) -> LedgerReconciliationSource:
    return LedgerReconciliationSource(
        source_id=source.source_id,
        binding=source.binding,
        txn_id=source.txn_id,
        effect=source.effect,
        qty=source.qty,
        amount=source.amount,
        currency=source.currency,
        receipt_digest=source.receipt_digest,
        logical_tick=source.logical_tick,
        source_digest=digest,
        schema_id=source.schema_id,
    )


def _policy_contract(policy: AfterSalesPolicyRevision) -> dict[str, Any]:
    return {
        "schema_id": policy.schema_id,
        "policy_id": policy.policy_id,
        "merchant_id": policy.merchant_id,
        "revision": policy.revision,
        "previous_digest": policy.previous_digest,
        "effective_tick": policy.effective_tick,
        "published_by_id": policy.published_by_id,
        "return_window_ticks": policy.return_window_ticks,
        "max_refund_bps": policy.max_refund_bps,
        "split_refund_bps": policy.split_refund_bps,
        "owner_paid_cancel_allowed": policy.owner_paid_cancel_allowed,
        "merchant_paid_cancel_allowed": policy.merchant_paid_cancel_allowed,
        "allowed_return_conditions": list(policy.allowed_return_conditions),
        "return_authorizer_ids": list(policy.return_authorizer_ids),
        "return_receiver_ids": list(policy.return_receiver_ids),
        "refund_decider_ids": list(policy.refund_decider_ids),
        "exchange_authorizer_ids": list(policy.exchange_authorizer_ids),
        "adjudicator_ids": list(policy.adjudicator_ids),
        "evidence_service_ids": list(policy.evidence_service_ids),
        "ledger_requester_ids": list(policy.ledger_requester_ids),
        "ledger_reconciler_ids": list(policy.ledger_reconciler_ids),
    }


def _replace_policy_digest(
    policy: AfterSalesPolicyRevision, digest: str
) -> AfterSalesPolicyRevision:
    return AfterSalesPolicyRevision(
        **{
            **{
                field: getattr(policy, field)
                for field in policy.__dataclass_fields__
                if field != "policy_digest"
            },
            "policy_digest": digest,
        }
    )


def _refund_source(
    source: RefundSource,
    binding: AfterSalesOrderBinding,
) -> tuple[str, str, int]:
    if isinstance(source, ReturnChain):
        if source.receipt is None or source.authorization is None:
            raise AfterSalesCoreTransitionError("refund return chain is incomplete")
        source_binding = source.request.binding
        result = (
            "return_receipt",
            source.receipt.record_digest,
            source.authorization.authorized_amount,
        )
    elif isinstance(source, CoreDisputeBundle):
        if source.ruling is None:
            raise AfterSalesCoreTransitionError("refund dispute has no ruling")
        source_binding = source.case.binding
        result = ("dispute_ruling", source.ruling.record_digest, source.ruling.refund_amount)
    elif isinstance(source, PaidCancellationRecord):
        validate_core_after_sales_record(source)
        source_binding = source.binding
        result = ("paid_cancellation", source.record_digest, source.refund_amount)
    else:
        raise AfterSalesCoreTransitionError("unsupported refund causal source")
    if source_binding.binding_digest != binding.binding_digest:
        raise AfterSalesCoreTransitionError("refund causal binding mismatch")
    return result


def _validate_authority_context(
    binding: AfterSalesOrderBinding, policy: AfterSalesPolicyRevision
) -> None:
    validate_after_sales_order_binding(binding)
    validate_after_sales_policy(policy)
    if (
        binding.policy_digest != policy.policy_digest
        or binding.policy_revision != policy.revision
        or binding.merchant_id != policy.merchant_id
    ):
        raise AfterSalesAuthorityError("binding/policy context mismatch")


def _validate_core_authority(
    binding: AfterSalesOrderBinding,
    policy: AfterSalesPolicyRevision,
    authority: AfterSalesCoreAuthority,
) -> None:
    _validate_authority_context(binding, policy)
    if not isinstance(authority, AfterSalesCoreAuthority):
        raise AfterSalesAuthorityError("authority must be World-derived")
    if (
        authority.binding_digest != binding.binding_digest
        or authority.policy_digest != policy.policy_digest
        or authority.owner_id != binding.owner_id
        or authority.merchant_id != binding.merchant_id
        or authority.protocol_authority.binding_digest != binding.binding_digest
    ):
        raise AfterSalesAuthorityError("World after-sales authority mismatch")


def _intent(
    value: Mapping[str, Any], op: str, binding: AfterSalesOrderBinding
) -> AfterSalesIntent:
    normalized = normalize_after_sales_intent(value)
    if normalized["op"] != op:
        raise AfterSalesIntentError(f"expected {op} intent")
    _require_order(normalized, binding)
    return normalized


def _operation(value: Mapping[str, Any], allowed: set[str]) -> str:
    if not isinstance(value, Mapping):
        raise AfterSalesIntentError("after-sales intent must be an object")
    op = value.get("op")
    if not isinstance(op, str) or op not in allowed:
        raise AfterSalesIntentError(
            "unexpected operation; expected one of " + ", ".join(sorted(allowed))
        )
    return op


def _require_order(intent: Mapping[str, Any], binding: AfterSalesOrderBinding) -> None:
    if intent.get("order_id") != binding.order_id:
        raise AfterSalesCoreTransitionError("intent order does not match binding")


def _derived_id(
    prefix: str,
    binding: AfterSalesOrderBinding,
    actor_id: str,
    idempotency_key: str,
) -> str:
    validate_after_sales_order_binding(binding)
    actor = _text(actor_id, "actor_id")
    key = _text(idempotency_key, "idempotency_key")
    suffix = _digest(
        {
            "prefix": prefix,
            "binding_digest": binding.binding_digest,
            "actor_id": actor,
            "idempotency_key": key,
        }
    )[:32]
    return f"{prefix}:{suffix}"


def _prorated_amount(
    binding: AfterSalesOrderBinding, qty: int, refund_bps: int
) -> int:
    quantity = _positive_int(qty, "qty")
    if quantity > binding.qty:
        raise AfterSalesCoreTransitionError("quantity exceeds paid binding")
    bps = _basis_points(refund_bps, "refund_bps")
    return (binding.amount * quantity * bps) // (binding.qty * 10_000)


def _money_minor_units(unit_amount: Decimal, qty: int) -> int:
    if not isinstance(unit_amount, Decimal) or not unit_amount.is_finite():
        raise AfterSalesCoreError("money amount must be finite Decimal")
    if unit_amount < 0:
        raise AfterSalesCoreError("money amount cannot be negative")
    quantity = _positive_int(qty, "qty")
    return int(
        (unit_amount * quantity * Decimal("100")).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )


def _decimal_string(value: Decimal) -> str:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise AfterSalesCoreError("money amount must be finite Decimal")
    return format(value, "f")


def _reject_nested_authority(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise AfterSalesIntentError(f"{path} must be an object")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise AfterSalesIntentError(f"{path} field names must be strings")
        if key.lower() in _AGENT_FORBIDDEN_FIELDS:
            raise AfterSalesIntentError(f"{path}.{key} is World-owned authority")
        if isinstance(item, Mapping):
            _reject_nested_authority(item, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                if isinstance(child, Mapping):
                    _reject_nested_authority(child, f"{path}.{key}[{index}]")


def _freeze_object(value: Any, path: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise AfterSalesIntentError(f"{path} must be an object")
    return cast(Mapping[str, JsonValue], _freeze_json(value, path))


def _freeze_json(value: Any, path: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AfterSalesIntentError(f"{path} contains non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise AfterSalesIntentError(f"{path} has non-string key")
            result[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise AfterSalesIntentError(f"{path} contains non-JSON value")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_texts(values: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise AfterSalesPolicyError(f"{label} must be an array")
    normalized = tuple(_text(value, label) for value in values)
    if len(normalized) != len(set(normalized)):
        raise AfterSalesPolicyError(f"{label} contains duplicates")
    return tuple(sorted(normalized))


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AfterSalesCoreError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AfterSalesCoreError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AfterSalesCoreError(f"{label} must be a non-negative integer")
    return int(value)


def _basis_points(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result > 10_000:
        raise AfterSalesCoreError(f"{label} cannot exceed 10000")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise AfterSalesCoreError(f"{label} must be boolean")
    return value


def _require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AfterSalesCoreError(f"{label} must be lowercase SHA-256")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AfterSalesCoreError(f"value is not canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "AFTER_SALES_INTENT_SCHEMA",
    "AFTER_SALES_POLICY_SCHEMA",
    "DISPUTE_RESPONSE_SCHEMA",
    "LEDGER_RECONCILIATION_REQUEST_SCHEMA",
    "LEDGER_RECONCILIATION_RESULT_SCHEMA",
    "LEDGER_RECONCILIATION_SOURCE_SCHEMA",
    "PAID_CANCELLATION_SCHEMA",
    "AfterSalesCoreAuthority",
    "AfterSalesCoreError",
    "AfterSalesCoreTransitionError",
    "AfterSalesIntent",
    "AfterSalesIntentError",
    "AfterSalesPolicyError",
    "AfterSalesPolicyRevision",
    "CoreDisputeBundle",
    "DisputeResponseRecord",
    "LedgerReconciliationRequest",
    "LedgerReconciliationResult",
    "LedgerReconciliationSource",
    "PaidCancellationRecord",
    "after_sales_intent_fingerprint",
    "after_sales_merchant",
    "after_sales_owner",
    "authoritative_order_digest",
    "authoritative_receipt_digest",
    "authoritative_shipment_digest",
    "build_after_sales_policy_revision",
    "core_after_sales_record_to_dict",
    "derive_after_sales_authority",
    "derive_after_sales_binding",
    "derive_dispute_evidence",
    "derive_dispute_open",
    "derive_dispute_response",
    "derive_dispute_ruling",
    "derive_exchange_case",
    "derive_ledger_reconciliation_request",
    "derive_ledger_reconciliation_result",
    "derive_ledger_reconciliation_source",
    "derive_paid_cancellation",
    "derive_refund_case",
    "derive_refund_decision",
    "derive_return_authorization",
    "derive_return_receipt",
    "derive_return_request",
    "normalize_after_sales_intent",
    "referenced_after_sales_evidence_ids",
    "validate_after_sales_policy",
    "validate_after_sales_policy_transition",
    "validate_core_after_sales_record",
    "validate_dispute_response_transition",
    "validate_ledger_reconciliation_transition",
    "validate_ledger_reconciliation_source",
    "validate_paid_cancellation_transition",
]
