"""Strict, replay-stable contracts for CommerceWorld after-sales workflows.

This module defines domain objects and pure state transitions.  It does not
register actions, mutate World, call a payment service, or persist anything.
Platform and World integrations must use the trusted order binding and server
logical clock when they later store these records.

The records are deliberately independent of benchmark tasks.  Return,
refund, exchange, and dispute state is bound to one order item, shipment,
owner, merchant, quantity, amount, currency, policy revision, and the exact
authoritative order/shipment digests.  Child records carry causal parent
digests, so a valid record cannot be replayed into another transaction.
Integer monetary amounts are expressed in the currency's minor unit.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, TypeAlias, cast

from protocol.errors import SchemaError


AFTER_SALES_BINDING_SCHEMA = "cwe.after-sales-order-binding.v1"
RETURN_REQUEST_SCHEMA = "cwe.return-request.v1"
RETURN_AUTHORIZATION_SCHEMA = "cwe.return-authorization.v1"
RETURN_RECEIPT_SCHEMA = "cwe.return-receipt.v1"
REFUND_CASE_SCHEMA = "cwe.refund-case.v1"
REFUND_DECISION_SCHEMA = "cwe.refund-decision.v1"
EXCHANGE_CASE_SCHEMA = "cwe.exchange-case.v1"
DISPUTE_EVIDENCE_SCHEMA = "cwe.dispute-evidence.v1"
DISPUTE_CASE_SCHEMA = "cwe.dispute-case.v1"
RULING_SCHEMA = "cwe.ruling.v1"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | tuple["JsonValue", ...]
ReturnOutcome: TypeAlias = Literal["authorized", "denied"]
RefundOutcome: TypeAlias = Literal["approved", "denied"]
ExchangeState: TypeAlias = Literal["requested", "authorized", "completed", "denied"]
DisputeState: TypeAlias = Literal["open", "under_review", "ruled", "withdrawn"]
RulingOutcome: TypeAlias = Literal["claim_upheld", "claim_denied", "split"]
RetryDisposition: TypeAlias = Literal["append", "idempotent"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_RETURN_OUTCOMES = frozenset({"authorized", "denied"})
_REFUND_OUTCOMES = frozenset({"approved", "denied"})
_EXCHANGE_STATES = frozenset({"requested", "authorized", "completed", "denied"})
_DISPUTE_STATES = frozenset({"open", "under_review", "ruled", "withdrawn"})
_RULING_OUTCOMES = frozenset({"claim_upheld", "claim_denied", "split"})


class AfterSalesError(SchemaError):
    """Base error for after-sales protocol contracts."""


class AfterSalesSchemaError(AfterSalesError):
    """A record is not an exact, valid canonical after-sales object."""


class AfterSalesDigestMismatch(AfterSalesSchemaError):
    """A record digest does not match canonical semantic content."""


class AfterSalesAuthorityError(AfterSalesError):
    """The actor or order binding is not authorized by trusted state."""


class AfterSalesTransitionError(AfterSalesError):
    """A valid record is illegal in the current after-sales state."""


class AfterSalesIdempotencyConflict(AfterSalesTransitionError):
    """A stable id or actor-scoped idempotency key has conflicting content."""


@dataclass(frozen=True, slots=True)
class AfterSalesOrderBinding:
    """Authoritative identity and policy snapshot for one purchased line."""

    order_id: str
    item_id: str
    sku_id: str
    shipment_id: str | None
    owner_id: str
    merchant_id: str
    qty: int
    amount: int
    currency: str
    policy_revision: int
    policy_digest: str
    order_digest: str
    shipment_digest: str | None
    binding_digest: str = ""
    schema_id: str = AFTER_SALES_BINDING_SCHEMA


@dataclass(frozen=True, slots=True)
class AfterSalesAuthority:
    """Trusted actor permissions for one bound order item.

    This object must be constructed from protected Platform or World policy,
    never from an untrusted after-sales record.
    """

    binding_digest: str
    owner_id: str
    merchant_id: str
    return_authorizer_ids: tuple[str, ...]
    return_receiver_ids: tuple[str, ...]
    refund_decider_ids: tuple[str, ...]
    exchange_authorizer_ids: tuple[str, ...]
    adjudicator_ids: tuple[str, ...]
    evidence_submitter_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_digest("authority.binding_digest", self.binding_digest)
        _require_text("authority.owner_id", self.owner_id)
        _require_text("authority.merchant_id", self.merchant_id)
        if self.owner_id == self.merchant_id:
            raise AfterSalesAuthorityError("owner and merchant must be distinct")
        for name in (
            "return_authorizer_ids",
            "return_receiver_ids",
            "refund_decider_ids",
            "exchange_authorizer_ids",
            "adjudicator_ids",
            "evidence_submitter_ids",
        ):
            _require_string_tuple(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class ReturnRequest:
    request_id: str
    binding: AfterSalesOrderBinding
    requested_qty: int
    requested_amount: int
    reason: str
    actor_id: str
    logical_tick: int
    idempotency_key: str
    evidence_ids: tuple[str, ...] = ()
    evidence_digests: tuple[str, ...] = ()
    record_digest: str = ""
    schema_id: str = RETURN_REQUEST_SCHEMA


@dataclass(frozen=True, slots=True)
class ReturnAuthorization:
    authorization_id: str
    request_id: str
    request_digest: str
    binding: AfterSalesOrderBinding
    outcome: ReturnOutcome
    authorized_qty: int
    authorized_amount: int
    reason: str
    actor_id: str
    logical_tick: int
    idempotency_key: str
    record_digest: str = ""
    schema_id: str = RETURN_AUTHORIZATION_SCHEMA


@dataclass(frozen=True, slots=True)
class ReturnReceipt:
    receipt_id: str
    request_id: str
    authorization_id: str
    authorization_digest: str
    binding: AfterSalesOrderBinding
    received_qty: int
    condition: str
    actor_id: str
    logical_tick: int
    idempotency_key: str
    record_digest: str = ""
    schema_id: str = RETURN_RECEIPT_SCHEMA


@dataclass(frozen=True, slots=True)
class RefundCase:
    case_id: str
    binding: AfterSalesOrderBinding
    source_kind: str
    causal_digest: str
    requested_amount: int
    reason: str
    actor_id: str
    logical_tick: int
    idempotency_key: str
    record_digest: str = ""
    schema_id: str = REFUND_CASE_SCHEMA


@dataclass(frozen=True, slots=True)
class RefundDecision:
    decision_id: str
    case_id: str
    case_digest: str
    binding: AfterSalesOrderBinding
    outcome: RefundOutcome
    approved_amount: int
    reason: str
    actor_id: str
    logical_tick: int
    idempotency_key: str
    record_digest: str = ""
    schema_id: str = REFUND_DECISION_SCHEMA


@dataclass(frozen=True, slots=True)
class ExchangeCase:
    case_id: str
    binding: AfterSalesOrderBinding
    return_receipt_digest: str
    replacement_item_id: str
    replacement_sku_id: str
    state: ExchangeState
    version: int
    previous_digest: str | None
    completion_order_digest: str | None
    reason: str
    actor_id: str
    logical_tick: int
    idempotency_key: str
    record_digest: str = ""
    schema_id: str = EXCHANGE_CASE_SCHEMA


@dataclass(frozen=True, slots=True)
class DisputeEvidence:
    evidence_id: str
    dispute_id: str
    dispute_digest: str
    binding: AfterSalesOrderBinding
    evidence_kind: str
    facts: Mapping[str, JsonValue]
    actor_id: str
    logical_tick: int
    idempotency_key: str
    source_version: int
    source_digest: str
    record_digest: str = ""
    schema_id: str = DISPUTE_EVIDENCE_SCHEMA


@dataclass(frozen=True, slots=True)
class DisputeCase:
    dispute_id: str
    binding: AfterSalesOrderBinding
    filed_by_id: str
    against_id: str
    reason: str
    state: DisputeState
    version: int
    previous_digest: str | None
    evidence_digests: tuple[str, ...]
    ruling_digest: str | None
    actor_id: str
    logical_tick: int
    idempotency_key: str
    record_digest: str = ""
    schema_id: str = DISPUTE_CASE_SCHEMA


@dataclass(frozen=True, slots=True)
class Ruling:
    ruling_id: str
    dispute_id: str
    dispute_digest: str
    binding: AfterSalesOrderBinding
    winner_id: str
    outcome: RulingOutcome
    refund_amount: int
    rationale: str
    evidence_digests: tuple[str, ...]
    actor_id: str
    logical_tick: int
    idempotency_key: str
    record_digest: str = ""
    schema_id: str = RULING_SCHEMA


AfterSalesRecord: TypeAlias = (
    ReturnRequest
    | ReturnAuthorization
    | ReturnReceipt
    | RefundCase
    | RefundDecision
    | ExchangeCase
    | DisputeEvidence
    | DisputeCase
    | Ruling
)


@dataclass(frozen=True, slots=True)
class ReturnChain:
    request: ReturnRequest
    authorization: ReturnAuthorization | None = None
    receipt: ReturnReceipt | None = None


@dataclass(frozen=True, slots=True)
class RefundChain:
    case: RefundCase
    decision: RefundDecision | None = None


@dataclass(frozen=True, slots=True)
class DisputeBundle:
    case: DisputeCase
    evidence: tuple[DisputeEvidence, ...] = ()
    ruling: Ruling | None = None


def build_after_sales_order_binding(
    *,
    order_id: str,
    item_id: str,
    sku_id: str,
    shipment_id: str | None,
    owner_id: str,
    merchant_id: str,
    qty: int,
    amount: int,
    currency: str,
    policy_revision: int,
    policy_digest: str,
    order_digest: str,
    shipment_digest: str | None,
) -> AfterSalesOrderBinding:
    unsigned = AfterSalesOrderBinding(
        order_id=order_id,
        item_id=item_id,
        sku_id=sku_id,
        shipment_id=shipment_id,
        owner_id=owner_id,
        merchant_id=merchant_id,
        qty=qty,
        amount=amount,
        currency=currency,
        policy_revision=policy_revision,
        policy_digest=policy_digest,
        order_digest=order_digest,
        shipment_digest=shipment_digest,
    )
    _validate_binding_fields(unsigned)
    sealed = _replace_binding_digest(unsigned, _digest(_binding_contract(unsigned)))
    validate_after_sales_order_binding(sealed)
    return sealed


def validate_after_sales_order_binding(binding: AfterSalesOrderBinding) -> None:
    if not isinstance(binding, AfterSalesOrderBinding):
        raise AfterSalesSchemaError("binding must be AfterSalesOrderBinding")
    _validate_binding_fields(binding)
    _require_digest("binding_digest", binding.binding_digest)
    if binding.binding_digest != _digest(_binding_contract(binding)):
        raise AfterSalesDigestMismatch("after-sales binding digest mismatch")


def after_sales_binding_to_dict(binding: AfterSalesOrderBinding) -> dict[str, Any]:
    validate_after_sales_order_binding(binding)
    return {**_binding_contract(binding), "binding_digest": binding.binding_digest}


def after_sales_binding_to_json(binding: AfterSalesOrderBinding) -> str:
    return _canonical_json(after_sales_binding_to_dict(binding))


def after_sales_binding_from_json(payload: str) -> AfterSalesOrderBinding:
    return _coerce_binding(_strict_json_loads(payload, "after-sales binding"))


def build_after_sales_authority(
    binding: AfterSalesOrderBinding,
    *,
    return_authorizer_ids: Iterable[str],
    return_receiver_ids: Iterable[str],
    refund_decider_ids: Iterable[str],
    exchange_authorizer_ids: Iterable[str],
    adjudicator_ids: Iterable[str],
    evidence_submitter_ids: Iterable[str],
) -> AfterSalesAuthority:
    validate_after_sales_order_binding(binding)
    return AfterSalesAuthority(
        binding_digest=binding.binding_digest,
        owner_id=binding.owner_id,
        merchant_id=binding.merchant_id,
        return_authorizer_ids=_canonical_strings(return_authorizer_ids),
        return_receiver_ids=_canonical_strings(return_receiver_ids),
        refund_decider_ids=_canonical_strings(refund_decider_ids),
        exchange_authorizer_ids=_canonical_strings(exchange_authorizer_ids),
        adjudicator_ids=_canonical_strings(adjudicator_ids),
        evidence_submitter_ids=_canonical_strings(evidence_submitter_ids),
    )


def build_return_request(
    binding: AfterSalesOrderBinding,
    *,
    request_id: str,
    requested_qty: int,
    requested_amount: int,
    reason: str,
    actor_id: str,
    logical_tick: int,
    idempotency_key: str,
    evidence_ids: Iterable[str] = (),
    evidence_digests: Iterable[str] = (),
) -> ReturnRequest:
    return cast(
        ReturnRequest,
        _seal_record(
            ReturnRequest(
                request_id=request_id,
                binding=binding,
                requested_qty=requested_qty,
                requested_amount=requested_amount,
                reason=reason,
                actor_id=actor_id,
                logical_tick=logical_tick,
                idempotency_key=idempotency_key,
                evidence_ids=tuple(evidence_ids),
                evidence_digests=tuple(evidence_digests),
            )
        ),
    )


def build_return_authorization(
    request: ReturnRequest,
    *,
    authorization_id: str,
    outcome: ReturnOutcome,
    authorized_qty: int,
    authorized_amount: int,
    reason: str,
    actor_id: str,
    logical_tick: int,
    idempotency_key: str,
) -> ReturnAuthorization:
    validate_after_sales_record(request)
    return cast(
        ReturnAuthorization,
        _seal_record(
            ReturnAuthorization(
                authorization_id=authorization_id,
                request_id=request.request_id,
                request_digest=request.record_digest,
                binding=request.binding,
                outcome=outcome,
                authorized_qty=authorized_qty,
                authorized_amount=authorized_amount,
                reason=reason,
                actor_id=actor_id,
                logical_tick=logical_tick,
                idempotency_key=idempotency_key,
            )
        ),
    )


def build_return_receipt(
    authorization: ReturnAuthorization,
    *,
    receipt_id: str,
    received_qty: int,
    condition: str,
    actor_id: str,
    logical_tick: int,
    idempotency_key: str,
) -> ReturnReceipt:
    validate_after_sales_record(authorization)
    return cast(
        ReturnReceipt,
        _seal_record(
            ReturnReceipt(
                receipt_id=receipt_id,
                request_id=authorization.request_id,
                authorization_id=authorization.authorization_id,
                authorization_digest=authorization.record_digest,
                binding=authorization.binding,
                received_qty=received_qty,
                condition=condition,
                actor_id=actor_id,
                logical_tick=logical_tick,
                idempotency_key=idempotency_key,
            )
        ),
    )


def build_refund_case(
    binding: AfterSalesOrderBinding,
    *,
    case_id: str,
    source_kind: str,
    causal_digest: str,
    requested_amount: int,
    reason: str,
    actor_id: str,
    logical_tick: int,
    idempotency_key: str,
) -> RefundCase:
    return cast(
        RefundCase,
        _seal_record(
            RefundCase(
                case_id=case_id,
                binding=binding,
                source_kind=source_kind,
                causal_digest=causal_digest,
                requested_amount=requested_amount,
                reason=reason,
                actor_id=actor_id,
                logical_tick=logical_tick,
                idempotency_key=idempotency_key,
            )
        ),
    )


def build_refund_decision(
    case: RefundCase,
    *,
    decision_id: str,
    outcome: RefundOutcome,
    approved_amount: int,
    reason: str,
    actor_id: str,
    logical_tick: int,
    idempotency_key: str,
) -> RefundDecision:
    validate_after_sales_record(case)
    return cast(
        RefundDecision,
        _seal_record(
            RefundDecision(
                decision_id=decision_id,
                case_id=case.case_id,
                case_digest=case.record_digest,
                binding=case.binding,
                outcome=outcome,
                approved_amount=approved_amount,
                reason=reason,
                actor_id=actor_id,
                logical_tick=logical_tick,
                idempotency_key=idempotency_key,
            )
        ),
    )


def build_exchange_case(
    binding: AfterSalesOrderBinding,
    *,
    case_id: str,
    return_receipt_digest: str,
    replacement_item_id: str,
    replacement_sku_id: str,
    state: ExchangeState,
    version: int,
    previous_digest: str | None,
    completion_order_digest: str | None,
    reason: str,
    actor_id: str,
    logical_tick: int,
    idempotency_key: str,
) -> ExchangeCase:
    return cast(
        ExchangeCase,
        _seal_record(
            ExchangeCase(
                case_id=case_id,
                binding=binding,
                return_receipt_digest=return_receipt_digest,
                replacement_item_id=replacement_item_id,
                replacement_sku_id=replacement_sku_id,
                state=state,
                version=version,
                previous_digest=previous_digest,
                completion_order_digest=completion_order_digest,
                reason=reason,
                actor_id=actor_id,
                logical_tick=logical_tick,
                idempotency_key=idempotency_key,
            )
        ),
    )


def build_dispute_case(
    binding: AfterSalesOrderBinding,
    *,
    dispute_id: str,
    filed_by_id: str,
    against_id: str,
    reason: str,
    state: DisputeState,
    version: int,
    previous_digest: str | None,
    evidence_digests: Iterable[str],
    ruling_digest: str | None,
    actor_id: str,
    logical_tick: int,
    idempotency_key: str,
) -> DisputeCase:
    return cast(
        DisputeCase,
        _seal_record(
            DisputeCase(
                dispute_id=dispute_id,
                binding=binding,
                filed_by_id=filed_by_id,
                against_id=against_id,
                reason=reason,
                state=state,
                version=version,
                previous_digest=previous_digest,
                evidence_digests=tuple(evidence_digests),
                ruling_digest=ruling_digest,
                actor_id=actor_id,
                logical_tick=logical_tick,
                idempotency_key=idempotency_key,
            )
        ),
    )


def build_dispute_evidence(
    case: DisputeCase,
    *,
    evidence_id: str,
    evidence_kind: str,
    facts: Mapping[str, Any],
    actor_id: str,
    logical_tick: int,
    idempotency_key: str,
    source_version: int,
    source_digest: str,
) -> DisputeEvidence:
    validate_after_sales_record(case)
    return cast(
        DisputeEvidence,
        _seal_record(
            DisputeEvidence(
                evidence_id=evidence_id,
                dispute_id=case.dispute_id,
                dispute_digest=case.record_digest,
                binding=case.binding,
                evidence_kind=evidence_kind,
                facts=_freeze_object(facts, "facts"),
                actor_id=actor_id,
                logical_tick=logical_tick,
                idempotency_key=idempotency_key,
                source_version=source_version,
                source_digest=source_digest,
            )
        ),
    )


def build_ruling(
    case: DisputeCase,
    *,
    ruling_id: str,
    winner_id: str,
    outcome: RulingOutcome,
    refund_amount: int,
    rationale: str,
    actor_id: str,
    logical_tick: int,
    idempotency_key: str,
) -> Ruling:
    validate_after_sales_record(case)
    return cast(
        Ruling,
        _seal_record(
            Ruling(
                ruling_id=ruling_id,
                dispute_id=case.dispute_id,
                dispute_digest=case.record_digest,
                binding=case.binding,
                winner_id=winner_id,
                outcome=outcome,
                refund_amount=refund_amount,
                rationale=rationale,
                evidence_digests=case.evidence_digests,
                actor_id=actor_id,
                logical_tick=logical_tick,
                idempotency_key=idempotency_key,
            )
        ),
    )


def validate_after_sales_record(record: AfterSalesRecord) -> None:
    if not isinstance(record, _RECORD_TYPES):
        raise AfterSalesSchemaError("unsupported after-sales record type")
    _validate_record_fields(record)
    _require_digest("record_digest", record.record_digest)
    if record.record_digest != _digest(_record_contract(record)):
        raise AfterSalesDigestMismatch(f"{type(record).__name__} record digest mismatch")


def after_sales_record_to_dict(record: AfterSalesRecord) -> dict[str, Any]:
    validate_after_sales_record(record)
    return {**_record_contract(record), "record_digest": record.record_digest}


def after_sales_record_to_json(record: AfterSalesRecord) -> str:
    return _canonical_json(after_sales_record_to_dict(record))


def after_sales_record_from_json(payload: str) -> AfterSalesRecord:
    value = _strict_json_loads(payload, "after-sales record")
    if not isinstance(value, Mapping):
        raise AfterSalesSchemaError("after-sales record must be an object")
    schema = value.get("schema_id")
    if not isinstance(schema, str):
        raise AfterSalesSchemaError("after-sales record schema_id must be a string")
    coercer = _SCHEMA_COERCERS.get(schema)
    if coercer is None:
        raise AfterSalesSchemaError(f"unsupported after-sales schema_id: {schema!r}")
    return coercer(value)


def classify_after_sales_retry(
    existing: AfterSalesRecord,
    candidate: AfterSalesRecord,
) -> RetryDisposition:
    validate_after_sales_record(existing)
    validate_after_sales_record(candidate)
    if type(existing) is not type(candidate):
        raise AfterSalesAuthorityError("retry record types do not match")
    same_stable_id = _stable_id(existing) == _stable_id(candidate)
    same_actor_key = (
        existing.binding.binding_digest,
        existing.actor_id,
        existing.idempotency_key,
    ) == (
        candidate.binding.binding_digest,
        candidate.actor_id,
        candidate.idempotency_key,
    )
    if not same_stable_id and not same_actor_key:
        raise AfterSalesAuthorityError("records do not share stable id or actor idempotency scope")
    if existing.record_digest == candidate.record_digest:
        return "idempotent"
    collision = "stable id" if same_stable_id else "actor idempotency key"
    raise AfterSalesIdempotencyConflict(f"conflicting reuse of {collision}")


def apply_return_record(
    chain: ReturnChain | None,
    record: ReturnRequest | ReturnAuthorization | ReturnReceipt,
    *,
    authority: AfterSalesAuthority,
    server_tick: int,
) -> tuple[RetryDisposition, ReturnChain]:
    validate_after_sales_record(record)
    _verify_authority_binding(record.binding, authority)
    _require_nonnegative_int("server_tick", server_tick)
    if chain is not None:
        for existing in (chain.request, chain.authorization, chain.receipt):
            if existing is None or type(existing) is not type(record):
                continue
            try:
                disposition = classify_after_sales_retry(existing, record)
            except AfterSalesAuthorityError:
                continue
            return disposition, chain
    _require_server_tick(record.logical_tick, server_tick)

    if chain is None:
        if not isinstance(record, ReturnRequest):
            raise AfterSalesTransitionError("return lifecycle must start with request")
        if record.actor_id != authority.owner_id:
            raise AfterSalesAuthorityError("only the order owner may request return")
        if record.binding.shipment_id is None:
            raise AfterSalesTransitionError("return requires a bound shipment")
        return "append", ReturnChain(record)

    if chain.receipt is not None:
        raise AfterSalesTransitionError("return lifecycle is terminal after receipt")
    if chain.authorization is None:
        if not isinstance(record, ReturnAuthorization):
            raise AfterSalesTransitionError("return request awaits authorization")
        if record.actor_id not in authority.return_authorizer_ids:
            raise AfterSalesAuthorityError("actor cannot authorize returns")
        if (
            record.request_id != chain.request.request_id
            or record.request_digest != chain.request.record_digest
        ):
            raise AfterSalesTransitionError("return authorization causal mismatch")
        if record.logical_tick < chain.request.logical_tick:
            raise AfterSalesTransitionError("return logical time regressed")
        if record.outcome == "authorized":
            if not 0 < record.authorized_qty <= chain.request.requested_qty:
                raise AfterSalesTransitionError("authorized return quantity is invalid")
            if not 0 <= record.authorized_amount <= chain.request.requested_amount:
                raise AfterSalesTransitionError("authorized return amount is invalid")
        elif record.authorized_qty != 0 or record.authorized_amount != 0:
            raise AfterSalesTransitionError("denied return must authorize zero quantity and amount")
        return "append", ReturnChain(chain.request, record)

    if chain.authorization.outcome == "denied":
        raise AfterSalesTransitionError("denied return is terminal")
    if not isinstance(record, ReturnReceipt):
        raise AfterSalesTransitionError("authorized return awaits physical receipt")
    if record.actor_id not in authority.return_receiver_ids:
        raise AfterSalesAuthorityError("actor cannot receive returns")
    if (
        record.request_id != chain.request.request_id
        or record.authorization_id != chain.authorization.authorization_id
        or record.authorization_digest != chain.authorization.record_digest
    ):
        raise AfterSalesTransitionError("return receipt causal mismatch")
    if record.received_qty > chain.authorization.authorized_qty:
        raise AfterSalesTransitionError("received quantity exceeds authorization")
    if record.logical_tick < chain.authorization.logical_tick:
        raise AfterSalesTransitionError("return logical time regressed")
    return "append", ReturnChain(chain.request, chain.authorization, record)


def replay_return_records(
    records: Iterable[ReturnRequest | ReturnAuthorization | ReturnReceipt],
    *,
    authority: AfterSalesAuthority,
) -> ReturnChain | None:
    chain: ReturnChain | None = None
    for record in records:
        _, chain = apply_return_record(
            chain, record, authority=authority, server_tick=record.logical_tick
        )
    return chain


def apply_refund_record(
    chain: RefundChain | None,
    record: RefundCase | RefundDecision,
    *,
    authority: AfterSalesAuthority,
    server_tick: int,
) -> tuple[RetryDisposition, RefundChain]:
    validate_after_sales_record(record)
    _verify_authority_binding(record.binding, authority)
    _require_nonnegative_int("server_tick", server_tick)
    if chain is not None:
        existing = chain.case if isinstance(record, RefundCase) else chain.decision
        if existing is not None:
            try:
                disposition = classify_after_sales_retry(existing, record)
            except AfterSalesAuthorityError:
                pass
            else:
                return disposition, chain
    _require_server_tick(record.logical_tick, server_tick)
    if chain is None:
        if not isinstance(record, RefundCase):
            raise AfterSalesTransitionError("refund lifecycle must start with case")
        allowed = {authority.owner_id, authority.merchant_id, *authority.refund_decider_ids}
        if record.actor_id not in allowed:
            raise AfterSalesAuthorityError("actor cannot open refund case")
        return "append", RefundChain(record)
    if chain.decision is not None:
        raise AfterSalesTransitionError("refund case already decided")
    if not isinstance(record, RefundDecision):
        raise AfterSalesTransitionError("refund case already exists")
    if record.actor_id not in authority.refund_decider_ids:
        raise AfterSalesAuthorityError("actor cannot decide refunds")
    if record.case_id != chain.case.case_id or record.case_digest != chain.case.record_digest:
        raise AfterSalesTransitionError("refund decision causal mismatch")
    if record.logical_tick < chain.case.logical_tick:
        raise AfterSalesTransitionError("refund logical time regressed")
    if record.outcome == "approved":
        if not 0 < record.approved_amount <= chain.case.requested_amount:
            raise AfterSalesTransitionError("approved refund amount is invalid")
    elif record.approved_amount != 0:
        raise AfterSalesTransitionError("denied refund must approve zero amount")
    return "append", RefundChain(chain.case, record)


def replay_refund_records(
    records: Iterable[RefundCase | RefundDecision],
    *,
    authority: AfterSalesAuthority,
) -> RefundChain | None:
    chain: RefundChain | None = None
    for record in records:
        _, chain = apply_refund_record(
            chain, record, authority=authority, server_tick=record.logical_tick
        )
    return chain


def apply_exchange_case(
    previous: ExchangeCase | None,
    candidate: ExchangeCase,
    *,
    authority: AfterSalesAuthority,
    server_tick: int,
) -> tuple[RetryDisposition, ExchangeCase]:
    validate_after_sales_record(candidate)
    _verify_authority_binding(candidate.binding, authority)
    _require_nonnegative_int("server_tick", server_tick)
    if previous is not None:
        validate_after_sales_record(previous)
        _verify_authority_binding(previous.binding, authority)
        if candidate.record_digest == previous.record_digest:
            return "idempotent", previous
        if candidate.version == previous.version:
            raise AfterSalesIdempotencyConflict("conflicting exchange case version")
        if (
            candidate.actor_id == previous.actor_id
            and candidate.idempotency_key == previous.idempotency_key
        ):
            raise AfterSalesIdempotencyConflict("conflicting exchange idempotency key")
    _require_server_tick(candidate.logical_tick, server_tick)
    if previous is None:
        if (
            candidate.state != "requested"
            or candidate.version != 1
            or candidate.previous_digest is not None
        ):
            raise AfterSalesTransitionError("exchange must start at requested version 1")
        if candidate.actor_id != authority.owner_id:
            raise AfterSalesAuthorityError("only owner may request exchange")
        return "append", candidate
    _verify_exchange_identity(previous, candidate)
    if previous.state in {"completed", "denied"}:
        raise AfterSalesTransitionError("exchange case is terminal")
    if (
        candidate.version != previous.version + 1
        or candidate.previous_digest != previous.record_digest
    ):
        raise AfterSalesTransitionError("exchange version or previous digest mismatch")
    if candidate.logical_tick < previous.logical_tick:
        raise AfterSalesTransitionError("exchange logical time regressed")
    if previous.state == "requested":
        if candidate.state not in {"authorized", "denied"}:
            raise AfterSalesTransitionError("requested exchange awaits authorization")
        if candidate.actor_id not in authority.exchange_authorizer_ids:
            raise AfterSalesAuthorityError("actor cannot authorize exchange")
    elif previous.state == "authorized":
        if candidate.state != "completed":
            raise AfterSalesTransitionError("authorized exchange must complete")
        if candidate.actor_id not in authority.exchange_authorizer_ids:
            raise AfterSalesAuthorityError("actor cannot complete exchange")
    return "append", candidate


def replay_exchange_cases(
    cases: Iterable[ExchangeCase],
    *,
    authority: AfterSalesAuthority,
) -> ExchangeCase | None:
    current: ExchangeCase | None = None
    for case in cases:
        _, current = apply_exchange_case(
            current, case, authority=authority, server_tick=case.logical_tick
        )
    return current


def apply_dispute_record(
    bundle: DisputeBundle | None,
    record: DisputeCase | DisputeEvidence | Ruling,
    *,
    authority: AfterSalesAuthority,
    server_tick: int,
) -> tuple[RetryDisposition, DisputeBundle]:
    validate_after_sales_record(record)
    _verify_authority_binding(record.binding, authority)
    _require_nonnegative_int("server_tick", server_tick)
    if bundle is not None:
        existing_records: tuple[AfterSalesRecord, ...] = (
            bundle.case,
            *bundle.evidence,
            *((bundle.ruling,) if bundle.ruling is not None else ()),
        )
        for existing in existing_records:
            if type(existing) is not type(record):
                continue
            try:
                disposition = classify_after_sales_retry(existing, record)
            except AfterSalesAuthorityError:
                continue
            return disposition, bundle
    _require_server_tick(record.logical_tick, server_tick)

    if bundle is None:
        if not isinstance(record, DisputeCase):
            raise AfterSalesTransitionError("dispute lifecycle must start with case")
        parties = {authority.owner_id, authority.merchant_id}
        if (
            record.state != "open"
            or record.version != 1
            or record.previous_digest is not None
            or record.evidence_digests
            or record.ruling_digest is not None
        ):
            raise AfterSalesTransitionError("new dispute must be empty OPEN version 1")
        if {record.filed_by_id, record.against_id} != parties:
            raise AfterSalesAuthorityError("dispute parties do not match order")
        if record.actor_id != record.filed_by_id:
            raise AfterSalesAuthorityError("filer actor mismatch")
        return "append", DisputeBundle(record)

    case = bundle.case
    if case.state in {"ruled", "withdrawn"}:
        raise AfterSalesTransitionError("dispute is terminal")
    if isinstance(record, DisputeEvidence):
        if record.actor_id not in authority.evidence_submitter_ids:
            raise AfterSalesAuthorityError("actor cannot submit dispute evidence")
        if record.dispute_id != case.dispute_id or record.dispute_digest != case.record_digest:
            raise AfterSalesTransitionError("dispute evidence causal mismatch")
        if record.logical_tick < case.logical_tick:
            raise AfterSalesTransitionError("dispute logical time regressed")
        updated = _advance_dispute_with_evidence(case, record)
        return "append", DisputeBundle(updated, (*bundle.evidence, record))
    if isinstance(record, Ruling):
        if record.actor_id not in authority.adjudicator_ids:
            raise AfterSalesAuthorityError("only configured adjudicator may rule")
        if record.dispute_id != case.dispute_id or record.dispute_digest != case.record_digest:
            raise AfterSalesTransitionError("ruling causal mismatch")
        if record.evidence_digests != case.evidence_digests:
            raise AfterSalesTransitionError("ruling evidence set mismatch")
        if record.winner_id not in {case.filed_by_id, case.against_id}:
            raise AfterSalesTransitionError("ruling winner is not a dispute party")
        if record.logical_tick < case.logical_tick:
            raise AfterSalesTransitionError("ruling logical time regressed")
        ruled = _advance_dispute_with_ruling(case, record)
        return "append", DisputeBundle(ruled, bundle.evidence, record)
    raise AfterSalesTransitionError(
        "direct DisputeCase updates are derived from evidence or ruling records"
    )


def replay_dispute_records(
    records: Iterable[DisputeCase | DisputeEvidence | Ruling],
    *,
    authority: AfterSalesAuthority,
) -> DisputeBundle | None:
    bundle: DisputeBundle | None = None
    for record in records:
        _, bundle = apply_dispute_record(
            bundle, record, authority=authority, server_tick=record.logical_tick
        )
    return bundle


def _advance_dispute_with_evidence(
    case: DisputeCase,
    evidence: DisputeEvidence,
) -> DisputeCase:
    return cast(
        DisputeCase,
        _seal_record(
            DisputeCase(
                dispute_id=case.dispute_id,
                binding=case.binding,
                filed_by_id=case.filed_by_id,
                against_id=case.against_id,
                reason=case.reason,
                state="under_review",
                version=case.version + 1,
                previous_digest=case.record_digest,
                evidence_digests=(*case.evidence_digests, evidence.record_digest),
                ruling_digest=None,
                actor_id=evidence.actor_id,
                logical_tick=evidence.logical_tick,
                idempotency_key=evidence.idempotency_key,
            )
        ),
    )


def _advance_dispute_with_ruling(case: DisputeCase, ruling: Ruling) -> DisputeCase:
    return cast(
        DisputeCase,
        _seal_record(
            DisputeCase(
                dispute_id=case.dispute_id,
                binding=case.binding,
                filed_by_id=case.filed_by_id,
                against_id=case.against_id,
                reason=case.reason,
                state="ruled",
                version=case.version + 1,
                previous_digest=case.record_digest,
                evidence_digests=case.evidence_digests,
                ruling_digest=ruling.record_digest,
                actor_id=ruling.actor_id,
                logical_tick=ruling.logical_tick,
                idempotency_key=ruling.idempotency_key,
            )
        ),
    )


def _seal_record(record: AfterSalesRecord) -> AfterSalesRecord:
    normalized = _replace_record_digest(record, "")
    _validate_record_fields(normalized)
    sealed = _replace_record_digest(normalized, _digest(_record_contract(normalized)))
    validate_after_sales_record(sealed)
    return sealed


def _validate_binding_fields(binding: AfterSalesOrderBinding) -> None:
    if binding.schema_id != AFTER_SALES_BINDING_SCHEMA:
        raise AfterSalesSchemaError("unsupported after-sales binding schema")
    for name in ("order_id", "item_id", "sku_id", "owner_id", "merchant_id"):
        _require_text(name, getattr(binding, name))
    if binding.owner_id == binding.merchant_id:
        raise AfterSalesSchemaError("owner and merchant must be distinct")
    if (binding.shipment_id is None) != (binding.shipment_digest is None):
        raise AfterSalesSchemaError("shipment id and digest must both be present or absent")
    if binding.shipment_id is not None:
        _require_text("shipment_id", binding.shipment_id)
        _require_digest("shipment_digest", binding.shipment_digest)
    _require_positive_int("qty", binding.qty)
    _require_nonnegative_int("amount", binding.amount)
    _require_currency(binding.currency)
    _require_nonnegative_int("policy_revision", binding.policy_revision)
    _require_digest("policy_digest", binding.policy_digest)
    _require_digest("order_digest", binding.order_digest)


def _validate_record_fields(record: AfterSalesRecord) -> None:
    validate_after_sales_order_binding(record.binding)
    _require_text("actor_id", record.actor_id)
    _require_nonnegative_int("logical_tick", record.logical_tick)
    _require_text("idempotency_key", record.idempotency_key)
    expected_schema = _TYPE_SCHEMAS[type(record)]
    if record.schema_id != expected_schema:
        raise AfterSalesSchemaError(
            f"unsupported {type(record).__name__} schema_id: {record.schema_id!r}"
        )
    if isinstance(record, ReturnRequest):
        _require_text("request_id", record.request_id)
        _require_positive_int("requested_qty", record.requested_qty)
        _require_nonnegative_int("requested_amount", record.requested_amount)
        _require_text("reason", record.reason)
        _require_string_tuple("evidence_ids", record.evidence_ids)
        _require_digest_tuple("evidence_digests", record.evidence_digests)
        if len(record.evidence_ids) != len(record.evidence_digests):
            raise AfterSalesSchemaError(
                "return evidence ids and digests must have equal length"
            )
        if (
            record.requested_qty > record.binding.qty
            or record.requested_amount > record.binding.amount
        ):
            raise AfterSalesSchemaError("return request exceeds bound order quantity or amount")
    elif isinstance(record, ReturnAuthorization):
        for name in ("authorization_id", "request_id", "reason"):
            _require_text(name, getattr(record, name))
        _require_digest("request_digest", record.request_digest)
        if record.outcome not in _RETURN_OUTCOMES:
            raise AfterSalesSchemaError("unsupported return authorization outcome")
        _require_nonnegative_int("authorized_qty", record.authorized_qty)
        _require_nonnegative_int("authorized_amount", record.authorized_amount)
    elif isinstance(record, ReturnReceipt):
        for name in ("receipt_id", "request_id", "authorization_id", "condition"):
            _require_text(name, getattr(record, name))
        _require_digest("authorization_digest", record.authorization_digest)
        _require_positive_int("received_qty", record.received_qty)
    elif isinstance(record, RefundCase):
        for name in ("case_id", "source_kind", "reason"):
            _require_text(name, getattr(record, name))
        _require_digest("causal_digest", record.causal_digest)
        _require_nonnegative_int("requested_amount", record.requested_amount)
        if record.requested_amount > record.binding.amount:
            raise AfterSalesSchemaError("refund request exceeds bound order amount")
    elif isinstance(record, RefundDecision):
        for name in ("decision_id", "case_id", "reason"):
            _require_text(name, getattr(record, name))
        _require_digest("case_digest", record.case_digest)
        if record.outcome not in _REFUND_OUTCOMES:
            raise AfterSalesSchemaError("unsupported refund outcome")
        _require_nonnegative_int("approved_amount", record.approved_amount)
        if record.approved_amount > record.binding.amount:
            raise AfterSalesSchemaError("refund decision exceeds order amount")
    elif isinstance(record, ExchangeCase):
        for name in (
            "case_id",
            "replacement_item_id",
            "replacement_sku_id",
            "reason",
        ):
            _require_text(name, getattr(record, name))
        _require_digest("return_receipt_digest", record.return_receipt_digest)
        if record.state not in _EXCHANGE_STATES:
            raise AfterSalesSchemaError("unsupported exchange state")
        _require_positive_int("version", record.version)
        if record.version == 1:
            if record.previous_digest is not None:
                raise AfterSalesSchemaError("first exchange previous_digest must be null")
        else:
            _require_digest("previous_digest", record.previous_digest)
        if record.state == "completed":
            _require_digest("completion_order_digest", record.completion_order_digest)
        elif record.completion_order_digest is not None:
            raise AfterSalesSchemaError("only completed exchange has completion digest")
    elif isinstance(record, DisputeEvidence):
        for name in ("evidence_id", "dispute_id", "evidence_kind"):
            _require_text(name, getattr(record, name))
        _require_digest("dispute_digest", record.dispute_digest)
        _require_positive_int("source_version", record.source_version)
        _require_digest("source_digest", record.source_digest)
        _freeze_object(record.facts, "facts")
    elif isinstance(record, DisputeCase):
        for name in ("dispute_id", "filed_by_id", "against_id", "reason"):
            _require_text(name, getattr(record, name))
        if record.filed_by_id == record.against_id:
            raise AfterSalesSchemaError("dispute parties must differ")
        if record.state not in _DISPUTE_STATES:
            raise AfterSalesSchemaError("unsupported dispute state")
        _require_positive_int("version", record.version)
        if record.version == 1:
            if record.previous_digest is not None:
                raise AfterSalesSchemaError("first dispute previous_digest must be null")
        else:
            _require_digest("previous_digest", record.previous_digest)
        _require_digest_tuple("evidence_digests", record.evidence_digests)
        if record.state == "ruled":
            _require_digest("ruling_digest", record.ruling_digest)
        elif record.ruling_digest is not None:
            raise AfterSalesSchemaError("only ruled dispute has ruling digest")
    elif isinstance(record, Ruling):
        for name in ("ruling_id", "dispute_id", "winner_id", "rationale"):
            _require_text(name, getattr(record, name))
        _require_digest("dispute_digest", record.dispute_digest)
        if record.outcome not in _RULING_OUTCOMES:
            raise AfterSalesSchemaError("unsupported ruling outcome")
        _require_nonnegative_int("refund_amount", record.refund_amount)
        if record.refund_amount > record.binding.amount:
            raise AfterSalesSchemaError("ruling refund exceeds order amount")
        _require_digest_tuple("evidence_digests", record.evidence_digests)


def _verify_authority_binding(
    binding: AfterSalesOrderBinding,
    authority: AfterSalesAuthority,
) -> None:
    validate_after_sales_order_binding(binding)
    if not isinstance(authority, AfterSalesAuthority):
        raise AfterSalesAuthorityError("authority must be trusted AfterSalesAuthority")
    mismatches = []
    if binding.binding_digest != authority.binding_digest:
        mismatches.append("binding_digest")
    if binding.owner_id != authority.owner_id:
        mismatches.append("owner_id")
    if binding.merchant_id != authority.merchant_id:
        mismatches.append("merchant_id")
    if mismatches:
        raise AfterSalesAuthorityError("after-sales authority mismatch: " + ", ".join(mismatches))


def _verify_exchange_identity(previous: ExchangeCase, candidate: ExchangeCase) -> None:
    fields = (
        "case_id",
        "return_receipt_digest",
        "replacement_item_id",
        "replacement_sku_id",
    )
    mismatches = [name for name in fields if getattr(previous, name) != getattr(candidate, name)]
    if previous.binding.binding_digest != candidate.binding.binding_digest:
        mismatches.append("binding_digest")
    if mismatches:
        raise AfterSalesTransitionError("exchange identity mismatch: " + ", ".join(mismatches))


def _binding_contract(binding: AfterSalesOrderBinding) -> dict[str, Any]:
    return {
        "schema_id": binding.schema_id,
        "order_id": binding.order_id,
        "item_id": binding.item_id,
        "sku_id": binding.sku_id,
        "shipment_id": binding.shipment_id,
        "owner_id": binding.owner_id,
        "merchant_id": binding.merchant_id,
        "qty": binding.qty,
        "amount": binding.amount,
        "currency": binding.currency,
        "policy_revision": binding.policy_revision,
        "policy_digest": binding.policy_digest,
        "order_digest": binding.order_digest,
        "shipment_digest": binding.shipment_digest,
    }


def _record_contract(record: AfterSalesRecord) -> dict[str, Any]:
    common = {
        "schema_id": record.schema_id,
        "binding": after_sales_binding_to_dict(record.binding),
        "actor_id": record.actor_id,
        "logical_tick": record.logical_tick,
        "idempotency_key": record.idempotency_key,
    }
    if isinstance(record, ReturnRequest):
        return {
            **common,
            "request_id": record.request_id,
            "requested_qty": record.requested_qty,
            "requested_amount": record.requested_amount,
            "reason": record.reason,
            "evidence_ids": list(record.evidence_ids),
            "evidence_digests": list(record.evidence_digests),
        }
    if isinstance(record, ReturnAuthorization):
        return {
            **common,
            "authorization_id": record.authorization_id,
            "request_id": record.request_id,
            "request_digest": record.request_digest,
            "outcome": record.outcome,
            "authorized_qty": record.authorized_qty,
            "authorized_amount": record.authorized_amount,
            "reason": record.reason,
        }
    if isinstance(record, ReturnReceipt):
        return {
            **common,
            "receipt_id": record.receipt_id,
            "request_id": record.request_id,
            "authorization_id": record.authorization_id,
            "authorization_digest": record.authorization_digest,
            "received_qty": record.received_qty,
            "condition": record.condition,
        }
    if isinstance(record, RefundCase):
        return {
            **common,
            "case_id": record.case_id,
            "source_kind": record.source_kind,
            "causal_digest": record.causal_digest,
            "requested_amount": record.requested_amount,
            "reason": record.reason,
        }
    if isinstance(record, RefundDecision):
        return {
            **common,
            "decision_id": record.decision_id,
            "case_id": record.case_id,
            "case_digest": record.case_digest,
            "outcome": record.outcome,
            "approved_amount": record.approved_amount,
            "reason": record.reason,
        }
    if isinstance(record, ExchangeCase):
        return {
            **common,
            "case_id": record.case_id,
            "return_receipt_digest": record.return_receipt_digest,
            "replacement_item_id": record.replacement_item_id,
            "replacement_sku_id": record.replacement_sku_id,
            "state": record.state,
            "version": record.version,
            "previous_digest": record.previous_digest,
            "completion_order_digest": record.completion_order_digest,
            "reason": record.reason,
        }
    if isinstance(record, DisputeEvidence):
        return {
            **common,
            "evidence_id": record.evidence_id,
            "dispute_id": record.dispute_id,
            "dispute_digest": record.dispute_digest,
            "evidence_kind": record.evidence_kind,
            "facts": _thaw_json(record.facts),
            "source_version": record.source_version,
            "source_digest": record.source_digest,
        }
    if isinstance(record, DisputeCase):
        return {
            **common,
            "dispute_id": record.dispute_id,
            "filed_by_id": record.filed_by_id,
            "against_id": record.against_id,
            "reason": record.reason,
            "state": record.state,
            "version": record.version,
            "previous_digest": record.previous_digest,
            "evidence_digests": list(record.evidence_digests),
            "ruling_digest": record.ruling_digest,
        }
    return {
        **common,
        "ruling_id": record.ruling_id,
        "dispute_id": record.dispute_id,
        "dispute_digest": record.dispute_digest,
        "winner_id": record.winner_id,
        "outcome": record.outcome,
        "refund_amount": record.refund_amount,
        "rationale": record.rationale,
        "evidence_digests": list(record.evidence_digests),
    }


def _replace_binding_digest(binding: AfterSalesOrderBinding, digest: str) -> AfterSalesOrderBinding:
    return AfterSalesOrderBinding(
        order_id=binding.order_id,
        item_id=binding.item_id,
        sku_id=binding.sku_id,
        shipment_id=binding.shipment_id,
        owner_id=binding.owner_id,
        merchant_id=binding.merchant_id,
        qty=binding.qty,
        amount=binding.amount,
        currency=binding.currency,
        policy_revision=binding.policy_revision,
        policy_digest=binding.policy_digest,
        order_digest=binding.order_digest,
        shipment_digest=binding.shipment_digest,
        binding_digest=digest,
        schema_id=binding.schema_id,
    )


def _replace_record_digest(record: AfterSalesRecord, digest: str) -> AfterSalesRecord:
    values = _record_contract_unvalidated(record)
    values["record_digest"] = digest
    values["schema_id"] = record.schema_id
    return type(record)(**values)


def _record_contract_unvalidated(record: AfterSalesRecord) -> dict[str, Any]:
    common: dict[str, Any] = {
        "binding": record.binding,
        "actor_id": record.actor_id,
        "logical_tick": record.logical_tick,
        "idempotency_key": record.idempotency_key,
    }
    if isinstance(record, ReturnRequest):
        return {
            **common,
            "request_id": record.request_id,
            "requested_qty": record.requested_qty,
            "requested_amount": record.requested_amount,
            "reason": record.reason,
            "evidence_ids": record.evidence_ids,
            "evidence_digests": record.evidence_digests,
        }
    if isinstance(record, ReturnAuthorization):
        return {
            **common,
            "authorization_id": record.authorization_id,
            "request_id": record.request_id,
            "request_digest": record.request_digest,
            "outcome": record.outcome,
            "authorized_qty": record.authorized_qty,
            "authorized_amount": record.authorized_amount,
            "reason": record.reason,
        }
    if isinstance(record, ReturnReceipt):
        return {
            **common,
            "receipt_id": record.receipt_id,
            "request_id": record.request_id,
            "authorization_id": record.authorization_id,
            "authorization_digest": record.authorization_digest,
            "received_qty": record.received_qty,
            "condition": record.condition,
        }
    if isinstance(record, RefundCase):
        return {
            **common,
            "case_id": record.case_id,
            "source_kind": record.source_kind,
            "causal_digest": record.causal_digest,
            "requested_amount": record.requested_amount,
            "reason": record.reason,
        }
    if isinstance(record, RefundDecision):
        return {
            **common,
            "decision_id": record.decision_id,
            "case_id": record.case_id,
            "case_digest": record.case_digest,
            "outcome": record.outcome,
            "approved_amount": record.approved_amount,
            "reason": record.reason,
        }
    if isinstance(record, ExchangeCase):
        return {
            **common,
            "case_id": record.case_id,
            "return_receipt_digest": record.return_receipt_digest,
            "replacement_item_id": record.replacement_item_id,
            "replacement_sku_id": record.replacement_sku_id,
            "state": record.state,
            "version": record.version,
            "previous_digest": record.previous_digest,
            "completion_order_digest": record.completion_order_digest,
            "reason": record.reason,
        }
    if isinstance(record, DisputeEvidence):
        return {
            **common,
            "evidence_id": record.evidence_id,
            "dispute_id": record.dispute_id,
            "dispute_digest": record.dispute_digest,
            "evidence_kind": record.evidence_kind,
            "facts": record.facts,
            "source_version": record.source_version,
            "source_digest": record.source_digest,
        }
    if isinstance(record, DisputeCase):
        return {
            **common,
            "dispute_id": record.dispute_id,
            "filed_by_id": record.filed_by_id,
            "against_id": record.against_id,
            "reason": record.reason,
            "state": record.state,
            "version": record.version,
            "previous_digest": record.previous_digest,
            "evidence_digests": record.evidence_digests,
            "ruling_digest": record.ruling_digest,
        }
    return {
        **common,
        "ruling_id": record.ruling_id,
        "dispute_id": record.dispute_id,
        "dispute_digest": record.dispute_digest,
        "winner_id": record.winner_id,
        "outcome": record.outcome,
        "refund_amount": record.refund_amount,
        "rationale": record.rationale,
        "evidence_digests": record.evidence_digests,
    }


_BINDING_FIELDS = frozenset(
    {
        "schema_id",
        "order_id",
        "item_id",
        "sku_id",
        "shipment_id",
        "owner_id",
        "merchant_id",
        "qty",
        "amount",
        "currency",
        "policy_revision",
        "policy_digest",
        "order_digest",
        "shipment_digest",
        "binding_digest",
    }
)
_COMMON_FIELDS = {
    "schema_id",
    "binding",
    "actor_id",
    "logical_tick",
    "idempotency_key",
    "record_digest",
}
_TYPE_FIELDS: dict[type[Any], frozenset[str]] = {
    ReturnRequest: frozenset(
        _COMMON_FIELDS
        | {
            "request_id",
            "requested_qty",
            "requested_amount",
            "reason",
            "evidence_ids",
            "evidence_digests",
        }
    ),
    ReturnAuthorization: frozenset(
        _COMMON_FIELDS
        | {
            "authorization_id",
            "request_id",
            "request_digest",
            "outcome",
            "authorized_qty",
            "authorized_amount",
            "reason",
        }
    ),
    ReturnReceipt: frozenset(
        _COMMON_FIELDS
        | {
            "receipt_id",
            "request_id",
            "authorization_id",
            "authorization_digest",
            "received_qty",
            "condition",
        }
    ),
    RefundCase: frozenset(
        _COMMON_FIELDS | {"case_id", "source_kind", "causal_digest", "requested_amount", "reason"}
    ),
    RefundDecision: frozenset(
        _COMMON_FIELDS
        | {"decision_id", "case_id", "case_digest", "outcome", "approved_amount", "reason"}
    ),
    ExchangeCase: frozenset(
        _COMMON_FIELDS
        | {
            "case_id",
            "return_receipt_digest",
            "replacement_item_id",
            "replacement_sku_id",
            "state",
            "version",
            "previous_digest",
            "completion_order_digest",
            "reason",
        }
    ),
    DisputeEvidence: frozenset(
        _COMMON_FIELDS
        | {
            "evidence_id",
            "dispute_id",
            "dispute_digest",
            "evidence_kind",
            "facts",
            "source_version",
            "source_digest",
        }
    ),
    DisputeCase: frozenset(
        _COMMON_FIELDS
        | {
            "dispute_id",
            "filed_by_id",
            "against_id",
            "reason",
            "state",
            "version",
            "previous_digest",
            "evidence_digests",
            "ruling_digest",
        }
    ),
    Ruling: frozenset(
        _COMMON_FIELDS
        | {
            "ruling_id",
            "dispute_id",
            "dispute_digest",
            "winner_id",
            "outcome",
            "refund_amount",
            "rationale",
            "evidence_digests",
        }
    ),
}
_TYPE_SCHEMAS = {
    ReturnRequest: RETURN_REQUEST_SCHEMA,
    ReturnAuthorization: RETURN_AUTHORIZATION_SCHEMA,
    ReturnReceipt: RETURN_RECEIPT_SCHEMA,
    RefundCase: REFUND_CASE_SCHEMA,
    RefundDecision: REFUND_DECISION_SCHEMA,
    ExchangeCase: EXCHANGE_CASE_SCHEMA,
    DisputeEvidence: DISPUTE_EVIDENCE_SCHEMA,
    DisputeCase: DISPUTE_CASE_SCHEMA,
    Ruling: RULING_SCHEMA,
}
_RECORD_TYPES = tuple(_TYPE_SCHEMAS)


def _coerce_binding(value: Any) -> AfterSalesOrderBinding:
    row = _exact_mapping(value, _BINDING_FIELDS, "after-sales binding")
    binding = AfterSalesOrderBinding(
        schema_id=_text(row, "schema_id"),
        order_id=_text(row, "order_id"),
        item_id=_text(row, "item_id"),
        sku_id=_text(row, "sku_id"),
        shipment_id=_optional_text(row, "shipment_id"),
        owner_id=_text(row, "owner_id"),
        merchant_id=_text(row, "merchant_id"),
        qty=_integer(row, "qty"),
        amount=_integer(row, "amount"),
        currency=_text(row, "currency"),
        policy_revision=_integer(row, "policy_revision"),
        policy_digest=_text(row, "policy_digest"),
        order_digest=_text(row, "order_digest"),
        shipment_digest=_optional_text(row, "shipment_digest"),
        binding_digest=_text(row, "binding_digest"),
    )
    validate_after_sales_order_binding(binding)
    return binding


def _coerce_record(value: Any, cls: type[Any]) -> AfterSalesRecord:
    row = _exact_mapping(value, _TYPE_FIELDS[cls], cls.__name__)
    if row["schema_id"] != _TYPE_SCHEMAS[cls]:
        raise AfterSalesSchemaError(f"unsupported {cls.__name__} schema")
    common = {
        "schema_id": _text(row, "schema_id"),
        "binding": _coerce_binding(row["binding"]),
        "actor_id": _text(row, "actor_id"),
        "logical_tick": _integer(row, "logical_tick"),
        "idempotency_key": _text(row, "idempotency_key"),
        "record_digest": _text(row, "record_digest"),
    }
    if cls is ReturnRequest:
        values = {
            **common,
            "request_id": _text(row, "request_id"),
            "requested_qty": _integer(row, "requested_qty"),
            "requested_amount": _integer(row, "requested_amount"),
            "reason": _text(row, "reason"),
            "evidence_ids": _canonical_text_tuple(
                row["evidence_ids"], "evidence_ids"
            ),
            "evidence_digests": _string_tuple(
                row["evidence_digests"], "evidence_digests"
            ),
        }
    elif cls is ReturnAuthorization:
        values = {
            **common,
            "authorization_id": _text(row, "authorization_id"),
            "request_id": _text(row, "request_id"),
            "request_digest": _text(row, "request_digest"),
            "outcome": _text(row, "outcome"),
            "authorized_qty": _integer(row, "authorized_qty"),
            "authorized_amount": _integer(row, "authorized_amount"),
            "reason": _text(row, "reason"),
        }
    elif cls is ReturnReceipt:
        values = {
            **common,
            "receipt_id": _text(row, "receipt_id"),
            "request_id": _text(row, "request_id"),
            "authorization_id": _text(row, "authorization_id"),
            "authorization_digest": _text(row, "authorization_digest"),
            "received_qty": _integer(row, "received_qty"),
            "condition": _text(row, "condition"),
        }
    elif cls is RefundCase:
        values = {
            **common,
            "case_id": _text(row, "case_id"),
            "source_kind": _text(row, "source_kind"),
            "causal_digest": _text(row, "causal_digest"),
            "requested_amount": _integer(row, "requested_amount"),
            "reason": _text(row, "reason"),
        }
    elif cls is RefundDecision:
        values = {
            **common,
            "decision_id": _text(row, "decision_id"),
            "case_id": _text(row, "case_id"),
            "case_digest": _text(row, "case_digest"),
            "outcome": _text(row, "outcome"),
            "approved_amount": _integer(row, "approved_amount"),
            "reason": _text(row, "reason"),
        }
    elif cls is ExchangeCase:
        values = {
            **common,
            "case_id": _text(row, "case_id"),
            "return_receipt_digest": _text(row, "return_receipt_digest"),
            "replacement_item_id": _text(row, "replacement_item_id"),
            "replacement_sku_id": _text(row, "replacement_sku_id"),
            "state": _text(row, "state"),
            "version": _integer(row, "version"),
            "previous_digest": _optional_text(row, "previous_digest"),
            "completion_order_digest": _optional_text(row, "completion_order_digest"),
            "reason": _text(row, "reason"),
        }
    elif cls is DisputeEvidence:
        values = {
            **common,
            "evidence_id": _text(row, "evidence_id"),
            "dispute_id": _text(row, "dispute_id"),
            "dispute_digest": _text(row, "dispute_digest"),
            "evidence_kind": _text(row, "evidence_kind"),
            "facts": _freeze_object(row["facts"], "facts"),
            "source_version": _integer(row, "source_version"),
            "source_digest": _text(row, "source_digest"),
        }
    elif cls is DisputeCase:
        values = {
            **common,
            "dispute_id": _text(row, "dispute_id"),
            "filed_by_id": _text(row, "filed_by_id"),
            "against_id": _text(row, "against_id"),
            "reason": _text(row, "reason"),
            "state": _text(row, "state"),
            "version": _integer(row, "version"),
            "previous_digest": _optional_text(row, "previous_digest"),
            "evidence_digests": _string_tuple(row["evidence_digests"], "evidence_digests"),
            "ruling_digest": _optional_text(row, "ruling_digest"),
        }
    else:
        values = {
            **common,
            "ruling_id": _text(row, "ruling_id"),
            "dispute_id": _text(row, "dispute_id"),
            "dispute_digest": _text(row, "dispute_digest"),
            "winner_id": _text(row, "winner_id"),
            "outcome": _text(row, "outcome"),
            "refund_amount": _integer(row, "refund_amount"),
            "rationale": _text(row, "rationale"),
            "evidence_digests": _string_tuple(row["evidence_digests"], "evidence_digests"),
        }
    record = cls(**values)
    validate_after_sales_record(record)
    return cast(AfterSalesRecord, record)


_SCHEMA_COERCERS = {
    schema: (lambda value, cls=cls: _coerce_record(value, cls))
    for cls, schema in _TYPE_SCHEMAS.items()
}


def _stable_id(record: AfterSalesRecord) -> str:
    for name in (
        "request_id",
        "authorization_id",
        "receipt_id",
        "decision_id",
        "case_id",
        "evidence_id",
        "dispute_id",
        "ruling_id",
    ):
        value = getattr(record, name, None)
        if isinstance(value, str):
            return value
    raise AfterSalesSchemaError("record has no stable id")


def _require_server_tick(record_tick: int, server_tick: int) -> None:
    if record_tick != server_tick:
        raise AfterSalesAuthorityError("record logical_tick is not trusted server tick")


def _exact_mapping(value: Any, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AfterSalesSchemaError(f"{label} must be an object")
    actual = frozenset(value.keys())
    if any(not isinstance(key, str) for key in actual):
        raise AfterSalesSchemaError(f"{label} has non-string fields")
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise AfterSalesSchemaError(
            f"{label} invalid fields: missing={missing!r}, unknown={unknown!r}"
        )
    return cast(Mapping[str, Any], value)


def _strict_json_loads(payload: str, label: str) -> Any:
    if not isinstance(payload, str):
        raise AfterSalesSchemaError(f"{label} JSON must be a string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AfterSalesSchemaError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def no_constant(value: str) -> None:
        raise AfterSalesSchemaError(f"non-finite JSON number: {value!r}")

    try:
        return json.loads(payload, object_pairs_hook=no_duplicates, parse_constant=no_constant)
    except json.JSONDecodeError as exc:
        raise AfterSalesSchemaError(f"invalid {label} JSON: {exc}") from exc


def _freeze_object(value: Any, path: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise AfterSalesSchemaError(f"{path} must be an object")
    frozen = _freeze_json(value, path)
    return cast(Mapping[str, JsonValue], frozen)


def _freeze_json(value: Any, path: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AfterSalesSchemaError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise AfterSalesSchemaError(f"{path} has a non-string key")
            result[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise AfterSalesSchemaError(f"{path} has non-JSON value {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise AfterSalesSchemaError(f"value is not canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row[key]
    _require_text(key, value)
    return cast(str, value)


def _optional_text(row: Mapping[str, Any], key: str) -> str | None:
    value = row[key]
    if value is None:
        return None
    _require_text(key, value)
    return cast(str, value)


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise AfterSalesSchemaError(f"{key} must be an integer")
    return cast(int, value)


def _canonical_strings(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise AfterSalesSchemaError("authority actor lists must be arrays")
    rows = tuple(values)
    for row in rows:
        _require_text("authority actor", row)
    if len(set(rows)) != len(rows):
        raise AfterSalesSchemaError("authority actor list contains duplicates")
    return tuple(sorted(rows))


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise AfterSalesSchemaError(f"{label} must be an array")
    rows = tuple(value)
    _require_digest_tuple(label, rows)
    return cast(tuple[str, ...], rows)


def _canonical_text_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise AfterSalesSchemaError(f"{label} must be an array")
    rows = tuple(value)
    for row in rows:
        _require_text(label, row)
    if rows != tuple(sorted(set(rows))):
        raise AfterSalesSchemaError(f"{label} must be unique and sorted")
    return cast(tuple[str, ...], rows)


def _require_string_tuple(label: str, values: Any) -> None:
    if not isinstance(values, tuple):
        raise AfterSalesAuthorityError(f"{label} must be a canonical tuple")
    for value in values:
        _require_text(label, value)
    if values != tuple(sorted(set(values))):
        raise AfterSalesAuthorityError(f"{label} must be unique and sorted")


def _require_digest_tuple(label: str, values: Any) -> None:
    if not isinstance(values, tuple):
        raise AfterSalesSchemaError(f"{label} must be a canonical tuple")
    for value in values:
        _require_digest(label, value)
    if len(set(values)) != len(values):
        raise AfterSalesSchemaError(f"{label} contains duplicate digests")


def _require_text(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise AfterSalesSchemaError(f"{name} must be a non-empty string")


def _require_digest(name: str, value: Any) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AfterSalesSchemaError(f"{name} must be a lowercase SHA-256 digest")


def _require_currency(value: Any) -> None:
    if not isinstance(value, str) or _CURRENCY_RE.fullmatch(value) is None:
        raise AfterSalesSchemaError("currency must be a three-letter uppercase code")


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AfterSalesSchemaError(f"{name} must be a positive integer")


def _require_nonnegative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AfterSalesSchemaError(f"{name} must be a non-negative integer")


__all__ = [
    "AFTER_SALES_BINDING_SCHEMA",
    "DISPUTE_CASE_SCHEMA",
    "DISPUTE_EVIDENCE_SCHEMA",
    "EXCHANGE_CASE_SCHEMA",
    "REFUND_CASE_SCHEMA",
    "REFUND_DECISION_SCHEMA",
    "RETURN_AUTHORIZATION_SCHEMA",
    "RETURN_RECEIPT_SCHEMA",
    "RETURN_REQUEST_SCHEMA",
    "RULING_SCHEMA",
    "AfterSalesAuthority",
    "AfterSalesAuthorityError",
    "AfterSalesDigestMismatch",
    "AfterSalesError",
    "AfterSalesIdempotencyConflict",
    "AfterSalesOrderBinding",
    "AfterSalesRecord",
    "AfterSalesSchemaError",
    "AfterSalesTransitionError",
    "DisputeBundle",
    "DisputeCase",
    "DisputeEvidence",
    "ExchangeCase",
    "RefundCase",
    "RefundChain",
    "RefundDecision",
    "ReturnAuthorization",
    "ReturnChain",
    "ReturnReceipt",
    "ReturnRequest",
    "Ruling",
    "after_sales_binding_from_json",
    "after_sales_binding_to_dict",
    "after_sales_binding_to_json",
    "after_sales_record_from_json",
    "after_sales_record_to_dict",
    "after_sales_record_to_json",
    "apply_dispute_record",
    "apply_exchange_case",
    "apply_refund_record",
    "apply_return_record",
    "build_after_sales_authority",
    "build_after_sales_order_binding",
    "build_dispute_case",
    "build_dispute_evidence",
    "build_exchange_case",
    "build_refund_case",
    "build_refund_decision",
    "build_return_authorization",
    "build_return_receipt",
    "build_return_request",
    "build_ruling",
    "classify_after_sales_retry",
    "replay_dispute_records",
    "replay_exchange_cases",
    "replay_refund_records",
    "replay_return_records",
    "validate_after_sales_order_binding",
    "validate_after_sales_record",
]
