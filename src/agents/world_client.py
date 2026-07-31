"""Platform-side VCP client to World (4c) — the drop-in for the platform's
former direct ``World`` handle.

After 4c the platform holds NO ``World`` object and shares NO lock: every world
access is a ``world.*`` envelope round-trip through the injected ``send``. The
method surface mirrors the slice of ``World`` the platform used
(``read`` / ``write`` / ``settle_order`` / ``refund_order``) plus
``search_catalog`` (replacing the aggregator's old ``_tables["catalog"].all()``
memory reach-in), so the policy bodies change only at ``self._world`` — not in
their logic.

``send(env) -> reply`` is the one seam the two topologies share:
  * in-process — ``WorldService.handle`` (a direct envelope-in/envelope-out call,
    no HTTP, no shared World object/lock);
  * 4-process — an HTTP POST to the world ``/vcp`` (e.g. ``VCPWorldClient.send``).

Typed rows are passed as objects wrapped under the world coercers' keys; the
``send`` serializes them over the wire (``to_json``) or hands the object straight
to the in-process service. Read replies come back as the typed object
(in-process) or a dict (HTTP); we normalize with the same world coercers — the
single source for dict→typed, so the two topologies can't drift.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable

import httpx

from protocol.envelope import Envelope
from protocol.cart_quote_request import (
    PersistentCartQuoteRequest,
    coerce_persistent_cart_quote_request,
)
from protocol.cart_quote_state import (
    CartQuoteStaleError,
    PersistentCartQuote,
    persistent_cart_quote_from_json,
)
from protocol.evidence_records import (
    EvidenceRecord,
    MandateRevision,
    MandateRevisionAuthority,
    coerce_evidence_record,
    coerce_mandate_revision,
    evidence_record_to_dict,
    mandate_revision_to_dict,
)
from protocol.event_receipts import (
    ProtocolEvent,
    ProtocolEventAuthorityError,
    ProtocolEventReceipt,
    ProtocolEventStaleError,
    protocol_event_receipt_to_dict,
    protocol_event_to_dict,
)
from protocol.matching import (
    MatchAcceptance,
    MatchAcceptanceRejected,
    MatchCertificate,
    SearchSession,
    coerce_match_acceptance,
    coerce_match_certificate,
    coerce_search_session,
    match_acceptance_to_wire,
    search_session_to_wire,
)
from protocol.listing_claims import (
    ListingClaim,
    coerce_listing_claim,
    listing_claim_to_wire,
)
from protocol.market_governance import (
    Campaign,
    GovernanceCase,
    MarketSignal,
    RankingContext,
    RemediationPlan,
    ReputationEvent,
    ReviewAggregate,
    ReviewEvidence,
)
from protocol.negotiation_state import (
    NegotiationEvent,
    NegotiationStateError,
    NegotiationThread,
    coerce_negotiation_event,
    coerce_negotiation_thread,
)
from protocol.pricing_policy import (
    PricingPolicyRevision,
    coerce_pricing_policy_revision,
)
from protocol.supply_authority import (
    DEFAULT_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS,
    SupplyPurchaseAuthority,
    coerce_supply_purchase_authority,
)
from world.evidence_contracts import (
    coerce_mandate_authority,
    mandate_authority_to_wire,
)
from world.after_sales_core import (
    AfterSalesCoreTransitionError,
    AfterSalesIntentError,
    AfterSalesPolicyRevision,
    after_sales_policy_from_dict,
)
from world.payment_fulfillment import (
    PackingRecord,
    PaymentStateRecord,
    packing_record_from_dict,
    payment_state_from_dict,
)
from world.market_governance_core import (
    AdsCampaignTerms,
    GovernanceBindingError,
    GovernanceIntentError,
    GovernanceResolutionDecision,
    GovernanceResponseAttestation,
    GovernanceTransitionError,
    RemediationBlueprint,
    ReputationPolicyRevision,
    ReviewAccountBinding,
)
from world.market_governance_persistence import (
    policy_payload_from_wire,
    record_payload_from_wire,
)
from world.errors import (
    AfterSalesReferenceRejected,
    CatalogMutationRejected,
    DisputeNotActionable,
    ExchangeNotActionable,
    FulfillmentNotActionable,
    IdempotencyConflict,
    InsufficientFunds,
    InvalidOrderTransition,
    LifecycleAuthorizationError,
    OrderIdentityMismatch,
    OrderNotRefundable,
    OrderNotSettleable,
    OutOfStock,
    ReturnWindowClosed,
    ShipmentNotActionable,
)
from world.service import (
    _coerce_allocation_batch,
    _coerce_order_group,
    _coerce_exchange,
    _coerce_dispute,
    _coerce_fulfillment,
    _coerce_inventory,
    _coerce_listing,
    _coerce_order,
    _coerce_order_timeline,
    _coerce_protocol_event,
    _coerce_protocol_event_receipt,
    _coerce_receipt,
    _coerce_reputation,
    _coerce_ruling,
    _coerce_shipment,
    _coerce_supply_state,
)
from world.types import (
    AgentId,
    AllocationBatch,
    Dispute,
    Exchange,
    ExchangeId,
    FulfillmentAllocation,
    InventoryRow,
    Listing,
    Order,
    OrderGroup,
    OrderId,
    OrderTimeline,
    Receipt,
    ReputationScore,
    Ruling,
    Shipment,
    ShipmentId,
    ShipmentResolution,
    ShipmentStatus,
    SkuId,
    SupplyState,
    TxnId,
)

#: WorldService.handle returns ``Envelope | list[Envelope] | None``; world.*
#: reads/writes are single-reply, but we accept the wider type and normalize.
SendFn = Callable[[Envelope], "Envelope | list[Envelope] | None"]

GovernancePolicyResult = (
    AdsCampaignTerms
    | ReviewAccountBinding
    | ReputationPolicyRevision
    | RemediationBlueprint
)
GovernanceRecordResult = (
    Campaign
    | ReviewEvidence
    | ReviewAggregate
    | MarketSignal
    | GovernanceCase
    | GovernanceResponseAttestation
    | GovernanceResolutionDecision
    | ReputationEvent
    | RemediationPlan
    | RankingContext
)

#: table -> (read kind, payload key, expected type, dict-coercer)
_READ: dict[str, tuple[str, str, type, Callable[[Any], Any]]] = {
    "catalog": ("world.read_catalog", "sku_id", Listing, _coerce_listing),
    "inventory": ("world.read_inventory", "sku_id", InventoryRow, _coerce_inventory),
    "orders": ("world.read_order", "order_id", Order, _coerce_order),
    "ledger": ("world.read_ledger", "txn_id", Receipt, _coerce_receipt),
    "fulfillments": (
        "world.read_fulfillment",
        "order_id",
        FulfillmentAllocation,
        _coerce_fulfillment,
    ),
    "reputation": ("world.read_reputation", "merchant_id", ReputationScore, _coerce_reputation),
    "disputes": ("world.read_dispute", "dispute_id", Dispute, _coerce_dispute),
    "rulings": ("world.read_ruling", "ruling_id", Ruling, _coerce_ruling),
    "order_timelines": (
        "world.read_order_timeline",
        "order_id",
        OrderTimeline,
        _coerce_order_timeline,
    ),
    "order_groups": (
        "world.read_order_group",
        "order_group_id",
        OrderGroup,
        _coerce_order_group,
    ),
    "shipments": (
        "world.read_shipment",
        "shipment_id",
        Shipment,
        _coerce_shipment,
    ),
    "protocol_events": (
        "world.read_protocol_event",
        "event_id",
        ProtocolEvent,
        _coerce_protocol_event,
    ),
}

#: table -> the wrapper key the world write-coercer reads the row from.
_WRITE_WRAP: dict[str, str] = {"catalog": "listing", "inventory": "inventory", "reputation": "score"}


class WorldClient:
    """VCP world access for the platform. ``send`` is the only transport seam."""

    def __init__(self, send: SendFn) -> None:
        self._send = send

    # --- transport -----------------------------------------------------

    def _roundtrip(self, *, from_: str, kind: str, payload: "dict[str, Any]",
                   idempotency_key: "str | None" = None) -> Any:
        env = Envelope(
            msg_id=f"pw:{uuid.uuid4()}", ts="1970-01-01T00:00:00Z",
            from_=from_, to="world", in_reply_to=None,
            idempotency_key=idempotency_key or f"pw:{uuid.uuid4()}",
            action={"kind": kind, "payload": payload},
        )
        try:
            reply = self._send(env)
        except httpx.HTTPStatusError as exc:
            _raise_typed_world_http_error(exc)
        if isinstance(reply, list):          # world.* is single-reply; normalize defensively
            reply = reply[0] if reply else None
        return reply.action.get("payload") if reply is not None else None

    # --- the World method surface the platform used --------------------

    def read(self, table: str, key: Any, *, caller: str) -> Any:
        kind, pk, typ, coerce = _READ[table]
        payload = self._roundtrip(from_=caller, kind=kind, payload={pk: str(key)})
        if payload is None:
            return None
        return payload if isinstance(payload, typ) else coerce(payload)

    def search_catalog(
        self,
        query: str,
        filters: "dict[str, Any] | None" = None,
        *,
        caller: str = "platform:aggregator",
        limit: int = 10,
    ) -> "list[Listing]":
        """Return a bounded catalog page through the same VCP read surface.

        The previous client hard-coded a one-million-row limit, causing every
        platform search to materialize the complete catalog. Callers must now
        choose an explicit candidate-pool bound; the World applies it before
        serializing the response.
        """
        if limit <= 0:
            return []
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_catalog",
            payload={"query": query, "filters": filters or {}, "limit": limit},
        )
        rows = payload or []
        return [r if isinstance(r, Listing) else _coerce_listing(r) for r in rows]

    def create_search_session(
        self,
        *,
        session: SearchSession,
        original_actor: str,
        idempotency_key: str,
    ) -> SearchSession:
        payload = self._roundtrip(
            from_="platform:aggregator",
            kind="world.create_search_session",
            payload={
                "session": search_session_to_wire(session),
                "original_actor": original_actor,
            },
            idempotency_key=idempotency_key,
        )
        return coerce_search_session(payload)

    def read_search_session(
        self,
        session_id: str,
        *,
        caller: str,
    ) -> SearchSession | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_search_session",
            payload={"session_id": session_id},
        )
        return None if payload is None else coerce_search_session(payload)

    def resolve_search_session(
        self,
        *,
        buyer_id: str,
        offer_id: str,
        caller: str = "platform:aggregator",
        unique_only: bool = True,
        current_only: bool = True,
    ) -> SearchSession | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.resolve_search_session",
            payload={
                "buyer_id": buyer_id,
                "offer_id": offer_id,
                "unique_only": unique_only,
                "current_only": current_only,
            },
        )
        return None if payload is None else coerce_search_session(payload)

    def issue_match_certificate(
        self,
        *,
        acceptance: MatchAcceptance,
        original_actor: str,
    ) -> MatchCertificate:
        payload = self._roundtrip(
            from_="platform:aggregator",
            kind="world.issue_match_certificate",
            payload={
                "acceptance": match_acceptance_to_wire(acceptance),
                "original_actor": original_actor,
            },
            idempotency_key=acceptance.idempotency_key,
        )
        return coerce_match_certificate(payload)

    def read_match_acceptance(
        self,
        *,
        buyer_id: str,
        idempotency_key: str,
        caller: str,
    ) -> MatchAcceptance | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_match_acceptance",
            payload={
                "buyer_id": buyer_id,
                "idempotency_key": idempotency_key,
            },
        )
        return None if payload is None else coerce_match_acceptance(payload)

    def read_match_certificate(
        self,
        cert_id: str,
        *,
        caller: str,
    ) -> MatchCertificate | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_match_certificate",
            payload={"cert_id": cert_id},
        )
        return None if payload is None else coerce_match_certificate(payload)

    def resolve_match_certificate(
        self,
        *,
        buyer_id: str,
        order_id: str,
        caller: str,
        current_only: bool = True,
    ) -> MatchCertificate | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_match_certificate",
            payload={
                "buyer_id": buyer_id,
                "order_id": order_id,
                "current_only": current_only,
            },
        )
        return None if payload is None else coerce_match_certificate(payload)

    def read_supply_state(
        self,
        sku_id: str,
        *,
        caller: str = "platform:supply",
    ) -> SupplyState:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_supply_state",
            payload={"sku_id": sku_id},
        )
        if payload is None:
            raise ValueError(f"unknown supply sku {sku_id!r}")
        return _coerce_supply_state(payload)

    def issue_supply_purchase_authorities(
        self,
        sku_ids: tuple[str, ...],
        *,
        original_actor: str,
        idempotency_key: str,
        ttl_ticks: int = DEFAULT_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS,
    ) -> tuple[SupplyPurchaseAuthority, ...]:
        payload = self._roundtrip(
            from_="platform:supply",
            kind="world.issue_supply_purchase_authority",
            payload={
                "sku_ids": list(sku_ids),
                "original_actor": original_actor,
                "ttl_ticks": ttl_ticks,
            },
            idempotency_key=idempotency_key,
        )
        if not isinstance(payload, list):
            raise ValueError("World returned invalid supply authorities")
        return tuple(coerce_supply_purchase_authority(row) for row in payload)

    def read_supply_purchase_authority(
        self,
        authority_id: str,
        *,
        caller: str,
    ) -> SupplyPurchaseAuthority | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_supply_purchase_authority",
            payload={"authority_id": authority_id},
        )
        return (
            None
            if payload is None
            else coerce_supply_purchase_authority(payload)
        )

    def read_order_protocol_state(
        self,
        order_id: OrderId | str,
        *,
        caller: str = "platform:events",
    ) -> dict[str, Any] | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_order_protocol_state",
            payload={"order_id": str(order_id)},
        )
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("World returned invalid order protocol state")
        return dict(payload)

    def read_protocol_event(
        self,
        event_id: str,
        *,
        caller: str,
    ) -> ProtocolEvent | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_protocol_event",
            payload={"event_id": event_id},
        )
        return None if payload is None else _coerce_protocol_event(payload)

    def read_protocol_events(
        self,
        binding_digest: str,
        *,
        caller: str,
    ) -> tuple[ProtocolEvent, ...]:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_protocol_events",
            payload={"binding_digest": binding_digest},
        )
        if not isinstance(payload, list):
            raise ValueError("World returned invalid protocol event stream")
        return tuple(_coerce_protocol_event(value) for value in payload)

    def read_protocol_receipts(
        self,
        binding_digest: str,
        *,
        order_id: OrderId | str,
        caller: str,
    ) -> tuple[ProtocolEventReceipt, ...]:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_protocol_receipts",
            payload={
                "binding_digest": binding_digest,
                "order_id": str(order_id),
            },
        )
        if not isinstance(payload, list):
            raise ValueError("World returned invalid protocol receipt stream")
        return tuple(_coerce_protocol_event_receipt(value) for value in payload)

    def apply_negotiation_intent(
        self,
        intent: dict[str, Any],
        *,
        original_actor: str,
        idempotency_key: str,
        max_rounds: int,
        deadline_ticks: int,
    ) -> NegotiationEvent:
        """Persist compact negotiation intent through the World authority API."""

        payload = self._roundtrip(
            from_="platform:negotiation",
            kind="world.apply_negotiation_intent",
            payload={
                "intent": intent,
                "original_actor": original_actor,
                "max_rounds": max_rounds,
                "deadline_ticks": deadline_ticks,
            },
            idempotency_key=idempotency_key,
        )
        return coerce_negotiation_event(payload)

    def read_negotiation_event(
        self, event_id: str, *, caller: str
    ) -> NegotiationEvent | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_negotiation_event",
            payload={"event_id": event_id},
        )
        return None if payload is None else coerce_negotiation_event(payload)

    def read_negotiation_thread(
        self, negotiation_id: str, *, caller: str
    ) -> NegotiationThread | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_negotiation_thread",
            payload={"negotiation_id": negotiation_id},
        )
        return None if payload is None else coerce_negotiation_thread(payload)

    def read_negotiation_events(
        self, negotiation_id: str, *, caller: str
    ) -> tuple[NegotiationEvent, ...]:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_negotiation_events",
            payload={"negotiation_id": negotiation_id},
        )
        if not isinstance(payload, list):
            raise ValueError("World returned invalid negotiation event stream")
        return tuple(coerce_negotiation_event(value) for value in payload)

    def publish_pricing_policy(
        self,
        intent: dict[str, Any],
        *,
        original_actor: str,
        idempotency_key: str,
    ) -> PricingPolicyRevision:
        """Submit only compact policy intent through the World authority API."""

        payload = self._roundtrip(
            from_="platform:pricing",
            kind="world.publish_pricing_policy",
            payload={"intent": intent, "original_actor": original_actor},
            idempotency_key=idempotency_key,
        )
        return coerce_pricing_policy_revision(payload)

    def pricing_policy_revisions(
        self,
        market_id: str,
        merchant_id: str,
        policy_id: str,
        *,
        caller: str,
    ) -> tuple[PricingPolicyRevision, ...]:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_pricing_policies",
            payload={
                "market_id": market_id,
                "merchant_id": merchant_id,
                "policy_id": policy_id,
            },
        )
        if not isinstance(payload, list):
            raise ValueError("World returned invalid pricing policy stream")
        return tuple(coerce_pricing_policy_revision(row) for row in payload)

    def create_cart_quote_request(
        self,
        intent: dict[str, Any],
        *,
        market_id: str,
        original_actor: str,
        idempotency_key: str,
        request_ttl_ticks: int = 10,
    ) -> PersistentCartQuoteRequest:
        payload = self._roundtrip(
            from_="platform:checkout",
            kind="world.create_cart_quote_request",
            payload={
                "intent": intent,
                "market_id": market_id,
                "original_actor": original_actor,
                "request_ttl_ticks": request_ttl_ticks,
            },
            idempotency_key=idempotency_key,
        )
        return coerce_persistent_cart_quote_request(payload)

    def issue_cart_quote(
        self,
        intent: dict[str, Any],
        *,
        market_id: str,
        original_actor: str,
        idempotency_key: str,
        quote_ttl_ticks: int = 10,
    ) -> PersistentCartQuote:
        payload = self._roundtrip(
            from_="platform:checkout",
            kind="world.issue_cart_quote",
            payload={
                "intent": intent,
                "market_id": market_id,
                "original_actor": original_actor,
                "quote_ttl_ticks": quote_ttl_ticks,
            },
            idempotency_key=idempotency_key,
        )
        return _coerce_persistent_cart_quote(payload)

    def issue_cart_quote_from_request(
        self,
        request_id: str,
        *,
        original_actor: str,
        idempotency_key: str,
        quote_ttl_ticks: int = 10,
    ) -> PersistentCartQuote:
        payload = self._roundtrip(
            from_="platform:checkout",
            kind="world.issue_cart_quote",
            payload={
                "request_id": request_id,
                "original_actor": original_actor,
                "quote_ttl_ticks": quote_ttl_ticks,
            },
            idempotency_key=idempotency_key,
        )
        return _coerce_persistent_cart_quote(payload)

    def read_cart_quote_request(
        self, request_id: str, *, caller: str
    ) -> PersistentCartQuoteRequest | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_cart_quote_request",
            payload={"request_id": request_id},
        )
        return (
            None
            if payload is None
            else coerce_persistent_cart_quote_request(payload)
        )

    def read_cart_quote(
        self, quote_id: str, *, caller: str
    ) -> PersistentCartQuote | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_cart_quote",
            payload={"quote_id": quote_id},
        )
        return None if payload is None else _coerce_persistent_cart_quote(payload)

    def publish_protocol_event(
        self,
        event: ProtocolEvent,
    ) -> ProtocolEvent:
        payload = self._roundtrip(
            from_="platform:events",
            kind="world.publish_protocol_event",
            payload={"event": protocol_event_to_dict(event)},
            idempotency_key=event.idempotency_key,
        )
        return _coerce_protocol_event(payload)

    def append_protocol_receipt(
        self,
        receipt: ProtocolEventReceipt,
        *,
        original_actor: str,
    ) -> ProtocolEventReceipt:
        payload = self._roundtrip(
            from_="platform:events",
            kind="world.append_protocol_receipt",
            payload={
                "receipt": protocol_event_receipt_to_dict(receipt),
                "original_actor": original_actor,
            },
            idempotency_key=receipt.idempotency_key,
        )
        return _coerce_protocol_event_receipt(payload)

    def process_protocol_event(
        self,
        *,
        event_id: str,
        original_actor: str,
        reason: str,
        idempotency_key: str,
    ) -> ProtocolEventReceipt:
        """Ask World to execute and receipt a registered commerce operation."""

        payload = self._roundtrip(
            from_="platform:events",
            kind="world.process_protocol_event",
            payload={
                "event_id": event_id,
                "original_actor": original_actor,
                "reason": reason,
            },
            idempotency_key=idempotency_key,
        )
        return _coerce_protocol_event_receipt(payload)

    def persist_evidence_record(
        self,
        record: EvidenceRecord,
        *,
        original_actor: str,
        idempotency_key: str,
    ) -> EvidenceRecord:
        payload = self._roundtrip(
            from_="platform:evidence",
            kind="world.persist_evidence_record",
            payload={
                "record": evidence_record_to_dict(record),
                "original_actor": original_actor,
            },
            idempotency_key=idempotency_key,
        )
        return coerce_evidence_record(payload)

    def read_evidence_record(
        self,
        record_id: str,
        *,
        caller: str,
        version: int | None = None,
        record_digest: str | None = None,
    ) -> EvidenceRecord | None:
        body: dict[str, Any] = {"record_id": record_id}
        if version is not None:
            body["version"] = version
        if record_digest is not None:
            body["record_digest"] = record_digest
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_evidence_record",
            payload=body,
        )
        return None if payload is None else coerce_evidence_record(payload)

    def register_mandate_authority(
        self,
        authority: MandateRevisionAuthority,
        *,
        original_actor: str,
        idempotency_key: str,
    ) -> MandateRevisionAuthority:
        payload = self._roundtrip(
            from_="platform:mandate",
            kind="world.register_mandate_authority",
            payload={
                "authority": mandate_authority_to_wire(authority),
                "original_actor": original_actor,
            },
            idempotency_key=idempotency_key,
        )
        return coerce_mandate_authority(payload)

    def append_mandate_revision(
        self,
        revision: MandateRevision,
        *,
        original_actor: str,
        idempotency_key: str,
    ) -> MandateRevision:
        payload = self._roundtrip(
            from_="platform:mandate",
            kind="world.append_mandate_revision",
            payload={
                "revision": mandate_revision_to_dict(revision),
                "original_actor": original_actor,
            },
            idempotency_key=idempotency_key,
        )
        return coerce_mandate_revision(payload)

    def read_mandate_revisions(
        self, mandate_id: str, *, caller: str
    ) -> tuple[MandateRevision, ...]:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_mandate_revisions",
            payload={"mandate_id": mandate_id},
        )
        if not isinstance(payload, list):
            raise ValueError("World returned invalid mandate history")
        return tuple(coerce_mandate_revision(value) for value in payload)

    def apply_listing_claim(
        self,
        claim: ListingClaim,
        *,
        original_actor: str,
        idempotency_key: str,
    ) -> ListingClaim:
        payload = self._roundtrip(
            from_="platform:claims",
            kind="world.apply_listing_claim",
            payload={
                "claim": listing_claim_to_wire(claim),
                "original_actor": original_actor,
            },
            idempotency_key=idempotency_key,
        )
        return coerce_listing_claim(payload)

    def read_listing_claim(
        self, claim_id: str, *, caller: str
    ) -> ListingClaim | None:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_listing_claim",
            payload={"claim_id": claim_id},
        )
        return None if payload is None else coerce_listing_claim(payload)

    def read_listing_claims(
        self, listing_id: str, *, caller: str
    ) -> tuple[ListingClaim, ...]:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_listing_claims",
            payload={"listing_id": listing_id},
        )
        if not isinstance(payload, list):
            raise ValueError("World returned invalid listing claim collection")
        return tuple(coerce_listing_claim(value) for value in payload)

    def apply_supply_event(
        self,
        *,
        sku_id: str,
        qty_delta: int = 0,
        eta_day: int | None = None,
        unit_price_cents: int | None = None,
        expected_version: int | None = None,
        original_actor: str,
        idempotency_key: str,
    ) -> SupplyState:
        body: dict[str, Any] = {
            "sku_id": str(SkuId(sku_id)),
            "qty_delta": qty_delta,
            "original_actor": original_actor,
        }
        if eta_day is not None:
            body["eta_day"] = eta_day
        if unit_price_cents is not None:
            body["unit_price_cents"] = unit_price_cents
        if expected_version is not None:
            body["expected_version"] = expected_version
        payload = self._roundtrip(
            from_="platform:supply",
            kind="world.apply_supply_event",
            payload=body,
            idempotency_key=idempotency_key,
        )
        return _coerce_supply_state(payload)

    def allocate_orders_atomic(
        self,
        *,
        allocation_id: str,
        merchant_id: AgentId,
        sku_id: SkuId,
        priority_order_ids: tuple[OrderId, ...],
        original_actor: str,
        idempotency_key: str,
    ) -> AllocationBatch:
        payload = self._roundtrip(
            from_="platform:fulfillment",
            kind="world.allocate_orders_atomic",
            payload={
                "allocation_id": allocation_id,
                "sku_id": str(sku_id),
                "priority_order_ids": [str(value) for value in priority_order_ids],
                "original_actor": original_actor,
            },
            idempotency_key=idempotency_key,
        )
        batch = _coerce_allocation_batch(payload or {})
        if batch.merchant_id != merchant_id:
            raise ValueError("World allocation owner did not match requested merchant")
        return batch

    def read_shipment(
        self,
        shipment_id: ShipmentId,
        *,
        caller: str = "platform:fulfillment",
    ) -> Shipment:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_shipment",
            payload={"shipment_id": str(shipment_id)},
        )
        if payload is None:
            raise ShipmentNotActionable(
                f"unknown or invisible shipment {shipment_id!r}"
            )
        return _coerce_shipment(payload)

    def record_shipment_status(
        self,
        *,
        shipment_id: ShipmentId,
        event_id: str,
        status: ShipmentStatus,
        original_actor: str,
        idempotency_key: str,
    ) -> Shipment:
        payload = self._roundtrip(
            from_="platform:fulfillment",
            kind="world.record_shipment_status",
            payload={
                "shipment_id": str(shipment_id),
                "event_id": event_id,
                "status": status.value,
                "original_actor": original_actor,
            },
            idempotency_key=idempotency_key,
        )
        return _coerce_shipment(payload or {})

    def resolve_shipment(
        self,
        *,
        shipment_id: ShipmentId,
        resolution: ShipmentResolution,
        replacement_sku_id: SkuId | None,
        original_actor: str,
        idempotency_key: str,
    ) -> Shipment:
        body: dict[str, Any] = {
            "shipment_id": str(shipment_id),
            "resolution": resolution.value,
            "original_actor": original_actor,
        }
        if replacement_sku_id is not None:
            body["replacement_sku_id"] = str(replacement_sku_id)
        payload = self._roundtrip(
            from_="platform:fulfillment",
            kind="world.resolve_shipment",
            payload=body,
            idempotency_key=idempotency_key,
        )
        return _coerce_shipment(payload or {})

    def write(self, table: str, key: Any, value: Any, *, by_action: str) -> None:
        # by_action IS the world.* write kind; value is a typed row wrapped under
        # the coercer's key so the world rebuilds it. Writes are platform-only.
        del key  # world derives the key from the row
        caller = "platform:reputation" if table == "reputation" else "platform:catalog"
        self._roundtrip(from_=caller, kind=by_action, payload={_WRITE_WRAP[table]: value})

    def apply_catalog_mutation(
        self,
        intent: "dict[str, Any]",
        *,
        original_actor: str,
        idempotency_key: str,
    ) -> Listing:
        """Apply a compact actor intent through the authoritative World API."""

        payload = self._roundtrip(
            from_="platform:catalog",
            kind="world.apply_catalog_mutation",
            payload={"intent": intent, "original_actor": original_actor},
            idempotency_key=idempotency_key,
        )
        return payload if isinstance(payload, Listing) else _coerce_listing(payload)

    def apply_settlement_reputation(
        self,
        *,
        merchant_id: AgentId,
        order_id: OrderId,
        txn_id: TxnId,
        original_actor: str,
        source_request_id: str,
        idempotency_key: str,
    ) -> ReputationScore:
        """Apply one authoritative settlement event through World VCP."""

        payload = self._roundtrip(
            from_="platform:reputation",
            kind="world.update_reputation",
            payload={
                "operation": "settlement_success",
                "merchant_id": str(merchant_id),
                "order_id": str(order_id),
                "txn_id": str(txn_id),
                "original_actor": original_actor,
                "source_request_id": source_request_id,
            },
            idempotency_key=idempotency_key,
        )
        return (
            payload
            if isinstance(payload, ReputationScore)
            else _coerce_reputation(payload or {})
        )

    # --- first-class payment, packing, and after-sales authority -------

    def publish_after_sales_policy(
        self,
        *,
        intent: dict[str, Any],
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesPolicyRevision:
        payload = self._roundtrip(
            from_="platform:policy",
            kind="world.publish_after_sales_policy",
            payload={"intent": intent, "original_actor": original_actor},
            idempotency_key=idempotency_key,
        )
        if isinstance(payload, AfterSalesPolicyRevision):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("World returned an invalid after-sales policy")
        return after_sales_policy_from_dict(payload)

    def apply_payment_intent(
        self,
        *,
        intent: dict[str, Any],
        original_actor: str,
        idempotency_key: str,
    ) -> PaymentStateRecord:
        payload = self._roundtrip(
            from_="platform:psp",
            kind="world.apply_payment_intent",
            payload={"intent": intent, "original_actor": original_actor},
            idempotency_key=idempotency_key,
        )
        if isinstance(payload, PaymentStateRecord):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("World returned an invalid payment state")
        return payment_state_from_dict(payload)

    def apply_packing_intent(
        self,
        *,
        intent: dict[str, Any],
        original_actor: str,
        idempotency_key: str,
    ) -> PackingRecord:
        payload = self._roundtrip(
            from_="platform:fulfillment",
            kind="world.apply_packing_intent",
            payload={"intent": intent, "original_actor": original_actor},
            idempotency_key=idempotency_key,
        )
        if isinstance(payload, PackingRecord):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("World returned an invalid packing record")
        return packing_record_from_dict(payload)

    def apply_after_sales_intent(
        self,
        *,
        intent: dict[str, Any],
        original_actor: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        payload = self._roundtrip(
            from_="platform:after-sales",
            kind="world.apply_after_sales_intent",
            payload={"intent": intent, "original_actor": original_actor},
            idempotency_key=idempotency_key,
        )
        return _require_mapping_payload(payload, "after-sales result")

    def complete_ledger_reconciliation(
        self,
        *,
        order_id: str,
        request_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        payload = self._roundtrip(
            from_="platform:accounting",
            kind="world.complete_ledger_reconciliation",
            payload={
                "order_id": order_id,
                "request_id": request_id,
                "original_actor": "platform:accounting",
            },
            idempotency_key=idempotency_key,
        )
        return _require_mapping_payload(payload, "reconciliation result")

    def read_after_sales_resource(
        self,
        *,
        resource: str,
        original_actor: str,
        order_id: str | None = None,
        merchant_id: str | None = None,
    ) -> dict[str, Any]:
        kinds = {
            "payment_history": "world.read_payment_history",
            "ledger_history": "world.read_ledger_history",
            "packing_history": "world.read_packing_history",
            "after_sales_history": "world.read_after_sales_history",
            "policy": "world.read_after_sales_policy",
        }
        try:
            kind = kinds[resource]
        except KeyError as exc:
            raise ValueError(f"unsupported after-sales resource {resource!r}") from exc
        if resource == "policy":
            if not merchant_id:
                raise ValueError("policy read requires merchant_id")
            request = {
                "merchant_id": merchant_id,
                "original_actor": original_actor,
            }
        else:
            if not order_id:
                raise ValueError("history read requires order_id")
            request = {
                "order_id": order_id,
                "original_actor": original_actor,
            }
        payload = self._roundtrip(
            from_="platform:after-sales",
            kind=kind,
            payload=request,
        )
        return _require_mapping_payload(payload, f"{resource} projection")

    def publish_governance_policy(
        self,
        policy_intent: dict[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> GovernancePolicyResult:
        payload = self._roundtrip(
            from_=by_actor,
            kind="world.publish_governance_policy",
            payload={
                "intent": dict(policy_intent),
                "original_actor": original_actor,
            },
            idempotency_key=idempotency_key,
        )
        result = _coerce_governance_result(payload)
        if not isinstance(
            result,
            (
                AdsCampaignTerms,
                ReviewAccountBinding,
                ReputationPolicyRevision,
                RemediationBlueprint,
            ),
        ):
            raise ValueError("World returned a governance record for a policy request")
        return result

    def apply_governance_intent(
        self,
        intent: dict[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> GovernanceRecordResult:
        payload = self._roundtrip(
            from_=by_actor,
            kind="world.apply_governance_intent",
            payload={"intent": dict(intent), "original_actor": original_actor},
            idempotency_key=idempotency_key,
        )
        result = _coerce_governance_result(payload)
        if isinstance(
            result,
            (
                AdsCampaignTerms,
                ReviewAccountBinding,
                ReputationPolicyRevision,
                RemediationBlueprint,
            ),
        ):
            raise ValueError("World returned a governance policy for an actor intent")
        return result

    def aggregate_reviews(
        self,
        sku_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> ReviewAggregate:
        payload = self._roundtrip(
            from_=by_actor,
            kind="world.aggregate_reviews",
            payload={"sku_id": sku_id, "original_actor": original_actor},
            idempotency_key=idempotency_key,
        )
        result = _coerce_governance_result(payload)
        if not isinstance(result, ReviewAggregate):
            raise ValueError("World returned an invalid review aggregate")
        return result

    def ingest_market_observation(
        self,
        record_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> GovernanceCase:
        payload = self._roundtrip(
            from_=by_actor,
            kind="world.ingest_market_observation",
            payload={"record_id": record_id, "original_actor": original_actor},
            idempotency_key=idempotency_key,
        )
        result = _coerce_governance_result(payload)
        if not isinstance(result, GovernanceCase):
            raise ValueError("World returned an invalid governance case")
        return result

    def ingest_review_observation(
        self,
        record_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> ReviewEvidence:
        payload = self._roundtrip(
            from_=by_actor,
            kind="world.ingest_review_observation",
            payload={"record_id": record_id, "original_actor": original_actor},
            idempotency_key=idempotency_key,
        )
        result = _coerce_governance_result(payload)
        if not isinstance(result, ReviewEvidence):
            raise ValueError("World returned invalid imported review evidence")
        return result

    def resolve_governance_case(
        self,
        decision_intent: dict[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> GovernanceCase:
        payload = self._roundtrip(
            from_=by_actor,
            kind="world.resolve_governance_case",
            payload={
                "intent": dict(decision_intent),
                "original_actor": original_actor,
            },
            idempotency_key=idempotency_key,
        )
        result = _coerce_governance_result(payload)
        if not isinstance(result, GovernanceCase):
            raise ValueError("World returned an invalid resolved governance case")
        return result

    def apply_governance_reputation(
        self,
        source_intent: dict[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> ReputationEvent:
        payload = self._roundtrip(
            from_=by_actor,
            kind="world.apply_governance_reputation",
            payload={
                "intent": dict(source_intent),
                "original_actor": original_actor,
            },
            idempotency_key=idempotency_key,
        )
        result = _coerce_governance_result(payload)
        if not isinstance(result, ReputationEvent):
            raise ValueError("World returned an invalid reputation event")
        return result

    def create_remediation_plan(
        self,
        plan_intent: dict[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> RemediationPlan:
        payload = self._roundtrip(
            from_=by_actor,
            kind="world.create_remediation_plan",
            payload={"intent": dict(plan_intent), "original_actor": original_actor},
            idempotency_key=idempotency_key,
        )
        result = _coerce_governance_result(payload)
        if not isinstance(result, RemediationPlan):
            raise ValueError("World returned an invalid remediation plan")
        return result

    def verify_remediation_step(
        self,
        plan_id: str,
        step_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> RemediationPlan:
        payload = self._roundtrip(
            from_=by_actor,
            kind="world.verify_remediation_step",
            payload={
                "plan_id": plan_id,
                "step_id": step_id,
                "original_actor": original_actor,
            },
            idempotency_key=idempotency_key,
        )
        result = _coerce_governance_result(payload)
        if not isinstance(result, RemediationPlan):
            raise ValueError("World returned an invalid verified remediation plan")
        return result

    def persist_ranking_context(
        self,
        ranking_result: dict[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> RankingContext:
        payload = self._roundtrip(
            from_=by_actor,
            kind="world.persist_ranking_context",
            payload={
                "ranking_result": dict(ranking_result),
                "original_actor": original_actor,
            },
            idempotency_key=idempotency_key,
        )
        result = _coerce_governance_result(payload)
        if not isinstance(result, RankingContext):
            raise ValueError("World returned an invalid ranking context")
        return result

    def ranking_context_projection(
        self,
        context_id: str,
        *,
        caller: str,
    ) -> dict[str, Any]:
        payload = self._roundtrip(
            from_=caller,
            kind="world.ranking_context_projection",
            payload={"context_id": context_id},
        )
        return _coerce_ranking_context_projection(payload)

    def governance_history(
        self,
        record_kind: str,
        stable_id: str,
        *,
        caller: str,
    ) -> tuple[GovernancePolicyResult | GovernanceRecordResult, ...]:
        payload = self._roundtrip(
            from_="platform:governance",
            kind="world.governance_history",
            payload={
                "record_kind": record_kind,
                "stable_id": stable_id,
                "original_actor": caller,
            },
        )
        projection = _require_mapping_payload(payload, "governance history")
        if (
            projection.get("resource") != "governance_history"
            or projection.get("record_kind") != record_kind
            or projection.get("stable_id") != stable_id
            or not isinstance(projection.get("records"), list)
        ):
            raise ValueError("World returned an invalid governance history")
        return tuple(
            _coerce_governance_payload(record_kind, row)
            for row in projection["records"]
        )

    def settle_order(self, *, order: Order, receipt: Any, by_role: str,
                     idempotency_key: str) -> "_TxnResult":
        payload = self._roundtrip(
            from_=f"{by_role}:psp", kind="world.settle_order",
            payload={"order": order, "receipt": receipt}, idempotency_key=idempotency_key)
        return _TxnResult(payload or {})

    def settle_order_partial(
        self,
        *,
        order: Order,
        fulfilled_qty: int,
        receipt: Any | None,
        original_actor: str,
        idempotency_key: str,
    ) -> "_TxnResult":
        body: dict[str, Any] = {
            "order": order,
            "fulfilled_qty": fulfilled_qty,
            "original_actor": original_actor,
        }
        if receipt is not None:
            body["receipt"] = receipt
        payload = self._roundtrip(
            from_="platform:psp",
            kind="world.settle_order_partial",
            payload=body,
            idempotency_key=idempotency_key,
        )
        return _TxnResult(payload or {})

    def checkout_cart(
        self,
        *,
        quote_id: str,
        original_actor: str,
        idempotency_key: str,
    ) -> OrderGroup:
        payload = self._roundtrip(
            from_="platform:checkout",
            kind="world.checkout_cart",
            payload={"quote_id": quote_id, "original_actor": original_actor},
            idempotency_key=idempotency_key,
        )
        return (
            payload
            if isinstance(payload, OrderGroup)
            else _coerce_order_group(payload or {})
        )

    def refund_order(self, *, order: Order, refund_receipt: Any, by_role: str,
                     idempotency_key: str) -> "_TxnResult":
        payload = self._roundtrip(
            from_=f"{by_role}:psp", kind="world.refund_order",
            payload={"order": order, "refund_receipt": refund_receipt}, idempotency_key=idempotency_key)
        return _TxnResult(payload or {})

    def dispatch_order(self, *, order_id: Any, original_actor: str) -> Order:
        return self._lifecycle_order(
            kind="world.dispatch_order",
            order_id=order_id,
            original_actor=original_actor,
        )

    def cancel_order(self, *, order_id: Any, original_actor: str) -> Order:
        return self._lifecycle_order(
            kind="world.cancel_order",
            order_id=order_id,
            original_actor=original_actor,
        )

    def mark_order_returned(self, *, order_id: Any, original_actor: str) -> Order:
        return self._lifecycle_order(
            kind="world.mark_order_returned",
            order_id=order_id,
            original_actor=original_actor,
        )

    def exchange_order(
        self,
        *,
        exchange_id: ExchangeId,
        original_order_id: OrderId,
        replacement_order: Order,
        idempotency_key: str,
    ) -> Exchange:
        payload = self._roundtrip(
            from_="platform:psp",
            kind="world.exchange_order",
            payload={
                "exchange_id": str(exchange_id),
                "original_order_id": str(original_order_id),
                "replacement_order": replacement_order,
            },
            idempotency_key=idempotency_key,
        )
        return payload if isinstance(payload, Exchange) else _coerce_exchange(payload or {})

    def read_logical_time(self, *, caller: str) -> int:
        payload = self._roundtrip(
            from_=caller,
            kind="world.read_clock",
            payload={},
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("logical_time"), int
        ):
            raise ValueError("World returned an invalid logical clock payload")
        return int(payload["logical_time"])

    def advance_logical_time(self, *, to_tick: int) -> int:
        payload = self._roundtrip(
            from_="runtime:clock",
            kind="world.advance_clock",
            payload={"to_tick": to_tick},
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("logical_time"), int
        ):
            raise ValueError("World returned an invalid logical clock payload")
        return int(payload["logical_time"])

    def _lifecycle_order(
        self,
        *,
        kind: str,
        order_id: Any,
        original_actor: str,
    ) -> Order:
        payload = self._roundtrip(
            from_="platform:psp",
            kind=kind,
            payload={
                "order_id": str(order_id),
                "original_actor": original_actor,
            },
        )
        return payload if isinstance(payload, Order) else _coerce_order(payload or {})

    def open_dispute(self, *, dispute: Dispute, original_actor: str) -> Dispute:
        payload = self._roundtrip(
            from_="platform:adjudicator",
            kind="world.open_dispute",
            payload={"dispute": dispute, "original_actor": original_actor},
        )
        return payload if isinstance(payload, Dispute) else _coerce_dispute(payload or {})

    def rule_dispute(self, *, ruling: Ruling, original_actor: str) -> Ruling:
        payload = self._roundtrip(
            from_="platform:adjudicator",
            kind="world.rule_dispute",
            payload={"ruling": ruling, "original_actor": original_actor},
        )
        return payload if isinstance(payload, Ruling) else _coerce_ruling(payload or {})


_GOVERNANCE_POLICY_KINDS = frozenset(
    {
        "ads_campaign_terms",
        "review_account_binding",
        "reputation_policy_revision",
        "remediation_blueprint",
    }
)
_GOVERNANCE_RECORD_KINDS = frozenset(
    {
        "campaign",
        "review_evidence",
        "review_aggregate",
        "market_signal",
        "governance_case",
        "governance_response_attestation",
        "governance_resolution_decision",
        "reputation_event",
        "remediation_plan",
        "ranking_context",
    }
)


def _coerce_governance_result(
    value: Any,
) -> GovernancePolicyResult | GovernanceRecordResult:
    row = _require_mapping_payload(value, "governance result")
    if set(row) != {"result_kind", "value"}:
        raise ValueError("World returned governance result fields that are not exact")
    kind = row["result_kind"]
    payload = row["value"]
    if not isinstance(kind, str) or not isinstance(payload, dict):
        raise ValueError("World returned an invalid governance result wrapper")
    return _coerce_governance_payload(kind, payload)


def _coerce_governance_payload(
    kind: str, value: Any
) -> GovernancePolicyResult | GovernanceRecordResult:
    if not isinstance(value, dict):
        raise ValueError("World returned a non-object governance payload")
    if kind in _GOVERNANCE_POLICY_KINDS:
        return policy_payload_from_wire(kind, value)  # type: ignore[arg-type]
    if kind in _GOVERNANCE_RECORD_KINDS:
        return record_payload_from_wire(kind, value)  # type: ignore[arg-type]
    raise ValueError(f"World returned unsupported governance kind {kind!r}")


def _coerce_ranking_context_projection(value: Any) -> dict[str, Any]:
    """Validate the exact public projection before Platform forwards it."""

    row = _require_mapping_payload(value, "ranking context projection")
    expected = {
        "schema_version",
        "context_id",
        "context_digest",
        "candidate_annotations",
    }
    if set(row) != expected:
        raise ValueError("World returned non-exact ranking projection fields")
    if row["schema_version"] != "cwe.governance-ranking-projection.v1":
        raise ValueError("World returned an unsupported ranking projection")
    for field in ("context_id", "context_digest"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"ranking projection {field} must be non-empty")
    annotations = row["candidate_annotations"]
    if not isinstance(annotations, list):
        raise ValueError("ranking projection annotations must be a list")
    for annotation in annotations:
        _validate_ranking_annotation(annotation)
    return row


def _validate_ranking_annotation(value: Any) -> None:
    annotation = _require_mapping_payload(value, "ranking annotation")
    if set(annotation) != {
        "sku_id",
        "merchant_id",
        "sponsored_placements",
        "review_summary",
        "resolved_cases",
        "reputation",
    }:
        raise ValueError("World returned non-exact ranking annotation fields")
    for field in ("sku_id", "merchant_id"):
        if not isinstance(annotation[field], str) or not annotation[field]:
            raise ValueError(f"ranking annotation {field} must be non-empty")
    placements = annotation["sponsored_placements"]
    cases = annotation["resolved_cases"]
    if not isinstance(placements, list) or not isinstance(cases, list):
        raise ValueError("ranking annotation placements and cases must be lists")
    for placement in placements:
        _validate_exact_public_object(
            placement,
            fields={
                "placement_id",
                "disclosure_status",
                "disclosure_text",
            },
            label="ranking placement",
        )
    for governance_case in cases:
        _validate_exact_public_object(
            governance_case,
            fields={"case_id", "case_kind", "resolution_code"},
            label="ranking case",
        )
    review = annotation["review_summary"]
    if review is not None:
        _validate_exact_public_object(
            review,
            fields={
                "review_count",
                "verified_review_count",
                "rating_sum",
                "verified_rating_sum",
            },
            label="ranking review summary",
        )
        if any(isinstance(item, bool) or not isinstance(item, int) for item in review.values()):
            raise ValueError("ranking review summary values must be integers")
    reputation = annotation["reputation"]
    if reputation is not None:
        _validate_exact_public_object(
            reputation,
            fields={"event_kind", "outcome_bps", "version"},
            label="ranking reputation",
        )
        if (
            isinstance(reputation["outcome_bps"], bool)
            or not isinstance(reputation["outcome_bps"], int)
            or isinstance(reputation["version"], bool)
            or not isinstance(reputation["version"], int)
        ):
            raise ValueError("ranking reputation numeric fields must be integers")


def _validate_exact_public_object(
    value: Any,
    *,
    fields: set[str],
    label: str,
) -> None:
    row = _require_mapping_payload(value, label)
    if set(row) != fields:
        raise ValueError(f"World returned non-exact {label} fields")


def _require_mapping_payload(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"World returned an invalid {label}")
    return value


def _coerce_persistent_cart_quote(value: Any) -> PersistentCartQuote:
    if isinstance(value, PersistentCartQuote):
        return value
    return persistent_cart_quote_from_json(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


class _TxnResult:
    """The world's settle/refund summary ({order_id, txn_id, status}), exposed as
    attributes so the PSP's ``settled.txn_id`` / ``.order_id`` read unchanged."""

    def __init__(self, payload: "dict[str, Any]") -> None:
        self.order_id = payload.get("order_id")
        self.txn_id = payload.get("txn_id")
        self.status = payload.get("status")
        self.requested_qty = payload.get("requested_qty")
        self.fulfilled_qty = payload.get("fulfilled_qty")
        self.backordered_qty = payload.get("backordered_qty")


def in_process_world_client(world: Any) -> WorldClient:
    """A ``WorldClient`` whose transport is a local ``WorldService.handle`` — the
    in-process topology. The platform still reaches World only through the
    ``world.*`` service surface (envelope in / envelope out): no HTTP, but also no
    shared World object and no shared lock (the lock lives inside ``World``,
    reached via ``world.settle_order`` etc., never held by the platform)."""
    from world.service import WorldService

    service = WorldService(world)
    return WorldClient(send=service.handle)


def _raise_typed_world_http_error(exc: httpx.HTTPStatusError) -> None:
    """Restore core World errors erased by the Platform-to-World HTTP seam."""

    try:
        payload = exc.response.json()
    except (TypeError, ValueError):
        raise exc
    if not isinstance(payload, dict):
        raise exc
    kind = payload.get("kind")
    message = str(payload.get("error", exc))
    if kind == "authorization_error":
        raise LifecycleAuthorizationError(message) from exc
    if kind == "idempotency_conflict":
        raise IdempotencyConflict(message) from exc
    if kind == "catalog_mutation_rejected":
        raise CatalogMutationRejected(message) from exc
    if kind == "match_acceptance_rejected":
        raise MatchAcceptanceRejected(message) from exc
    if kind == "cart_quote_stale":
        raise CartQuoteStaleError(message) from exc
    if kind == "after_sales_validation_error":
        raise AfterSalesIntentError(message) from exc
    if kind == "after_sales_transition_error":
        raise AfterSalesCoreTransitionError(message) from exc
    if kind == "after_sales_reference_rejected":
        raise AfterSalesReferenceRejected(message) from exc
    if kind == "governance_validation_error":
        raise GovernanceIntentError(message) from exc
    if kind == "governance_binding_error":
        raise GovernanceBindingError(message) from exc
    if kind == "governance_transition_error":
        raise GovernanceTransitionError(message) from exc
    if kind == "negotiation_state_error":
        raise NegotiationStateError(message) from exc
    if kind == "insufficient_funds":
        raise InsufficientFunds(message) from exc
    if kind == "out_of_stock":
        raise OutOfStock(message) from exc
    if kind == "order_not_settleable":
        raise OrderNotSettleable(message) from exc
    if kind == "order_not_refundable":
        raise OrderNotRefundable(message) from exc
    if kind == "return_window_closed":
        raise ReturnWindowClosed(message) from exc
    if kind == "invalid_order_transition":
        raise InvalidOrderTransition(message) from exc
    if kind == "dispute_not_actionable":
        raise DisputeNotActionable(message) from exc
    if kind == "fulfillment_not_actionable":
        raise FulfillmentNotActionable(message) from exc
    if kind == "shipment_not_actionable":
        raise ShipmentNotActionable(message) from exc
    if kind == "exchange_not_actionable":
        raise ExchangeNotActionable(message) from exc
    if kind == "order_identity_mismatch":
        raise OrderIdentityMismatch(message) from exc
    if kind == "protocol_event_authority":
        raise ProtocolEventAuthorityError(message) from exc
    if kind == "protocol_event_stale":
        raise ProtocolEventStaleError(message) from exc
    raise exc


__all__ = ["WorldClient", "SendFn", "in_process_world_client"]
