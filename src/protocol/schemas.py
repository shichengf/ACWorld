"""TypedDict payload schemas for ``commerce.*`` envelopes.

The merchant emits these payloads via ``agents.merchant_tools``; the
platform consumes them via ``agents.platform.CatalogPolicy``. Before
this module, the two sides agreed on payload shape by convention only —
a constructor wrote ``{"list_price": int}`` and the handler read
``payload.get("list_price")``, with no compile-time or type-checker
link between them. A wrong field name on either side silently produced
an ``invalid_list_price`` decline at runtime, or worse — a passing
write with stale defaults.

These TypedDicts are the single source of truth: both sides import
the same name, mypy / pyright can flag drift, and the schema is the
documentation of what ``commerce.*`` payloads look like on the wire.

Conventions (consistent with the rest of the codebase):

* All money fields are **integer minor units** (cents, fen). The
  envelope guard in :mod:`protocol.envelope` does not enforce this —
  the constructors do, and these TypedDicts pin it at the type level.
* All ``sku_id`` / ``op_id`` / etc. are strings on the wire (the
  ``SkuId`` newtype lives only in :mod:`world.types`).
* No PRIVATE_UTILITY field (``floor_price``, ``walk_away``, urgency
  cues) may appear in any payload here. The absence of such fields
  from these schemas is itself an enforcement layer: a constructor
  that tries to embed one is statically wrong, not just runtime-wrong.
* No payload here may carry a ``merchant_id`` — ownership is derived
  from ``env.from_`` by :class:`agents.platform.CatalogPolicy`. The
  schemas omit the field deliberately; the handler rejects envelopes
  that include it (defense in depth, see Phase 1 fix 0b).
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class CartRequestLinePayload(TypedDict):
    """One requested SKU/quantity pair for an authoritative cart quote."""

    sku_id: str
    qty: int


class RequestCartQuotePayload(TypedDict, total=False):
    """``commerce.request_cart_quote`` payload."""

    market_id: str
    mandate_id: str
    lines: list[CartRequestLinePayload]
    fill_policy: Literal["all_or_none", "allow_partial"]
    backorder_policy: Literal["reject", "allow"]
    request_id: str
    quote_ttl_ticks: int


class CreateCartQuoteRequestPayload(TypedDict, total=False):
    """Buyer/principal intent for a persisted merchant quote authorization."""

    market_id: str
    mandate_id: str
    lines: list[CartRequestLinePayload]
    fill_policy: Literal["all_or_none", "allow_partial"]
    backorder_policy: Literal["reject", "allow"]
    request_ttl_ticks: int


class CheckoutCartPayload(TypedDict):
    """``platform.checkout_cart`` payload; World loads the quote by id."""

    quote_id: str


class ReadSupplyStatePayload(TypedDict):
    """Request one or more World-authored supply projections."""

    sku_ids: list[str]


class UpdateSupplyPayload(TypedDict, total=False):
    """Owner-authored or trusted deterministic supply revision."""

    sku_id: str
    qty_delta: int
    eta_day: int
    unit_price_cents: int
    expected_version: int


class AllocateFulfillmentPayload(TypedDict):
    """Merchant request for one World-computed scarce-stock allocation."""

    allocation_id: str
    sku_id: str
    priority_order_ids: list[str]


class ReadShipmentPayload(TypedDict):
    shipment_id: str


class RecordShipmentStatusPayload(TypedDict):
    shipment_id: str
    event_id: str
    status: Literal["delayed", "missing_scan", "lost", "delivered"]


class ResolveShipmentPayload(TypedDict, total=False):
    shipment_id: str
    resolution: Literal["wait", "replacement", "refund"]
    replacement_sku_id: str


class ProtocolEventDecisionPayload(TypedDict):
    """Actor decision intent for one Platform-delivered persistent event.

    Identity, observed order state, revision, logical tick, and effect
    references are deliberately absent.  Platform derives those fields from
    the authenticated envelope and authoritative World state.
    """

    event_id: str
    reason: str


class IssueProtocolEventPayload(TypedDict, total=False):
    """Trusted control-plane request for one authoritative event.

    Scenario and scheduler code may choose the stream, recipient, semantic
    event kind, and time-to-live.  It cannot supply order parties, order state,
    state revision, logical time, or an operation-state digest; Platform reads
    those values from World.  ``reference_digest`` is accepted only for the
    certificate lane and is revalidated by World.
    """

    market_id: str
    stream_id: str
    order_id: str
    recipient_id: str
    event_id: str
    event_kind: str
    ttl_ticks: int
    reference_kind: Literal["operation", "certificate"]
    reference_digest: str


class AdvanceMarketClockPayload(TypedDict):
    """Trusted Runtime request; Platform forwards only the target tick."""

    to_tick: int


class ProcessProtocolEventPayload(TypedDict):
    """Trusted Platform-to-World protocol operation request."""

    event_id: str
    original_actor: str
    reason: str


class DeliverProtocolEventPayload(TypedDict):
    """``platform.deliver_protocol_event`` sealed event payload."""

    event: dict[str, Any]


class ProtocolEventReceiptPayload(TypedDict):
    """Platform response containing the exact persisted decision receipt."""

    receipt: dict[str, Any]


class ReadOrderProtocolStatePayload(TypedDict):
    order_id: str


class ReadProtocolEventPayload(TypedDict):
    event_id: str


class ReadProtocolEventsPayload(TypedDict):
    binding_digest: str


class ReadProtocolReceiptsPayload(TypedDict):
    binding_digest: str
    order_id: str


class NegotiationIntentPayload(TypedDict, total=False):
    """Compact actor intent; every authority field is World-derived."""

    action_kind: Literal[
        "commerce.propose_offer",
        "commerce.counter_offer",
        "commerce.accept_offer",
        "commerce.reject_offer",
        "commerce.withdraw_offer",
    ]
    negotiation_id: str
    offer_id: str
    sku_id: str
    counterparty_id: str
    unit_price: int
    round_no: int
    qty: int


class WorldApplyNegotiationIntentPayload(TypedDict):
    intent: NegotiationIntentPayload
    original_actor: str
    max_rounds: int
    deadline_ticks: int


class ReadNegotiationEventPayload(TypedDict):
    event_id: str


class ReadNegotiationThreadPayload(TypedDict):
    negotiation_id: str


class PublishProtocolEventPayload(TypedDict):
    event: dict[str, Any]


class AppendProtocolEventReceiptPayload(TypedDict):
    receipt: dict[str, Any]
    original_actor: str


class PersistEvidenceRecordPayload(TypedDict):
    record: dict[str, Any]
    original_actor: str


class ReadEvidenceRecordPayload(TypedDict, total=False):
    record_id: str
    version: int
    record_digest: str


class RegisterMandateAuthorityPayload(TypedDict):
    authority: dict[str, Any]
    original_actor: str


class AppendMandateRevisionPayload(TypedDict):
    revision: dict[str, Any]
    original_actor: str


class ReadMandateRevisionsPayload(TypedDict):
    mandate_id: str


class ApplyListingClaimPayload(TypedDict, total=False):
    """Actor-authored compact claim intent.

    Identity fields belong to the listing/claim.  Merchant, issuer, version,
    history, logical time, event identifiers, and digests are deliberately not
    accepted from an agent; Platform derives and seals them from World state.
    Exact required fields depend on ``operation``.
    """

    claim_id: str
    listing_id: str
    subject: str
    operation: Literal["draft", "publish", "correct", "retract"]
    content: dict[str, Any]
    evidence_record_ids: list[str]
    reason: str


class WorldApplyListingClaimPayload(TypedDict):
    """Trusted Platform-to-World sealed claim payload."""

    claim: dict[str, Any]
    original_actor: str


class ReadListingClaimPayload(TypedDict):
    claim_id: str


class ReadListingClaimsPayload(TypedDict):
    listing_id: str


class AfterSalesIntentPayload(TypedDict, total=False):
    """Compact actor-authored after-sales intent without authority fields."""

    order_id: str
    reason: str
    requested_qty: int
    evidence_ids: list[str]
    request_id: str
    authorization_id: str
    received_qty: int
    condition: str
    case_id: str
    replacement_sku_id: str
    dispute_id: str
    evidence_id: str
    position: Literal["accept", "contest"]


class WorldApplyAfterSalesIntentPayload(TypedDict):
    """Platform-to-World wrapper; World derives all mutable commerce state."""

    intent: dict[str, Any]
    original_actor: str


class WorldCompleteLedgerReconciliationPayload(TypedDict):
    """Service-only accounting request over World-authored ledger receipts."""

    order_id: str
    request_id: str
    original_actor: str


class AfterSalesUpdatedPayload(TypedDict):
    """Safe Platform acknowledgement of one committed or idempotent command."""

    operation: str
    order_id: str
    disposition: Literal["committed", "idempotent"]
    result_table: str
    result_key: str
    result_digest: str
    references: dict[str, str]


class AfterSalesHistoryReadPayload(TypedDict):
    """Actor request for an owner-scoped World lifecycle projection."""

    order_id: str


class AfterSalesPolicyReadPayload(TypedDict):
    """Actor request for one merchant's public after-sales terms."""

    merchant_id: str


class WorldAfterSalesHistoryReadPayload(TypedDict):
    """Trusted Platform read preserving the authenticated actor identity."""

    order_id: str
    original_actor: str


class WorldAfterSalesPolicyReadPayload(TypedDict):
    """Trusted Platform request for a sanitized public policy projection."""

    merchant_id: str
    original_actor: str


class WorldApplyPaymentIntentPayload(TypedDict):
    """PSP-only wrapper around a compact World payment transition."""

    intent: dict[str, Any]
    original_actor: str


class WorldApplyPackingIntentPayload(TypedDict):
    """Fulfillment-only wrapper around a compact World packing transition."""

    intent: dict[str, Any]
    original_actor: str


class WorldPublishAfterSalesPolicyPayload(TypedDict):
    """Policy-service-only wrapper around merchant-authored public terms."""

    intent: dict[str, Any]
    original_actor: str


class AfterSalesSnapshotPayload(TypedDict):
    """Safe result of one caller-scoped after-sales read."""

    resource: Literal[
        "payment_history",
        "ledger_history",
        "packing_history",
        "after_sales_history",
        "policy",
    ]
    records: list[dict[str, Any]]


class AdjustPricePayload(TypedDict):
    """``commerce.adjust_price`` — change a SKU's public list price.

    Forwarded by :class:`CatalogPolicy` as a compact
    ``world.apply_catalog_mutation`` intent.  World derives owner and revision.
    """

    sku_id: str
    list_price: int     # minor units, strictly positive


class ReceiveShipmentPayload(TypedDict, total=False):
    """``commerce.receive_shipment`` — record an inbound stock lot.

    Forwarded by :class:`CatalogPolicy` as
    ``world.update_inventory`` (``qty_available += qty``).

    ``lot_id`` is optional and carries a merchant-side bookkeeping id;
    the platform does not interpret it.
    """

    sku_id: str         # required
    qty: int            # required; strictly positive
    arrived_at: str     # required; ISO date
    product: str        # optional, merchant-side product label
    lot_id: str         # optional


class UpdateListingPayload(TypedDict, total=False):
    """``commerce.update_listing`` — author / edit / delist catalog entries.

    Forwarded by :class:`CatalogPolicy` as a compact
    ``world.apply_catalog_mutation`` intent.

    The ``fields`` shape varies by ``op``; see
    :data:`UpdateListingFieldsPublish` and friends below. ``op_id``
    is the idempotency token recorded by the ``listing-publish`` skill.
    """

    op: Literal["publish", "update", "delist"]  # required
    sku_id: str                                  # required
    product: str                                 # required
    fields: dict[str, Any]                       # required (may be {} for delist)
    op_id: str                                   # optional


class WorldApplyCatalogMutationPayload(TypedDict):
    """Trusted Platform-to-World wrapper around a compact actor intent.

    ``intent`` contains only ``op``, ``sku_id``, and ``fields``.  Actor
    identity and idempotency come from authenticated envelope fields; a model
    never supplies a sealed listing, owner, revision, or request fingerprint.
    """

    intent: dict[str, Any]
    original_actor: str


class WorldPublishPricingPolicyPayload(TypedDict):
    """Compact merchant intent; World seals owner, revision, digest, and tick."""

    intent: dict[str, Any]
    original_actor: str


class ReadPricingPoliciesPayload(TypedDict):
    """Read one exact market, merchant, and policy revision stream."""

    market_id: str
    merchant_id: str
    policy_id: str


class UpdateListingFieldsPublish(TypedDict, total=False):
    """``fields`` shape when ``op="publish"``.

    Note the deliberate absence of ``floor_price`` and ``merchant_id``:
    floor is PRIVATE_UTILITY ([[project-privacy-invariant]]) and never
    on the wire; merchant_id is derived from sender at the platform
    boundary.
    """

    list_price: int                       # required, minor units
    attributes: dict[str, Any]            # required (may be {})
    permitted_claims: list[str]           # required (may be [])
    must_not_claim: list[str]             # optional
    inventory: int                        # optional, default 0
    status: str                           # optional, default "active"
    category: str                         # optional
    name: str                             # optional
    currency: str                         # optional, default "USD"


class UpdateListingFieldsUpdate(TypedDict, total=False):
    """``fields`` shape when ``op="update"``.

    ``list_price`` is forbidden here — list price moves are owned by
    the markup / markdown skills, not by the authoring path.
    """

    attributes: dict[str, Any]
    permitted_claims: list[str]
    must_not_claim: list[str]
    status: str
    category: str
    name: str


class UpdateListingFieldsDelist(TypedDict):
    """``fields`` shape when ``op="delist"``. Always exactly ``{"status": "delisted"}``."""

    status: Literal["delisted"]


__all__ = [
    "CartRequestLinePayload",
    "CreateCartQuoteRequestPayload",
    "CheckoutCartPayload",
    "RequestCartQuotePayload",
    "ReadSupplyStatePayload",
    "UpdateSupplyPayload",
    "AllocateFulfillmentPayload",
    "ReadShipmentPayload",
    "RecordShipmentStatusPayload",
    "ResolveShipmentPayload",
    "IssueProtocolEventPayload",
    "AdvanceMarketClockPayload",
    "ProcessProtocolEventPayload",
    "PersistEvidenceRecordPayload",
    "ReadEvidenceRecordPayload",
    "RegisterMandateAuthorityPayload",
    "AppendMandateRevisionPayload",
    "ReadMandateRevisionsPayload",
    "ApplyListingClaimPayload",
    "WorldApplyListingClaimPayload",
    "ReadListingClaimPayload",
    "ReadListingClaimsPayload",
    "AfterSalesIntentPayload",
    "WorldApplyAfterSalesIntentPayload",
    "WorldCompleteLedgerReconciliationPayload",
    "AfterSalesUpdatedPayload",
    "AfterSalesHistoryReadPayload",
    "AfterSalesPolicyReadPayload",
    "WorldAfterSalesHistoryReadPayload",
    "WorldAfterSalesPolicyReadPayload",
    "WorldApplyPaymentIntentPayload",
    "WorldApplyPackingIntentPayload",
    "WorldPublishAfterSalesPolicyPayload",
    "AfterSalesSnapshotPayload",
    "AdjustPricePayload",
    "ReceiveShipmentPayload",
    "UpdateListingPayload",
    "WorldApplyCatalogMutationPayload",
    "WorldPublishPricingPolicyPayload",
    "ReadPricingPoliciesPayload",
    "UpdateListingFieldsPublish",
    "UpdateListingFieldsUpdate",
    "UpdateListingFieldsDelist",
]
