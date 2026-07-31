"""Passive data types for the world package.

Per the architecture skill: no methods beyond ``__post_init__`` placeholders.
Behavior lives in ``state.py`` / ``tools.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, NewType

if TYPE_CHECKING:
    from world.after_sales_core import AfterSalesPolicyRevision
    from world.after_sales_persistence import AfterSalesWrite
    from world.market_governance_persistence import (
        GovernancePolicyEnvelope,
        GovernanceRecordEnvelope,
    )
    from world.payment_fulfillment import PackingRecord, PaymentStateRecord
    from protocol.evidence_records import (
        EvidenceRecord,
        MandateRevision,
        MandateRevisionAuthority,
    )
    from protocol.event_receipts import ProtocolEvent, ProtocolEventReceipt
    from protocol.listing_claims import ListingClaim
    from protocol.matching import MatchAcceptance, MatchCertificate, SearchSession
    from protocol.negotiation_state import NegotiationEvent, NegotiationThread
    from protocol.cart_quote_state import PersistentCartQuote
    from protocol.cart_quote_request import PersistentCartQuoteRequest
    from protocol.pricing_policy import PricingPolicyRevision
    from protocol.supply_authority import SupplyPurchaseAuthority

# --- Identifiers --------------------------------------------------------

AgentId = NewType("AgentId", str)  # e.g. "buyer", "merchant", "buyer:intent"
SkuId = NewType("SkuId", str)
OrderId = NewType("OrderId", str)
TxnId = NewType("TxnId", str)
DisputeId = NewType("DisputeId", str)
RulingId = NewType("RulingId", str)
ReviewId = NewType("ReviewId", str)
ExchangeId = NewType("ExchangeId", str)
QuoteId = NewType("QuoteId", str)
OrderGroupId = NewType("OrderGroupId", str)
ShipmentId = NewType("ShipmentId", str)


# --- Money --------------------------------------------------------------


@dataclass(frozen=True)
class Money:
    """A monetary amount in a fixed currency (USD by default)."""

    amount: Decimal
    currency: str = "USD"


# --- Catalog / Inventory ----------------------------------------------


@dataclass(frozen=True)
class Listing:
    """A catalog row. Public surface, no private floor price exposed via tools."""

    sku_id: SkuId
    category: str
    name: str
    attributes: dict[str, Any]
    list_price: Money
    merchant_id: AgentId
    # Shared product identity across merchant-specific listings. ``sku_id``
    # remains the globally unique listing key for backward compatibility.
    product_id: str | None = None


@dataclass(frozen=True)
class InventoryRow:
    """Inventory for one SKU.

    ``qty_available`` is the public availability basis; exact rows are still
    only exposed through caller-scoped world reads.
    """

    sku_id: SkuId
    merchant_id: AgentId
    qty_available: int
    qty_reserved: int = 0
    # Additive supply metadata. Defaults preserve historical constructors.
    eta_day: int = 0
    version: int = 1
    # Durable per-SKU operation history.  This is part of authoritative World
    # state so an in-memory World rehydrated from a snapshot preserves scoped
    # idempotency exactly like the SQLite backend does across process restarts.
    supply_events: tuple["SupplyEventRecord", ...] = ()


@dataclass(frozen=True)
class SupplyState:
    """World-authored inventory/ETA/price projection for one SKU."""

    sku_id: SkuId
    merchant_id: AgentId
    available_qty: int
    reserved_qty: int
    eta_day: int
    unit_price_cents: int
    version: int


@dataclass(frozen=True)
class SupplyEventRecord:
    """One durable supply idempotency record embedded in its inventory row."""

    scope: str
    idempotency_key: str
    sku_id: SkuId
    qty_delta: int
    eta_day: int | None
    unit_price_cents: int | None
    expected_version: int | None
    original_actor: str
    outcome: SupplyState


# --- Orders / Receipts ------------------------------------------------


class OrderState(str, Enum):
    """Lifecycle states for an order."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    BACKORDERED = "backordered"
    PARTIALLY_SETTLED = "partially_settled"
    SETTLED = "settled"
    DISPATCHED = "dispatched"
    RETURNED = "returned"
    REFUNDED = "refunded"
    EXCHANGED = "exchanged"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Order:
    """An order through its lifecycle."""

    order_id: OrderId
    buyer_id: AgentId
    merchant_id: AgentId
    sku_id: SkuId
    qty: int
    agreed_price: Money
    state: OrderState
    # Authoritative arrival priority for scarce-stock batch allocation.  Zero
    # keeps legacy orders outside ordered T6 competition unless explicitly set.
    request_order: int = 0


@dataclass(frozen=True)
class Receipt:
    """A settlement receipt — the artifact ``settle`` produces."""

    txn_id: TxnId
    ts: str
    order_id: OrderId
    buyer_id: AgentId
    merchant_id: AgentId
    sku_id: SkuId
    qty: int
    price: Money
    idempotency_key: str
    # World-owned economic direction.  Legacy constructors remain charges;
    # every reversal path must set ``refund`` explicitly.  Reconciliation must
    # never infer this fact from transaction-id naming conventions.
    effect: Literal["charge", "refund"] = "charge"


# --- Multi-line cart quote / settlement -------------------------------


@dataclass(frozen=True)
class CartLine:
    """One authoritative priced line in a Platform cart quote."""

    sku_id: SkuId
    merchant_id: AgentId
    qty: int
    list_unit_price: Money
    unit_price: Money
    line_total: Money
    tier_min_qty: int | None = None
    bundle_discount_bps: int = 0


@dataclass(frozen=True)
class FeeComponent:
    """One named fee included in an authoritative cart total."""

    fee_id: str
    kind: str
    scope: str
    amount: Money
    basis: str


@dataclass(frozen=True)
class CartQuote:
    """Platform-authored immutable quote bound by ``quote_hash``."""

    quote_id: QuoteId
    buyer_id: AgentId
    requested_by: AgentId
    lines: tuple[CartLine, ...]
    subtotal: Money
    fee_breakdown: tuple[FeeComponent, ...]
    grand_total: Money
    issued_at_tick: int
    expires_at_tick: int
    quote_hash: str


@dataclass(frozen=True)
class OrderGroup:
    """Authoritative group payment record for one atomic cart checkout.

    ``txn_ids`` bind the line-item ledger receipts, ``fee_breakdown`` binds all
    non-line charges, and ``grand_total`` must equal the receipt subtotal plus
    those charges. ``quote_hash`` binds the exact World-issued quote.
    """

    order_group_id: OrderGroupId
    quote_id: QuoteId
    buyer_id: AgentId
    merchant_ids: tuple[AgentId, ...]
    order_ids: tuple[OrderId, ...]
    txn_ids: tuple[TxnId, ...]
    subtotal: Money
    fee_breakdown: tuple[FeeComponent, ...]
    grand_total: Money
    quote_hash: str
    idempotency_key: str
    state: str = "settled"


@dataclass(frozen=True)
class OrderTimeline:
    """Authoritative logical-time evidence for one order.

    Ticks are allocated only by the World; envelope/receipt timestamps are
    deliberately not used for lifecycle authorization. ``return_window_ticks``
    is captured from the listing at settlement and remains ``None`` for legacy
    orders, preserving the pre-window refund behavior.
    """

    order_id: OrderId
    buyer_id: AgentId
    merchant_id: AgentId
    settled_at_tick: int | None = None
    dispatched_at_tick: int | None = None
    return_window_ticks: int | None = None
    return_authorized_at_tick: int | None = None
    returned_at_tick: int | None = None
    refunded_at_tick: int | None = None


@dataclass(frozen=True)
class FulfillmentAllocation:
    """Deterministic quantity allocation for one order.

    ``Order.qty`` remains the requested quantity for backward compatibility.
    This record makes the actual paid/reserved quantity explicit and preserves
    the invariant ``requested_qty == fulfilled_qty + backordered_qty``.  A
    fully backordered allocation has no receipt transaction.
    """

    order_id: OrderId
    buyer_id: AgentId
    merchant_id: AgentId
    sku_id: SkuId
    requested_qty: int
    fulfilled_qty: int
    backordered_qty: int
    receipt_txn_id: TxnId | None
    created_by: AgentId
    idempotency_key: str
    # Present for World-computed merchant allocation batches; ``None`` for the
    # older single-order partial-settlement path.  Persisting it on each row
    # makes a batch reconstructible after an in-memory or SQLite restart.
    allocation_id: str | None = None


@dataclass(frozen=True)
class AllocationBatch:
    """One merchant-authored, World-computed atomic allocation outcome."""

    allocation_id: str
    merchant_id: AgentId
    sku_id: SkuId
    priority_order_ids: tuple[OrderId, ...]
    allocations: tuple[FulfillmentAllocation, ...]
    created_by: AgentId
    idempotency_key: str


class ShipmentStatus(str, Enum):
    """Authoritative logistics states in progression order."""

    IN_TRANSIT = "in_transit"
    DELAYED = "delayed"
    MISSING_SCAN = "missing_scan"
    LOST = "lost"
    DELIVERED = "delivered"


class ShipmentResolution(str, Enum):
    """Party decision after a shipment exception."""

    WAIT = "wait"
    REPLACEMENT = "replacement"
    REFUND = "refund"


@dataclass(frozen=True)
class ShipmentStatusEvent:
    """One World-timestamped, idempotent logistics observation."""

    event_id: str
    status: ShipmentStatus
    logical_time: int
    idempotency_key: str | None = None
    shipment_version: int | None = None


@dataclass(frozen=True)
class Shipment:
    """Shipment state plus its complete append-only status history."""

    shipment_id: ShipmentId
    order_id: OrderId
    buyer_id: AgentId
    merchant_id: AgentId
    original_sku_id: SkuId
    status: ShipmentStatus
    status_history: tuple[ShipmentStatusEvent, ...]
    resolution: ShipmentResolution | None = None
    replacement_sku_id: SkuId | None = None
    version: int = 1
    resolution_idempotency_key: str | None = None
    resolved_by: AgentId | None = None
    resolution_version: int | None = None
    resolution_history_length: int | None = None


@dataclass(frozen=True)
class Exchange:
    """A completed, like-for-like replacement linked to its original order.

    Exchange completion moves inventory but never creates a ledger receipt.
    The link is therefore the authoritative explanation for a replacement
    order that is dispatchable without a second payment.
    """

    exchange_id: ExchangeId
    original_order_id: OrderId
    replacement_order_id: OrderId
    buyer_id: AgentId
    merchant_id: AgentId
    original_sku_id: SkuId
    replacement_sku_id: SkuId
    qty: int
    created_by: AgentId
    idempotency_key: str


@dataclass(frozen=True)
class TableWrite:
    """One row mutation inside an atomic transaction: before/after snapshots and
    whether the write created or updated the row."""

    table: str
    key: str
    op: str  # "create" | "update"
    before: Any  # row snapshot before the write (None on create)
    after: Any  # row snapshot after the write


@dataclass(frozen=True)
class TransactionStateDiff:
    """Observability artifact for a multi-row atomic write (settle / refund):
    each table write's before/after + the invariants the transaction upheld.
    Recorded to a sidecar for audit/replay; the scorer NEVER reads it — it is not
    a judgment source (only the audit log + trace + snapshot are)."""

    txn: str  # "settle" | "refund"
    order_id: str
    table_writes: "tuple[TableWrite, ...]"
    invariants_held: "tuple[str, ...]"


WORLD_COMMIT_SCHEMA = "cwe.world-commit.v1"


@dataclass(frozen=True)
class WorldCommitRecord:
    """One authoritative, append-only World commit.

    A record is emitted only after a mutation has committed.  Single-row
    :meth:`world.state.World.write` calls produce ``commit_kind='write'``;
    atomic lifecycle operations produce one ``commit_kind='transaction'``
    record containing every row mutation in commit order.  Failed operations
    and idempotent no-op retries never produce records.

    ``authority_action`` names the World capability that authorized the write;
    ``actor_id`` names the authenticated caller when the transaction API has
    one.  They are deliberately separate so an action name is never presented
    as an actor identity. ``request_fingerprint`` binds compact actor intent
    when the authority method uses actor-scoped idempotency. Episode export adds
    a per-window SHA-256 chain to the canonical JSONL representation without
    mutating these World-owned records.
    """

    sequence: int
    commit_id: str
    commit_kind: str  # "write" | "transaction"
    operation: str
    authority_action: str
    actor_id: str | None
    idempotency_key: str | None
    subject_id: str
    table_writes: "tuple[TableWrite, ...]"
    invariants_held: "tuple[str, ...]"
    # Optional canonical request identity for actor-scoped authority methods.
    # The catalog mutation path sets this directly and also persists it in the
    # authority operation row, enabling exact audit-to-state joins.
    request_fingerprint: str | None = None
    schema_version: str = WORLD_COMMIT_SCHEMA


# --- Reputation -------------------------------------------------------


@dataclass(frozen=True)
class ReputationScore:
    """A merchant's reputation as seen on the public surface."""

    merchant_id: AgentId
    rolling_avg: float
    n_settled: int
    n_disputed: int


@dataclass(frozen=True)
class ReputationSettlementSource:
    """One authenticated request identity observed for a settlement event.

    A transport retry may present the same source identity again. A later
    request can use a different idempotency key for the same settlement. Both
    cases remain attached to one World-owned event instead of creating another
    reputation sample.
    """

    source_actor: str
    source_request_id: str
    source_idempotency_key: str


@dataclass(frozen=True)
class ReputationSettlement:
    """World-owned proof that one settled transaction affected reputation.

    ``event_id`` and ``txn_id`` identify one immutable commercial outcome.
    ``sources`` is the append-only set of authenticated request aliases that
    attempted to apply that outcome. ``outcome`` is the score after the single
    successful application and never changes when aliases are appended.
    """

    event_id: str
    order_id: OrderId
    txn_id: TxnId
    merchant_id: AgentId
    sources: tuple[ReputationSettlementSource, ...]
    outcome: ReputationScore


# --- Disputes / Rulings -----------------------------------------------


class DisputeState(str, Enum):
    """Lifecycle states for a dispute."""

    OPEN = "open"
    UNDER_REVIEW = "under_review"
    RULED = "ruled"
    WITHDRAWN = "withdrawn"


@dataclass(frozen=True)
class Dispute:
    """A dispute filed against an order."""

    dispute_id: DisputeId
    order_id: OrderId
    filed_by: AgentId
    against: AgentId
    reason: str
    state: DisputeState


@dataclass(frozen=True)
class Ruling:
    """An adjudicator's ruling on a dispute."""

    ruling_id: RulingId
    dispute_id: DisputeId
    in_favor_of: AgentId
    rationale: str
    refund_amount: Money | None


# --- Social commerce (item 7) -----------------------------------------


@dataclass(frozen=True)
class Friendship:
    """One buyer's friend list — the social graph as seen by that buyer.

    Stored per-buyer (key = ``buyer_id``) and read caller-scoped: a buyer sees
    only their own friends. ``friends`` are the buyer-side agent ids whose
    purchase reviews this buyer is allowed to consult.
    """

    buyer_id: AgentId
    friends: tuple[AgentId, ...]


@dataclass(frozen=True)
class Review:
    """A buyer's recorded experience with a product — a purchase + its rating.

    The social signal: a friend bought ``sku_id`` from ``merchant_id`` and rated
    it. A buyer reaches these only through friendship-gated reads
    (``WorldTools.get_friend_reviews``), so it models "ask your friends what
    they thought", not a public review wall.
    """

    review_id: ReviewId
    reviewer_id: AgentId  # the buyer who bought + reviewed (a friend)
    sku_id: SkuId
    merchant_id: AgentId
    rating: int  # 1..5
    text: str = ""


@dataclass(frozen=True)
class AuthorityOperationRecord:
    """Durable actor-scoped idempotency binding for authority subsystems."""

    operation_key: str
    scope: str
    actor_id: str
    idempotency_key: str
    request_fingerprint: str
    outcome_table: str
    outcome_key: str
    # Catalog rows are mutable, so pointing at ``catalog[sku_id]`` alone would
    # make a late retry return a newer operation's value.  The immutable result
    # snapshot preserves exact retry semantics without putting operation state
    # into ``Listing.attributes``.  Other authority scopes keep this ``None``.
    outcome_listing: Listing | None = None


# --- Snapshot ----------------------------------------------------------


@dataclass(frozen=True)
class WorldSnapshot:
    """A frozen view of every world table; used by the replay verifier.

    The shape is additive. New tables become new keys; existing keys never
    change shape.
    """

    catalog: tuple[Listing, ...] = ()
    inventory: dict[SkuId, InventoryRow | int] = field(default_factory=dict)
    orders: tuple[Order, ...] = ()
    ledger: tuple[Receipt, ...] = ()
    reputation: dict[AgentId, ReputationScore] = field(default_factory=dict)
    reputation_settlements: tuple[ReputationSettlement, ...] = ()
    disputes: tuple[Dispute, ...] = ()
    rulings: tuple[Ruling, ...] = ()
    # Authoritative social tables (item 7). Exposed so an external scorer can
    # observe the friend graph + reviews WITHOUT reading an agent's TRUST memory
    # (which is not a truth source). Default empty preserves backward compat for
    # every existing WorldSnapshot(...) construction.
    friendships: tuple[Friendship, ...] = ()
    reviews: tuple[Review, ...] = ()
    fulfillments: tuple[FulfillmentAllocation, ...] = ()
    exchanges: tuple[Exchange, ...] = ()
    # World-owned deterministic time. It never derives from wall-clock or an
    # agent-supplied envelope timestamp.
    logical_time: int = 0
    order_timelines: tuple[OrderTimeline, ...] = ()
    order_groups: tuple[OrderGroup, ...] = ()
    shipments: tuple[Shipment, ...] = ()
    # Discovery and certification are durable World facts.  Keeping them in
    # snapshots makes restart and replay verification cover the same trust
    # boundary that settlement consumes.
    search_sessions: tuple["SearchSession", ...] = ()
    match_acceptances: dict[str, "MatchAcceptance"] = field(default_factory=dict)
    match_certificates: tuple["MatchCertificate", ...] = ()
    supply_purchase_authorities: tuple["SupplyPurchaseAuthority", ...] = ()
    # Authority-issued protocol events and exact actor decisions are World
    # facts, not runtime chat artifacts.  Their sealed records are included in
    # snapshots so restart and replay compare the full T10 trust boundary.
    protocol_events: tuple["ProtocolEvent", ...] = ()
    protocol_receipts: tuple["ProtocolEventReceipt", ...] = ()
    # Negotiation events are append-only World facts.  The thread rows are
    # deterministic materializations whose accepted terminal state embeds the
    # agreement.  Both are participant-scoped on read.
    negotiation_events: tuple["NegotiationEvent", ...] = ()
    negotiation_threads: tuple["NegotiationThread", ...] = ()
    pricing_policy_revisions: tuple["PricingPolicyRevision", ...] = ()
    persistent_cart_quote_requests: tuple["PersistentCartQuoteRequest", ...] = ()
    persistent_cart_quotes: tuple["PersistentCartQuote", ...] = ()
    # Materialized from authoritative committed order writes.  Keeping the
    # ledger outside ``Order`` preserves the passive row schema while making
    # event freshness identical across snapshot/restart backends.
    order_state_revisions: dict[str, int] = field(default_factory=dict)
    # Evidence, principal-authorized mandate histories, and merchant listing
    # claims are authoritative World records. They are intentionally separate
    # from free-form listing attributes and agent memory.
    evidence_records: tuple["EvidenceRecord", ...] = ()
    mandate_authorities: tuple["MandateRevisionAuthority", ...] = ()
    mandate_revisions: tuple["MandateRevision", ...] = ()
    listing_claims: tuple["ListingClaim", ...] = ()
    authority_operations: tuple[AuthorityOperationRecord, ...] = ()
    # First-class payment, pre-dispatch fulfillment, policy, and typed
    # after-sales state.  These are authoritative World collections, never a
    # benchmark projection or scorer sidecar.
    payment_states: tuple["PaymentStateRecord", ...] = ()
    packing_records: tuple["PackingRecord", ...] = ()
    after_sales_policies: tuple["AfterSalesPolicyRevision", ...] = ()
    after_sales_records: tuple["AfterSalesWrite", ...] = ()
    governance_policies: tuple["GovernancePolicyEnvelope", ...] = ()
    governance_records: tuple["GovernanceRecordEnvelope", ...] = ()
