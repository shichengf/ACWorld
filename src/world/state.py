"""The world state service: the single writer of every world table.

See WORLD_CLASSES.md §2 for the surface; §7 for the replay invariant the
write/snapshot pair must preserve.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal, localcontext
from threading import RLock
from typing import Any, Literal, cast

from protocol.errors import SchemaError
from protocol.cart_quote_state import (
    CartQuoteStaleError,
    PersistentCartQuote,
    apply_cart_quote_record,
    assert_cart_quote_snapshots_current,
    assert_cart_quote_usable,
    validate_persistent_cart_quote,
)
from protocol.cart_quote_request import (
    CartQuoteRequestAuthorityError,
    CartQuoteRequestLine,
    CartQuoteRequestStaleError,
    PersistentCartQuoteRequest,
    assert_cart_quote_request_usable,
    build_persistent_cart_quote_request,
    validate_persistent_cart_quote_request,
)
from protocol.evidence_records import (
    EvidenceRecord,
    MandateRevision,
    MandateRevisionAuthority,
    validate_evidence_record,
    validate_mandate_revision,
    validate_mandate_revision_sequence,
)
from protocol.event_receipts import (
    ProtocolEvent,
    ProtocolEventAuthorityError,
    ProtocolEventReceipt,
    ProtocolEventSchemaError,
    ProtocolEventStaleError,
    apply_protocol_event,
    apply_protocol_receipt,
    build_protocol_event_receipt,
    replay_protocol_events,
    replay_protocol_receipts,
    validate_protocol_event,
    validate_protocol_event_receipt,
    validate_protocol_receipt_stream,
)
from protocol.matching import (
    MatchAcceptance,
    MatchAcceptanceRejected,
    MatchCertificate,
    MatchValidationError,
    SearchSession,
    canonical_digest,
    issue_match_certificate as build_match_certificate,
    validate_search_session,
)
from protocol.listing_claims import ListingClaim, validate_listing_claim
from protocol.negotiation_state import (
    PROPOSE_OFFER,
    NegotiationBinding,
    NegotiationEvent,
    NegotiationSchemaError,
    NegotiationThread,
    apply_negotiation_event,
    build_negotiation_event,
    build_next_negotiation_event,
    replay_negotiation_events,
    validate_negotiation_event,
    validate_negotiation_thread,
)
from protocol.pricing_policy import (
    GENESIS_PRICING_POLICY_DIGEST,
    PricingPolicyRevision,
    PricingPolicyStaleError,
    apply_pricing_policy_revision,
    build_pricing_policy_revision,
    replay_pricing_policy_revisions,
    resolve_active_pricing_policy,
    validate_pricing_policy_revision,
)
from protocol.supply_authority import (
    DEFAULT_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS,
    SupplyPurchaseAuthority,
    build_supply_purchase_authority,
    supply_purchase_authority_id,
    validate_supply_purchase_authority,
    validate_supply_purchase_authority_ttl_ticks,
)

from world.catalog_mutations import (
    CatalogMutationIntent,
    apply_catalog_mutation_intent,
    catalog_listings_semantically_equal,
    catalog_mutation_fingerprint,
    catalog_owner_for_actor,
    normalize_catalog_mutation_intent,
)
from world.after_sales_core import (
    AfterSalesCoreTransitionError,
    AfterSalesPolicyRevision,
    PaidCancellationRecord,
    after_sales_intent_fingerprint,
    authoritative_order_digest,
    build_after_sales_policy_revision,
    derive_ledger_reconciliation_source,
    normalize_after_sales_intent,
)
from world.after_sales_persistence import (
    AfterSalesCommit,
    AfterSalesOperationRecord,
    AfterSalesTables,
    AfterSalesWrite,
    CommerceEffect,
    build_after_sales_operation,
    canonical_digest as after_sales_digest,
    physical_after_sales_record_key,
    physical_after_sales_record_lookup_key,
    record_digest as after_sales_record_digest,
)
from world.after_sales_service import (
    AfterSalesCommandResult,
    AfterSalesContextConflict,
    AfterSalesPlanner,
    TrustedAfterSalesContext,
    TrustedReplacementContext,
    load_trusted_after_sales_evidence,
    resolve_current_versioned_record,
    stage_policy_publication,
)
from world.market_governance_world import MarketGovernanceWorldMixin
from world.market_governance_persistence import (
    GovernancePolicyEnvelope,
    GovernanceRecordEnvelope,
)
from world.cart_pricing import (
    CartQuoteIntent,
    PricingPolicyIntent,
    QuoteAuthority,
    build_authoritative_cart_quote,
    cart_quote_intent_fingerprint,
    catalog_snapshot_digest,
    inventory_snapshot_digest,
    normalize_pricing_policy_intent,
    normalize_cart_quote_intent,
    pricing_policy_set_digest,
    pricing_policy_intent_fingerprint,
    pricing_policy_revision_key,
)
from world.evidence_contracts import (
    authority_operation_key,
    authorize_mandate_read,
    authorize_persisted_evidence_read,
    evidence_record_key,
    mandate_authority_to_wire,
    mandate_revision_key,
    reject_embedded_authority_records,
    validate_evidence_append,
    validate_listing_claim_append,
    validate_mandate_append,
    validate_mandate_authority_registration,
)
from world.negotiations import (
    NegotiationIntent,
    negotiation_event_id,
    negotiation_intent_fingerprint,
    negotiation_listing_digest,
    normalize_negotiation_intent,
)
from world.errors import (
    AfterSalesReferenceRejected,
    DisputeNotActionable,
    ExchangeNotActionable,
    FulfillmentNotActionable,
    IdempotencyConflict,
    InvalidOrderTransition,
    LogicalTimeError,
    LifecycleAuthorizationError,
    OrderNotRefundable,
    OrderNotSettleable,
    OutOfStock,
    ReturnWindowClosed,
    ShipmentNotActionable,
    TableNotFound,
    WorldError,
    WriteNotAuthorized,
)
from world.reset import E0Reset, E1Reset
from world.payment_fulfillment import (
    PackingRecord,
    PaymentStateRecord,
    authoritative_payment_receipt_digest,
    derive_dispatch_packing_sequence,
    derive_packing_transition,
    derive_payment_authorization,
    derive_payment_capture,
    derive_payment_resolution,
    normalize_packing_intent,
    normalize_payment_intent,
    packing_record_key,
    payment_state_key,
)
from world.tables import (
    AfterSalesPolicyTable,
    AfterSalesRecordTable,
    AuthorityOperationTable,
    CatalogTable,
    DisputeTable,
    EvidenceRecordTable,
    ExchangeTable,
    FriendshipTable,
    GovernancePolicyTable,
    GovernanceRecordTable,
    FulfillmentTable,
    InventoryTable,
    LedgerTable,
    ListingClaimTable,
    MandateAuthorityTable,
    MandateRevisionTable,
    MatchAcceptanceTable,
    MatchCertificateTable,
    NegotiationEventTable,
    NegotiationThreadTable,
    OrderTable,
    OrderGroupTable,
    OrderTimelineTable,
    PackingRecordTable,
    PaymentStateTable,
    ProtocolEventTable,
    ProtocolReceiptTable,
    PricingPolicyRevisionTable,
    PersistentCartQuoteTable,
    PersistentCartQuoteRequestTable,
    ReputationSettlementTable,
    ReputationTable,
    ReviewTable,
    RulingTable,
    ShipmentTable,
    SearchSessionTable,
    SupplyPurchaseAuthorityTable,
    Table,
)
from world.transactions import (
    validate_listing_owner,
    validate_persisted_order_identity,
    validate_transaction_identity,
)
from world.types import (
    AgentId,
    AllocationBatch,
    AuthorityOperationRecord,
    Dispute,
    DisputeId,
    DisputeState,
    Exchange,
    ExchangeId,
    FeeComponent,
    FulfillmentAllocation,
    InventoryRow,
    Listing,
    Money,
    Order,
    OrderGroup,
    OrderGroupId,
    OrderId,
    OrderState,
    OrderTimeline,
    Receipt,
    QuoteId,
    ReputationScore,
    ReputationSettlement,
    ReputationSettlementSource,
    Ruling,
    RulingId,
    Shipment,
    ShipmentId,
    ShipmentResolution,
    ShipmentStatus,
    ShipmentStatusEvent,
    SkuId,
    SupplyEventRecord,
    SupplyState,
    TableWrite,
    TransactionStateDiff,
    WorldCommitRecord,
    WorldSnapshot,
    TxnId,
)

#: The settle precondition is an ALLOW-LIST (allowlist + default-deny): only an
#: order in one of these pre-settlement states may proceed to settle. Anything
#: not here is either an idempotent replay (_ALREADY_SETTLED) or refused — so a
#: CANCELLED order, or any OrderState added later, is denied until classified.
_SETTLEABLE: "frozenset[OrderState]" = frozenset({
    OrderState.PROPOSED,
    OrderState.ACCEPTED,
})

#: Order states that already imply a completed settle — re-settling one (even
#: under a fresh idempotency key) is a no-op returning the prior receipt, never
#: a second inventory reservation / ledger append.
_ALREADY_SETTLED: "frozenset[OrderState]" = frozenset({
    OrderState.PARTIALLY_SETTLED,
    OrderState.SETTLED,
    OrderState.DISPATCHED,
    OrderState.RETURNED,
    OrderState.REFUNDED,
})

#: Explicit default-deny states for the legacy all-or-nothing settle API.
#: BACKORDERED has already received a single-shot allocation and must not be
#: silently converted into a full settle; EXCHANGED has consumed its payment in
#: a linked replacement. New states must be consciously classified here, in
#: ``_SETTLEABLE``, or in ``_ALREADY_SETTLED``.
_SETTLE_DENIED: "frozenset[OrderState]" = frozenset({
    OrderState.BACKORDERED,
    OrderState.CANCELLED,
    OrderState.EXCHANGED,
})

#: Order states a refund may be applied to (allowlist + default-deny): only a
#: paid order (SETTLED, DISPATCHED, or physically RETURNED) can be refunded.
#: PROPOSED / ACCEPTED (never paid), CANCELLED, or any future state are refused.
_REFUNDABLE: "frozenset[OrderState]" = frozenset({
    OrderState.PARTIALLY_SETTLED,
    OrderState.SETTLED,
    OrderState.DISPATCHED,
    OrderState.RETURNED,
})

#: Paid lifecycle states from which a party may open a dispute. A refunded
#: transaction remains disputable because either side may contest the outcome.
_DISPUTABLE: "frozenset[OrderState]" = frozenset({
    OrderState.PARTIALLY_SETTLED,
    OrderState.SETTLED,
    OrderState.DISPATCHED,
    OrderState.RETURNED,
    OrderState.REFUNDED,
})

_SHIPMENT_TRANSITIONS: dict[ShipmentStatus, frozenset[ShipmentStatus]] = {
    ShipmentStatus.IN_TRANSIT: frozenset({
        ShipmentStatus.DELAYED,
        ShipmentStatus.DELIVERED,
    }),
    ShipmentStatus.DELAYED: frozenset({
        ShipmentStatus.MISSING_SCAN,
        ShipmentStatus.DELIVERED,
    }),
    ShipmentStatus.MISSING_SCAN: frozenset({
        ShipmentStatus.LOST,
        ShipmentStatus.DELIVERED,
    }),
    ShipmentStatus.LOST: frozenset(),
    ShipmentStatus.DELIVERED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class _IdempotencyRecord:
    operation: str
    order: Order
    request_receipt: Receipt
    outcome: Receipt


@dataclass(frozen=True, slots=True)
class _SupplyIdempotencyRecord:
    sku_id: SkuId
    qty_delta: int
    eta_day: int | None
    unit_price_cents: int | None
    expected_version: int | None
    original_actor: str
    outcome: SupplyState


@dataclass(frozen=True, slots=True)
class _AllocationIdempotencyRecord:
    allocation_id: str
    merchant_id: AgentId
    sku_id: SkuId
    priority_order_ids: tuple[OrderId, ...]
    original_actor: str
    outcome: AllocationBatch


@dataclass(frozen=True, slots=True)
class _ShipmentEventIdempotencyRecord:
    shipment_id: ShipmentId
    event_id: str
    status: ShipmentStatus
    original_actor: str
    outcome: Shipment


@dataclass(frozen=True, slots=True)
class _ShipmentResolutionIdempotencyRecord:
    shipment_id: ShipmentId
    resolution: ShipmentResolution
    replacement_sku_id: SkuId | None
    original_actor: str
    outcome: Shipment


class World(MarketGovernanceWorldMixin):
    """State service. Holds tables, gates writes by action, supports reset + snapshot.

    The world does **not** know about agents. Agents reach it only through the
    runtime (which constructs ``WorldTools`` per send and binds it to the
    receiving agent's id).
    """

    def __init__(self, *, mode: Literal["E0", "E1"] = "E0") -> None:
        """Construct an empty world. Tables are populated by ``apply``."""
        self._tables: dict[str, Table[Any, Any]] = {
            "catalog": CatalogTable(),
            "inventory": InventoryTable(),
            "orders": OrderTable(),
            "ledger": LedgerTable(),
            "reputation": ReputationTable(),
            "reputation_settlements": ReputationSettlementTable(),
            "disputes": DisputeTable(),
            "rulings": RulingTable(),
            "friendships": FriendshipTable(),
            "reviews": ReviewTable(),
            "fulfillments": FulfillmentTable(),
            "exchanges": ExchangeTable(),
            "shipments": ShipmentTable(),
            "order_timelines": OrderTimelineTable(),
            "order_groups": OrderGroupTable(),
            "search_sessions": SearchSessionTable(),
            "match_acceptances": MatchAcceptanceTable(),
            "match_certificates": MatchCertificateTable(),
            "supply_purchase_authorities": SupplyPurchaseAuthorityTable(),
            "negotiation_events": NegotiationEventTable(),
            "negotiation_threads": NegotiationThreadTable(),
            "pricing_policy_revisions": PricingPolicyRevisionTable(),
            "persistent_cart_quote_requests": PersistentCartQuoteRequestTable(),
            "persistent_cart_quotes": PersistentCartQuoteTable(),
            "protocol_events": ProtocolEventTable(),
            "protocol_receipts": ProtocolReceiptTable(),
            "evidence_records": EvidenceRecordTable(),
            "mandate_authorities": MandateAuthorityTable(),
            "mandate_revisions": MandateRevisionTable(),
            "listing_claims": ListingClaimTable(),
            "authority_operations": AuthorityOperationTable(),
            "payment_states": PaymentStateTable(),
            "packing_records": PackingRecordTable(),
            "after_sales_policies": AfterSalesPolicyTable(),
            "after_sales_records": AfterSalesRecordTable(),
            "governance_policies": GovernancePolicyTable(),
            "governance_records": GovernanceRecordTable(),
        }
        self._lock = RLock()
        self._idempotency_cache: dict[tuple[str, str], _IdempotencyRecord] = {}
        self._supply_idempotency_cache: dict[
            tuple[str, str], _SupplyIdempotencyRecord
        ] = {}
        self._allocation_idempotency_cache: dict[
            tuple[str, str], _AllocationIdempotencyRecord
        ] = {}
        self._shipment_event_idempotency_cache: dict[
            tuple[str, str], _ShipmentEventIdempotencyRecord
        ] = {}
        self._shipment_resolution_idempotency_cache: dict[
            tuple[str, str], _ShipmentResolutionIdempotencyRecord
        ] = {}
        #: Observability artifacts for the multi-row atomic writes (settle /
        #: refund), drained to a sidecar by the runner/launcher. NOT a scorer
        #: input — only the audit log + trace + snapshot judge an episode.
        self._txn_diffs: list[TransactionStateDiff] = []
        #: Complete authoritative mutation stream for the current evidence
        #: window.  Unlike ``_txn_diffs``, this includes ordinary single-row
        #: writes as well as one atomic record per transaction commit.
        self._commit_journal: list[WorldCommitRecord] = []
        #: Materialized authoritative order-state revision.  This is separate
        #: from ``Order`` so the passive row schema remains backward compatible.
        #: It advances only when the central World commit journal records an
        #: order create/update, and is snapshotted for restart parity.
        self._order_state_revisions: dict[str, int] = {}
        self._mode: Literal["E0", "E1"] = mode
        self._logical_time = 0

    def begin_evidence_window(self) -> int:
        """Return the commit cursor that starts a new episode evidence window.

        Episodes call this immediately after writing ``world.initial.json``.
        Initial seeding and E1 carry-over therefore form the replay baseline,
        while every later committed mutation receives a contiguous sequence.
        The authoritative commit journal itself is never cleared; callers export
        only records at or after the returned cursor.  ``_txn_diffs`` remains a
        legacy per-episode compatibility sidecar and is reset here.
        """

        with self._lock:
            self._txn_diffs.clear()
            return len(self._commit_journal)

    @property
    def commit_journal(self) -> tuple[WorldCommitRecord, ...]:
        """Return a defensive copy of the complete ordered commit journal."""

        with self._lock:
            return tuple(_copy_world_commit(row) for row in self._commit_journal)

    def commits_since(self, cursor: int) -> tuple[WorldCommitRecord, ...]:
        """Return committed records at or after a previously captured cursor."""

        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("commit cursor must be a non-negative integer")
        with self._lock:
            if cursor > len(self._commit_journal):
                raise ValueError("commit cursor is beyond the end of the journal")
            return tuple(
                _copy_world_commit(row) for row in self._commit_journal[cursor:]
            )

    @property
    def logical_time(self) -> int:
        """Current World-owned tick; no wall clock participates."""
        with self._lock:
            return self._logical_time

    def advance_logical_time(self, *, to_tick: int, by_actor: str) -> int:
        """Advance deterministic time from the trusted runtime control plane.

        Agent envelopes cannot choose time: only the exact ``runtime:clock``
        identity may advance it, and a regression is rejected without effect.
        """
        if by_actor != "runtime:clock":
            raise LogicalTimeError("only runtime:clock may advance World logical time")
        if isinstance(to_tick, bool) or not isinstance(to_tick, int) or to_tick < 0:
            raise LogicalTimeError("logical time must be a non-negative integer")
        with self._lock:
            if to_tick < self._logical_time:
                raise LogicalTimeError(
                    f"logical time cannot regress from {self._logical_time} to {to_tick}"
                )
            before = self._logical_time
            self._logical_time = to_tick
            if to_tick != before:
                self._append_transaction_commit(_make_diff(
                    "advance_clock",
                    "world",
                    [("logical_time", "world", "update", before, to_tick)],
                    ("runtime-clock-only", "monotonic", "deterministic"),
                ), authority_action="world.advance_logical_time", actor_id=by_actor)
            return self._logical_time

    def read(self, table: str, key: Any, *, caller: str | None = None) -> Any | None:
        """Return the row at ``(table, key)`` or ``None``.

        Raises:
            TableNotFound: ``table`` is not registered.

        Visibility is delegated to the per-table read policy; ``caller`` is
        passed through for partition-scoped tables.
        """
        row = self._table(table).read(key, caller=caller)
        if table != "order_groups" or row is None:
            return row
        if caller is None or caller.startswith("platform:") or caller == "runtime":
            return row
        if caller == str(row.buyer_id) or caller in {
            str(merchant_id) for merchant_id in row.merchant_ids
        }:
            return row
        return None

    def read_order_state_revision(
        self,
        order_id: OrderId | str,
        *,
        caller: str,
    ) -> int | None:
        """Return the World-owned revision of one caller-visible order.

        The revision is not supplied by an event sender.  It is materialized
        from committed order-row mutations and therefore shares the same
        authority boundary as the order itself.
        """

        with self._lock:
            order = cast(
                "Order | None",
                self._tables["orders"].read(order_id, caller=caller),
            )
            if order is None:
                return None
            return self._order_revision_for_existing(str(order.order_id))

    def read_order_operation_reference(
        self,
        order_id: OrderId | str,
        *,
        caller: str,
    ) -> str | None:
        """Return a digest of the exact visible order state and revision."""

        with self._lock:
            order = cast(
                "Order | None",
                self._tables["orders"].read(order_id, caller=caller),
            )
            if order is None:
                return None
            revision = self._order_revision_for_existing(str(order.order_id))
            return order_operation_reference_digest(order, revision)

    def protocol_events_for_stream(
        self,
        binding_digest: str,
        *,
        caller: str,
    ) -> tuple[ProtocolEvent, ...]:
        """Read one protocol stream through recipient-scoped table policy."""

        with self._lock:
            table = cast(ProtocolEventTable, self._tables["protocol_events"])
            return tuple(
                deepcopy(event)
                for event in table.by_stream(binding_digest)
                if table.read(event.event_id, caller=caller) is not None
            )

    def read_negotiation_event(
        self, event_id: str, *, caller: str
    ) -> NegotiationEvent | None:
        """Return one immutable event only to its participants or control plane."""

        with self._lock:
            event = self._tables["negotiation_events"].read(
                event_id, caller=caller
            )
            return None if event is None else deepcopy(event)

    def read_negotiation_thread(
        self, negotiation_id: str, *, caller: str
    ) -> NegotiationThread | None:
        """Return one materialized thread under participant-scoped ACL."""

        with self._lock:
            thread = self._tables["negotiation_threads"].read(
                negotiation_id, caller=caller
            )
            return None if thread is None else deepcopy(thread)

    def negotiation_events_for_thread(
        self, negotiation_id: str, *, caller: str
    ) -> tuple[NegotiationEvent, ...]:
        """Return a deterministic, sequence-ordered participant-visible stream."""

        with self._lock:
            table = cast(
                NegotiationEventTable, self._tables["negotiation_events"]
            )
            return tuple(
                deepcopy(event)
                for event in table.by_thread(negotiation_id)
                if table.read(event.event_id, caller=caller) is not None
            )

    def apply_negotiation_intent(
        self,
        intent: NegotiationIntent | Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        max_rounds: int,
        deadline_ticks: int,
    ) -> NegotiationEvent:
        """Derive and atomically persist one authenticated negotiation event.

        The actor cannot supply parties, listing ownership, sequence, clock,
        lineage digests, status, or an agreement.  Those facts are derived from
        the authoritative listing and current persisted thread inside the World
        lock.  Exact retries return the immutable prior event and create no
        commit.
        """

        if by_actor != "platform:negotiation":
            raise WriteNotAuthorized(
                "only platform:negotiation may apply negotiation intents"
            )
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise IdempotencyConflict(
                "negotiation idempotency key must be a non-empty string"
            )
        normalized = normalize_negotiation_intent(intent)
        fingerprint = negotiation_intent_fingerprint(
            normalized,
            max_rounds=max_rounds,
            deadline_ticks=deadline_ticks,
        )
        negotiation_id = normalized["negotiation_id"]
        operation_scope = f"negotiation-event:{negotiation_id}"
        with self._lock:
            replay = self._authority_operation_replay(
                scope=operation_scope,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if replay is not None:
                if not isinstance(replay, NegotiationEvent):
                    raise WorldError(
                        "negotiation idempotency outcome has wrong type"
                    )
                return deepcopy(replay)

            event_table = cast(
                NegotiationEventTable, self._tables["negotiation_events"]
            )
            thread_table = cast(
                NegotiationThreadTable, self._tables["negotiation_threads"]
            )
            operation_table = cast(
                AuthorityOperationTable, self._tables["authority_operations"]
            )
            before_thread = thread_table.read(negotiation_id, caller=None)
            event_id = negotiation_event_id(
                negotiation_id, original_actor, idempotency_key
            )
            before_tick = self._logical_time
            event_tick = before_tick + 1
            action_kind = normalized["action_kind"]

            if action_kind == PROPOSE_OFFER:
                if before_thread is not None:
                    raise NegotiationSchemaError(
                        "negotiation already exists"
                    )
                buyer_id, merchant_id = _negotiation_parties(
                    original_actor, normalized["counterparty_id"]
                )
                listing = cast(
                    "Listing | None",
                    self._tables["catalog"].read(
                        SkuId(normalized["sku_id"]), caller=None
                    ),
                )
                if listing is None:
                    raise NegotiationSchemaError("negotiation listing does not exist")
                if str(listing.merchant_id) != merchant_id:
                    raise WriteNotAuthorized(
                        "negotiation merchant does not own the listing"
                    )
                binding = NegotiationBinding(
                    negotiation_id=negotiation_id,
                    buyer_id=buyer_id,
                    merchant_id=merchant_id,
                    offer_id=normalized["offer_id"],
                    sku_id=normalized["sku_id"],
                    listing_digest=negotiation_listing_digest(listing),
                    listing_revision=_catalog_revision(listing),
                    currency=listing.list_price.currency,
                    qty=int(normalized.get("qty", 1)),
                    max_rounds=max_rounds,
                    opened_at_tick=event_tick,
                    expires_at_tick=event_tick + deadline_ticks,
                )
                claimed_round = int(normalized.get("round_no", 1))
                event = build_negotiation_event(
                    binding,
                    event_id=event_id,
                    action_kind=action_kind,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    unit_price=int(normalized["unit_price"]),
                    round_no=claimed_round,
                    sequence_no=1,
                    previous_digest=None,
                    server_tick=event_tick,
                )
            else:
                if before_thread is None:
                    raise NegotiationSchemaError("negotiation does not exist")
                if original_actor not in {
                    before_thread.buyer_id,
                    before_thread.merchant_id,
                }:
                    raise WriteNotAuthorized(
                        "actor is not a negotiation participant"
                    )
                expected_counterparty = (
                    before_thread.merchant_id
                    if original_actor == before_thread.buyer_id
                    else before_thread.buyer_id
                )
                if normalized["counterparty_id"] != expected_counterparty:
                    raise NegotiationSchemaError("negotiation counterparty mismatch")
                if (
                    normalized["offer_id"] != before_thread.offer_id
                    or normalized["sku_id"] != before_thread.sku_id
                ):
                    raise NegotiationSchemaError("negotiation lineage mismatch")
                if max_rounds != before_thread.max_rounds or (
                    deadline_ticks
                    != before_thread.expires_at_tick - before_thread.opened_at_tick
                ):
                    raise NegotiationSchemaError(
                        "negotiation policy configuration changed mid-thread"
                    )
                event = build_next_negotiation_event(
                    before_thread,
                    event_id=event_id,
                    action_kind=action_kind,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    server_tick=event_tick,
                    unit_price=normalized.get("unit_price"),
                    round_no=normalized.get("round_no"),
                )
                binding = NegotiationBinding(
                    negotiation_id=before_thread.negotiation_id,
                    buyer_id=before_thread.buyer_id,
                    merchant_id=before_thread.merchant_id,
                    offer_id=before_thread.offer_id,
                    sku_id=before_thread.sku_id,
                    listing_digest=before_thread.listing_digest,
                    listing_revision=before_thread.listing_revision,
                    currency=before_thread.currency,
                    qty=before_thread.qty,
                    max_rounds=before_thread.max_rounds,
                    opened_at_tick=before_thread.opened_at_tick,
                    expires_at_tick=before_thread.expires_at_tick,
                )

            transition = apply_negotiation_event(
                before_thread,
                event,
                binding=binding,
                server_tick=event_tick,
            )
            operation_key = authority_operation_key(
                operation_scope, original_actor, idempotency_key
            )
            operation = AuthorityOperationRecord(
                operation_key=operation_key,
                scope=operation_scope,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                outcome_table="negotiation_events",
                outcome_key=event.event_id,
            )
            diff_cursor = len(self._txn_diffs)
            commit_cursor = len(self._commit_journal)
            try:
                event_table.write(event.event_id, event)
                thread_table.write(negotiation_id, transition.thread)
                operation_table.write(operation_key, operation)
                self._logical_time = event_tick
                self._append_transaction_commit(
                    _make_diff(
                        "negotiation_event",
                        negotiation_id,
                        [
                            (
                                "negotiation_events",
                                event.event_id,
                                "create",
                                None,
                                event,
                            ),
                            (
                                "negotiation_threads",
                                negotiation_id,
                                "create" if before_thread is None else "update",
                                before_thread,
                                transition.thread,
                            ),
                            (
                                "authority_operations",
                                operation_key,
                                "create",
                                None,
                                operation,
                            ),
                            (
                                "logical_time",
                                "world",
                                "update",
                                before_tick,
                                event_tick,
                            ),
                        ],
                        (
                            "platform-mediated",
                            "participant-and-listing-owner-bound",
                            "actor-scoped-idempotency",
                            "world-clock-and-sequence-derived",
                            "event-thread-agreement-atomic",
                            "private-utility-excluded",
                        ),
                    ),
                    authority_action="world.apply_negotiation_intent",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
            except BaseException:
                self._logical_time = before_tick
                event_table.delete(event.event_id)
                thread_table.delete(negotiation_id)
                if before_thread is not None:
                    thread_table.write(negotiation_id, before_thread)
                operation_table.delete(operation_key)
                del self._txn_diffs[diff_cursor:]
                del self._commit_journal[commit_cursor:]
                raise
            return deepcopy(event)

    def protocol_receipts_for_stream(
        self,
        binding_digest: str,
        *,
        order_id: OrderId | str,
        caller: str,
    ) -> tuple[ProtocolEventReceipt, ...]:
        """Read recipient decisions for one exact protocol stream."""

        with self._lock:
            table = cast(ProtocolReceiptTable, self._tables["protocol_receipts"])
            return tuple(
                deepcopy(receipt)
                for receipt in table.by_order(str(order_id))
                if receipt.binding.binding_digest == binding_digest
                and table.read(receipt.receipt_id, caller=caller) is not None
            )

    def publish_protocol_event(
        self,
        event: ProtocolEvent,
        *,
        by_actor: str,
    ) -> ProtocolEvent:
        """Validate and durably publish an authority-issued protocol event.

        The event is accepted only when its order, parties, current state,
        revision, reference digest, and World logical tick all match
        authoritative state.  Free-form agent text cannot create this record.
        """

        if by_actor != "platform:events":
            raise WriteNotAuthorized(
                "only platform:events may publish protocol events"
            )
        with self._lock:
            order = cast(
                "Order | None",
                self._tables["orders"].read(event.binding.order_id, caller=None),
            )
            if order is None:
                raise ProtocolEventSchemaError(
                    f"protocol event order {event.binding.order_id!r} does not exist"
                )
            _validate_event_order_binding(event, order)
            revision = self._order_revision_for_existing(str(order.order_id))
            _validate_protocol_event_reference(
                event,
                order,
                revision=revision,
                logical_time=self._logical_time,
                certificates=cast(
                    MatchCertificateTable, self._tables["match_certificates"]
                ),
            )
            table = cast(ProtocolEventTable, self._tables["protocol_events"])
            events = table.by_stream(event.binding.binding_digest)
            disposition, _ = apply_protocol_event(
                events,
                event,
                binding=event.binding,
                server_tick=self._logical_time,
                current_order_state=order.state.value,
                current_state_revision=revision,
            )
            if disposition == "idempotent":
                return deepcopy(
                    next(
                        existing
                        for existing in events
                        if existing.event_digest == event.event_digest
                    )
                )

            try:
                self.write(
                    "protocol_events",
                    event.event_id,
                    event,
                    by_action="world.publish_protocol_event",
                    _record_commit=False,
                )
            except BaseException:
                _restore_row(table, event.event_id, None)
                raise
            self._append_transaction_commit(
                _make_diff(
                    "publish_protocol_event",
                    str(order.order_id),
                    [
                        (
                            "protocol_events",
                            event.event_id,
                            "create",
                            None,
                            event,
                        )
                    ],
                    (
                        "platform-events-only",
                        "authoritative-order-binding",
                        "authoritative-state-revision",
                        "reference-verified",
                        "append-only",
                    ),
                ),
                authority_action="world.publish_protocol_event",
                actor_id=by_actor,
                idempotency_key=event.idempotency_key,
            )
            return deepcopy(event)

    def append_protocol_receipt(
        self,
        receipt: ProtocolEventReceipt,
        *,
        by_actor: str,
        original_actor: str,
    ) -> ProtocolEventReceipt:
        """Persist one exact recipient decision for one persisted event.

        Acknowledge and reject are evidence-only operations.  Process is
        deliberately fail-closed until a real CommerceWorld operation handler
        can bind its effect digest to a committed business mutation.
        """

        if by_actor != "platform:events":
            raise WriteNotAuthorized(
                "only platform:events may append protocol receipts"
            )
        if original_actor != receipt.actor_id:
            raise LifecycleAuthorizationError(
                "authenticated receipt actor does not match sealed actor identity"
            )
        if receipt.decision == "process":
            raise WriteNotAuthorized(
                "protocol event processing requires a registered CommerceWorld "
                "operation handler and committed effect digest"
            )
        with self._lock:
            event_table = cast(
                ProtocolEventTable, self._tables["protocol_events"]
            )
            receipt_table = cast(
                ProtocolReceiptTable, self._tables["protocol_receipts"]
            )
            event = event_table.read(receipt.event_id, caller=None)
            if event is None or event.event_digest != receipt.event_digest:
                raise ProtocolEventSchemaError(
                    "receipt does not reference one exact persisted protocol event"
                )
            if original_actor != event.binding.recipient_id:
                raise LifecycleAuthorizationError(
                    "only the persisted event recipient may decide the event"
                )
            order = cast(
                "Order | None",
                self._tables["orders"].read(event.binding.order_id, caller=None),
            )
            if order is None:
                raise ProtocolEventSchemaError(
                    f"protocol receipt order {event.binding.order_id!r} does not exist"
                )
            _validate_event_order_binding(event, order)
            revision = self._order_revision_for_existing(str(order.order_id))
            events = event_table.by_stream(event.binding.binding_digest)
            receipts = tuple(
                prior
                for prior in receipt_table.by_order(str(order.order_id))
                if prior.binding.binding_digest == event.binding.binding_digest
            )
            disposition, _ = apply_protocol_receipt(
                receipts,
                events,
                receipt,
                binding=event.binding,
                server_tick=self._logical_time,
                current_order_state=order.state.value,
                current_state_revision=revision,
            )
            if disposition == "idempotent":
                return deepcopy(
                    next(
                        existing
                        for existing in receipts
                        if existing.receipt_digest == receipt.receipt_digest
                    )
                )

            try:
                self.write(
                    "protocol_receipts",
                    receipt.receipt_id,
                    receipt,
                    by_action="world.append_protocol_receipt",
                    _record_commit=False,
                )
            except BaseException:
                _restore_row(receipt_table, receipt.receipt_id, None)
                raise
            self._append_transaction_commit(
                _make_diff(
                    "append_protocol_receipt",
                    str(order.order_id),
                    [
                        (
                            "protocol_receipts",
                            receipt.receipt_id,
                            "create",
                            None,
                            receipt,
                        )
                    ],
                    (
                        "platform-events-mediated",
                        "recipient-identity-bound",
                        "exact-event-reference",
                        "authoritative-observed-state",
                        "evidence-only-no-business-effect",
                    ),
                ),
                authority_action="world.append_protocol_receipt",
                actor_id=original_actor,
                idempotency_key=receipt.idempotency_key,
            )
            return deepcopy(receipt)

    def process_protocol_event(
        self,
        *,
        event_id: str,
        by_actor: str,
        original_actor: str,
        reason: str,
        idempotency_key: str,
    ) -> ProtocolEventReceipt:
        """Execute one registered commerce operation and persist its receipt.

        The event selects a small, core-owned operation registry.  Scenarios
        cannot provide a state transition or an effect payload.  World checks
        the exact persisted event, recipient, pre-state, revision, and expiry;
        performs the real settlement, dispatch, or refund primitive; then
        writes a process receipt linked to the committed outcome identity.

        The business mutation and process receipt form one rollback boundary
        and one World commit.  The receipt is append-only and retry-safe.  A
        failed or stale precondition, operation, or receipt append has zero
        business effect and creates no receipt.
        """

        if by_actor != "platform:events":
            raise WriteNotAuthorized(
                "only platform:events may process protocol events"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ProtocolEventSchemaError("protocol process reason must be non-empty")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ProtocolEventSchemaError(
                "protocol process idempotency key must be non-empty"
            )
        with self._lock:
            event_table = cast(
                ProtocolEventTable, self._tables["protocol_events"]
            )
            receipt_table = cast(
                ProtocolReceiptTable, self._tables["protocol_receipts"]
            )
            event = event_table.read(event_id, caller=None)
            if event is None:
                raise ProtocolEventSchemaError("protocol event does not exist")
            if original_actor != event.binding.recipient_id:
                raise LifecycleAuthorizationError(
                    "only the persisted event recipient may process it"
                )
            receipts = tuple(
                prior
                for prior in receipt_table.by_order(event.binding.order_id)
                if prior.binding.binding_digest == event.binding.binding_digest
            )
            replay = _protocol_process_receipt_replay(
                receipts,
                event=event,
                actor_id=original_actor,
                reason=reason,
                idempotency_key=idempotency_key,
            )
            if replay is not None:
                return deepcopy(replay)
            order = cast(
                "Order | None",
                self._tables["orders"].read(event.binding.order_id, caller=None),
            )
            if order is None:
                raise ProtocolEventSchemaError("protocol event order does not exist")
            _validate_event_order_binding(event, order)
            revision = self._order_revision_for_existing(str(order.order_id))
            decision_tick = self._logical_time
            operation = _validate_protocol_process_precondition(
                event,
                order=order,
                revision=revision,
                logical_time=decision_tick,
                original_actor=original_actor,
                certificates=cast(
                    MatchCertificateTable, self._tables["match_certificates"]
                ),
            )

            checkpoint = self._protocol_process_checkpoint()
            try:
                outcome_table, outcome_key = self._execute_protocol_operation(
                    event,
                    order=order,
                    operation=operation,
                )
                outcome = self._tables[outcome_table].read(
                    outcome_key, caller=None
                )
                if outcome is None:
                    raise WorldError(
                        "processed protocol operation has no committed outcome row"
                    )
                effect_reference = protocol_operation_effect_reference_digest(
                    event,
                    operation=operation,
                    outcome_table=outcome_table,
                    outcome_key=str(outcome_key),
                )
                receipt = build_protocol_event_receipt(
                    event,
                    receipt_id=_protocol_process_receipt_id(
                        event,
                        actor_id=original_actor,
                        idempotency_key=idempotency_key,
                    ),
                    decision="process",
                    actor_id=original_actor,
                    observed_order_state=event.required_order_state,
                    observed_state_revision=event.required_state_revision,
                    reason=reason,
                    effect_reference_digests=(effect_reference,),
                    logical_tick=decision_tick,
                    idempotency_key=idempotency_key,
                )
                events = event_table.by_stream(event.binding.binding_digest)
                validate_protocol_receipt_stream(
                    (*receipts, receipt),
                    events=events,
                    binding=event.binding,
                )
                self._persist_protocol_process_receipt(receipt)
                self._merge_protocol_process_commit(
                    checkpoint,
                    receipt=receipt,
                    order_id=str(order.order_id),
                    operation=operation,
                    original_actor=original_actor,
                    idempotency_key=idempotency_key,
                )
            except BaseException:
                self._restore_protocol_process_checkpoint(checkpoint)
                raise
            return deepcopy(receipt)

    def _persist_protocol_process_receipt(
        self,
        receipt: ProtocolEventReceipt,
    ) -> None:
        """Write the receipt inside the caller's protocol-process boundary."""

        self.write(
            "protocol_receipts",
            receipt.receipt_id,
            receipt,
            by_action="world.process_protocol_event",
            _record_commit=False,
        )

    def _protocol_process_checkpoint(self) -> dict[str, Any]:
        """Capture every mutable surface touched by a registered operation."""

        table_names = (
            "orders",
            "inventory",
            "ledger",
            "order_timelines",
            "shipments",
            "payment_states",
            "packing_records",
            "protocol_receipts",
        )
        return {
            "tables": {
                name: tuple(
                    (deepcopy(key), _copy_world_value(value))
                    for key, value in self._tables[name].rows.items()
                )
                for name in table_names
            },
            "logical_time": self._logical_time,
            "idempotency_cache": dict(self._idempotency_cache),
            "txn_diffs": list(self._txn_diffs),
            "commit_journal": list(self._commit_journal),
            "order_state_revisions": dict(self._order_state_revisions),
        }

    def _restore_protocol_process_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
    ) -> None:
        """Restore a failed process attempt without leaving partial effects."""

        rows_by_table = cast(
            "Mapping[str, tuple[tuple[Any, Any], ...]]",
            checkpoint["tables"],
        )
        for name, rows in rows_by_table.items():
            table = self._tables[name]
            table.clear()
            for key, value in rows:
                table.write(key, _copy_world_value(value))
        self._logical_time = int(checkpoint["logical_time"])
        self._idempotency_cache = dict(checkpoint["idempotency_cache"])
        self._txn_diffs[:] = list(checkpoint["txn_diffs"])
        self._commit_journal[:] = list(checkpoint["commit_journal"])
        self._order_state_revisions = dict(checkpoint["order_state_revisions"])

    def _merge_protocol_process_commit(
        self,
        checkpoint: Mapping[str, Any],
        *,
        receipt: ProtocolEventReceipt,
        order_id: str,
        operation: str,
        original_actor: str,
        idempotency_key: str,
    ) -> None:
        """Expose the business effect and receipt as one authoritative commit."""

        diff_cursor = len(cast("list[Any]", checkpoint["txn_diffs"]))
        commit_cursor = len(cast("list[Any]", checkpoint["commit_journal"]))
        if (
            len(self._txn_diffs) != diff_cursor + 1
            or len(self._commit_journal) != commit_cursor + 1
        ):
            raise WorldError(
                "registered protocol operation did not produce one World commit"
            )
        business_diff = self._txn_diffs[diff_cursor]
        business_commit = self._commit_journal[commit_cursor]
        receipt_write = TableWrite(
            table="protocol_receipts",
            key=receipt.receipt_id,
            op="create",
            before=None,
            after=receipt,
        )
        combined_writes = (*business_diff.table_writes, receipt_write)
        invariants = (
            *business_diff.invariants_held,
            "platform-events-mediated",
            "recipient-identity-bound",
            "fresh-event-precondition",
            f"registered-operation:{operation}",
            "committed-effect-reference",
            "business-and-receipt-atomic",
            "append-only",
        )
        self._txn_diffs[diff_cursor] = TransactionStateDiff(
            txn="process_protocol_event",
            order_id=order_id,
            table_writes=combined_writes,
            invariants_held=invariants,
        )
        self._commit_journal[commit_cursor] = replace(
            business_commit,
            operation="process_protocol_event",
            authority_action="world.process_protocol_event",
            actor_id=original_actor,
            idempotency_key=idempotency_key,
            subject_id=order_id,
            table_writes=combined_writes,
            invariants_held=invariants,
        )

    def _execute_protocol_operation(
        self,
        event: ProtocolEvent,
        *,
        order: Order,
        operation: str,
    ) -> tuple[str, str]:
        """Dispatch one protocol event to a real core World primitive."""

        effect_key = _protocol_effect_idempotency_key(event)
        if operation == "settle_order":
            receipt = _protocol_payment_receipt(
                event,
                order=order,
                logical_time=self._logical_time,
                refund=False,
            )
            settled = self.settle_order(
                order=order,
                receipt=receipt,
                by_role="platform",
                idempotency_key=effect_key,
            )
            return "ledger", str(settled.txn_id)
        if operation == "dispatch_order":
            self.dispatch_order(
                order_id=order.order_id,
                by_actor=event.binding.recipient_id,
            )
            return "shipments", f"shipment:{order.order_id}"
        if operation == "refund_order":
            receipt = _protocol_payment_receipt(
                event,
                order=order,
                logical_time=self._logical_time,
                refund=True,
                paid_qty=_paid_quantity(
                    cast(FulfillmentTable, self._tables["fulfillments"]),
                    order,
                ),
            )
            refunded = self.refund_order(
                order=order,
                refund_receipt=receipt,
                by_role="platform",
                idempotency_key=effect_key,
            )
            return "ledger", str(refunded.txn_id)
        raise ProtocolEventSchemaError(
            f"unsupported protocol operation {operation!r}"
        )

    def apply_catalog_mutation(
        self,
        intent: CatalogMutationIntent | Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Listing:
        """Atomically apply one authenticated merchant catalog intent.

        The current row, ownership decision, revision stamp, idempotency
        binding, and commit evidence are all produced inside World.  Platform
        cannot submit a pre-sealed listing or a revision number.
        """

        if by_actor != "platform:catalog":
            raise WriteNotAuthorized(
                "only platform:catalog may apply actor catalog mutations"
            )
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise IdempotencyConflict(
                "catalog mutation idempotency key must not be blank"
            )
        normalized = normalize_catalog_mutation_intent(intent)
        fingerprint = catalog_mutation_fingerprint(normalized)
        sku_id = SkuId(normalized["sku_id"])
        with self._lock:
            replay = self._authority_operation_replay(
                scope="catalog-mutation",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if replay is not None:
                if not isinstance(replay, Listing):
                    raise WorldError(
                        "catalog mutation idempotency outcome has wrong type"
                    )
                return replay

            catalog = cast(CatalogTable, self._tables["catalog"])
            operations = cast(
                AuthorityOperationTable, self._tables["authority_operations"]
            )
            before = catalog.read(sku_id, caller=None)
            desired = apply_catalog_mutation_intent(
                before, normalized, original_actor=original_actor
            )
            changed = before is None or not catalog_listings_semantically_equal(
                desired, before
            )
            outcome = (
                _revisioned_listing(desired, before)
                if changed
                else cast(Listing, before)
            )
            operation_key = authority_operation_key(
                "catalog-mutation", original_actor, idempotency_key
            )
            operation = AuthorityOperationRecord(
                operation_key=operation_key,
                scope="catalog-mutation",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                outcome_table="catalog",
                outcome_key=str(sku_id),
                outcome_listing=deepcopy(outcome),
            )
            diff_cursor = len(self._txn_diffs)
            commit_cursor = len(self._commit_journal)
            try:
                if changed:
                    catalog.write(sku_id, outcome)
                operations.write(operation_key, operation)
                writes: list[tuple[str, str, str, Any, Any]] = []
                if changed:
                    writes.append(
                        (
                            "catalog",
                            str(sku_id),
                            "create" if before is None else "update",
                            before,
                            outcome,
                        )
                    )
                writes.append(
                    (
                        "authority_operations",
                        operation_key,
                        "create",
                        None,
                        operation,
                    )
                )
                self._append_transaction_commit(
                    _make_diff(
                        "catalog_mutation",
                        str(sku_id),
                        writes,
                        (
                            "authenticated-original-actor",
                            "merchant-owner-validated",
                            "actor-scoped-idempotency",
                            "world-owned-catalog-revision",
                            "semantic-noop-does-not-revise",
                        ),
                    ),
                    authority_action="world.apply_catalog_mutation",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
            except BaseException:
                _restore_row(catalog, sku_id, before)
                operations.delete(operation_key)
                del self._txn_diffs[diff_cursor:]
                del self._commit_journal[commit_cursor:]
                raise
            return deepcopy(outcome)

    def publish_pricing_policy(
        self,
        intent: PricingPolicyIntent | Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> PricingPolicyRevision:
        """Publish one World-stamped merchant pricing-policy revision."""

        if by_actor != "platform:pricing":
            raise WriteNotAuthorized(
                "only platform:pricing may publish pricing policies"
            )
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise IdempotencyConflict(
                "pricing policy idempotency key must not be blank"
            )
        normalized = normalize_pricing_policy_intent(intent)
        fingerprint = pricing_policy_intent_fingerprint(normalized)
        market_id = normalized["market_id"]
        scope = f"pricing-policy:{market_id}"
        with self._lock:
            replay = self._authority_operation_replay(
                scope=scope,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if replay is not None:
                if not isinstance(replay, PricingPolicyRevision):
                    raise WorldError(
                        "pricing policy idempotency outcome has wrong type"
                    )
                return deepcopy(replay)

            merchant_id = catalog_owner_for_actor(original_actor)
            currency = _pricing_policy_scope_currency(
                cast(CatalogTable, self._tables["catalog"]),
                merchant_id=merchant_id,
                listing_ids=normalized["listing_ids"],
                product_ids=normalized["product_ids"],
            )
            policies = cast(
                PricingPolicyRevisionTable,
                self._tables["pricing_policy_revisions"],
            )
            stream = policies.by_stream(
                market_id, merchant_id, normalized["policy_id"]
            )
            previous = stream[-1] if stream else None
            revision_number = 1 if previous is None else previous.revision + 1
            predecessor = (
                GENESIS_PRICING_POLICY_DIGEST
                if previous is None
                else previous.policy_digest
            )
            before_tick = self._logical_time
            event_tick = before_tick + 1
            expires_after = normalized["expires_after_ticks"]
            revision = build_pricing_policy_revision(
                market_id=market_id,
                merchant_id=merchant_id,
                owner_id=merchant_id,
                policy_id=normalized["policy_id"],
                revision=revision_number,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                currency=currency,
                listing_ids=normalized["listing_ids"],
                product_ids=normalized["product_ids"],
                quantity_tiers=normalized["quantity_tiers"],
                bundle_discounts=normalized["bundle_discounts"],
                bundle_stacking=normalized["bundle_stacking"],
                components=normalized["components"],
                published_at_tick=event_tick,
                effective_from_tick=(
                    event_tick + normalized["effective_after_ticks"]
                ),
                expires_at_tick=(
                    None
                    if expires_after is None
                    else event_tick + expires_after
                ),
                predecessor_digest=predecessor,
            )
            disposition, _ = apply_pricing_policy_revision(
                tuple(row for _, row in policies.all()),
                revision,
                trusted_owner_id=merchant_id,
                server_tick=event_tick,
            )
            if disposition != "append":
                raise WorldError("new pricing policy unexpectedly classified as retry")
            key = pricing_policy_revision_key(
                market_id, merchant_id, revision.policy_id, revision.revision
            )
            operation_key = authority_operation_key(
                scope, original_actor, idempotency_key
            )
            operation = AuthorityOperationRecord(
                operation_key=operation_key,
                scope=scope,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                outcome_table="pricing_policy_revisions",
                outcome_key=key,
            )
            operations = cast(
                AuthorityOperationTable, self._tables["authority_operations"]
            )
            diff_cursor = len(self._txn_diffs)
            commit_cursor = len(self._commit_journal)
            try:
                policies.write(key, revision)
                operations.write(operation_key, operation)
                self._logical_time = event_tick
                self._append_transaction_commit(
                    _make_diff(
                        "publish_pricing_policy",
                        key,
                        [
                            (
                                "pricing_policy_revisions",
                                key,
                                "create",
                                None,
                                revision,
                            ),
                            (
                                "authority_operations",
                                operation_key,
                                "create",
                                None,
                                operation,
                            ),
                            (
                                "logical_time",
                                "world",
                                "update",
                                before_tick,
                                event_tick,
                            ),
                        ],
                        (
                            "merchant-owner-derived",
                            "world-revision-and-clock-derived",
                            "append-only-digest-chain",
                            "actor-scoped-idempotency",
                            "catalog-scope-validated",
                        ),
                    ),
                    authority_action="world.publish_pricing_policy",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
            except BaseException:
                self._logical_time = before_tick
                policies.delete(key)
                operations.delete(operation_key)
                del self._txn_diffs[diff_cursor:]
                del self._commit_journal[commit_cursor:]
                raise
            return deepcopy(revision)

    def pricing_policy_revisions(
        self,
        market_id: str,
        merchant_id: str,
        policy_id: str,
        *,
        caller: str,
    ) -> tuple[PricingPolicyRevision, ...]:
        """Read a policy history through owner and control-plane ACLs."""

        with self._lock:
            table = cast(
                PricingPolicyRevisionTable,
                self._tables["pricing_policy_revisions"],
            )
            return tuple(
                deepcopy(row)
                for row in table.by_stream(market_id, merchant_id, policy_id)
                if table.read(
                    pricing_policy_revision_key(
                        row.market_id,
                        row.merchant_id,
                        row.policy_id,
                        row.revision,
                    ),
                    caller=caller,
                )
                is not None
            )

    def persist_evidence_record(
        self,
        record: EvidenceRecord,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> EvidenceRecord:
        """Append one issuer-authenticated evidence version to World."""

        if by_actor != "platform:evidence":
            raise WriteNotAuthorized(
                "only platform:evidence may persist evidence records"
            )
        if not idempotency_key:
            raise IdempotencyConflict("evidence idempotency key must not be blank")
        with self._lock:
            replay = self._authority_operation_replay(
                scope="evidence",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=record.record_digest,
            )
            if replay is not None:
                if not isinstance(replay, EvidenceRecord):
                    raise WorldError("evidence idempotency outcome has wrong type")
                return replay
            table = cast(EvidenceRecordTable, self._tables["evidence_records"])
            key = evidence_record_key(record.record_id, record.version)
            exact = table.rows.get(key)
            if exact is not None:
                if exact == record and record.issuer_id == original_actor:
                    operation = self._bind_authority_operation(
                        scope="evidence",
                        actor_id=original_actor,
                        idempotency_key=idempotency_key,
                        request_fingerprint=record.record_digest,
                        outcome_table="evidence_records",
                        outcome_key=key,
                    )
                    self._append_transaction_commit(
                        _make_diff(
                            "bind_evidence_idempotency",
                            record.record_id,
                            [
                                (
                                    "authority_operations",
                                    operation.operation_key,
                                    "create",
                                    None,
                                    operation,
                                )
                            ],
                            ("actor-scoped-idempotency", "zero-evidence-write"),
                        ),
                        authority_action="world.persist_evidence_record",
                        actor_id=original_actor,
                        idempotency_key=idempotency_key,
                        request_fingerprint=record.record_digest,
                    )
                    return exact
                raise IdempotencyConflict(
                    "evidence version retry differs from persisted content"
                )
            disposition = validate_evidence_append(
                table.current(record.record_id),
                record,
                original_actor=original_actor,
                logical_time=self._logical_time,
            )
            if disposition == "idempotent":
                return record
            table.write(key, record)
            operation = self._bind_authority_operation(
                scope="evidence",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=record.record_digest,
                outcome_table="evidence_records",
                outcome_key=key,
            )
            self._append_transaction_commit(
                _make_diff(
                    "persist_evidence_record",
                    record.record_id,
                    [
                        ("evidence_records", key, "create", None, record),
                        (
                            "authority_operations",
                            operation.operation_key,
                            "create",
                            None,
                            operation,
                        ),
                    ],
                    (
                        "issuer-authenticated",
                        "owner-acl-bound",
                        "version-contiguous",
                        "digest-verified",
                        "append-only",
                    ),
                ),
                authority_action="world.persist_evidence_record",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=record.record_digest,
            )
            return record

    def read_evidence_record(
        self,
        record_id: str,
        *,
        caller: str,
        version: int | None = None,
        record_digest: str | None = None,
    ) -> EvidenceRecord | None:
        """Read the current or exact historical version through persisted ACL."""

        if version is not None and record_digest is not None:
            raise ValueError("choose evidence version or digest, not both")
        with self._lock:
            table = cast(EvidenceRecordTable, self._tables["evidence_records"])
            if version is not None:
                row = table.rows.get(evidence_record_key(record_id, version))
            elif record_digest is not None:
                row = table.by_digest(record_id, record_digest)
            else:
                row = table.current(record_id)
            if row is None:
                return None
            return authorize_persisted_evidence_read(row, reader_id=caller)

    def register_mandate_authority(
        self,
        authority: MandateRevisionAuthority,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> MandateRevisionAuthority:
        """Register one immutable principal-to-buyer mandate authority."""

        if by_actor != "platform:mandate":
            raise WriteNotAuthorized(
                "only platform:mandate may register mandate authority"
            )
        if not idempotency_key:
            raise IdempotencyConflict("mandate idempotency key must not be blank")
        with self._lock:
            fingerprint = canonical_digest(mandate_authority_to_wire(authority))
            replay = self._authority_operation_replay(
                scope="mandate-authority",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if replay is not None:
                if not isinstance(replay, MandateRevisionAuthority):
                    raise WorldError(
                        "mandate authority idempotency outcome has wrong type"
                    )
                return replay
            table = cast(
                MandateAuthorityTable, self._tables["mandate_authorities"]
            )
            current = table.rows.get(authority.mandate_id)
            disposition = validate_mandate_authority_registration(
                current, authority, original_actor=original_actor
            )
            if disposition == "idempotent":
                outcome = cast(MandateRevisionAuthority, current)
            else:
                table.write(authority.mandate_id, authority)
                outcome = authority
            operation = self._bind_authority_operation(
                scope="mandate-authority",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                outcome_table="mandate_authorities",
                outcome_key=authority.mandate_id,
            )
            writes: list[tuple[str, str, str, Any, Any]] = []
            if disposition == "append":
                writes.append((
                    "mandate_authorities",
                    authority.mandate_id,
                    "create",
                    None,
                    authority,
                ))
            writes.append((
                "authority_operations",
                operation.operation_key,
                "create",
                None,
                operation,
            ))
            self._append_transaction_commit(
                _make_diff(
                    "register_mandate_authority",
                    authority.mandate_id,
                    writes,
                    (
                        "principal-authenticated",
                        "buyer-bound",
                        "immutable-authority",
                    ),
                ),
                authority_action="world.register_mandate_authority",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
            )
            return outcome

    def append_mandate_revision(
        self,
        revision: MandateRevision,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> MandateRevision:
        """Append one principal-authenticated, contiguous mandate revision."""

        if by_actor != "platform:mandate":
            raise WriteNotAuthorized(
                "only platform:mandate may append mandate revisions"
            )
        if not idempotency_key:
            raise IdempotencyConflict("mandate idempotency key must not be blank")
        with self._lock:
            replay = self._authority_operation_replay(
                scope="mandate-revision",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=revision.revision_digest,
            )
            if replay is not None:
                if not isinstance(replay, MandateRevision):
                    raise WorldError(
                        "mandate revision idempotency outcome has wrong type"
                    )
                return replay
            authority = cast(
                "MandateRevisionAuthority | None",
                self._tables["mandate_authorities"].read(
                    revision.mandate_id, caller=None
                ),
            )
            if authority is None:
                raise WriteNotAuthorized(
                    f"mandate {revision.mandate_id!r} has no registered authority"
                )
            table = cast(
                MandateRevisionTable, self._tables["mandate_revisions"]
            )
            key = mandate_revision_key(revision.mandate_id, revision.revision)
            exact = table.rows.get(key)
            if exact is not None:
                if exact == revision and revision.principal_id == original_actor:
                    operation = self._bind_authority_operation(
                        scope="mandate-revision",
                        actor_id=original_actor,
                        idempotency_key=idempotency_key,
                        request_fingerprint=revision.revision_digest,
                        outcome_table="mandate_revisions",
                        outcome_key=key,
                    )
                    self._append_transaction_commit(
                        _make_diff(
                            "bind_mandate_revision_idempotency",
                            revision.mandate_id,
                            [
                                (
                                    "authority_operations",
                                    operation.operation_key,
                                    "create",
                                    None,
                                    operation,
                                )
                            ],
                            ("actor-scoped-idempotency", "zero-revision-write"),
                        ),
                        authority_action="world.append_mandate_revision",
                        actor_id=original_actor,
                        idempotency_key=idempotency_key,
                    )
                    return exact
                raise IdempotencyConflict(
                    "mandate revision retry differs from persisted content"
                )
            disposition = validate_mandate_append(
                table.current(revision.mandate_id),
                revision,
                authority,
                original_actor=original_actor,
                logical_time=self._logical_time,
            )
            if disposition == "idempotent":
                return revision
            table.write(key, revision)
            operation = self._bind_authority_operation(
                scope="mandate-revision",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=revision.revision_digest,
                outcome_table="mandate_revisions",
                outcome_key=key,
            )
            self._append_transaction_commit(
                _make_diff(
                    "append_mandate_revision",
                    revision.mandate_id,
                    [
                        ("mandate_revisions", key, "create", None, revision),
                        (
                            "authority_operations",
                            operation.operation_key,
                            "create",
                            None,
                            operation,
                        ),
                    ],
                    (
                        "principal-authenticated",
                        "persisted-authority-bound",
                        "digest-chain-contiguous",
                        "actor-scoped",
                    ),
                ),
                authority_action="world.append_mandate_revision",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
            )
            return revision

    def mandate_revisions(
        self, mandate_id: str, *, caller: str
    ) -> tuple[MandateRevision, ...]:
        """Return one mandate history only to its principal, buyer, or services."""

        with self._lock:
            authority = cast(
                "MandateRevisionAuthority | None",
                self._tables["mandate_authorities"].read(
                    mandate_id, caller=None
                ),
            )
            if authority is None:
                return ()
            authorize_mandate_read(authority, reader_id=caller)
            table = cast(
                MandateRevisionTable, self._tables["mandate_revisions"]
            )
            return table.by_mandate(mandate_id)

    def apply_listing_claim(
        self,
        claim: ListingClaim,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> ListingClaim:
        """Apply one merchant-owned claim transition with exact evidence joins."""

        if by_actor != "platform:claims":
            raise WriteNotAuthorized(
                "only platform:claims may apply listing claims"
            )
        if not idempotency_key:
            raise IdempotencyConflict("claim idempotency key must not be blank")
        with self._lock:
            table = cast(ListingClaimTable, self._tables["listing_claims"])
            prior_for_key = table.by_actor_key(original_actor, idempotency_key)
            if prior_for_key is not None:
                prior_version = next(
                    version
                    for version in prior_for_key.versions
                    if version.idempotency_key == idempotency_key
                )
                if (
                    claim.claim_id == prior_for_key.claim_id
                    and claim.listing_id == prior_for_key.listing_id
                    and claim.merchant_id == prior_for_key.merchant_id
                    and claim.subject == prior_for_key.subject
                    and claim.issuer_id == prior_for_key.issuer_id
                    and claim.current.idempotency_key == idempotency_key
                    and claim.current.request_digest == prior_version.request_digest
                ):
                    return prior_for_key
                raise IdempotencyConflict(
                    "merchant claim idempotency key was reused for a different claim"
                )
            current = table.rows.get(claim.claim_id)
            listing = cast(
                "Listing | None",
                self._tables["catalog"].read(SkuId(claim.listing_id), caller=None),
            )
            evidence_table = cast(
                EvidenceRecordTable, self._tables["evidence_records"]
            )
            disposition = validate_listing_claim_append(
                current,
                claim,
                listing=listing,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                logical_time=self._logical_time,
                evidence_lookup=evidence_table.by_digest,
            )
            if disposition == "idempotent":
                return claim
            table.write(claim.claim_id, claim)
            self._append_transaction_commit(
                _make_diff(
                    "apply_listing_claim",
                    claim.claim_id,
                    [
                        (
                            "listing_claims",
                            claim.claim_id,
                            "create" if current is None else "update",
                            current,
                            claim,
                        )
                    ],
                    (
                        "merchant-owner-authenticated",
                        "listing-identity-bound",
                        "evidence-digest-subject-bound",
                        "append-only-version-history",
                        "idempotent",
                    ),
                ),
                authority_action="world.apply_listing_claim",
                actor_id=by_actor,
                idempotency_key=idempotency_key,
            )
            return claim

    def read_listing_claim(
        self, claim_id: str, *, caller: str
    ) -> ListingClaim | None:
        """Return a draft only to its owner; public histories expose state."""

        with self._lock:
            claim = cast(
                "ListingClaim | None",
                self._tables["listing_claims"].read(claim_id, caller=None),
            )
            if claim is None:
                return None
            trusted = caller.startswith(("platform:", "runtime:")) or caller == "runtime"
            if claim.state == "draft" and caller != claim.merchant_id and not trusted:
                return None
            return claim

    def listing_claims_for_listing(
        self, listing_id: str, *, caller: str
    ) -> tuple[ListingClaim, ...]:
        """Return deterministically ordered claims visible to one actor."""

        with self._lock:
            table = cast(ListingClaimTable, self._tables["listing_claims"])
            return tuple(
                claim
                for claim in table.by_listing(listing_id)
                if self.read_listing_claim(claim.claim_id, caller=caller) is not None
            )

    def _authority_operation_replay(
        self,
        *,
        scope: str,
        actor_id: str,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> Any | None:
        key = authority_operation_key(scope, actor_id, idempotency_key)
        operation = cast(
            "AuthorityOperationRecord | None",
            self._tables["authority_operations"].read(key, caller="runtime"),
        )
        if operation is None:
            return None
        if operation.request_fingerprint != request_fingerprint:
            raise IdempotencyConflict(
                f"idempotency key {idempotency_key!r} was reused for another request"
            )
        if operation.outcome_listing is not None:
            return deepcopy(operation.outcome_listing)
        outcome = self._tables[operation.outcome_table].read(
            operation.outcome_key, caller=None
        )
        if outcome is None:
            raise WorldError("authority idempotency outcome is missing")
        return outcome

    def _bind_authority_operation(
        self,
        *,
        scope: str,
        actor_id: str,
        idempotency_key: str,
        request_fingerprint: str,
        outcome_table: str,
        outcome_key: str,
    ) -> AuthorityOperationRecord:
        key = authority_operation_key(scope, actor_id, idempotency_key)
        operation = AuthorityOperationRecord(
            operation_key=key,
            scope=scope,
            actor_id=actor_id,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            outcome_table=outcome_table,
            outcome_key=outcome_key,
        )
        cast(
            AuthorityOperationTable, self._tables["authority_operations"]
        ).write(key, operation)
        return operation

    def _order_revision_for_existing(self, order_id: str) -> int:
        revision = self._order_state_revisions.get(order_id)
        if revision is None:
            # Legacy/seeded order rows predate the additive revision ledger.
            # Existing authoritative rows start at revision one.
            revision = 1
            self._order_state_revisions[order_id] = revision
        return revision

    def read_supply_state(self, sku_id: SkuId, *, caller: str) -> SupplyState:
        """Return one authoritative supply projection without duplicating state."""

        with self._lock:
            inventory = cast(
                "InventoryRow | None",
                self._tables["inventory"].read(sku_id, caller=None),
            )
            listing = cast(
                "Listing | None",
                self._tables["catalog"].read(sku_id, caller=None),
            )
            if inventory is None or listing is None:
                raise OutOfStock(f"unknown supply sku {sku_id}")
            role = caller.split(":", 1)[0]
            if role not in {"platform", "runtime"} and caller != str(inventory.merchant_id):
                raise LifecycleAuthorizationError(
                    f"actor {caller!r} cannot read exact supply for {sku_id}"
                )
            if inventory.merchant_id != listing.merchant_id:
                raise WorldError(f"catalog/inventory owner mismatch for {sku_id}")
            return _supply_projection(listing, inventory)

    def issue_supply_purchase_authorities(
        self,
        sku_ids: tuple[str, ...],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        ttl_ticks: int = DEFAULT_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS,
    ) -> tuple[SupplyPurchaseAuthority, ...]:
        """Mint one atomic set of buyer-scoped authorities from current supply.

        Exact retries return the originally sealed rows even after the market
        clock or inventory advances.  A new settlement must still pass PSP's
        current version and expiry checks before it can mutate commerce state.
        """

        if by_actor != "platform:supply":
            raise WriteNotAuthorized(
                "only platform:supply may issue supply purchase authorities"
            )
        if not isinstance(original_actor, str) or not original_actor.startswith(
            "buyer:"
        ):
            raise LifecycleAuthorizationError(
                "supply purchase authority requires an authenticated buyer"
            )
        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key.strip()
        ):
            raise IdempotencyConflict(
                "supply purchase authority requires an idempotency key"
            )
        try:
            validate_supply_purchase_authority_ttl_ticks(ttl_ticks)
        except SchemaError as exc:
            raise WorldError(str(exc)) from exc
        if (
            not isinstance(sku_ids, tuple)
            or not sku_ids
            or len(sku_ids) > 64
            or any(not isinstance(value, str) or not value for value in sku_ids)
            or len(set(sku_ids)) != len(sku_ids)
        ):
            raise WorldError(
                "supply purchase authority requires 1..64 unique sku ids"
            )

        authority_ids = tuple(
            supply_purchase_authority_id(
                buyer_id=original_actor,
                request_idempotency_key=idempotency_key,
                sku_id=sku_id,
            )
            for sku_id in sku_ids
        )
        with self._lock:
            table = cast(
                SupplyPurchaseAuthorityTable,
                self._tables["supply_purchase_authorities"],
            )
            existing = tuple(
                table.read(authority_id, caller="runtime")
                for authority_id in authority_ids
            )
            if any(row is not None for row in existing):
                if not all(row is not None for row in existing):
                    raise IdempotencyConflict(
                        "supply authority request has an incomplete durable replay"
                    )
                replay = cast(tuple[SupplyPurchaseAuthority, ...], existing)
                for row, sku_id in zip(replay, sku_ids, strict=True):
                    validate_supply_purchase_authority(row)
                    if (
                        row.buyer_id != original_actor
                        or row.sku_id != sku_id
                        or row.request_idempotency_key != idempotency_key
                        or row.expires_at_tick - row.issued_at_tick != ttl_ticks
                    ):
                        raise IdempotencyConflict(
                            "supply authority idempotency key was reused"
                        )
                return tuple(deepcopy(row) for row in replay)

            issued_at_tick = self._logical_time
            authorities: list[SupplyPurchaseAuthority] = []
            for sku_id in sku_ids:
                listing = cast(
                    "Listing | None",
                    self._tables["catalog"].read(SkuId(sku_id), caller=None),
                )
                inventory = cast(
                    "InventoryRow | None",
                    self._tables["inventory"].read(SkuId(sku_id), caller=None),
                )
                if listing is None or inventory is None:
                    raise OutOfStock(f"unknown supply sku {sku_id}")
                if listing.merchant_id != inventory.merchant_id:
                    raise WorldError(
                        f"catalog/inventory owner mismatch for {sku_id}"
                    )
                state = _supply_projection(listing, inventory)
                authorities.append(build_supply_purchase_authority(
                    buyer_id=original_actor,
                    merchant_id=str(listing.merchant_id),
                    sku_id=sku_id,
                    unit_price_cents=state.unit_price_cents,
                    currency=listing.list_price.currency,
                    available_qty=state.available_qty,
                    supply_version=state.version,
                    issued_at_tick=issued_at_tick,
                    expires_at_tick=issued_at_tick + ttl_ticks,
                    request_idempotency_key=idempotency_key,
                ))

            applied: list[str] = []
            commit_cursor = len(self._commit_journal)
            try:
                for authority in authorities:
                    table.write(authority.authority_id, authority)
                    applied.append(authority.authority_id)
                writes: list[tuple[str, str, str, Any, Any]] = [
                    (
                        "supply_purchase_authorities",
                        authority.authority_id,
                        "create",
                        None,
                        authority,
                    )
                    for authority in authorities
                ]
                self._append_transaction_commit(
                    _make_diff(
                        "issue_supply_purchase_authority",
                        authorities[0].authority_id,
                        writes,
                        (
                            "world-derived-listing-and-supply",
                            "buyer-scoped",
                            "sealed",
                            "bounded-expiry",
                            "atomic-batch",
                            "idempotent",
                            "market-clock-neutral",
                        ),
                    ),
                    authority_action="world.issue_supply_purchase_authority",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=canonical_digest({
                        "sku_ids": list(sku_ids),
                        "ttl_ticks": ttl_ticks,
                    }),
                )
            except BaseException:
                for authority_id in reversed(applied):
                    table.delete(authority_id)
                del self._commit_journal[commit_cursor:]
                raise
            return tuple(deepcopy(row) for row in authorities)

    def read_supply_purchase_authority(
        self,
        authority_id: str,
        *,
        caller: str,
    ) -> SupplyPurchaseAuthority | None:
        """Read one persisted authority under its buyer/control-plane ACL."""

        with self._lock:
            row = self._tables["supply_purchase_authorities"].read(
                authority_id,
                caller=caller,
            )
            return None if row is None else deepcopy(cast(SupplyPurchaseAuthority, row))

    def apply_settlement_reputation(
        self,
        *,
        merchant_id: AgentId,
        order_id: OrderId,
        txn_id: TxnId,
        by_actor: str,
        original_actor: str,
        source_request_id: str,
        idempotency_key: str,
    ) -> ReputationScore:
        """Apply one five-star sample for one authoritative settlement.

        Transaction identity is the primary idempotency boundary. The source
        actor and idempotency key form a second boundary that detects reuse
        across transactions. A new request key for an already applied
        transaction appends a replayable source alias but never changes the
        merchant score again.
        """

        if by_actor != "platform:reputation":
            raise WriteNotAuthorized(
                "only platform:reputation may apply settlement reputation"
            )
        if original_actor != "platform:psp":
            raise WriteNotAuthorized(
                "settlement reputation must originate from platform:psp"
            )
        if not source_request_id or not idempotency_key:
            raise IdempotencyConflict(
                "settlement reputation requires source request and idempotency keys"
            )

        source = ReputationSettlementSource(
            source_actor=original_actor,
            source_request_id=source_request_id,
            source_idempotency_key=idempotency_key,
        )
        with self._lock:
            order = cast(
                "Order | None", self._tables["orders"].read(order_id, caller=None)
            )
            receipt = cast(
                "Receipt | None", self._tables["ledger"].read(txn_id, caller=None)
            )
            _validate_reputation_settlement_identity(
                merchant_id=merchant_id,
                order_id=order_id,
                txn_id=txn_id,
                order=order,
                receipt=receipt,
            )
            event_table = cast(
                ReputationSettlementTable,
                self._tables["reputation_settlements"],
            )
            source_event = event_table.by_source(original_actor, idempotency_key)
            if source_event is not None:
                if (
                    source_event.order_id != order_id
                    or source_event.txn_id != txn_id
                    or source_event.merchant_id != merchant_id
                ):
                    raise IdempotencyConflict(
                        "reputation source idempotency key was reused for a different settlement"
                    )
                persisted_source = next(
                    item
                    for item in source_event.sources
                    if item.source_actor == original_actor
                    and item.source_idempotency_key == idempotency_key
                )
                if persisted_source.source_request_id != source_request_id:
                    raise IdempotencyConflict(
                        "reputation source request changed under an existing key"
                    )
                return deepcopy(source_event.outcome)

            existing = event_table.by_txn(str(txn_id))
            if existing is not None:
                if (
                    existing.order_id != order_id
                    or existing.merchant_id != merchant_id
                ):
                    raise IdempotencyConflict(
                        "reputation transaction identity conflicts with persisted event"
                    )
                updated = replace(
                    existing,
                    sources=tuple(
                        sorted(
                            (*existing.sources, source),
                            key=lambda item: (
                                item.source_actor,
                                item.source_idempotency_key,
                                item.source_request_id,
                            ),
                        )
                    ),
                )
                event_table.write(existing.event_id, updated)
                self._append_transaction_commit(
                    _make_diff(
                        "append_reputation_settlement_source",
                        existing.event_id,
                        [
                            (
                                "reputation_settlements",
                                existing.event_id,
                                "update",
                                existing,
                                updated,
                            )
                        ],
                        (
                            "authoritative-settlement",
                            "source-alias-only",
                            "reputation-unchanged",
                            "idempotent",
                        ),
                    ),
                    authority_action="world.update_reputation",
                    actor_id=by_actor,
                    idempotency_key=idempotency_key,
                )
                return deepcopy(existing.outcome)

            prior = cast(
                "ReputationScore | None",
                self._tables["reputation"].read(merchant_id, caller=None),
            )
            if prior is None:
                raise WorldError(
                    f"settlement merchant {merchant_id} has no tracked reputation row"
                )
            settled = int(prior.n_settled) + 1
            outcome = ReputationScore(
                merchant_id=prior.merchant_id,
                rolling_avg=(
                    float(prior.rolling_avg) * int(prior.n_settled) + 5.0
                )
                / settled,
                n_settled=settled,
                n_disputed=int(prior.n_disputed),
            )
            event = ReputationSettlement(
                event_id=_reputation_settlement_event_id(txn_id),
                order_id=order_id,
                txn_id=txn_id,
                merchant_id=merchant_id,
                sources=(source,),
                outcome=outcome,
            )
            reputation_table = self._tables["reputation"]
            try:
                reputation_table.write(merchant_id, outcome)
                event_table.write(event.event_id, event)
            except BaseException:
                _restore_row(reputation_table, merchant_id, prior)
                event_table.delete(event.event_id)
                raise
            self._append_transaction_commit(
                _make_diff(
                    "apply_settlement_reputation",
                    event.event_id,
                    [
                        (
                            "reputation",
                            str(merchant_id),
                            "update",
                            prior,
                            outcome,
                        ),
                        (
                            "reputation_settlements",
                            event.event_id,
                            "create",
                            None,
                            event,
                        ),
                    ],
                    (
                        "authoritative-settlement",
                        "one-sample-per-transaction",
                        "source-bound-idempotency",
                        "atomic",
                    ),
                ),
                authority_action="world.update_reputation",
                actor_id=by_actor,
                idempotency_key=idempotency_key,
            )
            return deepcopy(outcome)

    def create_search_session(
        self,
        *,
        session: SearchSession,
        by_actor: str,
        idempotency_key: str,
    ) -> SearchSession:
        """Validate and persist one immutable server-authored search result.

        The World rechecks every offer against current catalog and inventory
        state before recording it.  Exact retries return the persisted object;
        a changed request under the same buyer-scoped key has no state effect.
        """

        if by_actor != "platform:aggregator":
            raise WriteNotAuthorized(
                "only platform:aggregator may create search sessions"
            )
        validate_search_session(session)
        if session.search_idempotency_key != idempotency_key:
            raise IdempotencyConflict(
                "search envelope and session idempotency keys differ"
            )
        with self._lock:
            table = cast(SearchSessionTable, self._tables["search_sessions"])
            existing = table.by_request_key(
                session.buyer_id, session.search_idempotency_key
            )
            if existing is not None:
                if existing != session:
                    raise IdempotencyConflict(
                        "search idempotency key was reused for a different request"
                    )
                return deepcopy(existing)
            self._validate_search_session_against_world(session)
            table.write(session.session_id, deepcopy(session))
            self._append_transaction_commit(
                _make_diff(
                    "create_search_session",
                    session.session_id,
                    [("search_sessions", session.session_id, "create", None, session)],
                    (
                        "server-authored-session",
                        "catalog-revision-bound",
                        "inventory-revision-bound",
                        "buyer-scoped-idempotency",
                    ),
                ),
                authority_action="world.create_search_session",
                actor_id=by_actor,
                idempotency_key=idempotency_key,
            )
            return deepcopy(session)

    def resolve_search_session(
        self,
        *,
        buyer_id: str,
        offer_id: str,
        caller: str,
        unique_only: bool = True,
        current_only: bool = True,
    ) -> SearchSession | None:
        """Resolve a persisted buyer offer for a trusted platform caller.

        Legacy acceptance uses the default exact-current-and-unique contract.
        Other trusted services may disable uniqueness or liveness only to test
        whether matching state exists; they still cannot expose it to agents.
        """

        if not (caller.startswith("platform:") or caller.startswith("runtime:")):
            return None
        with self._lock:
            table = cast(SearchSessionTable, self._tables["search_sessions"])
            matches = sorted(
                (
                session
                for session in table.by_offer(buyer_id, offer_id)
                if not current_only or self._logical_time < session.expires_at_tick
                ),
                key=lambda session: session.session_id,
            )
            if not matches or (unique_only and len(matches) != 1):
                return None
            return deepcopy(matches[0])

    def issue_match_certificate(
        self,
        *,
        acceptance: MatchAcceptance,
        by_actor: str,
        original_actor: str,
    ) -> MatchCertificate:
        """Atomically persist an acceptance and its exact match certificate."""

        if by_actor != "platform:aggregator":
            raise WriteNotAuthorized(
                "only platform:aggregator may issue match certificates"
            )
        if original_actor != acceptance.buyer_id:
            raise MatchAcceptanceRejected(
                "acceptance buyer does not match original actor"
            )
        with self._lock:
            session = cast(
                "SearchSession | None",
                self._tables["search_sessions"].read(
                    acceptance.session_id, caller=by_actor
                ),
            )
            if session is None:
                raise MatchAcceptanceRejected(
                    "unknown persisted search session"
                )
            acceptance_key = _match_acceptance_key(
                acceptance.buyer_id, acceptance.idempotency_key
            )
            acceptance_table = cast(
                MatchAcceptanceTable, self._tables["match_acceptances"]
            )
            certificate_table = cast(
                MatchCertificateTable, self._tables["match_certificates"]
            )
            existing_acceptance = acceptance_table.read(
                acceptance_key, caller=by_actor
            )
            existing_certificate = (
                None
                if existing_acceptance is None
                else _certificate_for_acceptance(
                    certificate_table, existing_acceptance.acceptance_digest
                )
            )
            if existing_acceptance is not None and existing_acceptance != acceptance:
                raise IdempotencyConflict(
                    "match acceptance changed under an existing idempotency key"
                )
            listing, inventory = self._matching_rows(acceptance.sku_id)
            try:
                certificate = build_match_certificate(
                    session,
                    acceptance,
                    current_tick=self._logical_time,
                    current_catalog_revision=_catalog_revision(listing),
                    current_inventory_revision=inventory.version,
                    existing_certificate=existing_certificate,
                )
            except MatchValidationError as exc:
                raise MatchAcceptanceRejected(str(exc)) from exc
            if existing_certificate is not None:
                return deepcopy(existing_certificate)

            acceptance_table.write(acceptance_key, deepcopy(acceptance))
            try:
                certificate_table.write(
                    certificate.cert_id, deepcopy(certificate)
                )
            except BaseException:
                if existing_acceptance is None:
                    acceptance_table.delete(acceptance_key)
                raise
            self._append_transaction_commit(
                _make_diff(
                    "issue_match_certificate",
                    certificate.cert_id,
                    [
                        (
                            "match_acceptances",
                            acceptance_key,
                            "create",
                            None,
                            acceptance,
                        ),
                        (
                            "match_certificates",
                            certificate.cert_id,
                            "create",
                            None,
                            certificate,
                        ),
                    ],
                    (
                        "persisted-session-membership",
                        "exact-offer-binding",
                        "current-revisions",
                        "order-binding",
                        "atomic-certificate-write",
                    ),
                ),
                authority_action="world.issue_match_certificate",
                actor_id=by_actor,
                idempotency_key=acceptance.idempotency_key,
            )
            return deepcopy(certificate)

    def resolve_match_certificate(
        self,
        *,
        buyer_id: str,
        order_id: str,
        caller: str,
        current_only: bool = True,
    ) -> MatchCertificate | None:
        """Return a certificate only when buyer and order identify one row."""

        if not (caller.startswith("platform:") or caller.startswith("runtime:")):
            return None
        with self._lock:
            table = cast(
                MatchCertificateTable, self._tables["match_certificates"]
            )
            matches = sorted(
                (
                    certificate
                    for certificate in table.by_order(buyer_id, order_id)
                    if (
                        not current_only
                        or self._logical_time < certificate.expires_at_tick
                    )
                ),
                key=lambda certificate: certificate.cert_id,
            )
            return deepcopy(matches[0]) if len(matches) == 1 else None

    def _matching_rows(self, sku_id: str) -> tuple[Listing, InventoryRow]:
        listing = cast(
            "Listing | None",
            self._tables["catalog"].read(SkuId(sku_id), caller=None),
        )
        inventory = cast(
            "InventoryRow | None",
            self._tables["inventory"].read(SkuId(sku_id), caller=None),
        )
        if listing is None or inventory is None:
            raise MatchValidationError("matched SKU lacks catalog or inventory state")
        if str(listing.merchant_id) != str(inventory.merchant_id):
            raise MatchValidationError("catalog and inventory merchant mismatch")
        return listing, inventory

    def _validate_search_session_against_world(self, session: SearchSession) -> None:
        if session.issued_at_tick != self._logical_time:
            raise MatchValidationError("search session issued_at_tick is not current")
        if session.expires_at_tick <= self._logical_time:
            raise MatchValidationError("search session is already expired")
        for offer in session.offers:
            listing, inventory = self._matching_rows(offer.sku_id)
            if offer.session_id != session.session_id:
                raise MatchValidationError("offer session_id mismatch")
            if offer.issued_at_tick != session.issued_at_tick:
                raise MatchValidationError("offer issued_at_tick mismatch")
            if offer.expires_at_tick > session.expires_at_tick:
                raise MatchValidationError("offer expiry exceeds session expiry")
            if offer.merchant_id != str(listing.merchant_id):
                raise MatchValidationError("offer merchant does not own listing")
            if offer.unit_price_cents != _money_cents(listing.list_price):
                raise MatchValidationError("offer price differs from catalog")
            if offer.currency != listing.list_price.currency:
                raise MatchValidationError("offer currency differs from catalog")
            if offer.catalog_revision != _catalog_revision(listing):
                raise MatchValidationError("offer catalog revision is stale")
            if offer.inventory_revision != inventory.version:
                raise MatchValidationError("offer inventory revision is stale")
            if offer.qty > _available_qty(inventory):
                raise MatchValidationError("offer quantity exceeds available inventory")

    def apply_supply_event(
        self,
        *,
        sku_id: SkuId,
        qty_delta: int,
        eta_day: int | None,
        unit_price_cents: int | None,
        expected_version: int | None,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> SupplyState:
        """Atomically apply a trusted or owner-authored supply revision.

        Inventory quantity/ETA/version and an optional catalog price change are
        committed as one World transaction.  Version advances exactly once per
        accepted event, including an event that changes both rows.
        """

        if by_actor != "platform:supply":
            raise LifecycleAuthorizationError(
                f"only platform:supply may apply supply events; got {by_actor!r}"
            )
        if isinstance(qty_delta, bool) or not isinstance(qty_delta, int):
            raise ValueError("qty_delta must be an integer")
        if eta_day is not None and (
            isinstance(eta_day, bool) or not isinstance(eta_day, int) or eta_day < 0
        ):
            raise ValueError("eta_day must be a non-negative integer")
        if unit_price_cents is not None and (
            isinstance(unit_price_cents, bool)
            or not isinstance(unit_price_cents, int)
            or unit_price_cents <= 0
        ):
            raise ValueError("unit_price_cents must be a positive integer")
        if expected_version is not None and (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or expected_version <= 0
        ):
            raise ValueError("expected_version must be a positive integer")
        if not idempotency_key.strip():
            raise ValueError("supply event idempotency key must not be blank")

        cache_key = (by_actor, idempotency_key)
        requested = (
            sku_id,
            qty_delta,
            eta_day,
            unit_price_cents,
            expected_version,
            original_actor,
        )
        with self._lock:
            cached = self._supply_idempotency_cache.get(cache_key)
            if cached is not None:
                prior = (
                    cached.sku_id,
                    cached.qty_delta,
                    cached.eta_day,
                    cached.unit_price_cents,
                    cached.expected_version,
                    cached.original_actor,
                )
                if prior != requested:
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was reused for another supply event"
                    )
                return cached.outcome

            inventory_table = cast("InventoryTable", self._tables["inventory"])
            durable = inventory_table.supply_operation(by_actor, idempotency_key)
            if durable is not None:
                prior = (
                    durable.sku_id,
                    durable.qty_delta,
                    durable.eta_day,
                    durable.unit_price_cents,
                    durable.expected_version,
                    durable.original_actor,
                )
                if prior != requested:
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was reused for another supply event"
                    )
                return cast("SupplyState", durable.outcome)

            inventory = cast(
                "InventoryRow | None",
                self._tables["inventory"].read(sku_id, caller=None),
            )
            listing = cast(
                "Listing | None",
                self._tables["catalog"].read(sku_id, caller=None),
            )
            if inventory is None or listing is None:
                raise OutOfStock(f"unknown supply sku {sku_id}")
            if inventory.merchant_id != listing.merchant_id:
                raise WorldError(f"catalog/inventory owner mismatch for {sku_id}")
            if original_actor not in {str(inventory.merchant_id), "runtime:supply"}:
                raise LifecycleAuthorizationError(
                    f"actor {original_actor!r} does not own supply {sku_id}"
                )
            if expected_version is not None and inventory.version != expected_version:
                raise IdempotencyConflict(
                    f"stale supply version {expected_version}; current is {inventory.version}"
                )

            next_qty = inventory.qty_available + qty_delta
            if next_qty < inventory.qty_reserved or next_qty < 0:
                raise OutOfStock(
                    f"supply event would reduce {sku_id} below reserved inventory"
                )
            projected_inventory = replace(
                inventory,
                qty_available=next_qty,
                eta_day=inventory.eta_day if eta_day is None else eta_day,
                version=inventory.version + 1,
            )
            listing_after = listing
            if unit_price_cents is not None:
                listing_after = _revisioned_listing(
                    replace(
                        listing,
                        list_price=Money(
                            amount=(
                                listing.list_price.amount.__class__(unit_price_cents)
                                / 100
                            ),
                            currency=listing.list_price.currency,
                        ),
                    ),
                    listing,
                )

            outcome = _supply_projection(listing_after, projected_inventory)
            durable_record = SupplyEventRecord(
                scope=by_actor,
                idempotency_key=idempotency_key,
                sku_id=sku_id,
                qty_delta=qty_delta,
                eta_day=eta_day,
                unit_price_cents=unit_price_cents,
                expected_version=expected_version,
                original_actor=original_actor,
                outcome=outcome,
            )
            inventory_after = replace(
                projected_inventory,
                supply_events=inventory.supply_events + (durable_record,),
            )

            writes: list[tuple[str, str, str, Any, Any]] = [
                ("inventory", str(sku_id), "update", inventory, inventory_after)
            ]
            if listing_after != listing:
                writes.append(("catalog", str(sku_id), "update", listing, listing_after))
            try:
                self.write(
                    "inventory",
                    sku_id,
                    inventory_after,
                    by_action="world.update_inventory",
                    _record_commit=False,
                )
                if listing_after != listing:
                    self.write(
                        "catalog",
                        sku_id,
                        listing_after,
                        by_action="world.update_catalog",
                        _record_commit=False,
                    )
            except BaseException:
                _restore_row(self._tables["inventory"], sku_id, inventory)
                _restore_row(self._tables["catalog"], sku_id, listing)
                raise

            self._supply_idempotency_cache[cache_key] = _SupplyIdempotencyRecord(
                sku_id,
                qty_delta,
                eta_day,
                unit_price_cents,
                expected_version,
                original_actor,
                outcome,
            )
            self._append_transaction_commit(
                _make_diff(
                    "apply_supply_event",
                    sku_id,
                    writes,
                    (
                        "atomic",
                        "owner-or-runtime-supply",
                        "inventory-conservation",
                        "supply-version-monotonic",
                        "idempotent",
                    ),
                ),
                authority_action="world.apply_supply_event",
                actor_id=by_actor,
                idempotency_key=idempotency_key,
            )
            return outcome

    def write(
        self,
        table: str,
        key: Any,
        value: Any,
        *,
        by_action: str,
        _record_commit: bool = True,
    ) -> None:
        """Mutate ``(table, key)``. ``by_action`` must be authorized on this table.

        Raises:
            WriteNotAuthorized: ``by_action`` not in the table's allowed actions.
            TableNotFound: ``table`` is not registered.
        """
        target = self._table(table)
        action = by_action.removeprefix("world.")
        if by_action not in target.allowed_actions and action not in target.allowed_actions:
            raise WriteNotAuthorized(
                f"{by_action!r} is not authorized to write table {table!r}"
            )
        protected_tables = {
            "protocol_events",
            "protocol_receipts",
            "evidence_records",
            "mandate_authorities",
            "mandate_revisions",
            "listing_claims",
            "authority_operations",
            "governance_policies",
            "governance_records",
            "pricing_policy_revisions",
            "persistent_cart_quote_requests",
            "persistent_cart_quotes",
            "supply_purchase_authorities",
            "order_groups",
        }
        if _record_commit and table in protected_tables:
            raise WriteNotAuthorized(
                f"table {table!r} requires its authority-validating World method"
            )
        with self._lock:
            before = deepcopy(target.rows.get(key))
            stored_value = (
                _revisioned_listing(value, before)
                if table == "catalog" and isinstance(value, Listing)
                else value
            )
            target.write(key, stored_value)
            if _record_commit:
                write = TableWrite(
                    table=table,
                    key=str(key),
                    op="create" if before is None else "update",
                    before=before,
                    after=deepcopy(stored_value),
                )
                self._append_world_commit(
                    commit_kind="write",
                    operation=by_action,
                    authority_action=by_action,
                    actor_id=None,
                    idempotency_key=None,
                    subject_id=str(key),
                    table_writes=(write,),
                    invariants_held=("authorized-action", "single-row"),
                )

    def _append_transaction_commit(
        self,
        diff: TransactionStateDiff,
        *,
        authority_action: str,
        actor_id: str | None,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
    ) -> None:
        """Record a successful atomic transaction in both compatibility streams."""

        self._txn_diffs.append(diff)
        self._append_world_commit(
            commit_kind="transaction",
            operation=diff.txn,
            authority_action=authority_action,
            actor_id=actor_id,
            idempotency_key=idempotency_key,
            subject_id=diff.order_id,
            table_writes=diff.table_writes,
            invariants_held=diff.invariants_held,
            request_fingerprint=request_fingerprint,
        )

    def _append_world_commit(
        self,
        *,
        commit_kind: str,
        operation: str,
        authority_action: str,
        actor_id: str | None,
        idempotency_key: str | None,
        subject_id: str,
        table_writes: tuple[TableWrite, ...],
        invariants_held: tuple[str, ...],
        request_fingerprint: str | None = None,
    ) -> None:
        # Advance the materialized revision at the same single commit boundary
        # used by every World mutation.  An actor can neither supply nor bump
        # this value through an event payload.
        for write in table_writes:
            if write.table != "orders":
                continue
            order_key = str(write.key)
            if write.op == "create":
                self._order_state_revisions[order_key] = 1
            elif write.op == "update":
                self._order_state_revisions[order_key] = (
                    self._order_state_revisions.get(order_key, 1) + 1
                )
            else:
                raise WorldError(
                    f"unsupported order mutation op for revision ledger: {write.op!r}"
                )
        sequence = len(self._commit_journal)
        frozen_writes = tuple(
            TableWrite(
                table=write.table,
                key=write.key,
                op=write.op,
                before=_copy_world_value(write.before),
                after=_copy_world_value(write.after),
            )
            for write in table_writes
        )
        self._commit_journal.append(WorldCommitRecord(
            sequence=sequence,
            commit_id=f"world-commit:{sequence:08d}",
            commit_kind=commit_kind,
            operation=operation,
            authority_action=authority_action,
            actor_id=actor_id,
            idempotency_key=idempotency_key,
            subject_id=subject_id,
            table_writes=frozen_writes,
            invariants_held=tuple(invariants_held),
            request_fingerprint=request_fingerprint,
        ))

    def settle_order(
        self,
        *,
        order: Order,
        receipt: Receipt,
        by_role: str,
        idempotency_key: str,
    ) -> Receipt:
        """Atomically settle ``order``: mark it SETTLED, reserve inventory, and
        append the ledger ``receipt`` — ONE transaction under the world lock,
        idempotent on ``(by_role, idempotency_key)``.

        This is the cross-process settle primitive. The platform runs in a
        separate process and cannot hold this world's lock, so the whole
        read-check-write must live here. A retried settle (the dispatcher
        re-sending the same envelope) returns the original receipt without
        re-reserving inventory or double-appending the ledger — the bug the
        previously-dead ``_idempotency_cache`` was meant to prevent (a second
        ledger append would otherwise raise on the append-only table).

        Raises:
            OutOfStock: no inventory row, or available qty < ``order.qty``.
        """
        receipt = replace(receipt, effect="charge")
        cache_key = (by_role, idempotency_key)
        with self._lock:
            cached = self._idempotency_cache.get(cache_key)
            if cached is not None:
                if (
                    cached.operation != "settle"
                    or cached.order != order
                    or cached.request_receipt != receipt
                ):
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was already used "
                        "for a different request"
                    )
                return cached.outcome
            # Classify by the order's PERSISTED state (allowlist + default-deny):
            #  * already-settled -> idempotent replay of the prior receipt. The
            #    order's settled state + its ledger receipt are the durable
            #    idempotency signal (the key cache is only a fast path). This is
            #    also what stops a same-order/new-key re-settle from re-reserving
            #    inventory and partial-committing on the append-only ledger's
            #    txn_id collision.
            #  * NOT on the settleable allow-list (CANCELLED, or any future state)
            #    -> refuse before any write; never silently settle.
            #  * fresh order (None) or a settleable state -> proceed.
            existing = self.read("orders", order.order_id, caller=by_role)
            if existing is not None:
                validate_persisted_order_identity(existing, order)
                # A durable same-order replay may carry a new txn/key, but its
                # receipt must still describe the immutable requested order.
                validate_transaction_identity(
                    order, receipt, None, expected_effect="charge"
                )
                if existing.state in _ALREADY_SETTLED:
                    prior = self._receipt_for_order(order.order_id)
                    if prior is None:
                        raise WorldError(
                            f"order {order.order_id} is {existing.state.value} but has "
                            "no ledger receipt — refusing to re-settle"
                        )
                    self._idempotency_cache[cache_key] = _IdempotencyRecord(
                        "settle", order, receipt, prior
                    )
                    return prior
                if existing.state not in _SETTLEABLE:
                    raise OrderNotSettleable(
                        f"order {order.order_id} in state {existing.state.value!r} "
                        "is not settleable"
                    )
            inventory = self.read("inventory", order.sku_id, caller=by_role)
            if inventory is None:
                raise OutOfStock(f"no inventory for sku {order.sku_id}")
            validate_transaction_identity(
                order, receipt, inventory, expected_effect="charge"
            )
            listing = self.read("catalog", order.sku_id, caller=by_role)
            validate_listing_owner(order, listing)
            payment_history = cast(
                PaymentStateTable, self._tables["payment_states"]
            ).history_for_order(str(order.order_id))
            previous_payment = payment_history[-1] if payment_history else None
            if previous_payment is not None and previous_payment.state != "authorized":
                raise WorldError(
                    f"order {order.order_id} has payment state "
                    f"{previous_payment.state!r} before settlement"
                )
            reservation_already_held = (
                previous_payment is not None
                and previous_payment.state == "authorized"
            )
            if reservation_already_held:
                if inventory.qty_reserved < order.qty:
                    raise WorldError(
                        f"authorized payment for {order.order_id} has no matching "
                        "inventory reservation"
                    )
            elif _available_qty(inventory) < order.qty:
                raise OutOfStock(f"insufficient inventory for sku {order.sku_id}")
            if self.read("ledger", receipt.txn_id, caller=by_role) is not None:
                raise WriteNotAuthorized(f"ledger row already exists: {receipt.txn_id}")
            settled_order = replace(order, state=OrderState.SETTLED)
            reserved_inv = (
                inventory
                if reservation_already_held
                else _reserve_inventory(inventory, qty=order.qty)
            )
            timeline_before = self._tables["order_timelines"].read(
                order.order_id, caller=None
            )
            if timeline_before is not None:
                raise WorldError(
                    f"order {order.order_id} has timing evidence before its first settlement"
                )
            event_tick = self._logical_time + 1
            timeline = OrderTimeline(
                order_id=order.order_id,
                buyer_id=order.buyer_id,
                merchant_id=order.merchant_id,
                settled_at_tick=event_tick,
                return_window_ticks=_captured_return_window(listing),
            )
            payment = derive_payment_capture(
                settled_order,
                receipt,
                previous=previous_payment,
                original_actor="platform:psp",
                server_tick=event_tick,
                idempotency_key=f"settlement-capture:{idempotency_key}",
            )
            payment_key = payment_state_key(payment)
            diff_writes: list[tuple[str, str, str, Any, Any]] = [
                ("orders", str(order.order_id),
                 "update" if existing is not None else "create", existing, settled_order),
                ("ledger", str(receipt.txn_id), "create", None, receipt),
                (
                    "order_timelines",
                    str(order.order_id),
                    "create",
                    None,
                    timeline,
                ),
                ("payment_states", payment_key, "create", None, payment),
                ("logical_time", "world", "update", self._logical_time, event_tick),
            ]
            if reserved_inv != inventory:
                diff_writes.insert(
                    1,
                    (
                        "inventory",
                        str(order.sku_id),
                        "update",
                        inventory,
                        reserved_inv,
                    ),
                )
            diff = _make_diff(
                "settle",
                order.order_id,
                diff_writes,
                (
                    "atomic",
                    "allowlist:SETTLEABLE",
                    "idempotent",
                    "world-clock",
                    "first-class-payment-captured",
                    "single-inventory-reservation",
                ),
            )
            before_tick = self._logical_time
            diff_cursor = len(self._txn_diffs)
            commit_cursor = len(self._commit_journal)
            prior_revisions = dict(self._order_state_revisions)
            try:
                self.write(
                    "orders", order.order_id, settled_order,
                    by_action="world.create_order", _record_commit=False,
                )
                if reserved_inv != inventory:
                    self.write(
                        "inventory",
                        order.sku_id,
                        reserved_inv,
                        by_action="world.reserve_inventory",
                        _record_commit=False,
                    )
                self.write(
                    "ledger", receipt.txn_id, receipt,
                    by_action="world.update_ledger", _record_commit=False,
                )
                self.write(
                    "order_timelines",
                    order.order_id,
                    timeline,
                    by_action="world.record_order_timeline",
                    _record_commit=False,
                )
                self.write(
                    "payment_states",
                    payment_key,
                    payment,
                    by_action="world.settle_order",
                    _record_commit=False,
                )
                self._logical_time = event_tick
                self._append_transaction_commit(
                    diff,
                    authority_action="world.settle_order",
                    actor_id=by_role,
                    idempotency_key=idempotency_key,
                )
                self._idempotency_cache[cache_key] = _IdempotencyRecord(
                    "settle", order, receipt, receipt
                )
            except BaseException:
                _restore_row(self._tables["orders"], order.order_id, existing)
                _restore_row(self._tables["inventory"], order.sku_id, inventory)
                _restore_row(self._tables["ledger"], receipt.txn_id, None)
                _restore_row(
                    self._tables["order_timelines"], order.order_id, timeline_before
                )
                _restore_row(self._tables["payment_states"], payment_key, None)
                self._logical_time = before_tick
                self._order_state_revisions = prior_revisions
                del self._txn_diffs[diff_cursor:]
                del self._commit_journal[commit_cursor:]
                self._idempotency_cache.pop(cache_key, None)
                raise
            return receipt

    def _receipt_for_order(self, order_id: Any) -> "Receipt | None":
        """The ledger receipt for ``order_id`` — the durable record of a settle."""
        for _, rcpt in self._tables["ledger"].all():
            if (
                str(rcpt.order_id) == str(order_id)
                and not str(rcpt.txn_id).startswith("refund")
            ):
                return cast("Receipt", rcpt)
        return None

    def create_cart_quote_request(
        self,
        intent: CartQuoteIntent | Mapping[str, Any],
        *,
        market_id: str,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        request_ttl_ticks: int = 10,
    ) -> PersistentCartQuoteRequest:
        """Persist a buyer/principal authorization for merchant quote issuance.

        The compact intent contains only a mandate id, SKU quantities, and
        availability preferences.  World derives the buyer, principal,
        current mandate revision, listing owners, catalog snapshots, currency,
        clock, identifier, and digest.  Budget values are neither copied into
        the request nor checked here, so merchants cannot probe them.
        """

        if by_actor != "platform:checkout":
            raise LifecycleAuthorizationError(
                "only platform:checkout may create a cart quote request"
            )
        if not isinstance(market_id, str) or not market_id.strip():
            raise WorldError("cart quote request market_id must be non-empty")
        if (
            isinstance(request_ttl_ticks, bool)
            or not isinstance(request_ttl_ticks, int)
            or request_ttl_ticks <= 0
        ):
            raise WorldError("cart quote request ttl must be a positive integer")
        normalized = normalize_cart_quote_intent(intent)
        fingerprint = canonical_digest(
            {
                "intent": cart_quote_intent_fingerprint(
                    normalized,
                    market_id=market_id,
                    quote_ttl_ticks=request_ttl_ticks,
                ),
                "request_ttl_ticks": request_ttl_ticks,
            }
        )
        scope = f"cart-quote-request:{market_id}"
        with self._lock:
            replay = self._authority_operation_replay(
                scope=scope,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if replay is not None:
                if not isinstance(replay, PersistentCartQuoteRequest):
                    raise WorldError(
                        "cart quote request idempotency outcome has wrong type"
                    )
                return deepcopy(replay)

            authority = cast(
                "MandateRevisionAuthority | None",
                self._tables["mandate_authorities"].read(
                    normalized["mandate_id"], caller=None
                ),
            )
            if authority is None:
                raise WriteNotAuthorized("cart quote request mandate is missing")
            if original_actor not in {authority.buyer_id, authority.principal_id}:
                raise WriteNotAuthorized(
                    "cart quote request actor is not the mandate buyer or principal"
                )
            revisions = cast(
                MandateRevisionTable, self._tables["mandate_revisions"]
            ).by_mandate(normalized["mandate_id"])
            if not revisions:
                raise WriteNotAuthorized("cart quote request mandate has no revision")
            mandate = revisions[-1]
            if (
                mandate.buyer_id != authority.buyer_id
                or mandate.principal_id != authority.principal_id
            ):
                raise WriteNotAuthorized(
                    "cart quote request mandate authority is inconsistent"
                )

            lines: list[CartQuoteRequestLine] = []
            currencies: set[str] = set()
            for line in normalized["lines"]:
                listing = cast(
                    "Listing | None",
                    self._tables["catalog"].read(
                        SkuId(line["sku_id"]), caller=None
                    ),
                )
                if listing is None:
                    raise WorldError(
                        f"cart quote request listing {line['sku_id']!r} is missing"
                    )
                currencies.add(listing.list_price.currency)
                lines.append(
                    CartQuoteRequestLine(
                        sku_id=str(listing.sku_id),
                        product_id=listing.product_id,
                        merchant_id=str(listing.merchant_id),
                        qty=line["qty"],
                        catalog_revision=_catalog_revision(listing),
                        catalog_digest=catalog_snapshot_digest(listing),
                    )
                )
            if len(currencies) != 1:
                raise WorldError("cart quote request cannot mix currencies")
            if len({line.merchant_id for line in lines}) != 1:
                raise WorldError(
                    "merchant quote authorization requires one merchant; "
                    "buyers may request a direct cross-merchant quote"
                )
            event_tick = self._logical_time + 1
            request_id = "cart-request:" + canonical_digest(
                {
                    "market_id": market_id,
                    "created_by": original_actor,
                    "idempotency_key": idempotency_key,
                    "fingerprint": fingerprint,
                }
            )[:32]
            request = build_persistent_cart_quote_request(
                request_id=request_id,
                market_id=market_id,
                buyer_id=authority.buyer_id,
                principal_id=authority.principal_id,
                created_by=original_actor,
                issuer_id="world",
                mandate_id=authority.mandate_id,
                mandate_revision=mandate.revision,
                mandate_digest=mandate.revision_digest,
                idempotency_key=idempotency_key,
                fill_policy=normalized["fill_policy"],
                backorder_policy=normalized["backorder_policy"],
                currency=next(iter(currencies)),
                lines=lines,
                issued_at_tick=event_tick,
                expires_at_tick=event_tick + request_ttl_ticks,
            )
            requests = cast(
                PersistentCartQuoteRequestTable,
                self._tables["persistent_cart_quote_requests"],
            )
            operations = cast(
                AuthorityOperationTable, self._tables["authority_operations"]
            )
            operation_key = authority_operation_key(
                scope, original_actor, idempotency_key
            )
            operation = AuthorityOperationRecord(
                operation_key=operation_key,
                scope=scope,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                outcome_table="persistent_cart_quote_requests",
                outcome_key=request.request_id,
            )
            before_tick = self._logical_time
            diff_cursor = len(self._txn_diffs)
            commit_cursor = len(self._commit_journal)
            try:
                requests.write(request.request_id, request)
                operations.write(operation_key, operation)
                self._logical_time = event_tick
                self._append_transaction_commit(
                    _make_diff(
                        "create_cart_quote_request",
                        request.request_id,
                        [
                            (
                                "persistent_cart_quote_requests",
                                request.request_id,
                                "create",
                                None,
                                request,
                            ),
                            (
                                "authority_operations",
                                operation_key,
                                "create",
                                None,
                                operation,
                            ),
                            (
                                "logical_time",
                                "world",
                                "update",
                                before_tick,
                                event_tick,
                            ),
                        ],
                        (
                            "buyer-or-principal-authorized",
                            "mandate-revision-bound",
                            "catalog-owner-derived",
                            "budget-values-excluded",
                            "actor-scoped-idempotency",
                        ),
                    ),
                    authority_action="world.create_cart_quote_request",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
            except BaseException:
                self._logical_time = before_tick
                requests.delete(request.request_id)
                operations.delete(operation_key)
                del self._txn_diffs[diff_cursor:]
                del self._commit_journal[commit_cursor:]
                raise
            return deepcopy(request)

    def issue_cart_quote_from_request(
        self,
        request_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        quote_ttl_ticks: int = 10,
    ) -> PersistentCartQuote:
        """Let one authorized merchant price a persisted buyer request."""

        if by_actor != "platform:checkout":
            raise LifecycleAuthorizationError(
                "only platform:checkout may issue a requested cart quote"
            )
        if original_actor.split(":", 1)[0] != "merchant":
            raise LifecycleAuthorizationError(
                "requested cart quote actor must be a fully qualified merchant"
            )
        if not isinstance(request_id, str) or not request_id.strip():
            raise WorldError("cart quote request_id must be non-empty")
        with self._lock:
            request = cast(
                "PersistentCartQuoteRequest | None",
                self._tables["persistent_cart_quote_requests"].read(
                    request_id, caller=None
                ),
            )
            if request is None:
                raise WriteNotAuthorized("cart quote request is unknown or unauthorized")
            validate_persistent_cart_quote_request(request)
            normalized = normalize_cart_quote_intent(
                {
                    "mandate_id": request.mandate_id,
                    "lines": [
                        {"sku_id": line.sku_id, "qty": line.qty}
                        for line in request.lines
                    ],
                    "fill_policy": request.fill_policy,
                    "backorder_policy": request.backorder_policy,
                }
            )
            fingerprint = canonical_digest(
                {
                    "request_id": request.request_id,
                    "request_digest": request.request_digest,
                    "quote_ttl_ticks": quote_ttl_ticks,
                }
            )
            scope = f"cart-quote:{request.market_id}"
            replay = self._authority_operation_replay(
                scope=scope,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if replay is not None:
                if not isinstance(replay, PersistentCartQuote):
                    raise WorldError("cart quote idempotency outcome has wrong type")
                return deepcopy(replay)

            revisions = cast(
                MandateRevisionTable, self._tables["mandate_revisions"]
            ).by_mandate(request.mandate_id)
            if not revisions:
                raise WriteNotAuthorized("cart quote request mandate is missing")
            current_mandate = revisions[-1]
            event_tick = self._logical_time + 1
            try:
                assert_cart_quote_request_usable(
                    request,
                    merchant_id=original_actor,
                    mandate_revision=current_mandate.revision,
                    mandate_digest=current_mandate.revision_digest,
                    logical_tick=event_tick,
                )
            except CartQuoteRequestStaleError as exc:
                raise CartQuoteStaleError(str(exc)) from exc
            except CartQuoteRequestAuthorityError as exc:
                raise WriteNotAuthorized(str(exc)) from exc

            listings: dict[str, Listing] = {}
            inventory: dict[str, InventoryRow] = {}
            policies: dict[str, PricingPolicyRevision] = {}
            all_policies = tuple(
                row for _, row in self._tables["pricing_policy_revisions"].all()
            )
            for request_line in request.lines:
                listing = cast(
                    "Listing | None",
                    self._tables["catalog"].read(
                        SkuId(request_line.sku_id), caller=None
                    ),
                )
                stock = self._tables["inventory"].read(
                    SkuId(request_line.sku_id), caller="platform:checkout"
                )
                if listing is None or not isinstance(stock, InventoryRow):
                    raise OutOfStock("requested cart source is unavailable")
                if (
                    str(listing.merchant_id) != request_line.merchant_id
                    or _catalog_revision(listing) != request_line.catalog_revision
                    or catalog_snapshot_digest(listing) != request_line.catalog_digest
                ):
                    raise CartQuoteStaleError(
                        "cart quote request catalog binding changed"
                    )
                listings[request_line.sku_id] = listing
                inventory[request_line.sku_id] = stock
                policies[request_line.sku_id] = _active_pricing_policy_for_listing(
                    all_policies,
                    market_id=request.market_id,
                    listing=listing,
                    logical_tick=event_tick,
                )
            quote = build_authoritative_cart_quote(
                intent=normalized,
                market_id=request.market_id,
                requested_by=original_actor,
                issuer_id="world",
                idempotency_key=idempotency_key,
                authority=QuoteAuthority(
                    principal_id=request.principal_id,
                    buyer_id=request.buyer_id,
                    mandate_revision=request.mandate_revision,
                    mandate_digest=request.mandate_digest,
                ),
                listings=listings,
                inventory=inventory,
                policies_by_sku=policies,
                issued_at_tick=event_tick,
                quote_ttl_ticks=min(
                    quote_ttl_ticks, request.expires_at_tick - event_tick
                ),
                request_id=request.request_id,
            )
            # Deliberately no budget comparison here.  A merchant sees only the
            # commercial request and quote.  Budget enforcement remains at the
            # buyer-authenticated atomic checkout boundary.
            return self._persist_authoritative_cart_quote(
                quote,
                scope=scope,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                before_tick=self._logical_time,
                event_tick=event_tick,
            )

    def _persist_authoritative_cart_quote(
        self,
        quote: PersistentCartQuote,
        *,
        scope: str,
        actor_id: str,
        idempotency_key: str,
        request_fingerprint: str,
        before_tick: int,
        event_tick: int,
    ) -> PersistentCartQuote:
        """Commit one World-built quote and its authority operation atomically."""

        disposition, _ = apply_cart_quote_record(
            tuple(
                row for _, row in self._tables["persistent_cart_quotes"].all()
            ),
            quote,
            trusted_issuer_id="world",
            server_tick=event_tick,
        )
        if disposition != "append":
            raise WorldError("new cart quote unexpectedly classified as retry")
        quotes = cast(
            PersistentCartQuoteTable, self._tables["persistent_cart_quotes"]
        )
        operations = cast(
            AuthorityOperationTable, self._tables["authority_operations"]
        )
        operation_key = authority_operation_key(
            scope, actor_id, idempotency_key
        )
        operation = AuthorityOperationRecord(
            operation_key=operation_key,
            scope=scope,
            actor_id=actor_id,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            outcome_table="persistent_cart_quotes",
            outcome_key=quote.quote_id,
        )
        diff_cursor = len(self._txn_diffs)
        commit_cursor = len(self._commit_journal)
        try:
            quotes.write(quote.quote_id, quote)
            operations.write(operation_key, operation)
            self._logical_time = event_tick
            self._append_transaction_commit(
                _make_diff(
                    "issue_cart_quote",
                    quote.quote_id,
                    [
                        (
                            "persistent_cart_quotes",
                            quote.quote_id,
                            "create",
                            None,
                            quote,
                        ),
                        (
                            "authority_operations",
                            operation_key,
                            "create",
                            None,
                            operation,
                        ),
                        (
                            "logical_time",
                            "world",
                            "update",
                            before_tick,
                            event_tick,
                        ),
                    ],
                    (
                        "world-owned-pricing",
                        "principal-mandate-bound",
                        "catalog-inventory-policy-snapshots",
                        "actor-scoped-idempotency",
                    ),
                ),
                authority_action="world.issue_cart_quote",
                actor_id=actor_id,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
            )
        except BaseException:
            self._logical_time = before_tick
            quotes.delete(quote.quote_id)
            operations.delete(operation_key)
            del self._txn_diffs[diff_cursor:]
            del self._commit_journal[commit_cursor:]
            raise
        return deepcopy(quote)

    def issue_cart_quote(
        self,
        intent: CartQuoteIntent | Mapping[str, Any],
        *,
        market_id: str,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        quote_ttl_ticks: int = 10,
    ) -> PersistentCartQuote:
        """Build and persist a mandate-bound quote from authoritative tables."""

        if by_actor != "platform:checkout":
            raise LifecycleAuthorizationError(
                "only platform:checkout may request an authoritative cart quote"
            )
        if original_actor.split(":", 1)[0] != "buyer":
            raise LifecycleAuthorizationError(
                "cart quote original actor must be a fully qualified buyer"
            )
        if not isinstance(market_id, str) or not market_id.strip():
            raise WorldError("cart quote market_id must be non-empty")
        normalized = normalize_cart_quote_intent(intent)
        fingerprint = cart_quote_intent_fingerprint(
            normalized,
            market_id=market_id,
            quote_ttl_ticks=quote_ttl_ticks,
        )
        scope = f"cart-quote:{market_id}"
        with self._lock:
            replay = self._authority_operation_replay(
                scope=scope,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if replay is not None:
                if not isinstance(replay, PersistentCartQuote):
                    raise WorldError("cart quote idempotency outcome has wrong type")
                return deepcopy(replay)

            authorities = cast(
                MandateAuthorityTable, self._tables["mandate_authorities"]
            )
            authority = authorities.read(normalized["mandate_id"], caller=None)
            if authority is None:
                raise WriteNotAuthorized("cart quote mandate authority is missing")
            if authority.buyer_id != original_actor:
                raise WriteNotAuthorized(
                    "cart quote actor does not own the persisted mandate"
                )
            revisions = cast(
                MandateRevisionTable, self._tables["mandate_revisions"]
            ).by_mandate(normalized["mandate_id"])
            if not revisions:
                raise WriteNotAuthorized("cart quote mandate has no revision")
            current_mandate = revisions[-1]
            if (
                current_mandate.buyer_id != authority.buyer_id
                or current_mandate.principal_id != authority.principal_id
            ):
                raise WriteNotAuthorized("cart quote mandate authority is inconsistent")

            listings: dict[str, Listing] = {}
            inventory: dict[str, InventoryRow] = {}
            selected_policies: dict[str, PricingPolicyRevision] = {}
            event_tick = self._logical_time + 1
            all_policies = tuple(
                row for _, row in self._tables["pricing_policy_revisions"].all()
            )
            for line in normalized["lines"]:
                sku_id = line["sku_id"]
                listing = cast(
                    "Listing | None",
                    self._tables["catalog"].read(SkuId(sku_id), caller=None),
                )
                stock = self._tables["inventory"].read(
                    SkuId(sku_id), caller="platform:checkout"
                )
                if listing is None or not isinstance(stock, InventoryRow):
                    raise OutOfStock(
                        f"cart quote requires authoritative listing and inventory "
                        f"for {sku_id!r}"
                    )
                listings[sku_id] = listing
                inventory[sku_id] = stock
                selected_policies[sku_id] = _active_pricing_policy_for_listing(
                    all_policies,
                    market_id=market_id,
                    listing=listing,
                    logical_tick=event_tick,
                )

            quote = build_authoritative_cart_quote(
                intent=normalized,
                market_id=market_id,
                requested_by=original_actor,
                issuer_id="world",
                idempotency_key=idempotency_key,
                authority=QuoteAuthority(
                    principal_id=authority.principal_id,
                    buyer_id=authority.buyer_id,
                    mandate_revision=current_mandate.revision,
                    mandate_digest=current_mandate.revision_digest,
                ),
                listings=listings,
                inventory=inventory,
                policies_by_sku=selected_policies,
                issued_at_tick=event_tick,
                quote_ttl_ticks=quote_ttl_ticks,
            )
            if quote.grand_total_minor > _mandate_budget_minor(revisions):
                raise WriteNotAuthorized(
                    "authoritative cart quote exceeds the principal mandate budget"
                )
            disposition, _ = apply_cart_quote_record(
                tuple(
                    row for _, row in self._tables["persistent_cart_quotes"].all()
                ),
                quote,
                trusted_issuer_id="world",
                server_tick=event_tick,
            )
            if disposition != "append":
                raise WorldError("new cart quote unexpectedly classified as retry")

            quotes = cast(
                PersistentCartQuoteTable,
                self._tables["persistent_cart_quotes"],
            )
            operations = cast(
                AuthorityOperationTable, self._tables["authority_operations"]
            )
            operation_key = authority_operation_key(
                scope, original_actor, idempotency_key
            )
            operation = AuthorityOperationRecord(
                operation_key=operation_key,
                scope=scope,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                outcome_table="persistent_cart_quotes",
                outcome_key=quote.quote_id,
            )
            before_tick = self._logical_time
            diff_cursor = len(self._txn_diffs)
            commit_cursor = len(self._commit_journal)
            try:
                quotes.write(quote.quote_id, quote)
                operations.write(operation_key, operation)
                self._logical_time = event_tick
                self._append_transaction_commit(
                    _make_diff(
                        "issue_cart_quote",
                        quote.quote_id,
                        [
                            (
                                "persistent_cart_quotes",
                                quote.quote_id,
                                "create",
                                None,
                                quote,
                            ),
                            (
                                "authority_operations",
                                operation_key,
                                "create",
                                None,
                                operation,
                            ),
                            (
                                "logical_time",
                                "world",
                                "update",
                                before_tick,
                                event_tick,
                            ),
                        ],
                        (
                            "world-owned-pricing",
                            "principal-mandate-bound",
                            "catalog-inventory-policy-snapshots",
                            "actor-scoped-idempotency",
                        ),
                    ),
                    authority_action="world.issue_cart_quote",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
            except BaseException:
                self._logical_time = before_tick
                quotes.delete(quote.quote_id)
                operations.delete(operation_key)
                del self._txn_diffs[diff_cursor:]
                del self._commit_journal[commit_cursor:]
                raise
            return deepcopy(quote)

    def checkout_cart(
        self,
        *,
        quote_id: str,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> OrderGroup:
        """Atomically settle one World-persisted quote by identifier only."""

        if by_actor != "platform:checkout":
            raise LifecycleAuthorizationError(
                "only platform:checkout may checkout a persistent cart quote"
            )
        if not isinstance(quote_id, str) or not quote_id.strip():
            raise OrderNotSettleable("cart checkout quote_id must be non-empty")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise IdempotencyConflict("cart checkout idempotency key must be non-empty")
        with self._lock:
            quote = cast(
                "PersistentCartQuote | None",
                self._tables["persistent_cart_quotes"].read(
                    quote_id, caller="platform:checkout"
                ),
            )
            if quote is None:
                raise OrderNotSettleable("cart checkout references an unknown quote")
            validate_persistent_cart_quote(quote)
            if original_actor != quote.buyer_id:
                raise LifecycleAuthorizationError(
                    "cart checkout must be authenticated by the quoted buyer"
                )
            fingerprint = canonical_digest({"quote_id": quote_id})
            scope = f"cart-checkout:{quote.market_id}"
            replay = self._authority_operation_replay(
                scope=scope,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if replay is not None:
                if not isinstance(replay, OrderGroup):
                    raise WorldError("cart checkout idempotency outcome has wrong type")
                return deepcopy(replay)

            group_id = OrderGroupId(f"group:{quote.quote_id}")
            if self._tables["order_groups"].read(
                group_id, caller="platform:checkout"
            ) is not None:
                raise IdempotencyConflict(
                    "persistent cart quote was already checked out under another key"
                )
            if any(
                line.backorder_qty or line.unfilled_qty or not line.fulfill_now_qty
                for line in quote.lines
            ):
                raise OrderNotSettleable(
                    "checkout currently requires every persisted quote line to be "
                    "fully available"
                )

            authorities = cast(
                MandateAuthorityTable, self._tables["mandate_authorities"]
            )
            authority = authorities.read(quote.mandate_id, caller=None)
            revisions = cast(
                MandateRevisionTable, self._tables["mandate_revisions"]
            ).by_mandate(quote.mandate_id)
            if authority is None or not revisions:
                raise WriteNotAuthorized("cart checkout mandate authority is missing")
            current_mandate = revisions[-1]
            event_tick = self._logical_time + 1
            assert_cart_quote_usable(
                quote,
                buyer_id=authority.buyer_id,
                principal_id=authority.principal_id,
                mandate_id=authority.mandate_id,
                mandate_revision=current_mandate.revision,
                mandate_digest=current_mandate.revision_digest,
                logical_tick=event_tick,
            )
            if quote.grand_total_minor > _mandate_budget_minor(revisions):
                raise WriteNotAuthorized(
                    "cart checkout exceeds the current principal mandate budget"
                )

            listings: dict[str, Listing] = {}
            inventory: dict[str, InventoryRow] = {}
            selected_policies: dict[str, PricingPolicyRevision] = {}
            all_policies = tuple(
                row for _, row in self._tables["pricing_policy_revisions"].all()
            )
            for line in quote.lines:
                listing = cast(
                    "Listing | None",
                    self._tables["catalog"].read(SkuId(line.sku_id), caller=None),
                )
                stock = self._tables["inventory"].read(
                    SkuId(line.sku_id), caller="platform:checkout"
                )
                if listing is None or not isinstance(stock, InventoryRow):
                    raise CartQuoteStaleError(
                        f"authoritative cart source disappeared for {line.sku_id!r}"
                    )
                listings[line.sku_id] = listing
                inventory[line.sku_id] = stock
                selected_policies[line.sku_id] = _active_pricing_policy_for_listing(
                    all_policies,
                    market_id=quote.market_id,
                    listing=listing,
                    logical_tick=event_tick,
                )
            assert_cart_quote_snapshots_current(
                quote,
                catalog_revisions={
                    sku_id: _catalog_revision(listing)
                    for sku_id, listing in listings.items()
                },
                catalog_digests={
                    sku_id: catalog_snapshot_digest(listing)
                    for sku_id, listing in listings.items()
                },
                inventory_revisions={
                    sku_id: stock.version for sku_id, stock in inventory.items()
                },
                inventory_digests={
                    sku_id: inventory_snapshot_digest(stock)
                    for sku_id, stock in inventory.items()
                },
                merchant_by_sku={
                    sku_id: str(listing.merchant_id)
                    for sku_id, listing in listings.items()
                },
            )
            policies = tuple(
                {policy.policy_digest: policy for policy in selected_policies.values()}.values()
            )
            if (
                pricing_policy_set_digest(policies) != quote.pricing_policy_digest
                or max(policy.revision for policy in policies)
                != quote.pricing_policy_revision
            ):
                raise CartQuoteStaleError("cart pricing policy snapshot changed")

            orders: list[Order] = []
            receipts: list[Receipt] = []
            timelines: list[OrderTimeline] = []
            reserved: dict[str, InventoryRow] = {}
            for index, line in enumerate(quote.lines, start=1):
                stock = inventory[line.sku_id]
                if _available_qty(stock) < line.fulfill_now_qty:
                    raise OutOfStock(
                        f"insufficient inventory for quoted sku {line.sku_id!r}"
                    )
                order_id = OrderId(f"order:{quote.quote_id}:{index:03d}")
                txn_id = TxnId(f"txn:{quote.quote_id}:{index:03d}")
                if (
                    self._tables["orders"].read(order_id, caller="platform:checkout")
                    is not None
                    or self._tables["ledger"].read(
                        txn_id, caller="platform:checkout"
                    )
                    is not None
                ):
                    raise IdempotencyConflict(
                        f"cart checkout durable row collision for {order_id}"
                    )
                unit_price = Money(
                    Decimal(line.unit_price_minor) / Decimal(100), quote.currency
                )
                order = Order(
                    order_id=order_id,
                    buyer_id=AgentId(quote.buyer_id),
                    merchant_id=AgentId(line.merchant_id),
                    sku_id=SkuId(line.sku_id),
                    qty=line.fulfill_now_qty,
                    agreed_price=unit_price,
                    state=OrderState.SETTLED,
                )
                receipt = Receipt(
                    txn_id=txn_id,
                    ts=f"world-tick:{event_tick}",
                    order_id=order_id,
                    buyer_id=order.buyer_id,
                    merchant_id=order.merchant_id,
                    sku_id=order.sku_id,
                    qty=order.qty,
                    price=unit_price,
                    idempotency_key=idempotency_key,
                )
                timeline = OrderTimeline(
                    order_id=order_id,
                    buyer_id=order.buyer_id,
                    merchant_id=order.merchant_id,
                    settled_at_tick=event_tick,
                    return_window_ticks=_captured_return_window(
                        listings[line.sku_id]
                    ),
                )
                orders.append(order)
                receipts.append(receipt)
                timelines.append(timeline)
                reserved[line.sku_id] = _reserve_inventory(
                    stock, qty=line.fulfill_now_qty
                )

            fee_breakdown = tuple(
                FeeComponent(
                    fee_id=charge.charge_id,
                    kind=charge.kind,
                    scope=charge.scope,
                    amount=Money(
                        Decimal(charge.amount_minor) / Decimal(100), quote.currency
                    ),
                    basis=charge.basis,
                )
                for charge in quote.charges
            )
            group = OrderGroup(
                order_group_id=group_id,
                quote_id=QuoteId(quote.quote_id),
                buyer_id=AgentId(quote.buyer_id),
                merchant_ids=tuple(
                    AgentId(value)
                    for value in sorted({line.merchant_id for line in quote.lines})
                ),
                order_ids=tuple(order.order_id for order in orders),
                txn_ids=tuple(receipt.txn_id for receipt in receipts),
                subtotal=Money(
                    Decimal(quote.subtotal_minor) / Decimal(100), quote.currency
                ),
                fee_breakdown=fee_breakdown,
                grand_total=Money(
                    Decimal(quote.grand_total_minor) / Decimal(100), quote.currency
                ),
                quote_hash=quote.quote_digest,
                idempotency_key=idempotency_key,
            )
            _validate_group_payment_record(group, receipts, quote)
            operation_key = authority_operation_key(
                scope, original_actor, idempotency_key
            )
            operation = AuthorityOperationRecord(
                operation_key=operation_key,
                scope=scope,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                outcome_table="order_groups",
                outcome_key=str(group_id),
            )
            writes: list[tuple[str, str, str, Any, Any]] = []
            for order, receipt, timeline in zip(
                orders, receipts, timelines, strict=True
            ):
                writes.extend(
                    [
                        ("orders", str(order.order_id), "create", None, order),
                        (
                            "inventory",
                            str(order.sku_id),
                            "update",
                            inventory[str(order.sku_id)],
                            reserved[str(order.sku_id)],
                        ),
                        ("ledger", str(receipt.txn_id), "create", None, receipt),
                        (
                            "order_timelines",
                            str(order.order_id),
                            "create",
                            None,
                            timeline,
                        ),
                    ]
                )
            writes.extend(
                [
                    ("order_groups", str(group_id), "create", None, group),
                    (
                        "authority_operations",
                        operation_key,
                        "create",
                        None,
                        operation,
                    ),
                    (
                        "logical_time",
                        "world",
                        "update",
                        self._logical_time,
                        event_tick,
                    ),
                ]
            )
            diff = _make_diff(
                "checkout_cart_quote",
                group_id,
                writes,
                (
                    "world-loaded-quote",
                    "atomic-cross-merchant",
                    "mandate-budget-current",
                    "catalog-inventory-policy-fresh",
                    "inventory-no-oversell",
                    "line-receipts-plus-group-charges-equal-grand-total",
                    "quote-digest-bound-group-payment",
                    "actor-scoped-idempotency",
                ),
            )
            prior_revisions = dict(self._order_state_revisions)
            before_tick = self._logical_time
            diff_cursor = len(self._txn_diffs)
            commit_cursor = len(self._commit_journal)
            try:
                for order, receipt, timeline in zip(
                    orders, receipts, timelines, strict=True
                ):
                    self.write(
                        "orders",
                        order.order_id,
                        order,
                        by_action="world.create_order",
                        _record_commit=False,
                    )
                    self.write(
                        "inventory",
                        order.sku_id,
                        reserved[str(order.sku_id)],
                        by_action="world.reserve_inventory",
                        _record_commit=False,
                    )
                    self.write(
                        "ledger",
                        receipt.txn_id,
                        receipt,
                        by_action="world.update_ledger",
                        _record_commit=False,
                    )
                    self.write(
                        "order_timelines",
                        order.order_id,
                        timeline,
                        by_action="world.record_order_timeline",
                        _record_commit=False,
                    )
                self.write(
                    "order_groups",
                    group_id,
                    group,
                    by_action="world.checkout_cart",
                    _record_commit=False,
                )
                cast(
                    AuthorityOperationTable,
                    self._tables["authority_operations"],
                ).write(operation_key, operation)
                self._logical_time = event_tick
                self._append_transaction_commit(
                    diff,
                    authority_action="world.checkout_cart_quote",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
            except BaseException:
                for order, receipt in zip(orders, receipts, strict=True):
                    _restore_row(self._tables["orders"], order.order_id, None)
                    _restore_row(self._tables["ledger"], receipt.txn_id, None)
                    _restore_row(
                        self._tables["order_timelines"], order.order_id, None
                    )
                    _restore_row(
                        self._tables["inventory"],
                        order.sku_id,
                        inventory[str(order.sku_id)],
                    )
                _restore_row(self._tables["order_groups"], group_id, None)
                self._tables["authority_operations"].delete(operation_key)
                self._order_state_revisions = prior_revisions
                self._logical_time = before_tick
                del self._txn_diffs[diff_cursor:]
                del self._commit_journal[commit_cursor:]
                raise
            return deepcopy(group)

    def settle_order_partial(
        self,
        *,
        order: Order,
        fulfilled_qty: int,
        receipt: Receipt | None,
        by_actor: str,
        idempotency_key: str,
    ) -> FulfillmentAllocation:
        """Atomically allocate a requested quantity between fill and backorder.

        This additive path is deliberately separate from :meth:`settle_order`:
        legacy settlement continues to require ``Receipt.qty == Order.qty``.
        Here ``Order.qty`` remains the requested quantity while the receipt, if
        any, covers only ``fulfilled_qty``. A zero fill writes no receipt and
        never creates a settled-order state.

        The allocation is single-shot per order. Exact same-key retries return
        the persisted record; key reuse with another request raises
        :class:`IdempotencyConflict`; a second allocation for the same order
        raises :class:`FulfillmentNotActionable`.
        """
        if receipt is not None:
            receipt = replace(receipt, effect="charge")
        with self._lock:
            _authorize_psp(by_actor, operation="allocate fulfillment")
            _validate_requested_and_fulfilled(order, fulfilled_qty)
            if order.state not in _SETTLEABLE:
                raise OrderNotSettleable(
                    f"order {order.order_id} in state {order.state.value!r} "
                    "is not eligible for an initial fulfillment allocation"
                )
            backordered_qty = order.qty - fulfilled_qty
            allocation = FulfillmentAllocation(
                order_id=order.order_id,
                buyer_id=order.buyer_id,
                merchant_id=order.merchant_id,
                sku_id=order.sku_id,
                requested_qty=order.qty,
                fulfilled_qty=fulfilled_qty,
                backordered_qty=backordered_qty,
                receipt_txn_id=receipt.txn_id if receipt is not None else None,
                created_by=AgentId(by_actor),
                idempotency_key=idempotency_key,
            )
            replay = self._fulfillment_replay(
                allocation=allocation,
                receipt=receipt,
                order=order,
            )
            if replay is not None:
                return replay

            existing = cast(
                "Order | None",
                self._tables["orders"].read(order.order_id, caller=None),
            )
            if existing is not None:
                _validate_same_order_identity(existing, order)
                if existing.state not in _SETTLEABLE:
                    raise OrderNotSettleable(
                        f"order {order.order_id} in state {existing.state.value!r} "
                        "is not fulfillable"
                    )

            inventory = self._tables["inventory"].read(order.sku_id, caller=None)
            if inventory is None:
                raise OutOfStock(f"no inventory row for sku {order.sku_id}")
            listing = self._tables["catalog"].read(order.sku_id, caller=None)
            if listing is None:
                raise FulfillmentNotActionable(
                    f"listing {order.sku_id} does not exist"
                )
            validate_listing_owner(order, listing)
            if isinstance(inventory, InventoryRow) and (
                inventory.merchant_id != order.merchant_id
            ):
                raise FulfillmentNotActionable(
                    "inventory merchant does not own the requested order"
                )
            if _available_qty(inventory) < fulfilled_qty:
                raise OutOfStock(f"insufficient inventory for sku {order.sku_id}")
            _validate_partial_receipt(
                order=order,
                fulfilled_qty=fulfilled_qty,
                receipt=receipt,
                inventory=inventory,
                idempotency_key=idempotency_key,
            )
            if receipt is not None and self._tables["ledger"].read(
                receipt.txn_id,
                caller=None,
            ) is not None:
                raise WriteNotAuthorized(f"ledger row already exists: {receipt.txn_id}")

            target_state = (
                OrderState.BACKORDERED
                if fulfilled_qty == 0
                else OrderState.PARTIALLY_SETTLED
                if backordered_qty > 0
                else OrderState.SETTLED
            )
            persisted = replace(order, state=target_state)
            reserved = (
                _reserve_inventory(inventory, qty=fulfilled_qty)
                if fulfilled_qty > 0
                else inventory
            )
            writes: list[tuple[str, str, str, Any, Any]] = [
                (
                    "orders",
                    str(order.order_id),
                    "update" if existing is not None else "create",
                    existing,
                    persisted,
                ),
                (
                    "fulfillments",
                    str(order.order_id),
                    "create",
                    None,
                    allocation,
                ),
            ]
            if fulfilled_qty > 0:
                writes.insert(
                    1,
                    (
                        "inventory",
                        str(order.sku_id),
                        "update",
                        inventory,
                        reserved,
                    ),
                )
                assert receipt is not None
                writes.append(
                    ("ledger", str(receipt.txn_id), "create", None, receipt)
                )
            timing_before = self._tables["order_timelines"].read(
                order.order_id, caller=None
            )
            event_tick: int | None = None
            timing_after: OrderTimeline | None = None
            captured_payment: PaymentStateRecord | None = None
            if fulfilled_qty > 0:
                if timing_before is not None:
                    raise WorldError(
                        f"order {order.order_id} has timing evidence before settlement"
                    )
                event_tick = self._logical_time + 1
                timing_after = OrderTimeline(
                    order_id=order.order_id,
                    buyer_id=order.buyer_id,
                    merchant_id=order.merchant_id,
                    settled_at_tick=event_tick,
                    return_window_ticks=_captured_return_window(listing),
                )
                writes.append((
                    "order_timelines",
                    str(order.order_id),
                    "create",
                    None,
                    timing_after,
                ))
                if backordered_qty == 0:
                    assert receipt is not None
                    captured_payment = derive_payment_capture(
                        persisted,
                        receipt,
                        previous=None,
                        original_actor="platform:psp",
                        server_tick=event_tick,
                        idempotency_key=(
                            f"partial-settlement-capture:{idempotency_key}"
                        ),
                    )
                    writes.append((
                        "payment_states",
                        payment_state_key(captured_payment),
                        "create",
                        None,
                        captured_payment,
                    ))
                writes.append((
                    "logical_time",
                    "world",
                    "update",
                    self._logical_time,
                    event_tick,
                ))
            diff = _make_diff(
                "partial_settle",
                order.order_id,
                writes,
                (
                    "atomic",
                    "requested=fulfilled+backordered",
                    "no-oversell",
                    "zero-fill-no-payment",
                    "idempotent",
                ),
            )
            before_tick = self._logical_time
            diff_cursor = len(self._txn_diffs)
            commit_cursor = len(self._commit_journal)
            prior_revisions = dict(self._order_state_revisions)
            try:
                self.write(
                    "orders",
                    order.order_id,
                    persisted,
                    by_action="world.create_order",
                    _record_commit=False,
                )
                if fulfilled_qty > 0:
                    self.write(
                        "inventory",
                        order.sku_id,
                        reserved,
                        by_action="world.reserve_inventory",
                        _record_commit=False,
                    )
                    assert receipt is not None
                    self.write(
                        "ledger",
                        receipt.txn_id,
                        receipt,
                        by_action="world.update_ledger",
                        _record_commit=False,
                    )
                self.write(
                    "fulfillments",
                    order.order_id,
                    allocation,
                    by_action="world.record_fulfillment",
                    _record_commit=False,
                )
                if timing_after is not None:
                    self.write(
                        "order_timelines",
                        order.order_id,
                        timing_after,
                        by_action="world.record_order_timeline",
                        _record_commit=False,
                    )
                if captured_payment is not None:
                    self.write(
                        "payment_states",
                        payment_state_key(captured_payment),
                        captured_payment,
                        by_action="world.settle_order_partial",
                        _record_commit=False,
                    )
                if event_tick is not None:
                    self._logical_time = event_tick
                self._append_transaction_commit(
                    diff,
                    authority_action="world.settle_order_partial",
                    actor_id=by_actor,
                    idempotency_key=idempotency_key,
                )
            except BaseException:
                _restore_row(self._tables["orders"], order.order_id, existing)
                _restore_row(self._tables["inventory"], order.sku_id, inventory)
                if receipt is not None:
                    _restore_row(self._tables["ledger"], receipt.txn_id, None)
                _restore_row(self._tables["fulfillments"], order.order_id, None)
                _restore_row(
                    self._tables["order_timelines"], order.order_id, timing_before
                )
                if captured_payment is not None:
                    _restore_row(
                        self._tables["payment_states"],
                        payment_state_key(captured_payment),
                        None,
                    )
                self._logical_time = before_tick
                self._order_state_revisions = prior_revisions
                del self._txn_diffs[diff_cursor:]
                del self._commit_journal[commit_cursor:]
                raise
            return allocation

    def allocate_orders_atomic(
        self,
        *,
        allocation_id: str,
        merchant_id: AgentId,
        sku_id: SkuId,
        priority_order_ids: tuple[OrderId, ...],
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AllocationBatch:
        """Allocate one scarce SKU across every pending prioritized order.

        The merchant chooses to commit the batch, but cannot invent quantities or
        reorder buyers: World derives both from authoritative orders and stock.
        Every order, receipt, fulfillment row, reservation, and timeline is one
        transaction commit.
        """

        if by_actor != "platform:fulfillment":
            raise LifecycleAuthorizationError(
                f"only platform:fulfillment may allocate orders; got {by_actor!r}"
            )
        if original_actor != str(merchant_id):
            raise LifecycleAuthorizationError(
                "allocation original_actor must equal the owning merchant"
            )
        if not allocation_id.strip() or not idempotency_key.strip():
            raise FulfillmentNotActionable(
                "allocation id and idempotency key must not be blank"
            )
        if not priority_order_ids or len(set(priority_order_ids)) != len(
            priority_order_ids
        ):
            raise FulfillmentNotActionable(
                "priority_order_ids must be non-empty and unique"
            )

        cache_key = (by_actor, idempotency_key)
        requested = (
            allocation_id,
            merchant_id,
            sku_id,
            priority_order_ids,
            original_actor,
        )
        with self._lock:
            cached = self._allocation_idempotency_cache.get(cache_key)
            if cached is not None:
                prior = (
                    cached.allocation_id,
                    cached.merchant_id,
                    cached.sku_id,
                    cached.priority_order_ids,
                    cached.original_actor,
                )
                if prior != requested:
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was reused for another allocation"
                    )
                return cached.outcome

            fulfillment_table = cast(
                "FulfillmentTable", self._tables["fulfillments"]
            )
            durable_rows = fulfillment_table.by_operation_key(
                original_actor, idempotency_key
            )
            if durable_rows:
                order_table = cast("OrderTable", self._tables["orders"])
                durable_rows = tuple(
                    sorted(
                        durable_rows,
                        key=lambda row: (
                            cast(
                                "Order",
                                order_table.read(row.order_id, caller=None),
                            ).request_order,
                            str(row.buyer_id),
                            str(row.order_id),
                        ),
                    )
                )
                durable_priority = tuple(row.order_id for row in durable_rows)
                durable_allocation_id = durable_rows[0].allocation_id
                prior = (
                    durable_allocation_id,
                    durable_rows[0].merchant_id,
                    durable_rows[0].sku_id,
                    durable_priority,
                    str(durable_rows[0].created_by),
                )
                if prior != requested or any(
                    row.allocation_id != durable_allocation_id
                    or row.merchant_id != merchant_id
                    or row.sku_id != sku_id
                    for row in durable_rows
                ):
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was reused for another allocation"
                    )
                return AllocationBatch(
                    allocation_id=allocation_id,
                    merchant_id=merchant_id,
                    sku_id=sku_id,
                    priority_order_ids=durable_priority,
                    allocations=durable_rows,
                    created_by=AgentId(original_actor),
                    idempotency_key=idempotency_key,
                )
            if fulfillment_table.by_allocation_id(allocation_id) or any(
                record.allocation_id == allocation_id
                for record in self._allocation_idempotency_cache.values()
            ):
                raise IdempotencyConflict(
                    f"allocation id {allocation_id!r} already exists"
                )

            order_table = cast("OrderTable", self._tables["orders"])
            eligible = order_table.eligible_for_allocation(
                merchant_id=str(merchant_id),
                sku_id=str(sku_id),
                states=frozenset(state.value for state in _SETTLEABLE),
            )
            authoritative = tuple(
                row.order_id for row in eligible
            )
            if priority_order_ids != authoritative:
                raise FulfillmentNotActionable(
                    "priority_order_ids do not cover authoritative pending orders in priority order"
                )
            if not eligible:
                raise FulfillmentNotActionable("allocation has no eligible orders")

            inventory = cast(
                "InventoryRow | None",
                self._tables["inventory"].read(sku_id, caller=None),
            )
            listing = cast(
                "Listing | None",
                self._tables["catalog"].read(sku_id, caller=None),
            )
            if inventory is None or listing is None:
                raise OutOfStock(f"unknown allocation sku {sku_id}")
            if (
                inventory.merchant_id != merchant_id
                or listing.merchant_id != merchant_id
            ):
                raise FulfillmentNotActionable(
                    "allocation catalog/inventory is not owned by the merchant"
                )
            if any(
                self._tables["fulfillments"].read(order_id, caller=None) is not None
                for order_id in priority_order_ids
            ):
                raise FulfillmentNotActionable(
                    "an allocation order already has a fulfillment record"
                )

            remaining = _available_qty(inventory)
            event_tick = self._logical_time + 1
            allocations: list[FulfillmentAllocation] = []
            receipts: list[Receipt] = []
            persisted_orders: list[tuple[Order, Order]] = []
            timelines: list[tuple[OrderTimeline | None, OrderTimeline]] = []
            total_fulfilled = 0
            by_id = {order.order_id: order for order in eligible}
            for order_id in priority_order_ids:
                order = by_id[order_id]
                fulfilled = min(order.qty, remaining)
                backordered = order.qty - fulfilled
                remaining -= fulfilled
                total_fulfilled += fulfilled
                per_order_key = f"{idempotency_key}:{order.order_id}"
                receipt = None
                if fulfilled > 0:
                    receipt = Receipt(
                        txn_id=TxnId(
                            f"allocation:{allocation_id}:{order.order_id}"
                        ),
                        ts=f"world-tick:{event_tick}",
                        order_id=order.order_id,
                        buyer_id=order.buyer_id,
                        merchant_id=order.merchant_id,
                        sku_id=order.sku_id,
                        qty=fulfilled,
                        price=order.agreed_price,
                        idempotency_key=per_order_key,
                    )
                    if self._tables["ledger"].read(
                        receipt.txn_id, caller=None
                    ) is not None:
                        raise WriteNotAuthorized(
                            f"ledger row already exists: {receipt.txn_id}"
                        )
                    receipts.append(receipt)
                allocation = FulfillmentAllocation(
                    order_id=order.order_id,
                    buyer_id=order.buyer_id,
                    merchant_id=order.merchant_id,
                    sku_id=order.sku_id,
                    requested_qty=order.qty,
                    fulfilled_qty=fulfilled,
                    backordered_qty=backordered,
                    receipt_txn_id=receipt.txn_id if receipt is not None else None,
                    created_by=AgentId(original_actor),
                    idempotency_key=per_order_key,
                    allocation_id=allocation_id,
                )
                allocations.append(allocation)
                target = (
                    OrderState.BACKORDERED
                    if fulfilled == 0
                    else OrderState.PARTIALLY_SETTLED
                    if backordered > 0
                    else OrderState.SETTLED
                )
                persisted_orders.append((order, replace(order, state=target)))
                if fulfilled > 0:
                    before_timeline = cast(
                        "OrderTimeline | None",
                        self._tables["order_timelines"].read(
                            order.order_id, caller=None
                        ),
                    )
                    if before_timeline is not None:
                        raise FulfillmentNotActionable(
                            f"order {order.order_id} already has timeline evidence"
                        )
                    timelines.append((
                        before_timeline,
                        OrderTimeline(
                            order_id=order.order_id,
                            buyer_id=order.buyer_id,
                            merchant_id=order.merchant_id,
                            settled_at_tick=event_tick,
                            return_window_ticks=_captured_return_window(listing),
                        ),
                    ))

            inventory_after = _reserve_inventory(inventory, qty=total_fulfilled)
            batch = AllocationBatch(
                allocation_id=allocation_id,
                merchant_id=merchant_id,
                sku_id=sku_id,
                priority_order_ids=priority_order_ids,
                allocations=tuple(allocations),
                created_by=AgentId(original_actor),
                idempotency_key=idempotency_key,
            )
            writes: list[tuple[str, str, str, Any, Any]] = []
            for before_order, after_order in persisted_orders:
                writes.append((
                    "orders",
                    str(before_order.order_id),
                    "update",
                    before_order,
                    after_order,
                ))
            if total_fulfilled:
                writes.append((
                    "inventory",
                    str(sku_id),
                    "update",
                    inventory,
                    inventory_after,
                ))
            receipt_by_order = {row.order_id: row for row in receipts}
            for allocation in allocations:
                receipt = receipt_by_order.get(allocation.order_id)
                if receipt is not None:
                    writes.append((
                        "ledger",
                        str(receipt.txn_id),
                        "create",
                        None,
                        receipt,
                    ))
                writes.append((
                    "fulfillments",
                    str(allocation.order_id),
                    "create",
                    None,
                    allocation,
                ))
            for _, timeline in timelines:
                writes.append((
                    "order_timelines",
                    str(timeline.order_id),
                    "create",
                    None,
                    timeline,
                ))
            writes.append((
                "logical_time",
                "world",
                "update",
                self._logical_time,
                event_tick,
            ))

            try:
                for before_order, after_order in persisted_orders:
                    self.write(
                        "orders",
                        before_order.order_id,
                        after_order,
                        by_action="world.update_order_status",
                        _record_commit=False,
                    )
                if total_fulfilled:
                    self.write(
                        "inventory",
                        sku_id,
                        inventory_after,
                        by_action="world.update_inventory",
                        _record_commit=False,
                    )
                for receipt in receipts:
                    self.write(
                        "ledger",
                        receipt.txn_id,
                        receipt,
                        by_action="world.update_ledger",
                        _record_commit=False,
                    )
                for allocation in allocations:
                    self.write(
                        "fulfillments",
                        allocation.order_id,
                        allocation,
                        by_action="world.record_fulfillment",
                        _record_commit=False,
                    )
                for _, timeline in timelines:
                    self.write(
                        "order_timelines",
                        timeline.order_id,
                        timeline,
                        by_action="world.record_order_timeline",
                        _record_commit=False,
                    )
            except BaseException:
                for before_order, _ in persisted_orders:
                    _restore_row(
                        self._tables["orders"], before_order.order_id, before_order
                    )
                _restore_row(self._tables["inventory"], sku_id, inventory)
                for receipt in receipts:
                    _restore_row(self._tables["ledger"], receipt.txn_id, None)
                for allocation in allocations:
                    _restore_row(
                        self._tables["fulfillments"], allocation.order_id, None
                    )
                for before_timeline, timeline in timelines:
                    _restore_row(
                        self._tables["order_timelines"],
                        timeline.order_id,
                        before_timeline,
                    )
                raise

            self._logical_time = event_tick
            self._allocation_idempotency_cache[cache_key] = (
                _AllocationIdempotencyRecord(
                    allocation_id,
                    merchant_id,
                    sku_id,
                    priority_order_ids,
                    original_actor,
                    batch,
                )
            )
            self._append_transaction_commit(
                _make_diff(
                    "allocate_orders_atomic",
                    allocation_id,
                    writes,
                    (
                        "atomic",
                        "merchant-owner",
                        "priority-order",
                        "no-oversell",
                        "inventory-conservation",
                        "requested-equals-fulfilled-plus-backordered",
                        "idempotent",
                    ),
                ),
                authority_action="world.allocate_orders_atomic",
                actor_id=by_actor,
                idempotency_key=idempotency_key,
            )
            return batch

    def record_shipment_status(
        self,
        *,
        shipment_id: ShipmentId,
        event_id: str,
        status: ShipmentStatus,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Shipment:
        """Append one trusted logistics event to authoritative shipment history."""

        if by_actor != "platform:fulfillment":
            raise LifecycleAuthorizationError(
                f"only platform:fulfillment may record shipment status; got {by_actor!r}"
            )
        if original_actor != "runtime:logistics":
            raise LifecycleAuthorizationError(
                "shipment status original_actor must be runtime:logistics"
            )
        if not str(shipment_id).strip() or not event_id.strip() or not idempotency_key.strip():
            raise ShipmentNotActionable(
                "shipment id, event id, and idempotency key must not be blank"
            )
        cache_key = (by_actor, idempotency_key)
        with self._lock:
            cached = self._shipment_event_idempotency_cache.get(cache_key)
            if cached is not None:
                requested = (shipment_id, event_id, status, original_actor)
                prior = (
                    cached.shipment_id,
                    cached.event_id,
                    cached.status,
                    cached.original_actor,
                )
                if requested != prior:
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was reused for another shipment event"
                    )
                return cached.outcome

            shipment_table = cast("ShipmentTable", self._tables["shipments"])
            durable = shipment_table.event_operation(idempotency_key)
            if durable is not None:
                persisted, durable_event = durable
                requested = (shipment_id, event_id, status, original_actor)
                prior = (
                    persisted.shipment_id,
                    durable_event.event_id,
                    durable_event.status,
                    "runtime:logistics",
                )
                if requested != prior:
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was reused for another shipment event"
                    )
                position = persisted.status_history.index(durable_event)
                prefix = persisted.status_history[: position + 1]
                event_version = durable_event.shipment_version or position + 1
                resolution_was_present = (
                    persisted.resolution_version is not None
                    and persisted.resolution_version <= event_version
                )
                return replace(
                    persisted,
                    status=durable_event.status,
                    status_history=prefix,
                    resolution=(
                        persisted.resolution if resolution_was_present else None
                    ),
                    replacement_sku_id=(
                        persisted.replacement_sku_id
                        if resolution_was_present
                        else None
                    ),
                    version=event_version,
                    resolution_idempotency_key=(
                        persisted.resolution_idempotency_key
                        if resolution_was_present
                        else None
                    ),
                    resolved_by=(
                        persisted.resolved_by if resolution_was_present else None
                    ),
                    resolution_version=(
                        persisted.resolution_version
                        if resolution_was_present
                        else None
                    ),
                    resolution_history_length=(
                        persisted.resolution_history_length
                        if resolution_was_present
                        else None
                    ),
                )

            shipment = cast(
                "Shipment | None",
                self._tables["shipments"].read(shipment_id, caller=None),
            )
            if shipment is None:
                raise ShipmentNotActionable(
                    f"shipment {shipment_id} does not exist"
                )
            for event in shipment.status_history:
                if event.event_id != event_id:
                    continue
                if event.status != status:
                    raise IdempotencyConflict(
                        f"shipment event id {event_id!r} was reused with another status"
                    )
                self._shipment_event_idempotency_cache[cache_key] = (
                    _ShipmentEventIdempotencyRecord(
                        shipment_id, event_id, status, original_actor, shipment
                    )
                )
                return shipment
            if shipment.resolution in {
                ShipmentResolution.REPLACEMENT,
                ShipmentResolution.REFUND,
            }:
                raise ShipmentNotActionable(
                    "a replaced or refunded shipment cannot receive more status events"
                )
            if status not in _SHIPMENT_TRANSITIONS[shipment.status]:
                raise ShipmentNotActionable(
                    f"shipment cannot transition from {shipment.status.value!r} to {status.value!r}"
                )
            event_tick = self._logical_time + 1
            event = ShipmentStatusEvent(
                event_id,
                status,
                event_tick,
                idempotency_key=idempotency_key,
                shipment_version=shipment.version + 1,
            )
            updated = replace(
                shipment,
                status=status,
                status_history=shipment.status_history + (event,),
                version=shipment.version + 1,
            )
            try:
                self.write(
                    "shipments",
                    shipment_id,
                    updated,
                    by_action="world.record_shipment",
                    _record_commit=False,
                )
            except BaseException:
                _restore_row(self._tables["shipments"], shipment_id, shipment)
                raise
            self._logical_time = event_tick
            self._shipment_event_idempotency_cache[cache_key] = (
                _ShipmentEventIdempotencyRecord(
                    shipment_id, event_id, status, original_actor, updated
                )
            )
            self._append_transaction_commit(
                _make_diff(
                    "record_shipment_status",
                    shipment_id,
                    [
                        ("shipments", str(shipment_id), "update", shipment, updated),
                        (
                            "logical_time",
                            "world",
                            "update",
                            event_tick - 1,
                            event_tick,
                        ),
                    ],
                    (
                        "atomic",
                        "runtime-logistics-only",
                        "append-only-history",
                        "shipment-version-monotonic",
                        "idempotent-event-id",
                    ),
                ),
                authority_action="world.record_shipment_status",
                actor_id=by_actor,
                idempotency_key=idempotency_key,
            )
            return updated

    def resolve_shipment(
        self,
        *,
        shipment_id: ShipmentId,
        resolution: ShipmentResolution,
        replacement_sku_id: SkuId | None,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Shipment:
        """Atomically commit wait, replacement, or lost-shipment refund."""

        if by_actor != "platform:fulfillment":
            raise LifecycleAuthorizationError(
                f"only platform:fulfillment may resolve shipments; got {by_actor!r}"
            )
        if not str(shipment_id).strip() or not idempotency_key.strip():
            raise ShipmentNotActionable(
                "shipment id and idempotency key must not be blank"
            )
        cache_key = (by_actor, idempotency_key)
        with self._lock:
            cached = self._shipment_resolution_idempotency_cache.get(cache_key)
            if cached is not None:
                requested = (
                    shipment_id,
                    resolution,
                    replacement_sku_id,
                    original_actor,
                )
                prior = (
                    cached.shipment_id,
                    cached.resolution,
                    cached.replacement_sku_id,
                    cached.original_actor,
                )
                if requested != prior:
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was reused for another shipment resolution"
                    )
                return cached.outcome


            shipment_table = cast("ShipmentTable", self._tables["shipments"])
            durable = shipment_table.resolution_operation(idempotency_key)
            if durable is not None:
                requested = (
                    shipment_id,
                    resolution,
                    replacement_sku_id,
                    original_actor,
                )
                prior = (
                    durable.shipment_id,
                    durable.resolution,
                    durable.replacement_sku_id,
                    None if durable.resolved_by is None else str(durable.resolved_by),
                )
                if requested != prior:
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was reused for another shipment resolution"
                    )
                length = durable.resolution_history_length or len(
                    durable.status_history
                )
                prefix = durable.status_history[:length]
                return replace(
                    durable,
                    status=prefix[-1].status,
                    status_history=prefix,
                    version=durable.resolution_version or durable.version,
                )

            shipment = cast(
                "Shipment | None",
                self._tables["shipments"].read(shipment_id, caller=None),
            )
            if shipment is None:
                raise ShipmentNotActionable(
                    f"shipment {shipment_id} does not exist"
                )
            if original_actor not in {
                str(shipment.buyer_id),
                str(shipment.merchant_id),
            }:
                raise LifecycleAuthorizationError(
                    f"actor {original_actor!r} is not a party to shipment {shipment_id}"
                )
            if shipment.resolution is not None:
                if (
                    shipment.resolution == resolution
                    and shipment.replacement_sku_id == replacement_sku_id
                ):
                    self._shipment_resolution_idempotency_cache[cache_key] = (
                        _ShipmentResolutionIdempotencyRecord(
                            shipment_id,
                            resolution,
                            replacement_sku_id,
                            original_actor,
                            shipment,
                        )
                    )
                    return shipment
                raise ShipmentNotActionable(
                    f"shipment {shipment_id} already has a different resolution"
                )
            if resolution == ShipmentResolution.WAIT:
                if shipment.status not in {
                    ShipmentStatus.DELAYED,
                    ShipmentStatus.MISSING_SCAN,
                    ShipmentStatus.LOST,
                }:
                    raise ShipmentNotActionable(
                        "wait is valid only for an exceptional shipment"
                    )
                if replacement_sku_id is not None:
                    raise ShipmentNotActionable(
                        "wait resolution cannot name a replacement sku"
                    )
            else:
                if shipment.status != ShipmentStatus.LOST:
                    raise ShipmentNotActionable(
                        "replacement/refund requires authoritative lost status"
                    )
                if resolution == ShipmentResolution.REPLACEMENT:
                    if replacement_sku_id is None:
                        raise ShipmentNotActionable(
                            "replacement resolution requires replacement_sku_id"
                        )
                elif replacement_sku_id is not None:
                    raise ShipmentNotActionable(
                        "refund resolution cannot name a replacement sku"
                    )

            order = cast(
                "Order | None",
                self._tables["orders"].read(shipment.order_id, caller=None),
            )
            if order is None or (
                order.buyer_id != shipment.buyer_id
                or order.merchant_id != shipment.merchant_id
                or order.sku_id != shipment.original_sku_id
            ):
                raise ShipmentNotActionable(
                    "shipment does not match its authoritative order"
                )
            if order.state != OrderState.DISPATCHED:
                raise ShipmentNotActionable(
                    f"shipment order must be dispatched; found {order.state.value!r}"
                )
            paid_qty = _paid_quantity(self._tables["fulfillments"], order)
            event_tick = self._logical_time + 1
            updated = replace(
                shipment,
                resolution=resolution,
                replacement_sku_id=replacement_sku_id,
                version=shipment.version + 1,
                resolution_idempotency_key=idempotency_key,
                resolved_by=AgentId(original_actor),
                resolution_version=shipment.version + 1,
                resolution_history_length=len(shipment.status_history),
            )
            writes: list[tuple[str, str, str, Any, Any]] = [(
                "shipments", str(shipment_id), "update", shipment, updated
            )]
            original_inventory: InventoryRow | None = None
            original_after: InventoryRow | None = None
            replacement_inventory: InventoryRow | None = None
            replacement_after: InventoryRow | None = None
            refunded_order: Order | None = None
            refund_receipt: Receipt | None = None
            timeline_before: OrderTimeline | None = None
            timeline_after: OrderTimeline | None = None

            if resolution in {
                ShipmentResolution.REPLACEMENT,
                ShipmentResolution.REFUND,
            }:
                original_inventory = cast(
                    "InventoryRow | None",
                    self._tables["inventory"].read(
                        shipment.original_sku_id, caller=None
                    ),
                )
                if (
                    original_inventory is None
                    or original_inventory.merchant_id != shipment.merchant_id
                    or original_inventory.qty_reserved < paid_qty
                    or original_inventory.qty_available < paid_qty
                ):
                    raise ShipmentNotActionable(
                        "lost shipment lacks its authoritative inventory reservation"
                    )
                original_after = replace(
                    original_inventory,
                    qty_available=original_inventory.qty_available - paid_qty,
                    qty_reserved=original_inventory.qty_reserved - paid_qty,
                    version=original_inventory.version + 1,
                )

            if resolution == ShipmentResolution.REPLACEMENT:
                assert replacement_sku_id is not None
                replacement_inventory = cast(
                    "InventoryRow | None",
                    self._tables["inventory"].read(replacement_sku_id, caller=None),
                )
                replacement_listing = cast(
                    "Listing | None",
                    self._tables["catalog"].read(replacement_sku_id, caller=None),
                )
                if (
                    replacement_inventory is None
                    or replacement_listing is None
                    or replacement_inventory.merchant_id != shipment.merchant_id
                    or replacement_listing.merchant_id != shipment.merchant_id
                ):
                    raise ShipmentNotActionable(
                        "replacement sku is not owned by the shipment merchant"
                    )
                if replacement_sku_id == shipment.original_sku_id:
                    assert original_after is not None
                    if _available_qty(original_after) < paid_qty:
                        raise OutOfStock(
                            f"insufficient replacement inventory for {replacement_sku_id}"
                        )
                    replacement_after = replace(
                        original_after,
                        qty_reserved=original_after.qty_reserved + paid_qty,
                        version=original_inventory.version + 1,
                    )
                    original_after = replacement_after
                    replacement_inventory = original_inventory
                else:
                    if _available_qty(replacement_inventory) < paid_qty:
                        raise OutOfStock(
                            f"insufficient replacement inventory for {replacement_sku_id}"
                        )
                    replacement_after = replace(
                        replacement_inventory,
                        qty_reserved=replacement_inventory.qty_reserved + paid_qty,
                        version=replacement_inventory.version + 1,
                    )
                assert original_inventory is not None and original_after is not None
                writes.append((
                    "inventory",
                    str(shipment.original_sku_id),
                    "update",
                    original_inventory,
                    original_after,
                ))
                if replacement_sku_id != shipment.original_sku_id:
                    assert replacement_inventory is not None and replacement_after is not None
                    writes.append((
                        "inventory",
                        str(replacement_sku_id),
                        "update",
                        replacement_inventory,
                        replacement_after,
                    ))

            if resolution == ShipmentResolution.REFUND:
                assert original_inventory is not None and original_after is not None
                refunded_order = replace(order, state=OrderState.REFUNDED)
                refund_receipt = Receipt(
                    txn_id=TxnId(f"refund:shipment:{shipment_id}"),
                    ts=f"world-tick:{event_tick}",
                    order_id=order.order_id,
                    buyer_id=order.buyer_id,
                    merchant_id=order.merchant_id,
                    sku_id=order.sku_id,
                    qty=paid_qty,
                    price=order.agreed_price,
                    idempotency_key=idempotency_key,
                    effect="refund",
                )
                if self._tables["ledger"].read(
                    refund_receipt.txn_id, caller=None
                ) is not None:
                    raise WriteNotAuthorized(
                        f"ledger row already exists: {refund_receipt.txn_id}"
                    )
                timeline_before = cast(
                    "OrderTimeline | None",
                    self._tables["order_timelines"].read(order.order_id, caller=None),
                )
                timeline_after = (
                    OrderTimeline(
                        order_id=order.order_id,
                        buyer_id=order.buyer_id,
                        merchant_id=order.merchant_id,
                        refunded_at_tick=event_tick,
                    )
                    if timeline_before is None
                    else replace(timeline_before, refunded_at_tick=event_tick)
                )
                writes.extend([
                    (
                        "inventory",
                        str(shipment.original_sku_id),
                        "update",
                        original_inventory,
                        original_after,
                    ),
                    ("orders", str(order.order_id), "update", order, refunded_order),
                    (
                        "ledger",
                        str(refund_receipt.txn_id),
                        "create",
                        None,
                        refund_receipt,
                    ),
                    (
                        "order_timelines",
                        str(order.order_id),
                        "update" if timeline_before is not None else "create",
                        timeline_before,
                        timeline_after,
                    ),
                ])

            writes.append((
                "logical_time",
                "world",
                "update",
                event_tick - 1,
                event_tick,
            ))
            try:
                self.write(
                    "shipments",
                    shipment_id,
                    updated,
                    by_action="world.resolve_shipment",
                    _record_commit=False,
                )
                if original_inventory is not None and original_after is not None:
                    self.write(
                        "inventory",
                        original_inventory.sku_id,
                        original_after,
                        by_action="world.update_inventory",
                        _record_commit=False,
                    )
                if (
                    replacement_inventory is not None
                    and replacement_after is not None
                    and replacement_inventory.sku_id != shipment.original_sku_id
                ):
                    self.write(
                        "inventory",
                        replacement_inventory.sku_id,
                        replacement_after,
                        by_action="world.update_inventory",
                        _record_commit=False,
                    )
                if refunded_order is not None:
                    self.write(
                        "orders",
                        refunded_order.order_id,
                        refunded_order,
                        by_action="world.update_order_status",
                        _record_commit=False,
                    )
                if refund_receipt is not None:
                    self.write(
                        "ledger",
                        refund_receipt.txn_id,
                        refund_receipt,
                        by_action="world.update_ledger",
                        _record_commit=False,
                    )
                if timeline_after is not None:
                    self.write(
                        "order_timelines",
                        timeline_after.order_id,
                        timeline_after,
                        by_action="world.record_order_timeline",
                        _record_commit=False,
                    )
            except BaseException:
                _restore_row(self._tables["shipments"], shipment_id, shipment)
                if original_inventory is not None:
                    _restore_row(
                        self._tables["inventory"],
                        original_inventory.sku_id,
                        original_inventory,
                    )
                if replacement_inventory is not None:
                    _restore_row(
                        self._tables["inventory"],
                        replacement_inventory.sku_id,
                        replacement_inventory,
                    )
                _restore_row(self._tables["orders"], order.order_id, order)
                if refund_receipt is not None:
                    _restore_row(
                        self._tables["ledger"], refund_receipt.txn_id, None
                    )
                _restore_row(
                    self._tables["order_timelines"],
                    order.order_id,
                    timeline_before,
                )
                raise

            self._logical_time = event_tick
            self._shipment_resolution_idempotency_cache[cache_key] = (
                _ShipmentResolutionIdempotencyRecord(
                    shipment_id,
                    resolution,
                    replacement_sku_id,
                    original_actor,
                    updated,
                )
            )
            self._append_transaction_commit(
                _make_diff(
                    "resolve_shipment",
                    shipment_id,
                    writes,
                    (
                        "atomic",
                        "shipment-party",
                        "lost-goods-consumed",
                        "replacement-no-oversell",
                        "single-resolution",
                        "idempotent",
                    ),
                ),
                authority_action="world.resolve_shipment",
                actor_id=by_actor,
                idempotency_key=idempotency_key,
            )
            return updated

    def _fulfillment_replay(
        self,
        *,
        allocation: FulfillmentAllocation,
        receipt: Receipt | None,
        order: Order,
    ) -> FulfillmentAllocation | None:
        """Classify an exact allocation retry versus either conflict type."""
        for _, prior in self._tables["fulfillments"].all():
            if (
                str(prior.created_by) == str(allocation.created_by)
                and prior.idempotency_key == allocation.idempotency_key
            ):
                prior_receipt = (
                    None
                    if prior.receipt_txn_id is None
                    else self._tables["ledger"].read(
                        prior.receipt_txn_id,
                        caller=None,
                    )
                )
                persisted_order = self._tables["orders"].read(
                    prior.order_id,
                    caller=None,
                )
                if (
                    prior == allocation
                    and prior_receipt == receipt
                    and isinstance(persisted_order, Order)
                    and _order_identity_matches(persisted_order, order)
                ):
                    return cast("FulfillmentAllocation", prior)
                raise IdempotencyConflict(
                    f"idempotency key {allocation.idempotency_key!r} was already "
                    "used for a different fulfillment request"
                )
        existing = self._tables["fulfillments"].read(
            allocation.order_id,
            caller=None,
        )
        if existing is not None:
            raise FulfillmentNotActionable(
                f"order {allocation.order_id} already has a fulfillment allocation"
            )
        return None

    def refund_order(
        self,
        *,
        order: Order,
        refund_receipt: Receipt,
        by_role: str,
        idempotency_key: str,
    ) -> Receipt:
        """Atomically refund a settled ``order``: mark it REFUNDED, RESTOCK
        inventory (reverse the settle reservation), and append the reversing
        ledger ``refund_receipt`` — ONE transaction under the world lock,
        idempotent on ``(by_role, idempotency_key)``. The settle-family inverse
        of :meth:`settle_order`; same atomic-multi-row + allowlist discipline.

        Classified by the order's PERSISTED state (allowlist + default-deny):
          * already REFUNDED -> idempotent replay of the prior refund receipt
            (the REFUNDED order + its refund ledger entry are the durable signal,
            so a same-order/new-key re-refund never double-restocks).
          * NOT on the refundable allow-list (PROPOSED / ACCEPTED never paid,
            CANCELLED, or any future state) -> refuse before any write.
          * a refundable state (SETTLED / DISPATCHED / RETURNED) -> proceed.

        Raises:
            OrderNotRefundable: order missing, or not in a refundable state.
        """
        refund_receipt = replace(refund_receipt, effect="refund")
        cache_key = (by_role, idempotency_key)
        with self._lock:
            cached = self._idempotency_cache.get(cache_key)
            if cached is not None:
                if (
                    cached.operation != "refund"
                    or cached.order != order
                    or cached.request_receipt != refund_receipt
                ):
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was already used "
                        "for a different request"
                    )
                return cached.outcome
            existing = self.read("orders", order.order_id, caller=by_role)
            if existing is None:
                raise OrderNotRefundable(
                    f"order {order.order_id} does not exist — nothing to refund"
                )
            validate_persisted_order_identity(existing, order)
            paid_qty = _paid_quantity(self._tables["fulfillments"], existing)
            validate_transaction_identity(
                existing,
                refund_receipt,
                None,
                expected_qty=paid_qty,
                expected_effect="refund",
            )
            if existing.state == OrderState.REFUNDED:
                prior = self._refund_receipt_for_order(order.order_id)
                if prior is None:
                    raise WorldError(
                        f"order {order.order_id} is REFUNDED but has no refund receipt "
                        "— refusing to re-refund"
                    )
                self._idempotency_cache[cache_key] = _IdempotencyRecord(
                    "refund", order, refund_receipt, prior
                )
                return prior
            if existing.state not in _REFUNDABLE:
                raise OrderNotRefundable(
                    f"order {order.order_id} in state {existing.state.value!r} "
                    "is not refundable (only a settled/dispatched/returned order "
                    "can be refunded)"
                )
            if _exchange_for_replacement(
                self._tables["exchanges"],
                existing.order_id,
            ) is not None:
                raise OrderNotRefundable(
                    f"replacement order {existing.order_id} has no independent "
                    "payment to refund"
                )
            inventory = self.read("inventory", existing.sku_id, caller=by_role)
            validate_transaction_identity(
                existing,
                refund_receipt,
                inventory,
                expected_qty=paid_qty,
                expected_effect="refund",
            )
            validate_listing_owner(
                existing, self.read("catalog", existing.sku_id, caller=by_role)
            )
            if self.read("ledger", refund_receipt.txn_id, caller=by_role) is not None:
                raise WriteNotAuthorized(
                    f"ledger row already exists: {refund_receipt.txn_id}"
                )
            timeline_before = cast(
                "OrderTimeline | None",
                self._tables["order_timelines"].read(existing.order_id, caller=None),
            )
            event_tick = self._logical_time + 1
            _enforce_return_window(timeline_before, event_tick=event_tick)
            payment_history = cast(
                PaymentStateTable, self._tables["payment_states"]
            ).history_for_order(str(existing.order_id))
            previous_payment = payment_history[-1] if payment_history else None
            refunded_payment: PaymentStateRecord | None = None
            payment_key: str | None = None
            if previous_payment is not None:
                if previous_payment.state not in {
                    "captured",
                    "partially_refunded",
                }:
                    raise WorldError(
                        f"order {existing.order_id} has payment state "
                        f"{previous_payment.state!r} before refund"
                    )
                refunded_payment = derive_payment_resolution(
                    existing,
                    previous_payment,
                    outcome="refunded",
                    refund_receipt=refund_receipt,
                    original_actor="platform:psp",
                    server_tick=event_tick,
                    idempotency_key=f"refund-resolution:{idempotency_key}",
                )
                payment_key = payment_state_key(refunded_payment)
                fully_refunded = refunded_payment.state == "refunded"
            else:
                # Legacy settled rows without first-class payment history keep
                # their historical full-refund behavior.  Partial refunds
                # require the cumulative PaymentState v2 authority chain.
                if refund_receipt.price != existing.agreed_price:
                    raise OrderNotRefundable(
                        "partial refund requires first-class captured payment state"
                    )
                fully_refunded = True
            refunded_order = (
                replace(existing, state=OrderState.REFUNDED)
                if fully_refunded
                else existing
            )
            timeline = (
                None
                if not fully_refunded
                else (
                    OrderTimeline(
                        order_id=existing.order_id,
                        buyer_id=existing.buyer_id,
                        merchant_id=existing.merchant_id,
                        return_authorized_at_tick=event_tick,
                        refunded_at_tick=event_tick,
                    )
                    if timeline_before is None
                    else replace(
                        timeline_before,
                        return_authorized_at_tick=event_tick,
                        refunded_at_tick=event_tick,
                    )
                )
            )
            writes: list[tuple[str, str, str, Any, Any]] = []
            if fully_refunded:
                writes.append(
                    ("orders", str(order.order_id), "update", existing, refunded_order)
                )
            restocked = None
            if fully_refunded and inventory is not None:
                if (
                    isinstance(inventory, InventoryRow)
                    and inventory.qty_reserved < paid_qty
                ):
                    raise OrderNotRefundable(
                        "inventory reservation is below the fulfilled quantity"
                    )
                restocked = _restock_inventory(inventory, qty=paid_qty)
                writes.append(
                    ("inventory", str(existing.sku_id), "update", inventory, restocked)
                )
            writes.append(("ledger", str(refund_receipt.txn_id), "create", None, refund_receipt))
            if refunded_payment is not None and payment_key is not None:
                writes.append(
                    (
                        "payment_states",
                        payment_key,
                        "create",
                        None,
                        refunded_payment,
                    )
                )
            if timeline is not None:
                writes.append((
                    "order_timelines",
                    str(existing.order_id),
                    "update" if timeline_before is not None else "create",
                    timeline_before,
                    timeline,
                ))
            writes.append((
                "logical_time",
                "world",
                "update",
                self._logical_time,
                event_tick,
            ))
            diff = _make_diff(
                "refund",
                order.order_id,
                writes,
                (
                    "atomic",
                    "allowlist:REFUNDABLE",
                    "idempotent",
                    "world-clock",
                    *(
                        (
                            "captured-payment-refunded"
                            if fully_refunded
                            else "captured-payment-partially-refunded",
                        )
                        if refunded_payment is not None
                        else ()
                    ),
                ),
            )
            before_tick = self._logical_time
            diff_cursor = len(self._txn_diffs)
            commit_cursor = len(self._commit_journal)
            prior_revisions = dict(self._order_state_revisions)
            try:
                if fully_refunded:
                    self.write(
                        "orders",
                        order.order_id,
                        refunded_order,
                        by_action="world.update_order_status",
                        _record_commit=False,
                    )
                if restocked is not None:
                    self.write(
                        "inventory",
                        existing.sku_id,
                        restocked,
                        by_action="world.update_inventory",
                        _record_commit=False,
                    )
                self.write(
                    "ledger",
                    refund_receipt.txn_id,
                    refund_receipt,
                    by_action="world.update_ledger",
                    _record_commit=False,
                )
                if refunded_payment is not None and payment_key is not None:
                    self.write(
                        "payment_states",
                        payment_key,
                        refunded_payment,
                        by_action="world.refund_order",
                        _record_commit=False,
                    )
                if timeline is not None:
                    self.write(
                        "order_timelines",
                        existing.order_id,
                        timeline,
                        by_action="world.record_order_timeline",
                        _record_commit=False,
                    )
                self._logical_time = event_tick
                self._append_transaction_commit(
                    diff,
                    authority_action="world.refund_order",
                    actor_id=by_role,
                    idempotency_key=idempotency_key,
                )
                self._idempotency_cache[cache_key] = _IdempotencyRecord(
                    "refund", order, refund_receipt, refund_receipt
                )
            except BaseException:
                if fully_refunded:
                    _restore_row(self._tables["orders"], order.order_id, existing)
                if fully_refunded and inventory is not None:
                    _restore_row(self._tables["inventory"], existing.sku_id, inventory)
                _restore_row(self._tables["ledger"], refund_receipt.txn_id, None)
                if payment_key is not None:
                    _restore_row(self._tables["payment_states"], payment_key, None)
                if timeline is not None:
                    _restore_row(
                        self._tables["order_timelines"],
                        existing.order_id,
                        timeline_before,
                    )
                self._logical_time = before_tick
                self._order_state_revisions = prior_revisions
                del self._txn_diffs[diff_cursor:]
                del self._commit_journal[commit_cursor:]
                self._idempotency_cache.pop(cache_key, None)
                raise
            return refund_receipt

    def _refund_receipt_for_order(self, order_id: Any) -> "Receipt | None":
        """Return the authoritative reversing ledger receipt for an order."""
        for _, rcpt in self._tables["ledger"].all():
            if str(rcpt.order_id) == str(order_id) and rcpt.effect == "refund":
                return cast("Receipt", rcpt)
        return None

    def dispatch_order(self, *, order_id: OrderId, by_actor: str) -> Order:
        """Move a paid order from ``SETTLED`` to ``DISPATCHED``.

        Only the owning merchant (or a trusted platform service) may dispatch.
        Exact retries, including retries after the order progressed to RETURNED
        or REFUNDED, are non-regressing no-ops.
        """
        return self._transition_order(
            order_id=order_id,
            by_actor=by_actor,
            allowed_actor_sides=frozenset({"merchant"}),
            allowed_from=frozenset({
                OrderState.SETTLED,
                OrderState.PARTIALLY_SETTLED,
            }),
            replay_states=frozenset({
                OrderState.DISPATCHED,
                OrderState.RETURNED,
                OrderState.REFUNDED,
            }),
            target=OrderState.DISPATCHED,
            action="dispatch",
            timeline_field="dispatched_at_tick",
            create_shipment=True,
            create_packing=True,
        )

    def cancel_order(self, *, order_id: OrderId, by_actor: str) -> Order:
        """Cancel an unpaid order while it is PROPOSED or ACCEPTED.

        Either exact party may cancel; a third-party buyer or merchant cannot.
        Paid, dispatched, returned, and refunded orders are default-denied.
        """
        return self._transition_order(
            order_id=order_id,
            by_actor=by_actor,
            allowed_actor_sides=frozenset({"buyer", "merchant"}),
            allowed_from=frozenset({
                OrderState.PROPOSED,
                OrderState.ACCEPTED,
                OrderState.BACKORDERED,
            }),
            replay_states=frozenset({OrderState.CANCELLED}),
            target=OrderState.CANCELLED,
            action="cancel",
        )

    def mark_order_returned(self, *, order_id: OrderId, by_actor: str) -> Order:
        """Record physical return receipt: ``DISPATCHED`` -> ``RETURNED``.

        The owning merchant (or platform) confirms receipt. Inventory is not
        released here: the existing atomic refund operation remains the single
        place that releases the settlement reservation and appends the reversing
        ledger entry.
        """
        return self._transition_order(
            order_id=order_id,
            by_actor=by_actor,
            allowed_actor_sides=frozenset({"merchant"}),
            allowed_from=frozenset({OrderState.DISPATCHED}),
            replay_states=frozenset({OrderState.RETURNED, OrderState.REFUNDED}),
            target=OrderState.RETURNED,
            action="return",
            timeline_field="returned_at_tick",
        )

    def exchange_order(
        self,
        *,
        exchange_id: ExchangeId,
        original_order_id: OrderId,
        replacement_order: Order,
        by_actor: str,
        idempotency_key: str,
    ) -> Exchange:
        """Atomically replace one fully returned order without moving money.

        The replacement must be like-for-like in buyer, merchant, quantity,
        unit price, and currency. The operation releases the returned item's
        reservation, reserves the replacement SKU, marks the original
        ``EXCHANGED``, creates a dispatchable replacement order, and records the
        immutable link. It intentionally performs no ledger write.
        """
        with self._lock:
            original = cast(
                "Order | None",
                self._tables["orders"].read(original_order_id, caller=None),
            )
            if original is None:
                raise ExchangeNotActionable(
                    f"original order {original_order_id} does not exist"
                )
            _authorize_exchange_actor(original, by_actor)
            if replacement_order.state not in _SETTLEABLE:
                raise ExchangeNotActionable(
                    "replacement order must begin PROPOSED or ACCEPTED"
                )
            if not str(exchange_id).strip() or not idempotency_key.strip():
                raise ExchangeNotActionable(
                    "exchange id and idempotency key must not be blank"
                )
            qty = _paid_quantity(self._tables["fulfillments"], original)
            exchange = Exchange(
                exchange_id=exchange_id,
                original_order_id=original.order_id,
                replacement_order_id=replacement_order.order_id,
                buyer_id=original.buyer_id,
                merchant_id=original.merchant_id,
                original_sku_id=original.sku_id,
                replacement_sku_id=replacement_order.sku_id,
                qty=qty,
                created_by=AgentId(by_actor),
                idempotency_key=idempotency_key,
            )
            replay = self._exchange_replay(exchange, replacement_order)
            if replay is not None:
                return replay
            if original.state != OrderState.RETURNED:
                raise ExchangeNotActionable(
                    f"order {original.order_id} must be RETURNED before exchange; "
                    f"found {original.state.value!r}"
                )
            allocation = self._tables["fulfillments"].read(
                original.order_id,
                caller=None,
            )
            if allocation is not None and allocation.backordered_qty != 0:
                raise ExchangeNotActionable(
                    "an order with outstanding backordered units cannot be exchanged"
                )
            _validate_replacement_identity(original, replacement_order, qty=qty)
            if self._tables["orders"].read(
                replacement_order.order_id,
                caller=None,
            ) is not None:
                raise ExchangeNotActionable(
                    f"replacement order id {replacement_order.order_id} already exists"
                )

            original_listing = self._tables["catalog"].read(
                original.sku_id,
                caller=None,
            )
            replacement_listing = self._tables["catalog"].read(
                replacement_order.sku_id,
                caller=None,
            )
            _require_listing_owner(original, original_listing)
            _require_listing_owner(replacement_order, replacement_listing)
            original_inventory = self._tables["inventory"].read(
                original.sku_id,
                caller=None,
            )
            replacement_inventory = self._tables["inventory"].read(
                replacement_order.sku_id,
                caller=None,
            )
            if original_inventory is None or replacement_inventory is None:
                raise ExchangeNotActionable(
                    "both original and replacement SKUs require inventory rows"
                )
            _require_inventory_owner(original, original_inventory)
            _require_inventory_owner(replacement_order, replacement_inventory)
            if (
                isinstance(original_inventory, InventoryRow)
                and original_inventory.qty_reserved < qty
            ):
                raise ExchangeNotActionable(
                    "original inventory no longer holds the returned reservation"
                )

            inventory_writes: list[tuple[str, str, str, Any, Any]] = []
            inventory_after: dict[Any, Any] = {}
            if original.sku_id == replacement_order.sku_id:
                released = _restock_inventory(original_inventory, qty=qty)
                if _available_qty(released) < qty:
                    raise OutOfStock(
                        f"insufficient inventory for replacement sku {original.sku_id}"
                    )
                final_inventory = _reserve_inventory(released, qty=qty)
                if isinstance(final_inventory, InventoryRow):
                    final_inventory = replace(
                        final_inventory, version=original_inventory.version + 1
                    )
                inventory_after[original.sku_id] = final_inventory
                inventory_writes.append((
                    "inventory",
                    str(original.sku_id),
                    "update",
                    original_inventory,
                    final_inventory,
                ))
            else:
                if _available_qty(replacement_inventory) < qty:
                    raise OutOfStock(
                        "insufficient inventory for replacement sku "
                        f"{replacement_order.sku_id}"
                    )
                released = _restock_inventory(original_inventory, qty=qty)
                replacement_reserved = _reserve_inventory(
                    replacement_inventory,
                    qty=qty,
                )
                inventory_after[original.sku_id] = released
                inventory_after[replacement_order.sku_id] = replacement_reserved
                inventory_writes.extend((
                    (
                        "inventory",
                        str(original.sku_id),
                        "update",
                        original_inventory,
                        released,
                    ),
                    (
                        "inventory",
                        str(replacement_order.sku_id),
                        "update",
                        replacement_inventory,
                        replacement_reserved,
                    ),
                ))

            exchanged_original = replace(original, state=OrderState.EXCHANGED)
            settled_replacement = replace(
                replacement_order,
                state=OrderState.SETTLED,
            )
            writes: list[tuple[str, str, str, Any, Any]] = [
                (
                    "orders",
                    str(original.order_id),
                    "update",
                    original,
                    exchanged_original,
                ),
                (
                    "orders",
                    str(replacement_order.order_id),
                    "create",
                    None,
                    settled_replacement,
                ),
                *inventory_writes,
                (
                    "exchanges",
                    str(exchange.exchange_id),
                    "create",
                    None,
                    exchange,
                ),
            ]
            diff = _make_diff(
                "exchange",
                original.order_id,
                writes,
                (
                    "atomic",
                    "returned-original",
                    "like-for-like",
                    "no-ledger-write",
                    "no-oversell",
                    "idempotent",
                ),
            )
            try:
                self.write(
                    "orders",
                    original.order_id,
                    exchanged_original,
                    by_action="world.update_order_status",
                    _record_commit=False,
                )
                self.write(
                    "orders",
                    replacement_order.order_id,
                    settled_replacement,
                    by_action="world.create_order",
                    _record_commit=False,
                )
                for sku_id, value in inventory_after.items():
                    self.write(
                        "inventory",
                        sku_id,
                        value,
                        by_action="world.update_inventory",
                        _record_commit=False,
                    )
                self.write(
                    "exchanges",
                    exchange.exchange_id,
                    exchange,
                    by_action="world.record_exchange",
                    _record_commit=False,
                )
            except BaseException:
                _restore_row(
                    self._tables["orders"],
                    original.order_id,
                    original,
                )
                _restore_row(
                    self._tables["orders"],
                    replacement_order.order_id,
                    None,
                )
                _restore_row(
                    self._tables["inventory"],
                    original.sku_id,
                    original_inventory,
                )
                if original.sku_id != replacement_order.sku_id:
                    _restore_row(
                        self._tables["inventory"],
                        replacement_order.sku_id,
                        replacement_inventory,
                    )
                _restore_row(
                    self._tables["exchanges"],
                    exchange.exchange_id,
                    None,
                )
                raise
            self._append_transaction_commit(
                diff,
                authority_action="world.exchange_order",
                actor_id=by_actor,
                idempotency_key=idempotency_key,
            )
            return exchange

    def _exchange_replay(
        self,
        requested: Exchange,
        replacement_order: Order,
    ) -> Exchange | None:
        for _, prior in self._tables["exchanges"].all():
            if (
                str(prior.created_by) == str(requested.created_by)
                and prior.idempotency_key == requested.idempotency_key
            ):
                persisted_replacement = self._tables["orders"].read(
                    prior.replacement_order_id,
                    caller=None,
                )
                if (
                    prior == requested
                    and isinstance(persisted_replacement, Order)
                    and _order_identity_matches(
                        persisted_replacement,
                        replacement_order,
                    )
                ):
                    return cast("Exchange", prior)
                raise IdempotencyConflict(
                    f"idempotency key {requested.idempotency_key!r} was already "
                    "used for a different exchange request"
                )
            if prior.exchange_id == requested.exchange_id:
                raise ExchangeNotActionable(
                    f"exchange id {requested.exchange_id} already exists"
                )
            if prior.original_order_id == requested.original_order_id:
                raise ExchangeNotActionable(
                    f"order {requested.original_order_id} was already exchanged"
                )
            if prior.replacement_order_id == requested.replacement_order_id:
                raise ExchangeNotActionable(
                    f"order {requested.replacement_order_id} is already a replacement"
                )
        return None

    def _transition_order(
        self,
        *,
        order_id: OrderId,
        by_actor: str,
        allowed_actor_sides: frozenset[str],
        allowed_from: frozenset[OrderState],
        replay_states: frozenset[OrderState],
        target: OrderState,
        action: str,
        timeline_field: str | None = None,
        create_shipment: bool = False,
        create_packing: bool = False,
    ) -> Order:
        """Apply one allow-listed order-state transition under the world lock."""
        with self._lock:
            existing = cast(
                "Order | None",
                self._tables["orders"].read(order_id, caller=None),
            )
            if existing is None:
                raise InvalidOrderTransition(f"order {order_id} does not exist")
            _authorize_lifecycle_actor(existing, by_actor, allowed_actor_sides)
            if existing.state in replay_states:
                return existing
            if existing.state not in allowed_from:
                allowed = ", ".join(sorted(state.value for state in allowed_from))
                raise InvalidOrderTransition(
                    f"cannot {action} order {order_id} from {existing.state.value!r}; "
                    f"allowed states: {allowed}"
                )
            updated = replace(existing, state=target)
            timing_before = cast(
                "OrderTimeline | None",
                self._tables["order_timelines"].read(order_id, caller=None),
            )
            event_tick = self._logical_time + 1 if timeline_field is not None else None
            timing_after = timing_before
            if timeline_field is not None:
                if timing_before is None:
                    listing = self._tables["catalog"].read(existing.sku_id, caller=None)
                    timing_before_window = (
                        _captured_return_window(listing)
                        if timeline_field == "dispatched_at_tick"
                        else None
                    )
                    timing_after = OrderTimeline(
                        order_id=existing.order_id,
                        buyer_id=existing.buyer_id,
                        merchant_id=existing.merchant_id,
                        return_window_ticks=timing_before_window,
                        **{timeline_field: event_tick},
                    )
                else:
                    timing_after = replace(
                        timing_before,
                        **{timeline_field: event_tick},
                    )
            shipment_before: Shipment | None = None
            shipment_after: Shipment | None = None
            if create_shipment:
                assert event_tick is not None
                shipment_table = cast(
                    "ShipmentTable", self._tables["shipments"]
                )
                shipment_before = shipment_table.by_order(existing.order_id)
                if shipment_before is not None:
                    raise ShipmentNotActionable(
                        f"order {existing.order_id} already has shipment "
                        f"{shipment_before.shipment_id} before dispatch"
                    )
                shipment_id = ShipmentId(f"shipment:{existing.order_id}")
                if shipment_table.read(shipment_id, caller=None) is not None:
                    raise ShipmentNotActionable(
                        f"shipment id {shipment_id} already exists"
                    )
                shipment_after = Shipment(
                    shipment_id=shipment_id,
                    order_id=existing.order_id,
                    buyer_id=existing.buyer_id,
                    merchant_id=existing.merchant_id,
                    original_sku_id=existing.sku_id,
                    status=ShipmentStatus.IN_TRANSIT,
                    status_history=(
                        ShipmentStatusEvent(
                            event_id=f"shipment:{existing.order_id}:created",
                            status=ShipmentStatus.IN_TRANSIT,
                            logical_time=event_tick,
                            shipment_version=1,
                        ),
                    ),
                    version=1,
                )
            packing_rows: tuple[PackingRecord, ...] = ()
            if create_packing:
                if event_tick is None:
                    raise WorldError("packing dispatch requires a World event tick")
                payment_history = cast(
                    PaymentStateTable, self._tables["payment_states"]
                ).history_for_order(str(existing.order_id))
                if not payment_history:
                    if existing.state != OrderState.PARTIALLY_SETTLED:
                        raise WorldError(
                            "dispatch requires first-class captured payment state"
                        )
                else:
                    prior_packing = cast(
                        PackingRecordTable, self._tables["packing_records"]
                    ).history_for_order(str(existing.order_id))
                    packing_actor = (
                        by_actor
                        if by_actor == str(existing.merchant_id)
                        else "platform:fulfillment"
                    )
                    packing_rows = derive_dispatch_packing_sequence(
                        updated,
                        payment_history[-1],
                        previous=prior_packing[-1] if prior_packing else None,
                        original_actor=packing_actor,
                        server_tick=event_tick,
                        idempotency_key=f"dispatch:{existing.order_id}",
                    )
            before_tick = self._logical_time
            diff_cursor = len(self._txn_diffs)
            commit_cursor = len(self._commit_journal)
            prior_revisions = dict(self._order_state_revisions)
            try:
                self.write(
                    "orders",
                    order_id,
                    updated,
                    by_action="world.update_order_status",
                    _record_commit=False,
                )
                if timing_after is not None and timing_after != timing_before:
                    self.write(
                        "order_timelines",
                        order_id,
                        timing_after,
                        by_action="world.record_order_timeline",
                        _record_commit=False,
                    )
                if shipment_after is not None:
                    self.write(
                        "shipments",
                        shipment_after.shipment_id,
                        shipment_after,
                        by_action="world.record_shipment",
                        _record_commit=False,
                    )
                for packing in packing_rows:
                    self.write(
                        "packing_records",
                        packing_record_key(packing),
                        packing,
                        by_action="world.dispatch_order",
                        _record_commit=False,
                    )
                if event_tick is not None:
                    self._logical_time = event_tick
                diff_writes: list[tuple[str, str, str, Any, Any]] = [
                    ("orders", str(order_id), "update", existing, updated)
                ]
                if timing_after is not None and timing_after != timing_before:
                    diff_writes.append((
                        "order_timelines",
                        str(order_id),
                        "update" if timing_before is not None else "create",
                        timing_before,
                        timing_after,
                    ))
                if shipment_after is not None:
                    diff_writes.append((
                        "shipments",
                        str(shipment_after.shipment_id),
                        "create",
                        None,
                        shipment_after,
                    ))
                for packing in packing_rows:
                    diff_writes.append((
                        "packing_records",
                        packing_record_key(packing),
                        "create",
                        None,
                        packing,
                    ))
                if event_tick is not None:
                    diff_writes.append((
                        "logical_time",
                        "world",
                        "update",
                        event_tick - 1,
                        event_tick,
                    ))
                self._append_transaction_commit(
                    _make_diff(
                        action,
                        order_id,
                        diff_writes,
                        (
                            "atomic",
                            f"allowlist:{target.value.upper()}",
                            "non-regressing",
                            *((
                                "unique-shipment-per-order",
                            ) if create_shipment else ()),
                            *((
                                "complete-packing-history",
                            ) if packing_rows else ()),
                        ),
                    ),
                    authority_action=f"world.{action}",
                    actor_id=by_actor,
                )
            except BaseException:
                _restore_row(self._tables["orders"], order_id, existing)
                _restore_row(
                    self._tables["order_timelines"], order_id, timing_before
                )
                if shipment_after is not None:
                    _restore_row(
                        self._tables["shipments"],
                        shipment_after.shipment_id,
                        shipment_before,
                    )
                for packing in reversed(packing_rows):
                    _restore_row(
                        self._tables["packing_records"],
                        packing_record_key(packing),
                        None,
                    )
                self._logical_time = before_tick
                self._order_state_revisions = prior_revisions
                del self._txn_diffs[diff_cursor:]
                del self._commit_journal[commit_cursor:]
                raise
            return updated

    def open_dispute(self, *, dispute: Dispute, by_actor: str) -> Dispute:
        """Open one deterministic dispute for a paid order.

        The full caller id must equal ``filed_by`` and the two named parties must
        be the order's buyer and merchant. Replaying the same dispute id and
        immutable claim is idempotent, even after a ruling; conflicting reuse or
        a second dispute id for the same order is rejected before any write.
        """
        with self._lock:
            _validate_open_dispute_shape(dispute, by_actor)
            existing = cast(
                "Dispute | None",
                self._tables["disputes"].read(dispute.dispute_id, caller=None),
            )
            if existing is not None:
                if _same_dispute_claim(existing, dispute):
                    return existing
                raise DisputeNotActionable(
                    f"dispute id {dispute.dispute_id} already names a different claim"
                )
            order = cast(
                "Order | None",
                self._tables["orders"].read(dispute.order_id, caller=None),
            )
            _validate_dispute_order(dispute, order)
            for _, prior in self._tables["disputes"].all():
                if str(prior.order_id) == str(dispute.order_id):
                    raise DisputeNotActionable(
                        f"order {dispute.order_id} already has dispute {prior.dispute_id}"
                    )
            try:
                self.write(
                    "disputes",
                    dispute.dispute_id,
                    dispute,
                    by_action="world.open_dispute",
                    _record_commit=False,
                )
            except BaseException:
                _restore_row(self._tables["disputes"], dispute.dispute_id, None)
                raise
            self._append_transaction_commit(
                _make_diff(
                    "open_dispute",
                    dispute.order_id,
                    [("disputes", str(dispute.dispute_id), "create", None, dispute)],
                    ("atomic", "party-bound", "one-dispute-per-order"),
                ),
                authority_action="world.open_dispute",
                actor_id=by_actor,
            )
            return dispute

    def rule_dispute(self, *, ruling: Ruling, by_actor: str) -> Ruling:
        """Atomically persist a ruling and mark its dispute ``RULED``.

        Only ``platform:adjudicator`` may rule. The ruling may declare a refund
        amount, but does not move money; the atomic ``refund_order`` primitive
        remains the only financial mutation path.
        """
        with self._lock:
            if by_actor != "platform:adjudicator":
                raise LifecycleAuthorizationError(
                    "only platform:adjudicator may rule a dispute"
                )
            dispute = cast(
                "Dispute | None",
                self._tables["disputes"].read(ruling.dispute_id, caller=None),
            )
            if dispute is None:
                raise DisputeNotActionable(
                    f"dispute {ruling.dispute_id} does not exist"
                )
            existing_ruling = cast(
                "Ruling | None",
                self._tables["rulings"].read(ruling.ruling_id, caller=None),
            )
            if existing_ruling is not None:
                if existing_ruling == ruling and dispute.state == DisputeState.RULED:
                    return existing_ruling
                raise DisputeNotActionable(
                    f"ruling id {ruling.ruling_id} already names a different outcome"
                )
            for _, prior in self._tables["rulings"].all():
                if str(prior.dispute_id) == str(ruling.dispute_id):
                    raise DisputeNotActionable(
                        f"dispute {ruling.dispute_id} already has ruling {prior.ruling_id}"
                    )
            if dispute.state not in {DisputeState.OPEN, DisputeState.UNDER_REVIEW}:
                raise DisputeNotActionable(
                    f"dispute {ruling.dispute_id} in state {dispute.state.value!r} "
                    "cannot be ruled"
                )
            order = cast(
                "Order | None",
                self._tables["orders"].read(dispute.order_id, caller=None),
            )
            _validate_ruling(ruling, dispute, order)
            ruled_dispute = replace(dispute, state=DisputeState.RULED)
            try:
                self.write(
                    "disputes",
                    dispute.dispute_id,
                    ruled_dispute,
                    by_action="world.update_dispute",
                    _record_commit=False,
                )
                self.write(
                    "rulings",
                    ruling.ruling_id,
                    ruling,
                    by_action="world.update_ruling",
                    _record_commit=False,
                )
            except BaseException:
                _restore_row(self._tables["disputes"], dispute.dispute_id, dispute)
                _restore_row(self._tables["rulings"], ruling.ruling_id, None)
                raise
            self._append_transaction_commit(
                _make_diff(
                    "rule_dispute",
                    dispute.order_id,
                    [
                        (
                            "disputes",
                            str(dispute.dispute_id),
                            "update",
                            dispute,
                            ruled_dispute,
                        ),
                        ("rulings", str(ruling.ruling_id), "create", None, ruling),
                    ],
                    ("atomic", "adjudicator-only", "single-ruling"),
                ),
                authority_action="world.rule_dispute",
                actor_id=by_actor,
            )
            return ruling

    # -- First-class payment, packing, and after-sales authority ---------

    def authorize_payment(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> PaymentStateRecord:
        """Authorize an order and reserve inventory in one World commit.

        ``by_actor`` is the authenticated payment service. ``original_actor``
        remains the principal on whose behalf it acts. The payment core sees
        only the trusted service identity, never a buyer masquerading as PSP.
        """

        normalized = normalize_payment_intent(intent)
        if normalized["op"] != "authorize":
            raise WorldError("authorize_payment requires an authorize intent")
        if by_actor != "platform:psp":
            raise WriteNotAuthorized("only platform:psp may authorize payment")
        with self._lock:
            order = cast(
                "Order | None",
                self._tables["orders"].read(normalized["order_id"], caller=None),
            )
            if order is None:
                raise WorldError("payment order does not exist")
            if original_actor != str(order.buyer_id):
                raise WriteNotAuthorized("payment principal does not own order")
            fingerprint = after_sales_digest(
                {
                    "intent": normalized,
                    "service_actor": by_actor,
                    "original_actor": original_actor,
                }
            )
            replay = self._authority_operation_replay(
                scope="apply_payment_intent",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if replay is not None:
                return cast(PaymentStateRecord, replay)
            inventory = self._tables["inventory"].read(order.sku_id, caller=None)
            if inventory is None or _available_qty(inventory) < order.qty:
                raise OutOfStock(f"insufficient inventory for {order.sku_id}")
            _require_inventory_owner(order, inventory)
            tick = self._logical_time + 1
            payment = derive_payment_authorization(
                order,
                original_actor=by_actor,
                server_tick=tick,
                idempotency_key=idempotency_key,
            )
            payment_key = payment_state_key(payment)
            reserved = _reserve_inventory(inventory, qty=order.qty)
            self._commit_simple_authority_transition(
                operation="authorize_payment",
                authority_action="world.apply_payment_intent",
                scope="apply_payment_intent",
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                subject_id=str(order.order_id),
                outcome_table="payment_states",
                outcome_key=payment_key,
                tick=tick,
                mutations=(
                    ("inventory", str(order.sku_id), inventory, reserved),
                    ("payment_states", payment_key, None, payment),
                ),
                invariants=(
                    "psp-authenticated",
                    "principal-order-bound",
                    "inventory-reserved",
                    "single-clock-advance",
                ),
            )
            return payment

    def capture_payment(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> PaymentStateRecord:
        """Bind a captured payment version to the exact World ledger receipt."""

        normalized = normalize_payment_intent(intent)
        if normalized["op"] != "capture":
            raise WorldError("capture_payment requires a capture intent")
        if by_actor != "platform:psp":
            raise WriteNotAuthorized("only platform:psp may capture payment")
        with self._lock:
            order = cast(
                "Order | None",
                self._tables["orders"].read(normalized["order_id"], caller=None),
            )
            if order is None:
                raise WorldError("payment order does not exist")
            if original_actor != str(order.buyer_id):
                raise WriteNotAuthorized("payment principal does not own order")
            fingerprint = after_sales_digest(
                {
                    "intent": normalized,
                    "service_actor": by_actor,
                    "original_actor": original_actor,
                }
            )
            replay = self._authority_operation_replay(
                scope="apply_payment_intent",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if replay is not None:
                return cast(PaymentStateRecord, replay)
            receipt = _charge_receipt_for_order(
                self._tables["ledger"], str(order.order_id), payment=None
            )
            if receipt is None:
                raise WorldError("payment capture requires a persisted charge receipt")
            history = cast(
                PaymentStateTable, self._tables["payment_states"]
            ).history_for_order(str(order.order_id))
            previous = history[-1] if history else None
            tick = self._logical_time + 1
            payment = derive_payment_capture(
                order,
                receipt,
                previous=previous,
                original_actor=by_actor,
                server_tick=tick,
                idempotency_key=idempotency_key,
            )
            payment_key = payment_state_key(payment)
            self._commit_simple_authority_transition(
                operation="capture_payment",
                authority_action="world.apply_payment_intent",
                scope="apply_payment_intent",
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                subject_id=str(order.order_id),
                outcome_table="payment_states",
                outcome_key=payment_key,
                tick=tick,
                mutations=(("payment_states", payment_key, None, payment),),
                invariants=(
                    "psp-authenticated",
                    "principal-order-bound",
                    "ledger-receipt-bound",
                    "single-clock-advance",
                ),
            )
            return payment

    def apply_packing_intent(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> PackingRecord:
        """Apply one compact fulfillment intent to authoritative packing state."""

        normalized = normalize_packing_intent(intent)
        if by_actor != "platform:fulfillment" and not by_actor.startswith("merchant:"):
            raise WriteNotAuthorized(
                "packing requires platform:fulfillment or the owning merchant route"
            )
        with self._lock:
            order = cast(
                "Order | None",
                self._tables["orders"].read(normalized["order_id"], caller=None),
            )
            if order is None:
                raise WorldError("packing order does not exist")
            if original_actor != str(order.merchant_id):
                raise WriteNotAuthorized("packing principal is not order merchant")
            if by_actor.startswith("merchant:") and by_actor != original_actor:
                raise WriteNotAuthorized("merchant packing route is not owner bound")
            fingerprint = after_sales_digest(
                {
                    "intent": normalized,
                    "service_actor": by_actor,
                    "original_actor": original_actor,
                }
            )
            replay = self._authority_operation_replay(
                scope="apply_packing_intent",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if replay is not None:
                return cast(PackingRecord, replay)
            payments = cast(
                PaymentStateTable, self._tables["payment_states"]
            ).history_for_order(str(order.order_id))
            if not payments:
                raise WorldError("packing requires first-class payment state")
            packing_history = cast(
                PackingRecordTable, self._tables["packing_records"]
            ).history_for_order(str(order.order_id))
            previous = packing_history[-1] if packing_history else None
            target = {
                "create": "created",
                "pack": "packed",
                "cancel": "cancelled",
                "hand_off": "handed_off",
            }[normalized["op"]]
            tick = self._logical_time + 1
            packing = derive_packing_transition(
                order,
                payments[-1],
                previous=previous,
                target_state=cast(Any, target),
                original_actor=by_actor,
                server_tick=tick,
                idempotency_key=idempotency_key,
            )
            packing_key = packing_record_key(packing)
            self._commit_simple_authority_transition(
                operation=f"packing_{normalized['op']}",
                authority_action="world.apply_packing_intent",
                scope="apply_packing_intent",
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                subject_id=str(order.order_id),
                outcome_table="packing_records",
                outcome_key=packing_key,
                tick=tick,
                mutations=(("packing_records", packing_key, None, packing),),
                invariants=(
                    "fulfillment-service-authenticated",
                    "merchant-order-bound",
                    "payment-state-bound",
                    "single-clock-advance",
                ),
            )
            return packing

    def publish_after_sales_policy(
        self,
        policy_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesPolicyRevision:
        """Publish a compact merchant policy as a World-sealed revision."""

        if by_actor != "platform:policy":
            raise WriteNotAuthorized(
                "only platform:policy may publish after-sales policy"
            )
        normalized = _normalize_after_sales_policy_intent(policy_intent)
        with self._lock:
            fingerprint = after_sales_digest(
                {
                    "intent": normalized,
                    "service_actor": by_actor,
                    "original_actor": original_actor,
                }
            )
            replay = self._authority_operation_replay(
                scope="publish_after_sales_policy",
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if replay is not None:
                return cast(AfterSalesPolicyRevision, replay)
            table = cast(
                AfterSalesPolicyTable, self._tables["after_sales_policies"]
            )
            current = table.latest(original_actor)
            tick = self._logical_time + 1
            policy = build_after_sales_policy_revision(
                **normalized,
                merchant_id=original_actor,
                revision=1 if current is None else current.revision + 1,
                previous_digest=None if current is None else current.policy_digest,
                effective_tick=tick,
                published_by_id=by_actor,
            )
            plan = stage_policy_publication(
                self._after_sales_tables(),
                policy,
                original_actor=by_actor,
                server_tick=tick,
                trusted_publisher_ids=("platform:policy",),
            )
            self._commit_simple_authority_transition(
                operation="publish_after_sales_policy",
                authority_action="world.publish_after_sales_policy",
                scope="publish_after_sales_policy",
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                subject_id=original_actor,
                outcome_table="after_sales_policies",
                outcome_key=plan.write.key,
                tick=tick,
                mutations=((
                    "after_sales_policies",
                    plan.write.key,
                    None,
                    policy,
                ),),
                invariants=(
                    "policy-service-authenticated",
                    "merchant-principal-bound",
                    "contiguous-policy-revision",
                    "single-clock-advance",
                ),
            )
            return policy

    def apply_after_sales_intent(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesCommandResult:
        """Plan and atomically commit one compact after-sales command."""

        if by_actor != "platform:after-sales":
            raise WriteNotAuthorized(
                "only platform:after-sales may apply after-sales intents"
            )
        normalized = normalize_after_sales_intent(intent)
        with self._lock:
            context = self.load_after_sales_context(
                cast(str, normalized["order_id"]),
                intent=normalized,
                original_actor=original_actor,
            )
            binding = context.tables.binding_for_order(
                str(context.order.order_id), caller="world"
            )
            policy = _after_sales_policy_for_context(
                context.tables,
                merchant_id=str(context.order.merchant_id),
                binding=binding,
            )
            if binding is None:
                from world.after_sales_core import derive_after_sales_binding

                binding = derive_after_sales_binding(
                    order=context.order,
                    receipt=context.charge_receipt,
                    shipment=context.shipment,
                    policy=policy,
                    payment=context.payment,
                )
            fingerprint = after_sales_intent_fingerprint(
                normalized,
                binding=binding,
                policy=policy,
                original_actor=original_actor,
                evidence_records=context.evidence_records,
            )
            retry = self._after_sales_retry_result(
                operation=cast(str, normalized["op"]),
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            if retry is not None:
                return retry
            planned = AfterSalesPlanner().plan(
                context,
                normalized,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
            )
            if planned.commit is not None:
                self.commit_after_sales(planned.commit)
                # ``planned`` is the response for this successful commit.  A
                # post-commit lookup necessarily sees the new authority row
                # and would mislabel the first call as an idempotent replay.
                return planned
            return planned

    def load_after_sales_context(
        self,
        order_id: str,
        *,
        intent: Mapping[str, Any] | None = None,
        original_actor: str | None = None,
        evidence_digests: Mapping[str, str] | None = None,
    ) -> TrustedAfterSalesContext:
        """Load exact authoritative rows consumed by the after-sales planner."""

        with self._lock:
            order = cast(
                "Order | None", self._tables["orders"].read(order_id, caller=None)
            )
            if order is None:
                if (
                    isinstance(original_actor, str)
                    and original_actor.split(":", 1)[0] in {"buyer", "merchant"}
                ):
                    raise AfterSalesReferenceRejected(
                        "after_sales_order_not_found"
                    )
                raise WorldError(f"after-sales order does not exist: {order_id}")
            payments = cast(
                PaymentStateTable, self._tables["payment_states"]
            ).history_for_order(order_id)
            if not payments:
                raise AfterSalesCoreTransitionError(
                    "after-sales requires first-class payment state"
                )
            payment = payments[-1]
            charge = _charge_receipt_for_order(
                self._tables["ledger"], order_id, payment=payment, history=payments
            )
            packings = cast(
                PackingRecordTable, self._tables["packing_records"]
            ).history_for_order(order_id)
            packing = packings[-1] if packings else None
            shipment = cast(
                ShipmentTable, self._tables["shipments"]
            ).by_order(order_id)
            timeline = cast(
                "OrderTimeline | None",
                self._tables["order_timelines"].read(order_id, caller=None),
            )
            settled_at = (
                timeline.settled_at_tick
                if timeline is not None and timeline.settled_at_tick is not None
                else _receipt_tick(charge)
            )
            tables = self._after_sales_tables()
            binding = tables.binding_for_order(order_id, caller="world")
            policy = _after_sales_policy_for_context(
                tables,
                merchant_id=str(order.merchant_id),
                binding=binding,
            )
            evidence_records = load_trusted_after_sales_evidence(
                intent,
                order=order,
                policy=policy,
                logical_tick=self._logical_time,
                original_actor=original_actor,
                lookup=lambda record_id: (
                    cast(
                        EvidenceRecordTable,
                        self._tables["evidence_records"],
                    ).current(record_id)
                    if evidence_digests is None
                    or record_id not in evidence_digests
                    else cast(
                        EvidenceRecordTable,
                        self._tables["evidence_records"],
                    ).by_digest(record_id, evidence_digests[record_id])
                ),
            )
            if evidence_digests is not None and set(evidence_digests) != {
                record.record_id for record in evidence_records
            }:
                raise WorldError("trusted evidence digest bindings are incomplete")
            replacement = self._after_sales_replacement_context(
                order,
                intent=intent,
                tables=tables,
            )
            ledger_sources = ()
            if binding is not None:
                ledger_sources = tuple(
                    derive_ledger_reconciliation_source(
                        receipt,
                        binding=binding,
                        effect=receipt.effect,
                        server_tick=_receipt_tick(receipt),
                    )
                    for _, receipt in self._tables["ledger"].all()
                    if str(receipt.order_id) == order_id
                )
            return TrustedAfterSalesContext(
                logical_tick=self._logical_time,
                order=deepcopy(order),
                payment=deepcopy(payment),
                charge_receipt=deepcopy(charge),
                packing=deepcopy(packing),
                shipment=deepcopy(shipment),
                settled_at_tick=settled_at,
                tables=tables,
                replacement=replacement,
                evidence_records=evidence_records,
                ledger_sources=ledger_sources,
            )

    def _after_sales_replacement_context(
        self,
        order: Order,
        *,
        intent: Mapping[str, Any] | None,
        tables: AfterSalesTables,
    ) -> TrustedReplacementContext | None:
        if intent is None:
            return None
        candidate = intent.get("replacement_sku_id")
        case_id: str | None = None
        if candidate is None and intent.get("case_id") is not None:
            requested_case = str(intent["case_id"])
            rows = [
                row
                for _, row in tables.internal_all("exchange_cases")
            ]
            if rows:
                latest = resolve_current_versioned_record(
                    tables,
                    "exchange_cases",
                    rows,
                    reference=requested_case,
                    identity_field="case_id",
                )
                candidate = latest.replacement_sku_id
                case_id = latest.case_id
        if candidate is None:
            return None
        sku_id = str(candidate)
        listing = self._tables["catalog"].read(sku_id, caller=None)
        inventory = self._tables["inventory"].read(sku_id, caller=None)
        if not isinstance(listing, Listing) or not isinstance(
            inventory, InventoryRow
        ):
            return None
        replacement_order = Order(
            order_id=OrderId(
                f"replacement:{case_id}"
                if case_id is not None
                else f"replacement:pending:{order.order_id}:{sku_id}"
            ),
            buyer_id=order.buyer_id,
            merchant_id=order.merchant_id,
            sku_id=SkuId(sku_id),
            qty=order.qty,
            agreed_price=order.agreed_price,
            state=OrderState.SETTLED,
            request_order=order.request_order,
        )
        return TrustedReplacementContext(
            sku_id=sku_id,
            merchant_id=str(listing.merchant_id),
            available_qty=_available_qty(inventory),
            listing_digest=after_sales_digest(
                {
                    "sku_id": str(listing.sku_id),
                    "merchant_id": str(listing.merchant_id),
                    "product_id": listing.product_id,
                    "name": listing.name,
                    "category": listing.category,
                    "attributes": listing.attributes,
                    "list_price": str(listing.list_price.amount),
                    "currency": listing.list_price.currency,
                }
            ),
            inventory_digest=after_sales_digest(
                {
                    "sku_id": str(inventory.sku_id),
                    "merchant_id": str(inventory.merchant_id),
                    "qty_available": inventory.qty_available,
                    "qty_reserved": inventory.qty_reserved,
                    "eta_day": inventory.eta_day,
                    "version": inventory.version,
                }
            ),
            projected_order_digest=authoritative_order_digest(replacement_order),
        )

    def commit_after_sales(self, commit: AfterSalesCommit) -> None:
        """Commit planner output and every commercial effect under one lock."""

        with self._lock:
            current = self.load_after_sales_context(
                commit.operation.order_id,
                intent=_after_sales_context_hint(commit),
                original_actor=commit.operation.actor_id,
                evidence_digests=_after_sales_context_evidence_digests(commit),
            )
            if commit.expected_logical_tick != self._logical_time:
                raise AfterSalesContextConflict("World logical time changed")
            if commit.context_digest != current.context_digest:
                raise AfterSalesContextConflict("World after-sales context changed")
            if len(commit.commerce_effects) > 1:
                raise WorldError("after-sales commit has multiple commerce effects")
            if not commit.commerce_effects:
                self._commit_after_sales_records(commit)
                return
            effect = commit.commerce_effects[0]
            if effect.kind == "cancel_paid_order":
                self._commit_paid_cancellation(commit, effect, current)
            elif effect.kind == "mark_returned":
                self._commit_return_receipt(commit, effect, current)
            elif effect.kind == "issue_refund":
                self._commit_after_sales_refund(commit, effect, current)
            elif effect.kind == "complete_exchange":
                self._commit_after_sales_exchange(commit, effect, current)
            else:
                raise WorldError(
                    f"unsupported after-sales commerce effect: {effect.kind}"
                )

    def complete_ledger_reconciliation(
        self,
        order_id: str,
        request_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesCommandResult:
        """Derive one accounting result from authoritative typed ledger effects."""

        if by_actor != "platform:accounting":
            raise WriteNotAuthorized(
                "only platform:accounting may complete reconciliation"
            )
        if original_actor != by_actor:
            raise WriteNotAuthorized(
                "ledger reconciliation principal must be platform accounting"
            )
        with self._lock:
            context = self.load_after_sales_context(order_id)
            planned = AfterSalesPlanner().plan_reconciliation_result(
                context,
                request_id=request_id,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
            )
            replay = self._after_sales_retry_result(
                operation="complete_ledger_reconciliation",
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=planned.operation.request_fingerprint,
                scope="complete_ledger_reconciliation",
            )
            if replay is not None:
                return replay
            if planned.commit is None:
                raise WorldError("reconciliation plan has no commit")
            self._commit_after_sales_records(
                planned.commit,
                authority_action="world.complete_ledger_reconciliation",
                scope="complete_ledger_reconciliation",
            )
            return planned

    def payment_history(
        self, order_id: str, *, caller: str
    ) -> tuple[PaymentStateRecord, ...]:
        with self._lock:
            table = cast(PaymentStateTable, self._tables["payment_states"])
            return tuple(
                row
                for row in table.history_for_order(order_id)
                if table.read(payment_state_key(row), caller=caller) is not None
            )

    def ledger_history(self, order_id: str, *, caller: str) -> tuple[Receipt, ...]:
        """Return exact order receipts only to an order party or trusted service."""

        with self._lock:
            order = self._tables["orders"].read(order_id, caller=caller)
            if order is None:
                return ()
            ledger = cast(LedgerTable, self._tables["ledger"])
            rows = [
                receipt
                for key, receipt in ledger.all()
                if str(receipt.order_id) == order_id
                and ledger.read(key, caller=caller) is not None
            ]
            return tuple(sorted(rows, key=lambda row: str(row.txn_id)))

    def packing_history(
        self, order_id: str, *, caller: str
    ) -> tuple[PackingRecord, ...]:
        with self._lock:
            table = cast(PackingRecordTable, self._tables["packing_records"])
            return tuple(
                row
                for row in table.history_for_order(order_id)
                if table.read(packing_record_key(row), caller=caller) is not None
            )

    def after_sales_history(self, order_id: str, *, caller: str) -> tuple[Any, ...]:
        with self._lock:
            table = cast(
                AfterSalesRecordTable, self._tables["after_sales_records"]
            )
            return tuple(
                write.value
                for write in table.history_for_order(order_id)
                if table.read(
                    physical_after_sales_record_key(write), caller=caller
                )
                is not None
            )

    def after_sales_result_record(
        self,
        operation: AfterSalesOperationRecord,
        *,
        caller: str,
    ) -> AfterSalesWrite | None:
        """Resolve one operation's exact persisted result under row ACLs."""

        with self._lock:
            table = cast(
                AfterSalesRecordTable, self._tables["after_sales_records"]
            )
            physical_key = physical_after_sales_record_lookup_key(
                operation.result_table, operation.result_key
            )
            write = table.read(physical_key, caller=caller)
            if write is None:
                return None
            if (
                write.table != operation.result_table
                or write.key != operation.result_key
                or after_sales_record_digest(write.value)
                != operation.result_digest
            ):
                return None
            return write

    def after_sales_policy(
        self, merchant_id: str, *, caller: str
    ) -> AfterSalesPolicyRevision | None:
        """Return the latest policy to its owner or a trusted service.

        Agent-facing public projection is produced by ``WorldService`` after
        it verifies the authenticated ``original_actor``.  Returning the typed
        row here keeps policy selection in World and avoids Platform-owned
        lifecycle state.
        """

        with self._lock:
            table = cast(
                AfterSalesPolicyTable, self._tables["after_sales_policies"]
            )
            policy = table.latest(merchant_id)
            if policy is None:
                return None
            return table.read(
                f"{policy.merchant_id}:{policy.revision}", caller=caller
            )

    def _after_sales_tables(self) -> AfterSalesTables:
        projection = AfterSalesTables()
        for _, policy in self._tables["after_sales_policies"].all():
            projection.append(
                AfterSalesWrite(
                    "after_sales_policies",
                    f"{policy.merchant_id}:{policy.revision}",
                    policy,
                )
            )
        for _, write in self._tables["after_sales_records"].all():
            projection.append(write)
        return projection

    def _after_sales_retry_result(
        self,
        *,
        operation: str,
        original_actor: str,
        idempotency_key: str,
        request_fingerprint: str,
        scope: str = "apply_after_sales_intent",
    ) -> AfterSalesCommandResult | None:
        key = authority_operation_key(
            scope, original_actor, idempotency_key
        )
        authority = cast(
            "AuthorityOperationRecord | None",
            self._tables["authority_operations"].read(key, caller="runtime"),
        )
        if authority is None:
            return None
        if authority.request_fingerprint != request_fingerprint:
            raise IdempotencyConflict(
                "after-sales idempotency key was reused for another intent"
            )
        stored = cast(
            "AfterSalesWrite | None",
            self._tables["after_sales_records"].read(
                authority.outcome_key, caller="runtime"
            ),
        )
        if stored is None:
            raise WorldError("after-sales idempotency outcome is missing")
        effects = _effects_for_persisted_after_sales(stored.value)
        binding = getattr(stored.value, "binding", None)
        if binding is None:
            raise WorldError("after-sales result has no order binding")
        operation_row = build_after_sales_operation(
            operation=operation,
            order_id=binding.order_id,
            binding_digest=binding.binding_digest,
            actor_id=original_actor,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            logical_tick=getattr(stored.value, "logical_tick"),
            result_table=stored.table,
            result_key=stored.key,
            result_digest=after_sales_record_digest(stored.value),
            effects=effects,
        )
        return AfterSalesCommandResult("idempotent", operation_row, None)

    def _commit_after_sales_records(
        self,
        commit: AfterSalesCommit,
        *,
        extra_mutations: tuple[tuple[str, str, Any | None, Any], ...] = (),
        authority_action: str = "world.apply_after_sales_intent",
        scope: str = "apply_after_sales_intent",
        extra_invariants: tuple[str, ...] = (),
    ) -> None:
        """Persist typed rows, reused core projections, and authority atomically."""

        core_mutations = self._after_sales_core_projection_mutations(commit)
        mutations: list[tuple[str, str, Any | None, Any]] = [
            *extra_mutations,
            *core_mutations,
        ]
        for write in commit.writes:
            if write.table == "after_sales_policies":
                raise WorldError("after-sales command cannot publish policy")
            physical_key = physical_after_sales_record_key(write)
            mutations.append(("after_sales_records", physical_key, None, write))
        result_write = next(
            write
            for write in commit.writes
            if write.table == commit.operation.result_table
            and write.key == commit.operation.result_key
        )
        self._commit_simple_authority_transition(
            operation=commit.operation.operation,
            authority_action=authority_action,
            scope=scope,
            original_actor=commit.operation.actor_id,
            idempotency_key=commit.operation.idempotency_key,
            request_fingerprint=commit.operation.request_fingerprint,
            subject_id=commit.operation.order_id,
            outcome_table="after_sales_records",
            outcome_key=physical_after_sales_record_key(result_write),
            tick=commit.logical_tick,
            mutations=tuple(mutations),
            invariants=(
                "after-sales-context-cas",
                "typed-causal-records",
                "principal-provenance-preserved",
                "single-clock-advance",
                *extra_invariants,
            ),
        )

    def _after_sales_core_projection_mutations(
        self, commit: AfterSalesCommit
    ) -> tuple[tuple[str, str, Any | None, Any], ...]:
        """Project typed dispute facts into existing CommerceWorld tables."""

        mutations: list[tuple[str, str, Any | None, Any]] = []
        for write in commit.writes:
            value = write.value
            if write.table == "dispute_cases":
                dispute_id = str(value.dispute_id)
                before = self._tables["disputes"].read(dispute_id, caller=None)
                after = Dispute(
                    dispute_id=DisputeId(dispute_id),
                    order_id=OrderId(value.binding.order_id),
                    filed_by=AgentId(value.filed_by_id),
                    against=AgentId(value.against_id),
                    reason=value.reason,
                    state=DisputeState(value.state),
                )
                if before is not None and (
                    str(before.order_id) != value.binding.order_id
                    or str(before.filed_by) != value.filed_by_id
                    or str(before.against) != value.against_id
                    or before.reason != value.reason
                ):
                    raise AfterSalesContextConflict(
                        "typed dispute conflicts with core dispute identity"
                    )
                mutations.append(("disputes", dispute_id, before, after))
            elif write.table == "after_sales_rulings":
                ruling_id = str(value.ruling_id)
                refund_amount = (
                    None
                    if value.refund_amount == 0
                    else Money(
                        (Decimal(value.refund_amount) / Decimal(100)).quantize(
                            Decimal("0.01")
                        ),
                        value.binding.currency,
                    )
                )
                ruling = Ruling(
                    ruling_id=RulingId(ruling_id),
                    dispute_id=DisputeId(str(value.dispute_id)),
                    in_favor_of=AgentId(str(value.winner_id)),
                    rationale=str(value.rationale),
                    refund_amount=refund_amount,
                )
                before = self._tables["rulings"].read(ruling_id, caller=None)
                mutations.append(("rulings", ruling_id, before, ruling))
        return tuple(mutations)

    def _commit_return_receipt(
        self,
        commit: AfterSalesCommit,
        effect: CommerceEffect,
        context: TrustedAfterSalesContext,
    ) -> None:
        order = context.order
        if order.state != OrderState.DISPATCHED:
            raise AfterSalesContextConflict(
                "physical return receipt requires a dispatched order"
            )
        if effect.order_id != str(order.order_id) or effect.qty > order.qty:
            raise AfterSalesContextConflict("return receipt order identity changed")
        returned = replace(order, state=OrderState.RETURNED)
        timeline = cast(
            "OrderTimeline | None",
            self._tables["order_timelines"].read(order.order_id, caller=None),
        )
        updated_timeline = (
            OrderTimeline(
                order_id=order.order_id,
                buyer_id=order.buyer_id,
                merchant_id=order.merchant_id,
                returned_at_tick=commit.logical_tick,
            )
            if timeline is None
            else replace(timeline, returned_at_tick=commit.logical_tick)
        )
        self._commit_after_sales_records(
            commit,
            extra_mutations=(
                ("orders", str(order.order_id), order, returned),
                (
                    "order_timelines",
                    str(order.order_id),
                    timeline,
                    updated_timeline,
                ),
            ),
            extra_invariants=(
                "physical-return-recorded",
                "inventory-reservation-preserved",
                "payment-state-preserved",
            ),
        )

    def _commit_after_sales_refund(
        self,
        commit: AfterSalesCommit,
        effect: CommerceEffect,
        context: TrustedAfterSalesContext,
    ) -> None:
        order = context.order
        if order.state not in {
            OrderState.SETTLED,
            OrderState.DISPATCHED,
            OrderState.RETURNED,
        }:
            raise AfterSalesContextConflict("order is not refundable")
        if context.payment.record_digest != effect.payment_before_digest:
            raise AfterSalesContextConflict("payment changed before refund")
        if effect.amount <= 0:
            raise WorldError("refund commerce effect amount must be positive")
        inventory = self._tables["inventory"].read(order.sku_id, caller=None)
        if not isinstance(inventory, InventoryRow):
            raise WorldError("refund requires indexed inventory")
        if inventory.qty_reserved < effect.inventory_release_qty:
            raise WorldError("refund inventory reservation underflow")
        refund = _after_sales_refund_receipt(
            order,
            amount=effect.amount,
            tick=commit.logical_tick,
            idempotency_key=commit.operation.idempotency_key,
        )
        payment = derive_payment_resolution(
            order,
            context.payment,
            outcome="refunded",
            refund_receipt=refund,
            original_actor="platform:psp",
            server_tick=commit.logical_tick,
            idempotency_key=f"payment:{commit.operation.idempotency_key}",
        )
        fully_refunded = payment.state == "refunded"
        refunded_order = (
            replace(order, state=OrderState.REFUNDED) if fully_refunded else order
        )
        timeline = cast(
            "OrderTimeline | None",
            self._tables["order_timelines"].read(order.order_id, caller=None),
        )
        updated_timeline = (
            None
            if not fully_refunded
            else (
                OrderTimeline(
                    order_id=order.order_id,
                    buyer_id=order.buyer_id,
                    merchant_id=order.merchant_id,
                    refunded_at_tick=commit.logical_tick,
                )
                if timeline is None
                else replace(timeline, refunded_at_tick=commit.logical_tick)
            )
        )
        mutations: list[tuple[str, str, Any | None, Any]] = []
        if fully_refunded:
            mutations.append(
                ("orders", str(order.order_id), order, refunded_order)
            )
        if effect.inventory_release_qty:
            released = replace(
                inventory,
                qty_reserved=inventory.qty_reserved
                - effect.inventory_release_qty,
                version=inventory.version + 1,
            )
            mutations.append(
                ("inventory", str(order.sku_id), inventory, released)
            )
        mutations.extend(
            (
                ("ledger", str(refund.txn_id), None, refund),
                ("payment_states", payment_state_key(payment), None, payment),
            )
        )
        if updated_timeline is not None:
            mutations.append(
                (
                    "order_timelines",
                    str(order.order_id),
                    timeline,
                    updated_timeline,
                )
            )
        self._commit_after_sales_records(
            commit,
            extra_mutations=tuple(mutations),
            extra_invariants=(
                "refund-ledger-payment-atomic",
                (
                    "order-fully-refunded"
                    if fully_refunded
                    else "order-state-preserved-for-partial-refund"
                ),
                (
                    "returned-quantity-released"
                    if effect.inventory_release_qty
                    else "dispute-does-not-restock"
                ),
            ),
        )

    def _commit_after_sales_exchange(
        self,
        commit: AfterSalesCommit,
        effect: CommerceEffect,
        context: TrustedAfterSalesContext,
    ) -> None:
        order = context.order
        if order.state != OrderState.RETURNED:
            raise AfterSalesContextConflict(
                "exchange completion requires a returned order"
            )
        result_write = next(
            write
            for write in commit.writes
            if write.table == commit.operation.result_table
            and write.key == commit.operation.result_key
        )
        case = result_write.value
        if getattr(case, "state", None) != "completed":
            raise WorldError("exchange completion result is not completed")
        replacement_sku = str(effect.replacement_sku_id)
        original_inventory = self._tables["inventory"].read(
            order.sku_id, caller=None
        )
        replacement_inventory = self._tables["inventory"].read(
            replacement_sku, caller=None
        )
        replacement_listing = self._tables["catalog"].read(
            replacement_sku, caller=None
        )
        if not isinstance(original_inventory, InventoryRow) or not isinstance(
            replacement_inventory, InventoryRow
        ):
            raise WorldError("exchange requires both inventory rows")
        if not isinstance(replacement_listing, Listing) or str(
            replacement_listing.merchant_id
        ) != str(order.merchant_id):
            raise WorldError("replacement listing is not merchant-owned")
        if original_inventory.qty_reserved < effect.inventory_release_qty:
            raise WorldError("exchange original reservation underflow")
        if _available_qty(replacement_inventory) < effect.qty:
            raise OutOfStock("replacement inventory cannot satisfy exchange")
        replacement_order = Order(
            order_id=OrderId(f"replacement:{case.case_id}"),
            buyer_id=order.buyer_id,
            merchant_id=order.merchant_id,
            sku_id=SkuId(replacement_sku),
            qty=effect.qty,
            agreed_price=order.agreed_price,
            state=OrderState.SETTLED,
            request_order=order.request_order,
        )
        if authoritative_order_digest(replacement_order) != case.completion_order_digest:
            raise AfterSalesContextConflict(
                "replacement order does not match planned completion digest"
            )
        exchanged = replace(order, state=OrderState.EXCHANGED)
        exchange = Exchange(
            exchange_id=ExchangeId(str(case.case_id)),
            original_order_id=order.order_id,
            replacement_order_id=replacement_order.order_id,
            buyer_id=order.buyer_id,
            merchant_id=order.merchant_id,
            original_sku_id=order.sku_id,
            replacement_sku_id=SkuId(replacement_sku),
            qty=effect.qty,
            created_by=AgentId(commit.operation.actor_id),
            idempotency_key=commit.operation.idempotency_key,
        )
        if str(order.sku_id) == replacement_sku:
            final_inventory = replace(
                original_inventory,
                version=original_inventory.version + 1,
            )
            inventory_mutations = ((
                "inventory",
                str(order.sku_id),
                original_inventory,
                final_inventory,
            ),)
        else:
            released = replace(
                original_inventory,
                qty_reserved=original_inventory.qty_reserved
                - effect.inventory_release_qty,
                version=original_inventory.version + 1,
            )
            reserved = replace(
                replacement_inventory,
                qty_reserved=replacement_inventory.qty_reserved + effect.qty,
                version=replacement_inventory.version + 1,
            )
            inventory_mutations = (
                ("inventory", str(order.sku_id), original_inventory, released),
                (
                    "inventory",
                    replacement_sku,
                    replacement_inventory,
                    reserved,
                ),
            )
        self._commit_after_sales_records(
            commit,
            extra_mutations=(
                ("orders", str(order.order_id), order, exchanged),
                (
                    "orders",
                    str(replacement_order.order_id),
                    None,
                    replacement_order,
                ),
                *inventory_mutations,
                ("exchanges", str(exchange.exchange_id), None, exchange),
            ),
            extra_invariants=(
                "returned-original",
                "replacement-world-grounded",
                "no-second-payment",
                "no-oversell",
            ),
        )

    def _commit_paid_cancellation(
        self,
        commit: AfterSalesCommit,
        effect: CommerceEffect,
        context: TrustedAfterSalesContext,
    ) -> None:
        order = context.order
        inventory = self._tables["inventory"].read(order.sku_id, caller=None)
        if not isinstance(inventory, InventoryRow):
            raise WorldError("paid cancellation requires indexed inventory row")
        if inventory.qty_reserved < effect.inventory_release_qty:
            raise WorldError("paid cancellation reservation underflow")
        if context.payment.record_digest != effect.payment_before_digest:
            raise AfterSalesContextConflict("payment changed before cancellation")
        if effect.packing_before_digest is not None and (
            context.packing is None
            or context.packing.record_digest != effect.packing_before_digest
        ):
            raise AfterSalesContextConflict("packing changed before cancellation")
        cancelled_order = replace(order, state=OrderState.CANCELLED)
        released_inventory = replace(
            inventory,
            qty_reserved=inventory.qty_reserved - effect.inventory_release_qty,
            version=inventory.version + 1,
        )
        refund: Receipt | None = None
        if effect.financial_effect == "refund":
            refund = _after_sales_refund_receipt(
                order,
                amount=effect.amount,
                tick=commit.logical_tick,
                idempotency_key=commit.operation.idempotency_key,
            )
            payment = derive_payment_resolution(
                cancelled_order,
                context.payment,
                outcome="refunded",
                refund_receipt=refund,
                original_actor="platform:psp",
                server_tick=commit.logical_tick,
                idempotency_key=f"payment:{commit.operation.idempotency_key}",
            )
        else:
            payment = derive_payment_resolution(
                cancelled_order,
                context.payment,
                outcome="voided",
                refund_receipt=None,
                original_actor="platform:psp",
                server_tick=commit.logical_tick,
                idempotency_key=f"payment:{commit.operation.idempotency_key}",
            )
        packing: PackingRecord | None = None
        if context.packing is not None:
            packing = derive_packing_transition(
                cancelled_order,
                payment,
                previous=context.packing,
                target_state="cancelled",
                original_actor="platform:fulfillment",
                server_tick=commit.logical_tick,
                idempotency_key=f"packing:{commit.operation.idempotency_key}",
            )
        mutations: list[tuple[str, str, Any | None, Any]] = [
            ("orders", str(order.order_id), order, cancelled_order),
            ("inventory", str(order.sku_id), inventory, released_inventory),
        ]
        if refund is not None:
            mutations.append(("ledger", str(refund.txn_id), None, refund))
        mutations.append(
            ("payment_states", payment_state_key(payment), None, payment)
        )
        if packing is not None:
            mutations.append(
                ("packing_records", packing_record_key(packing), None, packing)
            )
        for write in commit.writes:
            if write.table == "after_sales_policies":
                raise WorldError("after-sales command cannot publish policy")
            physical_key = physical_after_sales_record_key(write)
            mutations.append(("after_sales_records", physical_key, None, write))
        result_write = next(
            write
            for write in commit.writes
            if write.table == commit.operation.result_table
            and write.key == commit.operation.result_key
        )
        self._commit_simple_authority_transition(
            operation=commit.operation.operation,
            authority_action="world.apply_after_sales_intent",
            scope="apply_after_sales_intent",
            original_actor=commit.operation.actor_id,
            idempotency_key=commit.operation.idempotency_key,
            request_fingerprint=commit.operation.request_fingerprint,
            subject_id=commit.operation.order_id,
            outcome_table="after_sales_records",
            outcome_key=physical_after_sales_record_key(result_write),
            tick=commit.logical_tick,
            mutations=tuple(mutations),
            invariants=(
                "after-sales-context-cas",
                "payment-packing-causal-binding",
                "atomic-commercial-effect",
                "principal-provenance-preserved",
                "single-clock-advance",
            ),
        )

    def _commit_simple_authority_transition(
        self,
        *,
        operation: str,
        authority_action: str,
        scope: str,
        original_actor: str,
        idempotency_key: str,
        request_fingerprint: str,
        subject_id: str,
        outcome_table: str,
        outcome_key: str,
        tick: int,
        mutations: tuple[tuple[str, str, Any | None, Any], ...],
        invariants: tuple[str, ...],
    ) -> None:
        if tick != self._logical_time + 1:
            raise LogicalTimeError("authority transaction must advance one World tick")
        authority_key = authority_operation_key(
            scope, original_actor, idempotency_key
        )
        authority = AuthorityOperationRecord(
            operation_key=authority_key,
            scope=scope,
            actor_id=original_actor,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            outcome_table=outcome_table,
            outcome_key=outcome_key,
        )
        all_mutations = (
            *mutations,
            ("authority_operations", authority_key, None, authority),
        )
        applied: list[tuple[str, str, Any | None]] = []
        before_tick = self._logical_time
        commit_cursor = len(self._commit_journal)
        revisions = dict(self._order_state_revisions)
        try:
            for table_name, key, before, after in all_mutations:
                current = self._tables[table_name].read(key, caller=None)
                if current != before:
                    raise AfterSalesContextConflict(
                        f"{table_name}:{key} changed before authority commit"
                    )
                self._tables[table_name].write(key, after)
                applied.append((table_name, key, before))
            self._logical_time = tick
            diff_rows = [
                (
                    table_name,
                    key,
                    "create" if before is None else "update",
                    before,
                    after,
                )
                for table_name, key, before, after in all_mutations
            ]
            diff_rows.append(
                ("logical_time", "world", "update", before_tick, tick)
            )
            self._append_transaction_commit(
                _make_diff(operation, subject_id, diff_rows, invariants),
                authority_action=authority_action,
                actor_id=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
            )
        except BaseException:
            for table_name, key, before in reversed(applied):
                _restore_row(self._tables[table_name], key, before)
            self._logical_time = before_tick
            self._order_state_revisions = revisions
            del self._commit_journal[commit_cursor:]
            raise

    def reset(self, mode: Literal["E0", "E1"]) -> None:
        """Clear state per the reset policy.

        E0: clear between every episode.
        E1: clear only between batches (cross-episode persistence enabled).
        """
        strategy = E0Reset() if mode == "E0" else E1Reset()
        with self._lock:
            strategy.apply(self)
            self._mode = mode
            if mode == "E0":
                self._idempotency_cache.clear()
                self._supply_idempotency_cache.clear()
                self._allocation_idempotency_cache.clear()
                self._shipment_event_idempotency_cache.clear()
                self._shipment_resolution_idempotency_cache.clear()
                self._order_state_revisions.clear()
                self._logical_time = 0

    def snapshot(self) -> "WorldSnapshot":
        """Return a deep, frozen view of every table.

        Order across collections must be deterministic so the replay verifier
        can compare snapshots byte-for-byte.
        """
        with self._lock:
            return WorldSnapshot(
                catalog=tuple(deepcopy(row) for _, row in self._tables["catalog"].all()),
                inventory={
                    key: deepcopy(row) for key, row in self._tables["inventory"].all()
                },
                orders=tuple(deepcopy(row) for _, row in self._tables["orders"].all()),
                ledger=tuple(deepcopy(row) for _, row in self._tables["ledger"].all()),
                reputation={
                    key: deepcopy(row) for key, row in self._tables["reputation"].all()
                },
                reputation_settlements=tuple(
                    deepcopy(row)
                    for _, row in self._tables["reputation_settlements"].all()
                ),
                disputes=tuple(deepcopy(row) for _, row in self._tables["disputes"].all()),
                rulings=tuple(deepcopy(row) for _, row in self._tables["rulings"].all()),
                # Social tables — deterministic (Table.all() sorts by key), deep-copied.
                friendships=tuple(
                    deepcopy(row) for _, row in self._tables["friendships"].all()
                ),
                reviews=tuple(deepcopy(row) for _, row in self._tables["reviews"].all()),
                fulfillments=tuple(
                    deepcopy(row) for _, row in self._tables["fulfillments"].all()
                ),
                exchanges=tuple(
                    deepcopy(row) for _, row in self._tables["exchanges"].all()
                ),
                logical_time=self._logical_time,
                order_timelines=tuple(
                    deepcopy(row)
                    for _, row in self._tables["order_timelines"].all()
                ),
                order_groups=tuple(
                    deepcopy(row) for _, row in self._tables["order_groups"].all()
                ),
                shipments=tuple(
                    deepcopy(row) for _, row in self._tables["shipments"].all()
                ),
                search_sessions=tuple(
                    deepcopy(row)
                    for _, row in self._tables["search_sessions"].all()
                ),
                match_acceptances={
                    str(key): deepcopy(row)
                    for key, row in self._tables["match_acceptances"].all()
                },
                match_certificates=tuple(
                    deepcopy(row)
                    for _, row in self._tables["match_certificates"].all()
                ),
                supply_purchase_authorities=tuple(
                    deepcopy(row)
                    for _, row in self._tables[
                        "supply_purchase_authorities"
                    ].all()
                ),
                protocol_events=tuple(
                    deepcopy(row)
                    for _, row in self._tables["protocol_events"].all()
                ),
                protocol_receipts=tuple(
                    deepcopy(row)
                    for _, row in self._tables["protocol_receipts"].all()
                ),
                negotiation_events=tuple(
                    deepcopy(row)
                    for _, row in self._tables["negotiation_events"].all()
                ),
                negotiation_threads=tuple(
                    deepcopy(row)
                    for _, row in self._tables["negotiation_threads"].all()
                ),
                pricing_policy_revisions=tuple(
                    deepcopy(row)
                    for _, row in self._tables["pricing_policy_revisions"].all()
                ),
                persistent_cart_quote_requests=tuple(
                    deepcopy(row)
                    for _, row in self._tables[
                        "persistent_cart_quote_requests"
                    ].all()
                ),
                persistent_cart_quotes=tuple(
                    deepcopy(row)
                    for _, row in self._tables["persistent_cart_quotes"].all()
                ),
                order_state_revisions=deepcopy(self._order_state_revisions),
                # These protocol values are recursively immutable. Returning
                # the sealed rows directly avoids attempting to pickle their
                # MappingProxyType payloads while preserving snapshot safety.
                evidence_records=tuple(
                    row for _, row in self._tables["evidence_records"].all()
                ),
                mandate_authorities=tuple(
                    row for _, row in self._tables["mandate_authorities"].all()
                ),
                mandate_revisions=tuple(
                    row for _, row in self._tables["mandate_revisions"].all()
                ),
                listing_claims=tuple(
                    row for _, row in self._tables["listing_claims"].all()
                ),
                authority_operations=tuple(
                    row for _, row in self._tables["authority_operations"].all()
                ),
                payment_states=tuple(
                    deepcopy(row) for _, row in self._tables["payment_states"].all()
                ),
                packing_records=tuple(
                    deepcopy(row) for _, row in self._tables["packing_records"].all()
                ),
                after_sales_policies=tuple(
                    deepcopy(row)
                    for _, row in self._tables["after_sales_policies"].all()
                ),
                after_sales_records=tuple(
                    row for _, row in self._tables["after_sales_records"].all()
                ),
                governance_policies=tuple(
                    row for _, row in self._tables["governance_policies"].all()
                ),
                governance_records=tuple(
                    row for _, row in self._tables["governance_records"].all()
                ),
            )

    def apply(self, initial_state: dict[str, Any]) -> None:
        """Bulk-load tables from a scenario's ``initial_state`` dict.

        Args:
            initial_state: ``{"catalog": [...], ...}`` mapping each table name
                to its optional seed rows.
        """
        with self._lock:
            revision_seed = initial_state.get("order_state_revisions")
            for table_name, rows in initial_state.items():
                if table_name == "order_state_revisions":
                    continue
                if table_name == "logical_time":
                    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
                        raise LogicalTimeError(
                            "seed logical_time must be a non-negative integer"
                        )
                    self._logical_time = rows
                    continue
                table = self._table(table_name)
                table.clear()
                if table_name == "orders":
                    self._order_state_revisions.clear()
                if rows is None:
                    continue
                if isinstance(rows, dict):
                    seed_items = rows.items()
                    if table_name == "governance_policies":
                        seed_items = sorted(
                            seed_items,
                            key=lambda item: (
                                item[1].kind,
                                item[1].stable_id,
                                item[1].revision,
                            ),
                        )
                    elif table_name == "governance_records":
                        seed_items = sorted(
                            seed_items,
                            key=lambda item: (
                                item[1].kind,
                                item[1].stable_id,
                                item[1].version,
                            ),
                        )
                    for key, value in seed_items:
                        _validate_protocol_seed_row(table_name, value)
                        table.write(key, value)
                        if table_name == "orders":
                            self._order_state_revisions[str(key)] = 1
                    continue
                seed_rows = rows
                if table_name == "governance_policies":
                    seed_rows = sorted(
                        rows,
                        key=lambda row: (row.kind, row.stable_id, row.revision),
                    )
                elif table_name == "governance_records":
                    seed_rows = sorted(
                        rows,
                        key=lambda row: (row.kind, row.stable_id, row.version),
                    )
                for row in seed_rows:
                    _validate_protocol_seed_row(table_name, row)
                    key = _row_key(table_name, row)
                    table.write(key, row)
                    if table_name == "orders":
                        self._order_state_revisions[str(key)] = 1
            if revision_seed is not None:
                if not isinstance(revision_seed, dict):
                    raise ValueError("order_state_revisions seed must be a mapping")
                existing_order_ids = {
                    str(key) for key, _ in self._tables["orders"].all()
                }
                for order_id, revision in revision_seed.items():
                    key = str(order_id)
                    if key not in existing_order_ids:
                        raise ValueError(
                            "order_state_revisions references a missing order: "
                            f"{key!r}"
                        )
                    if (
                        isinstance(revision, bool)
                        or not isinstance(revision, int)
                        or revision < 1
                    ):
                        raise ValueError(
                            "order state revisions must be positive integers"
                        )
                    self._order_state_revisions[key] = revision
            _validate_seeded_protocol_records(
                cast(ProtocolEventTable, self._tables["protocol_events"]),
                cast(ProtocolReceiptTable, self._tables["protocol_receipts"]),
            )
            _validate_seeded_negotiations(
                cast(
                    NegotiationEventTable,
                    self._tables["negotiation_events"],
                ),
                cast(
                    NegotiationThreadTable,
                    self._tables["negotiation_threads"],
                ),
            )
            _validate_seeded_evidence_contracts(
                catalog=cast(CatalogTable, self._tables["catalog"]),
                evidence=cast(EvidenceRecordTable, self._tables["evidence_records"]),
                authorities=cast(
                    MandateAuthorityTable, self._tables["mandate_authorities"]
                ),
                revisions=cast(
                    MandateRevisionTable, self._tables["mandate_revisions"]
                ),
                claims=cast(ListingClaimTable, self._tables["listing_claims"]),
                operations=cast(
                    AuthorityOperationTable, self._tables["authority_operations"]
                ),
                negotiation_events=cast(
                    NegotiationEventTable, self._tables["negotiation_events"]
                ),
                pricing_policies=cast(
                    PricingPolicyRevisionTable,
                    self._tables["pricing_policy_revisions"],
                ),
                cart_quote_requests=cast(
                    PersistentCartQuoteRequestTable,
                    self._tables["persistent_cart_quote_requests"],
                ),
                cart_quotes=cast(
                    PersistentCartQuoteTable,
                    self._tables["persistent_cart_quotes"],
                ),
                order_groups=cast(OrderGroupTable, self._tables["order_groups"]),
                ledger=cast(LedgerTable, self._tables["ledger"]),
                governance_policies=cast(
                    GovernancePolicyTable, self._tables["governance_policies"]
                ),
                governance_records=cast(
                    GovernanceRecordTable, self._tables["governance_records"]
                ),
                logical_time=self._logical_time,
            )

    def _table(self, table: str) -> Table[Any, Any]:
        try:
            return self._tables[table]
        except KeyError as exc:
            raise TableNotFound(f"unknown world table: {table}") from exc


def _normalize_after_sales_policy_intent(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the compact policy content that World is allowed to accept."""

    fields = {
        "policy_id",
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
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WorldError(
            "after-sales policy intent fields must be exactly: "
            + ", ".join(sorted(fields))
        )
    return {key: deepcopy(value[key]) for key in sorted(fields)}


def _after_sales_context_hint(commit: AfterSalesCommit) -> Mapping[str, Any] | None:
    """Reconstruct compact fields required to reload the exact CAS context."""

    operation = commit.operation.operation
    if operation not in {
        "request_return",
        "submit_dispute_evidence",
        "respond_to_dispute",
        "request_exchange",
        "authorize_exchange",
        "deny_exchange",
        "complete_exchange",
    }:
        return None
    result = next(
        write.value
        for write in commit.writes
        if write.table == commit.operation.result_table
        and write.key == commit.operation.result_key
    )
    if operation == "request_return":
        return {
            "op": operation,
            "order_id": result.binding.order_id,
            "requested_qty": result.requested_qty,
            "reason": result.reason,
            "evidence_ids": list(result.evidence_ids),
        }
    if operation == "submit_dispute_evidence":
        return {
            "op": operation,
            "order_id": result.binding.order_id,
            "dispute_id": result.dispute_id,
            "evidence_id": result.evidence_id,
        }
    if operation == "respond_to_dispute":
        citations = result.facts.get("evidence", ())
        return {
            "op": operation,
            "order_id": result.binding.order_id,
            "dispute_id": result.dispute_id,
            "evidence_ids": [row["record_id"] for row in citations],
            "position": result.facts["position"],
        }
    if operation == "request_exchange":
        return {"replacement_sku_id": result.replacement_sku_id}
    return {"case_id": result.case_id}


def _after_sales_context_evidence_digests(
    commit: AfterSalesCommit,
) -> Mapping[str, str] | None:
    """Recover exact historical EvidenceRecord digests bound by one commit."""

    operation = commit.operation.operation
    if operation not in {
        "request_return",
        "submit_dispute_evidence",
        "respond_to_dispute",
    }:
        return None
    result = next(
        write.value
        for write in commit.writes
        if write.table == commit.operation.result_table
        and write.key == commit.operation.result_key
    )
    if operation == "request_return":
        return dict(zip(result.evidence_ids, result.evidence_digests, strict=True))
    if operation == "submit_dispute_evidence":
        return {result.evidence_id: result.source_digest}
    citations = result.facts.get("evidence", ())
    return {row["record_id"]: row["record_digest"] for row in citations}


def _charge_receipt_for_order(
    ledger: Table[Any, Any],
    order_id: str,
    *,
    payment: PaymentStateRecord | None,
    history: tuple[PaymentStateRecord, ...] = (),
) -> Receipt | None:
    target_digest: str | None = None
    candidates = history or (() if payment is None else (payment,))
    for row in reversed(candidates):
        if row.state == "captured":
            target_digest = row.ledger_receipt_digest
            break
    receipts = [
        cast(Receipt, row)
        for _, row in ledger.all()
        if str(cast(Receipt, row).order_id) == order_id
    ]
    if target_digest is not None:
        for receipt in receipts:
            if authoritative_payment_receipt_digest(receipt) == target_digest:
                return receipt
        raise WorldError("captured payment's authoritative receipt is missing")
    if payment is not None and payment.state == "authorized":
        return None
    charges = [
        receipt
        for receipt in receipts
        if receipt.effect == "charge"
    ]
    if len(charges) > 1:
        raise WorldError("order has multiple candidate charge receipts")
    return charges[0] if charges else None


def _receipt_tick(receipt: Receipt | None) -> int:
    if receipt is None:
        return 0
    prefix = "world-tick:"
    if receipt.ts.startswith(prefix):
        try:
            return max(0, int(receipt.ts.removeprefix(prefix)))
        except ValueError:
            pass
    return 0


def _after_sales_policy_for_context(
    tables: AfterSalesTables,
    *,
    merchant_id: str,
    binding: Any | None,
) -> AfterSalesPolicyRevision:
    """Load the exact policy bound to an order, or the current pre-binding row."""

    if binding is None:
        policy = tables.latest_policy(merchant_id, caller="world")
        if policy is None:
            raise AfterSalesCoreTransitionError(
                "merchant has no published after-sales policy"
            )
        return policy
    matches = [
        cast(AfterSalesPolicyRevision, row)
        for _, row in tables.internal_all("after_sales_policies")
        if isinstance(row, AfterSalesPolicyRevision)
        and row.merchant_id == merchant_id
        and row.revision == binding.policy_revision
        and row.policy_digest == binding.policy_digest
    ]
    if len(matches) != 1:
        raise WorldError("order binding does not resolve to one policy revision")
    return matches[0]


def _after_sales_refund_receipt(
    order: Order,
    *,
    amount: int,
    tick: int,
    idempotency_key: str,
) -> Receipt:
    if amount <= 0 or order.qty <= 0:
        raise WorldError("after-sales refund amount or quantity is invalid")
    # Receipt keeps the order quantity for exact identity.  The per-unit
    # amount may therefore need sub-cent precision so that, for example, a
    # one-cent refund over quantity three is not rounded to zero or three
    # cents.  Preserve enough Decimal precision and verify the total roundtrip.
    with localcontext() as context:
        context.prec = max(28, len(str(amount)) + len(str(order.qty)) + 20)
        unit = Decimal(amount) / Decimal(100) / Decimal(order.qty)
        recovered = (unit * order.qty * Decimal(100)).quantize(Decimal("1"))
    if int(recovered) != amount:
        raise WorldError("after-sales refund amount cannot roundtrip through receipt")
    identity = after_sales_digest(
        {
            "order_id": str(order.order_id),
            "amount": amount,
            "tick": tick,
            "idempotency_key": idempotency_key,
        }
    )
    return Receipt(
        txn_id=TxnId(f"txn:z-after-sales-refund:{identity[:24]}"),
        ts=f"world-tick:{tick}",
        order_id=order.order_id,
        buyer_id=order.buyer_id,
        merchant_id=order.merchant_id,
        sku_id=order.sku_id,
        qty=order.qty,
        price=Money(unit, order.agreed_price.currency),
        idempotency_key=f"after-sales-refund:{idempotency_key}",
        effect="refund",
    )


def _effects_for_persisted_after_sales(value: Any) -> tuple[CommerceEffect, ...]:
    if isinstance(value, PaidCancellationRecord):
        return (
            CommerceEffect(
                kind="cancel_paid_order",
                order_id=value.binding.order_id,
                sku_id=value.binding.sku_id,
                qty=value.inventory_release_qty,
                amount=value.refund_amount,
                currency=value.binding.currency,
                inventory_release_qty=value.inventory_release_qty,
                payment_before_digest=value.payment_digest,
                packing_before_digest=value.packing_digest,
                fulfillment_stage=value.fulfillment_stage,
                financial_effect=value.financial_effect,
            ),
        )
    return ()


def _restore_row(table: Table[Any, Any], key: Any, before: Any | None) -> None:
    """Restore one in-memory row while unwinding a failed atomic transaction."""
    if before is None:
        table.delete(key)
    else:
        table.write(key, before)


def _captured_return_window(listing: Any | None) -> int | None:
    """Read the opt-in tick window once, at the authoritative lifecycle edge."""
    if listing is None:
        return None
    attributes = getattr(listing, "attributes", {}) or {}
    if "return_window_ticks" not in attributes:
        return None
    value = attributes["return_window_ticks"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OrderNotSettleable(
            "listing return_window_ticks must be a positive integer when present"
        )
    return value


def _enforce_return_window(
    timeline: OrderTimeline | None,
    *,
    event_tick: int,
) -> None:
    """Authorize a refund event against captured, World-owned timing only."""
    if timeline is None or timeline.return_window_ticks is None:
        return
    if timeline.dispatched_at_tick is None:
        raise ReturnWindowClosed(
            f"order {timeline.order_id} has an explicit return window but was not dispatched"
        )
    deadline = timeline.dispatched_at_tick + timeline.return_window_ticks
    if event_tick > deadline:
        raise ReturnWindowClosed(
            f"return window closed for order {timeline.order_id}: "
            f"authorization tick {event_tick} exceeds deadline tick {deadline}"
        )


def _is_platform_actor(actor: str) -> bool:
    return actor == "platform" or actor.startswith("platform:")


def _authorize_lifecycle_actor(
    order: Order,
    actor: str,
    allowed_sides: frozenset[str],
) -> None:
    """Bind a transition to the exact buyer/merchant id, with platform privilege."""
    if _is_platform_actor(actor):
        return
    allowed: set[str] = set()
    if "buyer" in allowed_sides:
        allowed.add(str(order.buyer_id))
    if "merchant" in allowed_sides:
        allowed.add(str(order.merchant_id))
    if actor not in allowed:
        raise LifecycleAuthorizationError(
            f"actor {actor!r} is not authorized for order {order.order_id}"
        )


def _validate_open_dispute_shape(dispute: Dispute, by_actor: str) -> None:
    if dispute.state != DisputeState.OPEN:
        raise DisputeNotActionable("a new dispute must start in OPEN state")
    if by_actor != str(dispute.filed_by):
        raise LifecycleAuthorizationError(
            f"actor {by_actor!r} cannot file as {dispute.filed_by!s}"
        )
    if not dispute.reason.strip():
        raise DisputeNotActionable("dispute reason must not be blank")


def _same_dispute_claim(existing: Dispute, requested: Dispute) -> bool:
    """Compare immutable claim fields while allowing the stored state to advance."""
    return (
        existing.dispute_id == requested.dispute_id
        and existing.order_id == requested.order_id
        and existing.filed_by == requested.filed_by
        and existing.against == requested.against
        and existing.reason == requested.reason
    )


def _validate_dispute_order(dispute: Dispute, order: Order | None) -> None:
    if order is None:
        raise DisputeNotActionable(f"order {dispute.order_id} does not exist")
    if order.state not in _DISPUTABLE:
        raise DisputeNotActionable(
            f"order {order.order_id} in state {order.state.value!r} is not disputable"
        )
    parties = {str(order.buyer_id), str(order.merchant_id)}
    if (
        str(dispute.filed_by) not in parties
        or str(dispute.against) not in parties
        or dispute.filed_by == dispute.against
    ):
        raise DisputeNotActionable(
            "dispute filed_by/against must be the order's opposing parties"
        )


def _validate_ruling(
    ruling: Ruling,
    dispute: Dispute,
    order: Order | None,
) -> None:
    if order is None:
        raise DisputeNotActionable(
            f"dispute {dispute.dispute_id} references missing order {dispute.order_id}"
        )
    if str(ruling.in_favor_of) not in {
        str(dispute.filed_by),
        str(dispute.against),
    }:
        raise DisputeNotActionable("ruling winner must be one of the dispute parties")
    if not ruling.rationale.strip():
        raise DisputeNotActionable("ruling rationale must not be blank")
    refund = ruling.refund_amount
    if refund is None:
        return
    if refund.amount < 0:
        raise DisputeNotActionable("ruling refund amount must be non-negative")
    if refund.currency != order.agreed_price.currency:
        raise DisputeNotActionable("ruling refund currency must match the order")
    max_refund = order.agreed_price.amount * order.qty
    if refund.amount > max_refund:
        raise DisputeNotActionable(
            f"ruling refund amount exceeds order value {max_refund}"
        )


def _row_key(table: str, row: Any) -> Any:
    key_attrs = {
        "catalog": "sku_id",
        "inventory": "sku_id",
        "orders": "order_id",
        "ledger": "txn_id",
        "reputation": "merchant_id",
        "reputation_settlements": "event_id",
        "disputes": "dispute_id",
        "rulings": "ruling_id",
        "friendships": "buyer_id",
        "reviews": "review_id",
        "fulfillments": "order_id",
        "exchanges": "exchange_id",
        "order_timelines": "order_id",
        "order_groups": "order_group_id",
        "shipments": "shipment_id",
        "search_sessions": "session_id",
        "negotiation_events": "event_id",
        "negotiation_threads": "negotiation_id",
        "persistent_cart_quotes": "quote_id",
        "protocol_events": "event_id",
        "protocol_receipts": "receipt_id",
        "mandate_authorities": "mandate_id",
        "listing_claims": "claim_id",
        "authority_operations": "operation_key",
        # Acceptance rows use a buyer-scoped idempotency key rather than a
        # field directly exposed by the protocol object.  Seed/replay callers
        # should not normally create these, but the mapping is deterministic.
        "match_certificates": "cert_id",
        "supply_purchase_authorities": "authority_id",
    }
    attr = key_attrs.get(table)
    if attr is None:
        if table in {"governance_policies", "governance_records"}:
            from world.market_governance_persistence import envelope_key

            return envelope_key(row)
        if table == "payment_states":
            return payment_state_key(row)
        if table == "packing_records":
            return packing_record_key(row)
        if table == "after_sales_policies":
            return f"{row.merchant_id}:{row.revision}"
        if table == "after_sales_records":
            return physical_after_sales_record_key(row)
        if table == "match_acceptances":
            return _match_acceptance_key(row.buyer_id, row.idempotency_key)
        if table == "evidence_records":
            return evidence_record_key(row.record_id, row.version)
        if table == "mandate_revisions":
            return mandate_revision_key(row.mandate_id, row.revision)
        if table == "pricing_policy_revisions":
            return pricing_policy_revision_key(
                row.market_id, row.merchant_id, row.policy_id, row.revision
            )
        raise TableNotFound(f"unknown world table: {table}")
    try:
        return getattr(row, attr)
    except AttributeError as exc:
        raise ValueError(f"{table} row missing key attribute {attr!r}") from exc


ORDER_OPERATION_REFERENCE_SCHEMA = "cwe.order-operation-reference.v1"
PROTOCOL_OPERATION_EFFECT_REFERENCE_SCHEMA = (
    "cwe.protocol-operation-effect-reference.v1"
)

_PROTOCOL_EVENT_OPERATIONS: dict[str, tuple[str, str]] = {
    # event kind -> (World operation, required recipient side)
    "payment.settle": ("settle_order", "buyer"),
    "fulfillment.dispatch": ("dispatch_order", "merchant"),
    "payment.refund": ("refund_order", "merchant"),
}

_PROTOCOL_OPERATION_STATES: dict[str, frozenset[OrderState]] = {
    "settle_order": _SETTLEABLE,
    "dispatch_order": frozenset(
        {OrderState.SETTLED, OrderState.PARTIALLY_SETTLED}
    ),
    "refund_order": _REFUNDABLE,
}


def registered_protocol_operation(event_kind: str) -> tuple[str, str]:
    """Return the core World operation and required recipient side.

    Protocol callbacks are executable only through this World-owned registry.
    Keeping the lookup public lets evidence validators and other transports
    verify the same contract without copying benchmark-local operation maps.
    """

    try:
        return _PROTOCOL_EVENT_OPERATIONS[event_kind]
    except KeyError as exc:
        raise ProtocolEventSchemaError(
            f"protocol event kind {event_kind!r} has no registered operation"
        ) from exc


def protocol_operation_outcome_identity(
    event: ProtocolEvent,
) -> tuple[str, str, str]:
    """Return ``(operation, table, key)`` for a registered event effect."""

    validate_protocol_event(event)
    operation, _required_side = registered_protocol_operation(event.event_kind)
    if operation == "settle_order":
        return operation, "ledger", f"txn:protocol:{event.event_id}"
    if operation == "refund_order":
        return operation, "ledger", f"refund:protocol:{event.event_id}"
    if operation == "dispatch_order":
        return operation, "shipments", f"shipment:{event.binding.order_id}"
    raise ProtocolEventSchemaError(
        f"unsupported registered protocol operation {operation!r}"
    )


def protocol_operation_effect_idempotency_key(event: ProtocolEvent) -> str:
    """Return the World-owned idempotency identity for one event effect."""

    validate_protocol_event(event)
    return f"protocol-effect:{event.event_digest}"


def order_operation_reference_digest(order: Order, revision: int) -> str:
    """Digest one exact authoritative order row and its World revision."""

    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("order operation reference revision must be positive")
    return canonical_digest({
        "schema_id": ORDER_OPERATION_REFERENCE_SCHEMA,
        "order_id": str(order.order_id),
        "buyer_id": str(order.buyer_id),
        "merchant_id": str(order.merchant_id),
        "sku_id": str(order.sku_id),
        "qty": order.qty,
        "unit_price_cents": _money_cents(order.agreed_price),
        "currency": order.agreed_price.currency,
        "state": order.state.value,
        "request_order": order.request_order,
        "state_revision": revision,
    })


def order_operation_reference_digest_from_row(
    row: Mapping[str, Any],
    revision: int,
) -> str:
    """Digest a serialized authoritative order using the core order contract.

    Evidence verifiers operate on transport-neutral World snapshots.  This
    adapter keeps their operation-reference check in the World module instead
    of reimplementing the order digest or importing a service-private coercer.
    It accepts only the exact passive ``Order`` row shape emitted by World.
    """

    required = {
        "order_id",
        "buyer_id",
        "merchant_id",
        "sku_id",
        "qty",
        "agreed_price",
        "state",
        "request_order",
    }
    if not isinstance(row, Mapping) or set(row) != required:
        raise ValueError("serialized order has an invalid field set")
    price = row.get("agreed_price")
    if not isinstance(price, Mapping) or set(price) != {"amount", "currency"}:
        raise ValueError("serialized order has an invalid agreed price")
    qty = row.get("qty")
    request_order = row.get("request_order")
    if (
        isinstance(qty, bool)
        or not isinstance(qty, int)
        or qty <= 0
        or isinstance(request_order, bool)
        or not isinstance(request_order, int)
        or request_order < 0
    ):
        raise ValueError("serialized order has invalid quantity or request order")
    try:
        order = Order(
            order_id=OrderId(str(row["order_id"])),
            buyer_id=AgentId(str(row["buyer_id"])),
            merchant_id=AgentId(str(row["merchant_id"])),
            sku_id=SkuId(str(row["sku_id"])),
            qty=qty,
            agreed_price=Money(
                amount=Decimal(str(price["amount"])),
                currency=str(price["currency"]),
            ),
            state=OrderState(str(row["state"])),
            request_order=request_order,
        )
    except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("serialized order cannot be normalized") from exc
    return order_operation_reference_digest(order, revision)


def protocol_operation_effect_reference_digest(
    event: ProtocolEvent,
    *,
    operation: str,
    outcome_table: str,
    outcome_key: str,
) -> str:
    """Reference one committed core-operation outcome from a process receipt.

    The reference is intentionally based on immutable authority identities,
    not a scenario score or agent claim.  World creates it only after the
    expected outcome row exists.  The event digest binds the effect to one
    exact delivery; the table/key binds it to a durable business fact.
    """

    return canonical_digest(
        {
            "schema_id": PROTOCOL_OPERATION_EFFECT_REFERENCE_SCHEMA,
            "event_digest": event.event_digest,
            "operation": operation,
            "outcome_table": outcome_table,
            "outcome_key": outcome_key,
        }
    )


def _protocol_effect_idempotency_key(event: ProtocolEvent) -> str:
    return protocol_operation_effect_idempotency_key(event)


def _protocol_process_receipt_id(
    event: ProtocolEvent,
    *,
    actor_id: str,
    idempotency_key: str,
) -> str:
    identity = canonical_digest(
        {
            "schema_id": "cwe.protocol-process-receipt-identity.v1",
            "binding_digest": event.binding.binding_digest,
            "event_digest": event.event_digest,
            "actor_id": actor_id,
            "idempotency_key": idempotency_key,
        }
    )
    return f"protocol-receipt:{identity}"


def _protocol_process_receipt_replay(
    receipts: tuple[ProtocolEventReceipt, ...],
    *,
    event: ProtocolEvent,
    actor_id: str,
    reason: str,
    idempotency_key: str,
) -> ProtocolEventReceipt | None:
    for receipt in receipts:
        same_key = (
            receipt.actor_id == actor_id
            and receipt.idempotency_key == idempotency_key
        )
        same_event = receipt.event_digest == event.event_digest
        if same_key:
            if (
                same_event
                and receipt.decision == "process"
                and receipt.reason == reason
            ):
                return receipt
            raise IdempotencyConflict(
                "protocol process idempotency key was reused with different intent"
            )
        if same_event:
            from protocol.event_receipts import ProtocolEventStaleError

            # A receipt consumes the event digest.  Only the exact same actor
            # idempotency key may replay the original process receipt above;
            # a fresh key is a new attempt against an already-decided event,
            # so it is stale rather than an idempotency-key conflict.
            raise ProtocolEventStaleError(
                "protocol event was already processed or decided"
            )
    return None


def _validate_protocol_process_precondition(
    event: ProtocolEvent,
    *,
    order: Order,
    revision: int,
    logical_time: int,
    original_actor: str,
    certificates: MatchCertificateTable,
) -> str:
    validate_protocol_event(event)
    _validate_event_order_binding(event, order)
    if original_actor != event.binding.recipient_id:
        raise LifecycleAuthorizationError(
            "protocol operation actor is not the event recipient"
        )
    operation, required_side = registered_protocol_operation(event.event_kind)
    if original_actor.split(":", 1)[0] != required_side:
        raise ProtocolEventAuthorityError(
            "protocol event recipient role cannot execute the registered operation"
        )
    if logical_time > event.expires_at_tick:
        raise ProtocolEventStaleError("cannot process an expired protocol event")
    if (
        order.state.value != event.required_order_state
        or revision != event.required_state_revision
    ):
        raise ProtocolEventStaleError(
            "cannot process stale protocol event state or revision"
        )
    # The event reference was bound to the authoritative order state at issue
    # time.  Once that required state or revision is stale, recomputing the
    # reference from the newer order would misclassify an ordinary concurrent
    # callback as schema corruption.  Persisted event integrity is validated at
    # publication and replay; reference authority is checked here only while
    # the event precondition is still current.
    _validate_protocol_event_reference(
        event,
        order,
        revision=revision,
        logical_time=logical_time,
        certificates=certificates,
        expired_certificate_is_stale=True,
    )
    if order.state not in _PROTOCOL_OPERATION_STATES[operation]:
        allowed = ", ".join(
            sorted(state.value for state in _PROTOCOL_OPERATION_STATES[operation])
        )
        raise ProtocolEventSchemaError(
            f"protocol operation {operation!r} cannot run from "
            f"{order.state.value!r}; allowed states: {allowed}"
        )
    return operation


def _protocol_payment_receipt(
    event: ProtocolEvent,
    *,
    order: Order,
    logical_time: int,
    refund: bool,
    paid_qty: int | None = None,
) -> Receipt:
    qty = order.qty if paid_qty is None else paid_qty
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        raise OrderNotRefundable(
            f"order {order.order_id} has no paid quantity for a protocol refund"
        )
    prefix = "refund:protocol" if refund else "txn:protocol"
    return Receipt(
        txn_id=TxnId(f"{prefix}:{event.event_id}"),
        ts=f"world-tick:{logical_time}",
        order_id=order.order_id,
        buyer_id=order.buyer_id,
        merchant_id=order.merchant_id,
        sku_id=order.sku_id,
        qty=qty,
        price=order.agreed_price,
        idempotency_key=_protocol_effect_idempotency_key(event),
        effect="refund" if refund else "charge",
    )


def _validate_event_order_binding(event: ProtocolEvent, order: Order) -> None:
    binding = event.binding
    if (
        binding.order_id != str(order.order_id)
        or binding.buyer_id != str(order.buyer_id)
        or binding.merchant_id != str(order.merchant_id)
        or binding.authority_id != "platform:events"
    ):
        raise ProtocolEventSchemaError(
            "protocol event binding disagrees with authoritative order parties "
            "or event authority"
        )


def _validate_protocol_event_reference(
    event: ProtocolEvent,
    order: Order,
    *,
    revision: int,
    logical_time: int,
    certificates: MatchCertificateTable,
    expired_certificate_is_stale: bool = False,
) -> None:
    if event.reference_kind == "operation":
        expected = order_operation_reference_digest(order, revision)
        if event.reference_digest != expected:
            raise ProtocolEventSchemaError(
                "protocol event operation reference is not authoritative"
            )
        return

    matching = [
        certificate
        for _, certificate in certificates.all()
        if certificate.certificate_digest == event.reference_digest
    ]
    if len(matching) != 1:
        raise ProtocolEventSchemaError(
            "protocol event certificate reference must resolve exactly once"
        )
    certificate = matching[0]
    if (
        certificate.order_id != str(order.order_id)
        or certificate.buyer_id != str(order.buyer_id)
        or certificate.merchant_id != str(order.merchant_id)
        or certificate.sku_id != str(order.sku_id)
        or certificate.qty != order.qty
        or certificate.currency != order.agreed_price.currency
        or certificate.unit_price_cents != _money_cents(order.agreed_price)
    ):
        raise ProtocolEventSchemaError(
            "protocol event certificate does not bind the authoritative order"
        )
    if logical_time < certificate.issued_at_tick:
        raise ProtocolEventSchemaError(
            "protocol event certificate is not valid at World logical time"
        )
    if logical_time >= certificate.expires_at_tick:
        if expired_certificate_is_stale:
            raise ProtocolEventStaleError(
                "protocol event certificate reference is stale"
            )
        raise ProtocolEventSchemaError(
            "protocol event certificate is not valid at World logical time"
        )


def _validate_protocol_seed_row(table_name: str, row: Any) -> None:
    if table_name == "protocol_events":
        validate_protocol_event(row)
    elif table_name == "protocol_receipts":
        validate_protocol_event_receipt(row)
    elif table_name == "negotiation_events":
        validate_negotiation_event(row)
    elif table_name == "negotiation_threads":
        validate_negotiation_thread(row)
    elif table_name == "pricing_policy_revisions":
        validate_pricing_policy_revision(row)
    elif table_name == "persistent_cart_quotes":
        if not isinstance(row, PersistentCartQuote):
            raise ValueError(
                "persistent cart quote seed row must be PersistentCartQuote"
            )
        validate_persistent_cart_quote(row)
    elif table_name == "persistent_cart_quote_requests":
        if not isinstance(row, PersistentCartQuoteRequest):
            raise ValueError(
                "persistent cart quote request seed row must be "
                "PersistentCartQuoteRequest"
            )
        validate_persistent_cart_quote_request(row)
    elif table_name == "supply_purchase_authorities":
        validate_supply_purchase_authority(row)
    elif table_name == "catalog":
        if not isinstance(row, Listing):
            raise ValueError("catalog seed row must be a Listing")
        reject_embedded_authority_records(row.attributes or {})
    elif table_name == "evidence_records":
        validate_evidence_record(row)
    elif table_name == "mandate_authorities":
        if not isinstance(row, MandateRevisionAuthority):
            raise ValueError(
                "mandate authority seed row must be MandateRevisionAuthority"
            )
        validate_mandate_authority_registration(
            None, row, original_actor=row.principal_id
        )
    elif table_name == "mandate_revisions":
        validate_mandate_revision(row)
    elif table_name == "listing_claims":
        validate_listing_claim(row)
    elif table_name == "authority_operations":
        if not isinstance(row, AuthorityOperationRecord):
            raise ValueError(
                "authority operation seed row must be AuthorityOperationRecord"
            )


def _validate_seeded_protocol_records(
    event_table: ProtocolEventTable,
    receipt_table: ProtocolReceiptTable,
) -> None:
    events_by_binding: dict[str, list[ProtocolEvent]] = {}
    for _, event in event_table.all():
        events_by_binding.setdefault(event.binding.binding_digest, []).append(event)
    for binding_digest, events in events_by_binding.items():
        ordered_events = tuple(sorted(events, key=lambda row: row.sequence))
        binding = ordered_events[0].binding
        if binding.binding_digest != binding_digest:
            raise ProtocolEventSchemaError("seeded protocol event binding mismatch")
        replay_protocol_events(binding, ordered_events)
        receipts = tuple(
            sorted(
                (
                    receipt
                    for _, receipt in receipt_table.all()
                    if receipt.binding.binding_digest == binding_digest
                ),
                key=lambda row: (row.logical_tick, row.receipt_id),
            )
        )
        replay_protocol_receipts(binding, ordered_events, receipts)
    orphan_receipts = [
        receipt.receipt_id
        for _, receipt in receipt_table.all()
        if receipt.binding.binding_digest not in events_by_binding
    ]
    if orphan_receipts:
        raise ProtocolEventSchemaError(
            f"seeded protocol receipts have no event stream: {orphan_receipts!r}"
        )


def _validate_seeded_negotiations(
    event_table: NegotiationEventTable,
    thread_table: NegotiationThreadTable,
) -> None:
    """Strictly replay every seeded negotiation and verify its materialization."""

    negotiation_ids = sorted(
        {event.negotiation_id for _, event in event_table.all()}
        | {thread.negotiation_id for _, thread in thread_table.all()}
    )
    for negotiation_id in negotiation_ids:
        events = event_table.by_thread(negotiation_id)
        thread = thread_table.read(negotiation_id, caller=None)
        if not events or thread is None:
            raise NegotiationSchemaError(
                "seeded negotiation requires both events and a materialized thread"
            )
        first = events[0]
        binding = NegotiationBinding(
            negotiation_id=first.negotiation_id,
            buyer_id=first.buyer_id,
            merchant_id=first.merchant_id,
            offer_id=first.offer_id,
            sku_id=first.sku_id,
            listing_digest=first.listing_digest,
            listing_revision=first.listing_revision,
            currency=first.currency,
            qty=first.qty,
            max_rounds=first.max_rounds,
            opened_at_tick=first.opened_at_tick,
            expires_at_tick=first.expires_at_tick,
        )
        replayed = replay_negotiation_events(binding, events).thread
        if replayed != thread:
            raise NegotiationSchemaError(
                "seeded negotiation thread does not match strict event replay"
            )


def _validate_seeded_evidence_contracts(
    *,
    catalog: CatalogTable,
    evidence: EvidenceRecordTable,
    authorities: MandateAuthorityTable,
    revisions: MandateRevisionTable,
    claims: ListingClaimTable,
    operations: AuthorityOperationTable,
    negotiation_events: NegotiationEventTable,
    pricing_policies: PricingPolicyRevisionTable,
    cart_quote_requests: PersistentCartQuoteRequestTable,
    cart_quotes: PersistentCartQuoteTable,
    order_groups: OrderGroupTable,
    ledger: LedgerTable,
    governance_policies: GovernancePolicyTable,
    governance_records: GovernanceRecordTable,
    logical_time: int,
) -> None:
    """Validate all cross-table authority joins after scenario seeding.

    Scenarios may seed valid domain records, but they do not get a weaker
    representation. The same ownership, ACL, digest, subject, and authority
    constraints used by live World writes are checked over the materialized
    seed state.
    """

    for _, listing in catalog.all():
        reject_embedded_authority_records(listing.attributes or {})

    for _, record in evidence.all():
        validate_evidence_record(record)
        if record.issued_at_tick > logical_time:
            raise LogicalTimeError(
                "seeded evidence cannot be issued after World logical time"
            )

    known_mandates: set[str] = set()
    for mandate_id, authority in authorities.all():
        known_mandates.add(str(mandate_id))
        history = revisions.by_mandate(authority.mandate_id)
        if validate_mandate_revision_sequence(history, authority) != history:
            raise WorldError("seeded mandate history contains duplicate revisions")
        if history and history[-1].logical_tick > logical_time:
            raise LogicalTimeError(
                "seeded mandate revision cannot be after World logical time"
            )
    orphan_mandates = sorted(
        {
            revision.mandate_id
            for _, revision in revisions.all()
            if revision.mandate_id not in known_mandates
        }
    )
    if orphan_mandates:
        raise WriteNotAuthorized(
            f"seeded mandate revisions have no authority: {orphan_mandates!r}"
        )

    for _, claim in claims.all():
        claim_listing = catalog.read(SkuId(claim.listing_id), caller=None)
        previous: ListingClaim | None = None
        for end, version in enumerate(claim.versions, start=1):
            prefix = ListingClaim(
                claim_id=claim.claim_id,
                listing_id=claim.listing_id,
                merchant_id=claim.merchant_id,
                subject=claim.subject,
                issuer_id=claim.issuer_id,
                versions=claim.versions[:end],
            )
            validate_listing_claim_append(
                previous,
                prefix,
                listing=claim_listing,
                original_actor=claim.merchant_id,
                idempotency_key=version.idempotency_key,
                logical_time=version.logical_tick,
                evidence_lookup=evidence.by_digest,
            )
            previous = prefix
        if claim.logical_tick > logical_time:
            raise LogicalTimeError(
                "seeded listing claim cannot be after World logical time"
            )

    seeded_policies = tuple(policy for _, policy in pricing_policies.all())
    owner_by_market_merchant: dict[tuple[str, str], str] = {}
    for policy in seeded_policies:
        validate_pricing_policy_revision(policy)
        if policy.published_at_tick > logical_time:
            raise LogicalTimeError(
                "seeded pricing policy cannot be published after World logical time"
            )
        currency = _pricing_policy_scope_currency(
            catalog,
            merchant_id=policy.merchant_id,
            listing_ids=policy.listing_ids,
            product_ids=policy.product_ids,
        )
        if policy.currency != currency:
            raise WorldError(
                "seeded pricing policy currency disagrees with its catalog scope"
            )
        owner_by_market_merchant[(policy.market_id, policy.merchant_id)] = (
            policy.merchant_id
        )
    if replay_pricing_policy_revisions(
        seeded_policies,
        trusted_owner_by_merchant=owner_by_market_merchant,
    ) != seeded_policies:
        raise WorldError("seeded pricing policies differ from strict replay")

    for _, request in cart_quote_requests.all():
        validate_persistent_cart_quote_request(request)
        if request.issuer_id != "world":
            raise WriteNotAuthorized(
                "persisted cart quote request has an untrusted issuer"
            )
        if request.issued_at_tick > logical_time:
            raise LogicalTimeError(
                "persisted cart quote request cannot be after World logical time"
            )
        authority = authorities.read(request.mandate_id, caller=None)
        history = revisions.by_mandate(request.mandate_id)
        bound = next(
            (
                revision
                for revision in history
                if revision.revision == request.mandate_revision
            ),
            None,
        )
        if (
            authority is None
            or bound is None
            or authority.buyer_id != request.buyer_id
            or authority.principal_id != request.principal_id
            or bound.revision_digest != request.mandate_digest
            or request.created_by not in {
                authority.buyer_id,
                authority.principal_id,
            }
        ):
            raise WriteNotAuthorized(
                "persisted cart quote request mandate binding is invalid"
            )

    for _, quote in cart_quotes.all():
        validate_persistent_cart_quote(quote)
        if quote.issuer_id != "world":
            raise WriteNotAuthorized("persisted cart quote has an untrusted issuer")
        if quote.issued_at_tick > logical_time:
            raise LogicalTimeError(
                "persisted cart quote cannot be after World logical time"
            )
        authority = authorities.read(quote.mandate_id, caller=None)
        bound = next(
            (
                revision
                for revision in revisions.by_mandate(quote.mandate_id)
                if revision.revision == quote.mandate_revision
            ),
            None,
        )
        if (
            authority is None
            or bound is None
            or authority.buyer_id != quote.buyer_id
            or authority.principal_id != quote.principal_id
            or bound.revision_digest != quote.mandate_digest
        ):
            raise WriteNotAuthorized(
                "persisted cart quote mandate binding is invalid"
            )
        if quote.requested_by.startswith("merchant:"):
            request = cart_quote_requests.read(quote.request_id, caller=None)
            if (
                request is None
                or quote.requested_by not in request.allowed_merchant_ids
                or request.buyer_id != quote.buyer_id
                or request.mandate_id != quote.mandate_id
                or request.mandate_revision != quote.mandate_revision
                or request.mandate_digest != quote.mandate_digest
            ):
                raise WriteNotAuthorized(
                    "merchant cart quote lacks its persisted buyer authorization"
                )

    for _, group in order_groups.all():
        quote = cart_quotes.read(str(group.quote_id), caller=None)
        receipts = tuple(
            ledger.read(txn_id, caller=None) for txn_id in group.txn_ids
        )
        if quote is None or any(receipt is None for receipt in receipts):
            raise WorldError(
                "persisted order group lacks its quote or line ledger receipts"
            )
        _validate_group_payment_record(
            group,
            tuple(cast(Receipt, receipt) for receipt in receipts),
            quote,
        )

    tables: dict[str, Table[Any, Any]] = {
        "catalog": catalog,
        "evidence_records": evidence,
        "mandate_authorities": authorities,
        "mandate_revisions": revisions,
        "listing_claims": claims,
        "negotiation_events": negotiation_events,
        "pricing_policy_revisions": pricing_policies,
        "persistent_cart_quote_requests": cart_quote_requests,
        "persistent_cart_quotes": cart_quotes,
        "order_groups": order_groups,
        "governance_policies": governance_policies,
        "governance_records": governance_records,
    }
    for _, operation in operations.all():
        if operation.scope == "catalog-mutation":
            if (
                operation.outcome_table != "catalog"
                or operation.outcome_listing is None
                or str(operation.outcome_listing.sku_id) != operation.outcome_key
                or str(operation.outcome_listing.merchant_id)
                != catalog_owner_for_actor(operation.actor_id)
                or not operation.request_fingerprint
            ):
                raise WorldError(
                    "seeded catalog mutation operation binding is invalid"
                )
            # Catalog is deliberately re-seeded between E1 episodes.  Exact
            # retry therefore binds to the operation's immutable result rather
            # than requiring the old mutable catalog row to remain current.
            continue
        table = tables.get(operation.outcome_table)
        outcome = (
            None
            if table is None
            else table.read(
                operation.outcome_key,
                caller=(
                    "world"
                    if operation.outcome_table
                    in {"governance_policies", "governance_records"}
                    else None
                ),
            )
        )
        if outcome is None:
            raise WorldError(
                "seeded authority operation references a missing outcome"
            )
        if operation.scope == "evidence":
            if (
                not isinstance(outcome, EvidenceRecord)
                or operation.outcome_table != "evidence_records"
                or operation.actor_id != outcome.issuer_id
                or operation.request_fingerprint != outcome.record_digest
            ):
                raise WorldError("seeded evidence operation binding is invalid")
        elif operation.scope == "mandate-authority":
            if (
                not isinstance(outcome, MandateRevisionAuthority)
                or operation.outcome_table != "mandate_authorities"
                or operation.actor_id != outcome.principal_id
                or operation.request_fingerprint
                != canonical_digest(mandate_authority_to_wire(outcome))
            ):
                raise WorldError(
                    "seeded mandate authority operation binding is invalid"
                )
        elif operation.scope == "mandate-revision":
            if (
                not isinstance(outcome, MandateRevision)
                or operation.outcome_table != "mandate_revisions"
                or operation.actor_id != outcome.principal_id
                or operation.request_fingerprint != outcome.revision_digest
            ):
                raise WorldError(
                    "seeded mandate revision operation binding is invalid"
                )
        elif operation.scope.startswith("negotiation-event:"):
            if (
                not isinstance(outcome, NegotiationEvent)
                or operation.outcome_table != "negotiation_events"
                or operation.actor_id != outcome.actor_id
                or operation.scope
                != f"negotiation-event:{outcome.negotiation_id}"
                or not operation.request_fingerprint
            ):
                raise WorldError(
                    "seeded negotiation operation binding is invalid"
                )
        elif operation.scope.startswith("pricing-policy:"):
            if (
                not isinstance(outcome, PricingPolicyRevision)
                or operation.outcome_table != "pricing_policy_revisions"
                or operation.actor_id != outcome.actor_id
                or operation.scope != f"pricing-policy:{outcome.market_id}"
                or not operation.request_fingerprint
            ):
                raise WorldError(
                    "seeded pricing policy operation binding is invalid"
                )
        elif operation.scope.startswith("cart-quote-request:"):
            if (
                not isinstance(outcome, PersistentCartQuoteRequest)
                or operation.outcome_table
                != "persistent_cart_quote_requests"
                or operation.actor_id != outcome.created_by
                or operation.scope
                != f"cart-quote-request:{outcome.market_id}"
                or not operation.request_fingerprint
            ):
                raise WorldError(
                    "seeded cart quote request operation binding is invalid"
                )
        elif operation.scope.startswith("cart-quote:"):
            if (
                not isinstance(outcome, PersistentCartQuote)
                or operation.outcome_table != "persistent_cart_quotes"
                or operation.actor_id != outcome.requested_by
                or operation.scope != f"cart-quote:{outcome.market_id}"
                or not operation.request_fingerprint
            ):
                raise WorldError("seeded cart quote operation binding is invalid")
        elif operation.scope.startswith("cart-checkout:"):
            if not isinstance(outcome, OrderGroup):
                raise WorldError("seeded cart checkout outcome has wrong type")
            quote = cart_quotes.read(str(outcome.quote_id), caller=None)
            if (
                quote is None
                or operation.outcome_table != "order_groups"
                or operation.actor_id != str(outcome.buyer_id)
                or operation.scope != f"cart-checkout:{quote.market_id}"
                or operation.request_fingerprint
                != canonical_digest({"quote_id": quote.quote_id})
            ):
                raise WorldError(
                    "seeded cart checkout operation binding is invalid"
                )
        elif operation.scope == "publish_governance_policy" or operation.scope in {
            "aggregate_reviews",
            "ingest_review_observation",
            "ingest_market_observation",
            "resolve_governance_case",
            "apply_governance_reputation",
            "create_remediation_plan",
            "verify_remediation_step",
            "persist_ranking_context",
        } or operation.scope.startswith("apply_governance_intent:"):
            if (
                not isinstance(
                    outcome, (GovernancePolicyEnvelope, GovernanceRecordEnvelope)
                )
                or operation.outcome_table
                not in {"governance_policies", "governance_records"}
                or operation.actor_id != outcome.original_actor
                or not operation.request_fingerprint
            ):
                raise WorldError(
                    "seeded governance operation binding is invalid"
                )
        else:
            raise WorldError(
                f"unsupported seeded authority operation scope {operation.scope!r}"
            )


def _available_qty(row: "InventoryRow | int") -> int:
    """Sellable units = available − reserved (or the bare int for legacy rows)."""
    if isinstance(row, InventoryRow):
        return row.qty_available - row.qty_reserved
    return int(row)


def _negotiation_parties(actor_id: str, counterparty_id: str) -> tuple[str, str]:
    """Return trusted buyer/merchant order for two opposite-side actor ids."""

    if not isinstance(actor_id, str) or not isinstance(counterparty_id, str):
        raise NegotiationSchemaError("negotiation participants must be actor ids")
    if actor_id == counterparty_id:
        raise NegotiationSchemaError("negotiation participants must be distinct")
    actor_role = actor_id.split(":", 1)[0]
    counterparty_role = counterparty_id.split(":", 1)[0]
    if {actor_role, counterparty_role} != {"buyer", "merchant"}:
        raise NegotiationSchemaError(
            "negotiation requires one buyer and one merchant"
        )
    if actor_role == "buyer":
        return actor_id, counterparty_id
    return counterparty_id, actor_id


def _catalog_revision(listing: Listing) -> int:
    value = (listing.attributes or {}).get("catalog_revision", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MatchValidationError("catalog revision must be a positive integer")
    return int(value)


def _pricing_policy_scope_currency(
    catalog: CatalogTable,
    *,
    merchant_id: str,
    listing_ids: tuple[str, ...],
    product_ids: tuple[str, ...],
) -> str:
    """Validate one merchant-owned policy scope and derive its currency."""

    scoped: dict[str, Listing] = {}
    for sku_id in listing_ids:
        listing = cast("Listing | None", catalog.read(SkuId(sku_id), caller=None))
        if listing is None:
            raise WriteNotAuthorized(
                f"pricing policy references unknown listing {sku_id!r}"
            )
        if str(listing.merchant_id) != merchant_id:
            raise WriteNotAuthorized(
                f"pricing policy actor does not own listing {sku_id!r}"
            )
        scoped[str(listing.sku_id)] = listing
    for product_id in product_ids:
        matches = [
            listing
            for _, listing in catalog.all()
            if listing.product_id == product_id
            and str(listing.merchant_id) == merchant_id
        ]
        if not matches:
            raise WriteNotAuthorized(
                f"pricing policy actor owns no listing for product {product_id!r}"
            )
        scoped.update((str(listing.sku_id), listing) for listing in matches)
    currencies = {listing.list_price.currency for listing in scoped.values()}
    if len(currencies) != 1:
        raise WorldError("pricing policy scope must use exactly one currency")
    return next(iter(currencies))


_MANDATE_BUDGET_FIELDS = (
    "budget.max_minor",
    "budget.max",
    "budget_minor",
    "max_total_minor",
)


def _effective_mandate_fields(
    revisions: tuple[MandateRevision, ...],
) -> dict[str, Any]:
    """Materialize flat principal-authored mandate patches in revision order."""

    if not revisions:
        raise WriteNotAuthorized("cart quote mandate has no persisted revision")
    values: dict[str, Any] = {}
    for revision in revisions:
        values.update(dict(revision.changes))
    return values


def _mandate_budget_minor(revisions: tuple[MandateRevision, ...]) -> int:
    values = _effective_mandate_fields(revisions)
    present = [field for field in _MANDATE_BUDGET_FIELDS if field in values]
    if len(present) != 1:
        raise WriteNotAuthorized(
            "cart mandate must define exactly one authoritative minor-unit budget"
        )
    value = values[present[0]]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WriteNotAuthorized("cart mandate budget must be a non-negative integer")
    return value


def _active_pricing_policy_for_listing(
    policies: tuple[PricingPolicyRevision, ...],
    *,
    market_id: str,
    listing: Listing,
    logical_tick: int,
) -> PricingPolicyRevision:
    """Resolve exactly one active World policy for an authoritative listing."""

    streams: dict[str, list[PricingPolicyRevision]] = {}
    for policy in policies:
        if (
            policy.market_id != market_id
            or policy.merchant_id != str(listing.merchant_id)
            or (
                str(listing.sku_id) not in policy.listing_ids
                and (
                    listing.product_id is None
                    or listing.product_id not in policy.product_ids
                )
            )
        ):
            continue
        streams.setdefault(policy.policy_id, []).append(policy)
    active: list[PricingPolicyRevision] = []
    for revisions in streams.values():
        ordered = tuple(sorted(revisions, key=lambda row: row.revision))
        try:
            active.append(
                resolve_active_pricing_policy(ordered, logical_tick=logical_tick)
            )
        except PricingPolicyStaleError:
            continue
    if len(active) != 1:
        raise WorldError(
            f"listing {listing.sku_id!s} requires exactly one active pricing "
            f"policy; found {len(active)}"
        )
    return active[0]


def _revisioned_listing(value: Listing, prior: Any | None) -> Listing:
    """Stamp only updates; trusted initial rows retain their declared revision."""

    attributes = dict(value.attributes or {})
    reject_embedded_authority_records(attributes)
    if prior is None:
        # Validate an explicitly seeded revision, while preserving historical
        # listing equality when a scenario omits the additive field.
        if "catalog_revision" in attributes:
            _catalog_revision(value)
        return value
    if not isinstance(prior, Listing):
        raise MatchValidationError("catalog row has an invalid prior value")
    attributes["catalog_revision"] = _catalog_revision(prior) + 1
    return replace(value, attributes=attributes)


def _money_cents(value: Money) -> int:
    # ``Money`` predates the typed matching protocol and permits sub-cent
    # catalog values.  Matching uses the same explicit half-up projection as
    # the aggregator, so a persisted offer and the World-side freshness check
    # cannot disagree about values such as 24.005 USD.
    return int(
        (value.amount * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )


def _validate_group_payment_record(
    group: OrderGroup,
    receipts: list[Receipt] | tuple[Receipt, ...],
    quote: PersistentCartQuote,
) -> None:
    """Prove the persisted group closes the complete quote accounting identity."""

    if group.quote_hash != quote.quote_digest:
        raise WorldError("order group does not bind the authoritative quote digest")
    if tuple(receipt.txn_id for receipt in receipts) != group.txn_ids:
        raise WorldError("order group transaction ids disagree with line receipts")
    receipt_subtotal = sum(
        _money_cents(receipt.price) * receipt.qty for receipt in receipts
    )
    charge_total = sum(_money_cents(component.amount) for component in group.fee_breakdown)
    if receipt_subtotal != quote.subtotal_minor:
        raise WorldError("line ledger receipts do not equal the quoted subtotal")
    if charge_total != (
        quote.shipping_minor + quote.tax_minor + quote.fee_minor
    ):
        raise WorldError("group charges do not equal the quote charge total")
    if _money_cents(group.subtotal) != receipt_subtotal:
        raise WorldError("order group subtotal disagrees with line receipts")
    if _money_cents(group.grand_total) != receipt_subtotal + charge_total:
        raise WorldError(
            "order group grand total must equal line receipts plus group charges"
        )
    if _money_cents(group.grand_total) != quote.grand_total_minor:
        raise WorldError("order group grand total disagrees with authoritative quote")


def _match_acceptance_key(buyer_id: str, key: str) -> str:
    return f"{buyer_id}\x1f{key}"


def _certificate_for_acceptance(
    table: MatchCertificateTable,
    acceptance_digest: str,
) -> MatchCertificate | None:
    matches = [
        certificate
        for _, certificate in table.all()
        if certificate.acceptance_digest == acceptance_digest
    ]
    if len(matches) > 1:
        raise WorldError("multiple certificates exist for one acceptance")
    return None if not matches else matches[0]


def _supply_projection(listing: Listing, inventory: InventoryRow) -> SupplyState:
    return SupplyState(
        sku_id=inventory.sku_id,
        merchant_id=inventory.merchant_id,
        available_qty=inventory.qty_available - inventory.qty_reserved,
        reserved_qty=inventory.qty_reserved,
        eta_day=inventory.eta_day,
        unit_price_cents=_money_cents(listing.list_price),
        version=inventory.version,
    )


def _authorize_psp(actor: str, *, operation: str) -> None:
    """Reserve payment/allocation commits for the exact PSP service actor."""
    if actor != "platform:psp":
        raise LifecycleAuthorizationError(
            f"only platform:psp may {operation}; got {actor!r}"
        )


def _authorize_exchange_actor(order: Order, actor: str) -> None:
    """Only the owning merchant or the exact PSP may complete replacement."""
    if actor not in {str(order.merchant_id), "platform:psp"}:
        raise LifecycleAuthorizationError(
            f"actor {actor!r} cannot exchange order {order.order_id}"
        )


def _validate_requested_and_fulfilled(order: Order, fulfilled_qty: int) -> None:
    if not isinstance(order.qty, int) or isinstance(order.qty, bool) or order.qty <= 0:
        raise FulfillmentNotActionable("requested quantity must be a positive integer")
    if (
        not isinstance(fulfilled_qty, int)
        or isinstance(fulfilled_qty, bool)
        or fulfilled_qty < 0
        or fulfilled_qty > order.qty
    ):
        raise FulfillmentNotActionable(
            "fulfilled quantity must be an integer between zero and requested quantity"
        )


def _validate_same_order_identity(existing: Order, requested: Order) -> None:
    fields = (
        "order_id",
        "buyer_id",
        "merchant_id",
        "sku_id",
        "qty",
        "agreed_price",
    )
    mismatches = [
        field
        for field in fields
        if getattr(existing, field) != getattr(requested, field)
    ]
    if mismatches:
        raise FulfillmentNotActionable(
            "persisted order identity mismatch: " + ", ".join(mismatches)
        )


def _order_identity_matches(existing: Order, requested: Order) -> bool:
    return all(
        getattr(existing, field) == getattr(requested, field)
        for field in (
            "order_id",
            "buyer_id",
            "merchant_id",
            "sku_id",
            "qty",
            "agreed_price",
        )
    )


def _validate_partial_receipt(
    *,
    order: Order,
    fulfilled_qty: int,
    receipt: Receipt | None,
    inventory: InventoryRow | int,
    idempotency_key: str,
) -> None:
    if not idempotency_key.strip():
        raise FulfillmentNotActionable("idempotency key must not be blank")
    if fulfilled_qty == 0:
        if receipt is not None:
            raise FulfillmentNotActionable(
                "a zero-fill backorder must not create a payment receipt"
            )
        return
    if receipt is None:
        raise FulfillmentNotActionable(
            "a positive fulfilled quantity requires a payment receipt"
        )
    if receipt.idempotency_key != idempotency_key:
        raise FulfillmentNotActionable(
            "receipt idempotency key must match the allocation request"
        )
    validate_transaction_identity(
        order,
        receipt,
        inventory,
        expected_qty=fulfilled_qty,
        expected_effect="charge",
    )


def _paid_quantity(table: Table[Any, Any], order: Order) -> int:
    allocation = table.read(order.order_id, caller=None)
    if allocation is None:
        return order.qty
    paid_qty = int(allocation.fulfilled_qty)
    if paid_qty <= 0:
        raise FulfillmentNotActionable(
            f"order {order.order_id} has no fulfilled units"
        )
    return paid_qty


def _exchange_for_replacement(
    table: Table[Any, Any],
    order_id: OrderId,
) -> Exchange | None:
    for _, exchange in table.all():
        if exchange.replacement_order_id == order_id:
            return cast("Exchange", exchange)
    return None


def _validate_replacement_identity(
    original: Order,
    replacement: Order,
    *,
    qty: int,
) -> None:
    mismatches: list[str] = []
    if replacement.order_id == original.order_id:
        mismatches.append("replacement.order_id")
    if replacement.buyer_id != original.buyer_id:
        mismatches.append("replacement.buyer_id")
    if replacement.merchant_id != original.merchant_id:
        mismatches.append("replacement.merchant_id")
    if replacement.qty != qty:
        mismatches.append("replacement.qty")
    if replacement.agreed_price != original.agreed_price:
        mismatches.append("replacement.agreed_price")
    if mismatches:
        raise ExchangeNotActionable(
            "exchange identity mismatch: " + ", ".join(mismatches)
        )


def _require_listing_owner(order: Order, listing: Any | None) -> None:
    if listing is None:
        raise ExchangeNotActionable(f"listing {order.sku_id} does not exist")
    if getattr(listing, "merchant_id", None) != order.merchant_id:
        raise ExchangeNotActionable(
            f"listing {order.sku_id} is not owned by {order.merchant_id}"
        )


def _require_inventory_owner(order: Order, inventory: InventoryRow | int) -> None:
    if isinstance(inventory, InventoryRow) and inventory.merchant_id != order.merchant_id:
        raise ExchangeNotActionable(
            f"inventory {order.sku_id} is not owned by {order.merchant_id}"
        )


def _reserve_inventory(row: "InventoryRow | int", *, qty: int) -> "InventoryRow | int":
    """Reserve ``qty`` units, returning the new row (bumps ``qty_reserved``)."""
    if isinstance(row, InventoryRow):
        return replace(
            row,
            qty_reserved=row.qty_reserved + qty,
            version=row.version + 1,
        )
    return int(row) - qty


def _restock_inventory(row: "InventoryRow | int", *, qty: int) -> "InventoryRow | int":
    """Restock ``qty`` units — the inverse of :func:`_reserve_inventory` (a
    refund returns the reserved units). Clamps at 0 so a refund never drives
    ``qty_reserved`` negative."""
    if isinstance(row, InventoryRow):
        return replace(
            row,
            qty_reserved=max(0, row.qty_reserved - qty),
            version=row.version + 1,
        )
    return int(row) + qty


def _make_diff(
    txn: str, order_id: Any, writes: "list[tuple[str, str, str, Any, Any]]",
    invariants: "tuple[str, ...]",
) -> TransactionStateDiff:
    """Snapshot a multi-row atomic write into a :class:`TransactionStateDiff` —
    deep-copying each before/after so the row state is frozen at transaction time
    (the rows keep mutating after)."""
    return TransactionStateDiff(
        txn=txn, order_id=str(order_id),
        table_writes=tuple(
            TableWrite(
                table=t,
                key=k,
                op=op,
                before=_copy_world_value(b),
                after=_copy_world_value(a),
            )
            for (t, k, op, b, a) in writes),
        invariants_held=tuple(invariants),
    )


def _copy_world_value(value: Any) -> Any:
    """Copy mutable rows while retaining recursively immutable protocol values."""

    if isinstance(
        value,
        (
            EvidenceRecord,
            MandateRevision,
            MandateRevisionAuthority,
            ListingClaim,
            AfterSalesWrite,
        ),
    ):
        return value
    return deepcopy(value)


def _copy_world_commit(record: WorldCommitRecord) -> WorldCommitRecord:
    """Defensively copy a commit without pickling sealed mapping proxies."""

    return replace(
        record,
        table_writes=tuple(
            TableWrite(
                table=write.table,
                key=write.key,
                op=write.op,
                before=_copy_world_value(write.before),
                after=_copy_world_value(write.after),
            )
            for write in record.table_writes
        ),
    )


def _reputation_settlement_event_id(txn_id: TxnId) -> str:
    """Return the canonical event key for one authoritative transaction."""

    return f"reputation-settlement:{txn_id}"


def _validate_reputation_settlement_identity(
    *,
    merchant_id: AgentId,
    order_id: OrderId,
    txn_id: TxnId,
    order: Order | None,
    receipt: Receipt | None,
) -> None:
    """Require a real persisted payment before reputation can advance."""

    if order is None or receipt is None:
        raise WorldError(
            "settlement reputation is not backed by an authoritative order and receipt"
        )
    mismatches: list[str] = []
    if order.order_id != order_id:
        mismatches.append("order.order_id")
    if order.merchant_id != merchant_id:
        mismatches.append("order.merchant_id")
    if receipt.txn_id != txn_id:
        mismatches.append("receipt.txn_id")
    if receipt.order_id != order.order_id:
        mismatches.append("receipt.order_id")
    if receipt.buyer_id != order.buyer_id:
        mismatches.append("receipt.buyer_id")
    if receipt.merchant_id != order.merchant_id:
        mismatches.append("receipt.merchant_id")
    if receipt.sku_id != order.sku_id:
        mismatches.append("receipt.sku_id")
    if receipt.price != order.agreed_price:
        mismatches.append("receipt.price")
    if isinstance(receipt.qty, bool) or receipt.qty <= 0 or receipt.qty > order.qty:
        mismatches.append("receipt.qty")
    if order.state not in _ALREADY_SETTLED:
        mismatches.append("order.state")
    if mismatches:
        raise WorldError(
            "invalid authoritative settlement for reputation: "
            + ", ".join(sorted(mismatches))
        )
