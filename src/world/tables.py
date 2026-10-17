"""Per-table primitives.

Each table encapsulates its visibility rule (read policy) and the set of
``ActionKind`` values authorized to mutate it. ``World`` consults the table
for visibility on read and for authorization on write.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from protocol.evidence_records import (
    EvidenceRecord,
    MandateRevision,
    MandateRevisionAuthority,
    validate_evidence_record,
    validate_mandate_revision,
)
from protocol.event_receipts import ProtocolEvent, ProtocolEventReceipt
from protocol.listing_claims import ListingClaim, validate_listing_claim
from protocol.matching import MatchAcceptance, MatchCertificate, SearchSession
from protocol.negotiation_state import (
    NegotiationEvent,
    NegotiationThread,
    validate_negotiation_event,
    validate_negotiation_thread,
)
from protocol.cart_quote_state import (
    PersistentCartQuote,
    validate_persistent_cart_quote,
)
from protocol.cart_quote_request import (
    PersistentCartQuoteRequest,
    validate_persistent_cart_quote_request,
)
from protocol.pricing_policy import (
    GENESIS_PRICING_POLICY_DIGEST,
    PricingPolicyRevision,
    validate_pricing_policy_revision,
)
from protocol.supply_authority import (
    SupplyPurchaseAuthority,
    validate_supply_purchase_authority,
)
from world.cart_pricing import pricing_policy_revision_key
from world.after_sales_core import AfterSalesPolicyRevision, validate_after_sales_policy
from world.after_sales_persistence import (
    AfterSalesWrite,
    physical_after_sales_record_key,
    validate_after_sales_write,
)
from world.market_governance_persistence import (
    GovernancePolicyEnvelope,
    GovernanceRecordEnvelope,
    GovernanceTables,
    GovernanceWrite,
    envelope_key,
    validate_policy_envelope,
    validate_record_envelope,
)
from world.evidence_contracts import (
    authority_operation_key,
    authorize_persisted_evidence_read,
    evidence_record_key,
    mandate_revision_key,
)
from world.types import (
    AuthorityOperationRecord,
    Dispute,
    Exchange,
    Friendship,
    FulfillmentAllocation,
    InventoryRow,
    Listing,
    Order,
    OrderGroup,
    OrderTimeline,
    Receipt,
    ReputationSettlement,
    Review,
    Shipment,
)
from world.payment_fulfillment import (
    PackingRecord,
    PaymentStateRecord,
    packing_record_key,
    payment_state_key,
    validate_packing_record,
    validate_packing_transition,
    validate_payment_state_record,
    validate_payment_transition,
)

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class Table(Generic[K, V]):
    """Base table. Subclasses override ``read`` for partition-scoped visibility."""

    name: str
    allowed_actions: frozenset[str]
    rows: dict[K, V] = field(default_factory=dict)

    def read(self, key: K, *, caller: str | None = None) -> V | None:
        """Default policy: public read. Subclasses override for partitioned tables."""
        return self.rows.get(key)

    def write(self, key: K, value: V) -> None:
        """Insert or replace the row. Authorization checked upstream by ``World``."""
        self.rows[key] = value

    def delete(self, key: K) -> None:
        """Delete one row, allowing indexed subclasses to keep parity."""

        self.rows.pop(key, None)

    def clear(self) -> None:
        """Clear every row, allowing indexed subclasses to reset indexes."""

        self.rows.clear()

    def all(self) -> "Iterator[tuple[K, V]]":
        """Iterate rows in deterministic order (sorted by key)."""
        yield from sorted(self.rows.items(), key=lambda kv: str(kv[0]))


class CatalogTable(Table[Any, Listing]):
    """Public catalog. Read by anyone; writes only by curator-side actions."""

    def __init__(self) -> None:
        """Initialize as an empty public table."""
        super().__init__(
            name="catalog",
            allowed_actions=frozenset({"update_catalog", "world.update_catalog"}),
        )

    def search(self, query: str, filters: dict[str, object] | None) -> "list[Listing]":
        """Return listings matching ``query``; deterministic order."""
        filters = filters or {}
        terms = query.casefold().split()
        matches: list[Listing] = []
        for _, row in self.all():
            haystack = " ".join(
                [
                    str(row.sku_id),
                    row.category,
                    row.name,
                    " ".join(f"{k} {v}" for k, v in sorted(row.attributes.items())),
                ]
            ).casefold()
            if terms and not all(term in haystack for term in terms):
                continue
            if not _matches_filters(row, filters):
                continue
            matches.append(row)
        return matches


class InventoryTable(Table[Any, InventoryRow | int]):
    """Inventory by SKU. Writable by platform reservation and fulfillment paths."""

    def __init__(self) -> None:
        """Initialize. Authorized actions: ``settle``."""
        super().__init__(
            name="inventory",
            allowed_actions=frozenset(
                {
                    "settle",
                    "reserve_inventory",
                    "update_inventory",
                    "world.reserve_inventory",
                    "world.update_inventory",
                }
            ),
        )
        self._supply_operations: dict[tuple[str, str], tuple[Any, Any]] = {}

    def write(self, key: Any, value: InventoryRow | int) -> None:
        before = self.rows.get(key)
        if isinstance(before, InventoryRow):
            self._drop_supply_indexes(key, before)
        super().write(key, value)
        if isinstance(value, InventoryRow):
            for record in value.supply_events:
                self._supply_operations[(record.scope, record.idempotency_key)] = (
                    key,
                    record,
                )

    def delete(self, key: Any) -> None:
        before = self.rows.get(key)
        if isinstance(before, InventoryRow):
            self._drop_supply_indexes(key, before)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._supply_operations.clear()

    def supply_operation(self, scope: str, key: str) -> Any | None:
        indexed = self._supply_operations.get((scope, key))
        return None if indexed is None else indexed[1]

    def _drop_supply_indexes(self, key: Any, row: InventoryRow) -> None:
        for record in row.supply_events:
            indexed = self._supply_operations.get((record.scope, record.idempotency_key))
            if indexed is not None and indexed[0] == key:
                self._supply_operations.pop((record.scope, record.idempotency_key), None)


class LedgerTable(Table[Any, Receipt]):
    """Append-only, caller-scoped ledger.

    A receipt is private to the buyer and merchant named on it.  Platform
    services may inspect every row for settlement/replay, matching the order
    table and the SQLite world's visibility rule.
    """

    def __init__(self) -> None:
        """Initialize. Authorized actions: ``settle``, ``issue_refund``."""
        super().__init__(
            name="ledger",
            allowed_actions=frozenset(
                {"settle", "issue_refund", "update_ledger", "world.update_ledger"}
            ),
        )

    def write(self, key: Any, value: Any) -> None:
        """Append-only ledger rows cannot be replaced."""
        if not isinstance(value, Receipt) or value.effect not in {"charge", "refund"}:
            raise ValueError("ledger receipt effect must be charge or refund")
        if key in self.rows:
            from world.errors import WriteNotAuthorized

            raise WriteNotAuthorized(f"ledger row already exists: {key}")
        super().write(key, value)

    def read(self, key: Any, *, caller: str | None = None) -> Receipt | None:
        """Return a receipt only to its two transaction parties or platform."""
        row = self.rows.get(key)
        if row is None or _is_platform(caller):
            return row
        return row if caller in (str(row.buyer_id), str(row.merchant_id)) else None


class OrderTable(Table[Any, Order]):
    """Orders are caller-scoped to buyer, merchant, and platform services."""

    def __init__(self) -> None:
        super().__init__(
            name="orders",
            allowed_actions=frozenset(
                {
                    "create_order",
                    "update_order_status",
                    "settle",
                    "dispatch",
                    "issue_refund",
                    "world.create_order",
                    "world.update_order_status",
                }
            ),
        )
        self._allocation_index: dict[tuple[str, str, str], dict[str, Order]] = {}

    def read(self, key: Any, *, caller: str | None = None) -> Order | None:
        row = self.rows.get(key)
        if row is None or _is_platform(caller):
            return row
        return row if caller in (str(row.buyer_id), str(row.merchant_id)) else None

    @staticmethod
    def _allocation_key(row: Order) -> tuple[str, str, str]:
        return (str(row.merchant_id), str(row.sku_id), row.state.value)

    def write(self, key: Any, value: Order) -> None:
        before = self.rows.get(key)
        if before is not None:
            bucket = self._allocation_index.get(self._allocation_key(before))
            if bucket is not None:
                bucket.pop(str(key), None)
                if not bucket:
                    self._allocation_index.pop(self._allocation_key(before), None)
        super().write(key, value)
        self._allocation_index.setdefault(self._allocation_key(value), {})[str(key)] = value

    def delete(self, key: Any) -> None:
        before = self.rows.get(key)
        if before is not None:
            bucket = self._allocation_index.get(self._allocation_key(before))
            if bucket is not None:
                bucket.pop(str(key), None)
                if not bucket:
                    self._allocation_index.pop(self._allocation_key(before), None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._allocation_index.clear()

    def eligible_for_allocation(
        self,
        *,
        merchant_id: str,
        sku_id: str,
        states: frozenset[str],
    ) -> tuple[Order, ...]:
        """Use the composite merchant/SKU/state index for scarce allocation.

        ``request_order`` is the leading deterministic priority inside each
        indexed scope; no call to :meth:`all` or traversal of unrelated orders
        participates in this query.
        """

        rows = [
            row
            for state in states
            for row in self._allocation_index.get((merchant_id, sku_id, state), {}).values()
            if row.request_order > 0
        ]
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.request_order,
                    str(row.buyer_id),
                    str(row.order_id),
                ),
            )
        )


class OrderGroupTable(Table[Any, OrderGroup]):
    """Atomic cart outcomes visible only to the buyer and participating merchants."""

    def __init__(self) -> None:
        super().__init__(
            name="order_groups",
            allowed_actions=frozenset({"checkout_cart", "world.checkout_cart"}),
        )

    def read(self, key: Any, *, caller: str | None = None) -> OrderGroup | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        participants = {str(row.buyer_id), *(str(value) for value in row.merchant_ids)}
        return row if caller in participants else None


class OrderTimelineTable(Table[Any, OrderTimeline]):
    """World-authored lifecycle timing, visible only to exact order parties."""

    def __init__(self) -> None:
        super().__init__(
            name="order_timelines",
            allowed_actions=frozenset(
                {
                    "record_order_timeline",
                    "world.record_order_timeline",
                }
            ),
        )

    def read(self, key: Any, *, caller: str | None = None) -> OrderTimeline | None:
        row = self.rows.get(key)
        if row is None or _is_platform(caller) or caller == "runtime:clock":
            return row
        return row if caller in (str(row.buyer_id), str(row.merchant_id)) else None


class FulfillmentTable(Table[Any, FulfillmentAllocation]):
    """One immutable quantity allocation per order, visible only to its parties."""

    def __init__(self) -> None:
        super().__init__(
            name="fulfillments",
            allowed_actions=frozenset(
                {
                    "record_fulfillment",
                    "world.record_fulfillment",
                }
            ),
        )
        self._allocation_ids: dict[str, set[Any]] = {}
        self._operation_keys: dict[tuple[str, str], set[Any]] = {}

    def read(
        self,
        key: Any,
        *,
        caller: str | None = None,
    ) -> FulfillmentAllocation | None:
        row = self.rows.get(key)
        if row is None or _is_platform(caller):
            return row
        return row if caller in (str(row.buyer_id), str(row.merchant_id)) else None

    def write(self, key: Any, value: FulfillmentAllocation) -> None:
        if key in self.rows:
            from world.errors import WriteNotAuthorized

            raise WriteNotAuthorized(f"fulfillment row already exists: {key}")
        super().write(key, value)
        if value.allocation_id is not None:
            self._allocation_ids.setdefault(value.allocation_id, set()).add(key)
        operation_key = self._batch_operation_key(value)
        if operation_key is not None:
            self._operation_keys.setdefault((str(value.created_by), operation_key), set()).add(key)

    def delete(self, key: Any) -> None:
        before = self.rows.get(key)
        if before is not None:
            if before.allocation_id is not None:
                keys = self._allocation_ids.get(before.allocation_id)
                if keys is not None:
                    keys.discard(key)
                    if not keys:
                        self._allocation_ids.pop(before.allocation_id, None)
            operation_key = self._batch_operation_key(before)
            if operation_key is not None:
                keys = self._operation_keys.get((str(before.created_by), operation_key))
                if keys is not None:
                    keys.discard(key)
                    if not keys:
                        self._operation_keys.pop((str(before.created_by), operation_key), None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._allocation_ids.clear()
        self._operation_keys.clear()

    def by_allocation_id(self, allocation_id: str) -> tuple[FulfillmentAllocation, ...]:
        return tuple(
            self.rows[key] for key in sorted(self._allocation_ids.get(allocation_id, ()), key=str)
        )

    def by_operation_key(
        self, created_by: str, idempotency_key: str
    ) -> tuple[FulfillmentAllocation, ...]:
        return tuple(
            self.rows[key]
            for key in sorted(
                self._operation_keys.get((created_by, idempotency_key), ()),
                key=str,
            )
        )

    @staticmethod
    def _batch_operation_key(value: FulfillmentAllocation) -> str | None:
        if value.allocation_id is None:
            return None
        suffix = f":{value.order_id}"
        if not value.idempotency_key.endswith(suffix):
            return None
        return value.idempotency_key[: -len(suffix)]


class ExchangeTable(Table[Any, Exchange]):
    """Completed order replacements, caller-scoped to their exact parties."""

    def __init__(self) -> None:
        super().__init__(
            name="exchanges",
            allowed_actions=frozenset({"record_exchange", "world.record_exchange"}),
        )

    def read(self, key: Any, *, caller: str | None = None) -> Exchange | None:
        row = self.rows.get(key)
        if row is None or _is_platform(caller):
            return row
        return row if caller in (str(row.buyer_id), str(row.merchant_id)) else None

    def write(self, key: Any, value: Exchange) -> None:
        if key in self.rows:
            from world.errors import WriteNotAuthorized

            raise WriteNotAuthorized(f"exchange row already exists: {key}")
        super().write(key, value)


class ShipmentTable(Table[Any, Shipment]):
    """Versioned shipment rows visible to exact order parties and Platform."""

    def __init__(self) -> None:
        super().__init__(
            name="shipments",
            allowed_actions=frozenset(
                {
                    "record_shipment",
                    "resolve_shipment",
                    "world.record_shipment",
                    "world.resolve_shipment",
                }
            ),
        )
        self._by_order: dict[str, Any] = {}
        self._event_operations: dict[str, tuple[Any, Any]] = {}
        self._resolution_operations: dict[str, Any] = {}

    def read(self, key: Any, *, caller: str | None = None) -> Shipment | None:
        row = self.rows.get(key)
        if row is None or _is_platform(caller) or caller == "runtime:logistics":
            return row
        return row if caller in (str(row.buyer_id), str(row.merchant_id)) else None

    def write(self, key: Any, value: Shipment) -> None:
        before = self.rows.get(key)
        other_key = self._by_order.get(str(value.order_id))
        if other_key is not None and other_key != key:
            from world.errors import WriteNotAuthorized

            raise WriteNotAuthorized(f"order {value.order_id} already has shipment {other_key}")
        if before is not None:
            self._drop_indexes(key, before)
        super().write(key, value)
        self._add_indexes(key, value)

    def delete(self, key: Any) -> None:
        before = self.rows.get(key)
        if before is not None:
            self._drop_indexes(key, before)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._by_order.clear()
        self._event_operations.clear()
        self._resolution_operations.clear()

    def by_order(self, order_id: Any) -> Shipment | None:
        key = self._by_order.get(str(order_id))
        return None if key is None else self.rows.get(key)

    def event_operation(self, idempotency_key: str) -> tuple[Shipment, Any] | None:
        indexed = self._event_operations.get(idempotency_key)
        if indexed is None:
            return None
        shipment_key, event = indexed
        shipment = self.rows.get(shipment_key)
        return None if shipment is None else (shipment, event)

    def resolution_operation(self, idempotency_key: str) -> Shipment | None:
        key = self._resolution_operations.get(idempotency_key)
        return None if key is None else self.rows.get(key)

    def _add_indexes(self, key: Any, value: Shipment) -> None:
        self._by_order[str(value.order_id)] = key
        for event in value.status_history:
            if event.idempotency_key:
                self._event_operations[event.idempotency_key] = (key, event)
        if value.resolution_idempotency_key:
            self._resolution_operations[value.resolution_idempotency_key] = key

    def _drop_indexes(self, key: Any, value: Shipment) -> None:
        if self._by_order.get(str(value.order_id)) == key:
            self._by_order.pop(str(value.order_id), None)
        for event in value.status_history:
            if (
                event.idempotency_key
                and self._event_operations.get(event.idempotency_key, (None, None))[0] == key
            ):
                self._event_operations.pop(event.idempotency_key, None)
        if (
            value.resolution_idempotency_key
            and self._resolution_operations.get(value.resolution_idempotency_key) == key
        ):
            self._resolution_operations.pop(value.resolution_idempotency_key, None)


class ReputationTable(Table[Any, Any]):
    """Public reputation. Writable only by ``update_reputation``."""

    def __init__(self) -> None:
        """Initialize. Authorized actions: ``update_reputation``."""
        super().__init__(
            name="reputation",
            allowed_actions=frozenset({"update_reputation", "world.update_reputation"}),
        )


class ReputationSettlementTable(Table[str, ReputationSettlement]):
    """Private, append-only settlement-to-reputation evidence.

    The table indexes both the authoritative transaction and every observed
    source idempotency identity. Updates may only append source aliases to the
    same immutable settlement and score outcome.
    """

    def __init__(self) -> None:
        super().__init__(
            name="reputation_settlements",
            allowed_actions=frozenset({"world.update_reputation"}),
        )
        self._by_txn: dict[str, str] = {}
        self._by_source: dict[tuple[str, str], str] = {}

    def read(self, key: str, *, caller: str | None = None) -> ReputationSettlement | None:
        row = self.rows.get(key)
        if row is None:
            return None
        if caller is None or _is_platform(caller) or caller == "runtime":
            return row
        return row if caller.startswith("runtime:") else None

    def write(self, key: str, value: ReputationSettlement) -> None:
        from world.errors import IdempotencyConflict

        if key != value.event_id:
            raise IdempotencyConflict("reputation settlement key does not match event")
        prior = self.rows.get(key)
        other_event = self._by_txn.get(str(value.txn_id))
        if other_event is not None and other_event != key:
            raise IdempotencyConflict(
                f"transaction {value.txn_id} already has reputation event {other_event}"
            )
        if prior is not None:
            if (
                prior.order_id != value.order_id
                or prior.txn_id != value.txn_id
                or prior.merchant_id != value.merchant_id
                or prior.outcome != value.outcome
                or not set(prior.sources).issubset(value.sources)
            ):
                raise IdempotencyConflict("reputation settlement identity or outcome cannot change")
        for source in value.sources:
            source_key = (source.source_actor, source.source_idempotency_key)
            indexed_event = self._by_source.get(source_key)
            if indexed_event is not None and indexed_event != key:
                raise IdempotencyConflict(
                    "reputation source idempotency key belongs to another settlement"
                )
        if prior is not None:
            self._drop_indexes(key, prior)
        super().write(key, value)
        self._add_indexes(key, value)

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            self._drop_indexes(key, prior)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._by_txn.clear()
        self._by_source.clear()

    def by_txn(self, txn_id: str) -> ReputationSettlement | None:
        key = self._by_txn.get(txn_id)
        return None if key is None else self.rows.get(key)

    def by_source(
        self, source_actor: str, source_idempotency_key: str
    ) -> ReputationSettlement | None:
        key = self._by_source.get((source_actor, source_idempotency_key))
        return None if key is None else self.rows.get(key)

    def _add_indexes(self, key: str, value: ReputationSettlement) -> None:
        self._by_txn[str(value.txn_id)] = key
        for source in value.sources:
            self._by_source[(source.source_actor, source.source_idempotency_key)] = key

    def _drop_indexes(self, key: str, value: ReputationSettlement) -> None:
        if self._by_txn.get(str(value.txn_id)) == key:
            self._by_txn.pop(str(value.txn_id), None)
        for source in value.sources:
            source_key = (source.source_actor, source.source_idempotency_key)
            if self._by_source.get(source_key) == key:
                self._by_source.pop(source_key, None)


class DisputeTable(Table[Any, Dispute]):
    """Disputes. Writable by ``open_dispute``, ``rule``, ``withdraw_dispute``."""

    def __init__(self) -> None:
        """Initialize. Caller-scoped to dispute parties."""
        super().__init__(
            name="disputes",
            allowed_actions=frozenset(
                {
                    "open_dispute",
                    "rule",
                    "withdraw_dispute",
                    "update_dispute",
                    "world.open_dispute",
                    "world.update_dispute",
                }
            ),
        )

    def read(self, key: Any, *, caller: str | None = None) -> Dispute | None:
        """Return the dispute only if ``caller`` is a party to it."""
        row = self.rows.get(key)
        if row is None or _is_platform(caller):
            return row
        return row if caller in (str(row.filed_by), str(row.against)) else None


class RulingTable(Table[Any, Any]):
    """Rulings. Writable only by ``rule`` from ``platform:adjudicator``."""

    def __init__(self) -> None:
        """Initialize. Authorized actions: ``rule``."""
        super().__init__(
            name="rulings",
            allowed_actions=frozenset({"rule", "update_ruling", "world.update_ruling"}),
        )


class FriendshipTable(Table[Any, Friendship]):
    """The social graph (item 7). Keyed by ``buyer_id``; caller-scoped read.

    A buyer sees only their own friend list — you cannot enumerate someone
    else's friends. Seeded from a scenario's ``initial_state.friendships``.
    """

    def __init__(self) -> None:
        super().__init__(
            name="friendships",
            allowed_actions=frozenset({"add_friend", "world.add_friend"}),
        )

    def read(self, key: Any, *, caller: str | None = None) -> Friendship | None:
        """Return the friend list only to its owning buyer (or platform)."""
        row = self.rows.get(key)
        if row is None or _is_platform(caller):
            return row
        return row if caller == str(row.buyer_id) else None


class ReviewTable(Table[Any, Review]):
    """Friends' product reviews (item 7). Keyed by ``review_id``.

    Row-key reads are open, but buyers reach reviews only through
    ``WorldTools.get_friend_reviews``, which filters to the caller's friends —
    that is where the social trust gate lives. Seeded from a scenario's
    ``initial_state.reviews``.
    """

    def __init__(self) -> None:
        super().__init__(
            name="reviews",
            allowed_actions=frozenset({"add_review", "world.add_review"}),
        )

    def by_friends(
        self,
        friends: "frozenset[str]",
        *,
        sku_id: str | None = None,
        merchant_id: str | None = None,
    ) -> "list[Review]":
        """Reviews authored by ``friends``, optionally filtered by sku/merchant."""
        out: list[Review] = []
        for _, row in self.all():
            if str(row.reviewer_id) not in friends:
                continue
            if sku_id is not None and str(row.sku_id) != str(sku_id):
                continue
            if merchant_id is not None and str(row.merchant_id) != str(merchant_id):
                continue
            out.append(row)
        return out


class SearchSessionTable(Table[str, SearchSession]):
    """Immutable Platform-authored discovery sessions in authoritative World."""

    def __init__(self) -> None:
        super().__init__(
            name="search_sessions",
            allowed_actions=frozenset({"world.create_search_session"}),
        )
        self._request_keys: dict[tuple[str, str], str] = {}
        self._offer_keys: dict[tuple[str, str], set[str]] = {}

    def write(self, key: str, value: SearchSession) -> None:
        prior = self.rows.get(key)
        if prior is not None and prior != value:
            from world.errors import IdempotencyConflict

            raise IdempotencyConflict(f"search session id {key!r} was reused")
        request_key = (value.buyer_id, value.search_idempotency_key)
        prior_id = self._request_keys.get(request_key)
        if prior_id is not None and prior_id != key:
            from world.errors import IdempotencyConflict

            raise IdempotencyConflict("search idempotency key was reused for a different session")
        super().write(key, value)
        self._request_keys[request_key] = key
        for offer in value.offers:
            self._offer_keys.setdefault((value.buyer_id, offer.offer_id), set()).add(key)

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            self._request_keys.pop((prior.buyer_id, prior.search_idempotency_key), None)
            for offer in prior.offers:
                offer_key = (prior.buyer_id, offer.offer_id)
                session_ids = self._offer_keys.get(offer_key)
                if session_ids is not None:
                    session_ids.discard(key)
                    if not session_ids:
                        self._offer_keys.pop(offer_key, None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._request_keys.clear()
        self._offer_keys.clear()

    def by_request_key(self, buyer_id: str, key: str) -> SearchSession | None:
        session_id = self._request_keys.get((buyer_id, key))
        return None if session_id is None else self.rows[session_id]

    def by_offer(self, buyer_id: str, offer_id: str) -> tuple[SearchSession, ...]:
        session_ids = sorted(self._offer_keys.get((buyer_id, offer_id), ()))
        return tuple(self.rows[session_id] for session_id in session_ids)

    def read(self, key: str, *, caller: str | None = None) -> SearchSession | None:
        return self.rows.get(key) if _is_trusted_matching_reader(caller) else None


class MatchAcceptanceTable(Table[str, MatchAcceptance]):
    """One immutable accepted-offer request per scoped idempotency key."""

    def __init__(self) -> None:
        super().__init__(
            name="match_acceptances",
            allowed_actions=frozenset({"world.issue_match_certificate"}),
        )

    def write(self, key: str, value: MatchAcceptance) -> None:
        prior = self.rows.get(key)
        if prior is not None and prior != value:
            from world.errors import IdempotencyConflict

            raise IdempotencyConflict(
                "match acceptance idempotency key was reused with different fields"
            )
        super().write(key, value)

    def read(self, key: str, *, caller: str | None = None) -> MatchAcceptance | None:
        return self.rows.get(key) if _is_trusted_matching_reader(caller) else None


class MatchCertificateTable(Table[str, MatchCertificate]):
    """Immutable certificates loaded by PSP from trusted World state."""

    def __init__(self) -> None:
        super().__init__(
            name="match_certificates",
            allowed_actions=frozenset({"world.issue_match_certificate"}),
        )
        self._order_keys: dict[tuple[str, str], set[str]] = {}

    def write(self, key: str, value: MatchCertificate) -> None:
        prior = self.rows.get(key)
        if prior is not None and prior != value:
            from world.errors import IdempotencyConflict

            raise IdempotencyConflict(f"match certificate id {key!r} was reused")
        super().write(key, value)
        self._order_keys.setdefault((value.buyer_id, value.order_id), set()).add(key)

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            order_key = (prior.buyer_id, prior.order_id)
            cert_ids = self._order_keys.get(order_key)
            if cert_ids is not None:
                cert_ids.discard(key)
                if not cert_ids:
                    self._order_keys.pop(order_key, None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._order_keys.clear()

    def by_order(self, buyer_id: str, order_id: str) -> tuple[MatchCertificate, ...]:
        cert_ids = sorted(self._order_keys.get((buyer_id, order_id), ()))
        return tuple(self.rows[cert_id] for cert_id in cert_ids)

    def read(self, key: str, *, caller: str | None = None) -> MatchCertificate | None:
        return self.rows.get(key) if _is_trusted_matching_reader(caller) else None


class SupplyPurchaseAuthorityTable(Table[str, SupplyPurchaseAuthority]):
    """Immutable World-issued supply authorities consumed by PSP."""

    def __init__(self) -> None:
        super().__init__(
            name="supply_purchase_authorities",
            allowed_actions=frozenset({"world.issue_supply_purchase_authority"}),
        )

    def write(self, key: str, value: SupplyPurchaseAuthority) -> None:
        from world.errors import IdempotencyConflict

        validate_supply_purchase_authority(value)
        if key != value.authority_id:
            raise IdempotencyConflict("supply purchase authority key mismatch")
        prior = self.rows.get(key)
        if prior is not None and prior != value:
            raise IdempotencyConflict(
                "supply purchase authority id was reused with different fields"
            )
        super().write(key, value)

    def read(
        self, key: str, *, caller: str | None = None
    ) -> SupplyPurchaseAuthority | None:
        row = self.rows.get(key)
        if row is None:
            return None
        if _is_protocol_control_reader(caller) or caller == row.buyer_id:
            return row
        return None


class AuthorityOperationTable(Table[str, AuthorityOperationRecord]):
    """Append-only actor-scoped idempotency bindings for authority writes."""

    def __init__(self) -> None:
        super().__init__(
            name="authority_operations",
            allowed_actions=frozenset(
                {
                    "world.persist_evidence_record",
                    "world.register_mandate_authority",
                    "world.append_mandate_revision",
                    "world.apply_catalog_mutation",
                    "world.apply_negotiation_intent",
                    "world.publish_pricing_policy",
                    "world.issue_cart_quote",
                    "world.checkout_cart_quote",
                    "world.publish_governance_policy",
                    "world.apply_governance_intent",
                    "world.aggregate_reviews",
                    "world.ingest_market_observation",
                    "world.resolve_governance_case",
                    "world.apply_governance_reputation",
                    "world.create_remediation_plan",
                    "world.verify_remediation_step",
                    "world.persist_ranking_context",
                }
            ),
        )

    def write(self, key: str, value: AuthorityOperationRecord) -> None:
        from world.errors import IdempotencyConflict

        expected = authority_operation_key(
            value.scope, value.actor_id, value.idempotency_key
        )
        if key != expected or value.operation_key != expected:
            raise IdempotencyConflict("authority operation key mismatch")
        prior = self.rows.get(key)
        if prior is not None and prior != value:
            raise IdempotencyConflict(
                "authority idempotency key was reused for another request"
            )
        super().write(key, value)

    def read(
        self, key: str, *, caller: str | None = None
    ) -> AuthorityOperationRecord | None:
        row = self.rows.get(key)
        return row if row is None or _is_protocol_control_reader(caller) else None


class PaymentStateTable(Table[str, PaymentStateRecord]):
    """Append-only World payment history, scoped to exact transaction parties."""

    def __init__(self) -> None:
        super().__init__(
            name="payment_states",
            allowed_actions=frozenset(
                {
                    "world.apply_payment_intent",
                    "world.apply_after_sales_intent",
                    "world.settle_order",
                    "world.settle_order_partial",
                    "world.dispatch_order",
                    "world.refund_order",
                }
            ),
        )
        self._by_order: dict[str, list[str]] = {}
        self._by_payment: dict[str, list[str]] = {}

    def write(self, key: str, value: PaymentStateRecord) -> None:
        from world.errors import IdempotencyConflict

        validate_payment_state_record(value)
        if key != payment_state_key(value):
            raise IdempotencyConflict("payment state key is not canonical")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict("payment state version already exists")
        history = self.history_for_payment(value.payment_id)
        previous = history[-1] if history else None
        validate_payment_transition(previous, value, server_tick=value.logical_tick)
        super().write(key, value)
        self._by_order.setdefault(value.order_id, []).append(key)
        self._by_payment.setdefault(value.payment_id, []).append(key)

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            for index, identity in (
                (self._by_order, prior.order_id),
                (self._by_payment, prior.payment_id),
            ):
                keys = index.get(identity)
                if keys is not None:
                    keys.remove(key)
                    if not keys:
                        index.pop(identity, None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._by_order.clear()
        self._by_payment.clear()

    def read(
        self, key: str, *, caller: str | None = None
    ) -> PaymentStateRecord | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        return row if caller in {row.owner_id, row.merchant_id} else None

    def history_for_order(self, order_id: str) -> tuple[PaymentStateRecord, ...]:
        return tuple(
            sorted(
                (self.rows[key] for key in self._by_order.get(order_id, ())),
                key=lambda row: row.version,
            )
        )

    def history_for_payment(self, payment_id: str) -> tuple[PaymentStateRecord, ...]:
        return tuple(
            sorted(
                (self.rows[key] for key in self._by_payment.get(payment_id, ())),
                key=lambda row: row.version,
            )
        )


class PackingRecordTable(Table[str, PackingRecord]):
    """Append-only pre-dispatch packing history with owner-aware reads."""

    def __init__(self) -> None:
        super().__init__(
            name="packing_records",
            allowed_actions=frozenset(
                {
                    "world.apply_packing_intent",
                    "world.apply_after_sales_intent",
                    "world.dispatch_order",
                }
            ),
        )
        self._by_order: dict[str, list[str]] = {}
        self._by_packing: dict[str, list[str]] = {}

    def write(self, key: str, value: PackingRecord) -> None:
        from world.errors import IdempotencyConflict

        validate_packing_record(value)
        if key != packing_record_key(value):
            raise IdempotencyConflict("packing record key is not canonical")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict("packing record version already exists")
        history = self.history_for_packing(value.packing_id)
        previous = history[-1] if history else None
        validate_packing_transition(previous, value, server_tick=value.logical_tick)
        super().write(key, value)
        self._by_order.setdefault(value.order_id, []).append(key)
        self._by_packing.setdefault(value.packing_id, []).append(key)

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            for index, identity in (
                (self._by_order, prior.order_id),
                (self._by_packing, prior.packing_id),
            ):
                keys = index.get(identity)
                if keys is not None:
                    keys.remove(key)
                    if not keys:
                        index.pop(identity, None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._by_order.clear()
        self._by_packing.clear()

    def read(self, key: str, *, caller: str | None = None) -> PackingRecord | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        return row if caller in {row.owner_id, row.merchant_id} else None

    def history_for_order(self, order_id: str) -> tuple[PackingRecord, ...]:
        return tuple(
            sorted(
                (self.rows[key] for key in self._by_order.get(order_id, ())),
                key=lambda row: row.version,
            )
        )

    def history_for_packing(self, packing_id: str) -> tuple[PackingRecord, ...]:
        return tuple(
            sorted(
                (self.rows[key] for key in self._by_packing.get(packing_id, ())),
                key=lambda row: row.version,
            )
        )


class AfterSalesPolicyTable(Table[str, AfterSalesPolicyRevision]):
    """Versioned merchant after-sales policies published by Platform policy."""

    def __init__(self) -> None:
        super().__init__(
            name="after_sales_policies",
            allowed_actions=frozenset({"world.publish_after_sales_policy"}),
        )
        self._by_merchant: dict[str, list[str]] = {}

    def write(self, key: str, value: AfterSalesPolicyRevision) -> None:
        from world.errors import IdempotencyConflict

        validate_after_sales_policy(value)
        expected = f"{value.merchant_id}:{value.revision}"
        if key != expected:
            raise IdempotencyConflict("after-sales policy key is not canonical")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict("after-sales policy revision already exists")
        super().write(key, value)
        self._by_merchant.setdefault(value.merchant_id, []).append(key)

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            keys = self._by_merchant.get(prior.merchant_id)
            if keys is not None:
                keys.remove(key)
                if not keys:
                    self._by_merchant.pop(prior.merchant_id, None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._by_merchant.clear()

    def read(
        self, key: str, *, caller: str | None = None
    ) -> AfterSalesPolicyRevision | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        return row if caller == row.merchant_id else None

    def latest(self, merchant_id: str) -> AfterSalesPolicyRevision | None:
        keys = self._by_merchant.get(merchant_id, ())
        if not keys:
            return None
        return max((self.rows[key] for key in keys), key=lambda row: row.revision)


class AfterSalesRecordTable(Table[str, AfterSalesWrite]):
    """One physical collection for exact typed after-sales domain records."""

    def __init__(self) -> None:
        super().__init__(
            name="after_sales_records",
            allowed_actions=frozenset({"world.apply_after_sales_intent"}),
        )
        self._by_order: dict[str, list[str]] = {}

    def write(self, key: str, value: AfterSalesWrite) -> None:
        from world.errors import IdempotencyConflict

        validate_after_sales_write(value)
        expected = physical_after_sales_record_key(value)
        if key != expected:
            raise IdempotencyConflict("after-sales record key is not canonical")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict("after-sales record already exists")
        super().write(key, value)
        self._by_order.setdefault(_after_sales_order_id(value), []).append(key)

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            order_id = _after_sales_order_id(prior)
            keys = self._by_order.get(order_id)
            if keys is not None:
                keys.remove(key)
                if not keys:
                    self._by_order.pop(order_id, None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._by_order.clear()

    def read(self, key: str, *, caller: str | None = None) -> AfterSalesWrite | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        owner_id, merchant_id = _after_sales_parties(row)
        return row if caller in {owner_id, merchant_id} else None

    def history_for_order(self, order_id: str) -> tuple[AfterSalesWrite, ...]:
        return tuple(
            self.rows[key] for key in sorted(self._by_order.get(order_id, ()))
        )


class GovernancePolicyTable(Table[str, GovernancePolicyEnvelope]):
    """Append-only marketplace-governance policies with protected reads."""

    def __init__(self) -> None:
        super().__init__(
            name="governance_policies",
            allowed_actions=frozenset({"world.publish_governance_policy"}),
        )

    def write(self, key: str, value: GovernancePolicyEnvelope) -> None:
        from world.errors import IdempotencyConflict

        validate_policy_envelope(value)
        if key != envelope_key(value):
            raise IdempotencyConflict("governance policy key is not canonical")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict("governance policy revision already exists")
        projection = GovernanceTables()
        # Physical table keys end in a textual version suffix, so lexical
        # order places revision 10 before revision 2.  Rebuild the typed
        # projection in stream and numeric revision order before validating a
        # new write.
        prior_rows = sorted(
            self.all(),
            key=lambda item: (
                item[1].kind,
                item[1].stable_id,
                item[1].revision,
            ),
        )
        for prior_key, prior_row in prior_rows:
            projection.append(
                GovernanceWrite("governance_policies", prior_key, prior_row)
            )
        projection.append(GovernanceWrite("governance_policies", key, value))
        super().write(key, value)

    def read(
        self, key: str, *, caller: str | None = None
    ) -> GovernancePolicyEnvelope | None:
        row = self.rows.get(key)
        if row is None or caller is None:
            return None
        if (
            caller == "world"
            or caller == row.service_actor
            or caller == "runtime"
            or caller.startswith("runtime:")
        ):
            return row
        visible = {*row.owner_ids, *row.subject_ids, row.original_actor}
        return row if caller in visible else None


class GovernanceRecordTable(Table[str, GovernanceRecordEnvelope]):
    """Append-only marketplace-governance outcomes with protected reads."""

    def __init__(self) -> None:
        super().__init__(
            name="governance_records",
            allowed_actions=frozenset(
                {
                    "world.apply_governance_intent",
                    "world.aggregate_reviews",
                    "world.ingest_market_observation",
                    "world.resolve_governance_case",
                    "world.apply_governance_reputation",
                    "world.create_remediation_plan",
                    "world.verify_remediation_step",
                    "world.persist_ranking_context",
                }
            ),
        )

    def write(self, key: str, value: GovernanceRecordEnvelope) -> None:
        from world.errors import IdempotencyConflict

        validate_record_envelope(value)
        if key != envelope_key(value):
            raise IdempotencyConflict("governance record key is not canonical")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict("governance record version already exists")
        projection = GovernanceTables()
        # A governance stream can exceed nine versions.  Never use the
        # physical table's lexical key order to reconstruct its lineage.
        prior_rows = sorted(
            self.all(),
            key=lambda item: (
                item[1].kind,
                item[1].stable_id,
                item[1].version,
            ),
        )
        for prior_key, prior_row in prior_rows:
            projection.append(
                GovernanceWrite("governance_records", prior_key, prior_row)
            )
        projection.append(GovernanceWrite("governance_records", key, value))
        super().write(key, value)

    def read(
        self, key: str, *, caller: str | None = None
    ) -> GovernanceRecordEnvelope | None:
        row = self.rows.get(key)
        if row is None or caller is None:
            return None
        if (
            caller == "world"
            or caller == row.service_actor
            or caller == "runtime"
            or caller.startswith("runtime:")
        ):
            return row
        visible = {*row.owner_ids, *row.subject_ids, row.original_actor}
        return row if caller in visible else None


def _after_sales_parties(write: AfterSalesWrite) -> tuple[str, str]:
    row = write.value
    binding = row if write.table == "after_sales_bindings" else getattr(row, "binding")
    return str(binding.owner_id), str(binding.merchant_id)


def _after_sales_order_id(write: AfterSalesWrite) -> str:
    row = write.value
    binding = row if write.table == "after_sales_bindings" else getattr(row, "binding")
    return str(binding.order_id)


class EvidenceRecordTable(Table[str, EvidenceRecord]):
    """Append-only evidence versions with record and digest indexes."""

    def __init__(self) -> None:
        super().__init__(
            name="evidence_records",
            allowed_actions=frozenset({"world.persist_evidence_record"}),
        )
        self._record_versions: dict[str, dict[int, str]] = {}
        self._digests: dict[tuple[str, str], str] = {}

    def write(self, key: str, value: EvidenceRecord) -> None:
        from world.errors import IdempotencyConflict, WorldError

        validate_evidence_record(value)
        if key != evidence_record_key(value.record_id, value.version):
            raise IdempotencyConflict("evidence table key does not match record version")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict(
                "evidence record version was reused for different content"
            )
        current = self.current(value.record_id)
        if current is None:
            if value.version != 1:
                raise WorldError("first evidence record version must be 1")
        else:
            if value.version != current.version + 1:
                raise WorldError("evidence record versions must be contiguous")
            for field in (
                "record_id",
                "kind",
                "subject_id",
                "issuer_id",
                "owner_id",
            ):
                if getattr(value, field) != getattr(current, field):
                    raise WorldError(f"evidence identity field {field} cannot change")
            if value.issued_at_tick <= current.issued_at_tick:
                raise WorldError("evidence issuance ticks must strictly increase")
        digest_key = (value.record_id, value.record_digest)
        if digest_key in self._digests:
            raise IdempotencyConflict("evidence digest already identifies a version")
        super().write(key, value)
        self._record_versions.setdefault(value.record_id, {})[value.version] = key
        self._digests[digest_key] = key

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            versions = self._record_versions.get(prior.record_id)
            if versions is not None:
                versions.pop(prior.version, None)
                if not versions:
                    self._record_versions.pop(prior.record_id, None)
            self._digests.pop((prior.record_id, prior.record_digest), None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._record_versions.clear()
        self._digests.clear()

    def read(
        self, key: str, *, caller: str | None = None
    ) -> EvidenceRecord | None:
        row = self.rows.get(key)
        if row is None or caller is None:
            return row
        return authorize_persisted_evidence_read(row, reader_id=caller)

    def current(self, record_id: str) -> EvidenceRecord | None:
        versions = self._record_versions.get(record_id)
        if not versions:
            return None
        return self.rows[versions[max(versions)]]

    def by_digest(self, record_id: str, digest: str) -> EvidenceRecord | None:
        key = self._digests.get((record_id, digest))
        return None if key is None else self.rows.get(key)

    def versions(self, record_id: str) -> tuple[EvidenceRecord, ...]:
        versions = self._record_versions.get(record_id, {})
        return tuple(self.rows[versions[number]] for number in sorted(versions))


class MandateAuthorityTable(Table[str, MandateRevisionAuthority]):
    """Immutable principal-to-buyer authority for one mandate."""

    def __init__(self) -> None:
        super().__init__(
            name="mandate_authorities",
            allowed_actions=frozenset({"world.register_mandate_authority"}),
        )

    def write(self, key: str, value: MandateRevisionAuthority) -> None:
        from world.errors import IdempotencyConflict

        if key != value.mandate_id:
            raise IdempotencyConflict("mandate authority key mismatch")
        prior = self.rows.get(key)
        if prior is not None and prior != value:
            raise IdempotencyConflict("mandate authority cannot change after registration")
        super().write(key, value)

    def read(
        self, key: str, *, caller: str | None = None
    ) -> MandateRevisionAuthority | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        return row if caller in {row.principal_id, row.buyer_id} else None


class MandateRevisionTable(Table[str, MandateRevision]):
    """Append-only, digest-linked revision histories keyed by mandate."""

    def __init__(self) -> None:
        super().__init__(
            name="mandate_revisions",
            allowed_actions=frozenset({"world.append_mandate_revision"}),
        )
        self._mandates: dict[str, dict[int, str]] = {}

    def write(self, key: str, value: MandateRevision) -> None:
        from world.errors import IdempotencyConflict, WorldError

        validate_mandate_revision(value)
        if key != mandate_revision_key(value.mandate_id, value.revision):
            raise IdempotencyConflict("mandate revision table key mismatch")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict("mandate revision number was reused")
        current = self.current(value.mandate_id)
        if current is None:
            if value.revision != 1:
                raise WorldError("first mandate revision must be 1")
        else:
            if value.revision != current.revision + 1:
                raise WorldError("mandate revisions must be contiguous")
            if value.previous_digest != current.revision_digest:
                raise WorldError("mandate previous digest does not match current revision")
            if value.logical_tick <= current.logical_tick:
                raise WorldError("mandate logical ticks must strictly increase")
            for field in ("principal_id", "buyer_id", "mandate_id"):
                if getattr(value, field) != getattr(current, field):
                    raise WorldError(f"mandate identity field {field} cannot change")
        super().write(key, value)
        self._mandates.setdefault(value.mandate_id, {})[value.revision] = key

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            revisions = self._mandates.get(prior.mandate_id)
            if revisions is not None:
                revisions.pop(prior.revision, None)
                if not revisions:
                    self._mandates.pop(prior.mandate_id, None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._mandates.clear()

    def read(
        self, key: str, *, caller: str | None = None
    ) -> MandateRevision | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        return row if caller in {row.principal_id, row.buyer_id} else None

    def current(self, mandate_id: str) -> MandateRevision | None:
        revisions = self._mandates.get(mandate_id)
        if not revisions:
            return None
        return self.rows[revisions[max(revisions)]]

    def by_mandate(self, mandate_id: str) -> tuple[MandateRevision, ...]:
        revisions = self._mandates.get(mandate_id, {})
        return tuple(self.rows[revisions[number]] for number in sorted(revisions))


class ListingClaimTable(Table[str, ListingClaim]):
    """Current claim aggregates with complete append-only version history."""

    def __init__(self) -> None:
        super().__init__(
            name="listing_claims",
            allowed_actions=frozenset({"world.apply_listing_claim"}),
        )
        self._listing_keys: dict[str, set[str]] = {}
        self._actor_keys: dict[tuple[str, str], str] = {}

    def write(self, key: str, value: ListingClaim) -> None:
        from world.errors import IdempotencyConflict

        validate_listing_claim(value)
        if key != value.claim_id:
            raise IdempotencyConflict("listing claim table key mismatch")
        prior = self.rows.get(key)
        if prior is not None:
            if value == prior:
                return
            if len(value.versions) != len(prior.versions) + 1:
                raise IdempotencyConflict("listing claim must append one version")
            if value.versions[:-1] != prior.versions:
                raise IdempotencyConflict("listing claim history cannot be rewritten")
        for version in value.versions:
            actor_key = (value.merchant_id, version.idempotency_key)
            indexed = self._actor_keys.get(actor_key)
            if indexed is not None and indexed != key:
                raise IdempotencyConflict(
                    "merchant claim idempotency key belongs to another claim"
                )
        if prior is not None:
            self._drop_indexes(key, prior)
        super().write(key, value)
        self._add_indexes(key, value)

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            self._drop_indexes(key, prior)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._listing_keys.clear()
        self._actor_keys.clear()

    def by_listing(self, listing_id: str) -> tuple[ListingClaim, ...]:
        return tuple(
            self.rows[key]
            for key in sorted(self._listing_keys.get(listing_id, ()))
        )

    def by_actor_key(
        self, merchant_id: str, idempotency_key: str
    ) -> ListingClaim | None:
        key = self._actor_keys.get((merchant_id, idempotency_key))
        return None if key is None else self.rows.get(key)

    def _add_indexes(self, key: str, value: ListingClaim) -> None:
        self._listing_keys.setdefault(value.listing_id, set()).add(key)
        for version in value.versions:
            self._actor_keys[(value.merchant_id, version.idempotency_key)] = key

    def _drop_indexes(self, key: str, value: ListingClaim) -> None:
        listing_keys = self._listing_keys.get(value.listing_id)
        if listing_keys is not None:
            listing_keys.discard(key)
            if not listing_keys:
                self._listing_keys.pop(value.listing_id, None)
        for version in value.versions:
            actor_key = (value.merchant_id, version.idempotency_key)
            if self._actor_keys.get(actor_key) == key:
                self._actor_keys.pop(actor_key, None)


class PricingPolicyRevisionTable(Table[str, PricingPolicyRevision]):
    """Append-only merchant pricing policies with stream and retry indexes."""

    def __init__(self) -> None:
        super().__init__(
            name="pricing_policy_revisions",
            allowed_actions=frozenset({"world.publish_pricing_policy"}),
        )
        self._streams: dict[tuple[str, str, str], dict[int, str]] = {}
        self._actor_keys: dict[tuple[str, str, str], str] = {}

    def write(self, key: str, value: PricingPolicyRevision) -> None:
        from world.errors import IdempotencyConflict

        validate_pricing_policy_revision(value)
        expected_key = pricing_policy_revision_key(
            value.market_id, value.merchant_id, value.policy_id, value.revision
        )
        if key != expected_key:
            raise IdempotencyConflict("pricing policy revision key mismatch")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict("pricing policy revision was reused")
        actor_key = (value.market_id, value.actor_id, value.idempotency_key)
        if actor_key in self._actor_keys:
            raise IdempotencyConflict(
                "pricing policy actor idempotency key belongs to another revision"
            )
        stream_key = (value.market_id, value.merchant_id, value.policy_id)
        revisions = self._streams.get(stream_key, {})
        current = self.rows[revisions[max(revisions)]] if revisions else None
        if current is None:
            if value.revision != 1:
                raise IdempotencyConflict("first pricing policy revision must be 1")
            if value.predecessor_digest != GENESIS_PRICING_POLICY_DIGEST:
                raise IdempotencyConflict("first pricing policy must bind genesis")
        else:
            if value.revision != current.revision + 1:
                raise IdempotencyConflict("pricing policy revisions must be contiguous")
            if value.predecessor_digest != current.policy_digest:
                raise IdempotencyConflict("pricing policy predecessor mismatch")
        super().write(key, value)
        self._streams.setdefault(stream_key, {})[value.revision] = key
        self._actor_keys[actor_key] = key

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            stream_key = (prior.market_id, prior.merchant_id, prior.policy_id)
            revisions = self._streams.get(stream_key)
            if revisions is not None:
                revisions.pop(prior.revision, None)
                if not revisions:
                    self._streams.pop(stream_key, None)
            self._actor_keys.pop(
                (prior.market_id, prior.actor_id, prior.idempotency_key), None
            )
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._streams.clear()
        self._actor_keys.clear()

    def read(
        self, key: str, *, caller: str | None = None
    ) -> PricingPolicyRevision | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        return row if caller in {row.owner_id, row.merchant_id} else None

    def by_stream(
        self, market_id: str, merchant_id: str, policy_id: str
    ) -> tuple[PricingPolicyRevision, ...]:
        revisions = self._streams.get((market_id, merchant_id, policy_id), {})
        return tuple(self.rows[revisions[number]] for number in sorted(revisions))

    def for_market_merchant(
        self, market_id: str, merchant_id: str
    ) -> tuple[PricingPolicyRevision, ...]:
        rows = (
            row
            for row in self.rows.values()
            if row.market_id == market_id and row.merchant_id == merchant_id
        )
        return tuple(sorted(rows, key=lambda row: (row.policy_id, row.revision)))


class PersistentCartQuoteTable(Table[str, PersistentCartQuote]):
    """Immutable World-issued cart quotes with owner-scoped visibility."""

    def __init__(self) -> None:
        super().__init__(
            name="persistent_cart_quotes",
            allowed_actions=frozenset({"world.issue_cart_quote"}),
        )
        self._actor_keys: dict[tuple[str, str, str], str] = {}
        self._request_keys: dict[tuple[str, str], str] = {}

    def write(self, key: str, value: PersistentCartQuote) -> None:
        from world.errors import IdempotencyConflict

        validate_persistent_cart_quote(value)
        if key != value.quote_id:
            raise IdempotencyConflict("persistent cart quote key mismatch")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict("persistent cart quote id was reused")
        actor_key = (value.market_id, value.requested_by, value.idempotency_key)
        if actor_key in self._actor_keys:
            raise IdempotencyConflict(
                "cart quote actor idempotency key belongs to another quote"
            )
        request_key = (value.market_id, value.request_id)
        if request_key in self._request_keys:
            raise IdempotencyConflict("cart quote request id was reused")
        super().write(key, value)
        self._actor_keys[actor_key] = key
        self._request_keys[request_key] = key

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            self._actor_keys.pop(
                (prior.market_id, prior.requested_by, prior.idempotency_key), None
            )
            self._request_keys.pop((prior.market_id, prior.request_id), None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._actor_keys.clear()
        self._request_keys.clear()

    def read(
        self, key: str, *, caller: str | None = None
    ) -> PersistentCartQuote | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        return (
            row
            if caller in {row.buyer_id, row.principal_id}
            else None
        )


class PersistentCartQuoteRequestTable(Table[str, PersistentCartQuoteRequest]):
    """World-issued merchant quote authorizations with participant ACLs."""

    def __init__(self) -> None:
        super().__init__(
            name="persistent_cart_quote_requests",
            allowed_actions=frozenset({"world.create_cart_quote_request"}),
        )
        self._actor_keys: dict[tuple[str, str, str], str] = {}

    def write(self, key: str, value: PersistentCartQuoteRequest) -> None:
        from world.errors import IdempotencyConflict

        validate_persistent_cart_quote_request(value)
        if key != value.request_id:
            raise IdempotencyConflict("persistent cart quote request key mismatch")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict("persistent cart quote request id was reused")
        actor_key = (value.market_id, value.created_by, value.idempotency_key)
        if actor_key in self._actor_keys:
            raise IdempotencyConflict(
                "cart quote request idempotency key belongs to another request"
            )
        super().write(key, value)
        self._actor_keys[actor_key] = key

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            self._actor_keys.pop(
                (prior.market_id, prior.created_by, prior.idempotency_key), None
            )
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._actor_keys.clear()

    def read(
        self, key: str, *, caller: str | None = None
    ) -> PersistentCartQuoteRequest | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        allowed = {
            row.buyer_id,
            row.principal_id,
            row.created_by,
        }
        return row if caller in allowed else None


class NegotiationEventTable(Table[str, NegotiationEvent]):
    """Append-only negotiation events visible only to exact participants."""

    def __init__(self) -> None:
        super().__init__(
            name="negotiation_events",
            allowed_actions=frozenset({"world.apply_negotiation_intent"}),
        )
        self._actor_keys: dict[tuple[str, str, str], str] = {}
        self._stream_sequences: dict[tuple[str, int], str] = {}
        self._thread_events: dict[str, set[str]] = {}

    def write(self, key: str, value: NegotiationEvent) -> None:
        from world.errors import IdempotencyConflict

        validate_negotiation_event(value)
        if key != value.event_id:
            raise IdempotencyConflict("negotiation event table key mismatch")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict("negotiation event id was reused")
        actor_key = (
            value.negotiation_id,
            value.actor_id,
            value.idempotency_key,
        )
        prior_id = self._actor_keys.get(actor_key)
        if prior_id is not None:
            raise IdempotencyConflict(
                "negotiation actor idempotency key belongs to another event"
            )
        sequence_key = (value.negotiation_id, value.sequence_no)
        prior_id = self._stream_sequences.get(sequence_key)
        if prior_id is not None:
            raise IdempotencyConflict(
                "negotiation sequence belongs to another event"
            )
        super().write(key, value)
        self._actor_keys[actor_key] = key
        self._stream_sequences[sequence_key] = key
        self._thread_events.setdefault(value.negotiation_id, set()).add(key)

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            self._actor_keys.pop(
                (prior.negotiation_id, prior.actor_id, prior.idempotency_key),
                None,
            )
            self._stream_sequences.pop(
                (prior.negotiation_id, prior.sequence_no), None
            )
            event_ids = self._thread_events.get(prior.negotiation_id)
            if event_ids is not None:
                event_ids.discard(key)
                if not event_ids:
                    self._thread_events.pop(prior.negotiation_id, None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._actor_keys.clear()
        self._stream_sequences.clear()
        self._thread_events.clear()

    def read(
        self, key: str, *, caller: str | None = None
    ) -> NegotiationEvent | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        return row if caller in {row.buyer_id, row.merchant_id} else None

    def by_actor_key(
        self, negotiation_id: str, actor_id: str, idempotency_key: str
    ) -> NegotiationEvent | None:
        key = self._actor_keys.get((negotiation_id, actor_id, idempotency_key))
        return None if key is None else self.rows.get(key)

    def by_thread(self, negotiation_id: str) -> tuple[NegotiationEvent, ...]:
        return tuple(
            sorted(
                (
                    self.rows[event_id]
                    for event_id in self._thread_events.get(negotiation_id, ())
                ),
                key=lambda event: (event.sequence_no, event.event_id),
            )
        )


class NegotiationThreadTable(Table[str, NegotiationThread]):
    """Materialized negotiation heads with participant-scoped visibility."""

    def __init__(self) -> None:
        super().__init__(
            name="negotiation_threads",
            allowed_actions=frozenset({"world.apply_negotiation_intent"}),
        )

    def write(self, key: str, value: NegotiationThread) -> None:
        from world.errors import IdempotencyConflict

        validate_negotiation_thread(value)
        if key != value.negotiation_id:
            raise IdempotencyConflict("negotiation thread table key mismatch")
        prior = self.rows.get(key)
        if prior is not None:
            if value.event_count != prior.event_count + 1:
                raise IdempotencyConflict(
                    "negotiation thread must advance by one event"
                )
            for field in (
                "negotiation_id",
                "buyer_id",
                "merchant_id",
                "offer_id",
                "sku_id",
                "listing_digest",
                "listing_revision",
                "currency",
                "qty",
                "max_rounds",
                "opened_at_tick",
                "expires_at_tick",
            ):
                if getattr(value, field) != getattr(prior, field):
                    raise IdempotencyConflict(
                        f"negotiation thread field {field} cannot change"
                    )
        super().write(key, value)

    def read(
        self, key: str, *, caller: str | None = None
    ) -> NegotiationThread | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        return row if caller in {row.buyer_id, row.merchant_id} else None


class ProtocolEventTable(Table[str, ProtocolEvent]):
    """Append-only authority events, visible only to their recipient/control plane."""

    def __init__(self) -> None:
        super().__init__(
            name="protocol_events",
            allowed_actions=frozenset({"world.publish_protocol_event"}),
        )
        self._stream_sequences: dict[tuple[str, int], str] = {}
        self._actor_keys: dict[tuple[str, str, str], str] = {}
        self._order_keys: dict[str, set[str]] = {}

    def write(self, key: str, value: ProtocolEvent) -> None:
        from world.errors import IdempotencyConflict

        if key != value.event_id:
            raise IdempotencyConflict("protocol event key does not match event_id")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict(f"protocol event id {key!r} was reused")
        sequence_key = (value.binding.binding_digest, value.sequence)
        prior_id = self._stream_sequences.get(sequence_key)
        if prior_id is not None:
            raise IdempotencyConflict(
                f"protocol event stream sequence already belongs to {prior_id!r}"
            )
        actor_key = (
            value.binding.binding_digest,
            value.actor_id,
            value.idempotency_key,
        )
        prior_id = self._actor_keys.get(actor_key)
        if prior_id is not None:
            raise IdempotencyConflict(
                f"protocol event actor idempotency key already belongs to {prior_id!r}"
            )
        super().write(key, value)
        self._stream_sequences[sequence_key] = key
        self._actor_keys[actor_key] = key
        self._order_keys.setdefault(value.binding.order_id, set()).add(key)

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            self._stream_sequences.pop((prior.binding.binding_digest, prior.sequence), None)
            self._actor_keys.pop(
                (
                    prior.binding.binding_digest,
                    prior.actor_id,
                    prior.idempotency_key,
                ),
                None,
            )
            order_ids = self._order_keys.get(prior.binding.order_id)
            if order_ids is not None:
                order_ids.discard(key)
                if not order_ids:
                    self._order_keys.pop(prior.binding.order_id, None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._stream_sequences.clear()
        self._actor_keys.clear()
        self._order_keys.clear()

    def read(self, key: str, *, caller: str | None = None) -> ProtocolEvent | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        return row if caller == row.binding.recipient_id else None

    def by_stream(self, binding_digest: str) -> tuple[ProtocolEvent, ...]:
        rows = (row for row in self.rows.values() if row.binding.binding_digest == binding_digest)
        return tuple(sorted(rows, key=lambda row: row.sequence))

    def by_order(self, order_id: str) -> tuple[ProtocolEvent, ...]:
        event_ids = sorted(self._order_keys.get(order_id, ()))
        return tuple(
            sorted(
                (self.rows[event_id] for event_id in event_ids),
                key=lambda row: (row.binding.stream_id, row.sequence, row.event_id),
            )
        )


class ProtocolReceiptTable(Table[str, ProtocolEventReceipt]):
    """One append-only recipient decision for each exact persisted event."""

    def __init__(self) -> None:
        super().__init__(
            name="protocol_receipts",
            allowed_actions=frozenset(
                {
                    "world.append_protocol_receipt",
                    "world.process_protocol_event",
                }
            ),
        )
        self._event_keys: dict[tuple[str, str], str] = {}
        self._actor_keys: dict[tuple[str, str, str], str] = {}
        self._order_keys: dict[str, set[str]] = {}

    def write(self, key: str, value: ProtocolEventReceipt) -> None:
        from world.errors import IdempotencyConflict

        if key != value.receipt_id:
            raise IdempotencyConflict("protocol receipt key does not match receipt_id")
        prior = self.rows.get(key)
        if prior is not None:
            if prior == value:
                return
            raise IdempotencyConflict(f"protocol receipt id {key!r} was reused")
        event_key = (value.binding.binding_digest, value.event_digest)
        prior_id = self._event_keys.get(event_key)
        if prior_id is not None:
            raise IdempotencyConflict(f"protocol event already has receipt {prior_id!r}")
        actor_key = (
            value.binding.binding_digest,
            value.actor_id,
            value.idempotency_key,
        )
        prior_id = self._actor_keys.get(actor_key)
        if prior_id is not None:
            raise IdempotencyConflict(
                f"protocol receipt actor idempotency key belongs to {prior_id!r}"
            )
        super().write(key, value)
        self._event_keys[event_key] = key
        self._actor_keys[actor_key] = key
        self._order_keys.setdefault(value.binding.order_id, set()).add(key)

    def delete(self, key: str) -> None:
        prior = self.rows.get(key)
        if prior is not None:
            self._event_keys.pop((prior.binding.binding_digest, prior.event_digest), None)
            self._actor_keys.pop(
                (
                    prior.binding.binding_digest,
                    prior.actor_id,
                    prior.idempotency_key,
                ),
                None,
            )
            receipt_ids = self._order_keys.get(prior.binding.order_id)
            if receipt_ids is not None:
                receipt_ids.discard(key)
                if not receipt_ids:
                    self._order_keys.pop(prior.binding.order_id, None)
        super().delete(key)

    def clear(self) -> None:
        super().clear()
        self._event_keys.clear()
        self._actor_keys.clear()
        self._order_keys.clear()

    def read(self, key: str, *, caller: str | None = None) -> ProtocolEventReceipt | None:
        row = self.rows.get(key)
        if row is None or _is_protocol_control_reader(caller):
            return row
        return row if caller == row.binding.recipient_id else None

    def by_event(self, binding_digest: str, event_digest: str) -> ProtocolEventReceipt | None:
        key = self._event_keys.get((binding_digest, event_digest))
        return None if key is None else self.rows.get(key)

    def by_order(self, order_id: str) -> tuple[ProtocolEventReceipt, ...]:
        receipt_ids = sorted(self._order_keys.get(order_id, ()))
        return tuple(
            sorted(
                (self.rows[receipt_id] for receipt_id in receipt_ids),
                key=lambda row: (row.logical_tick, row.receipt_id),
            )
        )


def _is_platform(caller: str | None) -> bool:
    return caller is None or caller == "platform" or caller.startswith("platform:")


def _is_trusted_matching_reader(caller: str | None) -> bool:
    return (
        _is_platform(caller)
        or caller == "runtime"
        or bool(caller and caller.startswith("runtime:"))
    )


def _is_protocol_control_reader(caller: str | None) -> bool:
    return (
        _is_platform(caller)
        or caller == "runtime"
        or bool(caller and caller.startswith("runtime:"))
    )


def _matches_filters(row: Listing, filters: dict[str, object]) -> bool:
    for key, expected in filters.items():
        actual = getattr(row, key, row.attributes.get(key))
        if isinstance(expected, (set, tuple, list, frozenset)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True
