"""The agent-facing read view of the world.

The runtime constructs one of these per send, bound to the receiving agent's
id. Partition rules apply automatically by caller scope; an agent cannot pass
a different agent's id (WORLD_CLASSES.md §2.2).

**Tools surface invariants:**
    1. Tools are read-only. Writes happen by emitting an envelope.
    2. Tool calls are caller-scoped automatically.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, cast

from world.types import Money, ReputationScore

if TYPE_CHECKING:
    from protocol.evidence_records import EvidenceRecord, MandateRevision
    from protocol.listing_claims import ListingClaim
    from world.state import World
    from world.types import (
        AgentId,
        Dispute,
        DisputeId,
        Listing,
        Money,
        Order,
        OrderId,
        ReputationScore,
        Review,
        Ruling,
        SkuId,
    )


class WorldTools:
    """Read-only, caller-scoped facade over ``World``.

    Constructed per send by the runtime; never instantiated by agent code.
    """

    def __init__(self, world: "World", *, caller_id: str) -> None:
        """Bind the tools to ``caller_id``; partition checks use this id."""
        self._world = world
        self._caller = caller_id

    # --- Catalog -----------------------------------------------------

    def search_catalog(
        self, query: str, filters: dict[str, object] | None = None, *, limit: int = 10
    ) -> "list[Listing]":
        """Return up to ``limit`` listings. Order is deterministic given the seed."""
        if limit <= 0:
            return []
        if hasattr(self._world, "search_catalog"):
            return cast(
                "list[Listing]",
                self._world.search_catalog(query, filters, limit=limit),
            )[:limit]
        catalog = self._world._tables["catalog"]
        results = cast("list[Listing]", catalog.search(query, filters))  # type: ignore[attr-defined]
        return results[:limit]

    def get_listing(self, sku_id: "SkuId") -> "Listing | None":
        """Return the listing or ``None``."""
        return self._world.read("catalog", sku_id, caller=self._caller)

    # --- Inventory (public surface only) -----------------------------

    def is_in_stock(self, sku_id: "SkuId", *, qty: int = 1) -> bool:
        """Return ``True`` iff at least ``qty`` units are available.

        Exact stock counts are intentionally not exposed.
        """
        row = self._world.read("inventory", sku_id, caller=self._caller)
        if row is None:
            return False
        available = getattr(row, "qty_available", row)
        reserved = getattr(row, "qty_reserved", 0)
        return int(available) - int(reserved) >= qty

    # --- Ledger / orders, caller-scoped ------------------------------

    def get_balance(self) -> "Money":
        """Return the calling agent's current ledger balance."""
        total = Decimal("0")
        currency = "USD"
        rows = (
            self._world.iter_table("ledger")
            if hasattr(self._world, "iter_table")
            else self._world._tables["ledger"].all()
        )
        for _, receipt in rows:
            price = receipt.price
            currency = price.currency
            amount = price.amount * receipt.qty
            if receipt.effect == "charge":
                buyer_delta = -amount
                merchant_delta = amount
            elif receipt.effect == "refund":
                buyer_delta = amount
                merchant_delta = -amount
            else:
                raise ValueError(
                    f"unsupported authoritative receipt effect: {receipt.effect!r}"
                )
            if str(receipt.buyer_id) == self._caller:
                total += buyer_delta
            if str(receipt.merchant_id) == self._caller:
                total += merchant_delta
        return Money(total, currency)

    def get_order(self, order_id: "OrderId") -> "Order | None":
        """Return the order if the caller is buyer or merchant on it, else ``None``."""
        return self._world.read("orders", order_id, caller=self._caller)

    def get_protocol_event_authority(self, event_id: str) -> dict[str, object]:
        """Return fresh caller-scoped authority for one delivered callback.

        This is an Agent-private prerequisite, never an LLM tool.  It is read
        when the Agent actually handles the delivery, so callbacks queued
        together cannot carry an authority snapshot captured before an earlier
        callback committed.
        """

        from protocol.event_receipts import (
            ProtocolEvent,
            protocol_event_receipt_to_dict,
            protocol_event_to_dict,
        )
        from world.types import OrderId

        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("protocol event authority event_id must be non-empty")
        event = self._world.read(
            "protocol_events",
            event_id,
            caller=self._caller,
        )
        if not isinstance(event, ProtocolEvent):
            raise ValueError("protocol event authority is not visible to the actor")
        if event.binding.recipient_id != self._caller:
            raise ValueError("protocol event authority belongs to another actor")
        order_id = OrderId(event.binding.order_id)
        order = self.get_order(order_id)
        if order is None:
            raise ValueError("protocol event order is not visible to the actor")
        revision = self._world.read_order_state_revision(
            order_id,
            caller=self._caller,
        )
        reference_digest = self._world.read_order_operation_reference(
            order_id,
            caller=self._caller,
        )
        if revision is None or reference_digest is None:
            raise RuntimeError("visible order has no authoritative protocol state")
        receipts = self._world.protocol_receipts_for_stream(
            event.binding.binding_digest,
            order_id=order_id,
            caller=self._caller,
        )
        receipt_by_event_id = {receipt.event_id: receipt for receipt in receipts}
        event_rows = (
            self._world.iter_table("protocol_events")
            if hasattr(self._world, "iter_table")
            else self._world._tables["protocol_events"].all()
        )
        prior_events = sorted(
            (
                prior_event
                for _, prior_event in event_rows
                if isinstance(prior_event, ProtocolEvent)
                and prior_event.binding == event.binding
                and prior_event.sequence < event.sequence
            ),
            key=lambda prior_event: prior_event.sequence,
        )
        prior_event_states: list[dict[str, object]] = []
        for prior_event in prior_events:
            receipt = receipt_by_event_id.get(prior_event.event_id)
            prior_event_states.append(
                {
                    "event_kind": prior_event.event_kind,
                    "reference_kind": prior_event.reference_kind,
                    "same_business_reference": (
                        prior_event.reference_digest == event.reference_digest
                    ),
                    "required_order_state": prior_event.required_order_state,
                    "required_state_revision": prior_event.required_state_revision,
                    "decision": None if receipt is None else receipt.decision,
                    "observed_order_state": (
                        None if receipt is None else receipt.observed_order_state
                    ),
                    "observed_state_revision": (
                        None if receipt is None else receipt.observed_state_revision
                    ),
                    "logical_tick": None if receipt is None else receipt.logical_tick,
                }
            )
        logical_tick = int(self._world.logical_time)
        reference_state: dict[str, object] = {"kind": event.reference_kind}
        if event.reference_kind == "certificate":
            certificate_rows = (
                self._world.iter_table("match_certificates")
                if hasattr(self._world, "iter_table")
                else self._world._tables["match_certificates"].all()
            )
            certificates = [
                certificate
                for _, certificate in certificate_rows
                if certificate.certificate_digest == event.reference_digest
            ]
            if len(certificates) != 1:
                raise RuntimeError(
                    "protocol event certificate reference is not uniquely visible"
                )
            certificate = certificates[0]
            if (
                certificate.order_id != str(order.order_id)
                or certificate.buyer_id != str(order.buyer_id)
                or certificate.merchant_id != str(order.merchant_id)
            ):
                raise RuntimeError(
                    "protocol event certificate crosses authoritative order parties"
                )
            reference_state.update(
                {
                    "valid_from_tick": certificate.issued_at_tick,
                    "valid_until_tick": certificate.expires_at_tick,
                }
            )
        return {
            "current_event": protocol_event_to_dict(event),
            "current_order_state": {
                "order_id": str(order.order_id),
                "buyer_id": str(order.buyer_id),
                "merchant_id": str(order.merchant_id),
                "state": order.state.value,
                "state_revision": revision,
                "operation_reference_digest": reference_digest,
                "logical_time": logical_tick,
            },
            "current_receipts": [
                protocol_event_receipt_to_dict(receipt)
                for receipt in receipts
            ],
            "current_reference_state": reference_state,
            "prior_event_states": prior_event_states,
            "current_tick": logical_tick,
        }

    def get_after_sales_authority(self, order_id: "OrderId | str") -> object:
        """Return one sealed, caller-scoped after-sales authority projection.

        This is an Agent-private framework read, not an LLM tool.  It derives
        the current business affordances from the same World context and
        planner used by real after-sales commits.  Candidate replacement and
        evidence rows are enumerated inside the caller-bound World facade, so
        neither their internal digests nor an unauthorized row can be supplied
        by model text.
        """

        from protocol.evidence_records import EvidenceRecord
        from world.after_sales_authority_projection import (
            issue_after_sales_authority_projection,
        )
        from world.after_sales_service import TrustedReplacementContext
        from world.types import Listing, OrderId

        normalized_order_id = str(order_id)
        if not normalized_order_id:
            raise ValueError("after-sales authority order_id must be non-empty")
        order = self.get_order(OrderId(normalized_order_id))
        if order is None:
            raise ValueError("after-sales authority order is not visible to the actor")
        loader = getattr(self._world, "load_after_sales_context", None)
        if not callable(loader):
            raise RuntimeError("World has no trusted after-sales context loader")
        context = loader(
            normalized_order_id,
            original_actor=self._caller,
        )

        replacements: dict[str, TrustedReplacementContext] = {}
        for _, candidate in _world_rows(self._world, "catalog"):
            if (
                not isinstance(candidate, Listing)
                or str(candidate.merchant_id) != str(order.merchant_id)
            ):
                continue
            candidate_context = loader(
                normalized_order_id,
                intent={
                    "op": "request_exchange",
                    "order_id": normalized_order_id,
                    "replacement_sku_id": str(candidate.sku_id),
                    "reason": "Agent authority projection",
                },
                original_actor=self._caller,
            )
            replacement = candidate_context.replacement
            if replacement is not None:
                replacements[replacement.sku_id] = replacement

        evidence: dict[str, EvidenceRecord] = {}
        for _, candidate in _world_rows(self._world, "evidence_records"):
            if (
                not isinstance(candidate, EvidenceRecord)
                or candidate.subject_id != normalized_order_id
            ):
                continue
            current = self._world.read_evidence_record(
                candidate.record_id,
                caller=self._caller,
            )
            if (
                isinstance(current, EvidenceRecord)
                and current.subject_id == normalized_order_id
            ):
                evidence[current.record_id] = current

        return issue_after_sales_authority_projection(
            context,
            original_actor=self._caller,
            replacement_options=tuple(replacements[key] for key in sorted(replacements)),
            evidence_records=tuple(evidence[key] for key in sorted(evidence)),
        )

    # --- Reputation --------------------------------------------------

    def get_merchant_reputation(self, merchant_id: "AgentId") -> "ReputationScore":
        """Return the merchant's current reputation. Public, all callers."""
        row = self._world.read("reputation", merchant_id, caller=self._caller)
        if row is not None:
            return cast("ReputationScore", row)
        return ReputationScore(
            merchant_id=merchant_id,
            rolling_avg=0.0,
            n_settled=0,
            n_disputed=0,
        )

    # --- Social commerce (item 7) ------------------------------------

    def get_friends(self) -> "tuple[str, ...]":
        """Return the calling buyer's friend ids (empty if none).

        Caller-scoped: a buyer can only read their own friend list.
        """
        row = self._world.read("friendships", self._caller, caller=self._caller)
        if row is None:
            return ()
        return tuple(str(f) for f in row.friends)

    def get_friend_reviews(
        self, sku_id: "SkuId | None" = None, merchant_id: "AgentId | None" = None
    ) -> "list[Review]":
        """Return reviews authored by the caller's friends.

        The social trust gate: only friends' reviews are returned, optionally
        filtered to one ``sku_id`` / ``merchant_id``. Returns ``[]`` when the
        caller has no friends or the world has no reviews table (e.g. a
        database-backed world without the social tables).
        """
        friends = frozenset(self.get_friends())
        if not friends:
            return []
        tables = getattr(self._world, "_tables", None)
        table = tables.get("reviews") if isinstance(tables, dict) else None
        if table is None or not hasattr(table, "by_friends"):
            return []
        return cast(
            "list[Review]",
            table.by_friends(
                friends,
                sku_id=None if sku_id is None else str(sku_id),
                merchant_id=None if merchant_id is None else str(merchant_id),
            ),
        )

    def get_review_evidence(
        self, sku_id: "SkuId | None" = None, merchant_id: "AgentId | None" = None
    ) -> "dict[str, object]":
        """Return caller-scoped reviews plus stable evidence identifiers.

        This is the structured provenance form used by deterministic benchmark
        lanes; it applies exactly the same friendship/row filtering as
        :meth:`get_friend_reviews` and adds no broader visibility.
        """
        reviews = self.get_friend_reviews(sku_id=sku_id, merchant_id=merchant_id)
        return {
            "sku_id": "" if sku_id is None else str(sku_id),
            "review_ids": [str(review.review_id) for review in reviews],
            "reviews": reviews,
        }

    # --- Durable evidence, mandates, and listing claims -------------

    def get_evidence_record(
        self,
        record_id: str,
        version: int | None = None,
        record_digest: str | None = None,
    ) -> "EvidenceRecord | None":
        """Read one durable evidence record through its persisted ACL.

        The caller identity is bound when this facade is constructed.  Agent
        code cannot substitute another reader.  ``version`` and
        ``record_digest`` are mutually exclusive selectors for historical
        versions; omitting both returns the current version.
        """

        return self._world.read_evidence_record(
            record_id,
            caller=self._caller,
            version=version,
            record_digest=record_digest,
        )

    def get_mandate_revisions(
        self, mandate_id: str
    ) -> "tuple[MandateRevision, ...]":
        """Return the caller-authorized, append-only mandate history."""

        return self._world.mandate_revisions(mandate_id, caller=self._caller)

    def get_listing_claim(self, claim_id: str) -> "ListingClaim | None":
        """Return an owner-visible draft or a claim visible to the public."""

        return self._world.read_listing_claim(claim_id, caller=self._caller)

    def list_listing_claims(self, listing_id: str) -> "tuple[ListingClaim, ...]":
        """Return published claims for one listing in deterministic order."""

        return self._world.listing_claims_for_listing(
            listing_id, caller=self._caller
        )

    # --- Disputes ----------------------------------------------------

    def get_dispute(self, dispute_id: "DisputeId") -> "Dispute | None":
        """Return the dispute if the caller is a party, else ``None``."""
        return self._world.read("disputes", dispute_id, caller=self._caller)

    def list_disputes_for_caller(self) -> "list[Dispute]":
        """Return all disputes the caller is party to."""
        rows = (
            self._world.iter_table("disputes")
            if hasattr(self._world, "iter_table")
            else self._world._tables["disputes"].all()
        )
        return [
            dispute
            for _, dispute in rows
            if self._caller in (str(dispute.filed_by), str(dispute.against))
        ]

    # --- Rulings -----------------------------------------------------

    def get_ruling(self, dispute_id: "DisputeId") -> "Ruling | None":
        """Return the ruling for the dispute, or ``None``."""
        rows = (
            self._world.iter_table("rulings")
            if hasattr(self._world, "iter_table")
            else self._world._tables["rulings"].all()
        )
        for _, ruling in rows:
            if ruling.dispute_id == dispute_id:
                dispute = self.get_dispute(ruling.dispute_id)
                is_platform = (
                    self._caller == "platform"
                    or self._caller.startswith("platform:")
                )
                return ruling if dispute is not None or is_platform else None
        return None


def _world_rows(world: object, table: str) -> tuple[tuple[object, object], ...]:
    """Read a World table through the common in-memory/SQLite iterator seam."""

    iterator = getattr(world, "iter_table", None)
    if callable(iterator):
        return tuple(iterator(table))
    tables = getattr(world, "_tables", None)
    selected = tables.get(table) if isinstance(tables, dict) else None
    all_rows = getattr(selected, "all", None)
    if not callable(all_rows):
        raise RuntimeError(f"World cannot enumerate {table!r} authority rows")
    return tuple(all_rows())
