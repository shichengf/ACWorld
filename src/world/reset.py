"""Reset strategies for the world.

E0: clear state between every episode (no cross-episode persistence).
E1: clear only between batches; persistence within a batch (lets reputation,
disputes, and rulings carry across episodes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from world.state import World


class ResetStrategy(Protocol):
    """Strategy interface used by ``World.reset``."""

    mode: str

    def apply(self, world: "World") -> None:
        """Reset the world according to the strategy."""
        ...


class E0Reset:
    """Clear catalog/inventory/ledger between every episode."""

    mode: str = "E0"

    def apply(self, world: "World") -> None:
        """Clear all per-episode tables; preserve only the schema, not the rows."""
        for table in world._tables.values():
            table.clear()


class E1Reset:
    """Clear episode-local supply data while preserving durable market history.

    Orders and ledger entries are part of the historical transaction record,
    not transient catalog state.  E1 batches therefore retain them alongside
    reputation and adjudication state. Catalog and current inventory are
    re-seeded for each episode; their mutations remain observable in the audit
    and transaction-diff artifacts.
    """

    mode: str = "E1"

    PERSISTENT_TABLES: frozenset[str] = frozenset({
        "orders",
        "ledger",
        "reputation",
        "reputation_settlements",
        "disputes",
        "rulings",
        "fulfillments",
        "exchanges",
        "order_timelines",
        "order_groups",
        "shipments",
        "search_sessions",
        "match_acceptances",
        "match_certificates",
        "supply_purchase_authorities",
        "protocol_events",
        "protocol_receipts",
        "negotiation_events",
        "negotiation_threads",
        "pricing_policy_revisions",
        "persistent_cart_quote_requests",
        "persistent_cart_quotes",
        "evidence_records",
        "mandate_authorities",
        "mandate_revisions",
        "listing_claims",
        "authority_operations",
        "payment_states",
        "packing_records",
        "after_sales_policies",
        "after_sales_records",
        "governance_policies",
        "governance_records",
        "reviews",
    })

    def apply(self, world: "World") -> None:
        """Clear non-persistent tables only; persistent ones survive within the batch."""
        for name, table in world._tables.items():
            if name not in self.PERSISTENT_TABLES:
                table.clear()
