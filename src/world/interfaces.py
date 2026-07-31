"""Cross-package contracts the world layer exposes.

The world-read interface is wire-stable. Other packages depend on these
Protocols (not on ``World`` or ``WorldTools`` directly) so a future
cross-process world drops in without lane changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from world.types import (
        AgentId,
        Dispute,
        DisputeId,
        Listing,
        Money,
        Order,
        OrderId,
        ReputationScore,
        Ruling,
        RulingId,
        SkuId,
        WorldSnapshot,
    )


# --- Core reader surface ----------------------------------------------

class CatalogReader(Protocol):
    """Read-only access to the catalog table."""

    def get_listing(self, sku_id: "SkuId") -> "Listing | None":
        """Return the listing or ``None``."""
        ...

    def search(self, query: str, *, limit: int = 10) -> "list[Listing]":
        """Return up to ``limit`` listings matching ``query``. Order is deterministic."""
        ...


class InventoryReader(Protocol):
    """Read-only access to the inventory table.

    Public surface only — exact stock counts are not exposed to agents.
    """

    def is_in_stock(self, sku_id: "SkuId", *, qty: int = 1) -> bool:
        """Return ``True`` iff at least ``qty`` units are available."""
        ...


class LedgerReader(Protocol):
    """Read-only access to the ledger and orders, caller-scoped."""

    def get_balance(self) -> "Money":
        """Return the calling agent's current ledger balance."""
        ...

    def get_order(self, order_id: "OrderId") -> "Order | None":
        """Return the order if the caller is buyer or merchant on it, else ``None``."""
        ...


# --- Reputation and dispute reader surface ----------------------------

class ReputationReader(Protocol):
    """Read-only access to the public reputation table."""

    def get_merchant_reputation(self, merchant_id: "AgentId") -> "ReputationScore":
        """Return the merchant's current reputation."""
        ...


class DisputeReader(Protocol):
    """Read-only access to the disputes table, caller-scoped to parties."""

    def get_dispute(self, dispute_id: "DisputeId") -> "Dispute | None":
        """Return the dispute if the caller is a party to it, else ``None``."""
        ...

    def list_disputes_for_caller(self) -> "list[Dispute]":
        """Return all disputes the caller is party to."""
        ...


# --- Ruling reader surface --------------------------------------------

class RulingReader(Protocol):
    """Read-only access to the rulings table."""

    def get_ruling(self, dispute_id: "DisputeId") -> "Ruling | None":
        """Return the ruling for the dispute, or ``None`` if not yet ruled."""
        ...

    def get_ruling_by_id(self, ruling_id: "RulingId") -> "Ruling | None":
        """Return the ruling by its own id, or ``None``."""
        ...


# --- Snapshot surface --------------------------------------------------

class SnapshotReader(Protocol):
    """Frozen-view read used by the replay verifier."""

    def snapshot(self) -> "WorldSnapshot":
        """Return a deep, frozen view of every world table."""
        ...
