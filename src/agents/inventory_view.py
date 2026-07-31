"""``InventoryView`` — the merchant lane's read-only window onto world inventory.

Resolves Q2 + Q3 of the ACWorld merchant-data authority design:

* **Surface**: pass-through reader exposing ``qty_available(sku_id)`` only.
  ``get_inventory_status`` composes the derived ``InventoryStatus`` shape
  (count + age + velocity + days_to_stockout) in the agent layer from
  this view plus :class:`agents.merchant_private_state.MerchantPrivateState`.
  No ``qty_reserved`` accessor — no current skill reads it; trivial to
  add when a future skill does.

* **Cache shape**: ``use_cache=False`` (default) is a pure pass-through —
  every ``qty_available`` call reads through to world. With
  ``use_cache=True`` the view caches per-sku and invalidates the whole
  cache on inbound envelopes whose kind is in
  :attr:`InventoryView._cache_invalidated_by` (today:
  ``platform.settlement_receipt`` from the PSP and
  ``platform.catalog_ack`` from the CatalogPolicy — both signal that
  world inventory may have changed for any sku the merchant cares about).

* **Ownership**: the owning :class:`agents.base.Agent` calls
  :meth:`InventoryView.on_envelope` as the first step of every
  ``receive()`` turn. Runtime stays unaware of the view — the agent
  owns its own cache. Buyer / consumer agents don't construct a view
  and pass ``None`` through, making the hook a no-op for them.

* **Pass-through cost**: a merchant turn that reads inventory once
  spends one world call. The cache is the optimization for skills that
  read it multiple times per turn (e.g. aging + restock + scarcity
  checks all on the same SKU); turn-fresh by construction since
  envelope dispatch happens after invalidation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from protocol.envelope import Envelope


#: Default inbound-envelope kinds that clear the inventory cache. The PSP's
#: settlement receipt means qty_available decremented; the platform's
#: catalog ack means it may have moved (receive_shipment increments,
#: update_listing may shift state).
_DEFAULT_INVALIDATING_KINDS: frozenset[str] = frozenset({
    "platform.settlement_receipt",
    "platform.catalog_ack",
})


@dataclass(frozen=True)
class _Entry:
    """One cached row. Stored as a value snapshot, not a live world reference."""
    qty_available: int

    @classmethod
    def from_row(cls, row: Any) -> "_Entry":
        # InventoryRow has qty_available; bare int rows (legacy) get coerced.
        qty = getattr(row, "qty_available", row)
        return cls(qty_available=int(qty))


class InventoryView:
    """Merchant-private read-only view of world inventory.

    Constructed by user code (e.g. a notebook driver or a real merchant
    factory) and passed into :class:`agents.base.Agent`. The agent calls
    :meth:`on_envelope` at the head of every receive turn; skills call
    :meth:`qty_available` from the tool layer.

    Args:
        world_reader: callable ``(sku_id: str) -> InventoryRow | None``.
            Wraps either a direct ``World`` reference (in-process) or an
            HTTP client (out-of-process). The view never holds a
            world handle itself.
        use_cache: enable per-sku caching with envelope-driven
            invalidation. Default off — single-process, single-merchant
            scope today doesn't need it; the flag is the migration
            point to a cached subclass when scale demands it (see RFD §5
            "Migration trigger to Option A").
        invalidating_kinds: override the envelope kinds that clear the
            cache. Tests can pass an empty set to assert the
            invalidation is what clears the cache, not test luck.
    """

    def __init__(
        self,
        world_reader: Callable[[str], Any | None],
        *,
        use_cache: bool = False,
        invalidating_kinds: frozenset[str] | None = None,
    ) -> None:
        self._reader = world_reader
        self._cache: dict[str, _Entry] | None = {} if use_cache else None
        self._invalidating: frozenset[str] = (
            invalidating_kinds
            if invalidating_kinds is not None
            else _DEFAULT_INVALIDATING_KINDS
        )

    def qty_available(self, sku_id: str) -> int:
        """Return current available qty for the SKU, or 0 if no row exists.

        Zero is the safe answer for "no row" because every skill that reads
        this is gating on threshold comparisons (``count <= stockout_threshold``,
        ``days_to_stockout``); treating "missing" as "0" is consistent with
        treating "missing" as "out of stock".
        """
        cache = self._cache
        if cache is not None:
            cached = cache.get(sku_id)
            if cached is not None:
                return cached.qty_available
        row = self._reader(sku_id)
        qty = 0 if row is None else int(getattr(row, "qty_available", row))
        if cache is not None:
            cache[sku_id] = _Entry(qty_available=qty)
        return qty

    def on_envelope(self, env: "Envelope") -> None:
        """Invalidate the cache if this envelope kind signals a world change.

        Called by the owning :class:`Agent` as the first line of
        :meth:`Agent.receive`. No-op when caching is disabled or the
        envelope kind is not invalidating.

        Whole-cache invalidation (not per-sku) is intentional for today's
        scope — the trigger conditions for per-sku invalidation (Option A
        in the RFD) haven't materialized yet, and over-invalidating is
        safe; under-invalidating is the bug class we're guarding against
        (the original "two inventory numbers" drift).
        """
        if self._cache is None:
            return
        if str(env.action.get("kind", "")) in self._invalidating:
            self._cache.clear()


__all__ = ["InventoryView"]
