"""Merchant-side private state — the half of the legacy ``Listing`` that
is NOT shared with world.

Resolves the "two ``inventory`` numbers" authority problem documented in
the ACWorld merchant-data authority design by establishing a single source of
truth for every datum the merchant reads:

* ``list_price`` / ``inventory`` / public ``attributes`` → world (read via
  ``world.read_catalog`` / ``world.read_inventory``).
* ``floor_price`` (PRIVATE_UTILITY), ``arrived_at``, ``inventory_age_days``,
  ``velocity_per_day``, ``private_attrs`` → here.

The public/private ``attributes`` split is allowlist-based:
:data:`PUBLIC_ATTRIBUTE_KEYS` enumerates the keys allowed on the wire.
Anything else (e.g. ``original_price_minor`` recorded by the CSV loader)
defaults to private — same default-deny posture as Phase 1's
``merchant_id_in_payload_forbidden`` decline.

Migration note: this dataclass coexists with the legacy
``agents.merchant_tools.Listing`` for one commit to keep the diff
reviewable. The ``Listing`` class is marked deprecated and removed in
the third commit of the B5 implementation plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


#: Allowlist of ``attributes`` keys allowed in the wire payload sent to
#: ``world.catalog``. Anything not in this set is **private** and stays in
#: :attr:`MerchantPrivateState.private_attrs`.
#:
#: All entries are formally public. ``certifications`` and ``tags`` /
#: ``flat_tags`` are read by the aggregator's claim-extraction path
#: (``_extract_claims`` in ``agents/platform.py``); ``options`` is the
#: variant-axis surface and carries no private bookkeeping in any
#: current loader. Adding a new key as public is an explicit allowlist
#: add — default-deny for everything else.
PUBLIC_ATTRIBUTE_KEYS: frozenset[str] = frozenset({
    "material",
    "variant",
    "key_features",
    "category",
    "display_name",
    "product_url",
    "certifications",
    "tags",
    "flat_tags",
    "options",
    # Data-lineage fields attached by the real-CSV catalog adapter.  They are
    # public source identifiers, not merchant utility or operational state.
    "source_label",
    "source_file",
    "source_merchant",
    "source_row_number",
    "source_sku",
    "source_product_name",
    "source_product_url",
})


def split_attributes(
    attrs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition a mixed attributes dict by :data:`PUBLIC_ATTRIBUTE_KEYS`.

    Returns ``(public, private)``. The split is default-deny: any key not
    in the allowlist lands in ``private``. Used by the CSV loader and the
    listing-publish skill so the same attributes payload doesn't have to
    be re-classified at each call site.
    """
    public: dict[str, Any] = {}
    private: dict[str, Any] = {}
    for k, v in attrs.items():
        if k in PUBLIC_ATTRIBUTE_KEYS:
            public[k] = v
        else:
            private[k] = v
    return public, private


@dataclass(frozen=True)
class MerchantPrivateState:
    """Merchant-side state for one SKU. Authoritative for floor / velocity /
    age / private bookkeeping. NOT authoritative for inventory or list_price
    (those live in world).

    Frozen so updates use ``dataclasses.replace()`` — see the
    [[velocity-update]] open question in the RFD: today's
    ``MerchantData.private_states`` is a mutable dict of frozen entries,
    so a velocity refresh is a one-line replace + dict assignment.

    Fields:
        product: merchant-side category label (key into ``MerchantData``
            collections). Distinct from the wire ``sku_id``.
        floor_price: minor units. PRIVATE_UTILITY — never on the wire,
            never a constructor parameter to any envelope helper.
        arrived_at: ISO date of the most recent restock, if known.
            Merchant's own bookkeeping; world tracks no restock timing.
        inventory_age_days: days since ``arrived_at`` (or seeded value).
            Used by ``aging-markdown`` / ``demand-driven-markup``.
        velocity_per_day: recent dispatch rate. Enables
            ``days_to_stockout = qty_available / velocity_per_day`` in
            :func:`agents.merchant_tools.get_inventory_status`.
        private_attrs: merchant-only attribute bookkeeping (e.g.
            ``original_price_minor`` for promo anchor tracking). Never
            shipped to world. The public counterpart lives in
            ``world.catalog.Listing.attributes`` and is set at publish
            time only.
        seed_list_price: spawn-time snapshot of the public list price
            (from CSV / seed config). **NOT authoritative** — world is.
            Lives here so tool-side sanity checks (``propose_markdown``'s
            "this is a markdown, not a markup" gate; peer-pricing's
            anchor) don't need a world round-trip on every call. A
            follow-up pass should refresh this on ``platform.catalog_ack``
            via the same pattern :class:`InventoryView` uses for
            inventory; today it is a static seed.
        seed_inventory: same shape as ``seed_list_price`` but for the
            spawn-time qty_available snapshot. Used by
            ``get_inventory_status`` only when ``MerchantData.inventory_view``
            is unset (e.g. tests without a world). When the view IS set,
            world is read through and ``seed_inventory`` is ignored.
    """
    product: str
    floor_price: int
    arrived_at: str | None = None
    inventory_age_days: int = 0
    velocity_per_day: float | None = None
    private_attrs: dict[str, Any] = field(default_factory=dict)
    seed_list_price: int = 0
    seed_inventory: int = 0


__all__ = [
    "MerchantPrivateState",
    "PUBLIC_ATTRIBUTE_KEYS",
    "split_attributes",
]
