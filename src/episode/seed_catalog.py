"""Item 5 — seed the World catalog from the real brand CSVs.

The research corpus needs a catalog with realistic, *groundable* product
attributes (``material: "wool, merino"``, ``noise_cancellation``, key-feature
prose) rather than the hand-written minimal listings in the spine scenarios.
This module materializes the public half of the brand CSVs (under
``data/raw_data/``) into World ``Listing`` + ``InventoryRow`` rows.

Single source of truth: the public listing view comes from
:func:`agents.merchant_data_csv.iter_public_listings`, which shares
``_row_listing_fields`` with the merchant-side ``load_from_csv``. So the
buyer-facing world catalog and the merchant's private bookkeeping never drift on
list_price / inventory / which rows are listings (verified in the tests). The
public/private attribute split is the allowlist in
:data:`agents.merchant_private_state.PUBLIC_ATTRIBUTE_KEYS` (default-deny).

Scale control — ``per_category_cap``: the brand CSVs run from dozens to
thousands of rows; an uncapped seed would make a single category (e.g.
Brooklinen sheets) dominate the world. The cap bounds the number of listings
per category, taken in deterministic CSV order, so a seeded world is small,
balanced, and reproducible. ``None`` means no cap (seed everything).

SKU keying: every World listing uses ``merchant:<brand>:sku:<source-sku>``.
The raw source SKU and exact CSV row remain in public provenance attributes.
Repeated source-SKU rows inside one CSV are canonicalized by the shared loader's
first-occurrence rule, so merchant-private and World-public views cannot choose
different variants. ``merchant_id`` is ``merchant:<brand>``; ``product_id`` is
derived from the source product URL (or product-name fallback) and groups variants
without inventing cross-brand equivalences.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import Any

from agents.merchant_data_csv import (
    KNOWN_MERCHANTS,
    catalog_source_statistics,
    iter_public_listings,
)
from episode.benchmark import CATALOG_SCALE_TARGETS, CatalogScale


def _scale(value: "str | CatalogScale | None") -> "CatalogScale | None":
    if value is None or isinstance(value, CatalogScale):
        return value
    try:
        return CatalogScale(str(value))
    except ValueError as exc:
        choices = ", ".join(s.value for s in CatalogScale)
        raise ValueError(f"unknown catalog_scale {value!r}; expected one of: {choices}") from exc


def _sample_key(seed: int, merchant: str, row: dict[str, Any]) -> bytes:
    """Stable cross-process ordering for bounded catalog profiles."""

    token = f"{seed}:{merchant}:{row['listing_sku_id']}".encode("utf-8")
    return hashlib.sha256(token).digest()


def seed_world_catalog(
    world: Any,
    *,
    merchants: "tuple[str, ...]" = KNOWN_MERCHANTS,
    data_root: str = "data/raw_data",
    seed: int = 42,
    per_category_cap: int | None = None,
    in_stock_only: bool = False,
    catalog_scale: "str | CatalogScale | None" = None,
) -> dict[str, Any]:
    """Seed ``world`` with the public catalog of the named brand CSVs.

    Args:
        world: a ``World`` (anything with ``.apply({"catalog": [...],
            "inventory": {...}})``).
        merchants: brand slugs to seed (default: all shipped brands).
        data_root: directory holding ``<brand>.csv``.
        seed: stock-assignment seed — same seed + same CSV ⇒ same world.
        per_category_cap: max listings per category (None = unbounded). Taken in
            deterministic CSV order across the merchant list.
        in_stock_only: if True, skip rows whose seeded inventory is 0 (so the
            buyer never sees an out-of-stock listing in discovery).
        catalog_scale: optional paper-facing scale profile. ``smoke`` seeds 5
            listings, ``standard`` 50, ``medium`` 250, and ``full`` everything.
            The bounded profiles use a SHA-256 order keyed by ``seed`` so the
            sample is reproducible across Python processes and machines.

    Returns:
        A summary dict: ``{listings, by_category, merchants, skipped_duplicate_sku,
        skipped_over_cap, skipped_out_of_stock}``.
    """
    from world.types import AgentId, InventoryRow, Listing, Money, SkuId

    scale = _scale(catalog_scale)
    target_listings = CATALOG_SCALE_TARGETS[scale] if scale is not None else None
    source_statistics = [
        catalog_source_statistics(merchant, data_root=data_root)
        for merchant in merchants
    ]

    candidates: list[tuple[str, Any, dict[str, Any]]] = []
    skipped_oos = 0
    for merchant in merchants:
        mid = AgentId(f"merchant:{merchant}")
        for row in iter_public_listings(merchant, data_root=data_root, seed=seed):
            if in_stock_only and int(row["inventory"]) <= 0:
                skipped_oos += 1
                continue
            candidates.append((merchant, mid, row))

    # Preserve the historical CSV order when no bounded scale is requested.
    # A scale profile intentionally samples across all six brands instead of
    # taking the first N rows (which would make a small catalog all-Allbirds).
    if target_listings is not None:
        candidates.sort(key=lambda item: _sample_key(seed, item[0], item[2]))

    catalog: list[Any] = []
    inventory: dict[Any, Any] = {}
    seen_sku: set[str] = set()
    per_cat: dict[str, int] = {}
    skipped_dup = skipped_cap = skipped_scale = 0

    for _merchant, mid, row in candidates:
        sku = row["listing_sku_id"]
        qty = int(row["inventory"])
        if sku in seen_sku:           # cross-merchant collision — first wins
            skipped_dup += 1
            continue
        cat = row["category"]
        if per_category_cap is not None and per_cat.get(cat, 0) >= per_category_cap:
            skipped_cap += 1
            continue
        if target_listings is not None and len(catalog) >= target_listings:
            skipped_scale += 1
            continue
        seen_sku.add(sku)
        per_cat[cat] = per_cat.get(cat, 0) + 1
        catalog.append(Listing(
            sku_id=SkuId(sku),
            category=cat,
            name=row["name"],
            attributes=dict(row["attributes"]),
            # CSV list_price is integer minor units (cents); Money.amount is
            # dollars (Decimal), matching the spine scenarios + the scorer.
            list_price=Money(amount=Decimal(int(row["list_price"])) / Decimal(100)),
            merchant_id=mid,
            product_id=str(row["product_id"]),
        ))
        inventory[SkuId(sku)] = InventoryRow(
            sku_id=SkuId(sku), merchant_id=mid, qty_available=qty)

    world.apply({"catalog": catalog, "inventory": inventory})
    return {
        "listings": len(catalog),
        "by_category": dict(per_cat),
        "merchants": list(merchants),
        "skipped_duplicate_sku": skipped_dup,
        "dropped_duplicate_source_sku_rows": sum(
            int(item["duplicate_source_sku_rows"]) for item in source_statistics
        ),
        "skipped_over_cap": skipped_cap,
        "skipped_over_scale": skipped_scale,
        "skipped_out_of_stock": skipped_oos,
        "catalog_scale": scale.value if scale is not None else None,
        "target_listings": target_listings,
        "data_label": "real_csv",
        "source_sellable_rows": sum(
            int(item["sellable_row_count"]) for item in source_statistics
        ),
        "source_canonical_listings": sum(
            int(item["canonical_listing_count"]) for item in source_statistics
        ),
        "source_provenance": source_statistics,
    }
