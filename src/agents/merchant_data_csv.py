"""CSV catalog loader for the merchant agent.

Reads one of the brand catalog CSVs under ``data/raw_data/<name>.csv`` and
builds a fully populated ``MerchantData`` the merchant lane's skills can
drive against. Six brand catalogs ship today:

    allbirds, brooklinen, death_wish_coffee, kylie_cosmetics, manduka, meowant

The CSVs carry only public catalog facts, as specified in
``WORLD_DATABASE_DESIGN.md`` §3. The merchant's private negotiating
utility fields, including ``floor_price``, ``target_band``,
``auto_accept_threshold``, per SKU velocity, and aging, are derived, not read
from the data. That keeps the public and private split clean: same CSV, same
public listings, different mandate policies per scenario.

Stock is not in the CSV (only ``In Stock`` / ``Out of Stock``). For
``In Stock`` rows we assign a deterministic seeded random count in
``mandate_policy.in_stock_qty_range`` (default 20-200). Out of stock
rows get 0. The seed is per ``(merchant, sku)`` so the same CSV always
materializes the same world.

Two CSV schemas:

* 25 columns (allbirds, brooklinen, death_wish_coffee, kylie_cosmetics,
  manduka): the original catalog layout.
* 27 columns (meowant): adds a leading ``No.`` index and a
  ``Capacity/Coverage`` column. Prices are stored as ``$23.99`` strings with a
  UTF-8 BOM on the file. ``csv.DictReader`` + ``utf-8-sig`` handle both.

Usage::

    from agents.merchant_data_csv import load_from_csv
    data = load_from_csv("meowant")
    print(len(data.private_states), "SKUs")
    print(data.private_states["PPB-YB04L25W00004"])

The returned ``MerchantData`` is the exact same shape ``merchant_tools``
already consumes. Every existing skill (pricing-negotiate, aging-markdown,
inquiry-handle, return-adjudicate, dispute-defense, etc.) works against
it unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agents.merchant_private_state import split_attributes
from agents.merchant_tools import MerchantData

# The six brand catalogs that ship in data/raw_data/. The loader is permissive:
# any name with a corresponding ``<name>.csv`` works, but these are the known
# ones.
KNOWN_MERCHANTS: tuple[str, ...] = (
    "allbirds",
    "brooklinen",
    "death_wish_coffee",
    "kylie_cosmetics",
    "manduka",
    "meowant",
)

# Files with a UTF-8 BOM at the start. ``open(..., encoding='utf-8-sig')``
# strips it; the others use plain utf-8.
_BOM_FILES: frozenset[str] = frozenset({"meowant"})


def namespaced_sku_id(merchant_name: str, source_sku: str) -> str:
    """Return the globally unique listing key used by the World catalog.

    Merchant-side data remains keyed by the source SKU by default for backward
    compatibility.  The World cannot make that assumption: two independent
    merchants are allowed to use the same source SKU, so the listing key carries
    the owning actor namespace.
    """

    merchant = str(merchant_name).strip().lower()
    sku = str(source_sku).strip()
    if not merchant or not sku:
        raise ValueError("merchant_name and source_sku must be non-empty")
    return f"merchant:{merchant}:sku:{sku}"


def source_product_id(
    merchant_name: str,
    *,
    product_url: str | None,
    product_name: str,
) -> str:
    """Derive a stable product-level identity without guessing cross-brand links.

    Variants sharing a source product URL receive the same ``product_id``.  A
    normalized product name is the deterministic fallback when a URL is absent.
    The merchant namespace is intentional: the shipped CSVs contain no verified
    cross-merchant product crosswalk, so equating similar names across brands
    would fabricate provenance.
    """

    merchant = str(merchant_name).strip().lower()
    url = str(product_url or "").strip().rstrip("/")
    fallback_name = " ".join(str(product_name).strip().lower().split())
    source_key = url or fallback_name
    if not merchant or not source_key:
        raise ValueError("merchant_name and product URL/name must be non-empty")
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:20]
    return f"merchant:{merchant}:product:{digest}"


# --- Default mandate policy ---------------------------------------------
#
# These knobs are merchant-side private state — never read from the CSV.
# The defaults are sensible for a research run; scenarios may override
# any of them via the ``mandate_policy`` arg.
#
# The negotiation surface is laid out so:
#
#     floor_price < target_band[0] < auto_accept_threshold <= target_band[1] = list_price
#
# i.e. the public ``list_price`` is the merchant's ceiling (they will not
# counter above what they themselves advertised); ``auto_accept_threshold``
# sits inside the band ("I'd take this without negotiating further");
# ``target_band[0]`` sits below ``auto_accept_threshold`` so
# ``reputation-aware-pricing`` / ``stockout-aware-pricing`` have room to
# shift counters within the legal zone; ``floor_price`` is private and
# strictly below the band.
DEFAULT_MANDATE_POLICY: dict[str, Any] = {
    "floor_pct_of_list":     0.60,   # private floor:        60% of list_price
    "target_band_low_pct":   0.80,   # public band low:      80% of list_price
    "auto_accept_pct":       0.95,   # auto-accept at:       95% of list_price
    # target_band[1] is always list_price (the merchant's ceiling).
    "must_not_claim": ("medical-grade", "FDA-approved", "cures-disease",
                       "miracle", "guaranteed weight-loss"),
    "in_stock_qty_range": (20, 200),
    "velocity_per_day_default": 0.5,
    "inventory_age_days_default": 7,
}


# Default stated refund + stock-management policy for the merchant. Mirrors
# the synthetic policy used in the demo notebook, calibrated to the new
# 30-day window (real merchants typically run 30-day returns, not 7).
DEFAULT_POLICY: dict[str, Any] = {
    "refund_policy": "30_day_return",
    "return_window_days": 30,
    "claim_aggressiveness": "neutral",
    "markdown_rules": {"after_days": 30, "step_interval_days": 14,
                        "step_pct": 5, "max_pct": 30},
    "scarcity_window_days": 7,
    "demand_floor_for_scarcity": 60,
    "stockout_threshold": 5,
    "markup_rules": {"step_pct": 5, "max_pct": 15, "cooldown_days": 7},
    "markup_demand_floor": 80,
    "markup_velocity_floor": 0.3,
    # Anti-endless-negotiation cap — merchant unilaterally rejects after
    # this many merchant counter_offers against the same (buyer, offer_id)
    # thread (reason="max_rounds_exceeded").
    "max_negotiation_rounds": 5,
    # Patience model — the merchant starts each (buyer, offer_id) thread with
    # initial_patience units; each round drains an amount scaled by how far
    # the buyer's offer sits below auto_accept_threshold. <=0 -> reject with
    # reason="patience_exhausted". Tunes the negotiation termination to be
    # OFFER-QUALITY-AWARE rather than purely round-count-based.
    "initial_patience": 100,
    "patience_decay_multiplier": 100,
    # Public minimum offer step (alongside target_band). The buyer must raise
    # by at least this fraction of list_price between consecutive offers in
    # the same (buyer, offer_id) thread. Default 2% — keeps haggling
    # bounded and prevents 1-cent tactical creep. 0.0 disables the rule.
    "min_offer_step_pct": 0.02,
}


# --- Money parsing ------------------------------------------------------

_PRICE_RE = re.compile(r"^\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*$")


def parse_money(s: Any) -> int | None:
    """Parse a CSV money cell to integer minor units (cents).

    Handles ``$23.99`` (meowant) and bare ``24.00`` (the other five) and
    blank / None. Returns ``None`` for blank or non-parseable; raises
    nothing — callers decide whether to skip.

    Per ``WORLD_DATABASE_DESIGN.md`` §4, money is integer minor units.
    ``$23.99`` -> ``2399``, ``24.00`` -> ``2400``, ``''`` -> ``None``.
    """
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    m = _PRICE_RE.match(t)
    if not m:
        return None
    return int(round(float(m.group(1).replace(",", "")) * 100))


# --- Tag parsing --------------------------------------------------------

_KV_RE = re.compile(r"^(?P<k>[^=]+?)\s*=>\s*(?P<v>.+)$")


def parse_tags(s: Any) -> dict[str, Any]:
    """Parse the Tags column into a structured dict.

    Allbirds uses both ``key => value`` pairs (``allbirds::carbon-score => 5.9``)
    and flat tags (``shoprunner``, ``YCRF_socks``) in the same cell.
    Brooklinen / Manduka / Kylie use flat comma-separated lists.
    The function handles both shapes per-token.

    Returns::

        {
            "<key>": "<value>",      # one per `key => value` token
            "flat_tags": [...],       # the bare tokens
        }
    """
    if not s:
        return {}
    out: dict[str, Any] = {}
    flat: list[str] = []
    for part in str(s).split(","):
        p = part.strip()
        if not p:
            continue
        m = _KV_RE.match(p)
        if m:
            out[m.group("k").strip()] = m.group("v").strip()
        else:
            flat.append(p)
    if flat:
        out["flat_tags"] = flat
    return out


# --- Attribute extraction -----------------------------------------------

# Columns the loader treats as product-attribute fields. The 27-col meowant
# variant additionally has "Capacity/Coverage"; csv.DictReader picks it up
# automatically if present.
_ATTR_SOURCE_COLUMNS: tuple[str, ...] = (
    "Item Weight",
    "Entrance Height",
    "Cat/Pet Weight Range",
    "Litter Bin Capacity",
    "Compatible Litter Type",
    "Dimensions (LxWxH)",
    "Material",
    "Safety Sensors",
    "App Control",
    "Certifications",
    "Warranty",
    "Capacity/Coverage",   # meowant-only — ignored on the other files
)


def _norm_attr_key(col: str) -> str:
    """Column header -> snake_case attribute key.

    ``'Item Weight'`` -> ``'item_weight'``;
    ``'Dimensions (LxWxH)'`` -> ``'dimensions_lxwxh'``;
    ``'Capacity/Coverage'`` -> ``'capacity_coverage'``.
    """
    s = col.lower()
    for ch in "()":
        s = s.replace(ch, "")
    s = s.replace("/", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def parse_attributes(row: dict[str, Any]) -> dict[str, Any]:
    """Bundle non-empty spec columns + parsed tags into a normalized dict.

    Empty / missing columns are skipped — the merchant's attribute surface
    is whatever the CSV actually carries, not a fixed schema. This keeps
    the dict compact for downstream skills (claim-truthfulness reads it).
    """
    attrs: dict[str, Any] = {}
    for col in _ATTR_SOURCE_COLUMNS:
        v = row.get(col)
        if v is None:
            continue
        v = str(v).strip()
        if not v:
            continue
        attrs[_norm_attr_key(col)] = v
    tags = parse_tags(row.get("Tags"))
    if tags:
        attrs["tags"] = tags
    return attrs


# --- Claim derivation ---------------------------------------------------

# Tag prefixes that look like internal flags / feed config, not buyer-visible
# claims. These are stripped from the permitted_claims set.
_INTERNAL_TAG_PREFIXES: tuple[str, ...] = (
    "algolia-", "algolia_", "feedonomics-", "feedonomics_",
    "YGroup_", "YCRF_", "YBlocklist", "YFiltered",
    "Cart Qty", "Discount Codes", "Fraud Risk", "redirect",
    "loop::", "Retail", "PDP", "shoprunner",
    "collection:",  # internal collection slugs
    "nosto", "_free_sample", "_gift_eligible",
    "Free Gift",  # internal gifting flag
)

_BANNED_CLAIM_WORDS: frozenset[str] = frozenset({
    "medical", "miracle", "cure", "cures", "treats", "diagnoses",
    "fda", "guaranteed",
})


def _safe_claim(token: str) -> bool:
    """A claim is OK to permit if it's short, non-internal, non-banned."""
    if not token or any(token.startswith(p) for p in _INTERNAL_TAG_PREFIXES):
        return False
    if len(token) < 3 or len(token) > 40:
        return False
    low = token.lower()
    if any(banned in low for banned in _BANNED_CLAIM_WORDS):
        return False
    return True


def derive_permitted_claims(row: dict[str, Any], attrs: dict[str, Any],
                            max_claims: int = 5) -> tuple[str, ...]:
    """Best-effort claim derivation from attributes + flat tags.

    Returns at most ``max_claims`` distinct public-safe claim phrases. The
    derivation is intentionally conservative — research-side overrides
    should pass an explicit ``permitted_claims`` policy when stronger
    claims are warranted.
    """
    claims: list[str] = []

    def add(c: str) -> None:
        c = c.strip()
        if c and c.lower() not in (x.lower() for x in claims):
            if _safe_claim(c):
                claims.append(c)

    # 1) Attribute-derived claims (the "factual" half — these are grounded
    # in the listing row itself, so claim-truthfulness will keep them).
    if attrs.get("app_control") in ("Yes", "yes", "true", "True", True):
        add("app control")
    if attrs.get("safety_sensors"):
        add("safety sensors")
    if attrs.get("material"):
        add(str(attrs["material"]))
    if attrs.get("warranty"):
        add("warranty included")
    if attrs.get("certifications"):
        add("certified")
    # 2) Flat-tag-derived claims. Safety check runs on the RAW tag so the
    # internal-prefix list catches feed-system tokens (``YCRF_socks``,
    # ``algolia-ignore``) before they get cosmetic-normalized.
    flat = ((attrs.get("tags") or {}).get("flat_tags")) or []
    for tag in flat:
        if len(claims) >= max_claims:
            break
        raw = tag.strip()
        if not _safe_claim(raw):
            continue
        norm = raw.replace("_", " ").strip()
        add(norm)
    return tuple(claims[:max_claims])


# --- Stock assignment ---------------------------------------------------

_IN_STOCK_VALUES: frozenset[str] = frozenset({
    "in stock", "instock", "in-stock", "available", "in_stock",
})


def assign_stock(sku_id: str, availability: Any, *,
                 seed: int = 42,
                 in_stock_range: tuple[int, int] = (20, 200)) -> int:
    """Deterministic random stock for in-stock items, 0 otherwise.

    The seed is per-``(merchant-run-seed, sku)`` so the same CSV always
    materializes the same world for the same run seed. Out of stock /
    sold out / discontinued / blank all map to 0.
    """
    avail = (str(availability) if availability is not None else "").strip().lower()
    if avail not in _IN_STOCK_VALUES:
        return 0
    rng = random.Random(f"{seed}:{sku_id}")
    lo, hi = in_stock_range
    return rng.randint(int(lo), int(hi))


# --- The main loader ----------------------------------------------------

def _row_listing_fields(raw: dict[str, Any]) -> "dict[str, Any] | None":
    """Per-row catalog fields shared by the merchant loader (private view) and
    the world catalog seeder (public view) — ONE source for *which* rows are
    listings and *how* Product Name / Variant / URL / Key Features / Type fold
    into the attributes dict, so the two views can never drift.

    Returns ``None`` for non-listing rows (blank SKU, or blank / zero /
    non-parseable Sale Price — the content / sample / free-gift rows). The
    returned ``attrs`` is the FULL (mixed public+private) attribute dict; callers
    apply :func:`split_attributes` (world) or keep the private half (merchant).
    """
    sku = (raw.get("SKU") or "").strip()
    if not sku:
        return None
    sale = parse_money(raw.get("Sale Price (USD)"))
    if sale is None or sale <= 0:
        return None
    attrs = parse_attributes(raw)
    name = (raw.get("Product Name") or "").strip()
    if name:
        attrs["display_name"] = name
    variant = (raw.get("Variant") or "").strip()
    if variant and variant.lower() not in ("default title",):
        attrs["variant"] = variant
    url = (raw.get("Product URL") or "").strip()
    if url:
        attrs["product_url"] = url
    kf = (raw.get("Key Features") or "").strip()
    if kf:
        attrs["key_features"] = kf
    ptype = (raw.get("Product Type") or "").strip() or "general"
    attrs["category"] = ptype
    return {"sku": sku, "sale": sale, "attrs": attrs, "name": name or sku, "category": ptype}


def _csv_path(merchant_name: str, data_root: Path) -> Path:
    return data_root / f"{merchant_name}.csv"


def _source_file_label(merchant_name: str, data_root: Path) -> str:
    """Portable repository label for shipped data; exact path for extensions."""

    path = _csv_path(merchant_name, data_root)
    if data_root.name == "raw_data" and data_root.parent.name == "data":
        return f"data/raw_data/{merchant_name}.csv"
    return path.as_posix()


def _open_csv(merchant_name: str, data_root: Path):
    enc = "utf-8-sig" if merchant_name in _BOM_FILES else "utf-8"
    return open(_csv_path(merchant_name, data_root), encoding=enc, newline="")


def _iter_canonical_listing_rows(
    merchant_name: str,
    data_root: Path,
) -> "Iterator[tuple[int, dict[str, Any], dict[str, Any]]]":
    """Yield the first sellable row for each source SKU.

    Two shipped sources contain repeated, non-unique values in the ``SKU``
    column.  Treating later rows as independent listings would make inventory
    and order keys collide; letting the public and private loaders choose
    different occurrences would be worse.  The canonical rule is therefore
    explicit and shared: first sellable occurrence in CSV order wins.

    ``source_row_number`` is one-based and includes the header, matching the
    logical spreadsheet row (not a physical text line, because cells may contain
    newlines) and making every materialized listing traceable to the source file.
    """

    seen_source_skus: set[str] = set()
    with _open_csv(merchant_name, data_root) as f:
        for source_row_number, raw in enumerate(csv.DictReader(f), start=2):
            fields = _row_listing_fields(raw)
            if fields is None or fields["sku"] in seen_source_skus:
                continue
            seen_source_skus.add(fields["sku"])
            yield source_row_number, raw, fields


def load_from_csv(
    merchant_name: str,
    *,
    data_root: str | Path = "data/raw_data",
    seed: int = 42,
    mandate_policy: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
    namespace_skus: bool = False,
) -> MerchantData:
    """Load one merchant's product catalog from CSV into ``MerchantData``.

    Given a ``merchant_name`` like ``"meowant"``, this reads
    ``<data_root>/<merchant_name>.csv`` and returns a ``MerchantData``
    with every sellable SKU populated:

    * ``listings[sku_id]`` — the public catalog row plus per-SKU derived
      private fields (``floor_price``, ``inventory``, ``inventory_age_days``,
      ``velocity_per_day``).
    * ``mandates[sku_id]`` — the ``OfferMandateView`` for negotiation,
      derived from ``mandate_policy`` (default 60% floor, 115% band high,
      auto-accept at full list).
    * ``policy`` — stated refund / markdown / scarcity rules.

    Skipped rows (not catalog listings):

    * Blank SKU.
    * ``Sale Price`` is blank, zero, or non-parseable (content rows).

    Stock for ``In Stock`` rows is a seeded random pick in
    ``mandate_policy['in_stock_qty_range']`` (default 20-200); out-of-stock
    rows get 0.

    Args:
        merchant_name: brand slug — matches a filename in ``data_root``
            (e.g. ``"meowant"`` reads ``data/raw_data/meowant.csv``).
        data_root: directory holding the CSVs.
        seed: reproducibility seed for the stock assignment. Same seed +
            same CSV -> same world.
        mandate_policy: overrides for the per-SKU private fields. Merged
            into ``DEFAULT_MANDATE_POLICY``.
        policy: overrides for the merchant's stated policy. Merged into
            ``DEFAULT_POLICY``.

    Returns:
        A ``MerchantData`` keyed by raw SKU (e.g. ``"PPB-YB04L25W00004"``)
        unless ``namespace_skus=True``.  Namespaced keys use the exact same
        identity function as the World catalog adapter.
    """
    mp = {**DEFAULT_MANDATE_POLICY, **(mandate_policy or {})}
    pol = {**DEFAULT_POLICY, **(policy or {})}
    root = Path(data_root)
    in_stock_range = tuple(mp["in_stock_qty_range"])

    listings_rows: list[dict[str, Any]] = []
    for _source_row_number, raw, fields in _iter_canonical_listing_rows(
        merchant_name, root
    ):
        source_sku = fields["sku"]
        sku = (
            namespaced_sku_id(merchant_name, source_sku)
            if namespace_skus else source_sku
        )
        sale = fields["sale"]
        attrs = fields["attrs"]
        orig = parse_money(raw.get("Original Price (USD)"))

        # Stock — deterministic per (seed, merchant, source sku).
        qty = assign_stock(
            f"{merchant_name}:{source_sku}",
            raw.get("Availability"),
            seed=seed,
            in_stock_range=in_stock_range,
        )

        # Public list = Sale Price. The merchant's ceiling for negotiation
        # is the public list_price — they will not counter above what they
        # themselves advertised. target_band = (band_low, list_price);
        # auto_accept_threshold sits inside the band; floor is below it.
        list_price = sale
        band_high = list_price
        band_low = int(round(sale * mp["target_band_low_pct"]))
        aat = int(round(sale * mp["auto_accept_pct"]))
        floor = int(round(sale * mp["floor_pct_of_list"]))

        # Original Price (if higher than Sale) is historical / promotional
        # info, not a negotiation ceiling — record it in attributes for
        # downstream skills that might want it, but keep target_band[1]
        # at list_price.
        if orig is not None and orig > sale:
            attrs["original_price_minor"] = int(orig)

        # Enforce the ordering invariant:
        #   floor < band_low < auto_accept <= band_high (= list_price)
        # Clamp aggressively if a low list_price makes any of these
        # collapse (e.g. a $1 SKU where 60% rounding hits 0).
        if band_low >= aat:
            band_low = max(1, aat - 1)
        if floor >= band_low:
            floor = max(1, band_low - 1)
        if aat > list_price:
            aat = list_price
        if floor < 1:
            floor = 1
        permitted = derive_permitted_claims(raw, attrs)

        listings_rows.append({
            "product": sku,
            "list_price": list_price,
            "floor_price": floor,
            "inventory": qty,
            "attributes": attrs,
            "auto_accept_threshold": aat,
            "target_band": (band_low, band_high),
            "permitted_claims": permitted,
            "must_not_claim": tuple(mp["must_not_claim"]),
            "arrived_at": None,
            "inventory_age_days": int(mp["inventory_age_days_default"]),
            "velocity_per_day": float(mp["velocity_per_day_default"]),
        })

    return MerchantData.from_rows(listings=listings_rows, policy=pol)


def iter_public_listings(
    merchant_name: str,
    *,
    data_root: str | Path = "data/raw_data",
    seed: int = 42,
    in_stock_range: tuple[int, int] = (20, 200),
) -> "Iterator[dict[str, Any]]":
    """Yield the PUBLIC catalog view of each sellable SKU in one merchant CSV.

    The complement of :func:`load_from_csv` (which returns the merchant's PRIVATE
    bookkeeping). Both share :func:`_row_listing_fields`, so they agree exactly on
    which rows are listings and how attributes are built. Here the attributes are
    narrowed to the allowlisted public subset (:func:`split_attributes`), and the
    deterministic per-``(seed, merchant, sku)`` stock is included — this is
    precisely what belongs on the world catalog a buyer can read.

    Duplicate source-SKU rows use the same first-occurrence canonicalization as
    :func:`load_from_csv`.  Yields dicts containing both the raw ``sku`` and the
    globally unique ``listing_sku_id``, plus a stable product-level ID and source
    coordinates.
    """
    root = Path(data_root)
    for source_row_number, raw, fields in _iter_canonical_listing_rows(
        merchant_name, root
    ):
        source_sku = fields["sku"]
        source_url = str(fields["attrs"].get("product_url") or "")
        public_attrs, _private = split_attributes(fields["attrs"])
        public_attrs.update({
            "source_label": "real_csv",
            "source_file": _source_file_label(merchant_name, root),
            "source_merchant": merchant_name,
            "source_row_number": source_row_number,
            "source_sku": source_sku,
            "source_product_name": fields["name"],
            "source_product_url": source_url or None,
        })
        qty = assign_stock(
            f"{merchant_name}:{source_sku}",
            raw.get("Availability"),
            seed=seed,
            in_stock_range=in_stock_range,
        )
        yield {
            "sku": source_sku,
            "listing_sku_id": namespaced_sku_id(merchant_name, source_sku),
            "product_id": source_product_id(
                merchant_name,
                product_url=source_url,
                product_name=fields["name"],
            ),
            "list_price": fields["sale"],   # integer minor units (cents)
            "category": fields["category"],
            "name": fields["name"],
            "attributes": public_attrs,
            "inventory": qty,
            "source_row_number": source_row_number,
        }


def catalog_source_statistics(
    merchant_name: str,
    *,
    data_root: str | Path = "data/raw_data",
) -> dict[str, Any]:
    """Return deterministic, machine-checkable statistics for one source CSV."""

    root = Path(data_root)
    path = _csv_path(merchant_name, root)
    row_count = sellable_rows = in_stock_sellable_rows = 0
    canonical_in_stock = 0
    seen_source_skus: set[str] = set()
    product_ids: set[str] = set()
    source_domains: set[str] = set()
    with _open_csv(merchant_name, root) as f:
        reader = csv.DictReader(f)
        column_count = len(reader.fieldnames or ())
        for raw in reader:
            row_count += 1
            fields = _row_listing_fields(raw)
            if fields is None:
                continue
            sellable_rows += 1
            in_stock = (
                str(raw.get("Availability") or "").strip().lower()
                in _IN_STOCK_VALUES
            )
            in_stock_sellable_rows += int(in_stock)
            if fields["sku"] in seen_source_skus:
                continue
            seen_source_skus.add(fields["sku"])
            canonical_in_stock += int(in_stock)
            product_ids.add(
                source_product_id(
                    merchant_name,
                    product_url=str(fields["attrs"].get("product_url") or ""),
                    product_name=fields["name"],
                )
            )
            product_url = str(fields["attrs"].get("product_url") or "").strip()
            if product_url and urlsplit(product_url).hostname:
                source_domains.add(str(urlsplit(product_url).hostname).lower())

    return {
        "merchant": merchant_name,
        "source_file": _source_file_label(merchant_name, root),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "column_count": column_count,
        "row_count": row_count,
        "sellable_row_count": sellable_rows,
        "canonical_listing_count": len(seen_source_skus),
        "duplicate_source_sku_rows": sellable_rows - len(seen_source_skus),
        "in_stock_sellable_row_count": in_stock_sellable_rows,
        "in_stock_canonical_listing_count": canonical_in_stock,
        "product_group_count": len(product_ids),
        "source_domains": sorted(source_domains),
    }


def list_merchants(data_root: str | Path = "data/raw_data") -> list[str]:
    """Discover available merchant CSVs in ``data_root``."""
    root = Path(data_root)
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.csv"))


def merchant_summary(data: MerchantData) -> dict[str, Any]:
    """Compact summary of a loaded MerchantData. Useful for sanity checks.

    Reads from ``private_states`` (B5). ``seed_list_price`` /
    ``seed_inventory`` are the spawn-time snapshots loaded from CSV; for
    live runtime numbers, query the world catalog / inventory directly.
    """
    states = list(data.private_states.values())
    if not states:
        return {"sku_count": 0}
    prices = [s.seed_list_price for s in states]
    invs = [s.seed_inventory for s in states]
    return {
        "sku_count": len(data.private_states),
        "in_stock_count": sum(1 for s in states if s.seed_inventory > 0),
        "out_of_stock_count": sum(1 for s in states if s.seed_inventory == 0),
        "list_price_minor_min": min(prices),
        "list_price_minor_max": max(prices),
        "list_price_minor_median": sorted(prices)[len(prices) // 2],
        "total_inventory_units": sum(invs),
        "sample_skus": list(data.private_states)[:3],
    }
