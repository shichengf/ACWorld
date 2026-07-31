"""``MerchantData`` — the per-merchant data store the tool functions read.

Split out of ``merchant_tools.py`` so the data shape (which the ``from_rows``
loader fills in with multi-source parsing) lives separately from the
(~700-line) catalog of tool functions. Same public surface — the existing
``from agents.merchant_tools import MerchantData`` import path still works
via the re-export shim in ``merchant_tools.py``.

The CSV / Excel loaders (``merchant_data.py``, ``merchant_data_csv.py``)
both call ``MerchantData.from_rows`` to build the in-memory store from
their respective row layouts. The tool functions in ``merchant_tools.py``
read from the resulting store but never construct it themselves.

B5 (ACWorld merchant-data authority design) is the active authority story
this module lives under:

* ``private_states`` is the new home for merchant-private per-SKU state.
* ``inventory_view`` is the live world-read hook used by
  ``get_inventory_status``.
* ``on_envelope`` refreshes ``seed_list_price`` snapshots on
  ``platform.catalog_ack`` (same invalidation pattern InventoryView
  uses for inventory).
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any

from agents.merchant_private_state import (
    MerchantPrivateState,
    split_attributes,
)
from agents.merchant_tools_types import (
    CompetitorInfo,
    DemandSignals,
    OfferMandateView,
    OpportunityGaps,
    Order,
    Policy,
    PriceBenchmark,
    Positioning,
    ProductNotFound,
    Reputation,
    Visibility,
)


# --- category aliases -------------------------------------------------
#
# The seller-side taxonomy and the competitor-analysis file use different
# Chinese terms for the same product (e.g. seller "自动猫砂盆" automatic litter
# box vs competitor "自清洁猫砂盆" self-cleaning litter box). Chinese has no
# word boundaries, so competitor->product matching uses an explicit alias
# list, not naive substring. Override via ``from_rows(category_aliases=...)``.
DEFAULT_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "自动猫砂盆": ("自动猫砂盆", "自清洁猫砂盆", "self clean", "robot litter", "automatic litter"),
    "猫砂盆通用": ("猫砂盆通用", "猫砂盆", "litter box"),
    "不锈钢猫砂盆": ("不锈钢猫砂盆", "不锈钢", "stainless steel litter"),
    "猫砂配件": ("猫砂配件", "猫砂垫", "猫砂"),
    "猫喂食器": ("猫喂食器", "喂食器", "feeder"),
    "猫饮水机": ("猫饮水机", "饮水机", "water fountain"),
    "空气净化器": ("空气净化器", "净化器", "air purifier"),
}


def _aliases_for(product: str, table: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    return table.get(product, (product,))


# --- internal parsing helpers ----------------------------------------

_PRICE_RE = re.compile(r"\$\s?(\d[\d,]*)\s*(?:[-–]\s*(\d[\d,]*))?")


def _pct(v: Any) -> float:
    """'6%' / 0.06 / 6 -> 0.06."""
    if isinstance(v, str) and v.strip().endswith("%"):
        return round(float(v.strip().rstrip("%")) / 100.0, 4)
    f = float(v or 0)
    return round(f / 100.0, 4) if f > 1 else round(f, 4)


def _parse_prices(text: str) -> list[int]:
    """Pull $amounts from competitor free text -> integer minor units (cents).

    Ranges (``$213-450``) expand to both endpoints.
    """
    out: list[int] = []
    for m in _PRICE_RE.finditer(text or ""):
        lo = int(m.group(1).replace(",", "")) * 100
        out.append(lo)
        if m.group(2):
            out.append(int(m.group(2).replace(",", "")) * 100)
    return out


def _benchmark(product: str, comps: list[CompetitorInfo]) -> PriceBenchmark:
    samples: list[tuple[str, int]] = []
    for c in comps:
        for px in _parse_prices(c.product_detail):
            samples.append((c.domain, px))
    prices = [p for _, p in samples]
    cov = round(len({d for d, _ in samples}) / len(comps), 2) if comps else 0.0
    return PriceBenchmark(
        product=product,
        band_low=min(prices) if prices else None,
        band_high=max(prices) if prices else None,
        samples=tuple(samples),
        coverage=cov,
    )


def _need(mapping: dict[str, Any], product: str, what: str) -> Any:
    if product not in mapping:
        raise ProductNotFound(f"no {what} for product {product!r}")
    return mapping[product]


# --- the injected store ----------------------------------------------

@dataclass
class MerchantData:
    """Everything the merchant tools read.

    Built either from the legacy Excel layout (``merchant_data.load_*``) or
    from in-memory rows (``MerchantData.from_rows``) for tests / zero-dep use.
    """
    # B5 (ACWorld merchant-data authority design): authoritative location
    # for merchant-private per-SKU state (floor, velocity, age,
    # private_attrs). World is authoritative for inventory (read via
    # ``inventory_view`` below) and for the current list_price (read via
    # ``world.read_catalog`` — currently only used by skill-side flow;
    # tool-side sanity checks use the ``seed_list_price`` snapshot).
    private_states: dict[str, MerchantPrivateState] = field(default_factory=dict)
    mandates: dict[str, OfferMandateView] = field(default_factory=dict)
    policy: Policy = field(default_factory=Policy)
    orders: dict[str, Order] = field(default_factory=dict)
    demand: dict[str, DemandSignals] = field(default_factory=dict)
    benchmarks: dict[str, PriceBenchmark] = field(default_factory=dict)
    competitors: dict[str, list[CompetitorInfo]] = field(default_factory=dict)
    visibility: dict[str, Visibility] = field(default_factory=dict)
    positioning: dict[str, Positioning] = field(default_factory=dict)
    gaps: dict[str, OpportunityGaps] = field(default_factory=dict)
    reputation: Reputation = field(default_factory=Reputation)
    # B5 (ACWorld merchant-data authority design): optional live read of world
    # inventory. When set, ``get_inventory_status`` reads the count through
    # this view (authoritative); when ``None`` (the test default), it falls
    # back to the spawn-time ``seed_inventory`` snapshot. Replace with a
    # real view in the agent factory or a notebook to get drift-free counts
    # after PSP settles / receive_shipment forwards.
    inventory_view: Any | None = None

    def on_envelope(self, env: Any) -> None:
        """Refresh ``seed_list_price`` snapshots from ``platform.catalog_ack``
        envelopes.

        ``seed_list_price`` on :class:`MerchantPrivateState` is a
        spawn-time snapshot that tool-side sanity checks (``propose_markdown``
        "this is a markdown not a markup" gate, peer-pricing anchor) read.
        It is **not** authoritative — world is. To keep it from going
        stale after a merchant successfully adjusts its own price, this
        hook listens for the CatalogPolicy's ack and replaces the seed
        with the new price.

        Follow-up to the B5 implementation plan: this is the same
        invalidation pattern :class:`InventoryView` uses for inventory,
        but applied to a different field. Called by ``Agent.receive``
        as the first thing in a turn (right after ``inventory_view.on_envelope``);
        no-op for any envelope kind that doesn't match a refreshable
        contract.

        Today only ``commerce.adjust_price`` acks are handled (they
        carry ``list_price_minor`` per
        :func:`agents.platform.CatalogPolicy._adjust_price`). Other
        catalog acks (``commerce.update_listing``, ``commerce.receive_shipment``)
        return different payloads — if their shape becomes relevant
        for cached merchant state, add cases here.

        Product lookup: ``sku_id`` in the ack payload is matched
        against ``MerchantPrivateState.product`` (the CSV loader sets
        them equal). Mismatch → silent no-op rather than crash, so a
        stray ack for a SKU this merchant doesn't own (shouldn't happen
        post-Phase 1's ownership check, but defense in depth) doesn't
        break the turn.
        """
        action = getattr(env, "action", None)
        if not isinstance(action, dict) or action.get("kind") != "platform.catalog_ack":
            return
        payload = action.get("payload") or {}
        if payload.get("kind") not in {
            "world.apply_catalog_mutation",
            # Read-only compatibility for pre-migration recorded acks.  New
            # Platform responses never claim the raw write occurred.
            "world.update_catalog",
        }:
            return
        sku_id = payload.get("sku_id")
        new_price = payload.get("list_price_minor")
        if not isinstance(sku_id, str) or not isinstance(new_price, int):
            return
        state = self.private_states.get(sku_id)
        if state is None:
            return
        self.private_states[sku_id] = dataclasses.replace(
            state, seed_list_price=new_price,
        )

    @staticmethod
    def from_rows(
        *,
        listings: list[dict[str, Any]] | None = None,
        pool_summary: list[dict[str, Any]] | None = None,
        keywords: list[dict[str, Any]] | None = None,
        competitors: list[dict[str, Any]] | None = None,
        orders: list[dict[str, Any]] | None = None,
        policy: dict[str, Any] | None = None,
        category_aliases: dict[str, tuple[str, ...]] | None = None,
        reputation: dict[str, Any] | None = None,
    ) -> "MerchantData":
        """Build from rows mirroring the legacy Excel column schema.

        * ``listings``   — cols: product, list_price, floor_price, inventory,
          attributes, auto_accept_threshold, target_band, permitted_claims,
          must_not_claim  (prices in minor units)
        * ``pool_summary`` — ``traffic-pools-summary.xlsx`` cols: category,
          keyword_count, comp_total_clicks, mw_covered, mw_not_covered,
          high_opp_count, coverage_rate
        * ``keywords``   — ``master-keyword-analysis`` / ``pool-L2-*`` cols:
          product(=category), keyword, volume, kd, intent, user_scenario,
          content_hook, channel, opp_score, mw_status
        * ``competitors`` — ``核心竞品分析`` cols: domain, productCategory,
          productDetail, totalTraffic, organicTraffic, relevance,
          maxOverlapScore
        """
        d = MerchantData()
        if reputation:
            d.reputation = Reputation(
                avg_rating=float(reputation.get("avg_rating", 0) or 0),
                review_count=int(reputation.get("review_count", 0) or 0),
            )
        if policy:
            _defaults = Policy()
            d.policy = Policy(
                refund_policy=policy.get("refund_policy", _defaults.refund_policy),
                return_window_days=int(policy.get("return_window_days", 0) or 0),
                claim_aggressiveness=policy.get("claim_aggressiveness", _defaults.claim_aggressiveness),
                markdown_rules=dict(policy.get("markdown_rules") or _defaults.markdown_rules),
                scarcity_window_days=int(policy.get("scarcity_window_days", _defaults.scarcity_window_days)),
                demand_floor_for_scarcity=int(policy.get("demand_floor_for_scarcity", _defaults.demand_floor_for_scarcity)),
                stockout_threshold=int(policy.get("stockout_threshold", _defaults.stockout_threshold)),
                markup_rules=dict(policy.get("markup_rules") or _defaults.markup_rules),
                markup_demand_floor=int(policy.get("markup_demand_floor", _defaults.markup_demand_floor)),
                markup_velocity_floor=float(policy.get("markup_velocity_floor", _defaults.markup_velocity_floor)),
                max_negotiation_rounds=int(policy.get("max_negotiation_rounds", _defaults.max_negotiation_rounds)),
                initial_patience=int(policy.get("initial_patience", _defaults.initial_patience)),
                patience_decay_multiplier=int(policy.get("patience_decay_multiplier", _defaults.patience_decay_multiplier)),
                min_offer_step_pct=float(policy.get("min_offer_step_pct", _defaults.min_offer_step_pct)),
                peer_pricing_strategy=str(policy.get("peer_pricing_strategy", _defaults.peer_pricing_strategy)),
                peer_pricing_step_pct=float(policy.get("peer_pricing_step_pct", _defaults.peer_pricing_step_pct)),
                peer_pricing_min_peers=int(policy.get("peer_pricing_min_peers", _defaults.peer_pricing_min_peers)),
                peer_pricing_cooldown_days=int(policy.get("peer_pricing_cooldown_days", _defaults.peer_pricing_cooldown_days)),
                peer_pricing_similarity_threshold=float(policy.get("peer_pricing_similarity_threshold", _defaults.peer_pricing_similarity_threshold)),
                peer_pricing_search_limit=int(policy.get("peer_pricing_search_limit", _defaults.peer_pricing_search_limit)),
            )
        for r in listings or []:
            p = r["product"]
            # ``attributes`` partitioned by PUBLIC_ATTRIBUTE_KEYS — the
            # private subset goes to MerchantPrivateState.private_attrs
            # where it cannot accidentally ship to world. The public
            # subset is dropped here (commit-3 of B5): world.read_catalog
            # is now authoritative for the public attributes a buyer
            # sees; merchant-side bookkeeping holds only private keys.
            full_attrs = dict(r.get("attributes", {}) or {})
            _public_attrs, private_attrs = split_attributes(full_attrs)
            d.private_states[p] = MerchantPrivateState(
                product=p,
                floor_price=int(r["floor_price"]),
                arrived_at=r.get("arrived_at"),
                inventory_age_days=int(r.get("inventory_age_days", 0) or 0),
                velocity_per_day=(float(r["velocity_per_day"])
                                  if r.get("velocity_per_day") is not None else None),
                private_attrs=private_attrs,
                seed_list_price=int(r["list_price"]),
                seed_inventory=int(r.get("inventory", 0) or 0),
            )
            band = r.get("target_band") or (int(r["floor_price"]), int(r["list_price"]))
            d.mandates[p] = OfferMandateView(
                product=p,
                floor_price=int(r["floor_price"]),
                auto_accept_threshold=int(r.get("auto_accept_threshold", r["list_price"])),
                target_band=(int(band[0]), int(band[1])),
                permitted_claims=tuple(r.get("permitted_claims", ()) or ()),
                must_not_claim=tuple(r.get("must_not_claim", ()) or ()),
            )
        for r in orders or []:
            d.orders[r["order_id"]] = Order(
                order_id=r["order_id"], product=r["product"],
                qty=int(r["qty"]), amount=int(r["amount"]),
                status=r.get("status", "created"),
                delivered_at=r.get("delivered_at"),
            )
        # demand + visibility
        by_cat_kw: dict[str, list[dict[str, Any]]] = {}
        for r in keywords or []:
            by_cat_kw.setdefault(r["product"], []).append(r)
        for cat, rows in by_cat_kw.items():
            vols = [int(x.get("volume", 0) or 0) for x in rows]
            kds = [float(x.get("kd", 0) or 0) for x in rows]
            intent_mix: dict[str, int] = {}
            for x in rows:
                intent_mix[x.get("intent", "?")] = intent_mix.get(x.get("intent", "?"), 0) + 1
            scen = [x.get("user_scenario") for x in rows if x.get("user_scenario")]
            d.demand[cat] = DemandSignals(
                product=cat,
                monthly_volume=sum(vols),
                keyword_count=len(rows),
                avg_kd=round(sum(kds) / len(kds), 1) if kds else 0.0,
                intent_mix=intent_mix,
                top_user_scenarios=tuple(dict.fromkeys(scen))[:5],
                opp_score=max((int(x.get("opp_score", 0) or 0) for x in rows), default=0),
            )
            d.positioning[cat] = Positioning(
                product=cat,
                content_hooks=tuple(dict.fromkeys(
                    x["content_hook"] for x in rows if x.get("content_hook")))[:5],
                channels=tuple(dict.fromkeys(
                    x["channel"] for x in rows if x.get("channel")))[:5],
                claim_candidates=(),
            )
        for r in pool_summary or []:
            cat = r["category"]
            d.visibility[cat] = Visibility(
                product=cat,
                coverage_rate=_pct(r.get("coverage_rate", 0)),
                mw_covered=int(r.get("mw_covered", 0) or 0),
                mw_not_covered=int(r.get("mw_not_covered", 0) or 0),
                high_opp_count=int(r.get("high_opp_count", 0) or 0),
                comp_total_clicks=int(r.get("comp_total_clicks", 0) or 0),
            )
        # competitors + price benchmarks (alias-aware category match)
        aliases = category_aliases or DEFAULT_CATEGORY_ALIASES
        cats = set(d.demand) | set(d.visibility) | set(d.private_states)
        for r in competitors or []:
            pdetail = str(r.get("productDetail", ""))
            pcat = str(r.get("productCategory", ""))
            hay = pcat + " " + pdetail
            info = CompetitorInfo(
                domain=r["domain"],
                product_detail=pdetail,
                total_traffic=int(r.get("totalTraffic", 0) or 0),
                organic_traffic=int(r.get("organicTraffic", 0) or 0),
                relevance=r.get("relevance", ""),
                overlap_score=int(r.get("maxOverlapScore", 0) or 0),
            )
            for cat in cats:
                if any(a and a in hay for a in _aliases_for(cat, aliases)):
                    d.competitors.setdefault(cat, []).append(info)
        for cat, comps in d.competitors.items():
            d.benchmarks[cat] = _benchmark(cat, comps)
        return d


__all__ = [
    "DEFAULT_CATEGORY_ALIASES",
    "MerchantData",
]
