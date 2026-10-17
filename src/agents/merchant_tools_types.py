"""Merchant-side data types: exception hierarchy + tier-1/tier-2 records.

Split out of ``merchant_tools.py`` so the shape definitions live separate
from the (much larger) set of tool functions that operate on them. Same
public surface — ``from agents.merchant_tools import OfferMandateView``
keeps working via the re-export shim in ``merchant_tools.py``.

``MerchantPrivateState`` and ``PUBLIC_ATTRIBUTE_KEYS`` continue to live
in :mod:`agents.merchant_private_state` (where B5 introduced them);
nothing here duplicates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- errors -----------------------------------------------------------

class ToolError(Exception):
    """Base error for merchant tool calls."""


class ProductNotFound(ToolError):
    """No listing / intel for the requested product (category)."""


class OrderNotFound(ToolError):
    """No order with the requested id."""


# --- Tier 1: transactional record types -------------------------------

@dataclass(frozen=True)
class OfferMandateView:
    """Tier 1. The merchant's authorized negotiating bounds."""
    product: str
    floor_price: int                       # PRIVATE_UTILITY (minor units)
    auto_accept_threshold: int             # minor units
    target_band: tuple[int, int]           # (low, high) minor units
    permitted_claims: tuple[str, ...] = ()
    must_not_claim: tuple[str, ...] = ()


@dataclass(frozen=True)
class Policy:
    """Tier 1. Refund / claim policy + stock-management rules.

    ``markdown_rules`` — used by ``aging-markdown`` + ``aged_stock_discount``:
        ``after_days``         : no markdown before this age
        ``step_interval_days`` : one step every N days past ``after_days``
        ``step_pct``           : percent off per step
        ``max_pct``            : absolute cap on total discount
    ``scarcity_window_days`` — ``stockout-aware-pricing`` fires when
        ``days_to_stockout <=`` this.
    ``demand_floor_for_scarcity`` — and only if ``opp_score >=`` this.
    ``stockout_threshold`` — ``restock-signal`` fires when ``count <=`` this.
    """
    refund_policy: str = "no_returns"
    return_window_days: int = 0
    claim_aggressiveness: str = "neutral"
    # Anti-endless-negotiation cap. After this many merchant counter_offers
    # against the same (buyer_id, offer_id) thread, the merchant unilaterally
    # rejects with reason="max_rounds_exceeded". 0 disables the cap (the
    # natural floor / auto_accept gates are the only termination).
    max_negotiation_rounds: int = 5
    # Patience model — a dynamic, offer-quality-aware drain that runs alongside
    # the round cap. The merchant starts each (buyer_id, offer_id) thread with
    # ``initial_patience`` units; every round drains an amount scaled by how
    # far the buyer's offer is below ``auto_accept_threshold`` (an offer right
    # at auto_accept drains 0; an offer at half of auto_accept drains
    # ``decay_multiplier / 2``). Patience <= 0 -> reject with
    # reason="patience_exhausted". 0 disables the patience model.
    initial_patience: int = 100
    patience_decay_multiplier: int = 100
    # Minimum offer step (PUBLIC like target_band) — fraction of list_price the
    # buyer must raise by between consecutive offers in the same (buyer_id,
    # offer_id) thread. 0.02 = 2% of list. An offer that raises less than this
    # is treated as bad-faith haggling and rejected with
    # reason="step_below_minimum". 0.0 disables the rule entirely. The
    # effective per-SKU step is ``compute_min_offer_step(list_price, pct)``.
    min_offer_step_pct: float = 0.02
    markdown_rules: dict[str, int] = field(default_factory=lambda: {
        "after_days": 30, "step_interval_days": 14, "step_pct": 5, "max_pct": 30})
    scarcity_window_days: int = 7
    demand_floor_for_scarcity: int = 60
    stockout_threshold: int = 2
    # demand-driven-markup (mirror of markdown for fresh+fast+high-demand)
    markup_rules: dict[str, int] = field(default_factory=lambda: {
        "step_pct": 5, "max_pct": 15, "cooldown_days": 7})
    markup_demand_floor: int = 80      # opp_score must be >= this
    markup_velocity_floor: float = 0.3  # units/day, must be >= this
    # peer-pricing — proactively align list_price with comparable items
    # already listed in the world catalog. ``strategy`` is public and
    # readable from policy; the value (median / quartile / etc.) is the
    # band anchor. ``min_peers`` is the floor on how many similar items
    # must exist before the signal is acted on (thin peer sets are
    # ignored). ``cooldown_days`` prevents back-to-back alignments.
    # ``similarity_threshold`` is the attribute-overlap bar a peer must
    # clear; ``search_limit`` caps the world.read_catalog page size.
    peer_pricing_strategy: str = "match_median"
    peer_pricing_step_pct: float = 0.05
    peer_pricing_min_peers: int = 3
    peer_pricing_cooldown_days: int = 14
    peer_pricing_similarity_threshold: float = 0.5
    peer_pricing_search_limit: int = 25


@dataclass(frozen=True)
class Order:
    """Tier 1. Minimal order record for fulfillment + returns.

    ``delivered_at`` (ISO date) starts the return-window clock for
    ``return-adjudicate``; ``None`` means the order has not been delivered
    yet (a return request would land outside the window of any policy).
    """
    order_id: str
    product: str
    qty: int
    amount: int                            # minor units
    status: str = "created"
    delivered_at: str | None = None


@dataclass(frozen=True)
class Reputation:
    """Merchant's own platform reputation snapshot.

    A merchant-side mirror of ``world.read_reputation`` (per VCP_SPEC §6).
    The reputation update path is platform-only — the merchant never writes
    its own; it only reads to inform offer-turn behavior
    (``reputation-aware-pricing``).
    """
    avg_rating: float = 0.0           # 0.0 .. 5.0
    review_count: int = 0


# --- Tier 2: market-intelligence views --------------------------------

@dataclass(frozen=True)
class DemandSignals:
    """Tier 2. Aggregated demand for a product (category)."""
    product: str
    monthly_volume: int
    keyword_count: int
    avg_kd: float
    intent_mix: dict[str, int] = field(default_factory=dict)
    top_user_scenarios: tuple[str, ...] = ()
    opp_score: int = 0


@dataclass(frozen=True)
class PriceBenchmark:
    """Tier 2. Competitor price anchors (minor units), parsed from free text.

    ``coverage`` = competitors with a parseable price / total competitors —
    the dataset's only price signal is weak, so callers should weight by it.
    """
    product: str
    band_low: int | None
    band_high: int | None
    samples: tuple[tuple[str, int], ...] = ()   # (domain, price_minor_units)
    coverage: float = 0.0


@dataclass(frozen=True)
class CompetitorInfo:
    """Tier 2. One competitor active in the product's category."""
    domain: str
    product_detail: str
    total_traffic: int
    organic_traffic: int
    relevance: str
    overlap_score: int


@dataclass(frozen=True)
class Visibility:
    """Tier 2. The seller's own market standing in a category."""
    product: str
    coverage_rate: float
    mw_covered: int
    mw_not_covered: int
    high_opp_count: int
    comp_total_clicks: int


@dataclass(frozen=True)
class Positioning:
    """Tier 2. Suggested angles / channels / claim candidates."""
    product: str
    content_hooks: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    claim_candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpportunityGaps:
    """Tier 2. Quick wins and gap keywords for a category."""
    product: str
    quick_wins: tuple[str, ...] = ()
    gap_keywords: tuple[str, ...] = ()


__all__ = [
    "CompetitorInfo",
    "DemandSignals",
    "OfferMandateView",
    "OpportunityGaps",
    "Order",
    "OrderNotFound",
    "Policy",
    "PriceBenchmark",
    "Positioning",
    "ProductNotFound",
    "Reputation",
    "ToolError",
    "Visibility",
]
