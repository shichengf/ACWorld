"""Pure pricing math — domain-agnostic helpers shared by every lane.

These functions are stateless, side-effect-free, and operate on **integer
minor units** per VCP_SPEC.md §3 (``$3.50 == 350``). They live in the
protocol layer because:

* every lane (merchant / buyer / platform / world) may need them when
  reasoning about price, scarcity, reputation, or the negotiation gate;
* keeping them lane-independent avoids cross-lane coupling — the merchant
  shouldn't own math that, e.g., a platform-side adjudicator skill might
  also want when sanity-checking a refund;
* they have no I/O, no clock, and no memory access, so they're trivially
  testable.

This module is intentionally narrow: only deterministic math. Anything that
constructs a VCP envelope lives in :mod:`protocol.envelopes`; anything that
reads or mutates lane-private state lives on that lane (``agents/`` or
``world/``).
"""

from __future__ import annotations

from typing import Any


# --- Negotiation counters --------------------------------------------

def compute_counter(offered: int, target_band_high: int, floor: int) -> int:
    """Stateless single-shot counter: midpoint of ``(offered, band high)``,
    clamped strictly above ``floor``.

    Pure; this is also the negotiation ablation baseline (the simplest
    possible deterministic counter). For multi-round negotiation where the
    merchant should concede across rounds (counter monotonically
    non-increasing), use :func:`compute_concession_counter` instead.
    """
    offered = int(offered)
    return max(int(floor) + 1, (offered + int(target_band_high)) // 2)


def compute_concession_counter(offered: int, target_band_high: int, floor: int,
                                *, previous_counter: int | None = None) -> int:
    """Multi-round-aware counter for negotiation threads.

    Round 1 (``previous_counter is None``): behaves identically to
    :func:`compute_counter` — midpoint of ``(offered, target_band_high)``,
    clamped above ``floor``.

    Rounds 2+ (``previous_counter`` set): new counter is the midpoint of
    ``(offered, previous_counter)`` — converging the gap between the buyer's
    offer and the merchant's prior position rather than walking back toward
    the band high. The result is clamped to satisfy two invariants:

    * **Floor invariant** — ``counter > floor`` (private-utility never
      crossed, never disclosed).
    * **Concession monotonicity** — ``counter <= previous_counter`` (the
      merchant never raises a counter against itself, no matter what the
      buyer does).

    Together these give realistic give-and-take: the buyer ratchets up, the
    merchant concedes (or holds), and the gap narrows each round.

    Raises nothing — all inputs are coerced to ``int``.
    """
    offered = int(offered)
    band_high = int(target_band_high)
    floor = int(floor)
    if previous_counter is None:
        return max(floor + 1, (offered + band_high) // 2)
    previous_counter = int(previous_counter)
    new_counter = (offered + previous_counter) // 2
    new_counter = max(new_counter, floor + 1)
    new_counter = min(new_counter, previous_counter)
    return new_counter


def compute_min_offer_step(list_price: int, pct: float, *,
                           absolute_minimum: int = 1) -> int:
    """Per-SKU effective minimum offer step (integer minor units).

    ``pct`` is a fraction of ``list_price`` (e.g. ``0.02`` = 2%). The result
    is clamped to ``absolute_minimum`` (default 1 minor unit) so a very
    cheap SKU still has a non-zero step. A non-positive ``pct`` disables
    the rule (returns ``0``).

    The negotiation gate rejects any buyer offer whose raise from the
    previous offer in the same ``(buyer_id, offer_id)`` thread is below
    this value, with ``reason="step_below_minimum"``. The value itself is
    PUBLIC — both sides compute it from ``list_price`` (public) and
    ``policy.min_offer_step_pct`` (public via ``world.read_policy``).
    """
    pct = float(pct)
    if pct <= 0:
        return 0
    value = int(round(int(list_price) * pct))
    return max(int(absolute_minimum), value)


def compute_patience_loss(offered: int, auto_accept_threshold: int, *,
                          decay_multiplier: int = 100) -> int:
    """Patience drained for one negotiation round, scaled by offer quality.

    Loss = ``(auto_accept - offered) / auto_accept * decay_multiplier``,
    integer-rounded. An offer at or above ``auto_accept`` drains zero (the
    merchant would auto-accept anyway). An offer at half of ``auto_accept``
    drains half the multiplier.

    Calibration with defaults (initial_patience=100, decay_multiplier=100):

        offer / auto_accept   ->   loss / round   ->  rounds until 0
        0.99 (close)              ~1                 ~100
        0.90                      ~10                ~10
        0.80                      ~20                ~5
        0.50 (lowball)            ~50                ~2
        0.10 (extreme)            ~90                ~1

    The pricing-negotiate skill subtracts the loss from
    ``memory[TRANSACTION][f"negotiation:{buyer_id}:{offer_id}"].patience_remaining``
    and rejects with ``reason="patience_exhausted"`` at zero.

    Returns:
        Integer loss in ``[0, decay_multiplier]``. Never negative.
    """
    offered = int(offered)
    auto_accept = max(1, int(auto_accept_threshold))
    if offered >= auto_accept:
        return 0
    distance = (auto_accept - offered) / auto_accept
    loss = int(round(distance * int(decay_multiplier)))
    return max(0, loss)


# --- Price-step helpers (aging markdown / demand markup) -------------

def aged_stock_discount(list_price: int, age_days: int,
                        floor_price: int, markdown_rules: dict[str, int]) -> int:
    """Deterministic markdown helper.

    Steps the public list price down per ``markdown_rules`` once age exceeds
    ``after_days``. Returns an integer minor-unit price strictly above
    ``floor_price``; when not yet eligible, returns ``list_price`` unchanged.

    ``markdown_rules`` keys:

    * ``after_days`` — no markdown before this age.
    * ``step_interval_days`` — one step every N days past ``after_days``.
    * ``step_pct`` — percent off per step.
    * ``max_pct`` — absolute cap on total discount.

    Pure — no I/O, no clock; the same inputs always yield the same output.
    """
    after = int(markdown_rules.get("after_days", 30))
    interval = int(markdown_rules.get("step_interval_days", 14))
    step_pct = int(markdown_rules.get("step_pct", 5))
    cap_pct = int(markdown_rules.get("max_pct", 30))
    age = int(age_days)
    if age <= after or interval <= 0 or step_pct <= 0:
        return int(list_price)
    steps = (age - after) // interval + 1
    disc_pct = min(steps * step_pct, cap_pct)
    new_price = int(list_price) - (int(list_price) * disc_pct) // 100
    return max(new_price, int(floor_price) + 1)


def demand_step_markup(list_price: int, ceiling: int,
                       markup_rules: dict[str, int]) -> int:
    """Deterministic markup helper (mirror of :func:`aged_stock_discount`).

    Bounds a single upward step on ``list_price``. The new price is the
    smaller of ``ceiling`` (the mandate's ``target_band[1]`` — never exceed)
    and ``list_price * (1 + step_pct/100)``, with ``step_pct`` itself
    clamped by ``max_pct`` of the original.

    ``markup_rules`` keys:

    * ``step_pct`` — percent up per step.
    * ``max_pct`` — absolute cap on a single step.

    Pure — no I/O, no clock.
    """
    if int(list_price) >= int(ceiling):
        return int(list_price)
    step_pct = int(markup_rules.get("step_pct", 5))
    max_pct = int(markup_rules.get("max_pct", 15))
    if step_pct <= 0:
        return int(list_price)
    step_pct = min(step_pct, max_pct)
    new_price = int(list_price) + (int(list_price) * step_pct) // 100
    return min(new_price, int(ceiling))


# --- Public qualitative bands ----------------------------------------

def availability_band(count: int, stockout_threshold: int) -> str:
    """Map a raw inventory count into a public qualitative band.

    Buyers must never see the integer count (effectively a scarcity-cue
    leak). Bands:

    * ``"out_of_stock"`` — ``count <= 0``
    * ``"limited"`` — at or below the policy stockout threshold
    * ``"in_stock"`` — above the threshold
    """
    c = int(count)
    if c <= 0:
        return "out_of_stock"
    if c <= int(stockout_threshold):
        return "limited"
    return "in_stock"


def reputation_band(avg_rating: float, review_count: int, *,
                    high_rating: float = 4.5, high_reviews: int = 50,
                    low_rating: float = 3.5, low_reviews: int = 10) -> str:
    """Map a reputation snapshot to a qualitative band.

    Bands:

    * ``"high"`` — strong rating AND meaningful review volume.
    * ``"low"`` — weak rating OR thin volume.
    * ``"medium"`` — anything in between (the no-bias default).

    Accepts primitives so the protocol layer doesn't import lane-specific
    dataclasses. Merchant-side callers unwrap their ``Reputation`` value:
    ``reputation_band(rep.avg_rating, rep.review_count)``.
    """
    if avg_rating >= high_rating and review_count >= high_reviews:
        return "high"
    if avg_rating < low_rating or review_count < low_reviews:
        return "low"
    return "medium"


# --- Negotiation-band biases -----------------------------------------

def apply_reputation_bias(target_band: tuple[int, int], band: str
                          ) -> tuple[int, int]:
    """Shift the effective negotiation band per qualitative reputation band.

    * ``"high"``   — top quarter: ``(hi - span/4, hi)``. Established sellers
      defend premium pricing on offer turns.
    * ``"low"``    — bottom quarter: ``(lo, lo + span/4)``. Lower-reputation
      sellers compete on price; the band collapses toward the floor end so
      the negotiation midpoint lands lower.
    * ``"medium"`` — ``target_band`` unchanged.

    Pure. Feeds the same path :func:`apply_pricing_bias` uses;
    reputation-aware-pricing writes the result into
    ``memory[PREFERENCE]['pricing_bias']`` so pricing-negotiate consumes it
    transparently.
    """
    lo, hi = int(target_band[0]), int(target_band[1])
    if hi <= lo:
        return (lo, hi)
    span = hi - lo
    quarter = max(1, span // 4)
    if band == "high":
        return (hi - quarter, hi)
    if band == "low":
        return (lo, lo + quarter)
    return (lo, hi)


def apply_pricing_bias(target_band: tuple[int, int],
                       bias: dict[str, Any] | None,
                       sku_id: str | None = None) -> tuple[int, int]:
    """Return the effective negotiation band, applying a scarcity bias if set.

    ``bias`` is the dict written by ``stockout-aware-pricing`` into
    ``MemoryType.PREFERENCE["pricing_bias"]``:
    ``{"sku_id"?: str, "effective_band": [low, high], "reason": str}``.
    When the bias does not match this SKU (or no bias is set), returns
    ``target_band`` unchanged — so pricing-negotiate's default midpoint
    behavior is preserved when scarcity does not apply.
    """
    if not bias:
        return (int(target_band[0]), int(target_band[1]))
    bsku = bias.get("sku_id")
    if sku_id is not None and bsku not in (None, sku_id):
        return (int(target_band[0]), int(target_band[1]))
    eb = bias.get("effective_band")
    if eb and len(eb) == 2:
        return (int(eb[0]), int(eb[1]))
    return (int(target_band[0]), int(target_band[1]))


# --- Peer pricing (compare-against-similar-items) --------------------
#
# Helpers a merchant skill uses after pulling competitor listings via
# ``world.read_catalog``: turn ``Money`` list-prices into integer minor
# units, summarize the peer distribution, score similarity against the
# merchant's own SKU, and recommend a list price under a chosen strategy.
#
# All math is pure and lane-agnostic — the buyer side could just as
# easily run a "is the merchant's price above its peers?" check using
# the same helpers.


def extract_list_prices_minor(
    listings: list[Any],
    *,
    money_to_minor: Any = None,
    exclude_sku_ids: tuple[str, ...] = (),
    exclude_merchant_ids: tuple[str, ...] = (),
) -> tuple[int, ...]:
    """Pull integer-minor list_prices from a list of world Listings.

    ``listings`` is the return of ``world.read_catalog`` /
    ``WorldTools.search_catalog`` — a list of objects with
    ``.sku_id``, ``.merchant_id`` and ``.list_price`` (a ``Money``).

    ``money_to_minor`` is a callable ``(Money) -> int`` provided by the
    caller. Money is currency-aware and lives in :mod:`world.types`; the
    protocol layer stays unaware of it. Callers typically pass
    ``lambda m: int((m.amount * 100).to_integral_exact())``.

    ``exclude_sku_ids`` / ``exclude_merchant_ids`` let the caller drop
    its own SKU (or its own merchant) from the peer set — the merchant
    must never compare itself against itself.

    Returns:
        Tuple of integer minor-unit prices, in input order. SKUs without
        a parseable price are silently dropped.
    """
    if money_to_minor is None:
        # Conservative default: assume ``list_price.amount`` is already
        # integer-minor (some test scaffolds skip the Money wrapper).
        def money_to_minor(m: Any) -> int:
            amount = getattr(m, "amount", m)
            return int(amount)
    out: list[int] = []
    excluded_skus = frozenset(exclude_sku_ids)
    excluded_merchants = frozenset(exclude_merchant_ids)
    for lst in listings:
        if str(getattr(lst, "sku_id", "")) in excluded_skus:
            continue
        if str(getattr(lst, "merchant_id", "")) in excluded_merchants:
            continue
        try:
            out.append(int(money_to_minor(lst.list_price)))
        except (AttributeError, TypeError, ValueError):
            continue
    return tuple(out)


def peer_price_stats(prices: tuple[int, ...] | list[int]) -> dict[str, int]:
    """Summarize a list of integer-minor peer prices.

    Returns ``{n, min, max, mean, median, p25, p75}`` — all integer
    minor units (mean / quartiles rounded down, the same direction
    integer division naturally takes). Empty input returns ``{n: 0}``.

    The quartiles use the "exclusive" rank method (sort + linear index)
    which is good enough for the small N (5–25 peers) we get from
    ``world.search_catalog``; we don't need numpy here.
    """
    xs = sorted(int(p) for p in prices)
    n = len(xs)
    if n == 0:
        return {"n": 0}

    def _percentile(rank: float) -> int:
        # rank in [0, 1], linear interpolation between adjacent samples.
        if n == 1:
            return xs[0]
        pos = rank * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return int(xs[lo] + frac * (xs[hi] - xs[lo]))

    return {
        "n": n,
        "min": xs[0],
        "max": xs[-1],
        "mean": sum(xs) // n,
        "median": _percentile(0.5),
        "p25": _percentile(0.25),
        "p75": _percentile(0.75),
    }


def peer_similarity_score(
    target_attrs: dict[str, Any],
    peer_attrs: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Return a similarity score in ``[0.0, 1.0]`` over attribute overlap.

    For each key in ``target_attrs``:

    * if the peer has the same value (``==``) → counts as a full match;
    * if the peer has a *different* value for that key → counts as a
      full mismatch (the key exists on both, so disagreement is a hard
      no);
    * if the peer is missing the key → counts as a *half match*
      (uncertain, neither penalize nor reward fully).

    Weights default to 1.0 per key; pass ``weights={"roast": 2.0, ...}``
    to make certain attributes count double / half. Numeric-typed
    attribute values match by equality only (no fuzzy bands here —
    that's the skill body's job if it wants a tolerance).

    Pure. Empty ``target_attrs`` returns ``0.0`` (no signal).
    """
    if not target_attrs:
        return 0.0
    weights = weights or {}
    total = 0.0
    matched = 0.0
    for key, tval in target_attrs.items():
        w = float(weights.get(key, 1.0))
        if w <= 0:
            continue
        total += w
        if key not in peer_attrs:
            matched += 0.5 * w
            continue
        if peer_attrs[key] == tval:
            matched += w
    if total == 0:
        return 0.0
    return round(matched / total, 4)


def filter_similar_peers(
    listings: list[Any],
    target_attrs: dict[str, Any],
    *,
    threshold: float = 0.5,
    weights: dict[str, float] | None = None,
) -> list[Any]:
    """Return the subset of ``listings`` whose attribute overlap with
    ``target_attrs`` meets ``threshold``.

    Pure. ``target_attrs`` is typically ``my_listing.attributes`` and
    each peer's ``listing.attributes`` is scored via
    :func:`peer_similarity_score`. The default ``0.5`` keeps "half-
    matched, half-uncertain" peers in the bucket; raise to ``0.8`` for
    a strict near-identical match.

    Returns the peers in their original order — callers that want
    similarity-ranked output should sort the result themselves.
    """
    if not target_attrs:
        return list(listings)
    out: list[Any] = []
    for lst in listings:
        peer_attrs = getattr(lst, "attributes", {}) or {}
        if peer_similarity_score(target_attrs, peer_attrs, weights=weights) >= threshold:
            out.append(lst)
    return out


# Strategy names accepted by :func:`recommend_list_price_from_peers`.
PEER_PRICING_STRATEGIES: frozenset[str] = frozenset({
    "match_median",     # set price exactly at peer median
    "match_mean",
    "undercut_median",  # one step below median (parametrised by ``step_pct``)
    "premium_median",   # one step above median
    "match_p25",        # value-tier — sit at 25th percentile
    "match_p75",        # premium-tier — sit at 75th percentile
})


def recommend_list_price_from_peers(
    stats: dict[str, int],
    *,
    strategy: str = "match_median",
    step_pct: float = 0.05,
    floor: int = 0,
    ceiling: int | None = None,
) -> int | None:
    """Pick a recommended integer-minor list price from peer ``stats``.

    ``stats`` is the return of :func:`peer_price_stats`. Returns ``None``
    when ``stats["n"] == 0`` (no peers — no recommendation; the caller
    keeps its current price).

    Strategies:

    * ``match_median`` / ``match_mean`` / ``match_p25`` / ``match_p75`` —
      sit at the named anchor (integer minor units).
    * ``undercut_median`` — ``median * (1 - step_pct)``, integer-rounded.
      Use when the merchant wants to win a price-sensitive buyer.
    * ``premium_median`` — ``median * (1 + step_pct)``. Use when the
      merchant has a quality / reputation reason to sit above peers.

    The result is clamped:

    * ``> floor`` (private-utility never crossed). Returns ``floor + 1``
      if the strategy would otherwise breach.
    * ``<= ceiling`` when provided (mandate ``target_band[1]`` or peer
      ``max``, whichever the caller wants enforced).

    Raises:
        ValueError: ``strategy`` not in :data:`PEER_PRICING_STRATEGIES`.
    """
    if strategy not in PEER_PRICING_STRATEGIES:
        raise ValueError(f"unknown peer-pricing strategy: {strategy!r}")
    if int(stats.get("n", 0)) <= 0:
        return None
    median = int(stats["median"])
    if strategy == "match_median":
        target = median
    elif strategy == "match_mean":
        target = int(stats["mean"])
    elif strategy == "match_p25":
        target = int(stats["p25"])
    elif strategy == "match_p75":
        target = int(stats["p75"])
    elif strategy == "undercut_median":
        target = median - int(round(median * float(step_pct)))
    else:  # premium_median
        target = median + int(round(median * float(step_pct)))
    target = max(target, int(floor) + 1)
    if ceiling is not None:
        target = min(target, int(ceiling))
    return int(target)


__all__ = [
    "PEER_PRICING_STRATEGIES",
    "aged_stock_discount",
    "apply_pricing_bias",
    "apply_reputation_bias",
    "availability_band",
    "compute_concession_counter",
    "compute_counter",
    "compute_min_offer_step",
    "compute_patience_loss",
    "demand_step_markup",
    "extract_list_prices_minor",
    "filter_similar_peers",
    "peer_price_stats",
    "peer_similarity_score",
    "recommend_list_price_from_peers",
    "reputation_band",
]
