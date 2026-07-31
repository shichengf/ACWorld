"""Shared deterministic primitives for the evaluators.

These are the SINGLE implementation of the feature/price semantics: thin public
wrappers that delegate to the legacy scorer's audited helpers, so the step
scorer and the episode oracle reuse exactly the same logic without copying it
and without touching :func:`evals.scorer.score_episode`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from evals import scorer as _s


def meets(feature: str, listing: Any) -> bool:
    """Is ``feature`` satisfied by ``listing`` (attribute-key truthiness first,
    else phrase match). Delegates to the legacy scorer helper."""
    return _s._meets(feature, listing)


def attrs_meet(feature: str, attrs: Any) -> "bool | None":
    """Does ``feature`` hold in a grounded-attributes dict? ``None`` if no
    evidence. Delegates to the legacy scorer helper."""
    return _s._attrs_meet(feature, attrs)


def soft_fit(listing: Any, soft: "list[tuple[str, int]]") -> int:
    """Sum of importance weights of the soft constraints ``listing`` meets."""
    return _s._soft_fit(listing, soft)


def price_cents(money: Any) -> "int | None":
    """Money-like (``.amount`` Decimal) -> integer cents, or ``None``."""
    return _s._price_cents(money)


def settled_order(snapshot: Any) -> Any:
    """The post-settlement order in ``snapshot`` (SETTLED/DISPATCHED/REFUNDED/…)
    or ``None``. Delegates to the legacy scorer helper."""
    return _s._settled_order(snapshot)


def listing_unit_cents(listing: Any) -> "int | None":
    """The listing's public list price in integer cents."""
    return price_cents(getattr(listing, "list_price", None))


def meets_all(listing: Any, features: "list[str]") -> bool:
    """True iff ``listing`` meets every feature in ``features``."""
    return all(meets(f, listing) for f in features)


def soft_pairs(soft_constraints: Any) -> "list[tuple[str, int]]":
    """Normalize a mandate ``soft_constraints`` list into ``(feature, weight)``
    pairs, most-important-first preserved. Accepts the corpus dict form
    ``{"feature": f, "importance": w}`` or a bare ``[feature, ...]``."""
    out: "list[tuple[str, int]]" = []
    for item in soft_constraints or []:
        if isinstance(item, dict):
            f = item.get("feature")
            if f is None:
                continue
            w = item.get("importance", 1)
            try:
                out.append((str(f), int(w)))
            except (TypeError, ValueError):
                out.append((str(f), 1))
        else:
            out.append((str(item), 1))
    return out


def cents(dollars: Any) -> int:
    """Dollars -> integer cents (Decimal-exact)."""
    return int(Decimal(str(dollars)) * 100)


__all__ = [
    "meets", "attrs_meet", "soft_fit", "price_cents", "settled_order",
    "listing_unit_cents", "meets_all", "soft_pairs", "cents",
]
