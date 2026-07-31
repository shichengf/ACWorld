"""World-authoritative social-choice oracle.

Friend ratings copied into an agent trace are claims, not truth.  This module
derives the only scored friend signal from the initial World snapshot's
buyer-scoped friendship and review tables.  It applies only as a genuine
tiebreak after hard feasibility, soft utility, and public price are tied.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from evals import primitives as P
from evals.episode_oracle import EpisodeOracle


@dataclass(frozen=True, slots=True)
class AuthoritativeFriendOracle:
    """Sanitized social-oracle result; raw ratings are intentionally omitted."""

    applicable: bool
    discriminating: bool | None
    candidate_skus: tuple[str, ...]
    preferred_skus: tuple[str, ...]
    review_ids: tuple[str, ...]
    reason: str


def _not_applicable(reason: str, candidates: tuple[str, ...] = ()) -> AuthoritativeFriendOracle:
    return AuthoritativeFriendOracle(
        applicable=False,
        discriminating=None,
        candidate_skus=candidates,
        preferred_skus=(),
        review_ids=(),
        reason=reason,
    )


def derive_authoritative_friend_oracle(
    oracle: EpisodeOracle,
    initial_snapshot: Any,
    *,
    buyer_id: str,
) -> AuthoritativeFriendOracle:
    """Derive the buyer's friend-preferred set from initial World state only.

    At least two friend reviews per candidate are required. Missing reviews do
    not become implicit zeroes, and an invalid rating is ignored rather than
    coerced. Equal averages produce a non-discriminating result that downstream
    scorers remove from both numerator and denominator.
    """

    candidates = tuple(dict.fromkeys(oracle.soft_optimal_skus))
    if len(candidates) < 2:
        return _not_applicable(
            "fewer than two hard-feasible, soft-optimal candidates", candidates
        )

    listings = {
        str(listing.sku_id): listing
        for listing in getattr(initial_snapshot, "catalog", ()) or ()
    }
    if any(sku not in listings for sku in candidates):
        return _not_applicable(
            "candidate pool is missing from the authoritative catalog", candidates
        )
    prices = {P.listing_unit_cents(listings[sku]) for sku in candidates}
    if len(prices) != 1:
        return _not_applicable(
            "candidate pool is not tied on an authoritative public price", candidates
        )

    friend_ids: set[str] = set()
    for row in getattr(initial_snapshot, "friendships", ()) or ():
        if str(row.buyer_id) == str(buyer_id):
            friend_ids.update(str(friend_id) for friend_id in row.friends)
    if not friend_ids:
        return _not_applicable(
            "the settled buyer has no authoritative World friendship row", candidates
        )

    ratings: dict[str, list[int]] = {sku: [] for sku in candidates}
    review_ids: list[str] = []
    for review in getattr(initial_snapshot, "reviews", ()) or ():
        sku = str(review.sku_id)
        if sku not in ratings or str(review.reviewer_id) not in friend_ids:
            continue
        rating = getattr(review, "rating", None)
        if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
            continue
        ratings[sku].append(rating)
        review_ids.append(str(review.review_id))

    if any(len(values) < 2 for values in ratings.values()):
        return _not_applicable(
            "not every tied candidate has at least two authoritative friend reviews",
            candidates,
        )

    scores = {sku: Fraction(sum(values), len(values)) for sku, values in ratings.items()}
    best = max(scores.values())
    preferred = tuple(sku for sku in candidates if scores[sku] == best)
    discriminating = len(set(scores.values())) > 1
    return AuthoritativeFriendOracle(
        applicable=True,
        discriminating=discriminating,
        candidate_skus=candidates,
        preferred_skus=preferred,
        review_ids=tuple(review_ids),
        reason=(
            "authoritative friend reviews distinguish the otherwise-tied candidates"
            if discriminating
            else "authoritative friend reviews are tied across all candidates"
        ),
    )


__all__ = ["AuthoritativeFriendOracle", "derive_authoritative_friend_oracle"]
