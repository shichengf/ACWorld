"""Example: a merchant-diversity policy independent of the platform core."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from episode.extensions import PLATFORM_POLICIES


def _rank(candidates: Iterable[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    by_merchant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_merchant[str(candidate["merchant_id"])].append(dict(candidate))
    for rows in by_merchant.values():
        rows.sort(
            key=lambda row: (int(row.get("unit_price", 0)), str(row.get("sku_id", "")))
        )
    result: list[dict[str, Any]] = []
    merchants = sorted(by_merchant)
    while len(result) < limit and any(by_merchant.values()):
        for merchant in merchants:
            if by_merchant[merchant] and len(result) < limit:
                result.append(by_merchant[merchant].pop(0))
    return result


@PLATFORM_POLICIES.decorator(
    "example.round-robin-ranking",
    description="Interleave merchants before repeating exposure",
)
def factory(config: Mapping[str, Any]) -> Any:
    default_limit = int(config.get("limit", 10))

    def rank(
        candidates: Iterable[Mapping[str, Any]], *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        return _rank(candidates, limit=default_limit if limit is None else limit)

    return rank
