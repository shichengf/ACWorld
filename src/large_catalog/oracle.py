"""Independent full-catalog oracles for the stress suite."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from large_catalog.database import CatalogDatabase
from large_catalog.models import CatalogListing


class OracleError(ValueError):
    """A task has no well-defined deterministic answer."""


@dataclass(frozen=True, slots=True)
class SelectionOracle:
    feasible_count: int
    objective_value: tuple[Any, ...] | None
    accepted_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "selection",
            "feasible_count": self.feasible_count,
            "objective_value": (
                None if self.objective_value is None else list(self.objective_value)
            ),
            "accepted_refs": list(self.accepted_refs),
            "tie_count": len(self.accepted_refs),
        }


@dataclass(frozen=True, slots=True)
class CartOracle:
    feasible_count: int
    objective_total_minor: int | None
    accepted_carts: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "cart",
            "feasible_count": self.feasible_count,
            "objective_total_minor": self.objective_total_minor,
            "accepted_carts": [list(cart) for cart in self.accepted_carts],
            "tie_count": len(self.accepted_carts),
        }


def selection_oracle(
    database: CatalogDatabase,
    *,
    query: str,
    filters: Mapping[str, Any],
    preference: Mapping[str, Any] | None = None,
) -> SelectionOracle:
    """Find all globally equivalent best listings after hard filtering."""

    candidates = database.full_candidates(query=query, filters=filters, sort="price_asc")
    if not candidates:
        return SelectionOracle(0, None, ())
    preference = dict(preference or {})
    kind = str(preference.get("kind", "lowest_price"))
    if kind == "lowest_price":
        def objective(row: CatalogListing) -> tuple[Any, ...]:
            return (row.price_minor,)
    elif kind == "category_then_price":
        preferred = str(preference.get("category", "")).casefold()

        def objective(row: CatalogListing) -> tuple[Any, ...]:
            return (0 if row.category.casefold() == preferred else 1, row.price_minor)
    elif kind == "feature_then_price":
        feature = str(preference.get("feature", "")).casefold()

        def objective(row: CatalogListing) -> tuple[Any, ...]:
            content = " ".join((row.name, row.tags, row.key_features, row.material)).casefold()
            return (0 if feature and feature in content else 1, row.price_minor)
    elif kind == "merchant_then_price":
        merchant = str(preference.get("merchant", ""))

        def objective(row: CatalogListing) -> tuple[Any, ...]:
            return (0 if row.merchant_id == merchant else 1, row.price_minor)
    elif kind == "category_premium":
        preferred = str(preference.get("category", "")).casefold()
        premium = int(preference.get("premium_minor", 0))
        cheapest = min(row.price_minor for row in candidates)
        preferred_rows = [
            row for row in candidates if row.category.casefold() == preferred
        ]
        choose_preferred = (
            bool(preferred_rows)
            and min(row.price_minor for row in preferred_rows) <= cheapest + premium
        )

        def objective(row: CatalogListing) -> tuple[Any, ...]:
            preferred_rank = 0 if row.category.casefold() == preferred else 1
            if not choose_preferred:
                preferred_rank = 0
            return (preferred_rank, row.price_minor)
    else:
        raise OracleError(f"unsupported selection preference: {kind!r}")
    best = min(objective(row) for row in candidates)
    accepted = tuple(
        sorted(row.listing_ref for row in candidates if objective(row) == best)
    )
    return SelectionOracle(len(candidates), best, accepted)


def comparison_oracle(
    database: CatalogDatabase,
    listing_refs: Sequence[str],
) -> dict[str, Any]:
    rows = tuple(database.listing(ref) for ref in listing_refs)
    if any(row is None for row in rows):
        raise OracleError("comparison references a missing listing")
    listings = tuple(row for row in rows if row is not None)
    if len(listings) != 2:
        raise OracleError("comparison requires exactly two listings")
    left, right = listings
    return {
        "kind": "comparison",
        "listing_refs": [left.listing_ref, right.listing_ref],
        "lower_price_ref": (
            left.listing_ref
            if left.price_minor < right.price_minor
            else right.listing_ref
            if right.price_minor < left.price_minor
            else None
        ),
        "same_category": left.category.casefold() == right.category.casefold(),
        "both_in_stock": left.in_stock and right.in_stock,
        "price_difference_minor": abs(left.price_minor - right.price_minor),
        "same_name": left.name.casefold() == right.name.casefold(),
        "same_variant": left.variant.casefold() == right.variant.casefold(),
    }


def cart_oracle(
    database: CatalogDatabase,
    *,
    requirements: Sequence[Mapping[str, Any]],
    constraints: Mapping[str, Any],
) -> CartOracle:
    """Solve a small cart exactly with branch and bound over complete sets."""

    if not 1 <= len(requirements) <= 5:
        raise OracleError("cart must have between one and five requirements")
    pools: list[tuple[CatalogListing, ...]] = []
    for requirement in requirements:
        queries = requirement.get("queries")
        if queries is None:
            queries = [str(requirement.get("query", ""))]
        if (
            not isinstance(queries, Sequence)
            or isinstance(queries, (str, bytes))
            or not queries
        ):
            raise OracleError("cart requirement queries must be a non-empty list")
        filters = dict(requirement.get("filters", {}))
        filters.setdefault("in_stock", True)
        by_ref: dict[str, CatalogListing] = {}
        for query in queries:
            for row in database.full_candidates(
                query=str(query),
                filters=filters,
                sort="price_asc",
            ):
                by_ref[row.listing_ref] = row
        rows = tuple(sorted(by_ref.values(), key=lambda row: (row.price_minor, row.listing_ref)))
        if not rows:
            return CartOracle(0, None, ())
        pools.append(rows)
    budget = constraints.get("budget_minor")
    if budget is not None and (type(budget) is not int or budget < 0):
        raise OracleError("cart budget must be a non-negative integer")
    min_merchants = int(constraints.get("min_merchants", 1))
    distinct = bool(constraints.get("distinct_listings", True))
    suffix_minimum = [0] * (len(pools) + 1)
    for index in range(len(pools) - 1, -1, -1):
        suffix_minimum[index] = suffix_minimum[index + 1] + pools[index][0].price_minor

    best_total: int | None = None
    accepted: set[tuple[str, ...]] = set()

    def visit(
        index: int,
        chosen: list[CatalogListing],
        total: int,
        merchants: set[str],
        refs: set[str],
    ) -> None:
        nonlocal best_total
        lower_bound = total + suffix_minimum[index]
        if best_total is not None and lower_bound > best_total:
            return
        if budget is not None and lower_bound > budget:
            return
        if index == len(pools):
            if len(merchants) < min_merchants:
                return
            cart = tuple(row.listing_ref for row in chosen)
            if best_total is None or total < best_total:
                best_total = total
                accepted.clear()
                accepted.add(cart)
            elif total == best_total:
                accepted.add(cart)
            return
        for row in pools[index]:
            next_total = total + row.price_minor
            if budget is not None and next_total > budget:
                break
            if best_total is not None and next_total + suffix_minimum[index + 1] > best_total:
                break
            if distinct and row.listing_ref in refs:
                continue
            chosen.append(row)
            refs.add(row.listing_ref)
            added_merchant = row.merchant_id not in merchants
            merchants.add(row.merchant_id)
            visit(index + 1, chosen, next_total, merchants, refs)
            if added_merchant and all(item.merchant_id != row.merchant_id for item in chosen[:-1]):
                merchants.remove(row.merchant_id)
            refs.remove(row.listing_ref)
            chosen.pop()

    visit(0, [], 0, set(), set())
    return CartOracle(
        feasible_count=_count_feasible_carts(
            pools,
            budget_minor=budget,
            min_merchants=min_merchants,
            distinct=distinct,
        ),
        objective_total_minor=best_total,
        accepted_carts=tuple(sorted(accepted)),
    )


def _count_feasible_carts(
    pools: Sequence[Sequence[CatalogListing]],
    *,
    budget_minor: int | None,
    min_merchants: int,
    distinct: bool,
) -> int:
    """Count all feasible carts without using the optimizer's pruning state."""

    reference_sets = [{row.listing_ref for row in pool} for pool in pools]
    disjoint = all(
        not reference_sets[left].intersection(reference_sets[right])
        for left in range(len(reference_sets))
        for right in range(left + 1, len(reference_sets))
    )
    if distinct and not disjoint:
        count = 0

        def visit(
            index: int,
            total: int,
            references: set[str],
            merchants: set[str],
        ) -> None:
            nonlocal count
            if index == len(pools):
                count += int(len(merchants) >= min_merchants)
                return
            for row in pools[index]:
                next_total = total + row.price_minor
                if budget_minor is not None and next_total > budget_minor:
                    continue
                if row.listing_ref in references:
                    continue
                visit(
                    index + 1,
                    next_total,
                    references | {row.listing_ref},
                    merchants | {row.merchant_id},
                )

        visit(0, 0, set(), set())
        return count

    def price_count(selected: Sequence[Sequence[CatalogListing]]) -> int:
        totals: Counter[int] = Counter({0: 1})
        for pool in selected:
            prices = Counter(row.price_minor for row in pool)
            next_totals: Counter[int] = Counter()
            for subtotal, ways in totals.items():
                for price, price_ways in prices.items():
                    combined = subtotal + price
                    if budget_minor is None or combined <= budget_minor:
                        next_totals[combined] += ways * price_ways
            totals = next_totals
        return sum(totals.values())

    all_count = price_count(pools)
    if min_merchants <= 1:
        return all_count
    if min_merchants != 2:
        raise OracleError("suite cart counting supports one or two Merchants")
    same_merchant = 0
    for merchant in {row.merchant_id for pool in pools for row in pool}:
        selected = [
            [row for row in pool if row.merchant_id == merchant] for pool in pools
        ]
        if all(selected):
            same_merchant += price_count(selected)
    return all_count - same_merchant


def independently_verify_cart(
    database: CatalogDatabase,
    *,
    requirements: Sequence[Mapping[str, Any]],
    constraints: Mapping[str, Any],
    expected: CartOracle,
) -> bool:
    """Cross-check the optimum with dynamic programming, not branch and bound."""

    states: dict[tuple[frozenset[str], frozenset[str]], tuple[int, tuple[str, ...]]] = {
        (frozenset(), frozenset()): (0, ())
    }
    budget = constraints.get("budget_minor")
    min_merchants = int(constraints.get("min_merchants", 1))
    distinct = bool(constraints.get("distinct_listings", True))
    for requirement in requirements:
        filters = dict(requirement.get("filters", {}))
        filters.setdefault("in_stock", True)
        queries = requirement.get("queries")
        if queries is None:
            queries = [str(requirement.get("query", ""))]
        by_ref: dict[str, CatalogListing] = {}
        for query in queries:
            for row in database.full_candidates(
                query=str(query),
                filters=filters,
                sort="price_asc",
            ):
                by_ref[row.listing_ref] = row
        candidates = tuple(
            sorted(by_ref.values(), key=lambda row: (row.price_minor, row.listing_ref))
        )
        next_states: dict[
            tuple[frozenset[str], frozenset[str]], tuple[int, tuple[str, ...]]
        ] = {}
        for (merchant_set, ref_set), (total, cart) in states.items():
            for row in candidates:
                if distinct and row.listing_ref in ref_set:
                    continue
                new_total = total + row.price_minor
                if budget is not None and new_total > budget:
                    continue
                key = (
                    merchant_set | {row.merchant_id},
                    ref_set | {row.listing_ref},
                )
                value = (new_total, cart + (row.listing_ref,))
                previous = next_states.get(key)
                if previous is None or value < previous:
                    next_states[key] = value
        states = next_states
    eligible = [
        value for (merchants, _refs), value in states.items() if len(merchants) >= min_merchants
    ]
    actual = None if not eligible else min(total for total, _cart in eligible)
    return actual == expected.objective_total_minor
