"""Finite public cart planning used only by ACWorld T5.

The planner deliberately accepts one self-contained public business problem.
It does not read a benchmark task, Scenario, World snapshot, scorer answer, or
runtime state.  The same rules operate on internal business identities while
building a task and on opaque provider-visible references while driving the
reviewed ideal policy.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


T5_CART_PLANNING_SCHEMA_V1 = "cwe.public-cart-planning.v1"
T5_CART_PLANNING_RULE_SET_V1 = "finite-cart-planning-v1"
T5_CART_ENUMERATION_LIMIT_V1 = 4096

ReferenceKind = Literal["internal", "public"]
CartLines = tuple[tuple[str, int], ...]


class T5CartPlanningError(ValueError):
    """The public planning problem is incomplete, ambiguous, or infeasible."""


@dataclass(frozen=True, slots=True)
class T5CartPlanEvaluationV1:
    lines: CartLines
    constraint_results: Mapping[str, bool]
    subtotal_minor: int | None
    charge_total_minor: int | None
    grand_total_minor: int | None
    max_delivery_days: int | None
    merchant_count: int | None
    objective_key: tuple[Any, ...] | None

    @property
    def hard_feasible(self) -> bool:
        return bool(self.constraint_results) and all(self.constraint_results.values())


@dataclass(frozen=True, slots=True)
class T5CartOracleV1:
    enumeration_count: int
    declared_plans: tuple[CartLines, ...]
    feasible_plans: tuple[T5CartPlanEvaluationV1, ...]
    acceptable_plans: tuple[CartLines, ...]

    @property
    def optimum(self) -> CartLines:
        if len(self.acceptable_plans) != 1:
            raise T5CartPlanningError("T5 cart oracle has no unique optimum")
        return self.acceptable_plans[0]

    @property
    def runner_up(self) -> CartLines:
        optimum = self.optimum
        rows = [row.lines for row in self.feasible_plans if row.lines != optimum]
        if not rows:
            raise T5CartPlanningError("T5 cart oracle has no feasible runner-up")
        return rows[0]


def _reference_fields(kind: ReferenceKind) -> dict[str, str]:
    if kind == "internal":
        return {
            "sku": "sku_id",
            "skus": "sku_ids",
            "merchant": "merchant_id",
            "eligible": "eligible_sku_ids",
            "trigger": "trigger_sku_id",
            "required": "required_sku_id",
        }
    if kind == "public":
        return {
            "sku": "sku_ref",
            "skus": "sku_refs",
            "merchant": "merchant_ref",
            "eligible": "eligible_sku_refs",
            "trigger": "trigger_sku_ref",
            "required": "required_sku_ref",
        }
    raise T5CartPlanningError("unsupported T5 reference kind")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) or not key for key in value):
        raise T5CartPlanningError(f"{label} must be an object")
    return value


def _rows(value: Any, label: str, *, nonempty: bool = False) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise T5CartPlanningError(f"{label} must be an array")
    if nonempty and not value:
        raise T5CartPlanningError(f"{label} must not be empty")
    if any(not isinstance(row, Mapping) for row in value):
        raise T5CartPlanningError(f"{label} contains a non-object row")
    return tuple(value)


def _exact(row: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(row) != fields:
        raise T5CartPlanningError(
            f"{label} fields differ: expected {sorted(fields)!r}, got {sorted(row)!r}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise T5CartPlanningError(f"{label} must be non-empty text")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise T5CartPlanningError(f"{label} must be an integer >= {minimum}")
    return value


def _optional_integer(value: Any, label: str, *, minimum: int = 0) -> int | None:
    return None if value is None else _integer(value, label, minimum=minimum)


def _canonical_lines(
    lines: Sequence[tuple[str, int]],
    *,
    public_reference: Callable[[str], str],
) -> CartLines:
    quantities: dict[str, int] = {}
    for identity, qty in lines:
        identity = _text(identity, "cart line reference")
        qty = _integer(qty, "cart line quantity", minimum=1)
        if identity in quantities:
            raise T5CartPlanningError("cart lines repeat one listing")
        quantities[identity] = qty
    return tuple(
        sorted(
            quantities.items(),
            key=lambda row: (public_reference(row[0]), row[1]),
        )
    )


def _problem_inventory(
    problem: Mapping[str, Any],
    *,
    reference_kind: ReferenceKind,
) -> tuple[
    dict[str, Mapping[str, Any]],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    names = _reference_fields(reference_kind)
    _exact(
        problem,
        {
            "schema_version",
            "rule_set",
            "currency",
            "listing_offers",
            "requirements",
            "relations",
            "pricing_terms",
            "hard_constraints",
            "calculation_rules",
            "objective",
        },
        "cart planning problem",
    )
    if problem.get("schema_version") != T5_CART_PLANNING_SCHEMA_V1:
        raise T5CartPlanningError("cart planning schema changed")
    if problem.get("rule_set") != T5_CART_PLANNING_RULE_SET_V1:
        raise T5CartPlanningError("cart planning rule set changed")
    _text(problem.get("currency"), "cart currency")

    offers: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(
        _rows(problem.get("listing_offers"), "listing_offers", nonempty=True)
    ):
        _exact(
            row,
            {
                names["sku"],
                names["merchant"],
                "product_family",
                "list_price_minor",
                "available_qty",
                "delivery_days",
            },
            f"listing_offers[{index}]",
        )
        sku = _text(row.get(names["sku"]), f"listing_offers[{index}].{names['sku']}")
        if sku in offers:
            raise T5CartPlanningError("listing_offers contains a duplicate listing")
        _text(row.get(names["merchant"]), "listing merchant")
        _text(row.get("product_family"), "listing product_family")
        _integer(row.get("list_price_minor"), "listing price", minimum=1)
        _integer(row.get("available_qty"), "listing inventory", minimum=0)
        _integer(row.get("delivery_days"), "listing delivery_days", minimum=0)
        offers[sku] = row

    requirements = _rows(problem.get("requirements"), "requirements", nonempty=True)
    requirement_keys: set[str] = set()
    eligible_seen: set[str] = set()
    for index, row in enumerate(requirements):
        _exact(
            row,
            {
                "requirement_key",
                "product_family",
                "required_qty",
                names["eligible"],
                "selection_rule",
            },
            f"requirements[{index}]",
        )
        key = _text(row.get("requirement_key"), "requirement_key")
        if key in requirement_keys:
            raise T5CartPlanningError("requirements contains a duplicate key")
        requirement_keys.add(key)
        family = _text(row.get("product_family"), "requirement product_family")
        _integer(row.get("required_qty"), "required_qty", minimum=1)
        if row.get("selection_rule") != "choose_exactly_one_substitute":
            raise T5CartPlanningError("requirement selection rule changed")
        eligible = row.get(names["eligible"])
        if (
            not isinstance(eligible, Sequence)
            or isinstance(eligible, (str, bytes, bytearray))
            or len(eligible) < 2
        ):
            raise T5CartPlanningError("each requirement needs at least two substitutes")
        normalized = tuple(_text(item, "eligible listing") for item in eligible)
        if len(normalized) != len(set(normalized)):
            raise T5CartPlanningError("one requirement repeats an eligible listing")
        if eligible_seen.intersection(normalized):
            raise T5CartPlanningError("one listing cannot satisfy two T5 requirements")
        eligible_seen.update(normalized)
        for sku in normalized:
            offer = offers.get(sku)
            if offer is None or offer.get("product_family") != family:
                raise T5CartPlanningError("requirement references an incompatible listing")

    if eligible_seen != set(offers):
        raise T5CartPlanningError("every listing offer must belong to one requirement")

    relations = _rows(problem.get("relations"), "relations")
    for index, row in enumerate(relations):
        kind = row.get("kind")
        if kind == "required_with":
            _exact(
                row,
                {"kind", names["trigger"], names["required"], "minimum_qty"},
                f"relations[{index}]",
            )
            refs = (row.get(names["trigger"]), row.get(names["required"]))
            if any(_text(ref, "relation listing") not in offers for ref in refs):
                raise T5CartPlanningError("relation references an unknown listing")
            _integer(row.get("minimum_qty"), "relation minimum_qty", minimum=1)
        elif kind == "complement_all_or_none":
            _exact(row, {"kind", names["skus"]}, f"relations[{index}]")
            refs = row.get(names["skus"])
            if (
                not isinstance(refs, Sequence)
                or isinstance(refs, (str, bytes, bytearray))
                or len(refs) < 2
            ):
                raise T5CartPlanningError("complement relation needs at least two listings")
            normalized = tuple(_text(ref, "complement listing") for ref in refs)
            if len(normalized) != len(set(normalized)) or any(
                ref not in offers for ref in normalized
            ):
                raise T5CartPlanningError("complement relation references invalid listings")
        else:
            raise T5CartPlanningError("unsupported cart relation")

    pricing_terms = _rows(problem.get("pricing_terms"), "pricing_terms", nonempty=True)
    covered: set[str] = set()
    for term_index, term in enumerate(pricing_terms):
        _exact(
            term,
            {
                names["skus"],
                "quantity_tiers",
                "bundle_discounts",
                "bundle_stacking",
                "charges",
            },
            f"pricing_terms[{term_index}]",
        )
        raw_skus = term.get(names["skus"])
        if (
            not isinstance(raw_skus, Sequence)
            or isinstance(raw_skus, (str, bytes, bytearray))
            or not raw_skus
        ):
            raise T5CartPlanningError("pricing term has no listing scope")
        skus = tuple(_text(item, "pricing listing") for item in raw_skus)
        if len(skus) != len(set(skus)) or covered.intersection(skus):
            raise T5CartPlanningError("pricing policies overlap or repeat listings")
        if any(sku not in offers for sku in skus):
            raise T5CartPlanningError("pricing term references an unknown listing")
        covered.update(skus)
        tiers = _rows(term.get("quantity_tiers"), "quantity_tiers", nonempty=True)
        previous_max = 0
        for tier_index, tier in enumerate(tiers):
            _exact(
                tier,
                {"minimum_quantity", "maximum_quantity", "unit_price_minor"},
                f"quantity_tiers[{tier_index}]",
            )
            minimum = _integer(tier.get("minimum_quantity"), "tier minimum", minimum=1)
            maximum = _optional_integer(tier.get("maximum_quantity"), "tier maximum", minimum=1)
            _integer(tier.get("unit_price_minor"), "tier unit price", minimum=1)
            if minimum != previous_max + 1 or (maximum is not None and maximum < minimum):
                raise T5CartPlanningError("quantity tiers are not contiguous")
            previous_max = maximum if maximum is not None else previous_max
            if maximum is None and tier_index != len(tiers) - 1:
                raise T5CartPlanningError("only the last quantity tier may be open-ended")
        if tiers[0].get("minimum_quantity") != 1:
            raise T5CartPlanningError("quantity tiers must start at one")
        for sku in skus:
            if offers[sku].get("list_price_minor") != tiers[0].get("unit_price_minor"):
                raise T5CartPlanningError("catalog price differs from the public first tier")
        if term.get("bundle_stacking") not in {"best_only", "cumulative"}:
            raise T5CartPlanningError("unsupported bundle stacking rule")
        for bundle in _rows(term.get("bundle_discounts"), "bundle_discounts"):
            _exact(bundle, {"conditions", "discount_minor", "discount_bps"}, "bundle")
            conditions = _rows(bundle.get("conditions"), "bundle conditions", nonempty=True)
            for condition in conditions:
                _exact(condition, {names["sku"], "minimum_quantity"}, "bundle condition")
                if _text(condition.get(names["sku"]), "bundle listing") not in offers:
                    raise T5CartPlanningError("bundle references an unknown listing")
                _integer(condition.get("minimum_quantity"), "bundle minimum", minimum=1)
            minor = bundle.get("discount_minor")
            bps = bundle.get("discount_bps")
            if (minor is None) == (bps is None):
                raise T5CartPlanningError("bundle needs exactly one discount form")
            if minor is not None:
                _integer(minor, "bundle discount_minor", minimum=1)
            if bps is not None:
                _integer(bps, "bundle discount_bps", minimum=1)
        for charge in _rows(term.get("charges"), "charges"):
            _exact(
                charge,
                {
                    "kind",
                    "fixed_minor",
                    "per_unit_minor",
                    "subtotal_rate_bps",
                    "minimum_subtotal_minor",
                    "maximum_subtotal_minor",
                },
                "charge",
            )
            if charge.get("kind") not in {"shipping", "tax", "fee"}:
                raise T5CartPlanningError("charge kind is invalid")
            for name in (
                "fixed_minor",
                "per_unit_minor",
                "subtotal_rate_bps",
                "minimum_subtotal_minor",
            ):
                _integer(charge.get(name), f"charge {name}", minimum=0)
            maximum = _optional_integer(
                charge.get("maximum_subtotal_minor"),
                "charge maximum_subtotal_minor",
                minimum=0,
            )
            if maximum is not None and maximum <= int(charge["minimum_subtotal_minor"]):
                raise T5CartPlanningError("charge subtotal interval is empty")
    if covered != set(offers):
        raise T5CartPlanningError("pricing terms do not exactly cover listing offers")

    hard = _mapping(problem.get("hard_constraints"), "hard_constraints")
    _exact(
        hard,
        {
            "budget_minor",
            "max_delivery_days",
            "inventory_rule",
            "requirement_rule",
            "relation_rule",
        },
        "hard_constraints",
    )
    _integer(hard.get("budget_minor"), "budget_minor", minimum=1)
    _integer(hard.get("max_delivery_days"), "max_delivery_days", minimum=0)
    if hard.get("inventory_rule") != "selected_qty_lte_available_qty":
        raise T5CartPlanningError("inventory rule changed")
    if hard.get("requirement_rule") != "exact_declared_demand":
        raise T5CartPlanningError("requirement rule changed")
    if hard.get("relation_rule") != "enforce_all_declared_relations":
        raise T5CartPlanningError("relation rule changed")

    calculation = _mapping(problem.get("calculation_rules"), "calculation_rules")
    if calculation != {
        "tier_scope": "sum_selected_quantity_within_pricing_term",
        "bundle_discount_scope": "selected_cart",
        "charge_basis": "base_subtotal_before_bundle_discount",
        "bps_rounding": "floor_minor_units",
        "grand_total": "discounted_line_subtotal_plus_active_charges",
    }:
        raise T5CartPlanningError("public cart calculation rules changed")
    objective = _mapping(problem.get("objective"), "objective")
    if objective != {
        "kind": "lexicographic_min",
        "criteria": [
            "grand_total_minor",
            "max_delivery_days",
            "merchant_count",
        ],
    }:
        raise T5CartPlanningError("public cart objective or tie-break changed")
    return offers, requirements, relations, pricing_terms


def _declared_plans(
    problem: Mapping[str, Any],
    *,
    reference_kind: ReferenceKind,
    public_reference: Callable[[str], str],
) -> tuple[int, tuple[CartLines, ...]]:
    names = _reference_fields(reference_kind)
    _offers, requirements, _relations, _terms = _problem_inventory(
        problem,
        reference_kind=reference_kind,
    )
    factors: list[tuple[tuple[str, int], ...]] = []
    count = 1
    for row in requirements:
        qty = int(row["required_qty"])
        choices = tuple((str(sku), qty) for sku in row[names["eligible"]])
        count *= len(choices)
        if count > T5_CART_ENUMERATION_LIMIT_V1:
            raise T5CartPlanningError("T5 cart enumeration exceeds 4096 plans")
        factors.append(choices)
    plans = tuple(
        _canonical_lines(choice, public_reference=public_reference)
        for choice in itertools.product(*factors)
    )
    if len(plans) != len(set(plans)):
        raise T5CartPlanningError("declared cart choices produce duplicate plans")
    return count, plans


def _active_tier(
    tiers: tuple[Mapping[str, Any], ...],
    quantity: int,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in tiers
        if quantity >= int(row["minimum_quantity"])
        and (row["maximum_quantity"] is None or quantity <= int(row["maximum_quantity"]))
    ]
    if len(matches) != 1:
        raise T5CartPlanningError("selected quantity has no unique public tier")
    return matches[0]


def _price_lines(
    lines: CartLines,
    *,
    problem: Mapping[str, Any],
    reference_kind: ReferenceKind,
    offers: Mapping[str, Mapping[str, Any]],
    pricing_terms: tuple[Mapping[str, Any], ...],
) -> tuple[int, int, int]:
    names = _reference_fields(reference_kind)
    quantities = dict(lines)
    if any(sku not in offers for sku in quantities):
        raise T5CartPlanningError("selected cart contains an unknown listing")
    subtotal = 0
    charges_total = 0
    for term in pricing_terms:
        selected = tuple(sorted(sku for sku in term[names["skus"]] if sku in quantities))
        if not selected:
            continue
        total_qty = sum(quantities[sku] for sku in selected)
        tier = _active_tier(
            _rows(term["quantity_tiers"], "quantity_tiers", nonempty=True),
            total_qty,
        )
        unit_price = int(tier["unit_price_minor"])
        base_subtotal = unit_price * total_qty
        satisfied_discounts: list[int] = []
        for bundle in term["bundle_discounts"]:
            if all(
                quantities.get(str(condition[names["sku"]]), 0)
                >= int(condition["minimum_quantity"])
                for condition in bundle["conditions"]
            ):
                value = (
                    int(bundle["discount_minor"])
                    if bundle["discount_minor"] is not None
                    else base_subtotal * int(bundle["discount_bps"]) // 10_000
                )
                satisfied_discounts.append(value)
        if term["bundle_stacking"] == "best_only":
            discount = max(satisfied_discounts, default=0)
        else:
            discount = sum(satisfied_discounts)
        discount = min(base_subtotal, discount)

        remaining = discount
        for sku in selected:
            qty = quantities[sku]
            reduction = min(unit_price - 1, remaining // qty)
            remaining -= reduction * qty
        if remaining:
            raise T5CartPlanningError(
                "public bundle discount cannot form integer World unit prices"
            )
        subtotal += base_subtotal - discount
        for charge in term["charges"]:
            upper = charge["maximum_subtotal_minor"]
            if base_subtotal < int(charge["minimum_subtotal_minor"]) or (
                upper is not None and base_subtotal >= int(upper)
            ):
                continue
            charges_total += (
                int(charge["fixed_minor"])
                + int(charge["per_unit_minor"]) * total_qty
                + base_subtotal * int(charge["subtotal_rate_bps"]) // 10_000
            )
    return subtotal, charges_total, subtotal + charges_total


def _evaluate_t5_cart_lines_v1(
    problem: Mapping[str, Any],
    lines: Sequence[tuple[str, int]],
    *,
    reference_kind: ReferenceKind,
    public_reference: Callable[[str], str],
    inventory: tuple[
        dict[str, Mapping[str, Any]],
        tuple[Mapping[str, Any], ...],
        tuple[Mapping[str, Any], ...],
        tuple[Mapping[str, Any], ...],
    ]
    | None = None,
    declared_plans: frozenset[CartLines] | None = None,
) -> T5CartPlanEvaluationV1:
    names = _reference_fields(reference_kind)
    offers, requirements, relations, pricing_terms = inventory or _problem_inventory(
        problem, reference_kind=reference_kind
    )
    try:
        canonical = _canonical_lines(lines, public_reference=public_reference)
    except T5CartPlanningError:
        canonical = tuple(lines)
    quantities = dict(canonical) if len(canonical) == len(set(canonical)) else {}
    if declared_plans is None:
        _count, declared = _declared_plans(
            problem,
            reference_kind=reference_kind,
            public_reference=public_reference,
        )
        declared_plans = frozenset(declared)

    requirement_ok = True
    for row in requirements:
        eligible = tuple(str(value) for value in row[names["eligible"]])
        selected = [(sku, quantities.get(sku, 0)) for sku in eligible if sku in quantities]
        if len(selected) != 1 or selected[0][1] != int(row["required_qty"]):
            requirement_ok = False
            break

    relation_ok = True
    for row in relations:
        if row["kind"] == "required_with":
            trigger = str(row[names["trigger"]])
            required = str(row[names["required"]])
            if quantities.get(trigger, 0) and quantities.get(required, 0) < int(row["minimum_qty"]):
                relation_ok = False
                break
        else:
            selected = [quantities.get(str(sku), 0) > 0 for sku in row[names["skus"]]]
            if any(selected) and not all(selected):
                relation_ok = False
                break

    inventory_ok = bool(quantities) and all(
        sku in offers and qty <= int(offers[sku]["available_qty"])
        for sku, qty in quantities.items()
    )
    max_delivery = (
        max(int(offers[sku]["delivery_days"]) for sku in quantities)
        if quantities and all(sku in offers for sku in quantities)
        else None
    )
    delivery_ok = bool(
        max_delivery is not None
        and max_delivery <= int(problem["hard_constraints"]["max_delivery_days"])
    )

    subtotal: int | None = None
    charge_total: int | None = None
    grand_total: int | None = None
    merchant_count: int | None = None
    try:
        subtotal, charge_total, grand_total = _price_lines(
            canonical,
            problem=problem,
            reference_kind=reference_kind,
            offers=offers,
            pricing_terms=pricing_terms,
        )
        merchant_count = len({str(offers[sku][names["merchant"]]) for sku in quantities})
    except (KeyError, T5CartPlanningError):
        pass
    budget_ok = bool(
        grand_total is not None and grand_total <= int(problem["hard_constraints"]["budget_minor"])
    )
    results = {
        "declared_discrete_plan": canonical in declared_plans,
        "exact_requirement_quantities": requirement_ok,
        "declared_relations": relation_ok,
        "inventory_available": inventory_ok,
        "delivery_deadline": delivery_ok,
        "fee_inclusive_budget": budget_ok,
    }
    objective_key = (
        (
            grand_total,
            max_delivery,
            merchant_count,
            tuple((public_reference(sku), qty) for sku, qty in canonical),
        )
        if grand_total is not None and max_delivery is not None and merchant_count is not None
        else None
    )
    return T5CartPlanEvaluationV1(
        lines=canonical,
        constraint_results=results,
        subtotal_minor=subtotal,
        charge_total_minor=charge_total,
        grand_total_minor=grand_total,
        max_delivery_days=max_delivery,
        merchant_count=merchant_count,
        objective_key=objective_key,
    )


def evaluate_t5_cart_lines_v1(
    problem: Mapping[str, Any],
    lines: Sequence[tuple[str, int]],
    *,
    reference_kind: ReferenceKind,
    public_reference: Callable[[str], str] | None = None,
) -> T5CartPlanEvaluationV1:
    """Evaluate arbitrary model-selected lines against public T5 facts."""

    public_reference = public_reference or (lambda value: value)
    return _evaluate_t5_cart_lines_v1(
        problem,
        lines,
        reference_kind=reference_kind,
        public_reference=public_reference,
    )


def build_t5_cart_oracle_v1(
    problem: Mapping[str, Any],
    *,
    reference_kind: ReferenceKind,
    public_reference: Callable[[str], str] | None = None,
) -> T5CartOracleV1:
    """Enumerate and rank the declared public discrete cart plans."""

    public_reference = public_reference or (lambda value: value)
    count, plans = _declared_plans(
        problem,
        reference_kind=reference_kind,
        public_reference=public_reference,
    )
    inventory = _problem_inventory(problem, reference_kind=reference_kind)
    declared_plans = frozenset(plans)
    evaluations = tuple(
        _evaluate_t5_cart_lines_v1(
            problem,
            lines,
            reference_kind=reference_kind,
            public_reference=public_reference,
            inventory=inventory,
            declared_plans=declared_plans,
        )
        for lines in plans
    )
    feasible = tuple(
        sorted(
            (row for row in evaluations if row.hard_feasible),
            key=lambda row: row.objective_key,
        )
    )
    if len(feasible) < 2:
        raise T5CartPlanningError("T5 Buyer task needs at least two executable plans")
    best = feasible[0].objective_key
    acceptable = tuple(row.lines for row in feasible if row.objective_key == best)
    if len(acceptable) != 1:
        raise T5CartPlanningError("T5 public objective does not have a unique optimum")
    return T5CartOracleV1(
        enumeration_count=count,
        declared_plans=plans,
        feasible_plans=feasible,
        acceptable_plans=acceptable,
    )


def public_t5_problem_copy_v1(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strict JSON-compatible defensive copy for task projections."""

    return copy.deepcopy(dict(problem))


__all__ = [
    "CartLines",
    "T5_CART_ENUMERATION_LIMIT_V1",
    "T5_CART_PLANNING_RULE_SET_V1",
    "T5_CART_PLANNING_SCHEMA_V1",
    "T5CartOracleV1",
    "T5CartPlanEvaluationV1",
    "T5CartPlanningError",
    "build_t5_cart_oracle_v1",
    "evaluate_t5_cart_lines_v1",
    "public_t5_problem_copy_v1",
]
