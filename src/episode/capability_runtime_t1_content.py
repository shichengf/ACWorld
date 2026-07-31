"""Frozen T1 problem content for the ACWorld runtime benchmark.

This is an execution-neutral task-data module.  It contains no simulator,
harness, policy, scorer, or trajectory code.  Formal T1 execution imports this
module so no historical direct-simulation family module enters its call stack.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from episode.capability_benchmark import TASK_REGISTRY_V2, TaskDefinitionV2


ConstraintOperatorT1 = Literal["at_most", "at_least", "equals"]
SelectionModeT1 = Literal["any_feasible", "best_feasible", "abstain_if_none"]


T1_PUBLIC_OBJECTIVE_WEIGHTS: Mapping[str, int] = MappingProxyType(
    {
        "quality_score": 1_000,
        "price_cents": -1,
        "shipping_days": -100,
        "warranty_months": 10,
    }
)
T1_PUBLIC_TIE_BREAK_FIELD = "sku_ref"


def _sha256(value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProductCandidateT1:
    sku_id: str
    name: str
    price_cents: int
    shipping_days: int
    warranty_months: int
    return_days: int
    energy_score: int
    quality_score: int
    in_stock: bool = True
    features: tuple[str, ...] = ()

    def value(self, field: str) -> int | bool:
        if field not in {
            "price_cents",
            "shipping_days",
            "warranty_months",
            "return_days",
            "energy_score",
            "in_stock",
        }:
            raise KeyError(f"unsupported T1 constraint field: {field!r}")
        value = getattr(self, field)
        if not isinstance(value, (int, bool)):
            raise TypeError(f"candidate field {field!r} is not scalar")
        return value


@dataclass(frozen=True)
class HardConstraintT1:
    constraint_id: str
    field: str
    operator: ConstraintOperatorT1
    value: int | bool
    description: str

    def satisfied_by(self, candidate: ProductCandidateT1) -> bool:
        observed = candidate.value(self.field)
        if self.operator == "at_most":
            return int(observed) <= int(self.value)
        if self.operator == "at_least":
            return int(observed) >= int(self.value)
        return observed == self.value


@dataclass(frozen=True)
class RuntimeTaskContentT1:
    task_id: str
    task_version: str
    capability_id: str
    difficulty_rank: int
    prompt: str
    candidates: tuple[ProductCandidateT1, ...]
    hard_constraints: tuple[HardConstraintT1, ...]
    selection_mode: SelectionModeT1
    soft_preferences: tuple[str, ...] = ()
    required_search_rounds: int = 0

    @property
    def content_sha256(self) -> str:
        return _sha256(asdict(self))

    def candidate(self, sku_id: str) -> ProductCandidateT1 | None:
        return next((row for row in self.candidates if row.sku_id == sku_id), None)

    def feasible(self, candidate: ProductCandidateT1) -> bool:
        return candidate.in_stock and all(
            constraint.satisfied_by(candidate) for constraint in self.hard_constraints
        )

    def utility(self, candidate: ProductCandidateT1) -> int:
        return sum(
            int(getattr(candidate, field)) * weight
            for field, weight in T1_PUBLIC_OBJECTIVE_WEIGHTS.items()
        )

    @property
    def feasible_skus(self) -> tuple[str, ...]:
        """All hard-constraint-satisfying business choices, independent of order."""

        return tuple(sorted(row.sku_id for row in self.candidates if self.feasible(row)))

    @property
    def acceptable_skus(self) -> tuple[str, ...]:
        """Derive the oracle set from fixture facts instead of a stored answer."""

        feasible = tuple(row for row in self.candidates if self.feasible(row))
        if self.selection_mode == "any_feasible":
            return tuple(sorted(row.sku_id for row in feasible))
        if self.selection_mode == "abstain_if_none":
            return ()
        if not feasible:
            return ()
        best_utility = max(self.utility(row) for row in feasible)
        # Fixtures reject objective ties below.  Runtime still applies the public
        # sku_ref tie-break, so a later malformed fixture cannot silently rely on
        # tuple position or an internal expected answer.
        return tuple(sorted(row.sku_id for row in feasible if self.utility(row) == best_utility))


def _constraint_set(count: int) -> tuple[HardConstraintT1, ...]:
    rows = (
        HardConstraintT1("c1", "price_cents", "at_most", 12_000, "price at most 12000 cents"),
        HardConstraintT1("c2", "shipping_days", "at_most", 5, "delivery within 5 days"),
        HardConstraintT1("c3", "warranty_months", "at_least", 18, "warranty at least 18 months"),
        HardConstraintT1("c4", "return_days", "at_least", 30, "return window at least 30 days"),
        HardConstraintT1("c5", "energy_score", "at_least", 4, "energy score at least 4"),
    )
    return rows[:count]


def _candidate(
    task_id: str,
    ordinal: int,
    *,
    price_cents: int = 9_000,
    shipping_days: int = 3,
    warranty_months: int = 24,
    return_days: int = 45,
    energy_score: int = 5,
    quality_score: int = 80,
    in_stock: bool = True,
    features: tuple[str, ...] = (),
) -> ProductCandidateT1:
    stem = task_id.casefold().replace("cwv2-", "").replace("-", "")
    return ProductCandidateT1(
        sku_id=f"merchant:benchmark:{stem}:p{ordinal:02d}",
        name=f"Catalog item {ordinal:02d}",
        price_cents=price_cents,
        shipping_days=shipping_days,
        warranty_months=warranty_months,
        return_days=return_days,
        energy_score=energy_score,
        quality_score=quality_score,
        in_stock=in_stock,
        features=features,
    )


_UTILITY_TEXT = (
    "Among eligible items, maximize quality_score*1000 - price_cents - "
    "shipping_days*100 + warranty_months*10; break an exact tie by the "
    "ascending public sku_ref."
)


def _base(definition: TaskDefinitionV2, **values: Any) -> RuntimeTaskContentT1:
    return RuntimeTaskContentT1(
        task_id=definition.task_id,
        task_version=definition.task_version,
        capability_id=definition.capability_id,
        difficulty_rank=definition.difficulty_rank,
        **values,
    )


def _build_basic(definition: TaskDefinitionV2) -> RuntimeTaskContentT1:
    count = int(dict(definition.difficulty_factors)["candidate_count"])
    constraints = (
        HardConstraintT1("c1", "price_cents", "at_most", 10_000, "price at most 10000 cents"),
        HardConstraintT1("c2", "shipping_days", "at_most", 5, "delivery within 5 days"),
    )
    candidates = tuple(
        _candidate(
            definition.task_id,
            index + 1,
            price_cents=(8_600 + index * 90 if index % 4 != 1 else 11_000 + index),
            shipping_days=2 + index % 3 if index % 4 != 2 else 8,
            quality_score=92 - index,
            in_stock=index % 4 != 3,
        )
        for index in range(count)
    )
    return _base(
        definition,
        prompt=(
            "Choose one in-stock catalog item satisfying every hard requirement. "
            "Select a sku, or abstain only if none qualify."
        ),
        candidates=candidates,
        hard_constraints=constraints,
        selection_mode="any_feasible",
    )


def _build_hard_constraints(definition: TaskDefinitionV2) -> RuntimeTaskContentT1:
    count = int(dict(definition.difficulty_factors)["hard_constraint_count"])
    constraints = _constraint_set(count)
    candidates = [
        _candidate(definition.task_id, 1, quality_score=96, price_cents=9_300),
        _candidate(definition.task_id, 2, quality_score=84, price_cents=8_700),
    ]
    violations: tuple[tuple[str, int], ...] = (
        ("price_cents", 12_001),
        ("shipping_days", 6),
        ("warranty_months", 17),
        ("return_days", 29),
        ("energy_score", 3),
    )
    for index, (field, value) in enumerate(violations[:count], start=3):
        candidates.append(
            replace(_candidate(definition.task_id, index, quality_score=99), **{field: value})
        )
    for index in range(count):
        candidates.append(
            _candidate(
                definition.task_id,
                len(candidates) + 1,
                price_cents=9_600 + index * 50,
                quality_score=78 - index,
            )
        )
    return _base(
        definition,
        prompt=(
            "Filter every hard requirement, then choose any eligible item. "
            "Every item satisfying all hard constraints is equally acceptable; "
            "abstain only if none qualify."
        ),
        candidates=tuple(candidates),
        hard_constraints=constraints,
        selection_mode="any_feasible",
    )


def _build_best(definition: TaskDefinitionV2) -> RuntimeTaskContentT1:
    count = int(dict(definition.difficulty_factors)["candidate_count"])
    constraints = _constraint_set(3)
    candidates = tuple(
        _candidate(
            definition.task_id,
            index + 1,
            price_cents=7_800 + ((index * 977 + count * 31) % 3_100),
            shipping_days=1 + ((index * 3 + count) % 5),
            warranty_months=18 + ((index * 7 + count) % 19),
            quality_score=70 + ((index * 11 + count * 3) % 27),
        )
        for index in range(count)
    )
    candidates = _place_unique_best_profile(
        definition.task_id,
        candidates,
        constraints,
    )
    return _base(
        definition,
        prompt=("Keep every hard requirement and choose the best available item. " + _UTILITY_TEXT),
        candidates=candidates,
        hard_constraints=constraints,
        selection_mode="best_feasible",
    )


def _utility(row: ProductCandidateT1) -> int:
    return (
        row.quality_score * 1_000
        - row.price_cents
        - row.shipping_days * 100
        + row.warranty_months * 10
    )


def _build_abstention(definition: TaskDefinitionV2) -> RuntimeTaskContentT1:
    subtlety = int(dict(definition.difficulty_factors)["conflict_subtlety"])
    constraints = (
        HardConstraintT1("c1", "price_cents", "at_most", 10_000, "price at most 10000 cents"),
        HardConstraintT1("c2", "warranty_months", "at_least", 18, "warranty at least 18 months"),
    )
    delta = {1: 5_000, 2: 1_000, 3: 100, 4: 1}[subtlety]
    candidates = tuple(
        _candidate(
            definition.task_id,
            index + 1,
            price_cents=10_000 + delta + index,
            warranty_months=24 + index,
            quality_score=88 + index,
        )
        for index in range(5)
    )
    return _base(
        definition,
        prompt=(
            "Check every item against all hard requirements. Select only if one "
            "satisfies all of them; otherwise abstain."
        ),
        candidates=candidates,
        hard_constraints=constraints,
        selection_mode="abstain_if_none",
    )


def _build_query(definition: TaskDefinitionV2) -> RuntimeTaskContentT1:
    rounds = int(dict(definition.difficulty_factors)["search_rounds"])
    constraints = (
        HardConstraintT1("c1", "price_cents", "at_most", 14_000, "price at most 14000 cents"),
        HardConstraintT1("c2", "warranty_months", "at_least", 12, "warranty at least 12 months"),
    )
    preferences = (
        "same_day_delivery",
        "local_pickup",
        "plastic_free_packaging",
        "matte_finish",
    )[: rounds - 1]
    candidates = [
        _candidate(definition.task_id, 1, price_cents=10_800, quality_score=96),
        _candidate(definition.task_id, 2, price_cents=9_100, quality_score=82),
    ]
    for index, _feature in enumerate(preferences, start=3):
        candidates.append(
            _candidate(
                definition.task_id,
                index,
                price_cents=12_500 + index,
                warranty_months=6,
                quality_score=99,
                features=preferences,
            )
        )
    candidates = list(
        _place_unique_best_profile(
            definition.task_id,
            tuple(candidates),
            constraints,
        )
    )
    return _base(
        definition,
        prompt=(
            "Search without weakening either hard requirement. Start with all "
            "optional filters, then remove exactly the lowest-priority filter "
            "after each empty result. Choose the best eligible result. " + _UTILITY_TEXT
        ),
        candidates=tuple(candidates),
        hard_constraints=constraints,
        selection_mode="best_feasible",
        soft_preferences=preferences,
        required_search_rounds=rounds,
    )


_BUILDERS = {
    "t1.basic_feasible_discovery": _build_basic,
    "t1.hard_constraint_filtering": _build_hard_constraints,
    "t1.best_feasible_selection": _build_best,
    "t1.correct_abstention": _build_abstention,
    "t1.query_reformulation": _build_query,
}


# The only T1 tasks with a unique product answer are the objective-bearing
# best-selection and query-reformulation tasks.  Their public fact profiles are
# deliberately assigned across the ranked business-reference positions.  This
# is fixture construction, not a scorer exception: the oracle is recomputed
# from the resulting public facts after placement.
_OBJECTIVE_TARGET_POSITION_BY_TASK: Mapping[str, int] = MappingProxyType(
    {
        "CWV2-T01-08": 1,
        "CWV2-T01-09": 3,
        "CWV2-T01-10": 6,
        "CWV2-T01-11": 11,
        "CWV2-T01-12": 19,
        "CWV2-T01-17": 1,
        "CWV2-T01-18": 2,
        "CWV2-T01-19": 4,
        "CWV2-T01-20": 6,
    }
)


def _profile_at_slot(
    slot: ProductCandidateT1,
    profile: ProductCandidateT1,
) -> ProductCandidateT1:
    """Assign public product facts to a stable business-reference slot."""

    return replace(
        slot,
        price_cents=profile.price_cents,
        shipping_days=profile.shipping_days,
        warranty_months=profile.warranty_months,
        return_days=profile.return_days,
        energy_score=profile.energy_score,
        quality_score=profile.quality_score,
        in_stock=profile.in_stock,
        features=profile.features,
    )


def _place_unique_best_profile(
    task_id: str,
    candidates: Sequence[ProductCandidateT1],
    constraints: Sequence[HardConstraintT1],
) -> tuple[ProductCandidateT1, ...]:
    """Place the unique optimum's facts at the task's balanced public position."""

    rows = tuple(sorted(candidates, key=lambda row: row.sku_id))
    feasible_indices = [
        index
        for index, row in enumerate(rows)
        if row.in_stock and all(rule.satisfied_by(row) for rule in constraints)
    ]
    if not feasible_indices:
        raise ValueError(f"T1 objective fixture {task_id} has no feasible candidate")
    best = max(_utility(rows[index]) for index in feasible_indices)
    best_indices = [index for index in feasible_indices if _utility(rows[index]) == best]
    if len(best_indices) != 1:
        raise ValueError(f"T1 objective fixture {task_id} must have a unique optimum")
    source_index = best_indices[0]
    target_index = _OBJECTIVE_TARGET_POSITION_BY_TASK[task_id] - 1
    if not 0 <= target_index < len(rows):
        raise ValueError(f"T1 objective target position is outside {task_id}'s catalog")
    if source_index == target_index:
        return rows
    output = list(rows)
    output[source_index] = _profile_at_slot(rows[source_index], rows[target_index])
    output[target_index] = _profile_at_slot(rows[target_index], rows[source_index])
    return tuple(output)


def validate_runtime_task_content_t1(
    tasks: Sequence[RuntimeTaskContentT1],
) -> None:
    """Fail construction when T1 facts, oracle semantics, or positions drift."""

    objective_positions: list[tuple[int, int]] = []
    for task in tasks:
        sku_ids = tuple(row.sku_id for row in task.candidates)
        if len(sku_ids) != len(set(sku_ids)):
            raise ValueError(f"T1 fixture {task.task_id} has duplicate SKU identities")
        feasible = task.feasible_skus
        acceptable = task.acceptable_skus
        reordered = replace(task, candidates=tuple(reversed(task.candidates)))
        if reordered.feasible_skus != feasible or reordered.acceptable_skus != acceptable:
            raise ValueError(f"T1 fixture {task.task_id} oracle depends on candidate order")
        if task.selection_mode == "any_feasible":
            if not feasible or acceptable != feasible:
                raise ValueError(
                    f"T1 discovery/filter fixture {task.task_id} must accept every feasible SKU"
                )
        elif task.selection_mode == "best_feasible":
            if len(acceptable) != 1:
                raise ValueError(
                    f"T1 objective fixture {task.task_id} must have one public optimum"
                )
            ranked = sorted(task.candidates, key=lambda row: row.sku_id)
            position = next(
                index for index, row in enumerate(ranked, start=1) if row.sku_id == acceptable[0]
            )
            objective_positions.append((position, len(ranked)))
        elif task.selection_mode == "abstain_if_none":
            if feasible or acceptable:
                raise ValueError(
                    f"T1 abstention fixture {task.task_id} unexpectedly has a feasible SKU"
                )
        else:
            raise ValueError(f"T1 fixture {task.task_id} has an unknown selection mode")

    buckets = [min(3, ((position - 1) * 4) // count) for position, count in objective_positions]
    bucket_counts = [buckets.count(index) for index in range(4)]
    if max(bucket_counts) - min(bucket_counts) > 1:
        raise ValueError(
            "T1 unique-answer positions are not balanced across ranked-list quartiles: "
            f"{bucket_counts}"
        )


def build_runtime_task_content_t1() -> tuple[RuntimeTaskContentT1, ...]:
    definitions = (
        definition for definition in TASK_REGISTRY_V2.values() if definition.family.value == "T1"
    )
    tasks = tuple(_BUILDERS[item.capability_id](item) for item in definitions)
    validate_runtime_task_content_t1(tasks)
    return tasks


T1_RUNTIME_TASKS: Mapping[str, RuntimeTaskContentT1] = MappingProxyType(
    {task.task_id: task for task in build_runtime_task_content_t1()}
)


__all__ = [
    "HardConstraintT1",
    "ProductCandidateT1",
    "RuntimeTaskContentT1",
    "SelectionModeT1",
    "T1_PUBLIC_OBJECTIVE_WEIGHTS",
    "T1_PUBLIC_TIE_BREAK_FIELD",
    "T1_RUNTIME_TASKS",
    "build_runtime_task_content_t1",
    "validate_runtime_task_content_t1",
]
