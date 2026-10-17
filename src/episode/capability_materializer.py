"""Deterministic, fail-closed materialization for the ACWorld capability benchmark.

The v2 registry is deliberately seed-free and describes *what* a task tests.
This module binds one :class:`TaskDefinitionV2` to a reviewed v1 scenario,
turns it into an explicit many-participant :class:`ScenarioSpec`, and exposes
only a seed-free public manifest.  The integer stored in ``ScenarioSpec.seed``
is a private compatibility detail required by the existing runtime; it is
derived from immutable task content and is never serialized publicly.

Materialization is conservative.  Existing catalog rows, authoritative state,
and the complete success oracle are copied without mutation.  Population and
catalog additions are deterministic distractors: they are explicitly out of
stock, over budget, and marked as benchmark distractors, so they cannot change
the reviewed template's answer.  If a smaller requested population cannot
retain every actor referenced by authoritative state or the oracle, generation
fails instead of silently producing an incoherent benchmark.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from episode.benchmark import BenchmarkTrack, VARIANT_REGISTRY
from episode.capability_benchmark import TASK_REGISTRY_V2, TaskDefinitionV2
from episode.errors import ScenarioInvalid
from episode.scenario import (
    from_yaml,
    kickoff_envelopes,
    population_for_scenario,
    seed_world,
)
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_LISTING_FACTOR_ABSOLUTE = frozenset({
    "candidate_count",
    "cart_line_count",
    "comparison_count",
    "merchant_count",
    "requested_sku_count",
    "sponsored_listing_count",
})
_LISTING_FACTOR_RELATIONAL = frozenset({
    "bundle_relation_count",
    "substitute_count",
})
_DROP = object()


class MaterializationError(ValueError):
    """A task cannot be mapped to a self-consistent reviewed scenario."""


def _public_scalar(value: Any, *, omit_seed: bool = False) -> Any:
    """Return canonical JSON data, optionally omitting every ``seed`` key."""

    if isinstance(value, Enum):
        return _public_scalar(value.value, omit_seed=omit_seed)
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return {
            field.name: _public_scalar(getattr(value, field.name), omit_seed=omit_seed)
            for field in fields(value)
            if not (omit_seed and field.name.casefold() == "seed")
        }
    if isinstance(value, Mapping):
        return {
            str(key): _public_scalar(item, omit_seed=omit_seed)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not (omit_seed and str(key).casefold() == "seed")
        }
    if isinstance(value, (tuple, list)):
        return [_public_scalar(item, omit_seed=omit_seed) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_public_scalar(item, omit_seed=omit_seed) for item in value),
            key=repr,
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def _canonical_json(value: Any, *, omit_seed: bool = False) -> str:
    return json.dumps(
        _public_scalar(value, omit_seed=omit_seed),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any, *, omit_seed: bool = False) -> str:
    return hashlib.sha256(
        _canonical_json(value, omit_seed=omit_seed).encode("utf-8")
    ).hexdigest()


def _contains_seed_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() == "seed" or _contains_seed_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_seed_key(item) for item in value)
    return False


@dataclass(frozen=True)
class MaterializedTaskV2:
    """One v2 task bound to an executable, explicit-population scenario."""

    task: TaskDefinitionV2
    scenario: ScenarioSpec
    source_scenario_id: str
    evaluated_actor_id: str
    determinism_key: str
    content_sha256: str

    def to_public_dict(self) -> dict[str, Any]:
        """Return the stable public identity; never expose the runtime seed."""

        return {
            "schema_version": "cwe.materialized-task.v2",
            "task": self.task.to_dict(),
            "scenario": {
                "scenario_id": self.scenario.scenario_id,
                "source_scenario_id": self.source_scenario_id,
                "evaluated_actor_id": self.evaluated_actor_id,
                "population": {
                    "buyers": self.task.buyers,
                    "merchants": self.task.merchants,
                },
                "content_sha256": self.content_sha256,
            },
            "determinism_key": self.determinism_key,
        }


def _scenario_path(repository_root: Path, variant_id: str) -> Path:
    if not variant_id.startswith("S") or not variant_id[1:].isdigit():
        raise MaterializationError(f"invalid template variant id {variant_id!r}")
    number = int(variant_id[1:])
    candidates = sorted((repository_root / "scenarios").glob(f"s{number}_*_42.yaml"))
    if len(candidates) != 1:
        rendered = ", ".join(path.name for path in candidates) or "none"
        raise MaterializationError(
            f"{variant_id} must have exactly one reviewed canonical template; "
            f"found {rendered}"
        )
    return candidates[0]


def _internal_key(task: TaskDefinitionV2, source: ScenarioSpec) -> str:
    return _sha256({
        "task": task.to_dict(),
        "source_scenario_id": source.scenario_id,
        "source_oracle": source.success_oracle,
    })


def _compatibility_seed(determinism_key: str) -> int:
    # Preserve a positive signed-32-bit value for existing catalog/runtime APIs.
    return int(determinism_key[:8], 16) % 2_147_483_646 + 1


def _scenario_id(task: TaskDefinitionV2, source: ScenarioSpec) -> str:
    source_stem = source.scenario_id.rsplit("_", 1)[0]
    return f"{task.task_id.lower().replace('-', '_')}__{source_stem}"


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _strings(item)


def _referenced_actors(
    source: ScenarioSpec,
    population: PopulationSpec,
    catalog_owner: Mapping[str, str],
) -> tuple[set[str], set[str]]:
    """Find actors that cannot be removed without changing authoritative data."""

    buyer_ids = {buyer.buyer_id for buyer in population.buyers}
    merchant_ids = {merchant.merchant_id for merchant in population.merchants}
    required_buyers: set[str] = set()
    required_merchants: set[str] = set()

    # Oracle actor/listing references are part of the answer key.
    for text in _strings(source.success_oracle):
        if text in buyer_ids:
            required_buyers.add(text)
        if text in merchant_ids:
            required_merchants.add(text)
        owner = catalog_owner.get(text)
        if owner is not None:
            required_merchants.add(owner)

    # Seeded orders/ledger are authoritative and may not be discarded.
    for table in ("orders", "ledger", "order_timelines"):
        for row in source.initial_state.get(table, ()) or ():
            if not isinstance(row, Mapping):
                continue
            buyer_id = str(row.get("buyer_id", ""))
            merchant_id = str(row.get("merchant_id", ""))
            if buyer_id in buyer_ids:
                required_buyers.add(buyer_id)
            if merchant_id in merchant_ids:
                required_merchants.add(merchant_id)

    # Directly addressed kickoff actors must remain.  Actor mentions nested in
    # candidate payloads are filterable distractors and are handled separately.
    for event in population.initial_events:
        for key in ("from", "from_", "to"):
            actor_id = str(event.get(key, ""))
            if actor_id in buyer_ids:
                required_buyers.add(actor_id)
            if actor_id in merchant_ids:
                required_merchants.add(actor_id)
    return required_buyers, required_merchants


def _choose_existing_ids(
    available: Sequence[str],
    required: set[str],
    desired: int,
    *,
    evaluated_primary: str | None,
    role: str,
) -> list[str]:
    must_keep = set(required)
    if evaluated_primary is not None:
        must_keep.add(evaluated_primary)
    unknown = must_keep - set(available)
    if unknown:
        raise MaterializationError(
            f"{role} answer key references undeclared actor(s): {sorted(unknown)!r}"
        )
    if len(must_keep) > desired:
        raise MaterializationError(
            f"requested {desired} {role}(s), but authoritative state requires "
            f"{sorted(must_keep)!r}"
        )
    chosen = sorted(must_keep)
    chosen.extend(item for item in sorted(available) if item not in must_keep)
    return chosen[:desired]


def _new_actor_id(role: str, occupied: set[str], ordinal: int) -> str:
    index = ordinal
    while True:
        candidate = f"{role}:v2{role[0]}{index}"
        if candidate not in occupied:
            return candidate
        index += 1


def _filter_event_payload(
    value: Any,
    *,
    allowed_merchants: set[str],
    allowed_skus: set[str],
) -> Any:
    """Remove candidate rows owned by actors removed from a smaller topology."""

    if isinstance(value, Mapping):
        merchant_id = value.get("merchant_id")
        sku_id = value.get("sku_id")
        if merchant_id is not None and str(merchant_id).startswith("merchant:"):
            if str(merchant_id) not in allowed_merchants:
                return _DROP
        if sku_id is not None and str(sku_id) not in allowed_skus:
            return _DROP
        result: dict[str, Any] = {}
        for key, item in value.items():
            filtered = _filter_event_payload(
                item,
                allowed_merchants=allowed_merchants,
                allowed_skus=allowed_skus,
            )
            if filtered is not _DROP:
                result[str(key)] = filtered
        return result
    if isinstance(value, (tuple, list)):
        result_list: list[Any] = []
        for item in value:
            filtered = _filter_event_payload(
                item,
                allowed_merchants=allowed_merchants,
                allowed_skus=allowed_skus,
            )
            if filtered is not _DROP:
                result_list.append(filtered)
        return result_list
    return copy.deepcopy(value)


def _listing_target(task: TaskDefinitionV2, current: int) -> int:
    target = max(current, task.merchants)
    for key, raw_value in task.difficulty_factors:
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            continue
        if key in _LISTING_FACTOR_ABSOLUTE:
            target = max(target, raw_value)
        elif key in _LISTING_FACTOR_RELATIONAL:
            # A relation/substitute is in addition to the reviewed target row.
            target = max(target, raw_value + 1)
    return target


def _distractor_kind(task: TaskDefinitionV2) -> str:
    keys = {key for key, _ in task.difficulty_factors}
    for key in (
        "substitute_count",
        "bundle_relation_count",
        "cart_line_count",
        "candidate_count",
        "comparison_count",
        "requested_sku_count",
        "sponsored_listing_count",
        "merchant_count",
    ):
        if key in keys:
            return key.removesuffix("_count")
    return "population"


def _max_budget_dollars(buyers: Sequence[BuyerSpec], source: ScenarioSpec) -> float:
    budgets: list[float] = []
    for buyer in buyers:
        hard = buyer.mandate.get("hard_constraints") or {}
        budget = hard.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool):
            budgets.append(float(budget) / 100.0)
    legacy_budget = source.buyer_goal.get("max_budget")
    if isinstance(legacy_budget, (int, float)) and not isinstance(legacy_budget, bool):
        budgets.append(float(legacy_budget))
    return max(budgets, default=100.0)


def _build_population_and_state(
    task: TaskDefinitionV2,
    source: ScenarioSpec,
) -> tuple[PopulationSpec, dict[str, Any], str]:
    base_population = population_for_scenario(source)
    base_buyers = {buyer.buyer_id: copy.deepcopy(buyer) for buyer in base_population.buyers}
    base_merchants = {
        merchant.merchant_id: copy.deepcopy(merchant)
        for merchant in base_population.merchants
    }
    if not base_buyers or not base_merchants:
        raise MaterializationError("canonical template has an empty population")

    source_catalog = copy.deepcopy(list(source.initial_state.get("catalog", ()) or ()))
    primary_merchant_id = sorted(base_merchants)[0]
    catalog_owner: dict[str, str] = {}
    for index, row in enumerate(source_catalog):
        if not isinstance(row, dict) or "sku_id" not in row:
            raise MaterializationError(f"catalog row {index} is not a listing mapping")
        sku = str(row["sku_id"])
        if sku in catalog_owner:
            raise MaterializationError(f"canonical template has duplicate sku_id {sku!r}")
        owner = str(row.get("merchant_id") or primary_merchant_id)
        if owner not in base_merchants:
            raise MaterializationError(
                f"canonical listing {sku!r} references undeclared merchant {owner!r}"
            )
        row["merchant_id"] = owner
        catalog_owner[sku] = owner

    required_buyers, required_merchants = _referenced_actors(
        source, base_population, catalog_owner
    )
    evaluated_buyer = sorted(base_buyers)[0] if task.evaluated_role == "buyer" else None
    evaluated_merchant = (
        sorted(base_merchants)[0] if task.evaluated_role == "merchant" else None
    )
    selected_buyer_ids = _choose_existing_ids(
        tuple(base_buyers),
        required_buyers,
        task.buyers,
        evaluated_primary=evaluated_buyer,
        role="buyer",
    )
    selected_merchant_ids = _choose_existing_ids(
        tuple(base_merchants),
        required_merchants,
        task.merchants,
        evaluated_primary=evaluated_merchant,
        role="merchant",
    )

    occupied = set(base_buyers) | set(base_merchants)
    while len(selected_buyer_ids) < task.buyers:
        actor_id = _new_actor_id("buyer", occupied, len(selected_buyer_ids) + 1)
        occupied.add(actor_id)
        selected_buyer_ids.append(actor_id)
    while len(selected_merchant_ids) < task.merchants:
        actor_id = _new_actor_id("merchant", occupied, len(selected_merchant_ids) + 1)
        occupied.add(actor_id)
        selected_merchant_ids.append(actor_id)

    buyer_prototype = base_buyers[sorted(base_buyers)[0]]
    buyers: list[BuyerSpec] = []
    for actor_id in selected_buyer_ids:
        source_buyer = base_buyers.get(actor_id, buyer_prototype)
        mandate = copy.deepcopy(source_buyer.mandate)
        if actor_id not in base_buyers:
            mandate["mandate_id"] = f"{task.task_id}:{actor_id}"
            mandate["benchmark_counterpart"] = True
        buyers.append(BuyerSpec(
            buyer_id=actor_id,
            persona=copy.deepcopy(source_buyer.persona),
            mandate=mandate,
            initial_state=copy.deepcopy(source_buyer.initial_state),
        ))

    merchant_prototype = base_merchants[sorted(base_merchants)[0]]
    merchants_without_scope: list[MerchantSpec] = []
    for actor_id in selected_merchant_ids:
        source_merchant = base_merchants.get(actor_id, merchant_prototype)
        policy = copy.deepcopy(source_merchant.policy)
        if actor_id not in base_merchants:
            policy["benchmark_counterpart"] = True
        merchants_without_scope.append(MerchantSpec(
            merchant_id=actor_id,
            persona=copy.deepcopy(source_merchant.persona),
            policy=policy,
            initial_state=copy.deepcopy(source_merchant.initial_state),
        ))

    selected_merchant_set = set(selected_merchant_ids)
    catalog = [
        row for row in source_catalog
        if str(row["merchant_id"]) in selected_merchant_set
    ]
    selected_skus = {str(row["sku_id"]) for row in catalog}
    for key in ("expected_sku", "cart_sku_1", "cart_sku_2"):
        target = source.success_oracle.get(key)
        if target is not None and str(target) not in selected_skus:
            raise MaterializationError(
                f"population reduction removed oracle target {key}={target!r}"
            )

    owners_with_listing = {str(row["merchant_id"]) for row in catalog}
    missing_owner_count = len(selected_merchant_set - owners_with_listing)
    listing_target = max(
        _listing_target(task, len(catalog)),
        len(catalog) + missing_owner_count,
    )
    high_price = max(
        _max_budget_dollars(buyers, source) + 100.0,
        max((float(row.get("list_price", 0)) for row in catalog), default=0.0) + 100.0,
    )
    kind = _distractor_kind(task)
    existing_skus = {str(row["sku_id"]) for row in catalog}
    ordinal = 1
    # Give newly introduced merchants a listing before adding further rows.
    owner_order = sorted(
        selected_merchant_ids,
        key=lambda actor_id: (
            any(str(row["merchant_id"]) == actor_id for row in catalog),
            actor_id,
        ),
    )
    while len(catalog) < listing_target:
        owner = owner_order[(ordinal - 1) % len(owner_order)]
        slug = task.task_id.lower().replace("-", ":")
        sku = f"{owner}:benchmark:{slug}:d{ordinal:02d}"
        while sku in existing_skus:
            ordinal += 1
            sku = f"{owner}:benchmark:{slug}:d{ordinal:02d}"
        catalog.append({
            "sku_id": sku,
            "merchant_id": owner,
            "product_id": f"benchmark:distractor:{slug}:{ordinal:02d}",
            "name": f"Ineligible benchmark distractor {ordinal}",
            "category": "benchmark_distractor",
            "list_price": high_price + ordinal,
            "floor_price": high_price,
            "inventory": 0,
            "attributes": {
                "in_stock": False,
                "shipping_days": 365,
                "benchmark_distractor": True,
                "distractor_kind": kind,
                "task_id": task.task_id,
                "difficulty_rank": task.difficulty_rank,
            },
        })
        existing_skus.add(sku)
        ordinal += 1

    scope_by_merchant: dict[str, list[str]] = {
        merchant_id: [] for merchant_id in selected_merchant_ids
    }
    for row in catalog:
        scope_by_merchant[str(row["merchant_id"])].append(str(row["sku_id"]))
    if any(not scope for scope in scope_by_merchant.values()):
        raise MaterializationError("every declared merchant must own at least one listing")

    merchants = tuple(MerchantSpec(
        merchant_id=merchant.merchant_id,
        persona=merchant.persona,
        policy=merchant.policy,
        catalog_scope=tuple(sorted(scope_by_merchant[merchant.merchant_id])),
        initial_state=merchant.initial_state,
    ) for merchant in merchants_without_scope)

    allowed_skus = set(existing_skus)
    events: list[dict[str, Any]] = []
    for event in base_population.initial_events:
        filtered = _filter_event_payload(
            event,
            allowed_merchants=selected_merchant_set,
            allowed_skus=allowed_skus,
        )
        if filtered is _DROP or not isinstance(filtered, dict):
            raise MaterializationError("initial event was removed during population filtering")
        target = str(filtered.get("to", ""))
        if target.startswith(("buyer", "merchant:")) and target not in {
            *selected_buyer_ids,
            *selected_merchant_ids,
        }:
            raise MaterializationError(
                f"initial event targets removed actor {target!r}"
            )
        events.append(filtered)

    initial_state = copy.deepcopy(source.initial_state)
    initial_state["catalog"] = catalog
    population = PopulationSpec(
        buyers=tuple(sorted(buyers, key=lambda buyer: buyer.buyer_id)),
        merchants=tuple(sorted(merchants, key=lambda merchant: merchant.merchant_id)),
        initial_events=tuple(events),
        matching={
            **copy.deepcopy(base_population.matching),
            "top_k": max(
                int(base_population.matching.get("top_k", 5)),
                min(listing_target, 20),
            ),
        },
        execution=copy.deepcopy(base_population.execution),
    )
    evaluated_actor_id = evaluated_buyer or evaluated_merchant
    if evaluated_actor_id is None:
        raise MaterializationError("task has no evaluated actor")
    return population, initial_state, evaluated_actor_id


def scenario_content_hash_v2(scenario: ScenarioSpec) -> str:
    """Hash executable scenario content while excluding the private seed field."""

    return _sha256(scenario, omit_seed=True)


def materialize(
    task: TaskDefinitionV2,
    *,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> MaterializedTaskV2:
    """Materialize one v2 definition from its same-family canonical template."""

    template = VARIANT_REGISTRY.get(task.template_variant_id)
    if template is None:
        raise MaterializationError(
            f"{task.task_id}: unknown template {task.template_variant_id!r}"
        )
    if template.track != BenchmarkTrack.AGENT or template.task_family != task.family:
        raise MaterializationError(
            f"{task.task_id}: template {task.template_variant_id} is not an agent "
            f"scenario in {task.family.value}"
        )

    root = Path(repository_root).resolve()
    source = from_yaml(_scenario_path(root, task.template_variant_id))
    # Work exclusively on detached values.  ``ScenarioSpec`` is frozen, but its
    # nested dictionaries are mutable and must never alias the source fixture.
    source_copy = copy.deepcopy(source)
    determinism_key = _internal_key(task, source_copy)
    population, initial_state, evaluated_actor_id = _build_population_and_state(
        task, source_copy
    )
    scenario = replace(
        source_copy,
        scenario_id=_scenario_id(task, source_copy),
        seed=_compatibility_seed(determinism_key),
        initial_state=initial_state,
        population=population,
    )
    materialized = MaterializedTaskV2(
        task=task,
        scenario=scenario,
        source_scenario_id=source.scenario_id,
        evaluated_actor_id=evaluated_actor_id,
        determinism_key=determinism_key,
        content_sha256=scenario_content_hash_v2(scenario),
    )
    issues = validate((materialized,), expected_tasks=1, repository_root=root)
    if issues:
        raise MaterializationError(
            f"{task.task_id}: invalid materialization: " + "; ".join(issues)
        )
    return materialized


def materialize_all(
    registry: Mapping[str, TaskDefinitionV2] | Iterable[TaskDefinitionV2] = TASK_REGISTRY_V2,
    *,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> tuple[MaterializedTaskV2, ...]:
    """Materialize a registry in canonical task-id order and validate it."""

    definitions = tuple(registry.values()) if isinstance(registry, Mapping) else tuple(registry)
    result = tuple(
        materialize(task, repository_root=repository_root)
        for task in sorted(definitions, key=lambda item: item.task_id)
    )
    issues = validate(
        result,
        expected_tasks=len(definitions),
        repository_root=repository_root,
    )
    if issues:
        raise MaterializationError("invalid materialized benchmark: " + "; ".join(issues))
    return result


def validate(
    materialized_tasks: Sequence[MaterializedTaskV2],
    *,
    expected_tasks: int = 200,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> tuple[str, ...]:
    """Return every structural or executable materialization problem."""

    issues: list[str] = []
    root = Path(repository_root).resolve()
    if len(materialized_tasks) != expected_tasks:
        issues.append(
            f"expected {expected_tasks} materialized task(s), got {len(materialized_tasks)}"
        )
    task_ids = [item.task.task_id for item in materialized_tasks]
    scenario_ids = [item.scenario.scenario_id for item in materialized_tasks]
    hashes = [item.content_sha256 for item in materialized_tasks]
    for label, values in (
        ("task ids", task_ids),
        ("scenario ids", scenario_ids),
        ("scenario content hashes", hashes),
    ):
        if len(values) != len(set(values)):
            issues.append(f"materialized {label} are not unique")

    for item in materialized_tasks:
        prefix = item.task.task_id
        scenario = item.scenario
        population = population_for_scenario(scenario)
        if len(population.buyers) != item.task.buyers:
            issues.append(f"{prefix}: buyer population mismatch")
        if len(population.merchants) != item.task.merchants:
            issues.append(f"{prefix}: merchant population mismatch")
        actor_ids = [buyer.buyer_id for buyer in population.buyers] + [
            merchant.merchant_id for merchant in population.merchants
        ]
        if len(actor_ids) != len(set(actor_ids)):
            issues.append(f"{prefix}: duplicate actor id")
        if item.evaluated_actor_id not in actor_ids:
            issues.append(f"{prefix}: evaluated actor is not declared")
        merchant_ids = {merchant.merchant_id for merchant in population.merchants}
        catalog = scenario.initial_state.get("catalog", ()) or ()
        skus: set[str] = set()
        owned: dict[str, set[str]] = {merchant_id: set() for merchant_id in merchant_ids}
        for row in catalog:
            if not isinstance(row, Mapping):
                issues.append(f"{prefix}: non-mapping catalog row")
                continue
            sku = str(row.get("sku_id", ""))
            owner = str(row.get("merchant_id", ""))
            if not sku or sku in skus:
                issues.append(f"{prefix}: empty or duplicate sku_id {sku!r}")
            skus.add(sku)
            if owner not in merchant_ids:
                issues.append(f"{prefix}: listing {sku!r} has undeclared owner {owner!r}")
            else:
                owned[owner].add(sku)
            attrs = row.get("attributes") or {}
            if isinstance(attrs, Mapping) and attrs.get("benchmark_distractor"):
                if int(row.get("inventory", -1)) != 0 or attrs.get("in_stock") is not False:
                    issues.append(f"{prefix}: distractor {sku!r} is accidentally feasible")
                if not sku.startswith(f"{owner}:"):
                    issues.append(f"{prefix}: added listing {sku!r} lacks owner namespace")
        for merchant in population.merchants:
            if set(merchant.catalog_scope) != owned.get(merchant.merchant_id, set()):
                issues.append(f"{prefix}: catalog scope disagrees for {merchant.merchant_id}")

        try:
            source = from_yaml(_scenario_path(root, item.task.template_variant_id))
        except (MaterializationError, ScenarioInvalid) as exc:
            issues.append(f"{prefix}: source template cannot be reloaded ({exc})")
        else:
            if scenario.success_oracle != source.success_oracle:
                issues.append(f"{prefix}: canonical success oracle was modified")

        expected_hash = scenario_content_hash_v2(scenario)
        if item.content_sha256 != expected_hash:
            issues.append(f"{prefix}: content hash does not match scenario")
        public = item.to_public_dict()
        if _contains_seed_key(public):
            issues.append(f"{prefix}: public manifest exposes seed")
        try:
            json.dumps(public, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            issues.append(f"{prefix}: public manifest is not JSON-compatible ({exc})")

        # This is the executable lower bound: owner/scope/order/ledger mistakes
        # must fail during construction, before any paid model call.
        try:
            from world import World

            seed_world(World(), scenario)
            kickoff_envelopes(scenario)
        except Exception as exc:  # noqa: BLE001 - aggregate all preflight failures
            issues.append(
                f"{prefix}: executable preflight failed ({type(exc).__name__}: {exc})"
            )
    return tuple(issues)


def export_public_manifest(
    materialized_tasks: Sequence[MaterializedTaskV2],
    *,
    indent: int | None = 2,
) -> str:
    """Serialize public materialization identities after validation."""

    issues = validate(materialized_tasks, expected_tasks=len(materialized_tasks))
    if issues:
        raise MaterializationError("invalid public manifest: " + "; ".join(issues))
    return json.dumps(
        [item.to_public_dict() for item in materialized_tasks],
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
    ) + "\n"


# Explicit aliases for callers that prefer versioned names.
materialize_task_v2 = materialize
materialize_all_v2 = materialize_all
validate_materialized_v2 = validate


__all__ = [
    "MaterializationError",
    "MaterializedTaskV2",
    "export_public_manifest",
    "materialize",
    "materialize_all",
    "materialize_all_v2",
    "materialize_task_v2",
    "scenario_content_hash_v2",
    "validate",
    "validate_materialized_v2",
]
