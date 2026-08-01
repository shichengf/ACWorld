"""ACWorld T6 inventory, fulfillment, and logistics tasks.

Every state transition in this module is requested through Runtime and
Platform and committed by World.  The module contains only fixed benchmark
materialization, deterministic conformance channels, and evidence scorers; it
does not import or call the retired direct T6 simulator.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from itertools import combinations
from typing import Any, Mapping, Sequence

from agents.agent_phase import public_reference_alias_v1
from agents.business_decision import LLM_BUSINESS_DECISION_SCHEMA_V1
from agents.inference import BusinessDecisionResponseV1, InferenceChannel
from episode.capability_benchmark import (
    TASK_REGISTRY_V2,
    TaskDefinitionV2,
    is_hardened_task_v2,
)
from episode.capability_runtime import (
    RuntimeBenchmarkIntegrityError,
    RuntimeEvidenceBundleV2,
    RuntimeEvidenceError,
    RuntimeMutationV2,
    RuntimeRubricCheckV2,
    RuntimeTaskBundleV2,
    RuntimeTaskScoreV3,
    canonical_sha256,
    renormalize_capability_checks_v2,
    require_runtime_benchmark_integrity_v2,
    score_checks,
)
from episode.capability_runtime_fixture import require_frozen_scenario_fixture_v2
from episode.capability_runtime_discovery import (
    verify_optional_discovery_prefix_v2,
)
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec
from runtime.supply_fulfillment_evidence import (
    SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
    VerifiedSupplyFulfillmentEvidence,
)
from runtime.tracker_evidence import (
    TrackerEvidenceError,
    VerifiedModelBusinessChoice,
    verified_model_business_choices,
)


T6_RUNTIME_SCHEMA_V2 = "cwe.runtime-task.t6.v4"
_BUYER_ID = "buyer:t6-benchmark"
_MERCHANT_ID = "merchant:t6-benchmark"
_T6_TASK_IDS = tuple(
    task_id for task_id, definition in TASK_REGISTRY_V2.items() if definition.family.value == "T6"
)


@dataclass(frozen=True)
class _CaseT6:
    definition: TaskDefinitionV2
    lane: str
    axis_name: str
    axis_value: int
    catalog: tuple[dict[str, Any], ...]
    read_skus: tuple[str, ...] = ()
    requested_qty: int = 0
    requested_quantities: tuple[int, ...] = ()
    expected_sku_id: str | None = None
    expected_fulfill: int = 0
    expected_backorder: int = 0
    expected_resolution: str | None = None
    expected_decision: str | None = None
    expected_report: tuple[dict[str, Any], ...] = ()
    event_count: int = 0
    buyer_ids: tuple[str, ...] = (_BUYER_ID,)
    priority_order_ids: tuple[str, ...] = ()
    expected_allocations: tuple[int, ...] = ()
    allocation_requests: tuple[dict[str, Any], ...] = ()
    shipment_id: str | None = None
    replacement_sku_id: str | None = None

    @property
    def evaluated_actor_id(self) -> str:
        return _BUYER_ID if self.definition.evaluated_role == "buyer" else _MERCHANT_ID

    @property
    def allocation_id(self) -> str | None:
        """Return the Agent-owned correlation for an allocation workflow."""

        if self.lane not in {
            "merchant_competing_commitment",
            "merchant_partial_backorder",
        }:
            return None
        return f"allocation:{self.definition.task_id.casefold()}"

    @property
    def semantic_contract(self) -> dict[str, Any]:
        return {
            "schema_version": T6_RUNTIME_SCHEMA_V2,
            "definition": self.definition.to_dict(),
            **(
                {"evaluation_profile": "hard-tier-step-attribution"}
                if is_hardened_task_v2(self.definition)
                else {}
            ),
            "lane": self.lane,
            "difficulty": {self.axis_name: self.axis_value},
            "catalog": self.catalog,
            "read_skus": self.read_skus,
            "requested_qty": self.requested_qty,
            **(
                {"requested_quantities": self.requested_quantities}
                if self.requested_quantities
                else {}
            ),
            "expected_sku_id": self.expected_sku_id,
            "expected_fulfill": self.expected_fulfill,
            "expected_backorder": self.expected_backorder,
            "expected_resolution": self.expected_resolution,
            "expected_decision": self.expected_decision,
            "expected_report": self.expected_report,
            "event_count": self.event_count,
            "buyer_ids": self.buyer_ids,
            "priority_order_ids": self.priority_order_ids,
            "expected_allocations": self.expected_allocations,
            **(
                {"allocation_requests": self.allocation_requests}
                if self.allocation_requests
                else {}
            ),
            "shipment_id": self.shipment_id,
            "replacement_sku_id": self.replacement_sku_id,
            "execution_path": (
                "agent",
                "platform:supply|platform:fulfillment|platform:psp",
                "world",
            ),
        }


def _axis(definition: TaskDefinitionV2) -> tuple[str, int]:
    values = [
        (name, value) for name, value in definition.difficulty_factors if name != "difficulty_level"
    ]
    if len(values) != 1:
        raise ValueError(f"{definition.task_id}: T6 needs one semantic axis")
    name, value = values[0]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{definition.task_id}: T6 axis must be positive")
    return name, value


def _sku(definition: TaskDefinitionV2, suffix: str) -> str:
    stem = definition.task_id.casefold().replace("cwv2-", "").replace("-", "")
    return f"merchant:t6:{stem}:{suffix}"


def _catalog_row(
    definition: TaskDefinitionV2,
    suffix: str,
    *,
    inventory: int,
    price_cents: int = 9_000,
    eta_day: int = 0,
    qty_reserved: int = 0,
) -> dict[str, Any]:
    sku_id = _sku(definition, suffix)
    return {
        "sku_id": sku_id,
        "product_id": f"product:{sku_id}",
        "merchant_id": _MERCHANT_ID,
        "category": "supply-item",
        "name": f"Supply item {suffix}",
        "list_price": str(Decimal(price_cents) / Decimal(100)),
        "inventory": inventory,
        "qty_reserved": qty_reserved,
        "eta_day": eta_day,
        "version": 1,
        "attributes": {},
    }


def _state_line(row: Mapping[str, Any], *, version: int = 1) -> dict[str, Any]:
    return {
        "sku_id": str(row["sku_id"]),
        "merchant_id": str(row["merchant_id"]),
        "available_qty": int(row["inventory"]) - int(row.get("qty_reserved", 0)),
        "reserved_qty": int(row.get("qty_reserved", 0)),
        "eta_day": int(row.get("eta_day", 0)),
        "unit_price_cents": int(Decimal(str(row["list_price"])) * 100),
        "version": version,
    }


def _hard_allocation_service_score(row: Mapping[str, Any]) -> int:
    """Compute the public service score used by hard merchant allocations."""

    tier = row.get("service_tier")
    promised_day = row.get("promised_day")
    prior_completed = row.get("prior_completed_orders")
    requested_qty = row.get("requested_qty")
    if (
        tier not in {"priority", "standard"}
        or not isinstance(promised_day, int)
        or isinstance(promised_day, bool)
        or promised_day <= 0
        or not isinstance(prior_completed, int)
        or isinstance(prior_completed, bool)
        or prior_completed < 0
        or not isinstance(requested_qty, int)
        or isinstance(requested_qty, bool)
        or requested_qty <= 0
    ):
        raise ValueError("hard T6 allocation facts are malformed")
    return (
        (30 if tier == "priority" else 0)
        + max(0, 8 - promised_day) * 5
        + min(prior_completed, 12) * 2
        - requested_qty
    )


def _hard_allocation_priority(
    requests: Sequence[Mapping[str, Any]],
    *,
    inventory: int,
    optimize_full_orders: bool,
) -> tuple[str, ...]:
    """Derive an auditable priority permutation from public business facts."""

    rows = tuple(dict(row) for row in requests)
    if not rows or inventory <= 0:
        raise ValueError("hard T6 allocation needs requests and positive inventory")
    if len({str(row.get("order_id", "")) for row in rows}) != len(rows):
        raise ValueError("hard T6 allocation order ids must be unique")

    def row_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
        arrival = row.get("arrival_sequence")
        if (
            not isinstance(arrival, int)
            or isinstance(arrival, bool)
            or arrival <= 0
        ):
            raise ValueError("hard T6 allocation arrival sequence is malformed")
        return (
            -_hard_allocation_service_score(row),
            arrival,
            str(row["order_id"]),
        )

    if not optimize_full_orders:
        return tuple(str(row["order_id"]) for row in sorted(rows, key=row_key))

    candidates: list[
        tuple[tuple[int, int, int, tuple[int, ...]], tuple[dict[str, Any], ...]]
    ] = []
    for size in range(len(rows) + 1):
        for selected in combinations(rows, size):
            used = sum(int(row["requested_qty"]) for row in selected)
            if used > inventory:
                continue
            metric = (
                -sum(_hard_allocation_service_score(row) for row in selected),
                -used,
                -len(selected),
                tuple(sorted(int(row["arrival_sequence"]) for row in selected)),
            )
            candidates.append((metric, tuple(selected)))
    if not candidates:
        raise ValueError("hard T6 allocation has no feasible full-order subset")
    _metric, selected = min(candidates, key=lambda row: row[0])
    selected_ids = {str(row["order_id"]) for row in selected}
    full = sorted(selected, key=row_key)
    deferred = sorted(
        (row for row in rows if str(row["order_id"]) not in selected_ids),
        key=row_key,
    )
    return tuple(str(row["order_id"]) for row in (*full, *deferred))


def _hard_allocation_requests(
    definition: TaskDefinitionV2,
    *,
    lane: str,
    unavailable_units: int,
) -> tuple[tuple[dict[str, Any], ...], int]:
    """Create shuffled, non-answer-encoding requests for hard T6 tasks."""

    if lane == "merchant_competing_commitment":
        facts = (
            (
                ("standard", 1, 8, 2, 3),
                ("priority", 8, 8, 2, 1),
                ("priority", 7, 5, 2, 2),
            )
            if unavailable_units == 3
            else (
                ("standard", 1, 10, 2, 4),
                ("priority", 8, 9, 2, 2),
                ("priority", 6, 3, 2, 1),
                ("standard", 2, 7, 2, 3),
            )
        )
        inventory = unavailable_units + 1
    elif lane == "merchant_partial_backorder":
        facts = (
            (
                ("priority", 6, 12, 5, 4),
                ("standard", 1, 12, 4, 1),
                ("priority", 8, 12, 4, 6),
                ("standard", 3, 7, 3, 3),
                ("standard", 7, 8, 2, 5),
                ("priority", 8, 9, 3, 2),
            )
            if unavailable_units == 3
            else (
                ("priority", 3, 11, 7, 7),
                ("standard", 1, 8, 6, 2),
                ("priority", 4, 10, 5, 8),
                ("priority", 6, 9, 4, 1),
                ("standard", 6, 6, 3, 6),
                ("standard", 7, 7, 2, 4),
                ("standard", 5, 6, 2, 3),
                ("standard", 7, 5, 1, 5),
            )
        )
        inventory = sum(row[3] for row in facts) - unavailable_units
    else:
        raise ValueError(f"unsupported hard allocation lane {lane!r}")

    requests = tuple(
        {
            "order_id": f"order:{definition.task_id.casefold()}:{index + 1}",
            "buyer_id": f"buyer:t6-b{index + 1}",
            "sku_id": _sku(definition, "scarce" if lane == "merchant_competing_commitment" else "partial"),
            "requested_qty": requested_qty,
            "order_state": "accepted",
            "service_tier": tier,
            "promised_day": promised_day,
            "prior_completed_orders": prior_completed,
            "arrival_sequence": arrival_sequence,
        }
        for index, (
            tier,
            promised_day,
            prior_completed,
            requested_qty,
            arrival_sequence,
        ) in enumerate(facts)
    )
    if len({int(row["arrival_sequence"]) for row in requests}) != len(requests):
        raise ValueError("hard T6 allocation arrival sequences must be unique")
    return tuple(sorted(requests, key=lambda row: int(row["arrival_sequence"]))), inventory


@lru_cache(maxsize=None)
def _case_for_t6(task_id: str) -> _CaseT6:
    definition = TASK_REGISTRY_V2[task_id]
    if definition.family.value != "T6":
        raise ValueError(f"{task_id} is not a T6 task")
    axis_name, count = _axis(definition)
    lane = definition.capability_id.removeprefix("t6.")

    if lane in {"buyer_partial_backorder", "merchant_partial_backorder"}:
        if (
            is_hardened_task_v2(definition)
            and lane == "merchant_partial_backorder"
        ):
            allocation_requests, inventory = _hard_allocation_requests(
                definition,
                lane=lane,
                unavailable_units=count,
            )
            order_ids = _hard_allocation_priority(
                allocation_requests,
                inventory=inventory,
                optimize_full_orders=True,
            )
            by_order = {
                str(row["order_id"]): row for row in allocation_requests
            }
            buyer_ids = tuple(
                str(by_order[order_id]["buyer_id"]) for order_id in order_ids
            )
            requested_quantities = tuple(
                int(by_order[order_id]["requested_qty"]) for order_id in order_ids
            )
            remaining = inventory
            allocations: list[int] = []
            for requested in requested_quantities:
                fulfilled = min(requested, remaining)
                allocations.append(fulfilled)
                remaining -= fulfilled
            row = _catalog_row(
                definition,
                "partial",
                inventory=inventory,
                eta_day=7,
            )
            return _CaseT6(
                definition,
                lane,
                axis_name,
                count,
                (row,),
                read_skus=(str(row["sku_id"]),),
                requested_qty=sum(requested_quantities),
                requested_quantities=requested_quantities,
                expected_sku_id=str(row["sku_id"]),
                expected_fulfill=inventory,
                expected_backorder=count,
                buyer_ids=buyer_ids,
                priority_order_ids=order_ids,
                expected_allocations=tuple(allocations),
                allocation_requests=allocation_requests,
            )
        row = _catalog_row(
            definition,
            "main" if lane.startswith("buyer") else "partial",
            inventory=3,
            eta_day=(
                7
                if is_hardened_task_v2(definition)
                or lane.startswith("merchant")
                else 6
            ),
        )
        return _CaseT6(
            definition,
            lane,
            axis_name,
            count,
            (row,),
            read_skus=(str(row["sku_id"]),),
            requested_qty=3 + count,
            expected_sku_id=str(row["sku_id"]),
            expected_fulfill=3,
            expected_backorder=count,
            expected_decision=(
                "decline"
                if is_hardened_task_v2(definition)
                and lane == "buyer_partial_backorder"
                else None
            ),
            priority_order_ids=(f"order:{task_id.casefold()}:partial",),
        )

    if lane == "buyer_stock_substitution":
        source = _catalog_row(definition, "desired", inventory=1, price_cents=8_500)
        if is_hardened_task_v2(definition):
            # Preserve the substitution objective while removing the previous
            # "choose the first row" shortcut.
            substitute_specs = (
                (1, 7_800, 1),
                (2, 8_300, 4),
                (2, 8_500, 2),
                (3, 8_900, 1),
                (2, 9_800, 0),
            )
            substitutes = tuple(
                _catalog_row(
                    definition,
                    f"sub{index + 1}",
                    inventory=inventory,
                    price_cents=price_cents,
                    eta_day=eta_day,
                )
                for index, (inventory, price_cents, eta_day) in enumerate(
                    substitute_specs
                )
            )
            requested_qty = 2
            expected_sku_id = str(substitutes[2]["sku_id"])
        else:
            substitutes = tuple(
                _catalog_row(
                    definition,
                    f"sub{index + 1}",
                    inventory=1,
                    price_cents=8_000 + index * 700,
                    eta_day=index + 1,
                )
                for index in range(count)
            )
            requested_qty = 1
            expected_sku_id = str(substitutes[0]["sku_id"])
        return _CaseT6(
            definition,
            lane,
            axis_name,
            count,
            (source, *substitutes),
            read_skus=tuple(str(row["sku_id"]) for row in (source, *substitutes)),
            requested_qty=requested_qty,
            expected_sku_id=expected_sku_id,
            event_count=1,
        )

    if lane.endswith("delivery_exception"):
        original = _catalog_row(
            definition,
            "original",
            inventory=1,
            qty_reserved=1,
        )
        if is_hardened_task_v2(definition):
            replacements = (
                _catalog_row(
                    definition,
                    "replacement-value",
                    inventory=2,
                    price_cents=8_800,
                    eta_day=3,
                ),
                _catalog_row(
                    definition,
                    "replacement-balanced",
                    inventory=5,
                    price_cents=9_200,
                    eta_day=2,
                ),
                _catalog_row(
                    definition,
                    "replacement-premium",
                    inventory=7,
                    price_cents=10_500,
                    eta_day=1,
                ),
            )
        else:
            replacements = (
                _catalog_row(definition, "replacement", inventory=2),
            )
        shipment_id = f"shipment:{task_id.casefold()}"
        return _CaseT6(
            definition,
            lane,
            axis_name,
            count,
            (original, *replacements),
            expected_sku_id=str(original["sku_id"]),
            expected_resolution="wait" if count == 1 else "replacement",
            event_count=count,
            shipment_id=shipment_id,
            replacement_sku_id=str(
                (
                    replacements[1]
                    if is_hardened_task_v2(definition) and count == 3
                    else replacements[0]
                )["sku_id"]
            ),
        )

    if lane in {"buyer_restock_price", "merchant_restock_price"}:
        if is_hardened_task_v2(definition):
            rows = (
                _catalog_row(
                    definition,
                    "restock",
                    inventory=0,
                    price_cents=10_000,
                    eta_day=5,
                ),
                _catalog_row(
                    definition,
                    "stable-value",
                    inventory=2,
                    price_cents=10_500,
                    eta_day=1,
                ),
                _catalog_row(
                    definition,
                    "late-low-price",
                    inventory=3,
                    price_cents=9_800,
                    eta_day=2,
                ),
                _catalog_row(
                    definition,
                    "shallow-stock",
                    inventory=1,
                    price_cents=9_000,
                    eta_day=0,
                ),
            )
        else:
            rows = (
                _catalog_row(
                    definition,
                    "restock",
                    inventory=0,
                    price_cents=10_000,
                    eta_day=5,
                ),
            )
        row = rows[0]
        final_qty = 2 if count == 2 else 4
        final_price = 9_500 if count == 2 else 12_500
        final = {
            **_state_line(row, version=count + 1),
            "available_qty": final_qty,
            "eta_day": 0,
            "unit_price_cents": final_price,
        }
        return _CaseT6(
            definition,
            lane,
            axis_name,
            count,
            rows,
            read_skus=tuple(str(value["sku_id"]) for value in rows),
            requested_qty=2 if is_hardened_task_v2(definition) else 1,
            expected_sku_id=str(
                (
                    rows[1]
                    if is_hardened_task_v2(definition) and lane.startswith("buyer")
                    else row
                )["sku_id"]
            ),
            expected_decision=(
                "select"
                if lane.startswith("buyer")
                and (
                    is_hardened_task_v2(definition)
                    or final_price <= 11_000
                )
                else "decline"
                if lane.startswith("buyer")
                else None
            ),
            expected_report=(final,) if lane.startswith("merchant") else (),
            event_count=count,
        )

    if lane == "merchant_inventory_eta":
        rows = tuple(
            _catalog_row(
                definition,
                f"item{index + 1}",
                inventory=2 + index * 2,
                price_cents=7_500 + index * 625,
                eta_day=index,
            )
            for index in range(count)
        )
        return _CaseT6(
            definition,
            lane,
            axis_name,
            count,
            rows,
            read_skus=tuple(str(row["sku_id"]) for row in rows),
            expected_report=tuple(_state_line(row) for row in rows),
        )

    if lane == "merchant_competing_commitment":
        if is_hardened_task_v2(definition):
            allocation_requests, inventory = _hard_allocation_requests(
                definition,
                lane=lane,
                unavailable_units=count,
            )
            order_ids = _hard_allocation_priority(
                allocation_requests,
                inventory=inventory,
                optimize_full_orders=False,
            )
            by_order = {
                str(value["order_id"]): value for value in allocation_requests
            }
            buyer_ids = tuple(
                str(by_order[order_id]["buyer_id"]) for order_id in order_ids
            )
            requested_quantities = tuple(
                int(by_order[order_id]["requested_qty"]) for order_id in order_ids
            )
        else:
            allocation_requests = ()
            inventory = count + 1
            buyer_ids = tuple(f"buyer:t6-b{index + 1}" for index in range(count))
            order_ids = tuple(
                f"order:{task_id.casefold()}:{index + 1}" for index in range(count)
            )
            requested_quantities = tuple(2 for _ in order_ids)
        row = _catalog_row(definition, "scarce", inventory=inventory)
        remaining = inventory
        allocations: list[int] = []
        for requested_qty in requested_quantities:
            qty = min(requested_qty, remaining)
            allocations.append(qty)
            remaining -= qty
        return _CaseT6(
            definition,
            lane,
            axis_name,
            count,
            (row,),
            read_skus=(str(row["sku_id"]),),
            expected_sku_id=str(row["sku_id"]),
            buyer_ids=buyer_ids,
            priority_order_ids=order_ids,
            expected_allocations=tuple(allocations),
            allocation_requests=allocation_requests,
        )

    raise ValueError(f"unsupported T6 capability {definition.capability_id!r}")


def _public_contract(case: _CaseT6) -> dict[str, Any]:
    """Return task inputs that are safe to expose to the evaluated actor.

    Expected decisions and final quantities intentionally stay in the private
    semantic contract/scorer.  The actor receives identifiers and the request,
    then has to obtain the changing state from Platform/World.
    """

    state_read_policy: dict[str, Any] = {
        "source": "authoritative_current_state",
        "scope": (
            "current_shipment" if case.lane.endswith("delivery_exception") else "all_listed_skus"
        ),
    }
    contract: dict[str, Any] = {
        "schema_version": T6_RUNTIME_SCHEMA_V2,
        "task_id": case.definition.task_id,
        "instruction": (
            "Check the current official record for where things actually stand, "
            "then make one call on stock, fulfilment, or shipping."
        ),
        "evaluated_role": case.definition.evaluated_role,
        "capability": case.lane,
        "difficulty": {case.axis_name: case.axis_value},
        "sku_ids": list(case.read_skus),
        "requested_qty": case.requested_qty,
        "buyer_ids": list(case.buyer_ids),
        "priority_order_ids": list(case.priority_order_ids),
        "shipment_id": case.shipment_id,
        "state_read_policy": state_read_policy,
        "authority": {
            "supply": "platform:supply -> world.read_supply_state",
            "allocation": "platform:fulfillment -> world.allocate_orders_atomic",
            "shipment": "platform:fulfillment -> world.resolve_shipment",
        },
    }
    if (
        is_hardened_task_v2(case.definition)
        and case.lane
        in {"merchant_competing_commitment", "merchant_partial_backorder"}
    ):
        # The allowed references remain in the typed action schema, but the
        # public contract no longer publishes their correct priority order as
        # a ready-to-copy list.
        contract.pop("priority_order_ids")
        contract.pop("buyer_ids")
    if case.allocation_id is not None:
        # This value is frozen Agent authority, not a model input.  The shared
        # public-reference bridge recognizes ``allocation_id`` as a framework
        # field, removes it from observations and model-facing schemas, and binds
        # it back only after validating the model's business choices.
        contract["allocation_id"] = case.allocation_id
        if case.allocation_requests:
            contract["allocation_requests"] = [
                copy.deepcopy(row) for row in case.allocation_requests
            ]
        else:
            requested_quantities = (
                case.requested_quantities
                if case.requested_quantities
                else (case.requested_qty,)
                if case.lane == "merchant_partial_backorder"
                else tuple(2 for _ in case.priority_order_ids)
            )
            contract["allocation_requests"] = [
                {
                    "order_id": order_id,
                    "buyer_id": buyer_id,
                    "sku_id": case.read_skus[0],
                    "requested_qty": requested_qty,
                    "request_order": request_order,
                    "order_state": "accepted",
                }
                for request_order, (order_id, buyer_id, requested_qty) in enumerate(
                    zip(
                        case.priority_order_ids,
                        case.buyer_ids,
                        requested_quantities,
                        strict=True,
                    ),
                    start=1,
                )
            ]
        contract["allocation_policy"] = {
            "eligible_order_state": "accepted",
            "priority": ["request_order_ascending", "order_ref_ascending"],
            "fulfillment": "fill_in_priority_order_and_backorder_remainder",
        }
    if case.lane in {"buyer_partial_backorder", "merchant_partial_backorder"}:
        contract["partial_backorder_policy"] = (
            {
                "purchase_requested_quantity": True,
                "allow_partial_fulfillment": True,
                "backorder_unavailable_remainder": True,
                "minimum_immediate_qty": 3,
                "maximum_backorder_qty": 4,
                "maximum_backorder_eta_day": 6,
                "on_ineligible": "decline_purchase",
            }
            if is_hardened_task_v2(case.definition)
            and case.lane == "buyer_partial_backorder"
            else {
                "purchase_requested_quantity": True,
                "allow_partial_fulfillment": True,
                "backorder_unavailable_remainder": True,
            }
        )
    if case.lane == "buyer_stock_substitution":
        contract["substitution_policy"] = {
            "required_qty": case.requested_qty,
            "eligible": "available_qty_at_least_required_qty",
            **(
                {
                    "maximum_unit_price_cents": 9_000,
                    "maximum_eta_day": 2,
                }
                if is_hardened_task_v2(case.definition)
                else {}
            ),
            "objective": [
                "unit_price_cents_ascending",
                "eta_day_ascending",
                "sku_ref_ascending",
            ],
            "allow_partial_fulfillment": False,
        }
    if case.lane.endswith("delivery_exception"):
        contract["delivery_remedy_policy"] = (
            {
                "status_remedy": {
                    "delayed": "wait",
                    "missing_scan": "replacement",
                    "lost": "replacement",
                },
                "minimum_replacement_available_qty": 3,
                "maximum_replacement_unit_price_cents": 9_500,
                "replacement_objective": [
                    "available_qty_descending",
                    "unit_price_cents_ascending",
                    "sku_ref_ascending",
                ],
            }
            if is_hardened_task_v2(case.definition)
            else {
                "status_remedy": {
                    "delayed": "wait",
                    "missing_scan": "replacement",
                    "lost": "replacement",
                },
                "replacement_eligibility": "available_qty_at_least_one",
                "replacement_objective": [
                    "unit_price_cents_ascending",
                    "sku_ref_ascending",
                ],
            }
        )
    if case.lane == "buyer_restock_price":
        # This is the buyer's business policy, not a scorer answer.  The model
        # needs the price boundary in order for the select/decline decision to
        # be an evaluated business choice rather than an undisclosed oracle.
        contract["purchase_policy"] = (
            {
                "required_qty": 2,
                "max_unit_price_cents": 11_000,
                "max_eta_day": 1,
                "objective": [
                    "unit_price_cents_ascending",
                    "eta_day_ascending",
                    "sku_ref_ascending",
                ],
                "on_no_eligible_option": "decline_purchase",
            }
            if is_hardened_task_v2(case.definition)
            else {"max_unit_price_cents": 11_000}
        )
    if case.lane in {"merchant_inventory_eta", "merchant_restock_price"}:
        contract["report_policy"] = {
            "scope": "all_observed_states",
            "order": "observed_state_order",
            "fields": [
                "sku_ref",
                "available_qty",
                "reserved_qty",
                "eta_day",
                "unit_price_cents",
            ],
            "transform": "none",
        }
    if (
        is_hardened_task_v2(case.definition)
        and case.lane
        in {"merchant_competing_commitment", "merchant_partial_backorder"}
    ):
        contract["allocation_policy"] = {
            "eligible_order_state": "accepted",
            "service_score": {
                "current_day": 8,
                "priority_tier_points": 30,
                "late_day_points": 5,
                "prior_completed_order_points": 2,
                "prior_completed_order_cap": 12,
                "requested_unit_penalty": 1,
                "formula": (
                    "tier_points + max(0,current_day-promised_day)*late_day_points "
                    "+ min(prior_completed_orders,cap)*history_points "
                    "- requested_qty*requested_unit_penalty"
                ),
            },
            "priority": [
                "service_score_descending",
                "arrival_sequence_ascending",
                "order_ref_ascending",
            ],
            "fulfillment": "fill_in_priority_order_and_backorder_remainder",
            **(
                {
                    "full_order_selection": {
                        "objective": (
                            "maximize_sum_service_score_of_fully_filled_orders"
                        ),
                        "capacity": "authoritative_available_qty",
                        "tie_breaks": [
                            "allocated_units_descending",
                            "selected_order_count_descending",
                            "selected_arrival_sequence_lexicographic",
                        ],
                    },
                    "output_order": (
                        "selected_full_orders_by_priority_then_deferred_orders_by_priority"
                    ),
                }
                if case.lane == "merchant_partial_backorder"
                else {}
            ),
        }
    _validate_public_contract_t6(case, contract)
    return contract


def _validate_public_contract_t6(
    case: _CaseT6,
    contract: Mapping[str, Any],
) -> None:
    """Fail task construction when a public T6 decision rule is incomplete."""

    expected_read_scope = (
        "current_shipment" if case.lane.endswith("delivery_exception") else "all_listed_skus"
    )
    if contract.get("state_read_policy") != {
        "source": "authoritative_current_state",
        "scope": expected_read_scope,
    }:
        raise ValueError(f"{case.definition.task_id}: missing public state-read policy")
    required_policy_by_lane = {
        "buyer_partial_backorder": "partial_backorder_policy",
        "merchant_partial_backorder": "partial_backorder_policy",
        "buyer_stock_substitution": "substitution_policy",
        "buyer_delivery_exception": "delivery_remedy_policy",
        "merchant_delivery_exception": "delivery_remedy_policy",
        "buyer_restock_price": "purchase_policy",
        "merchant_inventory_eta": "report_policy",
        "merchant_restock_price": "report_policy",
        "merchant_competing_commitment": "allocation_policy",
    }
    required_policy = required_policy_by_lane.get(case.lane)
    if required_policy is None or not isinstance(
        contract.get(required_policy),
        Mapping,
    ):
        raise ValueError(
            f"{case.definition.task_id}: missing public decision policy for {case.lane}"
        )
    if case.allocation_id is not None:
        requests = contract.get("allocation_requests")
        if not (
            isinstance(requests, list)
            and len(requests) == len(case.priority_order_ids) > 0
            and all(isinstance(row, Mapping) for row in requests)
        ):
            raise ValueError(f"{case.definition.task_id}: allocation requests are incomplete")
        order_ids = [row.get("order_id") for row in requests]
        quantities = [row.get("requested_qty") for row in requests]
        order_states = [row.get("order_state") for row in requests]
        hard_allocation = (
            is_hardened_task_v2(case.definition)
            and case.lane
            in {"merchant_competing_commitment", "merchant_partial_backorder"}
        )
        if not (
            len(set(order_ids)) == len(order_ids)
            and (
                set(order_ids) == set(case.priority_order_ids)
                if hard_allocation
                else order_ids == list(case.priority_order_ids)
            )
            and (
                sorted(row.get("arrival_sequence") for row in requests)
                == list(range(1, len(requests) + 1))
                if hard_allocation
                else [row.get("request_order") for row in requests]
                == list(range(1, len(requests) + 1))
            )
            and set(order_states) == {"accepted"}
            and all(
                isinstance(qty, int) and not isinstance(qty, bool) and qty > 0 for qty in quantities
            )
        ):
            raise ValueError(
                f"{case.definition.task_id}: allocation priority/quantity facts are ambiguous"
            )
        if hard_allocation and not all(
            row.get("service_tier") in {"priority", "standard"}
            and isinstance(row.get("promised_day"), int)
            and not isinstance(row.get("promised_day"), bool)
            and int(row["promised_day"]) > 0
            and isinstance(row.get("prior_completed_orders"), int)
            and not isinstance(row.get("prior_completed_orders"), bool)
            and int(row["prior_completed_orders"]) >= 0
            for row in requests
        ):
            raise ValueError(
                f"{case.definition.task_id}: allocation service priority is incomplete"
            )
        if hard_allocation:
            expected = _hard_allocation_priority(
                requests,
                inventory=int(case.catalog[0]["inventory"]),
                optimize_full_orders=case.lane == "merchant_partial_backorder",
            )
            if expected != case.priority_order_ids:
                raise ValueError(
                    f"{case.definition.task_id}: public allocation facts encode "
                    "a different priority order"
                )
            if [row.get("order_id") for row in requests] == list(
                case.priority_order_ids
            ):
                raise ValueError(
                    f"{case.definition.task_id}: public request order leaks the answer"
                )
    if case.lane == "buyer_stock_substitution":
        policy = contract["substitution_policy"]
        if not (
            policy.get("required_qty") == case.requested_qty > 0
            and policy.get("objective")
            == [
                "unit_price_cents_ascending",
                "eta_day_ascending",
                "sku_ref_ascending",
            ]
        ):
            raise ValueError(
                f"{case.definition.task_id}: substitution objective has no stable tie-break"
            )
        if is_hardened_task_v2(case.definition) and not (
            policy.get("maximum_unit_price_cents") == 9_000
            and policy.get("maximum_eta_day") == 2
        ):
            raise ValueError(
                f"{case.definition.task_id}: substitution constraints are incomplete"
            )
    if case.lane == "buyer_restock_price":
        purchase_policy = contract["purchase_policy"]
        max_price = purchase_policy.get("max_unit_price_cents")
        if not isinstance(max_price, int) or isinstance(max_price, bool) or max_price <= 0:
            raise ValueError(f"{case.definition.task_id}: purchase policy is incomplete")
        if is_hardened_task_v2(case.definition) and not (
            purchase_policy.get("required_qty") == case.requested_qty
            and purchase_policy.get("max_eta_day") == 1
            and purchase_policy.get("objective")
            == [
                "unit_price_cents_ascending",
                "eta_day_ascending",
                "sku_ref_ascending",
            ]
            and purchase_policy.get("on_no_eligible_option")
            == "decline_purchase"
        ):
            raise ValueError(
                f"{case.definition.task_id}: restock selection policy is incomplete"
            )
    if (
        case.lane == "buyer_partial_backorder"
        and is_hardened_task_v2(case.definition)
    ):
        policy = contract["partial_backorder_policy"]
        if not (
            policy.get("minimum_immediate_qty") == 3
            and policy.get("maximum_backorder_qty") == 4
            and policy.get("maximum_backorder_eta_day") == 6
            and policy.get("on_ineligible") == "decline_purchase"
        ):
            raise ValueError(
                f"{case.definition.task_id}: backorder eligibility is incomplete"
            )
    if case.lane.endswith("delivery_exception"):
        remedies = contract["delivery_remedy_policy"].get("status_remedy")
        if not (
            isinstance(remedies, Mapping) and set(remedies) == {"delayed", "missing_scan", "lost"}
        ):
            raise ValueError(f"{case.definition.task_id}: delivery remedies are incomplete")
        if is_hardened_task_v2(case.definition) and not (
            contract["delivery_remedy_policy"].get(
                "minimum_replacement_available_qty"
            )
            == 3
            and contract["delivery_remedy_policy"].get(
                "maximum_replacement_unit_price_cents"
            )
            == 9_500
            and contract["delivery_remedy_policy"].get(
                "replacement_objective"
            )
            == [
                "available_qty_descending",
                "unit_price_cents_ascending",
                "sku_ref_ascending",
            ]
        ):
            raise ValueError(
                f"{case.definition.task_id}: replacement selection policy is incomplete"
            )
    if case.lane in {"merchant_inventory_eta", "merchant_restock_price"}:
        fields = contract["report_policy"].get("fields")
        if not (isinstance(fields, list) and fields and len(fields) == len(set(fields))):
            raise ValueError(f"{case.definition.task_id}: report fields are ambiguous")


def _public_task_context(case: _CaseT6) -> dict[str, Any]:
    """Describe one public supply workflow without serializing its answer.

    Route selection depends only on the public capability lane and evaluated
    role.  The contract never consumes the private expected SKU, quantity,
    allocation, price decision, shipment resolution, or scorer oracle.
    """

    role = case.definition.evaluated_role
    shipment_lane = case.lane.endswith("delivery_exception")
    if shipment_lane:
        initial_route = {
            "action_kind": "commerce.read_shipment",
            "destination": "platform:fulfillment",
        }
        response_kind = "platform.shipment_state"
        response_routes = [
            {
                "action_kind": "commerce.resolve_shipment",
                "destination": "platform:fulfillment",
            },
            {
                "action_kind": "commerce.send_message",
                "destination": "@argument:recipient_id",
            },
        ]
    else:
        initial_route = {
            "action_kind": "commerce.read_supply_state",
            "destination": "platform:supply",
        }
        response_kind = "platform.supply_state"
        if role == "buyer":
            response_routes = [
                {
                    "action_kind": "platform.settle_payment",
                    "destination": "platform:psp",
                },
                {
                    "action_kind": "delegate.reject_purchase",
                    "destination": "@inbound_sender",
                },
            ]
        elif case.lane in {"merchant_inventory_eta", "merchant_restock_price"}:
            response_routes = [
                {
                    "action_kind": "commerce.send_message",
                    "destination": "@argument:recipient_id",
                }
            ]
        else:
            response_routes = [
                {
                    "action_kind": "commerce.allocate_fulfillment",
                    "destination": "platform:fulfillment",
                },
                {
                    "action_kind": "commerce.send_message",
                    "destination": "@argument:recipient_id",
                },
            ]

    initial_match = (
        {
            "actor_roles": ["buyer"],
            "inbound_action_kinds": ["delegate.create_purchase_mandate"],
            "inbound_sender_roles": ["consumer"],
        }
        if role == "buyer"
        else {
            "actor_roles": ["merchant"],
            "inbound_action_kinds": ["commerce.send_message"],
            "inbound_sender_roles": ["buyer"],
            "payload_equals": {"category": "t6_authoritative_request"},
        }
    )
    phases: list[dict[str, Any]] = [
        {
            "phase_id": f"{role}_authoritative_read",
            "match": initial_match,
            "allowed_routes": [initial_route],
            "world_reads": "deny",
            "finish": "forbid",
        },
        {
            "phase_id": f"{role}_{'shipment' if shipment_lane else 'supply'}_decision",
            "match": {
                "actor_roles": [role],
                "inbound_action_kinds": [response_kind],
                "inbound_sender_roles": ["platform"],
            },
            "allowed_routes": response_routes,
            "world_reads": "deny",
            "finish": "forbid",
        },
    ]
    terminal_kinds: list[str] = []
    if shipment_lane:
        terminal_kinds.append("platform.shipment_resolved")
    elif role == "buyer":
        terminal_kinds.extend(("platform.fulfillment_allocation", "platform.settlement_receipt"))
    elif case.lane in {
        "merchant_competing_commitment",
        "merchant_partial_backorder",
    }:
        terminal_kinds.append("platform.allocation_batch")
    if terminal_kinds:
        phases.append(
            {
                "phase_id": f"{role}_supply_terminal",
                "match": {
                    "actor_roles": [role],
                    "inbound_action_kinds": terminal_kinds,
                    "inbound_sender_roles": ["platform"],
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            }
        )
    counterpart_role: str | None = None
    counterpart_category: str | None = None
    if role == "merchant":
        counterpart_role = "buyer"
        counterpart_category = "shipment_resolution" if shipment_lane else "inventory_eta_report"
    elif shipment_lane:
        counterpart_role = "merchant"
        counterpart_category = "shipment_resolution"
    if counterpart_role is not None and counterpart_category is not None:
        phases.append(
            {
                "phase_id": f"{counterpart_role}_supply_report_notice",
                "match": {
                    "actor_roles": [counterpart_role],
                    "inbound_action_kinds": ["commerce.send_message"],
                    "inbound_sender_roles": [role],
                    "payload_equals": {"category": counterpart_category},
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            }
        )
    return {
        "schema_version": T6_RUNTIME_SCHEMA_V2,
        "task_id": case.definition.task_id,
        "capability": case.lane,
        "evaluated_role": role,
        "execution_contract": {
            "schema_version": "cwe.public-task-execution.v1",
            "phases": phases,
        },
    }


def _envelope(
    case: _CaseT6,
    *,
    ordinal: str,
    sender: str,
    recipient: str,
    kind: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    key = f"t6:{case.definition.task_id.casefold()}:{ordinal}"
    return {
        "msg_id": key,
        "ts": "2026-07-16T12:00:00Z",
        "from": sender,
        "to": recipient,
        "idempotency_key": key,
        "action": {"kind": kind, "payload": copy.deepcopy(dict(payload))},
    }


def _supply_events(case: _CaseT6) -> tuple[dict[str, Any], ...]:
    if case.lane == "buyer_stock_substitution":
        return (
            _envelope(
                case,
                ordinal="supply-stockout",
                sender="runtime:supply",
                recipient="platform:supply",
                kind="platform.apply_supply_event",
                payload={
                    "sku_id": case.read_skus[0],
                    "qty_delta": -1,
                    "expected_version": 1,
                },
            ),
        )
    if case.lane not in {"buyer_restock_price", "merchant_restock_price"}:
        return ()
    changes: tuple[dict[str, Any], ...] = (
        {"qty_delta": 2, "eta_day": 0},
        {"unit_price_cents": 9_500},
        {"qty_delta": 2},
        {"unit_price_cents": 12_500},
    )
    return tuple(
        _envelope(
            case,
            ordinal=f"supply-change-{index + 1}",
            sender="runtime:supply",
            recipient="platform:supply",
            kind="platform.apply_supply_event",
            payload={
                "sku_id": case.read_skus[0],
                "expected_version": index + 1,
                **changes[index],
            },
        )
        for index in range(case.event_count)
    )


def _shipment_events(case: _CaseT6) -> tuple[dict[str, Any], ...]:
    if not case.lane.endswith("delivery_exception"):
        return ()
    statuses = ("delayed", "missing_scan", "lost")
    assert case.shipment_id is not None
    return tuple(
        _envelope(
            case,
            ordinal=f"shipment-{status}",
            sender="runtime:logistics",
            recipient="platform:fulfillment",
            kind="platform.record_shipment_status",
            payload={
                "shipment_id": case.shipment_id,
                "event_id": f"event:{case.definition.task_id.casefold()}:{status}",
                "status": status,
            },
        )
        for status in statuses[: case.event_count]
    )


def _principal_id(buyer_id: str) -> str:
    return f"consumer:{buyer_id.split(':', 1)[-1]}"


def _buyer_kickoff(case: _CaseT6) -> dict[str, Any]:
    buyer_id = case.buyer_ids[0]
    return _envelope(
        case,
        ordinal="buyer-kickoff",
        sender=_principal_id(buyer_id),
        recipient=buyer_id,
        kind="delegate.create_purchase_mandate",
        payload={
            "mandate_id": case.definition.task_id,
            "goal": "get this order filled",
            "quantity": max(1, case.requested_qty),
            "return_after_purchase": False,
            "hard_constraints": {
                "budget": 100_003,
                "delivery_days": 30,
                "must_have": [],
            },
            "soft_constraints": [],
            "soft_preferences": {"style": [], "avoid": []},
            "authority": {
                "can_buy_without_confirmation": True,
                "must_not_share_with_merchant": ["budget"],
            },
            "intent_expiry": "2099-12-31T00:00:00Z",
            "benchmark_contract": _public_contract(case),
        },
    )


def _merchant_kickoff(case: _CaseT6) -> dict[str, Any]:
    return _envelope(
        case,
        ordinal="merchant-kickoff",
        sender=case.buyer_ids[0],
        recipient=_MERCHANT_ID,
        kind="commerce.send_message",
        payload={
            "category": "t6_authoritative_request",
            "benchmark_contract": _public_contract(case),
        },
    )


def _allocation_orders(case: _CaseT6) -> list[dict[str, Any]]:
    if case.lane not in {
        "merchant_competing_commitment",
        "merchant_partial_backorder",
    }:
        return []
    qtys = (
        case.requested_quantities
        if case.requested_quantities
        else (case.requested_qty,)
        if case.lane == "merchant_partial_backorder"
        else tuple(2 for _ in case.priority_order_ids)
    )
    return [
        {
            "order_id": order_id,
            "buyer_id": buyer_id,
            "merchant_id": _MERCHANT_ID,
            "sku_id": case.expected_sku_id,
            "qty": qty,
            "agreed_price": case.catalog[0]["list_price"],
            "state": "accepted",
            "request_order": index + 1,
        }
        for index, (order_id, buyer_id, qty) in enumerate(
            zip(case.priority_order_ids, case.buyer_ids, qtys, strict=True)
        )
    ]


def _shipment_seed(case: _CaseT6) -> dict[str, Any]:
    if not case.lane.endswith("delivery_exception"):
        return {}
    assert case.shipment_id is not None
    original = case.catalog[0]
    order_id = f"order:{case.definition.task_id.casefold()}:shipment"
    txn_id = f"txn:{case.definition.task_id.casefold()}:shipment"
    return {
        "logical_time": 1,
        "orders": [
            {
                "order_id": order_id,
                "buyer_id": _BUYER_ID,
                "merchant_id": _MERCHANT_ID,
                "sku_id": original["sku_id"],
                "qty": 1,
                "agreed_price": original["list_price"],
                "state": "dispatched",
            }
        ],
        "ledger": [
            {
                "txn_id": txn_id,
                "order_id": order_id,
                "buyer_id": _BUYER_ID,
                "merchant_id": _MERCHANT_ID,
                "sku_id": original["sku_id"],
                "qty": 1,
                "price": original["list_price"],
                "idempotency_key": f"seed:{case.definition.task_id}:shipment",
            }
        ],
        "order_timelines": [
            {
                "order_id": order_id,
                "buyer_id": _BUYER_ID,
                "merchant_id": _MERCHANT_ID,
                "settled_at_tick": 0,
                "dispatched_at_tick": 1,
            }
        ],
        "shipments": [
            {
                "shipment_id": case.shipment_id,
                "order_id": order_id,
                "buyer_id": _BUYER_ID,
                "merchant_id": _MERCHANT_ID,
                "original_sku_id": original["sku_id"],
                "status": "in_transit",
                "status_history": [
                    {
                        "event_id": f"event:{case.definition.task_id.casefold()}:created",
                        "status": "in_transit",
                        "logical_time": 1,
                    }
                ],
                "version": 1,
            }
        ],
    }


def scenario_for_t6(task_id: str) -> ScenarioSpec:
    """Materialize one T6 configuration over the shared CommerceWorld runtime."""

    case = _case_for_t6(task_id)
    contract = _public_contract(case)
    task_context = _public_task_context(case)
    buyers = tuple(
        BuyerSpec(
            buyer_id=buyer_id,
            persona={"name": f"T6 buyer {buyer_id}", "task_family": "T6"},
            mandate={
                "mandate_id": f"{task_id}:{buyer_id}",
                "goal": "place my order and see it through",
                "quantity": max(1, case.requested_qty),
                "return_after_purchase": False,
                "hard_constraints": {
                    "budget": 100_003,
                    "delivery_days": 30,
                    "must_have": [],
                },
                "soft_constraints": [],
                "soft_preferences": {"style": [], "avoid": []},
                "authority": {
                    "can_buy_without_confirmation": True,
                    "must_not_share_with_merchant": ["budget"],
                },
                "intent_expiry": "2099-12-31T00:00:00Z",
                "task_context": task_context,
                "benchmark_contract": contract,
            },
        )
        for buyer_id in case.buyer_ids
    )
    merchant = MerchantSpec(
        merchant_id=_MERCHANT_ID,
        persona={"name": "T6 benchmark merchant", "task_family": "T6"},
        policy={
            "floor_price": 731,
            "margin_target_bps": 1_000,
            "max_negotiation_rounds": 1,
            "refund_policy": "30_day_return",
            "claim_aggressiveness": "neutral",
            "task_context": task_context,
            "benchmark_contract": contract,
        },
        catalog_scope=tuple(str(row["sku_id"]) for row in case.catalog),
    )
    initial_events = (
        *_supply_events(case),
        *_shipment_events(case),
        _buyer_kickoff(case)
        if case.definition.evaluated_role == "buyer"
        else _merchant_kickoff(case),
    )
    initial_state: dict[str, Any] = {
        "catalog": [copy.deepcopy(row) for row in case.catalog],
        "orders": _allocation_orders(case),
        "ledger": [],
        "order_timelines": [],
        "shipments": [],
        **_shipment_seed(case),
    }
    population = PopulationSpec(
        buyers=buyers,
        merchants=(merchant,),
        initial_events=tuple(initial_events),
        matching={"top_k": max(1, len(case.catalog))},
        execution={"max_transactions_per_buyer": 4},
    )
    return ScenarioSpec(
        scenario_id=f"{task_id.casefold().replace('-', '_')}__runtime",
        seed=int(case.definition.canonical_hash[:8], 16) % 2_147_483_646 + 1,
        initial_state=initial_state,
        buyer_goal={},
        merchant_policy={},
        allowed_actions=(
            "send_message",
            "settle",
            "read_supply_state",
            "allocate_fulfillment",
            "read_shipment",
            "resolve_shipment",
        ),
        success_oracle={
            "schema_version": T6_RUNTIME_SCHEMA_V2,
            "task_id": task_id,
            "lane": case.lane,
        },
        platform_policy=None,
        population=population,
    )


def _business_request(user_prompt: str) -> dict[str, Any]:
    """Decode the exact provider-neutral request emitted by our Agent."""

    start = user_prompt.find("{")
    if start < 0:
        raise ValueError("Agent business prompt contains no request object")
    value = json.loads(user_prompt[start:])
    if not isinstance(value, dict):
        raise ValueError("Agent business request must be an object")
    return value


def _allowed_intent(request: Mapping[str, Any], intent: str) -> Mapping[str, Any]:
    rows = request.get("allowed_intents")
    if not isinstance(rows, list):
        raise ValueError("Agent business request has no allowed intents")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("intent") == intent]
    if len(matches) != 1:
        raise ValueError(f"business intent {intent!r} is not uniquely available")
    return matches[0]


def _intent_properties(request: Mapping[str, Any], intent: str) -> Mapping[str, Any]:
    parameters = _allowed_intent(request, intent).get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    if not isinstance(properties, Mapping):
        raise ValueError(f"business intent {intent!r} has no property schema")
    return properties


def _enum_values(
    request: Mapping[str, Any],
    intent: str,
    field: str,
    *,
    array: bool = False,
) -> tuple[str, ...]:
    schema = _intent_properties(request, intent).get(field)
    if not isinstance(schema, Mapping):
        return ()
    source = schema.get("items") if array else schema
    raw = source.get("enum") if isinstance(source, Mapping) else None
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(value, str) or not value for value in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValueError(f"{intent}.{field} has no finite public reference enum")
    return tuple(raw)


def _walk_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        rows.append(value)
        for item in value.values():
            rows.extend(_walk_mappings(item))
    elif isinstance(value, list):
        for item in value:
            rows.extend(_walk_mappings(item))
    return tuple(rows)


def _business_fact(request: Mapping[str, Any], name: str) -> Any:
    values = [row[name] for row in _walk_mappings(request.get("observations")) if name in row]
    canonical = {
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for value in values
    }
    if len(canonical) != 1:
        raise ValueError(f"business observations do not identify one {name}")
    return values[0]


def _observed_rows(request: Mapping[str, Any], name: str) -> tuple[Mapping[str, Any], ...]:
    batches = [
        row[name]
        for row in _walk_mappings(request.get("observations"))
        if isinstance(row.get(name), list) and all(isinstance(item, Mapping) for item in row[name])
    ]
    if len(batches) != 1:
        raise ValueError(f"business observations do not identify one {name} batch")
    return tuple(dict(item) for item in batches[0])


def _available_intents(request: Mapping[str, Any]) -> frozenset[str]:
    rows = request.get("allowed_intents")
    if not isinstance(rows, list):
        raise ValueError("Agent business request has no allowed intents")
    return frozenset(
        str(row["intent"])
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("intent"), str)
    )


def _public_policy(request: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = _business_fact(request, name)
    if not isinstance(value, Mapping):
        raise ValueError(f"T6 request has no public {name}")
    return copy.deepcopy(dict(value))


def _report_decision(
    request: Mapping[str, Any],
    *,
    mutated: bool,
) -> tuple[str, Mapping[str, Any]]:
    """Select a typed inventory report with model-owned business facts."""

    intent = "send_message"
    if intent not in _available_intents(request):
        raise ValueError("supply report intent is unavailable")
    policy = _public_policy(request, "report_policy")
    fields = policy.get("fields")
    if not (
        policy.get("scope") == "all_observed_states"
        and policy.get("order") == "observed_state_order"
        and policy.get("transform") == "none"
        and isinstance(fields, list)
        and fields
        and all(isinstance(name, str) and name for name in fields)
        and len(fields) == len(set(fields))
    ):
        raise ValueError("T6 report policy is incomplete or ambiguous")
    properties = _intent_properties(request, intent)
    if "states" not in properties:
        raise ValueError("supply report intent omits typed business states")
    states_schema = properties["states"]
    item_schema = states_schema.get("items") if isinstance(states_schema, Mapping) else None
    item_properties = item_schema.get("properties") if isinstance(item_schema, Mapping) else None
    if not isinstance(item_properties, Mapping):
        raise ValueError("supply report intent has no typed state schema")
    if set(fields) != set(item_properties):
        raise ValueError("T6 report policy and typed report fields disagree")
    states = [
        {str(name): copy.deepcopy(row[name]) for name in fields if name in row}
        for row in _observed_rows(request, "states")
    ]
    if any(set(row) != set(fields) for row in states):
        raise ValueError("T6 observed state omits a required report field")
    if mutated and states:
        states[0]["available_qty"] = int(states[0]["available_qty"]) + 1
    return intent, {"states": states}


def _supply_decision(
    request: Mapping[str, Any],
    *,
    mutated: bool,
) -> tuple[str, Mapping[str, Any]]:
    available = _available_intents(request)
    observed = _walk_mappings(request.get("observations"))
    if "settle_payment" in available:
        buyer_policies = {
            name
            for name in (
                "purchase_policy",
                "substitution_policy",
                "partial_backorder_policy",
            )
            if any(name in row for row in observed)
        }
        if len(buyer_policies) != 1:
            raise ValueError(
                "T6 buyer supply decision has a missing or ambiguous public decision policy"
            )
    if "settle_payment" in available and any("purchase_policy" in row for row in observed):
        policy = _business_fact(request, "purchase_policy")
        if not isinstance(policy, Mapping):
            raise ValueError("restock decision has no public purchase policy")
        states = _observed_rows(request, "states")
        if "required_qty" in policy:
            if not (
                policy.get("objective")
                == [
                    "unit_price_cents_ascending",
                    "eta_day_ascending",
                    "sku_ref_ascending",
                ]
                and policy.get("on_no_eligible_option")
                == "decline_purchase"
            ):
                raise ValueError("T6 restock selection policy is incomplete")
            required_qty = int(policy["required_qty"])
            max_price = int(policy["max_unit_price_cents"])
            max_eta = int(policy["max_eta_day"])
            choices = set(_enum_values(request, "settle_payment", "sku_ref"))
            state_by_ref = {
                str(row["sku_ref"]): row
                for row in states
                if isinstance(row.get("sku_ref"), str)
            }
            options = [
                row
                for row in _observed_rows(request, "purchase_options")
                if row.get("sku_ref") in choices
                and row.get("sku_ref") in state_by_ref
                and int(row["available_qty"]) >= required_qty
                and int(row["unit_price_cents"]) <= max_price
                and int(state_by_ref[str(row["sku_ref"])]["eta_day"])
                <= max_eta
            ]
            options.sort(
                key=lambda row: (
                    int(row["unit_price_cents"]),
                    int(state_by_ref[str(row["sku_ref"])]["eta_day"]),
                    str(row["sku_ref"]),
                )
            )
            select = bool(options)
            selected_ref = str(options[0]["sku_ref"]) if options else None
        else:
            if len(states) != 1:
                raise ValueError("baseline restock decision requires one state")
            required_qty = 1
            select = int(states[0]["unit_price_cents"]) <= int(
                policy["max_unit_price_cents"]
            )
            selected_ref = _enum_values(
                request, "settle_payment", "sku_ref"
            )[0]
        if mutated:
            select = not select
        if not select:
            _allowed_intent(request, "reject_purchase")
            return "reject_purchase", {
                "reason": "The restock price exceeds the stated purchase policy."
            }
        return "settle_payment", {
            "sku_ref": (
                selected_ref
                if selected_ref is not None
                else _enum_values(request, "settle_payment", "sku_ref")[0]
            ),
            "qty": required_qty,
            "allow_partial": False,
        }
    if "settle_payment" in available and any("substitution_policy" in row for row in observed):
        policy = _public_policy(request, "substitution_policy")
        objective = policy.get("objective")
        required_qty = policy.get("required_qty")
        if not (
            isinstance(required_qty, int)
            and not isinstance(required_qty, bool)
            and required_qty > 0
            and policy.get("eligible") == "available_qty_at_least_required_qty"
            and objective
            == [
                "unit_price_cents_ascending",
                "eta_day_ascending",
                "sku_ref_ascending",
            ]
            and policy.get("allow_partial_fulfillment") is False
        ):
            raise ValueError("T6 substitution policy is incomplete or ambiguous")
        choices = set(_enum_values(request, "settle_payment", "sku_ref"))
        state_by_ref = {
            str(row["sku_ref"]): row
            for row in _observed_rows(request, "states")
            if isinstance(row.get("sku_ref"), str)
        }
        options = [
            row
            for row in _observed_rows(request, "purchase_options")
            if row.get("sku_ref") in choices
            and isinstance(row.get("available_qty"), int)
            and int(row["available_qty"]) >= required_qty
            and row.get("sku_ref") in state_by_ref
            and (
                "maximum_unit_price_cents" not in policy
                or int(row["unit_price_cents"])
                <= int(policy["maximum_unit_price_cents"])
            )
            and (
                "maximum_eta_day" not in policy
                or int(state_by_ref[str(row["sku_ref"])]["eta_day"])
                <= int(policy["maximum_eta_day"])
            )
        ]
        if not options:
            raise ValueError("T6 substitution policy has no eligible public option")
        options.sort(
            key=lambda row: (
                int(row["unit_price_cents"]),
                int(state_by_ref[str(row["sku_ref"])]["eta_day"]),
                str(row["sku_ref"]),
            )
        )
        selected = options[1 if mutated else 0]
        return "settle_payment", {
            "sku_ref": str(selected["sku_ref"]),
            "qty": required_qty,
            "allow_partial": False,
        }
    if "settle_payment" in available and any("partial_backorder_policy" in row for row in observed):
        policy = _public_policy(request, "partial_backorder_policy")
        baseline_policy = {
            "purchase_requested_quantity": True,
            "allow_partial_fulfillment": True,
            "backorder_unavailable_remainder": True,
        }
        if not set(baseline_policy).issubset(policy) or any(
            policy[name] != value for name, value in baseline_policy.items()
        ):
            raise ValueError("T6 partial/backorder policy is incomplete or ambiguous")
        requested_qty = _business_fact(request, "requested_qty")
        if (
            not isinstance(requested_qty, int)
            or isinstance(requested_qty, bool)
            or requested_qty <= 0
        ):
            raise ValueError("T6 partial/backorder quantity is missing or invalid")
        eligible = True
        if "minimum_immediate_qty" in policy:
            [state] = _observed_rows(request, "states")
            immediate = int(state["available_qty"])
            backorder = requested_qty - immediate
            eligible = bool(
                immediate >= int(policy["minimum_immediate_qty"])
                and backorder <= int(policy["maximum_backorder_qty"])
                and int(state["eta_day"])
                <= int(policy["maximum_backorder_eta_day"])
            )
            if mutated:
                eligible = not eligible
            if not eligible:
                _allowed_intent(request, "reject_purchase")
                return "reject_purchase", {
                    "reason": "The partial-fill and backorder conditions are not satisfied."
                }
        return "settle_payment", {
            "sku_ref": _enum_values(request, "settle_payment", "sku_ref")[0],
            "qty": (
                requested_qty - 1
                if mutated and "minimum_immediate_qty" not in policy
                else requested_qty
            ),
            "allow_partial": True,
        }
    if "allocate_fulfillment" in available:
        if mutated:
            _allowed_intent(request, "send_message")
            return "send_message", {}
        policy = _public_policy(request, "allocation_policy")
        baseline_allocation = policy.get("priority") == [
            "request_order_ascending",
            "order_ref_ascending",
        ]
        hard_allocation = policy.get("priority") == [
            "service_score_descending",
            "arrival_sequence_ascending",
            "order_ref_ascending",
        ]
        if not (
            policy.get("eligible_order_state") == "accepted"
            and policy.get("fulfillment")
            == "fill_in_priority_order_and_backorder_remainder"
            and (baseline_allocation or hard_allocation)
        ):
            raise ValueError("T6 allocation policy is incomplete or ambiguous")
        requests = _business_fact(request, "allocation_requests")
        if not (
            isinstance(requests, list)
            and requests
            and all(isinstance(row, Mapping) for row in requests)
        ):
            raise ValueError("T6 allocation request facts are missing")
        allowed_order_refs = set(
            _enum_values(
                request,
                "allocate_fulfillment",
                "priority_order_refs",
                array=True,
            )
        )
        if hard_allocation:
            service_score = policy.get("service_score")
            if not (
                isinstance(service_score, Mapping)
                and service_score.get("current_day") == 8
                and service_score.get("priority_tier_points") == 30
                and service_score.get("late_day_points") == 5
                and service_score.get("prior_completed_order_points") == 2
                and service_score.get("prior_completed_order_cap") == 12
                and service_score.get("requested_unit_penalty") == 1
            ):
                raise ValueError("T6 hard allocation service score is incomplete")
            normalized = []
            for row in requests:
                order_ref = row.get("order_ref")
                if not isinstance(order_ref, str) or order_ref not in allowed_order_refs:
                    raise ValueError("T6 hard allocation order reference is malformed")
                normalized.append({**dict(row), "order_id": order_ref})
            optimize_full_orders = "full_order_selection" in policy
            if optimize_full_orders:
                [state] = _observed_rows(request, "states")
                inventory = int(state["available_qty"])
            else:
                inventory = sum(int(row["requested_qty"]) for row in normalized)
            priority_refs = _hard_allocation_priority(
                normalized,
                inventory=inventory,
                optimize_full_orders=optimize_full_orders,
            )
            if set(priority_refs) != allowed_order_refs:
                raise ValueError("T6 hard allocation did not cover all public orders")
            return "allocate_fulfillment", {
                "sku_ref": _enum_values(
                    request,
                    "allocate_fulfillment",
                    "sku_ref",
                )[0],
                "priority_order_refs": list(priority_refs),
            }
        normalized_requests: list[tuple[tuple[Any, ...], str]] = []
        for row in requests:
            order_ref = row.get("order_ref")
            request_order = row.get("request_order")
            requested_qty = row.get("requested_qty")
            order_state = row.get("order_state")
            if not (
                isinstance(order_ref, str)
                and order_ref in allowed_order_refs
                and isinstance(request_order, int)
                and not isinstance(request_order, bool)
                and request_order > 0
                and isinstance(requested_qty, int)
                and not isinstance(requested_qty, bool)
                and requested_qty > 0
                and order_state == policy["eligible_order_state"]
            ):
                raise ValueError("T6 allocation request facts are malformed")
            key = (request_order, order_ref)
            normalized_requests.append((key, order_ref))
        if (
            len({row[0] for row in normalized_requests})
            != len(normalized_requests)
            or {row[1] for row in normalized_requests} != allowed_order_refs
        ):
            raise ValueError("T6 allocation priorities are incomplete or non-unique")
        normalized_requests.sort(key=lambda row: row[0])
        return "allocate_fulfillment", {
            "sku_ref": _enum_values(request, "allocate_fulfillment", "sku_ref")[0],
            "priority_order_refs": [row[1] for row in normalized_requests],
        }
    if "send_message" in available:
        return _report_decision(request, mutated=mutated)
    raise ValueError("T6 supply decision has no complete public decision policy")


def _shipment_decision(
    request: Mapping[str, Any],
    *,
    mutated: bool,
) -> tuple[str, Mapping[str, Any]]:
    policy = _public_policy(request, "delivery_remedy_policy")
    status_remedy = policy.get("status_remedy")
    replacement_objective = policy.get("replacement_objective")
    hard_replacement = (
        policy.get("minimum_replacement_available_qty") is not None
    )
    if not (
        isinstance(status_remedy, Mapping)
        and status_remedy
        and set(status_remedy.values()).issubset({"wait", "replacement", "refund"})
        and (
            (
                hard_replacement
                and replacement_objective
                == [
                    "available_qty_descending",
                    "unit_price_cents_ascending",
                    "sku_ref_ascending",
                ]
            )
            or (
                not hard_replacement
                and policy.get("replacement_eligibility")
                == "available_qty_at_least_one"
                and replacement_objective
                == ["unit_price_cents_ascending", "sku_ref_ascending"]
            )
        )
    ):
        raise ValueError("T6 delivery remedy policy is incomplete or ambiguous")
    if mutated:
        _allowed_intent(request, "refund_shipment")
        return "refund_shipment", {}
    shipment_rows = [
        row["shipment"]
        for row in _walk_mappings(request.get("observations"))
        if isinstance(row.get("shipment"), Mapping)
    ]
    if len(shipment_rows) != 1:
        raise ValueError("shipment observations do not identify one shipment")
    status = shipment_rows[0].get("status")
    remedy = status_remedy.get(status)
    if remedy == "wait":
        _allowed_intent(request, "wait_for_shipment")
        return "wait_for_shipment", {}
    if remedy == "refund":
        _allowed_intent(request, "refund_shipment")
        return "refund_shipment", {}
    if remedy != "replacement":
        raise ValueError("T6 delivery policy has no remedy for current status")
    allowed_refs = set(
        _enum_values(
            request,
            "replace_shipment",
            "replacement_sku_ref",
        )
    )
    options = [
        row
        for row in _observed_rows(request, "replacement_options")
        if row.get("sku_ref") in allowed_refs
        and isinstance(row.get("available_qty"), int)
        and int(row["available_qty"])
        >= int(policy.get("minimum_replacement_available_qty", 1))
        and (
            "maximum_replacement_unit_price_cents" not in policy
            or int(row["unit_price_cents"])
            <= int(policy["maximum_replacement_unit_price_cents"])
        )
    ]
    if not options:
        raise ValueError("T6 delivery policy has no eligible replacement")
    options.sort(
        key=(
            lambda row: (
                -int(row["available_qty"]),
                int(row["unit_price_cents"]),
                str(row["sku_ref"]),
            )
            if hard_replacement
            else (
                int(row["unit_price_cents"]),
                str(row["sku_ref"]),
            )
        )
    )
    _allowed_intent(request, "replace_shipment")
    return "replace_shipment", {"replacement_sku_ref": str(options[0]["sku_ref"])}


def ideal_business_decision_t6(
    request: Mapping[str, Any],
    *,
    mutated: bool = False,
) -> tuple[str, Mapping[str, Any]]:
    """Derive one T6 decision from the complete provider request only."""

    available = _available_intents(request)
    if "read_supply_state" in available:
        policy = _public_policy(request, "state_read_policy")
        if policy != {
            "source": "authoritative_current_state",
            "scope": "all_listed_skus",
        }:
            raise ValueError("T6 supply read policy is incomplete or ambiguous")
        return "read_supply_state", {
            "sku_refs": list(
                _enum_values(
                    request,
                    "read_supply_state",
                    "sku_refs",
                    array=True,
                )
            )
        }
    if "read_shipment" in available:
        policy = _public_policy(request, "state_read_policy")
        if policy != {
            "source": "authoritative_current_state",
            "scope": "current_shipment",
        }:
            raise ValueError("T6 shipment read policy is incomplete or ambiguous")
        return "read_shipment", {
            "shipment_ref": _enum_values(
                request,
                "read_shipment",
                "shipment_ref",
            )[0]
        }
    if available.intersection({"wait_for_shipment", "replace_shipment", "refund_shipment"}):
        return _shipment_decision(request, mutated=mutated)
    return _supply_decision(request, mutated=mutated)


def _business_response(intent: str, arguments: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "schema_version": LLM_BUSINESS_DECISION_SCHEMA_V1,
            "intent": intent,
            "arguments": copy.deepcopy(dict(arguments)),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class _T6Channel:
    """Reviewed T6 policy over the only typed business-decision seam."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def __init__(self, case: _CaseT6, *, mutated: bool) -> None:
        self._actor_role = case.definition.evaluated_role
        self._mutated = mutated

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("T6 business channel requires Agent decision evidence")
        request = _business_request(user_prompt)
        if request.get("role") != self._actor_role:
            raise ValueError("T6 business request crossed actor roles")
        intent, arguments = ideal_business_decision_t6(
            request,
            mutated=self._mutated,
        )
        _allowed_intent(request, intent)
        content = _business_response(intent, arguments)
        return BusinessDecisionResponseV1(
            content=content,
            response_chars=len(content),
            response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )


class _UnexpectedT6Channel:
    """Fail if a framework-terminal counterpart ever reaches a provider."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt, user_prompt, decision_id
        raise AssertionError("framework-terminal T6 counterpart reached the provider")


def targeted_mutation_channel_t6(task_id: str) -> InferenceChannel:
    """Run one deliberate semantic error through the same real runtime."""

    return _T6Channel(_case_for_t6(task_id), mutated=True)


def _action_payload(envelope: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(envelope, Mapping):
        return {}
    action = envelope.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    return dict(payload) if isinstance(payload, Mapping) else {}


def _first_action(
    evidence: RuntimeEvidenceBundleV2,
    *,
    kind: str,
    sender: str | None = None,
    recipient: str | None = None,
) -> dict[str, Any] | None:
    for row in evidence.actions(kind=kind, actor_id=sender):
        if recipient is None or row.get("to") == recipient:
            return row
    return None


def _table(snapshot: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    tables = snapshot.get("tables")
    value = tables.get(name) if isinstance(tables, Mapping) else None
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _inventory(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tables = snapshot.get("tables")
    value = tables.get("inventory") if isinstance(tables, Mapping) else None
    if not isinstance(value, Mapping):
        return {}
    return {str(sku_id): dict(row) for sku_id, row in value.items() if isinstance(row, Mapping)}


def _catalog(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sku_id")): row for row in _table(snapshot, "catalog")}


def _snapshot_money_cents(value: Any) -> int:
    if not isinstance(value, Mapping) or "amount" not in value:
        return -1
    amount = Decimal(str(value["amount"])) * 100
    if amount != amount.to_integral_value():
        return -1
    return int(amount)


def _expected_supply_states(case: _CaseT6) -> tuple[dict[str, Any], ...]:
    states = {str(row["sku_id"]): _state_line(row) for row in case.catalog}
    if case.lane == "buyer_stock_substitution":
        source = states[case.read_skus[0]]
        source["available_qty"] = int(source["available_qty"]) - 1
        source["version"] = 2
    if case.lane in {"buyer_restock_price", "merchant_restock_price"}:
        state = states[case.read_skus[0]]
        state.update(
            available_qty=2 if case.event_count == 2 else 4,
            eta_day=0,
            unit_price_cents=9_500 if case.event_count == 2 else 12_500,
            version=case.event_count + 1,
        )
    return tuple(copy.deepcopy(states[sku_id]) for sku_id in case.read_skus)


def _unique_text_sequence(value: Any) -> tuple[str, ...] | None:
    if not (
        isinstance(value, list)
        and all(isinstance(item, str) and item for item in value)
        and len(value) == len(set(value))
    ):
        return None
    return tuple(value)


def _supply_grounding(
    case: _CaseT6,
    request: Mapping[str, Any] | None,
    response: Mapping[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    requested = _action_payload(request).get("sku_ids")
    observed = _action_payload(response).get("states")
    expected = _expected_supply_states(case)
    observed_states = (
        tuple(dict(row) for row in observed if isinstance(row, Mapping))
        if isinstance(observed, list)
        else ()
    )
    requested_skus = _unique_text_sequence(requested)
    expected_by_sku = {str(row["sku_id"]): row for row in expected}
    observed_sku_ids = tuple(str(row.get("sku_id", "")) for row in observed_states)
    observed_by_sku = {str(row.get("sku_id", "")): row for row in observed_states}
    requested_is_known_subset = bool(
        requested_skus is not None and set(requested_skus).issubset(expected_by_sku)
    )
    environment_fixture_match: bool | None
    if request is None or response is None:
        environment_fixture_match = None
    else:
        environment_fixture_match = bool(
            requested_is_known_subset
            and len(observed_sku_ids) == len(set(observed_sku_ids))
            and set(observed_sku_ids) == set(requested_skus or ())
            and all(
                observed_by_sku[sku_id] == expected_by_sku[sku_id]
                for sku_id in requested_skus or ()
            )
        )
    # The shared evidence contract has already proven that the response is an
    # exact projection of authoritative World state.  The separate fixture flag
    # below proves that state is also the state frozen by this benchmark task;
    # it is an invalid/no-score prerequisite, never rubric credit.  The model
    # check is set-based because read-target order has no business meaning.
    ok = bool(
        requested_skus is not None
        and len(requested_skus) == len(case.read_skus)
        and set(requested_skus) == set(case.read_skus)
    )
    return ok, {
        "requested_skus": requested,
        "expected_states": list(expected),
        "observed_states": list(observed_states),
        "environment_fixture_match": environment_fixture_match,
    }


def _expected_shipment_replacement_options(case: _CaseT6) -> tuple[dict[str, Any], ...]:
    if case.replacement_sku_id is None:
        return ()
    matches = [
        row
        for row in case.catalog
        if (
            row.get("sku_id") != case.expected_sku_id
            if is_hardened_task_v2(case.definition)
            else row.get("sku_id") == case.replacement_sku_id
        )
    ]
    if not matches or sum(
        row.get("sku_id") == case.replacement_sku_id for row in matches
    ) != 1:
        raise RuntimeBenchmarkIntegrityError(
            "T6 frozen shipment fixture has no unique expected replacement listing"
        )
    return tuple(
        {
            "sku_id": str(listing["sku_id"]),
            "merchant_id": str(listing["merchant_id"]),
            "available_qty": _state_line(listing)["available_qty"],
            "unit_price_cents": _state_line(listing)["unit_price_cents"],
            "currency": "USD",
            "supply_version": _state_line(listing)["version"],
        }
        for listing in matches
    )


def _shipment_grounding(
    case: _CaseT6,
    request: Mapping[str, Any] | None,
    response: Mapping[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    requested = _action_payload(request).get("shipment_id")
    shipment_value = _action_payload(response).get("shipment")
    shipment = dict(shipment_value) if isinstance(shipment_value, Mapping) else {}
    history_value = shipment.get("status_history")
    history = (
        [dict(row) for row in history_value if isinstance(row, Mapping)]
        if isinstance(history_value, list)
        else []
    )
    statuses = [str(row.get("status")) for row in history]
    expected_statuses = ["in_transit", "delayed"]
    if case.event_count == 3:
        expected_statuses.extend(["missing_scan", "lost"])
    response_payload = _action_payload(response)
    raw_options = response_payload.get("replacement_options")
    observed_options = (
        tuple(dict(row) for row in raw_options if isinstance(row, Mapping))
        if isinstance(raw_options, list)
        else ()
    )
    expected_options = _expected_shipment_replacement_options(case)
    expected_options_by_sku = {str(row["sku_id"]): row for row in expected_options}
    observed_options_by_sku = {str(row.get("sku_id", "")): row for row in observed_options}
    option_ids = tuple(str(row.get("sku_id", "")) for row in observed_options)
    environment_fixture_match: bool | None
    if request is None or response is None:
        environment_fixture_match = None
    else:
        environment_fixture_match = bool(
            shipment.get("shipment_id") == case.shipment_id
            and shipment.get("status") == expected_statuses[-1]
            and statuses == expected_statuses
            and shipment.get("version") == case.event_count + 1
            and len(option_ids) == len(set(option_ids))
            and set(option_ids) == set(expected_options_by_sku)
            and all(
                observed_options_by_sku[sku_id] == expected_options_by_sku[sku_id]
                for sku_id in expected_options_by_sku
            )
        )
    # Response fidelity is an environment invariant enforced by
    # VerifiedSupplyFulfillmentEvidence.  The fixture comparison additionally
    # proves that the authoritative state is the one frozen for this task.
    # Score only the model's read target.
    ok = requested == case.shipment_id
    return ok, {
        "requested_shipment_id": requested,
        "observed_status": shipment.get("status"),
        "observed_statuses": statuses,
        "observed_version": shipment.get("version"),
        "expected_statuses": expected_statuses,
        "observed_replacement_options": list(observed_options),
        "expected_replacement_options": list(expected_options),
        "environment_fixture_match": environment_fixture_match,
    }


def _require_t6_read_fixture_integrity(
    grounding_results: Sequence[tuple[bool, Mapping[str, Any]]],
) -> None:
    failures = [
        index
        for index, (_model_target_ok, evidence) in enumerate(grounding_results)
        if evidence.get("environment_fixture_match") is not True
    ]
    if failures:
        raise RuntimeBenchmarkIntegrityError(
            "T6 authoritative provider state differs from the frozen task fixture: "
            + json.dumps({"read_indexes": failures}, sort_keys=True)
        )


def _report_semantics(case: _CaseT6, evidence: RuntimeEvidenceBundleV2) -> bool:
    report = _first_action(
        evidence,
        kind="commerce.send_message",
        sender=_MERCHANT_ID,
        recipient=case.buyer_ids[0],
    )
    payload = _action_payload(report)
    raw = payload.get("states")
    states = (
        tuple(dict(row) for row in raw if isinstance(row, Mapping)) if isinstance(raw, list) else ()
    )
    return (
        payload.get("category") == "inventory_eta_report"
        and states == _expected_supply_states(case)
    )


_T6_SCORE_BEARING_ACTION_KINDS = frozenset(
    {
        "commerce.allocate_fulfillment",
        "commerce.read_shipment",
        "commerce.read_supply_state",
        "commerce.resolve_shipment",
        "commerce.send_message",
        "delegate.reject_purchase",
        "platform.settle_payment",
    }
)
_T6_SCORE_BEARING_MODEL_INTENTS = frozenset(
    {
        "allocate_fulfillment",
        "read_shipment",
        "read_supply_state",
        "refund_shipment",
        "reject_purchase",
        "replace_shipment",
        "send_message",
        "settle_payment",
        "wait_for_shipment",
    }
)


def _t6_score_bearing_wire_action(
    case: _CaseT6,
    envelope: Mapping[str, Any],
) -> bool:
    """Exclude scenario kickoffs while retaining every evaluated model action."""

    action = envelope.get("action")
    kind = action.get("kind") if isinstance(action, Mapping) else None
    destination = envelope.get("to")
    if (
        envelope.get("from") != case.evaluated_actor_id
        or kind not in _T6_SCORE_BEARING_ACTION_KINDS
    ):
        return False
    if kind == "commerce.read_supply_state":
        return destination == "platform:supply"
    if kind in {
        "commerce.allocate_fulfillment",
        "commerce.read_shipment",
        "commerce.resolve_shipment",
    }:
        return destination == "platform:fulfillment"
    if kind == "platform.settle_payment":
        return destination == "platform:psp"
    if kind == "delegate.reject_purchase":
        return isinstance(destination, str) and destination.startswith("consumer:")
    return kind == "commerce.send_message" and destination in set(case.buyer_ids)


def _t6_model_text_matches_wire(value: Any, wire_text: Any) -> bool:
    if not isinstance(wire_text, str):
        return False
    if isinstance(value, str):
        return value == wire_text
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"text_chars", "text_sha256"}
        and value.get("text_chars") == len(wire_text)
        and value.get("text_sha256") == hashlib.sha256(wire_text.encode("utf-8")).hexdigest()
    )


def _t6_model_choice_matches_wire(
    case: _CaseT6,
    choice: VerifiedModelBusinessChoice,
    envelope: Mapping[str, Any],
) -> bool:
    """Prove that Agent routing preserved the model's T6 business parameters."""

    action = envelope.get("action")
    action_kind = action.get("kind") if isinstance(action, Mapping) else None
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if not (
        isinstance(action_kind, str)
        and isinstance(payload, Mapping)
        and choice.emitted_msg_id == envelope.get("msg_id")
        and choice.action_kind == action_kind
        and choice.destination == envelope.get("to")
    ):
        return False
    arguments = choice.arguments

    if action_kind == "commerce.read_supply_state":
        sku_ids = payload.get("sku_ids")
        return bool(
            choice.intent == "read_supply_state"
            and set(arguments) == {"sku_refs"}
            and isinstance(sku_ids, list)
            and arguments.get("sku_refs")
            == [public_reference_alias_v1(str(sku_id)) for sku_id in sku_ids]
        )
    if action_kind == "commerce.read_shipment":
        shipment_id = payload.get("shipment_id")
        return bool(
            choice.intent == "read_shipment"
            and set(arguments) == {"shipment_ref"}
            and isinstance(shipment_id, str)
            and arguments.get("shipment_ref") == public_reference_alias_v1(shipment_id)
        )
    if action_kind == "platform.settle_payment":
        sku_id = payload.get("sku_id")
        return bool(
            choice.intent == "settle_payment"
            and set(arguments) == {"sku_ref", "qty", "allow_partial"}
            and isinstance(sku_id, str)
            and arguments.get("sku_ref") == public_reference_alias_v1(sku_id)
            and arguments.get("qty") == payload.get("qty")
            and arguments.get("allow_partial") == payload.get("allow_partial")
        )
    if action_kind == "delegate.reject_purchase":
        return bool(
            choice.intent == "reject_purchase"
            and set(arguments) == {"reason"}
            and _t6_model_text_matches_wire(arguments.get("reason"), payload.get("reason"))
        )
    if action_kind == "commerce.allocate_fulfillment":
        sku_id = payload.get("sku_id")
        order_ids = payload.get("priority_order_ids")
        return bool(
            choice.intent == "allocate_fulfillment"
            and set(arguments) == {"sku_ref", "priority_order_refs"}
            and isinstance(sku_id, str)
            and isinstance(order_ids, list)
            and arguments.get("sku_ref") == public_reference_alias_v1(sku_id)
            and arguments.get("priority_order_refs")
            == [public_reference_alias_v1(str(order_id)) for order_id in order_ids]
        )
    if action_kind == "commerce.resolve_shipment":
        resolution = payload.get("resolution")
        expected_intent = {
            "wait": "wait_for_shipment",
            "replacement": "replace_shipment",
            "refund": "refund_shipment",
        }.get(resolution)
        if choice.intent != expected_intent:
            return False
        if resolution != "replacement":
            return not arguments
        replacement_sku_id = payload.get("replacement_sku_id")
        return bool(
            set(arguments) == {"replacement_sku_ref"}
            and isinstance(replacement_sku_id, str)
            and arguments.get("replacement_sku_ref")
            == public_reference_alias_v1(replacement_sku_id)
        )
    if action_kind == "commerce.send_message":
        if choice.intent != "send_message":
            return False
        # Allocation lanes expose a legal, deliberately non-scoring generic
        # message alternative with no model parameters.  Its payload is bound
        # by Agent authority; report lanes instead keep every model state fact.
        if not arguments:
            return case.lane not in {
                "merchant_inventory_eta",
                "merchant_restock_price",
            }
        raw_states = payload.get("states")
        if set(arguments) != {"states"} or not isinstance(raw_states, list):
            return False
        public_states = []
        for row in raw_states:
            if not isinstance(row, Mapping) or not isinstance(row.get("sku_id"), str):
                return False
            public_states.append(
                {
                    "sku_ref": public_reference_alias_v1(str(row["sku_id"])),
                    "available_qty": row.get("available_qty"),
                    "reserved_qty": row.get("reserved_qty"),
                    "eta_day": row.get("eta_day"),
                    "unit_price_cents": row.get("unit_price_cents"),
                }
            )
        return bool(
            payload.get("category") == "inventory_eta_report"
            and arguments.get("states") == public_states
        )
    return False


def _require_t6_model_compilation_integrity(
    case: _CaseT6,
    evidence: RuntimeEvidenceBundleV2,
) -> None:
    """Exact-join every scored T6 wire action to the originating model choice."""

    try:
        choices = verified_model_business_choices(
            evidence,
            evaluated_actor_id=case.evaluated_actor_id,
        )
    except TrackerEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError(
            "T6 model business-choice provenance is invalid"
        ) from exc
    choice_by_msg_id = {
        row.emitted_msg_id: row
        for row in choices
        if row.intent in _T6_SCORE_BEARING_MODEL_INTENTS
        or row.action_kind in _T6_SCORE_BEARING_ACTION_KINDS
    }
    actions = tuple(row for row in evidence.envelopes if _t6_score_bearing_wire_action(case, row))
    action_by_msg_id = {
        str(row.get("msg_id")): row for row in actions if isinstance(row.get("msg_id"), str)
    }
    if len(action_by_msg_id) != len(actions) or set(action_by_msg_id) != set(choice_by_msg_id):
        raise RuntimeBenchmarkIntegrityError(
            "T6 score-bearing wire actions do not exactly match model choices"
        )
    mismatched = sorted(
        msg_id
        for msg_id, action in action_by_msg_id.items()
        if not _t6_model_choice_matches_wire(case, choice_by_msg_id[msg_id], action)
    )
    if mismatched:
        raise RuntimeBenchmarkIntegrityError(
            "T6 Agent compiler drifted from public model arguments for message(s): "
            + ", ".join(mismatched)
        )


def _decision_semantics(
    case: _CaseT6,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedSupplyFulfillmentEvidence,
) -> tuple[bool, dict[str, Any]]:
    actor = case.evaluated_actor_id
    rejected = verified.rejections_for(actor_id=actor)

    def verified_request(kind: str) -> dict[str, Any] | None:
        rows = verified.operations_for(action_kind=kind, actor_id=actor)
        return rows[0].exchange.request if len(rows) == 1 else None

    # World-changing decisions receive semantic credit only after the shared
    # evidence contract has joined them to an accepted Platform decision and a
    # unique authoritative World transaction.
    settle = verified_request("platform.settle_payment")
    resolve = verified_request("commerce.resolve_shipment")
    allocate = verified_request("commerce.allocate_fulfillment")
    decline = _first_action(
        evidence,
        kind="delegate.reject_purchase",
        sender=actor,
    )
    settle_payload = _action_payload(settle)
    resolve_payload = _action_payload(resolve)
    allocate_payload = _action_payload(allocate)

    if case.lane == "buyer_partial_backorder":
        if case.expected_decision == "decline":
            ok = decline is not None and settle is None
        else:
            ok = (
                settle_payload.get("sku_id") == case.expected_sku_id
                and settle_payload.get("qty") == case.requested_qty
                and settle_payload.get("allow_partial") is True
            )
    elif case.lane == "buyer_stock_substitution":
        ok = (
            settle_payload.get("sku_id") == case.expected_sku_id
            and settle_payload.get("qty") == case.requested_qty
            and settle_payload.get("allow_partial") is False
        )
    elif case.lane.endswith("delivery_exception"):
        ok = (
            resolve_payload.get("shipment_id") == case.shipment_id
            and resolve_payload.get("resolution") == case.expected_resolution
            and (
                resolve_payload.get("replacement_sku_id") == case.replacement_sku_id
                if case.expected_resolution == "replacement"
                else "replacement_sku_id" not in resolve_payload
            )
        )
    elif case.lane == "buyer_restock_price":
        if case.expected_decision == "select":
            ok = (
                settle_payload.get("sku_id") == case.expected_sku_id
                and settle_payload.get("qty") == case.requested_qty
                and settle_payload.get("allow_partial") is False
                and decline is None
            )
        else:
            ok = decline is not None and settle is None
    elif case.lane in {"merchant_inventory_eta", "merchant_restock_price"}:
        ok = _report_semantics(case, evidence)
    else:
        ok = allocate_payload.get("sku_id") == case.expected_sku_id and allocate_payload.get(
            "priority_order_ids"
        ) == list(case.priority_order_ids)
    return bool(ok and not rejected), {
        "lane": case.lane,
        "settle": settle_payload,
        "resolve": resolve_payload,
        "allocate": allocate_payload,
        "decline": decline is not None,
        "rejected_actions": [
            {
                "action_kind": row.decision.get("action_kind"),
                "reason_code": row.decision.get("reason_code"),
                "error_type": row.decision.get("error_type"),
            }
            for row in rejected
        ],
    }


def _allocation_world_outcome(
    case: _CaseT6,
    evidence: RuntimeEvidenceBundleV2,
) -> bool:
    fulfillments = {
        str(row.get("order_id")): row for row in _table(evidence.final_world, "fulfillments")
    }
    orders = {str(row.get("order_id")): row for row in _table(evidence.final_world, "orders")}
    ledger = _table(evidence.final_world, "ledger")
    inventory = _inventory(evidence.final_world).get(str(case.expected_sku_id), {})
    expected = (
        case.expected_allocations
        if case.expected_allocations
        else (case.expected_fulfill,)
        if case.lane == "merchant_partial_backorder"
        else case.expected_allocations
    )
    if len(fulfillments) != len(case.priority_order_ids):
        return False
    total = 0
    for order_id, fulfilled in zip(case.priority_order_ids, expected, strict=True):
        row = fulfillments.get(order_id)
        order = orders.get(order_id)
        requested = (
            case.requested_quantities[
                case.priority_order_ids.index(order_id)
            ]
            if case.requested_quantities
            else case.requested_qty
            if case.lane == "merchant_partial_backorder"
            else 2
        )
        backordered = requested - fulfilled
        expected_state = (
            "backordered" if fulfilled == 0 else "partially_settled" if backordered else "settled"
        )
        if (
            row is None
            or order is None
            or row.get("requested_qty") != requested
            or row.get("fulfilled_qty") != fulfilled
            or row.get("backordered_qty") != backordered
            or order.get("state") != expected_state
        ):
            return False
        total += fulfilled
        matching_receipts = [receipt for receipt in ledger if receipt.get("order_id") == order_id]
        if len(matching_receipts) != (1 if fulfilled else 0):
            return False
        if fulfilled and matching_receipts[0].get("qty") != fulfilled:
            return False
    return int(inventory.get("qty_reserved", -1)) == total


def _buyer_settlement_world_outcome(
    case: _CaseT6,
    evidence: RuntimeEvidenceBundleV2,
) -> bool:
    assert case.expected_sku_id is not None
    option_matches = [
        option
        for response in evidence.actions(kind="platform.supply_state")
        for option in (
            (response.get("action") or {}).get("payload", {}).get("purchase_options", [])
        )
        if isinstance(option, Mapping) and option.get("sku_id") == case.expected_sku_id
    ]
    if not option_matches:
        return False
    orders = {str(row.get("order_id")): row for row in _table(evidence.final_world, "orders")}
    all_ledger = _table(evidence.final_world, "ledger")
    all_fulfillments = _table(evidence.final_world, "fulfillments")
    settled_requests = evidence.actions(
        kind="platform.settle_payment",
        actor_id=_BUYER_ID,
    )
    if case.expected_decision == "decline":
        option_order_ids = {
            str(option.get("order_id"))
            for option in option_matches
            if isinstance(option.get("order_id"), str) and option.get("order_id")
        }
        return (
            not settled_requests
            and not (set(orders) & option_order_ids)
            and not any(row.get("order_id") in option_order_ids for row in all_ledger)
            and not any(row.get("order_id") in option_order_ids for row in all_fulfillments)
        )
    if len(settled_requests) != 1:
        return False
    settled_payload = (settled_requests[0].get("action") or {}).get("payload")
    if not isinstance(settled_payload, Mapping):
        return False
    selected_options = [
        option
        for option in option_matches
        if settled_payload.get("supply_authority_id") == option.get("authority_id")
        and settled_payload.get("supply_authority_digest") == option.get("authority_digest")
        and settled_payload.get("sku_id") == case.expected_sku_id
    ]
    if len(selected_options) != 1:
        return False
    option = selected_options[0]
    order_id = option.get("order_id")
    if not isinstance(order_id, str) or not order_id:
        return False
    ledger = [row for row in all_ledger if row.get("order_id") == order_id]
    fulfillment = next(
        (row for row in all_fulfillments if row.get("order_id") == order_id),
        None,
    )
    order = orders.get(order_id)
    if order is None or order.get("sku_id") != case.expected_sku_id:
        return False
    if case.lane == "buyer_partial_backorder":
        return bool(
            order.get("qty") == case.requested_qty
            and order.get("state") == "partially_settled"
            and fulfillment is not None
            and fulfillment.get("requested_qty") == case.requested_qty
            and fulfillment.get("fulfilled_qty") == case.expected_fulfill
            and fulfillment.get("backordered_qty") == case.expected_backorder
            and len(ledger) == 1
            and ledger[0].get("qty") == case.expected_fulfill
        )
    inventory = _inventory(evidence.final_world).get(str(case.expected_sku_id), {})
    expected_qty = (
        case.requested_qty
        if case.lane in {"buyer_stock_substitution", "buyer_restock_price"}
        else 1
    )
    return bool(
        order.get("qty") == expected_qty
        and order.get("state") == "settled"
        and len(ledger) == 1
        and ledger[0].get("qty") == expected_qty
        and int(inventory.get("qty_reserved", -1)) >= expected_qty
    )


def _shipment_world_outcome(
    case: _CaseT6,
    evidence: RuntimeEvidenceBundleV2,
) -> bool:
    shipment = next(
        (
            row
            for row in _table(evidence.final_world, "shipments")
            if row.get("shipment_id") == case.shipment_id
        ),
        None,
    )
    if shipment is None:
        return False
    if (
        shipment.get("resolution") != case.expected_resolution
        or shipment.get("version") != case.event_count + 2
    ):
        return False
    if case.expected_resolution == "wait":
        return shipment.get("replacement_sku_id") is None
    if shipment.get("replacement_sku_id") != case.replacement_sku_id:
        return False
    original = _inventory(evidence.final_world).get(str(case.expected_sku_id), {})
    replacement = _inventory(evidence.final_world).get(str(case.replacement_sku_id), {})
    return (
        original.get("qty_available") == 0
        and original.get("qty_reserved") == 0
        and replacement.get("qty_reserved") == 1
    )


def _supply_only_world_outcome(
    case: _CaseT6,
    evidence: RuntimeEvidenceBundleV2,
) -> bool:
    expected = _expected_supply_states(case)
    inventory = _inventory(evidence.final_world)
    catalog = _catalog(evidence.final_world)
    for state in expected:
        sku_id = str(state["sku_id"])
        row = inventory.get(sku_id)
        listing = catalog.get(sku_id)
        if row is None or listing is None:
            return False
        if (
            int(row.get("qty_available", -1)) - int(row.get("qty_reserved", -1))
            != int(state["available_qty"])
            or int(row.get("qty_reserved", -1)) != int(state["reserved_qty"])
            or int(row.get("eta_day", -1)) != int(state["eta_day"])
            or int(row.get("version", -1)) != int(state["version"])
            or _snapshot_money_cents(listing.get("list_price")) != int(state["unit_price_cents"])
        ):
            return False
    return True


def _world_outcome(case: _CaseT6, evidence: RuntimeEvidenceBundleV2) -> bool:
    if case.lane in {
        "buyer_partial_backorder",
        "buyer_stock_substitution",
        "buyer_restock_price",
    }:
        return _buyer_settlement_world_outcome(case, evidence)
    if case.lane.endswith("delivery_exception"):
        return _shipment_world_outcome(case, evidence)
    if case.lane in {
        "merchant_competing_commitment",
        "merchant_partial_backorder",
    }:
        return _allocation_world_outcome(case, evidence)
    return _supply_only_world_outcome(case, evidence)


def _conservation(evidence: RuntimeEvidenceBundleV2) -> tuple[bool, dict[str, Any]]:
    inventory = _inventory(evidence.final_world)
    fulfillments = _table(evidence.final_world, "fulfillments")
    ledger = {str(row.get("txn_id")): row for row in _table(evidence.final_world, "ledger")}
    inventory_ok = all(
        isinstance(row.get("qty_available"), int)
        and isinstance(row.get("qty_reserved"), int)
        and 0 <= int(row["qty_reserved"]) <= int(row["qty_available"])
        for row in inventory.values()
    )
    allocations_ok = True
    for row in fulfillments:
        requested = row.get("requested_qty")
        fulfilled = row.get("fulfilled_qty")
        backordered = row.get("backordered_qty")
        receipt_id = row.get("receipt_txn_id")
        if not all(isinstance(value, int) for value in (requested, fulfilled, backordered)):
            allocations_ok = False
            break
        if requested != fulfilled + backordered or min(fulfilled, backordered) < 0:
            allocations_ok = False
            break
        if (fulfilled > 0) != (str(receipt_id) in ledger):
            allocations_ok = False
            break
    sequences = [row.get("sequence") for row in evidence.world_events]
    commits_ok = len(sequences) == len(set(sequences)) and all(
        row.get("invariants_held") for row in evidence.world_events
    )
    manifest_ok = evidence.evidence_manifest is not None
    return inventory_ok and allocations_ok and commits_ok and manifest_ok, {
        "inventory_rows": len(inventory),
        "fulfillment_rows": len(fulfillments),
        "world_commit_count": len(evidence.world_events),
        "inventory_nonnegative": inventory_ok,
        "fulfillment_conservation": allocations_ok,
        "commit_invariants_present": commits_ok,
        "verified_manifest": manifest_ok,
    }


def _hard_t6_decision_components(
    case: _CaseT6,
    evidence: RuntimeEvidenceBundleV2,
    decision_evidence: Mapping[str, Any],
) -> tuple[tuple[str, float, dict[str, Any]], ...]:
    """Decompose hard-tier decisions without changing their business goal."""

    settle = decision_evidence.get("settle")
    settle_payload = dict(settle) if isinstance(settle, Mapping) else {}
    resolve = decision_evidence.get("resolve")
    resolve_payload = dict(resolve) if isinstance(resolve, Mapping) else {}
    allocate = decision_evidence.get("allocate")
    allocate_payload = dict(allocate) if isinstance(allocate, Mapping) else {}
    declined = decision_evidence.get("decline") is True

    if case.lane in {"merchant_inventory_eta", "merchant_restock_price"}:
        report = _first_action(
            evidence,
            kind="commerce.send_message",
            sender=_MERCHANT_ID,
            recipient=case.buyer_ids[0],
        )
        payload = _action_payload(report)
        raw_states = payload.get("states")
        states = (
            [dict(row) for row in raw_states if isinstance(row, Mapping)]
            if isinstance(raw_states, list)
            else []
        )
        expected = list(_expected_supply_states(case))
        expected_by_sku = {str(row["sku_id"]): row for row in expected}
        observed_by_sku = {
            str(row.get("sku_id", "")): row for row in states
            if isinstance(row.get("sku_id"), str)
        }
        target_credit = len(set(expected_by_sku) & set(observed_by_sku)) / max(
            len(expected_by_sku),
            1,
        )
        fields = ("available_qty", "reserved_qty", "eta_day", "unit_price_cents")
        matched_cells = sum(
            observed_by_sku.get(sku_id, {}).get(field) == row.get(field)
            for sku_id, row in expected_by_sku.items()
            for field in fields
        )
        parameter_credit = matched_cells / max(len(expected_by_sku) * len(fields), 1)
        action_credit = float(
            payload.get("category") == "inventory_eta_report"
            and isinstance(raw_states, list)
        )
        detail = {
            "expected_skus": sorted(expected_by_sku),
            "observed_skus": sorted(observed_by_sku),
            "matched_field_count": matched_cells,
            "expected_field_count": len(expected_by_sku) * len(fields),
        }
    elif case.lane.endswith("delivery_exception"):
        action_credit = float(bool(resolve_payload))
        target_credit = float(
            resolve_payload.get("shipment_id") == case.shipment_id
        )
        resolution_ok = resolve_payload.get("resolution") == case.expected_resolution
        replacement_ok = (
            resolve_payload.get("replacement_sku_id") == case.replacement_sku_id
            if case.expected_resolution == "replacement"
            else "replacement_sku_id" not in resolve_payload
        )
        parameter_credit = (float(resolution_ok) + float(replacement_ok)) / 2
        detail = {
            "expected_resolution": case.expected_resolution,
            "observed_resolution": resolve_payload.get("resolution"),
            "replacement_matches": replacement_ok,
        }
    elif case.lane in {
        "merchant_competing_commitment",
        "merchant_partial_backorder",
    }:
        action_credit = float(bool(allocate_payload))
        target_credit = float(
            allocate_payload.get("sku_id") == case.expected_sku_id
        )
        observed_order_ids = allocate_payload.get("priority_order_ids")
        observed = (
            tuple(str(value) for value in observed_order_ids)
            if isinstance(observed_order_ids, list)
            else ()
        )
        positional_matches = sum(
            observed_value == expected_value
            for observed_value, expected_value in zip(
                observed,
                case.priority_order_ids,
                strict=False,
            )
        )
        parameter_credit = positional_matches / max(len(case.priority_order_ids), 1)
        detail = {
            "expected_priority_order_ids": list(case.priority_order_ids),
            "observed_priority_order_ids": list(observed),
            "positional_match_count": positional_matches,
        }
    elif case.expected_decision == "decline":
        action_credit = float(declined or bool(settle_payload))
        target_credit = float(declined and not settle_payload)
        parameter_credit = float(declined and not settle_payload)
        detail = {
            "expected_decision": "decline",
            "declined": declined,
            "settlement_attempted": bool(settle_payload),
        }
    else:
        action_credit = float(bool(settle_payload))
        target_credit = float(
            settle_payload.get("sku_id") == case.expected_sku_id
        )
        expected_qty = (
            case.requested_qty
            if case.lane
            in {
                "buyer_partial_backorder",
                "buyer_stock_substitution",
                "buyer_restock_price",
            }
            else 1
        )
        quantity_ok = settle_payload.get("qty") == expected_qty
        partial_ok = (
            settle_payload.get("allow_partial") is True
            if case.lane == "buyer_partial_backorder"
            else settle_payload.get("allow_partial") is False
        )
        parameter_credit = (float(quantity_ok) + float(partial_ok)) / 2
        detail = {
            "expected_qty": expected_qty,
            "observed_qty": settle_payload.get("qty"),
            "partial_policy_matches": partial_ok,
        }

    return (
        ("decision_action_selection", action_credit, {
            **detail,
            "component": "action_selection",
        }),
        ("decision_target_grounding", target_credit, {
            **detail,
            "component": "target_grounding",
        }),
        ("decision_parameter_correctness", parameter_credit, {
            **detail,
            "component": "parameter_correctness",
        }),
    )


def _score_t6(case: _CaseT6, evidence: RuntimeEvidenceBundleV2) -> RuntimeTaskScoreV3:
    require_runtime_benchmark_integrity_v2(evidence)
    require_frozen_scenario_fixture_v2(
        evidence,
        scenario_for_t6(str(case.definition.task_id)),
        family="T6",
    )
    _require_t6_model_compilation_integrity(case, evidence)
    discovery = verify_optional_discovery_prefix_v2(
        evidence,
        buyer_id=_BUYER_ID,
    )
    read_kind = (
        "commerce.read_shipment"
        if case.lane.endswith("delivery_exception")
        else "commerce.read_supply_state"
    )
    verified: VerifiedSupplyFulfillmentEvidence | None = None
    verification_error: str | None = None
    try:
        result = evidence.verified_operation_evidence(
            SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
            options={
                "allow_rejected": True,
                "evaluated_actor_id": case.evaluated_actor_id,
                "expected_read_kind": read_kind,
                "preclaimed_commit_ids": discovery.commit_ids,
            },
        )
        if not isinstance(result, VerifiedSupplyFulfillmentEvidence):
            raise RuntimeEvidenceError(
                "supply and fulfillment evidence contract returned wrong type"
            )
        verified = result
    except RuntimeEvidenceError as exc:
        verification_error = str(exc)

    read_exchanges = (
        verified.reads_for(
            action_kind=read_kind,
            actor_id=case.evaluated_actor_id,
        )
        if verified is not None
        else ()
    )
    grounding_verifier = (
        _shipment_grounding if case.lane.endswith("delivery_exception") else _supply_grounding
    )
    grounding_results = [
        grounding_verifier(
            case,
            row.exchange.request,
            row.response,
        )
        for row in read_exchanges
    ]
    _require_t6_read_fixture_integrity(grounding_results)
    matching_read_indexes = [
        index
        for index, (matches_expected_state, _evidence) in enumerate(grounding_results)
        if matches_expected_state
    ]
    if matching_read_indexes:
        selected_read_index = matching_read_indexes[0]
    elif read_exchanges:
        selected_read_index = len(read_exchanges) - 1
    else:
        selected_read_index = None
    request = (
        read_exchanges[selected_read_index].exchange.request
        if selected_read_index is not None
        else None
    )
    if selected_read_index is not None:
        _selected_matches_expected_state, grounding_evidence = grounding_results[
            selected_read_index
        ]
        grounding_ok = bool(matching_read_indexes)
    else:
        grounding_ok, grounding_evidence = grounding_verifier(case, None, None)
    grounding_evidence.update(
        {
            "authoritative_read_count": len(read_exchanges),
            "matching_read_count": len(matching_read_indexes),
            "selected_request_msg_id": (
                request.get("msg_id") if isinstance(request, Mapping) else None
            ),
        }
    )

    if case.lane.endswith("delivery_exception"):
        event_operation = "record_shipment_status"
        event_action = "platform.record_shipment_status"
        authority_action = "world.record_shipment_status"
        event_actor = "runtime:logistics"
    else:
        event_operation = "apply_supply_event"
        event_action = "platform.apply_supply_event"
        authority_action = "world.apply_supply_event"
        event_actor = "runtime:supply"
    event_operations = (
        verified.operations_for(action_kind=event_action, actor_id=event_actor)
        if verified is not None
        else ()
    )
    event_requests = tuple(operation.exchange.request for operation in event_operations)
    event_commits = [operation.commit for operation in event_operations]
    events_ok = (
        verified is not None
        and len(event_requests) == case.event_count
        and len(event_commits) == case.event_count
        and all(
            row.get("operation") == event_operation
            and row.get("authority_action") == authority_action
            for row in event_commits
        )
    )
    if verified is None:
        decision_ok = False
        decision_evidence = {
            "lane": case.lane,
            "exact_evidence_error": verification_error,
        }
    else:
        decision_ok, decision_evidence = _decision_semantics(case, evidence, verified)
    world_ok = verified is not None and _world_outcome(case, evidence)
    conservation_ok, conservation_evidence = _conservation(evidence)
    conservation_evidence["exact_evidence_contract"] = SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT
    conservation_evidence["exact_evidence_verified"] = verified is not None
    conservation_evidence["exact_evidence_error"] = verification_error

    if not events_ok:
        raise RuntimeBenchmarkIntegrityError(
            "T6 exogenous supply or logistics events are not exactly Platform/World committed"
        )
    if not conservation_ok:
        raise RuntimeBenchmarkIntegrityError(
            "T6 World inventory, allocation, or commit conservation invariant failed"
        )
    if decision_ok and not world_ok:
        raise RuntimeBenchmarkIntegrityError(
            "T6 World outcome does not faithfully execute the verified agent decision"
        )

    supply_report_lane = case.lane in {
        "merchant_inventory_eta",
        "merchant_restock_price",
    }
    if supply_report_lane and not world_ok:
        raise RuntimeBenchmarkIntegrityError(
            "T6 deterministic supply events are not faithfully projected in World"
        )

    if is_hardened_task_v2(case.definition):
        components = _hard_t6_decision_components(case, evidence, decision_evidence)
        checks = (
            RuntimeRubricCheckV2(
                "supply_state_evidence_coverage",
                0.30,
                float(grounding_ok),
                grounding_evidence,
            ),
            RuntimeRubricCheckV2(
                "decision_semantics",
                0.55,
                float(decision_ok),
                decision_evidence,
            ),
            *(
                RuntimeRubricCheckV2(name, 0.05, credit, detail)
                for name, credit, detail in components
            ),
        )
    else:
        checks = (
            RuntimeRubricCheckV2(
                "supply_state_evidence_coverage",
                0.40,
                float(grounding_ok),
                grounding_evidence,
            ),
            RuntimeRubricCheckV2(
                "decision_semantics",
                0.60,
                float(decision_ok),
                decision_evidence,
            ),
        )
    return score_checks(
        case.definition,
        renormalize_capability_checks_v2(checks),
    )


def _mutation_targets(case: _CaseT6) -> tuple[str, ...]:
    if not is_hardened_task_v2(case.definition):
        return ("decision_semantics",)
    if case.lane in {
        "buyer_delivery_exception",
        "merchant_delivery_exception",
        "merchant_inventory_eta",
        "merchant_restock_price",
    }:
        return ("decision_semantics", "decision_parameter_correctness")
    if case.lane == "buyer_partial_backorder":
        return (
            "decision_semantics",
            "decision_target_grounding",
            "decision_parameter_correctness",
        )
    if case.lane == "buyer_stock_substitution":
        return ("decision_semantics", "decision_target_grounding")
    if case.lane == "buyer_restock_price":
        return (
            "decision_semantics",
            "decision_action_selection",
            "decision_target_grounding",
            "decision_parameter_correctness",
        )
    return (
        "decision_semantics",
        "decision_action_selection",
        "decision_target_grounding",
        "decision_parameter_correctness",
    )


def runtime_bundle_t6(task_id: str) -> RuntimeTaskBundleV2:
    """Bind one T6 data configuration to the real CommerceWorld Episode."""

    case = _case_for_t6(task_id)
    counterparts: dict[str, Any]
    if case.definition.evaluated_role == "buyer":
        counterparts = {_MERCHANT_ID: lambda: _UnexpectedT6Channel()}
    else:
        counterparts = {buyer_id: (lambda: _UnexpectedT6Channel()) for buyer_id in case.buyer_ids}
    return RuntimeTaskBundleV2(
        task=case.definition,
        scenario=scenario_for_t6(task_id),
        evaluated_actor_id=case.evaluated_actor_id,
        ideal_channel=lambda: _T6Channel(case, mutated=False),
        counterpart_channels=counterparts,
        scorer=lambda evidence: _score_t6(case, evidence),
        semantic_hash=canonical_sha256(case.semantic_contract),
        mutations=(
            RuntimeMutationV2(
                mutation_id=f"{task_id}:partial:01",
                channel=lambda: targeted_mutation_channel_t6(task_id),
                expected_changed_checks=_mutation_targets(case),
            ),
        ),
    )


def runtime_bundles_t6() -> tuple[RuntimeTaskBundleV2, ...]:
    return tuple(runtime_bundle_t6(task_id) for task_id in _T6_TASK_IDS)


__all__ = [
    "T6_RUNTIME_SCHEMA_V2",
    "ideal_business_decision_t6",
    "runtime_bundle_t6",
    "runtime_bundles_t6",
    "scenario_for_t6",
    "targeted_mutation_channel_t6",
]
