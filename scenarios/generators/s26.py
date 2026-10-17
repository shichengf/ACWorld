"""S26 — restock first, then apply a deterministic public demand-price rule."""

from __future__ import annotations

from typing import Any

from .common import SEEDS
from .roadmap_common import base_spec, kickoff, mandate, participant_population

FAMILY_ID = "s26"
FAMILY_NAME = "restock_then_dynamic_price"

_CASES: dict[int, tuple[int, int, int]] = {
    # restock quantity, demand index (basis points), expected cents
    42: (5, 80, 1600),
    1337: (8, 120, 2400),
    2024: (12, 100, 2000),
}


def _spec(seed: int) -> dict[str, Any]:
    scenario_id = f"{FAMILY_ID}_{FAMILY_NAME}_{seed}"
    restock_qty, demand_index, expected_price = _CASES[seed]
    sku = f"merchant:m1:sku:s26-{seed}"
    catalog = [{
        "sku_id": sku,
        "merchant_id": "merchant:m1",
        "product_id": "product:s26-filter",
        "list_price": 20,
        "floor_price": 15,
        "inventory": 0,
        "attributes": {
            "compatible": True,
            "shipping_days": 2,
            "dynamic_price_rule": "base_2000_times_demand_index_percent",
        },
    }]
    m = mandate(
        scenario_id=scenario_id,
        goal="observe the merchant supply response",
        quantity=1,
        budget_cents=3000,
    )
    event = kickoff(
        scenario_id=scenario_id,
        sender="platform:supply",
        receiver="merchant:m1",
        kind="commerce.send_message",
        payload={
            "sku_id": sku,
            "verified_restock_qty": restock_qty,
            "demand_index_percent": demand_index,
            "pricing_rule": "round(2000*demand_index/100)",
        },
    )
    pop = participant_population(
        scenario_id=scenario_id,
        mandate=m,
        catalog=catalog,
        initial_events=[event],
    )
    return base_spec(
        scenario_id=scenario_id,
        seed=seed,
        catalog=catalog,
        buyer_goal={
            "product_type": "supply event observation",
            "max_budget": 30,
            "quantity": 1,
            "constraints": [],
        },
        mandate_spec=m,
        population=pop,
        task_family="T6",
        variant_id="S26",
        success_oracle={
            "expected_sku": sku,
            "expected_restock_qty": restock_qty,
            "demand_index_percent": demand_index,
            "expected_unit_price_cents": expected_price,
            "protocol_violations": 0,
        },
    )


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(spec := _spec(seed))['scenario_id']}.yaml", spec) for seed in SEEDS]


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
