"""S21 — exact quantity-tier arithmetic at the requested quantity."""

from __future__ import annotations

from typing import Any

from .common import SEEDS
from .roadmap_common import base_spec, mandate

FAMILY_ID = "s21"
FAMILY_NAME = "quantity_discount"


def _spec(seed: int) -> dict[str, Any]:
    scenario_id = f"{FAMILY_ID}_{FAMILY_NAME}_{seed}"
    qty, expected = {42: (4, 1800), 1337: (6, 1700), 2024: (10, 1500)}[seed]
    sku = f"merchant:m1:sku:s21-{seed}"
    catalog = [{
        "sku_id": sku,
        "merchant_id": "merchant:m1",
        "product_id": "product:s21-notebook",
        "list_price": 20,
        "floor_price": 14,
        "inventory": 20,
        "attributes": {
            "recycled_paper": True,
            "shipping_days": 3,
            "quantity_price_tiers": {"1": 2000, "3": 1800, "5": 1700, "10": 1500},
        },
    }]
    m = mandate(
        scenario_id=scenario_id,
        goal="buy the requested notebook quantity at the applicable public tier",
        quantity=qty,
        budget_cents=20_000,
        must_have=["recycled_paper"],
    )
    return base_spec(
        scenario_id=scenario_id,
        seed=seed,
        catalog=catalog,
        buyer_goal={
            "product_type": "recycled notebook",
            "max_budget": 200,
            "quantity": qty,
            "constraints": ["recycled_paper"],
        },
        mandate_spec=m,
        task_family="T5",
        variant_id="S21",
        success_oracle={
            "expected_sku": sku,
            "expected_qty": qty,
            "expected_unit_price_cents": expected,
            "protocol_violations": 0,
        },
    )


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(spec := _spec(seed))['scenario_id']}.yaml", spec) for seed in SEEDS]


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
