"""S12 — exact structured grounding provenance.

The buyer must choose the only BPA-free bottle.  The answer is deliberately a
structured boolean attribute; neither the scenario nor the scorer asks an LLM
to infer facts from marketing prose.  The valid sku changes position across the
three seeds, preventing a fixed first-result policy from passing the corpus.

S12's headline lane additionally requires the settled decision to contain a
framework-executed listing-read result bound to that exact decision.  The HTTP
path cross-checks the same result against its audited ``world.response``.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s12"
FAMILY_NAME = "grounding_provenance"


def _spec(seed: int) -> dict[str, Any]:
    matching_number = 1 if seed % 2 == 0 else 2
    matching_sku = f"SKU_000{matching_number}"
    list_price = 32 + seed % 5
    floor_price = list_price - 10
    max_budget = list_price + 12

    catalog = []
    for number in (1, 2):
        sku_id = f"SKU_000{number}"
        catalog.append({
            "sku_id": sku_id,
            "list_price": list_price,
            "floor_price": floor_price,
            "inventory": 4,
            "attributes": {
                "bpa_free": sku_id == matching_sku,
                "food_grade_steel": True,
                "capacity_ml": 750,
                "shipping_days": 2,
            },
        })

    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {"catalog": catalog},
        "buyer_goal": {
            "product_type": "750ml stainless steel water bottle",
            "max_budget": max_budget,
            "quantity": 1,
            "constraints": ["bpa_free", "shipping_within_3_days"],
        },
        "merchant_policy": {
            "list_price": list_price,
            "floor_price": floor_price,
            "refund_policy": "30_day_return",
            "claim_aggressiveness": "neutral",
        },
        "allowed_actions": ALLOWED_ACTIONS,
        "success_oracle": {
            "product_match": True,
            "expected_sku": matching_sku,
            "final_price_lte": max_budget,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "ledger_entry_created": True,
            "protocol_violations": 0,
        },
        "benchmark": {
            "task_family": "T2",
            "variant_id": "S12",
            "track": "agent",
            "difficulty": "medium",
            "catalog_source": "inline",
            "catalog_scale": "smoke",
            "scenario_version": "1.0",
        },
    }


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(spec := _spec(seed))['scenario_id']}.yaml", spec) for seed in SEEDS]
