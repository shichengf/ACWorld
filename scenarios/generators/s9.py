"""S9 — multi-word material grounding.

Exactly one listing contains both words in the ``merino wool`` hard
constraint; the cheaper decoy uses a different, non-negated material.  Word
order, SKU assignment, catalog order, and prices vary across seeds while the
same deterministic phrase-matching contract remains under test.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s9"
FAMILY_NAME = "multiword_material"

_MATCHING = ("SKU_0001", "SKU_0002", "SKU_0001")
_CATALOG_ORDERS = (
    ("SKU_0001", "SKU_0002"),
    ("SKU_0001", "SKU_0002"),
    ("SKU_0002", "SKU_0001"),
)
_MATCHING_MATERIALS = (
    "wool, merino",
    "premium merino wool",
    "merino superfine wool",
)
_DECOY_MATERIALS = ("cotton", "bamboo viscose", "linen")


def _spec(seed: int, index: int) -> dict[str, Any]:
    matching = _MATCHING[index]
    decoy = "SKU_0002" if matching == "SKU_0001" else "SKU_0001"
    matching_price = (40, 44, 39)[index]
    decoy_price = matching_price - (8, 9, 6)[index]
    prices = {matching: matching_price, decoy: decoy_price}
    materials = {
        matching: _MATCHING_MATERIALS[index],
        decoy: _DECOY_MATERIALS[index],
    }

    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {
            "catalog": [
                {
                    "sku_id": sku_id,
                    "list_price": prices[sku_id],
                    "floor_price": prices[sku_id] - (12 if sku_id == matching else 10),
                    "inventory": 5 + index,
                    "attributes": {
                        "material": materials[sku_id],
                        "shipping_days": (2, 3, 1)[index],
                    },
                }
                for sku_id in _CATALOG_ORDERS[index]
            ]
        },
        "buyer_goal": {
            "product_type": "wool socks",
            "max_budget": matching_price + 10,
            "quantity": 1,
            "constraints": ["merino wool", "shipping_within_3_days"],
        },
        "merchant_policy": {
            "list_price": matching_price,
            "floor_price": matching_price - 12,
            "refund_policy": "30_day_return",
            "claim_aggressiveness": "neutral",
        },
        "allowed_actions": ALLOWED_ACTIONS,
        "success_oracle": {
            "product_match": True,
            "expected_sku": matching,
            "final_price_lte": matching_price + 10,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "ledger_entry_created": True,
            "protocol_violations": 0,
        },
        "benchmark": {
            "task_family": "T2",
            "variant_id": "S9",
            "track": "agent",
            "difficulty": "medium",
            "catalog_source": "inline",
            "catalog_scale": "smoke",
            "scenario_version": "1.0",
        },
    }


def generate() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for index, seed in enumerate(SEEDS):
        spec = _spec(seed, index)
        out.append((f"{spec['scenario_id']}.yaml", spec))
    return out
