"""S7 — soft compromise.

Both listings satisfy every hard constraint and have the same public price.
Neither satisfies all soft preferences: the unique optimum keeps the
higher-importance preference and relaxes only the lower-importance one.  SKU
assignment and catalog order vary independently across seeds.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s7"
FAMILY_NAME = "soft_compromise"

_PREFERRED = ("SKU_0001", "SKU_0002", "SKU_0001")
_CATALOG_ORDERS = (
    ("SKU_0001", "SKU_0002"),
    ("SKU_0001", "SKU_0002"),
    ("SKU_0002", "SKU_0001"),
)


def _spec(seed: int, index: int) -> dict[str, Any]:
    preferred = _PREFERRED[index]
    list_price = (60, 64, 57)[index]
    floor_price = list_price - (20, 19, 17)[index]
    high_importance = (2, 4, 3)[index]
    low_importance = (1, 1, 2)[index]

    def attributes(sku_id: str) -> dict[str, Any]:
        keeps_high_value = sku_id == preferred
        return {
            "waterproof": True,
            "bluetooth": keeps_high_value,
            "lightweight": not keeps_high_value,
            "shipping_days": (2, 3, 1)[index],
        }

    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {
            "catalog": [
                {
                    "sku_id": sku_id,
                    "list_price": list_price,
                    "floor_price": floor_price,
                    "inventory": 5 + index,
                    "attributes": attributes(sku_id),
                }
                for sku_id in _CATALOG_ORDERS[index]
            ]
        },
        "buyer_goal": {
            "product_type": "outdoor speaker",
            "max_budget": list_price + 10,
            "quantity": 1,
            "constraints": ["waterproof", "shipping_within_3_days"],
            "soft_constraints": [
                {"feature": "bluetooth", "importance": high_importance},
                {"feature": "lightweight", "importance": low_importance},
            ],
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
            "expected_sku": preferred,
            "final_price_lte": list_price + 10,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "ledger_entry_created": True,
            "protocol_violations": 0,
        },
        "benchmark": {
            "task_family": "T3",
            "variant_id": "S7",
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
