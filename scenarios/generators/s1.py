"""S1 — Direct purchase.

Stresses: list-price acceptance, settlement, dispatch.
One SKU that satisfies every constraint, priced at or under the buyer's
budget — the happy path a deterministic stub agent must close cleanly.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s1"
FAMILY_NAME = "direct_purchase"

_PRODUCTS = ["wireless earbuds", "desk lamp", "mechanical keyboard"]


def _spec(seed: int) -> dict[str, Any]:
    list_price = 50 + (seed % 4) * 10
    floor_price = list_price - 25
    max_budget = list_price + 20
    inventory = 4 + seed % 3
    product = _PRODUCTS[seed % len(_PRODUCTS)]
    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {
            "catalog": [
                {
                    "sku_id": "SKU_0001",
                    "list_price": list_price,
                    "floor_price": floor_price,
                    "inventory": inventory,
                    "attributes": {"in_stock": True, "shipping_days": 2},
                }
            ]
        },
        "buyer_goal": {
            "product_type": product,
            "max_budget": max_budget,
            "quantity": 1,
            "constraints": ["in_stock", "shipping_within_3_days"],
        },
        "merchant_policy": {
            "list_price": list_price,
            "floor_price": floor_price,
            "refund_policy": "7_day_return",
            "claim_aggressiveness": "neutral",
        },
        "allowed_actions": ALLOWED_ACTIONS,
        "success_oracle": {
            "product_match": True,
            "final_price_lte": max_budget,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "ledger_entry_created": True,
            "protocol_violations": 0,
        },
    }


def generate() -> list[tuple[str, dict[str, Any]]]:
    out = []
    for seed in SEEDS:
        spec = _spec(seed)
        out.append((f"{spec['scenario_id']}.yaml", spec))
    return out
