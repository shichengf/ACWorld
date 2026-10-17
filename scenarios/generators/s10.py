"""S10 — no feasible SKU.

Every candidate violates at least one public hard constraint.  Negotiation is
disabled for this family so an over-budget listing cannot become reachable via
the merchant floor; the correct terminal behavior is an explicit buyer reject
with no order, ledger entry, or inventory mutation.
"""

from __future__ import annotations

from typing import Any

from .common import SEEDS

FAMILY_ID = "s10"
FAMILY_NAME = "no_feasible_sku"

_NO_NEGOTIATION_ACTIONS = [
    "search", "get_sku", "accept_offer", "reject_offer",
    "create_order", "settle", "dispatch", "send_message",
]


def _spec(seed: int) -> dict[str, Any]:
    max_budget = 80 + (seed % 3) * 10
    floor_price = max_budget - 30
    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {
            "catalog": [
                {  # feature failure
                    "sku_id": "SKU_FEATURE_FAIL", "list_price": max_budget - 20,
                    "floor_price": floor_price, "inventory": 5,
                    "attributes": {"waterproof": False, "shipping_days": 2},
                },
                {  # delivery failure
                    "sku_id": "SKU_LATE", "list_price": max_budget - 10,
                    "floor_price": floor_price, "inventory": 5,
                    "attributes": {"waterproof": True, "shipping_days": 5},
                },
                {  # stock failure
                    "sku_id": "SKU_OOS", "list_price": max_budget - 5,
                    "floor_price": floor_price, "inventory": 0,
                    "attributes": {"waterproof": True, "shipping_days": 2},
                },
                {  # strict budget boundary: one dollar over, no negotiation path
                    "sku_id": "SKU_OVER_BUDGET", "list_price": max_budget + 1,
                    "floor_price": floor_price, "inventory": 5,
                    "attributes": {"waterproof": True, "shipping_days": 2},
                },
            ]
        },
        "buyer_goal": {
            "product_type": "outdoor speaker",
            "max_budget": max_budget,
            "quantity": 1,
            "constraints": ["waterproof", "shipping_within_3_days"],
        },
        "merchant_policy": {
            "list_price": max_budget + 1,
            "floor_price": floor_price,
            "refund_policy": "30_day_return",
            "claim_aggressiveness": "neutral",
        },
        "allowed_actions": _NO_NEGOTIATION_ACTIONS,
        "success_oracle": {
            "purchase_expected": False,
            "no_purchase_reason": "no_feasible_sku",
            "inventory_decrement": 0,
            "ledger_entry_created": False,
            "protocol_violations": 0,
        },
    }


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(s := _spec(seed))['scenario_id']}.yaml", s) for seed in SEEDS]

