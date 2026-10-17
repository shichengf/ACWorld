"""Example: add a product domain with only a scenario generator registration."""

from __future__ import annotations

from typing import Any, Mapping

from episode.extensions import SCENARIO_GENERATORS


@SCENARIO_GENERATORS.decorator(
    "example.pet-supplies",
    description="Synthetic many-to-many pet-supplies market",
)
def generate(seed: int, parameters: Mapping[str, Any]) -> dict[str, Any]:
    buyers = int(parameters.get("buyers", 2))
    merchants = int(parameters.get("merchants", 2))
    if buyers <= 0 or merchants <= 0:
        raise ValueError("buyers and merchants must be positive")
    feasible = parameters.get("feasible", True)
    if not isinstance(feasible, bool):
        raise ValueError("feasible must be a boolean")
    catalog = []
    merchant_specs = []
    for index in range(merchants):
        merchant_id = f"merchant:pet{index:03d}"
        sku = f"{merchant_id}:bed"
        catalog.append(
            {
                "sku_id": sku,
                "product_id": "pet-bed",
                "merchant_id": merchant_id,
                "category": "pet-supplies",
                "name": f"Pet bed {index}",
                "list_price": 40 + index,
                "inventory": 2,
                "attributes": {"washable": feasible, "in_stock": True},
            }
        )
        merchant_specs.append(
            {
                "merchant_id": merchant_id,
                "persona": {"name": f"Pet merchant {index}"},
                "policy": {
                    "floor_price": 2500 + index * 10,
                    "refund_policy": "7_day_return",
                },
                "catalog_scope": [sku],
            }
        )
    buyer_specs = [
        {
            "buyer_id": f"buyer:pet{index:03d}",
            "persona": {"name": f"Pet buyer {index}"},
            "mandate": {
                "mandate_id": f"pet:{seed}:{index}",
                "goal": "washable pet bed",
                "quantity": 1,
                "hard_constraints": {"budget": 6000, "must_have": ["washable"]},
                "soft_constraints": [],
                "soft_preferences": {"style": [], "avoid": []},
                "authority": {
                    "can_buy_without_confirmation": True,
                    "must_not_share_with_merchant": ["budget"],
                },
                "intent_expiry": "2099-01-01T00:00:00Z",
            },
        }
        for index in range(buyers)
    ]
    return {
        "scenario_id": (
            f"s18_pet_supplies_{seed}" if feasible else f"s18_pet_supplies_infeasible_{seed}"
        ),
        "seed": seed,
        "initial_state": {"catalog": catalog},
        "buyer_goal": {
            "product_type": "pet bed",
            "max_budget": 60,
            "quantity": 1,
            "constraints": ["washable"],
        },
        "merchant_policy": {"floor_price": 25, "refund_policy": "7_day_return"},
        "allowed_actions": [
            "search",
            "propose_offer",
            "counter_offer",
            "accept_offer",
            "create_order",
            "settle",
        ],
        "success_oracle": {"product_match": True, "final_price_lte": 60},
        "population": {
            "buyers": buyer_specs,
            "merchants": merchant_specs,
            "matching": {"top_k": min(5, merchants)},
            "execution": {"max_transactions_per_buyer": 1},
        },
    }
