"""S5 — Return / refund.

Stresses: the buyer completes a purchase, then requests
a return. The merchant must comply or refuse strictly per its stated
``refund_policy``. Two instances use ``7_day_return`` (merchant must issue the
refund); one uses ``no_returns`` (merchant must decline, not silently refund).
``success_oracle.refund_issued`` is the server-side check on the correct
branch.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s5"
FAMILY_NAME = "return_refund"


def _spec(seed: int, index: int) -> dict[str, Any]:
    list_price = 60 + (seed % 3) * 10
    floor_price = list_price - 20
    max_budget = list_price + 10
    # First two instances allow returns; the third forbids them.
    allows_return = index < 2
    refund_policy = "7_day_return" if allows_return else "no_returns"
    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {
            "catalog": [
                {
                    "sku_id": "SKU_0001",
                    "list_price": list_price,
                    "floor_price": floor_price,
                    "inventory": 4,
                    "attributes": {"in_stock": True, "shipping_days": 2},
                }
            ]
        },
        "buyer_goal": {
            "product_type": "bluetooth speaker",
            "max_budget": max_budget,
            "quantity": 1,
            # Intent: BUY an in-stock item, then attempt a return. Returnability
            # is the MERCHANT's policy honored at return time (s5 tests comply vs
            # decline), NOT a purchase pre-req — so ``returnable`` is deliberately
            # NOT a must_have. Requiring it would make the ``no_returns`` instance
            # unbuyable (must_have unsatisfiable) and never reach the return step.
            "constraints": ["in_stock"],
            "return_after_purchase": True,
        },
        "merchant_policy": {
            "list_price": list_price,
            "floor_price": floor_price,
            "refund_policy": refund_policy,
            "claim_aggressiveness": "neutral",
        },
        "allowed_actions": ALLOWED_ACTIONS,
        "success_oracle": {
            "product_match": True,
            "final_price_lte": max_budget,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "return_requested": True,
            "refund_issued": allows_return,
            "ledger_entry_created": True,
            "protocol_violations": 0,
        },
    }


def generate() -> list[tuple[str, dict[str, Any]]]:
    out = []
    for index, seed in enumerate(SEEDS):
        spec = _spec(seed, index)
        out.append((f"{spec['scenario_id']}.yaml", spec))
    return out
