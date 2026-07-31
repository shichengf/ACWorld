"""S14 — authoritative return-window decisions.

The episode begins at a genuine after-sales boundary: World already contains a
paid, dispatched order, its inventory reservation, ledger receipt, captured
return policy, and logical timeline.  The evaluated buyer receives the normal
settlement receipt and must request a return.  The World—not an envelope or
model timestamp—then authorizes an early/boundary request or atomically rejects
the late request.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s14"
FAMILY_NAME = "return_window"

_CURRENT_TICKS = (3, 4, 5)
_BRANCHES = ("authorized", "authorized", "rejected_late")
_RECEIPT_TIMESTAMPS = (
    "2999-12-31T23:59:59Z",  # future wall clock cannot close an early window
    "1970-01-01T00:00:00Z",  # old wall clock cannot move the exact boundary
    "1900-01-01T00:00:00Z",  # old wall clock cannot rescue a late request
)


def _spec(seed: int, index: int) -> dict[str, Any]:
    price = 60 + index * 10
    floor = price - 20
    sku = f"merchant:m1:sku:return-window:{seed}"
    order_id = f"order:s14:{seed}"
    txn_id = f"settle:s14:{seed}"
    branch = _BRANCHES[index]
    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {
            "catalog": [{
                "sku_id": sku,
                "product_id": "product:return-window",
                "list_price": price,
                "floor_price": floor,
                "inventory": 4,
                "qty_reserved": 1,
                "attributes": {
                    "in_stock": True,
                    "shipping_days": 2,
                    "return_window_ticks": 3,
                },
            }],
            "orders": [{
                "order_id": order_id,
                "buyer_id": "buyer:b1",
                "merchant_id": "merchant:m1",
                "sku_id": sku,
                "qty": 1,
                "agreed_price": price,
                "state": "dispatched",
            }],
            "ledger": [{
                "txn_id": txn_id,
                "ts": "seed-evidence-not-time-authority",
                "order_id": order_id,
                "buyer_id": "buyer:b1",
                "merchant_id": "merchant:m1",
                "sku_id": sku,
                "qty": 1,
                "price": price,
                "idempotency_key": f"seed:settle:s14:{seed}",
            }],
            "order_timelines": [{
                "order_id": order_id,
                "buyer_id": "buyer:b1",
                "merchant_id": "merchant:m1",
                "settled_at_tick": 1,
                "dispatched_at_tick": 2,
                "return_window_ticks": 3,
            }],
            "logical_time": _CURRENT_TICKS[index],
        },
        # The legacy fields stay present for the stable ScenarioSpec schema.
        "buyer_goal": {
            "product_type": "return-window test item",
            "max_budget": price + 10,
            "quantity": 1,
            "constraints": [],
            "return_after_purchase": True,
        },
        "merchant_policy": {
            "list_price": price,
            "floor_price": floor,
            "refund_policy": "7_day_return",
            "claim_aggressiveness": "neutral",
        },
        "population": {
            "buyers": [{
                "buyer_id": "buyer:b1",
                "persona": {"name": "Return-window buyer"},
                "mandate": {
                    "mandate_id": f"s14:{seed}",
                    "goal": "request the authorized return for the existing order",
                    "quantity": 1,
                    "return_after_purchase": True,
                    "hard_constraints": {"budget": (price + 10) * 100, "must_have": []},
                    "soft_constraints": [],
                    "authority": {"can_buy_without_confirmation": True},
                    "intent_expiry": "2099-12-31T00:00:00Z",
                },
                "initial_state": {
                    "transaction": {
                        "return_after_purchase": True,
                        "pending_settlement_order_id": order_id,
                    },
                },
            }],
            "merchants": [{
                "merchant_id": "merchant:m1",
                "persona": {"name": "Return-window merchant"},
                "policy": {
                    "floor_price": floor * 100,
                    "refund_policy": "7_day_return",
                    "max_negotiation_rounds": 3,
                },
                "catalog_scope": [sku],
            }],
            "initial_events": [{
                "msg_id": f"kickoff:s14:{seed}",
                "ts": _RECEIPT_TIMESTAMPS[index],
                "from": "platform:psp",
                "to": "buyer:b1",
                "idempotency_key": f"kickoff:s14:{seed}",
                "action": {
                    "kind": "platform.settlement_receipt",
                    "payload": {
                        "order_id": order_id,
                        "txn_id": txn_id,
                        "status": "settled",
                    },
                },
            }],
            "matching": {"top_k": 1},
            "execution": {"max_transactions_per_buyer": 1},
        },
        "allowed_actions": ALLOWED_ACTIONS,
        "success_oracle": {
            "expected_sku": sku,
            "product_match": True,
            "final_price_lte": price + 10,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "return_requested": True,
            "refund_issued": branch == "authorized",
            "ledger_entry_created": True,
            "return_window_branch": branch,
            "expected_order_id": order_id,
            "protocol_violations": 0,
        },
        "benchmark": {
            "task_family": "T7",
            "variant_id": "S14",
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


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
