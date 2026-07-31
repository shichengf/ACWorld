"""S22 — authoritative full, partial, and zero-fill allocation branches."""

from __future__ import annotations

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s22"
FAMILY_NAME = "partial_fill_backorder"

_CASES: dict[int, tuple[int, int, str]] = {
    42: (3, 5, "full"),
    1337: (4, 2, "partial"),
    2024: (2, 0, "backordered"),
}


def generate() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for seed in SEEDS:
        requested, inventory, branch = _CASES[seed]
        fulfilled = min(requested, inventory)
        backordered = requested - fulfilled
        sku = f"merchant:m1:sku:s22:{seed}"
        offer = f"s22-{seed}"
        mandate_id = f"s22:{seed}"
        order_id = f"ord-{mandate_id}-{offer}"
        spec = {
            "scenario_id": f"s22_partial_fill_backorder_{seed}",
            "seed": seed,
            "initial_state": {
                "catalog": [{
                    "sku_id": sku,
                    "merchant_id": "merchant:m1",
                    "product_id": "product:s22",
                    "list_price": 25,
                    "floor_price": 15,
                    "inventory": inventory,
                    "attributes": {
                        "in_stock": inventory > 0,
                        "shipping_days": 2,
                    },
                }],
            },
            "buyer_goal": {
                "product_type": "multi-unit benchmark item",
                "max_budget": 200,
                "quantity": requested,
                "constraints": [],
            },
            "merchant_policy": {
                "list_price": 25,
                "floor_price": 15,
                "refund_policy": "no_returns",
                "claim_aggressiveness": "neutral",
            },
            "population": {
                "buyers": [{
                    "buyer_id": "buyer:b1",
                    "persona": {"name": "S22 buyer"},
                    "mandate": {
                        "mandate_id": mandate_id,
                        "goal": "buy the available quantity and backorder the remainder",
                        "quantity": requested,
                        "hard_constraints": {"budget": 20000, "must_have": []},
                        "soft_constraints": [],
                        "authority": {"can_buy_without_confirmation": True},
                        "intent_expiry": "2099-12-31T00:00:00Z",
                    },
                    "initial_state": {
                        "private_utility": {
                            "max_budget": 20000,
                            "can_buy_without_confirmation": True,
                        },
                        "transaction": {
                            "mandate_id": mandate_id,
                            "cumulative_spend": 0,
                            "pending_settlement_order_id": order_id,
                            "selected_offer": {
                                "offer_id": offer,
                                "merchant_id": "merchant:m1",
                                "sku_id": sku,
                                "qty": requested,
                                "unit_price": 2500,
                                "rationale": "allow an authoritative partial allocation",
                            },
                        },
                    },
                }],
                "merchants": [{
                    "merchant_id": "merchant:m1",
                    "persona": {"name": "S22 merchant"},
                    "policy": {
                        "floor_price": 1500,
                        "refund_policy": "no_returns",
                        "max_negotiation_rounds": 3,
                    },
                    "catalog_scope": [sku],
                }],
                "initial_events": [{
                    "msg_id": f"kickoff:s22:{seed}",
                    "ts": "2026-07-14T00:00:00Z",
                    "from": "platform:aggregator",
                    "to": "buyer:b1",
                    "idempotency_key": f"kickoff:s22:{seed}",
                    "action": {
                        "kind": "platform.create_match_certificate",
                        "payload": {
                            "cert_id": f"cert:s22:{seed}",
                            "offer_id": offer,
                            "checks_passed": {
                                "budget": True,
                                "must_have": True,
                                "delivery": True,
                                "grounding": True,
                            },
                        },
                    },
                }],
                "matching": {"top_k": 1},
                "execution": {"max_transactions_per_buyer": 1},
            },
            "allowed_actions": list(ALLOWED_ACTIONS),
            "success_oracle": {
                "expected_sku": sku,
                "product_match": True,
                "fulfillment_branch": branch,
                "expected_order_id": order_id,
                "requested_qty": requested,
                "fulfilled_qty": fulfilled,
                "backordered_qty": backordered,
                "partial_fill": branch == "partial",
                "ledger_entry_created": fulfilled > 0,
                "protocol_violations": 0,
            },
            "benchmark": {
                "task_family": "T6",
                "variant_id": "S22",
                "track": "agent",
                "difficulty": "medium",
                "catalog_source": "inline",
                "catalog_scale": "smoke",
                "scenario_version": "1.0",
            },
        }
        out.append((f"s22_partial_fill_backorder_{seed}.yaml", spec))
    return out


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
