"""S18 — World-authoritative comparison across merchant-specific listings.

Three merchants sell the same ``product_id``.  One cheap listing fails the
buyer's minimum reputation, while two listings are feasible and the less
expensive of those is the unique optimum under the public S18 ordering.  The
winning merchant rotates across seeds, preventing fixed merchant/SKU policies
from passing the corpus.
"""

from __future__ import annotations

from typing import Any

from .common import SEEDS

FAMILY_ID = "s18"
FAMILY_NAME = "multi_merchant_price_quality"

_ALLOWED_ACTIONS = [
    "search",
    "get_sku",
    "accept_offer",
    "reject_offer",
    "create_order",
    "settle",
    "dispatch",
    "request_return",
    "issue_refund",
    "send_message",
]

_PROFILES: dict[str, dict[str, Any]] = {
    "low_trust": {
        "list_price": 42,
        "floor_price_cents": 3000,
        "shipping_days": 2,
        "merchant_reputation_bps": 380,
        "return_window_days": 30,
        "refund_policy": "30_day_return",
    },
    "best_value": {
        "list_price": 49,
        "floor_price_cents": 3600,
        "shipping_days": 3,
        "merchant_reputation_bps": 470,
        "return_window_days": 30,
        "refund_policy": "30_day_return",
    },
    "premium": {
        "list_price": 58,
        "floor_price_cents": 4200,
        "shipping_days": 1,
        "merchant_reputation_bps": 490,
        "return_window_days": 45,
        "refund_policy": "45_day_return",
    },
}

_ROTATIONS = {
    42: ("low_trust", "best_value", "premium"),
    1337: ("premium", "low_trust", "best_value"),
    2024: ("best_value", "premium", "low_trust"),
}


def _spec(seed: int) -> dict[str, Any]:
    assignments = _ROTATIONS[seed]
    catalog: list[dict[str, Any]] = []
    merchants: list[dict[str, Any]] = []
    winner = ""

    for index, profile_name in enumerate(assignments, start=1):
        merchant_id = f"merchant:m{index}"
        sku_id = f"{merchant_id}:travel-mug"
        profile = _PROFILES[profile_name]
        if profile_name == "best_value":
            winner = sku_id
        catalog.append({
            "sku_id": sku_id,
            "merchant_id": merchant_id,
            "product_id": "product:insulated-travel-mug",
            "list_price": profile["list_price"],
            "floor_price": profile["floor_price_cents"] // 100,
            "inventory": 4,
            "attributes": {
                "product_type": "insulated travel mug",
                "authenticity_verified": True,
                "shipping_days": profile["shipping_days"],
                "merchant_reputation_bps": profile["merchant_reputation_bps"],
                "return_window_days": profile["return_window_days"],
            },
        })
        merchants.append({
            "merchant_id": merchant_id,
            "persona": {"name": f"Merchant {index}"},
            "policy": {
                "floor_price": profile["floor_price_cents"],
                "refund_policy": profile["refund_policy"],
                "claim_aggressiveness": "neutral",
            },
            "catalog_scope": [sku_id],
        })

    scenario_id = f"{FAMILY_ID}_{FAMILY_NAME}_{seed}"
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "initial_state": {"catalog": catalog},
        # Required legacy surface.  The explicit population below is the
        # actor-native source used by runtime and scoring.
        "buyer_goal": {
            "product_type": "insulated travel mug",
            "max_budget": 65,
            "quantity": 1,
            "constraints": ["authenticity_verified", "shipping_within_4_days"],
        },
        "merchant_policy": {
            "list_price": 49,
            "floor_price": 30,
            "refund_policy": "30_day_return",
            "claim_aggressiveness": "neutral",
        },
        "population": {
            "buyers": [{
                "buyer_id": "buyer:b1",
                "persona": {"name": "Comparison buyer"},
                "mandate": {
                    "mandate_id": scenario_id,
                    "goal": "insulated travel mug",
                    "quantity": 1,
                    "hard_constraints": {
                        "budget": 6500,
                        "must_have": ["authenticity_verified"],
                        "delivery_days": 4,
                        "min_merchant_reputation_bps": 450,
                        "min_return_window_days": 30,
                    },
                    "soft_constraints": [],
                    "soft_preferences": {"style": [], "avoid": []},
                    "authority": {
                        "can_buy_without_confirmation": True,
                        "must_not_share_with_merchant": ["budget"],
                    },
                    "intent_expiry": "2099-12-31T00:00:00Z",
                },
            }],
            "merchants": merchants,
            "matching": {"top_k": 3},
            "execution": {"max_transactions_per_buyer": 1},
        },
        "allowed_actions": _ALLOWED_ACTIONS,
        "success_oracle": {
            "product_match": True,
            "expected_sku": winner,
            "final_price_lte": 65,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "ledger_entry_created": True,
            "protocol_violations": 0,
        },
        "benchmark": {
            "task_family": "T8",
            "variant_id": "S18",
            "track": "agent",
            "difficulty": "hard",
            "catalog_source": "inline",
            "catalog_scale": "smoke",
            "scenario_version": "1.0",
        },
    }


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(spec := _spec(seed))['scenario_id']}.yaml", spec) for seed in SEEDS]
