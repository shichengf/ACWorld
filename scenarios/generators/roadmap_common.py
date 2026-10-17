"""Shared deterministic builders for the S19--S29 benchmark tranche.

The public modules remain one-per-variant so the generator registry and paper
can refer to stable symbols.  This file only removes boilerplate around the
actor-native 1x1 population used by most of those variants.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS


def participant_population(
    *,
    scenario_id: str,
    mandate: dict[str, Any],
    catalog: list[dict[str, Any]],
    merchant_ids: tuple[str, ...] = ("merchant:m1",),
    initial_events: list[dict[str, Any]] | None = None,
    max_transactions: int = 1,
    buyer_initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a complete actor-ID population without role-only fallbacks."""

    return {
        "buyers": [{
            "buyer_id": "buyer:b1",
            "persona": {"name": f"{scenario_id} buyer"},
            "mandate": mandate,
            **(
                {"initial_state": buyer_initial_state}
                if buyer_initial_state
                else {}
            ),
        }],
        "merchants": [
            {
                "merchant_id": merchant_id,
                "persona": {"name": f"{scenario_id} {merchant_id}"},
                "policy": {
                    "floor_price": min(
                        int(round(float(row["floor_price"]) * 100))
                        for row in catalog
                        if row.get("merchant_id", "merchant:m1") == merchant_id
                    ),
                    "refund_policy": "30_day_return",
                    "claim_aggressiveness": "neutral",
                },
                "catalog_scope": [
                    str(row["sku_id"])
                    for row in catalog
                    if row.get("merchant_id", "merchant:m1") == merchant_id
                ],
            }
            for merchant_id in merchant_ids
        ],
        "initial_events": list(initial_events or []),
        "matching": {"top_k": max(1, len(catalog))},
        "execution": {"max_transactions_per_buyer": max_transactions},
    }


def mandate(
    *,
    scenario_id: str,
    goal: str,
    quantity: int,
    budget_cents: int,
    must_have: list[str] | None = None,
    delivery_days: int | None = None,
) -> dict[str, Any]:
    hard: dict[str, Any] = {
        "budget": budget_cents,
        "must_have": list(must_have or []),
    }
    if delivery_days is not None:
        hard["delivery_days"] = delivery_days
    return {
        "mandate_id": scenario_id,
        "goal": goal,
        "quantity": quantity,
        "hard_constraints": hard,
        "soft_constraints": [],
        "soft_preferences": {"style": [], "avoid": []},
        "authority": {
            "can_buy_without_confirmation": True,
            "must_not_share_with_merchant": ["budget"],
        },
        "intent_expiry": "2099-12-31T00:00:00Z",
    }


def base_spec(
    *,
    scenario_id: str,
    seed: int,
    catalog: list[dict[str, Any]],
    buyer_goal: dict[str, Any],
    mandate_spec: dict[str, Any],
    task_family: str,
    variant_id: str,
    success_oracle: dict[str, Any],
    initial_state_extra: dict[str, Any] | None = None,
    population: dict[str, Any] | None = None,
    allowed_actions: list[str] | None = None,
    difficulty: str = "hard",
) -> dict[str, Any]:
    initial = {"catalog": catalog, **dict(initial_state_extra or {})}
    merchant_floor = min(float(row["floor_price"]) for row in catalog)
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "initial_state": initial,
        "buyer_goal": buyer_goal,
        "merchant_policy": {
            "list_price": float(catalog[0]["list_price"]),
            "floor_price": merchant_floor,
            "refund_policy": "30_day_return",
            "claim_aggressiveness": "neutral",
        },
        "population": population or participant_population(
            scenario_id=scenario_id,
            mandate=mandate_spec,
            catalog=catalog,
        ),
        "allowed_actions": list(allowed_actions or ALLOWED_ACTIONS),
        "success_oracle": success_oracle,
        "benchmark": {
            "task_family": task_family,
            "variant_id": variant_id,
            "track": "agent",
            "difficulty": difficulty,
            "catalog_source": "inline",
            "catalog_scale": "smoke",
            "scenario_version": "1.0",
        },
    }


def kickoff(
    *,
    scenario_id: str,
    sender: str,
    receiver: str,
    kind: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "msg_id": f"kickoff:{scenario_id}",
        "ts": "2026-07-14T00:00:00Z",
        "from": sender,
        "to": receiver,
        "idempotency_key": f"kickoff:{scenario_id}",
        "action": {"kind": kind, "payload": payload},
    }


__all__ = ["base_spec", "kickoff", "mandate", "participant_population"]
