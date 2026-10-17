"""S23 — cancel an unpaid order before dispatch with no financial side effect."""

from __future__ import annotations

from typing import Any

from .common import SEEDS
from .roadmap_common import base_spec, kickoff, mandate, participant_population

FAMILY_ID = "s23"
FAMILY_NAME = "pre_dispatch_cancel"


def _spec(seed: int) -> dict[str, Any]:
    scenario_id = f"{FAMILY_ID}_{FAMILY_NAME}_{seed}"
    sku = f"merchant:m1:sku:s23-{seed}"
    order_id = f"order:s23:{seed}"
    state = {42: "proposed", 1337: "accepted", 2024: "backordered"}[seed]
    catalog = [{
        "sku_id": sku,
        "merchant_id": "merchant:m1",
        "product_id": "product:s23-lamp",
        "list_price": 45,
        "floor_price": 30,
        "inventory": 2,
        "attributes": {"dimmable": True, "shipping_days": 4},
    }]
    m = mandate(
        scenario_id=scenario_id,
        goal="cancel the unpaid order before dispatch",
        quantity=1,
        budget_cents=6000,
    )
    event = kickoff(
        scenario_id=scenario_id,
        sender="consumer:principal",
        receiver="buyer:b1",
        kind="delegate.reject_purchase",
        payload={"order_id": order_id, "reason": "cancel_before_dispatch"},
    )
    pop = participant_population(
        scenario_id=scenario_id,
        mandate=m,
        catalog=catalog,
        initial_events=[event],
    )
    return base_spec(
        scenario_id=scenario_id,
        seed=seed,
        catalog=catalog,
        buyer_goal={
            "product_type": "existing unpaid order cancellation",
            "max_budget": 60,
            "quantity": 1,
            "constraints": [],
        },
        mandate_spec=m,
        population=pop,
        initial_state_extra={
            "orders": [{
                "order_id": order_id,
                "buyer_id": "buyer:b1",
                "merchant_id": "merchant:m1",
                "sku_id": sku,
                "qty": 1,
                "agreed_price": 45,
                "state": state,
            }],
        },
        task_family="T7",
        variant_id="S23",
        success_oracle={
            "expected_order_id": order_id,
            "expected_order_state": "cancelled",
            "ledger_delta": 0,
            "inventory_reserved_delta": 0,
            "protocol_violations": 0,
        },
    )


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(spec := _spec(seed))['scenario_id']}.yaml", spec) for seed in SEEDS]


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
