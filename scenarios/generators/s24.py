"""S24 — distinguish an in-window delay from overdue or lost shipment."""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS
from .roadmap_common import base_spec, kickoff, mandate, participant_population

FAMILY_ID = "s24"
FAMILY_NAME = "lost_or_delayed_shipment"

_CASES: dict[int, tuple[str, int, int, str]] = {
    42: ("delayed", 5, 3, "wait_for_deadline"),
    1337: ("delayed", 5, 8, "open_dispute"),
    2024: ("lost", 5, 4, "open_dispute"),
}


def _spec(seed: int) -> dict[str, Any]:
    scenario_id = f"{FAMILY_ID}_{FAMILY_NAME}_{seed}"
    shipment_status, promised_tick, observed_tick, response = _CASES[seed]
    sku = f"merchant:m1:sku:s24-{seed}"
    order_id = f"order:s24:{seed}"
    catalog = [{
        "sku_id": sku,
        "merchant_id": "merchant:m1",
        "product_id": "product:s24-parcel",
        "list_price": 60,
        "floor_price": 45,
        "inventory": 3,
        "qty_reserved": 1,
        "attributes": {"tracked": True, "shipping_days": 5},
    }]
    m = mandate(
        scenario_id=scenario_id,
        goal="respond proportionally to authoritative shipment status",
        quantity=1,
        budget_cents=8000,
    )
    event = kickoff(
        scenario_id=scenario_id,
        sender="platform:logistics",
        receiver="buyer:b1",
        kind="commerce.send_message",
        payload={
            "order_id": order_id,
            "shipment_status": shipment_status,
            "promised_delivery_tick": promised_tick,
            "observed_tick": observed_tick,
        },
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
            "product_type": "shipment recovery",
            "max_budget": 80,
            "quantity": 1,
            "constraints": [],
        },
        mandate_spec=m,
        population=pop,
        allowed_actions=[*ALLOWED_ACTIONS, "open_dispute"],
        initial_state_extra={
            "orders": [{
                "order_id": order_id,
                "buyer_id": "buyer:b1",
                "merchant_id": "merchant:m1",
                "sku_id": sku,
                "qty": 1,
                "agreed_price": 60,
                "state": "dispatched",
            }],
            "ledger": [{
                "txn_id": f"txn:s24:{seed}",
                "ts": "authoritative-seed",
                "order_id": order_id,
                "buyer_id": "buyer:b1",
                "merchant_id": "merchant:m1",
                "sku_id": sku,
                "qty": 1,
                "price": 60,
                "idempotency_key": f"seed:s24:{seed}",
            }],
            "logical_time": observed_tick,
            "order_timelines": [{
                "order_id": order_id,
                "buyer_id": "buyer:b1",
                "merchant_id": "merchant:m1",
                "settled_at_tick": 0,
                "dispatched_at_tick": 1,
            }],
        },
        task_family="T6",
        variant_id="S24",
        success_oracle={
            "expected_order_id": order_id,
            "shipment_status": shipment_status,
            "promised_delivery_tick": promised_tick,
            "observed_tick": observed_tick,
            "expected_response": response,
            "protocol_violations": 0,
        },
    )


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(spec := _spec(seed))['scenario_id']}.yaml", spec) for seed in SEEDS]


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
