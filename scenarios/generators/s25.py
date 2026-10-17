"""S25 — like-for-like exchange after a physical return, without a second charge."""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS
from .roadmap_common import base_spec, kickoff, mandate, participant_population

FAMILY_ID = "s25"
FAMILY_NAME = "exchange_instead_of_refund"


def _spec(seed: int) -> dict[str, Any]:
    scenario_id = f"{FAMILY_ID}_{FAMILY_NAME}_{seed}"
    original = f"merchant:m1:sku:s25-original-{seed}"
    replacement = f"merchant:m1:sku:s25-replacement-{seed}"
    order_id = f"order:s25:{seed}"
    replacement_order_id = f"order:s25:{seed}:replacement"
    exchange_id = f"exchange:s25:{seed}"
    qty = {42: 1, 1337: 2, 2024: 3}[seed]
    catalog = [
        {
            "sku_id": original,
            "merchant_id": "merchant:m1",
            "product_id": "product:s25-shirt",
            "list_price": 35,
            "floor_price": 25,
            "inventory": 5,
            "qty_reserved": qty,
            "attributes": {"size": "M", "color": "blue", "shipping_days": 3},
        },
        {
            "sku_id": replacement,
            "merchant_id": "merchant:m1",
            "product_id": "product:s25-shirt",
            "list_price": 35,
            "floor_price": 25,
            "inventory": 5,
            "attributes": {"size": "M", "color": "green", "shipping_days": 3},
        },
    ]
    m = mandate(
        scenario_id=scenario_id,
        goal="exchange the returned item without a second payment",
        quantity=qty,
        budget_cents=35 * 100 * qty,
    )
    event = kickoff(
        scenario_id=scenario_id,
        sender="consumer:principal",
        receiver="buyer:b1",
        kind="delegate.create_purchase_mandate",
        payload={
            "order_id": order_id,
            "resolution": "exchange",
            "replacement_sku_id": replacement,
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
            "product_type": "returned shirt exchange",
            "max_budget": 35 * qty,
            "quantity": qty,
            "constraints": [],
        },
        mandate_spec=m,
        population=pop,
        initial_state_extra={
            "orders": [{
                "order_id": order_id,
                "buyer_id": "buyer:b1",
                "merchant_id": "merchant:m1",
                "sku_id": original,
                "qty": qty,
                "agreed_price": 35,
                "state": "returned",
            }],
            "ledger": [{
                "txn_id": f"txn:s25:{seed}",
                "ts": "authoritative-seed",
                "order_id": order_id,
                "buyer_id": "buyer:b1",
                "merchant_id": "merchant:m1",
                "sku_id": original,
                "qty": qty,
                "price": 35,
                "idempotency_key": f"seed:s25:{seed}",
            }],
        },
        task_family="T7",
        variant_id="S25",
        allowed_actions=[*ALLOWED_ACTIONS, "request_exchange"],
        success_oracle={
            "expected_order_id": order_id,
            "expected_replacement_order_id": replacement_order_id,
            "expected_exchange_id": exchange_id,
            "expected_replacement_sku": replacement,
            "expected_qty": qty,
            "ledger_delta": 0,
            "protocol_violations": 0,
        },
    )


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(spec := _spec(seed))['scenario_id']}.yaml", spec) for seed in SEEDS]


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
