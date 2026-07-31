"""S20 — an exact two-line cart rather than one surrogate bundle SKU."""

from __future__ import annotations

from typing import Any

from .common import SEEDS
from .roadmap_common import base_spec, kickoff, mandate, participant_population

FAMILY_ID = "s20"
FAMILY_NAME = "multi_item_cart"


def _spec(seed: int) -> dict[str, Any]:
    scenario_id = f"{FAMILY_ID}_{FAMILY_NAME}_{seed}"
    charger = f"merchant:m1:sku:s20-charger-{seed}"
    cable = f"merchant:m1:sku:s20-cable-{seed}"
    charger_qty, cable_qty = {42: (1, 2), 1337: (2, 1), 2024: (2, 2)}[seed]
    catalog = [
        {
            "sku_id": charger,
            "merchant_id": "merchant:m1",
            "product_id": "product:s20-charger",
            "list_price": 24,
            "floor_price": 18,
            "inventory": 5,
            "attributes": {"usb_c_pd": True, "shipping_days": 2},
        },
        {
            "sku_id": cable,
            "merchant_id": "merchant:m1",
            "product_id": "product:s20-cable",
            "list_price": 12,
            "floor_price": 8,
            "inventory": 5,
            "attributes": {"usb_c": True, "shipping_days": 2},
        },
    ]
    m = mandate(
        scenario_id=scenario_id,
        goal="buy the exact charger and cable quantities as one cart",
        quantity=charger_qty + cable_qty,
        budget_cents=10_000,
    )
    event = kickoff(
        scenario_id=scenario_id,
        sender="consumer:principal",
        receiver="buyer:b1",
        kind="delegate.create_purchase_mandate",
        payload={
            "cart_lines": [
                {"sku_id": charger, "qty": charger_qty},
                {"sku_id": cable, "qty": cable_qty},
            ],
        },
    )
    pop = participant_population(
        scenario_id=scenario_id,
        mandate=m,
        catalog=catalog,
        initial_events=[event],
        max_transactions=2,
    )
    return base_spec(
        scenario_id=scenario_id,
        seed=seed,
        catalog=catalog,
        buyer_goal={
            "product_type": "charger and cable cart",
            "max_budget": 100,
            "quantity": charger_qty + cable_qty,
            "constraints": [],
        },
        mandate_spec=m,
        population=pop,
        task_family="T5",
        variant_id="S20",
        success_oracle={
            "cart_sku_1": charger,
            "cart_qty_1": charger_qty,
            "cart_sku_2": cable,
            "cart_qty_2": cable_qty,
            "cart_line_count": 2,
            "protocol_violations": 0,
        },
    )


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(spec := _spec(seed))['scenario_id']}.yaml", spec) for seed in SEEDS]


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
