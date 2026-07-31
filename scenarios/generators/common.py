"""Shared helpers for deterministic scenario generators.

The emitter here writes exactly the constrained YAML subset that
``episode.scenario._loads`` parses (block maps, a block sequence of
``catalog`` items, inline flow lists/maps, bare scalars). Generator output
and the loader are round-trip verified in ``scenarios/generators/generate.py``
and in the test notebook, so the two must stay in lock-step.

Determinism: every value is a pure function of
the scenario seed — no RNG, no clock, no dict-ordering reliance. Re-running a
generator yields byte-identical files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The 12 v0.1 action kinds (protocol/actions.py ActionKind).
ALLOWED_ACTIONS: list[str] = [
    "search", "get_sku", "propose_offer", "counter_offer",
    "accept_offer", "reject_offer", "create_order", "settle",
    "dispatch", "request_return", "issue_refund", "send_message",
]

# Three deterministic instances per family (see scenarios/README.md).
SEEDS: list[int] = [42, 1337, 2024]


def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    # A comma is a structural separator in a flow list/map.  Quote such values
    # instead of weakening the data (S9 deliberately uses ``wool, merino`` to
    # test order-independent multi-word material grounding).  JSON string
    # syntax is a strict subset of the loader's supported double-quoted scalar
    # syntax and gives byte-stable escaping.
    if isinstance(v, str) and "," in v:
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _flow_value(value: Any) -> str:
    """Render the recursive flow subset understood by ``episode.scenario``.

    Most original scenario attributes are scalar, but explicit populations
    contain nested mandates, policies, and envelope actions.  Keeping those
    structures in flow form makes the generated files compact while preserving
    the dependency-free loader's deterministic grammar.
    """
    if isinstance(value, dict):
        return _flow_map(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_flow_value(item) for item in value) + "]"
    return _scalar(value)


def _flow_list(items: list[Any]) -> str:
    return "[" + ", ".join(_flow_value(x) for x in items) + "]"


def _flow_map(d: dict[str, Any]) -> str:
    return "{ " + ", ".join(f"{k}: {_flow_value(v)}" for k, v in d.items()) + " }"


def emit_yaml(spec: dict[str, Any]) -> str:
    """Serialize a scenario spec dict into the loader's YAML subset.

    Key order is fixed (matches the ScenarioSpec contract) so output is stable.
    """
    L: list[str] = []
    L.append(f"scenario_id: {spec['scenario_id']}")
    L.append(f"seed: {spec['seed']}")

    L.append("initial_state:")
    L.append("  catalog:")
    for sku in spec["initial_state"]["catalog"]:
        L.append(f"    - sku_id: {sku['sku_id']}")
        if "merchant_id" in sku:
            L.append(f"      merchant_id: {sku['merchant_id']}")
        if "product_id" in sku:
            L.append(f"      product_id: {sku['product_id']}")
        L.append(f"      list_price: {sku['list_price']}")
        L.append(f"      floor_price: {sku['floor_price']}")
        L.append(f"      inventory: {sku['inventory']}")
        if "qty_reserved" in sku:
            L.append(f"      qty_reserved: {sku['qty_reserved']}")
        L.append(f"      attributes: {_flow_map(sku['attributes'])}")

    orders = spec["initial_state"].get("orders") or []
    if orders:
        L.append("  orders:")
        for order in orders:
            L.append(f"    - order_id: {order['order_id']}")
            L.append(f"      buyer_id: {order['buyer_id']}")
            L.append(f"      merchant_id: {order['merchant_id']}")
            L.append(f"      sku_id: {order['sku_id']}")
            L.append(f"      qty: {order['qty']}")
            L.append(f"      agreed_price: {order['agreed_price']}")
            if "currency" in order:
                L.append(f"      currency: {order['currency']}")
            L.append(f"      state: {order['state']}")

    ledger = spec["initial_state"].get("ledger") or []
    if ledger:
        L.append("  ledger:")
        for receipt in ledger:
            L.append(f"    - txn_id: {receipt['txn_id']}")
            L.append(f"      ts: {receipt['ts']}")
            L.append(f"      order_id: {receipt['order_id']}")
            L.append(f"      buyer_id: {receipt['buyer_id']}")
            L.append(f"      merchant_id: {receipt['merchant_id']}")
            L.append(f"      sku_id: {receipt['sku_id']}")
            L.append(f"      qty: {receipt['qty']}")
            L.append(f"      price: {receipt['price']}")
            if "currency" in receipt:
                L.append(f"      currency: {receipt['currency']}")
            L.append(f"      idempotency_key: {receipt['idempotency_key']}")

    timelines = spec["initial_state"].get("order_timelines") or []
    if timelines:
        L.append("  order_timelines:")
        for timeline in timelines:
            L.append(f"    - order_id: {timeline['order_id']}")
            L.append(f"      buyer_id: {timeline['buyer_id']}")
            L.append(f"      merchant_id: {timeline['merchant_id']}")
            for key in (
                "settled_at_tick",
                "dispatched_at_tick",
                "return_window_ticks",
                "return_authorized_at_tick",
                "returned_at_tick",
                "refunded_at_tick",
            ):
                if timeline.get(key) is not None:
                    L.append(f"      {key}: {timeline[key]}")

    if "logical_time" in spec["initial_state"]:
        L.append(f"  logical_time: {spec['initial_state']['logical_time']}")

    friendships = spec["initial_state"].get("friendships") or []
    if friendships:
        L.append("  friendships:")
        for friendship in friendships:
            L.append(f"    - buyer_id: {friendship['buyer_id']}")
            L.append(f"      friends: {_flow_list(friendship['friends'])}")

    reviews = spec["initial_state"].get("reviews") or []
    if reviews:
        L.append("  reviews:")
        for review in reviews:
            L.append(f"    - review_id: {review['review_id']}")
            L.append(f"      reviewer_id: {review['reviewer_id']}")
            L.append(f"      sku_id: {review['sku_id']}")
            L.append(f"      merchant_id: {review['merchant_id']}")
            L.append(f"      rating: {review['rating']}")
            if "text" in review:
                # Review prose is data in the T9 adversarial lane.  Serialize
                # with the same scalar rules as every other value so commas in
                # an injection fixture cannot change the YAML structure.
                L.append(f"      text: {_scalar(review['text'])}")

    reputation = spec["initial_state"].get("reputation") or []
    if reputation:
        rows = reputation.values() if isinstance(reputation, dict) else reputation
        L.append("  reputation:")
        for score in rows:
            L.append(f"    - merchant_id: {score['merchant_id']}")
            L.append(f"      rolling_avg: {score['rolling_avg']}")
            L.append(f"      n_settled: {score['n_settled']}")
            L.append(f"      n_disputed: {score['n_disputed']}")

    bg = spec["buyer_goal"]
    L.append("buyer_goal:")
    L.append(f"  product_type: {bg['product_type']}")
    L.append(f"  max_budget: {bg['max_budget']}")
    L.append(f"  quantity: {bg['quantity']}")
    L.append(f"  constraints: {_flow_list(bg['constraints'])}")
    soft_constraints = bg.get("soft_constraints") or []
    if soft_constraints:
        L.append("  soft_constraints:")
        for constraint in soft_constraints:
            L.append(f"    - feature: {constraint['feature']}")
            L.append(f"      importance: {constraint.get('importance', 1)}")
    # Optional intent flag — emitted only when a family sets it (s5 return/refund),
    # so families that don't (s1–s4) stay byte-identical to prior output.
    if "return_after_purchase" in bg:
        L.append(f"  return_after_purchase: {_scalar(bg['return_after_purchase'])}")

    mp = spec["merchant_policy"]
    L.append("merchant_policy:")
    L.append(f"  list_price: {mp['list_price']}")
    L.append(f"  floor_price: {mp['floor_price']}")
    L.append(f"  refund_policy: {mp['refund_policy']}")
    L.append(f"  claim_aggressiveness: {mp['claim_aggressiveness']}")

    population = spec.get("population")
    if population:
        L.append("population:")
        L.append("  buyers:")
        for buyer in population["buyers"]:
            L.append(f"    - buyer_id: {buyer['buyer_id']}")
            L.append(f"      persona: {_flow_map(buyer.get('persona', {}))}")
            L.append(f"      mandate: {_flow_map(buyer['mandate'])}")
            if buyer.get("initial_state"):
                L.append(
                    f"      initial_state: {_flow_map(buyer['initial_state'])}"
                )
        L.append("  merchants:")
        for merchant in population["merchants"]:
            L.append(f"    - merchant_id: {merchant['merchant_id']}")
            L.append(f"      persona: {_flow_map(merchant.get('persona', {}))}")
            L.append(f"      policy: {_flow_map(merchant['policy'])}")
            if merchant.get("catalog_scope") is not None:
                L.append(
                    f"      catalog_scope: {_flow_list(merchant['catalog_scope'])}"
                )
            if merchant.get("initial_state"):
                L.append(
                    f"      initial_state: {_flow_map(merchant['initial_state'])}"
                )
        events = population.get("initial_events") or []
        if events:
            L.append("  initial_events:")
            for event in events:
                L.append(f"    - msg_id: {event['msg_id']}")
                L.append(f"      ts: {event['ts']}")
                L.append(f"      from: {event['from']}")
                L.append(f"      to: {event['to']}")
                if event.get("in_reply_to") is not None:
                    L.append(f"      in_reply_to: {event['in_reply_to']}")
                L.append(f"      idempotency_key: {event['idempotency_key']}")
                L.append(f"      action: {_flow_map(event['action'])}")
        L.append(f"  matching: {_flow_map(population.get('matching', {'top_k': 5}))}")
        L.append(
            "  execution: "
            + _flow_map(
                population.get("execution", {"max_transactions_per_buyer": 1})
            )
        )

    L.append(f"allowed_actions: {_flow_list(spec['allowed_actions'])}")

    L.append("success_oracle:")
    for k, v in spec["success_oracle"].items():
        # Market-only server-side answer keys contain nested structured
        # valuations/floors. Keep the historical scalar rendering byte-stable
        # for the 111 core files while allowing this explicitly named additive
        # block to use the loader's recursive flow grammar.
        rendered = (
            _flow_value(v)
            if k in {"market_oracle", "market_study"}
            else _scalar(v)
        )
        L.append(f"  {k}: {rendered}")

    benchmark = spec.get("benchmark")
    if benchmark:
        L.append("benchmark:")
        ordered = (
            "task_family", "variant_id", "track", "difficulty",
            "catalog_source", "catalog_scale", "catalog_merchants",
            "in_stock_only", "scenario_version",
        )
        for key in ordered:
            if key not in benchmark:
                continue
            value = benchmark[key]
            rendered = _flow_list(value) if isinstance(value, list) else _scalar(value)
            L.append(f"  {key}: {rendered}")

    return "\n".join(L) + "\n"


def write_scenario(out_dir: Path, filename: str, spec: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(emit_yaml(spec), encoding="utf-8", newline="\n")
    return path
