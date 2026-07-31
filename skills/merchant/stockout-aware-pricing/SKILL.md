---
name: stockout-aware-pricing
description: A pre-step for pricing-negotiate during an offer turn. When a SKU is both scarce (low days_to_stockout) and in-demand (high opp_score), write a memory bias that shrinks the effective target_band toward its upper end — so pricing-negotiate counters closer to the top, not the midpoint. Emits no envelope of its own. NOT a replacement for pricing-negotiate; always followed by it in the same turn.
when_to_use: Use when the inbound is propose_offer / counter_offer AND get_inventory_status(product).days_to_stockout <= policy.scarcity_window_days AND get_category_demand(product).opp_score >= policy.demand_floor_for_scarcity. Always load pricing-negotiate next in the same turn so it consumes the bias.
group: economic
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_catalog
  - world.read_inventory
  - memory.read
  - memory.write
context: main
---

# Stockout-Aware Pricing

## Native function guidance

Use authoritative listing and availability reads to decide whether genuine
scarcity should influence an active negotiation. Apply a bounded scarcity bias
only when the configured evidence supports it. Do not infer scarcity from a
buyer's urgency or disclose exact stock, floor price, target band, or internal
pressure. This preparatory skill does not itself change price or send a market
action. Let an active negotiation skill make the terminal decision, or finish
safely when no such action is authorized.

## Detailed workflow

A pre-step for `pricing-negotiate` on scarce + in-demand SKUs.

If the SKU is running out faster than restock can refill, and category
demand is high, the merchant should not meet the buyer at the midpoint of
`target_band` — they should hold closer to `target_band[1]`. This skill
encodes that as a memory hint and hands the actual accept/counter/reject
decision off to `pricing-negotiate` (which reads the hint).

## Procedure

1. Read state:
   - `inv    = get_inventory_status(product)` → `days_to_stockout`
   - `demand = get_category_demand(product)`  → `opp_score`
   - `pol    = world.read_policy()`           → `scarcity_window_days`,
                                                `demand_floor_for_scarcity`
2. Gate: if NOT (`inv.days_to_stockout <= scarcity_window_days` AND
   `demand.opp_score >= demand_floor_for_scarcity`) → emit `no_reply`
   (let `pricing-negotiate` proceed normally).
3. Shrink the band toward the top quarter:
   `band = read_offer_mandate(product).target_band`
   `effective_low = band[1] - max(1, (band[1] - band[0]) // 4)`
4. Write the hint:
   `memory.write(PREFERENCE, "pricing_bias",
                 {"sku_id": ..., "effective_band": [effective_low, band[1]],
                  "reason": "scarcity"})`
5. Emit `no_reply`. Then load `pricing-negotiate`, which calls
   `apply_pricing_bias(target_band, bias, sku_id)` and uses the returned
   `effective_band` in place of `target_band` for its midpoint.

## Never disclose

- Inventory counts, `days_to_stockout`, or any urgency cue in a payload
  or in chat. The bias only shifts the outbound `unit_price`; it must not
  narrate ("limited stock, my price is firmer"). Defer to
  `private-utility-guard`.

## Output

`no_reply`. `pricing-negotiate` emits the single offer-turn envelope.
