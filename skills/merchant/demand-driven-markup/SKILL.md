---
name: demand-driven-markup
description: Use proactively (not in response to a buyer offer) to take a single bounded step UP on a listing's public list_price when demand is strong AND stock is fresh-and-moving — capturing margin proportional to demand without crossing the mandate's authorized ceiling. Emits exactly one commerce.adjust_price; capped at policy.markup_rules.max_pct and never exceeds target_band[1]. Never used as surge pricing during scarcity. Never reveals the floor.
when_to_use: Use when no inbound offer needs a decision this turn AND get_category_demand(product).opp_score >= policy.markup_demand_floor AND get_inventory_status(product).velocity_per_day >= policy.markup_velocity_floor AND get_inventory_status(product).inventory_age_days <= policy.markdown_rules.after_days (i.e. NOT aged) AND list_price < target_band[1] AND no markup applied within policy.markup_rules.cooldown_days. Do NOT use during a propose_offer / counter_offer turn (stockout-aware-pricing owns scarcity-on-offer behavior). Do NOT use when stock is aging (defer to aging-markdown). Do NOT use during a scarcity event (defer; that is a different policy decision).
group: economic
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_catalog
  - world.read_inventory
  - world.read_policy
  - memory.read
  - memory.write
context: main
---

# Demand-Driven Markup

## Native function guidance

Read the authoritative listing and availability before considering a markup.
Raise the public price only when configured demand and movement evidence
supports it, the item is fresh, and the decision is not scarcity surge
pricing. Keep the result within the private target and policy caps. Never reveal
floor price, target rails, exact stock, demand internals, or urgency. Use one
supplied price adjustment when all conditions hold; otherwise finish safely.

## Detailed workflow

The mirror of `aging-markdown`. When a fresh, fast-moving SKU is sustaining
strong demand, take **one** bounded step up on its public `list_price` so
the merchant captures margin proportional to the demand signal — without
ever crossing the mandate's authorized ceiling (`target_band[1]`).

## Procedure

1. Read state:
   - `dem  = get_category_demand(product)`
   - `inv  = get_inventory_status(product)`
   - `lst  = get_listing(product)` — `list_price`, `floor_price` (PRIVATE)
   - `mand = read_offer_mandate(product)` — `target_band[1]` is the ceiling
   - `pol  = world.read_policy()` — `markup_rules`, `markup_demand_floor`,
     `markup_velocity_floor`, `markdown_rules`, `scarcity_window_days`
2. Gate (any false → `no_reply`):
   - `dem.opp_score >= pol.markup_demand_floor`
   - `inv.velocity_per_day` known and `>= pol.markup_velocity_floor`
   - `inv.inventory_age_days <= pol.markdown_rules.after_days` (not aged)
   - `inv.days_to_stockout > pol.scarcity_window_days` (not scarce)
   - `lst.list_price < mand.target_band[1]` (room to move)
   - no entry in `memory[PREFERENCE].last_markup[sku_id]` within
     `markup_rules.cooldown_days`
3. Compute the new list price deterministically:
   `new_lp = demand_step_markup(lst.list_price, mand.target_band[1],
                                pol.markup_rules)`.
   The helper caps the step at `markup_rules.max_pct` of the current
   price and clamps to `target_band[1]` — both bounds are absolute.
4. If `new_lp <= lst.list_price` → `no_reply` (already at the ceiling
   or rules say zero step).
5. Emit exactly one envelope:
   `{"kind": "commerce.adjust_price",
     "payload": {"sku_id": ..., "list_price": new_lp}}`.
6. Record the step:
   `memory.write(PREFERENCE, "last_markup",
                 {sku_id: {"price": new_lp, "applied_at_turn": ctx.turn}})`.

## Never disclose

- Neither `floor_price`, nor any cue about *why* the price went up.
  The new `list_price` is public; the reasoning (demand + velocity) is
  not. Markup must look identical to a routine policy update — not a
  "we know you want it" signal.
- `urgency_to_sell` — markup is policy-driven and bounded, not buyer-
  signaling. Defer to `private-utility-guard` before emitting.

## Bounds and ethics

- One bounded step per cooldown window. Never compound rapidly.
- Always strictly at or below `target_band[1]` — the mandate's
  authorized ceiling is the contract with the merchant's owner.
- Never raise during a scarcity event from this skill. If scarcity
  drives the merchant's pricing decision, that belongs in
  `stockout-aware-pricing` (offer-turn bias, not public list bump),
  not here. Surge-pricing the public list is a separate, deliberate
  policy decision, out of scope for this skill.

## Output

One `commerce.adjust_price` envelope (integer minor units) or `no_reply`.
