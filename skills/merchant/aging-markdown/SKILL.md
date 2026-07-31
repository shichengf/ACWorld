---
name: aging-markdown
description: Use proactively (not in response to a buyer offer) to take a single deterministic step down on a listing's public list_price when the listing has aged past the markdown threshold and stock is NOT scarce. Emits exactly one commerce.adjust_price; never reveals the floor or any urgency cue.
when_to_use: Use when no inbound offer needs a decision this turn AND get_inventory_status(product).inventory_age_days > policy.markdown_rules.after_days AND get_inventory_status(product).days_to_stockout > policy.scarcity_window_days AND the listing has not already been marked down for this aging step. Do NOT use during a propose_offer / counter_offer turn (pricing-negotiate owns that). Do NOT use to set or reset the target band (price-discovery owns that).
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

# Aging Markdown

## Native function guidance

Read the authoritative listing and availability before considering a markdown.
Lower the public price only when the configured age and movement evidence
supports it and stock is not scarce. Keep the new price within every private
floor and policy bound. Never reveal the floor, urgency, inventory count, or
the private reason for the change. Use one supplied price adjustment for a
valid bounded markdown; otherwise finish safely without changing the listing.

## Detailed workflow

Sell-side dynamic pricing for listings that are *aging* but not *scarce*.
When a SKU sits past `markdown_rules.after_days`, take **one** deterministic
step down on its public `list_price` so buyer-side discovery ranks it more
competitively — without ever signaling urgency or disclosing the floor.

## Procedure

1. Read state:
   - `inv = get_inventory_status(product)`
   - `lst = get_listing(product)` — list_price, floor_price (floor is PRIVATE)
   - `pol = world.read_policy()` — markdown_rules, scarcity_window_days
2. Gate (any false → `no_reply`):
   - `inv.inventory_age_days > pol.markdown_rules.after_days`
   - `inv.days_to_stockout > pol.scarcity_window_days`
   - no matching entry in `memory[PREFERENCE].last_markdown[sku_id]`
3. Compute the new list price deterministically:
   `new_lp = aged_stock_discount(lst.list_price, inv.inventory_age_days,
                                  lst.floor_price, pol.markdown_rules)`.
   The helper caps total discount at `max_pct` and guarantees
   `new_lp > floor`.
4. If `new_lp >= lst.list_price` → `no_reply` (nothing to do this step).
5. Emit exactly one envelope via:
   `propose_markdown(product, new_lp)` →
   `{"kind": "commerce.adjust_price",
     "payload": {"sku_id": ..., "list_price": new_lp}}`
6. Record the step:
   `memory.write(PREFERENCE, "last_markdown",
                 {sku_id: {"price": new_lp,
                           "age_step_days": inv.inventory_age_days}})`

## Never disclose

- Neither `floor_price` nor any number derived from it (`% above floor`,
  spread vs floor, …). The new `list_price` is public; how it was computed
  is not.
- `urgency_to_sell` — markdown depth follows `markdown_rules`, not buyer
  signaling. The `private-utility-guard` invariant runs on every emit;
  defer to it before sending.

## Output

One `commerce.adjust_price` envelope (integer minor units) or `no_reply`.
