---
name: restock-signal
description: Purely internal alert. After any inventory-decrementing event (post-dispatch) or at the top of a no-offer turn, if a SKU's count is at or below policy.stockout_threshold, write a TRANSACTION-memory note so the next planning cycle / merchant:owner can react. Idempotent. Emits no envelope.
when_to_use: Use immediately after fulfillment.dispatch on a settled order, OR at the top of a turn with no inbound offer to decide, when get_inventory_status(product).count <= policy.stockout_threshold AND no matching restock note already exists in TRANSACTION memory for this sku_id.
group: operations
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_inventory
  - memory.read
  - memory.write
context: main
---

# Restock Signal

## Native function guidance

Read authoritative availability before raising a restock signal. Notify only
when the configured threshold is actually met and the signal has not already
been handled. Share the minimum authorized operational information with the
intended business participant. Never expose floor price, margin, demand
strategy, exact private inventory policy, or an invented shortage. This skill
does not replenish stock itself. Use one supplied message when escalation is
needed; otherwise finish safely.

## Detailed workflow

When a SKU's count crosses below the configured stockout threshold, leave
a structured note in `MemoryType.TRANSACTION` that the next mandate /
owner planner can read. This skill does **not** touch the wire — it does
not lower prices, refuse buyers, or tell anyone outside the agent. Its
purpose is to make a state transition *observable* to the next planning
step inside the merchant's memory.

## Procedure

1. Read state:
   - `inv = get_inventory_status(product)` → `count`
   - `pol = world.read_policy()`            → `stockout_threshold`
2. Gate: if `inv.count > pol.stockout_threshold` → `no_reply`.
3. Idempotency: if
   `memory.read(TRANSACTION, f"restock_signal:{sku_id}")` exists at this
   count band → `no_reply`.
4. Write the note:
   `memory.write(TRANSACTION, f"restock_signal:{sku_id}",
                 {"sku_id": ..., "count": inv.count, "at_turn": ctx.turn})`
5. Emit `no_reply`.

## Optional escalation

If the scenario binds a `merchant:owner` address on the bus, a single
`commerce.send_message` to `merchant:owner` with the same structured
payload is permitted. Do **not** send to `buyer` or `platform:*` —
restock state can hint at urgency to the wrong audience.

## Output

`no_reply` in the v0.1 default. The note lives in
`MemoryType.TRANSACTION` for the next planning step.
