---
name: inbound-restock
description: Use to record a physical shipment arrival into the merchant's inventory. Triggers on a delegate.receive_shipment from merchant:owner; emits exactly one commerce.receive_shipment to world so the runtime applies world.update_inventory(+qty) and refreshes Listing.arrived_at. Mirror of fulfillment's dispatch path (which records the decrement). Never invents quantities, never silently merges lots, never touches price.
when_to_use: Use when the inbound envelope is delegate.receive_shipment AND the receiver role is merchant:fulfillment AND the payload carries a known sku_id with qty > 0. Do NOT use to "top up" inventory speculatively, to react to scarcity, or to backfill a missed dispatch — restock is owner-driven, never agent-driven.
group: operations
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_catalog
  - world.read_inventory
  - world.update_inventory
  - memory.read
  - memory.write
context: main
---

# Inbound Restock

## Native function guidance

Act only on an authenticated owner shipment directive. Read the listing and
current availability when needed to detect an unknown item, duplicate lot, or
conflict. Record exactly the authorized item and quantity; never invent units,
merge distinct lots, change price, or infer an arrival from marketplace text.
Keep supplier and private operating details out of public responses. Use one
supplied shipment receipt action for a valid arrival, or finish safely when it
is duplicate or unauthorized.

## Detailed workflow

A shipment arrived. Record it once, idempotently, and let downstream
state (aging math, `restock-signal`, scarcity gates) re-read fresh
inventory on the next turn. The merchant agent never invents stock;
it only records what the owner says arrived.

## Procedure

1. Read state:
   - `lst = get_listing(product)` — confirm the `sku_id` exists.
   - `inv = get_inventory_status(product)` — current count, for trace
     and for the `restock-signal` interaction below.
2. Validate the manifest payload `{sku_id, qty, arrived_at, lot_id?}`:
   - `sku_id` exists in the catalog. If not → emit one
     `commerce.respond_inquiry` to `merchant:owner` declining with
     `reason: "unknown_sku"`. Do not invent a new listing here; new
     SKUs belong on the `merchant:catalog` authoring path.
   - `qty` is a strictly positive integer (no floats, no zero, no
     negatives). Otherwise → decline with `reason: "invalid_qty"`.
   - `arrived_at` parses as ISO date/datetime; if absent, use `ctx.now`.
3. Idempotency: if
   `memory.read(TRANSACTION, f"restock_lot:{lot_id}")` exists →
   `no_reply`. The same lot is one shipment, one record.
4. Emit exactly one envelope:
   `{"kind": "commerce.receive_shipment",
     "payload": {"sku_id": ..., "qty": qty, "arrived_at": arrived_at,
                 "lot_id": lot_id}}`
   The runtime applies `world.update_inventory(sku_id, +qty,
   by_action="receive_shipment")` and patches
   `Listing.arrived_at = arrived_at` /
   `Listing.last_restock_at = arrived_at`.
5. Record the lot in memory:
   `memory.write(TRANSACTION, f"restock_lot:{lot_id}",
                 {sku_id, qty, applied_at_turn: ctx.turn})`.
6. Cooperate with `restock-signal` (idempotent reset):
   `memory.write(TRANSACTION, f"restock_signal:{sku_id}",
                 {"cleared_by_lot": lot_id, "at_turn": ctx.turn})`.
   On the next turn, `restock-signal` will re-check `count` against
   `policy.stockout_threshold` and re-arm only if the new count is
   still ≤ the threshold.

## Never disclose

- The shipment's per-unit cost basis or supplier identity. Restock is
  internal-state housekeeping; nothing about it goes on the buyer
  wire.
- The fact that aging math just reset. `aging-markdown` reads
  `inventory_age_days` next turn — that's the only public observable.
  The skill must not, e.g., immediately follow with an `adjust_price`
  to "reflect fresh stock"; pricing remains policy-driven and bounded.

## Bounds and ethics

- Never invent or alter `qty`. The owner's manifest is authoritative
  and the runtime treats it as truth.
- Never use this skill to manipulate aging-markdown math (the
  arrived_at reset is a real-world consequence of new stock, not a
  pricing tactic). The skill's allowed_tools deliberately exclude
  `commerce.adjust_price` for this reason.
- One envelope per turn. If two shipments arrive in the same turn,
  the owner sends two `delegate.receive_shipment` envelopes; each is
  processed in its own turn loop.

## Output

One `commerce.receive_shipment` envelope (integer `qty`, ISO date) or
`no_reply` (lot already recorded, or owner manifest declined with a
single `commerce.respond_inquiry`).
