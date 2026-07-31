---
name: order-intake
description: Use to record an order that the buyer (or platform) just created against this merchant, so the merchant's TRANSACTION memory has the order before settlement / dispatch / return / dispute paths need to find it. Triggers on a commerce.create_order inbound from buyer or platform; emits exactly one commerce.respond_inquiry acknowledging the record. Never modifies the catalog or inventory — those moves belong to fulfillment / world writes.
when_to_use: Use when the inbound envelope is commerce.create_order AND the merchant has previously accepted the matching offer (or the platform has confirmed the deal). Do NOT use to "accept" an offer (pricing-negotiate owns that). Do NOT use to dispatch (fulfillment owns that, after platform.settlement_receipt). Do NOT use during a return or dispute turn (those have their own skills and read the order this skill recorded).
group: operations
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_order
  - memory.read
  - memory.write
context: main
---

# Order Intake

## Native function guidance

Use authoritative listing, availability, and order reads to verify that an
inbound order belongs to this merchant and matches the current commercial
facts. Accept or acknowledge only a valid, nonduplicate order within the
merchant's authority. Do not treat intake as payment, dispatch, inventory
mutation, or permission to change terms. Never expose floor price or internal
policy. Make one supplied acknowledgement or acceptance when appropriate;
otherwise give the permitted public response or finish safely.

## Detailed workflow

The merchant's order-side bookkeeping. A buyer or platform announces an
order against this merchant — typically as the next envelope after the
merchant's own `commerce.accept_offer`. This skill **records** the order
into `MerchantData.orders` + `MemoryType.TRANSACTION` so every later
skill that needs to look it up — `fulfillment` (on settlement),
`return-adjudicate` (on return), `dispute-defense` (on a contested
dispute) — finds a single authoritative row.

The merchant agent's TRANSACTION memory is the source of truth for its
own orders. In the real runtime the row also lives in the World's
`orders` table per `world.create_order` (which the platform's PSP
emits); this skill's mirror is what the merchant's own skills operate
against, without needing a `world.read_order` round-trip per turn.

## Procedure

1. Validate the manifest payload `{order_id, product, qty, amount,
   buyer_id?, status?, delivered_at?}`:
   - `order_id` is non-empty and not already recorded in memory.
   - `product` resolves via `get_listing(product)`; if not, emit
     `commerce.respond_inquiry` declining with `reason: "unknown_sku"`.
   - `qty` is a strictly positive integer.
   - `amount` is a non-negative integer (minor units; zero allowed for
     gifted / sample SKUs).
2. Idempotency: if
   `memory.read(TRANSACTION, f"order:{order_id}")` exists → emit the
   stored ack envelope and `no_reply` no second write.
3. Record the row via:
   `record_order(order_id, product=..., qty=..., amount=...,
                  buyer_id=..., status=...)`.
   The helper enforces every boundary check above and refuses
   `ToolError("duplicate_order")` if the row already exists.
4. Persist the bookkeeping:
   `memory.write(TRANSACTION, f"order:{order_id}",
                 {order_id, product, qty, amount, status,
                  recorded_at_turn: ctx.turn})`.
5. Emit exactly one envelope:
   `{"kind": "commerce.respond_inquiry",
     "payload": {"sku_id": ..., "product": ...,
                 "category": "general",
                 "answer": {"ack": "order_recorded",
                             "order_id": ..., "qty": ..., "amount": ...}}}`.

## Never disclose

- The order's per-unit margin, supplier cost, or any private negotiation
  history. The wire only carries the public order fields (id, product,
  qty, amount, status).
- The fact that the merchant's `pricing-negotiate` previously emitted an
  accept_offer at any particular agreed price. The amount in this
  envelope is the public settled amount, not a derivation of the floor.

## Bounds and ethics

- One envelope per turn. A buyer creating two orders in one inbound is
  two turns, not two records.
- The merchant does not write the world's `orders` table directly — the
  platform's PSP owns `world.create_order` per the role partition. This
  skill is the merchant's *local mirror*, not the world write.
- A retried `commerce.create_order` for the same `order_id` is idempotent
  by design — the same ack is emitted and no second TRANSACTION write
  happens. The runtime layers an additional idempotency check via the
  `idempotency_cache` table (see `WORLD_DATABASE_DESIGN.md` §3.3).

## Output

One `commerce.respond_inquiry` envelope (ack with the recorded order
fields) on success, or one `commerce.respond_inquiry` declining with
a public reason (`"unknown_sku"`, `"duplicate_order"`).
