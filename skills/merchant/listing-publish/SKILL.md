---
name: listing-publish
description: Use to author catalog state — publish a new SKU, edit a listing's attributes / permitted_claims, or delist an existing SKU. Triggers on a delegate.update_listing from merchant:owner; emits exactly one commerce.update_listing to world so the runtime applies world.update_catalog. The only merchant:catalog-role skill — the agent's authoring path. Never used to lower or raise the public list_price (that is aging-markdown / demand-driven-markup).
when_to_use: Use when the inbound envelope is delegate.update_listing AND the receiver role is merchant:catalog AND the payload carries a valid op ∈ {"publish","update","delist"} with the fields that op requires. Do NOT use to react to inventory (inbound-restock owns +qty; fulfillment owns -qty), to demand (the markup/markdown skills own list_price), or to a buyer turn — catalog authoring is owner-driven, never buyer-driven.
group: discovery
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_catalog
  - world.update_catalog
  - memory.read
  - memory.write
context: main
---

# Listing Publish

## Native function guidance

Act only on an authenticated owner directive for this merchant's catalog. Read
the current listing, availability, or claim state when needed to distinguish a
publish, update, delist, duplicate, or conflict. Apply only owner-authorized
public fields and grounded claims. Do not use catalog editing to change
inventory, perform dynamic pricing, or expose private policy. Make one supplied
listing action when the directive is valid; otherwise give the permitted public
response or finish safely.

## Detailed workflow

The merchant's authoring path. The owner says "list this product", "edit
that attribute", or "take it down"; the agent emits exactly one
`commerce.update_listing` so the runtime updates the world catalog. The
agent never invents listings on its own — every change traces to an owner
directive (idempotent by `op_id`).

## Procedure

Validate the manifest payload `{op, product, sku_id?, fields?, op_id?}`:

1. `op` ∈ {`"publish"`, `"update"`, `"delist"`}. Anything else → emit
   one `commerce.respond_inquiry` to `merchant:owner` declining with
   `reason: "unknown_op"`.
2. Per-op fields:
   - **publish**: `product`, `list_price` (public, integer minor units),
     `attributes` (dict), `permitted_claims` (list), `must_not_claim`
     (list), optional `inventory` (defaults to 0 — restock adds stock).
     The `product` must NOT already exist in the catalog (`get_listing`
     raises `ProductNotFound`); if it does, decline with
     `reason: "already_listed"`. The merchant's private `floor_price`
     stays in `MerchantData` and `MemoryType.PRIVATE_UTILITY` only — it
     is **never** passed to `update_listing` and **never** appears in
     the payload. Wire payload carries public catalog fields only.
   - **update**: `product` and a `fields` dict with the subset to change
     (attributes / permitted_claims / must_not_claim / status). The
     `product` MUST already exist. Decline with `reason: "unknown_sku"`
     otherwise. Never include `list_price` in `fields` here — see
     "When NOT to fire" above. (`floor_price` is private and is never
     permitted in any catalog payload, publish or update.)
   - **delist**: `product` only. Sets `status: "delisted"`. Decline with
     `reason: "unknown_sku"` if the product does not exist.
3. Integer-money check: any monetary field in the payload is integer
   minor units (`int`, not float). Reject with `reason: "non_integer_money"`.
4. Idempotency: if
   `memory.read(TRANSACTION, f"listing_op:{op_id}")` exists → `no_reply`.
5. Emit exactly one envelope:
   `{"kind": "commerce.update_listing",
     "payload": {"op": op, "sku_id": ..., "product": ..., "fields": {...}}}`.
   The runtime applies `world.update_catalog` as a governed write; the
   agent only requests it via the envelope.
6. Record the directive in memory:
   `memory.write(TRANSACTION, f"listing_op:{op_id}",
                 {op, product, applied_at_turn: ctx.turn})`.

## Never disclose

- The supplier identity, per-unit cost basis, or any private margin math
  used to pick the seed `list_price` on a publish. The wire only carries
  the public catalog fields.
- The `floor_price` lives in `MerchantData` (set at seller-onboarding
  time, outside this skill) and in `MemoryType.PRIVATE_UTILITY`. It is
  never a parameter to any envelope constructor and never appears in a
  catalog payload — not on publish, not on any subsequent skill emit.
  `private-utility-guard` is a backstop, not the primary defense; the
  primary defense is that no constructor in this lane accepts
  `floor_price` as input.

## Bounds and ethics

- Never invent or alter the owner's manifest fields. The directive is
  authoritative; this skill validates and forwards, not authors.
- Never bundle multiple ops in one envelope. Two directives ⇒ two turns.
- A `delist` does not refund open orders or cancel in-flight dispatches —
  those have their own skills (`order-cancel` / `return-adjudicate`).
  Delist is a catalog-state change, nothing more.

## Output

One `commerce.update_listing` envelope (op + product + fields) or
`no_reply` (op already recorded, or owner manifest declined with a single
`commerce.respond_inquiry`).
