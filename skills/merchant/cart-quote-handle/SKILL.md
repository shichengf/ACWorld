---
name: cart-quote-handle
description: Participate in the mediated cart quote lifecycle by answering Platform quote requests and validating quote or settlement continuations.
when_to_use: Use on platform.cart_quote_request, platform.cart_quote, or platform.cart_settlement addressed to this merchant.
group: operations
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.get_listing
context: main
---

# Cart Quote Handling

## Native function guidance

Use authoritative listing reads to evaluate the exact line items in a
Platform-mediated cart request. Do not invent totals, reserve
stock locally, alter quantities, or treat a buyer message as an authoritative
quote or settlement. Use the supplied quote action only for a valid request
within this merchant's catalog. Treat later Platform quote and settlement
continuations as authoritative and finish safely when no further merchant
decision is required.

## Detailed workflow

Cart quotes are Platform-owned snapshots over authoritative catalog and
inventory state.  The merchant confirms its participation but does not create
orders, reserve inventory, or settle funds directly.

## Quote request

1. Read `payload.request`.  Require a non-empty `request_id` issued by
   Platform.  If lines are present, verify only this merchant's SKUs with
   `world.get_listing`; do not alter buyer quantities or fill policy.
2. Emit exactly one `commerce.request_cart_quote` to `platform:checkout` with
   `payload: {"request_id": <the exact returned id>}`.  Use a stable
   idempotency key.
3. Do not copy a buyer budget or any private price floor into the request.

## Quote and settlement continuation

On `platform.cart_quote`, treat `payload.quote` as an immutable Platform
projection.  A merchant normally records or inspects it and returns
`no_reply`; checkout authorization belongs to the buyer or authorized
principal.  On `platform.cart_settlement`, likewise verify that the receipt
references the expected quote and return `no_reply` unless the inbound
explicitly requests a public acknowledgement.

Never emit `platform.checkout_cart` for a buyer and never write cart, order,
inventory, or ledger state directly.  All commercial writes remain mediated
by Platform and committed by World.
