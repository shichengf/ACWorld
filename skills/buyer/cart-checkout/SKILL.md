---
name: cart-checkout
description: Select or preserve authorized cart lines, request and validate a World-authoritative multi-line quote, then check out by its exact identifier. Use for a mandate with specified lines or a bounded public cart-planning problem and quote authority, and for platform.cart_quote or platform.cart_settlement replies.
group: authorization
---

# Buyer · Cart Quote and Checkout

## Native function guidance

Use the supplied cart quote action in one of two explicit modes. For a
specified cart, preserve the authorized line items and quantities exactly. For
a bounded public planning problem, select lines only from its declared offers
by applying its public constraints, calculation rules, objective, and tie-break.
Before checkout, compare the World-backed quote with private budget, listing,
stock, and balance evidence when those reads are available. Reject stale,
altered, incomplete, unaffordable, or unauthorized quotes. Check out only by
the authoritative quote returned by Platform, and never reconstruct prices or
identifiers from text. After a confirmed cart settlement, finish without
creating a second purchase.

## Authority boundary

The buyer supplies line-item intent. It either preserves principal-specified
lines or derives them from a complete public planning problem. Platform
validates that intent and World owns executable pricing, quote state, inventory
reservation, checkout, orders, and ledger commits. A buyer may compare declared
candidate costs while planning, but must treat the World quote as authoritative
for execution. Never emit a `world.*` action.

## Start a cart quote

Use this path only when the principal's authorized purchase mandate contains a
market identifier and either:

- one or more requested lines, each with an exact `sku_id` and positive integer
  `qty`; or
- cart quote authority plus a complete bounded public planning problem whose
  offers, requirements, relations, charges, hard constraints, objective, and
  tie-break determine the selected lines.

1. Read every requested listing with `world.get_listing`. Reject the workflow
   if a listing is absent, belongs to an unexpected merchant, or conflicts with
   a hard mandate constraint.
2. In specified-cart mode, preserve the line set and quantities exactly; do not
   silently drop, split, or substitute. In planning mode, select only declared
   offers, satisfy every public hard constraint, and apply the declared
   objective and tie-break without using private or undeclared facts.
3. Emit one request to `platform:checkout`:

```yaml
step: emit_envelope
envelope:
  to: platform:checkout
  idempotency_key: "quote:<mandate_id>"
  action:
    kind: commerce.request_cart_quote
    payload:
      market_id: "<authorized market_id>"
      mandate_id: "<authorized mandate_id>"
      lines:
        - sku_id: "<grounded sku_id>"
          qty: <positive integer>
      fill_policy: all_or_none
      backorder_policy: reject
```

Copy any explicitly authorized fill or backorder policy instead of changing it.
Store the requested line tuple in `TRANSACTION.pending_cart_lines` so the reply
can be checked against the original intent.

## Handle `platform.cart_quote`

The payload contains a World-issued `quote`. Treat that object as immutable.

1. Require a nonempty `quote.quote_id`.
2. Verify that its market, mandate, buyer, lines, quantities, and currency agree
   with the pending request.
3. Check the World-derived total against the buyer's private budget locally.
   Never copy the private budget into an envelope.
4. Confirm the quote is still valid and permits the requested fill policy.
5. If every check passes, emit checkout with the quote identifier only:

```yaml
step: emit_envelope
envelope:
  to: platform:checkout
  idempotency_key: "checkout:<quote_id>"
  action:
    kind: platform.checkout_cart
    payload:
      quote_id: "<exact World-issued quote_id>"
```

Do not reproduce prices, totals, merchant identities, or line items in the
checkout payload. World resolves all of them from the persisted quote.

If the quote fails a mandate or authority check, emit `delegate.reject_purchase`
to the principal with a short non-private reason. Do not attempt checkout.

## Handle `platform.cart_settlement`

Verify that the response refers to the pending quote and expected buyer. Store
only the returned order-group and receipt references under `TRANSACTION`, clear
the pending cart state, and emit `no_reply`. A duplicate settlement reply is an
idempotent observation and must not trigger another checkout.
