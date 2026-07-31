---
name: return-refund
description: Post-purchase return/refund — when the mandate asked for a return, request it after the settlement receipt and close the loop on the refund.
---

# Buyer · Return / Refund

## Native function guidance

Request a return only when the mandate calls for it and an authoritative order
shows a completed purchase that is eligible for the requested remedy. Use the
exact order and public reason supported by the current facts. Do not invent a
refund, duplicate a request, or treat a merchant message as proof that money
moved. When the authoritative refund or lifecycle response arrives, verify that
it belongs to this buyer's order and then finish safely unless another supplied
action is explicitly required.

## What this skill is for

Some mandates are "buy it, then return it" (testing the return/refund path).
This skill runs *after* a successful purchase to exercise that path. It does
nothing on an ordinary buy — it only acts when the mandate explicitly asked for
a post-purchase return.

## When this activates

- You received a `platform.settlement_receipt` **and**
  `TRANSACTION.return_after_purchase` is true (mandate-parsing copies the
  mandate's `return_after_purchase` flag there). → request the return.
- You received a `commerce.issue_refund`. → the loop is closed; you are done.

If `TRANSACTION.return_after_purchase` is not set, do **nothing** here — the
purchase is complete, let `purchase-confirmation` finish the turn.

## What to do

### On the settlement receipt (return wanted)

Emit a single `commerce.request_return` to **`platform:psp`** for the order you
just settled:

```
{"step": "emit_envelope",
 "envelope": {
   "to": "platform:psp",
   "idempotency_key": "refund:<mandate_id>",
   "action": {"kind": "commerce.request_return",
              "payload": {"order_id": "<TRANSACTION.pending_settlement_order_id>",
                          "reason": "<short reason>"}}}}
```

Use the order id you settled (`TRANSACTION.pending_settlement_order_id`). A
distinct `idempotency_key` (the refund key, not the settle key) keeps the refund
a separate, idempotent transaction from the settle. The platform verifies the
sku is `returnable` (the same world attribute you grounded for a `returnable`
must_have) and runs the atomic refund (the order goes REFUNDED, inventory is
restocked, a reversing ledger entry is written), then replies
`commerce.issue_refund`.

### On the refund (`commerce.issue_refund`)

The return is complete — record it if useful and `no_reply`. Do not re-request.

## Boundary

A return is only honored for a `returnable` sku. You verified `returnable` by
grounding the listing (it is never satisfied by `offer.claims`); the platform
enforces the same attribute, so a non-returnable item's refund is refused.
