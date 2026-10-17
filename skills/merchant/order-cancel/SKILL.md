---
name: order-cancel
description: Use to adjudicate a pre-dispatch cancellation. Triggers on commerce.cancel_order; emits exactly one envelope — commerce.issue_refund (cancelable + full refund) or commerce.respond_inquiry of category "policy" (declined because the order already shipped, was already cancelled, or is in some other terminal status). Cancellation is a status-gated refund; the deterministic cancel_order helper is the source of truth.
when_to_use: Use when the inbound envelope is commerce.cancel_order with an order_id the merchant fulfilled. Do NOT use after the order has been dispatched (the buyer should use commerce.request_return -> return-adjudicate after delivery). Do NOT use to handle a return request on a delivered order (return-adjudicate owns that). Do NOT use to handle a platform.open_dispute (dispute-defense owns that).
group: operations
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_order
  - memory.read
  - memory.write
context: main
---

# Order Cancel

## Native function guidance

Read the authoritative order before deciding a cancellation. Confirm that it
belongs to this merchant and apply the public cancellation policy to its
current payment and dispatch status. Never infer cancellation from a message,
invent a refund, or alter an already terminal order. Keep private pricing and
operational details out of the response. Use one supplied cancellation or
public policy response when justified, and finish safely after the authoritative
continuation.

## Detailed workflow

The merchant's pre-dispatch cancellation path. A buyer (or the merchant
itself on fraud / oversold detection) wants to undo an order **before**
the merchant has dispatched it. The merchant emits exactly one
``commerce.issue_refund`` envelope for the full amount; the PSP
executes the ledger reverse.

This is the **only** path between order-created and order-shipped where
the merchant can give the buyer money back. After dispatch, the buyer's
recourse is ``commerce.request_return`` followed by
``return-adjudicate``; cancellation is no longer in scope.

## Procedure

1. Read state:
   - ``order = get_order(order_id)`` — must exist; carries ``status``,
     ``amount``, ``product``.
2. Run the deterministic adjudication:
   ``decision = cancel_order(order_id, reason=payload.reason
                              or "buyer_requested")``.
   The helper returns the action dict directly — emit it as-is.

### Status-driven decisions

| Status                                | Outcome                                       |
| ------------------------------------- | --------------------------------------------- |
| ``created`` / ``paid`` / ``authorized`` | approve full refund, ``cancellation:<reason>`` |
| ``shipped`` / ``delivered``           | decline ``order_already_shipped``             |
| ``cancelled``                         | decline ``already_cancelled``                 |
| (unknown)                             | decline ``not_cancellable``                   |
| (missing order)                       | decline ``unknown_order``                     |

### After the decision

3. Record in memory:
   ``memory.write(TRANSACTION, f"cancel:{order_id}",
                  {decision, reason, amount, applied_at_turn: ctx.turn})``.
4. Emit exactly one envelope.

## Never disclose

- The order's margin / per-unit cost basis on a decline. The decline
  reason is a public status fact ("already shipped"), not a private
  number. ``private-utility-guard`` runs on every emit.
- Internal restock state. If the cancellation lands inventory back, the
  world's update_inventory write is the visible effect; the merchant's
  envelope only carries the refund request.
- Whether the merchant has flagged the buyer or order as suspicious —
  that is the platform's adjudication surface, not a cancellation
  decline.

## Bounds and ethics

- Full refunds only. Cancellation is a binary status flip; partial
  refunds belong to ``return-adjudicate`` (post-delivery) when the
  item arrives in a degraded condition.
- One envelope per turn.
- Cancellation does not negotiate. A buyer asking "can I get most of
  it back and keep the order open?" is asking for a return that hasn't
  happened, not a cancellation; refuse via the normal decline path or
  defer to ``return-adjudicate``.

## Output

One ``commerce.issue_refund`` envelope (approved) or one
``commerce.respond_inquiry`` envelope of ``category: "policy"``
(declined).
