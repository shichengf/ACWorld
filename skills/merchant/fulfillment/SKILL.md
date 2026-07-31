---
name: fulfillment
description: Use after settlement to dispatch the order, and to handle return requests strictly per the stated refund policy.
when_to_use: Use when the inbound envelope signals a settled order to dispatch, or is commerce.request_return.
group: operations
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_policy
  - world.update_inventory
  - world.update_order_status
  - memory.read
  - memory.write
context: main
---

# Fulfillment & Returns

## Native function guidance

Read the authoritative order, listing, and availability before a fulfillment
or return decision. Dispatch only a settled eligible order for the exact
authorized quantity. For a return, apply the published policy and current order
state rather than inventing an exception or relying on message text. Keep
private operational and pricing details confidential. Make one supplied
dispatch, returned, refund, or public response action, then wait for the
authoritative continuation instead of assuming state changed.

## Detailed workflow

The post-deal half of the sell-side journey (VIBE_COMMERCE.md §4.2:
settle → dispatch → optional return).

## On a settled order

1. Confirm the order is settled (the inbound is the platform/PSP settlement
   notification — the merchant never moves money itself).
2. Emit `commerce.dispatch` with the shipment record `{ order_id, carrier }`.
   The runtime applies `world.update_inventory` (decrement) and
   `world.update_order_status` (→ shipped) as governed writes; the merchant
   only requests them via the envelope.
3. Persist the receipt: `memory.write(TRANSACTION, order_id, {...})` — needed
   if a return arrives later.

## On `commerce.request_return`

1. Read the stated refund policy (`world.read_policy` / the policy seeded
   into memory): window, returnable conditions.
2. If the policy **permits** the return → emit the refund request
   (`commerce.issue_refund_request`) for the original `order_id` / amount.
   The PSP executes the ledger move; the merchant does not.
3. If the policy **forbids** it → emit one `commerce.respond_inquiry`
   declining with the public policy reason
   (e.g. `decline_reason: "return_window_closed"`). Do not invent ad-hoc
   terms and do not negotiate the refund here.

Emit 0 or 1 envelope per turn. Never expose private utility (floor, urgency)
in a dispatch note, decline message, or receipt echoed on the wire.
