---
name: after-sales-lifecycle
description: Execute cancellation, return, refund, exchange, dispute, and reconciliation workflows using authoritative Platform references and World-backed history.
when_to_use: Use for an after-sales instruction and on platform.after_sales_updated or platform.after_sales_snapshot continuations.
group: operations
user_invocable: false
disable_model_invocation: false
allowed_tools: []
context: main
---

# After-Sales Lifecycle

## Native function guidance

Act only on an authoritative order owned by this merchant. Read order and
evidence state before choosing cancellation, return authorization, refund,
exchange, dispute, evidence, shipment, or reconciliation operations. Apply the
published policy and current lifecycle state; do not invent consent, receipts,
inspection results, or exceptions. Keep private pricing and operational details
confidential. Make one supplied lifecycle action, then use the Platform
continuation rather than assuming the state changed.

## Detailed workflow

After-sales work is a state machine, not a free-form conversation.  The agent
chooses an allowed intent.  Platform validates actor ownership and causal
preconditions.  World alone changes payment, packing, return, refund,
exchange, dispute, inventory, and ledger records.

## Read and continue

Use the caller-scoped Platform reads when the decision needs history:
`commerce.read_payment_history`, `commerce.read_ledger_history`,
`commerce.read_packing_history`, `commerce.read_after_sales_history`, or
`commerce.read_after_sales_policy`, all addressed to
`platform:after-sales`.  Treat `platform.after_sales_snapshot` as the only
authoritative response to those reads.

For mutations, emit one appropriate intent to `platform:after-sales`:

- `commerce.cancel_paid_order`
- `commerce.authorize_return` or `commerce.deny_return`
- `commerce.receive_return`
- `commerce.approve_refund` or `commerce.deny_refund`
- `commerce.authorize_exchange`, `commerce.deny_exchange`, or
  `commerce.complete_exchange`
- `commerce.open_dispute`, `commerce.submit_dispute_evidence`, or
  `commerce.respond_to_dispute`
- `commerce.request_ledger_reconciliation`

Use order, request, authorization, case, dispute, and evidence identifiers
from the inbound request or a prior `platform.after_sales_updated`.  Preserve
them exactly.  Do not jump over a missing lifecycle state.  If a response says
the transition was rejected, do not retry with fabricated identifiers.

One external turn should make one bounded decision.  Use ordinary
`commerce.send_message` only to pass safe returned references to the other
party when coordination is required.  Never emit a `world.*` write and never
claim that a refund, return, exchange, or ruling occurred until Platform has
returned the authoritative continuation.
