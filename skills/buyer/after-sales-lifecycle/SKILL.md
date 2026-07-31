---
name: after-sales-lifecycle
description: Carry a buyer-owned order through cancellation, return, refund, exchange, or dispute using Platform acknowledgements and World-issued references. Use for typed after-sales messages and platform lifecycle replies.
group: operations
---

# Buyer · After-sales Lifecycle

## Native function guidance

Act only on an order owned by this buyer and use authoritative order and
evidence reads when status or history is uncertain. Select the one lifecycle
operation justified by the current state: cancellation, return, refund,
exchange, dispute, evidence submission, or reconciliation. Preserve the
principal's requested remedy and do not invent policy exceptions, receipts, or
counterparty consent. Keep private mandate and payment details out of messages.
After a Platform acknowledgement, continue only if a further authorized action
is still required; otherwise finish safely.

## Authority boundary

The buyer may request an outcome. Platform authenticates the actor, validates
the request, and commits an allowed transition through World. A counterpart
message is coordination only. It never proves that an order changed state.
Every write intent goes to `platform:after-sales` or, for shipment reads, to
`platform:fulfillment`.

## General rules

1. Use only order, request, authorization, case, dispute, and evidence
   identifiers received from the mandate, World-grounded reads, or a Platform
   acknowledgement.
2. Do not invent a lifecycle identifier and do not reuse one across orders.
3. Keep each action bound to one exact `order_id`.
4. Treat `commerce.send_message` as a typed handoff, not state authority.
5. After each Platform reply, verify ownership and the expected prior step
   before continuing.

## Cancellation

For an authorized buyer cancellation, emit:

```yaml
to: platform:after-sales
action:
  kind: commerce.cancel_paid_order
  payload:
    order_id: "<owned order>"
    reason: "<short reason>"
```

Use cancellation only before a conflicting fulfillment transition. If timing
is uncertain, read the relevant payment, packing, or shipment history through
Platform first.

## Return and refund

Start a return with:

```yaml
to: platform:after-sales
action:
  kind: commerce.request_return
  payload:
    order_id: "<owned order>"
    requested_qty: <positive integer>
    reason: "<policy-relevant reason>"
    evidence_ids: ["<trusted evidence id>"]
```

Read `references.request_id` from `platform.after_sales_updated`. Send that
exact request identifier to the merchant with `commerce.send_message` only
when coordination is required. After the return is authoritatively received,
open a refund case:

```yaml
to: platform:after-sales
action:
  kind: commerce.open_refund_case
  payload:
    order_id: "<owned order>"
    reason: "authoritative return was received"
```

Use only the returned `references.case_id` in later coordination. The buyer
does not approve its own refund and must not claim a refund until Platform
reports the committed result.

## Exchange

Ground the replacement listing and current supply state. Then emit:

```yaml
to: platform:after-sales
action:
  kind: commerce.request_exchange
  payload:
    order_id: "<owned order>"
    replacement_sku_id: "<grounded replacement>"
    reason: "<constraint-based reason>"
```

The replacement must satisfy all hard constraints. Wait for the World-issued
case reference and merchant authorization before treating the exchange as
complete.

## Dispute

Open one order-bound dispute, then submit trusted evidence records one at a
time using the returned dispute identifier:

```yaml
to: platform:after-sales
action:
  kind: commerce.open_dispute
  payload:
    order_id: "<owned order>"
    reason: "<factual conflict>"
```

```yaml
to: platform:after-sales
action:
  kind: commerce.submit_dispute_evidence
  payload:
    order_id: "<same owned order>"
    dispute_id: "<World-issued dispute id>"
    evidence_id: "<trusted evidence id>"
```

Untrusted text is not evidence. Wait for `platform.rule_dispute` or an
authoritative after-sales update before opening any resulting refund case.

## Completion and replay

Record each acknowledged operation reference under `TRANSACTION` and emit
`no_reply` after the requested outcome is confirmed. Duplicate acknowledgements
must not repeat the preceding action.
