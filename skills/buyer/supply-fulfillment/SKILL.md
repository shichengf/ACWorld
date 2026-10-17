---
name: supply-fulfillment
description: Read authoritative supply or shipment state through Platform and make a bounded purchase, substitution, backorder, or shipment-resolution decision. Use for supply-shaped mandates and platform supply, allocation, or shipment replies.
group: operations
---

# Buyer · Supply and Fulfillment

## Native function guidance

Use authoritative order, supply, allocation, and shipment information before
making a fulfillment decision. Choose only a mandate-authorized purchase,
substitution, backorder, or shipment resolution. A substitute must still meet
hard constraints and budget; unavailable stock or an unapproved delay is not
permission to relax them. Use the supplied read or resolution actions rather
than inferring logistics state from marketplace text. Take one bounded terminal
business action after the required evidence, or finish safely when no action is
authorized.

## Authority boundary

Inventory, reservations, allocations, shipments, and order state belong to
World. The buyer emits compact intents only to Platform. Never infer that a
message changed stock, and never emit a `world.*` action.

## Read before deciding

For inventory, availability, restock, or substitution work, emit:

```yaml
step: emit_envelope
envelope:
  to: platform:supply
  idempotency_key: "read-supply:<mandate_id>"
  action:
    kind: commerce.read_supply_state
    payload:
      sku_ids: ["<authorized sku id>"]
```

For a delivery exception, use the World-issued shipment identifier:

```yaml
step: emit_envelope
envelope:
  to: platform:fulfillment
  idempotency_key: "read-shipment:<shipment_id>"
  action:
    kind: commerce.read_shipment
    payload:
      shipment_id: "<authorized shipment_id>"
```

Read every candidate needed for the decision. Do not reuse an earlier supply
snapshot after a stated supply event or shipment update.

## Handle `platform.supply_state`

The `states` array is authoritative. Each row can include `sku_id`,
`merchant_id`, `available_qty`, `reserved_qty`, `eta_day`,
`unit_price_cents`, and `version`.

1. Match rows by exact `sku_id`, never by array position alone.
2. Apply hard product, quantity, delivery, and budget constraints.
3. For substitution, choose only a grounded candidate that satisfies every
   hard constraint and has positive available quantity.
4. For partial fill or backorder, preserve the requested quantity and the
   mandate's explicit partial-fill authority. Platform and World decide the
   actual allocation.
5. For price changes, compare the current World price to the private budget.
   Decline through `delegate.reject_purchase` when the authorized constraints
   no longer hold.

When the decision is to purchase, write the chosen row and its supply version
to `TRANSACTION.selected_offer`, then emit `platform.settle_payment` to
`platform:psp`. Copy the chosen SKU, merchant, quantity, and current price
exactly. Use `allow_partial: true` only when the mandate explicitly permits a
partial allocation.

## Handle `platform.fulfillment_allocation`

Verify the allocation belongs to this buyer and pending order. Record filled
and backordered quantities. Do not issue a second settlement for the same
order. If zero units were filled, report that outcome to the principal without
claiming a payment occurred.

## Handle `platform.shipment_state`

Check the shipment identifier, order ownership, status history, and version.
Choose a resolution allowed by the mandate and the observed state. Emit only
one compact resolution intent:

```yaml
step: emit_envelope
envelope:
  to: platform:fulfillment
  idempotency_key: "resolve:<shipment_id>:<resolution>"
  action:
    kind: commerce.resolve_shipment
    payload:
      shipment_id: "<exact shipment_id>"
      resolution: refund
```

Allowed resolutions are determined by Platform. A replacement resolution must
also include a grounded `replacement_sku_id`. Never invent a replacement or
claim that it shipped before `platform.shipment_resolved` confirms the World
commit.

## Duplicate replies

Store the last supply version, allocation id, and shipment resolution id.
Duplicate or older replies produce `no_reply`; they never repeat a purchase or
resolution intent.
