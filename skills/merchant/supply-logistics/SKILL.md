---
name: supply-logistics
description: Read authoritative supply and shipment state, then request bounded inventory, allocation, or shipment operations through Platform.
when_to_use: Use for inventory, fulfillment, allocation, or shipment instructions and on Platform supply, allocation, or shipment continuations.
group: operations
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.get_listing
context: main
---

# Supply and Logistics

## Native function guidance

Read authoritative listing, availability, order, supply, and shipment state
before deciding. Make only a bounded inventory update, allocation, shipment
resolution, or operational message authorized for this merchant and consistent
with the current facts. Never invent supply, allocation, arrival, or delivery,
and do not expose private policy or supplier details. Use one supplied business
action after the required evidence, then wait for Platform and World to confirm
the new state.

## Detailed workflow

Make operational decisions from current World-backed projections.  Never infer
available quantity, allocation identity, shipment status, or version from an
earlier prompt when Platform can return the authoritative state.

## Read before deciding

- For inventory or allocation work, emit `commerce.read_supply_state` to
  `platform:supply` with the requested `sku_ids`.
- For delivery work, emit `commerce.read_shipment` to
  `platform:fulfillment` with the supplied `shipment_id`.

These are actor intents to Platform, not direct World calls.  Wait for
`platform.supply_state` or `platform.shipment_state` before selecting an
operation.

## Bounded operations

- Inventory event: emit `commerce.update_supply` to `platform:supply` using
  the exact SKU, event, quantity, and expected version authorized by the
  request.
- Competing orders: emit `commerce.allocate_fulfillment` to
  `platform:fulfillment` with an `allocation_id`, owned `sku_id`, and the
  ordered `priority_order_ids`.  Platform and World perform the atomic
  allocation and return `platform.allocation_batch`.
- Shipment exception: after reading `platform.shipment_state`, emit
  `commerce.resolve_shipment` to `platform:fulfillment` with the returned
  `shipment_id` and one supported resolution.  Include a replacement SKU only
  for a replacement resolution.
- If the request asks only for a report, send `commerce.send_message` with the
  exact safe projection returned by Platform.  Do not embellish quantities or
  estimated arrival times.

Treat `platform.supply_event_applied`, `platform.fulfillment_allocation`,
`platform.allocation_batch`, and `platform.shipment_resolved` as receipts.
Continue only when the receipt advertises a next action.  Never send a
`world.*` write, never decrement stock in memory as if it were authoritative,
and never invent allocation or shipment identifiers.
