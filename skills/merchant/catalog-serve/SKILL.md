---
name: catalog-serve
description: Use to answer buyer-side discovery — search hits and SKU/offer requests — with a truthful GroundedOffer that never exposes private utility.
when_to_use: Use when the inbound envelope is commerce.request_offer, commerce.get_sku, or a search-driven inquiry routed to the merchant.
group: discovery
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_catalog
  - world.read_inventory
  - world.read_policy
  - memory.read
context: main
---

# Catalog Serve

## Native function guidance

Answer discovery from the merchant's current authoritative listing and stock
evidence. Use supplied listing and availability reads when either fact is
uncertain. Return only public product attributes, permitted grounded claims,
price, and a truthful availability statement. Never reveal floor price, exact
inventory, urgency, margin, or private policy. Do not advertise a missing,
delisted, unsupported, or unavailable item as purchasable. Make one supplied
catalog or inquiry response, or finish safely if no truthful response is
possible.

## Detailed workflow

Answer buyer discovery with a concrete, **truthful** offer.

## Procedure

1. Resolve the requested `sku_id` / query against the catalog with
   `world.read_catalog` (and `world.read_inventory` for availability).
2. Build a `GroundedOffer` (VCP_SPEC.md §4.3): `offer_id`, `sku_id`, `qty`,
   `unit_price` (**integer minor units**), `fulfillment`, `claims`,
   `expires_at`, `idempotency_key`.
3. `claims` MUST be a subset of `OfferMandate.truthfulness.permitted_claims`
   and MUST NOT include anything in `must_not_claim`. The platform's
   `verify_claims` will score this; unsupported claims fail claim grounding.
4. Emit exactly one `commerce.respond_inquiry` (or `commerce.counter_offer`
   carrying the `GroundedOffer`) back to the requesting buyer role.

## Never disclose

- No `floor_price` / `min_acceptable_price` / `urgency_to_sell` (all
  **PRIVATE_UTILITY**) — not as a number, a hint, or a unit conversion.
- No raw inventory counts or stock-urgency signals beyond what the offer
  needs. Quote availability as "in stock" / "limited", never the integer.

Emit 0 or 1 envelope. If nothing matches, a single `commerce.respond_inquiry`
explaining no match is the correct reply.
