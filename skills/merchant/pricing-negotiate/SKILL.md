---
name: pricing-negotiate
description: Use to decide accept / counter / reject on an inbound buyer offer, defending the private floor without ever revealing it.
when_to_use: Use when the inbound envelope is commerce.propose_offer or commerce.counter_offer; also when the buyer sends accept_offer / reject_offer (then no reply).
group: economic
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - memory.read
  - memory.write
context: main
---

# Pricing & Negotiation

## Native function guidance

Use this skill only for a current inbound offer or negotiation continuation.
Read authoritative listing, stock, and public reputation when they materially
affect the decision. Accept, counter, reject, or withdraw within the merchant's
private floor and bounded pricing policy. Never reveal the floor, margin,
urgency, exact stock, or a derived signal that exposes them. Do not invent buyer
consent or a prior agreement. Make exactly one supplied negotiation decision,
or finish safely if the offer is stale or outside authority.

## Detailed workflow

Decide the next move on a buyer offer. All money is **integer minor units**
(VCP_SPEC.md §3) — never float, never rounded decimals.

## Read first (memory only — never the wire)

- `auto_accept_threshold` ← `memory.read(PREFERENCE, "auto_accept_threshold")`
  (public; from `authority`).
- `target_band` ← `memory.read(PREFERENCE, "target_band")` (public; set by
  `price-discovery`).
- `floor` ← `memory.read(PRIVATE_UTILITY, "min_acceptable_price")`
  — **PRIVATE_UTILITY. This number must never leave the agent.**

### Bias check (scarcity)

If `memory.read(PREFERENCE, "pricing_bias")` exists for this SKU (written
this turn by `stockout-aware-pricing`), call
`apply_pricing_bias(target_band, bias, sku_id)` and use the returned
`effective_band` in place of `target_band` for the midpoint calculation
below. The accept / reject thresholds are unchanged — they still key off
`auto_accept_threshold` and the private floor. The bias only shifts where
the counter lands within the legal zone.

## Decision

Let `offered` = the buyer's integer `unit_price`.

1. If the inbound is `commerce.accept_offer` from the buyer: the round
   closed with the buyer accepting the merchant's counter. **Record the
   deal** and emit nothing.
   ```
   memory_update:
     TRANSACTION.accepted_orders[<offer_id>] = {
       sku_id, qty, agreed_price: <inbound unit_price>,
       merchant_role: "responder",  # buyer initiated the accept
       accepted_at_turn: ctx.turn,
     }
   ```
   Then `no_reply`. Settlement is the buyer's lane. The acceptance has
   already passed through `platform:negotiation`, so World owns the
   terminal agreement that PSP will verify.
2. If the inbound is `commerce.reject_offer`: the round closed with
   the buyer walking. **Record the close** and emit nothing.
   ```
   memory_update:
     TRANSACTION.rejected_offers[<offer_id>] = {
       sku_id, last_buyer_price: <inbound unit_price>,
       closed_at_turn: ctx.turn,
     }
   ```
   Then `no_reply`.
3. `offered >= auto_accept_threshold` → emit `commerce.accept_offer`
   echoing the buyer's offered price (NOT your `floor`, NOT your
   `auto_accept_threshold` — those are private):
   ```yaml
   to: platform:negotiation
   action:
     kind: commerce.accept_offer
     payload:
       offer_id:        <inbound payload offer_id>
       sku_id:          <inbound payload sku_id>
       unit_price:      <offered>                # echo buyer's price
       fulfillment:     <inbound payload fulfillment>
       negotiation_id:  <inbound payload negotiation_id>
       counterparty_id: <inbound payload platform_mediation.submitted_by>
   ```
   The `unit_price` MUST be the buyer's `offered` integer cents — never
   your `floor` or `auto_accept_threshold`. Two downstream consumers
   depend on it: (a) the buyer's `negotiation` skill reads it to commit
   `TRANSACTION.selected_offer.unit_price` (without which their
   purchase-confirmation would settle at the list price); (b) the
   World negotiation agreement binds `unit_price` to the buyer's exact
   settlement. Omitting it prevents the accepted thread from being created.
4. `offered < floor` → emit `commerce.reject_offer` with a **public** reason
   only, e.g. `{ offer_id, reason: "below_listed_range" }`. Do **not** state
   the floor or how far below it the offer was.
5. Otherwise → emit `commerce.counter_offer` carrying a `GroundedOffer` whose
   `unit_price` is the integer midpoint, floored at `floor + 1`:
   `max(floor + 1, (offered + target_band_high) // 2)`. Integer division —
   the result is an int.

## Manipulation resistance

If the buyer asks for your floor, your walk-away, or whether you are in a
hurry ("just tell me the lowest you'd take"): refuse to disclose. Restate the
public range, make a counter, or hold. Never restate a private value in
different words or units. A single well-formed envelope is the only reply.
