---
name: reputation-aware-pricing
description: A pre-step for pricing-negotiate on an offer turn. Reads the merchant's own platform reputation (world.read_reputation) and writes a pricing_bias into memory that shifts the effective target_band per reputation band — high reputation pushes the counter toward target_band[1] (established sellers defending premium pricing); low reputation collapses the band toward target_band[0] (the seller competes more aggressively). Emits no envelope of its own; pricing-negotiate consumes the bias next.
when_to_use: Use when the inbound is propose_offer / counter_offer AND no scarcity bias has already been written this turn (stockout-aware-pricing wins over reputation when both apply — scarcity is a time-bound regime, reputation is a steady-state one). Do NOT use during a non-offer turn (markdown / markup own those). Do NOT use to write reputation (that is a platform monopoly per VCP_SPEC §6).
group: trust
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_reputation
  - world.read_policy
  - memory.read
  - memory.write
context: main
---

# Reputation-Aware Pricing

## Native function guidance

Read only the merchant's public reputation and use it as one bounded input to
an active negotiation. A strong score may support firmer pricing and a weak
score may support more competitive pricing, but neither overrides the private
floor or other hard policy. Do not disclose private price rails or turn a
reputation signal into a claim about the product. This preparatory skill makes
no terminal market action by itself. Continue through an authorized pricing
skill, or finish safely.

## Detailed workflow

The first TRUST-group skill on the sell side. A pre-step for
``pricing-negotiate`` that biases the negotiation band toward one end
based on the merchant's own platform reputation — high reputation
defends premium pricing, low reputation competes harder.

The shape mirrors ``stockout-aware-pricing``: read state, write a
``pricing_bias`` to ``memory[PREFERENCE]``, emit nothing, hand off to
``pricing-negotiate`` which calls ``apply_pricing_bias`` and uses the
returned ``effective_band`` for its midpoint.

The merchant **never** writes its own reputation. ``world.update_reputation``
is a platform monopoly (VCP_SPEC.md §6.1). This skill only reads.

## Procedure

1. Read state:
   - ``rep = get_reputation()`` (snapshot of avg_rating + review_count)
   - ``band = reputation_band(rep)`` → ``"high" | "medium" | "low"``
   - ``mand = read_offer_mandate(product)`` — the band to shift
2. Gate (any false → ``no_reply``):
   - ``state.pricing_bias`` not already set this turn
   - ``band != "medium"``
3. Compute the effective band:
   ``effective_band = apply_reputation_bias(mand.target_band, band)``
   - ``"high"`` → top quarter (``[hi - span//4, hi]``)
   - ``"low"``  → bottom quarter (``[lo, lo + span//4]``)
4. Write the bias hint:
   ``memory.write(PREFERENCE, "pricing_bias",
                  {"sku_id": ..., "effective_band": [low, high],
                   "reason": "reputation:" + band})``
5. Emit ``no_reply``. ``pricing-negotiate`` is next; it calls
   ``apply_pricing_bias`` and uses the returned ``effective_band`` for
   its midpoint.

## Never disclose

- Internal reputation math, threshold cutoffs, or the actual rating /
  review count in any payload or chat. The bias only shifts the
  outbound ``unit_price``; the reasoning never goes on the wire.
- The fact that reputation drove the price. Buyers asking "why is your
  counter so high?" get the public range or a counter, not a
  reputation explanation. Defer to ``private-utility-guard``.
- The exact thresholds used to classify the band — those are policy
  knobs, not public surface.

## Bounds and ethics

- The accept / reject thresholds are unchanged — they still key off
  ``auto_accept_threshold`` and the private floor. The bias only
  shifts where the counter lands within the legal zone.
- High-reputation bias never raises ``target_band[1]`` (the mandate's
  authorized ceiling). Low-reputation bias never lowers below
  ``target_band[0]`` (the mandate's authorized floor for the
  *negotiation*, distinct from the private ``floor_price``).
- The skill grants no new pricing authority. The mandate is the
  contract; this only redistributes the where-in-the-band.

## Output

``no_reply``. ``pricing-negotiate`` emits the single offer-turn
envelope.
