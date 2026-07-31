---
name: price-discovery
description: Use at the start of a sell-side journey to turn a fuzzy "fair price" intent into a concrete, defensible target band before any offer is made.
when_to_use: Use on the first turn of a sell mandate, or when pricing_intent changes, before responding to any commerce.request_offer / propose_offer.
group: interpretation
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_catalog
  - world.read_policy
  - memory.read
  - memory.write
context: main
---

# Price Discovery

## Native function guidance

Before making an offer, use authoritative listing, stock, and public reputation
reads when they are needed to judge market position. Form a defensible target
band from public evidence and the merchant's private policy. Keep floor price,
margin targets, urgency, and internal reasoning private. This preparatory skill
does not itself change commerce state. Continue with a terminal pricing action
only when another active skill authorizes it; otherwise finish safely without
inventing market comparisons.

## Detailed workflow

The supply-side analogue of taste grounding (VIBE_COMMERCE.md §4.2): the peer
seller arrives with `pricing_intent` ("fair price", recent comparables,
depreciation) but **no number**. Convert that into a defensible target band
the negotiation skill can defend.

## Inputs

- `SellMandate.pricing_intent` (method hints, optional target_band) and
  `SellMandate.min_acceptable_price` — the latter is **PRIVATE_UTILITY**.
- `OfferMandate.pricing.list_price` / `floor_price` if this is a B2B
  `OfferMandate` rather than a C2C `SellMandate`. `floor_price` is
  **PRIVATE_UTILITY**.
- Comparable listings via `world.read_catalog`.

## Procedure

1. If the mandate already states `pricing_intent.target_band`, use it.
2. Otherwise derive `[low, high]` from comparable catalog listings and the
   stated method (recent comparables / depreciation curve). Keep both bounds
   **>= the private floor** (`min_acceptable_price` / `floor_price`).
3. Write the result to memory so later turns are stable:
   - `memory.write(PREFERENCE, "target_band", [low, high])`
   - `memory.write(PREFERENCE, "auto_accept_threshold", <from authority>)`
   - `memory.write(PRIVATE_UTILITY, "min_acceptable_price", <floor>)`
4. All prices are **integer minor units** (VCP_SPEC.md §3) — never floats.

## Output

This is a setup skill. Emit **no envelope** (`no_reply`). The band lives in
memory; `pricing-negotiate` reads it on every offer. Never echo the floor or
any private value back on the wire.
