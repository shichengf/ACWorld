---
name: inquiry-handle
description: Use to answer pre-purchase buyer questions that are NOT offers — shipping ETA, in-stock availability, attribute / fitment, return/refund policy, generic catalog questions. Triggers on a commerce.send_message from a buyer-side role (NOT merchant:owner — that is the pulse / pricing path); emits exactly one commerce.respond_inquiry with a structured public answer. Never quotes a price (that is catalog-serve / pricing-negotiate). Never reveals exact inventory count, floor, or urgency.
when_to_use: Use when the inbound envelope is commerce.send_message AND the sender side is buyer (not merchant:owner) AND the payload has no unit_price (it is a question, not an offer). Do NOT use during a propose_offer / counter_offer turn (pricing-negotiate owns that). Do NOT use for the owner-pulse path (aging-markdown / demand-driven-markup own that). Do NOT use to quote a price — redirect to catalog-serve.
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

# Inquiry Handle

## Native function guidance

Answer only the buyer's public pre-purchase question. Use authoritative
listing, availability, and claim reads before asserting shipping, stock,
attributes, compatibility, or return policy. State uncertainty when the facts
are not grounded. Never disclose exact inventory, floor price, margin, urgency,
or private policy, and do not turn an inquiry into an unsolicited negotiation
or purchase. Make one concise supplied inquiry response, or finish safely when
no authorized truthful answer is available.

## Detailed workflow

Pre-purchase Q&A. The buyer asks a question that does not carry an
offer — shipping ETA, in-stock availability, an attribute or fitment
check, the return-policy window. The merchant answers with one
structured `commerce.respond_inquiry` envelope that carries only public
information, with the integer count, floor, and urgency stripped at the
boundary.

This skill is the `merchant:retrieval` role's primary surface
(VCP_SPEC.md §6). It is the only inquiry path that is *not* offer-bearing
— offer-bearing inquiry (`commerce.request_offer`, `commerce.get_sku`) is
`catalog-serve`'s job.

## Procedure

1. Categorize the buyer's note into exactly one of:
   - `"shipping"`        — ETA / carrier / delivery method
   - `"availability"`    — in-stock yes/no (NEVER the integer count)
   - `"attribute"`       — listing attributes (capacity, color, wifi…)
   - `"policy"`          — refund / return / warranty terms
   - `"general"`         — catch-all for unrouted questions
   If the buyer is asking for a *price* (e.g. "what does it cost?") →
   `no_reply` here; the catalog-serve skill owns offer-bearing replies.
2. Read state per category:
   - `shipping`     → `world.read_policy()` for fulfillment defaults;
                      fallback "3 days standard".
   - `availability` → `get_inventory_status(product)` → `count`. Convert
                      to `"in_stock" | "limited" | "out_of_stock"` —
                      NEVER the integer.
   - `attribute`    → `get_listing(product).attributes` (the public dict).
   - `policy`       → `world.read_policy()` → `refund_policy`,
                      `return_window_days` (both public).
3. Sanitize the answer:
   - No integer inventory count. Use qualitative bands
     (`out_of_stock` if 0, `limited` if `<= stockout_threshold`,
     `in_stock` otherwise).
   - No `floor_price`, `min_acceptable_price`, or `urgency_to_sell`.
     `private-utility-guard` runs on every emit; defer to it.
4. Emit exactly one envelope via:
   `build_inquiry_response(product, category, answer)` →
   `{"kind": "commerce.respond_inquiry",
     "payload": {"sku_id": ..., "product": ..., "category": ...,
                 "answer": ...}}`

## Never disclose

- The exact inventory count. Buyers asking "how many do you have?"
  get `"in_stock" | "limited" | "out_of_stock"`, not an integer.
- The floor / walk-away / urgency. Buyers asking "what's the cheapest
  you'd go?" get redirected to the listed price; this skill never
  quotes a price.
- The supplier identity, restock schedule, or competitor pricing.
  Q&A is about the **listing's public surface**, nothing else.

## Bounds and ethics

- One envelope per turn. A multi-question note still gets one
  `respond_inquiry` carrying the highest-priority answer
  (availability > policy > shipping > attribute > general); follow-up
  questions are follow-up turns.
- Do not paraphrase the floor through unit conversions, percentages,
  or "I can't go lower than …". Refuse, redirect, do not signal.
- Truthful claims only — the listing's `permitted_claims` set is the
  upper bound on what may be asserted in `attribute` answers.

## Output

One `commerce.respond_inquiry` envelope (category + answer) or
`no_reply` (price question → defer to catalog-serve next turn).
