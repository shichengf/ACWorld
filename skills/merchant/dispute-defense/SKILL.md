---
name: dispute-defense
description: Use to package the merchant's evidence brief when a buyer dispute lands. Triggers on platform.open_dispute notifications routed to the merchant; emits exactly one commerce.send_message to platform:adjudicator carrying a structured evidence object — order summary, claim record from the original offer, refund history, public policy. The adjudicator owns the ruling; this skill owns the merchant's defense surface. Never editorializes, never accuses the buyer, never reveals the floor.
when_to_use: Use when the inbound envelope is platform.open_dispute against an order the merchant fulfilled. Do NOT use to handle a return adjudication (return-adjudicate owns that, before a dispute is opened). Do NOT use to handle the adjudicator's eventual platform.rule (that is the runtime's; the merchant is bound by the ruling).
group: operations
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_order
  - world.read_policy
  - world.read_dispute
  - memory.read
context: main
---

# Dispute Defense

## Native function guidance

Read the merchant's authoritative order and available evidence before
responding to a dispute. Submit only relevant facts that can be tied to the
transaction, listing, fulfillment, or published policy. Do not fabricate
records, accuse the buyer, reveal floor price or internal strategy, or decide
the ruling yourself. Use one supplied dispute, evidence, response, or mediated
message action that advances the case. If no grounded response is authorized,
finish safely.

## Detailed workflow

The merchant's contested-path skill. A buyer opened a dispute; the
adjudicator wants a factual record of what happened. This skill builds
the merchant's evidence brief — order shipped, claims made, refund
history, public policy — and emits exactly one message back to the
platform.

The adjudicator owns the *ruling*. This skill owns the *defense surface*.
Once `platform.rule` lands, the merchant has no further say: refund
issuance / reputation update / restock are governed runtime writes.

## Procedure

1. Read state:
   - `order = get_order(order_id)` — must exist; carries product, qty,
     amount, status, delivered_at.
   - `pol   = world.read_policy()` — refund_policy, return_window_days.
   - `memory.read(TRANSACTION, f"return:{order_id}")` — prior refund
     adjudication record (if any). The decision the merchant made when
     the return was first requested matters here; absence of a record
     means no return was filed before the dispute.
   - `memory.read(TRANSACTION, order_id)` — the receipt persisted at
     dispatch (if any), with the `permitted_claims` the merchant
     attached to the original `GroundedOffer`.

2. Build the evidence brief via:
   `build_dispute_evidence(order_id, dispute_id,
                           buyer_claim=payload.buyer_claim,
                           refund_history=<from memory>,
                           claim_record=<from memory>)`
   The helper renders the exact payload shape the adjudicator parses.
   It contains only factual, public surface — order summary, policy,
   claim record, refund history.

3. Emit exactly one envelope:
   `commerce.send_message` to `platform:adjudicator` carrying the
   structured `evidence` payload.

## Never disclose

- The order's per-unit cost basis, margin, or any private negotiating
  utility. Evidence is **what happened**, not *what the merchant
  hoped to make*. `private-utility-guard` runs on every emit.
- Buyer accusations or character claims. The brief is factual: the
  product was shipped, the claims were these, the refund history was
  this. The adjudicator decides everything else.
- Internal reasoning about how the merchant *would have ruled* the
  dispute. The adjudicator owns the ruling.

## Bounds and ethics

- One envelope per turn. A multi-turn dispute (request_evidence →
  send → request_more) is multiple turns, each producing one envelope.
- Never include private-utility math (floor, margin) even as a
  percentage or unit-converted equivalent.
- Never editorialize a return adjudication that the buyer disputed —
  the original `return:{order_id}` memory record is the truth; this
  skill records it faithfully, regardless of whether the adjudicator
  is likely to side with the merchant.
- Truthful claims only. The `claim_record` field carries the original
  `permitted_claims` the merchant attached to the offer; if memory has
  no record, the field is an empty list, never a fabricated one.

## Output

One `commerce.send_message` envelope to `platform:adjudicator` whose
`payload.evidence` is the structured brief. `no_reply` only if the
dispute references an order the merchant did not fulfill (an internal
error, surfaced via the boundary).
