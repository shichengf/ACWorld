---
name: negotiation
description: Open and conduct a multi-round price negotiation with a merchant when the buyer's discovery filter found a SKU that passes every check except budget. Use this skill whenever (a) `discovery-search` Path B's 4-stage filter leaves zero fully-eligible candidates but at least one "budget-only failure" candidate exists, or (b) an inbound envelope is `commerce.counter_offer` / `commerce.accept_offer` / `commerce.reject_offer` from a merchant with `negotiation_id` matching the buyer's `TRANSACTION.pending_negotiation`. This is the only buyer skill that emits `commerce.propose_offer` or `commerce.counter_offer`.
group: economic
---

# Buyer · Negotiation

## Native function guidance

Negotiate only for an otherwise feasible item. Keep the buyer budget and
walk-away value private. Use authoritative listing, stock, reputation, or own
balance reads when needed. Make one bounded offer, counter, acceptance,
rejection, withdrawal, or settlement decision with the supplied functions.
Never accept or settle above the mandate budget. Do not invent a merchant
response, agreement, order, or protocol field.

## What this skill is for

The buyer's 4-stage discovery filter is a take-it-or-leave-it screen.
When the only thing standing between the buyer and a perfect-fit SKU
is the price tag, walking away is wasteful — there's room to haggle.
This skill turns that wasted opportunity into a bounded bargaining
loop: propose a counter, take the merchant's reply, iterate up to N
rounds, settle if the price lands at or under the buyer's max budget.

The privacy story is the load-bearing constraint. The buyer's
`max_budget` is in `PRIVATE_UTILITY` and may NEVER appear on the wire
— not as an integer, not as a restated value, not as a derived
quantity. Counter prices are anchored to the merchant's last offer
(public information), never to the budget. The buyer's local
arithmetic uses `max_budget` only as an inequality gate.

## Settlement requires World-backed merchant consent

**You cannot unilaterally settle below the listed price.** Every
proposal, counter, acceptance, rejection, or withdrawal is sent to
`platform:negotiation`. The intended peer is carried in
`counterparty_id`. Platform validates the actor and forwards a
server-authored relay. World persists the complete offer thread and
derives the terminal agreement. `PSPPolicy.settle` accepts a negotiated
price only when `negotiation_id` resolves to an unexpired World agreement
whose buyer, merchant, SKU, quantity, currency, price, listing revision,
and deterministic order identity all match the settlement.

In plain terms: **if you want to settle at $21.60 for a $24.00 listing,
you MUST submit $21.60 through `platform:negotiation`, and the merchant
must accept that exact persisted offer.** Audit text alone is never
settlement authority. Self-discounting fails before any order, inventory,
or ledger write.

This means: any time discovery hands you an offer over budget, the
ONLY way through is the full envelope round-trip described below.
Skipping the negotiation envelopes and just settling at a
self-computed price will fail loud, not silent.

## When to load this skill

Two triggers, exhaustively:

1. **After `discovery-search` Path B's filter dead-ends with a
   negotiable candidate.** "Dead-ends" means: zero candidates survived
   Stages 1–4 + variant grouping. "Negotiable" means: at least one
   candidate exists which would have passed Stages 2–4 (must_have via
   claims or grounding, eta within `delivery_days`, merchant not on
   avoid list, in stock, not expired) but failed Stage 1's budget
   check (`unit_price * qty + cumulative_spend > max_budget`) — AND
   `floor(unit_price * 0.7) ≤ max_budget` (a plausible discount toward
   the merchant's ~70 %-of-list floor could close the gap; a single 10 %
   ask is too strict and abandons winnable deals). `discovery-search`
   loads this skill in that precise case.

2. **Server-authored relay during an active negotiation.** The envelope
   comes from `platform:negotiation`. Its `platform_mediation.submitted_by`
   names the merchant, `platform_mediation.recipient_id` names this buyer,
   and `platform_mediation.status` matches the action. The envelope's
   `negotiation_id` matches
   `TRANSACTION.pending_negotiation.negotiation_id`. Three kinds:
   `commerce.counter_offer`, `commerce.accept_offer`,
   `commerce.reject_offer`. Each routes to a branch below.

## Memory model

| Bucket            | Holds                                                                                                            |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- |
| `PREFERENCE`      | (reads from `discovery-search` writes; nothing of its own)                                                       |
| `TRANSACTION`     | `pending_negotiation`, `selected_offer` (on success), `last_discovery_outcome`                                    |
| `PRIVATE_UTILITY` | reads `max_budget`, `max_negotiation_rounds` (default 3); never writes here                                       |

The `pending_negotiation` record shape:

```yaml
pending_negotiation:
  negotiation_id:  str         # generated client-side; persists across rounds
  offer_id:        str         # from the original rank_offers candidate
  merchant_id:     str
  sku_id:          str
  qty:             int
  fulfillment:     {method, eta_days}
  anchor_price:    int         # the original merchant offer, in cents
  last_counter:    int         # buyer's most recent counter, in cents
  round:           int         # 1, 2, 3, ...
```

## Privacy invariants — read this before writing any branch

- The literal value of `PRIVATE_UTILITY.max_budget` MUST NEVER appear
  in any outbound payload OR any `tool_call.args`.
- The boolean *result* of `floor(unit_price * 0.7) ≤ max_budget` is a
  private comparison: never put it in a payload, never echo it as
  reasoning text in any field other than the LLM's internal scratch.
- Any derived value (anchored counter price, midpoint between
  counters) must be computed from **public** operands (the merchant's
  offered price, the buyer's own previous counter). `max_budget` is
  used **only** as the upper-bound check before emitting.
- Edge case: if your derived counter equals `max_budget` literally
  (e.g. `floor(2778 * 0.9) = 2500` when `max_budget = 2500`),
  **subtract 1 cent** before emitting. The Router's payload scanner
  would otherwise raise `PrivateUtilityLeak`.

## What to do

### Branch N-A — initiate (loaded after discovery dead-end)

This branch fires when no `TRANSACTION.pending_negotiation` exists and
the load reason was discovery's dead-end. Inputs available in memory:

- The list of "budget-only failure" candidates surfaced by
  `discovery-search` (stored transiently in
  `TRANSACTION.negotiable_candidates`, written by discovery before
  loading this skill).
- `PRIVATE_UTILITY.max_budget`, `PRIVATE_UTILITY.max_negotiation_rounds`
  (default 3 if absent).

Steps:

1. Among `negotiable_candidates`, pick the one with the **lowest
   `unit_price`** (cheapest to close the gap). Tie-break by
   `offer_id` lex order for deterministic replay.

2. Compute the anchor:
   ```
   counter = min(floor(offered.unit_price * 0.9), max_budget)
   if counter >= max_budget:
       counter = max_budget - 1  # never propose above budget; avoid PrivateUtilityLeak
   ```

3. Generate a `negotiation_id` (`f"neg:{mandate_id}:{offered.offer_id}"`
   keeps it deterministic and tied to a single round of bargaining
   per offer).

4. Memory writes (one `memory_update`):
   ```
   TRANSACTION.pending_negotiation = {
     negotiation_id, offer_id, merchant_id, sku_id, qty,
     fulfillment, anchor_price: offered.unit_price,
     last_counter: counter, round: 1
   }
   ```

5. Emit `commerce.propose_offer` to `platform:negotiation`:
   ```yaml
   to: platform:negotiation
   action:
     kind: commerce.propose_offer
     payload:
       offer_id:        <from offered>
       sku_id:          <from offered>
       qty:             <from offered>
       unit_price:      <counter>           # the anchored price
       fulfillment:     <from offered>
       negotiation_id:  <generated above>
       counterparty_id: <offered.merchant_id>
   ```

### Branch N-B — merchant counter-counters (`commerce.counter_offer` inbound)

The merchant came back with a price above the buyer's last counter.
Read:

- `pending = TRANSACTION.pending_negotiation`
- `merchant_counter = inbound.payload.unit_price`
- `max_budget = PRIVATE_UTILITY.max_budget`

Decide:

1. **`merchant_counter ≤ max_budget` → accept.** The merchant met or
   beat the budget. Emit `commerce.accept_offer` to
   `platform:negotiation` echoing the merchant's price (NOT
   `max_budget`; same privacy-discipline reason as the merchant's
   accept branch):
   ```yaml
   to: platform:negotiation
   action:
     kind: commerce.accept_offer
     payload:
       offer_id:        <pending.offer_id>
       sku_id:          <pending.sku_id>
       unit_price:      <merchant_counter>
       fulfillment:     <pending.fulfillment>
       negotiation_id:  <pending.negotiation_id>
       counterparty_id: <pending.merchant_id>
   ```

   Memory writes (commit the negotiated offer into the same shape
   `purchase-confirmation` reads — Branch B kicks in next when the
   merchant's `commerce.accept_offer` arrives back at the buyer, or
   when the platform issues the match certificate):
   ```
   TRANSACTION.selected_offer = {
     offer_id, merchant_id, sku_id, qty,
     unit_price: merchant_counter,
     fulfillment, claims: null, expires_at: "<+1h from now>",
   }
   TRANSACTION.pending_negotiation = null
   TRANSACTION.last_discovery_outcome = "picked"
   ```

2. **`merchant_counter > max_budget` AND `pending.round <
   max_negotiation_rounds` → counter-counter.** Compute:
   ```
   counter = floor((pending.last_counter + merchant_counter) / 2)
   if counter == max_budget:
       counter = counter - 1
   if counter > max_budget:
       # merchant is converging slower than budget allows — give up.
       go to branch N-D-equivalent reject path.
   ```
   If the counter is valid, emit `commerce.counter_offer`:
   ```yaml
   to: platform:negotiation
   action:
     kind: commerce.counter_offer
     payload:
       offer_id:        <pending.offer_id>
       sku_id:          <pending.sku_id>
       unit_price:      <counter>
       fulfillment:     <pending.fulfillment>
       negotiation_id:  <pending.negotiation_id>
       counterparty_id: <pending.merchant_id>
   ```
   Memory:
   ```
   TRANSACTION.pending_negotiation.last_counter = counter
   TRANSACTION.pending_negotiation.round       += 1
   ```

3. **Otherwise → reject.** Out of rounds, or the converged counter
   would exceed budget. Emit `commerce.reject_offer`:
   ```yaml
   to: platform:negotiation
   action:
     kind: commerce.reject_offer
     payload:
       offer_id:        <pending.offer_id>
       negotiation_id:  <pending.negotiation_id>
       sku_id:          <pending.sku_id>
       counterparty_id: <pending.merchant_id>
       reason:          "rounds_exhausted"  # or "no_convergence"
   ```
   Memory:
   ```
   TRANSACTION.pending_negotiation         = null
   TRANSACTION.last_discovery_outcome      = "negotiation_failed"
   ```

### Branch N-C — mediated merchant acceptance (`commerce.accept_offer` inbound)

The merchant accepted the buyer's most recent counter (or its first
proposal). This is the success path. **This branch is mandatory on
this inbound** — the selector activates both `negotiation` AND
`purchase-confirmation` on a merchant's `commerce.accept_offer`, and
Branch N-C MUST run before purchase-confirmation Branch B so the
agreed price is committed to memory.

Steps:

1. Verify the envelope is from `platform:negotiation`; the mediation block
   names the pending merchant as submitter and this buyer as recipient; its
   status is `accepted`; and `inbound.payload.negotiation_id ==
   pending_negotiation.negotiation_id`. If any check fails, ignore it.
2. Commit the deal into `TRANSACTION.selected_offer` in the shape
   `purchase-confirmation` Branch B expects. The merchant's accept
   envelope carries the agreed `unit_price` (per the merchant
   `pricing-negotiate` SKILL's accept-emit contract); copy it verbatim
   — this is the integer cents the World agreement and the buyer's
   reconciler both key off:
   ```
   TRANSACTION.selected_offer = {
     offer_id:    pending.offer_id,
     merchant_id: pending.merchant_id,
     sku_id:      pending.sku_id,
     qty:         pending.qty,
     unit_price:  inbound.payload.unit_price,    # the agreed price (NOT max_budget)
     fulfillment: pending.fulfillment,
     claims:      null,
     expires_at:  "<inbound.payload.expires_at or +1h>",
   }
   TRANSACTION.pending_negotiation     = null
   TRANSACTION.last_discovery_outcome  = "picked"
   ```
3. Emit `no_reply`. Settlement is `purchase-confirmation` Branch B's
   job — it reads `TRANSACTION.selected_offer` (the dict you just
   wrote) and emits `platform.settle_payment`. If you skip step 2,
   `selected_offer` is empty, and the reconciler raises
   `BudgetReconcileViolation` before the wrong settle reaches the bus
   (the anomaly: pending_negotiation exists but selected_offer doesn't,
   meaning this commit step was missed). Never emit
   `platform.settle_payment` from this skill — settlement crosses a
   trust boundary and is owned by `purchase-confirmation`.

### Branch N-D — merchant rejects (`commerce.reject_offer` inbound)

The merchant walked away (typically "below_floor" — buyer's last
counter was insulting from the merchant's side).

Steps:

1. Verify `negotiation_id` matches.
2. Memory:
   ```
   TRANSACTION.pending_negotiation     = null
   TRANSACTION.last_discovery_outcome  = "negotiation_failed"
   ```
3. Emit `no_reply`. The buyer is stuck on this mandate; a future
   recovery skill might re-search with a relaxed query.

## Worked example — single-round success

**Memory state when Branch N-A activates** (discovery's dead-end):

```
PRIVATE_UTILITY.max_budget                  = 2200   # $22.00
PRIVATE_UTILITY.max_negotiation_rounds      = 3
TRANSACTION.negotiable_candidates           = [{
  offer_id: "agg:PCC1HONU302", merchant_id: "merchant:allbirds",
  sku_id: "PCC1HONU302", qty: 1, unit_price: 2400,
  fulfillment: {method: "standard", eta_days: 4}
}]
```

**Branch N-A computation:** `floor(2400 * 0.9) = 2160`; `2160 != 2200`,
so no -1 adjustment.

**Outbound:**

```yaml
to: platform:negotiation
action:
  kind: commerce.propose_offer
  payload:
    offer_id:        agg:PCC1HONU302
    sku_id:          PCC1HONU302
    qty:             1
    unit_price:      2160
    fulfillment:     {method: standard, eta_days: 4}
    negotiation_id:  neg:m-001:agg:PCC1HONU302
    counterparty_id: merchant:allbirds
```

**Memory writes:**

```
TRANSACTION.pending_negotiation = {
  negotiation_id: "neg:m-001:agg:PCC1HONU302",
  offer_id: "agg:PCC1HONU302", merchant_id: "merchant:allbirds",
  sku_id: "PCC1HONU302", qty: 1,
  fulfillment: {method: "standard", eta_days: 4},
  anchor_price: 2400, last_counter: 2160, round: 1,
}
```

The integer `2200` (max_budget) does not appear in the outbound. The
counter `2160` is anchored to the public offered price `2400`, not to
the budget.

If the merchant's `floor_price` is 2100, they accept at 2160 (≥ floor),
echoing `unit_price: 2160` — and the buyer's purchase-confirmation
Branch B takes over for the final settle.

## Worked example — edge case: counter equals max_budget

`offered.unit_price = 2778`, `max_budget = 2500`. `floor(2778 * 0.9) =
2500`. Without the -1 guard, the buyer would emit `unit_price: 2500`,
and the Router's payload scanner would raise `PrivateUtilityLeak`
(2500 is in PRIVATE_UTILITY). The skill emits `unit_price: 2499`
instead — semantically a tiny shading toward the buyer, mechanically
the only way to stay under the literal-equality check.

## Failure modes and what to do

- **`pending_negotiation.round >= max_negotiation_rounds` and merchant
  still over budget:** emit `commerce.reject_offer` with
  `reason: "rounds_exhausted"`. Set `last_discovery_outcome =
  "negotiation_failed"`. No further action — the discovery layer
  may or may not retry; that's not this skill's concern.
- **Inbound `negotiation_id` doesn't match
  `pending_negotiation.negotiation_id`:** stale envelope. Emit
  `no_reply`, don't touch memory.
- **No `negotiable_candidates` in memory when Branch N-A would fire:**
  the discovery skill loaded `negotiation` by mistake (or the
  candidates were cleared by a stale `awaiting_user_approval` reset).
  Emit `no_reply`.

## Why this is its own skill

Negotiation is the single most-iterated piece of buyer behavior in
real deployments. Pulling it out of `discovery-search` and
`purchase-confirmation` keeps three things honest:

1. **Ablation surface.** Swap in `negotiation-walk-away` (the trivial
   skill that always emits `no_reply`) and the buyer reverts to the
   take-it-or-leave-it behavior; A/B on the value of bargaining.
2. **Privacy clarity.** All max-budget-comparison logic lives in one
   file; reviewers don't have to chase the leak surface across three
   skills.
3. **Counter strategy iteration.** Anchor-based today; future
   alternatives (bisection, learned policy, multi-merchant
   simultaneous proposals) replace one skill body, no protocol
   changes.
