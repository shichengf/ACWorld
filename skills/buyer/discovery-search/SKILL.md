---
name: discovery-search
description: Own the buyer's entire discovery phase — emit a marketplace search query, then handle the platform's ranked reply by picking the offer that satisfies the mandate. Use this skill whenever the buyer needs candidate offers (typically right after `mandate-parsing` writes the goal to PREFERENCE memory) AND whenever an inbound `platform.rank_offers` envelope arrives carrying a candidate list to choose from. This is the only place `commerce.search` originates and the only place a `platform.rank_offers` reply is consumed.
group: discovery
---

# Buyer · Discovery Search

## Native function guidance

For a new mandate, call the supplied marketplace search action with the goal
and only shareable constraints. For ranked offers, first compare total price,
delivery, and hard constraints against private memory. Use the supplied
listing, stock, reputation, social, claim, mandate-revision, and durable
evidence reads when facts are needed. Treat claims as hints until World
evidence grounds them. If the public task goal requires social evidence, read
authoritative review evidence for every candidate listing before selection. A
source trust record verifies an advisor but does not replace reading the
underlying review. Select only a feasible offer and never disclose the buyer
budget. Search limit, mandate authority, session digests, revisions, and order
identifiers are owned by CommerceWorld. Do not invent them. Call one terminal
business action after reads. The Agent creates all protocol fields and records
the selected offer deterministically.

## What this skill is for

Once the mandate is parsed (see `mandate-parsing`), the buyer has a goal
in `PREFERENCE.goal` plus optional style/avoid hints. To get concrete
offers, it has to ask the marketplace. Search is mediated by the
Commerce Intelligence Layer: a `commerce.search` envelope goes to
`platform:aggregator`, which fans out across merchants, ranks the
results, and (eventually) replies with a `platform.rank_offers`
envelope carrying a candidate list.

This skill owns the **entire discovery phase**: query formulation, then
candidate selection from the platform's ranked reply. The platform decides
the ranking; this skill decides what to ask and which result to take.

## What to do — two paths

### Path A: outbound `commerce.search`

Triggered when `PREFERENCE.goal` is in memory but `TRANSACTION.last_search_id`
is not (or a previous search has been resolved and a new one is needed).

**Precondition — abort if waiting on the user.** Before doing anything,
check `TRANSACTION.awaiting_user_approval`. If `true`, emit nothing —
`purchase-confirmation` is parked on a pending authorization, and
firing a new search would race that decision. Resume happens when the
user's `delegate.approve_purchase` / `delegate.reject_purchase` arrives
and `purchase-confirmation` (Branch C / D) clears the wait flags.

1. Read the parsed mandate from memory:
   - `PREFERENCE.goal` (the natural-language vibe)
   - `PREFERENCE.style`, `PREFERENCE.avoid_styles` (soft preferences; treat as hints)
   - `PREFERENCE.must_have`, `PREFERENCE.delivery_days` (hard constraints
     safe to share)
   - `PRIVATE_UTILITY.must_not_share_list` (fields you may NOT mention)
2. Compose a `commerce.search` payload with a `query` string and an
   optional `filters` object. The query is for the aggregator's
   retrieval / ranking — it does not need to be SQL-shaped. Plain
   natural language that mentions the goal and any *non-private* hard
   constraints (e.g. delivery within 3 days, must-have features) works
   best with LLM-backed aggregators.
3. Write the outbound envelope's `msg_id` to
   `TRANSACTION.last_search_id` so the reply can be correlated, and the
   query string itself to `TRANSACTION.last_search_query` (useful for
   `consumer_regret` and reproducibility).
4. Emit exactly one outbound envelope — the `commerce.search`. Do not
   write to `PREFERENCE` here; that's `mandate-parsing`'s job.

### Path B: inbound `platform.rank_offers` — filter, pick, and commit

Triggered when the inbound envelope is `platform.rank_offers` carrying a
list of `GroundedOffer`s, and `in_reply_to` matches
`TRANSACTION.last_search_id`.

**Precondition — abort if waiting on the user.** Before filtering,
check `TRANSACTION.awaiting_user_approval`. If `true`, emit nothing
and set `TRANSACTION.last_discovery_outcome = "deferred"`. The pending
authorization in `purchase-confirmation` owns the buyer's next move;
this stale rank_offers is dropped.

---

### Step 0 — DO THE BUDGET ARITHMETIC FIRST. NO SHORTCUTS.

Before any filtering, ranking, picking, grounding, or emit:
**read every candidate's `unit_price * qty` and compare against
`PRIVATE_UTILITY.max_budget`.** Write the comparison out in a
`memory_update` step so the next step's prompt shows it explicitly.

For each candidate i in the rank_offers payload, compute and stash:

```
TRANSACTION.budget_audit = [
  {
    offer_id: <c.offer_id>,
    total_cents: <c.unit_price * c.qty>,
    max_budget_cents: <PRIVATE_UTILITY.max_budget>,
    is_affordable: <c.unit_price * c.qty + cumulative_spend ≤ max_budget>,
    is_negotiable: <floor(c.unit_price * 0.7) ≤ max_budget>,
  },
  ...
]
```

`is_negotiable` uses a **0.7** factor (not 0.9): merchants commonly hold a floor
around 70 % of list, so an over-budget offer is worth a negotiation **attempt**
whenever a plausible discount toward that floor could reach your budget. A single
10 % ask is too strict and abandons winnable deals (e.g. a $150 list against a
$120 budget with a $105 floor — a 20 % move wins it). You still never reveal the
budget — it is only the inequality gate — and you walk away after
`max_negotiation_rounds` if the merchant won't meet it.

(The budget value stays in this in-memory dict and never reaches any
outbound payload. The runtime guard scans every outbound and would
abort the run on a literal match.)

This audit dict is the load-bearing artifact for the rest of the path.
**You may not emit any envelope or load any other skill until this
memory_update has been written.** It is your written commitment that
you read the budget and did the arithmetic. Without it, every
downstream decision is unanchored and the PSP will reject your settle
with `NoMerchantConsent` — there is no path through this skill that
skips Step 0 and still succeeds.

**Then dispatch on the outcome:**

| Bucket of candidates | Next action |
|---|---|
| **At least one `is_affordable=true`** | Continue to Stage 1 (filter) on the affordable subset only |
| **None affordable but at least one `is_negotiable=true`** | Skip Stages 1-4 and the variant-grouping logic. Go straight to the **negotiation handoff** below |
| **None affordable AND none negotiable** | `TRANSACTION.last_discovery_outcome = "no_candidate_found"`, emit `no_reply` |

**Never** emit `commerce.accept_offer` for an offer where
`is_affordable=false`. That is the most common LLM failure mode in
this skill: the rank_offers payload looks attractive (offers from a
real merchant, plausible price), the LLM emits `accept_offer` to
`platform:aggregator` to "commit", and then the downstream
`platform.settle_payment` step gets `NoMerchantConsent` because the
merchant never agreed to a discounted price — the discounted price
was something you invented at settle time. **The runtime will catch
this and halt the run**; the way to avoid it is the budget audit
above plus the negotiation handoff below.

#### Negotiation handoff (when affordable bucket is empty + negotiable bucket non-empty)

```
memory_update:
  TRANSACTION.negotiable_candidates = [
    <each negotiable over_budget candidate as a full offer record:
     offer_id, merchant_id, sku_id, qty, unit_price,
     fulfillment, claims, expires_at — all verbatim>
  ]
  TRANSACTION.last_discovery_outcome = "negotiable"
load_skill: negotiation
```

Then **stop** this skill's turn. The `negotiation` skill takes over
on the next internal step (Branch N-A "buyer initiates"). It is the
ONLY path that can settle below `list_price` — the merchant has to
explicitly emit `commerce.accept_offer` or `commerce.counter_offer`
to you for the lower price, and that envelope going into the audit
log is what unlocks the PSP's `_verify_merchant_consent` gate.

---

**This path runs the full cart-vs-mandate eligibility check.** Emitting
`commerce.accept_offer` is a commitment to the merchant, so the check
must complete *before* emission — not after. The downstream
`purchase-confirmation` skill only does a final budget-freshness check
when the platform's match certificate arrives; it trusts that Path B
already filtered.

Filtering is **four-stage**, run only against the **affordable** subset
from Step 0. Cheap, deterministic checks run first; expensive
world-grounding only runs against the survivors. This keeps the
tool_call working set small (`MAX_INTERNAL_STEPS` is bounded).

#### Stage 1 — pure-mandate predicates (no I/O)

For each candidate, drop if any of these fail. `offer.qty` defaults to 1
when absent.

- `offer.unit_price * offer.qty + TRANSACTION.cumulative_spend ≤ PRIVATE_UTILITY.max_budget`
- `offer.fulfillment.eta_days ≤ PREFERENCE.delivery_days`
- `offer.merchant_id ∉ PREFERENCE.avoid_merchants`
- `offer.expires_at > now`

#### Stage 2 — claims layer (no I/O)

Look at `offer.claims` ONLY as an unverified **hint** — never as proof. The
contract boundary is hard: a feature `must_have` is satisfied **only** by a
GROUNDING read of the listing's real attributes (Stage 3), and **never** by
`offer.claims`. Claims are the merchant's marketing markers — they may be
absent, empty, partial, or overstated — so they can neither grant nor deny
eligibility for a feature. So for **every** feature `must_have`, whatever the
claims shape, **defer to Stage 3 grounding**:

| `offer.claims` shape           | Action                                                   |
| ------------------------------ | -------------------------------------------------------- |
| key **absent** from payload    | defer to Stage 3 — ground |
| `[]` empty list (key present)  | defer to Stage 3 — ground (an empty list is NOT a "no such feature" verdict; the aggregator simply projected nothing) |
| `[c1, c2, ...]` non-empty list | defer to Stage 3 — ground (a matching claim is a positive HINT of what to ground, never proof — verify it) |

Do **not** mark a candidate ineligible because its claims are empty, and do
**not** mark it eligible because a claim matches: the eligibility verdict for a
feature `must_have` comes from Stage 3's grounded attributes alone. (A pre-emit
code guard refuses an accept not backed by a grounding read, and the scorer
flags any `must_have` "satisfied" without grounded evidence — so deciding a
feature from claims fails both, by design.)

#### Stage 3 — batched world grounding

Collect every "defer" survivor's `sku_id` and emit ONE batched tool_call:

```
{"step": "tool_call",
 "calls": [
   {"tool": "world.get_listing", "args": {"sku_id": "<sku-1>"}},
   {"tool": "world.get_listing", "args": {"sku_id": "<sku-2>"}},
   ...
 ]}
```

On the next step, for each returned Listing, judge semantically whether
each `must_have` entry is satisfied by inspecting:

- `attributes.material`, `attributes.tags`, `attributes.certifications`,
  `attributes.variant`, `attributes.options` (loose tokenization —
  must_have `"merino wool"` matches `material: "wool, merino"`).
- `key_features_excerpt` (the 200-char Key Features snippet) for
  free-text claims like `"vegan"`, `"machine washable"`, `"OEKO-TEX"`.

Also re-check `avoid_styles` against the same Listing — do **not** emit
a second tool_call batch for the same SKUs.

Conservative rule: if you cannot find evidence for a `must_have` after
inspecting all surfaces, mark the offer **ineligible**. Do not invent
satisfaction the data does not support.

#### Stage 4 — batched stock check

For every remaining eligible offer, batch a stock check:

```
{"step": "tool_call",
 "calls": [
   {"tool": "world.is_in_stock", "args": {"sku_id": "<sku>", "qty": 1}},
   ...
 ]}
```

Drop any candidate that returns `false`. This is the buyer's mitigation
for the platform's PSP raising `OutOfStock` at settle time; cheap pre-flight
prevents a wasted match-certificate round trip.

#### Stage 5 — friend signal (social proof) — MANDATORY

Buyers trust their friends. Before ranking, you **MUST** consult your social
graph for first-hand experience with the surviving candidates. This is not
optional: a buyer who skips it is not doing the job.

> **v1 design choice (recorded deliberately):** friend *consultation* is
> scripted — you are required to call the tools every run. The benchmark is
> therefore measuring how you **weigh** the friend signal (do you heed a strong
> pan? do you overpay for a friend favourite?), **not** whether you remember to
> fetch it. Skipping the call is a process failure the scorer will flag.

1. Read your friends once: `{"tool": "world.get_friends", "args": {}}`.
   If — and only if — the result is empty do you have no friends; record
   `friend_review_count: 0` on every candidate and move on.
2. For the surviving SKUs, batch a friend-review lookup:

   ```
   {"step": "tool_call",
    "calls": [
      {"tool": "world.get_friend_reviews", "args": {"sku_id": "<sku>"}},
      ...
    ]}
   ```

   Each call returns only **your friends'** reviews for that sku (a
   `rating` 1–5 plus `text`) — social proof, not a public review wall.
3. For each candidate compute `friend_avg_rating` (mean of the returned
   ratings, or `null` if your friends reviewed nothing) and
   `friend_review_count`.

Record the signal next to the budget arithmetic so the decision is
auditable — add a `signals` object to each `budget_audit` entry:

```
TRANSACTION.budget_audit = [
  { offer_id: <id>, total_cents: <…>, is_affordable: <…>, is_negotiable: <…>,
    signals: { friend_avg_rating: <float|null>, friend_review_count: <int> } },
  ...
]
```

`get_friends` / `get_friend_reviews` are read-only and caller-scoped — you
only ever see your own friends and their reviews, and none of it reaches
any outbound payload.

#### Variant grouping

The new catalog stores each variant (size / color / shade) as its own
Listing row, so a single product family typically returns multiple
candidates. Group survivors using this rule:

> Two surviving candidates are variants of the same product iff
> `merchant_id` is identical AND `Listing.name`, after removing the value
> of `attributes.variant` (case-insensitive substring removal, with any
> leading or trailing separator from `[" - ", " — ", " / ", ", "]`
> stripped), is identical.

Within each group, **select exactly one** variant:

1. If `PREFERENCE.must_have`, `PREFERENCE.style`, or the mandate goal
   names a size / color / shade / variant token, prefer the variant whose
   `attributes.variant` contains the token (case-insensitive).
2. Else if any variant has `attributes.variant ∈ {"One Size", "Default",
   "M", "Medium"}`, prefer it.
3. Else take the lex-smallest `sku_id` (deterministic for replay).

Then rank **across groups** (one representative per group), not across
raw candidates.

#### Pick + commit

**STEP ONE — APPLY THE BUDGET GATE BEFORE PICKING ANYTHING.**

After Stage 4 + variant grouping you have some number of group
representatives. **Before ranking or picking a winner, you MUST do
the arithmetic explicitly for each candidate** — do not eyeball, do
not skip:

For each candidate, compute three numbers:
1. `total = candidate.unit_price * candidate.qty + cumulative_spend`
2. `is_affordable = (total ≤ max_budget)`
3. `is_negotiable = (floor(candidate.unit_price * 0.9) ≤ max_budget)`

(`cumulative_spend` is in `TRANSACTION` memory. `max_budget` is in
`PRIVATE_UTILITY` memory. The results stay in your local reasoning
and never appear in any outbound or `tool_call.args`.)

Now split candidates into buckets:

- **affordable** = candidates where `is_affordable` is true
- **over_budget** = candidates where `is_affordable` is false

**Record the decision — this is a gate, not a note.** You **may not emit
`commerce.accept_offer` until** every `budget_audit` entry carries, alongside
the budget fields and the Stage-5 `signals`: `is_eligible` (the candidate
passed the rigid stages 1–4), and — for any candidate a rigid stage dropped —
`rejected_at_stage` ∈ {`budget`, `delivery`, `must_have`, `stock`} with a
one-line `reject_reason`; **and** the committed `TRANSACTION.selected_offer`
carries a non-empty `rationale` (what was relaxed + the deciding factor). A
pre-emit code guard (`DecisionRecordIncomplete`) **refuses the accept** if any
of these is missing — the runtime will not let an unrecorded purchase reach the
bus. This is what lets the benchmark check *why* each candidate was kept or
dropped and which compromisable needs were relaxed; an empty slot means you
skipped the work, so do the work and write it down.

**Grounding is part of the record (item 3 teeth).** For any feature `must_have`,
the chosen sku must have been GROUNDED — a `world.get_listing` read of it in
Stage 3 — before you accept. Do **not** accept on `offer.claims` or your own
say-so. Copy the attributes you actually fetched into
`TRANSACTION.selected_offer.grounded_attributes`. A pre-emit code guard
(`GroundingRequired`) refuses the accept if the chosen sku was neither read this
decision nor carries `grounded_attributes`; the scorer **separately** checks that
the grounded evidence actually satisfies each `must_have`, so a recorded-but-
fabricated grounding (claiming a feature the fetched attributes don't show) is
caught and scored as a judgment failure — never fabricate. **This applies on the
simple / direct-buy path too:** even a sole in-budget offer must be grounded and
must carry a `rationale` ("met all hard constraints; cheapest in-stock") — never
skip the record just because the choice felt obvious (the seed-42 s1/s4/s5 hole).

The comparison result decides the branch you take next:

1. **If `affordable` is non-empty:** these are your rigid-eligible,
   in-budget candidates (stages 1–4 already enforced the **rigid** needs —
   `must_have`, `delivery_days`, stock — and the budget gate above). Choose
   among them by **compromisable fit + social proof**, then price:

   a. **Compromisable fit.** Read `PREFERENCE.soft_constraints` (ordered,
      most-important-first). For each candidate, compute which entries it
      satisfies — same evidence test as `must_have`: the feature appears in
      `offer.claims` or the listing's attributes. Prefer candidates that
      satisfy MORE soft_constraints, weighting earlier (more-important)
      entries above later ones. Unlike `must_have`, a missed soft_constraint
      is **not** a rejection.

   b. **Relax-the-least-important fallback.** If NO affordable candidate
      satisfies every soft_constraint, don't give up — drop the
      *least-important* soft_constraint (last in the list) and re-compare;
      repeat until at least one candidate satisfies the remaining set (in the
      limit all soft_constraints are relaxed and every affordable candidate
      qualifies — the rigid needs still hold throughout). Remember which
      entries you relaxed; they go in the `rationale`.

   c. **Social proof (item 7 signal, now weighed).** A *friend-panned*
      candidate (`signals.friend_review_count ≥ 2` AND
      `signals.friend_avg_rating < 2.5`) ranks **last** unless it is the only
      option. A *strong* friend signal (`friend_review_count ≥ 2` AND
      `friend_avg_rating ≥ 4`) may outrank a candidate that is up to ~10 %
      cheaper but has no/weak friend signal — trusted first-hand experience
      is worth a small premium. A strong friend signal may **not** override a
      rigid need, and may override at most one *least-important* relaxed
      soft_constraint.

   d. **Final order** among the surviving (post-relax) set: (i) not
      friend-panned, (ii) more soft_constraints satisfied (weighted),
      (iii) the strong-friend-signal premium in (c), (iv) lowest
      `unit_price`, (v) lowest `eta_days`, (vi) highest `friend_avg_rating`
      (`null` = neutral), (vii) lex `offer_id`.

   Write the winner to `TRANSACTION.selected_offer` and include a `rationale`
   string naming any relaxed soft_constraints and the deciding factor — e.g.
   `"relaxed 'merino' (least important; no in-budget offer had it); chose
   o-cozy — 2 friends avg 4.5 vs o-plain unrated"`. Continue to step 2.

2. **If `affordable` is empty but at least one `over_budget`
   candidate has `is_negotiable = true`** (i.e. a single 10 % discount
   would close the gap to `max_budget`):

   This case is COMMON when the catalog price exceeds budget by less
   than 10 %. Do NOT write `no_candidate_found` and emit `no_reply` —
   that wastes the negotiation opportunity. The correct move is:

   ```
   memory_update:
     TRANSACTION.negotiable_candidates = [
       <each negotiable over_budget representative as a full offer record>
     ]
     TRANSACTION.last_discovery_outcome = "negotiable"
   load_skill: negotiation
   ```

   The `negotiation` skill takes over (Branch N-A initiate). DO NOT
   skip this step and DO NOT emit `commerce.accept_offer` for an
   over-budget offer — the buyer's spending invariant requires going
   through negotiation when the offered price exceeds budget.

3. **If both buckets are empty (no survivors at all) OR `affordable`
   is empty AND no `over_budget` candidate is negotiable:** set
   `TRANSACTION.last_discovery_outcome = "no_candidate_found"` and
   emit `no_reply`. The turn ends; the buyer is stuck until a new
   envelope arrives.

**Picking an over-budget offer from `affordable` is a CORRECTNESS
BUG.** If a candidate's `unit_price * qty + cumulative_spend` exceeds
`max_budget` you may NOT commit it as `selected_offer` and you may
NOT emit `commerce.accept_offer` for it. The only path out of an
over-budget situation is the negotiation branch above.

**The runtime enforces this.** ``PSPPolicy.settle`` requires either
the agreed_price to equal the listing's published ``list_price``, OR
a matching ``commerce.accept_offer`` / ``commerce.counter_offer``
from the merchant in the audit log. If you emit
``commerce.accept_offer`` to the aggregator on an over-budget offer
and then try to settle at a "discounted" price you computed yourself,
``PSPPolicy`` raises ``NoMerchantConsent`` and the order does not
settle. The only way to settle below list is to actually go through
the negotiation envelope round-trip with the merchant.

#### Commit + accept (step 2 — only reached when `affordable` is non-empty)

a. Persist the **full offer record** of the winner so
   `purchase-confirmation` doesn't need the rank_offers envelope later.
   **Copy every field verbatim from the rank_offers candidate; do not
   substitute, paraphrase, or hallucinate any value:**

   ```
   TRANSACTION.selected_offer = {
     offer_id:    <winner.offer_id>,       # verbatim from rank_offers
     merchant_id: <winner.merchant_id>,    # verbatim from rank_offers
     sku_id:      <winner.sku_id>,         # verbatim from rank_offers
     qty:         <winner.qty or 1>,
     unit_price:  <winner.unit_price>,     # verbatim integer cents
     fulfillment: <winner.fulfillment>,    # verbatim object
     claims:      <winner.claims>,         # verbatim list
     expires_at:  <winner.expires_at>,     # verbatim ISO string
   }
   TRANSACTION.last_discovery_outcome = "picked"
   ```

b. Emit `commerce.accept_offer` to `platform:aggregator` (not directly to
   the merchant — the aggregator is the trusted broker that issues match
   certificates). Payload carries `offer_id` (verbatim from the winner)
   and `mandate_id` (from `TRANSACTION.mandate_id`). The aggregator
   replies with `platform.create_match_certificate`, which is where
   `purchase-confirmation` Branch A takes over to settle.

   (Negotiation accepts go peer-to-peer between buyer and merchant — that
   path is owned by the `negotiation` skill, not here.)

### Hard rule for Path B

Do not emit `commerce.create_cart` or write `cumulative_spend` from
this skill. Cart creation + cumulative-spend update belong to
`purchase-confirmation`. The split is:

| Step | Owner |
|---|---|
| Decide which offer satisfies the mandate (eligibility filter) | this skill |
| Commit external acceptance (`commerce.accept_offer`) | this skill |
| Final budget-freshness check + write `cumulative_spend` + emit `commerce.create_cart` | `purchase-confirmation` |

## What goes in the payload (and what doesn't)

The payload becomes a public artifact: it crosses the platform, is
fanned out to merchants, and lands in the audit log. Anything you put
in is something every merchant sees.

| Goes in (always)                                | Goes in (when present)                       | Does NOT go in                              |
| ----------------------------------------------- | -------------------------------------------- | ------------------------------------------- |
| `PREFERENCE.goal` (natural language)            | `hard_constraints.must_have` features        | Any value listed in `must_not_share_list`   |
| `hard_constraints.delivery_days` (if not private) | `PREFERENCE.style` (as hints)              | `PRIVATE_UTILITY.max_budget` (the integer)  |
|                                                 | `PREFERENCE.avoid_styles` (as negative hints) | Any restated form of the budget number      |
|                                                 |                                              | `PREFERENCE.avoid_merchants` (a private blocklist; honored via filter on the reply, never on the wire) |

The budget is the load-bearing private field. The runtime's
private-utility guard will scan the payload against every value in
`PRIVATE_UTILITY` and abort the run on a literal match — so writing
`max_price: 7500` when `PRIVATE_UTILITY.max_budget == 7500` is a hard
fail. You can convey "I'm price-sensitive" qualitatively if useful, but
the integer ceiling stays off the wire.

## Worked example — Path A (outbound search)

**Memory state when this skill activates:**

```
PREFERENCE.goal            = "wireless earbuds, noise-cancelling, low budget"
PREFERENCE.style           = ["minimalist"]
PREFERENCE.avoid_styles    = []
PREFERENCE.avoid_merchants = []
PRIVATE_UTILITY.max_budget = 7500
PRIVATE_UTILITY.must_not_share_list = ["budget"]
TRANSACTION.mandate_id     = "m-001"
```

**Outbound envelope:**

```yaml
from: buyer
to: platform:aggregator
action:
  kind: commerce.search
  payload:
    query: "wireless earbuds with active noise cancellation, minimalist design, delivery within 3 days"
    filters:
      must_have: [noise_cancellation]
      delivery_days_max: 3
    limit: 10
```

**Memory writes you also produce:**

```
TRANSACTION.last_search_id    = "<this envelope's msg_id>"
TRANSACTION.last_search_query = "wireless earbuds with active..."
```

Note: the payload says nothing about `$75` or the budget integer.
"Low budget" appears qualitatively in `goal` because that's already in
`PREFERENCE` — it crossed the prompt boundary at parse time, not now.
The integer ceiling never appears.

## Worked example — Path B (rank_offers reply)

Real-catalog scenario: a buyer is shopping Allbirds socks with a
moderate budget. The platform returns 4 candidates that exercise every
stage of the filter (claims-fast-path, claims-empty-list rejection,
world-grounding, variant grouping, stock pre-flight).

**Memory state when this skill activates:**

```
PREFERENCE.goal              = "wool crew socks, comfortable for daily wear"
PREFERENCE.must_have         = ["wool", "merino"]
PREFERENCE.style             = ["heathered", "size M"]
PREFERENCE.delivery_days     = 5
PRIVATE_UTILITY.max_budget   = 3000   # $30.00 in cents
TRANSACTION.cumulative_spend = 0
TRANSACTION.last_search_id   = "b-001"
```

**Inbound `platform.rank_offers` payload:**

```yaml
candidates:
  - offer_id: o-cozy-s
    sku_id:   PCC1HONU301
    merchant_id: merchant:allbirds
    unit_price: 2400          # $24.00
    qty: 1
    fulfillment: {method: standard, eta_days: 4}
    claims: [wool, merino]    # aggregator projected — fast path
    expires_at: "2026-05-26T12:00:00Z"

  - offer_id: o-cozy-m
    sku_id:   PCC1HONU302
    merchant_id: merchant:allbirds
    unit_price: 2400
    qty: 1
    fulfillment: {method: standard, eta_days: 4}
    # claims key ABSENT — must ground via world.get_listing
    expires_at: "2026-05-26T12:00:00Z"

  - offer_id: o-linen
    sku_id:   BL-SHEET-001
    merchant_id: merchant:brooklinen
    unit_price: 2100
    qty: 1
    fulfillment: {method: standard, eta_days: 3}
    claims: []                # explicit empty — ineligible, do NOT ground
    expires_at: "2026-05-26T12:00:00Z"

  - offer_id: o-overrun
    sku_id:   PCC1HONU399
    merchant_id: merchant:allbirds
    unit_price: 2400
    qty: 1
    fulfillment: {method: standard, eta_days: 4}
    claims: [wool, merino]
    expires_at: "2026-05-26T12:00:00Z"
```

**Stage 1 (mandate predicates):** all four survive — under budget,
within delivery window, no avoid-merchants list, expires in the future.

**Stage 2 (claims layer):**
- `o-cozy-s` → claims `{wool, merino} ⊇ must_have` → **eligible**.
- `o-cozy-m` → claims absent → **defer to Stage 3**.
- `o-linen`  → claims `[]` → **ineligible** (rejected without grounding).
- `o-overrun` → eligible (same as o-cozy-s).

**Stage 3 (grounding for o-cozy-m):**

```json
{"step": "tool_call",
 "calls": [{"tool": "world.get_listing",
            "args": {"sku_id": "PCC1HONU302"}}]}
```

History after the tool returns:

```
tool_call:
    world.get_listing({"sku_id": "PCC1HONU302"}) → {
      "sku_id": "PCC1HONU302",
      "name":   "Trino® Cozy Crew - Heathered Onyx",
      "category": "Socks",
      "list_price": {"amount": "24.00", "currency": "USD"},
      "merchant_id": "merchant:allbirds",
      "attributes": {
        "material": "wool, merino",
        "variant":  "M (W8-10)",
        "tags":     "allbirds::carbon-score => 5.9, collection:apr26"
      },
      "key_features_excerpt": "The Trino Cozy Crew is a breathable, ..."
    }
```

`must_have = ["wool", "merino"]` — both tokens appear in
`attributes.material` ("wool, merino"). **Eligible.**

**Stage 4 (stock pre-flight on the three survivors o-cozy-s, o-cozy-m,
o-overrun):**

```json
{"step": "tool_call",
 "calls": [
   {"tool": "world.is_in_stock", "args": {"sku_id": "PCC1HONU301", "qty": 1}},
   {"tool": "world.is_in_stock", "args": {"sku_id": "PCC1HONU302", "qty": 1}},
   {"tool": "world.is_in_stock", "args": {"sku_id": "PCC1HONU399", "qty": 1}}
 ]}
```

Suppose the world replies `true / true / false`. `o-overrun` drops.

**Variant grouping:**

The two survivors share `merchant_id: merchant:allbirds`. Their
`attributes.variant` values are `"S (W5-7)"` and `"M (W8-10)"`
respectively. Strip those substrings (with the trailing ` - ` separator)
from each `name`:

- `"Trino® Cozy Crew - Heathered Onyx"` (both — strip leaves the same
  base name).

→ One group. The mandate `PREFERENCE.style` includes the token
`"size M"`. The variant whose `attributes.variant` contains `M` is
`o-cozy-m`. **Winner: `o-cozy-m`.**

**Memory writes:**

```
TRANSACTION.selected_offer = {
  offer_id: "o-cozy-m",
  merchant_id: "merchant:allbirds",
  sku_id: "PCC1HONU302",
  qty: 1,
  unit_price: 2400,
  fulfillment: {method: "standard", eta_days: 4},
  claims: null,
  expires_at: "2026-05-26T12:00:00Z",
}
TRANSACTION.last_discovery_outcome = "picked"
```

**Outbound envelope:**

```yaml
from: buyer            # (runtime fills from; LLM does not set it)
to: merchant:allbirds
in_reply_to: <the rank_offers msg_id>
action:
  kind: commerce.accept_offer
  payload:
    offer_id: o-cozy-m
    mandate_id: m-001
```

The payload references the offer by id only — no price, no budget, no
private fields. The platform will follow with
`platform.create_match_certificate`, picked up by `purchase-confirmation`.

## Failure modes and what to do

### Analysis-only discovery mandates

Some mandates request an evidence-grounded comparison without authorizing a
purchase. When the mandate's public `task_context.action_schema` explicitly
requires `commerce.submit_decision_record`, complete the same budget,
grounding, and provenance checks above, but do not accept or settle an offer.
Emit one result to `runtime:evidence` with:

```yaml
action:
  kind: commerce.submit_decision_record
  payload:
    outcome: completed
    summary: <short public conclusion>
    details: <the schema-bound grounded decision>
```

The record is an auditable actor conclusion, not a World fact. Never copy
private utility into it. If the mandate does not explicitly request this
analysis endpoint, continue with the ordinary commerce flow.

### Path A (outbound search)

- **No `PREFERENCE.goal` in memory.** The mandate wasn't parsed yet —
  this skill is simply done; emit nothing. `mandate-parsing` runs on
  the mandate envelope and unblocks this skill on the next opportunity.
- **An item in `must_not_share_list` appears in your composed query.**
  Don't scrub on the fly — the runtime guard will catch it anyway and
  abort the run. Regenerate the query from `PREFERENCE` fields only.
  If you can't produce a clean query, emit nothing.
- **A previous `commerce.search` is still pending** (`TRANSACTION.last_search_id`
  is set and `last_discovery_outcome` is unset or "pending"). Don't
  fan out a second search; emit nothing and wait. Two searches in
  flight bloat the audit log without buying anything.

### Path B (inbound rank_offers)

- **No eligible offers after filtering.** Set
  `TRANSACTION.last_discovery_outcome = "no_candidate_found"` and emit
  no envelope. The buyer is stuck on this search; resolution is up to
  the next inbound envelope.
- **`in_reply_to` does not match `TRANSACTION.last_search_id`.** Stale
  reply — emit nothing, don't touch memory.

## Why this is its own skill

Query formulation is the single skill that gets the most upgrades over
the project's lifetime: taste grounding ("calm premium home office" →
attribute filters), multi-merchant ranking weights, learned ranking
policies. Pulling it out of negotiation / authorization keeps the
search-side ablation surface — "what changes if we swap query
formulation with `top-1-by-price`?" — clean, and it keeps those
downstream skills smaller because they don't need to know how the
candidate set was assembled.
