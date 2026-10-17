---
name: mandate-parsing
description: Parse the user's PurchaseMandate into the buyer's memory at the start of a buyer-side journey. Use this skill whenever an inbound `delegate.create_purchase_mandate` envelope arrives and no `mandate_id` is yet recorded in TRANSACTION memory — this is the very first thing the buyer does in every scenario, and downstream skills (discovery, negotiation, authorization) all read what this one writes.
group: interpretation
---

# Buyer · Mandate Parsing

## Native function guidance

The Agent has already copied the authoritative mandate into typed private
memory. Do not reproduce a legacy memory step. On a new mandate, identify the
goal and shareable hard constraints, keep the budget and protected fields
private, then call the supplied marketplace search action. If the mandate is
missing a safe or feasible objective, use the supplied rejection or result
function. Never invent protocol routing or identifiers.

## What this skill is for

A buyer-side journey starts with a `delegate.create_purchase_mandate`
envelope from the user (or, in pre-v1.0 scenarios, synthesized from the
scenario YAML). The envelope carries a `PurchaseMandate` — the user's
goal, hard constraints, soft preferences, taste, and authority block (see
the ACWorld typed-envelope contract).

The buyer cannot act on a raw mandate. Every other buyer skill —
discovery's query formulation, negotiation's budget tracking,
authorization's cart-vs-mandate check — reads structured fields out of
memory. This skill is the **one place** where the raw mandate gets
unpacked into those typed memory buckets, exactly once per session.

## What to do

1. Read `envelope.action.payload` as a `PurchaseMandate`.

2. **Idempotency branch — re-delivered mandate.** Check
   `TRANSACTION.mandate_id` in memory. If a value is already there, the
   mandate was parsed on a prior turn. Touch no bucket. Your next step
   is **`{"step": "no_reply"}`** — the turn ends with no outbound,
   because nothing downstream needs to happen this turn either.

3. **First-delivery branch — parse and continue to discovery.** Split
   the mandate fields into the four memory regions per the table below
   (one `memory_update` step, all writes batched).

   **DO NOT emit `no_reply` after the writes.** This is the load-bearing
   instruction: the bounded internal loop is still inside the same
   external turn, and there is no second turn that delivers "you finished
   parsing." The buyer must continue **in this same turn** with:

   ```
   {"step": "load_skill", "name": "discovery-search"}
   ```

   followed by `discovery-search`'s Path A (compose the
   `commerce.search` query) culminating in an `emit_envelope` to
   `platform:aggregator`. The full first-turn sequence is therefore
   approximately:

   ```
   load_skill mandate-parsing → memory_update <all parsed fields>
   → load_skill discovery-search → emit_envelope commerce.search
   ```

   If you stop after the memory_update with `no_reply`, the buyer
   never asks the aggregator for offers and the entire journey
   silently halts.

## Where each field goes

The memory boundaries here are load-bearing for the privacy story. A
runtime guard scans every outbound payload against the agent's
`PRIVATE_UTILITY` bucket; values misfiled in `PREFERENCE` or
`TRANSACTION` would skip that scan and leak.

| Memory bucket       | Fields written                                                              |
| ------------------- | --------------------------------------------------------------------------- |
| `PREFERENCE`        | `goal`, `style` (= `soft_preferences.style`), `avoid_styles` (= `soft_preferences.avoid`), `taste.*`, `must_have` (= `hard_constraints.must_have`), `delivery_days` (= `hard_constraints.delivery_days`), `soft_constraints` (= `soft_constraints`, ordered most-important-first; compromisable), `avoid_merchants` (= default `[]` unless mandate specifies) |
| `TRANSACTION`       | `mandate_id`, `intent_expiry`, `cumulative_spend: 0`, `awaiting_user_approval: false`, `pending_cert_id: null`, `pending_offer_id: null` (the three approval-gate fields are initialized so downstream skills can read them safely without null-checks), and `return_after_purchase` (= the mandate's `return_after_purchase`, default `false`) — the `return-refund` skill reads it to decide whether to request a return after settling |
| `PRIVATE_UTILITY`   | `max_budget` (= `hard_constraints.budget`), `must_not_share_list` (= `authority.must_not_share_with_merchant`), `can_buy_without_confirmation` (= `authority.can_buy_without_confirmation`) |
| `TRUST`             | nothing — populated later by reputation reads                               |

The polarity matters: `hard_constraints.budget` is the buyer's
ceiling. The *value* of that ceiling is private utility; the *fact* of
having a budget is fine to mention to merchants. So the integer goes into
`PRIVATE_UTILITY`, not `TRANSACTION`. Same for `delivery_days` — its
existence is fine, but if your scenario flags a deadline as
must-not-share, move it from `PREFERENCE` to `PRIVATE_UTILITY`.

`authority.can_buy_without_confirmation` belongs in `PRIVATE_UTILITY`
because it governs the buyer's internal kill-switch — a merchant who
learned this value could exploit it (e.g. flood with low-priced offers
hoping the buyer auto-accepts). Treat it as private even though it's
not strictly a price.

## Rigid vs. compromisable needs

The mandate carries needs at three strengths. Keep them distinct — the
discovery skill treats them very differently:

- **Rigid (`hard_constraints`)** — budget, `delivery_days`, `must_have`
  features. **Never relaxed.** An offer that violates any of these is
  *ineligible*, full stop.
- **Compromisable (`soft_constraints`)** — an **ordered** list (most
  important first) of features the buyer *wants* but will give up if
  nothing in budget has them. Each entry is a feature string, or an
  object `{feature, importance}` (higher importance = relaxed later).
  Write it to `PREFERENCE.soft_constraints` verbatim, preserving order.
  Discovery prefers offers meeting more of these, and **relaxes the
  least-important first** when no eligible offer satisfies them all.
- **Taste hints (`soft_preferences.style` / `.avoid`)** — pure
  ranking/variant nudges, never a requirement and never relaxed-from
  (there is nothing to relax — they only tilt ties).

If the mandate omits `soft_constraints`, write `PREFERENCE.soft_constraints
= []`. Do **not** fabricate compromisable needs from the goal text — only
record what the mandate states.

## Worked example

**Inbound envelope (action.payload):**

```yaml
mandate_id: m-001
goal: "wireless earbuds, noise-cancelling, low budget"
hard_constraints:            # RIGID — never relaxed
  budget: 7500          # $75.00, minor units
  delivery_days: 3
  must_have: [noise_cancellation]
soft_constraints:            # COMPROMISABLE — ordered, most-important first
  - feature: bluetooth_5_3
    importance: 2
  - feature: usb_c_charging   # least important — relaxed first if needed
    importance: 1
soft_preferences:            # taste hints only (ranking nudges)
  style: [minimalist]
  avoid: []
authority:
  can_buy_without_confirmation: true
  must_not_share_with_merchant: [budget]    # mandate field name(s); the strip rule scans hard_constraints by this list
intent_expiry: "2026-05-22T00:00:00Z"
```

**Memory writes you produce:**

```
PREFERENCE.goal              = "wireless earbuds, noise-cancelling, low budget"
PREFERENCE.style             = ["minimalist"]
PREFERENCE.avoid_styles      = []
PREFERENCE.avoid_merchants   = []
PREFERENCE.must_have         = ["noise_cancellation"]
PREFERENCE.delivery_days     = 3
PREFERENCE.soft_constraints  = [{"feature": "bluetooth_5_3", "importance": 2}, {"feature": "usb_c_charging", "importance": 1}]
TRANSACTION.mandate_id    = "m-001"
TRANSACTION.intent_expiry = "2026-05-22T00:00:00Z"
TRANSACTION.cumulative_spend = 0
PRIVATE_UTILITY.max_budget = 7500
PRIVATE_UTILITY.must_not_share_list = ["budget"]
PRIVATE_UTILITY.can_buy_without_confirmation = true
```

**Next step in the same turn (NOT `no_reply`):**

```
{"step": "load_skill", "name": "discovery-search"}
```

Then `discovery-search` Path A composes the `commerce.search` payload
and emits to `platform:aggregator`. The full first-turn sequence is
four internal steps: `load_skill mandate-parsing` → `memory_update`
(above) → `load_skill discovery-search` → `emit_envelope
commerce.search`. The terminal step of THIS external turn is the
`commerce.search` emit, not anything from `mandate-parsing`.

## Failure modes and what to do

For both rejection cases below, the envelope shape is:

```yaml
from: buyer
to: consumer:persona          # back to the principal that delegated
in_reply_to: <inbound msg_id>
action:
  kind: delegate.reject_purchase
  payload:
    mandate_id: <from inbound; echo for correlation>
    reason: "<canonical code, see below>"
```

The `reason` is a short canonical code, **not** prose. Never echo the
malformed value itself in the reason — that's how leaks creep in
("got: 75.00" defeats the whole privacy story). The bad field's *name*
is fine; its *value* is not.

- **Required field missing in payload** (`mandate_id`, `goal`, or
  `hard_constraints.budget`): emit `delegate.reject_purchase` with
  `reason: "incomplete_mandate"`. Do not partial-parse what you have —
  downstream skills assume a complete record.

- **Field type mismatch** (e.g. `budget` arrives as a float or a string
  instead of integer minor units): emit `delegate.reject_purchase`
  with `reason: "type_mismatch:<field>"`, e.g. `"type_mismatch:budget"`.
  Same no-partial-parse rule — write nothing to memory.

- **`mandate_id` already present in TRANSACTION** (duplicate / re-delivery):
  this skill is simply done; touch no bucket, emit no envelope. The
  earlier parse already wrote everything, and re-writing would clobber
  state that downstream skills have since updated (e.g. `cumulative_spend`
  is no longer zero). The turn loop continues; whatever skill should
  handle the current situation will take over.

- **Tie-break on multiple failures**: check in the order missing-required
  → type-mismatch. Report only the first failure that fires.

## Why this is its own skill

Parsing is upstream of every other buyer skill and runs at most once per
session. Pulling it out of the discovery / negotiation prompts keeps
those prompts shorter and makes the parse step ablatable on its own —
e.g. swap in a strict parser that rejects mandates without an
`intent_expiry`, and measure the resulting drop in `task_success`
without touching downstream code.
