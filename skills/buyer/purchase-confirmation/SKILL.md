---
name: purchase-confirmation
description: Validate a proposed deal against the mandate, send the settlement request to the platform PSP, and book the receipt. Use this skill whenever an inbound envelope is one of (a) `platform.create_match_certificate` — the platform's confirmation after the buyer accepted a ranked offer; (b) a server-authored `commerce.accept_offer` relay from `platform:negotiation` — the negotiation path; (c) `delegate.approve_purchase` from the user — resumes a previously-halted authorization that was waiting on user approval; (d) `delegate.reject_purchase` from the user — withdraws a previously-halted authorization; or (e) `platform.settlement_receipt` — the PSP's confirmation that the order settled. This is the only skill that emits `platform.settle_payment`, `commerce.reject_offer`, or its-own `delegate.reject_purchase` (when the buyer auto-rejects pre-halt); the user-direction `delegate.approve_purchase` and `delegate.reject_purchase` are inbound triggers, not outbound emissions.
group: authorization
---

# Buyer · Purchase Confirmation

## Native function guidance

Confirm that the proposed deal is grounded, within the current mandate budget,
still in stock, and authorized by the principal. Read authoritative listing,
stock, balance, or order state when needed. If authorization and evidence are
complete, call the supplied settlement action. Otherwise reject, decline, or
finish safely. A receipt is evidence of a completed settlement, not permission
to create a second payment. The Agent constructs protocol identity,
idempotency, and routing fields.

## What this skill is for

The buyer has a deal on the table. Three inbound paths reach this skill:

1. **Match-certificate path (typical).** Inbound is
   `platform.create_match_certificate`. The buyer earlier emitted
   `commerce.accept_offer` from `discovery-search` Path B, the platform
   verified the match against its own policies, and now hands the
   buyer a certificate. The full offer record is already in
   `TRANSACTION.selected_offer` thanks to discovery's write — this skill
   does NOT re-run the eligibility filter (discovery owns that).

2. **Negotiation path.** Inbound is a server-authored
   `commerce.accept_offer` relay from `platform:negotiation`. The
   `platform_mediation` block identifies the merchant that submitted the
   acceptance and the recipient. World has already persisted the terminal
   agreement. On this path, no discovery filter ran, so this skill DOES
   run the full check.

3. **Settlement-receipt path.** Inbound is `platform.settlement_receipt`
   from `platform:psp`, replying to a `platform.settle_payment` this
   skill emitted earlier. This is where the buyer records that money
   actually moved.

Paths 1 and 2 end the same way: emit `platform.settle_payment` to
`platform:psp`, or refuse if a final check fails. Path 3 just records
the receipt and emits `no_reply`. The skill is intentionally **thin**
— it owns the final budget-freshness check, the settle envelope, and
the receipt-booking; it does NOT own the upstream eligibility filter
(discovery owns that).

There is **no `commerce.create_cart` step.** The platform's PSP
([src/agents/platform.py PSPPolicy](../../../src/agents/platform.py))
recognizes only `platform.settle_payment` and replies with
`platform.settlement_receipt`. A "cart" object lives only as
`TRANSACTION.selected_offer` in buyer memory — there is no envelope
between accept and settle.

## What to do

There are **five** inbound triggers; A and B start a fresh authorization,
C and D resume a previously-halted one, and E books a settled receipt.

### Branch A — fresh authorization on `platform.create_match_certificate`

Read the `MatchCertificate` payload, then read the full offer record
from `TRANSACTION.selected_offer` (written by `discovery-search` Path B).
If the cert's `offer_id` doesn't match `selected_offer.offer_id`, emit
`commerce.reject_offer` with `reason: "offer_mismatch"`.

### Branch B — fresh authorization on a mediated `commerce.accept_offer`

Read the offer from `TRANSACTION.selected_offer`. The selector activates
both `negotiation` and `purchase-confirmation` on this inbound;
`negotiation` Branch N-C runs FIRST and commits `selected_offer` from
`inbound.payload.unit_price` (the merchant's echoed agreed price). Verify
the envelope is from `platform:negotiation`, its
`platform_mediation.status` is `accepted`, its recipient is this buyer,
and its `negotiation_id` matches the pending negotiation. If
`selected_offer` is missing on this branch, do **not** fall back to
guessing from the inbound payload — instead emit `no_reply` and stop;
the reconciler's `BudgetReconcileViolation` anomaly check will catch
the missing commit if you ever attempt the settle. The full eligibility
check (not just freshness) runs in this branch because no discovery
filter was upstream.

### Branch C — resume on `delegate.approve_purchase`

Only valid if `TRANSACTION.awaiting_user_approval == true`. Verify the
inbound `cert_id` matches `TRANSACTION.pending_cert_id`; if not, emit
`commerce.reject_offer` with `reason: "offer_mismatch"`. Otherwise
re-read `TRANSACTION.selected_offer` and re-run the **freshness check**
(price + expires_at + max_budget) — the user might have approved hours
later. On pass, emit `platform.settle_payment` (same shape as Branch A
success, see step 5 below) and clear the user-approval wait flags:

```
TRANSACTION.awaiting_user_approval = false
TRANSACTION.pending_cert_id        = null
TRANSACTION.pending_settlement_order_id = "ord-<mandate_id>-<offer_id>"
```

Do NOT increment `cumulative_spend` here — that happens in Branch E
when the receipt confirms money moved.

### Branch D — resume on `delegate.reject_purchase`

Only valid if `TRANSACTION.awaiting_user_approval == true`. Walk away
from the held deal: emit `commerce.reject_offer` to the merchant with
`reason: "user_confirmation_required"` (the user declined, but the
merchant doesn't need to know why), then clear the wait flags as in
Branch C. Do not touch `cumulative_spend`.

### Branch E — book settlement on `platform.settlement_receipt`

Triggered when inbound is `platform.settlement_receipt` from
`platform:psp`. The payload looks like:

```yaml
action:
  kind: platform.settlement_receipt
  payload:
    order_id: "ord-m-001-o-cozy-m"
    txn_id:   "txn:ord-m-001-o-cozy-m"
    status:   "settled"
```

Verify `payload.order_id == TRANSACTION.pending_settlement_order_id`
(stale receipts arriving after a session reset get dropped — emit
`no_reply` without touching memory). On match:

1. Read `TRANSACTION.selected_offer` for `unit_price` and `qty`.
2. Memory writes (batched in one `memory_update`):
   ```
   TRANSACTION.cumulative_spend            += unit_price * qty
   TRANSACTION.completed_orders            append {order_id, txn_id}
   TRANSACTION.pending_settlement_order_id  = null
   ```
3. Emit `no_reply`. The external turn ends; the buyer is ready for the
   next mandate.

There is **no `payment_failed` branch.** PSP raises `OutOfStock` instead
of replying — from the buyer's side that manifests as "no inbound ever
arrives." That's a Runtime-level concern; the Stage 4 stock pre-flight
in `discovery-search` is the buyer's mitigation.

### Shared steps (Branches A and B)

For **Branch A and Branch B only**. After identifying the offer:

1. Read the mandate slice from memory:
   - `PREFERENCE.goal`, `PREFERENCE.style` (subjective check, Branch B only)
   - `PREFERENCE.must_have`, `PREFERENCE.delivery_days` (hard constraints, Branch B only)
   - `TRANSACTION.mandate_id`, `TRANSACTION.cumulative_spend`
   - `PRIVATE_UTILITY.max_budget` (compare locally; never emit)
   - `PRIVATE_UTILITY.can_buy_without_confirmation` (governs the
     authority gate below)

2. Run the path-specific check:
   - **Branch A (match-certificate)**: do only the **final freshness check**:
     `offer.unit_price * offer.qty + TRANSACTION.cumulative_spend ≤ PRIVATE_UTILITY.max_budget`,
     and `offer.expires_at > now`, and (if certificate carries
     `checks_passed`) verify all four sub-checks are `true`. Trust the
     platform's match on must_have / delivery / claim grounding —
     re-running them here is wasted inference. If `checks_passed`
     shows any `false`, emit `commerce.reject_offer` with
     `reason: "certificate_checks_failed:<which>"`.
   - **Branch B (negotiation)**: do the **full** offer-vs-mandate check
     (the same 5 predicates discovery would have run — see table below).
     Must_have grounding uses the same `world.get_listing` tool_call
     pattern that `discovery-search` Stage 3 uses.

3. Apply the **authority gate** (this is the canonical halt path, not
   a failure):
   - If `PRIVATE_UTILITY.can_buy_without_confirmation == true`, fall
     through to step 4.
   - If `false`, the buyer is not authorized to settle alone. Emit
     `no_reply` this turn; write:
     ```
     TRANSACTION.awaiting_user_approval = true
     TRANSACTION.pending_cert_id        = <cert_id from Branch A; null in Branch B>
     TRANSACTION.pending_offer_id       = <offer.offer_id>
     ```
     Return; the agent halts. Resumption is owned by Branch C / D above.
     No new outbound envelope kind is introduced — the buyer's halt is
     pure memory state.

4. If the check (step 2) failed, branch by failure type — see
   "Failure modes" below.

5. On success (check passed AND authority gate cleared), emit
   `platform.settle_payment` to `platform:psp`.

   **CRITICAL — payload values are NOT free to choose. Every field
   except `order_id` is copied verbatim from
   `TRANSACTION.selected_offer` (or `{{AGENT_ID}}`). Do not paraphrase,
   substitute, or invent values. Re-emitting the wrong merchant_id or
   a fabricated unit_price will settle the wrong order against the
   wrong account.**

   **PRE-FLIGHT INVARIANT (verify before emit):** the
   `agreed_price.amount` you are about to send (interpreted as
   integer cents) MUST equal `TRANSACTION.selected_offer.unit_price`.
   If `selected_offer.unit_price = 2400` you MUST emit
   `agreed_price.amount = "24.00"` — not `"21.60"`, not any other
   number. For Branch B, the runtime's ``PSPPolicy`` resolves
   `negotiation_id` against the authoritative World thread. It verifies
   the exact buyer, merchant, SKU, quantity, price, currency, listing
   revision, validity interval, and deterministic order identity. Audit
   text is not settlement authority. Trying to "save money" by emitting
   a different `agreed_price` will fail before any commerce state changes.

   Field-by-field source table:

   | Payload field | Source — copy from this exact location |
   | --- | --- |
   | `order_id` | derived: `f"ord-{TRANSACTION.mandate_id}-{TRANSACTION.selected_offer.offer_id}"` |
   | `buyer_id` | `{{AGENT_ID}}` (literal text from your system prompt). NOT persona name. |
   | `merchant_id` | `TRANSACTION.selected_offer.merchant_id` — verbatim string |
   | `sku_id` | `TRANSACTION.selected_offer.sku_id` — verbatim string |
   | `qty` | `TRANSACTION.selected_offer.qty` — verbatim integer |
   | `agreed_price.amount` | `TRANSACTION.selected_offer.unit_price` converted from integer cents to decimal-string dollars: e.g. `2400` → `"24.00"`. **NEVER guess or round to a different value.** |
   | `agreed_price.currency` | always `"USD"` in this iteration |
   | `negotiation_id` | Branch B only: `inbound.payload.negotiation_id` from the accepted server relay; omit on the match-certificate path |

   Envelope shape (with placeholders the runtime fills):

   ```yaml
   to: platform:psp
   idempotency_key: "settle:<mandate_id>:<offer_id>"
   action:
     kind: platform.settle_payment
     payload:
       order_id:    "ord-<TRANSACTION.mandate_id>-<TRANSACTION.selected_offer.offer_id>"
       buyer_id:    "{{AGENT_ID}}"
       merchant_id: "<TRANSACTION.selected_offer.merchant_id>"
       sku_id:      "<TRANSACTION.selected_offer.sku_id>"
       qty:         <TRANSACTION.selected_offer.qty>
       agreed_price:
         amount:   "<TRANSACTION.selected_offer.unit_price as 'D.DD' string>"
         currency: "USD"
       negotiation_id: "<Branch B pending negotiation id; omit in Branch A>"
   ```

   Memory write (one `memory_update`):
   ```
   TRANSACTION.pending_settlement_order_id = <the order_id you just emitted>
   ```

   **Do NOT bump `cumulative_spend` here.** That belongs to Branch E —
   only after the PSP confirms money moved is the spend booked.

   Worked numeric example: if
   `TRANSACTION.selected_offer = {merchant_id: "merchant:allbirds",
   sku_id: "PCC1HONU302", qty: 1, unit_price: 2160, ...}` and
   `TRANSACTION.mandate_id = "m-001"`, the emitted payload MUST be:

   ```json
   {
     "order_id":    "ord-m-001-agg:PCC1HONU302",
     "buyer_id":    "buyer",
     "merchant_id": "merchant:allbirds",
     "sku_id":      "PCC1HONU302",
     "qty":         1,
     "agreed_price": {"amount": "21.60", "currency": "USD"},
     "negotiation_id": "neg:m-001:agg:PCC1HONU302"
   }
   ```

   Notice the merchant_id is the exact string from `selected_offer`,
   not a different merchant from your training data. The amount
   "21.60" is `2160 / 100`, not 21.6, not "21.6", and not any other
   number you find plausible.

## The offer-vs-mandate check

Five predicates, all must hold:

| Predicate                                                                | Source                                                       |
| ------------------------------------------------------------------------ | ------------------------------------------------------------ |
| `offer.unit_price * offer.qty + cumulative_spend ≤ max_budget`           | `PRIVATE_UTILITY.max_budget` + `TRANSACTION.cumulative_spend` |
| `offer.fulfillment.eta_days ≤ mandate.hard_constraints.delivery_days`    | mandate slice in `PREFERENCE` / `TRANSACTION`                |
| Every attribute in `mandate.hard_constraints.must_have` is satisfied by the SKU | `world.get_listing(offer.sku_id)` via a `tool_call` step |
| `offer.merchant_id` is not on the buyer's `PREFERENCE.avoid_merchants` list | `PREFERENCE.avoid_merchants` (default `[]`)               |
| `offer.expires_at > now`                                                 | offer payload                                                |

These are local computations (the must_have predicate may require a
`tool_call` to ground against `Listing.attributes`). If you're
implemented as a deterministic baseline, the same checks apply.

## Worked example — Branch A → settle → receipt (happy path)

The canonical buyer journey: discovery picked an offer, platform issued
a match certificate, buyer is auto-authorized (no user gate). Two
external turns: send settle, then book receipt.

### Turn 1 — inbound `platform.create_match_certificate`

**Memory state:**

```
PRIVATE_UTILITY.max_budget               = 3000     # $30.00 in cents
PRIVATE_UTILITY.can_buy_without_confirmation = true
TRANSACTION.cumulative_spend             = 0
TRANSACTION.mandate_id                   = "m-001"
TRANSACTION.selected_offer = {
  offer_id: "o-cozy-m",
  merchant_id: "merchant:allbirds",
  sku_id: "PCC1HONU302",
  qty: 1,
  unit_price: 2400,
  fulfillment: {method: "standard", eta_days: 4},
  expires_at: "2026-05-26T12:00:00Z"
}
```

**Inbound:**

```yaml
from: platform:aggregator
to: buyer
action:
  kind: platform.create_match_certificate
  payload:
    cert_id:  cert-001
    offer_id: o-cozy-m
    checks_passed: {budget: true, delivery: true, claims: true, expiry: true}
```

**Checks:**

```
cert.offer_id == selected_offer.offer_id         ✓
2400 + 0 ≤ 3000                                   ✓
expires_at in future                              ✓
checks_passed all true                            ✓
can_buy_without_confirmation == true              → fall through to step 5
```

**Outbound envelope** (note `{{AGENT_ID}}` is substituted by the system
prompt — it appears here literally as the buyer's own id, e.g. `"buyer"`):

```yaml
from: buyer                          # runtime fills from `{{AGENT_ID}}`
to: platform:psp
in_reply_to: <cert envelope msg_id>
idempotency_key: "settle:m-001:o-cozy-m"
action:
  kind: platform.settle_payment
  payload:
    order_id:    "ord-m-001-o-cozy-m"
    buyer_id:    "buyer"
    merchant_id: "merchant:allbirds"
    sku_id:      "PCC1HONU302"
    qty:         1
    agreed_price:
      amount:   "24.00"
      currency: "USD"
```

**Memory writes:**

```
TRANSACTION.pending_settlement_order_id = "ord-m-001-o-cozy-m"
```

Note: `cumulative_spend` is still 0; it bumps in Turn 2.

### Turn 2 — inbound `platform.settlement_receipt`

**Inbound:**

```yaml
from: platform:psp
to: buyer
in_reply_to: <the settle_payment msg_id>
action:
  kind: platform.settlement_receipt
  payload:
    order_id: "ord-m-001-o-cozy-m"
    txn_id:   "txn:ord-m-001-o-cozy-m"
    status:   "settled"
```

`payload.order_id == TRANSACTION.pending_settlement_order_id` ✓

**Memory writes:**

```
TRANSACTION.cumulative_spend            = 2400
TRANSACTION.completed_orders            = [{order_id: "ord-m-001-o-cozy-m", txn_id: "txn:..."}]
TRANSACTION.pending_settlement_order_id  = null
```

**Outbound:** `no_reply`. External turn ends; the buyer is idle and
ready for the next mandate.

The integer `2400` is the `unit_price`; the budget integer `3000` never
appears on the wire. `agreed_price.amount = "24.00"` is buyer-visible
market data, not the budget ceiling — the privacy invariant scans for
`"3000"` as a substring of the outbound, finds nothing, and the run
proceeds.

## Worked example — gated flow (Branches A → halt → C → settle → E)

When `PRIVATE_UTILITY.can_buy_without_confirmation == false`, the
authorization splits across three external turns: halt for approval,
resume + settle, then book receipt.

### Turn N — inbound `platform.create_match_certificate`

```
cert.offer_id == selected_offer.offer_id        ✓
freshness (2400 + 0 ≤ 3000, expires future)      ✓
checks_passed (all four true)                    ✓
can_buy_without_confirmation == false            → AUTHORITY GATE HALT
```

**Memory writes (no envelope):**

```
TRANSACTION.awaiting_user_approval = true
TRANSACTION.pending_cert_id        = "cert-001"
TRANSACTION.pending_offer_id       = "o-cozy-m"
```

**Outbound:** `no_reply`. Turn ends; the buyer is parked.

### Turn N+k — inbound `delegate.approve_purchase` carrying `cert_id: cert-001`

```
awaiting_user_approval == true                   ✓
inbound cert_id == pending_cert_id (both cert-001) ✓
re-freshness (selected_offer.expires_at > now)    ✓
re-budget (still 2400 + 0 ≤ 3000)                ✓
```

**Outbound:** the same `platform.settle_payment` envelope shown in the
happy-path Turn 1 above.

**Memory writes:**

```
TRANSACTION.awaiting_user_approval        = false
TRANSACTION.pending_cert_id               = null
TRANSACTION.pending_offer_id              = null
TRANSACTION.pending_settlement_order_id   = "ord-m-001-o-cozy-m"
```

### Turn N+k+1 — `platform.settlement_receipt`

Same as the happy-path Turn 2: book cumulative_spend, append the
completed order, clear pending flag, `no_reply`.

If instead `delegate.reject_purchase` had arrived (Branch D), the
buyer would emit `commerce.reject_offer` to `merchant:allbirds` with
`reason: "user_confirmation_required"` and clear the user-approval wait
flags without touching `cumulative_spend` or `pending_settlement_order_id`.

## Failure modes and what to do

### Decision-only completion

If the buyer mandate explicitly asks for a grounded decision record rather
than payment, a `platform.create_match_certificate` may close the comparison
without moving money. Revalidate that the certificate matches the selected
offer. Then emit `commerce.submit_decision_record` to `runtime:evidence` using
the public action schema in the mandate. Do not emit `platform.settle_payment`,
increment spend, or claim that an order exists. This branch is permitted only
when the mandate clearly selects decision-only completion.

On **every** failure branch below: do NOT increment
`TRANSACTION.cumulative_spend`, do NOT set
`TRANSACTION.pending_settlement_order_id`, do NOT emit
`platform.settle_payment`. Both money-state writes belong on the
success branch (settle send + receipt book).

The two rejection envelope shapes:

```yaml
# Walk away from this offer (typical for routine fails)
from: buyer
to: merchant                  # echo the offer's merchant_id
in_reply_to: <inbound msg_id>
action:
  kind: commerce.reject_offer
  payload:
    offer_id: <from offer>
    mandate_id: <from memory>
    reason: "<short canonical code; see below>"
```

```yaml
# Escalate to the principal (when authority requires user confirmation
# and the buyer auto-rejects pre-halt — rare; the typical halt path is
# pure memory + no_reply, see step 3 of "Shared steps")
from: buyer
to: consumer:persona
in_reply_to: <inbound msg_id>
action:
  kind: delegate.reject_purchase
  payload:
    mandate_id: <from memory>
    offer_id: <from offer>
    reason: "<short canonical code>"
```

The `reason` field is a short canonical code, never prose. Approved codes:
`over_budget`, `missing_must_have:<feature>`, `delivery_too_slow`,
`offer_expired`, `sku_not_found`, `user_confirmation_required`,
`offer_mismatch`, `certificate_checks_failed:<which>`.
**Never restate the price or the budget integer in the reason** —
that defeats the privacy story. The reason names what failed; the
specifics live in the trace, not on the wire.

### Branch table

- **Over budget** (`offer.unit_price + cumulative_spend > max_budget`):
  emit `commerce.reject_offer` with `reason: "over_budget"`. Don't
  escalate to `delegate.reject_purchase` unless
  `PRIVATE_UTILITY.can_buy_without_confirmation` is false — for a
  routine "too expensive" the buyer just walks.

- **Hard constraint violated** (missing `must_have` feature):
  `commerce.reject_offer` with `reason: "missing_must_have:<feature>"`.

- **Delivery too slow** (`eta_days > delivery_days`):
  `commerce.reject_offer` with `reason: "delivery_too_slow"`.

- **`PRIVATE_UTILITY.can_buy_without_confirmation == false`**: NOT a
  failure — see "Authority gate" in step 3 of the shared steps above.
  The buyer halts and waits for a subsequent inbound
  `delegate.approve_purchase` (resume via Branch C) or
  `delegate.reject_purchase` (withdraw via Branch D).

- **`offer.expires_at < now`**: stale offer. `commerce.reject_offer`
  with `reason: "offer_expired"`.

- **`ctx.world.get_listing(sku_id)` returns `None`**: SKU vanished
  from the catalog between offer and confirm. `commerce.reject_offer`
  with `reason: "sku_not_found"`.

- **Stale `platform.settlement_receipt`** (`payload.order_id` does not
  match `TRANSACTION.pending_settlement_order_id`): emit `no_reply`,
  do not touch memory. The receipt corresponds to a settled order from
  a prior session or a re-delivery; ignoring it is safe because the
  real receipt either already arrived or never will.

## Why this is its own skill

This is the only skill that owns the terminal money-moving action.
Keeping it small means three things:

1. **Auditability.** When an episode ends with the wrong order settled,
   the audit trail points at one skill — not at a tangled blob with
   discovery and negotiation also wired in.
2. **Ablation.** Swap this skill for the deterministic baseline
   `purchase-confirmation-hard-constraints-only` (a one-page bundle
   that only checks budget + must_have, skipping subjective hints) and
   you get a clean A/B on how much LLM-mediated authorization is
   worth.
3. **Trust boundary.** The check that emits the money-moving envelope
   sits in one prompt context, so the user (and the reviewer) can
   inspect exactly what the LLM has to consult before authorizing.
