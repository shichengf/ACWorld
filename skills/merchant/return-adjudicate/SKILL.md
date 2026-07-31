---
name: return-adjudicate
description: Use to adjudicate a buyer's return request strictly per the stated refund policy and the reason given. Triggers on commerce.request_return; emits exactly one envelope — either commerce.issue_refund (approve full or partial) or commerce.respond_inquiry of category "policy" (decline with a public decline reason). Never invents ad-hoc terms, never negotiates the refund. The deterministic adjudicate_return helper is the source of truth; the skill body only narrates the regime.
when_to_use: Use when the inbound envelope is commerce.request_return with an order_id the merchant fulfilled. Do NOT use during a propose_offer / counter_offer turn (that is pricing-negotiate's). Do NOT use to handle pre-purchase Q&A (inquiry-handle owns that). Do NOT use to handle dispute escalation (dispute-defense owns that once platform.open_dispute fires).
group: operations
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_order
  - world.read_policy
  - memory.read
  - memory.write
context: main
---

# Return Adjudicate

## Native function guidance

Read the authoritative order and relevant evidence before deciding a return.
Apply the published policy to the verified status, timing, and reason. Approve
only the supported full or partial remedy, or give the permitted public policy
decline. Do not negotiate ad hoc terms, invent damage or inspection evidence,
or expose private pricing and operations. Make exactly one supplied refund or
response decision, then rely on Platform and World for the resulting state.

## Detailed workflow

The merchant's `support` role for the return path (VCP_SPEC.md §6).
A buyer asks for a return; the merchant decides per policy, emits
exactly one envelope, and writes the decision into TRANSACTION memory
so a follow-up dispute can be defended with the original adjudication.

Adjudication is **deterministic**: the `adjudicate_return` helper is the
single source of truth. The skill body documents the regime so a reader
understands *why* a decision was made; the merchant agent does not author
ad-hoc terms ("I can do 30% off if you keep it"), does not negotiate the
refund amount, and does not invent reason codes outside the lookup set.

## Procedure

1. Read state:
   - `order = get_order(order_id)` — must exist; carries `delivered_at`,
     `amount`, `status`.
   - `pol   = world.read_policy()` — `refund_policy`,
     `return_window_days`, optional partial-refund factor.
2. Run the deterministic adjudication:
   `decision = adjudicate_return(order_id, reason, today=ctx.now,
                                 condition=payload.condition)`.
   The helper returns the action dict directly — emit it as-is.

### Reason codes (decision rules — deterministic)

| Reason code              | Condition       | Outcome             |
| ------------------------ | --------------- | ------------------- |
| `defective`              | any             | full refund         |
| `damaged_on_arrival`     | any             | full refund         |
| `wrong_item`             | any             | full refund         |
| `not_as_described`       | any             | full refund         |
| `changed_mind`           | `new` / unspec. | full refund         |
| `changed_mind`           | `used`/`opened` | partial (50% def.)  |
| (anything else)          | —               | decline             |

A request outside `return_window_days` is declined with
`return_window_closed`, **regardless** of reason. The window is the
contract; the reason is the modifier.

### After the decision

3. Record in memory:
   `memory.write(TRANSACTION, f"return:{order_id}",
                 {decision, reason, amount, adjudicated_at: ctx.turn})`.
4. Emit exactly one envelope:
   - approved → `commerce.issue_refund`
     `{order_id, amount, reason, product, sku_id}` (the PSP executes
     the ledger move; the merchant only **requests** it).
   - declined → `commerce.respond_inquiry`
     `{category: "policy", answer: {decision: "declined",
                                     decline_reason, order_id}}`.

## Never disclose

- The order's margin, the per-unit cost basis, or any private profit
  math used to seed the partial-refund factor. The wire carries only
  the public refund amount and a public decline reason.
- A decline that hints at "but we could've gone X% if you'd…" — the
  decision is policy, not negotiation. `private-utility-guard` runs
  on every emit; defer to it.
- Internal restock state — a return that lands back as inventory is
  the world's concern (the runtime applies the inventory write); the
  merchant only emits the refund request.

## Bounds and ethics

- One envelope per turn. A buyer asking "and can I also exchange?"
  is a second turn, not a second envelope.
- Partial refunds round **down** to integer minor units — never round
  up at the buyer's expense. The integer-money invariant (VCP_SPEC §3)
  is non-negotiable.
- Decline reasons are drawn from a fixed public set; never invent a
  decline reason that quotes a private value or accuses the buyer
  ("you damaged it"). Those belong in a dispute, not a refund decline.

## Output

One `commerce.issue_refund` envelope (approved) or one
`commerce.respond_inquiry` envelope of category `"policy"` (declined).
Never both, never zero.
