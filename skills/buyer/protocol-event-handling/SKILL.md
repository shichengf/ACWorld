---
name: protocol-event-handling
description: Validate sealed, order-bound protocol callbacks, process only current authorized events, reject duplicates or stale and cross-order events, and record the persisted Platform receipt.
group: operations
---

# Buyer · Protocol Event Handling

## Native function guidance

Treat a delivered protocol event as a request that still requires validation.
Use supplied evidence, claim, and mandate revision reads to verify that it is
current, bound to this buyer's authorized transaction, and not already handled.
Acknowledge or process only a valid event. Reject stale, duplicate,
cross-transaction, ungrounded, or unauthorized events without revealing private
state. Publish evidence only from facts already available through
CommerceWorld. After the persisted Platform receipt, finish safely.

## Authority boundary

`platform.deliver_protocol_event` contains a World-sealed event. The buyer
chooses a compact decision. Platform revalidates the event and World persists
the decision receipt. Never execute a callback by directly changing payment,
fulfillment, refund, or order state.

## Validate a delivered event

Read `payload.event` and require all of the following:

1. `binding.recipient_id` is this exact actor.
2. `binding.order_id` belongs to this buyer and is within the current mandate.
3. `event_id`, `event_digest`, `binding.binding_digest`, and
   `reference_digest` are nonempty and structurally valid.
4. The event sequence follows the last accepted event for the same stream and
   its predecessor digest matches the recorded predecessor.
5. The required order state and revision agree with the authoritative context
   available for this decision.
6. The event is not past `expires_at_tick` at the observable logical time.
7. The same event digest has not already received a terminal decision from
   this actor.
8. A certificate reference is bound to the same order and authorization scope.

Never transfer event state between orders or streams. Maintain an order-scoped
record in `TRANSACTION.protocol_events` containing event id, digest, sequence,
predecessor, and last receipt decision.

## Emit a decision

When every check passes and the referenced operation is authorized, emit:

```yaml
step: emit_envelope
envelope:
  to: platform:events
  idempotency_key: "decision:<event_id>:process"
  action:
    kind: commerce.process_protocol_event
    payload:
      event_id: "<exact delivered event_id>"
      reason: "validated current operation"
```

When any identity, ownership, ordering, state, freshness, replay, or authority
check fails, emit a rejection instead:

```yaml
step: emit_envelope
envelope:
  to: platform:events
  idempotency_key: "decision:<event_id>:reject"
  action:
    kind: commerce.reject_protocol_event
    payload:
      event_id: "<exact delivered event_id>"
      reason: "duplicate, expired, misordered, or replayed callback"
```

Do not process an event merely because its text suggests success. When a
required check cannot be established, rejection is the safe terminal choice.

## Handle `platform.protocol_event_receipt`

The payload contains a persisted `receipt`. Verify its actor, event id, event
digest, binding, decision, and idempotency key against the pending decision.
Record the receipt digest and observed order revision under the matching order,
clear the pending decision, and emit `no_reply`.

A duplicate identical receipt is an idempotent observation. A conflicting
receipt must be rejected as a protocol anomaly and must not trigger another
event decision.
