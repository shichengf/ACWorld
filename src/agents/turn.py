"""Re-entrant turn machinery for the process split (R1).

In the split topology an agent runs in its own process and cannot read World
in-process. A grounding read mid-turn becomes a ``world.read_*`` VCP envelope
routed back through the central dispatcher; the agent's turn SUSPENDS and
RESUMES when the matching ``world.response`` arrives. One logical decision thus
spans several dispatcher rounds, but its step history accumulates in ONE
:class:`TurnFrame` — so the reasoning trace stays one record per decision.

Transport-agnostic: the same frame threads through ``InProcessTransport``
(``remote_world=True``) and ``HttpTransport``. Correlation ids are derived
deterministically from ``(decision_id, seq)`` (never wall-clock / uuid) so the
audited envelope stream is identical across both topologies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Collection
from typing import TYPE_CHECKING, Any

from protocol.actions import ActionKind, is_send_allowed
from protocol.envelope import Envelope, validate
from protocol.errors import PartitionViolation

if TYPE_CHECKING:
    from agents.types import TurnTrace


#: Whitelisted world TOOL -> the ``world.read_*`` action it becomes on the wire
#: (the inverse of :meth:`world.service.WorldService._read`). ``world.get_balance``
#: has NO single-envelope equivalent (it aggregates the whole ledger
#: caller-scoped) and is intentionally absent — re-entrant mode rejects it until a
#: dedicated balance-read surface exists (tracked for the real-buyer increment).
_TOOL_TO_READ_KIND: dict[str, str] = {
    "world.get_listing": "world.read_catalog",
    "world.search_catalog": "world.read_catalog",
    # is_in_stock maps to the availability-only read (bool, no counts on the
    # wire) — NOT read_inventory, which would expose qty_available/qty_reserved.
    "world.is_in_stock": "world.read_availability",
    "world.get_merchant_reputation": "world.read_reputation",
    "world.get_order": "world.read_order",
    "world.get_friends": "world.read_friends",
    "world.get_friend_reviews": "world.read_friend_reviews",
    "world.get_review_evidence": "world.read_review_evidence",
    "world.get_evidence_record": "world.read_evidence_record",
    "world.get_mandate_revisions": "world.read_mandate_revisions",
    "world.get_listing_claim": "world.read_listing_claim",
    "world.list_listing_claims": "world.read_listing_claims",
    # Framework-only authority prerequisite.  It deliberately has no public
    # business intent and is issued by Agent before an after-sales business
    # decision in a split-process topology.
    "world.get_after_sales_authority": "world.read_after_sales_authority",
    "world.get_protocol_event_authority": "world.read_protocol_event_authority",
}

# Public audit boundary for framework-generated, actor-scoped World reads.
# Formal benchmark authority checks import this set so HTTP/re-entrant tool
# transport is not confused with an actor attempting a World mutation.  The
# envelope-shape and response-causality checks remain in the authority gate.
PUBLIC_WORLD_READ_ACTION_KINDS: frozenset[str] = frozenset(
    _TOOL_TO_READ_KIND.values()
)


@dataclass(frozen=True)
class ToolCall:
    """One Agent-compiled, read-only World operation within a turn."""

    tool: str
    args: dict[str, Any]


@dataclass
class TurnFrame:
    """Continuation of a suspended agent turn — one logical decision."""

    decision_id: str
    original_inbound: Envelope
    history: list[dict[str, Any]] = field(default_factory=list)
    activated: "tuple[str, ...] | None" = None
    trace: "TurnTrace | None" = None
    steps_used: int = 0
    read_seq: int = 0
    pending_calls: list[ToolCall] = field(default_factory=list)
    pending_results: list[dict[str, Any]] = field(default_factory=list)
    # Spawn-frozen actor roots and direct response ancestry determine whether
    # report-result functions are legal for this logical turn.  Retain the
    # decision across every re-entrant World read rather than recomputing it
    # from a ``world.response`` envelope.
    actor_reports_allowed: bool = False
    # Exact read batches already executed in this logical turn. A repeated
    # pure-read batch is an invalid model choice: the Agent offers the one
    # lifetime repair, then terminates a further repeat as protocol evidence.
    business_read_fingerprints: set[str] = field(default_factory=set)
    # Re-entrant reads suspend one call at a time. Keep the current fingerprint
    # until every response has arrived and the framework commits the tool step.
    business_pending_read_fingerprint: str | None = None
    # Exact request currently on the Runtime audit and awaiting a
    # world.response.  It remains framework-owned across suspension so a
    # forced flush can claim the audited read without inventing a result.
    pending_read_msg_id: str | None = None
    # Agent-owned after-sales authority prerequisites use the same audited
    # World read transport but are not model-authored business decisions. The flag keeps their
    # results out of provider conversation/history and routes them into the
    # private phase cache before inference begins.
    framework_authority_prerequisite: str | None = None


@dataclass
class TurnSuspended:
    """The turn emitted a world read mid-flight and awaits its ``world.response``.

    ``read_env`` is enqueued + audited + routed to World by the dispatcher like
    any envelope; ``frame`` is parked under ``read_env.msg_id`` until the matching
    response resumes the turn.
    """

    frame: TurnFrame
    read_env: Envelope


def finalize_forced_turn_trace(
    frame: TurnFrame,
    *,
    audited_read_msg_ids: Collection[str] = (),
) -> None:
    """Finalize a suspended decision with an exact pending-read claim.

    A forced flush may occur after the Runtime audited the World request but
    before its response was delivered.  Completed results in the current batch
    and the one issued request are real framework evidence.  Later unissued
    calls are deliberately omitted.  No result is fabricated for the pending
    request.
    """

    if frame.trace is None:
        frame.pending_calls = []
        frame.pending_results = []
        frame.pending_read_msg_id = None
        frame.business_pending_read_fingerprint = None
        frame.framework_authority_prerequisite = None
        return
    if frame.pending_read_msg_id is not None:
        if not frame.pending_calls:
            raise RuntimeError("suspended turn lost its pending ToolCall")
        results = list(frame.pending_results)
        # A read returned by the Agent can still be waiting in Runtime's queue.
        # Claim it only after the dispatcher has appended it to audit.jsonl.
        if frame.pending_read_msg_id in audited_read_msg_ids:
            call = frame.pending_calls[0]
            results.append(
                {
                    "tool": call.tool,
                    "args": call.args,
                    "pending_request_msg_id": frame.pending_read_msg_id,
                    "status": "pending_response",
                }
            )
        if results:
            frame.history.append(
                {
                    "step": "tool_call",
                    "results": results,
                    "incomplete": True,
                }
            )
        frame.pending_results = []
    frame.pending_calls = []
    frame.pending_read_msg_id = None
    frame.business_pending_read_fingerprint = None
    frame.framework_authority_prerequisite = None
    frame.trace.finalize(frame.history, result=None, terminal="forced_flush")


def decision_id_for(env: Envelope) -> str:
    """Stable correlation id for the decision a fresh inbound opens."""
    return env.idempotency_key or env.msg_id


def tool_call_to_read_envelope(
    call: "ToolCall", *, from_id: str, correlation_id: str, seq: int, ts: str
) -> Envelope:
    """Convert one whitelisted world-read tool call into its ``world.read_*``
    envelope. ``correlation_id`` is the triggering inbound's msg_id (always
    unique per envelope) so two turns that happen to share an idempotency_key
    cannot mint colliding read ids. ``msg_id`` / ``idempotency_key`` are
    deterministic (no wall-clock) so the audited stream matches across
    transports and the response correlates back via ``in_reply_to``."""
    kind = _TOOL_TO_READ_KIND.get(call.tool)
    if kind is None:
        raise ValueError(
            f"tool {call.tool!r} has no single-envelope world read mapping for "
            f"re-entrant grounding; supported: {sorted(_TOOL_TO_READ_KIND)}"
        )
    cid = f"{correlation_id}:read:{seq}"
    return Envelope(
        msg_id=cid,
        ts=ts,
        from_=from_id,
        to="world",
        # Preserve the causal edge to the external inbound that opened this
        # decision.  Besides replay/debugging, this lets deterministic progress
        # scoring distinguish a real model-authored read from a seeded
        # actor-shaped scenario event (which has no parent).
        in_reply_to=correlation_id,
        idempotency_key=cid,
        action={"kind": kind, "payload": _read_payload(call)},
    )


def validate_reentrant_read_partition(read_env: Envelope, *, actor_id: str) -> None:
    """Reject a framework-generated World read inside the owning Agent turn.

    Re-entrant tools suspend a decision before Runtime audits the generated
    ``world.read_*`` envelope.  The authoritative Router checks the envelope
    again at send time, but an unauthorized read must be rejected before the
    turn is suspended so Tracker can bind the protocol failure to the exact
    decision.  This check uses the same public partition table as Router and
    never relaxes World row-level access control.
    """

    validate(read_env)
    kind = ActionKind(str(read_env.action["kind"]))
    sender_role = actor_id.split(":", 1)[0]
    receiver_role = read_env.to.split(":", 1)[0]
    if not is_send_allowed(kind, sender_role, receiver_role):
        raise PartitionViolation(
            f"{sender_role!r} may not send {kind.value!r} to {receiver_role!r}"
        )


def finalize_rejected_tool_batch(frame: TurnFrame) -> None:
    """Preserve completed reads before a later read in the batch is rejected.

    The rejected request itself never reached the wire and therefore must not
    appear in Tracker.  Earlier results in the same model step may already be
    backed by audited World responses, so retain exactly those results as an
    incomplete tool step before the turn receives its failure terminal.
    """

    if frame.pending_results:
        frame.history.append(
            {
                "step": "tool_call",
                "results": list(frame.pending_results),
                "incomplete": True,
            }
        )
    frame.pending_results = []
    frame.pending_calls = []
    frame.pending_read_msg_id = None
    frame.business_pending_read_fingerprint = None
    frame.framework_authority_prerequisite = None


def reentrant_world_tool_allowed(tool: str, *, actor_id: str) -> bool:
    """Return whether ``actor_id`` may issue a mapped re-entrant World tool."""

    kind_value = _TOOL_TO_READ_KIND.get(tool)
    if kind_value is None:
        return False
    kind = ActionKind(kind_value)
    return is_send_allowed(
        kind,
        actor_id.split(":", 1)[0],
        "world",
    )


def _read_payload(call: "ToolCall") -> dict[str, Any]:
    a = _validated_tool_args(call)
    tool = call.tool
    if tool == "world.get_listing":
        return {"sku_id": a["sku_id"]}
    if tool == "world.search_catalog":
        return {k: a[k] for k in ("query", "filters", "limit") if k in a}
    if tool == "world.is_in_stock":
        return {"sku_id": a["sku_id"], "qty": a.get("qty", 1)}
    if tool == "world.get_merchant_reputation":
        return {"merchant_id": a["merchant_id"]}
    if tool == "world.get_order":
        return {"order_id": a["order_id"]}
    if tool == "world.get_friends":
        return {}
    if tool == "world.get_friend_reviews":
        return {k: a[k] for k in ("sku_id", "merchant_id") if k in a}
    if tool == "world.get_review_evidence":
        return {k: a[k] for k in ("sku_id", "merchant_id") if k in a}
    if tool == "world.get_evidence_record":
        return {
            key: a[key]
            for key in ("record_id", "version", "record_digest")
            if key in a
        }
    if tool == "world.get_mandate_revisions":
        return {"mandate_id": a["mandate_id"]}
    if tool == "world.get_listing_claim":
        return {"claim_id": a["claim_id"]}
    if tool == "world.list_listing_claims":
        return {"listing_id": a["listing_id"]}
    if tool == "world.get_after_sales_authority":
        return {"order_id": a["order_id"]}
    if tool == "world.get_protocol_event_authority":
        return {"event_id": a["event_id"]}
    raise ValueError(f"unsupported re-entrant world tool: {tool!r}")


def _validated_tool_args(call: "ToolCall") -> dict[str, Any]:
    """Validate mapped read arguments before a turn is suspended.

    Invalid arguments are reported through the existing recoverable tool-result
    path in :meth:`agents.base.Agent._advance_reads`.  World still validates
    every accepted envelope authoritatively; this early shape check only keeps
    model-caused argument errors inside the originating Agent decision.
    """

    args = call.args
    schemas: dict[str, tuple[frozenset[str], frozenset[str]]] = {
        "world.get_listing": (frozenset({"sku_id"}), frozenset({"sku_id"})),
        "world.search_catalog": (
            frozenset({"query", "filters", "limit"}),
            frozenset({"query"}),
        ),
        "world.is_in_stock": (
            frozenset({"sku_id", "qty"}),
            frozenset({"sku_id"}),
        ),
        "world.get_merchant_reputation": (
            frozenset({"merchant_id"}),
            frozenset({"merchant_id"}),
        ),
        "world.get_order": (frozenset({"order_id"}), frozenset({"order_id"})),
        "world.get_friends": (frozenset(), frozenset()),
        "world.get_friend_reviews": (
            frozenset({"sku_id", "merchant_id"}),
            frozenset(),
        ),
        "world.get_review_evidence": (
            frozenset({"sku_id", "merchant_id"}),
            frozenset(),
        ),
        "world.get_evidence_record": (
            frozenset({"record_id", "version", "record_digest"}),
            frozenset({"record_id"}),
        ),
        "world.get_mandate_revisions": (
            frozenset({"mandate_id"}),
            frozenset({"mandate_id"}),
        ),
        "world.get_listing_claim": (
            frozenset({"claim_id"}),
            frozenset({"claim_id"}),
        ),
        "world.list_listing_claims": (
            frozenset({"listing_id"}),
            frozenset({"listing_id"}),
        ),
        "world.get_after_sales_authority": (
            frozenset({"order_id"}),
            frozenset({"order_id"}),
        ),
        "world.get_protocol_event_authority": (
            frozenset({"event_id"}),
            frozenset({"event_id"}),
        ),
    }
    schema = schemas.get(call.tool)
    if schema is None:
        return args
    allowed, required = schema
    if not set(args).issubset(allowed) or not required.issubset(args):
        raise ValueError(f"invalid argument fields for {call.tool}")

    string_fields = {
        "sku_id",
        "merchant_id",
        "order_id",
        "record_id",
        "record_digest",
        "mandate_id",
        "claim_id",
        "listing_id",
        "event_id",
    }
    for field_name in string_fields.intersection(args):
        value = args[field_name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{call.tool} {field_name} must be a non-empty string"
            )
    if "query" in args and not isinstance(args["query"], str):
        raise ValueError("world.search_catalog query must be a string")
    if "filters" in args and not isinstance(args["filters"], dict):
        raise ValueError("world.search_catalog filters must be an object")
    for field_name in ("limit", "qty", "version"):
        if field_name not in args:
            continue
        value = args[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{call.tool} {field_name} must be a positive integer"
            )
    if "version" in args and "record_digest" in args:
        raise ValueError("world.get_evidence_record cannot combine version and digest")
    return args


def world_response_to_tool_result(call: "ToolCall", response: Envelope) -> dict[str, Any]:
    """Shape a ``world.response`` payload back into the tool_call result slot the
    in-process path would have produced — so step history (and the grounded
    evidence derived from it) is transport-identical.

    Two divergences this guards against: (1) over HTTP the payload arrives as a
    serialized dict, in-process as the live dataclass — re-hydrating both to the
    typed object means ``_format_tool_result`` projects them identically; (2) the
    synchronous ``WorldTools`` applies semantics the raw world read does not
    (``is_in_stock`` -> bool; an absent reputation row -> a zero-score default)."""
    from agents.base import _format_tool_result  # lazy: avoid an import cycle
    from protocol.tool_errors import world_tool_error_text

    payload = response.action.get("payload")
    tool_error = world_tool_error_text(payload)
    if tool_error is not None:
        result: Any = {"error": tool_error}
    elif call.tool == "world.is_in_stock":
        # read_availability returns {"sku_id", "in_stock"} — a bool, no counts.
        result = bool(payload.get("in_stock")) if isinstance(payload, dict) else False
    else:
        result = _format_tool_result(_rehydrate(call, payload))
    # source_msg_id: provenance of the grounded evidence (the world.response this
    # result came from). The synchronous path has no such envelope; absent there.
    return {"tool": call.tool, "args": call.args, "result": result,
            "source_msg_id": response.msg_id}


def _rehydrate(call: "ToolCall", payload: Any) -> Any:
    """Coerce a (possibly serialized) world.response payload back to its typed
    form, applying the same default the synchronous ``WorldTools`` would. Reuses
    ``world.service``'s coercion helpers as the single source of truth so the two
    transports cannot drift."""
    from world.service import _coerce_listing, _coerce_order, _coerce_reputation
    from world.types import AgentId, Listing, Order, ReputationScore

    tool = call.tool
    if tool == "world.get_merchant_reputation":
        if payload is None:  # WorldTools.get_merchant_reputation returns a default
            mid = AgentId(str(call.args.get("merchant_id", "")))
            return ReputationScore(merchant_id=mid, rolling_avg=0.0, n_settled=0, n_disputed=0)
        return payload if isinstance(payload, ReputationScore) else _coerce_reputation(payload)
    if payload is None:
        return None
    if tool == "world.get_listing":
        return payload if isinstance(payload, Listing) else _coerce_listing(payload)
    if tool == "world.search_catalog":
        items = payload if isinstance(payload, list) else [payload]
        return [it if isinstance(it, Listing) else _coerce_listing(it) for it in items]
    if tool == "world.get_order":
        return payload if isinstance(payload, Order) else _coerce_order(payload)
    if tool == "world.get_evidence_record":
        from protocol.evidence_records import coerce_evidence_record

        return coerce_evidence_record(payload)
    if tool == "world.get_after_sales_authority":
        from world.after_sales_authority_projection import (
            after_sales_authority_projection_from_wire,
        )

        return after_sales_authority_projection_from_wire(payload)
    if tool == "world.get_mandate_revisions":
        from protocol.evidence_records import coerce_mandate_revision

        items = payload if isinstance(payload, list) else [payload]
        return [coerce_mandate_revision(item) for item in items]
    if tool in {"world.get_listing_claim", "world.list_listing_claims"}:
        from protocol.listing_claims import coerce_listing_claim

        if tool == "world.get_listing_claim":
            return coerce_listing_claim(payload)
        items = payload if isinstance(payload, list) else [payload]
        return [coerce_listing_claim(item) for item in items]
    return payload  # friends (list[str]) / reviews — already JSON-shaped
