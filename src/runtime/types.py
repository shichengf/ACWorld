"""Passive data types for the runtime package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from protocol.envelope import Envelope


# --- Routing ----------------------------------------------------------

@dataclass(frozen=True)
class RouteEntry:
    """One row in the runtime's address table.

    Built at ``Runtime.register`` time; used by ``Runtime.send`` to deliver.
    """
    agent_id: str
    inbound_action_kinds: frozenset[str]
    outbound_action_kinds: frozenset[str]


# --- Audit log record -------------------------------------------------

@dataclass(frozen=True)
class AuditRecord:
    """One on-disk record per envelope. Written by ``AuditLog.append``.

    Fields are additive: new fields can be added; existing ones are never
    removed or renamed.
    """
    envelope: "Envelope"
    role: str | None = None
    skill: str | None = None
    outcome_tag: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    protocol_version: str = "1.0"


@dataclass(frozen=True)
class AuditRecordView:
    """Indexed, read-only metadata view over one :class:`AuditRecord`.

    ``sequence`` is the record's zero-based position in the global committed
    audit stream.  Partition queries return these views rather than copying
    records into per-market files, so callers can merge or page partitions
    without losing the canonical event order.

    Identifier tuples are derived from the envelope and its payload.  They are
    sorted and de-duplicated to keep the view deterministic across processes.
    The underlying on-disk ``audit.jsonl`` schema is deliberately unchanged.
    """

    sequence: int
    record: AuditRecord
    market_ids: tuple[str, ...] = ()
    actor_ids: tuple[str, ...] = ()
    transaction_ids: tuple[str, ...] = ()

    @property
    def envelope(self) -> "Envelope":
        """The original on-wire envelope for convenience."""

        return self.record.envelope


# --- Inference budget -------------------------------------------------

@dataclass(frozen=True)
class InferenceBudget:
    """Per-agent budget enforced by the inference channel at register time."""
    max_tokens_per_turn: int
    max_latency_ms_per_turn: int
    max_total_tokens: int


# --- Security event (blocked privacy leak) ----------------------------

@dataclass(frozen=True)
class SecurityEvent:
    """A SANITIZED record of a blocked privacy-leak attempt.

    Written to the security sidecar (``<stem>.security.jsonl``), NEVER to
    ``audit.jsonl`` — the rejected envelope did not reach the wire, so the
    audit/replay stream must not contain it (the leak attempt would otherwise
    be replayed). Deliberately carries NO raw secret value, NO copied payload,
    and NO agent memory — only enough metadata to audit the violation. ``ts`` is
    the offending envelope's own timestamp (no wall clock); ``run_id`` is stamped
    by the writer when blank."""
    msg_id: str
    sender_id: str
    action_kind: str
    secret_owner: str
    secret_name: str
    field_path: str
    reason: str
    recipient_id: str = ""   # sanitized: improves boundary attribution
    owner_role: str = ""     # side prefix of the secret owner
    ts: str = ""
    run_id: str = ""


# --- Reasoning trace (item 5) -----------------------------------------
#
# The trace is the benchmark-data lane: per-AGENT-TURN cognition (the step
# stream, the candidates weighed, the final choice, and the private inputs the
# decision used). It is written ONLY to the trace sidecar by ``AuditLog.append_
# trace`` — never to ``audit.jsonl`` — so (a) the on-wire log stays byte-stable
# and replay-safe, and (b) private content (max_budget, floor_price) is
# physically separated from anything that crosses the wire. It joins back to
# the on-wire log via ``(run_id, turn, emitted_msg_id == envelope.msg_id)``.

@dataclass(frozen=True)
class TraceStep:
    """One framework-owned step of an agent turn.

    ``data`` is deliberately open ended.  Besides the legacy
    ``load_skill``/``memory_update``/``tool_call`` entries, it can carry a
    sanitized business intent and the VCP route compiled from it. Raw
    provider prompts and responses do not belong in this sidecar entry.
    """
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceCandidate:
    """One option the agent weighed during a decision turn.

    item 5 fills the skeleton — id / price / affordability / eligibility — from
    the existing memory writes (``budget_audit``). ``rejected_at_stage`` +
    ``reject_reason`` (per-stage) and ``signals`` (friend/reputation, item 7)
    are reserved slots so item 8 / item 7 populate them with no schema
    migration: they restructure the decision skill to emit these directly.
    """
    offer_id: str | None = None
    sku_id: str | None = None
    unit_price: int | None = None
    total_cents: int | None = None
    eta_days: int | None = None
    qty: int | None = None
    affordable: bool | None = None
    negotiable: bool | None = None
    eligible: bool | None = None
    rejected_at_stage: str | None = None      # which filter stage dropped it (item 8)
    reject_reason: str | None = None          # free-text per-stage reason (item 8)
    rank: int | None = None
    signals: dict[str, Any] | None = None     # friend / reputation signal (item 7)
    # The attributes actually FETCHED for this sku by a framework-executed
    # grounding read in this decision. ``grounded_evidence_id`` binds the exact
    # tool-result slot to the stitched decision in both transports.  HTTP/VCP
    # additionally supplies ``grounded_source_msg_id`` so the scorer can match
    # the slot to its audited world.response.  The synchronous path has no
    # response envelope, but retains the same decision-bound public evidence.
    # None of these fields are populated from LLM-authored rationale text.
    grounded_attributes: dict[str, Any] | None = None
    grounded_evidence_id: str | None = None
    grounded_source_msg_id: str | None = None


@dataclass(frozen=True)
class TraceChoice:
    """The turn's final decision and why."""
    decision: str                             # accept|counter|reject|negotiate|search|settle|no_reply|other
    offer_id: str | None = None
    price: int | None = None
    rationale: str | None = None              # free-text deciding factor (item 8 fills)


@dataclass(frozen=True)
class TraceRecord:
    """One agent-turn of cognition. Sidecar-only; never on the wire log."""
    run_id: str
    turn: int
    agent_id: str
    inbound_msg_id: str | None
    emitted_msg_id: str | None
    # Normal: emit_envelope/no_reply.  Incomplete: forced_flush.  A failed,
    # scoreable Agent decision uses one of the typed terminals declared by
    # runtime.turn_failure and is bound exactly by termination.json.
    terminal: str
    skill: str | None
    steps: tuple[TraceStep, ...] = ()
    considered: tuple[TraceCandidate, ...] = ()
    chosen: TraceChoice | None = None
    private_context: dict[str, Any] | None = None
    protocol_version: str = "1.0"
    # R1: a decision can span several dispatcher rounds (emit world.read ->
    # suspend -> resume). ``decision_id`` is the stitch key tying those rounds
    # into this one record. ``forced_flush``/``incomplete`` mark a decision
    # force-finalized mid-grounding (e.g. at teardown) — semantically NOT a
    # completed no_reply, so a scorer must not read it as "considered N then
    # chose not to buy". Same family as the trace_integrity flag.
    decision_id: str | None = None
    forced_flush: bool = False
    incomplete: bool = False
    # Appended after every legacy field so positional construction remains
    # backward compatible.  Values are the content digests of every
    # selector-approved skill loaded for this decision, in first-load order
    # with duplicates removed.  ``skill`` remains the legacy projection of the
    # last loaded skill name.
    skill_digests: tuple[str, ...] = ()
