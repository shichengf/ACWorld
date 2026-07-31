"""Strict causal verification for the agent Tracker sidecar.

The Tracker is evidence, not model-authored narration.  This verifier binds an
evaluated actor's completed decision records to the exact Runtime audit stream
and to the hash-covered episode manifest.  In particular, a trace row cannot
invent its inbound message, emitted action, or a World tool result.

The verifier is task agnostic.  Formal task preflight calls it for the ideal
policy, while other environments may reuse it for any actor whose trace is
expected to be complete.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agents.business_decision import (
    MODEL_BUSINESS_CHOICE_SCHEMA_V1,
    canonical_business_value_digest,
)
from agents.turn import (
    PUBLIC_WORLD_READ_ACTION_KINDS,
    ToolCall,
    tool_call_to_read_envelope,
    world_response_to_tool_result,
)
from protocol.envelope import Envelope, from_json, to_json
from runtime.trace import normalized_decision_for_action_kind
from runtime.turn_failure import (
    PROTOCOL_ERROR_TERMINAL,
    RESOURCE_LIMIT_TERMINAL,
    SECURITY_ERROR_TERMINAL,
    SCOREABLE_FAILURE_TERMINALS,
)


TRACKER_TRACE_ARTIFACT = "audit.trace.jsonl"
_FAILURE_STOP_BY_TERMINAL = {
    PROTOCOL_ERROR_TERMINAL: "model_protocol_error",
    RESOURCE_LIMIT_TERMINAL: RESOURCE_LIMIT_TERMINAL,
    SECURITY_ERROR_TERMINAL: "security_guard",
}

_TRACKER_ROW_FIELDS = frozenset(
    {
        "run_id",
        "turn",
        "agent_id",
        "inbound_msg_id",
        "emitted_msg_id",
        "terminal",
        "skill",
        "steps",
        "considered",
        "chosen",
        "private_context",
        "protocol_version",
        "decision_id",
        "forced_flush",
        "incomplete",
        "skill_digests",
    }
)

_PUBLIC_TASK_COMMON_FIELDS = frozenset(
    {
        "phase_id",
        "phase_index",
        "decision_id",
        "execution_contract_sha256",
        "semantic_authority_revision",
        "authority_source_msg_ids",
    }
)
_PUBLIC_TASK_BUSINESS_FIELDS = _PUBLIC_TASK_COMMON_FIELDS | frozenset(
    {
        "interface",
        "llm_decision_surface_sha256",
    }
)
_FRAMEWORK_AUTHORITY_PREREQUISITE_FIELDS = frozenset(
    {
        "interface",
        "authority_kind",
        "source_msg_ids",
        "authority_sha256",
    }
)

_TRACKER_STEP_FIELDS = {
    "public_task_execution": _PUBLIC_TASK_BUSINESS_FIELDS,
    "load_skill": frozenset({"name", "result", "digest", "source"}),
    "memory_update": frozenset({"writes", "source"}),
    "tool_call": frozenset({"results", "interface", "incomplete"}),
    "semantic_transport_retry": frozenset(
        {
            "interface",
            "error_code",
            "retry_index",
            "same_decision",
        }
    ),
    "framework_protocol_continuation": frozenset(
        {
            "interface",
            "trigger_action_kind",
            "action_kind",
            "destination",
            "authority",
            "accepted_envelope_sha256",
            "certificate_id_sha256",
        }
    ),
    "framework_protocol_receipt": frozenset(
        {
            "interface",
            "trigger_action_kind",
            "settled_envelope_sha256",
            "terminal",
        }
    ),
    "framework_public_task_continuation": frozenset(
        {
            "interface",
            "action_kind",
            "destination",
            "payload_chars",
            "payload_sha256",
        }
    ),
    "framework_public_task_terminal": frozenset(
        {
            "interface",
            "trigger_action_kind",
            "terminal",
        }
    ),
    "framework_authority_prerequisite": _FRAMEWORK_AUTHORITY_PREREQUISITE_FIELDS,
    "semantic_read_request": frozenset(
        {
            "interface",
            "function",
            "business_intent",
            "compiler_validated",
            "arguments_chars",
            "arguments_sha256",
            "model_business_choice",
            "agent_argument_binding",
            "response_chars",
            "response_sha256",
            "compiled_world_read",
        }
    ),
    "semantic_terminal_deferred": frozenset(
        {
            "interface",
            "calls",
            "instruction",
        }
    ),
    "semantic_redundant_reads_elided": frozenset(
        {
            "interface",
            "read_batch_sha256",
            "elided_read_count",
            "calls",
        }
    ),
    "semantic_finish": frozenset(
        {
            "interface",
            "function",
            "arguments_chars",
            "arguments_sha256",
        }
    ),
    "semantic_validation_error": frozenset(
        {
            "interface",
            "error",
            "error_code",
            "calls",
            "instruction",
        }
    ),
    "semantic_response_error": frozenset(
        {
            "interface",
            "error",
            "error_code",
            "response_chars",
            "response_sha256",
            "instruction",
        }
    ),
    "semantic_action": frozenset(
        {
            "interface",
            "function",
            "business_intent",
            "compiler_validated",
            "observation_class",
            "expects_platform_exchange",
            "arguments_chars",
            "arguments_sha256",
            "model_business_choice",
            "agent_argument_binding",
            "response_chars",
            "response_sha256",
            "compiled_vcp",
        }
    ),
}

_FORBIDDEN_TRACKER_STEP_FIELDS = frozenset(
    {
        "raw_prompt",
        "raw_response",
        "assistant_content",
        "semantic_messages",
        "provider_body",
        "reasoning",
        "reasoning_details",
    }
)


class TrackerEvidenceError(ValueError):
    """Raised when Tracker evidence cannot be exactly joined to the audit."""


@dataclass(frozen=True)
class TrackerEvidenceVerdict:
    """Structured verdict suitable for preflight and paid-run compatibility."""

    evaluated_actor_id: str
    strict_ideal: bool
    verified: bool
    issues: tuple[str, ...]
    record_count: int
    complete_record_count: int
    emitted_envelope_count: int
    world_tool_result_count: int
    trace_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated_actor_id": self.evaluated_actor_id,
            "strict_ideal": self.strict_ideal,
            "verified": self.verified,
            "issues": list(self.issues),
            "record_count": self.record_count,
            "complete_record_count": self.complete_record_count,
            "emitted_envelope_count": self.emitted_envelope_count,
            "world_tool_result_count": self.world_tool_result_count,
            "trace_sha256": self.trace_sha256,
        }


@dataclass(frozen=True)
class VerifiedModelBusinessChoice:
    """A safe model business choice exactly joined to one audited envelope."""

    emitted_msg_id: str
    inbound_msg_id: str
    intent: str
    arguments: Mapping[str, Any]
    model_arguments_sha256: str
    resolved_arguments_sha256: str
    argument_binding_sha256: str
    destination: str
    action_kind: str
    payload_chars: int
    payload_sha256: str
    expects_platform_exchange: bool
    response_chars: int
    response_sha256: str


@dataclass(frozen=True)
class VerifiedModelWorldRead:
    """One model-requested World read exactly joined to audited execution."""

    inbound_msg_id: str
    intent: str
    arguments: Mapping[str, Any]
    model_arguments_sha256: str
    resolved_arguments_sha256: str
    argument_binding_sha256: str
    tool: str
    args: Mapping[str, Any]
    args_chars: int
    args_sha256: str
    source_msg_id: str | None
    response_chars: int
    response_sha256: str


@dataclass(frozen=True)
class AllActiveActorTrackerVerdict:
    """Tracker closure across every declared actor that received a real turn."""

    evaluated_actor_id: str
    active_actor_ids: tuple[str, ...]
    actor_verdicts: tuple[TrackerEvidenceVerdict, ...]
    issues: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return not self.issues and all(row.verified for row in self.actor_verdicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated_actor_id": self.evaluated_actor_id,
            "evaluated_actor_strict": (
                next(
                    (
                        row.strict_ideal
                        for row in self.actor_verdicts
                        if row.evaluated_actor_id == self.evaluated_actor_id
                    ),
                    None,
                )
            ),
            "strict_ideal": (
                next(
                    (
                        row.strict_ideal
                        for row in self.actor_verdicts
                        if row.evaluated_actor_id == self.evaluated_actor_id
                    ),
                    None,
                )
            ),
            "verified": self.verified,
            "active_actor_ids": list(self.active_actor_ids),
            "verified_actor_count": sum(row.verified for row in self.actor_verdicts),
            "record_count": sum(row.record_count for row in self.actor_verdicts),
            "emitted_envelope_count": sum(
                row.emitted_envelope_count for row in self.actor_verdicts
            ),
            "world_tool_result_count": sum(
                row.world_tool_result_count for row in self.actor_verdicts
            ),
            "issues": list(self.issues),
            "actors": [row.to_dict() for row in self.actor_verdicts],
        }


def verify_all_active_actor_tracker_evidence(
    evidence: object,
    *,
    declared_actor_ids: tuple[str, ...] | list[str] | set[str] | frozenset[str],
    evaluated_actor_id: str,
    evaluated_actor_strict: bool,
) -> AllActiveActorTrackerVerdict:
    """Verify Tracker coverage for all declared actors that received a turn.

    Active actors are derived from both the sidecar and audited non-World
    decision inbounds.  Taking their union makes a missing Tracker row visible
    instead of defining activity from the evidence whose completeness is under
    test. Counterparts are always strict. Only the evaluated actor may use
    scoreable non-ideal termination semantics.
    """

    declared = frozenset(declared_actor_ids)
    issues: list[str] = []
    if (
        not declared
        or any(not isinstance(actor_id, str) or not actor_id for actor_id in declared)
        or evaluated_actor_id not in declared
        or not isinstance(evaluated_actor_strict, bool)
    ):
        issues.append("declared/evaluated Tracker actor identity is invalid")
    trace_rows = getattr(evidence, "trace_rows", ())
    envelopes = getattr(evidence, "envelopes", ())
    traced = {
        str(row.get("agent_id"))
        for row in trace_rows
        if isinstance(row, Mapping) and isinstance(row.get("agent_id"), str) and row.get("agent_id")
    }
    audited_inbounds = {
        str(envelope.get("to"))
        for envelope in envelopes
        if isinstance(envelope, Mapping)
        and envelope.get("to") in declared
        and isinstance(envelope.get("action"), Mapping)
        and envelope["action"].get("kind") != "world.response"
    }
    active_actor_ids = tuple(sorted(traced | audited_inbounds))
    undeclared = set(active_actor_ids) - declared
    if undeclared:
        issues.append("Tracker contains undeclared active actors: " + ", ".join(sorted(undeclared)))
    if not active_actor_ids:
        issues.append("CommerceWorld episode has no active Tracker actor")
    if evaluated_actor_id not in active_actor_ids:
        issues.append("evaluated actor never entered the Tracker decision loop")
    verdicts = tuple(
        verify_tracker_evidence(
            evidence,
            evaluated_actor_id=actor_id,
            strict_ideal=(evaluated_actor_strict if actor_id == evaluated_actor_id else True),
        )
        for actor_id in active_actor_ids
        if actor_id in declared
    )
    issues.extend(
        f"{verdict.evaluated_actor_id}: {issue}" for verdict in verdicts for issue in verdict.issues
    )
    return AllActiveActorTrackerVerdict(
        evaluated_actor_id=evaluated_actor_id,
        active_actor_ids=active_actor_ids,
        actor_verdicts=verdicts,
        issues=tuple(issues),
    )


def tracker_row_has_usable_completed_steps(row: Mapping[str, Any]) -> bool:
    """Whether a scorer may credit completed framework steps in one row.

    A scoreable Agent guard finalizes the decision as ``incomplete`` because no
    envelope crossed Runtime. Reads and memory operations completed before the
    guard are nevertheless real, hash-covered Tracker evidence and should earn
    partial credit. A forced flush has no such terminal guarantee and remains
    excluded. Legacy complete rows without explicit flags retain their prior
    behavior.
    """

    if row.get("forced_flush") is True:
        return False
    if row.get("incomplete") is not True:
        return True
    return row.get("forced_flush") is False and row.get("terminal") in SCOREABLE_FAILURE_TERMINALS


def verify_tracker_evidence(
    evidence: object,
    *,
    evaluated_actor_id: str,
    strict_ideal: bool = True,
) -> TrackerEvidenceVerdict:
    """Verify one evaluated actor's complete Tracker record.

    ``evidence`` is intentionally structural rather than tied to a benchmark
    family.  It must expose the standard ``RuntimeEvidenceBundleV2`` fields:
    ``episode_dir``, ``evidence_manifest``, ``audit_rows``, ``envelopes``, and
    ``trace_rows``.
    """

    try:
        return _verify_tracker_evidence_or_raise(
            evidence,
            evaluated_actor_id=evaluated_actor_id,
            strict_ideal=strict_ideal,
        )
    except Exception as exc:  # fail closed at the public evidence boundary
        try:
            summary = _trace_summary(evidence, evaluated_actor_id=evaluated_actor_id)
        except Exception:
            summary = {
                "record_count": 0,
                "complete_record_count": 0,
                "emitted_envelope_count": 0,
                "world_tool_result_count": 0,
                "trace_sha256": None,
            }
        issue = (
            str(exc) if isinstance(exc, TrackerEvidenceError) else (f"{type(exc).__name__}: {exc}")
        )
        return TrackerEvidenceVerdict(
            evaluated_actor_id=evaluated_actor_id,
            strict_ideal=strict_ideal,
            verified=False,
            issues=(issue,),
            **summary,
        )


def verified_tracker_emitted_msg_ids(
    evidence: object,
    *,
    evaluated_actor_id: str,
    strict_ideal: bool = False,
) -> tuple[str, ...]:
    """Return only hash-covered envelopes emitted by the evaluated actor.

    Actor-shaped scenario setup messages may legitimately appear in the
    Runtime audit.  They are not model actions.  This helper first verifies the
    complete Tracker sidecar against the immutable audit and its evidence
    manifest, then extracts the exact ``emitted_msg_id`` values from completed
    evaluated-actor decisions.  Callers can therefore join protocol credit to
    actual agent output instead of trusting an envelope's ``from`` field alone.
    """

    verdict = verify_tracker_evidence(
        evidence,
        evaluated_actor_id=evaluated_actor_id,
        strict_ideal=strict_ideal,
    )
    if not verdict.verified:
        detail = "; ".join(verdict.issues[:3]) or "unknown Tracker defect"
        raise TrackerEvidenceError("evaluated actor Tracker evidence is invalid: " + detail)

    rows = getattr(evidence, "trace_rows", ())
    emitted: list[str] = []
    for row in rows:
        if (
            isinstance(row, Mapping)
            and row.get("agent_id") == evaluated_actor_id
            and row.get("terminal") == "emit_envelope"
        ):
            msg_id = row.get("emitted_msg_id")
            if not isinstance(msg_id, str) or not msg_id:
                # The full verifier above makes this unreachable.  Keep the
                # extraction fail closed if its invariants ever change.
                raise TrackerEvidenceError("verified Tracker emit record has no message id")
            emitted.append(msg_id)
    if len(emitted) != len(set(emitted)):
        raise TrackerEvidenceError("verified Tracker emitted ids are ambiguous")
    return tuple(emitted)


def verified_model_business_choices(
    evidence: object,
    *,
    evaluated_actor_id: str,
    strict_ideal: bool = False,
) -> tuple[VerifiedModelBusinessChoice, ...]:
    """Extract only model-authored choices exact-joined to emitted messages.

    Deterministic framework continuations are intentionally absent.  The full
    Tracker verifier runs first, so any ambiguous claim, payload drift, or
    malformed provenance record fails closed before a scorer sees a choice.
    """

    verdict = verify_tracker_evidence(
        evidence,
        evaluated_actor_id=evaluated_actor_id,
        strict_ideal=strict_ideal,
    )
    if not verdict.verified:
        detail = "; ".join(verdict.issues[:3]) or "unknown Tracker defect"
        raise TrackerEvidenceError("evaluated actor Tracker evidence is invalid: " + detail)
    rows = getattr(evidence, "trace_rows", ())
    output: list[VerifiedModelBusinessChoice] = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or row.get("agent_id") != evaluated_actor_id
            or row.get("terminal") != "emit_envelope"
        ):
            continue
        semantic = [
            step.get("data")
            for step in row.get("steps", ())
            if isinstance(step, Mapping) and step.get("kind") == "semantic_action"
        ]
        if not semantic:
            continue
        if len(semantic) != 1 or not isinstance(semantic[0], Mapping):
            raise TrackerEvidenceError("verified Tracker semantic-action extraction is ambiguous")
        data = semantic[0]
        choice = data.get("model_business_choice")
        binding = data.get("agent_argument_binding")
        compiled = data.get("compiled_vcp")
        emitted_msg_id = row.get("emitted_msg_id")
        inbound_msg_id = row.get("inbound_msg_id")
        if not all(
            (
                isinstance(choice, Mapping),
                isinstance(binding, Mapping),
                isinstance(compiled, Mapping),
                isinstance(emitted_msg_id, str),
                isinstance(inbound_msg_id, str),
            )
        ):
            raise TrackerEvidenceError("verified Tracker model-choice extraction is incomplete")
        output.append(
            VerifiedModelBusinessChoice(
                emitted_msg_id=emitted_msg_id,
                inbound_msg_id=inbound_msg_id,
                intent=str(choice["intent"]),
                arguments=copy.deepcopy(dict(choice["arguments"])),
                model_arguments_sha256=str(choice["arguments_sha256"]),
                resolved_arguments_sha256=str(binding["resolved_arguments_sha256"]),
                argument_binding_sha256=str(binding["binding_sha256"]),
                destination=str(compiled["to"]),
                action_kind=str(compiled["action_kind"]),
                payload_chars=int(compiled["payload_chars"]),
                payload_sha256=str(compiled["payload_sha256"]),
                expects_platform_exchange=bool(data["expects_platform_exchange"]),
                response_chars=int(data["response_chars"]),
                response_sha256=str(data["response_sha256"]),
            )
        )
    if len({row.emitted_msg_id for row in output}) != len(output):
        raise TrackerEvidenceError("verified model business choices are ambiguous")
    return tuple(output)


def verified_model_world_reads(
    evidence: object,
    *,
    evaluated_actor_id: str,
    strict_ideal: bool = False,
) -> tuple[VerifiedModelWorldRead, ...]:
    """Extract completed model-requested reads exact-joined to World evidence.

    Framework authority prerequisites are deliberately absent.  The generic
    Tracker verifier first proves that every typed compiler claim matches the
    recorded ToolCall.  For re-entrant execution it additionally proves that
    the ToolCall exactly reconstructs an audited World request/result pair;
    direct in-process reads have ``source_msg_id=None`` and remain bound to the
    recorded local result.  The returned ``arguments`` are the sanitized
    provider-authored public choice; ``args`` are the actual Agent-owned World
    arguments.  A family scorer can therefore validate its explicit public
    reference relation without trusting either side in isolation.
    """

    verdict = verify_tracker_evidence(
        evidence,
        evaluated_actor_id=evaluated_actor_id,
        strict_ideal=strict_ideal,
    )
    if not verdict.verified:
        detail = "; ".join(verdict.issues[:3]) or "unknown Tracker defect"
        raise TrackerEvidenceError("evaluated actor Tracker evidence is invalid: " + detail)

    output: list[VerifiedModelWorldRead] = []
    rows = getattr(evidence, "trace_rows", ())
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or row.get("agent_id") != evaluated_actor_id
            or not tracker_row_has_usable_completed_steps(row)
        ):
            continue
        pending: Mapping[str, Any] | None = None
        for step in row.get("steps", ()):
            if not isinstance(step, Mapping):  # unreachable after verification
                continue
            kind = step.get("kind")
            data = step.get("data")
            if kind == "semantic_read_request" and isinstance(data, Mapping):
                pending = data
                continue
            if kind != "tool_call" or pending is None or not isinstance(data, Mapping):
                continue
            results = data.get("results")
            result = results[0] if isinstance(results, list) and len(results) == 1 else None
            if not isinstance(result, Mapping):  # unreachable after verification
                raise TrackerEvidenceError("verified model World-read extraction is ambiguous")
            source_msg_id = result.get("source_msg_id")
            if result.get("pending_request_msg_id") is not None:
                # Pending forced-flush reads were already excluded above.
                pending = None
                continue
            if source_msg_id is not None and (
                not isinstance(source_msg_id, str) or not source_msg_id
            ):
                raise TrackerEvidenceError("verified model World read has a malformed source id")
            choice = pending.get("model_business_choice")
            binding = pending.get("agent_argument_binding")
            compiled = pending.get("compiled_world_read")
            args = result.get("args")
            inbound_msg_id = row.get("inbound_msg_id")
            if not all(
                (
                    isinstance(choice, Mapping),
                    isinstance(binding, Mapping),
                    isinstance(compiled, Mapping),
                    isinstance(args, Mapping),
                    isinstance(inbound_msg_id, str),
                )
            ):
                raise TrackerEvidenceError("verified model World-read extraction is incomplete")
            output.append(
                VerifiedModelWorldRead(
                    inbound_msg_id=inbound_msg_id,
                    intent=str(choice["intent"]),
                    arguments=copy.deepcopy(dict(choice["arguments"])),
                    model_arguments_sha256=str(choice["arguments_sha256"]),
                    resolved_arguments_sha256=str(binding["resolved_arguments_sha256"]),
                    argument_binding_sha256=str(binding["binding_sha256"]),
                    tool=str(compiled["tool"]),
                    args=copy.deepcopy(dict(args)),
                    args_chars=int(compiled["args_chars"]),
                    args_sha256=str(compiled["args_sha256"]),
                    source_msg_id=source_msg_id,
                    response_chars=int(pending["response_chars"]),
                    response_sha256=str(pending["response_sha256"]),
                )
            )
            pending = None
    sourced = [row.source_msg_id for row in output if row.source_msg_id is not None]
    if len(set(sourced)) != len(sourced):
        raise TrackerEvidenceError("verified model World reads are ambiguous")
    return tuple(output)


def verified_model_local_finish(
    evidence: object,
    *,
    evaluated_actor_id: str,
    strict_ideal: bool = False,
) -> bool:
    """Prove one model-owned local finish with no emitted commerce action.

    A valid ``finish`` is a typed model business decision, but by definition it
    produces no Runtime-to-Platform envelope.  This helper is intentionally
    narrower than a generic ``no_reply`` check: the hash-covered Tracker must
    contain exactly one evaluated-actor row, one business-decision execution
    binding, and one well-formed ``semantic_finish`` terminal.  Deterministic
    continuations, semantic actions, incomplete terminals, and framework-only
    no-reply rows do not satisfy the proof.
    """

    verdict = verify_tracker_evidence(
        evidence,
        evaluated_actor_id=evaluated_actor_id,
        strict_ideal=strict_ideal,
    )
    if not verdict.verified:
        detail = "; ".join(verdict.issues[:3]) or "unknown Tracker defect"
        raise TrackerEvidenceError("evaluated actor Tracker evidence is invalid: " + detail)
    actor_rows = [
        row
        for row in getattr(evidence, "trace_rows", ())
        if isinstance(row, Mapping) and row.get("agent_id") == evaluated_actor_id
    ]
    if len(actor_rows) != 1:
        return False
    row = actor_rows[0]
    if (
        row.get("terminal") != "no_reply"
        or row.get("emitted_msg_id") is not None
        or row.get("forced_flush") is not False
        or row.get("incomplete") is not False
        or row.get("chosen")
        != {
            "decision": "no_reply",
            "offer_id": None,
            "price": None,
            "rationale": None,
        }
    ):
        return False
    steps = row.get("steps")
    if not isinstance(steps, list):  # already rejected by the verifier
        return False
    execution_steps = [
        step.get("data")
        for step in steps
        if isinstance(step, Mapping) and step.get("kind") == "public_task_execution"
    ]
    semantic_steps = [
        step
        for step in steps
        if isinstance(step, Mapping)
        and isinstance(step.get("kind"), str)
        and str(step["kind"]).startswith("semantic_")
    ]
    if len(execution_steps) != 1 or len(semantic_steps) != 1:
        return False
    execution = execution_steps[0]
    finish = semantic_steps[0]
    finish_data = finish.get("data")
    if (
        not isinstance(execution, Mapping)
        or execution.get("interface") != "llm_business_decision_v1"
        or execution.get("decision_id") != row.get("decision_id")
        or finish.get("kind") != "semantic_finish"
        or not isinstance(finish_data, Mapping)
        or set(finish_data) != _TRACKER_STEP_FIELDS["semantic_finish"]
        or finish_data.get("interface") != "business_decision"
        or finish_data.get("function") != "agent_finish_turn"
        or not _valid_length_digest_pair(
            finish_data.get("arguments_chars"),
            finish_data.get("arguments_sha256"),
        )
        or steps[-1] is not finish
    ):
        return False
    forbidden_kinds = {
        "semantic_action",
        "framework_protocol_continuation",
        "framework_public_task_continuation",
    }
    return not any(
        isinstance(step, Mapping) and step.get("kind") in forbidden_kinds for step in steps
    )


def _verify_tracker_evidence_or_raise(
    evidence: object,
    *,
    evaluated_actor_id: str,
    strict_ideal: bool,
) -> TrackerEvidenceVerdict:
    if not isinstance(evaluated_actor_id, str) or not evaluated_actor_id:
        raise TrackerEvidenceError("evaluated actor id must be non-empty")
    if not isinstance(strict_ideal, bool):
        raise TrackerEvidenceError("strict_ideal must be boolean")

    root = Path(getattr(evidence, "episode_dir"))
    manifest = getattr(evidence, "evidence_manifest", None)
    if not isinstance(manifest, Mapping):
        raise TrackerEvidenceError("Tracker verification requires an episode manifest")
    trace_rows = _verified_trace_artifact(root, manifest)
    for ordinal, row in enumerate(trace_rows, start=1):
        _validate_tracker_row_schema(row, ordinal=ordinal)
    termination = _verified_termination_artifact(root, manifest)
    loaded_trace_rows = getattr(evidence, "trace_rows", None)
    if not isinstance(loaded_trace_rows, tuple) or tuple(trace_rows) != loaded_trace_rows:
        raise TrackerEvidenceError("Tracker rows differ from the hash-covered trace artifact")

    audit_rows = getattr(evidence, "audit_rows", None)
    envelopes = getattr(evidence, "envelopes", None)
    if not isinstance(audit_rows, tuple) or not isinstance(envelopes, tuple):
        raise TrackerEvidenceError("Runtime evidence has no immutable audit streams")
    if len(audit_rows) != len(envelopes):
        raise TrackerEvidenceError("audit rows and decoded envelopes have different lengths")

    audited_by_id: dict[str, Mapping[str, Any]] = {}
    audit_record_by_id: dict[str, Mapping[str, Any]] = {}
    for ordinal, (record, envelope) in enumerate(zip(audit_rows, envelopes, strict=True), start=1):
        if not isinstance(record, Mapping) or not isinstance(envelope, Mapping):
            raise TrackerEvidenceError(f"audit row {ordinal} is not an object")
        msg_id = envelope.get("msg_id")
        if not isinstance(msg_id, str) or not msg_id:
            raise TrackerEvidenceError(f"audit row {ordinal} has no message id")
        if msg_id in audited_by_id:
            raise TrackerEvidenceError(f"audit message id is ambiguous: {msg_id}")
        _action_kind(envelope, label=f"audit row {ordinal}")
        audited_by_id[msg_id] = envelope
        audit_record_by_id[msg_id] = record

    scenario_id = manifest.get("scenario_id") if isinstance(manifest, Mapping) else None
    actor_rows = [row for row in trace_rows if row.get("agent_id") == evaluated_actor_id]
    if not actor_rows:
        raise TrackerEvidenceError(f"evaluated actor {evaluated_actor_id!r} has no Tracker record")

    termination_binding: Mapping[str, Any] | None = None
    if termination is not None:
        binding = termination.get("tracker_binding")
        classification = termination.get("classification")
        if (
            termination.get("schema_version") != "cwe.episode-termination.v1"
            or termination.get("status") != "aborted"
            or termination.get("phase") != "runtime"
            or termination.get("scoreable") is not True
            or not isinstance(classification, str)
            or termination.get("stop_reason") != classification
            or not isinstance(binding, Mapping)
            or binding.get("artifact") != TRACKER_TRACE_ARTIFACT
        ):
            raise TrackerEvidenceError(
                "episode termination is not a scoreable Tracker-bound Agent failure"
            )
        # The episode has one terminal failure, but a formal preflight verifies
        # every actor that actually took a turn.  A failure bound to actor A is
        # irrelevant to actor B's row semantics.  It is nevertheless global
        # episode evidence, so validate its exact hash-covered Tracker row
        # before continuing with B instead of either rejecting B or ignoring a
        # forged foreign binding.
        if binding.get("agent_id") == evaluated_actor_id:
            termination_binding = binding
        else:
            _verify_foreign_termination_binding(
                trace_rows,
                binding=binding,
                scenario_id=scenario_id,
                classification=classification,
            )

    tracked_inbound_ids: set[str] = set()
    tracked_emitted_ids: set[str] = set()
    claimed_world_response_ids: set[str] = set()
    claimed_world_request_ids: set[str] = set()
    tool_result_by_source: dict[str, Mapping[str, Any]] = {}
    failure_record_count = 0

    for ordinal, row in enumerate(actor_rows, start=1):
        label = f"Tracker row {ordinal} for {evaluated_actor_id}"
        if row.get("run_id") != scenario_id:
            raise TrackerEvidenceError(f"{label} has the wrong run id")
        turn = row.get("turn")
        if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
            raise TrackerEvidenceError(f"{label} has an invalid turn")
        forced_flag = row.get("forced_flush")
        incomplete_flag = row.get("incomplete")
        if not isinstance(forced_flag, bool) or not isinstance(incomplete_flag, bool):
            raise TrackerEvidenceError(f"{label} has invalid completion flags")
        row_terminal = row.get("terminal")
        if strict_ideal and row_terminal in ({"forced_flush"} | SCOREABLE_FAILURE_TERMINALS):
            raise TrackerEvidenceError(f"{label} is forced or incomplete")

        inbound_id = row.get("inbound_msg_id")
        if not isinstance(inbound_id, str) or not inbound_id:
            raise TrackerEvidenceError(f"{label} has no inbound message id")
        if inbound_id in tracked_inbound_ids:
            raise TrackerEvidenceError(f"Tracker inbound is claimed twice: {inbound_id}")
        inbound = audited_by_id.get(inbound_id)
        if inbound is None:
            raise TrackerEvidenceError(f"Tracker inbound is absent from audit: {inbound_id}")
        if inbound.get("to") != evaluated_actor_id:
            raise TrackerEvidenceError(
                f"Tracker inbound is addressed to another actor: {inbound_id}"
            )
        if _action_kind(inbound, label=f"inbound {inbound_id}") == "world.response":
            raise TrackerEvidenceError(
                f"Tracker opens a second decision for a World response: {inbound_id}"
            )
        if row.get("decision_id") != inbound.get("idempotency_key"):
            raise TrackerEvidenceError(f"{label} decision id contradicts its inbound")
        audit_protocol = audit_record_by_id[inbound_id].get("protocol_version")
        if row.get("protocol_version") != audit_protocol:
            raise TrackerEvidenceError(f"{label} protocol version contradicts its inbound")
        tracked_inbound_ids.add(inbound_id)

        steps = row.get("steps")
        if not isinstance(steps, list):
            raise TrackerEvidenceError(f"{label} steps must be an array")
        semantic_action_claims: list[Mapping[str, Any]] = []
        continuation_claims: list[tuple[str, Mapping[str, Any]]] = []
        pending_semantic_read: Mapping[str, Any] | None = None
        sourced_sequence = 0
        for step_index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise TrackerEvidenceError(f"{label} step {step_index} is not an object")
            kind = step.get("kind")
            data = step.get("data")
            if not isinstance(kind, str) or not isinstance(data, Mapping):
                raise TrackerEvidenceError(f"{label} step {step_index} is malformed")
            if kind in {"emit_envelope", "no_reply"}:
                raise TrackerEvidenceError(
                    f"{label} embeds a terminal step instead of using terminal metadata"
                )
            if pending_semantic_read is not None and kind != "tool_call":
                raise TrackerEvidenceError(
                    f"{label} typed World-read claim is not followed by its tool result"
                )
            if kind == "semantic_action":
                semantic_action_claims.append(data)
            elif kind == "semantic_read_request":
                if pending_semantic_read is not None:  # pragma: no cover - guarded above
                    raise TrackerEvidenceError(f"{label} has overlapping typed World-read claims")
                pending_semantic_read = data
            elif kind in {
                "framework_protocol_continuation",
                "framework_public_task_continuation",
            }:
                continuation_claims.append((kind, data))
            if kind == "framework_authority_prerequisite":
                source_msg_ids = data.get("source_msg_ids")
                if not isinstance(source_msg_ids, list):
                    raise TrackerEvidenceError(
                        f"{label} framework authority step has no source responses"
                    )
                for source_id in source_msg_ids:
                    response = audited_by_id.get(str(source_id))
                    if (
                        response is None
                        or response.get("from") != "world"
                        or response.get("to") != evaluated_actor_id
                        or _action_kind(
                            response,
                            label=f"framework authority response {source_id}",
                        )
                        != "world.response"
                    ):
                        raise TrackerEvidenceError(
                            "framework authority source is not an audited World response"
                        )
                    request_id = response.get("in_reply_to")
                    request = audited_by_id.get(str(request_id))
                    if (
                        request is None
                        or request.get("from") != evaluated_actor_id
                        or request.get("to") != "world"
                        or request.get("in_reply_to") != inbound_id
                        or _action_kind(
                            request,
                            label=f"framework authority request {request_id}",
                        )
                        not in PUBLIC_WORLD_READ_ACTION_KINDS
                    ):
                        raise TrackerEvidenceError(
                            "framework authority source has no correlated World read"
                        )
                    if request_id in claimed_world_request_ids:
                        raise TrackerEvidenceError(
                            f"World read is claimed more than once: {request_id}"
                        )
                    if source_id in claimed_world_response_ids:
                        raise TrackerEvidenceError(
                            f"World response is claimed more than once: {source_id}"
                        )
                    claimed_world_request_ids.add(str(request_id))
                    claimed_world_response_ids.add(str(source_id))
                    # Agent-owned authority prerequisite reads and subsequent
                    # model-requested reads share the same per-inbound World
                    # request sequence.  Advance past every prerequisite so a
                    # later Tracker tool result is compared with read:1 (or
                    # later), not incorrectly reconstructed as read:0.
                    sourced_sequence += 1
                continue
            if kind != "tool_call":
                continue
            results = data.get("results")
            if not isinstance(results, list):
                raise TrackerEvidenceError(f"{label} tool step has no result array")
            if pending_semantic_read is not None:
                if len(results) != 1 or not isinstance(results[0], Mapping):
                    raise TrackerEvidenceError(
                        f"{label} typed World read does not have exactly one tool result"
                    )
                if data.get("interface") not in {None, "business_decision"}:
                    raise TrackerEvidenceError(
                        f"{label} typed World read has the wrong tool interface"
                    )
                _verify_semantic_read_tool_result(
                    pending_semantic_read,
                    results[0],
                    label=f"{label} step {step_index}",
                )
                pending_semantic_read = None
            elif data.get("interface") == "business_decision":
                raise TrackerEvidenceError(
                    f"{label} business-decision World result has no typed read claim"
                )
            for result_index, result in enumerate(results):
                if not isinstance(result, Mapping):
                    raise TrackerEvidenceError(
                        f"{label} tool result {result_index} is not an object"
                    )
                source_id = result.get("source_msg_id")
                pending_request_id = result.get("pending_request_msg_id")
                if pending_request_id is not None:
                    if (
                        row_terminal != "forced_flush"
                        or forced_flag is not True
                        or incomplete_flag is not True
                        or data.get("incomplete") is not True
                        or set(data) != {"results", "incomplete"}
                        or set(result) != {"tool", "args", "pending_request_msg_id", "status"}
                        or result.get("status") != "pending_response"
                        or source_id is not None
                        or not isinstance(pending_request_id, str)
                        or not pending_request_id
                    ):
                        raise TrackerEvidenceError(
                            f"{label} has a malformed pending World request claim"
                        )
                    if pending_request_id in claimed_world_request_ids:
                        raise TrackerEvidenceError(
                            f"World read is claimed more than once: {pending_request_id}"
                        )
                    request = audited_by_id.get(pending_request_id)
                    tool = result.get("tool")
                    args = result.get("args")
                    if (
                        request is None
                        or not isinstance(tool, str)
                        or not isinstance(args, Mapping)
                    ):
                        raise TrackerEvidenceError(
                            f"pending World read is absent or malformed: {pending_request_id}"
                        )
                    call = ToolCall(tool=tool, args=dict(args))
                    expected_request = tool_call_to_read_envelope(
                        call,
                        from_id=evaluated_actor_id,
                        correlation_id=inbound_id,
                        seq=sourced_sequence,
                        ts=str(request.get("ts", "")),
                    )
                    if _envelope_dict(expected_request) != dict(request):
                        raise TrackerEvidenceError(
                            "pending World read contradicts Tracker ToolCall"
                        )
                    if any(
                        envelope.get("from") == "world"
                        and envelope.get("to") == evaluated_actor_id
                        and envelope.get("in_reply_to") == pending_request_id
                        and _action_kind(
                            envelope, label=f"pending response to {pending_request_id}"
                        )
                        == "world.response"
                        for envelope in audited_by_id.values()
                    ):
                        raise TrackerEvidenceError(
                            "pending World read already has an audited response"
                        )
                    claimed_world_request_ids.add(pending_request_id)
                    sourced_sequence += 1
                    continue
                if source_id is None:
                    continue
                if not isinstance(source_id, str) or not source_id:
                    raise TrackerEvidenceError(f"{label} has an invalid World source id")
                if source_id in claimed_world_response_ids:
                    raise TrackerEvidenceError(
                        f"World response is claimed by multiple tool results: {source_id}"
                    )
                response = audited_by_id.get(source_id)
                if response is None:
                    raise TrackerEvidenceError(
                        f"tool result source is absent from audit: {source_id}"
                    )
                if (
                    response.get("from") != "world"
                    or response.get("to") != evaluated_actor_id
                    or _action_kind(response, label=f"World response {source_id}")
                    != "world.response"
                ):
                    raise TrackerEvidenceError(
                        f"tool result source is not an audited World response: {source_id}"
                    )

                request_id = response.get("in_reply_to")
                request = audited_by_id.get(str(request_id))
                if request is None:
                    raise TrackerEvidenceError(
                        f"World response has no audited request: {source_id}"
                    )
                if request_id in claimed_world_request_ids:
                    raise TrackerEvidenceError(
                        f"World read is claimed by multiple tool results: {request_id}"
                    )
                tool = result.get("tool")
                args = result.get("args")
                if not isinstance(tool, str) or not isinstance(args, Mapping):
                    raise TrackerEvidenceError(f"{label} has a malformed sourced tool result")
                call = ToolCall(tool=tool, args=dict(args))
                expected_request = tool_call_to_read_envelope(
                    call,
                    from_id=evaluated_actor_id,
                    correlation_id=inbound_id,
                    seq=sourced_sequence,
                    ts=str(request.get("ts", "")),
                )
                if _envelope_dict(expected_request) != dict(request):
                    raise TrackerEvidenceError(
                        f"World read request contradicts Tracker tool call: {request_id}"
                    )
                response_envelope = _mapping_to_envelope(response)
                expected_result = world_response_to_tool_result(call, response_envelope)
                if dict(result) != expected_result:
                    raise TrackerEvidenceError(
                        f"Tracker tool result contradicts audited World response: {source_id}"
                    )
                claimed_world_response_ids.add(source_id)
                claimed_world_request_ids.add(str(request_id))
                tool_result_by_source[source_id] = result
                sourced_sequence += 1

        if pending_semantic_read is not None:
            raise TrackerEvidenceError(
                f"{label} typed World-read claim has no recorded tool request/result"
            )

        terminal = row.get("terminal")
        emitted_id = row.get("emitted_msg_id")
        chosen = row.get("chosen")
        if not isinstance(chosen, Mapping) or not isinstance(chosen.get("decision"), str):
            raise TrackerEvidenceError(f"{label} has no normalized choice")
        if terminal == "emit_envelope":
            if forced_flag is not False or incomplete_flag is not False:
                raise TrackerEvidenceError(f"{label} complete emit has invalid flags")
            if not isinstance(emitted_id, str) or not emitted_id:
                raise TrackerEvidenceError(f"{label} emitted no envelope")
            if emitted_id in tracked_emitted_ids:
                raise TrackerEvidenceError(f"Tracker emitted id is claimed twice: {emitted_id}")
            emitted = audited_by_id.get(emitted_id)
            if emitted is None:
                raise TrackerEvidenceError(
                    f"Tracker emitted envelope is absent from audit: {emitted_id}"
                )
            if emitted.get("from") != evaluated_actor_id:
                raise TrackerEvidenceError("Tracker emitted envelope belongs to another actor")
            if emitted.get("in_reply_to") != inbound_id:
                raise TrackerEvidenceError("Tracker emitted envelope has the wrong causal parent")
            _verify_emitted_action_claim(
                label=label,
                evaluated_actor_id=evaluated_actor_id,
                inbound=inbound,
                emitted=emitted,
                audited_by_id=audited_by_id,
                semantic_claims=semantic_action_claims,
                continuation_claims=continuation_claims,
            )
            emitted_kind = _action_kind(emitted, label=f"emitted {emitted_id}")
            if chosen.get("decision") != normalized_decision_for_action_kind(emitted_kind):
                raise TrackerEvidenceError(
                    f"Tracker choice contradicts emitted action {emitted_kind!r}"
                )
            payload = emitted.get("action", {}).get("payload")
            if (
                isinstance(payload, Mapping)
                and "offer_id" in payload
                and chosen.get("offer_id") is not None
                and chosen.get("offer_id") != payload.get("offer_id")
            ):
                raise TrackerEvidenceError("Tracker chosen offer contradicts emitted action")
            tracked_emitted_ids.add(emitted_id)
        elif terminal == "no_reply":
            if (
                forced_flag is not False
                or incomplete_flag is not False
                or emitted_id is not None
                or chosen.get("decision") != "no_reply"
            ):
                raise TrackerEvidenceError(f"{label} no_reply metadata is contradictory")
            if semantic_action_claims or continuation_claims:
                raise TrackerEvidenceError(
                    f"{label} claims an emitted action on a no-reply terminal"
                )
        elif not strict_ideal and terminal == "forced_flush":
            if (
                emitted_id is not None
                or forced_flag is not True
                or incomplete_flag is not True
                or dict(chosen)
                != {
                    "decision": "no_reply",
                    "offer_id": None,
                    "price": None,
                    "rationale": None,
                }
            ):
                raise TrackerEvidenceError(f"{label} incomplete terminal emitted an envelope")
            _verify_forced_flush_action_claims(
                label=label,
                semantic_claims=semantic_action_claims,
                continuation_claims=continuation_claims,
                has_bound_model_termination=termination_binding is not None,
            )
        elif not strict_ideal and terminal in SCOREABLE_FAILURE_TERMINALS:
            expected_stop = _FAILURE_STOP_BY_TERMINAL.get(str(terminal))
            if (
                emitted_id is not None
                or forced_flag is not False
                or incomplete_flag is not True
                or dict(chosen)
                != {
                    "decision": terminal,
                    "offer_id": None,
                    "price": None,
                    "rationale": None,
                }
                or termination_binding is None
                or termination is None
                or termination.get("classification") != expected_stop
                or termination_binding.get("run_id") != row.get("run_id")
                or termination_binding.get("turn") != row.get("turn")
                or termination_binding.get("agent_id") != row.get("agent_id")
                or termination_binding.get("inbound_msg_id") != inbound_id
                or termination_binding.get("decision_id") != row.get("decision_id")
                or termination_binding.get("terminal") != terminal
                or termination_binding.get("row_sha256") != _canonical_row_sha256(row)
            ):
                raise TrackerEvidenceError(
                    f"{label} failure terminal is not exactly bound by termination"
                )
            failure_record_count += 1
            _verify_scoreable_failure_action_claims(
                label=label,
                terminal=str(terminal),
                semantic_claims=semantic_action_claims,
                continuation_claims=continuation_claims,
            )
        else:
            raise TrackerEvidenceError(f"{label} has non-complete terminal {terminal!r}")

        considered = row.get("considered")
        if not isinstance(considered, list):
            raise TrackerEvidenceError(f"{label} considered candidates must be an array")
        for candidate in considered:
            if not isinstance(candidate, Mapping):
                raise TrackerEvidenceError(f"{label} has a malformed candidate")
            source_id = candidate.get("grounded_source_msg_id")
            if source_id is None:
                continue
            result = tool_result_by_source.get(str(source_id))
            if result is None:
                raise TrackerEvidenceError(
                    "grounded candidate cites a World response outside its tool results"
                )
            _verify_grounded_candidate(candidate, result, source_id=str(source_id))

    audited_turn_inbounds = {
        msg_id
        for msg_id, envelope in audited_by_id.items()
        if envelope.get("to") == evaluated_actor_id
        and _action_kind(envelope, label=f"audit envelope {msg_id}") != "world.response"
    }
    if tracked_inbound_ids != audited_turn_inbounds:
        raise TrackerEvidenceError(
            "evaluated actor Tracker coverage does not match audited decision inbounds"
        )

    audited_world_requests = {
        msg_id
        for msg_id, envelope in audited_by_id.items()
        if envelope.get("from") == evaluated_actor_id
        and envelope.get("to") == "world"
        and _action_kind(envelope, label=f"audit envelope {msg_id}")
        in PUBLIC_WORLD_READ_ACTION_KINDS
    }
    if claimed_world_request_ids != audited_world_requests:
        raise TrackerEvidenceError(
            "evaluated actor Tracker does not claim every audited World read"
        )

    audited_world_responses = {
        msg_id
        for msg_id, envelope in audited_by_id.items()
        if envelope.get("from") == "world"
        and envelope.get("to") == evaluated_actor_id
        and _action_kind(envelope, label=f"audit envelope {msg_id}") == "world.response"
    }
    if claimed_world_response_ids != audited_world_responses:
        raise TrackerEvidenceError(
            "evaluated actor Tracker does not claim every audited World response"
        )

    audited_terminal_outputs = {
        msg_id
        for msg_id, envelope in audited_by_id.items()
        if envelope.get("from") == evaluated_actor_id
        and envelope.get("in_reply_to") is not None
        and not (
            envelope.get("to") == "world"
            and _action_kind(envelope, label=f"audit envelope {msg_id}")
            in PUBLIC_WORLD_READ_ACTION_KINDS
        )
    }
    if tracked_emitted_ids != audited_terminal_outputs:
        raise TrackerEvidenceError(
            "evaluated actor Tracker emitted coverage differs from audited actions"
        )

    if termination_binding is not None and failure_record_count != 1:
        raise TrackerEvidenceError(
            "scoreable episode termination must bind exactly one evaluated-actor failure"
        )

    manifest_artifacts = manifest.get("artifacts")
    assert isinstance(manifest_artifacts, Mapping)
    trace_descriptor = manifest_artifacts.get("audit_trace")
    assert isinstance(trace_descriptor, Mapping)
    complete_records = sum(
        row.get("forced_flush") is False
        and row.get("incomplete") is False
        and row.get("terminal") in {"emit_envelope", "no_reply"}
        for row in actor_rows
    )
    return TrackerEvidenceVerdict(
        evaluated_actor_id=evaluated_actor_id,
        strict_ideal=strict_ideal,
        verified=True,
        issues=(),
        record_count=len(actor_rows),
        complete_record_count=complete_records,
        emitted_envelope_count=len(tracked_emitted_ids),
        world_tool_result_count=len(claimed_world_response_ids),
        trace_sha256=str(trace_descriptor.get("sha256")),
    )


def _verify_emitted_action_claim(
    *,
    label: str,
    evaluated_actor_id: str,
    inbound: Mapping[str, Any],
    emitted: Mapping[str, Any],
    audited_by_id: Mapping[str, Mapping[str, Any]],
    semantic_claims: list[Mapping[str, Any]],
    continuation_claims: list[tuple[str, Mapping[str, Any]]],
) -> None:
    """Exact-join one emit to either model semantics or framework continuation."""

    if semantic_claims and continuation_claims:
        raise TrackerEvidenceError(
            f"{label} ambiguously mixes model semantics and framework continuation"
        )
    if len(semantic_claims) == 1 and not continuation_claims:
        _verify_semantic_action_claim(semantic_claims[0], emitted=emitted)
        return
    if len(continuation_claims) == 1 and not semantic_claims:
        kind, data = continuation_claims[0]
        _verify_framework_continuation_claim(
            kind,
            data,
            evaluated_actor_id=evaluated_actor_id,
            inbound=inbound,
            emitted=emitted,
            audited_by_id=audited_by_id,
        )
        return
    raise TrackerEvidenceError(
        f"{label} must have exactly one semantic-action or continuation claim"
    )


def _verify_scoreable_failure_action_claims(
    *,
    label: str,
    terminal: str,
    semantic_claims: list[Mapping[str, Any]],
    continuation_claims: list[tuple[str, Mapping[str, Any]]],
) -> None:
    """Validate action provenance on one bound no-emit failure.

    A security guard can reject a business action after the Agent bridge has
    compiled it but before an ``Envelope`` enters the immutable audit. Its
    single ``semantic_action`` step is therefore compilation provenance, not a
    claim that Runtime emitted the action. A model-protocol failure occurs
    before a valid business action exists, and deterministic continuations are
    never legal on a failed turn.
    """

    if continuation_claims:
        raise TrackerEvidenceError(f"{label} failed terminal claims a framework continuation")
    if not semantic_claims:
        return
    if terminal != SECURITY_ERROR_TERMINAL:
        raise TrackerEvidenceError(f"{label} failed terminal claims a compiled business action")
    if len(semantic_claims) != 1:
        raise TrackerEvidenceError(
            f"{label} guarded failure must have exactly one semantic-action claim"
        )

    claim = semantic_claims[0]
    _validate_semantic_action_data(
        claim,
        label=f"{label} failed semantic-action claim",
    )
    compiled = claim["compiled_vcp"]
    assert isinstance(compiled, Mapping)
    expected_exchange = str(compiled.get("to", "")).startswith("platform:")
    if claim.get("expects_platform_exchange") is not expected_exchange:
        raise TrackerEvidenceError(
            f"{label} failed semantic-action platform-exchange expectation contradicts destination"
        )


def _verify_forced_flush_action_claims(
    *,
    label: str,
    semantic_claims: list[Mapping[str, Any]],
    continuation_claims: list[tuple[str, Mapping[str, Any]]],
    has_bound_model_termination: bool,
) -> None:
    """Validate a compiled action interrupted by a scoreable model abort.

    A multi-inbound episode can have one or more evaluated-actor turns parked
    after the Agent has compiled a model choice but before Runtime emits the
    Platform envelope.  If another turn produces the episode's exact-bound
    scoreable model terminal, teardown force-finalizes those parked turns.
    Their semantic action is hashes-only compilation provenance, not evidence
    that the action crossed Runtime or should receive execution credit.
    """

    if continuation_claims:
        raise TrackerEvidenceError(
            f"{label} incomplete terminal claims a framework continuation"
        )
    if not semantic_claims:
        return
    if not has_bound_model_termination or len(semantic_claims) != 1:
        raise TrackerEvidenceError(
            f"{label} incomplete terminal claims an unbound compiled business action"
        )
    claim = semantic_claims[0]
    _validate_semantic_action_data(
        claim,
        label=f"{label} abort-flushed semantic-action claim",
    )
    compiled = claim["compiled_vcp"]
    assert isinstance(compiled, Mapping)
    if (
        claim.get("expects_platform_exchange") is not True
        or not str(compiled.get("to", "")).startswith("platform:")
    ):
        raise TrackerEvidenceError(
            f"{label} abort-flushed action is not a pending Platform exchange"
        )


def _verify_semantic_action_claim(
    data: Mapping[str, Any],
    *,
    emitted: Mapping[str, Any],
) -> None:
    _validate_semantic_action_data(data, label="semantic-action claim")
    compiled = data["compiled_vcp"]
    assert isinstance(compiled, Mapping)
    emitted_action = emitted.get("action")
    if not isinstance(emitted_action, Mapping):
        raise TrackerEvidenceError("semantic-action emitted envelope has no action")
    payload = emitted_action.get("payload")
    if not isinstance(payload, Mapping):
        raise TrackerEvidenceError("semantic-action emitted envelope has no object payload")
    payload_chars, payload_sha256 = canonical_business_value_digest(payload)
    if compiled.get("to") != emitted.get("to"):
        raise TrackerEvidenceError(
            "semantic-action destination contradicts audited emitted envelope"
        )
    if compiled.get("action_kind") != emitted_action.get("kind"):
        raise TrackerEvidenceError("semantic-action kind contradicts audited emitted envelope")
    if (
        compiled.get("payload_chars") != payload_chars
        or compiled.get("payload_sha256") != payload_sha256
        or compiled.get("payload_projection") != _safe_compiled_payload_projection(payload)
    ):
        raise TrackerEvidenceError("semantic-action payload contradicts audited emitted envelope")
    expected_exchange = str(emitted.get("to", "")).startswith("platform:")
    if data.get("expects_platform_exchange") is not expected_exchange:
        raise TrackerEvidenceError(
            "semantic-action platform-exchange expectation contradicts destination"
        )


def _safe_compiled_payload_projection(value: Any) -> Any:
    """Reconstruct the Agent's strings-digested compiled payload projection."""

    if isinstance(value, Mapping):
        return {str(name): _safe_compiled_payload_projection(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_safe_compiled_payload_projection(item) for item in value]
    if isinstance(value, str):
        return {
            "text_chars": len(value),
            "text_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
    if value is None or isinstance(value, (bool, int, float)):
        return copy.deepcopy(value)
    raise TrackerEvidenceError("compiled business payload is not strict JSON")


def _verify_framework_continuation_claim(
    kind: str,
    data: Mapping[str, Any],
    *,
    evaluated_actor_id: str,
    inbound: Mapping[str, Any],
    emitted: Mapping[str, Any],
    audited_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    emitted_action = emitted.get("action")
    if not isinstance(emitted_action, Mapping):
        raise TrackerEvidenceError("framework continuation emitted no action")
    payload = emitted_action.get("payload")
    if not isinstance(payload, Mapping):
        raise TrackerEvidenceError("framework continuation emitted no object payload")
    if kind == "framework_public_task_continuation":
        destination = data.get("destination")
        self_continuation = destination == "@self"
        if (
            data.get("interface") != "deterministic_agent_controller"
            or data.get("action_kind") != emitted_action.get("kind")
            or not isinstance(destination, str)
            or not destination
            or (self_continuation and emitted.get("to") != evaluated_actor_id)
            or (
                not self_continuation
                and (
                    destination != emitted.get("to")
                    or destination == evaluated_actor_id
                    or destination.split(":", 1)[0] not in {"buyer", "merchant"}
                )
            )
        ):
            raise TrackerEvidenceError(
                "public-task continuation contradicts audited emitted envelope"
            )
        if not self_continuation:
            inbound_action = inbound.get("action")
            inbound_payload = (
                inbound_action.get("payload") if isinstance(inbound_action, Mapping) else None
            )
            accepted_id = inbound.get("in_reply_to")
            accepted = audited_by_id.get(accepted_id) if isinstance(accepted_id, str) else None
            accepted_action = accepted.get("action") if isinstance(accepted, Mapping) else None
            accepted_payload = (
                accepted_action.get("payload") if isinstance(accepted_action, Mapping) else None
            )
            if (
                inbound.get("from") != "platform:after-sales"
                or not isinstance(inbound_action, Mapping)
                or inbound_action.get("kind") != "platform.after_sales_updated"
                or not isinstance(inbound_payload, Mapping)
                or emitted.get("in_reply_to") != inbound.get("msg_id")
                or not isinstance(accepted, Mapping)
                or accepted.get("from") != evaluated_actor_id
                or accepted.get("to") != "platform:after-sales"
                or not isinstance(accepted_action, Mapping)
                or not str(accepted_action.get("kind", "")).startswith("commerce.")
                or not isinstance(accepted_payload, Mapping)
                or accepted_payload.get("order_id") != inbound_payload.get("order_id")
            ):
                raise TrackerEvidenceError(
                    "bound-counterparty continuation has no exact accepted lineage"
                )
        payload_chars, payload_sha256 = canonical_business_value_digest(payload)
        if (
            data.get("payload_chars") != payload_chars
            or data.get("payload_sha256") != payload_sha256
        ):
            raise TrackerEvidenceError(
                "public-task continuation payload contradicts audited emitted envelope"
            )
        return
    if kind != "framework_protocol_continuation":
        raise TrackerEvidenceError("unsupported framework continuation claim")
    inbound_action = inbound.get("action")
    cert_id = payload.get("cert_id") if isinstance(payload, Mapping) else None
    if (
        data.get("interface") != "deterministic_agent_controller"
        or not isinstance(inbound_action, Mapping)
        or data.get("trigger_action_kind") != inbound_action.get("kind")
        or data.get("action_kind") != emitted_action.get("kind")
        or data.get("destination") != emitted.get("to")
        or set(payload) != {"cert_id"}
        or not isinstance(cert_id, str)
        or hashlib.sha256(cert_id.encode("utf-8")).hexdigest() != data.get("certificate_id_sha256")
    ):
        raise TrackerEvidenceError("protocol continuation contradicts audited emitted envelope")
    accepted_id = inbound.get("in_reply_to")
    accepted = audited_by_id.get(str(accepted_id))
    if (
        accepted is None
        or accepted.get("from") != evaluated_actor_id
        or _wire_envelope_sha256(accepted) != data.get("accepted_envelope_sha256")
    ):
        raise TrackerEvidenceError(
            "protocol continuation has no exact audited accepted-envelope authority"
        )


def _wire_envelope_sha256(value: Mapping[str, Any]) -> str:
    try:
        rendered = to_json(_mapping_to_envelope(value))
    except Exception as exc:
        raise TrackerEvidenceError("audited continuation envelope is malformed") from exc
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _argument_binding_sha256(
    *,
    intent: str,
    model_arguments_sha256: str,
    resolved_arguments_sha256: str,
) -> str:
    _chars, digest = canonical_business_value_digest(
        {
            "schema_version": "cwe.agent-argument-binding.v1",
            "intent": intent,
            "model_arguments_sha256": model_arguments_sha256,
            "resolved_arguments_sha256": resolved_arguments_sha256,
        }
    )
    return digest


def _validate_semantic_action_data(
    data: Mapping[str, Any],
    *,
    label: str,
) -> None:
    compiled = data.get("compiled_vcp")
    intent = data.get("business_intent")
    response_chars = data.get("response_chars")
    if (
        set(data) != _TRACKER_STEP_FIELDS["semantic_action"]
        or data.get("interface") != "business_decision"
        or not isinstance(data.get("function"), str)
        or not data.get("function")
        or not isinstance(intent, str)
        or not intent
        or data.get("compiler_validated") is not True
        or not isinstance(data.get("observation_class"), str)
        or not data.get("observation_class")
        or not isinstance(data.get("expects_platform_exchange"), bool)
        or isinstance(response_chars, bool)
        or not isinstance(response_chars, int)
        or not 0 <= response_chars <= 1_048_576
        or not _is_sha256(data.get("response_sha256"))
        or not _valid_length_digest_pair(data.get("arguments_chars"), data.get("arguments_sha256"))
        or not isinstance(compiled, Mapping)
    ):
        raise TrackerEvidenceError(f"{label} has invalid typed provenance")
    _validate_model_choice_argument_binding(data, intent=intent, label=label)
    if (
        set(compiled)
        != {
            "to",
            "action_kind",
            "payload_chars",
            "payload_sha256",
            "payload_projection",
        }
        or not isinstance(compiled.get("to"), str)
        or not compiled.get("to")
        or not isinstance(compiled.get("action_kind"), str)
        or not compiled.get("action_kind")
        or not _valid_length_digest_pair(
            compiled.get("payload_chars"), compiled.get("payload_sha256")
        )
        or not isinstance(compiled.get("payload_projection"), Mapping)
    ):
        raise TrackerEvidenceError(f"{label} compiled VCP binding is malformed")
    _validate_sanitized_choice_value(
        compiled["payload_projection"],
        label=f"{label}.compiled_vcp.payload_projection",
    )


def _validate_semantic_read_request_data(
    data: Mapping[str, Any],
    *,
    label: str,
) -> None:
    compiled = data.get("compiled_world_read")
    intent = data.get("business_intent")
    response_chars = data.get("response_chars")
    if (
        set(data) != _TRACKER_STEP_FIELDS["semantic_read_request"]
        or data.get("interface") != "business_decision"
        or not isinstance(data.get("function"), str)
        or not data.get("function")
        or not isinstance(intent, str)
        or not intent
        or data.get("compiler_validated") is not True
        or isinstance(response_chars, bool)
        or not isinstance(response_chars, int)
        or not 0 <= response_chars <= 1_048_576
        or not _is_sha256(data.get("response_sha256"))
        or not _valid_length_digest_pair(data.get("arguments_chars"), data.get("arguments_sha256"))
        or not isinstance(compiled, Mapping)
    ):
        raise TrackerEvidenceError(f"{label} has invalid typed World-read provenance")
    _validate_model_choice_argument_binding(data, intent=intent, label=label)
    if (
        set(compiled) != {"tool", "args_chars", "args_sha256"}
        or not isinstance(compiled.get("tool"), str)
        or not compiled.get("tool")
        or not _valid_length_digest_pair(compiled.get("args_chars"), compiled.get("args_sha256"))
    ):
        raise TrackerEvidenceError(f"{label} compiled World read is malformed")


def _verify_semantic_read_tool_result(
    claim: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Bind an Agent compiler claim to the exact recorded World ToolCall."""

    compiled = claim.get("compiled_world_read")
    tool = result.get("tool")
    args = result.get("args")
    if (
        not isinstance(compiled, Mapping)
        or not isinstance(tool, str)
        or not isinstance(args, Mapping)
        or tool != compiled.get("tool")
    ):
        raise TrackerEvidenceError(
            f"{label} typed World-read compiler claim contradicts its tool result"
        )
    try:
        args_chars, args_sha256 = canonical_business_value_digest(dict(args))
    except Exception as exc:
        raise TrackerEvidenceError(
            f"{label} typed World-read tool arguments are not canonical JSON"
        ) from exc
    if args_chars != compiled.get("args_chars") or args_sha256 != compiled.get("args_sha256"):
        raise TrackerEvidenceError(
            f"{label} typed World-read arguments drifted after Agent compilation"
        )


def _validate_model_choice_argument_binding(
    data: Mapping[str, Any],
    *,
    intent: str,
    label: str,
) -> None:
    choice = data.get("model_business_choice")
    binding = data.get("agent_argument_binding")
    if not isinstance(choice, Mapping) or not isinstance(binding, Mapping):
        raise TrackerEvidenceError(f"{label} has no typed Agent argument binding")
    if set(choice) != {
        "schema_version",
        "intent",
        "arguments",
        "arguments_chars",
        "arguments_sha256",
    }:
        raise TrackerEvidenceError(f"{label} model choice has schema drift")
    arguments = choice.get("arguments")
    if not isinstance(arguments, Mapping):
        raise TrackerEvidenceError(f"{label} model choice arguments are malformed")
    choice_chars, choice_sha256 = canonical_business_value_digest(arguments)
    if (
        choice.get("schema_version") != MODEL_BUSINESS_CHOICE_SCHEMA_V1
        or choice.get("intent") != intent
        or choice.get("arguments_chars") != choice_chars
        or choice.get("arguments_sha256") != choice_sha256
    ):
        raise TrackerEvidenceError(f"{label} model choice binding is inconsistent")
    _validate_sanitized_choice_value(arguments, label=f"{label}.arguments")
    if set(binding) != {
        "model_arguments_sha256",
        "resolved_arguments_chars",
        "resolved_arguments_sha256",
        "binding_sha256",
    }:
        raise TrackerEvidenceError(f"{label} Agent argument binding has schema drift")
    if (
        binding.get("model_arguments_sha256") != choice_sha256
        or binding.get("resolved_arguments_chars") != data.get("arguments_chars")
        or binding.get("resolved_arguments_sha256") != data.get("arguments_sha256")
        or binding.get("binding_sha256")
        != _argument_binding_sha256(
            intent=intent,
            model_arguments_sha256=choice_sha256,
            resolved_arguments_sha256=str(data.get("arguments_sha256")),
        )
    ):
        raise TrackerEvidenceError(f"{label} Agent argument binding is inconsistent")


def _valid_length_digest_pair(length: object, digest: object) -> bool:
    return bool(
        isinstance(length, int)
        and not isinstance(length, bool)
        and 0 <= length <= 1_048_576
        and _is_sha256(digest)
    )


def _validate_sanitized_choice_value(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        if set(value) == {"text_chars", "text_sha256"}:
            if not _valid_length_digest_pair(value.get("text_chars"), value.get("text_sha256")):
                raise TrackerEvidenceError(f"{label} has an invalid text digest")
            return
        for key, item in value.items():
            if not isinstance(key, str):
                raise TrackerEvidenceError(f"{label} has a non-text argument name")
            _validate_sanitized_choice_value(item, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_sanitized_choice_value(item, label=f"{label}[{index}]")
        return
    if isinstance(value, str):
        if len(value) > 512 or any(ord(character) < 32 for character in value):
            raise TrackerEvidenceError(f"{label} retains unsafe model-authored text")
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    raise TrackerEvidenceError(f"{label} is not strict sanitized JSON")


def _validate_tracker_row_schema(row: object, *, ordinal: int) -> None:
    """Reject schema drift and provider-text fields in formal Tracker rows.

    ``TraceStep.data`` is extensible inside the environment, but formal
    benchmark evidence needs an explicit projection.  New instrumentation must
    be reviewed before it can enter a paper artifact, and raw model messages or
    reasoning must never appear there.  World results remain values under the
    already-authorized ``results`` field.
    """

    label = f"Tracker row {ordinal}"
    if not isinstance(row, Mapping) or set(row) != _TRACKER_ROW_FIELDS:
        raise TrackerEvidenceError(f"{label} has an unknown or missing field")
    steps = row.get("steps")
    if not isinstance(steps, list):
        raise TrackerEvidenceError(f"{label} steps must be an array")
    for step_index, step in enumerate(steps):
        if not isinstance(step, Mapping) or set(step) != {"kind", "data"}:
            raise TrackerEvidenceError(f"{label} step {step_index} has an unknown or missing field")
        kind = step.get("kind")
        data = step.get("data")
        allowed = _TRACKER_STEP_FIELDS.get(kind) if isinstance(kind, str) else None
        if allowed is None or not isinstance(data, Mapping) or not set(data) <= allowed:
            raise TrackerEvidenceError(
                f"{label} step {step_index} has an unsupported evidence shape"
            )
        if {str(key).casefold() for key in data} & _FORBIDDEN_TRACKER_STEP_FIELDS:
            raise TrackerEvidenceError(f"{label} step {step_index} exposes provider model I/O")
        if kind == "public_task_execution" and (
            set(data)
            not in {
                _PUBLIC_TASK_COMMON_FIELDS,
                _PUBLIC_TASK_BUSINESS_FIELDS,
            }
            or not isinstance(data.get("phase_id"), str)
            or not data.get("phase_id")
            or isinstance(data.get("phase_index"), bool)
            or not isinstance(data.get("phase_index"), int)
            or data.get("phase_index", -1) < 0
            or not isinstance(data.get("decision_id"), str)
            or not data.get("decision_id")
            or not _is_sha256(data.get("execution_contract_sha256"))
            or (
                set(data) == _PUBLIC_TASK_BUSINESS_FIELDS
                and (
                    data.get("interface") != "llm_business_decision_v1"
                    or not _is_sha256(data.get("llm_decision_surface_sha256"))
                )
            )
            or isinstance(data.get("semantic_authority_revision"), bool)
            or not isinstance(data.get("semantic_authority_revision"), int)
            or data.get("semantic_authority_revision", -1) < 0
            or not isinstance(data.get("authority_source_msg_ids"), list)
            or any(
                not isinstance(source_id, str) or not source_id
                for source_id in data.get("authority_source_msg_ids", ())
            )
            or len(data.get("authority_source_msg_ids", ()))
            != len(set(data.get("authority_source_msg_ids", ())))
        ):
            raise TrackerEvidenceError(
                f"{label} step {step_index} has an invalid public task binding"
            )
        if kind == "framework_public_task_continuation" and (
            set(data) != _TRACKER_STEP_FIELDS["framework_public_task_continuation"]
            or data.get("interface") != "deterministic_agent_controller"
            or not isinstance(data.get("destination"), str)
            or not data.get("destination")
            or (
                data.get("destination") != "@self"
                and str(data.get("destination")).split(":", 1)[0] not in {"buyer", "merchant"}
            )
            or not isinstance(data.get("action_kind"), str)
            or not data.get("action_kind")
            or not _is_sha256(data.get("payload_sha256"))
            or isinstance(data.get("payload_chars"), bool)
            or not isinstance(data.get("payload_chars"), int)
            or data.get("payload_chars", -1) < 0
        ):
            raise TrackerEvidenceError(
                f"{label} step {step_index} has an invalid continuation digest"
            )
        if kind == "framework_protocol_continuation" and (
            set(data) != _TRACKER_STEP_FIELDS["framework_protocol_continuation"]
            or data.get("interface") != "deterministic_agent_controller"
            or not isinstance(data.get("trigger_action_kind"), str)
            or not data.get("trigger_action_kind")
            or not isinstance(data.get("action_kind"), str)
            or not data.get("action_kind")
            or not isinstance(data.get("destination"), str)
            or not data.get("destination")
            or not isinstance(data.get("authority"), str)
            or not data.get("authority")
            or not _is_sha256(data.get("accepted_envelope_sha256"))
            or not _is_sha256(data.get("certificate_id_sha256"))
        ):
            raise TrackerEvidenceError(
                f"{label} step {step_index} has an invalid protocol continuation"
            )
        if kind == "semantic_action":
            _validate_semantic_action_data(
                data,
                label=f"{label} step {step_index}",
            )
        if kind == "semantic_read_request":
            _validate_semantic_read_request_data(
                data,
                label=f"{label} step {step_index}",
            )
        if kind == "framework_authority_prerequisite":
            source_msg_ids = data.get("source_msg_ids")
            authority_sha256 = data.get("authority_sha256")
            if (
                set(data) != _FRAMEWORK_AUTHORITY_PREREQUISITE_FIELDS
                or data.get("interface") != "agent_world_authority_prerequisite"
                or data.get("authority_kind") not in {"after_sales", "protocol_event"}
                or not isinstance(source_msg_ids, list)
                or not source_msg_ids
                or any(
                    not isinstance(source_id, str) or not source_id for source_id in source_msg_ids
                )
                or len(source_msg_ids) != len(set(source_msg_ids))
                or not isinstance(authority_sha256, list)
                or len(authority_sha256) != len(source_msg_ids)
                or any(not _is_sha256(digest) for digest in authority_sha256)
            ):
                raise TrackerEvidenceError(
                    f"{label} step {step_index} has an invalid framework authority binding"
                )


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_foreign_termination_binding(
    trace_rows: tuple[dict[str, Any], ...],
    *,
    binding: Mapping[str, Any],
    scenario_id: object,
    classification: str,
) -> None:
    """Prove a scoreable termination bound to another active actor exactly."""

    actor_id = binding.get("agent_id")
    inbound_msg_id = binding.get("inbound_msg_id")
    decision_id = binding.get("decision_id")
    terminal = binding.get("terminal")
    row_sha256 = binding.get("row_sha256")
    run_id = binding.get("run_id")
    turn = binding.get("turn")
    if (
        not all(
            isinstance(value, str) and value
            for value in (
                actor_id,
                inbound_msg_id,
                decision_id,
                terminal,
                row_sha256,
                run_id,
            )
        )
        or terminal not in SCOREABLE_FAILURE_TERMINALS
        or isinstance(turn, bool)
        or not isinstance(turn, int)
        or turn < 0
        or classification != _FAILURE_STOP_BY_TERMINAL.get(str(terminal))
    ):
        raise TrackerEvidenceError("foreign termination Tracker binding is malformed")
    matches = [
        row
        for row in trace_rows
        if row.get("agent_id") == actor_id
        and row.get("inbound_msg_id") == inbound_msg_id
        and row.get("decision_id") == decision_id
        and row.get("terminal") == terminal
    ]
    if len(matches) != 1:
        raise TrackerEvidenceError("foreign termination does not identify exactly one Tracker row")
    row = matches[0]
    if (
        row.get("run_id") != scenario_id
        or run_id != row.get("run_id")
        or turn != row.get("turn")
        or row.get("forced_flush") is not False
        or row.get("incomplete") is not True
        or _canonical_row_sha256(row) != row_sha256
    ):
        raise TrackerEvidenceError("foreign termination Tracker row is not manifest-valid")


def _trace_summary(
    evidence: object,
    *,
    evaluated_actor_id: str,
) -> dict[str, Any]:
    rows = getattr(evidence, "trace_rows", ())
    actor_rows = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("agent_id") == evaluated_actor_id
    ]
    complete = [
        row
        for row in actor_rows
        if row.get("forced_flush") is False
        and row.get("incomplete") is False
        and row.get("terminal") in {"emit_envelope", "no_reply"}
    ]
    emitted = {
        row.get("emitted_msg_id")
        for row in actor_rows
        if isinstance(row.get("emitted_msg_id"), str)
    }
    sources = {
        result.get("source_msg_id")
        for row in actor_rows
        for step in row.get("steps", [])
        if isinstance(step, Mapping)
        and step.get("kind") == "tool_call"
        and isinstance(step.get("data"), Mapping)
        for result in step["data"].get("results", [])
        if isinstance(result, Mapping) and isinstance(result.get("source_msg_id"), str)
    }
    manifest = getattr(evidence, "evidence_manifest", None)
    artifacts = manifest.get("artifacts") if isinstance(manifest, Mapping) else None
    descriptor = artifacts.get("audit_trace") if isinstance(artifacts, Mapping) else None
    digest = descriptor.get("sha256") if isinstance(descriptor, Mapping) else None
    return {
        "record_count": len(actor_rows),
        "complete_record_count": len(complete),
        "emitted_envelope_count": len(emitted),
        "world_tool_result_count": len(sources),
        "trace_sha256": str(digest) if isinstance(digest, str) else None,
    }


def _verified_trace_artifact(
    root: Path,
    manifest: object,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(manifest, Mapping):
        raise TrackerEvidenceError("Tracker verification requires an episode manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TrackerEvidenceError("episode manifest has no artifact inventory")
    descriptor = artifacts.get("audit_trace")
    if not isinstance(descriptor, Mapping):
        raise TrackerEvidenceError("audit.trace.jsonl is not hash-covered by the manifest")
    path = root / TRACKER_TRACE_ARTIFACT
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TrackerEvidenceError(f"cannot read Tracker artifact: {exc}") from exc
    lines = [line for line in raw.splitlines() if line.strip()]
    expected = {
        "path": TRACKER_TRACE_ARTIFACT,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "record_count": len(lines),
    }
    if dict(descriptor) != expected:
        raise TrackerEvidenceError("Tracker artifact does not match its manifest descriptor")
    rows: list[dict[str, Any]] = []
    for ordinal, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TrackerEvidenceError(f"Tracker row {ordinal} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise TrackerEvidenceError(f"Tracker row {ordinal} is not an object")
        rows.append(value)
    return tuple(rows)


def _verified_termination_artifact(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TrackerEvidenceError("episode manifest has no artifact inventory")
    descriptor = artifacts.get("termination")
    if descriptor is None:
        if (root / "termination.json").exists():
            raise TrackerEvidenceError("termination exists outside the episode manifest")
        return None
    if not isinstance(descriptor, Mapping):
        raise TrackerEvidenceError("termination descriptor is malformed")
    path = root / "termination.json"
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackerEvidenceError("cannot read episode termination artifact") from exc
    expected = {
        "path": "termination.json",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if dict(descriptor) != expected or not isinstance(value, dict):
        raise TrackerEvidenceError("termination artifact does not match its manifest descriptor")
    return value


def _canonical_row_sha256(row: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(row),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrackerEvidenceError("Tracker failure row is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _action_kind(envelope: Mapping[str, Any], *, label: str) -> str:
    action = envelope.get("action")
    kind = action.get("kind") if isinstance(action, Mapping) else None
    if not isinstance(kind, str) or not kind:
        raise TrackerEvidenceError(f"{label} has no typed action")
    return kind


def _mapping_to_envelope(value: Mapping[str, Any]) -> Envelope:
    try:
        return from_json(
            json.dumps(
                dict(value),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except Exception as exc:
        raise TrackerEvidenceError("audited World response is not a valid envelope") from exc


def _envelope_dict(value: Envelope) -> dict[str, Any]:
    decoded = json.loads(to_json(value))
    if not isinstance(decoded, dict):  # pragma: no cover - to_json is typed
        raise AssertionError("Envelope did not serialize to an object")
    return decoded


def _verify_grounded_candidate(
    candidate: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    source_id: str,
) -> None:
    payload = result.get("result")
    items = payload if isinstance(payload, list) else [payload]
    sku_id = candidate.get("sku_id")
    matching = [
        item for item in items if isinstance(item, Mapping) and item.get("sku_id") == sku_id
    ]
    if len(matching) != 1:
        raise TrackerEvidenceError(
            f"grounded candidate does not uniquely match World result {source_id}"
        )
    if candidate.get("grounded_attributes") != matching[0].get("attributes"):
        raise TrackerEvidenceError(
            f"grounded candidate attributes contradict World result {source_id}"
        )


__all__ = [
    "TRACKER_TRACE_ARTIFACT",
    "AllActiveActorTrackerVerdict",
    "TrackerEvidenceError",
    "TrackerEvidenceVerdict",
    "VerifiedModelBusinessChoice",
    "VerifiedModelWorldRead",
    "tracker_row_has_usable_completed_steps",
    "verify_tracker_evidence",
    "verify_all_active_actor_tracker_evidence",
    "verified_model_business_choices",
    "verified_model_local_finish",
    "verified_model_world_reads",
    "verified_tracker_emitted_msg_ids",
]
