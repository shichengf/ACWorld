"""Fail-closed skill-path evidence for deterministic benchmark references.

The formal ideal policy is useful only when it exercises the same Agent skill
gate that a live model sees.  A scripted channel that emits an otherwise valid
envelope without first loading an activated skill can make an incomplete skill
surface look healthy.  This verifier closes that gap from hash-covered Runtime
artifacts.

For every model-authored business envelope emitted by the evaluated actor in a
clean ideal episode it
joins the Tracker decision to the exact audited inbound and outbound envelopes
and requires at least one successful ``load_skill`` step.  Agent performs the
selector check before recording such a step, so a recorded load is dynamic
evidence that the role selector activated a real, enabled skill for that exact
inbound action kind.  The verifier does not infer coverage from source text or
from a separately maintained list of action names.

No-reply turns and deterministic Agent protocol continuations do not need a
model skill.  The latter exercise Agent interface logic, not LLM capability,
and are identified from framework-owned Tracker steps rather than action-name
allowlists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class ReferenceSkillEvidenceSource(Protocol):
    """Minimal hash-covered artifact surface consumed by this verifier."""

    envelopes: tuple[dict[str, Any], ...]
    trace_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ReferenceSkillCoverageV2:
    """Dynamic skill coverage for one evaluated reference policy."""

    evaluated_actor_id: str
    emitted_turn_count: int
    verified_turn_count: int
    inbound_action_kinds: tuple[str, ...]
    outbound_action_kinds: tuple[str, ...]
    loaded_skill_names: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def verified(self) -> bool:
        """Return whether every emitting reference turn used a real skill."""

        return (
            self.emitted_turn_count > 0
            and self.verified_turn_count == self.emitted_turn_count
            and not self.issues
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready report projection."""

        return {
            "evaluated_actor_id": self.evaluated_actor_id,
            "emitted_turn_count": self.emitted_turn_count,
            "verified_turn_count": self.verified_turn_count,
            "inbound_action_kinds": list(self.inbound_action_kinds),
            "outbound_action_kinds": list(self.outbound_action_kinds),
            "loaded_skill_names": list(self.loaded_skill_names),
            "issues": list(self.issues),
            "verified": self.verified,
        }


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _action_kind(envelope: Mapping[str, Any]) -> str | None:
    action = envelope.get("action")
    if not isinstance(action, Mapping):
        return None
    return _text(action.get("kind"))


def _envelopes_by_id(
    envelopes: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Mapping[str, Any]], tuple[str, ...]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    issues: list[str] = []
    for ordinal, envelope in enumerate(envelopes, start=1):
        if not isinstance(envelope, Mapping):
            issues.append(f"audit envelope {ordinal} is not an object")
            continue
        msg_id = _text(envelope.get("msg_id"))
        if msg_id is None:
            issues.append(f"audit envelope {ordinal} has no msg_id")
            continue
        if msg_id in by_id:
            issues.append(f"audit envelope msg_id {msg_id!r} is not unique")
            continue
        if _action_kind(envelope) is None:
            issues.append(f"audit envelope {msg_id!r} has no typed action")
            continue
        by_id[msg_id] = envelope
    return by_id, tuple(issues)


def _loaded_skills(trace: Mapping[str, Any]) -> tuple[str, ...]:
    loaded: list[str] = []
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return ()
    for step in steps:
        if not isinstance(step, Mapping) or step.get("kind") != "load_skill":
            continue
        data = step.get("data")
        if not isinstance(data, Mapping):
            continue
        name = _text(data.get("name"))
        result = _text(data.get("result"))
        # Agent appends the history row only after selector enforcement and a
        # successful SkillLoader.load.  Requiring its framework-owned result
        # prevents a model-authored trace-like object from counting as a load.
        if name is not None and result is not None and result.startswith("loaded ("):
            loaded.append(name)
    return tuple(loaded)


def _has_model_business_action(trace: Mapping[str, Any]) -> bool:
    """Return whether this emitting turn came through the typed LLM bridge."""

    steps = trace.get("steps")
    if not isinstance(steps, list):
        return False
    return any(
        isinstance(step, Mapping)
        and step.get("kind") == "semantic_action"
        and isinstance(step.get("data"), Mapping)
        and step["data"].get("interface") == "business_decision"
        for step in steps
    )


def verify_reference_skill_evidence_v2(
    evidence: ReferenceSkillEvidenceSource,
    *,
    evaluated_actor_id: str,
) -> ReferenceSkillCoverageV2:
    """Verify exact inbound to selector to skill to outbound reference turns.

    The supplied evidence must already come from a complete CommerceWorld
    episode.  Tracker and manifest verification remain separate mandatory
    gates.  This function adds the narrower guarantee that scripted ideal
    responses cannot bypass Agent skills while still passing those gates.
    """

    issues: list[str] = []
    by_id, envelope_issues = _envelopes_by_id(evidence.envelopes)
    issues.extend(envelope_issues)

    emitting: list[Mapping[str, Any]] = []
    inbound_kinds: set[str] = set()
    outbound_kinds: set[str] = set()
    skill_names: set[str] = set()
    verified_turns = 0
    seen_emitted_ids: set[str] = set()

    for ordinal, trace in enumerate(evidence.trace_rows, start=1):
        if not isinstance(trace, Mapping) or trace.get("agent_id") != evaluated_actor_id:
            continue
        if trace.get("terminal") != "emit_envelope":
            continue
        if not _has_model_business_action(trace):
            continue
        emitting.append(trace)

        inbound_id = _text(trace.get("inbound_msg_id"))
        emitted_id = _text(trace.get("emitted_msg_id"))
        label = f"evaluated actor trace {ordinal}"
        inbound = by_id.get(inbound_id or "")
        outbound = by_id.get(emitted_id or "")
        turn_ok = True

        if inbound_id is None or inbound is None:
            issues.append(f"{label} does not join to one audited inbound envelope")
            turn_ok = False
        elif inbound.get("to") != evaluated_actor_id:
            issues.append(f"{label} inbound envelope is addressed to another actor")
            turn_ok = False
        else:
            inbound_kind = _action_kind(inbound)
            if inbound_kind is not None:
                inbound_kinds.add(inbound_kind)

        if emitted_id is None or outbound is None:
            issues.append(f"{label} does not join to one audited emitted envelope")
            turn_ok = False
        elif outbound.get("from") != evaluated_actor_id:
            issues.append(f"{label} emitted envelope belongs to another actor")
            turn_ok = False
        elif emitted_id in seen_emitted_ids:
            issues.append(f"{label} reuses emitted envelope {emitted_id!r}")
            turn_ok = False
        else:
            seen_emitted_ids.add(emitted_id)
            outbound_kind = _action_kind(outbound)
            if outbound_kind is not None:
                outbound_kinds.add(outbound_kind)

        loaded = _loaded_skills(trace)
        if not loaded:
            inbound_kind = _action_kind(inbound) if inbound is not None else None
            issues.append(
                f"{label} emitted after {inbound_kind or 'unknown inbound'!r} "
                "without a successful selector-approved load_skill step"
            )
            turn_ok = False
        else:
            skill_names.update(loaded)
            recorded_last = _text(trace.get("skill"))
            if recorded_last != loaded[-1]:
                issues.append(f"{label} last-loaded skill does not match Tracker projection")
                turn_ok = False

        if turn_ok:
            verified_turns += 1

    if not emitting:
        issues.append("evaluated ideal policy emitted no typed model business action")

    return ReferenceSkillCoverageV2(
        evaluated_actor_id=evaluated_actor_id,
        emitted_turn_count=len(emitting),
        verified_turn_count=verified_turns,
        inbound_action_kinds=tuple(sorted(inbound_kinds)),
        outbound_action_kinds=tuple(sorted(outbound_kinds)),
        loaded_skill_names=tuple(sorted(skill_names)),
        issues=tuple(issues),
    )


__all__ = [
    "ReferenceSkillCoverageV2",
    "ReferenceSkillEvidenceSource",
    "verify_reference_skill_evidence_v2",
]
