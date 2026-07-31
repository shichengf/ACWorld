"""Offline causal verification for Runtime-accepted actor evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from protocol.actor_result import ActorResultBinding
from protocol.envelope import Envelope, from_json, to_json
from runtime.actor_context_artifact import (
    ActorContextRegistrations,
    load_actor_contexts,
)
from runtime.actor_evidence import load_actor_evidence


class ActorEvidenceCausalityError(ValueError):
    """The audit, trusted contexts, and accepted journal do not join exactly."""


_TERMINATION_SCHEMA = "cwe.episode-termination.v1"
_TRACE_ARTIFACT = "audit.trace.jsonl"
_CONTEXTS_ARTIFACT = "actor.contexts.json"
_STOP_REASON_BY_FAILURE_TERMINAL = {
    "protocol_error": "model_protocol_error",
    "resource_limit": "resource_limit",
    "security_error": "security_guard",
}


@dataclass(frozen=True, slots=True)
class ActorEvidenceCausalityVerification:
    """Counts from one successful independent actor-evidence replay."""

    audit_envelopes: int
    registered_contexts: int
    accepted_actor_evidence: int
    context_roots_observed: int
    evidence_requests_replayed: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def verify_actor_evidence_causality(
    *,
    audit_path: str | Path,
    contexts_path: str | Path,
    journal_path: str | Path,
    expected_episode_context_id: str | None = None,
    strict: bool = True,
    termination_path: str | Path | None = None,
) -> ActorEvidenceCausalityVerification:
    """Rebuild the resolver, replay audit order, and re-derive every binding.

    The accepted journal is treated only as a claim to check.  Actor,
    principal, task, mandate, episode, action, idempotency, parent, request id,
    and request digest are all derived again from the persisted registrations
    plus the exact audited causal graph.
    """

    try:
        return _verify_actor_evidence_causality(
            audit_path=Path(audit_path),
            contexts_path=Path(contexts_path),
            journal_path=Path(journal_path),
            expected_episode_context_id=expected_episode_context_id,
            strict=strict,
            termination_path=(
                Path(termination_path) if termination_path is not None else None
            ),
        )
    except ActorEvidenceCausalityError:
        raise
    except Exception as exc:
        raise ActorEvidenceCausalityError(
            f"actor evidence causal verification failed: {exc}"
        ) from exc


def _verify_actor_evidence_causality(
    *,
    audit_path: Path,
    contexts_path: Path,
    journal_path: Path,
    expected_episode_context_id: str | None,
    strict: bool,
    termination_path: Path | None,
) -> ActorEvidenceCausalityVerification:
    registry = load_actor_contexts(contexts_path, strict=strict)
    if (
        expected_episode_context_id is not None
        and registry.episode_context_id != expected_episode_context_id
    ):
        raise ActorEvidenceCausalityError(
            "actor-context artifact belongs to another episode"
        )
    records = load_actor_evidence(journal_path, strict=strict)
    records_by_key = {
        (str(record["actor_id"]), str(record["idempotency_key"])): record
        for record in records
    }
    if len(records_by_key) != len(records):
        raise ActorEvidenceCausalityError(
            "actor evidence contains duplicate actor idempotency keys"
        )

    resolver = registry.build_resolver()
    registrations = {
        context.root_msg_id: context for context in registry.registrations
    }
    observed: dict[str, Envelope] = {}
    observed_registration_roots: set[str] = set()
    audited_context_roots: set[str] = set()
    seen_evidence_keys: set[tuple[str, str]] = set()
    audit_count = 0
    evidence_request_count = 0

    try:
        raw_lines = audit_path.read_bytes().splitlines(keepends=True)
    except FileNotFoundError as exc:
        raise ActorEvidenceCausalityError(
            f"missing actor-evidence audit stream: {audit_path}"
        ) from exc
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            if strict:
                raise ActorEvidenceCausalityError(
                    f"blank audit record at {audit_path}:{line_number}"
                )
            continue
        envelope = _audit_envelope(
            audit_path,
            line_number=line_number,
            raw_line=raw_line,
            strict=strict,
        )
        audit_count += 1
        existing = observed.get(envelope.msg_id)
        if existing is None:
            observed[envelope.msg_id] = envelope
        if (
            envelope.in_reply_to is None
            and str(envelope.action.get("kind", "")) in registry.root_action_kinds
        ):
            audited_context_roots.add(envelope.msg_id)
        if envelope.msg_id in registrations:
            observed_registration_roots.add(envelope.msg_id)

        # ingest() enforces the same exact endpoint/action rules as live Runtime:
        # report actions must terminate at runtime:evidence, and that endpoint
        # cannot accept an unrelated action kind.
        binding = resolver.ingest(envelope)
        if binding is None:
            continue
        evidence_request_count += 1
        root_msg_id = _causal_root_msg_id(envelope, observed)
        if root_msg_id not in registrations:
            raise ActorEvidenceCausalityError(
                "accepted actor evidence has no persisted immutable root context"
            )
        key = (binding.actor_id, binding.idempotency_key)
        record = records_by_key.get(key)
        if record is None:
            raise ActorEvidenceCausalityError(
                "audited actor result has no accepted Runtime evidence record"
            )
        _verify_record_binding(
            record,
            request=envelope,
            binding=binding,
            registry=registry,
        )
        seen_evidence_keys.add(key)

    missing_contexts = sorted(audited_context_roots - set(registrations))
    if missing_contexts:
        raise ActorEvidenceCausalityError(
            "audited trusted roots have no persisted actor context: "
            + ", ".join(missing_contexts)
        )
    extra_contexts = sorted(set(registrations) - audited_context_roots)
    if extra_contexts:
        _verify_scoreable_termination_boundary(
            termination_path=termination_path,
            trace_path=audit_path.with_name(_TRACE_ARTIFACT),
            contexts_path=contexts_path,
            registry=registry,
            registrations=registrations,
            observed=observed,
        )
    if seen_evidence_keys != set(records_by_key):
        raise ActorEvidenceCausalityError(
            "actor evidence journal contains a report absent from audit"
        )
    return ActorEvidenceCausalityVerification(
        audit_envelopes=audit_count,
        registered_contexts=len(registrations),
        accepted_actor_evidence=len(records),
        context_roots_observed=len(observed_registration_roots),
        evidence_requests_replayed=evidence_request_count,
    )


def _verify_scoreable_termination_boundary(
    *,
    termination_path: Path | None,
    trace_path: Path,
    contexts_path: Path,
    registry: ActorContextRegistrations,
    registrations: Mapping[str, Any],
    observed: Mapping[str, Envelope],
) -> None:
    """Authorize declared but unexecuted roots only for one exact Agent abort.

    Registrations are frozen before execution, while Runtime delivers kickoff
    roots sequentially.  A scoreable model failure may therefore stop the
    episode before later declared roots reach the audit.  This exception to
    declaration/audit equality is valid only when the termination artifact is
    bound to the exact context bytes and to one canonical failure Tracker row
    whose causal root was actually audited.
    """

    if termination_path is None:
        raise ActorEvidenceCausalityError(
            "actor-context artifact contains roots absent from audit without "
            "a scoreable termination boundary"
        )
    try:
        same_root = (
            termination_path.parent.resolve()
            == trace_path.parent.resolve()
            == contexts_path.parent.resolve()
        )
    except OSError as exc:
        raise ActorEvidenceCausalityError(
            "cannot resolve actor-evidence termination boundary"
        ) from exc
    if not same_root:
        raise ActorEvidenceCausalityError(
            "actor-evidence termination artifacts must share one episode directory"
        )

    termination = _load_json_object(termination_path, label="termination")
    tracker = termination.get("tracker_binding")
    contexts_binding = termination.get("actor_contexts_binding")
    classification = termination.get("classification")
    if (
        termination.get("schema_version") != _TERMINATION_SCHEMA
        or termination.get("status") != "aborted"
        or termination.get("phase") != "runtime"
        or termination.get("scoreable") is not True
        or not isinstance(classification, str)
        or termination.get("stop_reason") != classification
        or not isinstance(tracker, Mapping)
        or not isinstance(contexts_binding, Mapping)
    ):
        raise ActorEvidenceCausalityError(
            "unexecuted actor contexts require a scoreable Tracker-bound termination"
        )

    try:
        contexts_raw = contexts_path.read_bytes()
    except OSError as exc:
        raise ActorEvidenceCausalityError(
            "cannot read actor-context declaration bytes"
        ) from exc
    expected_contexts_binding = {
        "artifact": _CONTEXTS_ARTIFACT,
        "bytes": len(contexts_raw),
        "sha256": hashlib.sha256(contexts_raw).hexdigest(),
    }
    if dict(contexts_binding) != expected_contexts_binding:
        raise ActorEvidenceCausalityError(
            "scoreable termination does not bind the exact actor-context registry"
        )

    terminal = tracker.get("terminal")
    expected_stop = _STOP_REASON_BY_FAILURE_TERMINAL.get(str(terminal))
    if (
        tracker.get("artifact") != _TRACE_ARTIFACT
        or tracker.get("run_id") != registry.episode_context_id
        or expected_stop is None
        or classification != expected_stop
    ):
        raise ActorEvidenceCausalityError(
            "scoreable termination has an invalid Tracker classification"
        )
    trace_row = _bound_failure_trace_row(trace_path, tracker=tracker)

    inbound_id = tracker.get("inbound_msg_id")
    agent_id = tracker.get("agent_id")
    inbound = observed.get(str(inbound_id))
    if inbound is None or inbound.to != agent_id:
        raise ActorEvidenceCausalityError(
            "scoreable termination is not bound to an audited actor inbound"
        )
    root_id = _causal_root_msg_id(inbound, dict(observed))
    context = registrations.get(root_id)
    if context is None or context.actor_id != agent_id:
        raise ActorEvidenceCausalityError(
            "scoreable termination has no executed immutable actor context"
        )
    if trace_row.get("decision_id") != inbound.idempotency_key:
        raise ActorEvidenceCausalityError(
            "scoreable termination decision contradicts its audited inbound"
        )


def _bound_failure_trace_row(
    path: Path,
    *,
    tracker: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise ActorEvidenceCausalityError(
            "scoreable termination is missing its Tracker artifact"
        ) from exc
    expected_sha = tracker.get("row_sha256")
    if not _is_sha256(expected_sha):
        raise ActorEvidenceCausalityError(
            "scoreable termination has an invalid Tracker row digest"
        )
    matches: list[dict[str, Any]] = []
    for ordinal, line in enumerate(lines, start=1):
        if not line.strip():
            raise ActorEvidenceCausalityError(
                f"blank Tracker record at {path}:{ordinal}"
            )
        row = _decode_json_object(line, label=f"Tracker row {ordinal}")
        canonical = json.dumps(
            row,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if line != canonical:
            raise ActorEvidenceCausalityError(
                f"non-canonical Tracker record at {path}:{ordinal}"
            )
        if hashlib.sha256(line).hexdigest() == expected_sha:
            matches.append(row)
    if len(matches) != 1:
        raise ActorEvidenceCausalityError(
            "scoreable termination does not bind exactly one Tracker row"
        )
    row = matches[0]
    terminal = tracker.get("terminal")
    chosen = row.get("chosen")
    identity_fields = (
        "run_id",
        "turn",
        "agent_id",
        "inbound_msg_id",
        "decision_id",
        "terminal",
    )
    if (
        any(row.get(field) != tracker.get(field) for field in identity_fields)
        or row.get("emitted_msg_id") is not None
        or row.get("forced_flush") is not False
        or row.get("incomplete") is not True
        or not isinstance(chosen, Mapping)
        or dict(chosen)
        != {
            "decision": terminal,
            "offer_id": None,
            "price": None,
            "rationale": None,
        }
    ):
        raise ActorEvidenceCausalityError(
            "scoreable termination Tracker row has contradictory failure metadata"
        )
    return row


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ActorEvidenceCausalityError(f"cannot read {label} artifact") from exc
    return _decode_json_object(raw, label=label)


def _decode_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ActorEvidenceCausalityError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise ActorEvidenceCausalityError(f"{label} artifact must be an object")
    return value


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _audit_envelope(
    path: Path,
    *,
    line_number: int,
    raw_line: bytes,
    strict: bool,
) -> Envelope:
    try:
        text = raw_line.decode("utf-8")
        row = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActorEvidenceCausalityError(
            f"invalid audit record at {path}:{line_number}: {exc}"
        ) from exc
    encoded = row.get("envelope") if isinstance(row, dict) else None
    if not isinstance(encoded, str):
        raise ActorEvidenceCausalityError(
            f"invalid audit envelope at {path}:{line_number}"
        )
    try:
        envelope = from_json(encoded)
    except Exception as exc:
        raise ActorEvidenceCausalityError(
            f"invalid audited envelope at {path}:{line_number}: {exc}"
        ) from exc
    if encoded != to_json(envelope):
        raise ActorEvidenceCausalityError(
            f"non-canonical audited envelope at {path}:{line_number}"
        )
    if strict:
        canonical_row = (
            json.dumps(row, allow_nan=False, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("utf-8")
        if raw_line != canonical_row:
            raise ActorEvidenceCausalityError(
                f"non-canonical audit record at {path}:{line_number}"
            )
    return envelope


def _causal_root_msg_id(
    report: Envelope,
    observed: dict[str, Envelope],
) -> str:
    current = report
    seen = {report.msg_id}
    while current.in_reply_to is not None:
        parent_id = current.in_reply_to
        if parent_id in seen:
            raise ActorEvidenceCausalityError(
                "cycle detected while locating actor-evidence root"
            )
        seen.add(parent_id)
        try:
            current = observed[parent_id]
        except KeyError as exc:  # resolver normally raises first; defensive join
            raise ActorEvidenceCausalityError(
                f"actor-evidence chain references absent parent {parent_id!r}"
            ) from exc
    return current.msg_id


def _verify_record_binding(
    record: dict[str, Any],
    *,
    request: Envelope,
    binding: ActorResultBinding,
    registry: ActorContextRegistrations,
) -> None:
    expected: dict[str, Any] = {
        "run_id": registry.episode_context_id,
        "actor_id": binding.actor_id,
        "principal_id": binding.principal_id,
        "action_kind": binding.action_kind,
        "idempotency_key": binding.idempotency_key,
        "in_reply_to": binding.in_reply_to,
        "task_id": binding.task_id,
        "mandate_id": binding.mandate_id,
        "context_id": binding.context_id,
        "request_msg_id": request.msg_id,
        "request_envelope_sha256": hashlib.sha256(
            to_json(request).encode("utf-8")
        ).hexdigest(),
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ActorEvidenceCausalityError(
                f"actor evidence does not match causal binding field {field!r}"
            )


__all__ = [
    "ActorEvidenceCausalityError",
    "ActorEvidenceCausalityVerification",
    "verify_actor_evidence_causality",
]
