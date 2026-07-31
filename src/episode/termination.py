"""Shared, sanitized classification for scoreable Episode early termination."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from runtime.turn_failure import (
    RemoteAgentTurnFailure,
    classify_agent_turn_exception,
    trace_identity,
)


EPISODE_TERMINATION_ARTIFACT = "termination.json"
EPISODE_TERMINATION_SCHEMA = "cwe.episode-termination.v1"
ACTOR_CONTEXTS_ARTIFACT = "actor.contexts.json"

RESOURCE_LIMIT_STOP_REASON = "resource_limit"
MODEL_PROTOCOL_STOP_REASON = "model_protocol_error"
CAPABILITY_GUARD_STOP_REASON = "capability_guard"
SECURITY_GUARD_STOP_REASON = "security_guard"
UNEXPECTED_RUNTIME_CLASSIFICATION = "unexpected_runtime_error"
SCOREABLE_STOP_REASONS = frozenset({
    RESOURCE_LIMIT_STOP_REASON,
    MODEL_PROTOCOL_STOP_REASON,
    SECURITY_GUARD_STOP_REASON,
})
_STOP_REASON_BY_TERMINAL = {
    "protocol_error": MODEL_PROTOCOL_STOP_REASON,
    "security_error": SECURITY_GUARD_STOP_REASON,
    "resource_limit": RESOURCE_LIMIT_STOP_REASON,
}


def write_termination_artifact(out_dir: Path, exc: Exception) -> None:
    """Persist a machine-readable abort marker without raw exception content."""

    classification, stop_reason = _classify_runtime_abort(exc)
    tracker_binding = None
    if stop_reason is not None:
        tracker_binding = _resolve_tracker_binding(out_dir, exc)
        if tracker_binding is None:
            # A scoreable stop without a causal, already-written Tracker row is
            # infrastructure.  Do not silently convert it to a model score.
            classification = UNEXPECTED_RUNTIME_CLASSIFICATION
            stop_reason = None
    payload = {
        "schema_version": EPISODE_TERMINATION_SCHEMA,
        "status": "aborted",
        "phase": "runtime",
        "classification": classification,
        "stop_reason": stop_reason,
        "scoreable": stop_reason is not None,
        "exception": _sanitized_exception(exc),
    }
    if tracker_binding is not None:
        payload["tracker_binding"] = tracker_binding
    actor_contexts_binding = _artifact_binding(
        out_dir / ACTOR_CONTEXTS_ARTIFACT,
        artifact=ACTOR_CONTEXTS_ARTIFACT,
    )
    if actor_contexts_binding is not None:
        # The registry is frozen before the first kickoff is delivered.  Bind
        # the abort marker to those exact declaration bytes so offline evidence
        # can distinguish a legitimate unexecuted kickoff from a context added
        # after the failure.
        payload["actor_contexts_binding"] = actor_contexts_binding
    (out_dir / EPISODE_TERMINATION_ARTIFACT).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_verified_scoreable_termination(out_dir: str | Path) -> dict[str, Any] | None:
    """Return a strictly Tracker-bound scoreable termination, if present.

    Episode evidence uses this before its manifest exists.  It therefore
    verifies the immutable Tracker row directly rather than trusting a boolean
    in ``termination.json``.  An absent or explicitly unscoreable termination
    returns ``None``; malformed scoreable claims fail closed.
    """

    root = Path(out_dir)
    path = root / EPISODE_TERMINATION_ARTIFACT
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("episode termination artifact is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("episode termination artifact must be an object")
    if payload.get("scoreable") is not True:
        return None
    binding = payload.get("tracker_binding")
    classification = payload.get("classification")
    stop_reason = payload.get("stop_reason")
    if (
        payload.get("schema_version") != EPISODE_TERMINATION_SCHEMA
        or payload.get("status") != "aborted"
        or payload.get("phase") != "runtime"
        or not isinstance(classification, str)
        or stop_reason not in SCOREABLE_STOP_REASONS
        or classification != stop_reason
        or not isinstance(binding, dict)
        or binding.get("artifact") != "audit.trace.jsonl"
    ):
        raise ValueError("scoreable episode termination is malformed")
    terminal = binding.get("terminal")
    if _STOP_REASON_BY_TERMINAL.get(terminal) != stop_reason:
        raise ValueError("scoreable episode termination contradicts its terminal")
    expected_identity = {
        key: binding.get(key)
        for key in (
            "run_id",
            "turn",
            "agent_id",
            "inbound_msg_id",
            "decision_id",
            "terminal",
        )
    }
    if (
        isinstance(expected_identity["turn"], bool)
        or not isinstance(expected_identity["turn"], int)
        or expected_identity["turn"] < 0
        or not all(
            isinstance(expected_identity[key], str) and expected_identity[key]
            for key in (
                "run_id",
                "agent_id",
                "inbound_msg_id",
                "decision_id",
                "terminal",
            )
        )
    ):
        raise ValueError("scoreable episode termination identity is malformed")
    row_sha256 = binding.get("row_sha256")
    if (
        not isinstance(row_sha256, str)
        or len(row_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in row_sha256)
    ):
        raise ValueError("scoreable episode termination row digest is malformed")
    try:
        raw_lines = (root / "audit.trace.jsonl").read_bytes().splitlines()
    except OSError as exc:
        raise ValueError("scoreable episode termination has no Tracker artifact") from exc
    matches: list[dict[str, Any]] = []
    for raw_line in raw_lines:
        if hashlib.sha256(raw_line).hexdigest() != row_sha256:
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("scoreable termination Tracker row is malformed") from exc
        if isinstance(row, dict):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError("scoreable termination Tracker binding is not unique")
    row = matches[0]
    if any(row.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("scoreable termination Tracker identity does not match")
    chosen = row.get("chosen")
    if (
        row.get("emitted_msg_id") is not None
        or row.get("forced_flush") is not False
        or row.get("incomplete") is not True
        or not isinstance(chosen, dict)
        or chosen
        != {
            "decision": terminal,
            "offer_id": None,
            "price": None,
            "rationale": None,
        }
    ):
        raise ValueError("scoreable termination Tracker row is not a no-emit failure")
    return payload


def _safe_exception_identifier(
    value: object,
    *,
    default: str,
    allow_dot: bool,
) -> str:
    text = value if isinstance(value, str) else ""
    punctuation = "._" if allow_dot else "_"
    if (
        text
        and len(text) <= 128
        and text[0].isascii()
        and (text[0].isalpha() or text[0] == "_")
        and all(
            char.isascii() and (char.isalnum() or char in punctuation)
            for char in text
        )
    ):
        return text
    return default


def _classify_runtime_abort(exc: Exception) -> tuple[str, str | None]:
    """Return ``(classification, scoreable stop_reason)`` for one abort."""

    failure = classify_agent_turn_exception(exc)
    if failure is not None:
        return failure.classification, failure.stop_reason
    return UNEXPECTED_RUNTIME_CLASSIFICATION, None


def _resolve_tracker_binding(out_dir: Path, exc: Exception) -> dict[str, Any] | None:
    identity = trace_identity(exc)
    if identity is None:
        return None
    path = out_dir / "audit.trace.jsonl"
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    matches: list[tuple[dict[str, Any], bytes]] = []
    for raw_line in raw.splitlines():
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(row, dict):
            return None
        if (
            row.get("turn") == identity.turn
            and row.get("agent_id") == identity.agent_id
            and row.get("inbound_msg_id") == identity.inbound_msg_id
            and row.get("decision_id") == identity.decision_id
            and row.get("terminal") == identity.terminal
        ):
            matches.append((row, raw_line))
    if len(matches) != 1:
        return None
    row, raw_line = matches[0]
    chosen = row.get("chosen")
    if (
        row.get("emitted_msg_id") is not None
        or row.get("forced_flush") is not False
        or row.get("incomplete") is not True
        or not isinstance(chosen, dict)
        or chosen.get("decision") != identity.terminal
        or not isinstance(row.get("run_id"), str)
        or not row.get("run_id")
    ):
        return None
    return {
        "artifact": "audit.trace.jsonl",
        "run_id": row["run_id"],
        "turn": identity.turn,
        "agent_id": identity.agent_id,
        "inbound_msg_id": identity.inbound_msg_id,
        "decision_id": identity.decision_id,
        "terminal": identity.terminal,
        "row_sha256": hashlib.sha256(raw_line).hexdigest(),
    }


def _artifact_binding(path: Path, *, artifact: str) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return {
        "artifact": artifact,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _sanitized_exception(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, RemoteAgentTurnFailure):
        return {
            "type": _safe_exception_identifier(
                exc.remote_exception_type, default="Exception", allow_dot=False
            ),
            "module": _safe_exception_identifier(
                exc.remote_exception_module, default="unknown", allow_dot=True
            ),
            "code": _safe_code(exc.error_code),
            "message_chars": exc.remote_message_chars,
            "message_sha256": exc.remote_message_sha256,
        }
    message = str(exc)
    return {
        "type": _safe_exception_identifier(
            type(exc).__name__, default="Exception", allow_dot=False
        ),
        "module": _safe_exception_identifier(
            type(exc).__module__, default="unknown", allow_dot=True
        ),
        "code": _safe_code(getattr(exc, "error_code", None)),
        "message_chars": len(message),
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
    }


def _safe_code(value: object) -> str | None:
    if (
        isinstance(value, str)
        and value
        and len(value) <= 64
        and all(char.isascii() and (char.isalnum() or char in "._-") for char in value)
    ):
        return value
    return None


__all__ = [
    "CAPABILITY_GUARD_STOP_REASON",
    "EPISODE_TERMINATION_ARTIFACT",
    "EPISODE_TERMINATION_SCHEMA",
    "MODEL_PROTOCOL_STOP_REASON",
    "RESOURCE_LIMIT_STOP_REASON",
    "SCOREABLE_STOP_REASONS",
    "SECURITY_GUARD_STOP_REASON",
    "UNEXPECTED_RUNTIME_CLASSIFICATION",
    "load_verified_scoreable_termination",
    "write_termination_artifact",
]
