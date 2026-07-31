"""Typed, sanitized evidence for failures inside one agent decision.

This module is deliberately part of the CommerceWorld runtime rather than an
experiment runner.  A benchmark may score a failed model decision only when
the failure happened inside a real agent turn and the runtime wrote a causal
Tracker record for that turn.  Provider and network failures are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


TURN_FAILURE_WIRE_SCHEMA = "cwe.agent-turn-failure.v1"

PROTOCOL_ERROR_TERMINAL = "protocol_error"
CAPABILITY_ERROR_TERMINAL = "capability_error"
SECURITY_ERROR_TERMINAL = "security_error"
RESOURCE_LIMIT_TERMINAL = "resource_limit"
BUDGET_EXCEEDED_TERMINAL = "budget_exceeded"

SCOREABLE_FAILURE_TERMINALS = frozenset(
    {
        PROTOCOL_ERROR_TERMINAL,
        RESOURCE_LIMIT_TERMINAL,
        SECURITY_ERROR_TERMINAL,
    }
)


@dataclass(frozen=True)
class AgentTurnFailureClass:
    """Public classification and Tracker terminal for one safe failure."""

    classification: str
    stop_reason: str
    terminal: str


@dataclass(frozen=True)
class AgentTurnTraceIdentity:
    """Identity of the Tracker row written before an exception is re-raised."""

    turn: int
    agent_id: str
    inbound_msg_id: str
    decision_id: str
    terminal: str


class RemoteAgentTurnFailure(RuntimeError):
    """Sanitized reconstruction of a scoreable failure returned over VCP.

    The remote endpoint never sends the original exception message.  These
    fields are sufficient for termination classification and fingerprinting;
    the authoritative evidence remains the dispatcher-owned Tracker sidecar.
    """

    def __init__(
        self,
        *,
        failure_class: AgentTurnFailureClass,
        exception_type: str,
        exception_module: str,
        error_code: str | None,
        message_chars: int,
        message_sha256: str,
    ) -> None:
        super().__init__("remote agent turn failed")
        self.failure_class = failure_class
        self.remote_exception_type = exception_type
        self.remote_exception_module = exception_module
        self.error_code = error_code
        self.remote_message_chars = message_chars
        self.remote_message_sha256 = message_sha256


_TRACE_IDENTITY_ATTR = "_commerceworld_agent_turn_trace_identity"


def classify_agent_turn_exception(exc: Exception) -> AgentTurnFailureClass | None:
    """Classify only failures safe to attribute to the current agent turn.

    In particular, a model-provider timeout or HTTP error is infrastructure.
    It must never be converted into a low benchmark score merely because the
    channel error inherits from :class:`AgentError`.
    """

    from agents.inference import (
        ChannelTransportError,
        ModelDecisionParseError,
    )
    from agents.decision_errors import FrameworkAuthorityError, ModelBusinessDecisionError
    from protocol.envelopes import MandateLeak
    from runtime.errors import (
        PrivateUtilityLeak,
    )

    if isinstance(exc, RemoteAgentTurnFailure):
        try:
            return _failure_class_from_public_fields(
                exc.failure_class.classification,
                exc.failure_class.stop_reason,
                exc.failure_class.terminal,
            )
        except ValueError:
            return None
    if (
        type(exc).__module__ == "experiments.benchmark_executor"
        and type(exc).__name__ == "ModelCallLimitExceeded"
        and getattr(exc, "error_code", None) == RESOURCE_LIMIT_TERMINAL
    ):
        return AgentTurnFailureClass(
            classification=RESOURCE_LIMIT_TERMINAL,
            stop_reason=RESOURCE_LIMIT_TERMINAL,
            terminal=RESOURCE_LIMIT_TERMINAL,
        )
    if isinstance(exc, ChannelTransportError):
        return None
    # The local high-level Agent boundary deliberately separates model business
    # choices from framework authority. Only the former is a scoreable model
    # protocol failure.  Missing Platform/World authority is an environment
    # defect and must escape to the experiment runner as infrastructure.
    if isinstance(exc, FrameworkAuthorityError):
        return None
    if isinstance(exc, ModelBusinessDecisionError):
        return AgentTurnFailureClass(
            classification="model_protocol_error",
            stop_reason="model_protocol_error",
            terminal=PROTOCOL_ERROR_TERMINAL,
        )
    if isinstance(exc, (PrivateUtilityLeak, MandateLeak)):
        return AgentTurnFailureClass(
            classification="security_guard",
            stop_reason="security_guard",
            terminal=SECURITY_ERROR_TERMINAL,
        )
    if isinstance(exc, ModelDecisionParseError):
        if not exc.provider_response_received:
            # A channel that returned the wrong Python object, or a local
            # wrapper that failed before producing hashes for actual model
            # text, violated the Provider/Agent interface.  It is
            # infrastructure, not evidence of a bad model decision.
            return None
        return AgentTurnFailureClass(
            classification="model_protocol_error",
            stop_reason="model_protocol_error",
            terminal=PROTOCOL_ERROR_TERMINAL,
        )
    return None


def tracker_terminal_for_exception(exc: Exception, *, current: str | None) -> str | None:
    """Return only a typed model-attributable failure terminal."""

    del current
    failure = classify_agent_turn_exception(exc)
    if failure is None:
        return None
    return failure.terminal


def attach_trace_identity(
    exc: Exception,
    *,
    turn: int,
    agent_id: str,
    inbound_msg_id: str,
    decision_id: str,
    terminal: str,
) -> None:
    """Bind an already-written Tracker row to the propagated exception."""

    identity = AgentTurnTraceIdentity(
        turn=turn,
        agent_id=agent_id,
        inbound_msg_id=inbound_msg_id,
        decision_id=decision_id,
        terminal=terminal,
    )
    try:
        setattr(exc, _TRACE_IDENTITY_ATTR, identity)
    except Exception:
        # An exotic immutable exception cannot carry causal evidence.  It will
        # consequently fail closed as unbound infrastructure at termination.
        return


def trace_identity(exc: Exception) -> AgentTurnTraceIdentity | None:
    value = getattr(exc, _TRACE_IDENTITY_ATTR, None)
    return value if isinstance(value, AgentTurnTraceIdentity) else None


def sanitized_failure_wire(
    exc: Exception,
    *,
    identity: AgentTurnTraceIdentity,
) -> dict[str, Any]:
    """Serialize a scoreable remote turn failure without its message."""

    import hashlib

    failure = classify_agent_turn_exception(exc)
    if failure is None:
        raise ValueError("non-scoreable exception has no agent-turn failure wire form")
    message = str(exc)
    code = _safe_code(getattr(exc, "error_code", None))
    return {
        "schema_version": TURN_FAILURE_WIRE_SCHEMA,
        "classification": failure.classification,
        "stop_reason": failure.stop_reason,
        "terminal": identity.terminal,
        "identity": {
            "turn": identity.turn,
            "agent_id": identity.agent_id,
            "inbound_msg_id": identity.inbound_msg_id,
            "decision_id": identity.decision_id,
        },
        "exception": {
            "type": _safe_identifier(type(exc).__name__, default="Exception", allow_dot=False),
            "module": _safe_identifier(type(exc).__module__, default="unknown", allow_dot=True),
            "code": code,
            "message_chars": len(message),
            "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        },
    }


def remote_failure_from_wire(
    value: object,
    *,
    trace: Mapping[str, Any],
    delivered_to: str,
    delivered_inbound_msg_id: str,
    delivered_kind: str,
) -> RemoteAgentTurnFailure:
    """Validate a VCP failure sidecar against its exact returned Tracker row."""

    if not isinstance(value, Mapping) or value.get("schema_version") != TURN_FAILURE_WIRE_SCHEMA:
        raise ValueError("remote agent failure has an invalid schema")
    identity = value.get("identity")
    exception = value.get("exception")
    if not isinstance(identity, Mapping) or not isinstance(exception, Mapping):
        raise ValueError("remote agent failure is missing identity metadata")
    terminal = value.get("terminal")
    classification = value.get("classification")
    stop_reason = value.get("stop_reason")
    expected = _failure_class_from_public_fields(classification, stop_reason, terminal)
    turn = identity.get("turn")
    agent_id = identity.get("agent_id")
    inbound_msg_id = identity.get("inbound_msg_id")
    decision_id = identity.get("decision_id")
    if (
        isinstance(turn, bool)
        or not isinstance(turn, int)
        or turn < 0
        or not all(
            isinstance(item, str) and item
            for item in (
                agent_id,
                inbound_msg_id,
                decision_id,
            )
        )
        or agent_id != delivered_to
        or trace.get("turn") != turn
        or trace.get("agent_id") != agent_id
        or trace.get("inbound_msg_id") != inbound_msg_id
        or trace.get("decision_id") != decision_id
        or trace.get("terminal") != terminal
        or trace.get("emitted_msg_id") is not None
        or trace.get("forced_flush") is not False
        or trace.get("incomplete") is not True
    ):
        raise ValueError("remote agent failure contradicts its Tracker identity")
    if delivered_kind != "world.response":
        if inbound_msg_id != delivered_inbound_msg_id:
            raise ValueError("remote agent failure is bound to another inbound envelope")
    elif not _trace_claims_world_response(trace, delivered_inbound_msg_id):
        raise ValueError("remote agent failure does not claim the delivered World response")
    chosen = trace.get("chosen")
    if not isinstance(chosen, Mapping) or dict(chosen) != {
        "decision": terminal,
        "offer_id": None,
        "price": None,
        "rationale": None,
    }:
        raise ValueError("remote agent failure has contradictory choice metadata")

    exc_type = exception.get("type")
    exc_module = exception.get("module")
    code = exception.get("code")
    chars = exception.get("message_chars")
    digest = exception.get("message_sha256")
    if (
        not isinstance(exc_type, str)
        or not exc_type
        or not isinstance(exc_module, str)
        or not exc_module
        or not (code is None or isinstance(code, str))
        or isinstance(chars, bool)
        or not isinstance(chars, int)
        or chars < 0
        or not _is_sha256(digest)
    ):
        raise ValueError("remote agent failure has invalid exception metadata")
    return RemoteAgentTurnFailure(
        failure_class=expected,
        exception_type=exc_type,
        exception_module=exc_module,
        error_code=code,
        message_chars=chars,
        message_sha256=digest,
    )


def failure_trace_dict(trace: Mapping[str, Any], *, terminal: str) -> dict[str, Any]:
    """Convert a not-yet-sidecarred successful remote trace to a rejected turn."""

    if terminal not in SCOREABLE_FAILURE_TERMINALS:
        raise ValueError(f"unknown failure terminal {terminal!r}")
    out = dict(trace)
    out.update(
        {
            "emitted_msg_id": None,
            "terminal": terminal,
            "chosen": {
                "decision": terminal,
                "offer_id": None,
                "price": None,
                "rationale": None,
            },
            "forced_flush": False,
            "incomplete": True,
        }
    )
    return out


def _failure_class_from_public_fields(
    classification: object,
    stop_reason: object,
    terminal: object,
) -> AgentTurnFailureClass:
    candidates = {
        ("model_protocol_error", "model_protocol_error", PROTOCOL_ERROR_TERMINAL),
        (RESOURCE_LIMIT_TERMINAL, RESOURCE_LIMIT_TERMINAL, RESOURCE_LIMIT_TERMINAL),
        ("security_guard", "security_guard", SECURITY_ERROR_TERMINAL),
    }
    triple = (classification, stop_reason, terminal)
    if triple not in candidates:
        raise ValueError("remote agent failure classification is inconsistent")
    return AgentTurnFailureClass(
        classification=str(classification),
        stop_reason=str(stop_reason),
        terminal=str(terminal),
    )


def _trace_claims_world_response(trace: Mapping[str, Any], response_msg_id: str) -> bool:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, Mapping) or not isinstance(step.get("data"), Mapping):
            continue
        data = step["data"]
        if step.get("kind") == "tool_call" and isinstance(data.get("results"), list):
            if any(
                isinstance(result, Mapping) and result.get("source_msg_id") == response_msg_id
                for result in data["results"]
            ):
                return True
        if step.get("kind") == "framework_authority_prerequisite" and isinstance(
            data.get("source_msg_ids"), list
        ):
            if response_msg_id in data["source_msg_ids"]:
                return True
    return False


def _safe_identifier(value: object, *, default: str, allow_dot: bool) -> str:
    text = value if isinstance(value, str) else ""
    punctuation = "._" if allow_dot else "_"
    if (
        text
        and len(text) <= 128
        and text[0].isascii()
        and (text[0].isalpha() or text[0] == "_")
        and all(char.isascii() and (char.isalnum() or char in punctuation) for char in text)
    ):
        return text
    return default


def _safe_code(value: object) -> str | None:
    if (
        isinstance(value, str)
        and value
        and len(value) <= 64
        and all(char.isascii() and (char.isalnum() or char in "._-") for char in value)
    ):
        return value
    return None


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


__all__ = [
    "AgentTurnFailureClass",
    "AgentTurnTraceIdentity",
    "BUDGET_EXCEEDED_TERMINAL",
    "CAPABILITY_ERROR_TERMINAL",
    "PROTOCOL_ERROR_TERMINAL",
    "RESOURCE_LIMIT_TERMINAL",
    "RemoteAgentTurnFailure",
    "SCOREABLE_FAILURE_TERMINALS",
    "SECURITY_ERROR_TERMINAL",
    "TURN_FAILURE_WIRE_SCHEMA",
    "attach_trace_identity",
    "classify_agent_turn_exception",
    "failure_trace_dict",
    "remote_failure_from_wire",
    "sanitized_failure_wire",
    "trace_identity",
    "tracker_terminal_for_exception",
]
