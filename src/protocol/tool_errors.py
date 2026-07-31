"""Typed, side-effect-free error replies for re-entrant Agent tools.

An actor-scoped World read can be a valid protocol request and still be
denied by row ownership.  That is a normal tool outcome, not a Runtime
failure.  This small wire shape lets World answer the audited request and
resume the parked Agent turn without exposing the protected row.
"""

from __future__ import annotations

from typing import Any, Mapping


WORLD_TOOL_ERROR_KEY = "_commerceworld_tool_error"


def world_tool_error_payload(*, error_type: str, message: str) -> dict[str, object]:
    """Build the exact payload used by a denied, read-only World tool call."""

    return {
        WORLD_TOOL_ERROR_KEY: {
            "type": error_type,
            "message": message,
        }
    }


def world_tool_error_text(payload: Any) -> str | None:
    """Return a stable tool-result error string, or ``None`` for normal data."""

    if not isinstance(payload, Mapping) or set(payload) != {WORLD_TOOL_ERROR_KEY}:
        return None
    error = payload.get(WORLD_TOOL_ERROR_KEY)
    if not isinstance(error, Mapping) or set(error) != {"type", "message"}:
        return None
    error_type = error.get("type")
    message = error.get("message")
    if not isinstance(error_type, str) or not error_type:
        return None
    if not isinstance(message, str) or not message:
        return None
    return f"{error_type}: {message}"


__all__ = [
    "WORLD_TOOL_ERROR_KEY",
    "world_tool_error_payload",
    "world_tool_error_text",
]
