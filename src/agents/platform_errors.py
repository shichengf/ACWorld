"""Typed, sanitized outcomes for actor requests rejected by Platform policy.

The exception is internal control flow between a concrete Platform policy and
``PlatformService.handle``.  It means that the actor emitted a recognizable
commerce action, but authoritative business state rejected that action.  The
Platform records one deterministic rejected decision and the Runtime continues;
this is benchmark evidence, not an infrastructure failure.

Only narrow, actor-facing policy boundaries may create this marker.  Schema
errors, corrupt World responses, privacy violations, transport failures, and
generic ``WorldError`` values must continue to propagate.
"""

from __future__ import annotations

from typing import Literal


RejectionCategory = Literal[
    "business",
    "conflict",
    "identity",
    "stale",
    "state",
]


class CommerceActionRejected(Exception):
    """A recognized actor action was denied by a deterministic commerce rule."""

    def __init__(
        self,
        reason_code: str,
        *,
        category: RejectionCategory,
        source_error_type: str,
    ) -> None:
        if not reason_code or not reason_code.replace("_", "").isalnum():
            raise ValueError("commerce rejection reason_code must be stable snake case")
        if not source_error_type or not source_error_type.isidentifier():
            raise ValueError("commerce rejection source_error_type must be an identifier")
        self.reason_code = reason_code
        self.category = category
        self.source_error_type = source_error_type
        super().__init__(reason_code)


__all__ = ["CommerceActionRejected", "RejectionCategory"]
