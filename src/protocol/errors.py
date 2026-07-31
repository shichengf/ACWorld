"""Exception hierarchy for the protocol package."""

from __future__ import annotations


class ProtocolError(Exception):
    """Base error for the protocol package."""


class SchemaError(ProtocolError):
    """An envelope failed schema validation (missing field, wrong type, …)."""


class PartitionViolation(ProtocolError):
    """A sender attempted to emit an action it is not permitted to send.

    Raised by the router on the basis of the partition allow-table in
    ``protocol.actions``.
    """


class IdempotencyConflict(ProtocolError):
    """A repeat envelope reused an idempotency key with a different payload."""


class UnknownActionKind(ProtocolError):
    """The envelope's ``action.kind`` is not in the registered ``ActionKind`` enum."""
