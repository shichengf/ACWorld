"""Cross-package contracts the protocol layer exposes.

Anything that wants to validate or sign envelopes depends on these Protocols,
not on a concrete class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from protocol.envelope import Envelope


class EnvelopeValidator(Protocol):
    """Schema + partition validation for envelopes.

    Implementations:
        * ``protocol.envelope.validate`` — pure schema check.
        * ``runtime.router.Router`` — schema + role-partition + private-utility.
    """

    def validate(self, env: "Envelope") -> None:
        """Raise ``SchemaError`` / ``PartitionViolation`` / ``UnknownActionKind``
        if ``env`` is invalid; return ``None`` otherwise.
        """
        ...


class EnvelopeSigner(Protocol):
    """Optional envelope signing.

    Off by default; enabled per scenario for hostile-cell (A3) adversarials.
    """

    def sign(self, env: "Envelope") -> "Envelope":
        """Return ``env`` with the ``signature`` field populated."""
        ...

    def verify(self, env: "Envelope") -> bool:
        """Return ``True`` iff ``env.signature`` validates against ``env.from_``."""
        ...
