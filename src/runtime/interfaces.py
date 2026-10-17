"""Cross-package contracts the runtime layer exposes.

The ``Bus`` Protocol is the single seam that lets a cross-process transport
drop in without lane changes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agents.base import Agent
    from protocol.envelope import Envelope
    from runtime.types import AuditRecord, InferenceBudget


class Bus(Protocol):
    """The send/register surface. ``Runtime`` is the in-process implementation;
    a cross-process transport implements the same protocol.
    """

    def register(self, agent: "Agent", *, budget: "InferenceBudget | None" = None) -> None:
        """Register ``agent`` and validate its declared action partitions."""
        ...

    def unregister(self, agent_id: str) -> None:
        """Remove the agent from the registry; retire its inference channel."""
        ...

    def send(self, env: "Envelope") -> None:
        """Deliver ``env`` to ``env.to``. May trigger 0+ further sends."""
        ...

    def run(self) -> None:
        """Drain the queue until terminal action observed or quiesced."""
        ...


class Router(Protocol):
    """Role-partition + private-utility guard. Pure function over envelopes."""

    def check_outbound(self, env: "Envelope", sender_id: str) -> None:
        """Raise ``PartitionViolation`` if ``sender_id`` may not emit ``env.action.kind``."""
        ...

    def check_payload(self, env: "Envelope", sender_id: str) -> None:
        """Raise ``PrivateUtilityLeak`` if the payload contains private-utility memory."""
        ...


class AuditWriter(Protocol):
    """Append-only audit sink. Field semantics live in ``runtime.types.AuditRecord``."""

    def append(self, record: "AuditRecord") -> None:
        """Append one record to the on-disk audit log."""
        ...

    def replay(self) -> Iterator["Envelope"]:
        """Yield envelopes in committed order. Drives the replay verifier."""
        ...
