"""Exception hierarchy for the runtime package.

Named ``BusError`` (not ``RuntimeError``) to avoid clashing with the stdlib
builtin of the same name.
"""

from __future__ import annotations


class BusError(Exception):
    """Base error for the runtime package."""


class AgentNotRegistered(BusError):
    """An envelope was sent to ``env.to`` but no agent is registered at that id.

    Per WORLD_CLASSES.md §9 Q4, this aborts the run with
    ``protocol_correctness = False``; no silent drops.
    """


class DuplicateAgentId(BusError):
    """An ``Agent`` with the same id was already registered."""


class ActorEvidenceUnavailable(BusError):
    """An actor report reached Runtime without its trusted acceptance service."""


class BudgetExceeded(BusError):
    """An agent's per-turn token or latency budget was exceeded."""


class PrivateUtilityLeak(BusError):
    """An outbound envelope's payload would disclose a registered private secret
    across a counterparty boundary.

    Per AGENT_CLASSES.md §4, this is the one non-negotiable invariant; a violation
    aborts the run with ``protocol_correctness = False``.

    Carries an optional sanitized ``finding`` (a :class:`runtime.privacy.LeakFinding`)
    so the bus can write a security-event sidecar record WITHOUT parsing the
    exception string. ``finding`` is ``None`` for legacy raises.
    """

    def __init__(self, message: str = "", *, finding: object | None = None) -> None:
        super().__init__(message)
        self.finding = finding
