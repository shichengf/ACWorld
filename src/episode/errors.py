"""Exception hierarchy for the episode package."""

from __future__ import annotations


class EpisodeError(Exception):
    """Base error for the episode package."""


class ScenarioInvalid(EpisodeError):
    """The scenario file failed validation (missing required field, bad types, …)."""


class KickoffMissing(EpisodeError):
    """The scenario did not declare a ``kickoff_envelope`` and no agent emitted one."""


class TerminalNotReached(EpisodeError):
    """The runtime turn loop exited without observing a terminal action.

    Reported in the scenario report rather than re-raised in production runs;
    raised in tests that assert termination.
    """
