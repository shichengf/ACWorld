"""Exception hierarchy for the evals package."""

from __future__ import annotations


class EvalError(Exception):
    """Base error for the evals package."""


class MetricUndefined(EvalError):
    """A named metric was requested but is not registered."""


class ProbeConfigError(EvalError):
    """A probe was instantiated with a config it cannot run (e.g. wrong scenario family)."""
