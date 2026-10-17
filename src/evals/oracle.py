"""The default ``Oracle`` implementation: state-diff plus the core metrics.

The oracle is a pure function over ``(scenario, audit_log, final_state)``.
Server-side only — never injected into agent prompts.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from episode.types import Report, ScenarioSpec
    from protocol.envelope import Envelope
    from world.types import WorldSnapshot


def score(
    scenario: "ScenarioSpec",
    audit: "Iterable[Envelope]",
    final_state: "WorldSnapshot",
) -> "Report":
    """Return the per-scenario report.

    Fills the core metrics (task_success, budget_adherence, merchant_margin,
    state_correctness, protocol_correctness). Report fields are additive.
    """
    raise NotImplementedError(
        "settled = find_settle(audit); compute each metric in metrics.py; "
        "assemble Report(scenario_id, seed, **metric_results)"
    )


def find_settle(audit: "Iterable[Envelope]") -> "Envelope | None":
    """Return the first ``settle`` envelope in ``audit``, or ``None`` if absent."""
    raise NotImplementedError(
        "for env in audit: if env.action['kind'] == 'settle': return env; return None"
    )


def count_violations(audit: "Iterable[Envelope]") -> int:
    """Count envelopes rejected by the runtime for partition or schema reasons.

    Drives ``protocol_correctness``: zero violations -> True.
    """
    raise NotImplementedError(
        "scan audit for envelopes whose outcome_tag in {'partition_violation', 'schema_error'}"
    )
