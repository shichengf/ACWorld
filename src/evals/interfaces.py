"""Cross-package contracts the evals layer exposes.

The probe interface is wire-stable. All probes satisfy ``Probe`` so a single
runner drives them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from episode.types import Report, ScenarioSpec
    from evals.types import MetricResult, ProbeReport
    from protocol.envelope import Envelope
    from world.types import WorldSnapshot


class Metric(Protocol):
    """One named metric. Pure function over (scenario, audit, snapshot)."""

    name: str

    def compute(
        self,
        scenario: "ScenarioSpec",
        audit: "Iterable[Envelope]",
        final_state: "WorldSnapshot",
    ) -> "MetricResult":
        """Return the metric's value plus the audit pointer that justifies it."""
        ...


class Probe(Protocol):
    """One ablation probe.

    The probe interface is wire-stable: new probes can be added without
    changing the contract.
    """

    probe_id: str

    def applies_to(self, scenario: "ScenarioSpec") -> bool:
        """Return ``True`` iff this probe is meaningful for ``scenario``."""
        ...

    def run(
        self,
        scenarios: "Iterable[ScenarioSpec]",
        *,
        n_seeds: int,
    ) -> "ProbeReport":
        """Execute the probe over the scenarios and return an aggregated report."""
        ...


class Reporter(Protocol):
    """Write per-scenario reports to disk."""

    def write(self, report: "Report", *, run_id: str) -> None:
        """Persist ``report`` under ``reports/<run_id>/<scenario_id>.json``."""
        ...
