"""Cross-package contracts the episode layer exposes.

The scenario file shape is wire-stable. Anything that loads scenarios depends
on ``ScenarioLoader``; anything that scores them depends on ``Oracle``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agents.base import Agent
    from episode.types import Report, ScenarioSpec
    from protocol.envelope import Envelope
    from world.types import WorldSnapshot


class ScenarioLoader(Protocol):
    """Hydrate ``ScenarioSpec`` from on-disk YAML.

    The default implementation lives in ``episode.scenario.ScenarioSpec.from_yaml``.
    A future ``src/scenarios/`` package can drop in a richer loader (with
    generator-driven YAML synthesis) without touching ``Episode``.
    """

    def load(self, path: Path) -> "ScenarioSpec":
        """Parse one YAML file into a ``ScenarioSpec``."""
        ...

    def load_all(self, root: Path) -> "Iterable[ScenarioSpec]":
        """Recursively load every YAML under ``root`` in deterministic order."""
        ...


class Oracle(Protocol):
    """Score a finished episode.

    The default implementation is ``evals.oracle.score``; lane owners (and probes)
    plug in alternate oracles by satisfying this Protocol.
    """

    def score(
        self,
        scenario: "ScenarioSpec",
        audit: "Iterable[Envelope]",
        final_state: "WorldSnapshot",
    ) -> "Report":
        """Return the per-scenario report."""
        ...


class AgentBuilder(Protocol):
    """Build the list of agents for a scenario. Implemented by ``ScenarioSpec.build_agents``."""

    def build_agents(self) -> "list[Agent]":
        """Return agents in registration order."""
        ...
