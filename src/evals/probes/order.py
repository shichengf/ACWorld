"""Order-of-actions ablation probe.

For negotiation scenarios, randomize the order of buyer/merchant first move
across N=10 seeds; report mean and std of each metric. See DESIGN.md §7.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from episode.types import ScenarioSpec
    from evals.types import ProbeReport


class OrderAblationProbe:
    """Causal probe: does buyer-first vs merchant-first matter on negotiation runs?"""

    probe_id: str = "order_ablation"

    def __init__(self, *, n_seeds: int = 10) -> None:
        """Configure number of seeds per scenario; deterministic seed sequence."""
        raise NotImplementedError("self._n_seeds = n_seeds")

    def applies_to(self, scenario: "ScenarioSpec") -> bool:
        """True iff the scenario is in the negotiation family.

        Negotiation is identified by ``scenario.scenario_id`` prefix or by
        presence of both ``propose_offer`` and ``counter_offer`` in
        ``allowed_actions``.
        """
        raise NotImplementedError(
            "return scenario.scenario_id.startswith('negotiation_') or "
            "{'propose_offer', 'counter_offer'} <= set(scenario.allowed_actions)"
        )

    def run(
        self,
        scenarios: "Iterable[ScenarioSpec]",
        *,
        n_seeds: int,
    ) -> "ProbeReport":
        """For each scenario × seed × first-mover, run an Episode and aggregate.

        Returns a ``ProbeReport`` with per-seed metric vectors and mean/std summary.
        """
        raise NotImplementedError(
            "for scenario in scenarios where self.applies_to(scenario): "
            "  for seed in derive_seeds(self._n_seeds): "
            "    for first_mover in ('buyer', 'merchant'): "
            "      run an Episode with permuted kickoff; collect Report; "
            "aggregate mean/std per metric across (seed, first_mover); "
            "return ProbeReport(probe_id='order_ablation', ...)"
        )
