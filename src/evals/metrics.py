"""The named metric implementations.

Each metric is a pure function over (scenario, audit, snapshot) returning a
``MetricResult``. They satisfy the ``evals.interfaces.Metric`` Protocol.

Metrics are additive; renames require a metric-version bump in the eval layer.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from agents.types import SkillGroup  # noqa: F401  (forward ref by name in role-localization metrics)

if TYPE_CHECKING:
    from episode.types import ScenarioSpec
    from evals.types import MetricResult
    from protocol.envelope import Envelope
    from world.types import WorldSnapshot


# --- Core transaction metrics ----------------------------------------

class TaskSuccess:
    """True iff the final purchased SKU matches buyer goal + constraints."""

    name: str = "task_success"

    def compute(
        self,
        scenario: "ScenarioSpec",
        audit: "Iterable[Envelope]",
        final_state: "WorldSnapshot",
    ) -> "MetricResult":
        """Cross-check settled SKU against ``scenario.buyer_goal``."""
        raise NotImplementedError(
            "settled = find_settle(audit); if not settled: return MetricResult(name, None); "
            "match = product_match(scenario.buyer_goal, settled, scenario.initial_state); "
            "return MetricResult(name, match, audit_pointer=settled.msg_id)"
        )


class BudgetAdherence:
    """True iff settled price ≤ buyer max_budget. None if no settlement."""

    name: str = "budget_adherence"

    def compute(
        self,
        scenario: "ScenarioSpec",
        audit: "Iterable[Envelope]",
        final_state: "WorldSnapshot",
    ) -> "MetricResult":
        """Compare settled price against ``scenario.buyer_goal['max_budget']``."""
        raise NotImplementedError(
            "settled = find_settle(audit); if not settled: return MetricResult(name, None); "
            "ok = settled.action['payload']['price'] <= scenario.buyer_goal['max_budget']; "
            "return MetricResult(name, ok, audit_pointer=settled.msg_id)"
        )


class MerchantMargin:
    """True iff settled price ≥ merchant floor_price. None if no settlement."""

    name: str = "merchant_margin"

    def compute(
        self,
        scenario: "ScenarioSpec",
        audit: "Iterable[Envelope]",
        final_state: "WorldSnapshot",
    ) -> "MetricResult":
        """Compare settled price against ``scenario.merchant_policy['floor_price']``."""
        raise NotImplementedError(
            "settled = find_settle(audit); if not settled: return MetricResult(name, None); "
            "ok = settled.action['payload']['price'] >= scenario.merchant_policy['floor_price']; "
            "return MetricResult(name, ok, audit_pointer=settled.msg_id)"
        )


class StateCorrectness:
    """True iff every ``success_oracle.*_decrement`` / ``*_created`` predicate holds."""

    name: str = "state_correctness"

    def compute(
        self,
        scenario: "ScenarioSpec",
        audit: "Iterable[Envelope]",
        final_state: "WorldSnapshot",
    ) -> "MetricResult":
        """Walk every key in ``scenario.success_oracle`` and check against ``final_state``."""
        raise NotImplementedError(
            "for predicate, expected in scenario.success_oracle.items(): "
            "  if not check_predicate(predicate, expected, final_state): return MetricResult(name, False, ...); "
            "return MetricResult(name, True)"
        )


class ProtocolCorrectness:
    """True iff zero envelopes were rejected by the runtime."""

    name: str = "protocol_correctness"

    def compute(
        self,
        scenario: "ScenarioSpec",
        audit: "Iterable[Envelope]",
        final_state: "WorldSnapshot",
    ) -> "MetricResult":
        """Count rejection envelopes; True iff count == 0."""
        raise NotImplementedError(
            "n = count_violations(audit); return MetricResult(name, n == 0, notes=f'{n} violations')"
        )


# --- Localization and competition metrics (signatures only) ----------

class SubRoleLocalization:
    """Maps audit-log outcomes back to roles."""

    name: str = "sub_role_localization"

    def compute(
        self,
        scenario: "ScenarioSpec",
        audit: "Iterable[Envelope]",
        final_state: "WorldSnapshot",
    ) -> "MetricResult":
        """Per-role attribution of the failure tag."""
        raise NotImplementedError("walk audit; group outcome_tags by role; return MetricResult")


class ConsumerRegret:
    """Quantify how often the consumer picked a dominated option."""

    name: str = "consumer_regret"

    def compute(
        self,
        scenario: "ScenarioSpec",
        audit: "Iterable[Envelope]",
        final_state: "WorldSnapshot",
    ) -> "MetricResult":
        """Compare chosen option against the dominant baseline."""
        raise NotImplementedError(
            "find chosen and dominated; return regret = (chosen_cost - best_cost)/best_cost"
        )
