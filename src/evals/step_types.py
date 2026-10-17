"""Passive types for the deterministic step scorer (reward.v1alpha1).

Frozen dataclasses + string enums with JSON-stable values. No behavior beyond
construction. The step scorer (``evals.step_scorer``) produces a
:class:`StepScoreReport`; ``evals.reward_io`` serializes it to ``reward.json``.

Design invariants (see reward/scoring_design.md):
- a :class:`StageReward` belongs to exactly one ``(stage, side)``; never both,
  never "integrity";
- :class:`EpisodeIntegrity` is a separate artifact, not a side;
- ``legacy_failure_mode`` is a compatibility artifact; ``effective_failure_mode``
  is derived from TRUSTED new gates + completion + integrity, NOT an
  unconditional merge of every legacy verdict;
- this schema is explicitly ``reward.v1alpha1`` (not final).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = "reward.v1alpha1"


class FailureMode(str, Enum):
    """The four categorical verdicts (shared priority with the legacy scorer)."""
    OK = "ok"
    CAPABILITY = "capability"
    JUDGMENT = "judgment"
    SECURITY = "security"


#: Categorical priority: a higher index is "worse" and wins a merge.
_PRIORITY = {FailureMode.OK: 0, FailureMode.JUDGMENT: 1,
             FailureMode.CAPABILITY: 2, FailureMode.SECURITY: 3}


def worst(*modes: "FailureMode") -> "FailureMode":
    """Return the most severe failure mode (SECURITY > CAPABILITY > JUDGMENT > OK)."""
    out = FailureMode.OK
    for m in modes:
        if _PRIORITY[m] > _PRIORITY[out]:
            out = m
    return out


class Side(str, Enum):
    """A StageReward subject side. Never 'integrity' (that is an episode artifact)."""
    BUYER = "buyer"
    MERCHANT = "merchant"


class StageId(str, Enum):
    """Stage identifiers used by the step scorer."""
    S0_PRIVACY = "S0_privacy"
    S1_DISCOVERY = "S1_discovery"
    S2_GROUNDING = "S2_grounding"
    S3_SELECTION = "S3_selection"
    S4_NEGOTIATION = "S4_negotiation"
    S5_OUTCOME = "S5_outcome"
    S6_RETURN = "S6_return"


@dataclass(frozen=True)
class EvidenceRef:
    """A pointer into the audit/world artifacts — never a copied envelope/secret."""
    kind: str                       # "envelope" | "order" | "security_event" | "snapshot" | ...
    ref: str                        # msg_id / order_id / txn_id / sku_id / "<event>"
    note: str = ""


@dataclass(frozen=True)
class Subreward:
    """One scored sub-check within a stage.

    ``applicable=False`` (N/A) removes the subreward from BOTH numerator and
    denominator. ``discriminating`` records whether the case actually tested the
    lane. ``hard_gate`` marks a gate that zeroes its stage on failure; ``gated``
    records that it fired.
    """
    name: str
    earned: float = 0.0
    maximum: float = 0.0
    applicable: bool = True
    discriminating: "bool | None" = None
    hard_gate: bool = False
    gated: bool = False
    passed: "bool | None" = None
    reasons: tuple[str, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class StageReward:
    """All subrewards for exactly one ``(stage, side)``."""
    stage: StageId
    side: Side
    subrewards: tuple[Subreward, ...] = ()
    points_earned: float = 0.0
    points_max: float = 0.0
    discriminating: "bool | None" = None
    gated_by: "str | None" = None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateEvent:
    """A hard-gate firing that contributes to the categorical verdict."""
    gate: str                       # "buyer_privacy" | "merchant_floor_breach" | ...
    side: "Side | None"             # None for episode-integrity-level gates
    failure_mode: FailureMode
    reason: str
    evidence: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class IntegrityIssue:
    """An episode/platform integrity problem (not a buyer/merchant point loss)."""
    code: str                       # "missing_merchant_consent" | "fabricated_settlement" | ...
    actor: str                      # the actual offending actor id, or "unknown"
    reason: str
    evidence: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class EpisodeIntegrity:
    """Episode-level integrity verdict. Separate from any StageReward side."""
    ok: bool = True
    issues: tuple[IntegrityIssue, ...] = ()


@dataclass(frozen=True)
class LegacyDisagreement:
    """A recorded place where the legacy verdict diverges from the new trusted
    evaluation (e.g. legacy JUDGMENT on a stale handwritten expected_sku while
    the chosen sku is in the derived acceptable set)."""
    field: str
    legacy: str
    effective: str
    reason: str


@dataclass(frozen=True)
class RewardWeightProfile:
    """A versioned, injectable weight profile. Weights are DRAFT, not final;
    totals are diagnostic only."""
    profile_id: str = "draft.v0"
    weights: tuple[tuple[str, float], ...] = ()   # (subreward-name -> weight), stable order

    def weight_for(self, name: str, default: float = 1.0) -> float:
        for k, v in self.weights:
            if k == name:
                return v
        return default


@dataclass(frozen=True)
class RewardTotals:
    """Diagnostic totals for one side (normalized over applicable subrewards)."""
    points_earned: float = 0.0
    points_max: float = 0.0
    normalized: "float | None" = None   # earned/max over non-N/A, or None if denom 0


@dataclass(frozen=True)
class StepScoreReport:
    """The full per-episode step-score report (reward.v1alpha1)."""
    scenario_id: str
    schema_version: str = SCHEMA_VERSION
    weight_profile_id: str = "draft.v0"
    legacy_failure_mode: str = "ok"
    stage_gate_verdict: FailureMode = FailureMode.OK
    effective_failure_mode: FailureMode = FailureMode.OK
    task_completed: bool = False
    # "settled" | "rejected_no_zopa" | "rejected_no_feasible_offer" |
    # "settled_inconsistent" | "incomplete"
    completion_kind: str = "incomplete"
    buyer_stages: tuple[StageReward, ...] = ()
    merchant_stages: tuple[StageReward, ...] = ()
    buyer_total: RewardTotals = field(default_factory=RewardTotals)
    merchant_total: RewardTotals = field(default_factory=RewardTotals)
    integrity: EpisodeIntegrity = field(default_factory=EpisodeIntegrity)
    gate_events: tuple[GateEvent, ...] = ()
    legacy_disagreements: tuple[LegacyDisagreement, ...] = ()
    oracle_summary: tuple[tuple[str, str], ...] = ()   # sanitized (key -> value) pairs
    coverage: tuple[tuple[str, str], ...] = ()         # (lane -> READY|TODO|N/A|BLOCKED)


__all__ = [
    "SCHEMA_VERSION", "FailureMode", "worst", "Side", "StageId", "EvidenceRef",
    "Subreward", "StageReward", "GateEvent", "IntegrityIssue", "EpisodeIntegrity",
    "LegacyDisagreement", "RewardWeightProfile", "RewardTotals", "StepScoreReport",
]
