"""Passive data types for the episode package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from episode.benchmark import BenchmarkMetadata


# --- Scenario specification ------------------------------------------

@dataclass(frozen=True)
class BuyerSpec:
    """One buyer participant in a many-to-many market.

    ``mandate`` uses the agent-native representation (money values are integer
    minor units).  ``initial_state`` is non-authoritative participant-local
    data such as memory and preferences.  It must never carry orders, pricing,
    lifecycle, governance, or other commerce state owned by World.
    """

    buyer_id: str
    persona: dict[str, Any]
    mandate: dict[str, Any]
    initial_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MerchantSpec:
    """One merchant participant in a many-to-many market.

    ``policy`` is private agent input and uses integer minor units for monetary
    fields such as ``floor_price``. ``catalog_scope`` contains globally unique
    listing ``sku_id`` values owned by this merchant; an empty tuple means the
    scope is inferred from the seeded catalog.
    """

    merchant_id: str
    persona: dict[str, Any]
    policy: dict[str, Any]
    catalog_scope: tuple[str, ...] = ()
    initial_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PopulationSpec:
    """Declarative participants and sparse-market execution parameters.

    ``initial_events`` contains external principal, actor, or environment input
    envelopes.  It may not impersonate Platform or World output.  When it is
    empty, the scenario layer deterministically creates one mandate-delegation
    kickoff per buyer. ``matching`` and ``execution`` are intentionally additive
    policy dictionaries; their stable v1 defaults are supplied by
    :func:`episode.scenario.population_for_scenario`.
    """

    buyers: tuple[BuyerSpec, ...]
    merchants: tuple[MerchantSpec, ...]
    initial_events: tuple[dict[str, Any], ...] = ()
    matching: dict[str, Any] = field(default_factory=lambda: {"top_k": 5})
    execution: dict[str, Any] = field(
        default_factory=lambda: {"max_transactions_per_buyer": 1}
    )


@dataclass(frozen=True)
class WorldEventSpec:
    """One deterministic invocation of a registered world-event handler.

    Events are ordered by ``(logical_time, event_id)`` before agent execution.
    The handler name is resolved through ``WORLD_EVENT_HANDLERS`` at run time,
    so adding an event implementation never requires an edit to the runner.
    """

    event_id: str
    handler: str
    payload: dict[str, Any] = field(default_factory=dict)
    logical_time: int = 0


@dataclass(frozen=True)
class ExtensionEvaluationSpec:
    """One deterministic invocation of a registered metric or oracle.

    Runtime-bound extension callables receive an
    ``ExtensionEvaluationContext`` keyword plus these explicit ``arguments``.
    Their JSON-compatible return value is written to ``extensions.json`` and is
    deliberately separate from the benchmark's headline score.
    """

    evaluation_id: str
    kind: Literal["market_metric", "oracle_primitive"]
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlServiceSpec:
    """One non-agent Runtime service declared by an environment scenario.

    Control services consume authenticated Platform requests.  They never
    appear in the buyer or merchant population and therefore cannot acquire
    agent private state or inference channels.
    """

    service_id: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioSpec:
    """Declarative description of one scenario run.

    Hydrated from a YAML file by ``scenario.ScenarioSpec.from_yaml``. Fields
    are additive: adding a new field is fine; renaming or removing one is not.

    ``allowed_actions`` is legacy task-oracle metadata.  It describes the
    action vocabulary directly relevant to a scenario's expected solution and
    is used by deterministic evaluation helpers.  It is never an actor
    permission list.  Runtime, Platform, and World authorization comes from
    the typed protocol, actor identity, private ownership, and current World
    state, so every snapshot retains the complete CommerceWorld action surface.
    """
    scenario_id: str
    seed: int
    initial_state: dict[str, Any]
    buyer_goal: dict[str, Any]
    merchant_policy: dict[str, Any]
    allowed_actions: tuple[str, ...]
    success_oracle: dict[str, Any]            # server-side only; never in prompts
    platform_policy: dict[str, Any] | None = None
    persona_axes: dict[str, Any] | None = None
    # Paper-facing taxonomy + parameter axes. Older/programmatic specs may leave
    # this unset; the YAML loader always infers it from the scenario's sN id.
    benchmark: BenchmarkMetadata | None = None
    # Optional many-to-many population. ``None`` is the wire-stable legacy 1x1
    # representation and is hydrated lazily by ``population_for_scenario``.
    population: PopulationSpec | None = None
    # Optional registered extension invocations. Empty tuples preserve the
    # behavior and artifacts of every existing scenario.
    world_events: tuple[WorldEventSpec, ...] = ()
    extension_evaluations: tuple[ExtensionEvaluationSpec, ...] = ()
    control_services: tuple[ControlServiceSpec, ...] = ()


# --- Batch -----------------------------------------------------------

@dataclass(frozen=True)
class BatchSpec:
    """Group of scenarios that share an E1 reset boundary.

    World resets between batches, not between scenarios within a batch.
    """
    batch_id: str
    scenarios: tuple[ScenarioSpec, ...]


# --- Report ----------------------------------------------------------

@dataclass(frozen=True)
class Report:
    """Per-scenario report. Written to ``reports/<run_id>/<scenario_id>.json``.

    ``None`` means "not applicable to this run" (e.g. no settlement happened).
    Fields are additive.
    """
    scenario_id: str
    seed: int
    # Core transaction metrics
    task_success: bool | None = None
    budget_adherence: bool | None = None
    merchant_margin: bool | None = None
    state_correctness: bool | None = None
    protocol_correctness: bool | None = None
    # Localization
    sub_role_localization: dict[str, Any] | None = None
    failure_tag: str | None = None
    # Competition
    consumer_regret: float | None = None
    # Persistence
    repeat_recognition_rate: float | None = None
    reputation_conditioned_share: float | None = None
    dispute_resolution_latency_ms: int | None = None
    stockout_handling: bool | None = None
    # Platform
    fairness_index: float | None = None
    bias_index: float | None = None
    gameability_index: float | None = None
    # Persona / adversarial
    trait_recoverability: float | None = None
    private_utility_leakage: float | None = None
    drift: float | None = None


# --- Run metadata ----------------------------------------------------

@dataclass(frozen=True)
class RunMetadata:
    """Identifies one ``cwe-run`` invocation across N scenarios."""
    run_id: str
    started_at: str
    seed: int
    mode: Literal["E0", "E1"] = "E0"
    audit_log_path: str = ""
