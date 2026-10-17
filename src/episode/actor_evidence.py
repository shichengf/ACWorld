"""Build per-episode Runtime actor-evidence services from trusted inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from episode.scenario import kickoff_envelopes, population_for_scenario
from episode.types import ScenarioSpec
from runtime.actor_context import (
    DEFAULT_ROOT_ACTION_KINDS,
    ActorContextResolver,
    RegisteredActorContext,
)
from runtime.actor_context_artifact import (
    ACTOR_CONTEXTS_ARTIFACT,
    write_actor_contexts,
)
from runtime.actor_evidence import (
    ACTOR_EVIDENCE_ARTIFACT,
    ActorEvidenceJournal,
)


def build_actor_evidence_services(
    scenario: ScenarioSpec,
    *,
    out_dir: str | Path,
) -> tuple[ActorContextResolver, ActorEvidenceJournal]:
    """Create one causal resolver and journal for a CommerceWorld episode.

    Scenarios configure roots and participant mandates, but the acceptance
    logic remains Runtime-owned.  Registrations are frozen before execution and
    are derived only from server-side scenario data plus the exact kickoff
    envelopes that Runtime will later audit.  Actor report payloads are never
    consulted.
    """

    contexts = actor_contexts_for_scenario(scenario)
    root_kinds = set(DEFAULT_ROOT_ACTION_KINDS)
    root_kinds.update(context.root_action_kind for context in contexts)

    frozen_contexts = contexts
    frozen_root_kinds = frozenset(root_kinds)
    persisted = write_actor_contexts(
        Path(out_dir) / ACTOR_CONTEXTS_ARTIFACT,
        episode_context_id=scenario.scenario_id,
        actor_contexts=frozen_contexts,
        root_action_kinds=frozen_root_kinds,
    )
    # Live acceptance and offline verification now instantiate from the exact
    # same canonical bytes, eliminating configuration drift between the two.
    resolver = persisted.build_resolver()
    journal = ActorEvidenceJournal(
        Path(out_dir) / ACTOR_EVIDENCE_ARTIFACT,
        run_id=scenario.scenario_id,
    )
    return resolver, journal


def actor_contexts_for_scenario(
    scenario: ScenarioSpec,
) -> tuple[RegisteredActorContext, ...]:
    """Derive the immutable actor-directed kickoff registrations.

    This is the single scenario-side source for both Runtime actor evidence and
    the Agent's spawn-frozen report-tool roots.  Platform- or World-directed
    initial events are deliberately excluded.  Their hidden internal ancestry
    must never manufacture an actor reporting authority.
    """

    population = population_for_scenario(scenario)
    participants = {
        buyer.buyer_id for buyer in population.buyers
    } | {
        merchant.merchant_id for merchant in population.merchants
    }
    buyer_mandates = {
        buyer.buyer_id: _text_or_none(buyer.mandate.get("mandate_id"))
        for buyer in population.buyers
    }
    task_id = _scenario_task_id(scenario)
    contexts: list[RegisteredActorContext] = []
    for root in kickoff_envelopes(scenario):
        if root.to not in participants:
            continue
        action_kind = str(root.action.get("kind", "")).strip()
        if not action_kind:
            raise ValueError(
                f"{scenario.scenario_id}: actor kickoff {root.msg_id!r} "
                "has no action kind"
            )
        payload = root.action.get("payload")
        payload_map = payload if isinstance(payload, Mapping) else {}
        mandate_id = (
            _context_value(payload_map, "mandate_id")
            or buyer_mandates.get(root.to)
            or f"{task_id}:{root.to}"
        )
        contexts.append(
            RegisteredActorContext(
                root_msg_id=root.msg_id,
                root_action_kind=action_kind,
                actor_id=root.to,
                principal_id=root.from_,
                task_id=task_id,
                mandate_id=mandate_id,
                context_id=scenario.scenario_id,
            )
        )
    return tuple(sorted(contexts, key=lambda context: context.root_msg_id))


def _scenario_task_id(scenario: ScenarioSpec) -> str:
    value = _text_or_none(scenario.success_oracle.get("task_id"))
    return value or scenario.scenario_id


def _context_value(payload: Mapping[str, Any], field: str) -> str | None:
    direct = _text_or_none(payload.get(field))
    nested = payload.get("delegation_context")
    nested_value = (
        _text_or_none(nested.get(field)) if isinstance(nested, Mapping) else None
    )
    if direct is not None and nested_value is not None and direct != nested_value:
        raise ValueError(f"actor kickoff has conflicting {field!r} context")
    return direct or nested_value


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


__all__ = ["actor_contexts_for_scenario", "build_actor_evidence_services"]
