"""Thin many-participant HTTP launcher.

Runs one scenario across one VCP ``/vcp`` server per buyer and merchant plus the
platform and world, wired to a dispatcher (:class:`~runtime.bus.Runtime` + ``HttpTransport``)
that relays validated envelopes between them and owns the single audit log + the
trace sidecar. The audited stream is identical to the in-process
:class:`~episode.runner.Episode`; what differs is that every agent boundary now
crosses the real HTTP/VCP wire, which is the interop evidence the contract is
meant to provide. Research numbers for the social/compromise corpus run here (the
final topology), per the project's process-split goal.

RESIDUAL (item 4c, deferred): the platform's own world access (settle / catalog
reads) is still an in-process call, so the platform server shares the SAME
``World`` object the world server fronts. The buyer/merchant ↔ platform ↔ buyer
message paths ARE over the wire; only the platform's internal world access is
co-located. True platform process isolation lands when ``world.settle_order`` /
catalog reads move over VCP.

Transport: in-process FastAPI ``TestClient``s by default — fast and exercises the
real HTTP/VCP code path (ASGI), so it is suitable for stub plumbing checks and
for the live discovery run alike. (Separate OS processes would swap the
TestClients for real ``httpx.Client``s pointed at running servers; the dispatcher
code is identical.)
"""

from __future__ import annotations

import sys
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from episode.extension_runtime import (
    apply_registered_world_events,
    evaluate_registered_extensions,
    write_extension_artifact,
)
from episode.runner import (
    ChannelFactory,
    _write_txn_diffs,
)
from episode.termination import (
    EPISODE_TERMINATION_ARTIFACT,
    write_termination_artifact,
)
from episode.scenario import (
    build_agents,
    build_secret_registry,
    kickoff_envelopes,
    population_for_scenario,
    seed_world,
)

if TYPE_CHECKING:
    from episode.types import ScenarioSpec


def run_http_episode(
    *,
    scenario: "ScenarioSpec",
    channels: ChannelFactory,
    out_dir: "str | Path",
    mode: "Literal['E0', 'E1']" = "E0",
    strict_skill_selection: bool = False,
) -> None:
    """Seed → stand up the four ``/vcp`` servers → dispatch the kickoff → drain →
    flush → snapshot → replay → seal evidence."""
    from fastapi.testclient import TestClient

    from agents.api import create_agent_app
    from agents.platform import PlatformService
    from agents.platform_api import create_platform_app
    from agents.world_client import WorldClient
    from runtime import AuditLog, Router, Runtime
    from runtime.transport import HttpTransport
    from world.api import create_app as create_world_app
    from world.client import VCPWorldClient
    from world.service import WorldService
    from world.state import World

    from episode.artifacts import prepare_out_dir

    # An episode OWNS its out_dir: clear prior episode artifacts so a re-run into
    # the same directory starts clean (AuditLog opens append-mode). Shared with
    # the in-process path so both remove the identical owned-artifact set.
    out = prepare_out_dir(out_dir)

    from evals.serialize import dump_snapshot

    world = World(mode=mode)
    seed_world(world, scenario)
    extension_events = apply_registered_world_events(world, scenario)
    # Authoritative pre-settlement snapshot, before any agent acts.
    initial_snapshot = world.snapshot()
    dump_snapshot(initial_snapshot, out / "world.initial.json", phase="initial")
    commit_cursor = world.begin_evidence_window()
    audit = AuditLog(out / "audit.jsonl", run_id=scenario.scenario_id)
    from episode.actor_evidence import build_actor_evidence_services

    actor_context_resolver, actor_evidence_journal = build_actor_evidence_services(
        scenario, out_dir=out
    )

    agents = build_agents(
        scenario,
        channels=channels,
        strict_skill_selection=strict_skill_selection,
    )

    with ExitStack() as stack:
        # 4c: the platform reaches World over VCP (HTTP), not a shared object. Stand
        # the world server up first, then build the platform with a WorldClient
        # whose transport POSTs to that world /vcp — so EVERY platform->world
        # access (settle/refund/catalog/inventory) crosses the wire.
        world_tc = stack.enter_context(TestClient(create_world_app(WorldService(world))))
        platform_world = WorldClient(send=VCPWorldClient(http_client=world_tc, agent_id="platform").send)
        population = population_for_scenario(scenario)
        secrets = build_secret_registry(scenario)
        participant_ids = frozenset(
            [buyer.buyer_id for buyer in population.buyers]
            + [merchant.merchant_id for merchant in population.merchants]
        )
        platform = PlatformService(
            world_client=platform_world,
            audit=audit,
            platform_policy=scenario.platform_policy,
            max_search_results=(
                int(population.matching["top_k"])
                if scenario.population is not None
                else None
            ),
            max_transactions_per_buyer=(
                int(population.execution["max_transactions_per_buyer"])
                if scenario.population is not None
                else None
            ),
            secrets=secrets,
            negotiation_participants=participant_ids,
            negotiation_max_rounds=int(
                population.execution.get(
                    "max_negotiation_rounds",
                    max(
                        (
                            int(merchant.policy.get("max_negotiation_rounds", 1))
                            for merchant in population.merchants
                        ),
                        default=8,
                    ),
                )
            ),
            negotiation_deadline_ticks=int(
                population.execution.get("negotiation_deadline_ticks", 32)
            ),
        )
        clients: dict[str, Any] = {
            "platform": stack.enter_context(TestClient(create_platform_app(platform))),
            "world": world_tc,
        }
        role_counts = {
            role: sum(agent.id.split(":", 1)[0] == role for agent in agents)
            for role in ("buyer", "merchant")
        }
        for agent in agents:
            client = stack.enter_context(TestClient(create_agent_app(agent)))
            clients[agent.id] = client
            role = agent.id.split(":", 1)[0]
            # Preserve the legacy role-only fallback only when it is unambiguous.
            if role_counts.get(role) == 1:
                clients[role] = client
        transport = HttpTransport(clients=clients, trace_sink=audit.append_trace_dict)
        # Enforce the scenario-derived privacy boundary on the HTTP launch path too
        # (secrets from the answer key, not agent memory).
        runtime = Runtime(world=world, router=Router(secrets),
                          audit=audit, transport=transport,
                          actor_context_resolver=actor_context_resolver,
                          actor_evidence_journal=actor_evidence_journal)
        if scenario.control_services:
            from runtime.control_services import build_control_service

            for service_spec in scenario.control_services:
                runtime.register_control_service(
                    build_control_service(service_spec, world=world)
                )
        try:
            for kickoff in kickoff_envelopes(scenario):
                runtime.send(kickoff)
            runtime.run()
        except Exception as exc:  # noqa: BLE001 — score the partial run, like Episode
            write_termination_artifact(out, exc)
            print(f"[http episode {scenario.scenario_id}] aborted: "
                  f"{type(exc).__name__}; sanitized details in "
                  f"{EPISODE_TERMINATION_ARTIFACT}", file=sys.stderr)
        finally:
            # Flush any decision parked mid-grounding on an agent server (the
            # cross-process forced_flush), then close the audit.
            runtime.flush_suspended()
            audit.close()

    final_snapshot = world.snapshot()
    dump_snapshot(final_snapshot, out / "world.final.json", phase="final")
    _write_txn_diffs(out, world)
    extension_evaluations = evaluate_registered_extensions(
        scenario,
        initial_snapshot=initial_snapshot,
        final_snapshot=final_snapshot,
        world_events=extension_events,
    )
    write_extension_artifact(
        out / "extensions.json",
        scenario=scenario,
        world_events=extension_events,
        evaluations=extension_evaluations,
    )
    from episode.evidence import finalize_episode_evidence

    finalize_episode_evidence(
        out,
        scenario_id=scenario.scenario_id,
        world=world,
        commit_cursor=commit_cursor,
        transport="http_vcp",
    )
    return None
