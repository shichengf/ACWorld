"""``Episode`` and ``EpisodeBatch`` — run a scenario end to end.

Per WORLD_CLASSES.md §5 an Episode is thin: seed the world from the scenario,
register the agents, send the kickoff, drain the runtime, snapshot, and seal
replay-verifiable evidence. Benchmark scoring happens later through the
task-specific capability scorer; Episode itself never emits an environment
score or reward.

Channels are injected via an actor-specific ``channels(agent_id, role)`` factory.
Every channel exposes only the typed business-decision transport; Agent owns
CommerceWorld communication and protocol compilation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from episode.extension_runtime import (
    apply_registered_world_events,
    evaluate_registered_extensions,
    write_extension_artifact,
)
from episode.scenario import (
    build_agents,
    build_secret_registry,
    kickoff_envelopes,
    population_for_scenario,
    seed_world,
)
from episode.termination import (
    EPISODE_TERMINATION_ARTIFACT,
    write_termination_artifact,
)
if TYPE_CHECKING:
    from runtime.audit import AuditLog
    from runtime.bus import Runtime
    from world.state import World

    from episode.types import ScenarioSpec

#: ``(agent_id, role) -> fresh typed business-decision channel``.
ChannelFactory = Callable[..., Any]

class Episode:
    """One scenario run with replay-verifiable evidence.

    The caller supplies an (unseeded) ``world``, an ``audit`` log, and a
    ``runtime`` already bound to that world + a PlatformService; the Episode
    seeds the world, registers the agents, drives the bus, and scores.
    """

    def __init__(
        self,
        *,
        scenario: "ScenarioSpec",
        world: "World",
        runtime: "Runtime",
        audit: "AuditLog",
        channels: ChannelFactory,
        out_dir: "str | Path",
        mode: "Literal['E0', 'E1']" = "E0",
        transport: str = "in_process",
        strict_skill_selection: bool = False,
    ) -> None:
        self._spec = scenario
        self._world = world
        self._runtime = runtime
        self._audit = audit
        self._channels = channels
        self._out_dir = Path(out_dir)
        self._mode = mode
        self._transport = transport
        self._strict_skill_selection = strict_skill_selection

    def run(self) -> None:
        """Seed -> capture initial snapshot -> register -> kickoff -> drain ->
        flush -> capture final snapshot -> replay -> seal evidence."""
        from evals.serialize import dump_snapshot

        self._out_dir.mkdir(parents=True, exist_ok=True)
        # ``Episode`` is also used directly, without ``prepare_out_dir``.  Never
        # let a prior run's abort marker contaminate a clean rerun.
        (self._out_dir / EPISODE_TERMINATION_ARTIFACT).unlink(missing_ok=True)
        # Direct Episode callers bypass EpisodeBatch's Runtime service builder,
        # but the immutable registration input is still a required evidence
        # artifact. Persist it before any envelope is audited. This does not
        # manufacture an acceptance service for a Runtime that omitted one:
        # any actor report on such a Runtime still fails closed.
        from runtime.actor_context_artifact import ACTOR_CONTEXTS_ARTIFACT

        if not (self._out_dir / ACTOR_CONTEXTS_ARTIFACT).exists():
            from episode.actor_evidence import build_actor_evidence_services

            build_actor_evidence_services(self._spec, out_dir=self._out_dir)
        seed_world(self._world, self._spec)
        try:
            extension_events = apply_registered_world_events(self._world, self._spec)
        except Exception:
            # Event configuration errors happen before the normal drain/finally
            # block. Close append-mode audit handles before surfacing them.
            self._audit.close()
            raise
        # Authoritative pre-settlement snapshot, captured before any agent acts.
        initial_snapshot = self._world.snapshot()
        dump_snapshot(initial_snapshot, self._out_dir / "world.initial.json",
                      phase="initial")
        commit_cursor = self._world.begin_evidence_window()
        if self._spec.control_services:
            from runtime.control_services import build_control_service

            for service_spec in self._spec.control_services:
                self._runtime.register_control_service(
                    build_control_service(service_spec, world=self._world)
                )
        for agent in build_agents(
            self._spec,
            channels=self._channels,
            strict_skill_selection=self._strict_skill_selection,
        ):
            self._runtime.register(agent)

        # A misbehaving agent (e.g. a decision-record / budget guard) may raise
        # mid-drain. Don't lose the episode: capture, then score the partial
        # artifacts — the scorer classifies an unfinished run as CAPABILITY.
        runtime_aborted = False
        teardown_error: Exception | None = None
        try:
            for kickoff in kickoff_envelopes(self._spec):
                self._runtime.send(kickoff)
            self._runtime.run()
        except Exception as exc:  # noqa: BLE001 — score the partial run, don't abort the batch
            runtime_aborted = True
            write_termination_artifact(self._out_dir, exc)
            print(f"[episode {self._spec.scenario_id}] aborted: "
                  f"{type(exc).__name__}; sanitized details in "
                  f"{EPISODE_TERMINATION_ARTIFACT}", file=sys.stderr)
        finally:
            # Flush any decision still parked on an unanswered world read so a
            # suspended/incomplete decision is recorded (forced_flush), matching
            # the HTTP path, before the audit handles close.
            try:
                self._runtime.flush_suspended()
            except Exception as exc:  # noqa: BLE001 - strict evidence decides usability below
                teardown_error = exc
            finally:
                self._audit.close()

        from evals.serialize import dump_snapshot

        final_snapshot = self._world.snapshot()
        dump_snapshot(final_snapshot, self._out_dir / "world.final.json", phase="final")
        _write_txn_diffs(self._out_dir, self._world)
        extension_evaluations = evaluate_registered_extensions(
            self._spec,
            initial_snapshot=initial_snapshot,
            final_snapshot=final_snapshot,
            world_events=extension_events,
        )
        write_extension_artifact(
            self._out_dir / "extensions.json",
            scenario=self._spec,
            world_events=extension_events,
            evaluations=extension_evaluations,
        )
        from episode.evidence import finalize_episode_evidence

        finalize_episode_evidence(
            self._out_dir,
            scenario_id=self._spec.scenario_id,
            world=self._world,
            commit_cursor=commit_cursor,
            transport=self._transport,
        )
        if teardown_error is not None:
            # A Runtime abort can race with teardown after the failing Agent
            # turn has already produced its Tracker-bound no-emit terminal.
            # Do not discard a model-attributed outcome merely because a
            # cleanup hook then raised: the manifest above is written only
            # after strict actor causality, Platform coverage, World replay,
            # and every artifact digest all verify.  A clean episode or an
            # unscoreable abort still fails closed on the teardown exception.
            from episode.termination import load_verified_scoreable_termination

            if (
                not runtime_aborted
                or load_verified_scoreable_termination(self._out_dir) is None
            ):
                raise teardown_error
        return None


def _read_jsonl(path: Path) -> "list[dict[str, Any]]":
    if not path.exists():
        return []
    return [json.loads(li) for li in path.read_text(encoding="utf-8").splitlines() if li.strip()]


def _write_txn_diffs(out_dir: Path, world: Any) -> None:
    """Drain the world's atomic multi-row write diffs (settle / refund) to a
    sidecar — an observability / replay artifact. NOT a scorer input (the scorer
    reads only the audit log + trace + snapshot)."""
    diffs = getattr(world, "_txn_diffs", None)
    if not diffs:
        return
    from evals.serialize import to_canonical

    (out_dir / "txn_diffs.jsonl").write_text(
        "\n".join(json.dumps(to_canonical(d)) for d in diffs) + "\n",
        encoding="utf-8",
    )


class EpisodeBatch:
    """Run scenarios under the selected world-lifetime policy.

    E0 constructs a fresh world for every episode. E1 constructs one world for
    the batch and performs an E1 reset between episodes, preserving durable
    orders, ledger entries, reputation, disputes, and rulings while catalog and
    current inventory are re-seeded. Each episode still receives a fresh
    runtime, agent registry, privacy registry, and audit stream.

    Returns one ``(scenario_id, None)`` row per executed scenario. Task-specific
    capability scoring consumes the sealed evidence separately.
    """

    def __init__(
        self,
        *,
        scenarios: "list[ScenarioSpec]",
        channels: ChannelFactory,
        out_root: "str | Path",
        mode: "Literal['E0', 'E1']" = "E0",
        remote_world: bool = False,
        strict_skill_selection: bool = False,
        strict_tracker_capture: bool = False,
    ) -> None:
        self._scenarios = list(scenarios)
        self._channels = channels
        self._out_root = Path(out_root)
        self._mode = mode
        self._remote_world = remote_world
        self._strict_skill_selection = strict_skill_selection
        self._strict_tracker_capture = strict_tracker_capture

    def run(self) -> "list[tuple[str, None]]":
        from agents.platform import PlatformService
        from runtime import AuditLog, Router, Runtime
        from world.state import World

        from episode.artifacts import prepare_out_dir

        results: list[tuple[str, None]] = []
        shared_world = World(mode="E1") if self._mode == "E1" else None
        for index, spec in enumerate(self._scenarios):
            # Remove episode-owned artifacts BEFORE AuditLog opens its append-mode
            # streams, so a re-run does not duplicate/contaminate (shared with the
            # HTTP path).
            out_dir = prepare_out_dir(self._out_root / spec.scenario_id)
            if shared_world is None:
                world = World(mode="E0")
            else:
                world = shared_world
                if index:
                    world.reset("E1")
            audit = AuditLog(out_dir / "audit.jsonl", run_id=spec.scenario_id)
            from episode.actor_evidence import build_actor_evidence_services

            actor_context_resolver, actor_evidence_journal = (
                build_actor_evidence_services(spec, out_dir=out_dir)
            )
            population = population_for_scenario(spec)
            secrets = build_secret_registry(spec)
            participant_ids = frozenset(
                [buyer.buyer_id for buyer in population.buyers]
                + [merchant.merchant_id for merchant in population.merchants]
            )
            top_k = int(population.matching["top_k"]) if spec.population is not None else None
            max_transactions = (
                int(population.execution["max_transactions_per_buyer"])
                if spec.population is not None
                else None
            )
            # Router enforces the scenario-derived privacy boundary on the normal
            # launch path (previously Router() registered nothing, so the check
            # was inert). Secrets come from the scenario answer key, not memory.
            negotiation_max_rounds = int(
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
            )
            negotiation_deadline_ticks = int(
                population.execution.get("negotiation_deadline_ticks", 32)
            )
            runtime = Runtime(world=world, router=Router(secrets),
                              audit=audit,
                              platform=PlatformService(
                                  world=world,
                                  audit=audit,
                                  platform_policy=spec.platform_policy,
                                  max_search_results=top_k,
                                  max_transactions_per_buyer=max_transactions,
                                  secrets=secrets,
                                  negotiation_participants=participant_ids,
                                  negotiation_max_rounds=negotiation_max_rounds,
                                  negotiation_deadline_ticks=negotiation_deadline_ticks,
                              ),
                              actor_context_resolver=actor_context_resolver,
                              actor_evidence_journal=actor_evidence_journal,
                              remote_world=self._remote_world,
                              strict_tracker_capture=self._strict_tracker_capture)
            ep = Episode(scenario=spec, world=world, runtime=runtime, audit=audit,
                         channels=self._channels, out_dir=out_dir, mode=self._mode,
                         strict_skill_selection=self._strict_skill_selection)
            results.append((spec.scenario_id, ep.run()))
        return results
