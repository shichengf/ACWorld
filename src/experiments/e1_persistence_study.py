"""Three-Episode E1 persistence study over one reopened DatabaseWorld.

The study starts with one proposed order and advances the same durable order
lineage through payment, fulfillment, and refund in three independent
Episodes.  Each Episode owns a fresh Runtime, Agent registry, memory, channel,
authority context, and audit stream.  Between Episodes the SQLite World is
closed, reopened, and reset in E1 mode: transient catalog/inventory state is
reseeded while the order, ledger, shipment, protocol event, and receipt tables
remain authoritative.

Every lifecycle transition is selected as a provider-neutral business intent
by a scripted Agent and compiled through Platform's registered protocol-event
route.  This module never calls a World mutation or settlement method.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents.business_decision import LLMBusinessDecisionV1
from episode.capability_materializer import scenario_content_hash_v2
from episode.capability_runtime import RuntimeEvidenceBundleV2, canonical_sha256
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec
from experiments.environment_smoke import decision_log_artifact
from experiments.environment_study import (
    ARTIFACT_INDEX_NAME,
    DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES,
    ENVIRONMENT_STUDY_ARTIFACT_ORDER,
    ActorCoverageV1,
    EnvironmentStudyContractV1,
    EnvironmentStudyReportV1,
    InvariantResultV1,
    SourceManifestV1,
    build_artifact_index,
    build_environment_study_contract,
    build_source_manifest,
    capture_git_metadata,
    iter_loaded_module_files,
    network_disabled,
    repo_source_paths,
    unlisted_loaded_sources,
    verify_environment_study_contract,
    write_json_artifact,
)
from experiments.multiagent_preflight import MultiAgentRunV1, run_multiagent_episode
from experiments.scripted_channel import (
    ScriptedChannelFactoryV1,
    ScriptedDecisionContextV1,
)
from runtime.exact_join import (
    PROTOCOL_EVENT_EVIDENCE_CONTRACT,
    VerifiedProtocolEventEvidence,
    VerifiedProtocolEventJoin,
)
from runtime.tracker_evidence import (
    verified_model_business_choices,
    verify_all_active_actor_tracker_evidence,
)


E1_STUDY_ID = "CWENV-E1-01"
E1_SEED = 2_026_0722
E1_MARKET_ID = "market:cwenv-e1-01"
E1_ORDER_ID = "order:cwenv-e1-01"
E1_BUYER_ID = "buyer:cwenv-e1-01"
E1_MERCHANT_ID = "merchant:cwenv-e1-01"
E1_SKU_ID = f"{E1_MERCHANT_ID}:sku:persistent-01"
E1_STREAM_ID = "stream:cwenv-e1-01:order"
_REPO_ROOT = Path(__file__).resolve().parents[2]

_STEPS = (
    ("payment", "payment.settle", E1_BUYER_ID, "settled"),
    ("fulfillment", "fulfillment.dispatch", E1_MERCHANT_ID, "dispatched"),
    ("refund", "payment.refund", E1_MERCHANT_ID, "refunded"),
)

_INVARIANTS = (
    "scenario_sequence_frozen",
    "three_episode_e1_sequence",
    "database_reopened_after_each_episode",
    "order_lineage_persisted",
    "fulfillment_lineage_persisted",
    "refund_ledger_closed",
    "protocol_event_exact_join",
    "tracker_causal_closure",
    "agent_platform_world_lineage",
    "strict_replay_all_episodes",
    "episodes_completed_without_termination",
    "zero_provider_calls",
    "network_egress_disabled",
    "no_direct_world_settlement",
    "execution_sources_declared",
)

_EVIDENCE_REFS: Mapping[str, tuple[str, ...]] = {
    "scenario_sequence_frozen": ("contract.json",),
    "three_episode_e1_sequence": ("payment/world.final.json", "refund/world.final.json"),
    "database_reopened_after_each_episode": ("report.json",),
    "order_lineage_persisted": (
        "payment/world.final.json",
        "fulfillment/world.initial.json",
        "refund/world.initial.json",
    ),
    "fulfillment_lineage_persisted": (
        "fulfillment/world.final.json",
        "refund/world.initial.json",
    ),
    "refund_ledger_closed": ("refund/world.final.json",),
    "protocol_event_exact_join": (
        "platform.decisions.jsonl",
        "world.commits.jsonl",
    ),
    "tracker_causal_closure": ("actor.evidence.jsonl", "audit.trace.jsonl"),
    "agent_platform_world_lineage": (
        "platform.decisions.jsonl",
        "world.commits.jsonl",
    ),
    "strict_replay_all_episodes": ("evidence-manifest.json",),
    "episodes_completed_without_termination": ("termination.json-absent",),
    "zero_provider_calls": ("hashes-only-decision-log.json",),
    "network_egress_disabled": ("network-guard.json",),
    "no_direct_world_settlement": ("audit.trace.jsonl", "world.commits.jsonl"),
    "execution_sources_declared": ("source-manifest.json", "loaded-module-audit"),
}


class E1PersistenceStudyError(RuntimeError):
    """The three-Episode E1 evidence could not prove its invariants."""


def _execution_contract() -> dict[str, Any]:
    return {
        "schema_version": "cwe.public-task-execution.v1",
        "phases": [
            {
                "phase_id": "persistent_protocol_event_decision",
                "match": {
                    "actor_roles": ["buyer", "merchant"],
                    "inbound_action_kinds": ["platform.deliver_protocol_event"],
                    "inbound_senders": ["platform:events"],
                },
                "allowed_routes": [
                    {
                        "action_kind": "commerce.process_protocol_event",
                        "destination": "platform:events",
                    },
                    {
                        "action_kind": "commerce.reject_protocol_event",
                        "destination": "platform:events",
                    },
                ],
                "world_reads": "skill_scoped",
                "finish": "forbid",
            },
            {
                "phase_id": "persistent_protocol_event_receipt",
                "match": {
                    "actor_roles": ["buyer", "merchant"],
                    "inbound_action_kinds": ["platform.protocol_event_receipt"],
                    "inbound_senders": ["platform:events"],
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
        ],
    }


def _walk(value: Any) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        rows.append(value)
        for item in value.values():
            rows.extend(_walk(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            rows.extend(_walk(item))
    return tuple(rows)


class ProcessCurrentProtocolEventPolicyV1:
    """Process only a provider-visible current lifecycle callback."""

    def __call__(self, context: ScriptedDecisionContextV1) -> LLMBusinessDecisionV1:
        rows = _walk(context.observations)
        current = [
            row
            for row in rows
            if {
                "event_kind",
                "required_order_state",
                "current_order_state",
                "required_state_snapshot_is_current",
                "same_event_already_decided",
                "current_tick",
                "expires_at_tick",
            }.issubset(row)
        ]
        visible = {
            (
                row.get("event_kind"),
                row.get("required_order_state"),
                row.get("current_order_state"),
                row.get("required_state_snapshot_is_current"),
                row.get("same_event_already_decided"),
                row.get("current_tick"),
                row.get("expires_at_tick"),
            )
            for row in current
        }
        process = False
        if len(visible) == 1:
            (
                _kind,
                required_state,
                current_state,
                snapshot_current,
                already_decided,
                current_tick,
                expires_at_tick,
            ) = next(iter(visible))
            process = bool(
                required_state == current_state
                and snapshot_current is True
                and already_decided is False
                and isinstance(current_tick, int)
                and not isinstance(current_tick, bool)
                and isinstance(expires_at_tick, int)
                and not isinstance(expires_at_tick, bool)
                and current_tick <= expires_at_tick
            )
        if process and "process_protocol_event" in context.allowed_intent_names:
            return LLMBusinessDecisionV1(
                intent="process_protocol_event",
                arguments={"reason": "The current lifecycle callback is valid."},
            )
        if "reject_protocol_event" in context.allowed_intent_names:
            return LLMBusinessDecisionV1(
                intent="reject_protocol_event",
                arguments={"reason": "The callback is not current for this order."},
            )
        raise E1PersistenceStudyError("protocol-event phase exposes no valid decision")


def _listing(*, qty_reserved: int) -> dict[str, Any]:
    return {
        "sku_id": E1_SKU_ID,
        "product_id": "product:cwenv-e1-01",
        "merchant_id": E1_MERCHANT_ID,
        "category": "environment-study",
        "name": "Persistent lifecycle item",
        "list_price": "10.00",
        "inventory": 4,
        "qty_reserved": qty_reserved,
        "attributes": {"persistence_study": True, "shipping_days": 2},
    }


def _order() -> dict[str, Any]:
    return {
        "order_id": E1_ORDER_ID,
        "buyer_id": E1_BUYER_ID,
        "merchant_id": E1_MERCHANT_ID,
        "sku_id": E1_SKU_ID,
        "qty": 1,
        "agreed_price": "10.00",
        "currency": "USD",
        "state": "proposed",
    }


def _issue_event(step: str, event_kind: str, recipient_id: str) -> dict[str, Any]:
    event_id = f"event:cwenv-e1-01:{step}"
    key = f"issue:{event_id}"
    return {
        "msg_id": key,
        "ts": "2026-07-22T12:00:00Z",
        "from": "runtime:events",
        "to": "platform:events",
        "idempotency_key": key,
        "action": {
            "kind": "platform.issue_protocol_event",
            "payload": {
                "market_id": E1_MARKET_ID,
                "stream_id": E1_STREAM_ID,
                "order_id": E1_ORDER_ID,
                "recipient_id": recipient_id,
                "event_id": event_id,
                "event_kind": event_kind,
                "ttl_ticks": 32,
            },
        },
    }


def _buyer(task_context: Mapping[str, Any]) -> BuyerSpec:
    return BuyerSpec(
        buyer_id=E1_BUYER_ID,
        persona={"name": "E1 persistence buyer", "role": "buyer"},
        mandate={
            "mandate_id": "mandate:cwenv-e1-01",
            "goal": (
                "Process only current order-bound payment, fulfillment, and refund callbacks."
            ),
            "hard_constraints": {"budget": 10_000, "must_have": []},
            "soft_constraints": [],
            "soft_preferences": {"style": [], "avoid": []},
            "authority": {
                "can_buy_without_confirmation": True,
                "must_not_share_with_merchant": ["budget"],
            },
            "intent_expiry": "2099-12-31T00:00:00Z",
            "task_context": copy.deepcopy(dict(task_context)),
            "benchmark_contract": {
                "schema_version": "cwe.environment-study-order-scope.v1",
                "order_ids": [E1_ORDER_ID],
            },
        },
    )


def _merchant(task_context: Mapping[str, Any]) -> MerchantSpec:
    return MerchantSpec(
        merchant_id=E1_MERCHANT_ID,
        persona={"name": "E1 persistence merchant", "role": "merchant"},
        policy={
            "floor_price": 0,
            "margin_target_bps": 0,
            "max_negotiation_rounds": 1,
            "refund_policy": "30_day_return",
            "claim_aggressiveness": "neutral",
            "task_context": copy.deepcopy(dict(task_context)),
            "benchmark_contract": {
                "schema_version": "cwe.environment-study-order-scope.v1",
                "order_ids": [E1_ORDER_ID],
            },
        },
        catalog_scope=(E1_SKU_ID,),
    )


def build_e1_scenarios() -> tuple[ScenarioSpec, ...]:
    """Freeze the three independent Episodes sharing one durable order."""

    task_context = {
        "schema_version": "cwe.environment-study-e1-protocol.v1",
        "execution_contract": _execution_contract(),
    }
    scenarios: list[ScenarioSpec] = []
    for index, (step, event_kind, recipient_id, _state) in enumerate(_STEPS):
        # E1 intentionally resets current inventory between Episodes.  Once
        # payment has settled, the reseeded inventory must reconstruct the
        # durable order's one-unit reservation so dispatch/refund retain the
        # same order-inventory closure instead of manufacturing free stock.
        initial_state: dict[str, Any] = {
            "catalog": [_listing(qty_reserved=0 if index == 0 else 1)]
        }
        if index == 0:
            initial_state.update({"orders": [_order()], "logical_time": 0})
        scenarios.append(
            ScenarioSpec(
                scenario_id=f"cwenv_e1_01__{index + 1}_{step}",
                seed=E1_SEED + index,
                initial_state=initial_state,
                buyer_goal={},
                merchant_policy={},
                allowed_actions=("settle", "dispatch", "refund"),
                success_oracle={"schema_version": "cwe.environment-study-no-score.v1"},
                population=PopulationSpec(
                    buyers=(_buyer(task_context),),
                    merchants=(_merchant(task_context),),
                    initial_events=(_issue_event(step, event_kind, recipient_id),),
                    matching={"top_k": 1},
                    execution={"max_transactions_per_buyer": 1},
                ),
            )
        )
    return tuple(scenarios)


def make_e1_factory() -> ScriptedChannelFactoryV1:
    return ScriptedChannelFactoryV1(
        policies={},
        actor_policies={
            E1_BUYER_ID: ProcessCurrentProtocolEventPolicyV1(),
            E1_MERCHANT_ID: ProcessCurrentProtocolEventPolicyV1(),
        },
    )


def _scenario_sequence_digest(scenarios: Sequence[ScenarioSpec]) -> str:
    return str(
        canonical_sha256(
            {
                scenario.scenario_id: scenario_content_hash_v2(scenario)
                for scenario in scenarios
            }
        )
    )


def build_e1_contract(
    *,
    scenarios: Sequence[ScenarioSpec],
    repo_root: str | Path = _REPO_ROOT,
) -> tuple[EnvironmentStudyContractV1, SourceManifestV1]:
    manifest = build_source_manifest(repo_root, DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES)
    policy_digest = hashlib.sha256(
        inspect.getsource(ProcessCurrentProtocolEventPolicyV1).encode("utf-8")
    ).hexdigest()
    contract = build_environment_study_contract(
        git=capture_git_metadata(repo_root),
        scenario_digest=_scenario_sequence_digest(scenarios),
        actor_policy_digests=(
            (E1_BUYER_ID, policy_digest),
            (E1_MERCHANT_ID, policy_digest),
        ),
        component_digests=manifest.component_digests(),
        data_provenance_digest=str(
            canonical_sha256(
                {
                    "schema_version": "cwe.environment-study-e1-workflow.v1",
                    "order_id": E1_ORDER_ID,
                    "stream_id": E1_STREAM_ID,
                    "steps": [row[1] for row in _STEPS],
                    "synthetic_fixture": True,
                }
            )
        ),
        backend="database_world_e1_reopen_each_episode",
        transport="in_process",
        seed=E1_SEED,
        invariants=_INVARIANTS,
    )
    verify_environment_study_contract(contract, manifest, repo_root)
    return contract, manifest


@dataclass(frozen=True, slots=True)
class E1EpisodeRunV1:
    step: str
    event_kind: str
    expected_state: str
    recipient_id: str
    run: MultiAgentRunV1


def run_e1_sequence(
    *,
    scenarios: Sequence[ScenarioSpec],
    out_root: str | Path,
    database_path: str | Path,
) -> tuple[E1EpisodeRunV1, ...]:
    if len(scenarios) != len(_STEPS):
        raise E1PersistenceStudyError("E1 scenario sequence must contain exactly three Episodes")
    results: list[E1EpisodeRunV1] = []
    for index, (scenario, step_row) in enumerate(zip(scenarios, _STEPS, strict=True)):
        step, event_kind, recipient_id, expected_state = step_row
        factory = make_e1_factory()
        run = run_multiagent_episode(
            scenario=scenario,
            factory=factory,
            out_root=out_root,
            database_path=database_path,
            world_mode="E1",
            reset_before_run=index > 0,
        )
        results.append(
            E1EpisodeRunV1(
                step=step,
                event_kind=event_kind,
                expected_state=expected_state,
                recipient_id=recipient_id,
                run=run,
            )
        )
    return tuple(results)


def _tables(evidence: RuntimeEvidenceBundleV2, *, final: bool) -> Mapping[str, Any]:
    snapshot = evidence.final_world if final else evidence.initial_world
    tables = snapshot.get("tables")
    if not isinstance(tables, Mapping):
        raise E1PersistenceStudyError("E1 evidence snapshot has no table object")
    return tables


def _table_rows(
    evidence: RuntimeEvidenceBundleV2, table: str, *, final: bool
) -> tuple[Mapping[str, Any], ...]:
    rows = _tables(evidence, final=final).get(table, [])
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise E1PersistenceStudyError(f"E1 {table} table is malformed")
    return tuple(rows)


def _order_state(evidence: RuntimeEvidenceBundleV2, *, final: bool) -> str | None:
    matches = [
        row
        for row in _table_rows(evidence, "orders", final=final)
        if row.get("order_id") == E1_ORDER_ID
    ]
    return str(matches[0].get("state")) if len(matches) == 1 else None


def _verified_protocol_join(row: E1EpisodeRunV1) -> VerifiedProtocolEventJoin | None:
    event_id = f"event:cwenv-e1-01:{row.step}"
    try:
        verified = row.run.evidence.verified_operation_evidence(
            PROTOCOL_EVENT_EVIDENCE_CONTRACT,
            options={"expected_event_ids": [event_id]},
        )
    except Exception:  # noqa: BLE001 - exact evidence is a validity gate
        return None
    if not isinstance(verified, VerifiedProtocolEventEvidence):
        return None
    by_event = verified.by_event_id()
    return by_event.get(event_id) if set(by_event) == {event_id} else None


def _tracker(row: E1EpisodeRunV1) -> Any:
    try:
        verdict = verify_all_active_actor_tracker_evidence(
            row.run.evidence,
            declared_actor_ids=row.run.factory.actor_ids(),
            evaluated_actor_id=row.recipient_id,
            evaluated_actor_strict=True,
        )
    except Exception:  # noqa: BLE001 - tracker closure is a validity gate
        return None
    return verdict if verdict.verified else None


def _combined_decision_log(rows: Sequence[E1EpisodeRunV1]) -> dict[str, Any]:
    return {
        "schema_version": "cwe.e1-scripted-decision-log.v1",
        "provider_calls": 0,
        "episodes": [
            {
                "scenario_id": row.run.scenario.scenario_id,
                "step": row.step,
                "records": decision_log_artifact(row.run.factory)["records"],
            }
            for row in rows
        ],
    }


def _aggregate_coverage(
    rows: Sequence[E1EpisodeRunV1],
    trackers: Sequence[Any],
    joins: Sequence[VerifiedProtocolEventJoin | None],
) -> tuple[ActorCoverageV1, ...]:
    coverage: list[ActorCoverageV1] = []
    for actor_id, role in ((E1_BUYER_ID, "buyer"), (E1_MERCHANT_ID, "merchant")):
        scripted = 0
        verified = 0
        complete = 0
        accepted = 0
        commits = 0
        for row, tracker, join in zip(rows, trackers, joins, strict=True):
            channel = row.run.factory.channel_for(actor_id)
            scripted += len(channel.decision_log)
            actor_verdict = next(
                (
                    value
                    for value in tracker.actor_verdicts
                    if value.evaluated_actor_id == actor_id
                ),
                None,
            ) if tracker is not None else None
            if actor_verdict is not None:
                verified += len(
                    verified_model_business_choices(
                        row.run.evidence,
                        evaluated_actor_id=actor_id,
                        strict_ideal=False,
                    )
                )
                complete += int(actor_verdict.complete_record_count)
            accepted += len(
                row.run.evidence.accepted_platform_exchanges(actor_id=actor_id)
            )
            if (
                join is not None
                and join.receipt_commit is not None
                and join.receipt_commit.get("actor_id") == actor_id
            ):
                commits += 1
        coverage.append(
            ActorCoverageV1(
                actor_id=actor_id,
                role=role,
                scripted_business_decisions=scripted,
                verified_business_decisions=verified,
                complete_tracker_records=complete,
                accepted_platform_operations=accepted,
                causally_linked_world_commits=commits,
            )
        )
    return tuple(coverage)


def build_e1_report(
    *,
    contract: EnvironmentStudyContractV1,
    manifest: SourceManifestV1,
    scenarios: Sequence[ScenarioSpec],
    rows: Sequence[E1EpisodeRunV1],
    network_guard: Mapping[str, Any],
    loaded_sources: Sequence[str],
) -> EnvironmentStudyReportV1:
    from episode.replay import verify_episode_replay

    joins = tuple(_verified_protocol_join(row) for row in rows)
    trackers = tuple(_tracker(row) for row in rows)
    replays = tuple(
        verify_episode_replay(row.run.evidence.episode_dir, strict=True) for row in rows
    )
    final_states = tuple(_order_state(row.run.evidence, final=True) for row in rows)
    carried_states = tuple(
        _order_state(rows[index].run.evidence, final=False)
        for index in range(1, len(rows))
    )
    reopen_ok = all(
        row.run.reopened_digest == row.run.evidence.final_digest
        for row in rows
    )
    final = rows[-1].run.evidence
    ledger = _table_rows(final, "ledger", final=True)
    shipments = _table_rows(final, "shipments", final=True)
    protocol_events = _table_rows(final, "protocol_events", final=True)
    protocol_receipts = _table_rows(final, "protocol_receipts", final=True)
    order_ids = {
        str(row.get("order_id"))
        for evidence_row in rows
        for row in _table_rows(evidence_row.run.evidence, "orders", final=True)
    }
    linked_joins = all(
        join is not None
        and join.event.event_kind == row.event_kind
        and join.event.binding.order_id == E1_ORDER_ID
        and join.event.binding.recipient_id == row.recipient_id
        and join.receipt is not None
        and join.receipt.decision == "process"
        and join.receipt_commit is not None
        and join.receipt_commit.get("actor_id") == row.recipient_id
        and join.operation is not None
        and join.outcome_row is not None
        for row, join in zip(rows, joins, strict=True)
    )
    choices_ok = all(
        tuple(
            choice.intent
            for choice in verified_model_business_choices(
                row.run.evidence,
                evaluated_actor_id=row.recipient_id,
                strict_ideal=False,
            )
        )
        == ("process_protocol_event",)
        for row in rows
        if _tracker(row) is not None
    ) and all(tracker is not None for tracker in trackers)
    process_commits = [
        commit
        for row in rows
        for commit in row.run.evidence.world_events
        if commit.get("operation") == "process_protocol_event"
    ]
    unlisted = unlisted_loaded_sources(manifest, loaded_sources)
    checks = {
        "scenario_sequence_frozen": (
            _scenario_sequence_digest(scenarios) == contract.scenario_digest
            and contract.seed == E1_SEED
        ),
        "three_episode_e1_sequence": (
            len(rows) == 3
            and final_states == tuple(step[3] for step in _STEPS)
        ),
        "database_reopened_after_each_episode": reopen_ok,
        "order_lineage_persisted": (
            order_ids == {E1_ORDER_ID}
            and carried_states == ("settled", "dispatched")
        ),
        "fulfillment_lineage_persisted": (
            len(shipments) == 1
            and shipments[0].get("order_id") == E1_ORDER_ID
            and _table_rows(rows[2].run.evidence, "shipments", final=False)
            == shipments
        ),
        "refund_ledger_closed": (
            len(ledger) == 2
            and {row.get("effect") for row in ledger} == {"charge", "refund"}
            and all(row.get("order_id") == E1_ORDER_ID for row in ledger)
        ),
        "protocol_event_exact_join": (
            linked_joins
            and len(protocol_events) == len(protocol_receipts) == 3
        ),
        "tracker_causal_closure": all(tracker is not None for tracker in trackers),
        "agent_platform_world_lineage": choices_ok and len(process_commits) == 3,
        "strict_replay_all_episodes": all(replay.replay_ok for replay in replays),
        "episodes_completed_without_termination": all(
            not (Path(row.run.evidence.episode_dir) / "termination.json").exists()
            for row in rows
        ),
        "zero_provider_calls": all(row.run.factory.provider_calls == 0 for row in rows),
        "network_egress_disabled": (
            network_guard.get("blocked_connect_count") == 0
            and network_guard.get("egress_free") is True
        ),
        "no_direct_world_settlement": (
            linked_joins
            and len(process_commits) == 3
            and all(
                "business-and-receipt-atomic" in commit.get("invariants_held", ())
                for commit in process_commits
            )
        ),
        "execution_sources_declared": not unlisted,
    }
    invariants = tuple(
        InvariantResultV1(
            name=name,
            passed=checks[name],
            evidence_refs=_EVIDENCE_REFS[name],
            failure_reason=(
                None
                if checks[name]
                else (
                    "undeclared loaded source: " + ", ".join(unlisted)
                    if name == "execution_sources_declared" and unlisted
                    else f"{name} not proven by CWENV-E1-01 evidence"
                )
            ),
        )
        for name in _INVARIANTS
    )
    return EnvironmentStudyReportV1(
        study_id=E1_STUDY_ID,
        contract_id=contract.contract_id,
        valid=all(checks.values()),
        invariants=invariants,
        actor_coverage=_aggregate_coverage(rows, trackers, joins),
        transaction_summary={
            "episodes": len(rows),
            "orders": len(order_ids),
            "order_id_sha256": hashlib.sha256(E1_ORDER_ID.encode("utf-8")).hexdigest(),
            "state_progression": list(final_states),
        },
        inventory_summary={"catalog_reseeded_each_episode": True},
        ledger_summary={
            "entries": len(ledger),
            "charge_entries": sum(row.get("effect") == "charge" for row in ledger),
            "refund_entries": sum(row.get("effect") == "refund" for row in ledger),
        },
        fulfillment_summary={
            "shipments": len(shipments),
            "persisted_into_refund_episode": bool(
                _table_rows(rows[2].run.evidence, "shipments", final=False)
            ),
        },
        after_sales_summary={
            "refund_completed": final_states[-1] == "refunded",
            "disputes": len(_table_rows(final, "disputes", final=True)),
        },
        replay={
            "all_replay_ok": all(replay.replay_ok for replay in replays),
            "episodes": [
                {
                    "step": row.step,
                    "transactions_replayed": int(replay.transactions_replayed),
                    "commits_replayed": int(replay.commits_replayed),
                }
                for row, replay in zip(rows, replays, strict=True)
            ],
        },
        diagnostics={
            "database_reopen_checks": len(rows),
            "reopened_snapshot_digests": [row.run.reopened_digest for row in rows],
            "loaded_source_count": len(set(loaded_sources)),
            "unlisted_source_count": len(unlisted),
        },
        data_scope={
            "fixture_kind": "synthetic_protocol_lifecycle",
            "database_mode": "E1",
            "episode_count": 3,
            "provider_data_sent": False,
            "paper_result": True,
        },
    )


@dataclass(frozen=True, slots=True)
class PersistedE1StudyV1:
    report: EnvironmentStudyReportV1
    artifacts_dir: str
    artifact_index: Mapping[str, Any]
    network_guard: Mapping[str, Any]


def run_persisted_e1_study(
    *,
    out_root: str | Path,
    artifacts_dir: str | Path,
    repo_root: str | Path = _REPO_ROOT,
) -> PersistedE1StudyV1:
    root = Path(repo_root).resolve(strict=True)
    scenarios = build_e1_scenarios()
    contract, manifest = build_e1_contract(scenarios=scenarios, repo_root=root)
    artifacts = Path(artifacts_dir)
    write_json_artifact(manifest.to_dict(), artifacts / "source-manifest.json")
    write_json_artifact(contract.to_dict(), artifacts / "contract.json")
    with network_disabled() as guard:
        rows = run_e1_sequence(
            scenarios=scenarios,
            out_root=out_root,
            database_path=Path(out_root) / "cwenv-e1-01.sqlite3",
        )
    loaded = repo_source_paths(iter_loaded_module_files(), root)
    report = build_e1_report(
        contract=contract,
        manifest=manifest,
        scenarios=scenarios,
        rows=rows,
        network_guard=guard.to_dict(),
        loaded_sources=tuple(loaded),
    )
    write_json_artifact(
        _combined_decision_log(rows), artifacts / "hashes-only-decision-log.json"
    )
    write_json_artifact(guard.to_dict(), artifacts / "network-guard.json")
    write_json_artifact(report.to_dict(), artifacts / "report.json")
    index = build_artifact_index(artifacts, ENVIRONMENT_STUDY_ARTIFACT_ORDER)
    write_json_artifact(index, artifacts / ARTIFACT_INDEX_NAME)
    return PersistedE1StudyV1(
        report=report,
        artifacts_dir=str(artifacts),
        artifact_index=index,
        network_guard=guard.to_dict(),
    )


__all__ = [
    "E1_BUYER_ID",
    "E1_MERCHANT_ID",
    "E1_ORDER_ID",
    "E1_STUDY_ID",
    "E1PersistenceStudyError",
    "PersistedE1StudyV1",
    "ProcessCurrentProtocolEventPolicyV1",
    "build_e1_contract",
    "build_e1_report",
    "build_e1_scenarios",
    "make_e1_factory",
    "run_e1_sequence",
    "run_persisted_e1_study",
]
