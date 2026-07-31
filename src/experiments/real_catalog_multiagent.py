"""Local-only 5x5 Agent study over the repository's real CSV catalog.

The catalog still enters through ``catalog_provenance -> merchant_data_csv ->
seed_world_catalog``.  This module only selects one already-canonical public
listing per selected merchant to define five independent quote requests.  Ten
actor-bound scripted channels then drive the normal Agent -> Platform -> World
path; neither this harness nor its policies receive a World handle or call a
settlement API.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from episode.benchmark import (
    BenchmarkMetadata,
    BenchmarkTrack,
    CatalogScale,
    CatalogSource,
    Difficulty,
    TaskFamily,
)
from episode.capability_materializer import scenario_content_hash_v2
from episode.catalog_provenance import DEFAULT_MANIFEST
from episode.seed_catalog import seed_world_catalog
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec
from experiments.catalog_feasibility import (
    CatalogFeasibilityReportV1,
    build_catalog_feasibility_report,
    verify_catalog_feasibility_report,
)
from experiments.data_environment_study import (
    _minor_units,
    _selected_merchant_slugs,
    verify_catalog_backend_parity,
)
from experiments.environment_smoke import decision_log_artifact
from experiments.environment_study import (
    ARTIFACT_INDEX_NAME,
    DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES,
    ENVIRONMENT_STUDY_ARTIFACT_ORDER,
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
from experiments.multiagent_preflight import (
    CartBuyerPolicyV1,
    CartMerchantPolicyV1,
    MultiAgentRunV1,
    _actor_coverage,
    _cart_execution_contract,
    _causal_commit_links,
    _tracker_verdict,
    run_multiagent_episode,
)
from experiments.scripted_channel import ScriptedChannelFactoryV1
from protocol.evidence_records import (
    MandateRevisionAuthority,
    build_mandate_revision,
    mandate_revision_to_dict,
)
from world.evidence_contracts import mandate_authority_to_wire
from world.state import World


REAL_CATALOG_M2M_STUDY_ID = "CWENV-DATA-M2M-01"
REAL_CATALOG_M2M_SCENARIO_ID = "cwenv_data_m2m_01__real_csv_5x5"
REAL_CATALOG_M2M_SEED = 42
REAL_CATALOG_M2M_MARKET_ID = "market:cwenv-data-m2m-01"
_REPO_ROOT = Path(__file__).resolve().parents[2]

_INVARIANTS = (
    "scenario_content_frozen",
    "catalog_provenance_verified",
    "real_catalog_database_reopen_parity",
    "ten_actor_boundary",
    "actor_channel_isolation",
    "tracker_causal_closure",
    "agent_platform_world_lineage",
    "five_quote_bound_transactions",
    "inventory_and_ledger_closed",
    "owner_sku_namespace_closed",
    "strict_replay",
    "episode_completed_without_termination",
    "zero_provider_calls",
    "network_egress_blocked",
    "raw_csv_rows_not_persisted",
    "no_direct_settlement",
    "execution_sources_declared",
)

_EVIDENCE_REFS: Mapping[str, tuple[str, ...]] = {
    "scenario_content_frozen": ("contract.json", "world.initial.json"),
    "catalog_provenance_verified": ("catalog-feasibility.v1.json",),
    "real_catalog_database_reopen_parity": ("report.json",),
    "ten_actor_boundary": ("actor.evidence.jsonl",),
    "actor_channel_isolation": ("hashes-only-decision-log.json",),
    "tracker_causal_closure": ("audit.trace.jsonl",),
    "agent_platform_world_lineage": (
        "platform.decisions.jsonl",
        "world.commits.jsonl",
    ),
    "five_quote_bound_transactions": ("world.final.json", "world.commits.jsonl"),
    "inventory_and_ledger_closed": ("world.initial.json", "world.final.json"),
    "owner_sku_namespace_closed": ("world.final.json",),
    "strict_replay": ("replay.json",),
    "episode_completed_without_termination": ("termination.json-absent",),
    "zero_provider_calls": ("hashes-only-decision-log.json",),
    "network_egress_blocked": ("network-guard.json",),
    "raw_csv_rows_not_persisted": ("artifact-index.json",),
    "no_direct_settlement": ("audit.trace.jsonl", "world.commits.jsonl"),
    "execution_sources_declared": ("source-manifest.json", "loaded-module-audit"),
}


class RealCatalogMultiAgentError(RuntimeError):
    """The local real-catalog 5x5 evidence could not be proven."""


def _buyer_id(index: int) -> str:
    return f"buyer:cwenv-data-m2m-b{index}"


def _principal_id(index: int) -> str:
    return f"consumer:cwenv-data-m2m-b{index}"


def _mandate_id(index: int) -> str:
    return f"mandate:cwenv-data-m2m-b{index}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _selected_listings(
    feasibility: CatalogFeasibilityReportV1,
) -> tuple[Any, ...]:
    world = World()
    seed_world_catalog(
        world,
        merchants=_selected_merchant_slugs(feasibility),
        seed=42,
        catalog_scale="medium",
        in_stock_only=True,
    )
    by_merchant: dict[str, list[Any]] = {
        merchant_id: [] for merchant_id in feasibility.selected_merchant_ids
    }
    for listing in world.snapshot().catalog:
        merchant_id = str(listing.merchant_id)
        if merchant_id in by_merchant:
            by_merchant[merchant_id].append(listing)
    selected: list[Any] = []
    for merchant_id in feasibility.selected_merchant_ids:
        candidates = sorted(
            by_merchant[merchant_id],
            key=lambda row: (_minor_units(row.list_price.amount), str(row.sku_id)),
        )
        if not candidates:
            raise RealCatalogMultiAgentError(
                f"selected merchant has no medium-profile listing: {merchant_id}"
            )
        selected.append(candidates[0])
    return tuple(selected)


def _pricing_terms(listing: Any) -> list[dict[str, Any]]:
    unit = _minor_units(listing.list_price.amount)
    return [
        {
            "sku_ids": [str(listing.sku_id)],
            "quantity_tiers": [
                {
                    "minimum_quantity": 1,
                    "maximum_quantity": None,
                    "unit_price_minor": unit,
                }
            ],
            "bundle_discounts": [],
            "bundle_stacking": "best_only",
            "charges": [],
        }
    ]


def _pricing_fixture(listing: Any, index: int) -> dict[str, Any]:
    unit = _minor_units(listing.list_price.amount)
    return {
        "merchant_id": str(listing.merchant_id),
        "idempotency_key": f"seed:cwenv-data-m2m:pricing:{index}",
        "intent": {
            "market_id": REAL_CATALOG_M2M_MARKET_ID,
            "policy_id": f"policy:cwenv-data-m2m:{index}",
            "listing_ids": [str(listing.sku_id)],
            "product_ids": [],
            "quantity_tiers": [
                {
                    "minimum_quantity": 1,
                    "maximum_quantity": None,
                    "unit_price_minor": unit,
                }
            ],
            "bundle_discounts": [],
            "bundle_stacking": "best_only",
            "components": [],
            "effective_after_ticks": 0,
            "expires_after_ticks": None,
        },
    }


def _mandate_rows(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = MandateRevisionAuthority(
        principal_id=_principal_id(index),
        buyer_id=_buyer_id(index),
        mandate_id=_mandate_id(index),
        allowed_fields=("budget_minor",),
    )
    revision = build_mandate_revision(
        principal_id=_principal_id(index),
        buyer_id=_buyer_id(index),
        mandate_id=_mandate_id(index),
        revision=1,
        previous_digest=None,
        changes={"budget_minor": 100_000_000},
        authorized_fields=("budget_minor",),
        logical_tick=0,
    )
    return mandate_authority_to_wire(authority), dict(
        mandate_revision_to_dict(revision)
    )


def _initial_event(listing: Any, index: int) -> dict[str, Any]:
    return {
        "msg_id": f"kickoff:cwenv-data-m2m:cart:{index}",
        "ts": "2026-07-21T12:00:00Z",
        "from": _buyer_id(index),
        "to": "platform:checkout",
        "idempotency_key": f"authorize:cwenv-data-m2m:{index}",
        "action": {
            "kind": "commerce.create_cart_quote_request",
            "payload": {
                "market_id": REAL_CATALOG_M2M_MARKET_ID,
                "mandate_id": _mandate_id(index),
                "lines": [{"sku_id": str(listing.sku_id), "qty": 1}],
                "fill_policy": "all_or_none",
                "backorder_policy": "reject",
            },
        },
    }


def build_real_catalog_multiagent_scenario(
    feasibility: CatalogFeasibilityReportV1 | None = None,
) -> ScenarioSpec:
    report = feasibility or build_catalog_feasibility_report()
    verify_catalog_feasibility_report(report)
    if report.real_csv_5x5_status != "executable_local_only":
        raise RealCatalogMultiAgentError(report.real_csv_5x5_reason)
    listings = _selected_listings(report)
    execution_contract = _cart_execution_contract()
    buyers: list[BuyerSpec] = []
    merchants: list[MerchantSpec] = []
    authorities: list[dict[str, Any]] = []
    revisions: list[dict[str, Any]] = []
    for index, listing in enumerate(listings, start=1):
        authority, revision = _mandate_rows(index)
        authorities.append(authority)
        revisions.append(revision)
        buyers.append(
            BuyerSpec(
                buyer_id=_buyer_id(index),
                persona={"name": f"Local catalog buyer {index}", "role": "buyer"},
                mandate={
                    "mandate_id": _mandate_id(index),
                    "goal": "Checkout the independently authorized real-catalog quote.",
                    "quantity": 1,
                    "return_after_purchase": False,
                    "hard_constraints": {
                        "budget": 100_000_000,
                        "delivery_days": 30,
                        "must_have": [],
                    },
                    "soft_constraints": [],
                    "soft_preferences": {"style": [], "avoid": []},
                    "authority": {
                        "can_buy_without_confirmation": True,
                        "must_not_share_with_merchant": ["budget"],
                    },
                    "intent_expiry": "2099-12-31T00:00:00Z",
                    "task_context": {
                        "schema_version": "cwe.environment-study-real-catalog-cart.v1",
                        "execution_contract": copy.deepcopy(execution_contract),
                    },
                },
            )
        )
        merchants.append(
            MerchantSpec(
                merchant_id=str(listing.merchant_id),
                persona={
                    "name": f"Local catalog merchant {index}",
                    "role": "merchant",
                },
                policy={
                    "floor_price": 0,
                    "margin_target_bps": 0,
                    "max_negotiation_rounds": 1,
                    "refund_policy": "30_day_return",
                    "claim_aggressiveness": "neutral",
                    "task_context": {
                        "schema_version": "cwe.environment-study-real-catalog-cart.v1",
                        "execution_contract": copy.deepcopy(execution_contract),
                        "pricing_terms": _pricing_terms(listing),
                    },
                },
                catalog_scope=(str(listing.sku_id),),
            )
        )
    benchmark = BenchmarkMetadata(
        variant_id=REAL_CATALOG_M2M_STUDY_ID,
        task_family=TaskFamily.CART_QUANTITY,
        track=BenchmarkTrack.AGENT,
        difficulty=Difficulty.BASELINE,
        catalog_scale=CatalogScale.MEDIUM,
        catalog_source=CatalogSource.REAL_CSV,
        catalog_merchants=_selected_merchant_slugs(report),
        in_stock_only=True,
        scenario_version="1.0",
    )
    return ScenarioSpec(
        scenario_id=REAL_CATALOG_M2M_SCENARIO_ID,
        seed=REAL_CATALOG_M2M_SEED,
        initial_state={
            "catalog": [],
            "pricing_policy_fixtures": [
                _pricing_fixture(listing, index)
                for index, listing in enumerate(listings, start=1)
            ],
            "mandate_authorities": authorities,
            "mandate_revisions": revisions,
        },
        buyer_goal={},
        merchant_policy={},
        allowed_actions=(),
        success_oracle={"schema_version": "cwe.environment-study-no-score.v1"},
        benchmark=benchmark,
        population=PopulationSpec(
            buyers=tuple(buyers),
            merchants=tuple(merchants),
            initial_events=tuple(
                _initial_event(listing, index)
                for index, listing in enumerate(listings, start=1)
            ),
            matching={"top_k": 20},
            execution={"max_transactions_per_buyer": 1},
        ),
    )


def make_real_catalog_multiagent_factory() -> ScriptedChannelFactoryV1:
    return ScriptedChannelFactoryV1(
        policies={},
        actor_policies={
            **{
                _buyer_id(index): CartBuyerPolicyV1(checkout=True)
                for index in range(1, 6)
            },
            **{
                merchant_id: CartMerchantPolicyV1()
                for merchant_id in build_catalog_feasibility_report().selected_merchant_ids
            },
        },
    )


def _build_contract(
    *,
    scenario: ScenarioSpec,
    feasibility: CatalogFeasibilityReportV1,
    repo_root: Path,
) -> tuple[EnvironmentStudyContractV1, SourceManifestV1]:
    manifest = build_source_manifest(
        repo_root, DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES
    )
    buyer_digest = _sha256_text(inspect.getsource(CartBuyerPolicyV1))
    merchant_digest = _sha256_text(inspect.getsource(CartMerchantPolicyV1))
    actor_digests = tuple(
        sorted(
            [(_buyer_id(index), buyer_digest) for index in range(1, 6)]
            + [
                (merchant_id, merchant_digest)
                for merchant_id in feasibility.selected_merchant_ids
            ]
        )
    )
    contract = build_environment_study_contract(
        git=capture_git_metadata(repo_root),
        scenario_digest=scenario_content_hash_v2(scenario),
        actor_policy_digests=actor_digests,
        component_digests=manifest.component_digests(),
        data_provenance_digest=feasibility.report_id,
        backend="world_episode_with_real_csv_and_sqlite_seed_parity",
        transport="in_process",
        seed=scenario.seed,
        invariants=_INVARIANTS,
    )
    verify_environment_study_contract(contract, manifest, repo_root)
    return contract, manifest


def _build_report(
    *,
    contract: EnvironmentStudyContractV1,
    manifest: SourceManifestV1,
    feasibility: CatalogFeasibilityReportV1,
    run: MultiAgentRunV1,
    parity: Any,
    network_guard: Mapping[str, Any],
    loaded_sources: Sequence[str],
) -> EnvironmentStudyReportV1:
    from episode.replay import verify_episode_replay

    evidence = run.evidence
    replay = verify_episode_replay(evidence.episode_dir, strict=True)
    tracker = _tracker_verdict(run)
    try:
        commit_links = _causal_commit_links(evidence)
    except Exception:  # noqa: BLE001 - exact-join failure invalidates the report
        commit_links = {}
    coverage = _actor_coverage(run, tracker, commit_links)
    initial_tables = evidence.initial_world.get("tables", {})
    final_tables = evidence.final_world.get("tables", {})
    orders = [
        row
        for row in final_tables.get("orders", [])
        if isinstance(row, Mapping)
        and str(row.get("buyer_id", "")).startswith("buyer:cwenv-data-m2m-")
    ]
    ledger = [
        row
        for row in final_tables.get("ledger", [])
        if isinstance(row, Mapping)
        and str(row.get("buyer_id", "")).startswith("buyer:cwenv-data-m2m-")
    ]
    groups = [
        row
        for row in final_tables.get("order_groups", [])
        if isinstance(row, Mapping)
        and str(row.get("buyer_id", "")).startswith("buyer:cwenv-data-m2m-")
    ]
    before_inventory = initial_tables.get("inventory", {})
    after_inventory = final_tables.get("inventory", {})
    selected_skus = {str(row.get("sku_id")) for row in orders}
    inventory_changed = bool(selected_skus) and all(
        isinstance(before_inventory, Mapping)
        and isinstance(after_inventory, Mapping)
        and sku in before_inventory
        and sku in after_inventory
        and before_inventory[sku] != after_inventory[sku]
        for sku in selected_skus
    )
    actor_ids = {row.actor_id for row in coverage}
    expected_actor_ids = {
        *(_buyer_id(index) for index in range(1, 6)),
        *feasibility.selected_merchant_ids,
    }
    log = decision_log_artifact(run.factory)
    expected_log_fields = {
        "sequence",
        "actor_id",
        "role",
        "decision_id",
        "request_sha256",
        "decision_sha256",
    }
    no_raw_rows = all(
        isinstance(row, Mapping) and set(row) == expected_log_fields
        for row in log.get("records", [])
    )
    unlisted = unlisted_loaded_sources(manifest, loaded_sources)
    checks = {
        "scenario_content_frozen": (
            scenario_content_hash_v2(run.scenario) == contract.scenario_digest
            and run.scenario.seed == contract.seed
        ),
        "catalog_provenance_verified": (
            feasibility.report_id == contract.data_provenance_digest
        ),
        "real_catalog_database_reopen_parity": bool(parity.parity_ok),
        "ten_actor_boundary": actor_ids == expected_actor_ids and len(coverage) == 10,
        "actor_channel_isolation": (
            run.factory.actor_ids() == tuple(sorted(expected_actor_ids))
            and len({id(run.factory.channel_for(actor)) for actor in expected_actor_ids})
            == 10
        ),
        "tracker_causal_closure": tracker is not None,
        "agent_platform_world_lineage": (
            len(commit_links) == 10
            and sum(len(values) for values in commit_links.values())
            == len(evidence.world_events)
        ),
        "five_quote_bound_transactions": (
            len(orders) == len(ledger) == len(groups) == 5
            and all(row.get("state") == "settled" for row in orders)
        ),
        "inventory_and_ledger_closed": inventory_changed and len(ledger) == 5,
        "owner_sku_namespace_closed": len(orders) == 5
        and all(
            str(row.get("sku_id", "")).startswith(f"{row.get('merchant_id')}:sku:")
            for row in orders
        ),
        "strict_replay": bool(replay.replay_ok),
        "episode_completed_without_termination": not (
            Path(evidence.episode_dir) / "termination.json"
        ).exists(),
        "zero_provider_calls": run.factory.provider_calls == 0,
        "network_egress_blocked": (
            network_guard.get("blocked_connect_count") == 0
            and network_guard.get("egress_free") is True
        ),
        "raw_csv_rows_not_persisted": no_raw_rows,
        "no_direct_settlement": (
            set(commit_links) == expected_actor_ids
            and sum(len(values) for values in commit_links.values())
            == len(evidence.world_events)
            and all(
                len(commit_links.get(_buyer_id(index), ())) == 2
                for index in range(1, 6)
            )
            and all(
                len(commit_links.get(merchant_id, ())) == 1
                for merchant_id in feasibility.selected_merchant_ids
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
                    else f"{name} not proven by CWENV-DATA-M2M-01 evidence"
                )
            ),
        )
        for name in _INVARIANTS
    )
    return EnvironmentStudyReportV1(
        study_id=REAL_CATALOG_M2M_STUDY_ID,
        contract_id=contract.contract_id,
        valid=all(checks.values()),
        invariants=invariants,
        actor_coverage=coverage,
        transaction_summary={
            "order_groups": len(groups),
            "orders": len(orders),
            "settled_orders": sum(row.get("state") == "settled" for row in orders),
            "quote_bound": len(groups) == 5,
        },
        inventory_summary={
            "initial_listings": parity.listing_count,
            "selected_listing_rows_changed": sum(
                before_inventory.get(sku) != after_inventory.get(sku)
                for sku in selected_skus
            ),
        },
        ledger_summary={"entries": len(ledger), "order_bound": len(ledger) == 5},
        fulfillment_summary={},
        after_sales_summary={},
        replay={
            "replay_ok": bool(replay.replay_ok),
            "transactions_replayed": int(replay.transactions_replayed),
            "commits_replayed": int(replay.commits_replayed),
        },
        diagnostics={
            "loaded_source_count": len(set(loaded_sources)),
            "unlisted_source_count": len(unlisted),
            "selected_listing_count": len(selected_skus),
        },
        data_scope={
            "dataset_id": feasibility.dataset_id,
            "catalog_profile": "medium",
            "in_stock_only": True,
            "selected_merchant_ids": list(feasibility.selected_merchant_ids),
            "cross_merchant_product_linkage": False,
            "raw_rows_embedded": False,
            "paper_result": False,
            "publication_reason": "license_and_permission_unverified",
        },
    )


@dataclass(frozen=True, slots=True)
class PersistedRealCatalogMultiAgentV1:
    report: EnvironmentStudyReportV1
    artifacts_dir: str
    artifact_index: Mapping[str, Any]
    network_guard: Mapping[str, Any]


def run_persisted_real_catalog_multiagent(
    *,
    out_root: str | Path,
    artifacts_dir: str | Path,
    repo_root: str | Path = _REPO_ROOT,
) -> PersistedRealCatalogMultiAgentV1:
    root = Path(repo_root).resolve(strict=True)
    feasibility = build_catalog_feasibility_report(
        manifest_path=DEFAULT_MANIFEST,
        repository_root=root,
    )
    verify_catalog_feasibility_report(feasibility, repository_root=root)
    scenario = build_real_catalog_multiagent_scenario(feasibility)
    factory = make_real_catalog_multiagent_factory()
    contract, manifest = _build_contract(
        scenario=scenario,
        feasibility=feasibility,
        repo_root=root,
    )
    artifacts = Path(artifacts_dir)
    write_json_artifact(manifest.to_dict(), artifacts / "source-manifest.json")
    write_json_artifact(contract.to_dict(), artifacts / "contract.json")
    parity = verify_catalog_backend_parity(
        feasibility=feasibility,
        database_path=Path(out_root) / "real-catalog-5x5-parity.sqlite3",
    )
    with network_disabled() as guard:
        run = run_multiagent_episode(
            scenario=scenario,
            factory=factory,
            out_root=out_root,
        )
    loaded = repo_source_paths(iter_loaded_module_files(), root)
    report = _build_report(
        contract=contract,
        manifest=manifest,
        feasibility=feasibility,
        run=run,
        parity=parity,
        network_guard=guard.to_dict(),
        loaded_sources=tuple(loaded),
    )
    write_json_artifact(
        feasibility.to_dict(), artifacts / "catalog-feasibility.v1.json"
    )
    write_json_artifact(
        decision_log_artifact(factory), artifacts / "hashes-only-decision-log.json"
    )
    write_json_artifact(guard.to_dict(), artifacts / "network-guard.json")
    write_json_artifact(report.to_dict(), artifacts / "report.json")
    index = build_artifact_index(artifacts, ENVIRONMENT_STUDY_ARTIFACT_ORDER)
    write_json_artifact(index, artifacts / ARTIFACT_INDEX_NAME)
    return PersistedRealCatalogMultiAgentV1(
        report=report,
        artifacts_dir=str(artifacts),
        artifact_index=index,
        network_guard=guard.to_dict(),
    )


__all__ = [
    "REAL_CATALOG_M2M_SCENARIO_ID",
    "REAL_CATALOG_M2M_STUDY_ID",
    "PersistedRealCatalogMultiAgentV1",
    "RealCatalogMultiAgentError",
    "build_real_catalog_multiagent_scenario",
    "make_real_catalog_multiagent_factory",
    "run_persisted_real_catalog_multiagent",
]
