"""CWENV-DATA-01: real CSV -> Agent -> Platform -> World environment study.

The catalog enters CommerceWorld through the repository's sole production
adapter: ``BenchmarkMetadata(catalog_source=real_csv)`` causes
``episode.scenario.seed_world`` to invoke ``seed_world_catalog``, which itself
uses ``merchant_data_csv.iter_public_listings``.  The experiment never parses a
CSV and never settles an order directly.  A scripted buyer issues only public
business intents; Agent owns reference resolution, World reads, route binding,
authority, envelopes, idempotency, and continuation.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents.business_decision import LLMBusinessDecisionV1
from episode.benchmark import (
    BenchmarkMetadata,
    BenchmarkTrack,
    CatalogScale,
    CatalogSource,
    Difficulty,
    TaskFamily,
)
from episode.capability_materializer import scenario_content_hash_v2
from episode.capability_runtime import RuntimeEvidenceBundleV2, canonical_sha256
from episode.catalog_provenance import DEFAULT_MANIFEST
from episode.runner import EpisodeBatch
from episode.seed_catalog import seed_world_catalog
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec
from evals.serialize import to_canonical
from experiments.catalog_feasibility import (
    CatalogFeasibilityReportV1,
    build_catalog_feasibility_report,
    verify_catalog_feasibility_report,
)
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
from experiments.scripted_channel import (
    ScriptedChannelFactoryV1,
    ScriptedDecisionContextV1,
)
from protocol.evidence_records import (
    MandateRevisionAuthority,
    build_mandate_revision,
    mandate_revision_to_dict,
)
from runtime.cart_evidence import CART_EVIDENCE_CONTRACT, VerifiedCartEvidence
from runtime.tracker_evidence import (
    verified_model_business_choices,
    verified_model_world_reads,
    verify_all_active_actor_tracker_evidence,
)
from world.evidence_contracts import mandate_authority_to_wire
from world.state import World
from world.store import DatabaseWorld


DATA_STUDY_ID = "CWENV-DATA-01"
DATA_SCENARIO_ID = "cwenv_data_01__real_csv_runtime"
DATA_BUYER_ID = "buyer:cwenv-data-01"
DATA_PRINCIPAL_ID = "consumer:cwenv-data-01"
DATA_MANDATE_ID = "mandate:cwenv-data-01"
DATA_MARKET_ID = "market:cwenv-data-01"
DATA_SEED = 42
_REPO_ROOT = Path(__file__).resolve().parents[2]

_DATA_INVARIANTS = (
    "scenario_content_frozen",
    "catalog_provenance_verified",
    "world_database_seed_parity",
    "database_reopen_parity",
    "agent_business_intent_chain",
    "cart_authority_exact_join",
    "quote_bound_checkout",
    "inventory_and_ledger_closed",
    "owner_sku_namespace_closed",
    "tracker_causal_closure",
    "episode_completed_without_termination",
    "strict_replay",
    "zero_provider_calls",
    "execution_sources_declared",
    "raw_csv_rows_not_persisted",
    "no_direct_settlement",
)

_DATA_EVIDENCE_REFS: Mapping[str, tuple[str, ...]] = {
    "scenario_content_frozen": ("contract.json", "world.initial.json"),
    "catalog_provenance_verified": ("catalog-feasibility.v1.json",),
    "world_database_seed_parity": ("report.json",),
    "database_reopen_parity": ("report.json",),
    "agent_business_intent_chain": ("audit.trace.jsonl",),
    "cart_authority_exact_join": ("audit.jsonl", "world.commits.jsonl"),
    "quote_bound_checkout": ("audit.jsonl", "world.commits.jsonl"),
    "inventory_and_ledger_closed": ("world.initial.json", "world.final.json"),
    "owner_sku_namespace_closed": ("world.final.json",),
    "tracker_causal_closure": ("audit.trace.jsonl",),
    "episode_completed_without_termination": ("termination.json-absent",),
    "strict_replay": ("replay.json",),
    "zero_provider_calls": ("hashes-only-decision-log.json",),
    "execution_sources_declared": ("source-manifest.json", "loaded-module-audit"),
    "raw_csv_rows_not_persisted": ("artifact-index.json",),
    "no_direct_settlement": ("audit.trace.jsonl", "world.commits.jsonl"),
}


class DataEnvironmentStudyError(RuntimeError):
    """CWENV-DATA-01 could not prove its environment-only invariants."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _catalog_inventory_payload(snapshot: Any) -> dict[str, Any]:
    catalog = sorted(
        (
            {
                "sku_id": str(listing.sku_id),
                "category": str(listing.category),
                "name": str(listing.name),
                "attributes": to_canonical(listing.attributes),
                "list_price": {
                    "amount": format(
                        Decimal(listing.list_price.amount).quantize(Decimal("0.01")),
                        ".2f",
                    ),
                    "currency": str(listing.list_price.currency),
                },
                "merchant_id": str(listing.merchant_id),
                "product_id": str(listing.product_id),
            }
            for listing in snapshot.catalog
        ),
        key=lambda row: str(row["sku_id"]),
    )
    inventory = sorted(
        (
            {
                "sku_id": str(row.sku_id),
                "merchant_id": str(row.merchant_id),
                "qty_available": int(row.qty_available),
                "qty_reserved": int(row.qty_reserved),
                "eta_day": int(row.eta_day),
                "version": int(row.version),
            }
            for row in snapshot.inventory.values()
        ),
        key=lambda row: str(row["sku_id"]),
    )
    return {"catalog": catalog, "inventory": inventory}


def _minor_units(amount: Any) -> int:
    value = Decimal(str(amount)) * Decimal(100)
    if value != value.to_integral_value():
        raise DataEnvironmentStudyError("real listing price is not integral minor units")
    return int(value)


def _selected_merchant_slugs(report: CatalogFeasibilityReportV1) -> tuple[str, ...]:
    return tuple(
        merchant_id.removeprefix("merchant:")
        for merchant_id in report.selected_merchant_ids
    )


def _seed_reference_world(
    feasibility: CatalogFeasibilityReportV1,
) -> tuple[dict[str, Any], tuple[Any, Any]]:
    world = World()
    seed_world_catalog(
        world,
        merchants=_selected_merchant_slugs(feasibility),
        seed=DATA_SEED,
        catalog_scale="medium",
        in_stock_only=True,
    )
    snapshot = world.snapshot()
    candidates = sorted(
        snapshot.catalog,
        key=lambda listing: (
            _minor_units(listing.list_price.amount),
            str(listing.merchant_id),
            str(listing.sku_id),
        ),
    )
    if len(candidates) < 2:
        raise DataEnvironmentStudyError("medium real catalog has fewer than two candidates")
    first = candidates[0]
    second = next(
        (
            listing
            for listing in candidates[1:]
            if str(listing.merchant_id) != str(first.merchant_id)
            and _minor_units(listing.list_price.amount)
            != _minor_units(first.list_price.amount)
        ),
        None,
    )
    if second is None:
        raise DataEnvironmentStudyError(
            "real catalog has no distinct-price cross-merchant alternative"
        )
    return _catalog_inventory_payload(snapshot), (first, second)


def _pricing_fixture(listing: Any, ordinal: int) -> dict[str, Any]:
    sku_id = str(listing.sku_id)
    return {
        "merchant_id": str(listing.merchant_id),
        "idempotency_key": f"seed:cwenv-data-01:policy:{ordinal:02d}",
        "intent": {
            "market_id": DATA_MARKET_ID,
            "policy_id": f"policy:cwenv-data-01:{ordinal:02d}",
            "listing_ids": [sku_id],
            "product_ids": [],
            "quantity_tiers": [
                {
                    "minimum_quantity": 1,
                    "maximum_quantity": None,
                    "unit_price_minor": _minor_units(listing.list_price.amount),
                }
            ],
            "bundle_discounts": [],
            "bundle_stacking": "best_only",
            "components": [],
            "effective_after_ticks": 0,
            "expires_after_ticks": None,
        },
    }


def _planning_problem(candidates: Sequence[Any]) -> dict[str, Any]:
    offers = [
        {
            "sku_id": str(listing.sku_id),
            "merchant_id": str(listing.merchant_id),
            "product_family": "real-catalog-choice",
            "list_price_minor": _minor_units(listing.list_price.amount),
            "available_qty": 1,
            "delivery_days": 3,
        }
        for listing in candidates
    ]
    terms = [
        {
            "sku_ids": [str(listing.sku_id)],
            "quantity_tiers": [
                {
                    "minimum_quantity": 1,
                    "maximum_quantity": None,
                    "unit_price_minor": _minor_units(listing.list_price.amount),
                }
            ],
            "bundle_discounts": [],
            "bundle_stacking": "best_only",
            "charges": [],
        }
        for listing in candidates
    ]
    return {
        "schema_version": "cwe.public-cart-planning.v1",
        "rule_set": "finite-cart-planning-v1",
        "currency": "USD",
        "listing_offers": offers,
        "requirements": [
            {
                "requirement_key": "real-catalog-choice",
                "product_family": "real-catalog-choice",
                "required_qty": 1,
                "eligible_sku_ids": [str(listing.sku_id) for listing in candidates],
                "selection_rule": "choose_exactly_one_substitute",
            }
        ],
        "relations": [],
        "pricing_terms": terms,
        "hard_constraints": {
            "budget_minor": 100_000,
            "max_delivery_days": 14,
            "inventory_rule": "selected_qty_lte_available_qty",
            "requirement_rule": "exact_declared_demand",
            "relation_rule": "enforce_all_declared_relations",
        },
        "calculation_rules": {
            "tier_scope": "sum_selected_quantity_within_pricing_term",
            "bundle_discount_scope": "selected_cart",
            "charge_basis": "base_subtotal_before_bundle_discount",
            "bps_rounding": "floor_minor_units",
            "grand_total": "discounted_line_subtotal_plus_active_charges",
        },
        "objective": {
            "kind": "lexicographic_min",
            "criteria": ["grand_total_minor", "max_delivery_days", "merchant_count"],
            "tie_break": "canonical_selected_sku_refs_then_qty_ascending",
        },
    }


def _execution_contract() -> dict[str, Any]:
    return {
        "schema_version": "cwe.public-task-execution.v1",
        "phases": [
            {
                "phase_id": "buyer_real_catalog_search",
                "match": {
                    "actor_roles": ["buyer"],
                    "inbound_action_kinds": ["delegate.create_purchase_mandate"],
                    "inbound_sender_roles": ["consumer"],
                },
                "allowed_routes": [
                    {
                        "action_kind": "commerce.request_cart_quote",
                        "destination": "platform:checkout",
                    }
                ],
                "world_reads": "skill_scoped",
                "finish": "forbid",
            },
            {
                "phase_id": "buyer_real_catalog_quote",
                "match": {
                    "actor_roles": ["buyer"],
                    "inbound_action_kinds": ["platform.cart_quote"],
                    "inbound_sender_roles": ["platform"],
                },
                "allowed_routes": [
                    {
                        "action_kind": "platform.checkout_cart",
                        "destination": "platform:checkout",
                    }
                ],
                "world_reads": "skill_scoped",
                "finish": "forbid",
            },
            {
                "phase_id": "buyer_real_catalog_settlement",
                "match": {
                    "actor_roles": ["buyer"],
                    "inbound_action_kinds": ["platform.cart_settlement"],
                    "inbound_sender_roles": ["platform"],
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
        ],
    }


def build_data_scenario(
    feasibility: CatalogFeasibilityReportV1 | None = None,
) -> ScenarioSpec:
    report = feasibility or build_catalog_feasibility_report()
    verify_catalog_feasibility_report(report)
    _seed_payload, candidates = _seed_reference_world(report)
    task_context = {
        "schema_version": "cwe.environment-study-data-task.v1",
        "capability": "real_catalog_quote_bound_checkout",
        "evaluated_role": "buyer",
        "execution_contract": _execution_contract(),
        "cart_planning_problem": _planning_problem(candidates),
    }
    authority = MandateRevisionAuthority(
        principal_id=DATA_PRINCIPAL_ID,
        buyer_id=DATA_BUYER_ID,
        mandate_id=DATA_MANDATE_ID,
        allowed_fields=("budget_minor",),
    )
    revision = build_mandate_revision(
        principal_id=DATA_PRINCIPAL_ID,
        buyer_id=DATA_BUYER_ID,
        mandate_id=DATA_MANDATE_ID,
        revision=1,
        previous_digest=None,
        changes={"budget_minor": 100_000},
        authorized_fields=("budget_minor",),
        logical_tick=0,
    )
    merchants = tuple(
        MerchantSpec(
            merchant_id=merchant_id,
            persona={"name": f"Local catalog merchant {merchant_id.removeprefix('merchant:')}"},
            policy={
                "floor_price": 0,
                "margin_target_bps": 1_000,
                "max_negotiation_rounds": 1,
                "refund_policy": "30_day_return",
                "claim_aggressiveness": "neutral",
            },
        )
        for merchant_id in report.selected_merchant_ids
    )
    buyer = BuyerSpec(
        buyer_id=DATA_BUYER_ID,
        persona={"name": "Local real-catalog buyer", "role": "buyer"},
        mandate={
            "mandate_id": DATA_MANDATE_ID,
            "goal": (
                "Search the public catalog, inspect the best eligible listing, request a "
                "World-authoritative quote, and checkout only by quote reference."
            ),
            "quantity": 1,
            "return_after_purchase": False,
            "hard_constraints": {"budget": 100_000, "delivery_days": 14, "must_have": []},
            "soft_constraints": [],
            "soft_preferences": {"style": [], "avoid": []},
            "authority": {
                "can_buy_without_confirmation": True,
                "must_not_share_with_merchant": ["budget"],
            },
            "intent_expiry": "2099-12-31T00:00:00Z",
            "cart_quote_authority": {
                "market_id": DATA_MARKET_ID,
                "mandate_id": DATA_MANDATE_ID,
                "fill_policy": "all_or_none",
                "backorder_policy": "reject",
            },
            "task_context": task_context,
        },
    )
    benchmark = BenchmarkMetadata(
        variant_id=DATA_STUDY_ID,
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
        scenario_id=DATA_SCENARIO_ID,
        seed=DATA_SEED,
        initial_state={
            "catalog": [],
            "pricing_policy_fixtures": [
                _pricing_fixture(listing, ordinal)
                for ordinal, listing in enumerate(candidates, start=1)
            ],
            "mandate_authorities": [mandate_authority_to_wire(authority)],
            "mandate_revisions": [mandate_revision_to_dict(revision)],
        },
        buyer_goal={},
        merchant_policy={},
        allowed_actions=(
            "observe_search_catalog",
            "observe_listing",
            "request_cart_quote",
            "checkout_cart",
        ),
        success_oracle={"schema_version": "cwe.environment-study-no-capability-oracle.v1"},
        benchmark=benchmark,
        population=PopulationSpec(
            buyers=(buyer,),
            merchants=merchants,
            matching={"top_k": 20},
            execution={"max_transactions_per_buyer": 1},
        ),
    )


def _walk_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        rows.append(value)
        for item in value.values():
            rows.extend(_walk_mappings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            rows.extend(_walk_mappings(item))
    return tuple(rows)


def _public_cheapest_sku_ref(context: ScriptedDecisionContextV1) -> str:
    offers: list[tuple[int, str]] = []
    for row in _walk_mappings(context.observations):
        if row.get("rule_set") != "finite-cart-planning-v1":
            continue
        raw_offers = row.get("listing_offers")
        if not isinstance(raw_offers, (list, tuple)):
            continue
        for offer in raw_offers:
            if not isinstance(offer, Mapping):
                continue
            sku_ref = offer.get("sku_ref")
            price = offer.get("list_price_minor")
            if (
                isinstance(sku_ref, str)
                and isinstance(price, int)
                and not isinstance(price, bool)
            ):
                offers.append((price, sku_ref))
    if not offers:
        raise DataEnvironmentStudyError("provider-visible planning problem has no offers")
    return min(offers)[1]


class DataBuyerPolicyV1:
    """Stateful only for one public listing read; consumes no World/oracle handle."""

    def __init__(self) -> None:
        self._searched = False
        self._read_sku_refs: set[str] = set()

    def __call__(self, context: ScriptedDecisionContextV1) -> LLMBusinessDecisionV1:
        names = context.allowed_intent_names
        if "observe_search_catalog" in names and not self._searched:
            self._searched = True
            return LLMBusinessDecisionV1(
                intent="observe_search_catalog",
                arguments={"query": "", "filters": {}, "limit": 20},
            )
        if "checkout_cart" in names:
            return LLMBusinessDecisionV1(intent="checkout_cart", arguments={})
        if "request_cart_quote" in names:
            sku_ref = _public_cheapest_sku_ref(context)
            if "observe_listing" in names and sku_ref not in self._read_sku_refs:
                self._read_sku_refs.add(sku_ref)
                return LLMBusinessDecisionV1(
                    intent="observe_listing", arguments={"sku_ref": sku_ref}
                )
            return LLMBusinessDecisionV1(
                intent="request_cart_quote",
                arguments={"lines": [{"sku_ref": sku_ref, "qty": 1}]},
            )
        raise DataEnvironmentStudyError(
            f"data buyer has no expected business intent in phase {context.phase!r}"
        )


def data_merchant_policy(context: ScriptedDecisionContextV1) -> LLMBusinessDecisionV1:
    del context
    return LLMBusinessDecisionV1(
        intent="finish",
        arguments={"reason": "No merchant-side decision is required for direct cart quote."},
    )


def make_data_factory() -> ScriptedChannelFactoryV1:
    return ScriptedChannelFactoryV1(
        policies={"buyer": DataBuyerPolicyV1(), "merchant": data_merchant_policy}
    )


def build_data_contract(
    *,
    scenario: ScenarioSpec,
    feasibility: CatalogFeasibilityReportV1,
    repo_root: str | Path,
) -> tuple[EnvironmentStudyContractV1, SourceManifestV1]:
    manifest = build_source_manifest(
        repo_root, DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES
    )
    buyer_digest = _sha256_text(inspect.getsource(DataBuyerPolicyV1))
    merchant_digest = _sha256_text(inspect.getsource(data_merchant_policy))
    contract = build_environment_study_contract(
        git=capture_git_metadata(repo_root),
        scenario_digest=scenario_content_hash_v2(scenario),
        actor_policy_digests=(
            (DATA_BUYER_ID, buyer_digest),
            *((merchant_id, merchant_digest) for merchant_id in feasibility.selected_merchant_ids),
        ),
        component_digests=manifest.component_digests(),
        data_provenance_digest=feasibility.report_id,
        backend="world_episode_with_sqlite_seed_parity",
        transport="in_process",
        seed=scenario.seed,
        invariants=_DATA_INVARIANTS,
    )
    verify_environment_study_contract(contract, manifest, repo_root)
    return contract, manifest


@dataclass(frozen=True)
class CatalogBackendParityV1:
    world_sha256: str
    database_sha256: str
    reopened_database_sha256: str
    listing_count: int
    parity_ok: bool


def verify_catalog_backend_parity(
    *, feasibility: CatalogFeasibilityReportV1, database_path: str | Path
) -> CatalogBackendParityV1:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    world = World()
    seed_world_catalog(
        world,
        merchants=_selected_merchant_slugs(feasibility),
        seed=DATA_SEED,
        catalog_scale="medium",
        in_stock_only=True,
    )
    memory_payload = _catalog_inventory_payload(world.snapshot())
    database = DatabaseWorld(path)
    try:
        seed_world_catalog(
            database,
            merchants=_selected_merchant_slugs(feasibility),
            seed=DATA_SEED,
            catalog_scale="medium",
            in_stock_only=True,
        )
        database_payload = _catalog_inventory_payload(database.snapshot())
    finally:
        database.close()
    reopened = DatabaseWorld(path)
    try:
        reopened_payload = _catalog_inventory_payload(reopened.snapshot())
    finally:
        reopened.close()
    world_sha = canonical_sha256(memory_payload)
    database_sha = canonical_sha256(database_payload)
    reopened_sha = canonical_sha256(reopened_payload)
    listing_count = len(memory_payload["catalog"] or [])
    return CatalogBackendParityV1(
        world_sha256=world_sha,
        database_sha256=database_sha,
        reopened_database_sha256=reopened_sha,
        listing_count=listing_count,
        parity_ok=(world_sha == database_sha == reopened_sha and listing_count == 250),
    )


def _tracker_verdict(
    evidence: RuntimeEvidenceBundleV2, factory: ScriptedChannelFactoryV1
) -> Any:
    try:
        verdict = verify_all_active_actor_tracker_evidence(
            evidence,
            declared_actor_ids=factory.actor_ids(),
            evaluated_actor_id=DATA_BUYER_ID,
            evaluated_actor_strict=True,
        )
    except Exception:  # noqa: BLE001 - an evidence verifier failure invalidates the study
        return None
    return verdict if verdict.verified else None


def _actor_coverage(
    evidence: RuntimeEvidenceBundleV2,
    factory: ScriptedChannelFactoryV1,
    tracker: Any,
    cart: VerifiedCartEvidence | None,
) -> tuple[ActorCoverageV1, ...]:
    verdict_by_actor = (
        {row.evaluated_actor_id: row for row in tracker.actor_verdicts}
        if tracker is not None
        else {}
    )
    cart_commit_ids = (
        {
            str(cart.checkout_commit["commit_id"]),
            *(str(row["commit_id"]) for row in cart.quote_commits),
        }
        if cart is not None
        else set()
    )
    rows: list[ActorCoverageV1] = []
    for actor_id in factory.actor_ids():
        channel = factory.channel_for(actor_id)
        actor_verdict = verdict_by_actor.get(actor_id)
        choices = (
            verified_model_business_choices(
                evidence, evaluated_actor_id=actor_id, strict_ideal=False
            )
            if actor_verdict is not None
            else ()
        )
        rows.append(
            ActorCoverageV1(
                actor_id=actor_id,
                role=channel.role,
                scripted_business_decisions=len(channel.decision_log),
                verified_business_decisions=len(choices),
                complete_tracker_records=(
                    int(actor_verdict.complete_record_count)
                    if actor_verdict is not None
                    else 0
                ),
                accepted_platform_operations=len(
                    evidence.accepted_platform_exchanges(actor_id=actor_id)
                ),
                causally_linked_world_commits=(
                    len(cart_commit_ids) if actor_id == DATA_BUYER_ID else 0
                ),
            )
        )
    return tuple(rows)


def build_data_report(
    *,
    contract: EnvironmentStudyContractV1,
    manifest: SourceManifestV1,
    scenario: ScenarioSpec,
    feasibility: CatalogFeasibilityReportV1,
    factory: ScriptedChannelFactoryV1,
    evidence: RuntimeEvidenceBundleV2,
    parity: CatalogBackendParityV1,
    loaded_sources: Sequence[str],
) -> EnvironmentStudyReportV1:
    from episode.replay import verify_episode_replay

    episode_dir = Path(evidence.episode_dir)
    replay = verify_episode_replay(episode_dir, strict=True)
    try:
        raw_cart = evidence.verified_operation_evidence(
            CART_EVIDENCE_CONTRACT,
            options={
                "market_id": DATA_MARKET_ID,
                "buyer_id": DATA_BUYER_ID,
                "evaluated_actor_id": DATA_BUYER_ID,
            },
        )
        cart = raw_cart if isinstance(raw_cart, VerifiedCartEvidence) else None
    except Exception:  # noqa: BLE001 - exact evidence is a validity gate
        cart = None
    tracker = _tracker_verdict(evidence, factory)
    choices = (
        verified_model_business_choices(
            evidence, evaluated_actor_id=DATA_BUYER_ID, strict_ideal=False
        )
        if tracker is not None
        else ()
    )
    reads = (
        verified_model_world_reads(
            evidence, evaluated_actor_id=DATA_BUYER_ID, strict_ideal=False
        )
        if tracker is not None
        else ()
    )
    emitted_intents = tuple(choice.intent for choice in choices)
    read_intents = tuple(read.intent for read in reads)
    intents = read_intents + emitted_intents
    required_intents = (
        "observe_search_catalog",
        "observe_listing",
        "request_cart_quote",
        "checkout_cart",
    )
    final_tables = evidence.final_world.get("tables", {})
    initial_tables = evidence.initial_world.get("tables", {})
    orders = final_tables.get("orders", [])
    ledger = final_tables.get("ledger", [])
    groups = final_tables.get("order_groups", [])
    inventory_before = initial_tables.get("inventory", {})
    inventory_after = final_tables.get("inventory", {})
    order_rows = [row for row in orders if isinstance(row, Mapping)]
    namespace_ok = bool(order_rows) and all(
        str(row.get("sku_id", "")).startswith(f"{row.get('merchant_id')}:sku:")
        for row in order_rows
    )
    selected_skus = [str(row.get("sku_id")) for row in order_rows]
    inventory_changed = all(
        isinstance(inventory_before, Mapping)
        and isinstance(inventory_after, Mapping)
        and sku in inventory_before
        and sku in inventory_after
        and inventory_before[sku] != inventory_after[sku]
        for sku in selected_skus
    )
    cart_closed = bool(
        cart is not None
        and len(groups) == 1
        and order_rows
        and len(ledger) == len(order_rows)
        and inventory_changed
    )
    unlisted = unlisted_loaded_sources(manifest, loaded_sources)
    decision_artifact = decision_log_artifact(factory)
    expected_log_fields = {
        "sequence",
        "actor_id",
        "role",
        "decision_id",
        "request_sha256",
        "decision_sha256",
    }
    raw_rows_absent = (
        decision_artifact.get("provider_calls") == 0
        and all(
            set(row) == expected_log_fields
            for row in decision_artifact.get("records", [])
            if isinstance(row, Mapping)
        )
    )
    checks = {
        "scenario_content_frozen": (
            scenario_content_hash_v2(scenario) == contract.scenario_digest
            and scenario.seed == contract.seed
        ),
        "catalog_provenance_verified": feasibility.report_id
        == contract.data_provenance_digest,
        "world_database_seed_parity": parity.parity_ok
        and parity.world_sha256 == parity.database_sha256,
        "database_reopen_parity": parity.parity_ok
        and parity.database_sha256 == parity.reopened_database_sha256,
        "agent_business_intent_chain": intents == required_intents,
        "cart_authority_exact_join": cart is not None,
        "quote_bound_checkout": bool(
            cart is not None
            and cart.checkout_exchange.request.get("from") == DATA_BUYER_ID
            and cart.checkout_exchange.request.get("action", {}).get("payload")
            == {"quote_id": cart.quote["quote_id"]}
        ),
        "inventory_and_ledger_closed": cart_closed,
        "owner_sku_namespace_closed": namespace_ok,
        "tracker_causal_closure": tracker is not None,
        "episode_completed_without_termination": not (episode_dir / "termination.json").exists(),
        "strict_replay": bool(replay.replay_ok),
        "zero_provider_calls": factory.provider_calls == 0,
        "execution_sources_declared": not unlisted,
        "raw_csv_rows_not_persisted": raw_rows_absent,
        "no_direct_settlement": bool(
            cart is not None
            and any(choice.intent == "checkout_cart" for choice in choices)
            and all(
                row.get("actor_id") in {DATA_BUYER_ID, "platform:checkout"}
                for row in (*cart.quote_commits, cart.checkout_commit)
            )
        ),
    }
    invariants = tuple(
        InvariantResultV1(
            name=name, passed=True, evidence_refs=_DATA_EVIDENCE_REFS[name]
        )
        if checks[name]
        else InvariantResultV1(
            name=name,
            passed=False,
            evidence_refs=_DATA_EVIDENCE_REFS[name],
            failure_reason=(
                "undeclared loaded source: " + ", ".join(unlisted)
                if name == "execution_sources_declared" and unlisted
                else f"{name} not proven by CWENV-DATA-01 evidence"
            ),
        )
        for name in _DATA_INVARIANTS
    )
    return EnvironmentStudyReportV1(
        study_id=DATA_STUDY_ID,
        contract_id=contract.contract_id,
        valid=all(checks.values()),
        invariants=invariants,
        actor_coverage=_actor_coverage(evidence, factory, tracker, cart),
        transaction_summary={
            "order_groups": len(groups),
            "orders": len(order_rows),
            "settled_orders": sum(row.get("state") == "settled" for row in order_rows),
            "quote_bound": cart is not None,
        },
        inventory_summary={
            "initial_listings": parity.listing_count,
            "selected_listing_rows_changed": sum(
                1
                for sku in selected_skus
                if inventory_before.get(sku) != inventory_after.get(sku)
            ),
        },
        ledger_summary={"entries": len(ledger), "order_bound": len(ledger) == len(order_rows)},
        fulfillment_summary={},
        after_sales_summary={},
        replay={
            "replay_ok": bool(replay.replay_ok),
            "transactions_replayed": int(replay.transactions_replayed),
            "commits_replayed": int(replay.commits_replayed),
        },
        diagnostics={
            "business_intents": list(intents),
            "loaded_source_count": len(set(loaded_sources)),
            "unlisted_source_count": len(unlisted),
        },
        data_scope={
            "dataset_id": feasibility.dataset_id,
            "catalog_profile": "medium",
            "in_stock_only": True,
            "selected_merchant_ids": list(feasibility.selected_merchant_ids),
            "cross_merchant_product_linkage": False,
            "raw_rows_embedded": False,
            "paper_result": False,
        },
    )


@dataclass(frozen=True)
class PersistedDataStudyV1:
    report: EnvironmentStudyReportV1
    artifacts_dir: str
    artifact_index: Mapping[str, Any]
    network_guard: Mapping[str, Any]


def run_persisted_data_study(
    *,
    out_root: str | Path,
    artifacts_dir: str | Path,
    repo_root: str | Path = _REPO_ROOT,
) -> PersistedDataStudyV1:
    root = Path(repo_root)
    feasibility = build_catalog_feasibility_report(
        manifest_path=DEFAULT_MANIFEST, repository_root=root
    )
    verify_catalog_feasibility_report(feasibility, repository_root=root)
    scenario = build_data_scenario(feasibility)
    factory = make_data_factory()
    contract, manifest = build_data_contract(
        scenario=scenario, feasibility=feasibility, repo_root=root
    )
    adir = Path(artifacts_dir)
    write_json_artifact(manifest.to_dict(), adir / "source-manifest.json")
    write_json_artifact(contract.to_dict(), adir / "contract.json")

    parity = verify_catalog_backend_parity(
        feasibility=feasibility,
        database_path=Path(out_root) / "catalog-parity.sqlite3",
    )
    with network_disabled() as guard:
        with factory.claim_for_episode(scenario.scenario_id):
            EpisodeBatch(
                scenarios=[scenario],
                channels=factory,
                out_root=out_root,
                remote_world=True,
                strict_skill_selection=True,
                strict_tracker_capture=True,
            ).run()
    evidence = RuntimeEvidenceBundleV2.load(Path(out_root) / scenario.scenario_id)
    loaded = repo_source_paths(iter_loaded_module_files(), root)
    report = build_data_report(
        contract=contract,
        manifest=manifest,
        scenario=scenario,
        feasibility=feasibility,
        factory=factory,
        evidence=evidence,
        parity=parity,
        loaded_sources=tuple(loaded),
    )
    write_json_artifact(
        decision_log_artifact(factory), adir / "hashes-only-decision-log.json"
    )
    write_json_artifact(guard.to_dict(), adir / "network-guard.json")
    write_json_artifact(report.to_dict(), adir / "report.json")
    index = build_artifact_index(adir, ENVIRONMENT_STUDY_ARTIFACT_ORDER)
    write_json_artifact(index, adir / ARTIFACT_INDEX_NAME)
    return PersistedDataStudyV1(
        report=report,
        artifacts_dir=str(adir),
        artifact_index=index,
        network_guard=guard.to_dict(),
    )


__all__ = [
    "DATA_STUDY_ID",
    "CatalogBackendParityV1",
    "DataBuyerPolicyV1",
    "DataEnvironmentStudyError",
    "PersistedDataStudyV1",
    "build_data_contract",
    "build_data_report",
    "build_data_scenario",
    "make_data_factory",
    "run_persisted_data_study",
    "verify_catalog_backend_parity",
]
