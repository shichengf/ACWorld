"""A minimal 1x1 environment smoke: scripted buyer + merchant -> settled order.

This proves the :class:`~experiments.scripted_channel.ScriptedBusinessDecisionChannelV1`
drives a real ``Episode -> Runtime -> Agent -> Platform -> World`` path to a
committed settlement with zero provider calls, executed UNDER A CONTRACT FROZEN
BEFORE THE RUN, and assembles an
:class:`~experiments.environment_study.EnvironmentStudyReportV1` whose invariants
verify the exact causal settlement chain (not merely table counts) via the
repository's existing exact-evidence contracts.

The buyer/merchant policies are deliberately generic and read only the public
decision context: the buyer searches with empty filters, accepts the top ranked
offer, then finishes; the merchant takes no counterpart action.

Scope note: the 1x1 world fixture is currently derived from the benchmark T1
bundle (a known-good single-buyer / single-merchant / single-listing scenario).
A bespoke synthetic fixture and freezing the FULL execution-path closure are
Gate 1b-2 / Gate 3 follow-ups.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agents.business_decision import LLMBusinessDecisionV1
from episode.capability_materializer import scenario_content_hash_v2
from episode.scenario import population_for_scenario
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
from runtime.match_certificate_evidence import MATCH_CERTIFICATE_EVIDENCE_CONTRACT
from runtime.supply_fulfillment_evidence import (
    SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
    verified_settlement_order,
)
from runtime.tracker_evidence import (
    verified_model_business_choices,
    verify_all_active_actor_tracker_evidence,
)

# Repository root (``.../src/experiments/environment_smoke.py`` -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The invariant NAMES this study declares in its contract and reports on.
_SMOKE_INVARIANTS = (
    "scenario_content_frozen",
    "tracker_causal_closure",
    "episode_completed_without_termination",
    "match_certificate_verified",
    "settlement_verified",
    "certificate_settlement_exact_binding",
    "execution_sources_declared",
    "strict_replay",
    "zero_provider_calls",
)

# The sealed source each invariant is actually proven from (persisted artifact
# paths land with the Gate 1b-2 persistence step; these are the logical sources).
_INVARIANT_EVIDENCE_REFS: Mapping[str, tuple[str, ...]] = {
    "scenario_content_frozen": ("environment-study-contract", "source-manifest"),
    "tracker_causal_closure": ("audit.trace.jsonl",),
    "episode_completed_without_termination": ("termination.json-absent",),
    "match_certificate_verified": ("world.commits.jsonl", "audit.jsonl"),
    "settlement_verified": ("world.commits.jsonl", "audit.jsonl"),
    "certificate_settlement_exact_binding": ("world.commits.jsonl", "audit.jsonl"),
    "execution_sources_declared": ("source-manifest", "loaded-module-audit"),
    "strict_replay": ("replay.json",),
    "zero_provider_calls": ("scripted-decision-log",),
}


@dataclass(frozen=True)
class SmokeRunV1:
    """The immutable result of one 1x1 smoke episode.

    Bundles everything a report is derived from so callers thread a single
    frozen object instead of a positional tuple whose ``episode_dir``/replay
    could be spliced apart.  ``episode_dir`` is retained only for reference --
    the report derives its own from ``evidence.episode_dir`` -- and
    ``loaded_sources`` is the repo-relative ``src`` closure to audit against the
    frozen source manifest: this run's newly-imported delta by default, or the
    full loaded closure for the authoritative persisted study.
    """

    evidence: Any
    factory: ScriptedChannelFactoryV1
    episode_dir: str
    scenario: Any
    loaded_sources: frozenset[str]


# --- policies --------------------------------------------------------


def _first_offer_ref(observations: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the first ``offer_ref`` found anywhere in the observations."""

    def walk(value: Any) -> str | None:
        if isinstance(value, Mapping):
            ref = value.get("offer_ref")
            if isinstance(ref, str) and ref:
                return ref
            for item in value.values():
                found = walk(item)
                if found is not None:
                    return found
            return None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                found = walk(item)
                if found is not None:
                    return found
        return None

    for observation in observations:
        found = walk(observation)
        if found is not None:
            return found
    return None


def smoke_buyer_policy(context: ScriptedDecisionContextV1) -> LLMBusinessDecisionV1:
    """Search with empty filters, then accept the top ranked offer, then finish."""

    names = context.allowed_intent_names
    offer_ref = _first_offer_ref(context.observations)
    if "accept_ranked_offer" in names and offer_ref is not None:
        return LLMBusinessDecisionV1(
            intent="accept_ranked_offer",
            arguments={"offer_ref": offer_ref, "reason": "Accept the ranked offer."},
        )
    if "search" in names:
        return LLMBusinessDecisionV1(intent="search", arguments={"query": "", "filters": {}})
    if "observe_search_catalog" in names:
        return LLMBusinessDecisionV1(
            intent="observe_search_catalog",
            arguments={"query": "", "filters": {}, "limit": 64},
        )
    return LLMBusinessDecisionV1(
        intent="finish", arguments={"reason": "No further business action is needed."}
    )


def smoke_merchant_policy(context: ScriptedDecisionContextV1) -> LLMBusinessDecisionV1:
    """The merchant takes no counterpart action in the direct-purchase path."""

    del context
    return LLMBusinessDecisionV1(
        intent="finish", arguments={"reason": "No counterpart business action is required."}
    )


def make_smoke_factory() -> ScriptedChannelFactoryV1:
    return ScriptedChannelFactoryV1(
        policies={"buyer": smoke_buyer_policy, "merchant": smoke_merchant_policy}
    )


def smoke_scenario(task_id: str = "CWV2-T01-06") -> Any:
    """Return the known-good 1x1 world fixture (benchmark T1 bundle scenario)."""

    from episode.capability_runtime_t1 import runtime_bundle_t1

    return runtime_bundle_t1(task_id).scenario


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _actor_ids(scenario: Any) -> tuple[list[str], list[str]]:
    population = population_for_scenario(scenario)
    buyers = [buyer.buyer_id for buyer in population.buyers]
    merchants = [merchant.merchant_id for merchant in population.merchants]
    return buyers, merchants


# --- contract (frozen BEFORE execution) ------------------------------


def build_smoke_contract(
    *,
    scenario: Any,
    repo_root: str | Path,
    transport: str = "in_process",
) -> tuple[EnvironmentStudyContractV1, SourceManifestV1]:
    """Freeze a contract describing the ACTUAL run, before the Episode starts.

    ``scenario_digest`` is the repository's ``scenario_content_hash_v2`` (which
    covers the mandate, merchant policy, kickoff, allowed actions, population,
    and platform policy, omitting only the seed); ``seed`` is the scenario's own
    seed; and each actor policy digest is keyed by the REAL actor id.  Component
    digests are derived from a source manifest and the contract is verified
    against on-disk bytes before it is returned.
    """

    manifest = build_source_manifest(repo_root, DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES)
    buyers, merchants = _actor_ids(scenario)
    buyer_digest = _sha256_text(inspect.getsource(smoke_buyer_policy))
    merchant_digest = _sha256_text(inspect.getsource(smoke_merchant_policy))
    actor_policy_digests = [(buyer_id, buyer_digest) for buyer_id in buyers] + [
        (merchant_id, merchant_digest) for merchant_id in merchants
    ]
    contract = build_environment_study_contract(
        git=capture_git_metadata(repo_root),
        scenario_digest=scenario_content_hash_v2(scenario),
        actor_policy_digests=actor_policy_digests,
        component_digests=manifest.component_digests(),
        data_provenance_digest=_sha256_text("synthetic:benchmark-t1-1x1-smoke"),
        backend="in_process",
        transport=transport,
        seed=scenario.seed,
        invariants=_SMOKE_INVARIANTS,
    )
    verify_environment_study_contract(contract, manifest, repo_root)
    return contract, manifest


# --- causal closure (exact-evidence contracts) -----------------------


def verify_causal_settlement(evidence: Any, *, buyer_id: str) -> dict[str, Any] | None:
    """Verify the exact certificate->settlement->order->commit chain.

    Uses only the repository's registered exact-evidence contracts (no benchmark
    scorer, no oracle).  Returns the bound identifiers and World commit ids, or
    ``None`` if the exact join does not hold.
    """

    try:
        verified_match = evidence.verified_operation_evidence(
            MATCH_CERTIFICATE_EVIDENCE_CONTRACT, options={"allow_rejected": False}
        )
        preclaimed: list[str] = []
        for join in verified_match.certificates:
            preclaimed.append(str(join.search_commit["commit_id"]))
            preclaimed.append(str(join.commit["commit_id"]))
        verified_txn = evidence.verified_operation_evidence(
            SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
            options={
                "allow_rejected": True,
                "evaluated_actor_id": buyer_id,
                "preclaimed_commit_ids": tuple(preclaimed),
            },
        )
        settlements = verified_txn.operations_for(
            action_kind="platform.settle_payment", actor_id=buyer_id
        )
        if len(verified_match.certificates) != 1 or len(settlements) != 1:
            return None
        join = verified_match.certificates[0]
        cert = join.certificate
        settlement = settlements[0]
        order = verified_settlement_order(settlement)
        exact_binding = (
            order["order_id"] == cert.order_id
            and order["buyer_id"] == cert.buyer_id
            and order["merchant_id"] == cert.merchant_id
            and order["sku_id"] == cert.sku_id
            and order["qty"] == cert.qty
        )
        commit_ids = sorted(
            {
                str(join.search_commit["commit_id"]),
                str(join.commit["commit_id"]),
                str(settlement.commit["commit_id"]),
            }
        )
        return {
            "exact_binding": bool(exact_binding),
            "cert_id": cert.cert_id,
            "order_id": order["order_id"],
            "sku_id": order["sku_id"],
            "commit_ids": commit_ids,
        }
    except Exception:  # noqa: BLE001 - any join failure means the chain is unproven
        return None


def verify_tracker_closure(evidence: Any, factory: ScriptedChannelFactoryV1, *, buyer_id: str) -> Any:
    """Return the all-active Tracker verdict, or ``None`` if it does not verify.

    Uses ``verify_all_active_actor_tracker_evidence``, which derives active
    actors from BOTH the sidecar and audited inbounds, so a missing or tampered
    Tracker for any active actor is visible.  When this returns ``None`` the
    study must be invalid and no Tracker summary count may be trusted.
    """

    try:
        verdict = verify_all_active_actor_tracker_evidence(
            evidence,
            declared_actor_ids=factory.actor_ids(),
            evaluated_actor_id=buyer_id,
            evaluated_actor_strict=True,
        )
    except Exception:  # noqa: BLE001 - any verifier failure means the Tracker is untrusted
        return None
    return verdict if verdict.verified else None


def _actor_verdict(tracker_verdict: Any, actor_id: str) -> Any:
    if tracker_verdict is None:
        return None
    for row in tracker_verdict.actor_verdicts:
        if row.evaluated_actor_id == actor_id:
            return row
    return None


def _actor_coverage(
    evidence: Any,
    factory: ScriptedChannelFactoryV1,
    *,
    buyer_id: str,
    causal: dict[str, Any] | None,
    tracker_verdict: Any,
) -> tuple[ActorCoverageV1, ...]:
    rows: list[ActorCoverageV1] = []
    for actor_id in factory.actor_ids():
        channel = factory.channel_for(actor_id)
        causal_commits = (
            len(causal["commit_ids"]) if (causal is not None and actor_id == buyer_id) else 0
        )
        # Tracker-derived counts are only read for an actor the verified
        # all-active Tracker verdict actually covers (an inactive counterpart
        # with no Tracker record contributes zero, not an error).
        actor_verdict = _actor_verdict(tracker_verdict, actor_id)
        if actor_verdict is not None:
            verified_choices = len(
                verified_model_business_choices(
                    evidence, evaluated_actor_id=actor_id, strict_ideal=False
                )
            )
            complete_records = int(actor_verdict.complete_record_count)
        else:
            verified_choices = 0
            complete_records = 0
        rows.append(
            ActorCoverageV1(
                actor_id=actor_id,
                role=channel.role,
                scripted_business_decisions=len(channel.decision_log),
                verified_business_decisions=verified_choices,
                complete_tracker_records=complete_records,
                accepted_platform_operations=len(
                    evidence.accepted_platform_exchanges(actor_id=actor_id)
                ),
                causally_linked_world_commits=causal_commits,
            )
        )
    return tuple(rows)


# --- report ----------------------------------------------------------


def _invariant_failure_reason(name: str, unlisted: tuple[str, ...]) -> str:
    if name == "execution_sources_declared" and unlisted:
        return "undeclared execution modules were imported: " + ", ".join(unlisted)
    return f"{name} not proven by the sealed evidence"


def build_smoke_report(
    *,
    study_id: str,
    contract: EnvironmentStudyContractV1,
    manifest: SourceManifestV1,
    run: SmokeRunV1,
) -> EnvironmentStudyReportV1:
    """Assemble a capability-free report whose invariants prove causal closure.

    Every derived input comes from ``run``, and the run-vs-contract match is
    enforced here, so a caller cannot splice a mismatched episode directory,
    replay verdict, or scenario into a report.  ``episode_dir`` -- hence the
    termination state and the replay artifact -- is derived from
    ``evidence.episode_dir`` as a single source of truth, and the ``src`` modules
    the run imported are audited against the frozen source manifest.
    """

    from episode.replay import verify_episode_replay

    evidence = run.evidence
    scenario = run.scenario
    factory = run.factory
    _assert_evidence_matches_contract(scenario=scenario, contract=contract, evidence=evidence)

    episode_dir = Path(evidence.episode_dir)
    replay = verify_episode_replay(episode_dir, strict=True)
    buyers, _merchants = _actor_ids(scenario)
    buyer_id = buyers[0]
    causal = verify_causal_settlement(evidence, buyer_id=buyer_id)
    tracker_verdict = verify_tracker_closure(evidence, factory, buyer_id=buyer_id)
    unlisted = unlisted_loaded_sources(manifest, run.loaded_sources)

    checks = {
        # Proves the run happened under the pre-frozen contract's scenario.
        "scenario_content_frozen": scenario_content_hash_v2(scenario) == contract.scenario_digest,
        # A missing/tampered Agent Tracker for any active actor invalidates the study.
        "tracker_causal_closure": tracker_verdict is not None,
        "episode_completed_without_termination": not (episode_dir / "termination.json").exists(),
        "match_certificate_verified": causal is not None,
        "settlement_verified": causal is not None,
        "certificate_settlement_exact_binding": bool(causal and causal["exact_binding"]),
        # Every executed src module must already be frozen in the manifest.
        "execution_sources_declared": not unlisted,
        "strict_replay": bool(replay.replay_ok),
        "zero_provider_calls": factory.provider_calls == 0,
    }
    invariants = tuple(
        InvariantResultV1(
            name=name, passed=True, evidence_refs=_INVARIANT_EVIDENCE_REFS[name]
        )
        if checks[name]
        else InvariantResultV1(
            name=name, passed=False, failure_reason=_invariant_failure_reason(name, unlisted)
        )
        for name in _SMOKE_INVARIANTS
    )

    final_tables = evidence.final_world.get("tables", {})
    orders = final_tables.get("orders", [])
    ledger = final_tables.get("ledger", [])
    certs = final_tables.get("match_certificates", [])

    return EnvironmentStudyReportV1(
        study_id=study_id,
        contract_id=contract.contract_id,
        valid=all(checks.values()),
        invariants=invariants,
        actor_coverage=_actor_coverage(
            evidence, factory, buyer_id=buyer_id, causal=causal, tracker_verdict=tracker_verdict
        ),
        transaction_summary={
            "orders": len(orders),
            "settled_orders": sum(
                1 for order in orders if isinstance(order, Mapping) and order.get("state") == "settled"
            ),
            "verified_certificate": bool(causal),
            "bound_order_id": (causal or {}).get("order_id", ""),
        },
        inventory_summary={"listings": len(final_tables.get("inventory", {}))},
        ledger_summary={"entries": len(ledger)},
        fulfillment_summary={},
        after_sales_summary={},
        replay={
            "replay_ok": bool(replay.replay_ok),
            "transactions_replayed": int(replay.transactions_replayed),
            "commits_replayed": int(replay.commits_replayed),
        },
        diagnostics={
            "match_certificates": len(certs),
            "causal_commit_ids": len((causal or {}).get("commit_ids", [])),
            "declared_execution_sources": len(run.loaded_sources),
        },
        data_scope={"fixture": "benchmark_t1_1x1", "publication": "internal_smoke"},
    )


# --- run -------------------------------------------------------------


def run_smoke_episode(
    *,
    out_root: str | Path,
    scenario: Any = None,
    factory: ScriptedChannelFactoryV1 | None = None,
    task_id: str = "CWV2-T01-06",
    repo_root: str | Path | None = None,
) -> SmokeRunV1:
    """Run one 1x1 episode with the GIVEN scripted channels and scenario.

    Callers that already froze a contract must pass the same ``scenario`` and
    ``factory`` object they froze it from, so the Episode is not silently rebuilt
    from ``task_id``.  The factory is claimed for exactly this Episode and sealed
    afterward (so a second episode cannot accumulate onto the same actor ids),
    and the ``src`` modules the run newly imports are captured for the
    loaded-source audit.  In a clean process that delta is the full execution
    closure; in a shared test process it is this run's own contribution.
    """

    from episode.capability_runtime import RuntimeEvidenceBundleV2
    from episode.runner import EpisodeBatch

    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    if scenario is None:
        scenario = smoke_scenario(task_id)
    if factory is None:
        factory = make_smoke_factory()

    before = set(iter_loaded_module_files())
    with factory.claim_for_episode(scenario.scenario_id):
        EpisodeBatch(
            scenarios=[scenario],
            channels=factory,
            out_root=str(out_root),
            remote_world=True,
            strict_tracker_capture=True,
        ).run()
    loaded_sources = repo_source_paths(set(iter_loaded_module_files()) - before, root)

    episode_dir = Path(out_root) / scenario.scenario_id
    evidence = RuntimeEvidenceBundleV2.load(episode_dir)
    return SmokeRunV1(
        evidence=evidence,
        factory=factory,
        episode_dir=str(episode_dir),
        scenario=scenario,
        loaded_sources=loaded_sources,
    )


def _assert_evidence_matches_contract(
    *,
    scenario: Any,
    contract: EnvironmentStudyContractV1,
    evidence: Any,
) -> None:
    """Require the sealed run to match every frozen field of the contract."""

    manifest = getattr(evidence, "evidence_manifest", None) or {}
    mismatches: list[str] = []
    if scenario_content_hash_v2(scenario) != contract.scenario_digest:
        mismatches.append("scenario content hash")
    if scenario.seed != contract.seed:
        mismatches.append("scenario seed")
    if manifest.get("scenario_id") != scenario.scenario_id:
        mismatches.append("evidence scenario_id")
    if manifest.get("transport") != contract.transport:
        mismatches.append("evidence transport")
    if mismatches:
        raise RuntimeError(
            "executed run does not match the frozen contract: " + ", ".join(mismatches)
        )


def run_smoke_and_report(
    *,
    out_root: str | Path,
    repo_root: str | Path,
    task_id: str = "CWV2-T01-06",
    study_id: str = "CWENV-SMOKE-1X1",
) -> EnvironmentStudyReportV1:
    """Freeze a contract, run the 1x1 smoke under it, and return the report.

    Order of operations matters: the contract and source manifest are built and
    verified from the scenario BEFORE the Episode runs (with the SAME
    scenario/factory objects), so the report proves execution under a frozen
    contract rather than an after-the-fact description of it.
    """

    scenario = smoke_scenario(task_id)
    factory = make_smoke_factory()
    contract, manifest = build_smoke_contract(
        scenario=scenario, repo_root=repo_root, transport="in_process"
    )
    run = run_smoke_episode(
        out_root=out_root, scenario=scenario, factory=factory, repo_root=repo_root
    )
    return build_smoke_report(study_id=study_id, contract=contract, manifest=manifest, run=run)


# --- in-process <-> HTTP/VCP semantic parity -------------------------


def run_http_smoke_episode(
    *,
    out_root: str | Path,
    scenario: Any = None,
    factory: ScriptedChannelFactoryV1 | None = None,
    task_id: str = "CWV2-T01-06",
    repo_root: str | Path | None = None,
) -> SmokeRunV1:
    """Run the same 1x1 smoke over the HTTP/VCP transport (in-memory ASGI).

    Mirrors :func:`run_smoke_episode` but drives the episode through
    ``run_http_episode``, which routes the buyer/merchant<->platform hops across
    the FastAPI ``/vcp`` boundary and seals evidence with ``transport='http_vcp'``.
    The result is the same :class:`RuntimeEvidenceBundleV2` type, so a report or a
    parity comparison consumes it identically.  The factory is single-use here
    too, and the run's ``src`` closure is captured for the loaded-source audit.
    """

    from episode.capability_runtime import RuntimeEvidenceBundleV2
    from episode.http_launcher import run_http_episode

    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    if scenario is None:
        scenario = smoke_scenario(task_id)
    if factory is None:
        factory = make_smoke_factory()

    episode_dir = Path(out_root) / scenario.scenario_id
    before = set(iter_loaded_module_files())
    with factory.claim_for_episode(scenario.scenario_id):
        run_http_episode(scenario=scenario, channels=factory, out_dir=episode_dir)
    loaded_sources = repo_source_paths(set(iter_loaded_module_files()) - before, root)

    evidence = RuntimeEvidenceBundleV2.load(episode_dir)
    return SmokeRunV1(
        evidence=evidence,
        factory=factory,
        episode_dir=str(episode_dir),
        scenario=scenario,
        loaded_sources=loaded_sources,
    )


# The semantic (transport-invariant) result fields two transports must agree on.
_PARITY_FIELDS = (
    "scenario_content",
    "seed",
    "final_world_digest",
    "causal_settlement",
    "accepted_platform_operations",
    "tracker_coverage",
    "replay",
    "decision_logs",
)


@dataclass(frozen=True)
class TransportParityV1:
    """The outcome of comparing one scenario run over two transports.

    ``parity_ok`` requires that every semantic field matched AND the two runs
    used genuinely different transports (comparing a transport with itself is not
    parity).  Transport-specific fields (``transport``, the contract id) are
    deliberately excluded from the comparison -- they are EXPECTED to differ.
    """

    transports: tuple[str, str]
    matched_fields: tuple[str, ...]
    mismatched_fields: tuple[str, ...]
    parity_ok: bool


def _semantic_fingerprint(run: SmokeRunV1, *, buyer_id: str) -> dict[str, Any]:
    """Extract the transport-invariant semantic result of one run."""

    from episode.replay import verify_episode_replay

    evidence = run.evidence
    tracker = verify_tracker_closure(evidence, run.factory, buyer_id=buyer_id)
    replay = verify_episode_replay(Path(evidence.episode_dir), strict=True)
    return {
        "scenario_content": scenario_content_hash_v2(run.scenario),
        "seed": run.scenario.seed,
        "final_world_digest": evidence.final_digest,
        "causal_settlement": verify_causal_settlement(evidence, buyer_id=buyer_id),
        "accepted_platform_operations": len(
            evidence.accepted_platform_exchanges(actor_id=buyer_id)
        ),
        "tracker_coverage": (
            {v.evaluated_actor_id: int(v.complete_record_count) for v in tracker.actor_verdicts}
            if tracker is not None
            else None
        ),
        "replay": (
            bool(replay.replay_ok),
            int(replay.transactions_replayed),
            int(replay.commits_replayed),
        ),
        "decision_logs": {
            actor_id: sorted(
                (rec.request_sha256, rec.decision_sha256)
                for rec in run.factory.channel_for(actor_id).decision_log
            )
            for actor_id in run.factory.actor_ids()
        },
    }


def _run_transport(run: SmokeRunV1) -> str:
    return str((run.evidence.evidence_manifest or {}).get("transport"))


def verify_transport_parity(in_process: SmokeRunV1, http: SmokeRunV1) -> TransportParityV1:
    """Compare two runs of the same scenario over different transports.

    Compares SEMANTIC results only -- final world digest, the exact causal
    settlement (certificate/order/sku binding and World commit ids), accepted
    Platform operations, Tracker coverage, strict replay, and the scripted
    request/decision hashes -- and requires the two transports to differ.  It
    does NOT require equal transport fields or contract ids: those are expected
    to differ (the transport is part of the frozen contract identity).
    """

    buyer_ip = _actor_ids(in_process.scenario)[0][0]
    buyer_http = _actor_ids(http.scenario)[0][0]
    fp_ip = _semantic_fingerprint(in_process, buyer_id=buyer_ip)
    fp_http = _semantic_fingerprint(http, buyer_id=buyer_http)

    matched = tuple(field for field in _PARITY_FIELDS if fp_ip[field] == fp_http[field])
    mismatched = tuple(field for field in _PARITY_FIELDS if fp_ip[field] != fp_http[field])
    transports = (_run_transport(in_process), _run_transport(http))
    parity_ok = not mismatched and transports[0] != transports[1]
    return TransportParityV1(
        transports=transports,
        matched_fields=matched,
        mismatched_fields=mismatched,
        parity_ok=parity_ok,
    )


# --- persisted study (network-disabled, ordered artifacts) -----------

_DECISION_LOG_SCHEMA = "cwe.environment-study-decision-log.v1"


def decision_log_artifact(factory: ScriptedChannelFactoryV1) -> dict[str, Any]:
    """Return the hashes-only scripted decision log (no raw prompt/decision text).

    Each record carries only the sequence, actor id, role, decision id, and the
    request/decision SHA-256 -- the same hashes the channel already keeps, so the
    persisted log leaks no business content.
    """

    records: list[dict[str, Any]] = []
    for actor_id in factory.actor_ids():
        for rec in factory.channel_for(actor_id).decision_log:
            records.append(
                {
                    "sequence": rec.sequence,
                    "actor_id": rec.actor_id,
                    "role": rec.role,
                    "decision_id": rec.decision_id,
                    "request_sha256": rec.request_sha256,
                    "decision_sha256": rec.decision_sha256,
                }
            )
    records.sort(key=lambda row: (row["actor_id"], row["sequence"]))
    return {"schema_version": _DECISION_LOG_SCHEMA, "provider_calls": 0, "records": records}


@dataclass(frozen=True)
class PersistedSmokeStudyV1:
    """A persisted study: the report plus the sealed artifact directory + index."""

    report: EnvironmentStudyReportV1
    artifacts_dir: str
    artifact_index: Mapping[str, Any]
    network_guard: Mapping[str, Any]


def run_persisted_smoke_study(
    *,
    out_root: str | Path,
    repo_root: str | Path,
    artifacts_dir: str | Path,
    task_id: str = "CWV2-T01-06",
    study_id: str = "CWENV-SMOKE-1X1",
) -> PersistedSmokeStudyV1:
    """Freeze, run (network-disabled), report, and persist the ordered artifacts.

    Artifact order: the source manifest and contract are written BEFORE the run;
    then the hashes-only decision log, the network-guard record, and the report
    are written after; finally an artifact index records the byte length and
    SHA-256 of those five files.  The episode runs inside ``network_disabled`` so
    the study attests it made no outbound network connection.  Run in a fresh
    process, the report's ``execution_sources_declared`` audit sees the full
    loaded closure and is therefore strict.
    """

    scenario = smoke_scenario(task_id)
    factory = make_smoke_factory()
    contract, manifest = build_smoke_contract(
        scenario=scenario, repo_root=repo_root, transport="in_process"
    )

    adir = Path(artifacts_dir)
    # Pre-run artifacts: the frozen source manifest and contract.
    write_json_artifact(manifest.to_dict(), adir / "source-manifest.json")
    write_json_artifact(contract.to_dict(), adir / "contract.json")

    with network_disabled() as guard:
        run = run_smoke_episode(
            out_root=out_root, scenario=scenario, factory=factory, repo_root=repo_root
        )
    # Authoritative audit: in a dedicated study process the FULL loaded src
    # closure IS the execution closure (no cross-test import pollution), so audit
    # every loaded module -- not just this run's newly-imported delta -- against
    # the frozen manifest.  A module the harness imports at load time but the
    # manifest omits is then caught too.
    full_closure = repo_source_paths(iter_loaded_module_files(), repo_root)
    audited_run = replace(run, loaded_sources=run.loaded_sources | full_closure)
    report = build_smoke_report(
        study_id=study_id, contract=contract, manifest=manifest, run=audited_run
    )

    # Post-run artifacts, in the canonical order, then the index over the five.
    write_json_artifact(
        decision_log_artifact(run.factory), adir / "hashes-only-decision-log.json"
    )
    write_json_artifact(guard.to_dict(), adir / "network-guard.json")
    write_json_artifact(report.to_dict(), adir / "report.json")
    index = build_artifact_index(adir, ENVIRONMENT_STUDY_ARTIFACT_ORDER)
    write_json_artifact(index, adir / ARTIFACT_INDEX_NAME)

    return PersistedSmokeStudyV1(
        report=report,
        artifacts_dir=str(adir),
        artifact_index=index,
        network_guard=guard.to_dict(),
    )
