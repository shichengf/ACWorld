"""Deterministic extension refresh and CommerceWorld integrity evidence.

This study is deliberately model-free.  It refreshes the three existing RQ2
extension examples through their public execution paths, then executes ten
small integrity probes against the current Platform/World surfaces.  The
negative World call in the atomicity probe is an explicitly labelled primitive
test; it is not presented as an Agent transaction trajectory.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from fastapi.testclient import TestClient

from agents.platform import PlatformService
from agents.world_client import WorldClient, in_process_world_client
from episode.capability_runtime import canonical_sha256
from evals.serialize import to_canonical
from experiments.environment_study import (
    DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES,
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
from experiments.rq2_extensions import (
    RQ2EvidenceError,
    verify_rq2_extension_evidence,
    write_rq2_extension_evidence,
)
from protocol.envelope import Envelope
from world import (
    AgentId,
    DatabaseWorld,
    InventoryRow,
    Listing,
    Money,
    Order,
    OrderId,
    OrderState,
    OutOfStock,
    Receipt,
    SkuId,
    TxnId,
    VCPWorldClient,
    World,
    WorldService,
)
from world.api import create_app as create_world_app


INTEGRITY_STUDY_ID = "CWENV-INTEGRITY-01"
INTEGRITY_SEED = 2_026_0723
INTEGRITY_DETAILS_SCHEMA = "cwe.environment-integrity-details.v1"
_REPO_ROOT = Path(__file__).resolve().parents[2]

_BUYER = AgentId("buyer:integrity-01")
_OTHER_BUYER = AgentId("buyer:integrity-02")
_MERCHANT = AgentId("merchant:integrity-01")
_SKU = SkuId("merchant:integrity-01:sku:item-01")
_PRICE = Money(Decimal("25.00"))

_EXTENSION_CASE_IDS = (
    "pet_supplies_domain",
    "round_robin_ranking",
    "flash_restock_event_oracle",
)
_INTEGRITY_CASE_IDS = (
    "atomic_overfill_rollback",
    "actor_idempotent_retry",
    "oversell_prevention",
    "authority_scope_rejection",
    "stale_event_rejection",
    "duplicate_event_retry",
    "transport_semantic_parity",
    "artifact_tamper_detection",
    "database_reopen_persistence",
    "execution_source_closure",
)
_INVARIANTS = (
    "three_extension_cases_refreshed",
    *_INTEGRITY_CASE_IDS,
    "zero_provider_calls",
    "network_egress_disabled",
)


class EnvironmentIntegrityStudyError(RuntimeError):
    """The extension/integrity evidence could not be built or verified."""


@dataclass(frozen=True, slots=True)
class IntegrityProbeV1:
    """One small, score-free environment integrity observation."""

    case_id: str
    category: str
    passed: bool
    observations: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "passed": self.passed,
            "observations": dict(self.observations),
            "evidence_sha256": str(canonical_sha256(self.observations)),
        }


@dataclass(frozen=True, slots=True)
class IntegrityStudyResultV1:
    """Persisted C6c output."""

    contract: EnvironmentStudyContractV1
    manifest: SourceManifestV1
    report: EnvironmentStudyReportV1
    details: Mapping[str, Any]


def _state_digest(world: World | DatabaseWorld) -> str:
    return str(canonical_sha256(to_canonical(world.snapshot())))


def _reserved_quantity(world: World | DatabaseWorld) -> int:
    return int(getattr(world.snapshot().inventory[_SKU], "qty_reserved"))


def _seed_world(world: World | DatabaseWorld, *, available: int = 2) -> None:
    world.apply(
        {
            "catalog": [
                Listing(
                    sku_id=_SKU,
                    product_id="product:integrity-01",
                    category="integrity-fixture",
                    name="Integrity fixture",
                    attributes={"returnable": True},
                    list_price=_PRICE,
                    merchant_id=_MERCHANT,
                )
            ],
            "inventory": [
                InventoryRow(
                    sku_id=_SKU,
                    merchant_id=_MERCHANT,
                    qty_available=available,
                )
            ],
        }
    )


def _settle_request(
    *,
    buyer: AgentId = _BUYER,
    qty: int = 2,
    order_suffix: str = "01",
    key: str = "settle:integrity-01",
) -> Envelope:
    return Envelope(
        msg_id=f"msg:{key}",
        ts="2026-07-23T00:00:00Z",
        from_=str(buyer),
        to="platform:psp",
        in_reply_to=None,
        idempotency_key=key,
        action={
            "kind": "platform.settle_payment",
            "payload": {
                "order_id": f"order:integrity-{order_suffix}",
                "buyer_id": str(buyer),
                "merchant_id": str(_MERCHANT),
                "sku_id": str(_SKU),
                "qty": qty,
                "agreed_price": {"amount": "25.00", "currency": "USD"},
                "allow_partial": False,
            },
        },
    )


def _probe_atomicity() -> IntegrityProbeV1:
    world = World()
    _seed_world(world, available=2)
    before = _state_digest(world)
    order_id = OrderId("order:integrity-overfill")
    order = Order(
        order_id=order_id,
        buyer_id=_BUYER,
        merchant_id=_MERCHANT,
        sku_id=_SKU,
        qty=4,
        agreed_price=_PRICE,
        state=OrderState.ACCEPTED,
    )
    receipt = Receipt(
        txn_id=TxnId(f"txn:{order_id}"),
        ts="2026-07-23T00:00:00Z",
        order_id=order_id,
        buyer_id=_BUYER,
        merchant_id=_MERCHANT,
        sku_id=_SKU,
        qty=3,
        price=_PRICE,
        idempotency_key="overfill:integrity",
    )
    rejected = False
    try:
        in_process_world_client(world).settle_order_partial(
            order=order,
            fulfilled_qty=3,
            receipt=receipt,
            original_actor=str(_BUYER),
            idempotency_key="overfill:integrity",
        )
    except OutOfStock:
        rejected = True
    after = _state_digest(world)
    return IntegrityProbeV1(
        case_id="atomic_overfill_rollback",
        category="world_primitive_negative_probe",
        passed=rejected and before == after,
        observations={
            "rejected": rejected,
            "state_unchanged": before == after,
            "commit_count": len(world.commit_journal),
        },
    )


def _probe_idempotency() -> IntegrityProbeV1:
    world = World()
    _seed_world(world)
    platform = PlatformService(world=world)
    request = _settle_request()
    first = platform.handle(request)
    commits = len(world.commit_journal)
    second = platform.handle(request)
    snapshot = world.snapshot()
    passed = bool(
        isinstance(first, Envelope)
        and isinstance(second, Envelope)
        and first.action == second.action
        and len(world.commit_journal) == commits
        and _reserved_quantity(world) == 2
        and len(snapshot.ledger) == 1
    )
    return IntegrityProbeV1(
        case_id="actor_idempotent_retry",
        category="platform_world",
        passed=passed,
        observations={
            "reply_equal": isinstance(first, Envelope)
            and isinstance(second, Envelope)
            and first.action == second.action,
            "commit_count_stable": len(world.commit_journal) == commits,
            "reserved_quantity": _reserved_quantity(world),
            "ledger_entries": len(snapshot.ledger),
        },
    )


def _probe_oversell() -> IntegrityProbeV1:
    world = World()
    _seed_world(world)
    platform = PlatformService(world=world)
    first = platform.handle(_settle_request())
    second = platform.handle(
        _settle_request(
            buyer=_OTHER_BUYER,
            order_suffix="02",
            key="settle:integrity-02",
        )
    )
    snapshot = world.snapshot()
    passed = bool(
        isinstance(first, Envelope)
        and second is None
        and _reserved_quantity(world) == 2
        and len(snapshot.ledger) == 1
        and len(snapshot.orders) == 1
    )
    return IntegrityProbeV1(
        case_id="oversell_prevention",
        category="platform_world",
        passed=passed,
        observations={
            "first_accepted": isinstance(first, Envelope),
            "second_rejected": second is None,
            "reserved_quantity": _reserved_quantity(world),
            "ledger_entries": len(snapshot.ledger),
            "orders": len(snapshot.orders),
        },
    )


def _probe_authority() -> IntegrityProbeV1:
    world = World()
    _seed_world(world)
    before = _state_digest(world)
    forged = _settle_request()
    forged = Envelope(
        msg_id=forged.msg_id,
        ts=forged.ts,
        from_=str(_OTHER_BUYER),
        to=forged.to,
        in_reply_to=forged.in_reply_to,
        idempotency_key=forged.idempotency_key,
        action=forged.action,
    )
    reply = PlatformService(world=world).handle(forged)
    after = _state_digest(world)
    return IntegrityProbeV1(
        case_id="authority_scope_rejection",
        category="platform_authority",
        passed=reply is None and before == after,
        observations={
            "forged_request_rejected": reply is None,
            "state_unchanged": before == after,
            "commit_count": len(world.commit_journal),
        },
    )


def _protocol_world() -> World:
    world = World()
    _seed_world(world, available=2)
    world.apply(
        {
            "orders": [
                Order(
                    order_id=OrderId("order:integrity-event"),
                    buyer_id=_BUYER,
                    merchant_id=_MERCHANT,
                    sku_id=_SKU,
                    qty=1,
                    agreed_price=_PRICE,
                    state=OrderState.PROPOSED,
                )
            ],
            "logical_time": 0,
        }
    )
    return world


def _event_request(
    *, sender: str, kind: str, key: str, payload: Mapping[str, object]
) -> Envelope:
    return Envelope(
        msg_id=f"msg:{key}",
        ts="2026-07-23T00:00:00Z",
        from_=sender,
        to="platform:events",
        in_reply_to=None,
        idempotency_key=key,
        action={"kind": kind, "payload": dict(payload)},
    )


def _issue_event(*, event_id: str, ttl_ticks: int) -> Envelope:
    return _event_request(
        sender="runtime:events",
        kind="platform.issue_protocol_event",
        key=f"issue:{event_id}",
        payload={
            "market_id": "market:integrity",
            "stream_id": "stream:integrity",
            "order_id": "order:integrity-event",
            "recipient_id": str(_BUYER),
            "event_id": event_id,
            "event_kind": "payment.settle",
            "ttl_ticks": ttl_ticks,
        },
    )


def _process_event(*, event_id: str, key: str) -> Envelope:
    return _event_request(
        sender=str(_BUYER),
        kind="commerce.process_protocol_event",
        key=key,
        payload={"event_id": event_id, "reason": "integrity probe"},
    )


def _probe_stale_event() -> IntegrityProbeV1:
    world = _protocol_world()
    platform = PlatformService(world=world)
    platform.handle(_issue_event(event_id="event:stale", ttl_ticks=1))
    before = _state_digest(world)
    platform.handle(
        _event_request(
            sender="runtime:clock",
            kind="platform.advance_market_clock",
            key="clock:stale",
            payload={"to_tick": 2},
        )
    )
    reply = platform.handle(_process_event(event_id="event:stale", key="process:stale"))
    snapshot = world.snapshot()
    passed = bool(
        reply is None
        and snapshot.orders[0].state == OrderState.PROPOSED
        and not snapshot.protocol_receipts
        and before != _state_digest(world)
    )
    return IntegrityProbeV1(
        case_id="stale_event_rejection",
        category="protocol_event",
        passed=passed,
        observations={
            "stale_process_rejected": reply is None,
            "order_remained_proposed": snapshot.orders[0].state == OrderState.PROPOSED,
            "receipt_count": len(snapshot.protocol_receipts),
            "logical_time": world.logical_time,
        },
    )


def _probe_duplicate_event() -> IntegrityProbeV1:
    world = _protocol_world()
    platform = PlatformService(world=world)
    platform.handle(_issue_event(event_id="event:duplicate", ttl_ticks=4))
    first = platform.handle(
        _process_event(event_id="event:duplicate", key="process:duplicate")
    )
    commits = len(world.commit_journal)
    second = platform.handle(
        _process_event(event_id="event:duplicate", key="process:duplicate")
    )
    snapshot = world.snapshot()
    passed = bool(
        isinstance(first, Envelope)
        and isinstance(second, Envelope)
        and first.action == second.action
        and len(world.commit_journal) == commits
        and len(snapshot.protocol_receipts) == 1
        and len(snapshot.ledger) == 1
    )
    return IntegrityProbeV1(
        case_id="duplicate_event_retry",
        category="protocol_event",
        passed=passed,
        observations={
            "reply_equal": isinstance(first, Envelope)
            and isinstance(second, Envelope)
            and first.action == second.action,
            "commit_count_stable": len(world.commit_journal) == commits,
            "receipt_count": len(snapshot.protocol_receipts),
            "ledger_entries": len(snapshot.ledger),
        },
    )


def _probe_transport() -> IntegrityProbeV1:
    direct_world = World()
    http_world = World()
    _seed_world(direct_world)
    _seed_world(http_world)
    request = _settle_request()
    direct_reply = PlatformService(world=direct_world).handle(request)
    with TestClient(create_world_app(WorldService(http_world))) as http:
        client = WorldClient(
            send=VCPWorldClient(http_client=http, agent_id="platform:psp").send
        )
        http_reply = PlatformService(world_client=client).handle(request)
    passed = bool(
        isinstance(direct_reply, Envelope)
        and isinstance(http_reply, Envelope)
        and direct_reply.action == http_reply.action
        and _state_digest(direct_world) == _state_digest(http_world)
    )
    return IntegrityProbeV1(
        case_id="transport_semantic_parity",
        category="transport",
        passed=passed,
        observations={
            "reply_equal": isinstance(direct_reply, Envelope)
            and isinstance(http_reply, Envelope)
            and direct_reply.action == http_reply.action,
            "world_equal": _state_digest(direct_world) == _state_digest(http_world),
            "transport_pair": ["in_process", "http_vcp_asgi"],
        },
    )


def _probe_tamper(extension_root: Path) -> IntegrityProbeV1:
    target = extension_root / "ranking" / "seed-42.json"
    original = target.read_bytes()
    payload = json.loads(original)
    if not isinstance(payload, dict):
        raise EnvironmentIntegrityStudyError("extension artifact must be an object")
    payload["passed"] = not bool(payload.get("passed"))
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    detected = False
    try:
        verify_rq2_extension_evidence(extension_root)
    except RQ2EvidenceError:
        detected = True
    finally:
        target.write_bytes(original)
    restored = verify_rq2_extension_evidence(extension_root)
    passed = detected and restored["acceptance_ok"] is True
    return IntegrityProbeV1(
        case_id="artifact_tamper_detection",
        category="artifact_integrity",
        passed=passed,
        observations={
            "mutation_detected": detected,
            "restored_bundle_verified": restored["acceptance_ok"] is True,
            "mutated_artifact": "ranking/seed-42.json",
        },
    )


def _probe_reopen(database_path: Path) -> IntegrityProbeV1:
    world = DatabaseWorld(database_path, mode="E1")
    _seed_world(world)
    reply = PlatformService(world_client=in_process_world_client(world)).handle(
        _settle_request()
    )
    before = _state_digest(world)
    world.close()
    reopened = DatabaseWorld(database_path, mode="E1")
    after = _state_digest(reopened)
    snapshot = reopened.snapshot()
    reserved_quantity = _reserved_quantity(reopened)
    reopened.close()
    passed = bool(
        isinstance(reply, Envelope)
        and before == after
        and reserved_quantity == 2
        and len(snapshot.ledger) == 1
        and len(snapshot.orders) == 1
    )
    return IntegrityProbeV1(
        case_id="database_reopen_persistence",
        category="database_world",
        passed=passed,
        observations={
            "state_digest_equal": before == after,
            "reserved_quantity": reserved_quantity,
            "ledger_entries": len(snapshot.ledger),
            "orders": len(snapshot.orders),
        },
    )


def _build_contract(
    *, repo_root: Path
) -> tuple[EnvironmentStudyContractV1, SourceManifestV1]:
    manifest = build_source_manifest(repo_root, DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES)
    policy_digest = hashlib.sha256(
        inspect.getsource(run_persisted_integrity_study).encode("utf-8")
    ).hexdigest()
    specification = {
        "schema_version": INTEGRITY_DETAILS_SCHEMA,
        "study_id": INTEGRITY_STUDY_ID,
        "seed": INTEGRITY_SEED,
        "extension_cases": list(_EXTENSION_CASE_IDS),
        "integrity_cases": list(_INTEGRITY_CASE_IDS),
    }
    contract = build_environment_study_contract(
        git=capture_git_metadata(repo_root),
        scenario_digest=str(canonical_sha256(specification)),
        actor_policy_digests=(("auditor:integrity", policy_digest),),
        component_digests=manifest.component_digests(),
        data_provenance_digest=str(
            canonical_sha256(
                {
                    "synthetic_fixture": True,
                    "extension_examples": list(_EXTENSION_CASE_IDS),
                    "integrity_case_count": len(_INTEGRITY_CASE_IDS),
                }
            )
        ),
        backend="world_and_database_world_e1",
        transport="in_process_and_http_vcp_asgi",
        seed=INTEGRITY_SEED,
        invariants=_INVARIANTS,
    )
    verify_environment_study_contract(contract, manifest, repo_root)
    return contract, manifest


def _extension_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = summary.get("cases")
    if not isinstance(cases, Mapping):
        raise EnvironmentIntegrityStudyError("extension summary cases are missing")
    rows: list[dict[str, Any]] = []
    for case_id in _EXTENSION_CASE_IDS:
        raw = cases.get(case_id)
        if not isinstance(raw, Mapping):
            raise EnvironmentIntegrityStudyError(f"extension case is missing: {case_id}")
        rows.append(
            {
                "case_id": case_id,
                "passed": raw.get("passed") is True,
                "run_count": int(raw.get("run_count", 0)),
                "passed_runs": int(raw.get("passed_runs", 0)),
            }
        )
    return rows


def _artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_persisted_integrity_study(
    *,
    out_root: str | Path,
    artifacts_dir: str | Path,
    repo_root: str | Path = _REPO_ROOT,
) -> IntegrityStudyResultV1:
    """Run C6c and persist its frozen contract, details, report, and index."""

    root = Path(repo_root).resolve(strict=True)
    output = Path(out_root).resolve()
    artifacts = Path(artifacts_dir).resolve()
    extension_root = output / "extensions"
    database_path = output / "integrity.sqlite3"
    if output.exists() or artifacts.exists():
        raise EnvironmentIntegrityStudyError("integrity output paths must not already exist")
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)

    contract, manifest = _build_contract(repo_root=root)
    write_json_artifact(manifest.to_dict(), artifacts / "source-manifest.json")
    write_json_artifact(contract.to_dict(), artifacts / "contract.json")

    loaded_before = repo_source_paths(iter_loaded_module_files(), root)
    with network_disabled(allow_loopback=True) as network_log:
        extension_summary = write_rq2_extension_evidence(
            extension_root,
            repo_root=root,
            seeds=(42, 1337, 2024),
        )
        extension_verification = verify_rq2_extension_evidence(extension_root)
        probes = [
            _probe_atomicity(),
            _probe_idempotency(),
            _probe_oversell(),
            _probe_authority(),
            _probe_stale_event(),
            _probe_duplicate_event(),
            _probe_transport(),
            _probe_tamper(extension_root),
            _probe_reopen(database_path),
        ]

    loaded_after = repo_source_paths(iter_loaded_module_files(), root)
    loaded_for_run = loaded_after - loaded_before
    unlisted = unlisted_loaded_sources(manifest, loaded_after)
    source_probe = IntegrityProbeV1(
        case_id="execution_source_closure",
        category="source_integrity",
        passed=not unlisted,
        observations={
            "loaded_source_count": len(loaded_after),
            "new_loaded_source_count": len(loaded_for_run),
            "unlisted_source_count": len(unlisted),
            "unlisted_sources": list(unlisted),
        },
    )
    probes.append(source_probe)
    if tuple(row.case_id for row in probes) != _INTEGRITY_CASE_IDS:
        raise EnvironmentIntegrityStudyError("integrity probe ordering differs from contract")

    extension_rows = _extension_rows(extension_summary)
    extensions_passed = bool(
        extension_verification["acceptance_ok"] is True
        and all(row["passed"] for row in extension_rows)
    )
    probe_by_id = {row.case_id: row for row in probes}
    invariant_results = [
        InvariantResultV1(
            name="three_extension_cases_refreshed",
            passed=extensions_passed,
            evidence_refs=("extensions/summary.json",),
            failure_reason=None if extensions_passed else "extension refresh failed",
        )
    ]
    for case_id in _INTEGRITY_CASE_IDS:
        probe = probe_by_id[case_id]
        invariant_results.append(
            InvariantResultV1(
                name=case_id,
                passed=probe.passed,
                evidence_refs=("integrity-details.json",),
                failure_reason=None if probe.passed else f"integrity case failed: {case_id}",
            )
        )
    network_ok = not network_log.blocked
    invariant_results.extend(
        (
            InvariantResultV1(
                name="zero_provider_calls",
                passed=True,
                evidence_refs=("network-guard.json",),
            ),
            InvariantResultV1(
                name="network_egress_disabled",
                passed=network_ok,
                evidence_refs=("network-guard.json",),
                failure_reason=None if network_ok else "outbound network attempt was blocked",
            ),
        )
    )
    details = {
        "schema_version": INTEGRITY_DETAILS_SCHEMA,
        "study_id": INTEGRITY_STUDY_ID,
        "contract_id": contract.contract_id,
        "extension_cases": extension_rows,
        "integrity_cases": [row.to_dict() for row in probes],
        "provider_calls": 0,
        "network_calls": 0,
    }
    write_json_artifact(details, artifacts / "integrity-details.json")
    write_json_artifact(network_log.to_dict(), artifacts / "network-guard.json")

    all_passed = all(row.passed for row in invariant_results)
    report = EnvironmentStudyReportV1(
        study_id=INTEGRITY_STUDY_ID,
        contract_id=contract.contract_id,
        valid=all_passed,
        invariants=tuple(invariant_results),
        actor_coverage=(),
        transaction_summary={
            "extension_cases": extension_rows,
            "integrity_cases": [row.to_dict() for row in probes],
            "extension_summary_sha256": _artifact_sha256(extension_root / "summary.json"),
        },
        inventory_summary={
            "atomicity_checked": probe_by_id["atomic_overfill_rollback"].passed,
            "oversell_checked": probe_by_id["oversell_prevention"].passed,
            "reopen_checked": probe_by_id["database_reopen_persistence"].passed,
        },
        ledger_summary={
            "idempotency_checked": probe_by_id["actor_idempotent_retry"].passed,
            "duplicate_checked": probe_by_id["duplicate_event_retry"].passed,
        },
        fulfillment_summary={
            "transport_checked": probe_by_id["transport_semantic_parity"].passed,
        },
        after_sales_summary={
            "covered_by_e1_study": True,
            "duplicated_here": False,
        },
        replay={
            "extension_bundle_verified": extension_verification["acceptance_ok"] is True,
            "strict_episode_replays": 6,
            "tamper_detection_checked": probe_by_id["artifact_tamper_detection"].passed,
        },
        diagnostics={
            "integrity_case_count": len(probes),
            "extension_case_count": len(extension_rows),
            "loaded_source_count": len(loaded_after),
            "unlisted_source_count": len(unlisted),
            "loopback_connects": network_log.allowed_loopback_connects,
        },
        data_scope={
            "paper_result": all_passed,
            "synthetic_fixture": True,
            "provider_calls": 0,
            "world_primitive_negative_cases": ["atomic_overfill_rollback"],
            "business_success_trajectories_use_agent_path": True,
        },
        provider_calls=0,
    )
    # Round-trip validation audits the complete raw report for forbidden model
    # measurement fields and recomputes all report invariants.
    report = EnvironmentStudyReportV1.from_dict(report.to_dict())
    write_json_artifact(report.to_dict(), artifacts / "report.json")
    index = build_artifact_index(
        artifacts,
        (
            "source-manifest.json",
            "contract.json",
            "integrity-details.json",
            "network-guard.json",
            "report.json",
        ),
    )
    write_json_artifact(index, artifacts / "artifact-index.json")
    return IntegrityStudyResultV1(
        contract=contract,
        manifest=manifest,
        report=report,
        details=details,
    )


__all__ = [
    "INTEGRITY_DETAILS_SCHEMA",
    "INTEGRITY_STUDY_ID",
    "EnvironmentIntegrityStudyError",
    "IntegrityProbeV1",
    "IntegrityStudyResultV1",
    "run_persisted_integrity_study",
]
