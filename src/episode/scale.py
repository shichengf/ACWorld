"""Deterministic sparse-market generator and scripted scale probe.

The probe separates environment scalability from paid model inference. It
materializes exactly ``buyers * top_k`` candidate edges, settles at most one
transaction per buyer through the durable World transaction API, writes a
self-contained event log, and rebuilds an independent database by consuming
that log. It never constructs the buyer×merchant Cartesian product.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
import tracemalloc
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from world import (
    AgentId,
    DatabaseWorld,
    InventoryRow,
    Listing,
    Money,
    Order,
    OrderId,
    OrderState,
    Receipt,
    SkuId,
    TxnId,
)
from world.errors import OutOfStock


SCALE_EVENT_SCHEMA = "cwe.scale-event.v1"
SCALE_REPORT_SCHEMA = "cwe.scale-report.v1"
SCALE_SUITE_SCHEMA = "cwe.scale-suite.v1"
SCALE_RUN_MANIFEST_SCHEMA = "cwe.scale-run-manifest.v1"
SCALE_WORKLOAD_SCHEMA = "cwe.sparse-market-workload.v1"
SCALE_PROVENANCE_SCHEMA = "cwe.scale-provenance.v1"

SCALE_RAW_ARTIFACTS = (
    ("primary_database", "world.sqlite3", "measured_world_state"),
    ("replay_database", "world.replay.sqlite3", "captured_replay_state"),
    ("event_log", "events.jsonl", "independent_replay_source"),
    ("run_report", "scale-report.json", "measured_run_report"),
)


@dataclass(frozen=True, slots=True)
class ScaleConfig:
    buyers: int
    merchants: int
    top_k: int = 5
    seed: int = 42
    max_transactions_per_buyer: int = 1

    def __post_init__(self) -> None:
        for name in ("buyers", "merchants", "top_k", "max_transactions_per_buyer"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.top_k > self.merchants:
            raise ValueError("top_k cannot exceed merchant population")
        if self.max_transactions_per_buyer != 1:
            raise ValueError("v1 scale probe supports at most one transaction per buyer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True, slots=True)
class SparseEdge:
    buyer_id: str
    merchant_id: str
    sku_id: str
    rank: int


@dataclass(frozen=True, slots=True)
class SparseMarket:
    buyer_ids: tuple[str, ...]
    merchant_ids: tuple[str, ...]
    listings: tuple[Listing, ...]
    inventory: tuple[InventoryRow, ...]
    edges: tuple[SparseEdge, ...]

    @property
    def cartesian_pairs(self) -> int:
        return len(self.buyer_ids) * len(self.merchant_ids)

    @property
    def materialized_edges(self) -> int:
        return len(self.edges)


@dataclass(frozen=True, slots=True)
class ScaleProbeResult:
    config: ScaleConfig
    materialized_edges: int
    cartesian_pairs: int
    completed_transactions: int
    events_recorded: int
    registration_seconds: float
    query_seconds: float
    settlement_seconds: float
    peak_memory_bytes: int
    database_bytes: int
    event_log_bytes: int
    state_digest: str
    event_digest: str
    replay_state_digest: str
    replay_event_digest: str
    replay_ok: bool
    report_path: str
    replay_seconds: float = 0.0
    run_manifest_path: str | None = None
    schema_version: str = SCALE_REPORT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": asdict(self.config),
            "materialized_edges": self.materialized_edges,
            "cartesian_pairs": self.cartesian_pairs,
            "sparsity_ratio": self.materialized_edges / self.cartesian_pairs,
            "completed_transactions": self.completed_transactions,
            "events_recorded": self.events_recorded,
            "registration_seconds": self.registration_seconds,
            "query_seconds": self.query_seconds,
            "settlement_seconds": self.settlement_seconds,
            "replay_seconds": self.replay_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "database_bytes": self.database_bytes,
            "event_log_bytes": self.event_log_bytes,
            "state_digest": self.state_digest,
            "event_digest": self.event_digest,
            "replay_state_digest": self.replay_state_digest,
            "replay_event_digest": self.replay_event_digest,
            "replay_ok": self.replay_ok,
            "report_path": self.report_path,
            "run_manifest_path": self.run_manifest_path,
            "workload": scale_workload_definition(self.config),
            "evidence_boundary": scale_evidence_boundary(),
        }


@dataclass(frozen=True, slots=True)
class _RunResult:
    market: SparseMarket
    events: tuple[dict[str, Any], ...]
    registration_seconds: float
    query_seconds: float
    settlement_seconds: float
    completed_transactions: int
    state_digest: str
    event_digest: str


@dataclass(frozen=True, slots=True)
class _ReplayResult:
    completed_transactions: int
    state_digest: str
    event_digest: str


class ScaleReplayError(ValueError):
    """A scale event stream is missing, reordered, corrupt, or unsupported."""


def collect_scale_environment() -> dict[str, Any]:
    """Return a non-identifying, dependency-free reproducibility card.

    Total memory is best-effort because ``sysconf`` is not available on every
    supported platform. Host names and hardware serial identifiers are never
    collected.
    """
    memory_bytes: int | None = None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
        memory_bytes = page_size * physical_pages
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "memory_bytes": memory_bytes,
    }


def _scale_repository_root(start: str | Path | None = None) -> Path | None:
    """Find the source checkout without assuming that the package is editable."""

    candidates = [Path(start).resolve()] if start is not None else [Path.cwd().resolve()]
    candidates.append(Path(__file__).resolve())
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate.is_file():
            candidate = candidate.parent
        for directory in (candidate, *candidate.parents):
            if directory in seen:
                continue
            seen.add(directory)
            if (directory / ".git").exists() and (directory / "pyproject.toml").is_file():
                return directory
    return None


def _git_output(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def collect_scale_provenance(
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Capture source and dependency identities for a frozen scale suite.

    The card deliberately excludes paths outside the repository and never
    records user or host identifiers.  Missing VCS metadata is represented
    explicitly so paper-result promotion can fail closed.
    """

    root = _scale_repository_root(repository_root)
    if root is None:
        return {
            "schema_version": SCALE_PROVENANCE_SCHEMA,
            "source_revision": {"vcs": "git", "commit": None, "dirty": None},
            "dependency_lock": None,
            "project_manifest": None,
        }

    revision = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    dependency_lock = root / "uv.lock"
    project_manifest = root / "pyproject.toml"
    return {
        "schema_version": SCALE_PROVENANCE_SCHEMA,
        "source_revision": {
            "vcs": "git",
            "commit": revision,
            "dirty": None if status is None else bool(status),
        },
        "dependency_lock": (
            scale_file_descriptor(dependency_lock, relative_to=root)
            if dependency_lock.is_file()
            else None
        ),
        "project_manifest": (
            scale_file_descriptor(project_manifest, relative_to=root)
            if project_manifest.is_file()
            else None
        ),
    }


def scale_workload_definition(config: ScaleConfig) -> dict[str, Any]:
    """Return the complete, deterministic workload contract for one run.

    The contract describes what is measured without treating a timing sample
    as a paper result.  In particular, the probe materializes a sparse
    candidate graph and never creates the buyer-by-merchant Cartesian matrix.
    """

    return {
        "schema_version": SCALE_WORKLOAD_SCHEMA,
        "population": {
            "buyers": config.buyers,
            "merchants": config.merchants,
        },
        "catalog": {
            "listings_per_merchant": 1,
            "listing_domain": "deterministic_synthetic_scale",
            "initial_inventory_per_listing": config.buyers,
        },
        "candidate_graph": {
            "algorithm": "sha256_start_coprime_stride_v1",
            "seed": config.seed,
            "top_k_per_buyer": config.top_k,
            "materializes_cartesian_matrix": False,
            "expected_materialized_edges": config.buyers * config.top_k,
            "cartesian_pairs_for_reference_only": config.buyers * config.merchants,
        },
        "interaction": {
            "agent_kind": "deterministic_scripted",
            "model_inference": False,
            "maximum_transactions_per_buyer": config.max_transactions_per_buyer,
            "query": "first_available_ranked_candidate",
            "settlement": "database_world_atomic_settle_order",
        },
        "measurement": {
            "registration": "catalog_and_inventory_apply_wall_time",
            "query": "sum_of_catalog_query_wall_times",
            "settlement": "sum_of_settlement_wall_times",
            "replay": "fresh_database_rebuild_wall_time",
            "memory": "python_tracemalloc_peak_for_run_and_replay",
        },
        "replay": {
            "source": "cwe.scale-event.v1_jsonl",
            "event_order": "contiguous_logical_time_from_zero",
            "success": "state_event_and_completed_transaction_digests_match",
        },
    }


def scale_evidence_boundary() -> dict[str, Any]:
    """Describe the archival boundary without making an evidence claim."""

    return {
        "paper_result": False,
        "bundle_kind": "external_raw_artifact_bundle",
        "summary_without_raw_artifacts_is_independently_verifiable": False,
        "required_raw_artifacts": [name for name, _, _ in SCALE_RAW_ARTIFACTS],
        "note": (
            "SQLite and JSONL files need not be committed to source control, but "
            "a frozen external bundle must retain every manifest-listed artifact."
        ),
    }


def scale_file_descriptor(
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
    name: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic SHA-256/size descriptor for a regular file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"scale artifact is missing or not a file: {source}")
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    if relative_to is None:
        display_path = source.name
    else:
        display_path = source.relative_to(Path(relative_to)).as_posix()
    descriptor: dict[str, Any] = {
        "path": display_path,
        "bytes": size,
        "sha256": digest.hexdigest(),
    }
    if name is not None:
        descriptor["name"] = name
    if role is not None:
        descriptor["role"] = role
    return descriptor


def build_scale_run_manifest(
    root: str | Path,
    result: ScaleProbeResult,
) -> dict[str, Any]:
    """Build a relocation-safe manifest for a completed raw run bundle."""

    directory = Path(root)
    artifacts = [
        scale_file_descriptor(
            directory / filename,
            relative_to=directory,
            name=name,
            role=role,
        )
        for name, filename, role in SCALE_RAW_ARTIFACTS
    ]
    return {
        "schema_version": SCALE_RUN_MANIFEST_SCHEMA,
        "config": asdict(result.config),
        "workload": scale_workload_definition(result.config),
        "expected": {
            "completed_transactions": result.completed_transactions,
            "events_recorded": result.events_recorded,
            "event_digest": result.event_digest,
            "state_digest": result.state_digest,
        },
        "artifacts": artifacts,
        "evidence_boundary": scale_evidence_boundary(),
    }


def build_scale_suite_manifest(
    root: str | Path,
    results: tuple[ScaleProbeResult, ...] | list[ScaleProbeResult],
    *,
    environment: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a suite manifest that cryptographically anchors every run manifest.

    Paths are relative to ``root`` so a complete directory can be frozen in an
    external artifact store and later verified after relocation.  The suite
    file itself is anchored by the SHA-256 descriptor emitted by the CLI.
    """

    directory = Path(root)
    ordered = sorted(
        results,
        key=lambda result: (
            result.config.buyers,
            result.config.merchants,
            result.config.seed,
            result.config.top_k,
        ),
    )
    run_manifests: list[dict[str, Any]] = []
    for result in ordered:
        if result.run_manifest_path is None:
            raise ValueError("scale result has no run manifest path")
        run_manifests.append({
            "config": asdict(result.config),
            "manifest": scale_file_descriptor(
                result.run_manifest_path,
                relative_to=directory,
                name="run_manifest",
                role="run_integrity_root",
            ),
        })
    return {
        "schema_version": SCALE_SUITE_SCHEMA,
        "environment": environment if environment is not None else collect_scale_environment(),
        "provenance": provenance if provenance is not None else collect_scale_provenance(),
        "suite_workload": {
            "schema_version": SCALE_WORKLOAD_SCHEMA,
            "run_count": len(ordered),
            "configurations": [asdict(result.config) for result in ordered],
            "all_runs_use_deterministic_scripted_agents": True,
            "all_runs_exclude_model_inference": True,
        },
        # Preserve the original report-oriented field for downstream analysis.
        "runs": [result.to_dict() for result in ordered],
        "run_manifests": run_manifests,
        "evidence_boundary": scale_evidence_boundary(),
    }


def generate_sparse_market(config: ScaleConfig) -> SparseMarket:
    """Generate one listing per merchant and exactly top-k edges per buyer."""
    buyer_ids = tuple(f"buyer:b{index:06d}" for index in range(config.buyers))
    merchant_ids = tuple(f"merchant:m{index:06d}" for index in range(config.merchants))
    listings: list[Listing] = []
    inventory: list[InventoryRow] = []
    for index, merchant_id in enumerate(merchant_ids):
        sku = SkuId(f"{merchant_id}:sku0")
        price_minor = 2_000 + (index % 101)
        listings.append(Listing(
            sku_id=sku,
            product_id=f"synthetic:product:{index % max(1, min(100, config.merchants))}",
            category="synthetic-scale",
            name=f"Synthetic listing {index}",
            attributes={"synthetic": True, "in_stock": True},
            list_price=Money(Decimal(price_minor) / Decimal(100)),
            merchant_id=AgentId(merchant_id),
        ))
        # Capacity B is deliberate: scale measures actor/routing/state overhead,
        # while the separate contention test measures scarce-stock correctness.
        inventory.append(InventoryRow(sku, AgentId(merchant_id), config.buyers))

    edges: list[SparseEdge] = []
    for buyer_index, buyer_id in enumerate(buyer_ids):
        token = hashlib.sha256(f"{config.seed}:{buyer_id}".encode()).digest()
        start = int.from_bytes(token[:8], "big") % config.merchants
        stride = _coprime_stride(
            int.from_bytes(token[8:16], "big"), config.merchants
        )
        seen: set[int] = set()
        step = 0
        while len(seen) < config.top_k:
            merchant_index = (start + step * stride) % config.merchants
            step += 1
            if merchant_index in seen:
                continue
            seen.add(merchant_index)
            merchant_id = merchant_ids[merchant_index]
            edges.append(SparseEdge(
                buyer_id=buyer_id,
                merchant_id=merchant_id,
                sku_id=f"{merchant_id}:sku0",
                rank=len(seen),
            ))
    return SparseMarket(
        buyer_ids=buyer_ids,
        merchant_ids=merchant_ids,
        listings=tuple(listings),
        inventory=tuple(inventory),
        edges=tuple(edges),
    )


def _coprime_stride(raw: int, modulus: int) -> int:
    if modulus == 1:
        return 1
    stride = raw % modulus or 1
    while math.gcd(stride, modulus) != 1:
        stride = (stride + 1) % modulus or 1
    return stride


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        default=str,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def scale_database_state_digest(database: str | Path) -> str:
    """Read a scale database and return the canonical World snapshot digest.

    Verifiers should call this on a temporary copy so opening an archived
    SQLite file can never modify the frozen artifact.
    """

    world = DatabaseWorld(Path(database))
    try:
        return _digest(asdict(world.snapshot()))
    finally:
        world.close()


def _execute_once(path: Path, config: ScaleConfig) -> _RunResult:
    if path.exists():
        raise FileExistsError(f"scale probe refuses to overwrite database: {path}")
    market = generate_sparse_market(config)
    events: list[dict[str, Any]] = []
    logical_time = 0
    for listing, inventory in zip(market.listings, market.inventory):
        events.append(_seed_event(listing, inventory, logical_time=logical_time))
        logical_time += 1

    world = DatabaseWorld(path)
    registration_started = time.perf_counter()
    world.apply({
        "catalog": market.listings,
        "inventory": {row.sku_id: row for row in market.inventory},
    })
    registration_seconds = time.perf_counter() - registration_started

    edges_by_buyer: dict[str, list[SparseEdge]] = {
        buyer_id: [] for buyer_id in market.buyer_ids
    }
    for edge in market.edges:
        edges_by_buyer[edge.buyer_id].append(edge)

    query_seconds = 0.0
    settlement_seconds = 0.0
    completed_transactions = 0
    for buyer_id in market.buyer_ids:
        selected: SparseEdge | None = None
        for edge in edges_by_buyer[buyer_id]:
            query_started = time.perf_counter()
            rows = world.search_catalog(
                "", {"merchant_id": edge.merchant_id}, limit=1
            )
            query_seconds += time.perf_counter() - query_started
            if rows:
                selected = edge
                break
        if selected is None:
            continue
        listing = world.read("catalog", SkuId(selected.sku_id), caller="platform:scale")
        assert listing is not None
        order = Order(
            order_id=OrderId(f"scale-order:{config.seed}:{buyer_id}"),
            buyer_id=AgentId(buyer_id),
            merchant_id=AgentId(selected.merchant_id),
            sku_id=SkuId(selected.sku_id),
            qty=1,
            agreed_price=listing.list_price,
            state=OrderState.PROPOSED,
        )
        receipt = Receipt(
            txn_id=TxnId(f"scale-txn:{config.seed}:{buyer_id}"),
            ts=f"scale:{logical_time:08d}",
            order_id=order.order_id,
            buyer_id=order.buyer_id,
            merchant_id=order.merchant_id,
            sku_id=order.sku_id,
            qty=1,
            price=order.agreed_price,
            idempotency_key=f"scale:{config.seed}:{buyer_id}",
        )
        settle_started = time.perf_counter()
        try:
            world.settle_order(
                order=order,
                receipt=receipt,
                by_role="platform",
                idempotency_key=receipt.idempotency_key,
            )
        except OutOfStock:
            settlement_seconds += time.perf_counter() - settle_started
            continue
        settlement_seconds += time.perf_counter() - settle_started
        events.append({
            "schema_version": SCALE_EVENT_SCHEMA,
            "event_id": f"scale-event:{config.seed}:{logical_time:08d}",
            "logical_time": logical_time,
            "kind": "world.settle_order",
            "payload": {
                "order": _order_payload(order),
                "receipt": _receipt_payload(receipt),
                "by_role": "platform",
                "idempotency_key": receipt.idempotency_key,
            },
        })
        logical_time += 1
        completed_transactions += 1

    snapshot = asdict(world.snapshot())
    world.close()
    return _RunResult(
        market=market,
        events=tuple(events),
        registration_seconds=registration_seconds,
        query_seconds=query_seconds,
        settlement_seconds=settlement_seconds,
        completed_transactions=completed_transactions,
        state_digest=_digest(snapshot),
        event_digest=_digest(events),
    )


def _money_payload(value: Money) -> dict[str, str]:
    return {"amount": str(value.amount), "currency": value.currency}


def _seed_event(
    listing: Listing,
    inventory: InventoryRow,
    *,
    logical_time: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCALE_EVENT_SCHEMA,
        "event_id": f"scale-seed:{logical_time:08d}",
        "logical_time": logical_time,
        "kind": "world.seed_listing",
        "payload": {
            "listing": {
                "sku_id": str(listing.sku_id),
                "product_id": listing.product_id,
                "category": listing.category,
                "name": listing.name,
                "attributes": dict(listing.attributes),
                "list_price": _money_payload(listing.list_price),
                "merchant_id": str(listing.merchant_id),
            },
            "inventory": {
                "sku_id": str(inventory.sku_id),
                "merchant_id": str(inventory.merchant_id),
                "qty_available": inventory.qty_available,
                "qty_reserved": inventory.qty_reserved,
            },
        },
    }


def _order_payload(order: Order) -> dict[str, Any]:
    return {
        "order_id": str(order.order_id),
        "buyer_id": str(order.buyer_id),
        "merchant_id": str(order.merchant_id),
        "sku_id": str(order.sku_id),
        "qty": order.qty,
        "agreed_price": _money_payload(order.agreed_price),
        "state": order.state.value,
    }


def _receipt_payload(receipt: Receipt) -> dict[str, Any]:
    return {
        "txn_id": str(receipt.txn_id),
        "ts": receipt.ts,
        "order_id": str(receipt.order_id),
        "buyer_id": str(receipt.buyer_id),
        "merchant_id": str(receipt.merchant_id),
        "sku_id": str(receipt.sku_id),
        "qty": receipt.qty,
        "price": _money_payload(receipt.price),
        "idempotency_key": receipt.idempotency_key,
    }


def _money_from_payload(raw: Any) -> Money:
    if not isinstance(raw, dict) or "amount" not in raw:
        raise ScaleReplayError("money payload must contain amount and currency")
    return Money(Decimal(str(raw["amount"])), str(raw.get("currency", "USD")))


def _apply_scale_event(world: DatabaseWorld, event: dict[str, Any]) -> bool:
    kind = event.get("kind")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise ScaleReplayError("event payload must be an object")
    if kind == "world.seed_listing":
        listing_raw = payload.get("listing")
        inventory_raw = payload.get("inventory")
        if not isinstance(listing_raw, dict) or not isinstance(inventory_raw, dict):
            raise ScaleReplayError("seed event needs listing and inventory objects")
        listing = Listing(
            sku_id=SkuId(str(listing_raw["sku_id"])),
            product_id=(
                str(listing_raw["product_id"])
                if listing_raw.get("product_id") is not None
                else None
            ),
            category=str(listing_raw["category"]),
            name=str(listing_raw["name"]),
            attributes=dict(listing_raw.get("attributes", {})),
            list_price=_money_from_payload(listing_raw["list_price"]),
            merchant_id=AgentId(str(listing_raw["merchant_id"])),
        )
        inventory = InventoryRow(
            sku_id=SkuId(str(inventory_raw["sku_id"])),
            merchant_id=AgentId(str(inventory_raw["merchant_id"])),
            qty_available=int(inventory_raw["qty_available"]),
            qty_reserved=int(inventory_raw.get("qty_reserved", 0)),
        )
        world.write(
            "catalog", listing.sku_id, listing, by_action="world.update_catalog"
        )
        world.write(
            "inventory",
            inventory.sku_id,
            inventory,
            by_action="world.update_inventory",
        )
        return False
    if kind == "world.settle_order":
        order_raw = payload.get("order")
        receipt_raw = payload.get("receipt")
        if not isinstance(order_raw, dict) or not isinstance(receipt_raw, dict):
            raise ScaleReplayError("settle event needs order and receipt objects")
        order = Order(
            order_id=OrderId(str(order_raw["order_id"])),
            buyer_id=AgentId(str(order_raw["buyer_id"])),
            merchant_id=AgentId(str(order_raw["merchant_id"])),
            sku_id=SkuId(str(order_raw["sku_id"])),
            qty=int(order_raw["qty"]),
            agreed_price=_money_from_payload(order_raw["agreed_price"]),
            state=OrderState(str(order_raw["state"])),
        )
        receipt = Receipt(
            txn_id=TxnId(str(receipt_raw["txn_id"])),
            ts=str(receipt_raw["ts"]),
            order_id=OrderId(str(receipt_raw["order_id"])),
            buyer_id=AgentId(str(receipt_raw["buyer_id"])),
            merchant_id=AgentId(str(receipt_raw["merchant_id"])),
            sku_id=SkuId(str(receipt_raw["sku_id"])),
            qty=int(receipt_raw["qty"]),
            price=_money_from_payload(receipt_raw["price"]),
            idempotency_key=str(receipt_raw["idempotency_key"]),
        )
        world.settle_order(
            order=order,
            receipt=receipt,
            by_role=str(payload.get("by_role", "platform")),
            idempotency_key=str(payload["idempotency_key"]),
        )
        return True
    raise ScaleReplayError(f"unsupported scale event kind: {kind!r}")


def replay_scale_event_log(event_log: str | Path, database: str | Path) -> _ReplayResult:
    """Rebuild a World solely from a captured scale event stream.

    Event ids must be unique and logical times contiguous. Missing, reordered,
    malformed, or unsupported events fail loudly before a replay can be marked
    successful.
    """
    source = Path(event_log)
    target = Path(database)
    if target.exists():
        raise FileExistsError(f"scale replay refuses to overwrite database: {target}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScaleReplayError(f"invalid JSON at line {line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise ScaleReplayError(f"event at line {line_number} must be an object")
        events.append(event)
    if not events:
        raise ScaleReplayError("event log is empty")

    seen: set[str] = set()
    world = DatabaseWorld(target)
    completed = 0
    try:
        for expected_time, event in enumerate(events):
            if event.get("schema_version") != SCALE_EVENT_SCHEMA:
                raise ScaleReplayError("unsupported or missing event schema_version")
            event_id = str(event.get("event_id", ""))
            if not event_id or event_id in seen:
                raise ScaleReplayError(f"missing or duplicate event_id: {event_id!r}")
            seen.add(event_id)
            if event.get("logical_time") != expected_time:
                raise ScaleReplayError(
                    f"event {event_id!r} has logical_time {event.get('logical_time')!r}; "
                    f"expected {expected_time}"
                )
            completed += int(_apply_scale_event(world, event))
        state_digest = _digest(asdict(world.snapshot()))
    finally:
        world.close()
    return _ReplayResult(
        completed_transactions=completed,
        state_digest=state_digest,
        event_digest=_digest(events),
    )


def run_scale_probe(config: ScaleConfig, out_dir: str | Path) -> ScaleProbeResult:
    """Run, independently replay, and manifest one sparse scripted market.

    The output directory is a self-contained raw bundle.  Existing artifacts
    are never overwritten, including the manifest itself.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    database = root / "world.sqlite3"
    replay_database = root / "world.replay.sqlite3"
    event_log = root / "events.jsonl"
    report_path = root / "scale-report.json"
    run_manifest_path = root / "run-manifest.json"
    occupied = [
        path
        for path in (
            database,
            replay_database,
            event_log,
            report_path,
            run_manifest_path,
        )
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(f"scale probe refuses to overwrite artifacts: {occupied}")

    tracemalloc.start()
    try:
        primary = _execute_once(database, config)
        event_body = b"\n".join(_canonical(event) for event in primary.events) + b"\n"
        event_log.write_bytes(event_body)
        replay_started = time.perf_counter()
        replay = replay_scale_event_log(event_log, replay_database)
        replay_seconds = time.perf_counter() - replay_started
        _, peak_memory = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    replay_ok = (
        primary.state_digest == replay.state_digest
        and primary.event_digest == replay.event_digest
        and primary.completed_transactions == replay.completed_transactions
    )
    result = ScaleProbeResult(
        config=config,
        materialized_edges=primary.market.materialized_edges,
        cartesian_pairs=primary.market.cartesian_pairs,
        completed_transactions=primary.completed_transactions,
        events_recorded=len(primary.events),
        registration_seconds=primary.registration_seconds,
        query_seconds=primary.query_seconds,
        settlement_seconds=primary.settlement_seconds,
        peak_memory_bytes=peak_memory,
        database_bytes=os.path.getsize(database),
        event_log_bytes=len(event_body),
        state_digest=primary.state_digest,
        event_digest=primary.event_digest,
        replay_state_digest=replay.state_digest,
        replay_event_digest=replay.event_digest,
        replay_ok=replay_ok,
        report_path=str(report_path),
        replay_seconds=replay_seconds,
        run_manifest_path=str(run_manifest_path),
    )
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_dict(), indent=2) + "\n")
    manifest = build_scale_run_manifest(root, result)
    with run_manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    return result


__all__ = [
    "SCALE_EVENT_SCHEMA",
    "SCALE_REPORT_SCHEMA",
    "SCALE_RUN_MANIFEST_SCHEMA",
    "SCALE_SUITE_SCHEMA",
    "SCALE_WORKLOAD_SCHEMA",
    "SCALE_RAW_ARTIFACTS",
    "SCALE_PROVENANCE_SCHEMA",
    "ScaleConfig",
    "ScaleProbeResult",
    "ScaleReplayError",
    "build_scale_run_manifest",
    "build_scale_suite_manifest",
    "collect_scale_environment",
    "collect_scale_provenance",
    "SparseEdge",
    "SparseMarket",
    "generate_sparse_market",
    "replay_scale_event_log",
    "run_scale_probe",
    "scale_evidence_boundary",
    "scale_database_state_digest",
    "scale_file_descriptor",
    "scale_workload_definition",
]
