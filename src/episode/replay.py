"""Strict, offline verification of CommerceWorld replay artifacts.

There are two supported replay lanes:

* Episode directories are verified by parsing the canonical audit stream and
  rebuilding the final world snapshot from ``world.initial.json`` plus the
  complete ordered commits in ``world.commits.jsonl``.  Legacy bundles without
  that journal fall back to ``txn_diffs.jsonl``.
* Scale probes are rebuilt into a fresh SQLite database from their
  self-contained ``events.jsonl`` stream.

The episode verifier is deliberately fail-closed.  A world mutation that is
not represented by the transaction-diff stream makes the reconstructed state
differ from ``world.final.json`` and therefore fails verification.  It never
labels an audit hash check alone as state replay.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evals.serialize import WORLD_SCHEMA_VERSION
from protocol.envelope import from_json, to_json
from world.types import WORLD_COMMIT_SCHEMA

from episode.scale import (
    SCALE_RAW_ARTIFACTS,
    SCALE_REPORT_SCHEMA,
    SCALE_RUN_MANIFEST_SCHEMA,
    SCALE_SUITE_SCHEMA,
    ScaleConfig,
    ScaleReplayError,
    replay_scale_event_log,
    scale_database_state_digest,
    scale_evidence_boundary,
    scale_workload_definition,
)


REPLAY_REPORT_SCHEMA = "cwe.replay-report.v1"

_SEQUENCE_TABLE_KEYS = {
    "catalog": "sku_id",
    "orders": "order_id",
    "ledger": "txn_id",
    "disputes": "dispute_id",
    "rulings": "ruling_id",
    "friendships": "buyer_id",
    "reviews": "review_id",
    "fulfillments": "order_id",
    "exchanges": "exchange_id",
    "order_timelines": "order_id",
    "order_groups": "order_group_id",
    "shipments": "shipment_id",
    "search_sessions": "session_id",
    "match_certificates": "cert_id",
    "supply_purchase_authorities": "authority_id",
    "reputation_settlements": "event_id",
    "protocol_events": "event_id",
    "protocol_receipts": "receipt_id",
    "negotiation_events": "event_id",
    "negotiation_threads": "negotiation_id",
    "persistent_cart_quote_requests": "request_id",
    "persistent_cart_quotes": "quote_id",
    "mandate_authorities": "mandate_id",
    "listing_claims": "claim_id",
    "authority_operations": "operation_key",
}
_COMPOSITE_SEQUENCE_TABLES = frozenset(
    {
        "evidence_records",
        "mandate_revisions",
        "pricing_policy_revisions",
        "payment_states",
        "packing_records",
        "after_sales_policies",
        "after_sales_records",
        "governance_policies",
        "governance_records",
    }
)
_MAPPING_TABLES = frozenset(
    {"inventory", "reputation", "match_acceptances", "order_state_revisions"}
)
_SCALAR_TABLE_KEYS = {"logical_time": "world"}


class ReplayVerificationError(ValueError):
    """Raised when an artifact set cannot be replayed exactly."""


@dataclass(frozen=True, slots=True)
class ReplayVerificationResult:
    """Machine-readable result for one independently verified run."""

    kind: str
    target: str
    replay_ok: bool
    events_verified: int
    transactions_replayed: int
    audit_digest: str
    expected_state_digest: str
    replay_state_digest: str
    strict: bool
    schema_version: str = REPLAY_REPORT_SCHEMA
    commits_replayed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by the CLI."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScaleBundleVerificationResult:
    """Integrity and independent-replay result for a frozen scale bundle."""

    bundle_kind: str
    target: str
    raw_bundle_complete: bool
    runs_verified: int
    artifacts_verified: int
    artifact_bytes_verified: int
    events_verified: int
    transactions_replayed: int
    independent_replay_seconds: float
    manifest_sha256: str
    manifest_bytes: int
    paper_result: bool = False
    schema_version: str = "cwe.scale-bundle-verification.v1"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready verification record."""

        return asdict(self)


def _digest(value: Any) -> str:
    body = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReplayVerificationError(f"missing replay artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReplayVerificationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayVerificationError(f"expected a JSON object in {path}")
    return value


def _read_snapshot(path: Path, *, phase: str) -> dict[str, Any]:
    value = _read_json_object(path)
    if value.get("schema_version") != WORLD_SCHEMA_VERSION:
        raise ReplayVerificationError(
            f"unsupported snapshot schema in {path}: {value.get('schema_version')!r}"
        )
    if value.get("phase") != phase:
        raise ReplayVerificationError(
            f"snapshot {path} has phase {value.get('phase')!r}; expected {phase!r}"
        )
    tables = value.get("tables")
    if not isinstance(tables, dict):
        raise ReplayVerificationError(f"snapshot {path} has no object-valued tables")
    return tables


def _verify_audit(path: Path, *, strict: bool) -> tuple[int, str]:
    """Parse every audit record and verify the nested envelope serialization."""

    try:
        raw_lines = path.read_bytes().splitlines(keepends=True)
    except FileNotFoundError as exc:
        raise ReplayVerificationError(f"missing replay artifact: {path}") from exc

    envelopes: list[str] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            if strict:
                raise ReplayVerificationError(
                    f"blank audit record at {path}:{line_number} in strict mode"
                )
            continue
        try:
            text = raw_line.decode("utf-8")
            record = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReplayVerificationError(
                f"invalid audit record at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict) or not isinstance(record.get("envelope"), str):
            raise ReplayVerificationError(
                f"audit record at {path}:{line_number} has no string envelope"
            )
        encoded = record["envelope"]
        try:
            canonical_envelope = to_json(from_json(encoded))
        except Exception as exc:  # protocol errors have several public subclasses
            raise ReplayVerificationError(
                f"invalid envelope at {path}:{line_number}: {exc}"
            ) from exc
        if encoded != canonical_envelope:
            raise ReplayVerificationError(
                f"non-canonical envelope bytes at {path}:{line_number}"
            )
        if strict:
            canonical_record = (
                json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            ).encode("utf-8")
            if raw_line != canonical_record:
                raise ReplayVerificationError(
                    f"non-canonical audit record bytes at {path}:{line_number}"
                )
        envelopes.append(canonical_envelope)

    return len(envelopes), _digest(envelopes)


def _row_index(table: str, value: Any) -> tuple[dict[str, Any], str]:
    """Return a mutable row index and its canonical materialization kind."""

    if table in _MAPPING_TABLES:
        if not isinstance(value, dict):
            raise ReplayVerificationError(f"table {table!r} must be an object")
        return copy.deepcopy(value), "mapping"

    scalar_key = _SCALAR_TABLE_KEYS.get(table)
    if scalar_key is not None:
        if table == "logical_time" and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ReplayVerificationError("logical_time must be a non-negative integer")
        return {scalar_key: copy.deepcopy(value)}, "scalar"

    key_field = _SEQUENCE_TABLE_KEYS.get(table)
    if key_field is None and table not in _COMPOSITE_SEQUENCE_TABLES:
        raise ReplayVerificationError(
            f"transaction diff references unsupported table {table!r}"
        )
    if not isinstance(value, list):
        raise ReplayVerificationError(f"table {table!r} must be an array")
    index: dict[str, Any] = {}
    for position, row in enumerate(value):
        if not isinstance(row, dict):
            raise ReplayVerificationError(
                f"table {table!r} row {position} must be an object"
            )
        if key_field is not None:
            if not isinstance(row.get(key_field), str):
                raise ReplayVerificationError(
                    f"table {table!r} row {position} has no string {key_field!r}"
                )
            key = row[key_field]
        else:
            key = _composite_sequence_key(table, row, position=position)
        if key in index:
            raise ReplayVerificationError(f"duplicate key {key!r} in table {table!r}")
        index[key] = copy.deepcopy(row)
    return index, "sequence"


def _composite_sequence_key(
    table: str, row: dict[str, Any], *, position: int
) -> str:
    """Reconstruct the exact World table key for versioned protocol rows."""

    if table == "evidence_records":
        identity_field, version_field = "record_id", "version"
    elif table == "mandate_revisions":
        identity_field, version_field = "mandate_id", "revision"
    elif table == "payment_states":
        payment_id = row.get("payment_id")
        version = row.get("version")
        if not isinstance(payment_id, str) or not payment_id:
            raise ReplayVerificationError(
                f"table {table!r} row {position} has no string 'payment_id'"
            )
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ReplayVerificationError(
                f"table {table!r} row {position} has invalid 'version'"
            )
        return f"{payment_id}:v{version}"
    elif table == "packing_records":
        packing_id = row.get("packing_id")
        version = row.get("version")
        if not isinstance(packing_id, str) or not packing_id:
            raise ReplayVerificationError(
                f"table {table!r} row {position} has no string 'packing_id'"
            )
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ReplayVerificationError(
                f"table {table!r} row {position} has invalid 'version'"
            )
        return f"{packing_id}:v{version}"
    elif table == "after_sales_policies":
        merchant_id = row.get("merchant_id")
        revision = row.get("revision")
        if not isinstance(merchant_id, str) or not merchant_id:
            raise ReplayVerificationError(
                f"table {table!r} row {position} has no string 'merchant_id'"
            )
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
        ):
            raise ReplayVerificationError(
                f"table {table!r} row {position} has invalid 'revision'"
            )
        return f"{merchant_id}:{revision}"
    elif table == "after_sales_records":
        record_table = row.get("table")
        record_key = row.get("key")
        if not isinstance(record_table, str) or not record_table:
            raise ReplayVerificationError(
                f"table {table!r} row {position} has no string 'table'"
            )
        if not isinstance(record_key, str) or not record_key:
            raise ReplayVerificationError(
                f"table {table!r} row {position} has no string 'key'"
            )
        return (
            f"{len(record_table)}:{record_table}:"
            f"{len(record_key)}:{record_key}"
        )
    elif table in {"governance_policies", "governance_records"}:
        kind = row.get("kind")
        stable_id = row.get("stable_id")
        version_field = "revision" if table == "governance_policies" else "version"
        version = row.get(version_field)
        if not isinstance(kind, str) or not kind:
            raise ReplayVerificationError(
                f"table {table!r} row {position} has no string 'kind'"
            )
        if not isinstance(stable_id, str) or not stable_id:
            raise ReplayVerificationError(
                f"table {table!r} row {position} has no string 'stable_id'"
            )
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ReplayVerificationError(
                f"table {table!r} row {position} has invalid {version_field!r}"
            )
        return f"{kind}:{stable_id}:{version}"
    elif table == "pricing_policy_revisions":
        for field in ("market_id", "merchant_id", "policy_id"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ReplayVerificationError(
                    f"table {table!r} row {position} has no string {field!r}"
                )
        version = row.get("revision")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ReplayVerificationError(
                f"table {table!r} row {position} has invalid 'revision'"
            )
        market_id = row["market_id"]
        merchant_id = row["merchant_id"]
        policy_id = row["policy_id"]
        return (
            f"{len(market_id)}:{market_id}:"
            f"{len(merchant_id)}:{merchant_id}:"
            f"{len(policy_id)}:{policy_id}:{version:020d}"
        )
    else:  # guarded by _COMPOSITE_SEQUENCE_TABLES
        raise ReplayVerificationError(f"unsupported composite table {table!r}")
    identity = row.get(identity_field)
    version = row.get(version_field)
    if not isinstance(identity, str) or not identity:
        raise ReplayVerificationError(
            f"table {table!r} row {position} has no string {identity_field!r}"
        )
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ReplayVerificationError(
            f"table {table!r} row {position} has invalid {version_field!r}"
        )
    return f"{len(identity)}:{identity}:{version:020d}"


def _materialize(
    indexes: dict[str, dict[str, Any]],
    kinds: dict[str, str],
) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for table, index in indexes.items():
        kind = kinds[table]
        if kind == "mapping":
            tables[table] = {key: index[key] for key in sorted(index)}
        elif kind == "sequence":
            # World table snapshots use deterministic key order (not mutation
            # order); e.g. a ``refund:*`` ledger row sorts before ``txn:*``.
            tables[table] = [index[key] for key in sorted(index)]
        elif kind == "scalar":
            scalar_key = _SCALAR_TABLE_KEYS[table]
            if set(index) != {scalar_key}:
                raise ReplayVerificationError(
                    f"scalar table {table!r} has invalid replay keys {sorted(index)}"
                )
            tables[table] = index[scalar_key]
        else:  # defensive: kinds are produced only by _row_index
            raise ReplayVerificationError(
                f"table {table!r} has unknown materialization kind {kind!r}"
            )
    return tables


def _load_transaction_diffs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    diffs: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayVerificationError(
                f"invalid transaction diff at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict) or not isinstance(value.get("table_writes"), list):
            raise ReplayVerificationError(
                f"invalid transaction diff shape at {path}:{line_number}"
            )
        diffs.append(value)
    return diffs


def _load_world_commits(
    path: Path,
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    """Load the canonical complete World commit journal fail-closed."""

    if not path.exists():
        raise ReplayVerificationError(f"missing replay artifact: {path}")
    commits: list[dict[str, Any]] = []
    first_sequence: int | None = None
    previous_digest: str | None = None
    for line_number, raw in enumerate(
        path.read_bytes().splitlines(keepends=True), start=1
    ):
        if not raw.strip():
            if strict:
                raise ReplayVerificationError(
                    f"blank world commit at {path}:{line_number} in strict mode"
                )
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReplayVerificationError(
                f"invalid world commit at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ReplayVerificationError(
                f"world commit at {path}:{line_number} is not an object"
            )
        if value.get("schema_version") != WORLD_COMMIT_SCHEMA:
            raise ReplayVerificationError(
                f"unsupported world commit schema at {path}:{line_number}"
            )
        sequence = value.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ReplayVerificationError(
                f"invalid world commit sequence at {path}:{line_number}"
            )
        if first_sequence is None:
            first_sequence = sequence
        if sequence != first_sequence + len(commits):
            raise ReplayVerificationError(
                f"non-contiguous world commit sequence at {path}:{line_number}"
            )
        if value.get("commit_id") != f"world-commit:{sequence:08d}":
            raise ReplayVerificationError(
                f"world commit id mismatch at {path}:{line_number}"
            )
        if value.get("commit_kind") not in {"write", "transaction"}:
            raise ReplayVerificationError(
                f"invalid world commit kind at {path}:{line_number}"
            )
        if not isinstance(value.get("operation"), str) or not isinstance(
            value.get("authority_action"), str
        ):
            raise ReplayVerificationError(
                f"invalid world commit authority at {path}:{line_number}"
            )
        if not isinstance(value.get("table_writes"), list) or not value["table_writes"]:
            raise ReplayVerificationError(
                f"world commit at {path}:{line_number} has no table writes"
            )
        if value.get("previous_commit_sha256") != previous_digest:
            raise ReplayVerificationError(
                f"broken world commit hash chain at {path}:{line_number}"
            )
        observed_digest = value.get("commit_sha256")
        digest_payload = dict(value)
        digest_payload.pop("commit_sha256", None)
        expected_digest = _digest(digest_payload)
        if observed_digest != expected_digest:
            raise ReplayVerificationError(
                f"world commit digest mismatch at {path}:{line_number}"
            )
        if strict:
            canonical = (
                json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
                + "\n"
            ).encode("utf-8")
            if raw != canonical:
                raise ReplayVerificationError(
                    f"non-canonical world commit bytes at {path}:{line_number}"
                )
        commits.append(value)
        previous_digest = observed_digest
    return commits


def _transaction_projection(commits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "txn": commit["operation"],
            "order_id": commit["subject_id"],
            "table_writes": commit["table_writes"],
            "invariants_held": commit.get("invariants_held", []),
        }
        for commit in commits
        if commit.get("commit_kind") == "transaction"
    ]


def _rebuild_episode_state(
    initial: dict[str, Any],
    diffs: list[dict[str, Any]],
) -> dict[str, Any]:
    indexes: dict[str, dict[str, Any]] = {}
    kinds: dict[str, str] = {}
    for table_name, value in initial.items():
        indexes[table_name], kinds[table_name] = _row_index(table_name, value)

    for txn_position, diff in enumerate(diffs):
        writes = diff["table_writes"]
        if not writes:
            raise ReplayVerificationError(
                f"transaction diff {txn_position} contains no table writes"
            )
        for write_position, write in enumerate(writes):
            where = f"transaction {txn_position} write {write_position}"
            if not isinstance(write, dict):
                raise ReplayVerificationError(f"{where} is not an object")
            table = write.get("table")
            key = write.get("key")
            op = write.get("op")
            if not isinstance(table, str) or not isinstance(key, str):
                raise ReplayVerificationError(f"{where} has an invalid table or key")
            if table not in indexes:
                raise ReplayVerificationError(f"{where} references unknown table {table!r}")
            if op not in {"create", "update"}:
                raise ReplayVerificationError(f"{where} has unsupported operation {op!r}")
            index = indexes[table]
            current = index.get(key)
            before = write.get("before")
            after = write.get("after")
            if current != before:
                raise ReplayVerificationError(
                    f"{where} before-state mismatch for {table}[{key!r}]"
                )
            if op == "create" and (before is not None or key in index):
                raise ReplayVerificationError(
                    f"{where} cannot create existing row {table}[{key!r}]"
                )
            if op == "update" and key not in index:
                raise ReplayVerificationError(
                    f"{where} cannot update missing row {table}[{key!r}]"
                )
            if after is None:
                raise ReplayVerificationError(f"{where} has no after-state")
            if kinds[table] == "scalar" and key != _SCALAR_TABLE_KEYS[table]:
                raise ReplayVerificationError(
                    f"{where} uses invalid scalar key {key!r} for {table!r}"
                )
            index[key] = copy.deepcopy(after)
            # Order revisions are a World-owned materialized ledger. They are
            # advanced at the same commit boundary as an authoritative order
            # write, not accepted as a second actor-supplied table mutation.
            if table == "orders" and "order_state_revisions" in indexes:
                revision_index = indexes["order_state_revisions"]
                if op == "create":
                    revision_index[key] = 1
                else:
                    current_revision = revision_index.get(key, 1)
                    if (
                        isinstance(current_revision, bool)
                        or not isinstance(current_revision, int)
                        or current_revision < 1
                    ):
                        raise ReplayVerificationError(
                            f"{where} found invalid materialized order revision"
                        )
                    revision_index[key] = current_revision + 1

    return _materialize(indexes, kinds)


def verify_episode_replay(
    directory: str | Path,
    *,
    strict: bool = True,
) -> ReplayVerificationResult:
    """Verify one episode directory and independently reconstruct final state.

    ``ReplayVerificationError`` is raised on the first malformed, missing,
    reordered, or state-inconsistent artifact.
    """

    root = Path(directory)
    if not root.is_dir():
        raise ReplayVerificationError(f"episode target is not a directory: {root}")
    initial = _read_snapshot(root / "world.initial.json", phase="initial")
    expected = _read_snapshot(root / "world.final.json", phase="final")
    events, audit_digest = _verify_audit(root / "audit.jsonl", strict=strict)
    commit_path = root / "world.commits.jsonl"
    legacy_path = root / "txn_diffs.jsonl"
    if commit_path.exists():
        commits = _load_world_commits(commit_path, strict=strict)
        transactions = _transaction_projection(commits)
        if legacy_path.exists():
            legacy = _load_transaction_diffs(legacy_path)
            # Validate the compatibility stream semantically before comparing
            # it to the authoritative journal.  This preserves the precise
            # before-state diagnostics existing replay consumers rely on.
            _rebuild_episode_state(initial, legacy)
            if legacy != transactions:
                raise ReplayVerificationError(
                    "txn_diffs.jsonl differs from the transaction projection of "
                    "world.commits.jsonl"
                )
        replayed = _rebuild_episode_state(initial, commits)
    else:
        # Backward compatibility for existing evidence bundles.  New Episode
        # finalization always emits world.commits.jsonl, including for zero writes.
        transactions = _load_transaction_diffs(legacy_path)
        commits = transactions
        replayed = _rebuild_episode_state(initial, transactions)
    if replayed != expected:
        raise ReplayVerificationError(
            "reconstructed episode state differs from world.final.json; "
            "a World commit may be missing, reordered, or corrupt"
        )
    return ReplayVerificationResult(
        kind="episode",
        target=str(root.resolve()),
        replay_ok=True,
        events_verified=events,
        transactions_replayed=len(transactions),
        audit_digest=audit_digest,
        expected_state_digest=_digest(expected),
        replay_state_digest=_digest(replayed),
        strict=strict,
        commits_replayed=len(commits),
    )


def verify_scale_replay(directory: str | Path) -> ReplayVerificationResult:
    """Rebuild a scale probe from events in a fresh temporary database."""

    root = Path(directory)
    events_path = root / "events.jsonl"
    report = _read_json_object(root / "scale-report.json")
    if report.get("schema_version") != SCALE_REPORT_SCHEMA:
        raise ReplayVerificationError(
            f"unsupported scale report schema: {report.get('schema_version')!r}"
        )
    try:
        with tempfile.TemporaryDirectory(prefix="cwe-replay-") as temporary:
            replay = replay_scale_event_log(events_path, Path(temporary) / "world.sqlite3")
    except (ScaleReplayError, OSError, ValueError) as exc:
        raise ReplayVerificationError(f"scale event replay failed: {exc}") from exc

    expected_state = report.get("state_digest")
    expected_events = report.get("event_digest")
    expected_completed = report.get("completed_transactions")
    if (
        replay.state_digest != expected_state
        or replay.event_digest != expected_events
        or replay.completed_transactions != expected_completed
    ):
        raise ReplayVerificationError(
            "scale replay result differs from scale-report.json"
        )
    event_count = sum(
        1 for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    return ReplayVerificationResult(
        kind="scale",
        target=str(root.resolve()),
        replay_ok=True,
        events_verified=event_count,
        transactions_replayed=replay.completed_transactions,
        audit_digest=replay.event_digest,
        expected_state_digest=str(expected_state),
        replay_state_digest=replay.state_digest,
        strict=True,
    )


@dataclass(frozen=True, slots=True)
class _VerifiedScaleRun:
    config: dict[str, Any]
    report: dict[str, Any]
    artifacts_verified: int
    artifact_bytes_verified: int
    events_verified: int
    transactions_replayed: int
    independent_replay_seconds: float
    manifest_sha256: str
    manifest_bytes: int


def _hash_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except FileNotFoundError as exc:
        raise ReplayVerificationError(f"missing scale bundle artifact: {path}") from exc
    return digest.hexdigest(), size


def _safe_manifest_path(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ReplayVerificationError("scale artifact descriptor has no path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReplayVerificationError(f"unsafe scale artifact path: {raw!r}")
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ReplayVerificationError(f"scale artifact escapes bundle root: {raw!r}") from exc
    if candidate.is_symlink():
        raise ReplayVerificationError(f"scale bundle artifact may not be a symlink: {raw!r}")
    if not candidate.is_file():
        raise ReplayVerificationError(f"missing scale bundle artifact: {candidate}")
    return candidate


def _verify_file_descriptor(
    root: Path,
    descriptor: Any,
    *,
    expected_path: str | None = None,
) -> tuple[Path, int]:
    if not isinstance(descriptor, dict):
        raise ReplayVerificationError("scale artifact descriptor must be an object")
    raw_path = descriptor.get("path")
    if expected_path is not None and raw_path != expected_path:
        raise ReplayVerificationError(
            f"scale artifact path is {raw_path!r}; expected {expected_path!r}"
        )
    path = _safe_manifest_path(root, raw_path)
    actual_digest, actual_size = _hash_and_size(path)
    expected_digest = descriptor.get("sha256")
    expected_size = descriptor.get("bytes")
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise ReplayVerificationError(f"invalid SHA-256 descriptor for {raw_path!r}")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise ReplayVerificationError(f"invalid byte-size descriptor for {raw_path!r}")
    if actual_digest != expected_digest or actual_size != expected_size:
        raise ReplayVerificationError(
            f"scale artifact digest/size mismatch for {raw_path!r}"
        )
    return path, actual_size


def _scale_config(raw: Any) -> ScaleConfig:
    if not isinstance(raw, dict):
        raise ReplayVerificationError("scale manifest config must be an object")
    try:
        config = ScaleConfig(**raw)
    except (TypeError, ValueError) as exc:
        raise ReplayVerificationError(f"invalid scale manifest config: {exc}") from exc
    if asdict(config) != raw:
        raise ReplayVerificationError("scale manifest config contains unsupported fields")
    return config


def _copy_and_digest_scale_database(source: Path, target: Path) -> str:
    try:
        shutil.copyfile(source, target)
        return scale_database_state_digest(target)
    except (OSError, ValueError) as exc:
        raise ReplayVerificationError(
            f"cannot read frozen scale database {source}: {exc}"
        ) from exc


def _verify_scale_run_bundle(root: Path) -> _VerifiedScaleRun:
    manifest_path = root / "run-manifest.json"
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema_version") != SCALE_RUN_MANIFEST_SCHEMA:
        raise ReplayVerificationError(
            f"unsupported scale run manifest schema: {manifest.get('schema_version')!r}"
        )
    config = _scale_config(manifest.get("config"))
    config_dict = asdict(config)
    workload = scale_workload_definition(config)
    if manifest.get("workload") != workload:
        raise ReplayVerificationError("scale run manifest workload does not match config")
    if manifest.get("evidence_boundary") != scale_evidence_boundary():
        raise ReplayVerificationError("scale run manifest has an invalid evidence boundary")

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ReplayVerificationError("scale run manifest artifacts must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for descriptor in raw_artifacts:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("name"), str):
            raise ReplayVerificationError("scale run artifact has no stable name")
        name = descriptor["name"]
        if name in by_name:
            raise ReplayVerificationError(f"duplicate scale artifact name: {name!r}")
        by_name[name] = descriptor
    expected_names = {name for name, _, _ in SCALE_RAW_ARTIFACTS}
    if set(by_name) != expected_names:
        missing = sorted(expected_names - set(by_name))
        extra = sorted(set(by_name) - expected_names)
        raise ReplayVerificationError(
            f"scale raw artifact set differs from contract; missing={missing}, extra={extra}"
        )

    paths: dict[str, Path] = {}
    artifact_bytes = 0
    for name, filename, role in SCALE_RAW_ARTIFACTS:
        descriptor = by_name[name]
        if descriptor.get("role") != role:
            raise ReplayVerificationError(f"scale artifact {name!r} has the wrong role")
        path, size = _verify_file_descriptor(root, descriptor, expected_path=filename)
        paths[name] = path
        artifact_bytes += size

    report = _read_json_object(paths["run_report"])
    if report.get("schema_version") != SCALE_REPORT_SCHEMA:
        raise ReplayVerificationError(
            f"unsupported scale report schema: {report.get('schema_version')!r}"
        )
    if report.get("config") != config_dict or report.get("workload") != workload:
        raise ReplayVerificationError("scale report config/workload differs from manifest")
    if report.get("evidence_boundary") != scale_evidence_boundary():
        raise ReplayVerificationError("scale report has an invalid evidence boundary")
    replay_seconds = report.get("replay_seconds")
    if (
        isinstance(replay_seconds, bool)
        or not isinstance(replay_seconds, (int, float))
        or replay_seconds < 0
    ):
        raise ReplayVerificationError("scale report has no valid replay wall time")

    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        raise ReplayVerificationError("scale run manifest has no expected replay result")
    expected_state = expected.get("state_digest")
    expected_events = expected.get("event_digest")
    expected_completed = expected.get("completed_transactions")
    expected_events_recorded = expected.get("events_recorded")
    if (
        report.get("state_digest") != expected_state
        or report.get("replay_state_digest") != expected_state
        or report.get("event_digest") != expected_events
        or report.get("replay_event_digest") != expected_events
        or report.get("completed_transactions") != expected_completed
        or (
            expected_events_recorded is not None
            and report.get("events_recorded") != expected_events_recorded
        )
        or report.get("replay_ok") is not True
    ):
        raise ReplayVerificationError("scale report differs from run manifest expectations")
    if report.get("database_bytes") != by_name["primary_database"].get("bytes"):
        raise ReplayVerificationError("scale report primary database size is inconsistent")
    if report.get("event_log_bytes") != by_name["event_log"].get("bytes"):
        raise ReplayVerificationError("scale report event log size is inconsistent")

    with tempfile.TemporaryDirectory(prefix="cwe-scale-bundle-") as temporary:
        temp = Path(temporary)
        primary_digest = _copy_and_digest_scale_database(
            paths["primary_database"], temp / "primary.sqlite3"
        )
        captured_replay_digest = _copy_and_digest_scale_database(
            paths["replay_database"], temp / "captured-replay.sqlite3"
        )
        started = time.perf_counter()
        try:
            replay = replay_scale_event_log(
                paths["event_log"], temp / "independent-replay.sqlite3"
            )
        except (ScaleReplayError, OSError, ValueError) as exc:
            raise ReplayVerificationError(
                f"independent scale bundle replay failed: {exc}"
            ) from exc
        independent_replay_seconds = time.perf_counter() - started

    if primary_digest != expected_state or captured_replay_digest != expected_state:
        raise ReplayVerificationError(
            "frozen primary or captured replay database differs from expected state"
        )
    if (
        replay.state_digest != expected_state
        or replay.event_digest != expected_events
        or replay.completed_transactions != expected_completed
    ):
        raise ReplayVerificationError(
            "independent event replay differs from run manifest expectations"
        )
    events = sum(
        1
        for line in paths["event_log"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if expected_events_recorded is not None and events != expected_events_recorded:
        raise ReplayVerificationError(
            "scale event count differs from run manifest expectations"
        )
    manifest_digest, manifest_bytes = _hash_and_size(manifest_path)
    return _VerifiedScaleRun(
        config=config_dict,
        report=report,
        artifacts_verified=len(SCALE_RAW_ARTIFACTS),
        artifact_bytes_verified=artifact_bytes,
        events_verified=events,
        transactions_replayed=int(expected_completed),
        independent_replay_seconds=independent_replay_seconds,
        manifest_sha256=manifest_digest,
        manifest_bytes=manifest_bytes,
    )


def _reject_identifying_environment(value: Any) -> None:
    forbidden = {
        "host",
        "hostname",
        "user",
        "username",
        "serial",
        "serial_number",
        "mac_address",
        "home",
        "home_directory",
    }
    if not isinstance(value, dict):
        raise ReplayVerificationError("scale suite environment card must be an object")
    for key, nested in value.items():
        if str(key).lower() in forbidden:
            raise ReplayVerificationError(
                f"scale suite environment card contains identifying field {key!r}"
            )
        if isinstance(nested, dict):
            _reject_identifying_environment(nested)


def verify_scale_artifact_bundle(
    target: str | Path,
) -> ScaleBundleVerificationResult:
    """Verify a raw run or suite and independently replay every event stream.

    A summary JSON file is intentionally insufficient.  The verifier requires
    both SQLite databases, the JSONL event source, the run report, and the
    manifest chain.  It is suitable for a relocated external artifact bundle;
    it does not require the binary SQLite files to live in source control.
    """

    path = Path(target)
    if path.is_file():
        if path.name not in {"scale-suite.json", "run-manifest.json"}:
            raise ReplayVerificationError(f"unsupported scale bundle manifest: {path}")
        root = path.parent
        bundle_kind = "suite" if path.name == "scale-suite.json" else "run"
    elif path.is_dir():
        root = path
        if (root / "scale-suite.json").is_file():
            path = root / "scale-suite.json"
            bundle_kind = "suite"
        elif (root / "run-manifest.json").is_file():
            path = root / "run-manifest.json"
            bundle_kind = "run"
        else:
            raise ReplayVerificationError(
                f"scale bundle has no scale-suite.json or run-manifest.json: {root}"
            )
    else:
        raise ReplayVerificationError(f"scale bundle target does not exist: {path}")

    manifest_digest, manifest_bytes = _hash_and_size(path)
    if bundle_kind == "run":
        run = _verify_scale_run_bundle(root)
        return ScaleBundleVerificationResult(
            bundle_kind="run",
            target=str(root.resolve()),
            raw_bundle_complete=True,
            runs_verified=1,
            artifacts_verified=run.artifacts_verified,
            artifact_bytes_verified=run.artifact_bytes_verified,
            events_verified=run.events_verified,
            transactions_replayed=run.transactions_replayed,
            independent_replay_seconds=run.independent_replay_seconds,
            manifest_sha256=manifest_digest,
            manifest_bytes=manifest_bytes,
        )

    suite = _read_json_object(path)
    if suite.get("schema_version") != SCALE_SUITE_SCHEMA:
        raise ReplayVerificationError(
            f"unsupported scale suite schema: {suite.get('schema_version')!r}"
        )
    if suite.get("evidence_boundary") != scale_evidence_boundary():
        raise ReplayVerificationError("scale suite has an invalid evidence boundary")
    _reject_identifying_environment(suite.get("environment"))
    manifest_entries = suite.get("run_manifests")
    reports = suite.get("runs")
    workload = suite.get("suite_workload")
    if not isinstance(manifest_entries, list) or not manifest_entries:
        raise ReplayVerificationError("scale suite has no run manifest entries")
    if not isinstance(reports, list) or len(reports) != len(manifest_entries):
        raise ReplayVerificationError("scale suite report/run-manifest counts differ")
    if not isinstance(workload, dict):
        raise ReplayVerificationError("scale suite has no explicit workload matrix")

    verified: list[_VerifiedScaleRun] = []
    seen_paths: set[Path] = set()
    manifest_artifact_bytes = 0
    for position, entry in enumerate(manifest_entries):
        if not isinstance(entry, dict):
            raise ReplayVerificationError(f"scale suite run entry {position} is not an object")
        descriptor = entry.get("manifest")
        run_manifest_path, size = _verify_file_descriptor(root, descriptor)
        if run_manifest_path.name != "run-manifest.json":
            raise ReplayVerificationError("suite run descriptor must point to run-manifest.json")
        if run_manifest_path in seen_paths:
            raise ReplayVerificationError("scale suite contains a duplicate run manifest")
        seen_paths.add(run_manifest_path)
        run = _verify_scale_run_bundle(run_manifest_path.parent)
        if entry.get("config") != run.config:
            raise ReplayVerificationError("suite entry config differs from run manifest")
        descriptor_digest = descriptor.get("sha256") if isinstance(descriptor, dict) else None
        if descriptor_digest != run.manifest_sha256 or size != run.manifest_bytes:
            raise ReplayVerificationError("suite descriptor differs from verified run manifest")
        verified.append(run)
        manifest_artifact_bytes += size

    expected_configs = [run.config for run in verified]
    if (
        workload.get("run_count") != len(verified)
        or workload.get("configurations") != expected_configs
        or workload.get("all_runs_use_deterministic_scripted_agents") is not True
        or workload.get("all_runs_exclude_model_inference") is not True
    ):
        raise ReplayVerificationError("scale suite workload matrix differs from its runs")
    if reports != [run.report for run in verified]:
        raise ReplayVerificationError("scale suite embedded reports differ from raw run reports")

    return ScaleBundleVerificationResult(
        bundle_kind="suite",
        target=str(root.resolve()),
        raw_bundle_complete=True,
        runs_verified=len(verified),
        artifacts_verified=sum(run.artifacts_verified for run in verified)
        + len(verified),
        artifact_bytes_verified=sum(
            run.artifact_bytes_verified for run in verified
        )
        + manifest_artifact_bytes,
        events_verified=sum(run.events_verified for run in verified),
        transactions_replayed=sum(run.transactions_replayed for run in verified),
        independent_replay_seconds=sum(
            run.independent_replay_seconds for run in verified
        ),
        manifest_sha256=manifest_digest,
        manifest_bytes=manifest_bytes,
    )


def resolve_replay_targets(target: str | Path) -> tuple[Path, ...]:
    """Resolve a run directory, artifact file, or batch root deterministically."""

    path = Path(target)
    if path.is_file():
        if path.name in {"audit.jsonl", "events.jsonl", "scale-report.json"}:
            path = path.parent
        else:
            raise ReplayVerificationError(f"unsupported replay artifact: {path}")
    if not path.is_dir():
        raise ReplayVerificationError(f"replay target does not exist: {path}")
    if (path / "events.jsonl").is_file() and (path / "scale-report.json").is_file():
        return (path,)
    if (path / "audit.jsonl").is_file():
        return (path,)
    children = sorted(
        {
            candidate.parent
            for candidate in path.rglob("audit.jsonl")
            if (candidate.parent / "world.initial.json").is_file()
            and (candidate.parent / "world.final.json").is_file()
        }
        | {
            candidate.parent
            for candidate in path.rglob("events.jsonl")
            if (candidate.parent / "scale-report.json").is_file()
        },
        key=lambda candidate: str(candidate),
    )
    if not children:
        raise ReplayVerificationError(f"no replayable runs below {path}")
    return tuple(children)


def verify_replay_target(
    target: str | Path,
    *,
    strict: bool = True,
) -> tuple[ReplayVerificationResult, ...]:
    """Verify every run resolved from ``target`` in deterministic path order."""

    results: list[ReplayVerificationResult] = []
    for path in resolve_replay_targets(target):
        if (path / "events.jsonl").is_file() and (path / "scale-report.json").is_file():
            results.append(verify_scale_replay(path))
        else:
            results.append(verify_episode_replay(path, strict=strict))
    return tuple(results)


__all__ = [
    "REPLAY_REPORT_SCHEMA",
    "ReplayVerificationError",
    "ReplayVerificationResult",
    "ScaleBundleVerificationResult",
    "resolve_replay_targets",
    "verify_episode_replay",
    "verify_replay_target",
    "verify_scale_artifact_bundle",
    "verify_scale_replay",
]
