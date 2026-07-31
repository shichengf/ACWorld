"""Canonical journal for Runtime-accepted actor-authored result evidence.

Actor reports are neither Platform decisions nor World facts.  They still need
one trusted acceptance boundary so a benchmark scorer cannot select an
arbitrary message that merely resembles a result.  This module verifies the
strict :mod:`protocol.actor_result` contract against Runtime-supplied binding
facts, then appends the accepted report to a replay-stable JSONL journal.

The journal owns idempotency.  An exact retry returns the existing record and
does not append another line.  Reusing an actor-scoped idempotency key with a
different envelope or report fails closed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, cast

from protocol.actor_result import (
    ACTOR_EVIDENCE_AUTHORITY,
    ActorResultBinding,
    actor_report_submission_to_dict,
    actor_result_from_json,
    actor_result_to_dict,
    actor_result_to_json,
    build_actor_result_report,
    coerce_actor_report_submission,
    coerce_actor_result_report,
    verify_actor_result_binding,
)
from protocol.envelope import Envelope, to_json
from protocol.errors import IdempotencyConflict, SchemaError


ACTOR_EVIDENCE_JOURNAL_SCHEMA = "cwe.actor-evidence-acceptance.v1"
ACTOR_EVIDENCE_ARTIFACT = "actor.evidence.jsonl"

_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "evidence_id",
        "run_id",
        "request_msg_id",
        "request_envelope_sha256",
        "actor_id",
        "principal_id",
        "action_kind",
        "idempotency_key",
        "in_reply_to",
        "task_id",
        "mandate_id",
        "context_id",
        "authority",
        "report",
        "report_sha256",
    }
)


class ActorEvidenceJournalError(SchemaError):
    """An accepted actor-evidence journal is malformed or inconsistent."""


class ActorEvidenceJournal:
    """Append-only, actor-scoped idempotent journal of accepted reports."""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("actor evidence run_id must be non-empty")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        self._run_id = run_id
        self._lock = RLock()
        existing = load_actor_evidence(self._path, strict=True)
        self._records = list(existing)
        if any(record["run_id"] != run_id for record in existing):
            raise ActorEvidenceJournalError(
                "actor evidence journal contains another run_id"
            )
        self._by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for record in existing:
            key = (str(record["actor_id"]), str(record["idempotency_key"]))
            if key in self._by_key:
                raise ActorEvidenceJournalError(
                    "duplicate actor evidence idempotency key"
                )
            self._by_key[key] = record

    @property
    def path(self) -> Path:
        return self._path

    def append(
        self,
        *,
        request: Envelope,
        binding: ActorResultBinding,
    ) -> dict[str, Any]:
        """Validate ``request`` and append one accepted actor report.

        ``binding`` must be constructed from trusted Runtime/scenario context.
        The report payload must never be used to construct it.
        """

        _verify_envelope_binding(request, binding)
        payload = request.action.get("payload")
        if not isinstance(payload, Mapping):
            raise ActorEvidenceJournalError(
                "actor result action payload must be an object"
            )
        # A policy may submit only actor-authored content.  Runtime supplies and
        # seals every identity/causal field, so an LLM never has to compute a
        # digest or self-assert its principal.  The fully sealed wire remains
        # accepted for SDK/reference-policy compatibility, but is still checked
        # against the same trusted binding.
        if frozenset(payload) == frozenset({"outcome", "summary", "details"}):
            submission = coerce_actor_report_submission(payload)
            report = build_actor_result_report(
                binding=binding,
                outcome=submission.outcome,
                summary=submission.summary,
                details=actor_report_submission_to_dict(submission)["details"],
            )
        else:
            report = coerce_actor_result_report(payload)
            verify_actor_result_binding(report, binding)
        report_body = actor_result_to_json(report)
        request_body = to_json(request)
        report_sha256 = _sha256(report_body.encode("utf-8"))
        request_sha256 = _sha256(request_body.encode("utf-8"))
        key = (request.from_, request.idempotency_key)

        with self._lock:
            previous = self._by_key.get(key)
            if previous is not None:
                if (
                    previous["request_envelope_sha256"] == request_sha256
                    and previous["report_sha256"] == report_sha256
                    and previous["report"] == actor_result_to_dict(report)
                ):
                    return _copy_record(previous)
                raise IdempotencyConflict(
                    "actor evidence idempotency key was reused with a different report"
                )

            sequence = len(self._records)
            record: dict[str, Any] = {
                "schema_version": ACTOR_EVIDENCE_JOURNAL_SCHEMA,
                "sequence": sequence,
                "evidence_id": f"{self._run_id}:actor-evidence:{sequence:08d}",
                "run_id": self._run_id,
                "request_msg_id": request.msg_id,
                "request_envelope_sha256": request_sha256,
                "actor_id": request.from_,
                "principal_id": binding.principal_id,
                "action_kind": binding.action_kind,
                "idempotency_key": request.idempotency_key,
                "in_reply_to": binding.in_reply_to,
                "task_id": binding.task_id,
                "mandate_id": binding.mandate_id,
                "context_id": binding.context_id,
                "authority": ACTOR_EVIDENCE_AUTHORITY,
                "report": actor_result_to_dict(report),
                "report_sha256": report_sha256,
            }
            encoded = _canonical_line(record)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
            self._records.append(record)
            self._by_key[key] = record
            return _copy_record(record)


def load_actor_evidence(
    path: str | Path,
    *,
    strict: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Load and independently validate a canonical actor-evidence journal."""

    journal = Path(path)
    if not journal.exists():
        raise ActorEvidenceJournalError(
            f"missing actor evidence journal: {journal}"
        )
    records: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    for line_number, raw in enumerate(
        journal.read_bytes().splitlines(keepends=True), start=1
    ):
        if not raw.strip():
            if strict:
                raise ActorEvidenceJournalError(
                    f"blank actor evidence record at {journal}:{line_number}"
                )
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ActorEvidenceJournalError(
                f"invalid actor evidence at {journal}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ActorEvidenceJournalError(
                f"actor evidence at {journal}:{line_number} is not an object"
            )
        _validate_record(value, expected_sequence=len(records))
        key = (str(value["actor_id"]), str(value["idempotency_key"]))
        if key in keys:
            raise ActorEvidenceJournalError(
                f"duplicate actor evidence idempotency key at {journal}:{line_number}"
            )
        keys.add(key)
        if strict and raw != _canonical_line(value).encode("utf-8"):
            raise ActorEvidenceJournalError(
                f"non-canonical actor evidence bytes at {journal}:{line_number}"
            )
        records.append(value)
    return tuple(_copy_record(record) for record in records)


def _verify_envelope_binding(
    request: Envelope,
    binding: ActorResultBinding,
) -> None:
    action_kind = str(request.action.get("kind", ""))
    mismatches: list[str] = []
    if action_kind != binding.action_kind:
        mismatches.append("action_kind")
    if request.from_ != binding.actor_id:
        mismatches.append("actor_id")
    if request.in_reply_to != binding.in_reply_to:
        mismatches.append("in_reply_to")
    if request.idempotency_key != binding.idempotency_key:
        mismatches.append("idempotency_key")
    if mismatches:
        raise ActorEvidenceJournalError(
            "actor result envelope binding mismatch: " + ", ".join(mismatches)
        )


def _validate_record(value: dict[str, Any], *, expected_sequence: int) -> None:
    actual_fields = frozenset(value)
    if actual_fields != _RECORD_FIELDS:
        raise ActorEvidenceJournalError(
            "actor evidence record fields differ from the exact schema"
        )
    if value.get("schema_version") != ACTOR_EVIDENCE_JOURNAL_SCHEMA:
        raise ActorEvidenceJournalError("unsupported actor evidence journal schema")
    if value.get("sequence") != expected_sequence:
        raise ActorEvidenceJournalError("non-contiguous actor evidence sequence")
    for field in (
        "evidence_id",
        "run_id",
        "request_msg_id",
        "request_envelope_sha256",
        "actor_id",
        "principal_id",
        "action_kind",
        "idempotency_key",
        "in_reply_to",
        "task_id",
        "mandate_id",
        "context_id",
        "authority",
        "report_sha256",
    ):
        if not isinstance(value.get(field), str) or not str(value[field]).strip():
            raise ActorEvidenceJournalError(
                f"actor evidence {field} must be non-empty"
            )
    if value["authority"] != ACTOR_EVIDENCE_AUTHORITY:
        raise ActorEvidenceJournalError("actor evidence authority marker mismatch")
    for digest_field in ("request_envelope_sha256", "report_sha256"):
        digest = str(value[digest_field])
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ActorEvidenceJournalError(
                f"actor evidence {digest_field} is not a SHA-256 digest"
            )
    report_value = value.get("report")
    if not isinstance(report_value, Mapping):
        raise ActorEvidenceJournalError("actor evidence report must be an object")
    try:
        report = coerce_actor_result_report(report_value)
        report_body = actor_result_to_json(report)
    except SchemaError as exc:
        raise ActorEvidenceJournalError(
            f"invalid actor evidence report: {exc}"
        ) from exc
    if value["report_sha256"] != _sha256(report_body.encode("utf-8")):
        raise ActorEvidenceJournalError("actor evidence report digest mismatch")
    # Reparse canonical report JSON to exercise duplicate-key-safe validation
    # independently of the surrounding journal parser.
    try:
        actor_result_from_json(report_body)
    except SchemaError as exc:  # pragma: no cover - guarded by coercion above
        raise ActorEvidenceJournalError(
            f"invalid canonical actor evidence report: {exc}"
        ) from exc
    binding = ActorResultBinding(
        action_kind=report.action_kind,
        actor_id=str(value["actor_id"]),
        principal_id=str(value["principal_id"]),
        task_id=str(value["task_id"]),
        mandate_id=str(value["mandate_id"]),
        context_id=str(value["context_id"]),
        in_reply_to=str(value["in_reply_to"]),
        idempotency_key=str(value["idempotency_key"]),
    )
    try:
        verify_actor_result_binding(report, binding)
    except SchemaError as exc:
        raise ActorEvidenceJournalError(
            f"actor evidence record binding mismatch: {exc}"
        ) from exc


def _canonical_line(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ActorEvidenceJournalError(
            f"actor evidence is not canonical JSON: {exc}"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _copy_record(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_canonical_line(value)))


__all__ = [
    "ACTOR_EVIDENCE_ARTIFACT",
    "ACTOR_EVIDENCE_JOURNAL_SCHEMA",
    "ActorEvidenceJournal",
    "ActorEvidenceJournalError",
    "load_actor_evidence",
]
