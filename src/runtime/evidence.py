"""Canonical runtime evidence journals that are separate from on-wire audit.

``audit.jsonl`` proves which envelopes reached a service.  Platform validation
is a different fact: every request addressed to a Platform endpoint must also
produce exactly one accepted/rejected decision record, including requests that
raise before a reply can be emitted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Iterable

from protocol.envelope import to_json

if TYPE_CHECKING:
    from protocol.envelope import Envelope


PLATFORM_DECISION_SCHEMA = "cwe.platform-validation-decision.v1"
PLATFORM_RESPONSE_DISPOSITION_SCHEMA = "cwe.platform-response-disposition.v1"
PLATFORM_RESPONSE_DISPOSITIONS_ARTIFACT = "platform.response-dispositions.jsonl"
_DECISIONS = frozenset({"accepted", "rejected"})
_RESPONSE_STATES = frozenset({"enqueued", "audited", "not_audited_at_shutdown"})


@dataclass(frozen=True, slots=True)
class PlatformResponseOccurrence:
    """Immutable identity of one Platform response occurrence in Runtime.

    A Platform decision proves that the response bytes were produced.  This
    identity lets Runtime append a separate lifecycle without mutating that
    decision: first ``enqueued``, then either ``audited`` or
    ``not_audited_at_shutdown``.
    """

    occurrence_id: str
    decision_sequence: int
    request_msg_id: str
    response_index: int
    response_msg_id: str
    response_envelope_sha256: str
    response_kind: str


class PlatformResponseDispositionJournal:
    """Append-only Runtime observations for Platform response delivery."""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        self._run_id = run_id
        self._lock = RLock()
        existing = load_platform_response_dispositions(
            self._path,
            strict=True,
            require_terminal=False,
        )
        self._next_sequence = len(existing)
        self._occurrences: dict[str, PlatformResponseOccurrence] = {}
        self._terminal_occurrences: set[str] = set()
        for record in existing:
            occurrence = _platform_response_occurrence_from_record(record)
            if record["state"] == "enqueued":
                self._occurrences[occurrence.occurrence_id] = occurrence
            else:
                self._terminal_occurrences.add(occurrence.occurrence_id)

    @property
    def path(self) -> Path:
        return self._path

    def append_enqueued(
        self,
        *,
        decision_sequence: int,
        request_msg_id: str,
        response_index: int,
        response: "Envelope",
    ) -> PlatformResponseOccurrence:
        """Record that Runtime accepted one produced response into its queue."""

        occurrence = PlatformResponseOccurrence(
            occurrence_id=(
                f"{self._run_id}:platform-response:"
                f"{decision_sequence:08d}:{response_index:04d}"
            ),
            decision_sequence=decision_sequence,
            request_msg_id=request_msg_id,
            response_index=response_index,
            response_msg_id=response.msg_id,
            response_envelope_sha256=_sha256(to_json(response).encode("utf-8")),
            response_kind=str(response.action.get("kind", "")),
        )
        with self._lock:
            if occurrence.occurrence_id in self._occurrences:
                raise ValueError("Platform response occurrence is already enqueued")
            self._append(occurrence, state="enqueued")
            self._occurrences[occurrence.occurrence_id] = occurrence
        return occurrence

    def append_terminal(
        self,
        occurrence: PlatformResponseOccurrence,
        *,
        state: str,
    ) -> None:
        """Append the one terminal observation for an enqueued response."""

        if state not in {"audited", "not_audited_at_shutdown"}:
            raise ValueError("Platform response terminal state is invalid")
        with self._lock:
            expected = self._occurrences.get(occurrence.occurrence_id)
            if (
                expected != occurrence
                or occurrence.occurrence_id in self._terminal_occurrences
            ):
                raise ValueError("Platform response occurrence cannot transition")
            self._append(occurrence, state=state)
            self._terminal_occurrences.add(occurrence.occurrence_id)

    def _append(
        self,
        occurrence: PlatformResponseOccurrence,
        *,
        state: str,
    ) -> None:
        if state not in _RESPONSE_STATES:
            raise ValueError("Platform response state is invalid")
        if occurrence.decision_sequence < 0 or occurrence.response_index < 0:
            raise ValueError("Platform response occurrence index is invalid")
        if not occurrence.request_msg_id or not occurrence.response_msg_id:
            raise ValueError("Platform response occurrence identity is blank")
        if not occurrence.response_kind:
            raise ValueError("Platform response kind is blank")
        with self._lock:
            sequence = self._next_sequence
            record = {
                "schema_version": PLATFORM_RESPONSE_DISPOSITION_SCHEMA,
                "sequence": sequence,
                "event_id": (
                    f"{self._run_id}:platform-response-event:{sequence:08d}"
                ),
                "run_id": self._run_id,
                "occurrence_id": occurrence.occurrence_id,
                "decision_sequence": occurrence.decision_sequence,
                "request_msg_id": occurrence.request_msg_id,
                "response_index": occurrence.response_index,
                "response_msg_id": occurrence.response_msg_id,
                "response_envelope_sha256": (
                    occurrence.response_envelope_sha256
                ),
                "response_kind": occurrence.response_kind,
                "state": state,
            }
            encoded = _canonical_line(record)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
            self._next_sequence += 1


class PlatformDecisionJournal:
    """Append-only canonical JSONL writer for Platform validation outcomes."""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        self._run_id = run_id
        self._lock = RLock()
        # A caller may reopen a journal deliberately; continuing from the count
        # preserves append-only sequence semantics rather than overwriting it.
        self._next_sequence = sum(
            1
            for line in self._path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    @property
    def path(self) -> Path:
        return self._path

    def append(
        self,
        *,
        request: "Envelope",
        decision: str,
        reason_code: str,
        response_kinds: Iterable[str] = (),
        response_msg_ids: Iterable[str] = (),
        response_sha256s: Iterable[str] = (),
        decision_metadata: dict[str, Any] | None = None,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        """Append exactly one decision for ``request`` and return its record.

        Exception messages are intentionally excluded.  ``normalized_action``
        is the JSON-normalized action that survived envelope and runtime gates;
        its digest independently binds the record back to ``audit.jsonl``.
        """

        if decision not in _DECISIONS:
            raise ValueError(f"unsupported platform decision: {decision!r}")
        if not reason_code or not _safe_identifier(reason_code):
            raise ValueError("reason_code must be a non-empty safe identifier")
        normalized_action = _json_normalize(request.action)
        action_body = _canonical_json(normalized_action)
        envelope_body = to_json(request)
        with self._lock:
            sequence = self._next_sequence
            record: dict[str, Any] = {
                "schema_version": PLATFORM_DECISION_SCHEMA,
                "sequence": sequence,
                "decision_id": f"{self._run_id}:platform-decision:{sequence:08d}",
                "run_id": self._run_id,
                "request_msg_id": request.msg_id,
                "request_envelope_sha256": _sha256(envelope_body.encode("utf-8")),
                "actor_id": request.from_,
                "platform_endpoint": request.to,
                "action_kind": str(normalized_action.get("kind", "")),
                "idempotency_key": request.idempotency_key,
                "normalized_action": normalized_action,
                "normalized_action_sha256": _sha256(action_body),
                "decision": decision,
                "reason_code": reason_code,
                "response_kinds": sorted(str(kind) for kind in response_kinds),
                "response_msg_ids": [str(value) for value in response_msg_ids],
                "response_sha256s": [str(value) for value in response_sha256s],
                "decision_metadata": _json_normalize(decision_metadata or {}),
                "error_type": error_type,
            }
            encoded = _canonical_line(record)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
            self._next_sequence += 1
            return record


def load_platform_decisions(
    path: str | Path,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Parse and integrity-check a Platform decision journal."""

    journal = Path(path)
    if not journal.exists():
        raise ValueError(f"missing platform decision journal: {journal}")
    records: list[dict[str, Any]] = []
    raw_lines = journal.read_bytes().splitlines(keepends=True)
    for line_number, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            if strict:
                raise ValueError(
                    f"blank platform decision record at {journal}:{line_number}"
                )
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid platform decision at {journal}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"platform decision at {journal}:{line_number} is not an object"
            )
        expected_sequence = len(records)
        if value.get("schema_version") != PLATFORM_DECISION_SCHEMA:
            raise ValueError(
                f"unsupported platform decision schema at {journal}:{line_number}"
            )
        if value.get("sequence") != expected_sequence:
            raise ValueError(
                f"non-contiguous platform decision sequence at {journal}:{line_number}"
            )
        if value.get("decision") not in _DECISIONS:
            raise ValueError(
                f"invalid platform decision outcome at {journal}:{line_number}"
            )
        action = value.get("normalized_action")
        if not isinstance(action, dict):
            raise ValueError(
                f"platform decision at {journal}:{line_number} has no normalized action"
            )
        if value.get("normalized_action_sha256") != _sha256(_canonical_json(action)):
            raise ValueError(
                f"platform decision action digest mismatch at {journal}:{line_number}"
            )
        response_ids = value.get("response_msg_ids", [])
        response_hashes = value.get("response_sha256s", [])
        if (
            not isinstance(response_ids, list)
            or not all(isinstance(item, str) and item for item in response_ids)
            or not isinstance(response_hashes, list)
            or not all(
                isinstance(item, str)
                and len(item) == 64
                and all(ch in "0123456789abcdef" for ch in item)
                for item in response_hashes
            )
            or len(response_ids) != len(response_hashes)
        ):
            raise ValueError(
                f"invalid platform response linkage at {journal}:{line_number}"
            )
        if not isinstance(value.get("decision_metadata", {}), dict):
            raise ValueError(
                f"invalid platform decision metadata at {journal}:{line_number}"
            )
        if strict and raw != _canonical_line(value).encode("utf-8"):
            raise ValueError(
                f"non-canonical platform decision bytes at {journal}:{line_number}"
            )
        records.append(value)
    return records


def load_platform_response_dispositions(
    path: str | Path,
    *,
    strict: bool = True,
    require_terminal: bool = True,
) -> list[dict[str, Any]]:
    """Parse and integrity-check Runtime's Platform response lifecycle.

    Every occurrence has exactly two append-only records.  ``enqueued`` is
    followed by one terminal state.  The common identity fields must remain
    byte-for-byte equal across the transition.
    """

    journal = Path(path)
    if not journal.exists():
        raise ValueError(f"missing Platform response disposition journal: {journal}")
    records: list[dict[str, Any]] = []
    first_by_occurrence: dict[str, dict[str, Any]] = {}
    terminal_occurrences: set[str] = set()
    raw_lines = journal.read_bytes().splitlines(keepends=True)
    identity_fields = {
        "run_id",
        "occurrence_id",
        "decision_sequence",
        "request_msg_id",
        "response_index",
        "response_msg_id",
        "response_envelope_sha256",
        "response_kind",
    }
    for line_number, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            if strict:
                raise ValueError(
                    f"blank Platform response record at {journal}:{line_number}"
                )
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid Platform response record at {journal}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"Platform response record at {journal}:{line_number} is not an object"
            )
        expected_sequence = len(records)
        if value.get("schema_version") != PLATFORM_RESPONSE_DISPOSITION_SCHEMA:
            raise ValueError(
                f"unsupported Platform response schema at {journal}:{line_number}"
            )
        if value.get("sequence") != expected_sequence:
            raise ValueError(
                f"non-contiguous Platform response sequence at {journal}:{line_number}"
            )
        state = value.get("state")
        if state not in _RESPONSE_STATES:
            raise ValueError(
                f"invalid Platform response state at {journal}:{line_number}"
            )
        decision_sequence = value.get("decision_sequence")
        response_index = value.get("response_index")
        if (
            isinstance(decision_sequence, bool)
            or not isinstance(decision_sequence, int)
            or decision_sequence < 0
            or isinstance(response_index, bool)
            or not isinstance(response_index, int)
            or response_index < 0
        ):
            raise ValueError(
                f"invalid Platform response indexes at {journal}:{line_number}"
            )
        if not all(
            isinstance(value.get(field), str) and value.get(field)
            for field in (
                "event_id",
                "run_id",
                "occurrence_id",
                "request_msg_id",
                "response_msg_id",
                "response_kind",
            )
        ):
            raise ValueError(
                f"invalid Platform response identity at {journal}:{line_number}"
            )
        digest = value.get("response_envelope_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise ValueError(
                f"invalid Platform response digest at {journal}:{line_number}"
            )
        occurrence_id = str(value["occurrence_id"])
        first = first_by_occurrence.get(occurrence_id)
        if state == "enqueued":
            if first is not None:
                raise ValueError(
                    f"duplicate Platform response enqueue at {journal}:{line_number}"
                )
            first_by_occurrence[occurrence_id] = value
        else:
            if first is None or occurrence_id in terminal_occurrences:
                raise ValueError(
                    f"invalid Platform response transition at {journal}:{line_number}"
                )
            if any(value.get(field) != first.get(field) for field in identity_fields):
                raise ValueError(
                    f"Platform response identity changed at {journal}:{line_number}"
                )
            terminal_occurrences.add(occurrence_id)
        if strict and raw != _canonical_line(value).encode("utf-8"):
            raise ValueError(
                f"non-canonical Platform response bytes at {journal}:{line_number}"
            )
        records.append(value)
    if require_terminal and set(first_by_occurrence) != terminal_occurrences:
        raise ValueError("Platform response disposition journal has pending occurrences")
    return records


def _platform_response_occurrence_from_record(
    record: dict[str, Any],
) -> PlatformResponseOccurrence:
    return PlatformResponseOccurrence(
        occurrence_id=str(record["occurrence_id"]),
        decision_sequence=int(record["decision_sequence"]),
        request_msg_id=str(record["request_msg_id"]),
        response_index=int(record["response_index"]),
        response_msg_id=str(record["response_msg_id"]),
        response_envelope_sha256=str(record["response_envelope_sha256"]),
        response_kind=str(record["response_kind"]),
    )


def _response_kinds(result: Any) -> tuple[str, ...]:
    if result is None:
        return ()
    values = result if isinstance(result, list) else [result]
    return tuple(
        str(value.action.get("kind", ""))
        for value in values
        if value is not None and isinstance(getattr(value, "action", None), dict)
    )


def _json_normalize(value: Any) -> Any:
    return json.loads(
        json.dumps(value, allow_nan=False, default=str, separators=(",", ":"), sort_keys=True)
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_line(value: Any) -> str:
    return _canonical_json(value).decode("utf-8") + "\n"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_identifier(value: str) -> bool:
    return all(ch.isalnum() or ch in {"_", "-", ".", ":"} for ch in value)


__all__ = [
    "PLATFORM_DECISION_SCHEMA",
    "PLATFORM_RESPONSE_DISPOSITION_SCHEMA",
    "PLATFORM_RESPONSE_DISPOSITIONS_ARTIFACT",
    "PlatformDecisionJournal",
    "PlatformResponseDispositionJournal",
    "PlatformResponseOccurrence",
    "load_platform_decisions",
    "load_platform_response_dispositions",
]
