"""The append-only protocol audit log.

The audit log is the canonical record of on-wire envelopes.  State replay uses
the official artifact bundle: canonical initial snapshot, this ordered audit
stream, and ordered authoritative transaction diffs (or, for scale probes, the
self-contained world-event stream).  An audit digest by itself is never treated
as proof that final World state was reconstructed.

Field growth is additive: new fields (role, skill, budget, protocol_version,
…) can be appended; no field is ever removed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, fields
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from protocol.envelope import from_json, to_json
from runtime.types import AuditRecord, AuditRecordView

if TYPE_CHECKING:
    from protocol.envelope import Envelope
    from runtime.types import TraceRecord


DEFAULT_QUERY_LIMIT = 1_000
MAX_QUERY_LIMIT = 10_000

_MARKET_KEYS = frozenset({"market_id", "market_ids"})
_ACTOR_KEYS = frozenset(
    {
        "actor_id",
        "actor_ids",
        "buyer_id",
        "buyer_ids",
        "merchant_id",
        "merchant_ids",
        "owner_id",
        "owner_ids",
        "participant_id",
        "participant_ids",
        "seller_id",
        "seller_ids",
    }
)
_TRANSACTION_KEYS = frozenset(
    {"order_id", "order_ids", "transaction_id", "transaction_ids", "txn_id", "txn_ids"}
)
_MAX_METADATA_DEPTH = 8
_MAX_METADATA_NODES = 1_024


class AuditLog:
    """JSONL on-disk writer + replay iterator.

    Two streams, one owner:

    * ``audit.jsonl`` — one record per on-wire envelope. ``append`` / ``replay``
      are byte-deterministic. The state-replay verifier combines it with the
      initial snapshot and authoritative transaction-diff stream.
    * ``<stem>.trace.jsonl`` — the reasoning sidecar (item 5), written by
      ``append_trace``. Per-agent-turn cognition; may contain private content;
      excluded from the replay/determinism gate. Created lazily on first trace.
    """

    def __init__(self, path: Path, *, run_id: str | None = None) -> None:
        """Open ``path`` for append. Creates parent dirs if needed.

        ``run_id`` stamps every trace record so the sidecar joins back to the
        on-wire log; it defaults to the audit file's stem (item 9's episode
        runner will pass an explicit episode id).
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        self._run_id = run_id or path.stem
        self._trace_path = path.with_name(f"{path.stem}.trace{path.suffix}")
        self._trace_fh: "IO[str] | None" = None  # opened lazily on first trace
        # Security sidecar — sanitized records of BLOCKED leak attempts. Separate
        # stream from audit.jsonl because a rejected envelope must never enter the
        # replayable on-wire log. Opened lazily on first event.
        self._security_path = path.with_name(f"{path.stem}.security{path.suffix}")
        self._security_fh: "IO[str] | None" = None

    @property
    def path(self) -> Path:
        """Filesystem path of the canonical on-wire audit stream."""

        return self._path

    @property
    def run_id(self) -> str:
        """Stable run identifier stamped on this log's evidence sidecars."""

        return self._run_id

    def append(self, record: "AuditRecord") -> None:
        """Append one record as one JSON line; flush after write.

        The on-disk byte sequence must be deterministic given the same envelopes
        in the same order — this is what makes byte-identical replay possible.
        """
        data = asdict(record)
        data["envelope"] = to_json(record.envelope)
        self._fh.write(json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n")
        self._fh.flush()

    def append_trace(self, record: "TraceRecord") -> None:
        """Append one per-turn reasoning record to the trace sidecar.

        Sidecar-only: this never touches ``audit.jsonl``, so the on-wire log
        stays byte-stable and replay-safe regardless of trace content. The
        sidecar may hold non-deterministic LLM text and private values, so it
        is deliberately excluded from any byte-identical / replay assertion.
        ``default=str`` coerces exotic memory values (e.g. ``Decimal``) since
        the sidecar is for analysis, not byte-exact replay.
        """
        self.append_trace_dict(asdict(record))

    def append_trace_dict(self, data: "dict[str, Any]") -> None:
        """Append an already-serialized trace record (a dict) to the sidecar.

        The cross-process path: a remote agent assembles its own ``TraceRecord``
        and returns it as a dict over ``/vcp`` (the trace is control-plane —
        agent→dispatcher only — and carries the private reasoning context the
        merchant must never see). The dispatcher writes it here without
        reconstructing the dataclass. Same on-disk shape as :meth:`append_trace`;
        sidecar-only, never ``audit.jsonl``.
        """
        if self._trace_fh is None:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_fh = self._trace_path.open("a", encoding="utf-8")
        data = dict(data)
        if not data.get("run_id"):
            data["run_id"] = self._run_id
        self._trace_fh.write(
            json.dumps(
                data,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        self._trace_fh.flush()

    def append_security(self, event: "Any") -> None:
        """Append one sanitized :class:`~runtime.types.SecurityEvent` to the
        security sidecar (``<stem>.security.jsonl``).

        Sidecar-only: this NEVER touches ``audit.jsonl`` (a blocked envelope did
        not reach the wire and must stay out of the replay stream). Stamps
        ``run_id`` when the event leaves it blank, mirroring ``append_trace``.
        The event type guarantees no raw secret / payload is present.
        """
        if self._security_fh is None:
            self._security_path.parent.mkdir(parents=True, exist_ok=True)
            self._security_fh = self._security_path.open("a", encoding="utf-8")
        data = asdict(event)
        if not data.get("run_id"):
            data["run_id"] = self._run_id
        self._security_fh.write(
            json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n"
        )
        self._security_fh.flush()

    def replay(self) -> Iterator["Envelope"]:
        """Yield envelopes in committed order. Used by the replay verifier.

        Reads ``audit.jsonl`` only — the trace sidecar is never consulted here,
        so reasoning content cannot affect replay.
        """
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                yield from_json(record["envelope"])

    def records(self) -> Iterator[AuditRecordView]:
        """Yield indexed audit records in canonical committed order.

        This is the unbounded, streaming primitive used by replay-oriented
        tooling.  Most callers should use :meth:`query`, whose result count is
        bounded and cursor-pageable.  Blank lines are ignored and do not
        consume a sequence number, matching :meth:`replay` semantics.
        """

        if not self._path.exists():
            return
        sequence = 0
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                data = json.loads(line)
                record = _audit_record_from_dict(data)
                yield _record_view(
                    sequence=sequence,
                    record=record,
                    default_market_id=self._run_id,
                )
                sequence += 1

    def query(
        self,
        *,
        market_id: str | None = None,
        actor_id: str | None = None,
        transaction_id: str | None = None,
        order_id: str | None = None,
        after_sequence: int = -1,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> Iterator[AuditRecordView]:
        """Yield a bounded partition while retaining global event positions.

        Supplied filters are combined with AND semantics.  ``actor_id`` may be
        an envelope endpoint or an actor named in a payload; ``transaction_id``
        matches order, transaction, or ``txn_id`` fields (including nested
        fields). ``order_id`` is a convenience alias for querying that same
        identifier set. ``market_id`` comes from the payload when present and
        falls back to this log's ``run_id`` for legacy one-market episodes.

        Results always follow global append order.  Use the last result's
        ``sequence`` as ``after_sequence`` to fetch the next page.  The hard
        upper bound prevents an accidental materialization of an entire large
        market log; :meth:`records` remains available for intentional streaming
        scans.
        """

        _validate_query_bounds(after_sequence=after_sequence, limit=limit)
        if limit == 0:
            return
        emitted = 0
        for view in self.records():
            if view.sequence <= after_sequence:
                continue
            if market_id is not None and market_id not in view.market_ids:
                continue
            if actor_id is not None and actor_id not in view.actor_ids:
                continue
            if transaction_id is not None and transaction_id not in view.transaction_ids:
                continue
            if order_id is not None and order_id not in view.transaction_ids:
                continue
            yield view
            emitted += 1
            if emitted >= limit:
                return

    def close(self) -> None:
        """Flush + close the underlying file handle(s)."""
        self._fh.flush()
        self._fh.close()
        if self._trace_fh is not None:
            self._trace_fh.flush()
            self._trace_fh.close()
            self._trace_fh = None
        if self._security_fh is not None:
            self._security_fh.flush()
            self._security_fh.close()
            self._security_fh = None


def _audit_record_from_dict(data: Any) -> AuditRecord:
    if not isinstance(data, dict):
        raise ValueError("audit record must be a JSON object")
    raw_envelope = data.get("envelope")
    if not isinstance(raw_envelope, str):
        raise ValueError("audit record envelope must be canonical JSON text")
    known = {field.name for field in fields(AuditRecord) if field.name != "envelope"}
    kwargs = {name: data[name] for name in known if name in data}
    return AuditRecord(envelope=from_json(raw_envelope), **kwargs)


def _record_view(
    *,
    sequence: int,
    record: AuditRecord,
    default_market_id: str,
) -> AuditRecordView:
    env = record.envelope
    market_ids = _metadata_values(env.action, _MARKET_KEYS)
    if not market_ids:
        market_ids = (default_market_id,)

    endpoint_ids = (str(env.from_), str(env.to))
    payload_actor_ids = _metadata_values(env.action, _ACTOR_KEYS)
    actor_ids = tuple(sorted(set(endpoint_ids + payload_actor_ids)))

    transaction_ids = _metadata_values(env.action, _TRANSACTION_KEYS)
    transaction_ids = _with_transaction_aliases(transaction_ids)
    return AuditRecordView(
        sequence=sequence,
        record=record,
        market_ids=market_ids,
        actor_ids=actor_ids,
        transaction_ids=transaction_ids,
    )


def _metadata_values(root: Any, keys: frozenset[str]) -> tuple[str, ...]:
    """Collect known identifiers from bounded nested JSON data."""

    found: set[str] = set()
    stack: list[tuple[Any, int]] = [(root, 0)]
    visited = 0
    while stack and visited < _MAX_METADATA_NODES:
        value, depth = stack.pop()
        visited += 1
        if depth > _MAX_METADATA_DEPTH:
            continue
        if isinstance(value, dict):
            for key in sorted(value, reverse=True):
                child = value[key]
                normalized_key = str(key)
                if _identifier_key_matches(normalized_key, keys):
                    _add_identifier(found, child)
                if isinstance(child, (dict, list, tuple)):
                    stack.append((child, depth + 1))
        elif isinstance(value, (list, tuple)):
            for child in reversed(value):
                if isinstance(child, (dict, list, tuple)):
                    stack.append((child, depth + 1))
    return tuple(sorted(found))


def _identifier_key_matches(key: str, keys: frozenset[str]) -> bool:
    if key in keys:
        return True
    return any(key.endswith(f"_{candidate}") for candidate in keys)


def _add_identifier(found: set[str], value: Any) -> None:
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        identifier = str(value)
        if identifier:
            found.add(identifier)
    elif isinstance(value, (list, tuple)):
        for item in value[:_MAX_METADATA_NODES]:
            if isinstance(item, (str, int)) and not isinstance(item, bool):
                identifier = str(item)
                if identifier:
                    found.add(identifier)


def _with_transaction_aliases(values: tuple[str, ...]) -> tuple[str, ...]:
    expanded = set(values)
    for value in values:
        for prefix in ("txn:", "refund:"):
            if value.startswith(prefix) and value[len(prefix):]:
                expanded.add(value[len(prefix):])
    return tuple(sorted(expanded))


def _validate_query_bounds(*, after_sequence: int, limit: int) -> None:
    if not isinstance(after_sequence, int) or isinstance(after_sequence, bool):
        raise TypeError("after_sequence must be an integer")
    if after_sequence < -1:
        raise ValueError("after_sequence must be >= -1")
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise TypeError("limit must be an integer")
    if limit < 0 or limit > MAX_QUERY_LIMIT:
        raise ValueError(f"limit must be between 0 and {MAX_QUERY_LIMIT}")


__all__ = [
    "AuditLog",
    "DEFAULT_QUERY_LIMIT",
    "MAX_QUERY_LIMIT",
]
