"""Deterministic index over an episode's audit artifacts.

Parses ``audit.jsonl`` (one ``AuditRecord`` per line, envelope serialized as a
JSON string), ``audit.trace.jsonl`` (reasoning sidecar), and
``audit.security.jsonl`` (sanitized blocked-leak events). Builds msg-id /
kind / sender / recipient / reply indexes plus the higher-level lookups the step
scorer needs (settle envelopes, merchant price commitments for consent, ranked
candidates, world reads/responses, grounding source refs, security events).

Causal linkage uses ``in_reply_to`` / ``msg_id`` where present, not file order.
Private ``private_context`` in the trace is never treated as truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from protocol.consent import PriceCommitment
from protocol.envelope import Envelope, from_json

_SETTLE_KINDS = ("platform.settle_payment", "settle")
_REJECT_KINDS = ("commerce.reject_offer",)


def _payload(env: Envelope) -> "dict[str, Any]":
    if not isinstance(env.action, dict):
        return {}
    p = env.action.get("payload")
    return p if isinstance(p, dict) else {}


def _kind(env: Envelope) -> str:
    return str(env.action.get("kind", "")) if isinstance(env.action, dict) else ""


@dataclass
class AuditIndex:
    """Indexed, read-only view of one episode's audit streams."""
    envelopes: tuple[Envelope, ...] = ()
    trace: tuple[dict[str, Any], ...] = ()
    security_events: tuple[dict[str, Any], ...] = ()
    by_msg_id: dict[str, Envelope] = field(default_factory=dict)
    by_kind: dict[str, list[Envelope]] = field(default_factory=dict)
    by_sender: dict[str, list[Envelope]] = field(default_factory=dict)
    by_recipient: dict[str, list[Envelope]] = field(default_factory=dict)
    children: dict[str, list[Envelope]] = field(default_factory=dict)
    _pos: dict[str, int] = field(default_factory=dict)

    # --- construction --------------------------------------------------

    @classmethod
    def from_records(
        cls,
        audit_records: "list[dict[str, Any]]",
        trace_records: "list[dict[str, Any]] | None" = None,
        security_records: "list[dict[str, Any]] | None" = None,
    ) -> "AuditIndex":
        envs: list[Envelope] = []
        for rec in audit_records:
            raw = rec.get("envelope") if isinstance(rec, dict) else None
            if isinstance(raw, str):
                envs.append(from_json(raw))
            elif isinstance(raw, dict):
                data = dict(raw)
                if "from" in data:
                    data["from_"] = data.pop("from")
                envs.append(Envelope(**data))
        idx = cls(envelopes=tuple(envs), trace=tuple(trace_records or ()),
                  security_events=tuple(security_records or ()))
        for i, e in enumerate(envs):
            idx._pos[e.msg_id] = i
            idx.by_msg_id[e.msg_id] = e
            idx.by_kind.setdefault(_kind(e), []).append(e)
            idx.by_sender.setdefault(str(e.from_), []).append(e)
            idx.by_recipient.setdefault(str(e.to), []).append(e)
            if e.in_reply_to:
                idx.children.setdefault(str(e.in_reply_to), []).append(e)
        return idx

    @classmethod
    def from_dir(cls, out_dir: "str | Path") -> "AuditIndex":
        out = Path(out_dir)
        return cls.from_records(
            _read_jsonl(out / "audit.jsonl"),
            _read_jsonl(out / "audit.trace.jsonl"),
            _read_jsonl(out / "audit.security.jsonl"),
        )

    # --- lookups -------------------------------------------------------

    def position(self, msg_id: str) -> int:
        """Committed position of ``msg_id`` (for causal ordering), or -1."""
        return self._pos.get(msg_id, -1)

    def of_kind(self, *kinds: str) -> "list[Envelope]":
        out: list[Envelope] = []
        for k in kinds:
            out.extend(self.by_kind.get(k, []))
        return out

    def settle_envelopes(self) -> "list[Envelope]":
        return self.of_kind(*_SETTLE_KINDS)

    def reject_envelopes(self) -> "list[Envelope]":
        return self.of_kind(*_REJECT_KINDS)

    def price_commitments(self, *, before_msg_id: "str | None" = None) -> "list[PriceCommitment]":
        """Merchant-authored price commitments (accept/counter) for consent. If
        ``before_msg_id`` is given, only commitments that causally precede it
        (by committed position) are returned."""
        cutoff = self.position(before_msg_id) if before_msg_id else len(self.envelopes)
        out: list[PriceCommitment] = []
        for e in self.of_kind("commerce.accept_offer", "commerce.counter_offer"):
            if cutoff >= 0 and self.position(e.msg_id) >= cutoff and before_msg_id is not None:
                continue
            p = _payload(e)
            out.append(PriceCommitment(
                from_id=str(e.from_), kind=_kind(e), sku_id=str(p.get("sku_id")),
                unit_price=p.get("unit_price"), msg_id=e.msg_id))
        return out

    def ranked_candidates(self) -> "list[dict[str, Any]]":
        """Flattened candidate dicts from every ``platform.rank_offers``."""
        out: list[dict[str, Any]] = []
        for e in self.by_kind.get("platform.rank_offers", []):
            for c in _payload(e).get("candidates", []) or []:
                if isinstance(c, dict):
                    out.append(c)
        return out

    def world_responses(self) -> "list[Envelope]":
        return self.by_kind.get("world.response", [])

    def trace_by_decision(self) -> "dict[str, dict[str, Any]]":
        out: dict[str, dict[str, Any]] = {}
        for rec in self.trace:
            did = rec.get("decision_id")
            if did:
                out[str(did)] = rec
        return out


def _read_jsonl(path: Path) -> "list[dict[str, Any]]":
    if not path.exists():
        return []
    return [json.loads(li) for li in path.read_text(encoding="utf-8").splitlines() if li.strip()]


__all__ = ["AuditIndex"]
