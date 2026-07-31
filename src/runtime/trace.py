"""Assembling a per-turn :class:`~runtime.types.TraceRecord` (item 5).

The transport calls :func:`build_trace_record` after an agent turn, with the
turn's :class:`~agents.types.TurnTrace` recorder and the agent's ``Memory``. It
projects the step stream into ``TraceStep`` s and reads the agent's *private*
memory (``TRANSACTION`` for the candidate skeleton + chosen offer,
``PRIVATE_UTILITY`` for the private snapshot) to fill ``considered`` /
``chosen`` / ``private_context``.

Everything here is **sidecar-only**: the values read from private memory
(``max_budget``, ``floor_price``, the ``budget_audit`` that embeds the budget)
are written exclusively to the trace file by ``AuditLog.append_trace`` and
never flow through any path that touches ``audit.jsonl``. Extraction is
best-effort and never raises — a tracing bug must never break a turn.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agents.types import MemoryType
from runtime.turn_failure import SCOREABLE_FAILURE_TERMINALS
from runtime.types import TraceCandidate, TraceChoice, TraceRecord, TraceStep

if TYPE_CHECKING:
    from agents.interfaces import Memory
    from agents.types import TurnTrace


#: Emitted action kind -> the decision label recorded in ``chosen``. Inferred
#: from the on-wire envelope (stable), not from memory shape. Unmapped kinds
#: fall through to the raw kind string so nothing is silently lost.
_DECISION_BY_KIND: dict[str, str] = {
    "commerce.accept_offer": "accept",
    "commerce.counter_offer": "counter",
    "commerce.reject_offer": "reject",
    "commerce.search": "search",
    "commerce.request_offer": "discover",
    "commerce.get_sku": "discover",
    "platform.settle_payment": "settle",
}


class TrackerCaptureError(RuntimeError):
    """Formal Tracker state could not be captured without guessing."""

    error_code = "tracker_capture_failure"


def normalized_decision_for_action_kind(kind: str) -> str:
    """Return the Tracker decision label for one emitted action kind.

    This is deliberately shared with the strict Tracker evidence verifier.
    Keeping normalization here prevents the writer and verifier from drifting
    into two subtly different interpretations of the same audited action.
    """

    return _DECISION_BY_KIND.get(kind, kind)


def build_trace_record(
    *,
    run_id: str,
    turn: int,
    agent_id: str,
    inbound_msg_id: str | None,
    recorder: "TurnTrace",
    memory: "Memory",
    forced_flush: bool = False,
    incomplete: bool = False,
    strict_memory_capture: bool = False,
    selected_offer_override: Mapping[str, Any] | None = None,
) -> TraceRecord:
    """Assemble one turn's :class:`TraceRecord` from its recorder + private memory.

    Under R1 a decision may span several dispatcher rounds (emit world.read ->
    suspend -> resume); ``recorder.steps`` already accumulates them into one
    history, so the record is per-DECISION, stitched by ``recorder.decision_id``.
    ``forced_flush``/``incomplete`` mark a decision force-finalized mid-grounding
    (e.g. at teardown) so a scorer never reads a half-trace as a completed
    no_reply. The grounding reads' fetched attributes are joined onto the
    weighed candidates (the evidence a fabricated-grounding check uses)."""
    steps = tuple(_to_step(h) for h in recorder.steps)
    decision_id = getattr(recorder, "decision_id", None)
    grounded = _grounded_from_steps(recorder.steps, decision_id=decision_id)
    if not isinstance(strict_memory_capture, bool):
        raise TypeError("strict_memory_capture must be boolean")
    chosen = _chosen(
        memory,
        recorder.result,
        strict_memory_capture=strict_memory_capture,
        selected_offer_override=selected_offer_override,
    )
    if recorder.terminal in SCOREABLE_FAILURE_TERMINALS:
        # A rejected/failed decision emitted nothing.  Never retain an offer,
        # price, or free-text rationale from a candidate choice that did not
        # cross the authoritative Runtime boundary.
        chosen = TraceChoice(decision=str(recorder.terminal))
    elif recorder.terminal == "forced_flush":
        # An interrupted decision made no terminal choice.  Do not carry a
        # previously selected offer or rationale into an incomplete record.
        chosen = TraceChoice(decision="no_reply")
    return TraceRecord(
        run_id=run_id,
        turn=turn,
        agent_id=agent_id,
        inbound_msg_id=inbound_msg_id,
        emitted_msg_id=getattr(recorder.result, "msg_id", None),
        terminal=recorder.terminal or "no_reply",
        skill=_last_loaded_skill(recorder.steps),
        skill_digests=_loaded_skill_digests(recorder.steps),
        steps=steps,
        considered=_considered_from_memory(
            memory,
            grounded,
            strict_memory_capture=strict_memory_capture,
        ),
        chosen=chosen,
        private_context=_private_snapshot(
            memory,
            strict_memory_capture=strict_memory_capture,
        ),
        decision_id=decision_id,
        forced_flush=forced_flush,
        incomplete=incomplete,
    )


def _grounded_from_steps(
    steps: "tuple[dict[str, Any], ...]",
    *,
    decision_id: "str | None" = None,
) -> "dict[str, dict[str, Any]]":
    """Return exact, decision-bound grounding evidence by sku.

    The result slots were created by the framework after actually executing
    ``get_listing`` / ``search_catalog``; they are not fields the model may
    self-report.  ``evidence_id`` is transport-neutral, while ``source_msg_id``
    is present only when the read crossed VCP and can be joined to an audited
    ``world.response``.
    """
    grounded: dict[str, dict[str, Any]] = {}
    for step_index, entry in enumerate(steps):
        if entry.get("step") != "tool_call":
            continue
        for result_index, r in enumerate(entry.get("results", [])):
            if not isinstance(r, dict):
                continue
            if r.get("tool") not in ("world.get_listing", "world.search_catalog"):
                continue
            res = r.get("result")
            items = res if isinstance(res, list) else [res]
            for item_index, item in enumerate(items):
                if isinstance(item, dict) and item.get("sku_id"):
                    evidence_id = None
                    if decision_id:
                        evidence_id = (
                            f"{decision_id}:grounding:{step_index}:"
                            f"{result_index}:{item_index}"
                        )
                    grounded[str(item["sku_id"])] = {
                        "attributes": item.get("attributes"),
                        "evidence_id": evidence_id,
                        "source_msg_id": r.get("source_msg_id"),
                    }
    return grounded


def _to_step(entry: "dict[str, Any]") -> TraceStep:
    kind = str(entry.get("step", "?"))
    data = {k: v for k, v in entry.items() if k != "step"}
    return TraceStep(kind=kind, data=data)


def _last_loaded_skill(steps: "tuple[dict[str, Any], ...]") -> str | None:
    name: str | None = None
    for entry in steps:
        if entry.get("step") == "load_skill" and entry.get("name"):
            name = str(entry["name"])
    return name


def _loaded_skill_digests(
    steps: "tuple[dict[str, Any], ...]",
) -> tuple[str, ...]:
    """Project selector-approved skill content digests from turn history.

    The step stream is already deterministically ordered.  Preserve that order
    while removing repeated loads of the same content, and ignore legacy
    ``load_skill`` entries that predate digest recording.  Trace extraction is
    intentionally best effort, so malformed values are skipped rather than
    failing the commerce turn.
    """

    out: list[str] = []
    seen: set[str] = set()
    for entry in steps:
        if entry.get("step") != "load_skill":
            continue
        raw_digest = entry.get("digest")
        if not isinstance(raw_digest, str):
            continue
        digest = raw_digest.strip()
        if not digest or digest in seen:
            continue
        seen.add(digest)
        out.append(digest)
    return tuple(out)


def _sku_from_offer_id(offer_id: Any) -> "str | None":
    """Recover the sku the aggregator encoded in an offer id (``agg:<sku>``) when
    the buyer's ``budget_audit`` entry didn't echo ``sku_id`` (a live model often
    keeps only ``offer_id``). Keeps the trace candidate's sku populated — the
    scorer keys grounding / soft / friend on it, so a null sku makes every such
    check vacuous.  Strip only the ``agg:`` transport prefix; the remainder is
    the complete SKU and may itself contain a merchant namespace."""
    if not isinstance(offer_id, str) or not offer_id.startswith("agg:"):
        return None
    sku = offer_id.removeprefix("agg:")
    return sku or None


def _considered_from_memory(
    memory: "Memory",
    grounded: "dict[str, dict[str, Any]] | None" = None,
    *,
    strict_memory_capture: bool = False,
) -> "tuple[TraceCandidate, ...]":
    """Skeleton candidates from the buyer's ``budget_audit`` write, joined to the
    grounding evidence (``grounded`` keyed by sku_id) this decision fetched.

    Fills id / price / affordability today; ``eligible`` and the reject/stage
    slots stay ``None`` until item 8 emits them directly. ``grounded_attributes``
    is the authoritative fetched evidence for the candidate's sku (single source
    of truth — the result the agent saw). Returns ``()`` for any agent (e.g. the
    merchant) that doesn't keep a ``budget_audit``.
    """
    grounded = grounded or {}
    audit = _safe_read(
        memory,
        MemoryType.TRANSACTION,
        "budget_audit",
        strict=strict_memory_capture,
    )
    if not isinstance(audit, list):
        return ()
    out: list[TraceCandidate] = []
    for c in audit:
        if not isinstance(c, dict):
            continue
        signals = c.get("signals")
        # Derive the sku from the offer id when the buyer didn't echo sku_id, so
        # the candidate's sku is always populated (the grounded join below + the
        # scorer's grounding/soft/friend all key on it).
        sku = c.get("sku_id") or _sku_from_offer_id(c.get("offer_id"))
        g = grounded.get(str(sku)) if sku else None
        out.append(
            TraceCandidate(
                offer_id=c.get("offer_id"),
                sku_id=sku,
                unit_price=c.get("unit_price"),
                total_cents=c.get("total_cents"),
                eta_days=c.get("eta_days"),
                qty=c.get("qty"),
                affordable=c.get("is_affordable"),
                negotiable=c.get("is_negotiable"),
                eligible=c.get("is_eligible"),
                rejected_at_stage=c.get("rejected_at_stage"),
                reject_reason=c.get("reject_reason"),
                rank=c.get("rank"),
                signals=signals if isinstance(signals, dict) else None,
                grounded_attributes=g.get("attributes") if g else None,
                grounded_evidence_id=g.get("evidence_id") if g else None,
                grounded_source_msg_id=g.get("source_msg_id") if g else None,
            )
        )
    return tuple(out)


def _chosen(
    memory: "Memory",
    result: Any,
    *,
    strict_memory_capture: bool = False,
    selected_offer_override: Mapping[str, Any] | None = None,
) -> TraceChoice | None:
    """The turn's decision: label inferred from the emitted envelope kind,
    offer/price from ``selected_offer`` when present."""
    kind: str | None = None
    if result is not None:
        action = getattr(result, "action", None)
        if isinstance(action, dict):
            kind = action.get("kind")
    decision = (
        "no_reply"
        if result is None
        else normalized_decision_for_action_kind(str(kind))
    )

    sel = (
        dict(selected_offer_override)
        if selected_offer_override is not None
        else _safe_read(
            memory,
            MemoryType.TRANSACTION,
            "selected_offer",
            strict=strict_memory_capture,
        )
    )
    offer_id = sel.get("offer_id") if isinstance(sel, dict) else None
    price = sel.get("unit_price") if isinstance(sel, dict) else None
    rationale = sel.get("rationale") if isinstance(sel, dict) else None

    if result is None and offer_id is None:
        return TraceChoice(decision="no_reply", rationale=rationale)
    return TraceChoice(decision=decision, offer_id=offer_id, price=price, rationale=rationale)


def _private_snapshot(
    memory: "Memory",
    *,
    strict_memory_capture: bool = False,
) -> "dict[str, Any] | None":
    """Snapshot the whole PRIVATE_UTILITY bucket — for the sidecar only."""
    keys_fn = getattr(memory, "keys", None)
    if not callable(keys_fn):
        if strict_memory_capture:
            raise TrackerCaptureError("Tracker memory has no keys interface")
        return None
    try:
        names = list(keys_fn(MemoryType.PRIVATE_UTILITY))
    except Exception as exc:
        if strict_memory_capture:
            raise TrackerCaptureError(
                "Tracker could not enumerate private utility memory"
            ) from exc
        return None
    snap = {
        name: _safe_read(
            memory,
            MemoryType.PRIVATE_UTILITY,
            name,
            strict=strict_memory_capture,
        )
        for name in names
    }
    return snap or None


def _safe_read(
    memory: "Memory",
    mtype: MemoryType,
    key: str,
    *,
    strict: bool = False,
) -> Any:
    try:
        return memory.read(mtype, key)
    except Exception as exc:
        if strict:
            raise TrackerCaptureError(
                f"Tracker could not read {mtype.value}.{key}"
            ) from exc
        return None


__all__ = [
    "TrackerCaptureError",
    "build_trace_record",
    "normalized_decision_for_action_kind",
]
