"""FastAPI adapter fronting ONE :class:`~agents.base.Agent` behind ``/vcp``.

Lets a buyer / merchant run as a separate process (R1 sub-step 1b). The agent's
re-entrant turn state (:class:`~agents.turn.TurnFrame`) lives HERE, server-side:
a grounding ``tool_call`` is returned to the dispatcher as a ``world.read_*``
envelope, and the matching ``world.response`` (POSTed back by the dispatcher)
resumes the parked frame. The dispatcher never holds the frame — it only relays
envelopes — so the audited stream is identical to the in-process topology.

Trace assembly over HTTP (returning the per-decision ``TraceRecord`` to the
dispatcher's sidecar) is deferred to the trace-stitching proof; this adapter
assembles no trace, which does not affect the on-wire / audited envelopes.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

from agents.remote_world import RemoteWorldTools
from agents.turn import (
    TurnFrame,
    TurnSuspended,
    finalize_forced_turn_trace,
    finalize_rejected_tool_batch,
)
from agents.types import AgentContext, TurnTrace
from protocol.envelope import Envelope, to_json, validate
from protocol.errors import PartitionViolation, SchemaError, UnknownActionKind
from runtime.trace import build_trace_record
from runtime.turn_failure import (
    AgentTurnTraceIdentity,
    SCOREABLE_FAILURE_TERMINALS,
    sanitized_failure_wire,
    tracker_terminal_for_exception,
)


def create_agent_app(agent: Any) -> FastAPI:
    """A ``/vcp`` server for ``agent`` that holds its own re-entrant continuations."""
    app = FastAPI(title=f"Commerce Agent {agent.id}")
    #: read_env.msg_id -> the suspended TurnFrame awaiting that read's response.
    suspended: dict[str, TurnFrame] = {}
    #: Exact terminal candidates returned to the dispatcher but not yet
    #: acknowledged as audited.  This is control-plane state only.  It keeps
    #: the remote topology equivalent to the in-process Runtime, where Agent
    #: state is committed only after ``audit.append`` accepts the same
    #: canonical envelope.
    pending_terminal: dict[str, Envelope] = {}

    @app.exception_handler(PartitionViolation)
    async def _partition(_req: Any, exc: PartitionViolation) -> JSONResponse:
        return JSONResponse(status_code=403, content={"error": str(exc),
                                                      "kind": "partition_violation"})

    @app.exception_handler(SchemaError)
    async def _schema(_req: Any, exc: SchemaError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc),
                                                      "kind": "schema_error"})

    @app.exception_handler(UnknownActionKind)
    async def _kind(_req: Any, exc: UnknownActionKind) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc),
                                                      "kind": "unknown_action_kind"})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/vcp")
    def vcp(payload: dict[str, Any]) -> Any:
        env = _envelope_from_body(payload)
        validate(env)
        kind = str(env.action.get("kind", ""))
        # Resume a parked turn if this is the world.response it awaited; reuse
        # that turn's recorder so the trace stays ONE record across the rounds.
        frame = suspended.pop(env.in_reply_to or "", None) if kind == "world.response" else None
        if kind == "world.response" and frame is None:
            # Stray/duplicate world.response with no parked turn — not a valid
            # fresh-turn trigger; drop it rather than run a spurious new turn.
            return _wrap(None, None)
        recorder = frame.trace if frame is not None else TurnTrace()
        inbound = frame.original_inbound if frame is not None else env
        ctx = AgentContext(
            world=RemoteWorldTools(caller_id=agent.id),
            turn=0, scratchpad=(), trace=recorder, remote_world=True,
        )
        try:
            result = agent.receive(env, ctx, frame=frame)
        except Exception as exc:
            terminal = tracker_terminal_for_exception(
                exc,
                current=recorder.terminal if recorder is not None else None,
            )
            if terminal is None or recorder is None:
                raise
            recorder.finalize_failure(terminal=terminal)
            trace_rec = build_trace_record(
                run_id="",
                turn=0,
                agent_id=agent.id,
                inbound_msg_id=inbound.msg_id,
                recorder=recorder,
                memory=agent.memory,
                incomplete=True,
            )
            if trace_rec.decision_id is None:
                raise RuntimeError("scoreable Agent failure has no decision identity") from exc
            identity = AgentTurnTraceIdentity(
                turn=trace_rec.turn,
                agent_id=trace_rec.agent_id,
                inbound_msg_id=str(trace_rec.inbound_msg_id),
                decision_id=trace_rec.decision_id,
                terminal=trace_rec.terminal,
            )
            return _wrap(
                None,
                trace_rec,
                error=sanitized_failure_wire(exc, identity=identity),
            )
        if isinstance(result, TurnSuspended):
            # Park the frame server-side; hand the read back for the dispatcher
            # to route to World. The world.response will POST back here to resume.
            suspended[result.read_env.msg_id] = result.frame
            return _wrap(result.read_env, None)
        # Terminal: assemble the per-decision TraceRecord HERE (only this
        # process owns the Agent's memory).  Native Agent guards restore their
        # temporary selected-offer overlay before returning, so Tracker must
        # read the exact digest-bound provisional choice instead of ordinary
        # live memory.  The candidate remains provisional until the dispatcher
        # acknowledges the exact envelope after its authoritative audit append.
        try:
            terminal = _single_terminal_envelope(result)
            selected_offer_override = None
            if terminal is not None:
                provisional_choice = getattr(
                    agent,
                    "provisional_selected_offer_for_outbound",
                    None,
                )
                if callable(provisional_choice):
                    selected_offer_override = provisional_choice(terminal)
            trace_rec = None
            if recorder is not None and recorder.finalized:
                trace_rec = build_trace_record(
                    run_id="",
                    turn=0,
                    agent_id=agent.id,
                    inbound_msg_id=inbound.msg_id,
                    recorder=recorder,
                    memory=agent.memory,
                    selected_offer_override=selected_offer_override,
                )
            wrapped = _wrap(result, trace_rec)
        except Exception:
            _reject_terminal_candidate(agent, result)
            raise
        if terminal is not None:
            if terminal.msg_id in pending_terminal:
                _reject_terminal_candidate(agent, terminal)
                raise RuntimeError("remote Agent terminal message id is ambiguous")
            pending_terminal[terminal.msg_id] = terminal
        return wrapped

    @app.post("/provisional/commit")
    def commit_provisional(payload: dict[str, Any]) -> Any:
        """Commit one exact terminal only after the dispatcher audited it."""

        env = _envelope_from_body(payload)
        validate(env)
        candidate = pending_terminal.get(env.msg_id)
        if (
            env.from_ != agent.id
            or candidate is None
            or to_json(candidate) != to_json(env)
        ):
            return JSONResponse(
                status_code=409,
                content={"committed": False, "reason": "candidate_mismatch"},
            )
        commit = getattr(agent, "commit_provisional_outbound", None)
        if callable(commit):
            try:
                commit(env)
            except Exception:
                return JSONResponse(
                    status_code=409,
                    content={"committed": False, "reason": "commit_rejected"},
                )
        del pending_terminal[env.msg_id]
        return {"committed": True}

    @app.post("/provisional/reject")
    def reject_provisional(payload: dict[str, Any]) -> Any:
        """Idempotently discard one exact terminal that was not audited."""

        env = _envelope_from_body(payload)
        validate(env)
        candidate = pending_terminal.get(env.msg_id)
        if candidate is None:
            return {"rejected": False}
        if env.from_ != agent.id or to_json(candidate) != to_json(env):
            return JSONResponse(
                status_code=409,
                content={"rejected": False, "reason": "candidate_mismatch"},
            )
        reject = getattr(agent, "reject_provisional_outbound", None)
        if callable(reject):
            reject(env)
        del pending_terminal[env.msg_id]
        return {"rejected": True}

    @app.post("/flush")
    def flush(
        payload: dict[str, Any] | None = Body(default=None),
    ) -> list[Any]:
        """Forced-flush every turn still suspended at teardown.

        The dispatcher calls this so a decision parked mid-grounding when the
        episode ends is NOT silently dropped over HTTP: this assembles a
        forced_flush/incomplete trace record per parked frame (only this process
        owns the agent's memory) and returns them for the dispatcher to sidecar
        — the cross-process equivalent of ``Runtime.flush_suspended``."""
        raw_ids = (payload or {}).get("audited_pending_read_msg_ids", [])
        if (
            not isinstance(raw_ids, list)
            or any(not isinstance(item, str) or not item for item in raw_ids)
            or len(set(raw_ids)) != len(raw_ids)
        ):
            raise ValueError("flush audited read ids must be unique non-empty strings")
        audited_read_msg_ids = frozenset(raw_ids)
        out: list[Any] = []
        for frame in suspended.values():
            if frame.trace is None:
                continue
            finalize_forced_turn_trace(
                frame,
                audited_read_msg_ids=audited_read_msg_ids,
            )
            rec = build_trace_record(
                run_id="", turn=0, agent_id=agent.id,
                inbound_msg_id=frame.original_inbound.msg_id, recorder=frame.trace,
                memory=agent.memory, forced_flush=True, incomplete=True,
            )
            out.append(json.loads(json.dumps(asdict(rec), default=str)))
        suspended.clear()
        # A terminal candidate can be stranded in the dispatcher's queue when
        # another turn aborts first.  It never reached the authoritative audit,
        # so teardown must discard its remote provisional state without
        # manufacturing a state transition or an additional Tracker row.
        for candidate in tuple(pending_terminal.values()):
            _reject_terminal_candidate(agent, candidate)
        pending_terminal.clear()
        return out

    @app.post("/reject-suspended")
    def reject_suspended(payload: dict[str, Any]) -> Any:
        """Finalize one centrally rejected read with a causal Tracker row.

        The dispatcher owns Router and the scenario SecretRegistry.  When it
        rejects a provisional re-entrant read, the Agent process must close the
        parked turn without claiming that the rejected envelope reached the
        wire.  Only the sanitized failure terminal and exact read identity
        cross this control-plane endpoint.
        """

        read_msg_id = payload.get("read_msg_id")
        terminal = payload.get("terminal")
        if not isinstance(read_msg_id, str) or not read_msg_id:
            raise ValueError("rejected suspended read id must be non-empty")
        if terminal not in SCOREABLE_FAILURE_TERMINALS:
            raise ValueError("rejected suspended terminal is not scoreable")
        frame = suspended.pop(read_msg_id, None)
        if frame is None or frame.pending_read_msg_id != read_msg_id:
            raise ValueError("rejected suspended read does not match a parked turn")
        if frame.trace is None:
            raise RuntimeError("rejected suspended read has no Tracker recorder")
        finalize_rejected_tool_batch(frame)
        frame.trace.finalize_failure(terminal=terminal)
        rec = build_trace_record(
            run_id="",
            turn=0,
            agent_id=agent.id,
            inbound_msg_id=frame.original_inbound.msg_id,
            recorder=frame.trace,
            memory=agent.memory,
            incomplete=True,
        )
        return json.loads(json.dumps(asdict(rec), default=str))

    return app


def _single_terminal_envelope(
    result: "Envelope | list[Envelope] | None",
) -> Envelope | None:
    """Return the Agent's sole terminal envelope, rejecting ambiguous lists."""

    if result is None:
        return None
    if isinstance(result, Envelope):
        return result
    values = [value for value in result if value is not None]
    if len(values) != 1 or not isinstance(values[0], Envelope):
        raise RuntimeError("Agent terminal output must contain exactly one envelope")
    return values[0]


def _reject_terminal_candidate(
    agent: Any,
    result: "Envelope | list[Envelope] | None",
) -> None:
    """Discard an exact candidate when remote post-processing fails."""

    try:
        terminal = _single_terminal_envelope(result)
    except RuntimeError:
        return
    reject = getattr(agent, "reject_provisional_outbound", None)
    if terminal is not None and callable(reject):
        reject(terminal)


def _wrap(
    reply: "Envelope | list[Envelope] | None",
    trace_rec: Any,
    *,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The agent /vcp response envelope: the reply to route + the trace to
    sidecar. ``trace`` carries private reasoning context (control-plane,
    agent→dispatcher only) and is JSON-coerced (``default=str``) so exotic memory
    values (Decimal) serialize cleanly."""
    trace_json = (
        json.loads(json.dumps(asdict(trace_rec), default=str))
        if trace_rec is not None else None
    )
    wrapped = {"reply": _encode(reply), "trace": trace_json}
    if error is not None:
        wrapped["error"] = error
    return wrapped


def _envelope_from_body(payload: dict[str, Any]) -> Envelope:
    data = dict(payload)
    if "from" in data:
        data["from_"] = data.pop("from")
    return Envelope(**data)


def _encode(result: "Envelope | list[Envelope] | None") -> Any:
    if result is None:
        return None
    if isinstance(result, list):
        return [json.loads(to_json(e)) for e in result]
    return json.loads(to_json(result))
