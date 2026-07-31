"""Transport strategies for the Runtime dispatch step.

The :class:`~runtime.bus.Runtime` owns the queue, the single audit log, and
the partition checks — the deterministic, replayable bus. *How* a validated
envelope reaches its target is the transport's job:

* :class:`InProcessTransport` calls the target object directly (the original
  behavior — fast unit tests, byte-exact replay, the demos).
* :class:`HttpTransport` POSTs the envelope to the target's ``/vcp`` endpoint,
  so World / Platform / Buyer / Merchant can each run as a **separate process**
  while the dispatcher keeps one ordered audit trail.

Both satisfy the :class:`Transport` protocol, so ``Runtime`` never needs to
know which topology it is running. This is the "central dispatcher, additive"
design: the bus stays single and authoritative; only the last delivery hop
becomes pluggable.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agents.turn import PUBLIC_WORLD_READ_ACTION_KINDS, TurnSuspended
from agents.types import AgentContext, TurnTrace
from protocol.envelope import Envelope, from_json, to_json
from runtime.errors import AgentNotRegistered
from runtime.trace import build_trace_record
from runtime.turn_failure import (
    attach_trace_identity,
    failure_trace_dict,
    remote_failure_from_wire,
    tracker_terminal_for_exception,
)
from world.tools import WorldTools

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

    from agents.base import Agent
    from agents.platform import PlatformService
    from agents.turn import TurnFrame
    from runtime.types import TraceRecord
    from world.service import WorldService
    from world.state import World


@runtime_checkable
class Transport(Protocol):
    """Deliver one already-validated, already-audited envelope to its target.

    Returns the target's reply (one envelope, several, or ``None``). Must not
    re-run partition or payload checks — the Runtime did those before calling.
    """

    def deliver(
        self, env: Envelope, *, turn: int, frame: "TurnFrame | None" = None
    ) -> "Envelope | list[Envelope] | TurnSuspended | None": ...


class InProcessTransport:
    """Dispatch by direct method call — the original ``Runtime`` behavior.

    Holds a reference to the live ``agents`` dict (not a copy) so agents
    registered after construction are visible.

    ``remote_world=True`` runs agents in R1 re-entrant mode: a grounding
    ``tool_call`` suspends and its world read goes back through the bus, exactly
    as it would over HTTP. The default (``False``) keeps the original synchronous
    behavior (grounding reads call ``WorldTools`` in-process).
    """

    def __init__(
        self,
        *,
        world: "World",
        world_service: "WorldService",
        platform: "PlatformService | None",
        agents: "dict[str, Agent]",
        trace_sink: "Callable[[TraceRecord], None] | None" = None,
        remote_world: bool = False,
        output_validator: "Callable[[Envelope, str], None] | None" = None,
        strict_tracker_capture: bool = False,
    ) -> None:
        if not isinstance(strict_tracker_capture, bool):
            raise TypeError("strict_tracker_capture must be boolean")
        self._world = world
        self._world_service = world_service
        self._platform = platform
        self._agents = agents
        self._trace_sink = trace_sink
        self._remote_world = remote_world
        self._output_validator = output_validator
        self._strict_tracker_capture = strict_tracker_capture

    def configure_agent_output_validator(
        self, validator: "Callable[[Envelope, str], None]"
    ) -> None:
        """Install Runtime's Router-backed pre-commit validator."""

        self._output_validator = validator

    def configure_trace_sink(self, sink: "Callable[[TraceRecord], None]") -> None:
        """Let Runtime stage output-bound Tracker rows until wire audit."""

        self._trace_sink = sink

    def known_agent_addresses(self) -> frozenset[str]:
        """Return currently registered in-process actor ids."""

        return frozenset(self._agents)

    def deliver(
        self, env: Envelope, *, turn: int, frame: "TurnFrame | None" = None
    ) -> "Envelope | list[Envelope] | TurnSuspended | None":
        to = env.to
        # A consumer principal is an external authority, not an executable
        # commerce agent.  Buyer status/rejection notifications terminate at
        # this audited boundary instead of requiring a registered inference
        # channel.  Runtime already validated the delegate partition before
        # delivery, so this cannot be used as an arbitrary message sink.
        if (
            to == "consumer"
            or to.startswith("consumer:")
            or to == "runtime"
            or to.startswith("runtime:")
        ):
            return None
        if to == "world":
            result = self._world_service.handle(env)
            if str(env.action.get("kind", "")) == "world.read_inventory":
                return _project_inventory_reply(result)
            return result
        if to.startswith("platform:"):
            if self._platform is None:
                raise AgentNotRegistered(to)
            return self._platform.handle(env)
        agent = self._agents.get(to)
        if agent is None:
            raise AgentNotRegistered(to)
        if frame is None and str(env.action.get("kind", "")) == "world.response":
            # A world.response with no parked turn is a stray/duplicate — never a
            # valid fresh-turn trigger. Drop it (in-process, a legitimate resume
            # always arrives with its frame); otherwise it would be run as a new
            # turn and emit a spurious read.
            return None
        if frame is not None:
            # R1 resume: reuse the suspended turn's recorder; world reads keep
            # going over the bus (ctx.world is unused on the re-entrant path).
            ctx = AgentContext(
                world=WorldTools(self._world, caller_id=to),
                turn=turn,
                scratchpad=(),
                trace=frame.trace,
                remote_world=True,
            )
            recorder = frame.trace
            inbound = frame.original_inbound
        else:
            recorder = TurnTrace() if self._trace_sink is not None else None
            ctx = AgentContext(
                world=WorldTools(self._world, caller_id=to),
                turn=turn,
                scratchpad=(),
                trace=recorder,
                remote_world=self._remote_world,
            )
            inbound = env
        try:
            result = (
                agent.receive(env, ctx, frame=frame)
                if frame is not None
                else agent.receive(env, ctx)
            )
            if isinstance(result, TurnSuspended):
                try:
                    self._validate_agent_result(result.read_env, agent_id=to)
                except Exception:
                    from agents.turn import finalize_rejected_tool_batch

                    finalize_rejected_tool_batch(result.frame)
                    raise
            else:
                try:
                    self._validate_agent_result(result, agent_id=to)
                except Exception:
                    self._reject_provisional_agent_result(agent, result)
                    raise
        except Exception as exc:
            terminal = tracker_terminal_for_exception(
                exc,
                current=recorder.terminal if recorder is not None else None,
            )
            if terminal is not None and recorder is not None:
                recorder.finalize_failure(terminal=terminal)
                record = self._flush_trace(
                    recorder,
                    agent=agent,
                    turn=turn,
                    agent_id=to,
                    inbound=inbound,
                    required=True,
                    incomplete=True,
                )
                if record is not None and record.decision_id is not None:
                    attach_trace_identity(
                        exc,
                        turn=record.turn,
                        agent_id=record.agent_id,
                        inbound_msg_id=str(record.inbound_msg_id),
                        decision_id=record.decision_id,
                        terminal=record.terminal,
                    )
            raise
        if isinstance(result, TurnSuspended):
            return result  # turn not done — don't flush the trace yet
        try:
            self._flush_trace(
                recorder,
                agent=agent,
                turn=turn,
                agent_id=to,
                inbound=inbound,
            )
        except Exception:
            self._reject_provisional_agent_result(agent, result)
            raise
        return result

    @staticmethod
    def _reject_provisional_agent_result(
        agent: "Agent",
        result: "Envelope | list[Envelope] | None",
    ) -> None:
        reject = getattr(agent, "reject_provisional_outbound", None)
        if not callable(reject) or result is None:
            return
        values = result if isinstance(result, list) else [result]
        for value in values:
            reject(value)

    def _validate_agent_result(
        self,
        result: "Envelope | list[Envelope] | None",
        *,
        agent_id: str,
    ) -> None:
        if self._output_validator is None or result is None:
            return
        values = result if isinstance(result, list) else [result]
        for value in values:
            self._output_validator(value, agent_id)

    def _flush_trace(
        self,
        recorder: "TurnTrace | None",
        *,
        agent: "Agent",
        turn: int,
        agent_id: str,
        inbound: Envelope,
        required: bool = False,
        incomplete: bool = False,
    ) -> "TraceRecord | None":
        """Build + emit this turn's reasoning record. Sidecar-only; best-effort.

        Reads the agent's private memory here (never on the on-wire path) and
        swallows any error — a tracing bug must never break the simulation.
        """
        required = required or self._strict_tracker_capture
        if self._trace_sink is None or recorder is None or not recorder.finalized:
            if required:
                raise RuntimeError("scoreable Agent failure has no writable Tracker sink")
            return None
        try:
            selected_offer_override = None
            result = recorder.result
            provisional_choice = getattr(
                agent,
                "provisional_selected_offer_for_outbound",
                None,
            )
            if isinstance(result, Envelope) and callable(provisional_choice):
                selected_offer_override = provisional_choice(result)
            record = build_trace_record(
                run_id="",  # AuditLog.append_trace stamps the run id
                turn=turn,
                agent_id=agent_id,
                inbound_msg_id=inbound.msg_id,
                recorder=recorder,
                memory=agent.memory,
                incomplete=incomplete,
                strict_memory_capture=self._strict_tracker_capture,
                selected_offer_override=selected_offer_override,
            )
            self._trace_sink(record)
            return record
        except Exception as exc:  # noqa: BLE001 — tracing must never break a turn
            if required:
                raise
            print(f"[trace] capture failed for {agent_id!r} turn {turn}: {exc}", file=sys.stderr)
            return None


class HttpTransport:
    """Dispatch by POSTing to each participant's ``/vcp`` endpoint.

    ``clients`` maps a VCP address — a full ``side:role`` id or a bare
    ``side`` — to an :class:`httpx.Client` whose ``base_url`` points at that
    participant's server (a real socket, or a :class:`fastapi.testclient.
    TestClient` in tests). Resolution tries the exact ``env.to`` first, then
    its side prefix, so ``{"world": ..., "platform": ..., "buyer": ...,
    "merchant": ...}`` covers ``platform:aggregator`` and
    ``merchant:fulfillment`` alike.
    """

    def __init__(
        self,
        *,
        clients: "dict[str, httpx.Client]",
        trace_sink: "Callable[[dict[str, Any]], None] | None" = None,
        output_validator: "Callable[[Envelope, str], None] | None" = None,
    ) -> None:
        self._clients = dict(clients)
        # Agent /vcp returns {reply, trace}; the dispatcher sidecars the trace
        # here (it is control-plane, agent→dispatcher only). World returns a bare
        # envelope and never carries a trace.
        self._trace_sink = trace_sink
        self._output_validator = output_validator

    def configure_agent_output_validator(
        self, validator: "Callable[[Envelope, str], None]"
    ) -> None:
        """Install Runtime's Router-backed pre-commit validator."""

        self._output_validator = validator

    def configure_trace_sink(self, sink: "Callable[[dict[str, Any]], None]") -> None:
        """Let Runtime stage output-bound Tracker rows until wire audit."""

        self._trace_sink = sink

    def known_agent_addresses(self) -> frozenset[str]:
        """Return dispatcher-configured VCP addresses for security metadata."""

        return frozenset(self._clients)

    def commit_provisional_outbound(self, env: Envelope) -> None:
        """Acknowledge one exact remote Agent envelope after audit commit."""

        response = self._client_for(env.from_).post(
            "/provisional/commit",
            content=to_json(env),
            headers={"content-type": "application/json"},
        )
        if response.status_code != 200:
            raise RuntimeError("remote Agent rejected an audited provisional envelope")
        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - control data must be exact
            raise RuntimeError("remote Agent provisional commit response is not JSON") from exc
        if not isinstance(payload, dict) or payload.get("committed") is not True:
            raise RuntimeError("remote Agent did not confirm provisional envelope commit")

    def reject_provisional_outbound(self, env: Envelope) -> bool:
        """Ask a remote Agent to discard one exact unaudited candidate."""

        response = self._client_for(env.from_).post(
            "/provisional/reject",
            content=to_json(env),
            headers={"content-type": "application/json"},
        )
        if response.status_code == 409:
            # An altered envelope must never discard the real staged candidate.
            return False
        if response.status_code != 200:
            raise RuntimeError("remote Agent provisional rejection control request failed")
        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - control data must be exact
            raise RuntimeError("remote Agent provisional rejection response is not JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("rejected"), bool):
            raise RuntimeError("remote Agent provisional rejection response is malformed")
        return bool(payload["rejected"])

    def flush(
        self,
        *,
        audited_read_msg_ids: frozenset[str] = frozenset(),
    ) -> None:
        """Tell each agent server to forced-flush turns still suspended at
        teardown; sidecar the returned records (cross-process equivalent of
        ``Runtime.flush_suspended``). Best-effort: a server with no ``/flush``
        route (the World) returns 404 and is skipped; one client mapped under
        several addresses is flushed once."""
        seen: set[int] = set()
        for client in self._clients.values():
            if id(client) in seen:
                continue
            seen.add(id(client))
            try:
                resp = client.post(
                    "/flush",
                    json={"audited_pending_read_msg_ids": sorted(audited_read_msg_ids)},
                )
            except Exception:  # noqa: BLE001 — teardown is best-effort
                continue
            if resp.status_code != 200:
                continue
            try:
                records = resp.json()
            except Exception:  # noqa: BLE001
                continue
            for rec in records or []:
                if not isinstance(rec, dict):
                    raise RuntimeError("remote forced-flush trace is not an object")
                _validate_remote_forced_trace(rec)
                if self._trace_sink is not None:
                    self._trace_sink(rec)

    def _client_for(self, address: str) -> "httpx.Client":
        if address in self._clients:
            return self._clients[address]
        client = self._clients.get(address.split(":", 1)[0])
        if client is None:
            raise AgentNotRegistered(address)
        return client

    def deliver(
        self, env: Envelope, *, turn: int, frame: "TurnFrame | None" = None
    ) -> "Envelope | list[Envelope] | TurnSuspended | None":
        del turn, frame  # remote server tracks its own turn; R1 resume = 1b work
        # Keep HTTP and in-process topology identical at the external principal
        # boundary.  There is deliberately no /vcp process for a human owner.
        if (
            env.to == "consumer"
            or env.to.startswith("consumer:")
            or env.to == "runtime"
            or env.to.startswith("runtime:")
        ):
            return None
        resp = self._client_for(env.to).post(
            "/vcp",
            content=to_json(env),
            headers={"content-type": "application/json"},
        )
        resp.raise_for_status()
        body = resp.text
        if not body or body == "null":
            return None
        data: Any = json.loads(body)
        # Agent /vcp wraps its result: {"reply": <envelope|null>, "trace": <rec|null>}.
        # Split the trace to the sidecar and route the reply. A bare envelope
        # (World, or an unwrapped reply) has an "action" key and falls through.
        if isinstance(data, dict) and "reply" in data and "action" not in data:
            trace = data.get("trace")
            reply = data.get("reply")
            result: Envelope | list[Envelope] | None
            if reply is None:
                result = None
            elif isinstance(reply, list):
                result = [_envelope_from_dict(i) for i in reply if i is not None]
            else:
                result = _envelope_from_dict(reply)

            remote_error = data.get("error")
            if remote_error is not None:
                if result is not None:
                    raise RuntimeError("remote Agent failure cannot also return an envelope")
                if not isinstance(trace, dict):
                    raise RuntimeError("remote scoreable Agent failure has no Tracker row")
                failure = remote_failure_from_wire(
                    remote_error,
                    trace=trace,
                    delivered_to=env.to,
                    delivered_inbound_msg_id=env.msg_id,
                    delivered_kind=str(env.action.get("kind", "")),
                )
                self._write_required_remote_failure_trace(trace, failure)
                raise failure

            try:
                self._validate_remote_agent_result(result, agent_id=env.to)
            except Exception as exc:
                terminal = tracker_terminal_for_exception(exc, current=None)
                if terminal is None:
                    self._reject_remote_terminal_candidate(result)
                    raise
                if not isinstance(trace, dict):
                    failed_trace = self._reject_remote_suspended_read(
                        result,
                        delivered_agent_id=env.to,
                        delivered_inbound_msg_id=env.msg_id,
                        delivered_kind=str(env.action.get("kind", "")),
                        terminal=terminal,
                        exc=exc,
                    )
                    self._write_required_remote_failure_trace(failed_trace, exc)
                    raise
                self._reject_remote_terminal_candidate(result)
                _validate_remote_success_trace(
                    trace,
                    agent_id=env.to,
                    result=result,
                    delivered_inbound_msg_id=env.msg_id,
                    delivered_kind=str(env.action.get("kind", "")),
                )
                failed_trace = failure_trace_dict(trace, terminal=terminal)
                self._write_required_remote_failure_trace(failed_trace, exc)
                raise

            if trace is not None:
                try:
                    if not isinstance(trace, dict):
                        raise RuntimeError("remote Agent trace is not an object")
                    _validate_remote_success_trace(
                        trace,
                        agent_id=env.to,
                        result=result,
                        delivered_inbound_msg_id=env.msg_id,
                        delivered_kind=str(env.action.get("kind", "")),
                    )
                    if self._trace_sink is not None:
                        self._trace_sink(trace)
                except Exception:
                    self._reject_remote_terminal_candidate(result)
                    raise
            return result
        if isinstance(data, list):
            return [_envelope_from_dict(item) for item in data if item is not None]
        return from_json(body)

    def _validate_remote_agent_result(
        self,
        result: Envelope | list[Envelope] | None,
        *,
        agent_id: str,
    ) -> None:
        if self._output_validator is None or result is None:
            return
        values = result if isinstance(result, list) else [result]
        for value in values:
            self._output_validator(value, agent_id)

    def _reject_remote_terminal_candidate(
        self,
        result: "Envelope | list[Envelope] | None",
    ) -> None:
        """Best-effort exact cleanup after remote terminal post-processing."""

        if result is None:
            return
        values = result if isinstance(result, list) else [result]
        for value in values:
            if (
                value.to == "world"
                and str(value.action.get("kind", "")) in PUBLIC_WORLD_READ_ACTION_KINDS
            ):
                continue
            self.reject_provisional_outbound(value)

    def _write_required_remote_failure_trace(
        self,
        trace: dict[str, Any],
        exc: Exception,
    ) -> None:
        if self._trace_sink is None:
            raise RuntimeError("scoreable remote Agent failure has no Tracker sink") from exc
        self._trace_sink(trace)
        turn = trace.get("turn")
        agent_id = trace.get("agent_id")
        inbound_msg_id = trace.get("inbound_msg_id")
        decision_id = trace.get("decision_id")
        terminal = trace.get("terminal")
        if (
            isinstance(turn, bool)
            or not isinstance(turn, int)
            or not all(
                isinstance(value, str) and value
                for value in (agent_id, inbound_msg_id, decision_id, terminal)
            )
        ):
            raise RuntimeError("remote failure Tracker identity is malformed") from exc
        attach_trace_identity(
            exc,
            turn=turn,
            agent_id=agent_id,
            inbound_msg_id=inbound_msg_id,
            decision_id=decision_id,
            terminal=terminal,
        )

    def _reject_remote_suspended_read(
        self,
        result: "Envelope | list[Envelope] | None",
        *,
        delivered_agent_id: str,
        delivered_inbound_msg_id: str,
        delivered_kind: str,
        terminal: str,
        exc: Exception,
    ) -> dict[str, Any]:
        """Ask the Agent process to close one rejected provisional World read."""

        from agents.turn import PUBLIC_WORLD_READ_ACTION_KINDS

        if (
            not isinstance(result, Envelope)
            or result.from_ != delivered_agent_id
            or result.to != "world"
            or str(result.action.get("kind", "")) not in PUBLIC_WORLD_READ_ACTION_KINDS
        ):
            raise RuntimeError("rejected remote Agent output has no terminal Tracker row") from exc
        response = self._client_for(delivered_agent_id).post(
            "/reject-suspended",
            json={"read_msg_id": result.msg_id, "terminal": terminal},
        )
        if response.status_code != 200:
            raise RuntimeError("remote Agent could not finalize rejected suspended read") from exc
        try:
            trace = response.json()
        except Exception as decode_exc:  # noqa: BLE001 - fail closed on control data
            raise RuntimeError("remote rejected suspended Tracker row is not JSON") from decode_exc
        _validate_remote_rejected_suspension_trace(
            trace,
            agent_id=delivered_agent_id,
            delivered_inbound_msg_id=delivered_inbound_msg_id,
            delivered_kind=delivered_kind,
            terminal=terminal,
        )
        return trace


def _envelope_from_dict(payload: "dict[str, Any]") -> Envelope:
    """Rebuild an Envelope from a parsed ``/vcp`` reply dict (``from`` → ``from_``)."""
    data = dict(payload)
    if "from" in data:
        data["from_"] = data.pop("from")
    return Envelope(**data)


def _validate_remote_success_trace(
    trace: dict[str, Any],
    *,
    agent_id: str,
    result: Envelope | list[Envelope] | None,
    delivered_inbound_msg_id: str,
    delivered_kind: str,
) -> None:
    """Validate the terminal remote trace before converting a rejected output.

    This is a narrow pre-transform guard.  The full Tracker verifier later
    joins the transformed row to the hash-covered Runtime audit and episode
    termination artifact.
    """

    from runtime.trace import normalized_decision_for_action_kind

    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError("Agent terminal trace cannot bind multiple envelopes")
        emitted = result[0]
    else:
        emitted = result
    turn = trace.get("turn")
    inbound_id = trace.get("inbound_msg_id")
    decision_id = trace.get("decision_id")
    if (
        isinstance(turn, bool)
        or not isinstance(turn, int)
        or turn < 0
        or trace.get("agent_id") != agent_id
        or not isinstance(inbound_id, str)
        or not inbound_id
        or not isinstance(decision_id, str)
        or not decision_id
        or trace.get("forced_flush") is not False
        or trace.get("incomplete") is not False
    ):
        raise RuntimeError("remote Agent terminal trace has invalid identity")
    if delivered_kind != "world.response":
        if inbound_id != delivered_inbound_msg_id:
            raise RuntimeError("remote Agent trace is bound to another inbound")
    elif not _trace_claims_world_response(trace, delivered_inbound_msg_id):
        raise RuntimeError("remote resumed trace does not claim the delivered World response")
    chosen = trace.get("chosen")
    if emitted is None:
        if (
            trace.get("terminal") != "no_reply"
            or trace.get("emitted_msg_id") is not None
            or not isinstance(chosen, dict)
            or chosen.get("decision") != "no_reply"
        ):
            raise RuntimeError("remote no_reply trace is contradictory")
        return
    action_kind = str(emitted.action.get("kind", ""))
    if (
        trace.get("terminal") != "emit_envelope"
        or trace.get("emitted_msg_id") != emitted.msg_id
        or emitted.from_ != agent_id
        or emitted.in_reply_to != inbound_id
        or not isinstance(chosen, dict)
        or chosen.get("decision") != normalized_decision_for_action_kind(action_kind)
    ):
        raise RuntimeError("remote emitted trace contradicts its Agent output")


def _trace_claims_world_response(trace: dict[str, Any], response_msg_id: str) -> bool:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("data"), dict):
            continue
        data = step["data"]
        if step.get("kind") == "tool_call" and isinstance(data.get("results"), list):
            if any(
                isinstance(result, dict) and result.get("source_msg_id") == response_msg_id
                for result in data["results"]
            ):
                return True
        if step.get("kind") == "framework_authority_prerequisite" and isinstance(
            data.get("source_msg_ids"), list
        ):
            if response_msg_id in data["source_msg_ids"]:
                return True
    return False


def _validate_remote_rejected_suspension_trace(
    trace: object,
    *,
    agent_id: str,
    delivered_inbound_msg_id: str,
    delivered_kind: str,
    terminal: str,
) -> None:
    """Fail closed on the control-plane row returned by `/reject-suspended`."""

    if not isinstance(trace, dict):
        raise RuntimeError("remote rejected suspended Tracker row is not an object")
    turn = trace.get("turn")
    inbound_id = trace.get("inbound_msg_id")
    decision_id = trace.get("decision_id")
    chosen = trace.get("chosen")
    if (
        isinstance(turn, bool)
        or not isinstance(turn, int)
        or turn < 0
        or trace.get("agent_id") != agent_id
        or not isinstance(inbound_id, str)
        or not inbound_id
        or not isinstance(decision_id, str)
        or not decision_id
        or trace.get("terminal") != terminal
        or trace.get("emitted_msg_id") is not None
        or trace.get("forced_flush") is not False
        or trace.get("incomplete") is not True
        or not isinstance(chosen, dict)
        or chosen
        != {
            "decision": terminal,
            "offer_id": None,
            "price": None,
            "rationale": None,
        }
    ):
        raise RuntimeError("remote rejected suspended Tracker row is contradictory")
    if delivered_kind != "world.response":
        if inbound_id != delivered_inbound_msg_id:
            raise RuntimeError("remote rejected suspended row binds another inbound")
    elif not _trace_claims_world_response(trace, delivered_inbound_msg_id):
        raise RuntimeError("remote rejected suspended row omits the delivered World response")
    steps = trace.get("steps")
    if not isinstance(steps, list) or any(
        isinstance(result, dict) and "pending_request_msg_id" in result
        for step in steps
        if isinstance(step, dict)
        and step.get("kind") == "tool_call"
        and isinstance(step.get("data"), dict)
        and isinstance(step["data"].get("results"), list)
        for result in step["data"]["results"]
    ):
        raise RuntimeError("remote rejected suspended row claims an unaudited read")


def _validate_remote_forced_trace(trace: dict[str, Any]) -> None:
    """Reject a forged/internally inconsistent trace returned by ``/flush``."""

    chosen = trace.get("chosen")
    turn = trace.get("turn")
    if (
        isinstance(turn, bool)
        or not isinstance(turn, int)
        or turn < 0
        or not isinstance(trace.get("agent_id"), str)
        or not trace.get("agent_id")
        or not isinstance(trace.get("inbound_msg_id"), str)
        or not trace.get("inbound_msg_id")
        or not isinstance(trace.get("decision_id"), str)
        or not trace.get("decision_id")
        or trace.get("terminal") != "forced_flush"
        or trace.get("emitted_msg_id") is not None
        or trace.get("forced_flush") is not True
        or trace.get("incomplete") is not True
        or not isinstance(chosen, dict)
        or chosen
        != {
            "decision": "no_reply",
            "offer_id": None,
            "price": None,
            "rationale": None,
        }
    ):
        raise RuntimeError("remote forced-flush trace has invalid terminal metadata")


def _project_inventory_reply(
    result: Envelope | list[Envelope] | None,
) -> Envelope | list[Envelope] | None:
    """Clone inventory replies into the legacy four-field VCP projection.

    ``WorldService`` intentionally keeps returning its typed ``InventoryRow``
    in direct in-process calls.  A Runtime delivery is a VCP transport boundary,
    so it must expose the same public shape as the HTTP adapter.
    """
    if result is None:
        return None
    if isinstance(result, list):
        return [
            projected
            for item in result
            if (projected := _project_inventory_envelope(item)) is not None
        ]
    return _project_inventory_envelope(result)


def _project_inventory_envelope(env: Envelope) -> Envelope:
    payload = env.action.get("payload")
    if payload is None:
        return env
    if isinstance(payload, dict):
        source = payload
    else:
        source = {
            field: getattr(payload, field)
            for field in (
                "sku_id",
                "merchant_id",
                "qty_available",
                "qty_reserved",
            )
            if hasattr(payload, field)
        }
    projected = {
        field: source[field]
        for field in (
            "sku_id",
            "merchant_id",
            "qty_available",
            "qty_reserved",
        )
        if field in source
    }
    return replace(env, action={**env.action, "payload": projected})


__all__ = ["Transport", "InProcessTransport", "HttpTransport"]
