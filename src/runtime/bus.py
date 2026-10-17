"""The in-process bus.

The runtime is the only thing that calls ``World.write``. Other code reaches
the world only through the read-only ``WorldTools`` view that the runtime
constructs per send.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Mapping

from agents.turn import (
    PUBLIC_WORLD_READ_ACTION_KINDS,
    TurnSuspended,
    finalize_forced_turn_trace,
)
from protocol.actor_terminal import validate_actor_terminal_action
from protocol.envelope import validate
from protocol.envelope import to_json
from runtime.errors import DuplicateAgentId, PrivateUtilityLeak
from runtime.evidence import (
    PLATFORM_RESPONSE_DISPOSITIONS_ARTIFACT,
    PlatformResponseDispositionJournal,
    PlatformResponseOccurrence,
    load_platform_decisions,
)
from runtime.trace import build_trace_record
from runtime.transport import InProcessTransport
from runtime.types import AuditRecord, SecurityEvent
from world.service import WorldService

if TYPE_CHECKING:
    from agents.base import Agent
    from agents.platform import PlatformService
    from agents.turn import TurnFrame
    from protocol.envelope import Envelope
    from runtime.actor_context import ActorContextResolver
    from runtime.actor_evidence import ActorEvidenceJournal
    from runtime.audit import AuditLog
    from runtime.router import Router
    from runtime.transport import Transport
    from runtime.control_services import RuntimeControlService
    from runtime.types import InferenceBudget
    from world.state import World


class Runtime:
    """Bus + registrar. Three responsibilities, one class.

    See WORLD_CLASSES.md §3 for the routing contract.
    """

    def __init__(
        self,
        *,
        world: "World",
        router: "Router",
        audit: "AuditLog",
        platform: "PlatformService | None" = None,
        world_service: WorldService | None = None,
        transport: "Transport | None" = None,
        max_quiescent_turns: int = 32,
        remote_world: bool = False,
        strict_tracker_capture: bool = False,
        actor_context_resolver: "ActorContextResolver | None" = None,
        actor_evidence_journal: "ActorEvidenceJournal | None" = None,
    ) -> None:
        """Wire the runtime to its world, router, and audit log.

        ``transport`` decides how a validated envelope reaches its target.
        It defaults to :class:`~runtime.transport.InProcessTransport` (direct
        method calls). Pass a :class:`~runtime.transport.HttpTransport` to run
        World / Platform / agents as separate ``/vcp`` processes while this
        Runtime keeps the single queue + audit trail.
        """
        if not isinstance(strict_tracker_capture, bool):
            raise TypeError("strict_tracker_capture must be boolean")
        self._world = world
        self._router = router
        self._audit = audit
        self._platform = platform
        self._world_service = world_service or WorldService(world)
        self._max_quiescent_turns = max_quiescent_turns
        if (actor_context_resolver is None) != (actor_evidence_journal is None):
            raise ValueError(
                "actor context resolver and evidence journal must be configured together"
            )
        self._actor_context_resolver = actor_context_resolver
        self._actor_evidence_journal = actor_evidence_journal
        self.agents: dict[str, Agent] = {}
        self._control_services: dict[str, RuntimeControlService] = {}
        self._queue: deque[Envelope] = deque()
        self._budgets: dict[str, InferenceBudget] = {}
        self._turn = 0
        self._draining_queue = False
        self._audited_world_read_ids: set[str] = set()
        self._strict_tracker_capture = strict_tracker_capture
        # A successful Agent trace may name an output that has passed Runtime's
        # pre-commit gates but is still waiting in the deterministic queue.  Do
        # not claim that output in Tracker until audit.jsonl commits the same
        # message.  If another turn aborts first, teardown converts the staged
        # row to a truthful forced non-emission instead of leaving an unaudited
        # emitted_msg_id in formal evidence.
        self._pending_agent_traces: dict[str, dict[str, Any]] = {}
        # The central dispatcher also retains the exact terminal envelope for
        # each staged Tracker row.  In-process Agents can recover it from their
        # provisional table, but an HTTP Agent lives behind the transport and
        # must receive the same canonical envelope for commit or teardown
        # rejection.
        self._pending_agent_outbounds: dict[str, Envelope] = {}
        decision_path = audit.path.with_name("platform.decisions.jsonl")
        self._platform_request_sequence = (
            len(load_platform_decisions(decision_path, strict=True))
            if decision_path.exists()
            else 0
        )
        self._platform_response_journal = PlatformResponseDispositionJournal(
            audit.path.with_name(PLATFORM_RESPONSE_DISPOSITIONS_ARTIFACT),
            run_id=audit.run_id,
        )
        self._pending_platform_responses: list[PlatformResponseOccurrence] = []
        # R1: parked turns awaiting a world.read_* reply, keyed by the read's
        # msg_id (== the world.response's in_reply_to). See runtime.transport.
        self._suspended: dict[str, "TurnFrame"] = {}
        self._transport: Transport = transport or InProcessTransport(
            world=world,
            world_service=self._world_service,
            platform=platform,
            agents=self.agents,
            trace_sink=self._stage_agent_trace,
            remote_world=remote_world,
            strict_tracker_capture=strict_tracker_capture,
        )
        configure_trace_sink = getattr(self._transport, "configure_trace_sink", None)
        if callable(configure_trace_sink):
            configure_trace_sink(self._stage_agent_trace)
        # Agent terminal outputs are checked by the SAME Router before their
        # Tracker row is committed.  This lets a rejected protocol/security
        # action become a truthful no-emit failure record.  Runtime.send still
        # performs the authoritative checks again immediately before audit;
        # this hook changes no policy and never appends the candidate envelope.
        configure_output_validator = getattr(
            self._transport, "configure_agent_output_validator", None
        )
        if callable(configure_output_validator):
            configure_output_validator(self._prevalidate_agent_output)

    def register(self, agent: "Agent", *, budget: "InferenceBudget | None" = None) -> None:
        """Validate and add ``agent`` to the registry. The only path to exposure.

        Validates: id uniqueness, well-formedness (``<side>[:<role>]``), and
        outbound-action partition (router checks the agent's declared kinds at
        register time, not just at send).

        Raises:
            DuplicateAgentId: agent.id collides with an already-registered agent.
            PartitionViolation: agent's outbound kinds violate the allow-table.
        """
        if agent.id in self.agents:
            raise DuplicateAgentId(agent.id)
        self._router.validate_register(
            agent.id,
            agent.outbound_action_kinds,
            agent.inbound_action_kinds,
        )
        self.agents[agent.id] = agent
        if budget is not None:
            self._budgets[agent.id] = budget

    def unregister(self, agent_id: str) -> None:
        """Reverse of register. Persist agent's memory if E1 mode."""
        self.agents.pop(agent_id, None)
        self._budgets.pop(agent_id, None)

    def register_control_service(self, service: "RuntimeControlService") -> None:
        """Register one deterministic runtime:* service outside agent population."""

        service_id = service.service_id
        if (
            not service_id.startswith("runtime:")
            or service_id in self._control_services
            or service_id in self.agents
        ):
            raise DuplicateAgentId(service_id)
        self._control_services[service_id] = service

    def send(self, env: "Envelope") -> "Envelope | list[Envelope] | None":
        """The bus call.

        Path: validate envelope -> router.check_outbound -> router.check_payload
        -> audit.append -> transport.deliver (world / platform / agent, in-process
        or over HTTP) -> enqueue any non-null reply so ``run()`` can deliver it next.

        Raises:
            AgentNotRegistered: ``env.to`` has no registered agent (or
                ``platform:*`` without a registered ``PlatformService``).
        """
        try:
            validate(env)
            self._router.check_outbound(env, env.from_)
        except Exception:
            self._reject_provisional_agent_output(env)
            raise
        # Privacy gate: a blocked leak is recorded to the SANITIZED security
        # sidecar and re-raised; the rejected envelope is NEVER appended to
        # audit.jsonl (it did not reach the wire) and is NOT delivered.
        try:
            self._router.check_payload(env, env.from_)
        except PrivateUtilityLeak as leak:
            self._record_security_event(env, leak)
            self._reject_provisional_agent_output(env)
            raise
        try:
            self._audit.append(
                AuditRecord(envelope=env, role=env.from_.split(":", 1)[0])
            )
        except Exception:
            self._reject_provisional_agent_output(env)
            raise
        self._commit_provisional_agent_output(env)
        self._commit_staged_agent_trace(env)
        if _is_platform_address(env.from_):
            self._record_audited_platform_response(env)
        if (
            env.to == "world"
            and str(env.action.get("kind", "")) in PUBLIC_WORLD_READ_ACTION_KINDS
        ):
            self._audited_world_read_ids.add(env.msg_id)

        # Actor-authored results terminate at the central Runtime. They are not
        # Platform decisions and never cross an HTTP delivery hop. Runtime
        # observes every audited envelope in order so the report binding is
        # reconstructed from the real causal chain, not report payload fields.
        binding = None
        if self._actor_context_resolver is not None:
            binding = self._actor_context_resolver.ingest(env)
        kind = str(env.action.get("kind", ""))
        if kind in {
            "delegate.report_result",
            "commerce.submit_decision_record",
        }:
            from runtime.errors import ActorEvidenceUnavailable

            if (
                env.to != "runtime:evidence"
                or self._actor_evidence_journal is None
                or binding is None
            ):
                raise ActorEvidenceUnavailable(
                    "actor result has no trusted Runtime evidence acceptance path"
                )
            self._actor_evidence_journal.append(request=env, binding=binding)
            return None

        # R1: a world.response routes back into the suspended turn it answers
        # (matched by in_reply_to == the read's msg_id), resuming that turn
        # rather than starting a fresh one.
        frame = None
        if str(env.action.get("kind", "")) == "world.response":
            frame = self._suspended.pop(env.in_reply_to or "", None)

        control_service = self._control_services.get(env.to)
        platform_decision_sequence: int | None = None
        if _is_platform_address(env.to):
            platform_decision_sequence = self._platform_request_sequence
            self._platform_request_sequence += 1
        if control_service is not None:
            if kind not in control_service.inbound_action_kinds:
                from protocol.errors import SchemaError

                raise SchemaError(
                    f"Runtime control service {env.to!r} rejects action {kind!r}"
                )
            result = control_service.handle(env)
        else:
            if kind == "platform.remediation_audit_request":
                from runtime.errors import AgentNotRegistered

                raise AgentNotRegistered(env.to)
            result = self._transport.deliver(env, turn=self._turn, frame=frame)

        # R1: the turn emitted a world read mid-flight. Park the frame under the
        # read's msg_id and enqueue the read so it is audited + routed like any
        # envelope — the single authoritative trail covers every grounding hop.
        if isinstance(result, TurnSuspended):
            self._suspended[result.read_env.msg_id] = result.frame
            self._queue.append(result.read_env)
            return None

        # Enqueue any non-null reply — services and agents both feed the bus.
        # A terminal Agent output is the completion of the decision currently
        # being tracked.  Put it at the head so Runtime audits that output
        # before opening another actor turn already waiting in the queue.  This
        # prevents a later security/capability guard from stranding a prior
        # successful Tracker row with an unaudited emitted_msg_id.  Platform
        # and World responses retain the existing deterministic FIFO order.
        actor_terminal_result = self._draining_queue and _is_agent_address(env.to)
        if isinstance(result, list):
            responses = [reply for reply in result if reply is not None]
            for response_index, reply in enumerate(responses):
                self._remember_pending_agent_outbound(reply)
                self._record_enqueued_platform_response(
                    request=env,
                    response=reply,
                    decision_sequence=platform_decision_sequence,
                    response_index=response_index,
                )
            if actor_terminal_result:
                self._queue.extendleft(reversed(responses))
            else:
                self._queue.extend(responses)
        elif result is not None:
            self._remember_pending_agent_outbound(result)
            self._record_enqueued_platform_response(
                request=env,
                response=result,
                decision_sequence=platform_decision_sequence,
                response_index=0,
            )
            if actor_terminal_result:
                self._queue.appendleft(result)
            else:
                self._queue.append(result)
        return result

    def _remember_pending_agent_outbound(self, env: "Envelope") -> None:
        """Retain an exact Agent terminal already bound to a staged trace."""

        if env.msg_id not in self._pending_agent_traces:
            return
        if env.msg_id in self._pending_agent_outbounds:
            raise RuntimeError("pending Agent outbound message id is ambiguous")
        self._pending_agent_outbounds[env.msg_id] = env

    def _record_enqueued_platform_response(
        self,
        *,
        request: "Envelope",
        response: "Envelope",
        decision_sequence: int | None,
        response_index: int,
    ) -> None:
        """Bind a produced Platform response to the Runtime queue occurrence."""

        if decision_sequence is None:
            return
        occurrence = self._platform_response_journal.append_enqueued(
            decision_sequence=decision_sequence,
            request_msg_id=request.msg_id,
            response_index=response_index,
            response=response,
        )
        self._pending_platform_responses.append(occurrence)

    def _record_audited_platform_response(self, env: "Envelope") -> None:
        """Terminally mark one exact queued response after audit append.

        The mark is intentionally written only after ``audit.jsonl`` commits
        the envelope.  A response still in the queue therefore cannot acquire
        the audited state.
        """

        digest = hashlib.sha256(to_json(env).encode("utf-8")).hexdigest()
        for index, occurrence in enumerate(self._pending_platform_responses):
            if (
                occurrence.response_msg_id == env.msg_id
                and occurrence.response_envelope_sha256 == digest
            ):
                self._platform_response_journal.append_terminal(
                    occurrence,
                    state="audited",
                )
                del self._pending_platform_responses[index]
                return

    def _record_security_event(self, env: "Envelope", leak: PrivateUtilityLeak) -> None:
        """Write a sanitized :class:`~runtime.types.SecurityEvent` for a blocked
        leak. No-op if the audit log has no security sidecar (older logs)."""
        append = getattr(self._audit, "append_security", None)
        if append is None:
            return
        from runtime.privacy import sanitize_field_path_for_audit

        finding = getattr(leak, "finding", None)
        kind = str(env.action.get("kind", "")) if isinstance(env.action, dict) else ""
        append(SecurityEvent(
            msg_id=env.msg_id,
            sender_id=env.from_,
            action_kind=kind,
            secret_owner=str(getattr(finding, "secret_owner", "") or ""),
            secret_name=str(getattr(finding, "secret_name", "") or ""),
            field_path=sanitize_field_path_for_audit(
                str(getattr(finding, "field_path", "") or "")
            ),
            reason=str(getattr(finding, "reason", "") or "private_utility_leak"),
            recipient_id=self._sanitized_security_recipient(env.to),
            owner_role=str(getattr(finding, "owner_role", "") or ""),
            ts=env.ts,
        ))

    def _stage_agent_trace(self, record: object) -> None:
        """Commit no-output/failure rows now; stage emitted rows until audit.

        The Transport owns Agent execution, but Runtime owns the wire audit.
        Buffering at this boundary makes a Tracker ``emitted_msg_id`` a claim
        about an audited envelope rather than merely a model-produced value.
        """

        if is_dataclass(record) and not isinstance(record, type):
            data = asdict(record)
        elif isinstance(record, Mapping):
            data = dict(record)
        else:
            raise RuntimeError("Agent Tracker row is not serializable")
        terminal = data.get("terminal")
        emitted_msg_id = data.get("emitted_msg_id")
        if terminal != "emit_envelope":
            self._audit.append_trace_dict(data)
            return
        if not isinstance(emitted_msg_id, str) or not emitted_msg_id:
            raise RuntimeError("emitted Agent Tracker row has no message id")
        if emitted_msg_id in self._pending_agent_traces:
            raise RuntimeError("Agent Tracker emitted message id is ambiguous")
        self._pending_agent_traces[emitted_msg_id] = data

    def _commit_staged_agent_trace(self, env: "Envelope") -> None:
        data = self._pending_agent_traces.get(env.msg_id)
        if data is None:
            return
        if data.get("agent_id") != env.from_ or data.get("emitted_msg_id") != env.msg_id:
            raise RuntimeError("staged Agent Tracker row contradicts audited envelope")
        pending = self._pending_agent_outbounds.get(env.msg_id)
        if pending is not None and to_json(pending) != to_json(env):
            raise RuntimeError("staged Agent outbound differs from audited envelope")
        del self._pending_agent_traces[env.msg_id]
        self._pending_agent_outbounds.pop(env.msg_id, None)
        self._audit.append_trace_dict(data)

    def _flush_pending_agent_traces(self) -> None:
        """Close validated but never-audited Agent outputs without inventing wire effects."""

        for emitted_msg_id, data in self._pending_agent_traces.items():
            agent = self.agents.get(str(data.get("agent_id", "")))
            reject = getattr(agent, "reject_provisional_outbound", None)
            provisional_envelope = getattr(
                agent,
                "provisional_outbound_envelope",
                None,
            )
            if callable(reject) and callable(provisional_envelope):
                exact_outbound = provisional_envelope(emitted_msg_id)
                if exact_outbound is not None:
                    reject(exact_outbound)
            exact_outbound = self._pending_agent_outbounds.get(emitted_msg_id)
            if exact_outbound is not None and agent is None:
                remote_reject = getattr(
                    self._transport,
                    "reject_provisional_outbound",
                    None,
                )
                if callable(remote_reject):
                    remote_reject(exact_outbound)
            forced = dict(data)
            forced.update(
                {
                    "emitted_msg_id": None,
                    "terminal": "forced_flush",
                    "chosen": {
                        "decision": "no_reply",
                        "offer_id": None,
                        "price": None,
                        "rationale": None,
                    },
                    "forced_flush": True,
                    "incomplete": True,
                }
            )
            self._audit.append_trace_dict(forced)
        self._pending_agent_traces.clear()
        self._pending_agent_outbounds.clear()

    def _commit_provisional_agent_output(self, env: "Envelope") -> None:
        if env.msg_id not in self._pending_agent_traces:
            # Scenario kickoffs may legitimately be authored with an actor's
            # identity, but they were not emitted by the registered Agent in
            # this Runtime.  Only Transport-produced terminal outputs have a
            # staged Tracker row and provisional Agent state to commit.
            return
        if (
            env.to == "world"
            and str(env.action.get("kind", "")) in PUBLIC_WORLD_READ_ACTION_KINDS
        ):
            # Re-entrant native reads are framework-owned intermediate
            # envelopes, not terminal semantic outputs with staged Agent state.
            return
        agent = self.agents.get(env.from_)
        commit = getattr(agent, "commit_provisional_outbound", None)
        if callable(commit):
            commit(env)
            return
        remote_commit = getattr(
            self._transport,
            "commit_provisional_outbound",
            None,
        )
        if callable(remote_commit):
            remote_commit(env)

    def _reject_provisional_agent_output(self, env: "Envelope") -> None:
        agent = self.agents.get(env.from_)
        reject = getattr(agent, "reject_provisional_outbound", None)
        if callable(reject):
            reject(env)
            return
        remote_reject = getattr(
            self._transport,
            "reject_provisional_outbound",
            None,
        )
        if callable(remote_reject) and env.msg_id in self._pending_agent_traces:
            remote_reject(env)

    def _sanitized_security_recipient(self, recipient_id: str) -> str:
        """Keep known actor ids; digest any model-invented address suffix."""

        known = set(self.agents)
        transport_known = getattr(self._transport, "known_agent_addresses", None)
        if callable(transport_known):
            known.update(transport_known())
        if recipient_id in known:
            return recipient_id
        role = recipient_id.split(":", 1)[0]
        if role not in {
            "buyer",
            "merchant",
            "platform",
            "world",
            "consumer",
            "runtime",
        }:
            role = "unknown"
        digest = hashlib.sha256(recipient_id.encode("utf-8")).hexdigest()[:16]
        return f"{role}:<unregistered chars={len(recipient_id)} sha256={digest}>"

    def _prevalidate_agent_output(self, env: "Envelope", agent_id: str) -> None:
        """Run Runtime's existing schema/partition/privacy gates pre-trace.

        The rejected envelope is not appended to ``audit.jsonl``.  In
        particular, privacy failures still use the existing sanitized security
        sidecar path, preserving Router as the sole privacy-policy authority.
        """

        from protocol.errors import SchemaError

        if env.from_ != agent_id:
            raise SchemaError("Agent output sender differs from delivered actor")
        validate(env)
        self._router.check_outbound(env, agent_id)
        try:
            self._router.check_payload(env, agent_id)
        except PrivateUtilityLeak as leak:
            self._record_security_event(env, leak)
            raise
        # The Router checks action partitions and private-utility disclosure.
        # The actor-terminal contract then previews the canonical service route
        # and model-authored payload while the Agent turn is still open.  It is
        # intentionally pure: Platform market rules and World state validation
        # still happen after Runtime commits the accepted envelope.
        # Re-entrant World reads are framework-owned provisional outputs, not
        # model terminal actions.  They have their own partition, privacy, and
        # payload contract and must remain eligible for turn suspension.
        if not (
            env.to == "world"
            and str(env.action.get("kind", ""))
            in PUBLIC_WORLD_READ_ACTION_KINDS
        ):
            validate_actor_terminal_action(env)
        # Preserve privacy-guard precedence when one model output is both a
        # leak and addressed to an unreachable endpoint.  The rejected payload
        # must still acquire its sanitized security sidecar; destination
        # validation remains pre-trace and pre-wire immediately afterwards.
        self._prevalidate_agent_destination(env)
        kind = str(env.action.get("kind", ""))
        if kind in {
            "delegate.report_result",
            "commerce.submit_decision_record",
        }:
            from runtime.errors import ActorEvidenceUnavailable

            if self._actor_context_resolver is None:
                raise ActorEvidenceUnavailable(
                    "actor result has no trusted Runtime evidence acceptance path"
                )
            # This is a pure preview over the already-audited causal graph.
            # Invalid model-authored reports fail inside the Agent turn, before
            # audit/Tracker can claim a successful emission. Runtime.send still
            # performs the authoritative ingest and journal append afterwards.
            self._actor_context_resolver.validate_report_candidate(env)

    def _prevalidate_agent_destination(self, env: "Envelope") -> None:
        """Reject unreachable or privileged endpoint identities before trace commit."""

        from runtime.errors import AgentNotRegistered

        kind = str(env.action.get("kind", ""))
        if kind in {
            "delegate.report_result",
            "commerce.submit_decision_record",
        } and env.to != "runtime:evidence":
            raise AgentNotRegistered("actor result must target runtime:evidence")
        recipient_role = env.to.split(":", 1)[0]
        if recipient_role == "world" and env.to != "world":
            raise AgentNotRegistered(
                "Agent output targets a non-canonical World endpoint"
            )
        if env.to == "platform":
            raise AgentNotRegistered(
                "Agent output must target a typed platform service endpoint"
            )
        if recipient_role not in {"buyer", "merchant"}:
            return
        known = set(self.agents)
        transport_known = getattr(self._transport, "known_agent_addresses", None)
        if callable(transport_known):
            known.update(transport_known())
        if env.to not in known:
            raise AgentNotRegistered(
                "Agent output targets an unregistered participant identity"
            )

    def run(self) -> None:
        """Drain the queue.

        Termination: queue empty for ``max_quiescent_turns`` consecutive turns,
        OR a terminal action (``settle`` / ``rule`` / refusal) is observed.
        """
        quiescent_turns = 0
        self._draining_queue = True
        try:
            while quiescent_turns < self._max_quiescent_turns:
                if not self._queue:
                    quiescent_turns += 1
                    continue
                quiescent_turns = 0
                self._turn += 1
                env = self._queue.popleft()
                self.send(env)
        finally:
            self._draining_queue = False

    def flush_suspended(self) -> None:
        """Emit a forced-flush trace record for every turn still parked on an
        unanswered world read (e.g. at episode teardown).

        Marks the record ``forced_flush``/``incomplete`` with ``terminal=
        "forced_flush"`` so a scorer never reads a half-finished decision as a
        completed ``no_reply`` ("considered N candidates then chose not to buy").
        Over HTTP each independent local Agent service owns its suspended
        frames and provisional terminal state.  The transport's control-plane
        flush closes those frames and discards any terminal candidate that the
        dispatcher never audited, preserving the same authority boundary as
        the in-process topology."""
        for frame in self._suspended.values():
            agent = self.agents.get(frame.original_inbound.to)
            if frame.trace is None or agent is None:
                continue
            finalize_forced_turn_trace(
                frame,
                audited_read_msg_ids=self._audited_world_read_ids,
            )
            self._audit.append_trace(build_trace_record(
                run_id="",
                turn=self._turn,
                agent_id=frame.original_inbound.to,
                inbound_msg_id=frame.original_inbound.msg_id,
                recorder=frame.trace,
                memory=agent.memory,
                forced_flush=True,
                incomplete=True,
                strict_memory_capture=self._strict_tracker_capture,
            ))
        self._suspended.clear()
        # Over HTTP the suspended frames live in the agent servers, not here;
        # ask the transport to flush them too (no-op for InProcessTransport).
        transport_flush = getattr(self._transport, "flush", None)
        if callable(transport_flush):
            transport_flush(
                audited_read_msg_ids=frozenset(self._audited_world_read_ids)
            )
        self._flush_pending_agent_traces()
        # A scoreable Agent abort can leave Platform responses behind the
        # failing envelope in Runtime's queue.  Preserve that fact without
        # appending those responses to the on-wire audit: their lifecycle ends
        # in the append-only disposition stream as not audited at shutdown.
        for occurrence in self._pending_platform_responses:
            self._platform_response_journal.append_terminal(
                occurrence,
                state="not_audited_at_shutdown",
            )
        self._pending_platform_responses.clear()


def _is_platform_address(value: str) -> bool:
    return value == "platform" or value.startswith("platform:")


def _is_agent_address(value: str) -> bool:
    role = value.split(":", 1)[0]
    return role in {"buyer", "merchant"}
