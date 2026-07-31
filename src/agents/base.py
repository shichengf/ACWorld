"""The typed business-decision ``Agent`` used by CommerceWorld.

Per AGENT_CLASSES.md §1, ``Agent`` is one of the primitives. It is a concrete
class (§11 Q1); variance is in the enabled skill manifests, skill bundles,
model config, and scenario inputs — never in subclasses.

Each model response is a provider-neutral business intent.  The Agent owns
phase authority, World reads, internal identifiers, routes, envelopes,
idempotency, correlation, retry, and deterministic protocol continuation.
The business loop is capped by :data:`MAX_INTERNAL_STEPS`; exceeding it raises
:class:`BudgetExceeded`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from agents.errors import (
    BudgetExceeded,
    BudgetReconcileViolation,
    DecisionRecordIncomplete,
    GroundingRequired,
    SkillNotActivated,
)
from agents.agent_phase import (
    AgentDecisionBridge,
    AgentPhaseContract,
    CompiledBusinessAction,
    CompiledPhaseDecision,
    CompiledWorldRead,
    LocalControlDecision,
)
from agents.business_decision import (
    BusinessDecisionContractError,
    ModelBusinessChoiceV1,
    business_repair_error_v1,
)
from agents.commerce_grounding import (
    AuthorityPrerequisiteResolver,
    ResolvedAuthorityPrerequisites,
)
from agents.inference import (
    BusinessDecisionResponseV1,
    ChannelTransportError,
    InvalidAgentEnvelope,
    ModelDecisionParseError,
)
from agents.business_prompt import BUSINESS_DECISION_SYSTEM_PROMPT_V1
from agents.public_task_execution import (
    phase_actor_context,
    public_task_execution_contract_digest,
    resolve_public_task_phase,
    validate_public_task_execution_contract,
)
from agents.negotiation_turn_authority import NegotiationTurnAuthority
from agents.governance_turn_authority import GovernanceProjectionCache
from agents.after_sales_turn_authority import AFTER_SALES_ROUTE_OPERATIONS
from agents.agent_routes import (
    AFTER_SALES_ORDER_BOUND_OPERATIONS,
    DEFAULT_AGENT_ROUTE_REGISTRY,
    AfterSalesOrderAuthority,
    SemanticAction,
    SemanticDecisionError,
    canonical_semantic_digest,
    resolve_semantic_destination,
    semantic_action_observation_class,
)
from agents.business_phase_adapter import (
    CompositePhaseAdapter,
    AgentRoutePhaseAdapter,
    public_actor_task_facts,
    public_business_observations,
)
from agents.domain_phase_adapters import (
    AfterSalesPhaseAdapter,
    BoundDomainPhase,
    GovernancePhaseAdapter,
    NegotiationPhaseAdapter,
    ProtocolEventPhaseAdapter,
    RankedOfferPhaseAdapter,
    SupplyPhaseAdapter,
    SupplyReportPhaseAdapter,
)
from agents.decision_errors import ModelBusinessDecisionError, PlatformContractError
from agents.turn import (
    PUBLIC_WORLD_READ_ACTION_KINDS,
    ToolCall,
    TurnFrame,
    TurnSuspended,
    decision_id_for,
    finalize_rejected_tool_batch,
    reentrant_world_tool_allowed,
    tool_call_to_read_envelope,
    validate_reentrant_read_partition,
    world_response_to_tool_result,
)
from agents.types import ALLOWED_WORLD_TOOLS, MemoryType
from protocol.envelope import Envelope, from_json, to_json
from protocol.cart_quote_state import persistent_cart_quote_from_json
from protocol.negotiation_state import (
    NEGOTIATION_ACTIONS,
    negotiated_order_id,
)

# Every model-authored business-contract failure shares one model-visible,
# hashes-only repair opportunity across the whole Agent lifetime.  This covers
# malformed responses, invalid/duplicate World reads, and non-progressing
# Platform observations.  A second failure is terminal model capability
# evidence; transport retries are independent because no model decision was
# produced.  The Agent never parses prose into an action.
_MAX_BUSINESS_REPAIRS = 1

# Business decisions are intentionally one intent per provider response.  The
# largest frozen discovery task evaluates 20 public candidates with one
# listing and one stock observation each, so its legitimate grounded turn has
# 40 distinct reads before the terminal choice.  This is an Agent transport
# ceiling, not a model-quality criterion.
_MAX_BUSINESS_DISTINCT_READ_DECISIONS_PER_TURN = 48

# Stateless provider channels receive no conversation transcript. Retain a
# deliberately small, provider-neutral record of prior validated choices so a
# later request remains solvable from its current provider surface. The record
# never contains Agent-restored ids, routes, response bodies, authority, or
# integrity digests.
_MAX_PUBLIC_BUSINESS_CHOICE_HISTORY = 32

# A transport failure before a model decision has produced no Agent emission,
# Platform acceptance, or World commit. Retrying that exact decision is
# therefore safe. Each exact decision has at most three provider attempts.
# Retry state is local to that request and cannot carry a failed request into a
# later decision. The persistent formal call ledger remains the Episode-wide
# ceiling. The channel owns the sanitized taxonomy; unknown failures remain
# terminal and are never guessed retryable.
_RETRYABLE_BUSINESS_TRANSPORT_CODES = frozenset(
    {
        "network",
        "provider_timeout",
        "provider_429",
        "provider_5xx",
        # A complete HTTP exchange can still yield no usable Chat
        # Completions envelope (for example, an upstream error without a
        # status/type or a truncated JSON object).  No model decision, Agent
        # emission, or commerce commit exists at this point, so the same
        # bounded exact-decision retry is safe.
        "malformed_provider_response",
    }
)
_BUSINESS_TRANSPORT_RETRY_BACKOFF_SECONDS = (60.0, 300.0)
_MAX_BUSINESS_TRANSPORT_RETRIES_PER_DECISION = len(
    _BUSINESS_TRANSPORT_RETRY_BACKOFF_SECONDS
)

if TYPE_CHECKING:
    from agents.inference import InferenceChannel
    from agents.interfaces import Memory, SkillLoader
    from agents.types import AgentContext, AgentInputs, SkillManifest


#: Maximum internal steps per external turn before :class:`BudgetExceeded`.
#: Sized for the buyer's longest journey with Agent-owned World observations
#: and typed business decisions.  This is an Agent loop ceiling, not a score.
#: Fifty leaves headroom for bounded observation and repair cycles without
#: allowing an unbounded provider loop.
MAX_INTERNAL_STEPS: int = 50


def _safe_compiled_payload_projection(value: Any) -> Any:
    """Project compiled wire semantics without retaining any raw string.

    Tracker already retains the canonical payload digest.  This projection is
    deliberately more useful to integrity scorers while remaining safe for
    persistent evidence: numbers/booleans/container shape remain comparable,
    and every string (including internal ids and free text) becomes only a
    length plus SHA-256 commitment.
    """

    if isinstance(value, Mapping):
        return {str(name): _safe_compiled_payload_projection(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_compiled_payload_projection(item) for item in value]
    if isinstance(value, str):
        return {
            "text_chars": len(value),
            "text_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
    if value is None or isinstance(value, (bool, int, float)):
        return copy.deepcopy(value)
    raise RuntimeError("compiled business payload is not strict JSON")


def _agent_message_payload_authority(
    inbound: "Envelope",
    *,
    benchmark_contract: Mapping[str, Any] | None,
    reply_recipient_id: str | None,
) -> dict[str, Any] | None:
    """Compile deterministic participant-message content from current authority.

    Correlation IDs, Platform state echoes, and fixed lifecycle hand-off
    markers are Agent communication work and are never transcribed through
    the provider.
    """

    kind = str(inbound.action.get("kind", ""))
    raw_payload = inbound.action.get("payload")
    payload = raw_payload if isinstance(raw_payload, Mapping) else {}
    if kind == "platform.supply_state":
        states = payload.get("states")
        if isinstance(states, list) and all(isinstance(row, Mapping) for row in states):
            return {
                "category": "inventory_eta_report",
                "states": copy.deepcopy(states),
            }
        return None
    if kind == "platform.after_sales_updated":
        references = payload.get("references")
        operation = payload.get("operation")
        if not isinstance(references, Mapping) or not isinstance(operation, str):
            return None
        selected: tuple[str, str] | None = {
            "request_return": ("request_id", "request_id"),
            "submit_dispute_evidence": ("dispute_id", "dispute_id"),
            "open_refund_case": ("case_id", "case_id"),
            "request_exchange": ("case_id", "case_id"),
        }.get(operation)
        if selected is not None:
            output_name, source_name = selected
            value = references.get(source_name)
            if isinstance(value, str) and value.strip():
                return {output_name: value}
            return None
        if operation == "receive_return":
            return {"event": "return_received"}
        if operation == "respond_to_dispute":
            return {"event": "ruling_committed"}
        return None
    if kind == "commerce.respond_inquiry":
        sku_id = payload.get("sku_id")
        if isinstance(sku_id, str) and sku_id.strip():
            return {"category": "response_received", "sku_id": sku_id}
        return None
    if kind == "platform.aggregate_reviews":
        return {"review_setup_ready": True}
    if kind in {"platform.after_sales_snapshot", "platform.shipment_state"}:
        order_id: Any = None
        if kind == "platform.shipment_state":
            shipment = payload.get("shipment")
            order_id = shipment.get("order_id") if isinstance(shipment, Mapping) else None
        else:
            records = payload.get("records")
            values = {
                row.get("order_id")
                for row in records or ()
                if isinstance(row, Mapping)
                and isinstance(row.get("order_id"), str)
                and str(row["order_id"]).strip()
            }
            if len(values) == 1:
                order_id = next(iter(values))
        if isinstance(order_id, str) and order_id.strip():
            return {
                "instruction": "begin the authoritative after-sales workflow",
                "order_id": order_id,
            }
        return None
    if kind == "commerce.send_message" and payload.get("category") == "after_sales":
        order_id = payload.get("order_id")
        authorized = (
            benchmark_contract.get("order_ids") if isinstance(benchmark_contract, Mapping) else None
        )
        if not isinstance(authorized, (list, tuple)):
            fallback = (
                benchmark_contract.get("order_id")
                if isinstance(benchmark_contract, Mapping)
                else None
            )
            authorized = [fallback] if fallback is not None else []
        if isinstance(order_id, str) and order_id in authorized:
            return {
                "instruction": "begin the authoritative after-sales workflow",
                "order_id": order_id,
            }
        return None
    if kind in {"platform.governance_updated", "platform.governance_case_notice"}:
        task_id = (
            benchmark_contract.get("task_id") if isinstance(benchmark_contract, Mapping) else None
        )
        if not isinstance(task_id, str) or not task_id.strip():
            return None
        instruction = (
            "publish campaign"
            if isinstance(reply_recipient_id, str) and reply_recipient_id.startswith("merchant:")
            else "inspect governance projection and select an offer"
        )
        return {"instruction": instruction, "task_id": task_id}
    return None


def _agent_reply_recipient_authority(
    inbound: "Envelope",
    *,
    benchmark_contract: Mapping[str, Any] | None,
    lineage_recipient_id: str | None,
) -> str | None:
    """Resolve a participant destination without asking the model for an ID.

    Ordinary replies follow the authenticated message lineage.  Governance
    campaigns can instead carry a spawn-frozen next participant in the Agent's
    private benchmark contract; this is needed for multi-party hand-offs where
    the next recipient is deliberately not the previous sender.
    """

    kind = str(inbound.action.get("kind", ""))
    if kind not in {"platform.governance_updated", "platform.governance_case_notice"}:
        return lineage_recipient_id
    if not isinstance(benchmark_contract, Mapping) or "recipient_id" not in benchmark_contract:
        return lineage_recipient_id
    recipient_id = benchmark_contract.get("recipient_id")
    if (
        not isinstance(recipient_id, str)
        or not recipient_id.strip()
        or recipient_id != recipient_id.strip()
        or recipient_id == inbound.to
        or recipient_id.split(":", 1)[0] not in {"buyer", "merchant"}
        or ":" not in recipient_id
        or not recipient_id.split(":", 1)[1]
    ):
        raise PlatformContractError("spawn-frozen participant recipient authority is malformed")
    return recipient_id


def _collect_business_content_ids(value: Any) -> tuple[str, ...]:
    """Return ordered, unique content identities from one participant message."""

    rows: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, Mapping):
            content_id = item.get("content_id")
            if isinstance(content_id, str) and content_id.strip() and content_id not in rows:
                rows.append(content_id)
            for child in item.values():
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    return tuple(rows)


def _collect_after_sales_order_ids(value: Any) -> tuple[str, ...]:
    """Collect finite business order references without interpreting a task family."""

    rows: list[str] = []

    def append(candidate: Any) -> None:
        if isinstance(candidate, str) and candidate.strip() and candidate not in rows:
            rows.append(candidate)

    def collect(item: Any) -> None:
        if isinstance(item, Mapping):
            append(item.get("order_id"))
            order_ids = item.get("order_ids")
            if isinstance(order_ids, (list, tuple)):
                for candidate in order_ids:
                    append(candidate)
            for child in item.values():
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    return tuple(rows)


def _is_after_sales_business_inbound(env: "Envelope") -> bool:
    kind = str(env.action.get("kind", ""))
    if kind in {
        "platform.after_sales_updated",
        "platform.after_sales_snapshot",
    }:
        return True
    payload = env.action.get("payload")
    if not isinstance(payload, Mapping):
        return False
    return bool(
        kind == "commerce.send_message"
        and (
            payload.get("category") == "after_sales"
            or any(field in payload for field in ("request_id", "case_id", "dispute_id"))
            or payload.get("event") in {"return_received", "ruling_committed"}
        )
    )


def _pending_pre_action_read_gate(
    gate: Any,
    history: list[dict[str, Any]],
) -> str | None:
    """Return the action kind hidden until exact successful reads complete."""

    if gate is None:
        return None
    if not isinstance(gate, Mapping):
        raise RuntimeError("pre-action read gate is malformed")
    action_kind = gate.get("action_kind")
    requirements = gate.get("requirements")
    if (
        not isinstance(action_kind, str)
        or not action_kind
        or not isinstance(requirements, list)
        or not requirements
    ):
        raise RuntimeError("pre-action read gate is malformed")

    completed: dict[str, list[tuple[Mapping[str, Any], Any]]] = {}
    for step in history:
        if (
            not isinstance(step, Mapping)
            or step.get("step") != "tool_call"
            or step.get("interface") != "business_decision"
        ):
            continue
        rows = step.get("results")
        if not isinstance(rows, list):
            raise RuntimeError("business World-read history is malformed")
        for row in rows:
            if not isinstance(row, Mapping):
                raise RuntimeError("business World-read result is malformed")
            tool = row.get("tool")
            args = row.get("args")
            result = row.get("result")
            if (
                isinstance(tool, str)
                and isinstance(args, Mapping)
                and isinstance(result, (Mapping, list, tuple))
            ):
                completed.setdefault(tool, []).append((args, result))

    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            raise RuntimeError("pre-action World-read requirement is malformed")
        tool = requirement.get("tool")
        argument_field = requirement.get("argument_field")
        required_values = requirement.get("required_values")
        minimum_successes = requirement.get("minimum_successes")
        if (
            not isinstance(tool, str)
            or not isinstance(required_values, list)
            or isinstance(minimum_successes, bool)
            or not isinstance(minimum_successes, int)
        ):
            raise RuntimeError("pre-action World-read requirement is malformed")
        rows = completed.get(tool, [])
        if argument_field is None:
            unique_calls = {
                json.dumps(
                    dict(args),
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for args, _result in rows
            }
            if len(unique_calls) < minimum_successes:
                return action_kind
            continue
        if not isinstance(argument_field, str) or not argument_field:
            raise RuntimeError("pre-action World-read argument field is malformed")
        observed = {
            str(args[argument_field])
            for args, _result in rows
            if isinstance(args.get(argument_field), str)
        }
        if not set(required_values).issubset(observed):
            return action_kind
    return None


@dataclass(frozen=True, slots=True)
class _ProvisionalMemoryUpdate:
    """One temporary guard overlay and its eventual audited value."""

    bucket: MemoryType
    key: str
    existed: bool
    previous: Any
    proposed: Any


@dataclass(frozen=True, slots=True)
class _ProvisionalOutbound:
    """Agent state staged until Runtime audits one exact envelope."""

    envelope_json: str
    envelope_sha256: str
    memory_update: _ProvisionalMemoryUpdate | None
    principal_id: str | None
    reply_recipient_id: str | None
    actor_reports_allowed: bool
    observation_choice_fingerprint: str | None
    continuation_source: tuple[str, str] | None = None


@dataclass(frozen=True, slots=True)
class _CommittedSemanticOutbound:
    """Minimal audited intent retained for exact protocol continuations."""

    action_kind: str
    destination: str
    envelope_sha256: str
    payload_sha256: str
    order_id: str | None


@dataclass(frozen=True, slots=True)
class _BusinessPhaseDecision:
    """One provider-neutral choice compiled entirely inside the Agent.

    The provider response is retained only as hashes/length metadata for the
    repair and Tracker contracts. No CommerceWorld route or wire payload is
    reconstructed from provider-authored fields.
    """

    compiled: CompiledPhaseDecision
    intent: str
    arguments: Mapping[str, Any]
    message: str | None
    source_name: str
    call_id: str
    response_chars: int
    response_sha256: str
    model_choice: ModelBusinessChoiceV1
    agent_argument_binding: Mapping[str, Any]
    observation_scope_sha256: str


class Agent:
    """A repository-owned commerce participant. Concrete; not subclassed.

    An optional provider model supplies only typed business decisions through
    ``channel``; it is not registered with Runtime and is never a bus actor.

    Lifecycle:
        construct → ``Runtime.register`` → many ``receive`` calls →
        ``Runtime.unregister``.

    At spawn, the agent sees only the lightweight ``enabled_skills`` manifests
    (AGENT_CLASSES.md §1, §4.1). Full ``SkillDocument`` bodies are pulled
    through ``skill_loader.load(name)`` inside the turn that activates them.
    """

    id: str
    skill_loader: "SkillLoader"
    enabled_skills: "tuple[SkillManifest, ...]"
    memory: "Memory"
    channel: "InferenceChannel"
    inbound_action_kinds: frozenset[str]
    outbound_action_kinds: frozenset[str]

    def __init__(
        self,
        *,
        id: str,
        skill_loader: "SkillLoader",
        enabled_skills: "tuple[SkillManifest, ...]",
        memory: "Memory",
        inputs: "AgentInputs",
        channel: "InferenceChannel",
        inbound_action_kinds: frozenset[str],
        outbound_action_kinds: frozenset[str],
        inventory_view: "Any | None" = None,
        merchant_data: "Any | None" = None,
        selector: "Any | None" = None,
        actor_report_root_msg_ids: frozenset[str] = frozenset(),
        semantic_search_limit: int | None = None,
        semantic_root_principals: Mapping[str, str] | None = None,
        provider_retry_sleeper: Callable[[float], None] | None = None,
    ) -> None:
        """Build one typed business-decision Agent.

        Private mandate and policy inputs remain Agent-owned compiler state.
        The provider receives only the fixed business system prompt plus the
        per-phase :class:`LLMDecisionRequestV1` projection.
        """
        if not id or not isinstance(id, str):
            raise ValueError(f"agent id must be a non-empty string, got {id!r}")

        self.id = id
        self.skill_loader = skill_loader
        self.enabled_skills = enabled_skills
        self.memory = memory
        self.channel = channel
        if not bool(getattr(channel, "supports_business_decisions", False)) or not callable(
            getattr(channel, "complete_business_decision", None)
        ):
            raise TypeError("Agent requires the typed business-decision channel")
        if provider_retry_sleeper is not None and not callable(provider_retry_sleeper):
            raise TypeError("provider_retry_sleeper must be callable or null")
        # Dependency injection keeps the production policy on wall-clock time
        # while allowing scoped tests to record the exact schedule instantly.
        self._provider_retry_sleeper = (
            time.sleep if provider_retry_sleeper is None else provider_retry_sleeper
        )
        self.inbound_action_kinds = frozenset(inbound_action_kinds)
        self.outbound_action_kinds = frozenset(outbound_action_kinds)
        if any(
            not isinstance(root_msg_id, str) or not root_msg_id.strip()
            for root_msg_id in actor_report_root_msg_ids
        ):
            raise ValueError("actor_report_root_msg_ids must contain non-empty strings")
        # This authority is derived from the scenario before Runtime delivers
        # any kickoff and never changes afterward.  Only outbound message ids
        # emitted on an already-authorized turn are accumulated below, so an
        # unrelated Platform notification cannot mint report authority.
        self._actor_report_root_msg_ids = frozenset(actor_report_root_msg_ids)
        self._reportable_actor_outbound_msg_ids: set[str] = set()
        root_principals = dict(semantic_root_principals or {})
        if any(
            not isinstance(root_msg_id, str)
            or not root_msg_id.strip()
            or not isinstance(principal_id, str)
            or not principal_id.strip()
            for root_msg_id, principal_id in root_principals.items()
        ):
            raise ValueError(
                "semantic_root_principals must map non-empty message ids to non-empty actor ids"
            )
        if not set(root_principals).issubset(self._actor_report_root_msg_ids):
            raise ValueError("semantic_root_principals must be registered actor report roots")
        # A semantic action may cross Platform before the actor receives the
        # response that completes its original business conversation.  Keep
        # the principal and participant reply routes on the exact outbound
        # message lineage.  This is deterministic framework state. The prompt
        # may describe the inbound route, but the model never authors these
        # reply destinations in its function output.
        self._semantic_root_principals = root_principals
        self._semantic_principal_by_outbound_msg_id: dict[str, str] = {}
        self._semantic_reply_recipient_by_outbound_msg_id: dict[str, str] = {}
        # Optional per-agent cache invalidation hooks — both typed
        # loosely (Any) to avoid forward imports; the contract is a
        # callable ``on_envelope(env)``. Today only the merchant lane
        # uses these (ACWorld merchant-data authority design). Pass-through
        # agents leave them None; ``receive`` is a no-op in that case.
        # ``inventory_view`` invalidates the world-inventory cache on
        # settle / catalog acks; ``merchant_data`` refreshes the
        # merchant-side seed snapshots (e.g. ``seed_list_price``) from
        # catalog acks. Two parallel hooks rather than one combined hook
        # so the responsibilities stay separable — view is per-SKU
        # world reads, data is per-agent private bookkeeping.
        self._inventory_view = inventory_view
        self._merchant_data = merchant_data
        # B4 (ACWorld skill-selection design): optional per-agent skill
        # selector. When set, ``receive`` calls ``select(...)`` at the
        # head of each turn and constrains the LLM to load only the
        # activated skills. ``None`` selects every enabled skill as a
        # deterministic business-default set.
        self._selector = selector
        if semantic_search_limit is not None and (
            isinstance(semantic_search_limit, bool)
            or not isinstance(semantic_search_limit, int)
            or semantic_search_limit <= 0
        ):
            raise ValueError("semantic_search_limit must be a positive integer or null")
        # Scenario-built Agents receive population.matching.top_k here.  It is
        # framework authority, not a field for the model to guess.
        self._semantic_search_limit = semantic_search_limit
        self._business_repairs_used = 0
        # A semantic decision record is needed transiently by Agent guards.
        # Exact candidate state lives here, outside public Memory and route
        # authority, until Runtime appends that envelope to the audit.
        self._provisional_outbound_by_msg_id: dict[str, _ProvisionalOutbound] = {}
        self._committed_semantic_outbound_by_msg_id: dict[str, _CommittedSemanticOutbound] = {}
        self._committed_observation_choice_fingerprints: set[str] = set()
        # Successful Platform receipts advance this actor-private cursor.  It
        # binds sequential after-sales order references without asking the
        # provider to manufacture or repeat framework-known identifiers.
        self._reconciled_after_sales_order_ids: set[str] = set()
        self._processed_after_sales_receipts: dict[str, tuple[str, str, str, str]] = {}
        # Sealed World projections are private, per-inbound compiler inputs.
        # They never enter Memory or provider observations.  A fresh inbound
        # always replaces its slot, preventing stale authority reuse.
        self._after_sales_projections_by_inbound: dict[str, tuple[object, ...]] = {}
        self._protocol_event_authority_by_inbound: dict[str, Mapping[str, Any]] = {}
        self._inquiry_authority_by_request_id: dict[str, dict[str, Any]] = {}
        # Stateless providers still need actor-visible peer content in later
        # business decisions.  Retain only the sanitized public observation,
        # scoped to this Agent/Episode; never cache a classification or oracle
        # answer. Correlated merchant inquiries use the narrower request map
        # above instead of this general history.
        self._persistent_public_content_by_inbound: dict[str, dict[str, Any]] = {}
        self._public_business_choice_history: list[dict[str, Any]] = []
        # Dynamic governance identities are learned only from authenticated
        # Platform projections.  They remain Agent-private across the
        # plan-notice -> independent-evidence continuation and are never
        # copied into provider observations or memory.
        self._governance_projection_cache = GovernanceProjectionCache(actor_id=id)
        self._commerce_action_grounding = AuthorityPrerequisiteResolver(actor_id=id)
        # The Agent initializes private scenario authority deterministically instead of
        # asking the model to transcribe private scenario inputs into memory.
        # Keep isolated copies so neither prompt rendering nor a scenario
        # caller can mutate the Agent's authority inputs after construction.
        self._semantic_mandate = copy.deepcopy(inputs.mandate or {})
        self._semantic_policy = copy.deepcopy(inputs.policy or {})
        # item 3 grounding teeth: the buyer's FEATURE must_haves (from its
        # mandate). Accepting a purchase that carries these requires the chosen
        # sku to have been grounded — enforced in ``_build_outbound``. Empty for
        # the merchant (no mandate), so the teeth never fire there.
        _mandate = inputs.mandate or {}
        self._mandate_must_have: tuple[str, ...] = tuple(
            str(x) for x in ((_mandate.get("hard_constraints") or {}).get("must_have") or [])
        )

    # --- Receive: the bounded internal turn loop --------------------

    def receive(
        self, env: "Envelope", ctx: "AgentContext", *, frame: "TurnFrame | None" = None
    ) -> "Envelope | TurnSuspended | None":
        """Run one typed business turn and return zero or one Agent envelope."""
        # R1 resume: a suspended turn's world.read_* reply (env is the
        # world.response) came back. Skip the fresh-start hooks/selector — they
        # already ran when the turn opened; everything else lives in ``frame``.
        if frame is not None:
            return self._resume_reentrant_turn(env, ctx, frame)

        # B5: cache-invalidation hooks live on the Agent, not the Runtime —
        # the bus stays unaware of any merchant-side cache. Run as the
        # FIRST lines so any subsequent world read OR private-state read
        # this turn sees fresh state when an invalidating envelope
        # (settle / catalog_ack) just arrived.
        if self._inventory_view is not None:
            self._inventory_view.on_envelope(env)
        if self._merchant_data is not None:
            self._merchant_data.on_envelope(env)

        # A model chooses whether to accept an offer.  Once that audited choice
        # has produced an authoritative match certificate, payment submission
        # is protocol plumbing when the principal already granted purchase
        # authority.  Handle that exact continuation locally instead of asking
        # the model to transcribe a Platform-owned certificate identifier.
        # Observe governance projections before deterministic terminal
        # handling.  An active remediation-plan notice is intentionally a
        # no-inference wait turn, but it must still bind the exact private
        # plan/step/digest authority used by the later auditor evidence.
        # This observer never emits an action and therefore cannot mask a
        # legitimate participant ``send_message`` continuation.
        self._governance_projection_cache.observe(env)
        handled, continuation = self._framework_protocol_continuation(env, ctx)
        if handled:
            return continuation
        handled, continuation = self._framework_public_task_continuation(
            env,
            ctx,
        )
        if handled:
            return continuation
        if self._framework_public_task_terminal(env, ctx):
            return None

        # B4: skill selector runs after cache hooks, before the turn loop.
        # ``None`` means no selector is attached; the business loop then loads
        # every spawn-enabled skill deterministically.
        activated: tuple[str, ...] | None = None
        if self._selector is not None:
            activated = self._selector.select(
                env,
                memory=self.memory,
                merchant_data=self._merchant_data,
                inventory_view=self._inventory_view,
            )

        return self._receive_business(env, ctx, activated=activated)

    # --- Typed business decision loop -------------------------------

    @property
    def actor_report_root_msg_ids(self) -> frozenset[str]:
        """Return the spawn-frozen scenario roots for actor-result tools."""

        return self._actor_report_root_msg_ids

    def _actor_reports_allowed_for(self, inbound: "Envelope") -> bool:
        """Return whether ``inbound`` remains on a registered actor lineage.

        A fresh authority starts only at an exact actor-directed scenario root.
        Later turns inherit it only when the inbound directly replies to an
        outbound this Agent emitted on an authorized turn.  Platform-internal
        hops are intentionally opaque and therefore fail closed here; Runtime's
        actor-context resolver remains the authoritative acceptance boundary.
        """

        if inbound.msg_id in self._actor_report_root_msg_ids:
            return True
        return (
            inbound.in_reply_to is not None
            and inbound.in_reply_to in self._reportable_actor_outbound_msg_ids
        )

    def _public_task_trace_step(
        self,
        scope: Mapping[str, Any],
        *,
        decision_id: str,
        phase_index: int,
        business_surface_sha256: str | None = None,
    ) -> dict[str, Any] | None:
        """Return hashes-only Tracker evidence for one provider decision."""

        phase_id = scope.get("task_phase_id")
        contract_sha256 = scope.get("task_execution_contract_sha256")
        if phase_id is None and contract_sha256 is None:
            return None
        if (
            not isinstance(phase_id, str)
            or not phase_id
            or not isinstance(contract_sha256, str)
            or len(contract_sha256) != 64
            or any(character not in "0123456789abcdef" for character in contract_sha256)
            or not isinstance(decision_id, str)
            or not decision_id
            or isinstance(phase_index, bool)
            or not isinstance(phase_index, int)
            or phase_index < 0
        ):
            raise RuntimeError("public task trace binding is malformed")
        grounding_authority = scope.get("grounding_authority")
        if grounding_authority is not None and not isinstance(
            grounding_authority,
            ResolvedAuthorityPrerequisites,
        ):
            raise RuntimeError("public task grounding authority is malformed")
        authority_revision = (
            grounding_authority.decision_surface_revision if grounding_authority is not None else 0
        )
        if (
            isinstance(authority_revision, bool)
            or not isinstance(authority_revision, int)
            or authority_revision < 0
        ):
            raise RuntimeError("public task authority revision is malformed")
        source_ids = (
            grounding_authority.decision_surface_source_msg_ids
            if grounding_authority is not None
            else ()
        )
        if (
            not isinstance(source_ids, (list, tuple, frozenset))
            or any(
                not isinstance(source_id, str) or not source_id.strip() for source_id in source_ids
            )
            or len(source_ids) != len(set(source_ids))
        ):
            raise RuntimeError("public task authority sources are malformed")
        common = {
            "step": "public_task_execution",
            "phase_id": phase_id,
            "phase_index": phase_index,
            "decision_id": decision_id,
            "execution_contract_sha256": contract_sha256,
            "semantic_authority_revision": authority_revision,
            "authority_source_msg_ids": list(source_ids),
        }
        if business_surface_sha256 is not None:
            if len(business_surface_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in business_surface_sha256
            ):
                raise RuntimeError("business decision surface digest is malformed")
            return {
                **common,
                "interface": "llm_business_decision_v1",
                "llm_decision_surface_sha256": business_surface_sha256,
            }
        return common

    def _framework_public_task_continuation(
        self,
        env: "Envelope",
        ctx: "AgentContext",
    ) -> tuple[bool, Envelope | None]:
        """Emit one contract-declared deterministic continuation.

        A bound-counterparty continuation is legal only for the exact Platform
        acknowledgement of an already Runtime-committed after-sales action.
        The participant destination and correlation payload are reconstructed
        from that authenticated lineage; neither is supplied by the model.
        """

        scope = self._semantic_task_execution_scope(env)
        if not isinstance(scope, Mapping):
            return False, None
        continuation = scope.get("task_framework_continuation")
        if not isinstance(continuation, Mapping):
            return False, None
        action_kind = continuation.get("action_kind")
        declared_destination = continuation.get("destination")
        declared_payload = continuation.get("payload")
        if (
            not isinstance(action_kind, str)
            or action_kind != "commerce.send_message"
            or action_kind not in self.outbound_action_kinds
            or declared_destination not in {"@self", "@bound_counterparty"}
            or not isinstance(declared_payload, Mapping)
        ):
            raise RuntimeError("framework continuation is outside Agent authority")
        destination = self.id
        payload = copy.deepcopy(dict(declared_payload))
        continuation_source: tuple[str, str] | None = None
        if declared_destination == "@bound_counterparty":
            if (
                env.from_ != "platform:after-sales"
                or env.action.get("kind") != "platform.after_sales_updated"
                or not isinstance(env.in_reply_to, str)
            ):
                raise RuntimeError(
                    "bound-counterparty continuation lacks an after-sales acknowledgement"
                )
            accepted = self._committed_semantic_outbound_by_msg_id.get(env.in_reply_to)
            acknowledgement = env.action.get("payload")
            if accepted is None or not isinstance(acknowledgement, Mapping):
                raise RuntimeError(
                    "bound-counterparty continuation has no committed action lineage"
                )
            operation = acknowledgement.get("operation")
            disposition = acknowledgement.get("disposition")
            order_id = acknowledgement.get("order_id")
            if (
                not isinstance(operation, str)
                or disposition not in {"committed", "idempotent"}
                or not isinstance(order_id, str)
                or not order_id
            ):
                raise RuntimeError("bound-counterparty continuation acknowledgement is malformed")
            route_operation = "request_order_return" if operation == "request_return" else operation
            route = DEFAULT_AGENT_ROUTE_REGISTRY.business_route_registry().by_operation(
                route_operation
            )
            if (
                route_operation not in AFTER_SALES_ROUTE_OPERATIONS
                or self.id.split(":", 1)[0] not in route.roles
                or accepted.action_kind != route.action_kind
                or accepted.destination != route.destination
            ):
                raise RuntimeError(
                    "bound-counterparty continuation crossed accepted action authority"
                )
            if accepted.order_id != order_id:
                raise RuntimeError(
                    "bound-counterparty acknowledgement crossed committed request order"
                )
            mandate_contract = self._semantic_mandate.get("benchmark_contract")
            policy_contract = self._semantic_policy.get("benchmark_contract")
            if (
                mandate_contract is not None
                and policy_contract is not None
                and mandate_contract != policy_contract
            ):
                raise RuntimeError("bound-counterparty task authority is ambiguous")
            benchmark_contract = (
                mandate_contract if mandate_contract is not None else policy_contract
            )
            if not isinstance(benchmark_contract, Mapping):
                raise RuntimeError("bound-counterparty task authority is missing")
            authorized_order_ids = benchmark_contract.get("order_ids")
            if not isinstance(authorized_order_ids, (list, tuple)):
                authorized_order_id = benchmark_contract.get("order_id")
                authorized_order_ids = (
                    [authorized_order_id] if isinstance(authorized_order_id, str) else []
                )
            if (
                not authorized_order_ids
                or any(not isinstance(value, str) or not value for value in authorized_order_ids)
                or order_id not in authorized_order_ids
            ):
                raise RuntimeError(
                    "bound-counterparty acknowledgement crossed task order authority"
                )

            gate = continuation.get("accepted_reference_gate")
            if gate is not None:
                if not isinstance(gate, Mapping):
                    raise RuntimeError("accepted-reference continuation gate is malformed")
                reference_field = gate.get("reference_field")
                required_values = gate.get("required_values")
                references = acknowledgement.get("references")
                if (
                    not isinstance(reference_field, str)
                    or not isinstance(required_values, (list, tuple))
                    or not required_values
                    or any(not isinstance(value, str) or not value for value in required_values)
                    or len(required_values) != len(set(required_values))
                    or not isinstance(references, Mapping)
                ):
                    raise RuntimeError("accepted-reference continuation gate is malformed")
                observed = references.get(reference_field)
                if not isinstance(observed, str) or observed not in required_values:
                    raise RuntimeError(
                        "accepted-reference continuation lies outside its public gate"
                    )
                _gate_chars, gate_key = canonical_semantic_digest(
                    {
                        "phase_id": scope.get("task_phase_id"),
                        "actor_id": self.id,
                        "order_id": order_id,
                        "reference_field": reference_field,
                        "required_values": list(required_values),
                    }
                )
                cache = getattr(self, "_framework_accepted_reference_values", None)
                if cache is None:
                    cache = {}
                    setattr(self, "_framework_accepted_reference_values", cache)
                if not isinstance(cache, dict):
                    raise RuntimeError("accepted-reference continuation cache is malformed")
                accepted_values = set(cache.get(gate_key, ()))
                accepted_values.add(observed)
                cache[gate_key] = frozenset(accepted_values)
                complete = accepted_values == set(required_values)
                if not complete:
                    return False, None
                completion_sources = getattr(
                    self,
                    "_framework_accepted_reference_completion_sources",
                    None,
                )
                if completion_sources is None:
                    completion_sources = {}
                    setattr(
                        self,
                        "_framework_accepted_reference_completion_sources",
                        completion_sources,
                    )
                if not isinstance(completion_sources, dict):
                    raise RuntimeError("accepted-reference completion source cache is malformed")
                completion_source = completion_sources.setdefault(
                    gate_key,
                    env.in_reply_to,
                )
                if completion_source != env.in_reply_to:
                    # Only the accepted operation that closed the public gate
                    # owns the peer handoff. Replaying an earlier accepted
                    # reference cannot mint another continuation source.
                    return True, None

            destination = self._semantic_reply_recipient_for(env) or ""
            if (
                destination.split(":", 1)[0] not in {"buyer", "merchant"}
                or destination == self.id
                or destination.split(":", 1)[0] == self.id.split(":", 1)[0]
            ):
                raise RuntimeError(
                    "bound-counterparty continuation has no authenticated participant"
                )
            dynamic_payload = _agent_message_payload_authority(
                env,
                benchmark_contract=None,
                reply_recipient_id=destination,
            )
            if dynamic_payload is None:
                raise RuntimeError(
                    "bound-counterparty continuation has no accepted business correlation"
                )
            payload.setdefault("category", "after_sales")
            payload.setdefault("order_id", order_id)
            if payload.get("category") != "after_sales" or payload.get("order_id") != order_id:
                raise RuntimeError(
                    "bound-counterparty continuation payload crossed order authority"
                )
            for key, value in dynamic_payload.items():
                if key in payload and payload[key] != value:
                    raise RuntimeError(
                        "bound-counterparty continuation payload conflicts with acknowledgement"
                    )
                payload[key] = copy.deepcopy(value)
            _source_chars, source_identity = canonical_semantic_digest(
                {
                    "accepted_outbound_msg_id": env.in_reply_to,
                    "operation": operation,
                    "order_id": order_id,
                    "recipient_id": destination,
                    "payload": payload,
                }
            )
            continuation_source = (env.in_reply_to, source_identity)
        elif scope.get("task_finish_policy") != "framework_continue":
            raise RuntimeError("self continuation requires framework_continue")
        public_step = self._public_task_trace_step(
            scope,
            decision_id=decision_id_for(env),
            phase_index=0,
        )
        assert public_step is not None
        payload_chars, payload_sha256 = canonical_semantic_digest(dict(payload))
        history = [
            public_step,
            {
                "step": "framework_public_task_continuation",
                "interface": "deterministic_agent_controller",
                "action_kind": action_kind,
                "destination": ("@self" if declared_destination == "@self" else destination),
                "payload_chars": payload_chars,
                "payload_sha256": payload_sha256,
            },
        ]
        trace = getattr(ctx, "trace", None)
        if trace is not None:
            trace.decision_id = decision_id_for(env)
            trace.bind_steps(history)
        if continuation_source is not None:
            source_key, source_identity = continuation_source
            committed_sources = getattr(
                self,
                "_framework_committed_continuation_sources",
                {},
            )
            if not isinstance(committed_sources, dict):
                raise RuntimeError("committed continuation source cache is malformed")
            committed_identity = committed_sources.get(source_key)
            if committed_identity is not None:
                if committed_identity != source_identity:
                    raise RuntimeError(
                        "accepted after-sales lineage changed after continuation commit"
                    )
                if trace is not None:
                    trace.finalize(history, result=None)
                return True, None
            provisional_sources = getattr(
                self,
                "_framework_provisional_continuation_sources",
                {},
            )
            if not isinstance(provisional_sources, dict):
                raise RuntimeError("provisional continuation source cache is malformed")
            provisional_source = provisional_sources.get(source_key)
            if provisional_source is not None:
                if (
                    not isinstance(provisional_source, tuple)
                    or len(provisional_source) != 2
                    or provisional_source[0] != source_identity
                    or not isinstance(provisional_source[1], str)
                ):
                    raise RuntimeError(
                        "accepted after-sales lineage changed before continuation commit"
                    )
                candidate = self.provisional_outbound_envelope(provisional_source[1])
                if candidate is None:
                    raise RuntimeError("provisional continuation source lost its exact envelope")
                if trace is not None:
                    trace.finalize(history, result=candidate)
                return True, candidate
        try:
            out = self._build_outbound(
                {
                    "to": destination,
                    "action": {
                        "kind": action_kind,
                        "payload": copy.deepcopy(dict(payload)),
                    },
                },
                env,
                history,
            )
        except ValueError as exc:
            raise InvalidAgentEnvelope(str(exc)) from exc
        if out.msg_id in self._committed_semantic_outbound_by_msg_id:
            # A repeated exact Platform acknowledgement deterministically
            # produces the same message identity.  Once Runtime committed it,
            # acknowledge the duplicate without emitting another peer turn.
            if trace is not None:
                trace.finalize(history, result=None)
            return True, None
        self._stage_provisional_outbound(
            out,
            memory_update=None,
            principal_id=self._semantic_principal_for(env),
            reply_recipient_id=self._semantic_reply_recipient_for(env),
            actor_reports_allowed=self._actor_reports_allowed_for(env),
            continuation_source=continuation_source,
        )
        if trace is not None:
            trace.finalize(history, result=out)
        return True, out

    def _framework_public_task_terminal(
        self,
        env: "Envelope",
        ctx: "AgentContext",
    ) -> bool:
        """End one explicitly framework-terminal public phase without inference."""

        scope = self._semantic_task_execution_scope(env)
        if (
            not isinstance(scope, Mapping)
            or scope.get("task_finish_policy") != "framework_terminal"
        ):
            return False
        public_step = self._public_task_trace_step(
            scope,
            decision_id=decision_id_for(env),
            phase_index=0,
        )
        assert public_step is not None
        history = [
            public_step,
            {
                "step": "framework_public_task_terminal",
                "interface": "deterministic_agent_controller",
                "trigger_action_kind": env.action.get("kind"),
                "terminal": "no_reply",
            },
        ]
        trace = getattr(ctx, "trace", None)
        if trace is not None:
            trace.decision_id = decision_id_for(env)
            trace.bind_steps(history)
            trace.finalize(history, result=None)
        return True

    def _framework_protocol_continuation(
        self,
        env: "Envelope",
        ctx: "AgentContext",
    ) -> "tuple[bool, Envelope | None]":
        """Compile an authorized certificate reply without another model call.

        This path never chooses an offer.  It only continues an exact, already
        audited aggregator acceptance when the buyer's frozen mandate permits
        purchase without another principal confirmation.
        """

        if not self.id.startswith("buyer"):
            return False, None
        kind = env.action.get("kind")
        if kind == "platform.settlement_receipt":
            return self._framework_settlement_receipt(env, ctx)
        if (
            env.from_ != "platform:aggregator"
            or kind != "platform.create_match_certificate"
            or "platform.settle_payment" not in self.outbound_action_kinds
        ):
            return False, None
        authority = self._semantic_mandate.get("authority")
        if (
            not isinstance(authority, Mapping)
            or authority.get("can_buy_without_confirmation") is not True
        ):
            return False, None
        if not isinstance(env.in_reply_to, str):
            return False, None
        accepted = self._committed_semantic_outbound_by_msg_id.get(env.in_reply_to)
        if (
            accepted is None
            or accepted.action_kind != "commerce.accept_offer"
            or accepted.destination != "platform:aggregator"
        ):
            return False, None
        payload = env.action.get("payload")
        cert_id = payload.get("cert_id") if isinstance(payload, Mapping) else None
        if not isinstance(cert_id, str) or not cert_id.strip():
            return False, None

        history = [
            {
                "step": "framework_protocol_continuation",
                "interface": "deterministic_agent_controller",
                "trigger_action_kind": "platform.create_match_certificate",
                "action_kind": "platform.settle_payment",
                "destination": "platform:psp",
                "authority": "can_buy_without_confirmation",
                "accepted_envelope_sha256": accepted.envelope_sha256,
                "certificate_id_sha256": hashlib.sha256(cert_id.encode("utf-8")).hexdigest(),
            }
        ]
        trace = getattr(ctx, "trace", None)
        if trace is not None:
            trace.decision_id = decision_id_for(env)
            trace.bind_steps(history)
        try:
            out = self._build_outbound(
                {
                    "to": "platform:psp",
                    "action": {
                        "kind": "platform.settle_payment",
                        "payload": {"cert_id": cert_id},
                    },
                },
                env,
                history,
            )
        except ValueError as exc:
            raise InvalidAgentEnvelope(str(exc)) from exc
        self._stage_provisional_outbound(
            out,
            memory_update=None,
            principal_id=self._semantic_principal_for(env),
            reply_recipient_id=self._semantic_reply_recipient_for(env),
            actor_reports_allowed=self._actor_reports_allowed_for(env),
        )
        if trace is not None:
            trace.finalize(history, result=out)
        return True, out

    def _framework_settlement_receipt(
        self,
        env: "Envelope",
        ctx: "AgentContext",
    ) -> "tuple[bool, None]":
        """Acknowledge an exact successful payment receipt without inference."""

        if (
            env.from_ != "platform:psp"
            or self._semantic_mandate.get("return_after_purchase") is True
            or self._settlement_receipt_requires_model_decision(env)
            or not isinstance(env.in_reply_to, str)
        ):
            return False, None
        settled = self._committed_semantic_outbound_by_msg_id.get(env.in_reply_to)
        if (
            settled is None
            or settled.action_kind != "platform.settle_payment"
            or settled.destination != "platform:psp"
        ):
            return False, None
        payload = env.action.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        completed_order: dict[str, str] = {}
        for name in ("order_id", "txn_id"):
            value = payload.get(name)
            if isinstance(value, str) and value.strip():
                completed_order[name] = value
        writes = []
        existing_completed = self.memory.read(
            MemoryType.TRANSACTION,
            "completed_orders",
        )
        if existing_completed is None:
            completed_orders: list[dict[str, str]] = []
        elif isinstance(existing_completed, list) and all(
            isinstance(row, Mapping) for row in existing_completed
        ):
            completed_orders = [dict(row) for row in existing_completed]
        else:
            raise RuntimeError("TRANSACTION.completed_orders is not a structured order list")
        already_recorded = bool(completed_order) and any(
            (
                isinstance(completed_order.get("order_id"), str)
                and row.get("order_id") == completed_order["order_id"]
            )
            or (
                isinstance(completed_order.get("txn_id"), str)
                and row.get("txn_id") == completed_order["txn_id"]
            )
            for row in completed_orders
        )
        if completed_order and not already_recorded:
            completed_orders.append(completed_order)
            self.memory.write(
                MemoryType.TRANSACTION,
                "completed_orders",
                completed_orders,
            )
            writes.append(
                {
                    "bucket": MemoryType.TRANSACTION.value,
                    "key": "completed_orders",
                    "value": completed_orders,
                }
            )
            selected = self.memory.read(MemoryType.TRANSACTION, "selected_offer")
            unit_price = selected.get("unit_price") if isinstance(selected, Mapping) else None
            quantity = selected.get("qty", 1) if isinstance(selected, Mapping) else 1
            cumulative = self.memory.read(
                MemoryType.TRANSACTION,
                "cumulative_spend",
            )
            if (
                isinstance(unit_price, int)
                and not isinstance(unit_price, bool)
                and isinstance(quantity, int)
                and not isinstance(quantity, bool)
                and quantity > 0
                and isinstance(cumulative, int)
                and not isinstance(cumulative, bool)
            ):
                updated_cumulative = cumulative + unit_price * quantity
                self.memory.write(
                    MemoryType.TRANSACTION,
                    "cumulative_spend",
                    updated_cumulative,
                )
                writes.append(
                    {
                        "bucket": MemoryType.TRANSACTION.value,
                        "key": "cumulative_spend",
                        "value": updated_cumulative,
                    }
                )
        self.memory.write(
            MemoryType.TRANSACTION,
            "pending_settlement_order_id",
            None,
        )
        writes.append(
            {
                "bucket": MemoryType.TRANSACTION.value,
                "key": "pending_settlement_order_id",
                "value": None,
            }
        )
        history: list[dict[str, Any]] = []
        task_scope = self._semantic_task_execution_scope(env)
        if isinstance(task_scope, Mapping):
            public_step = self._public_task_trace_step(
                task_scope,
                decision_id=decision_id_for(env),
                phase_index=0,
            )
            if public_step is not None:
                history.append(public_step)
        if writes:
            history.append(
                {
                    "step": "memory_update",
                    "source": "framework_protocol_receipt",
                    "writes": writes,
                }
            )
        history.append(
            {
                "step": "framework_protocol_receipt",
                "interface": "deterministic_agent_controller",
                "trigger_action_kind": "platform.settlement_receipt",
                "settled_envelope_sha256": settled.envelope_sha256,
                "terminal": "no_reply",
            }
        )
        trace = getattr(ctx, "trace", None)
        if trace is not None:
            trace.decision_id = decision_id_for(env)
            trace.bind_steps(history)
            trace.finalize(history, result=None)
        return True, None

    def _settlement_receipt_requires_model_decision(
        self,
        env: "Envelope",
    ) -> bool:
        """Keep a declared post-settlement business decision model-owned.

        Some environment tasks explicitly require the actor to submit a
        structured result only after observing the authoritative settlement
        receipt.  That requirement is public mandate context, not a hidden
        oracle.  The deterministic receipt controller must therefore yield to
        the semantic Agent instead of silently ending the turn.
        """

        if env.action.get("kind") != "platform.settlement_receipt":
            return False
        task_scope = self._semantic_task_execution_scope(env)
        result_contract = (
            task_scope.get("task_result_contract") if isinstance(task_scope, Mapping) else None
        )
        return bool(isinstance(result_contract, Mapping) and result_contract.get("active") is True)

    def _receive_business(
        self,
        env: "Envelope",
        ctx: "AgentContext",
        *,
        activated: "tuple[str, ...] | None",
    ) -> "Envelope | TurnSuspended | None":
        """Run the sole typed LLM decision loop for one Agent turn."""

        history = self._business_turn_state(activated)
        actor_reports_allowed = self._actor_reports_allowed_for(env)
        trace = getattr(ctx, "trace", None)
        decision_id = decision_id_for(env)
        if trace is not None:
            trace.decision_id = decision_id
            trace.bind_steps(history)

        authority_order_ids = self._prepare_after_sales_authority(env, ctx)
        protocol_event_id = self._prepare_protocol_event_authority(env, ctx)
        if authority_order_ids and protocol_event_id is not None:
            raise PlatformContractError(
                "one business turn cannot mix authority prerequisite domains"
            )
        if getattr(ctx, "remote_world", False):
            frame = TurnFrame(
                decision_id=decision_id,
                original_inbound=env,
                history=history,
                activated=activated,
                trace=trace,
                actor_reports_allowed=actor_reports_allowed,
            )
            if authority_order_ids:
                frame.framework_authority_prerequisite = "after_sales"
                frame.pending_calls = [
                    ToolCall(
                        tool="world.get_after_sales_authority",
                        args={"order_id": order_id},
                    )
                    for order_id in authority_order_ids
                ]
                suspended = self._advance_reads(frame)
                if suspended is None:
                    raise RuntimeError("after-sales authority prerequisite issued no World read")
                return suspended
            if protocol_event_id is not None:
                frame.framework_authority_prerequisite = "protocol_event"
                frame.pending_calls = [
                    ToolCall(
                        tool="world.get_protocol_event_authority",
                        args={"event_id": protocol_event_id},
                    )
                ]
                suspended = self._advance_reads(frame)
                if suspended is None:
                    raise RuntimeError("protocol-event authority prerequisite issued no World read")
                return suspended
            return self._business_reentrant_loop(ctx, frame)

        read_fingerprints: set[str] = set()
        finish_allowed = self._semantic_finish_allowed(env)
        for _ in range(MAX_INTERNAL_STEPS):
            try:
                decision = self._business_decide(
                    env,
                    ctx,
                    history,
                    activated_skill_names=activated,
                    allow_actor_reports=actor_reports_allowed,
                    include_finish=finish_allowed,
                    decision_id=decision_id,
                )
            except ModelDecisionParseError as exc:
                if not exc.provider_response_received or not self._claim_business_repair():
                    raise
                self._record_business_response_error(history, exc)
                continue
            compiled = decision.compiled
            if isinstance(compiled, CompiledWorldRead):
                calls = (
                    ToolCall(
                        tool=compiled.tool,
                        args=copy.deepcopy(dict(compiled.args)),
                    ),
                )
                try:
                    fingerprint = self._semantic_read_batch_fingerprint(
                        calls,
                        decision=decision,
                    )
                except ModelDecisionParseError as exc:
                    self._record_business_validation_error(history, decision, exc)
                    if self._claim_business_repair():
                        continue
                    raise
                if fingerprint in read_fingerprints:
                    exc = ModelDecisionParseError(
                        "business decision repeated an identical World read",
                        response_chars=decision.response_chars,
                        response_sha256=decision.response_sha256,
                    )
                    self._record_business_validation_error(history, decision, exc)
                    if self._claim_business_repair():
                        continue
                    raise exc
                if len(read_fingerprints) >= (_MAX_BUSINESS_DISTINCT_READ_DECISIONS_PER_TURN):
                    exc = ModelDecisionParseError(
                        "business decision exceeds the distinct World read limit",
                        response_chars=decision.response_chars,
                        response_sha256=decision.response_sha256,
                    )
                    self._record_business_validation_error(history, decision, exc)
                    if self._claim_business_repair():
                        continue
                    raise exc
                read_fingerprints.add(fingerprint)
                self._record_business_read_request(history, decision)
                results = [self._dispatch_tool(call, ctx) for call in calls]
                self._commerce_action_grounding.observe_read_results(results)
                history.append(
                    {
                        "step": "tool_call",
                        "interface": "business_decision",
                        "results": results,
                    }
                )
                continue
            if isinstance(compiled, LocalControlDecision):
                self._record_business_finish(history, decision)
                if trace is not None:
                    trace.finalize(history, result=None)
                return None
            if not isinstance(compiled, CompiledBusinessAction):
                raise RuntimeError("Agent bridge returned an unsupported business disposition")
            observation_fingerprint, retry = self._platform_observation_repeat_guard(
                history,
                decision,
            )
            if retry:
                continue
            try:
                out = self._emit_semantic_action(
                    self._business_action_for(decision),
                    inbound=env,
                    history=history,
                    actor_reports_allowed=actor_reports_allowed,
                    decision=decision,
                    observation_choice_fingerprint=observation_fingerprint,
                )
            except GroundingRequired as grounding_error:
                exc = self._model_grounding_decision_error(
                    decision,
                    grounding_error,
                )
                self._record_business_validation_error(history, decision, exc)
                if self._claim_business_repair():
                    continue
                raise exc from grounding_error
            if trace is not None:
                trace.finalize(history, result=out)
            return out

        if trace is not None:
            trace.finalize(history, result=None, terminal="budget_exceeded")
        raise BudgetExceeded(
            f"agent {self.id!r} exceeded {MAX_INTERNAL_STEPS} business decisions "
            "without a terminal commerce action"
        )

    def _business_reentrant_loop(
        self,
        ctx: "AgentContext",
        frame: "TurnFrame",
    ) -> "Envelope | TurnSuspended | None":
        """Typed business loop whose Agent-owned World reads suspend via VCP."""

        while frame.steps_used < MAX_INTERNAL_STEPS:
            frame.steps_used += 1
            try:
                decision = self._business_decide(
                    frame.original_inbound,
                    ctx,
                    frame.history,
                    activated_skill_names=frame.activated,
                    allow_actor_reports=frame.actor_reports_allowed,
                    include_finish=self._semantic_finish_allowed(frame.original_inbound),
                    decision_id=frame.decision_id,
                )
            except ModelDecisionParseError as exc:
                if not exc.provider_response_received or not self._claim_business_repair():
                    raise
                self._record_business_response_error(frame.history, exc)
                continue
            compiled = decision.compiled
            if isinstance(compiled, CompiledWorldRead):
                calls = (
                    ToolCall(
                        tool=compiled.tool,
                        args=copy.deepcopy(dict(compiled.args)),
                    ),
                )
                try:
                    fingerprint = self._semantic_read_batch_fingerprint(
                        calls,
                        decision=decision,
                    )
                except ModelDecisionParseError as exc:
                    self._record_business_validation_error(
                        frame.history,
                        decision,
                        exc,
                    )
                    if self._claim_business_repair():
                        continue
                    raise
                if fingerprint in frame.business_read_fingerprints:
                    exc = ModelDecisionParseError(
                        "business decision repeated an identical World read",
                        response_chars=decision.response_chars,
                        response_sha256=decision.response_sha256,
                    )
                    self._record_business_validation_error(
                        frame.history,
                        decision,
                        exc,
                    )
                    if self._claim_business_repair():
                        continue
                    raise exc
                if len(frame.business_read_fingerprints) >= (
                    _MAX_BUSINESS_DISTINCT_READ_DECISIONS_PER_TURN
                ):
                    exc = ModelDecisionParseError(
                        "business decision exceeds the distinct World read limit",
                        response_chars=decision.response_chars,
                        response_sha256=decision.response_sha256,
                    )
                    self._record_business_validation_error(
                        frame.history,
                        decision,
                        exc,
                    )
                    if self._claim_business_repair():
                        continue
                    raise exc
                frame.business_read_fingerprints.add(fingerprint)
                self._record_business_read_request(frame.history, decision)
                frame.pending_calls = list(calls)
                frame.pending_results = []
                frame.business_pending_read_fingerprint = fingerprint
                suspended = self._advance_reads(frame)
                if suspended is not None:
                    return suspended
                self._commit_tool_call(frame)
                continue
            if isinstance(compiled, LocalControlDecision):
                self._record_business_finish(frame.history, decision)
                if frame.trace is not None:
                    frame.trace.finalize(frame.history, result=None)
                return None
            if not isinstance(compiled, CompiledBusinessAction):
                raise RuntimeError("Agent bridge returned an unsupported business disposition")
            observation_fingerprint, retry = self._platform_observation_repeat_guard(
                frame.history,
                decision,
            )
            if retry:
                continue
            try:
                out = self._emit_semantic_action(
                    self._business_action_for(decision),
                    inbound=frame.original_inbound,
                    history=frame.history,
                    actor_reports_allowed=frame.actor_reports_allowed,
                    decision=decision,
                    observation_choice_fingerprint=observation_fingerprint,
                )
            except GroundingRequired as grounding_error:
                exc = self._model_grounding_decision_error(
                    decision,
                    grounding_error,
                )
                self._record_business_validation_error(
                    frame.history,
                    decision,
                    exc,
                )
                if self._claim_business_repair():
                    continue
                raise exc from grounding_error
            if frame.trace is not None:
                frame.trace.finalize(frame.history, result=out)
            return out

        if frame.trace is not None:
            frame.trace.finalize(
                frame.history,
                result=None,
                terminal="budget_exceeded",
            )
        raise BudgetExceeded(
            f"agent {self.id!r} exceeded {MAX_INTERNAL_STEPS} business decisions "
            "across re-entrant suspensions"
        )

    def _business_turn_state(
        self,
        activated: "tuple[str, ...] | None",
    ) -> list[dict[str, Any]]:
        """Bootstrap private memory and auto-load selector-approved skills."""

        history: list[dict[str, Any]] = []
        self._bootstrap_semantic_memory(history)
        names = (
            tuple(manifest.name for manifest in self.enabled_skills)
            if activated is None
            else activated
        )
        manifests = {manifest.name: manifest for manifest in self.enabled_skills}
        for name in names:
            manifest = manifests.get(name)
            if manifest is None:
                raise SkillNotActivated(
                    f"selector activated skill {name!r} outside agent {self.id!r}"
                )
            document = self.skill_loader.load(name)
            history.append(
                {
                    "step": "load_skill",
                    "name": name,
                    "digest": manifest.digest,
                    "source": "selector" if activated is not None else "business_default",
                    "result": f"loaded ({len(document.instructions)} chars)",
                }
            )
        return history

    def _semantic_finish_allowed(self, env: "Envelope") -> bool:
        """Return whether an explicit no-action terminal is valid this turn.

        A fresh principal mandate always requires a typed continuation: search,
        an operational request, an evidence result, or an explicit rejection.
        An active public task-result phase likewise requires the one result
        function named by its spawn-frozen contract.  Advertising finish there
        would let a provider silently abandon a task after observing the facts
        needed to answer it.
        Runtime handles idempotent duplicate envelopes before this boundary, so
        other inbound continuations retain finish for completed receipts,
        untrusted messages, and safe no-candidate branches.
        """

        task_scope = self._semantic_task_execution_scope(env)
        finish_policy = (
            task_scope.get("task_finish_policy") if isinstance(task_scope, Mapping) else None
        )
        if isinstance(finish_policy, str):
            return finish_policy == "allow_wait"
        if finish_policy is not None:
            raise ValueError("public task finish policy must be text")
        if env.action.get("kind") == "delegate.create_purchase_mandate":
            return False
        result_contract = (
            task_scope.get("task_result_contract") if isinstance(task_scope, Mapping) else None
        )
        return not (isinstance(result_contract, Mapping) and result_contract.get("active") is True)

    def _public_after_sales_order_authority(
        self,
        *,
        env: "Envelope",
        benchmark_contract: Mapping[str, Any] | None,
        public_routes: Any,
        route_registry: Any,
    ) -> AfterSalesOrderAuthority | None:
        """Bind current after-sales orders from public task and receipt state.

        The operation-to-route relationship is projected from the one internal
        ``RouteRegistry``.  This helper therefore owns no service-address or
        action-kind table of its own.  Ordered task references are public
        business inputs; only an authenticated Platform acknowledgement may
        advance the private sequential cursor.
        """

        active_routes = {
            tuple(route)
            for route in public_routes or ()
            if isinstance(route, (list, tuple)) and len(route) == 2
        }
        bound_routes = tuple(
            binding
            for binding in route_registry.bindings
            if binding.operation in AFTER_SALES_ORDER_BOUND_OPERATIONS
        )
        relevant_routes = {(binding.action_kind, binding.destination) for binding in bound_routes}
        if active_routes.isdisjoint(relevant_routes):
            return None
        if not isinstance(benchmark_contract, Mapping):
            raise PlatformContractError(
                "public after-sales order phase has no task business authority"
            )

        raw_order_ids = benchmark_contract.get("order_ids")
        raw_order_id = benchmark_contract.get("order_id")
        if raw_order_ids is None:
            raw_order_ids = [raw_order_id] if raw_order_id is not None else None
        if (
            not isinstance(raw_order_ids, (list, tuple))
            or not raw_order_ids
            or any(
                not isinstance(order_id, str) or not order_id.strip() for order_id in raw_order_ids
            )
            or len(raw_order_ids) != len(set(raw_order_ids))
        ):
            raise PlatformContractError("public after-sales task order authority is malformed")
        task_order_ids = tuple(raw_order_ids)

        payload = env.action.get("payload")
        operation = payload.get("operation") if isinstance(payload, Mapping) else None
        if (
            env.action.get("kind") == "platform.after_sales_updated"
            and operation == "request_ledger_reconciliation"
        ):
            service_ids = {binding.destination for binding in bound_routes}
            order_id = payload.get("order_id") if isinstance(payload, Mapping) else None
            disposition = payload.get("disposition") if isinstance(payload, Mapping) else None
            if (
                env.from_ not in service_ids
                or not isinstance(order_id, str)
                or order_id not in task_order_ids
                or disposition not in {"committed", "idempotent"}
            ):
                raise PlatformContractError(
                    "ledger reconciliation receipt is outside task order authority"
                )
            receipt_identity = (env.from_, operation, order_id, disposition)
            prior_receipt = self._processed_after_sales_receipts.get(env.msg_id)
            if prior_receipt is not None:
                if prior_receipt != receipt_identity:
                    raise PlatformContractError(
                        "ledger reconciliation receipt identity changed within one turn"
                    )
            else:
                remaining_before = tuple(
                    candidate
                    for candidate in task_order_ids
                    if candidate not in self._reconciled_after_sales_order_ids
                )
                if order_id in self._reconciled_after_sales_order_ids:
                    if disposition != "idempotent":
                        raise PlatformContractError(
                            "ledger reconciliation receipt repeats a committed order"
                        )
                elif not remaining_before or order_id != remaining_before[0]:
                    raise PlatformContractError(
                        "ledger reconciliation receipt advanced orders out of sequence"
                    )
                else:
                    self._reconciled_after_sales_order_ids.add(order_id)
                self._processed_after_sales_receipts[env.msg_id] = receipt_identity

        if not self._reconciled_after_sales_order_ids.issubset(task_order_ids):
            raise RuntimeError("after-sales order cursor crossed task authority")
        remaining = tuple(
            order_id
            for order_id in task_order_ids
            if order_id not in self._reconciled_after_sales_order_ids
        )
        return AfterSalesOrderAuthority(
            actor_id=self.id,
            task_order_ids=task_order_ids,
            current_order_ids=remaining[:1],
        )

    def _after_sales_route_operations(
        self,
        env: "Envelope",
        task_scope: Mapping[str, Any] | None,
    ) -> frozenset[str]:
        """Resolve this turn's after-sales registry closure without task-family logic."""

        routes = DEFAULT_AGENT_ROUTE_REGISTRY.business_route_registry()
        public_routes = (
            task_scope.get("task_allowed_routes") if isinstance(task_scope, Mapping) else None
        )
        registered = (
            frozenset(
                tuple(route)
                for route in public_routes
                if isinstance(route, (list, tuple)) and len(route) == 2
            )
            if isinstance(public_routes, (list, tuple, frozenset))
            else None
        )
        operations = {
            binding.operation
            for binding in routes.bindings
            if binding.operation in AFTER_SALES_ROUTE_OPERATIONS
            and self.id.split(":", 1)[0] in binding.roles
            and binding.action_kind in self.outbound_action_kinds
            and (registered is None or (binding.action_kind, binding.destination) in registered)
        }
        explicitly_after_sales = _is_after_sales_business_inbound(env)
        # With no task route closure, an explicit after-sales inbound may use
        # the actor's registered after-sales surface.  A generic shipment (or
        # any unrelated business turn) must not acquire that authority merely
        # because the actor supports after-sales actions globally.  When a
        # public task closure exists, the registered domain routes themselves
        # are the activation signal; an empty/unrelated closure stays closed.
        if registered is None:
            if not explicitly_after_sales:
                return frozenset()
        elif not operations:
            return frozenset()
        return frozenset(operations)

    def _after_sales_turn_order_ids(
        self,
        env: "Envelope",
        *,
        task_scope: Mapping[str, Any] | None,
        route_operations: frozenset[str],
    ) -> tuple[str, ...]:
        """Bind finite current order references for an Agent authority read."""

        public_routes = (
            task_scope.get("task_allowed_routes") if isinstance(task_scope, Mapping) else None
        )
        benchmark_contract = (
            self._semantic_mandate.get("benchmark_contract")
            if self.id.startswith("buyer")
            else self._semantic_policy.get("benchmark_contract")
        )
        task_business_ids = (
            _collect_after_sales_order_ids(benchmark_contract)
            if isinstance(benchmark_contract, Mapping)
            else ()
        )
        inbound_ids = _collect_after_sales_order_ids(env.action.get("payload"))
        if task_business_ids and not set(inbound_ids).issubset(task_business_ids):
            raise PlatformContractError(
                "after-sales inbound order lies outside public business authority"
            )
        ordered = (
            self._public_after_sales_order_authority(
                env=env,
                benchmark_contract=benchmark_contract,
                public_routes=public_routes,
                route_registry=DEFAULT_AGENT_ROUTE_REGISTRY.business_route_registry(),
            )
            if isinstance(benchmark_contract, Mapping)
            else None
        )
        if "request_ledger_reconciliation" in route_operations and ordered is not None:
            order_ids = ordered.current_order_ids
        elif task_business_ids:
            order_ids = task_business_ids
        else:
            order_ids = inbound_ids
        if not order_ids or len(order_ids) != len(set(order_ids)) or len(order_ids) > 64:
            raise PlatformContractError(
                "after-sales turn has no finite unique business order authority"
            )
        return tuple(order_ids)

    def _prepare_after_sales_authority(
        self,
        env: "Envelope",
        ctx: "AgentContext",
    ) -> tuple[str, ...]:
        """Load local sealed projections or return ids for audited R1 reads."""

        task_scope = self._semantic_task_execution_scope(env)
        operations = self._after_sales_route_operations(env, task_scope)
        if not operations:
            self._after_sales_projections_by_inbound.pop(env.msg_id, None)
            return ()
        order_ids = self._after_sales_turn_order_ids(
            env,
            task_scope=task_scope,
            route_operations=operations,
        )
        self._after_sales_projections_by_inbound.pop(env.msg_id, None)
        if getattr(ctx, "remote_world", False):
            return order_ids
        projections = tuple(ctx.world.get_after_sales_authority(order_id) for order_id in order_ids)
        self._after_sales_projections_by_inbound[env.msg_id] = projections
        return ()

    def _prepare_protocol_event_authority(
        self,
        env: "Envelope",
        ctx: "AgentContext",
    ) -> str | None:
        """Bind a fresh current-event projection at actual Agent handling time."""

        self._protocol_event_authority_by_inbound.pop(env.msg_id, None)
        if env.action.get("kind") != "platform.deliver_protocol_event":
            return None
        payload = env.action.get("payload")
        event = payload.get("event") if isinstance(payload, Mapping) else None
        event_id = event.get("event_id") if isinstance(event, Mapping) else None
        if not isinstance(event_id, str) or not event_id.strip():
            raise PlatformContractError("protocol event delivery has no finite event identity")
        if getattr(ctx, "remote_world", False):
            return event_id
        authority = ctx.world.get_protocol_event_authority(event_id)
        if not isinstance(authority, Mapping):
            raise PlatformContractError("World protocol event authority response is malformed")
        self._protocol_event_authority_by_inbound[env.msg_id] = copy.deepcopy(dict(authority))
        return None

    def _semantic_actor_context(
        self,
        env: "Envelope",
        *,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return framework-owned fields used by the semantic compiler."""

        context: dict[str, Any] = {"actor_id": self.id}
        domain_phases: list[BoundDomainPhase] = []
        domain_routes = DEFAULT_AGENT_ROUTE_REGISTRY.business_route_registry()
        negotiation_authority: NegotiationTurnAuthority | None = None
        task_scope = self._semantic_task_execution_scope(env)
        if task_scope is not None:
            task_scope = dict(task_scope)
            read_gate = task_scope.get("task_pre_action_read_gate")
            gated_action_kind = _pending_pre_action_read_gate(
                read_gate,
                history or [],
            )
            if gated_action_kind is not None:
                if (
                    isinstance(read_gate, Mapping)
                    and read_gate.get("exclusive_until_complete") is True
                ):
                    task_scope["task_allowed_action_kinds"] = ()
                    task_scope["task_allowed_routes"] = ()
                else:
                    task_scope["task_blocked_action_kinds"] = (gated_action_kind,)
            context.update(task_scope)
        public_routes = (
            task_scope.get("task_allowed_routes") if isinstance(task_scope, Mapping) else None
        )
        inbound_payload = env.action.get("payload")
        registered_routes = tuple(
            tuple(route)
            for route in public_routes or ()
            if isinstance(route, (list, tuple)) and len(route) == 2
        )
        registered_route_set = frozenset(registered_routes)
        after_sales_route_operations = self._after_sales_route_operations(
            env,
            task_scope,
        )
        complete_after_sales_phase = bool(after_sales_route_operations)
        send_message_route = domain_routes.by_operation("send_message")
        allocate_fulfillment_route = domain_routes.by_operation("allocate_fulfillment")
        supply_report_route_exclusive = bool(
            env.action.get("kind") == "platform.supply_state"
            and public_routes is not None
            and (
                send_message_route.action_kind,
                send_message_route.destination,
            )
            in registered_route_set
            and (
                allocate_fulfillment_route.action_kind,
                allocate_fulfillment_route.destination,
            )
            not in registered_route_set
        )

        def operation_is_registered(operation: str) -> bool:
            binding = domain_routes.by_operation(operation)
            return bool(
                binding is not None
                and (binding.action_kind, binding.destination) in registered_route_set
            )

        if env.action.get("kind") == "commerce.send_message":
            persistent_content_ids = _collect_business_content_ids(inbound_payload)
            if persistent_content_ids and not operation_is_registered("get_sku"):
                public_content_observations = public_business_observations(
                    inbound_kind="commerce.send_message",
                    inbound_payload=(
                        inbound_payload if isinstance(inbound_payload, Mapping) else {}
                    ),
                    history=(),
                )
                sender_role = env.from_.split(":", 1)[0]
                if len(public_content_observations) != 1 or sender_role not in {
                    "buyer",
                    "merchant",
                    "consumer",
                }:
                    raise PlatformContractError(
                        "peer content has no unique public source observation"
                    )
                retained = {
                    "source_role": sender_role,
                    "provenance": "authenticated_inbound_business_message",
                    "observation": copy.deepcopy(public_content_observations[0]),
                }
                previous_retained = self._persistent_public_content_by_inbound.get(env.msg_id)
                if previous_retained is not None and previous_retained != retained:
                    raise PlatformContractError("peer public content changed across retries")
                self._persistent_public_content_by_inbound[env.msg_id] = retained
                if len(self._persistent_public_content_by_inbound) > 64:
                    raise PlatformContractError(
                        "peer public content history exceeds its bounded scope"
                    )

        if operation_is_registered("get_sku") and env.action.get("kind") == "commerce.send_message":
            # The lookup response is correlated to this authenticated
            # participant request.  The model chooses the SKU; Agent owns the
            # request identity carried through Platform.
            context["catalog_lookup_request_id"] = env.msg_id
            content_ids = _collect_business_content_ids(inbound_payload)
            if not content_ids:
                raise PlatformContractError(
                    "current inquiry has no finite business content authority"
                )
            public_content_observations = public_business_observations(
                inbound_kind=str(env.action.get("kind", "commerce.send_message")),
                inbound_payload=inbound_payload,
                history=(),
            )
            if len(public_content_observations) != 1:
                raise PlatformContractError(
                    "current inquiry has no unique public content observation"
                )
            sender_role = env.from_.split(":", 1)[0]
            if sender_role not in {"buyer", "merchant", "consumer"}:
                raise PlatformContractError("current inquiry sender has no public business role")
            inquiry_authority = {
                "request_id": env.msg_id,
                "sender_id": env.from_,
                "content_ids": content_ids,
                # Preserve the provider-visible business content, not a
                # benchmark answer or an opaque content-id-only cache.  A
                # later correlated Platform response reprojects this exact
                # sanitized observation so stateless providers can classify
                # the source material in the decision turn that uses it.
                "public_content_observation": copy.deepcopy(public_content_observations[0]),
                "source_role": sender_role,
            }
            previous = self._inquiry_authority_by_request_id.get(env.msg_id)
            if previous is not None and previous != inquiry_authority:
                raise PlatformContractError(
                    "current inquiry business authority changed across retries"
                )
            self._inquiry_authority_by_request_id[env.msg_id] = inquiry_authority
        if (
            operation_is_registered("respond_inquiry")
            and env.action.get("kind") == "platform.catalog_listing"
        ):
            response_payload = env.action.get("payload")
            request_id = (
                response_payload.get("request_id")
                if isinstance(response_payload, Mapping)
                else None
            )
            inquiry_authority = (
                self._inquiry_authority_by_request_id.get(request_id)
                if isinstance(request_id, str)
                else None
            )
            if inquiry_authority is None:
                raise PlatformContractError(
                    "catalog response has no authenticated inquiry authority"
                )
            context["inquiry_response_authority"] = copy.deepcopy(inquiry_authority)
        prerequisites = self._commerce_action_grounding.resolve_phase(
            inbound_id=env.msg_id,
            in_reply_to=env.in_reply_to,
            action_kind=str(env.action.get("kind", "")),
            payload=inbound_payload,
            registered_routes=registered_routes,
        )
        # One immutable resolver result is the sole authority prerequisite
        # shared by the Agent bridge and semantic compiler.  Domain code must
        # not reconstruct it from independent route, revision, ID, or payload
        # context keys.
        context["grounding_authority"] = prerequisites
        mandate_id = self._semantic_mandate.get("mandate_id")
        if isinstance(mandate_id, str) and mandate_id.strip():
            context["mandate_id"] = mandate_id
        benchmark_contract = (
            self._semantic_mandate.get("benchmark_contract")
            if self.id.startswith("buyer")
            else self._semantic_policy.get("benchmark_contract")
        )
        if isinstance(benchmark_contract, Mapping):
            benchmark_task_id = benchmark_contract.get("task_id")
            if isinstance(benchmark_task_id, str) and benchmark_task_id.strip():
                # Benchmark identity is frozen scenario context.  It may tag a
                # World search session, but a provider must never be asked to
                # guess or echo it.
                context["benchmark_task_id"] = benchmark_task_id
            candidate_search = benchmark_contract.get("candidate_search")
            if (
                isinstance(candidate_search, Mapping)
                and candidate_search.get("scope") == "complete_candidate_set"
                and candidate_search.get("filters") == "forbidden"
            ):
                # Some tasks score comparison across the complete published
                # candidate set.  On those tasks a model-authored search filter
                # would hide candidates before the required World grounding
                # phase and make the public execution contract impossible.
                context["search_filters_allowed"] = False
            constraints = benchmark_contract.get("constraints")
            if isinstance(constraints, list):
                hard_constraint_ids = tuple(
                    constraint_id
                    for row in constraints
                    if isinstance(row, Mapping)
                    and isinstance(
                        constraint_id := row.get("constraint_id"),
                        str,
                    )
                    and constraint_id.strip()
                )
                if hard_constraint_ids:
                    # Constraint identity belongs to the frozen mandate.  The
                    # provider chooses its query and filters, while the Agent
                    # binds the complete authority set to the World search.
                    context["hard_constraint_ids"] = hard_constraint_ids
        after_sales_order_authority = (
            self._public_after_sales_order_authority(
                env=env,
                benchmark_contract=benchmark_contract,
                public_routes=public_routes,
                route_registry=domain_routes,
            )
            if isinstance(benchmark_contract, Mapping)
            else None
        )
        if after_sales_order_authority is not None and not complete_after_sales_phase:
            context["after_sales_order_authority"] = after_sales_order_authority
        principal_id = self._semantic_principal_for(env)
        if principal_id is not None:
            context["principal_id"] = principal_id
        reply_recipient_id = _agent_reply_recipient_authority(
            env,
            benchmark_contract=(
                benchmark_contract if isinstance(benchmark_contract, Mapping) else None
            ),
            lineage_recipient_id=self._semantic_reply_recipient_for(env),
        )
        if reply_recipient_id is not None:
            context["reply_recipient_id"] = reply_recipient_id
        message_payload_authority = (
            None
            if supply_report_route_exclusive
            else _agent_message_payload_authority(
                env,
                benchmark_contract=(
                    benchmark_contract if isinstance(benchmark_contract, Mapping) else None
                ),
                reply_recipient_id=reply_recipient_id,
            )
        )
        if message_payload_authority is not None:
            context["message_payload_authority"] = message_payload_authority
        if self._semantic_search_limit is not None:
            context["search_limit"] = self._semantic_search_limit
        if self.id.startswith("buyer"):
            cart_quote_authority = self._semantic_mandate.get("cart_quote_authority")
            if isinstance(cart_quote_authority, Mapping):
                # This is public mandate authority frozen at Agent spawn.  It
                # is compiler input, not a model-authored protocol payload.
                context["cart_quote_authority"] = copy.deepcopy(dict(cart_quote_authority))
        if env.action.get("kind") == "platform.cart_quote_request":
            if env.from_ != "platform:checkout":
                raise PlatformContractError("cart quote request must come from platform:checkout")
            payload = env.action.get("payload")
            request = payload.get("request") if isinstance(payload, Mapping) else None
            request_id = request.get("request_id") if isinstance(request, Mapping) else None
            request_lines = request.get("lines") if isinstance(request, Mapping) else None
            if (
                not isinstance(request_id, str)
                or not request_id.strip()
                or not isinstance(request_lines, list)
                or not request_lines
                or any(
                    not isinstance(line, Mapping)
                    or not isinstance(line.get("sku_id"), str)
                    or not str(line["sku_id"]).strip()
                    or isinstance(line.get("qty"), bool)
                    or not isinstance(line.get("qty"), int)
                    or int(line["qty"]) <= 0
                    for line in request_lines
                )
            ):
                raise PlatformContractError("Platform cart quote request is malformed")
            if self.id.split(":", 1)[0] == "merchant":
                # Platform owns this request id.  Merchant model output only
                # chooses whether to answer the current request.
                context["cart_quote_request_id"] = request_id
        if env.action.get("kind") == "platform.rank_offers":
            payload = env.action.get("payload")
            if not isinstance(payload, Mapping):
                raise PlatformContractError("Platform ranked response payload is malformed")
            route_set = registered_route_set
            authority_routes = {
                (
                    binding.action_kind,
                    binding.destination,
                )
                for operation in ("accept_ranked_offer", "propose_offer")
                for binding in (domain_routes.by_operation(operation),)
            }
            issued_at_tick = payload.get("issued_at_tick")
            if route_set.intersection(authority_routes) and (
                isinstance(issued_at_tick, bool)
                or not isinstance(issued_at_tick, int)
                or issued_at_tick < 0
            ):
                raise PlatformContractError("Platform ranked response has no valid World tick")
            # Ephemeral authoritative input for compiling a high-level choice.
            # Which compiler is needed is determined by the public route, not
            # by a benchmark family or expected answer.
            if operation_is_registered("accept_ranked_offer"):
                blocked_action_kinds = context.get("task_blocked_action_kinds", ())
                adapter = RankedOfferPhaseAdapter(
                    domain_routes,
                    allow_accept=(
                        "commerce.accept_offer" not in blocked_action_kinds
                    ),
                )
                authority = adapter.build_authority(
                    inbound=env,
                    actor_id=self.id,
                    current_tick=issued_at_tick,
                )
                domain_phases.append(BoundDomainPhase(adapter, authority))
            if operation_is_registered("propose_offer"):
                adapter = NegotiationPhaseAdapter(domain_routes)
                authority = adapter.build_authority(
                    inbound=env,
                    actor_id=self.id,
                    current_tick=issued_at_tick,
                )
                negotiation_authority = authority.value
                domain_phases.append(BoundDomainPhase(adapter, authority))
            context["rank_offers"] = payload
        if env.from_ == "platform:negotiation" and env.action.get("kind") in NEGOTIATION_ACTIONS:
            payload = env.action.get("payload")
            thread = (
                payload.get("world_thread_projection") if isinstance(payload, Mapping) else None
            )
            adapter = NegotiationPhaseAdapter(domain_routes)
            authority = adapter.build_authority(
                inbound=env,
                actor_id=self.id,
                current_thread=thread,
            )
            negotiation_authority = authority.value
            domain_phases.append(BoundDomainPhase(adapter, authority))
        if (
            self.id.startswith("buyer")
            and env.from_ == "platform:negotiation"
            and env.action.get("kind") == "commerce.accept_offer"
        ):
            authority = negotiation_authority
            if authority is None:
                raise PlatformContractError(
                    "accepted negotiation lacks current World thread authority"
                )
            required = (
                authority.negotiation_id,
                authority.offer_id,
                authority.peer_id,
                authority.sku_id,
                authority.currency,
            )
            if (
                any(not isinstance(value, str) or not value for value in required)
                or authority.qty is None
                or authority.current_unit_price is None
            ):
                raise PlatformContractError("accepted negotiation compiler authority is incomplete")
            negotiation_id = str(authority.negotiation_id)
            offer_id = str(authority.offer_id)
            context["negotiated_settlement_authority"] = {
                "payload": {
                    "negotiation_id": negotiation_id,
                    "order_id": negotiated_order_id(negotiation_id, offer_id),
                    "buyer_id": self.id,
                    "merchant_id": str(authority.peer_id),
                    "sku_id": str(authority.sku_id),
                    "qty": int(authority.qty),
                    "agreed_price": {
                        "amount": format(
                            Decimal(int(authority.current_unit_price)) / Decimal(100),
                            ".2f",
                        ),
                        "currency": str(authority.currency),
                    },
                }
            }
        if env.action.get("kind") == "platform.supply_state":
            payload = env.action.get("payload")
            states = payload.get("states") if isinstance(payload, Mapping) else None
            settlement_options = (
                payload.get("purchase_options") if isinstance(payload, Mapping) else None
            )
            if self.id.split(":", 1)[0] == "buyer":
                if (
                    not isinstance(states, list)
                    or not states
                    or not all(isinstance(row, Mapping) for row in states)
                    or not isinstance(settlement_options, list)
                    or not all(isinstance(row, Mapping) for row in settlement_options)
                ):
                    raise PlatformContractError(
                        "Platform supply reply has no complete World authority batch"
                    )
                if any(
                    not isinstance(row.get("sku_id"), str)
                    or not str(row["sku_id"]).strip()
                    or isinstance(row.get("available_qty"), bool)
                    or not isinstance(row.get("available_qty"), int)
                    or int(row["available_qty"]) < 0
                    for row in states
                ):
                    raise PlatformContractError("Platform supply state batch is malformed")
                state_by_sku = {str(row["sku_id"]): row for row in states}
                option_by_sku = {str(row.get("sku_id")): row for row in settlement_options}
                purchasable_skus = {
                    sku_id for sku_id, row in state_by_sku.items() if int(row["available_qty"]) > 0
                }
                if (
                    len(state_by_sku) != len(states)
                    or len(option_by_sku) != len(settlement_options)
                    or purchasable_skus != set(option_by_sku)
                ):
                    raise PlatformContractError(
                        "Platform supply authority batch does not match purchasable supply"
                    )
                for sku_id in purchasable_skus:
                    state = state_by_sku[sku_id]
                    option = option_by_sku[sku_id]
                    text_fields = (
                        "authority_id",
                        "authority_digest",
                        "order_id",
                        "merchant_id",
                        "currency",
                    )
                    integer_fields = (
                        "unit_price_cents",
                        "available_qty",
                        "supply_version",
                        "expires_at_tick",
                    )
                    if (
                        any(
                            not isinstance(option.get(name), str) or not str(option[name]).strip()
                            for name in text_fields
                        )
                        or any(
                            isinstance(option.get(name), bool)
                            or not isinstance(option.get(name), int)
                            or int(option[name]) < 0
                            for name in integer_fields
                        )
                        or option.get("merchant_id") != state.get("merchant_id")
                        or option.get("unit_price_cents") != state.get("unit_price_cents")
                        or option.get("available_qty") != state.get("available_qty")
                        or int(option.get("available_qty", 0)) <= 0
                        or option.get("supply_version") != state.get("version")
                    ):
                        raise PlatformContractError(
                            "Platform supply authority binding is malformed"
                        )
                context["supply_settlement_authority"] = {
                    "buyer_id": self.id,
                    "states": copy.deepcopy(states),
                    "settlement_options": copy.deepcopy(settlement_options),
                }
                if settlement_options:
                    adapter = SupplyPhaseAdapter(domain_routes)
                    authority = adapter.build_authority(
                        buyer_id=self.id,
                        options=settlement_options,
                    )
                    domain_phases.append(BoundDomainPhase(adapter, authority))
            elif self.id.split(":", 1)[0] == "merchant":
                if supply_report_route_exclusive:
                    if (
                        not isinstance(states, list)
                        or not states
                        or not all(isinstance(row, Mapping) for row in states)
                        or not isinstance(reply_recipient_id, str)
                        or not reply_recipient_id.startswith("buyer:")
                    ):
                        raise PlatformContractError(
                            "merchant supply report has no complete participant authority"
                        )
                    adapter = SupplyReportPhaseAdapter(domain_routes)
                    authority = adapter.build_authority(
                        merchant_id=self.id,
                        recipient_id=reply_recipient_id,
                        states=states,
                    )
                    domain_phases.append(BoundDomainPhase(adapter, authority))
        if env.action.get("kind") == "platform.shipment_state":
            payload = env.action.get("payload")
            shipment = payload.get("shipment") if isinstance(payload, Mapping) else None
            shipment_id = shipment.get("shipment_id") if isinstance(shipment, Mapping) else None
            raw_options = (
                payload.get("replacement_options") if isinstance(payload, Mapping) else None
            )
            candidates = (
                [
                    row["sku_id"]
                    for row in raw_options
                    if isinstance(row, Mapping)
                    and isinstance(row.get("sku_id"), str)
                    and row["sku_id"].strip()
                ]
                if isinstance(raw_options, list)
                else None
            )
            if (
                isinstance(shipment_id, str)
                and shipment_id.strip()
                and isinstance(candidates, list)
            ):
                context["shipment_resolution_authority"] = {
                    "shipment_id": shipment_id,
                    "replacement_candidate_sku_ids": copy.deepcopy(candidates),
                }
        if env.action.get("kind") == "platform.create_match_certificate":
            payload = env.action.get("payload")
            cert_id = payload.get("cert_id") if isinstance(payload, Mapping) else None
            if isinstance(cert_id, str) and cert_id.strip():
                # Calling the settlement function is the model's business
                # decision.  The certificate identity itself is authoritative
                # Platform context and is bound by the Agent.
                context["match_certificate_id"] = cert_id
        if env.action.get("kind") == "platform.cart_quote":
            if env.from_ != "platform:checkout":
                raise PlatformContractError("cart quote must come from platform:checkout")
            payload = env.action.get("payload")
            quote = payload.get("quote") if isinstance(payload, Mapping) else None
            quote_id = quote.get("quote_id") if isinstance(quote, Mapping) else None
            if self.id.split(":", 1)[0] == "buyer":
                if not isinstance(quote, Mapping):
                    raise PlatformContractError("Platform cart quote payload is malformed")
                try:
                    sealed_quote = persistent_cart_quote_from_json(
                        json.dumps(
                            dict(quote),
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise PlatformContractError(
                        "Platform cart quote is not a valid sealed World quote"
                    ) from exc
                if sealed_quote.buyer_id != self.id:
                    raise PlatformContractError("Platform cart quote belongs to a different buyer")
                quote_id = sealed_quote.quote_id
                context["cart_quote_id"] = quote_id
        if env.action.get("kind") == "platform.deliver_protocol_event":
            authority_package = self._protocol_event_authority_by_inbound.get(env.msg_id)
            if not isinstance(authority_package, Mapping) or set(authority_package) != {
                "current_event",
                "current_order_state",
                "current_receipts",
                "current_reference_state",
                "prior_event_states",
                "current_tick",
            }:
                raise PlatformContractError(
                    "protocol event turn has no fresh Agent-owned World authority"
                )
            receipts = authority_package["current_receipts"]
            state = authority_package["current_order_state"]
            if not isinstance(receipts, list) or not isinstance(state, Mapping):
                raise PlatformContractError("protocol event World authority package is malformed")
            adapter = ProtocolEventPhaseAdapter(domain_routes)
            authority = adapter.build_authority(
                inbound=env,
                actor_id=self.id,
                current_event=authority_package["current_event"],
                current_order_state=state,
                current_receipts=receipts,
                current_tick=authority_package["current_tick"],
            )
            domain_phases.append(BoundDomainPhase(adapter, authority))
        if complete_after_sales_phase:
            projections = self._after_sales_projections_by_inbound.get(env.msg_id)
            if not projections:
                raise PlatformContractError(
                    "after-sales business phase has no Agent-owned World authority"
                )
            adapter = AfterSalesPhaseAdapter(domain_routes)
            authority = adapter.build_authority(
                inbound=env,
                actor_id=self.id,
                projections=projections,
                allowed_route_operations=tuple(sorted(after_sales_route_operations)),
            )
            domain_phases.append(BoundDomainPhase(adapter, authority))
        if self._governance_projection_cache.has_business_turn(env):
            adapter = GovernancePhaseAdapter(domain_routes)
            authority = adapter.build_authority(
                inbound=env,
                actor_id=self.id,
                projection_cache=self._governance_projection_cache,
            )
            domain_phases.append(BoundDomainPhase(adapter, authority))
        if domain_phases:
            context["domain_phases"] = tuple(domain_phases)
        return context

    def _semantic_task_execution_scope(
        self,
        env: "Envelope",
    ) -> dict[str, Any] | None:
        """Bind one value-free public task contract to the current phase.

        This consumes only the actor's spawn-frozen public ``task_context``.
        It never reads ``success_oracle`` and never supplies an expected answer.
        The contract can restrict an initial turn to a Platform action and can
        bind a later high-level result payload to exactly one Runtime evidence
        action without asking the model to author protocol wrappers.
        """

        mandate_context = self._semantic_mandate.get("task_context")
        policy_context = self._semantic_policy.get("task_context")
        if (
            mandate_context is not None
            and policy_context is not None
            and mandate_context != policy_context
        ):
            raise ValueError("mandate and policy contain conflicting public task context")
        task_context = mandate_context if mandate_context is not None else policy_context
        execution = (
            task_context.get("execution_contract") if isinstance(task_context, Mapping) else None
        )
        if not isinstance(execution, Mapping):
            return None
        if execution.get("schema_version") != "cwe.public-task-execution.v1":
            raise ValueError("public task execution contract has an unsupported schema")
        if "phases" in execution:
            contract = validate_public_task_execution_contract(execution)
            phase = resolve_public_task_phase(
                contract,
                actor_id=self.id,
                inbound=env,
            )
            output = phase_actor_context(phase)
            claim_templates = task_context.get("claim_compilation_templates")
            if claim_templates is not None:
                if not isinstance(claim_templates, Mapping):
                    raise ValueError("private claim compilation templates must be an object")
                output["claim_compilation_templates"] = copy.deepcopy(dict(claim_templates))
                authorized_claim_ids = task_context.get("claim_ids")
                if (
                    not isinstance(authorized_claim_ids, list)
                    or not authorized_claim_ids
                    or any(
                        not isinstance(claim_id, str) or not claim_id.strip()
                        for claim_id in authorized_claim_ids
                    )
                    or len(authorized_claim_ids) != len(set(authorized_claim_ids))
                ):
                    raise ValueError(
                        "private claim compilation binding has no authorized claim set"
                    )
                output["claim_compilation_authorized_claim_ids"] = tuple(authorized_claim_ids)
            output["task_execution_contract_sha256"] = public_task_execution_contract_digest(
                execution
            )
            return output
        decision = execution.get("decision_evidence")
        if not isinstance(decision, Mapping):
            raise ValueError("public task execution contract has no decision evidence")
        triggers = decision.get("trigger_action_kinds")
        payload_schema = decision.get("payload_schema")
        result_format = decision.get("result_format", "named_submission")
        submission_kind = decision.get("submission_kind")
        result_reads_allowed = decision.get("world_reads_allowed", True)
        result_action_exclusive = decision.get("exclusive_action", False)
        if (
            not isinstance(triggers, list)
            or not triggers
            or any(not isinstance(value, str) or not value.strip() for value in triggers)
            or len(triggers) != len(set(triggers))
            or decision.get("endpoint") != "runtime:evidence"
            or decision.get("action_kind")
            not in {"commerce.submit_decision_record", "delegate.report_result"}
            or result_format not in {"named_submission", "direct_details"}
            or (
                result_format == "named_submission"
                and (not isinstance(submission_kind, str) or not submission_kind.strip())
            )
            or (result_format == "direct_details" and submission_kind is not None)
            or not isinstance(result_reads_allowed, bool)
            or not isinstance(result_action_exclusive, bool)
            or not isinstance(payload_schema, Mapping)
        ):
            raise ValueError("public task decision-evidence contract is malformed")
        current_kind = str(env.action.get("kind", ""))
        output: dict[str, Any] = {
            "task_result_contract": {
                "active": current_kind in triggers,
                "action_kind": str(decision["action_kind"]),
                "endpoint": "runtime:evidence",
                "result_format": str(result_format),
                "submission_kind": submission_kind,
                "payload_schema": copy.deepcopy(dict(payload_schema)),
            }
        }
        if current_kind in triggers:
            output["task_world_reads_allowed"] = result_reads_allowed
            if result_action_exclusive:
                output["task_allowed_action_kinds"] = (str(decision["action_kind"]),)
        initial = execution.get("initial_phase")
        if initial is not None:
            if not isinstance(initial, Mapping):
                raise ValueError("public task initial-phase contract is malformed")
            trigger = initial.get("trigger_action_kind")
            required = initial.get("required_business_action_kind")
            reads_allowed = initial.get("world_reads_allowed")
            if (
                not isinstance(trigger, str)
                or not trigger.strip()
                or not isinstance(required, str)
                or not required.strip()
                or not isinstance(reads_allowed, bool)
            ):
                raise ValueError("public task initial-phase contract is malformed")
            if current_kind == trigger:
                output["task_allowed_action_kinds"] = (required,)
                output["task_world_reads_allowed"] = reads_allowed
        return output

    def _semantic_principal_for(self, inbound: "Envelope") -> str | None:
        """Resolve the principal on the exact registered message lineage."""

        direct = self._semantic_root_principals.get(inbound.msg_id)
        if direct is not None:
            return direct
        if inbound.in_reply_to is None:
            return None
        return self._semantic_principal_by_outbound_msg_id.get(inbound.in_reply_to)

    def _semantic_reply_recipient_for(self, inbound: "Envelope") -> str | None:
        """Resolve the participant awaiting a reply across a Platform hop."""

        sender_role = inbound.from_.split(":", 1)[0]
        if sender_role in {"buyer", "merchant"} and inbound.from_ != self.id:
            return inbound.from_
        if inbound.in_reply_to is None:
            return None
        return self._semantic_reply_recipient_by_outbound_msg_id.get(inbound.in_reply_to)

    def _bootstrap_semantic_memory(self, history: list[dict[str, Any]]) -> None:
        """Deterministically project spawn authority inputs into Agent memory."""

        writes: list[tuple[MemoryType, str, Any]] = []

        def stage(bucket: MemoryType, key: str, value: Any) -> None:
            if value is not None and self.memory.read(bucket, key) is None:
                writes.append((bucket, key, copy.deepcopy(value)))

        if self.id.startswith("buyer"):
            mandate = self._semantic_mandate
            hard = mandate.get("hard_constraints")
            hard = hard if isinstance(hard, Mapping) else {}
            authority = mandate.get("authority")
            authority = authority if isinstance(authority, Mapping) else {}
            soft_preferences = mandate.get("soft_preferences")
            soft_preferences = soft_preferences if isinstance(soft_preferences, Mapping) else {}
            stage(MemoryType.PREFERENCE, "goal", mandate.get("goal"))
            stage(MemoryType.PREFERENCE, "must_have", hard.get("must_have", []))
            stage(MemoryType.PREFERENCE, "delivery_days", hard.get("delivery_days"))
            stage(MemoryType.PREFERENCE, "soft_constraints", mandate.get("soft_constraints", []))
            stage(MemoryType.PREFERENCE, "style", soft_preferences.get("style", []))
            stage(MemoryType.PREFERENCE, "avoid_styles", soft_preferences.get("avoid", []))
            stage(MemoryType.TRANSACTION, "mandate_id", mandate.get("mandate_id"))
            stage(MemoryType.TRANSACTION, "intent_expiry", mandate.get("intent_expiry"))
            stage(MemoryType.TRANSACTION, "cumulative_spend", 0)
            stage(
                MemoryType.TRANSACTION,
                "return_after_purchase",
                mandate.get("return_after_purchase", False),
            )
            stage(MemoryType.PRIVATE_UTILITY, "max_budget", hard.get("budget"))
            stage(
                MemoryType.PRIVATE_UTILITY,
                "must_not_share_list",
                authority.get("must_not_share_with_merchant", []),
            )
            stage(
                MemoryType.PRIVATE_UTILITY,
                "can_buy_without_confirmation",
                authority.get("can_buy_without_confirmation"),
            )
        elif self.id.startswith("merchant"):
            policy = self._semantic_policy
            stage(MemoryType.PRIVATE_UTILITY, "floor_price", policy.get("floor_price"))
            stage(MemoryType.PRIVATE_UTILITY, "min_acceptable_price", policy.get("floor_price"))
            stage(MemoryType.PREFERENCE, "margin_target_bps", policy.get("margin_target_bps"))
            stage(
                MemoryType.PREFERENCE,
                "max_negotiation_rounds",
                policy.get("max_negotiation_rounds"),
            )

        if not writes:
            return
        for bucket, key, value in writes:
            self.memory.write(bucket, key, value)
        history.append(
            {
                "step": "memory_update",
                "source": "scenario_authority",
                "writes": [
                    {"bucket": bucket.value, "key": key, "value": value}
                    for bucket, key, value in writes
                ],
            }
        )

    def _business_decide(
        self,
        env: "Envelope",
        ctx: "AgentContext",
        history: list[dict[str, Any]],
        *,
        activated_skill_names: tuple[str, ...] | None,
        allow_actor_reports: bool,
        include_finish: bool,
        decision_id: str,
    ) -> _BusinessPhaseDecision:
        """Build one composite phase and request one typed business choice."""

        role = self.id.split(":", 1)[0]
        actor_context = self._semantic_actor_context(env, history=history)
        task_allowed = actor_context.get("task_allowed_action_kinds")
        allowed_action_kinds = self.outbound_action_kinds
        if isinstance(task_allowed, (list, tuple, frozenset)):
            allowed_action_kinds = self.outbound_action_kinds & frozenset(
                str(value) for value in task_allowed
            )
        task_blocked = actor_context.get("task_blocked_action_kinds")
        if isinstance(task_blocked, (list, tuple, frozenset)):
            allowed_action_kinds = allowed_action_kinds - frozenset(
                str(value) for value in task_blocked
            )
        active_allowed_routes = actor_context.get("task_allowed_routes")
        if (
            isinstance(active_allowed_routes, (list, tuple, frozenset))
            and isinstance(task_blocked, (list, tuple, frozenset))
        ):
            blocked = frozenset(str(value) for value in task_blocked)
            active_allowed_routes = tuple(
                route
                for route in active_allowed_routes
                if not (
                    isinstance(route, (list, tuple))
                    and len(route) == 2
                    and route[0] in blocked
                )
            )
        generic_intent_context = copy.deepcopy(dict(actor_context))
        if "domain_phases" in generic_intent_context:
            # Domain ownership belongs exclusively to CompositePhaseAdapter.
            # The retained generic catalogue must not also reverse-match a
            # route and specialize the same choice through the Agent-private
            # business-intent registry.
            generic_intent_context.pop("domain_phases", None)
        intent_specs = DEFAULT_AGENT_ROUTE_REGISTRY.intent_specs_for(
            role,
            allowed_action_kinds,
            include_world_reads=(actor_context.get("task_world_reads_allowed") is not False),
            activated_skill_names=activated_skill_names,
            allow_actor_reports=allow_actor_reports,
            include_finish=include_finish,
            task_result_contract=actor_context.get("task_result_contract"),
            allowed_routes=active_allowed_routes,
            bind_cart_quote_authority=(
                isinstance(actor_context.get("cart_quote_authority"), Mapping)
                or (
                    isinstance(actor_context.get("cart_quote_request_id"), str)
                    and bool(str(actor_context["cart_quote_request_id"]).strip())
                )
            ),
            actor_context=generic_intent_context,
        )
        if getattr(ctx, "remote_world", False):
            intent_specs = tuple(
                intent_spec
                for intent_spec in intent_specs
                if intent_spec.category != "read"
                or (
                    (
                        world_tool := DEFAULT_AGENT_ROUTE_REGISTRY.world_read_tool_name(
                            intent_spec.name
                        )
                    )
                    is not None
                    and reentrant_world_tool_allowed(world_tool, actor_id=self.id)
                )
            )
        return self._business_semantic_decide(
            env,
            history,
            intent_specs=tuple(intent_specs),
            actor_context=actor_context,
            allowed_action_kinds=allowed_action_kinds,
            activated_skill_names=activated_skill_names,
            allow_actor_reports=allow_actor_reports,
            decision_id=decision_id,
        )

    def _business_semantic_decide(
        self,
        env: "Envelope",
        history: list[dict[str, Any]],
        *,
        intent_specs: tuple[Any, ...],
        actor_context: Mapping[str, Any],
        allowed_action_kinds: frozenset[str],
        activated_skill_names: tuple[str, ...] | None,
        allow_actor_reports: bool,
        decision_id: str,
    ) -> _BusinessPhaseDecision:
        """Run one provider-neutral business decision through Agent bridge."""

        generic_adapter = AgentRoutePhaseAdapter(DEFAULT_AGENT_ROUTE_REGISTRY)
        adapter = CompositePhaseAdapter(generic_adapter)
        business_actor_context = copy.deepcopy(dict(actor_context))
        raw_domain_phases = business_actor_context.pop("domain_phases", ())
        if not isinstance(raw_domain_phases, tuple) or any(
            not isinstance(row, BoundDomainPhase) for row in raw_domain_phases
        ):
            raise RuntimeError("Agent domain phase inventory is malformed")
        if any(
            DEFAULT_AGENT_ROUTE_REGISTRY.business_operation_for_source(intent_spec.name) == "search"
            for intent_spec in intent_specs
        ):
            business_actor_context.setdefault(
                "search_limit",
                self._semantic_search_limit or 10,
            )
        authority = adapter.build_authority(
            intent_specs=intent_specs,
            role=self.id.split(":", 1)[0],
            allowed_action_kinds=allowed_action_kinds,
            activated_skill_names=activated_skill_names,
            allow_actor_reports=allow_actor_reports,
            actor_context=business_actor_context,
            domain_phases=raw_domain_phases,
        )
        specs = adapter.decision_schema(authority)
        route_bindings = adapter.route_bindings(authority)
        public_observations = public_business_observations(
            inbound_kind=str(env.action.get("kind", "business_event")),
            inbound_payload=(
                env.action.get("payload") if isinstance(env.action.get("payload"), Mapping) else {}
            ),
            history=history,
            phase_id=str(actor_context.get("task_phase_id") or "business_turn"),
        )
        protocol_event_observation = self._protocol_event_business_observation(env)
        if protocol_event_observation is not None:
            public_observations = (
                *public_observations,
                protocol_event_observation,
            )
        task_facts = public_actor_task_facts(
            self._semantic_mandate if self.id.startswith("buyer") else self._semantic_policy,
            phase_id=str(actor_context.get("task_phase_id") or "business_turn"),
        )
        if task_facts:
            public_observations = (
                *public_observations,
                {"persistent_task_business_facts": task_facts},
            )
        retained_content = tuple(
            copy.deepcopy(row)
            for inbound_id, row in self._persistent_public_content_by_inbound.items()
            if inbound_id != env.msg_id
        )
        if retained_content:
            public_observations = (
                *public_observations,
                {
                    "retained_untrusted_business_content": list(retained_content),
                },
            )
        if self._public_business_choice_history:
            public_observations = (
                *public_observations,
                {
                    "prior_validated_business_choices": copy.deepcopy(
                        self._public_business_choice_history
                    )
                },
            )
        inquiry_authority = actor_context.get("inquiry_response_authority")
        if isinstance(inquiry_authority, Mapping):
            content_observation = inquiry_authority.get("public_content_observation")
            source_role = inquiry_authority.get("source_role")
            if isinstance(content_observation, Mapping) and isinstance(source_role, str):
                public_observations = (
                    *public_observations,
                    {
                        "correlated_inquiry_business_content": {
                            "source_role": source_role,
                            "provenance": "authenticated_inbound_business_message",
                            "observation": copy.deepcopy(dict(content_observation)),
                        }
                    },
                )
        contract = AgentPhaseContract(
            phase_id=str(actor_context.get("task_phase_id") or "business_turn"),
            role=self.id.split(":", 1)[0],
            goal=self._business_decision_goal(),
            observations=public_observations,
            authority=authority,
            adapter=adapter,
            allowed_operations=frozenset(row.intent for row in specs),
            route_bindings=route_bindings,
            framework_observations=(
                {
                    "inbound_msg_id": env.msg_id,
                    **(
                        {"recipient_id": business_actor_context["reply_recipient_id"]}
                        if isinstance(
                            business_actor_context.get("reply_recipient_id"),
                            str,
                        )
                        and str(business_actor_context["reply_recipient_id"]).strip()
                        else {}
                    ),
                },
                copy.deepcopy(
                    dict(env.action.get("payload"))
                    if isinstance(env.action.get("payload"), Mapping)
                    else {}
                ),
                copy.deepcopy(business_actor_context),
            ),
        )
        bridge = AgentDecisionBridge()
        request = bridge.request(contract, decision_id=decision_id)
        _scope_chars, observation_scope_sha256 = canonical_semantic_digest(
            {
                "phase_id": contract.phase_id,
                "inbound_action_kind": str(env.action.get("kind", "")),
                "inbound_payload": (
                    copy.deepcopy(dict(env.action["payload"]))
                    if isinstance(env.action.get("payload"), Mapping)
                    else {}
                ),
            }
        )
        existing_public_steps = [
            row for row in history if row.get("step") == "public_task_execution"
        ]
        public_step = self._public_task_trace_step(
            actor_context,
            decision_id=decision_id,
            phase_index=len(existing_public_steps),
            business_surface_sha256=request.public_sha256,
        )
        if public_step is not None:
            if any(
                row.get("phase_id") != public_step["phase_id"]
                or row.get("execution_contract_sha256") != public_step["execution_contract_sha256"]
                or row.get("decision_id") != decision_id
                for row in existing_public_steps
            ):
                raise RuntimeError("public business phase identity changed within one Agent turn")
            history.append(public_step)

        response = self._complete_business_decision_with_transport_retry(
            system_prompt=self._business_decision_system_prompt(),
            user_prompt=request.to_prompt(),
            decision_id=decision_id,
            history=history,
        )
        if not isinstance(response, BusinessDecisionResponseV1):
            raise ModelDecisionParseError(
                "business channel returned an unsupported response object"
            )
        try:
            bridged_decision = bridge.parse_with_provenance(
                self._business_decision_response_content(response.content),
                contract,
            )
            business_decision = bridged_decision.decision
            compiled = bridge.compile(contract, business_decision)
            arguments_sha256 = bridged_decision.resolved_arguments_sha256
            call_id = (
                "business:"
                + hashlib.sha256(
                    (
                        f"{decision_id}:{len(existing_public_steps)}:"
                        f"{business_decision.intent}:{arguments_sha256}:"
                        f"{response.response_sha256}"
                    ).encode("utf-8")
                ).hexdigest()[:24]
            )
        except ModelBusinessDecisionError as exc:
            raise ModelDecisionParseError(
                str(exc),
                response_chars=response.response_chars,
                response_sha256=response.response_sha256,
            ) from exc
        except BusinessDecisionContractError as exc:
            raise ModelDecisionParseError(
                str(exc),
                response_chars=response.response_chars,
                response_sha256=response.response_sha256,
            ) from exc
        phase_decision = _BusinessPhaseDecision(
            compiled=compiled,
            intent=business_decision.intent,
            arguments=copy.deepcopy(dict(business_decision.arguments)),
            message=business_decision.message,
            source_name=compiled.source_name,
            call_id=call_id,
            response_chars=response.response_chars,
            response_sha256=response.response_sha256,
            model_choice=bridged_decision.model_choice,
            agent_argument_binding=bridged_decision.argument_binding_dict(),
            observation_scope_sha256=observation_scope_sha256,
        )
        self._remember_public_business_choice(phase_decision.model_choice)
        return phase_decision

    def _remember_public_business_choice(
        self,
        choice: ModelBusinessChoiceV1,
    ) -> None:
        """Retain one bounded platform-neutral choice for later requests."""

        def without_integrity_metadata(value: Any) -> Any:
            if isinstance(value, Mapping):
                output: dict[str, Any] = {}
                for raw_key, item in value.items():
                    key = str(raw_key)
                    normalized = key.casefold()
                    if normalized in {
                        "digest",
                        "fingerprint",
                        "hash",
                    } or normalized.endswith(
                        ("_digest", "_fingerprint", "_hash", "_sha256")
                    ):
                        continue
                    output[key] = without_integrity_metadata(item)
                return output
            if isinstance(value, list):
                return [without_integrity_metadata(item) for item in value]
            return copy.deepcopy(value)

        arguments = without_integrity_metadata(choice.arguments)
        if not isinstance(arguments, Mapping):
            raise RuntimeError("public business choice arguments lost their object shape")
        self._public_business_choice_history.append(
            {
                "intent": choice.intent,
                "arguments": dict(arguments),
            }
        )
        if len(self._public_business_choice_history) > _MAX_PUBLIC_BUSINESS_CHOICE_HISTORY:
            del self._public_business_choice_history[
                : -_MAX_PUBLIC_BUSINESS_CHOICE_HISTORY
            ]

    def _complete_business_decision_with_transport_retry(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str,
        history: list[dict[str, Any]],
    ) -> BusinessDecisionResponseV1:
        """Request one exact decision with the bounded Agent retry policy."""

        complete = getattr(self.channel, "complete_business_decision", None)
        if not callable(complete):
            raise TypeError("business-decision channel has no completion method")
        retry_index = 0
        while True:
            try:
                return complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    decision_id=decision_id,
                )
            except ChannelTransportError as exc:
                error_code = getattr(exc, "error_code", None)
                if (
                    error_code not in _RETRYABLE_BUSINESS_TRANSPORT_CODES
                    or retry_index
                    >= _MAX_BUSINESS_TRANSPORT_RETRIES_PER_DECISION
                ):
                    raise
                retry_index += 1
                self._provider_retry_sleeper(
                    _BUSINESS_TRANSPORT_RETRY_BACKOFF_SECONDS[retry_index - 1]
                )
                history.append(
                    {
                        "step": "semantic_transport_retry",
                        "interface": "business_decision",
                        "error_code": str(error_code),
                        "retry_index": retry_index,
                        "same_decision": True,
                    }
                )

    def _protocol_event_business_observation(
        self,
        env: "Envelope",
    ) -> Mapping[str, Any] | None:
        """Project fresh callback authority into platform-neutral facts."""

        authority = self._protocol_event_authority_by_inbound.get(env.msg_id)
        if authority is None:
            return None
        event = authority.get("current_event")
        state = authority.get("current_order_state")
        receipts = authority.get("current_receipts")
        reference = authority.get("current_reference_state")
        prior_event_states = authority.get("prior_event_states")
        tick = authority.get("current_tick")
        if (
            not isinstance(event, Mapping)
            or not isinstance(state, Mapping)
            or not isinstance(receipts, list)
            or not isinstance(reference, Mapping)
            or not isinstance(prior_event_states, list)
            or isinstance(tick, bool)
            or not isinstance(tick, int)
        ):
            raise PlatformContractError(
                "fresh protocol-event authority cannot form business observations"
            )
        event_id = event.get("event_id")
        prior: list[dict[str, Any]] = []
        same_event_already_decided = False
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                raise PlatformContractError("fresh protocol-event receipt history is malformed")
            receipt_event_id = receipt.get("event_id")
            same_event_already_decided = bool(
                same_event_already_decided
                or (isinstance(event_id, str) and receipt_event_id == event_id)
            )
        for prior_event in prior_event_states:
            if not isinstance(prior_event, Mapping):
                raise PlatformContractError("fresh prior callback history is malformed")
            prior.append(
                {
                    "decision": prior_event.get("decision"),
                    "observed_order_state": prior_event.get("observed_order_state"),
                    "logical_tick": prior_event.get("logical_tick"),
                    "event_kind": prior_event.get("event_kind"),
                    "reference_kind": prior_event.get("reference_kind"),
                    "same_business_reference": prior_event.get("same_business_reference"),
                    "required_order_state": prior_event.get("required_order_state"),
                    "same_required_state_snapshot": bool(
                        prior_event.get("required_order_state") == event.get("required_order_state")
                        and prior_event.get("required_state_revision")
                        == event.get("required_state_revision")
                    ),
                }
            )
        return {
            "current_callback_business_state": {
                "event_kind": event.get("event_kind"),
                "sequence": event.get("sequence"),
                "required_order_state": event.get("required_order_state"),
                "reference_kind": event.get("reference_kind"),
                "authorization_valid_from_tick": reference.get("valid_from_tick"),
                "authorization_valid_until_tick": reference.get("valid_until_tick"),
                "issued_at_tick": event.get("issued_at_tick"),
                "expires_at_tick": event.get("expires_at_tick"),
                "current_order_state": state.get("state"),
                "required_state_snapshot_is_current": bool(
                    event.get("required_state_revision") == state.get("state_revision")
                ),
                "current_tick": tick,
                "same_event_already_decided": same_event_already_decided,
                "prior_callback_decisions": prior,
            }
        }

    def _business_decision_goal(self) -> str:
        source = self._semantic_mandate if self.id.startswith("buyer") else self._semantic_policy
        for key in ("goal", "objective", "instruction"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
        context = source.get("task_context")
        if isinstance(context, Mapping):
            for key in ("goal", "objective", "instruction"):
                value = context.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return "Make the best permitted commerce decision for the current actor."

    def _business_decision_system_prompt(self) -> str:
        """Apply an explicit task-declared response mode without global drift."""

        source = self._semantic_mandate if self.id.startswith("buyer") else self._semantic_policy
        benchmark = source.get("benchmark_contract")
        protocol = benchmark.get("model_action_protocol") if isinstance(benchmark, Mapping) else None
        if isinstance(protocol, Mapping) and protocol.get("response_mode") == "strict_json_only":
            return (
                BUSINESS_DECISION_SYSTEM_PROMPT_V1
                + "\nPerform all reasoning silently. Never reveal calculations or analysis. "
                "Your complete response must be only the JSON object, with no Markdown fence "
                "and no text before or after it."
            )
        return BUSINESS_DECISION_SYSTEM_PROMPT_V1

    def _business_decision_response_content(self, raw: str) -> str:
        """Apply one task-declared, ambiguity-rejecting JSON extraction mode."""

        source = self._semantic_mandate if self.id.startswith("buyer") else self._semantic_policy
        benchmark = source.get("benchmark_contract")
        mode = (
            benchmark.get("response_extraction_mode")
            if isinstance(benchmark, Mapping)
            else None
        )
        stripped = raw.strip()
        if (
            mode != "single_fenced_json_fallback"
            or stripped.startswith("{")
            or stripped.startswith("```")
        ):
            return raw
        matches = re.findall(
            r"```(?:json)?[ \t]*\r?\n(\{.*?\})[ \t]*\r?\n```",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return matches[0] if len(matches) == 1 else raw

    @staticmethod
    def _business_action_for(
        decision: _BusinessPhaseDecision,
    ) -> SemanticAction:
        compiled = decision.compiled
        if not isinstance(compiled, CompiledBusinessAction):
            raise RuntimeError("business decision is not a compiled action")
        rationale = next(
            (
                str(value).strip()
                for value in (
                    decision.message,
                    decision.arguments.get("decision_rationale"),
                    decision.arguments.get("reason"),
                )
                if isinstance(value, str) and value.strip()
            ),
            None,
        )
        return SemanticAction(
            destination=compiled.destination,
            action_kind=compiled.action_kind,
            payload=copy.deepcopy(dict(compiled.payload)),
            decision_rationale=rationale,
        )

    def _record_business_read_request(
        self,
        history: list[dict[str, Any]],
        decision: _BusinessPhaseDecision,
    ) -> None:
        compiled = decision.compiled
        if not isinstance(compiled, CompiledWorldRead):
            raise RuntimeError("business decision is not a compiled World read")
        arguments_chars, arguments_digest = canonical_semantic_digest(dict(decision.arguments))
        compiled_args_chars, compiled_args_digest = canonical_semantic_digest(dict(compiled.args))
        history.append(
            {
                "step": "semantic_read_request",
                "interface": "business_decision",
                "function": decision.source_name,
                "business_intent": decision.intent,
                "compiler_validated": True,
                "arguments_chars": arguments_chars,
                "arguments_sha256": arguments_digest,
                "model_business_choice": decision.model_choice.to_dict(),
                "agent_argument_binding": copy.deepcopy(dict(decision.agent_argument_binding)),
                "response_chars": decision.response_chars,
                "response_sha256": decision.response_sha256,
                "compiled_world_read": {
                    "tool": compiled.tool,
                    "args_chars": compiled_args_chars,
                    "args_sha256": compiled_args_digest,
                },
            }
        )

    def _record_business_finish(
        self,
        history: list[dict[str, Any]],
        decision: _BusinessPhaseDecision,
    ) -> None:
        if not isinstance(decision.compiled, LocalControlDecision):
            raise RuntimeError("business decision is not local control")
        chars, digest = canonical_semantic_digest(dict(decision.arguments))
        history.append(
            {
                "step": "semantic_finish",
                "interface": "business_decision",
                "function": decision.source_name,
                "arguments_chars": chars,
                "arguments_sha256": digest,
            }
        )

    def _record_business_validation_error(
        self,
        history: list[dict[str, Any]],
        decision: _BusinessPhaseDecision,
        exc: ModelDecisionParseError,
    ) -> None:
        chars, digest = canonical_semantic_digest(dict(decision.arguments))
        error_code, error_message = business_repair_error_v1(exc)
        history.append(
            {
                "step": "semantic_validation_error",
                "interface": "business_decision",
                "error": error_message,
                "error_code": error_code,
                "calls": [
                    {
                        "function": decision.source_name,
                        "arguments_chars": chars,
                        "arguments_sha256": digest,
                    }
                ],
                "instruction": (
                    "Choose a valid supplied business intent and correct its "
                    "arguments. Do not repeat the rejected choice unchanged."
                ),
            }
        )

    def _claim_business_repair(self) -> bool:
        """Consume the Agent-lifetime repair shared by all business failures."""

        if self._business_repairs_used >= _MAX_BUSINESS_REPAIRS:
            return False
        self._business_repairs_used += 1
        return True

    @staticmethod
    def _model_grounding_decision_error(
        decision: _BusinessPhaseDecision,
        error: GroundingRequired,
    ) -> ModelDecisionParseError:
        """Bind a rejected model-selected purchase to provider provenance.

        ``GroundingRequired`` also protects legacy and framework paths, so it
        remains an unscoreable generic Agent exception everywhere else.  Only
        the typed business loop knows that the guarded action came directly
        from this model response and can safely offer the one Agent-lifetime
        correction before emitting a model-attributed protocol terminal.
        """

        del error
        return ModelDecisionParseError(
            "business decision requires public listing grounding before acceptance",
            response_chars=decision.response_chars,
            response_sha256=decision.response_sha256,
        )

    @staticmethod
    def _semantic_read_batch_fingerprint(
        calls: tuple[ToolCall, ...],
        *,
        decision: _BusinessPhaseDecision,
    ) -> str:
        """Hash effective read identities independent of provider ordering.

        Fingerprints use the compiled World tool rather than the provider
        function name.  Omitted arguments are filled with the WorldTools
        defaults that affect behavior.  Provider order remains untouched for
        execution, while sorted identities prevent reordering or default
        spelling from bypassing the per-turn distinct-batch bound.
        """

        identities: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        for call in calls:
            args = dict(call.args)
            if call.tool == "world.search_catalog":
                args.setdefault("filters", {})
                args.setdefault("limit", 10)
            elif call.tool == "world.is_in_stock":
                args.setdefault("qty", 1)
            identity = {"tool": call.tool, "args": args}
            _identity_chars, identity_digest = canonical_semantic_digest(identity)
            if identity_digest in seen:
                raise ModelDecisionParseError(
                    "business decision contains a duplicate World read identity",
                    response_chars=decision.response_chars,
                    response_sha256=decision.response_sha256,
                )
            seen.add(identity_digest)
            identities.append((identity_digest, identity))
        identities.sort(key=lambda row: row[0])
        _chars, digest = canonical_semantic_digest(
            {
                "calls": [identity for _digest, identity in identities],
            }
        )
        return digest

    def _record_business_response_error(
        self,
        history: list[dict[str, Any]],
        exc: ModelDecisionParseError,
    ) -> None:
        """Record one provider-response repair without retaining its body."""

        error_code, error_message = business_repair_error_v1(exc)
        row: dict[str, Any] = {
            "step": "semantic_response_error",
            "interface": "business_decision",
            "error": error_message,
            "response_chars": exc.response_chars,
            "response_sha256": exc.response_sha256,
            "instruction": (
                "The previous response was rejected. Return exactly one allowed "
                "business decision as strict JSON now; do not include prose."
            ),
        }
        row["error_code"] = error_code
        history.append(row)

    def _emit_semantic_action(
        self,
        semantic: SemanticAction,
        *,
        inbound: "Envelope",
        history: list[dict[str, Any]],
        actor_reports_allowed: bool,
        decision: _BusinessPhaseDecision,
        observation_choice_fingerprint: str | None,
    ) -> "Envelope":
        """Compile one action and stage its state for exact Runtime commit."""

        decision_name = decision.source_name
        if not isinstance(decision_name, str) or not decision_name.strip():
            raise RuntimeError("compiled business action lost its decision name")
        function_arguments = dict(decision.arguments)
        semantic = self._bind_semantic_actor_context(semantic)
        history_mark = len(history)
        provisional = self._prepare_semantic_decision_memory(
            semantic,
            inbound=inbound,
            history=history,
        )
        arguments_chars, arguments_digest = canonical_semantic_digest(function_arguments)
        payload_chars, payload_digest = canonical_semantic_digest(dict(semantic.payload))
        try:
            try:
                destination = resolve_semantic_destination(
                    semantic,
                    inbound_sender=inbound.from_,
                )
            except SemanticDecisionError as exc:
                raise InvalidAgentEnvelope(str(exc)) from exc
            semantic_entry = {
                "step": "semantic_action",
                "interface": "business_decision",
                "function": decision_name,
                "business_intent": decision.intent,
                # This proves only that the typed business choice passed the
                # Agent compiler. Runtime/Platform/World evidence decides
                # whether it was accepted or changed commerce state.
                "compiler_validated": True,
                "observation_class": semantic_action_observation_class(
                    semantic,
                    resolved_destination=destination,
                ),
                "expects_platform_exchange": destination.startswith("platform:"),
                "arguments_chars": arguments_chars,
                "arguments_sha256": arguments_digest,
                "model_business_choice": decision.model_choice.to_dict(),
                "agent_argument_binding": copy.deepcopy(dict(decision.agent_argument_binding)),
                "response_chars": decision.response_chars,
                "response_sha256": decision.response_sha256,
                "compiled_vcp": {
                    "to": destination,
                    "action_kind": semantic.action_kind,
                    "payload_chars": payload_chars,
                    "payload_sha256": payload_digest,
                    "payload_projection": _safe_compiled_payload_projection(semantic.payload),
                },
            }
            try:
                out = self._build_outbound(
                    {
                        "to": destination,
                        "action": {
                            "kind": semantic.action_kind,
                            "payload": dict(semantic.payload),
                        },
                    },
                    inbound,
                    history,
                )
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise InvalidAgentEnvelope(str(exc)) from exc
                raise
            # A deterministic Agent guard may reject the compiled candidate
            # before it becomes an outbound envelope.  Record the compiled
            # action only after those guards pass so a failed Tracker terminal
            # never claims an action that did not reach the Runtime boundary.
            history.append(semantic_entry)
        except Exception:
            # The candidate never crossed the Agent boundary.  Its temporary
            # semantic memory and compiled-action trace entries are not facts
            # about the failed turn; retain only the validation error added by
            # the caller after rollback.
            del history[history_mark:]
            raise
        finally:
            # The selected-offer write exists only while deterministic guards
            # inspect the candidate. It must not be observable by another turn
            # until Runtime audits the exact envelope below.
            self._restore_provisional_memory(provisional)

        final_payload = out.action.get("payload")
        if isinstance(final_payload, Mapping):
            chars, digest = canonical_semantic_digest(dict(final_payload))
            semantic_entry["compiled_vcp"]["payload_chars"] = chars
            semantic_entry["compiled_vcp"]["payload_sha256"] = digest
            semantic_entry["compiled_vcp"]["payload_projection"] = (
                _safe_compiled_payload_projection(final_payload)
            )
        reply_recipient_id = self._semantic_reply_recipient_for(inbound)
        if semantic.action_kind == "commerce.send_message" and destination.split(":", 1)[0] in {
            "buyer",
            "merchant",
        }:
            reply_recipient_id = destination
        self._stage_provisional_outbound(
            out,
            memory_update=provisional,
            principal_id=self._semantic_principal_for(inbound),
            reply_recipient_id=reply_recipient_id,
            actor_reports_allowed=actor_reports_allowed,
            observation_choice_fingerprint=observation_choice_fingerprint,
        )
        return out

    def _platform_observation_repeat_guard(
        self,
        history: list[dict[str, Any]],
        decision: _BusinessPhaseDecision,
    ) -> tuple[str | None, bool]:
        """Bound identical Platform observations across continuation turns."""

        compiled = decision.compiled
        if (
            not isinstance(compiled, CompiledBusinessAction)
            or not compiled.destination.startswith("platform:")
            or not compiled.action_kind.startswith("commerce.read_")
        ):
            return None, False
        _chars, fingerprint = canonical_semantic_digest(
            {
                "observation_scope_sha256": decision.observation_scope_sha256,
                "business_intent": decision.intent,
                "action_kind": compiled.action_kind,
                "destination": compiled.destination,
                "payload": copy.deepcopy(dict(compiled.payload)),
            }
        )
        if fingerprint not in self._committed_observation_choice_fingerprints:
            return fingerprint, False
        exc = ModelDecisionParseError(
            "business decision repeated an unchanged platform observation",
            response_chars=decision.response_chars,
            response_sha256=decision.response_sha256,
        )
        self._record_business_validation_error(history, decision, exc)
        if self._claim_business_repair():
            return None, True
        raise exc

    def _bind_semantic_actor_context(
        self,
        semantic: SemanticAction,
    ) -> SemanticAction:
        """Bind native business actions to spawn-frozen actor authority.

        A model may choose the high-level search arguments, but it does not
        own buyer identity or mandate authority.  Governed search therefore
        receives the buyer's active mandate from the immutable-at-spawn Agent
        input before the payload is hashed and compiled into VCP.  Keeping
        this enrichment at the Agent boundary leaves the role-only semantic
        tool compiler generic and prevents a model-authored mandate id from
        escaping the buyer's actor context.
        """

        if not (self.id.startswith("buyer") and semantic.action_kind == "commerce.search"):
            return semantic

        mandate_id = self._semantic_mandate.get("mandate_id")
        if not isinstance(mandate_id, str) or not mandate_id.strip():
            return semantic
        payload = dict(semantic.payload)
        payload["mandate_id"] = mandate_id
        return SemanticAction(
            destination=semantic.destination,
            action_kind=semantic.action_kind,
            payload=payload,
            decision_rationale=semantic.decision_rationale,
        )

    def _prepare_semantic_decision_memory(
        self,
        semantic: SemanticAction,
        *,
        inbound: "Envelope",
        history: list[dict[str, Any]],
    ) -> _ProvisionalMemoryUpdate | None:
        """Temporarily overlay a framework decision record for Agent guards."""

        if not self.id.startswith("buyer"):
            return None
        if semantic.action_kind not in {
            "commerce.accept_offer",
            "platform.settle_payment",
        }:
            return None
        bucket = MemoryType.TRANSACTION
        key = "selected_offer"
        prior_keys = self.memory.keys(bucket)
        existed = key in prior_keys
        existing = self.memory.read(MemoryType.TRANSACTION, "selected_offer")
        selected = dict(existing) if isinstance(existing, Mapping) else {}
        sources: list[Mapping[str, Any]] = [semantic.payload]
        inbound_payload = inbound.action.get("payload")
        if isinstance(inbound_payload, Mapping):
            sources.append(inbound_payload)
            target = semantic.payload.get("offer_id")
            for field in ("offers", "candidates"):
                offers = inbound_payload.get(field)
                if isinstance(offers, list):
                    sources.extend(
                        row
                        for row in offers
                        if isinstance(row, Mapping) and row.get("offer_id") == target
                    )
        aliases = {
            "offer_id": ("offer_id",),
            "sku_id": ("sku_id",),
            "merchant_id": ("merchant_id",),
            "order_id": ("order_id",),
            "qty": ("qty",),
            "unit_price": ("unit_price", "unit_price_cents"),
        }
        for target, candidates in aliases.items():
            if target in selected:
                continue
            for source in sources:
                found = next((source[name] for name in candidates if name in source), None)
                if found is not None:
                    selected[target] = copy.deepcopy(found)
                    break
        if semantic.decision_rationale:
            # The free-text business argument is model output. Formal
            # runs retain only hashes and lengths, so persist a proof that a
            # non-empty rationale was supplied rather than its raw text.  The
            # existing decision-record guard needs a non-empty marker, while
            # scorers can still distinguish supplied from omitted rationale.
            rationale_bytes = semantic.decision_rationale.encode("utf-8")
            selected["rationale"] = (
                "model-rationale:"
                f"{len(semantic.decision_rationale)}:"
                f"{hashlib.sha256(rationale_bytes).hexdigest()}"
            )
        if not selected:
            return None
        previous = copy.deepcopy(existing)
        self.memory.write(bucket, key, selected)
        history.append(
            {
                "step": "memory_update",
                "source": "semantic_compiler",
                "writes": [
                    {
                        "bucket": MemoryType.TRANSACTION.value,
                        "key": "selected_offer",
                        "value": selected,
                    }
                ],
            }
        )
        return _ProvisionalMemoryUpdate(
            bucket=bucket,
            key=key,
            existed=existed,
            previous=previous,
            proposed=copy.deepcopy(selected),
        )

    def _stage_provisional_outbound(
        self,
        outbound: Envelope,
        *,
        memory_update: _ProvisionalMemoryUpdate | None,
        principal_id: str | None,
        reply_recipient_id: str | None,
        actor_reports_allowed: bool,
        observation_choice_fingerprint: str | None = None,
        continuation_source: tuple[str, str] | None = None,
    ) -> None:
        """Bind all candidate Agent state to one canonical envelope digest."""

        if outbound.msg_id in self._provisional_outbound_by_msg_id:
            raise RuntimeError("provisional Agent outbound message id is ambiguous")
        provisional_sources: dict[str, tuple[str, str]] | None = None
        if continuation_source is not None:
            if (
                not isinstance(continuation_source, tuple)
                or len(continuation_source) != 2
                or any(not isinstance(value, str) or not value for value in continuation_source)
            ):
                raise RuntimeError("provisional continuation source is malformed")
            source_key, source_identity = continuation_source
            provisional_sources = getattr(
                self,
                "_framework_provisional_continuation_sources",
                None,
            )
            if provisional_sources is None:
                provisional_sources = {}
                setattr(
                    self,
                    "_framework_provisional_continuation_sources",
                    provisional_sources,
                )
            if not isinstance(provisional_sources, dict):
                raise RuntimeError("provisional continuation source cache is malformed")
            if source_key in provisional_sources:
                raise RuntimeError("provisional continuation source is ambiguous")
            committed_sources = getattr(
                self,
                "_framework_committed_continuation_sources",
                {},
            )
            if not isinstance(committed_sources, dict):
                raise RuntimeError("committed continuation source cache is malformed")
            if source_key in committed_sources:
                raise RuntimeError("continuation source was already committed")
        envelope_json = to_json(outbound)
        self._provisional_outbound_by_msg_id[outbound.msg_id] = _ProvisionalOutbound(
            envelope_json=envelope_json,
            envelope_sha256=hashlib.sha256(envelope_json.encode("utf-8")).hexdigest(),
            memory_update=memory_update,
            principal_id=principal_id,
            reply_recipient_id=reply_recipient_id,
            actor_reports_allowed=actor_reports_allowed,
            observation_choice_fingerprint=observation_choice_fingerprint,
            continuation_source=continuation_source,
        )
        if provisional_sources is not None:
            provisional_sources[source_key] = (source_identity, outbound.msg_id)

    @staticmethod
    def _envelope_matches_provisional(
        outbound: Envelope,
        provisional: _ProvisionalOutbound,
    ) -> bool:
        try:
            envelope_json = to_json(outbound)
        except (TypeError, ValueError):
            return False
        return (
            envelope_json == provisional.envelope_json
            and hashlib.sha256(envelope_json.encode("utf-8")).hexdigest()
            == provisional.envelope_sha256
        )

    def provisional_outbound_envelope(self, msg_id: str) -> Envelope | None:
        """Return an isolated exact candidate for deterministic teardown."""

        provisional = self._provisional_outbound_by_msg_id.get(msg_id)
        if provisional is None:
            return None
        return from_json(provisional.envelope_json)

    def provisional_selected_offer_for_outbound(
        self,
        outbound: Envelope,
    ) -> Mapping[str, Any] | None:
        """Return a Tracker-only choice bound to the exact candidate envelope."""

        provisional = self._provisional_outbound_by_msg_id.get(outbound.msg_id)
        if provisional is None:
            raise RuntimeError("typed Agent output has no provisional state")
        if not self._envelope_matches_provisional(outbound, provisional):
            raise RuntimeError("provisional Agent outbound envelope digest mismatch")
        update = provisional.memory_update
        if update is None:
            return None
        return copy.deepcopy(update.proposed)

    def commit_provisional_outbound(self, outbound: Envelope) -> None:
        """Atomically apply state after Runtime audits the exact envelope."""

        provisional = self._provisional_outbound_by_msg_id.get(outbound.msg_id)
        if provisional is None:
            raise RuntimeError("typed Agent outbound has no provisional commit")
        if not self._envelope_matches_provisional(outbound, provisional):
            raise RuntimeError("provisional Agent outbound envelope digest mismatch")

        update = provisional.memory_update
        current_memory: tuple[bool, Any] | None = None
        if update is not None:
            current_memory = self._memory_value_snapshot(update.bucket, update.key)
        principal_existed = outbound.msg_id in self._semantic_principal_by_outbound_msg_id
        principal_previous = self._semantic_principal_by_outbound_msg_id.get(outbound.msg_id)
        reply_existed = outbound.msg_id in self._semantic_reply_recipient_by_outbound_msg_id
        reply_previous = self._semantic_reply_recipient_by_outbound_msg_id.get(outbound.msg_id)
        report_existed = outbound.msg_id in self._reportable_actor_outbound_msg_ids
        observation_fingerprint = provisional.observation_choice_fingerprint
        observation_existed = bool(
            observation_fingerprint is not None
            and observation_fingerprint in self._committed_observation_choice_fingerprints
        )
        continuation_source = provisional.continuation_source
        provisional_sources: dict[str, tuple[str, str]] | None = None
        committed_sources: dict[str, str] | None = None
        if continuation_source is not None:
            source_key, source_identity = continuation_source
            provisional_sources = getattr(
                self,
                "_framework_provisional_continuation_sources",
                None,
            )
            committed_sources = getattr(
                self,
                "_framework_committed_continuation_sources",
                None,
            )
            if committed_sources is None:
                committed_sources = {}
                setattr(
                    self,
                    "_framework_committed_continuation_sources",
                    committed_sources,
                )
            if (
                not isinstance(provisional_sources, dict)
                or provisional_sources.get(source_key) != (source_identity, outbound.msg_id)
                or not isinstance(committed_sources, dict)
                or source_key in committed_sources
            ):
                raise RuntimeError("provisional continuation commit binding is malformed")

        try:
            if update is not None:
                self.memory.write(
                    update.bucket,
                    update.key,
                    copy.deepcopy(update.proposed),
                )
            if provisional.principal_id is not None:
                self._semantic_principal_by_outbound_msg_id[outbound.msg_id] = (
                    provisional.principal_id
                )
            if provisional.reply_recipient_id is not None:
                self._semantic_reply_recipient_by_outbound_msg_id[outbound.msg_id] = (
                    provisional.reply_recipient_id
                )
            if provisional.actor_reports_allowed:
                self._reportable_actor_outbound_msg_ids.add(outbound.msg_id)
            committed_payload = outbound.action.get("payload")
            if not isinstance(committed_payload, Mapping):
                raise RuntimeError("committed Agent outbound has no object payload")
            _payload_chars, committed_payload_sha256 = canonical_semantic_digest(
                dict(committed_payload)
            )
            committed_order_id = committed_payload.get("order_id")
            if committed_order_id is not None and (
                not isinstance(committed_order_id, str) or not committed_order_id
            ):
                raise RuntimeError("committed Agent outbound has malformed order authority")
            self._committed_semantic_outbound_by_msg_id[outbound.msg_id] = (
                _CommittedSemanticOutbound(
                    action_kind=str(outbound.action.get("kind", "")),
                    destination=outbound.to,
                    envelope_sha256=provisional.envelope_sha256,
                    payload_sha256=committed_payload_sha256,
                    order_id=committed_order_id,
                )
            )
            self._commerce_action_grounding.record_committed_outbound(
                msg_id=outbound.msg_id,
                action_kind=str(outbound.action.get("kind", "")),
                payload=outbound.action.get("payload"),
            )
            if observation_fingerprint is not None:
                self._committed_observation_choice_fingerprints.add(observation_fingerprint)
        except Exception:
            if update is not None and current_memory is not None:
                current_existed, current_value = current_memory
                self._restore_memory_value(
                    update.bucket,
                    update.key,
                    existed=current_existed,
                    previous=current_value,
                )
            self._restore_mapping_value(
                self._semantic_principal_by_outbound_msg_id,
                outbound.msg_id,
                existed=principal_existed,
                previous=principal_previous,
            )
            self._restore_mapping_value(
                self._semantic_reply_recipient_by_outbound_msg_id,
                outbound.msg_id,
                existed=reply_existed,
                previous=reply_previous,
            )
            if not report_existed:
                self._reportable_actor_outbound_msg_ids.discard(outbound.msg_id)
            if observation_fingerprint is not None and not observation_existed:
                self._committed_observation_choice_fingerprints.discard(observation_fingerprint)
            raise
        if continuation_source is not None:
            assert provisional_sources is not None
            assert committed_sources is not None
            committed_sources[source_key] = source_identity
            del provisional_sources[source_key]
        del self._provisional_outbound_by_msg_id[outbound.msg_id]

    def reject_provisional_outbound(self, outbound: Envelope) -> bool:
        """Idempotently discard only an exact rejected candidate envelope."""

        provisional = self._provisional_outbound_by_msg_id.get(outbound.msg_id)
        if provisional is None or not self._envelope_matches_provisional(
            outbound,
            provisional,
        ):
            return False
        continuation_source = provisional.continuation_source
        if continuation_source is not None:
            source_key, source_identity = continuation_source
            provisional_sources = getattr(
                self,
                "_framework_provisional_continuation_sources",
                None,
            )
            if not isinstance(provisional_sources, dict) or provisional_sources.get(source_key) != (
                source_identity,
                outbound.msg_id,
            ):
                raise RuntimeError("provisional continuation reject binding is malformed")
            del provisional_sources[source_key]
        del self._provisional_outbound_by_msg_id[outbound.msg_id]
        return True

    def _memory_value_snapshot(
        self,
        bucket: MemoryType,
        key: str,
    ) -> tuple[bool, Any]:
        existed = key in self.memory.keys(bucket)
        return existed, copy.deepcopy(self.memory.read(bucket, key))

    def _restore_memory_value(
        self,
        bucket: MemoryType,
        key: str,
        *,
        existed: bool,
        previous: Any,
    ) -> None:
        if existed:
            self.memory.write(bucket, key, copy.deepcopy(previous))
            return
        self.memory.delete(bucket, key)

    @staticmethod
    def _restore_mapping_value(
        mapping: dict[str, str],
        key: str,
        *,
        existed: bool,
        previous: str | None,
    ) -> None:
        if existed and previous is not None:
            mapping[key] = previous
        else:
            mapping.pop(key, None)

    def _restore_provisional_memory(
        self,
        provisional: _ProvisionalMemoryUpdate | None,
    ) -> None:
        if provisional is None:
            return
        self._restore_memory_value(
            provisional.bucket,
            provisional.key,
            existed=provisional.existed,
            previous=provisional.previous,
        )

    def _advance_reads(self, frame: "TurnFrame") -> "TurnSuspended | None":
        """Suspend for the next Agent-validated typed business read."""
        while frame.pending_calls:
            call = frame.pending_calls[0]
            try:
                read_env = tool_call_to_read_envelope(
                    call,
                    from_id=self.id,
                    correlation_id=frame.original_inbound.msg_id,
                    seq=frame.read_seq,
                    ts=frame.original_inbound.ts,
                )
            except (ValueError, KeyError) as exc:  # unmapped tool / missing arg
                finalize_rejected_tool_batch(frame)
                raise RuntimeError(
                    "validated business read lost its re-entrant World mapping"
                ) from exc
            try:
                validate_reentrant_read_partition(read_env, actor_id=self.id)
            except Exception:
                finalize_rejected_tool_batch(frame)
                raise
            frame.read_seq += 1
            frame.pending_read_msg_id = read_env.msg_id
            return TurnSuspended(frame=frame, read_env=read_env)
        return None

    def _commit_tool_call(self, frame: "TurnFrame") -> None:
        """Record the completed tool_call step (every call resolved)."""
        if frame.pending_calls or frame.pending_read_msg_id is not None:
            raise RuntimeError(
                "business re-entrant tool batch committed before all reads completed"
            )
        if frame.business_pending_read_fingerprint is None:
            raise RuntimeError("business re-entrant tool batch lost its read fingerprint")
        self._commerce_action_grounding.observe_read_results(frame.pending_results)
        frame.history.append(
            {
                "step": "tool_call",
                "interface": "business_decision",
                "results": frame.pending_results,
            }
        )
        frame.business_pending_read_fingerprint = None
        frame.pending_results = []
        frame.pending_read_msg_id = None

    def _resume_reentrant_turn(
        self, env: "Envelope", ctx: "AgentContext", frame: "TurnFrame"
    ) -> "Envelope | TurnSuspended | None":
        """Resume on a ``world.response``: fill the head pending call's result,
        then either suspend for the next batched read or re-enter the loop."""
        if frame.pending_read_msg_id != env.in_reply_to:
            from protocol.errors import SchemaError

            raise SchemaError("World response does not match the suspended read")
        frame.pending_read_msg_id = None
        call = frame.pending_calls.pop(0)
        frame.pending_results.append(world_response_to_tool_result(call, env))
        suspended = self._advance_reads(frame)
        if suspended is not None:
            return suspended
        if frame.framework_authority_prerequisite is not None:
            self._commit_framework_authority_reads(frame)
        else:
            self._commit_tool_call(frame)
        return self._business_reentrant_loop(ctx, frame)

    def _commit_framework_authority_reads(self, frame: "TurnFrame") -> None:
        """Dispatch one private prerequisite batch through the shared seam."""

        if frame.framework_authority_prerequisite == "after_sales":
            self._commit_framework_after_sales_authority_reads(frame)
            return
        if frame.framework_authority_prerequisite == "protocol_event":
            self._commit_framework_protocol_event_authority_read(frame)
            return
        raise RuntimeError("framework authority prerequisite kind is unknown")

    def _commit_framework_after_sales_authority_reads(
        self,
        frame: "TurnFrame",
    ) -> None:
        """Validate hidden World results and bind them to the original turn."""

        from world.after_sales_authority_projection import (
            AfterSalesAuthorityProjection,
            after_sales_authority_projection_from_wire,
            validate_after_sales_authority_projection,
        )

        if frame.pending_calls or frame.pending_read_msg_id is not None:
            raise RuntimeError("after-sales authority batch committed before all World reads")
        projections: list[AfterSalesAuthorityProjection] = []
        expected_order_ids: list[str] = []
        for row in frame.pending_results:
            if row.get("tool") != "world.get_after_sales_authority":
                raise RuntimeError("after-sales authority batch contains another tool")
            args = row.get("args")
            result = row.get("result")
            order_id = args.get("order_id") if isinstance(args, Mapping) else None
            if not isinstance(order_id, str) or not order_id:
                raise RuntimeError("after-sales authority read lost its order identity")
            if not isinstance(result, Mapping):
                raise PlatformContractError(
                    "World after-sales authority response is missing or malformed"
                )
            projection = after_sales_authority_projection_from_wire(result)
            validate_after_sales_authority_projection(projection)
            if projection.actor_id != self.id or projection.order_id != order_id:
                raise PlatformContractError(
                    "World after-sales authority response crossed actor or order"
                )
            expected_order_ids.append(order_id)
            projections.append(projection)
        if not projections or len(expected_order_ids) != len(set(expected_order_ids)):
            raise PlatformContractError("World after-sales authority response batch is incomplete")
        self._after_sales_projections_by_inbound[frame.original_inbound.msg_id] = tuple(projections)
        frame.history.append(
            {
                "step": "framework_authority_prerequisite",
                "interface": "agent_world_authority_prerequisite",
                "authority_kind": "after_sales",
                "source_msg_ids": [str(row.get("source_msg_id")) for row in frame.pending_results],
                "authority_sha256": [projection.projection_digest for projection in projections],
            }
        )
        frame.pending_results = []
        frame.pending_read_msg_id = None
        frame.framework_authority_prerequisite = None

    def _commit_framework_protocol_event_authority_read(
        self,
        frame: "TurnFrame",
    ) -> None:
        """Validate one fresh callback projection and bind it to the turn."""

        if frame.pending_calls or frame.pending_read_msg_id is not None:
            raise RuntimeError("protocol-event authority committed before its World read completed")
        if len(frame.pending_results) != 1:
            raise PlatformContractError("protocol-event authority response batch is incomplete")
        row = frame.pending_results[0]
        if row.get("tool") != "world.get_protocol_event_authority":
            raise RuntimeError("protocol-event authority batch contains another tool")
        args = row.get("args")
        result = row.get("result")
        event_id = args.get("event_id") if isinstance(args, Mapping) else None
        if not isinstance(event_id, str) or not event_id:
            raise RuntimeError("protocol-event authority read lost its event identity")
        if not isinstance(result, Mapping) or set(result) != {
            "current_event",
            "current_order_state",
            "current_receipts",
            "current_reference_state",
            "prior_event_states",
            "current_tick",
        }:
            raise PlatformContractError(
                "World protocol-event authority response is missing or malformed"
            )
        current_event = result.get("current_event")
        if not isinstance(current_event, Mapping) or current_event.get("event_id") != event_id:
            raise PlatformContractError("World protocol-event authority crossed event identities")
        authority = copy.deepcopy(dict(result))
        self._protocol_event_authority_by_inbound[frame.original_inbound.msg_id] = authority
        _chars, digest = canonical_semantic_digest(authority)
        source_msg_id = row.get("source_msg_id")
        if not isinstance(source_msg_id, str) or not source_msg_id:
            raise PlatformContractError("World protocol-event authority response has no provenance")
        frame.history.append(
            {
                "step": "framework_authority_prerequisite",
                "interface": "agent_world_authority_prerequisite",
                "authority_kind": "protocol_event",
                "source_msg_ids": [source_msg_id],
                "authority_sha256": [digest],
            }
        )
        frame.pending_results = []
        frame.pending_read_msg_id = None
        frame.framework_authority_prerequisite = None

    # --- Helpers ----------------------------------------------------

    def _dispatch_tool(self, call: "ToolCall", ctx: "AgentContext") -> "dict[str, Any]":
        """Invoke one whitelisted world tool, formatting the result for history.

        The Agent compiler already binds a registered World read, and the
        dispatcher re-verifies the allowlist so this guard does not depend on
        adapter state.

        Exceptions raised by the tool are captured into the result slot
        (``{"error": ...}``) so the skill can react on the next step rather
        than aborting the whole turn.
        """
        if call.tool not in ALLOWED_WORLD_TOOLS:
            return {
                "tool": call.tool,
                "args": call.args,
                "result": {"error": f"tool not whitelisted: {call.tool!r}"},
            }
        method_name = call.tool.removeprefix("world.")
        method = getattr(ctx.world, method_name, None)
        if not callable(method):
            return {
                "tool": call.tool,
                "args": call.args,
                "result": {"error": f"WorldTools has no method {method_name!r}"},
            }
        try:
            raw = method(**call.args)
        except Exception as exc:  # noqa: BLE001 — surfaced into result slot
            return {
                "tool": call.tool,
                "args": call.args,
                "result": {"error": f"{type(exc).__name__}: {exc}"},
            }
        return {
            "tool": call.tool,
            "args": call.args,
            "result": _format_tool_result(raw),
        }

    def _build_outbound(
        self,
        compiled_envelope: "dict[str, Any]",
        inbound: "Envelope",
        history: "list[dict[str, Any]] | None" = None,
    ) -> "Envelope":
        """Seal an Agent-compiled route and action into a real ``Envelope``.

        ``compiled_envelope`` is produced only by a phase adapter or a
        deterministic Agent continuation. The provider never supplies its
        destination, action kind, payload shape, or bookkeeping fields.
        Before sealing it, the Agent reconciles framework-owned settlement
        facts against committed local state and rejects contradictions.
        """
        to = compiled_envelope.get("to")
        action = compiled_envelope.get("action")
        if not isinstance(to, str) or not to:
            raise ValueError(f"compiled envelope missing 'to' string: {compiled_envelope!r}")
        if not isinstance(action, dict) or "kind" not in action:
            raise ValueError(
                f"compiled envelope missing valid 'action.kind': {compiled_envelope!r}"
            )
        if str(action.get("kind", "")) in PUBLIC_WORLD_READ_ACTION_KINDS:
            raise ValueError(
                "World reads must use the Agent observation path; a compiled "
                "outbound action cannot create a re-entrant read"
            )

        # Tracker derives the buyer's chosen offer from TRANSACTION memory,
        # while Runtime audits the actual emitted action.  Letting those two
        # sources disagree would create a trace that claims the buyer chose one
        # offer even though another offer crossed the authoritative boundary.
        # Refuse the contradictory action before any normalization or Platform
        # delivery.  This is deliberately a guard, not a reconciler: code must
        # not silently replace the model's commercial choice.
        _assert_outbound_choice_consistent(action, self.memory, self.id)

        settlement_corrections = _bind_negotiated_settlement(
            action=action,
            inbound=inbound,
        )
        corrections = [
            *settlement_corrections,
            *_reconcile_outbound_payload(action, self.memory, self.id),
        ]
        if corrections:
            print(
                f"  [reconcile] {self.id} {action.get('kind')}: " + "; ".join(corrections),
                file=sys.stderr,
            )

        _assert_decision_recorded(
            action,
            self.memory,
            self.id,
            must_have=self._mandate_must_have,
            history=history or [],
        )

        msg_id, idempotency_key = _deterministic_outbound_identity(
            agent_id=self.id,
            inbound=inbound,
            recipient=to,
            action=action,
            supplied_idempotency_key=compiled_envelope.get("idempotency_key"),
        )
        return Envelope(
            msg_id=msg_id,
            # Logical ordering comes from Runtime. Reusing the triggering
            # envelope timestamp keeps the same decision byte-identical across
            # in-process and HTTP agent processes.
            ts=inbound.ts,
            from_=self.id,
            to=to,
            in_reply_to=inbound.msg_id,
            idempotency_key=idempotency_key,
            action=action,
        )


def _deterministic_outbound_identity(
    *,
    agent_id: str,
    inbound: "Envelope",
    recipient: str,
    action: "dict[str, Any]",
    supplied_idempotency_key: Any,
) -> tuple[str, str]:
    """Derive stable message/idempotency identities for one agent decision.

    One external turn can emit at most one envelope. The actual inbound
    ``msg_id`` plus the canonical semantic action therefore form a stable retry
    identity without a process-local counter or random UUID. A caller-supplied
    idempotency key remains authoritative and is committed into ``msg_id`` so a
    key change cannot silently reuse the same message identity.
    """

    if supplied_idempotency_key is not None and (
        not isinstance(supplied_idempotency_key, str) or not supplied_idempotency_key.strip()
    ):
        raise ValueError("emit envelope idempotency_key must be a non-empty string")
    contract = {
        "schema": "cwe.agent-outbound-identity.v1",
        "agent_id": agent_id,
        "inbound_msg_id": inbound.msg_id,
        "recipient": recipient,
        "action": action,
    }
    try:
        canonical = json.dumps(
            contract,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("emit envelope action must be canonical JSON") from exc
    semantic_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    idempotency_key = supplied_idempotency_key or f"agent-idem:{semantic_digest}"
    message_digest = hashlib.sha256(f"{canonical}\n{idempotency_key}".encode("utf-8")).hexdigest()
    return f"agent-msg:{message_digest}", idempotency_key


def _assert_decision_recorded(
    action: "dict[str, Any]",
    memory: "Memory",
    agent_id: str,
    must_have: "tuple[str, ...]" = (),
    history: "list[dict[str, Any]] | None" = None,
) -> None:
    """Refuse to emit a purchase commit with an incomplete bound decision.

    On ``commerce.accept_offer`` the buyer must have a structured
    ``selected_offer`` with a non-empty ``offer_id``.  When a discovery
    ``budget_audit`` exists, every entry must also record its eligibility (and
    the rejection stage/reason for ineligible candidates).  A free-text
    rationale is deliberately optional: ranked-offer authority already binds
    the model's structured choice to the emitted offer, and rationale is not a
    benchmark scoring predicate.

    Raises:
        DecisionRecordIncomplete: a required decision artifact is missing.
    """
    if action.get("kind") != "commerce.accept_offer":
        return
    if not agent_id.startswith("buyer"):
        return  # the recording is a buyer-side discovery artifact; a merchant
        # emitting accept_offer (accepting a counter) has no budget_audit.

    # Universal: every buy must retain the authority-bound structured choice.
    # ``_assert_outbound_choice_consistent`` performs the exact emitted/recorded
    # offer comparison on the production path; keep this local check so direct
    # callers cannot accidentally bypass the presence invariant.
    sel = memory.read(MemoryType.TRANSACTION, "selected_offer")
    selected_offer_id = sel.get("offer_id") if isinstance(sel, dict) else None
    if not isinstance(selected_offer_id, str) or not selected_offer_id.strip():
        raise DecisionRecordIncomplete(
            f"agent {agent_id!r}: TRANSACTION.selected_offer needs a non-empty "
            "'offer_id' bound to the accepted offer."
        )

    # When a candidate audit exists (the discovery path), each entry must show
    # its eligibility — and any ineligible one must say which rigid stage cut it.
    audit = memory.read(MemoryType.TRANSACTION, "budget_audit")
    if isinstance(audit, list) and audit:
        for i, cand in enumerate(audit):
            if not isinstance(cand, dict) or "is_eligible" not in cand:
                raise DecisionRecordIncomplete(
                    f"agent {agent_id!r}: budget_audit[{i}] is missing 'is_eligible' "
                    f"— mark every candidate eligible-or-not before accepting."
                )
            if not cand.get("is_eligible") and not (
                cand.get("rejected_at_stage") and cand.get("reject_reason")
            ):
                raise DecisionRecordIncomplete(
                    f"agent {agent_id!r}: ineligible candidate {cand.get('offer_id')!r} "
                    f"must record rejected_at_stage + reject_reason."
                )

    # item 3 grounding teeth: a feature must_have requires the chosen sku to have
    # been grounded — a real listing read THIS decision (in-turn / re-entrant), or
    # a recorded ``selected_offer.grounded_attributes`` (the cross-turn case) —
    # before accepting. PRESENCE only; whether the grounded evidence actually
    # satisfies the need is the scorer's fabricated-grounding check. Skipped when
    # the mandate carries no feature must_have.
    if must_have:
        chosen_sku = sel.get("sku_id") if isinstance(sel, dict) else None
        if not _grounded_for_accept(chosen_sku, sel, history or []):
            raise GroundingRequired(
                f"agent {agent_id!r}: accepting a purchase with feature must_have "
                f"{list(must_have)} but the chosen sku {chosen_sku!r} was not grounded "
                f"(no get_listing/search_catalog read this decision, and no "
                f"selected_offer.grounded_attributes). Ground before accepting."
            )


def _assert_outbound_choice_consistent(
    action: "dict[str, Any]",
    memory: "Memory",
    agent_id: str,
) -> None:
    """Bind a buyer's emitted acceptance to its recorded selected offer.

    Tracker records the chosen offer from ``TRANSACTION.selected_offer``.  A
    buyer ``commerce.accept_offer`` must therefore name that same offer in the
    action that Runtime will audit.  Optional identity fields are compared only
    when both the action and the recorded offer explicitly carry them.  Price
    accepts either protocol spelling, ``unit_price`` or
    ``unit_price_cents``.  Missing optional fields remain the Platform's schema
    concern; a contradictory explicit value is rejected here.

    The guard never rewrites the action.  ``DecisionRecordIncomplete`` is the
    existing scoreable Agent capability guard for an unusable decision record,
    so Runtime finalizes a bound ``capability_error`` Tracker row before
    propagating the exception.

    Merchant acceptances are negotiation decisions with merchant-owned state,
    not buyer discovery records, and are intentionally exempt.

    Raises:
        DecisionRecordIncomplete: the required offer binding is absent or an
            explicitly emitted choice field contradicts ``selected_offer``.
    """
    if action.get("kind") != "commerce.accept_offer":
        return
    if agent_id.split(":", 1)[0] != "buyer":
        return

    selected = memory.read(MemoryType.TRANSACTION, "selected_offer")
    if not isinstance(selected, dict):
        raise DecisionRecordIncomplete(
            f"agent {agent_id!r}: buyer commerce.accept_offer requires a "
            "structured TRANSACTION.selected_offer bound to the emitted offer."
        )

    payload = action.get("payload")
    selected_offer_id = selected.get("offer_id")
    emitted_offer_id = payload.get("offer_id") if isinstance(payload, dict) else None
    if (
        not isinstance(selected_offer_id, str)
        or not selected_offer_id
        or not isinstance(emitted_offer_id, str)
        or not emitted_offer_id
    ):
        raise DecisionRecordIncomplete(
            f"agent {agent_id!r}: buyer commerce.accept_offer requires a "
            "non-empty payload.offer_id bound to "
            "TRANSACTION.selected_offer.offer_id."
        )
    if emitted_offer_id != selected_offer_id:
        raise DecisionRecordIncomplete(
            f"agent {agent_id!r}: commerce.accept_offer payload.offer_id "
            "contradicts TRANSACTION.selected_offer.offer_id; refusing to "
            "write a false Tracker choice."
        )

    assert isinstance(payload, dict)  # narrowed by the required binding above
    for field in ("sku_id", "merchant_id"):
        if field not in payload or field not in selected:
            continue
        if payload[field] != selected[field]:
            raise DecisionRecordIncomplete(
                f"agent {agent_id!r}: commerce.accept_offer payload.{field} "
                f"contradicts TRANSACTION.selected_offer.{field}; refusing "
                "to write a false Tracker choice."
            )

    # Tracker's TraceChoice reads ``selected_offer.unit_price`` today, so that
    # spelling is preferred when a record happens to contain both aliases.
    selected_price_field = next(
        (name for name in ("unit_price", "unit_price_cents") if name in selected),
        None,
    )
    for emitted_price_field in ("unit_price", "unit_price_cents"):
        if emitted_price_field not in payload or selected_price_field is None:
            continue
        if payload[emitted_price_field] != selected[selected_price_field]:
            raise DecisionRecordIncomplete(
                f"agent {agent_id!r}: commerce.accept_offer "
                f"payload.{emitted_price_field} contradicts "
                f"TRANSACTION.selected_offer.{selected_price_field}; refusing "
                "to write a false Tracker choice."
            )


def _grounded_for_accept(sku: "str | None", sel: "Any", history: "list[dict[str, Any]]") -> bool:
    """Was the chosen sku grounded this decision? True if ``selected_offer``
    records ``grounded_attributes`` (the cross-turn case), or the turn history
    has a ``world.get_listing`` / ``world.search_catalog`` read of the sku (the
    in-turn / re-entrant case)."""
    if isinstance(sel, dict) and isinstance(sel.get("grounded_attributes"), dict):
        return True
    if not sku:
        return False
    for entry in history:
        if entry.get("step") != "tool_call":
            continue
        for r in entry.get("results", []):
            if not isinstance(r, dict) or r.get("tool") not in (
                "world.get_listing",
                "world.search_catalog",
            ):
                continue
            if str((r.get("args") or {}).get("sku_id") or "") == sku:
                return True
            res = r.get("result")
            for it in res if isinstance(res, list) else [res]:
                if isinstance(it, dict) and str(it.get("sku_id") or "") == sku:
                    return True
    return False


def _bind_negotiated_settlement(
    *,
    action: dict[str, Any],
    inbound: Envelope,
) -> list[str]:
    """Bind settlement only to an accepted server-mediated negotiation."""

    if action.get("kind") != "platform.settle_payment":
        return []
    if (
        inbound.from_ != "platform:negotiation"
        or str(inbound.action.get("kind", "")) != "commerce.accept_offer"
    ):
        return []
    inbound_payload = inbound.action.get("payload")
    if not isinstance(inbound_payload, dict):
        return []
    metadata = inbound_payload.get("platform_mediation")
    if (
        not isinstance(metadata, dict)
        or metadata.get("mediated_by") != "platform:negotiation"
        or metadata.get("status") != "accepted"
        or metadata.get("recipient_id") != inbound.to
    ):
        return []
    negotiation_id = inbound_payload.get("negotiation_id")
    if not isinstance(negotiation_id, str) or not negotiation_id:
        return []
    payload = action.setdefault("payload", {})
    if not isinstance(payload, dict):
        return []
    before = payload.get("negotiation_id")
    payload["negotiation_id"] = negotiation_id
    if before == negotiation_id:
        return []
    return [f"negotiation_id: {before!r} -> authoritative accepted thread"]


def _reconcile_outbound_payload(
    action: "dict[str, Any]",
    memory: "Memory",
    agent_id: str,
) -> list[str]:
    """Force critical outbound payload fields to match memory truth.

    Memory is the source of truth; the LLM is allowed to *decide* what
    kind of envelope to emit, but it cannot *invent* the values of
    fields that are deterministically derivable from memory. This
    function scans the outbound payload for the active action kind and
    overwrites such fields in place, returning a list of
    human-readable correction descriptors (empty if the LLM got
    everything right).

    Currently covers ``platform.settle_payment`` only. Negotiation
    envelopes (``commerce.propose_offer`` / ``commerce.counter_offer``)
    carry the LLM's actual choice in ``unit_price`` and are
    intentionally NOT reconciled — overwriting those would mean code,
    not LLM, is deciding the counter price.

    Args:
        action: the envelope's ``action`` dict (``kind`` + ``payload``).
            Mutated in place.
        memory: the agent's memory store.
        agent_id: the agent's own bus id, used for ``buyer_id`` fields.

    Returns:
        List of ``"<field>: <old> → <new>"`` strings, one per actual
        change. Empty list when no corrections were made.
    """
    kind = action.get("kind")
    if kind != "platform.settle_payment":
        return []

    raw_payload = action.get("payload")
    if isinstance(raw_payload, dict) and (
        set(raw_payload) <= {"cert_id"} or "supply_authority_id" in raw_payload
    ):
        # Certificate- and supply-authority-backed settlements are separate
        # Platform contracts.  In particular, an empty payload is a valid
        # negative-conformance probe and must reach the Platform unchanged.
        # Rebuilding either shape from selected_offer would manufacture a
        # payment intent that the actor did not emit and would make a missing
        # certificate succeed.  The authoritative Platform remains
        # responsible for validating the certificate and deriving the order.
        return []

    sel = memory.read(MemoryType.TRANSACTION, "selected_offer")
    if not isinstance(sel, dict):
        # Settle without selected_offer is anomalous: discovery's happy
        # path writes selected_offer on a chosen offer, and the
        # negotiation skill's Branch N-C commits selected_offer from
        # the merchant's accept_offer envelope. Reaching settle without
        # either having run means the upstream skill body skipped its
        # commit — let through, the PSP's consent guard would accept
        # the LLM's hallucinated agreed_price against the catalog
        # list_price (silent overpay, surfaced by the 2026-05 live
        # driver). Raise so the failure is loud at the agent boundary.
        pending = memory.read(MemoryType.TRANSACTION, "pending_negotiation")
        if pending is not None:
            raise BudgetReconcileViolation(
                f"agent {agent_id!r} emitting platform.settle_payment "
                f"with pending_negotiation in memory but no "
                f"selected_offer. negotiation's Branch N-C (commit on "
                f"merchant accept_offer) must run BEFORE "
                f"purchase-confirmation emits settle — otherwise the "
                f"reconciler has no source of truth and the buyer would "
                f"settle at the LLM's invented price."
            )
        raise BudgetReconcileViolation(
            f"agent {agent_id!r} emitting platform.settle_payment with "
            f"no selected_offer AND no pending_negotiation in "
            f"TRANSACTION memory. Either discovery-search Path B should "
            f"have written selected_offer (happy path) or negotiation's "
            f"Branch N-C should have committed it (negotiation path). "
            f"Without either, this settle has no upstream commerce "
            f"context — refusing rather than letting the PSP accept an "
            f"arbitrary list-price settle."
        )

    mandate_id = memory.read(MemoryType.TRANSACTION, "mandate_id") or ""
    offer_id = sel.get("offer_id", "")
    unit_cents = sel.get("unit_price")
    negotiation_id = raw_payload.get("negotiation_id") if isinstance(raw_payload, dict) else None
    expected_order_id = (
        negotiated_order_id(negotiation_id, str(offer_id))
        if isinstance(negotiation_id, str) and negotiation_id
        else (
            str(sel["order_id"])
            if isinstance(sel.get("order_id"), str) and sel["order_id"]
            else f"ord-{mandate_id}-{offer_id}"
        )
    )

    expected: dict[str, Any] = {
        "buyer_id": agent_id,
        "merchant_id": sel.get("merchant_id"),
        "sku_id": sel.get("sku_id"),
        "qty": int(sel.get("qty", 1)),
        "order_id": expected_order_id,
    }
    if isinstance(unit_cents, int):
        # Budget invariant: refuse to reconcile a settle that would put
        # cumulative_spend over max_budget. This catches an over-budget
        # selected_offer that slipped past business-decision validation
        # budget gate. Without this check, the reconciler would silently
        # promote the LLM's hallucinated low price to the over-budget
        # selected_offer.unit_price and the PSP would accept it as a
        # list-price settle, masking the budget violation.
        max_budget = memory.read(MemoryType.PRIVATE_UTILITY, "max_budget")
        if isinstance(max_budget, int):
            cumulative = memory.read(MemoryType.TRANSACTION, "cumulative_spend") or 0
            qty = int(sel.get("qty", 1))
            total = unit_cents * qty + int(cumulative)
            if total > max_budget:
                raise BudgetReconcileViolation(
                    f"agent {agent_id!r} would settle at {unit_cents} cents × "
                    f"qty {qty} = {unit_cents * qty}, cumulative {cumulative}, "
                    f"total {total} > max_budget {max_budget}. "
                    f"Discovery-search picked an over-budget offer "
                    f"({sel.get('offer_id')!r}) as selected_offer; the "
                    f"reconciler refuses to silently push the LLM's "
                    f"made-up agreed_price to the selected offer's "
                    f"unit_price when doing so would exceed budget. "
                    "The over-budget choice must be rejected or negotiated "
                    "instead of being committed for settlement."
                )
        expected["agreed_price"] = {
            "amount": f"{unit_cents / 100:.2f}",
            "currency": "USD",
        }

    payload = action.setdefault("payload", {})
    if not isinstance(payload, dict):
        # Garbage payload — wipe and rebuild from memory expectations.
        payload = {}
        action["payload"] = payload

    corrections: list[str] = []
    for key, want in expected.items():
        actual = payload.get(key)
        if actual != want:
            corrections.append(f"{key}: {actual!r} → {want!r}")
            payload[key] = want
    return corrections


def _envelope_to_wire(env: "Envelope") -> "dict[str, Any]":
    """Render an ``Envelope`` as a dict with the wire ``from`` field.

    Local fork of the not-yet-implemented ``protocol.envelope.to_json``
    helper; used to feed the LLM a readable representation of the inbound.
    """
    return _wire_json_value(
        {
            "msg_id": env.msg_id,
            "ts": env.ts,
            "from": env.from_,
            "to": env.to,
            "in_reply_to": env.in_reply_to,
            "idempotency_key": env.idempotency_key,
            "action": env.action,
        }
    )


def _wire_json_value(value: "Any") -> "Any":
    """Normalize typed envelope values to their lossless JSON wire shape.

    In-process services may keep protocol payloads as frozen dataclasses until
    the envelope reaches an agent.  The prompt is still a wire view, so mirror
    ``protocol.envelope.to_json`` here instead of falling back to ``repr`` or
    losing nested fields.  Decimal amounts stay exact as strings.
    """

    if is_dataclass(value):
        return {field.name: _wire_json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _wire_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_json_value(item) for item in value]
    return value


# --- Tool result serialization ----------------------------------------

_KEY_FEATURES_MAX_CHARS: int = 200


def _format_tool_result(value: Any) -> Any:
    """Project a world-tool result into a compact JSON-friendly shape.

    ``Listing`` is the load-bearing case: its ``attributes`` dict can carry
    every CSV column (25 today), including a long ``Key Features`` marketing
    paragraph. Pushing all of that into step history for every grounding
    lookup would blow the prompt budget. We keep only the fields the
    discovery-search skill actually consults during matching, plus a 200-char
    excerpt of Key Features for free-text claims.
    """
    # Lazy import to avoid pulling world.types into agents at module load.
    from world.types import Listing, Money, ReputationScore

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Money):
        return {"amount": str(value.amount), "currency": value.currency}
    if isinstance(value, ReputationScore):
        return {
            "rolling_avg": value.rolling_avg,
            "n_settled": value.n_settled,
            "n_disputed": value.n_disputed,
        }
    if isinstance(value, Listing):
        attrs = value.attributes or {}
        # Grounding truth: expose the full structured attributes (decision #3 — a
        # feature must_have may be satisfied only by a grounded attribute, so the
        # buyer must be able to SEE it, and the trace records exactly what it
        # saw). The one free-text field (``key_features``) is excerpted
        # separately to bound the prompt; every other attribute
        # (noise_cancellation, shipping_days, material, …) passes through
        # verbatim. (Prompt size for very wide CSV catalogs is a later
        # optimization; the v1 corpus carries a handful of attributes.)
        projected = {
            k: v for k, v in attrs.items() if k.lower().replace(" ", "_") != "key_features"
        }
        key_features = attrs.get("key_features") or attrs.get("Key Features") or ""
        out: dict[str, Any] = {
            "sku_id": str(value.sku_id),
            "category": value.category,
            "name": value.name,
            "list_price": _format_tool_result(value.list_price),
            "merchant_id": str(value.merchant_id),
            "attributes": projected,
        }
        if isinstance(key_features, str) and key_features:
            out["key_features_excerpt"] = key_features[:_KEY_FEATURES_MAX_CHARS]
        return out
    if isinstance(value, (list, tuple)):
        return [_format_tool_result(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _format_tool_result(v) for k, v in value.items()}
    if is_dataclass(value):
        return {
            field.name: _format_tool_result(getattr(value, field.name)) for field in fields(value)
        }
    return str(value)
