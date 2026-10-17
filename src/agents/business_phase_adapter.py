"""Production composition for provider-neutral Agent phase decisions.

The top-level composite combines the generic business-intent catalogue with sealed
domain authorities before a model request is built.  Existing registry action
compilers remain private implementation details; World reads and local control
choices compile directly to explicit Agent-owned dispositions.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agents.agent_phase import (
    CompiledBusinessAction,
    CompiledPhaseDecision,
    CompiledWorldRead,
    LocalControlDecision,
    RouteBinding,
    RouteRegistry,
)
from agents.business_decision import BusinessIntentSpec, LLMBusinessDecisionV1
from agents.domain_phase_adapters import BoundDomainPhase
from agents.decision_errors import FrameworkAuthorityError
from agents.provider_boundary_policy import (
    BENCHMARK_INTERNAL_METADATA_FIELDS_V1,
    benchmark_internal_namespace_field_v1,
    benchmark_metadata_field_is_private_v1,
    provider_boundary_field_is_private_v1,
    provider_boundary_public_field_name_v1,
    provider_boundary_semantic_path_v1,
)
from agents.agent_routes import (
    AgentBusinessIntentSpec,
    AgentRouteRegistry,
)
from agents.types import ALLOWED_WORLD_TOOLS


_INTERNAL_DESCRIPTION_RE = re.compile(
    r"\b(?:CommerceWorld|VCP|Runtime)\b|(?:platform|runtime|world):[A-Za-z0-9_-]+",
    re.IGNORECASE,
)
_INTERNAL_ACTION_NAME_RE = re.compile(
    r"\b(?:commerce|platform|delegate)\.[a-z0-9_]+\b",
    re.IGNORECASE,
)


# Single source of truth for the model-facing meaning of a completed Agent
# World read.  Internal tool identities, transport provenance, and correlation
# fields never cross the provider boundary.  Adding a new Agent business read
# without first defining its platform-neutral observation fails at import time
# instead of silently exposing the internal tool row.
_PUBLIC_OBSERVATION_KIND_BY_WORLD_TOOL_V1: Mapping[str, str] = {
    "world.search_catalog": "catalog_search",
    "world.get_listing": "listing",
    "world.is_in_stock": "stock_availability",
    "world.get_merchant_reputation": "merchant_reputation",
    "world.get_balance": "account_balance",
    "world.get_order": "order",
    "world.get_friends": "social_connections",
    "world.get_friend_reviews": "friend_reviews",
    "world.get_review_evidence": "review_evidence",
    "world.get_evidence_record": "evidence_record",
    "world.get_mandate_revisions": "mandate_revisions",
    "world.get_listing_claim": "listing_claim",
    "world.list_listing_claims": "listing_claims",
}
if frozenset(_PUBLIC_OBSERVATION_KIND_BY_WORLD_TOOL_V1) != ALLOWED_WORLD_TOOLS:
    raise RuntimeError(
        "Agent business World-read inventory lacks an exact public observation mapping"
    )


@dataclass(frozen=True, slots=True)
class AgentRoutePhaseAuthority:
    """Agent-private inputs needed by the route compiler."""

    intent_specs: tuple[AgentBusinessIntentSpec, ...]
    role: str
    allowed_action_kinds: frozenset[str]
    activated_skill_names: tuple[str, ...] | None
    allow_actor_reports: bool
    actor_context: Mapping[str, Any]


class AgentRoutePhaseAdapter:
    """Expose provider-neutral intents while reusing exact domain compilers."""

    def __init__(self, registry: AgentRouteRegistry) -> None:
        self._registry = registry

    def build_authority(self, **inputs: Any) -> AgentRoutePhaseAuthority:
        """Build the route authority behind the unified phase seam."""

        try:
            return AgentRoutePhaseAuthority(**inputs)
        except TypeError as exc:
            raise FrameworkAuthorityError(
                "Agent route phase authority inputs are incomplete"
            ) from exc

    def decision_schema(
        self,
        authority: object,
    ) -> tuple[BusinessIntentSpec, ...]:
        resolved = _authority(authority)
        rows: list[BusinessIntentSpec] = []
        for intent_spec in resolved.intent_specs:
            operation = self._registry.business_operation_for_source(intent_spec.name)
            category = {
                "read": "observe",
                "action": "act",
                "control": "control",
            }[intent_spec.category]
            schema = intent_spec.business_parameters or intent_spec.parameters
            description = _public_description(
                intent_spec.description,
                operation=operation,
            )
            rows.append(
                BusinessIntentSpec(
                    intent=operation,
                    description=description,
                    parameters=copy.deepcopy(dict(schema)),
                    category=category,
                    source_name=intent_spec.name,
                )
            )
        if len({row.intent for row in rows}) != len(rows):
            raise FrameworkAuthorityError("active business intent surface is ambiguous")
        return tuple(rows)

    def route_bindings(
        self,
        authority: object,
    ) -> tuple[RouteBinding, ...]:
        """Return only routes reachable from the current business surface."""

        resolved = _authority(authority)
        routes = tuple(
            binding
            for intent_spec in resolved.intent_specs
            if (binding := self._registry.business_route_binding_for_source(intent_spec.name))
            is not None
        )
        if len(routes) != len(frozenset(routes)):
            raise FrameworkAuthorityError("active business route surface is ambiguous")
        return routes

    def compile(
        self,
        authority: object,
        decision: LLMBusinessDecisionV1,
    ) -> CompiledPhaseDecision:
        resolved = _authority(authority)
        spec = self._spec_for_decision(resolved, decision)
        arguments = _decision_arguments(spec, decision)
        route = self._registry.business_route_binding_for_source(spec.source_name)
        if spec.category == "observe":
            if route is not None:
                raise FrameworkAuthorityError("business observation unexpectedly owns a wire route")
            world_tool = self._registry.world_read_tool_name(spec.source_name)
            if world_tool is None:
                raise FrameworkAuthorityError(
                    "business observation has no internal World read binding"
                )
            return CompiledWorldRead(
                operation=decision.intent,
                tool=world_tool,
                args=copy.deepcopy(arguments),
                source_name=spec.source_name,
            )
        if spec.category == "control":
            if (
                route is not None
                or self._registry.world_read_tool_name(spec.source_name) is not None
            ):
                raise FrameworkAuthorityError(
                    "local business control unexpectedly owns an external binding"
                )
            reason = arguments.get("reason")
            return LocalControlDecision(
                intent=decision.intent,
                reason=(
                    str(reason).strip()
                    if isinstance(reason, str) and reason.strip()
                    else decision.message.strip()
                    if isinstance(decision.message, str) and decision.message.strip()
                    else None
                ),
                source_name=spec.source_name,
            )
        if route is None:
            raise FrameworkAuthorityError("business action has no internal route binding")
        action = self._registry.compile_action(
            role=resolved.role,
            source_name=spec.source_name,
            arguments=arguments,
            allowed_action_kinds=resolved.allowed_action_kinds,
            activated_skill_names=resolved.activated_skill_names,
            allow_actor_reports=resolved.allow_actor_reports,
            actor_context=resolved.actor_context,
        )
        return CompiledBusinessAction(
            operation=route.operation,
            action_kind=action.action_kind,
            destination=action.destination,
            payload=copy.deepcopy(dict(action.payload)),
            source_name=spec.source_name,
        )

    def route_binding_for_intent(
        self,
        authority: object,
        intent: str,
    ) -> RouteBinding | None:
        """Return the generic route directly associated with one intent."""

        resolved = _authority(authority)
        matches = [row for row in self.decision_schema(resolved) if row.intent == intent]
        if len(matches) != 1:
            raise FrameworkAuthorityError(
                "generic business intent has no unique source inventory entry"
            )
        return self._registry.business_route_binding_for_source(matches[0].source_name)

    def _spec_for_decision(
        self,
        authority: AgentRoutePhaseAuthority,
        decision: LLMBusinessDecisionV1,
    ) -> BusinessIntentSpec:
        matches = [row for row in self.decision_schema(authority) if row.intent == decision.intent]
        if len(matches) != 1:
            raise FrameworkAuthorityError("parsed business intent lost its Agent compiler source")
        return matches[0]


@dataclass(frozen=True, slots=True)
class CompositePhaseAuthority:
    """One generic business-intent surface plus authenticated domain owners."""

    generic: AgentRoutePhaseAuthority
    domains: tuple[BoundDomainPhase, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.generic, AgentRoutePhaseAuthority):
            raise FrameworkAuthorityError("composite phase has the wrong generic authority")
        if not isinstance(self.domains, tuple) or any(
            not isinstance(row, BoundDomainPhase) for row in self.domains
        ):
            raise FrameworkAuthorityError("composite phase has malformed domain authorities")


@dataclass(frozen=True, slots=True)
class _CompositeIntentOwner:
    spec: BusinessIntentSpec
    adapter: object
    authority: object
    routes: tuple[RouteBinding, ...]


@dataclass(frozen=True, slots=True)
class _CompositeSurface:
    specs: tuple[BusinessIntentSpec, ...]
    routes: tuple[RouteBinding, ...]
    owner_by_intent: Mapping[str, _CompositeIntentOwner]


class CompositePhaseAdapter:
    """Compose the generic catalogue with exact domain-owned authorities.

    Domain ownership is declared by each phase's complete route closure, not
    reconstructed from a selected action's destination.  This is important
    for high-level choices such as after-sales approve/deny branches and for
    route-less choices such as rejecting a ranked batch locally.
    """

    def __init__(self, generic: AgentRoutePhaseAdapter) -> None:
        if not isinstance(generic, AgentRoutePhaseAdapter):
            raise TypeError("composite phase requires an Agent route phase adapter")
        self._generic = generic

    def build_authority(self, **inputs: Any) -> CompositePhaseAuthority:
        raw_domains = inputs.pop("domain_phases", ())
        if not isinstance(raw_domains, tuple) or any(
            not isinstance(row, BoundDomainPhase) for row in raw_domains
        ):
            raise FrameworkAuthorityError("composite domain phases must be an authenticated tuple")
        generic = self._generic.build_authority(**inputs)
        authority = CompositePhaseAuthority(
            generic=generic,
            domains=raw_domains,
        )
        # Resolve once at authority construction so ambiguous phase ownership
        # fails before a provider request is possible.
        self._surface(authority)
        return authority

    def decision_schema(self, authority: object) -> tuple[BusinessIntentSpec, ...]:
        return self._surface(_composite_authority(authority)).specs

    def route_bindings(self, authority: object) -> tuple[RouteBinding, ...]:
        return self._surface(_composite_authority(authority)).routes

    def compile(
        self,
        authority: object,
        decision: LLMBusinessDecisionV1,
    ) -> CompiledPhaseDecision:
        resolved = _composite_authority(authority)
        surface = self._surface(resolved)
        try:
            owner = surface.owner_by_intent[decision.intent]
        except KeyError as exc:
            raise FrameworkAuthorityError("business intent has no composite phase owner") from exc
        # Dispatch is solely by the frozen intent inventory.  In particular,
        # it never guesses a domain by reverse-matching action kind or
        # destination after compilation.
        compiled = owner.adapter.compile(owner.authority, decision)
        if isinstance(compiled, CompiledBusinessAction):
            candidates = [route for route in owner.routes if route.operation == compiled.operation]
            if len(candidates) != 1:
                raise FrameworkAuthorityError(
                    "composite owner compiled outside its exact route closure"
                )
        elif isinstance(compiled, (CompiledWorldRead, LocalControlDecision)):
            if (
                compiled.source_name != owner.spec.source_name
                or (
                    isinstance(compiled, CompiledWorldRead)
                    and compiled.operation != owner.spec.intent
                )
                or (
                    isinstance(compiled, LocalControlDecision)
                    and compiled.intent != owner.spec.intent
                )
            ):
                raise FrameworkAuthorityError(
                    "composite owner compiled outside its source inventory"
                )
        else:
            raise FrameworkAuthorityError(
                "composite owner returned an unsupported compile disposition"
            )
        return compiled

    def _surface(self, authority: CompositePhaseAuthority) -> _CompositeSurface:
        generic_specs = self._generic.decision_schema(authority.generic)
        generic_routes = {
            spec.intent: self._generic.route_binding_for_intent(
                authority.generic,
                spec.intent,
            )
            for spec in generic_specs
        }

        domain_owners: dict[str, _CompositeIntentOwner] = {}
        domain_route_owner: dict[RouteBinding, int] = {}
        all_domain_routes: list[RouteBinding] = []
        for index, phase in enumerate(authority.domains):
            if phase.authority.role != authority.generic.role:
                raise FrameworkAuthorityError("domain phase differs from the generic actor role")
            routes = phase.adapter.route_bindings(phase.authority)
            if len(routes) != len(frozenset(routes)):
                raise FrameworkAuthorityError("domain phase route closure contains duplicates")
            for route in routes:
                prior = domain_route_owner.get(route)
                if prior is not None and prior != index:
                    raise FrameworkAuthorityError("composite domain route ownership is ambiguous")
                domain_route_owner[route] = index
                all_domain_routes.append(route)
            for spec in phase.adapter.decision_schema(phase.authority):
                if spec.intent in domain_owners:
                    raise FrameworkAuthorityError("composite domain intent ownership is ambiguous")
                domain_owners[spec.intent] = _CompositeIntentOwner(
                    spec=spec,
                    adapter=phase.adapter,
                    authority=phase.authority,
                    routes=routes,
                )

        domain_route_set = frozenset(all_domain_routes)
        owners: dict[str, _CompositeIntentOwner] = {}
        retained_generic_routes: list[RouteBinding] = []
        specs: list[BusinessIntentSpec] = []
        for spec in generic_specs:
            route = generic_routes[spec.intent]
            # An authenticated domain spec wins an identical business intent.
            # Its entire route closure also suppresses protocol-shaped generic
            # siblings, including one-to-many approve/deny branches.
            if spec.intent in domain_owners or (route is not None and route in domain_route_set):
                continue
            owner_routes = () if route is None else (route,)
            owners[spec.intent] = _CompositeIntentOwner(
                spec=spec,
                adapter=self._generic,
                authority=authority.generic,
                routes=owner_routes,
            )
            specs.append(spec)
            if route is not None:
                retained_generic_routes.append(route)

        for intent, owner in domain_owners.items():
            if intent in owners:
                raise FrameworkAuthorityError(
                    "composite business intent source inventory is ambiguous"
                )
            owners[intent] = owner
            specs.append(owner.spec)

        routes = tuple((*retained_generic_routes, *all_domain_routes))
        if len(routes) != len(frozenset(routes)):
            raise FrameworkAuthorityError("composite route closure is ambiguous")
        try:
            RouteRegistry(routes)
        except ValueError as exc:
            raise FrameworkAuthorityError(
                "composite route closure has conflicting registry identities"
            ) from exc
        if len(owners) != len(specs):
            raise FrameworkAuthorityError(
                "composite business intent source inventory is incomplete"
            )
        source_names = [row.source_name for row in specs]
        if len(source_names) != len(frozenset(source_names)):
            raise FrameworkAuthorityError(
                "composite business intent compiler sources are ambiguous"
            )
        return _CompositeSurface(
            specs=tuple(specs),
            routes=tuple(sorted(routes, key=lambda row: row.operation)),
            owner_by_intent=owners,
        )


def public_business_observations(
    *,
    inbound_kind: str,
    inbound_payload: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    phase_id: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Project actor-visible business facts without framework wire metadata."""

    rows: list[Mapping[str, Any]] = [
        {
            "event": _business_event_name(inbound_kind),
            "facts": _public_value(
                inbound_payload,
                benchmark_phase_id=phase_id,
            ),
        }
    ]
    for item in history:
        if item.get("step") == "tool_call":
            results = item.get("results")
            if (
                not isinstance(results, Sequence)
                or isinstance(results, (str, bytes, bytearray))
                or not results
            ):
                raise FrameworkAuthorityError(
                    "completed Agent World-read history is malformed"
                )
            rows.append(
                {
                    "observed_business_facts": [
                        _public_world_read_observation(result) for result in results
                    ]
                }
            )
        elif item.get("step") in {
            "semantic_response_error",
            "semantic_validation_error",
        }:
            rows.append(
                {
                    "previous_decision_rejected": True,
                    "code": str(item.get("error_code", "business_validation_rejected")),
                    "reason": str(item.get("error", "invalid business decision"))[:500],
                }
            )
    return tuple(rows)


def _public_world_read_observation(value: Any) -> Mapping[str, Any]:
    """Project one completed internal World read to public business facts.

    The input row is Agent-owned step history.  Select fields explicitly so a
    transport addition such as ``source_msg_id`` can never become provider
    context by accident.  An unregistered tool is an Agent/framework defect,
    not a model-visible observation, and therefore fails closed.
    """

    if not isinstance(value, Mapping):
        raise FrameworkAuthorityError(
            "completed Agent World-read result is not an object"
        )
    tool = value.get("tool")
    observation_kind = (
        _PUBLIC_OBSERVATION_KIND_BY_WORLD_TOOL_V1.get(tool)
        if isinstance(tool, str)
        else None
    )
    if observation_kind is None:
        raise FrameworkAuthorityError(
            "completed Agent World read has no public business observation mapping"
        )
    criteria = value.get("args")
    if not isinstance(criteria, Mapping) or "result" not in value:
        raise FrameworkAuthorityError(
            "completed Agent World-read result lacks criteria or facts"
        )
    return {
        "observation_kind": observation_kind,
        "criteria": _public_value(
            criteria,
            semantic_path=("observed_business_facts", "criteria"),
        ),
        "facts": _public_value(
            value["result"],
            semantic_path=("observed_business_facts", "facts"),
        ),
    }


def public_actor_task_facts(
    actor_input: Mapping[str, Any],
    *,
    phase_id: str | None = None,
) -> Mapping[str, Any]:
    """Project spawn-frozen public business facts for every Agent turn.

    Task execution routes and result schemas are compiled by the Agent and are
    never repeated to the model.  Resource candidates, instructions, business
    constraints, and evidence references remain available across later
    Platform turns so the public-reference bridge can preserve authority.
    """

    output: dict[str, Any] = {}
    task_context = actor_input.get("task_context")
    task_facts = _public_task_facts(task_context)
    if task_facts:
        output["task"] = task_facts
    benchmark_contract = actor_input.get("benchmark_contract")
    benchmark_facts = _public_benchmark_facts(
        benchmark_contract,
        phase_id=phase_id,
    )
    if benchmark_facts:
        output["benchmark"] = benchmark_facts
    return output


_PRIVATE_KEYS = frozenset(
    {
        "action_kind",
        "allowed_routes",
        "authority",
        "authority_id",
        "benchmark_task_id",
        "cert_id",
        "claim_compilation_authorized_claim_ids",
        "claim_compilation_templates",
        "contract_id",
        "destination",
        "expected_answer",
        "ground_truth",
        "ideal_trajectory",
        "idempotency_key",
        "in_reply_to",
        "msg_id",
        "oracle",
        "request_msg_id",
        "response_actions",
        "route",
        "result_key",
        "result_table",
        "success_oracle",
        "task_blocked_action_kinds",
        "task_pre_action_read_gate",
        "task_id",
        "world_thread_projection",
    }
)


def _public_value(
    value: Any,
    *,
    semantic_path: tuple[str, ...] = (),
    benchmark_phase_id: str | None = None,
) -> Any:
    if isinstance(value, Mapping):
        semantic_path = provider_boundary_semantic_path_v1(semantic_path, value)
        output: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.casefold()
            public_key = provider_boundary_public_field_name_v1(
                semantic_path,
                key,
            )
            if public_key != key:
                output[public_key] = _public_value(
                    item,
                    semantic_path=(*semantic_path, public_key),
                    benchmark_phase_id=benchmark_phase_id,
                )
                continue
            if normalized == "task_context":
                # Runtime task context mixes useful public business facts with
                # Agent-owned execution routes and duplicated JSON schemas.
                # The latter are already compiled into ``allowed_intents``;
                # retaining them here both leaks framework structure and makes
                # schema property names look like observed resource IDs.
                projected = _public_task_facts(item)
                if projected:
                    output["task_facts"] = projected
                continue
            if normalized == "benchmark_contract":
                # Initial Runtime messages may repeat the frozen benchmark
                # contract beside ordinary business facts.  Project it through
                # the same allowlist used for persistent Agent inputs so route
                # destinations and execution mechanics never reach the LLM.
                projected = _public_benchmark_facts(
                    item,
                    phase_id=benchmark_phase_id,
                )
                if projected:
                    output["benchmark_facts"] = projected
                continue
            if (
                normalized in _PRIVATE_KEYS
                or benchmark_internal_namespace_field_v1(normalized)
                or benchmark_metadata_field_is_private_v1(
                    semantic_path,
                    normalized,
                )
                or provider_boundary_field_is_private_v1(
                    semantic_path,
                    normalized,
                )
            ):
                continue
            if normalized.endswith(("_id", "_ids")):
                # Structured identities are deliberately left intact for the
                # shared AgentPhase public-reference bridge, which either
                # hides framework ids or replaces finite business ids with
                # reversible opaque refs.  Sanitizing their string values here
                # would destroy that authority binding.
                output[key] = copy.deepcopy(item)
                continue
            output[key] = _public_value(
                item,
                semantic_path=(*semantic_path, normalized),
                benchmark_phase_id=benchmark_phase_id,
            )
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _public_value(
                item,
                semantic_path=semantic_path,
                benchmark_phase_id=benchmark_phase_id,
            )
            for item in value
        ]
    if isinstance(value, str):
        projected = value
        for prefix in (
            "buyer:",
            "consumer:",
            "merchant:",
            "platform:",
            "runtime:",
            "world:",
        ):
            projected = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(prefix)}[^\s,;\]\[(){{}}]+",
                "marketplace_service",
                projected,
                flags=re.IGNORECASE,
            )
        return projected
    return copy.deepcopy(value)


_PRIVATE_TASK_CONTEXT_KEYS = (
    frozenset(
        {
            "action_schema",
            "claim_compilation_authorized_claim_ids",
            "claim_compilation_templates",
            "execution_contract",
            "schema_version",
            "task_id",
        }
    )
    | BENCHMARK_INTERNAL_METADATA_FIELDS_V1
)


def _public_task_facts(value: Any) -> dict[str, Any]:
    """Keep task business inputs while dropping Agent execution metadata."""

    if not isinstance(value, Mapping):
        return {}
    selected = {
        str(key): copy.deepcopy(item)
        for key, item in value.items()
        if str(key).casefold() not in _PRIVATE_TASK_CONTEXT_KEYS
    }
    projected = _public_value(selected, semantic_path=("task_facts",))
    return dict(projected) if isinstance(projected, Mapping) else {}


_PRIVATE_BENCHMARK_CONTEXT_KEYS = (
    frozenset(
        {
            "authority",
            "execution_contract",
            "schema_version",
            "scoring",
            "task_id",
            "provider_visibility",
            "response_extraction_mode",
        }
    )
    | BENCHMARK_INTERNAL_METADATA_FIELDS_V1
)


def _public_benchmark_facts(
    value: Any,
    *,
    phase_id: str | None = None,
) -> dict[str, Any]:
    """Keep frozen business inputs without benchmark/framework mechanics."""

    if not isinstance(value, Mapping):
        return {}
    selected = {
        str(key): copy.deepcopy(item)
        for key, item in value.items()
        if str(key).casefold() not in _PRIVATE_BENCHMARK_CONTEXT_KEYS
    }
    visibility = value.get("provider_visibility")
    if phase_id is not None and isinstance(visibility, Mapping):
        phase_visibility = visibility.get(phase_id)
        if isinstance(phase_visibility, Mapping):
            excluded = phase_visibility.get("exclude_fields", ())
            if not isinstance(excluded, Sequence) or isinstance(
                excluded,
                (str, bytes, bytearray),
            ):
                raise FrameworkAuthorityError(
                    "benchmark provider visibility exclude_fields must be an array"
                )
            for field in excluded:
                if not isinstance(field, str) or not field:
                    raise FrameworkAuthorityError(
                        "benchmark provider visibility contains an invalid field"
                    )
                selected.pop(field, None)
    projected = _public_value(selected, semantic_path=("benchmark_facts",))
    return dict(projected) if isinstance(projected, Mapping) else {}


def _business_event_name(value: str) -> str:
    name = str(value).split(".", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_") or "business_event"


def _public_description(value: str, *, operation: str) -> str:
    if _INTERNAL_ACTION_NAME_RE.search(value):
        return f"Perform the {operation.replace('_', ' ')} business choice."
    description = _INTERNAL_DESCRIPTION_RE.sub("marketplace", value).strip()
    return description or f"Perform the {operation.replace('_', ' ')} business choice."


def _decision_arguments(
    spec: BusinessIntentSpec,
    decision: LLMBusinessDecisionV1,
) -> dict[str, Any]:
    arguments = copy.deepcopy(dict(decision.arguments))
    if decision.message is not None and "message" not in arguments:
        properties = spec.parameters.get("properties")
        if isinstance(properties, Mapping) and "message" in properties:
            arguments["message"] = decision.message
    return arguments


def _authority(value: object) -> AgentRoutePhaseAuthority:
    if not isinstance(value, AgentRoutePhaseAuthority):
        raise FrameworkAuthorityError("Agent route phase adapter received the wrong authority type")
    return value


def _composite_authority(value: object) -> CompositePhaseAuthority:
    if not isinstance(value, CompositePhaseAuthority):
        raise FrameworkAuthorityError("composite phase adapter received the wrong authority type")
    return value


__all__ = [
    "CompositePhaseAdapter",
    "CompositePhaseAuthority",
    "AgentRoutePhaseAuthority",
    "AgentRoutePhaseAdapter",
    "public_actor_task_facts",
    "public_business_observations",
]
