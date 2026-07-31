"""Unified internal phase and decision bridge for CommerceWorld Agents."""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agents.business_decision import (
    BusinessDecisionContractError,
    BusinessIntentSpec,
    LLMBusinessDecisionV1,
    LLMDecisionRequestV1,
    ModelBusinessChoiceV1,
    PUBLIC_REFERENCE_POLICY_V1,
    canonical_business_value_digest,
)
from agents.business_intent_schema_policy import (
    PUBLIC_REFERENCE_ALIAS_HEX_CHARS,
    PUBLIC_REFERENCE_ALIAS_PREFIX,
    framework_owned_reference_field,
    require_business_intent_authority_safe,
)
from agents.decision_errors import (
    DeterministicCompilerError,
    FrameworkAuthorityError,
    ModelBusinessDecisionError,
)
from agents.provider_boundary_policy import (
    benchmark_internal_namespace_field_v1,
    provider_boundary_field_is_private_v1,
    provider_boundary_public_field_name_v1,
    provider_boundary_semantic_path_v1,
)


_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_PRIVATE_ADDRESS_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:buyer|consumer|merchant|platform|runtime|world):"
    r"[^\s,;\]\[(){}]+",
    flags=re.IGNORECASE,
)
_PRIVATE_ACTION_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:commerce|platform|delegate)\.[a-z0-9_]+",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RouteBinding:
    """The single internal wire route for one business operation."""

    operation: str
    action_kind: str
    destination: str
    roles: frozenset[str]
    source_name: str

    def __post_init__(self) -> None:
        if not _OPERATION_RE.fullmatch(self.operation):
            raise ValueError("route binding operation is invalid")
        if not self.action_kind or not self.destination or not self.source_name:
            raise ValueError("route binding is incomplete")
        if not self.roles or not self.roles <= {"buyer", "merchant"}:
            raise ValueError("route binding roles are invalid")


class RouteRegistry:
    """Deterministic, immutable-by-convention route registry."""

    def __init__(self, bindings: Sequence[RouteBinding]) -> None:
        rows = tuple(bindings)
        by_operation = {row.operation: row for row in rows}
        by_source = {row.source_name: row for row in rows}
        if len(by_operation) != len(rows):
            raise ValueError("business operations must map to one route each")
        if len(by_source) != len(rows):
            raise ValueError("route source names must be unique")
        self._by_operation = by_operation
        self._by_source = by_source

    def by_operation(self, operation: str) -> RouteBinding:
        try:
            return self._by_operation[operation]
        except KeyError as exc:
            raise FrameworkAuthorityError(
                f"business operation {operation!r} has no internal route"
            ) from exc

    def by_source(self, source_name: str) -> RouteBinding:
        try:
            return self._by_source[source_name]
        except KeyError as exc:
            raise FrameworkAuthorityError(
                "Agent compiler source has no internal route binding"
            ) from exc

    def by_route(
        self,
        action_kind: str,
        destination: str,
        *,
        role: str,
        source_name: str | None = None,
    ) -> RouteBinding:
        """Resolve one compiled wire route through the sole registry.

        Semantic variants may intentionally share a wire action kind and
        destination.  Such routes must supply their registered compiler source
        so the lookup stays unambiguous.
        """

        matches = [
            row
            for row in self._by_operation.values()
            if row.action_kind == action_kind
            and _destination_matches(row.destination, destination)
            and role in row.roles
            and (source_name is None or row.source_name == source_name)
        ]
        if len(matches) != 1:
            raise FrameworkAuthorityError(
                "compiled domain choice has no unique internal route binding"
            )
        return matches[0]

    @property
    def bindings(self) -> tuple[RouteBinding, ...]:
        return tuple(sorted(self._by_operation.values(), key=lambda row: row.operation))


@dataclass(frozen=True, slots=True)
class CompiledBusinessAction:
    """One exact Agent-owned wire action compiled from a model business choice."""

    operation: str
    action_kind: str
    destination: str
    payload: Mapping[str, Any]
    source_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise ValueError("compiled business payload must be an object")


@dataclass(frozen=True, slots=True)
class CompiledWorldRead:
    """One Agent-owned World read selected through a business observation.

    ``tool`` is the existing internal WorldTools operation.  It is deliberately
    absent from :class:`LLMDecisionRequestV1`; the model selects ``operation``
    and business arguments while the Agent binds that choice to this exact
    read implementation.
    """

    operation: str
    tool: str
    args: Mapping[str, Any]
    source_name: str

    def __post_init__(self) -> None:
        if not _OPERATION_RE.fullmatch(self.operation):
            raise ValueError("compiled World read operation is invalid")
        if not self.tool.startswith("world.") or not self.source_name:
            raise ValueError("compiled World read source is incomplete")
        if not isinstance(self.args, Mapping):
            raise ValueError("compiled World read arguments must be an object")


@dataclass(frozen=True, slots=True)
class LocalControlDecision:
    """A validated business choice that emits neither a wire action nor a read."""

    intent: str
    reason: str | None
    source_name: str

    def __post_init__(self) -> None:
        if not _OPERATION_RE.fullmatch(self.intent) or not self.source_name:
            raise ValueError("local control decision source is incomplete")
        if self.reason is not None and (
            not isinstance(self.reason, str) or not self.reason.strip()
        ):
            raise ValueError("local control decision reason must be non-empty text")


CompiledPhaseDecision = CompiledBusinessAction | CompiledWorldRead | LocalControlDecision


@runtime_checkable
class PhaseAdapter(Protocol):
    """Domain seam retained by every Agent phase implementation."""

    def build_authority(self, **inputs: Any) -> object:
        """Validate authenticated Agent inputs into domain authority."""
        ...

    def decision_schema(self, authority: object) -> tuple[BusinessIntentSpec, ...]:
        """Return the provider-neutral choices authorized for this phase."""
        ...

    def route_bindings(self, authority: object) -> tuple[RouteBinding, ...]:
        """Return the closed set of internal routes this authority may compile."""
        ...

    def compile(
        self,
        authority: object,
        decision: LLMBusinessDecisionV1,
    ) -> CompiledPhaseDecision:
        """Compile one model choice into an explicit Agent-owned disposition."""
        ...


@dataclass(frozen=True, slots=True)
class AgentPhaseContract:
    """Agent-private authority and adapter selected for one inbound phase."""

    phase_id: str
    role: str
    goal: str
    observations: tuple[Mapping[str, Any], ...]
    authority: object
    adapter: PhaseAdapter
    allowed_operations: frozenset[str]
    route_bindings: tuple[RouteBinding, ...]
    # Authenticated Agent context used only to bind framework-owned schema
    # fields.  These rows are never copied into ``LLMDecisionRequestV1`` and
    # cannot contribute model-selectable business references.
    framework_observations: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.phase_id or not self.goal:
            raise FrameworkAuthorityError("Agent phase identity is incomplete")
        if self.role not in {"buyer", "merchant"}:
            raise FrameworkAuthorityError("Agent phase role is invalid")
        if not isinstance(self.adapter, PhaseAdapter):
            raise FrameworkAuthorityError("Agent phase adapter is invalid")
        specs = self.adapter.decision_schema(self.authority)
        operations = frozenset(row.intent for row in specs)
        if operations != self.allowed_operations:
            raise FrameworkAuthorityError(
                "Agent phase operations disagree with its decision adapter"
            )
        adapter_routes = self.adapter.route_bindings(self.authority)
        if len(adapter_routes) != len(self.route_bindings) or frozenset(
            adapter_routes
        ) != frozenset(self.route_bindings):
            raise FrameworkAuthorityError("Agent phase routes disagree with its authority adapter")
        if any(self.role not in row.roles for row in self.route_bindings):
            raise FrameworkAuthorityError("Agent phase contains a route outside the actor role")
        # Construct once during validation so ambiguous operations or compiler
        # sources fail before any provider request.
        RouteRegistry(self.route_bindings)

    @property
    def route_registry(self) -> RouteRegistry:
        return RouteRegistry(self.route_bindings)


_INTERNAL_SINGULAR_SUFFIX = str(PUBLIC_REFERENCE_POLICY_V1["internal_singular_suffix"])
_INTERNAL_PLURAL_SUFFIX = str(PUBLIC_REFERENCE_POLICY_V1["internal_plural_suffix"])
_PUBLIC_SINGULAR_SUFFIX = str(PUBLIC_REFERENCE_POLICY_V1["public_singular_suffix"])
_PUBLIC_PLURAL_SUFFIX = str(PUBLIC_REFERENCE_POLICY_V1["public_plural_suffix"])
_PUBLIC_ALIAS_PREFIX = PUBLIC_REFERENCE_ALIAS_PREFIX
_PUBLIC_ALIAS_HEX_CHARS = PUBLIC_REFERENCE_ALIAS_HEX_CHARS
_PUBLIC_ALIAS_DOMAIN = str(PUBLIC_REFERENCE_POLICY_V1["alias_domain_separator"])
if (
    _PUBLIC_ALIAS_PREFIX != PUBLIC_REFERENCE_POLICY_V1["alias_prefix"]
    or _PUBLIC_ALIAS_HEX_CHARS != PUBLIC_REFERENCE_POLICY_V1["alias_hash_hex_chars"]
):
    raise RuntimeError("public business reference policy constants disagree")

_SEMANTIC_REFERENCE_ROOTS = frozenset(
    {
        "buyer",
        "claim",
        "dispute",
        "evidence",
        "exchange",
        "inquiry",
        "listing",
        "merchant",
        "offer",
        "order",
        "product",
        "recipient",
        "record",
        "refund",
        "return",
        "review",
        "shipment",
        "sku",
    }
)

# Frozen CommerceWorld resource relations.  These are business-reference
# semantics, not wire routes: a target result field may select only values
# already observed under one of its explicitly related structured fields.
# Keeping the table here avoids each benchmark family inventing its own ID
# rewrite while avoiding unsafe arbitrary suffix guessing.
_REFERENCE_STEM_RELATIONS: Mapping[str, frozenset[str]] = {
    "listing": frozenset({"sku"}),
    "considered_listing": frozenset({"candidate_listing", "listing", "sku"}),
    "used_signal": frozenset({"signal", "social_signal"}),
    "applied_update": frozenset({"update", "preference_update"}),
    "governing_instruction": frozenset({"instruction", "authority_instruction"}),
    "discarded_input": frozenset(
        {
            "signal",
            "social_signal",
            "update",
            "preference_update",
            "instruction",
            "authority_instruction",
        }
    ),
    "source": frozenset({"record", "evidence_record"}),
    "evidence_source": frozenset({"record", "evidence_record"}),
}

# Actor-visible business records may carry storage ACLs, wire signatures, or
# replay fingerprints beside their semantic facts.  Those fields are useful to
# the Agent when validating authority, but they are not business observations
# for the LLM and can contain raw participant/service addresses.  Strip them
# during the same shared projection that converts resource ids to opaque refs.
_AGENT_PRIVATE_OBSERVATION_FIELDS = frozenset(
    {
        "action_kind",
        "authority",
        "destination",
        "oracle",
        "read_acl",
        "route",
        "routes",
        "signature",
        "success_oracle",
        "write_acl",
    }
)


class _UnavailableBusinessReferenceSurface(Exception):
    """An otherwise valid intent has no actor-visible resource candidate yet."""


@dataclass(frozen=True, slots=True)
class _PublicReferenceProjection:
    """One deterministic, reversible model-facing reference surface."""

    public_goal: str
    public_observations: tuple[Mapping[str, Any], ...]
    public_specs: tuple[BusinessIntentSpec, ...]
    internal_by_alias: Mapping[str, str]
    internal_specs: Mapping[str, BusinessIntentSpec]
    hidden_by_intent: Mapping[str, Mapping[tuple[str, ...], Any]]

    @classmethod
    def build(
        cls,
        *,
        goal: str,
        observations: Sequence[Mapping[str, Any]],
        specs: Sequence[BusinessIntentSpec],
        framework_observations: Sequence[Mapping[str, Any]] = (),
    ) -> "_PublicReferenceProjection":
        internal_specs = {row.intent: row for row in specs}
        if len(internal_specs) != len(specs):
            raise FrameworkAuthorityError("Agent public-reference surface has duplicate intents")

        candidates_by_stem: dict[str, set[str]] = {}
        all_internal: set[str] = set()
        hidden_observed_by_name: dict[str, list[Any]] = {}
        hidden_goal_values: set[str] = set()
        for observation in observations:
            _collect_observation_references(
                observation,
                candidates_by_stem=candidates_by_stem,
                all_internal=all_internal,
                hidden_observed_by_name=hidden_observed_by_name,
                hidden_goal_values=hidden_goal_values,
                path="observations",
            )
        for observation in framework_observations:
            _collect_framework_reference_bindings(
                observation,
                hidden_observed_by_name=hidden_observed_by_name,
                hidden_goal_values=hidden_goal_values,
                path="framework_observations",
            )
        for spec in specs:
            _collect_schema_enum_references(
                spec.parameters,
                candidates_by_stem=candidates_by_stem,
                all_internal=all_internal,
                path=f"intent[{spec.intent}]",
            )

        # Hash aliases are stable across request retries and across later turns
        # whose visible candidate set has grown.  Collisions fail closed rather
        # than silently making two authorities indistinguishable.
        aliases_by_internal: dict[str, str] = {}
        internal_by_alias: dict[str, str] = {}
        for internal in sorted(all_internal):
            alias = _public_alias(internal)
            if alias in all_internal:
                raise FrameworkAuthorityError(
                    "an internal business identity collides with the public reference namespace"
                )
            previous = internal_by_alias.get(alias)
            if previous is not None and previous != internal:
                raise FrameworkAuthorityError("public business reference alias collision")
            aliases_by_internal[internal] = alias
            internal_by_alias[alias] = internal

        public_observations = tuple(
            _project_observation_references(
                observation,
                aliases_by_internal=aliases_by_internal,
                path=f"observations[{index}]",
                semantic_path=("observations",),
            )
            for index, observation in enumerate(observations)
        )
        public_specs: list[BusinessIntentSpec] = []
        hidden_by_intent: dict[str, Mapping[tuple[str, ...], Any]] = {}
        for spec in specs:
            hidden_bindings: dict[tuple[str, ...], Any] = {}
            try:
                public_parameters = _project_reference_schema(
                    spec.parameters,
                    candidates_by_stem=candidates_by_stem,
                    aliases_by_internal=aliases_by_internal,
                    hidden_observed_by_name=hidden_observed_by_name,
                    hidden_bindings=hidden_bindings,
                    property_path=(),
                    path=f"intent[{spec.intent}]",
                )
            except _UnavailableBusinessReferenceSurface:
                # Read-after-observe capabilities are dormant until an
                # authenticated observation supplies a finite reference set.
                # One dormant optional intent must not invalidate unrelated
                # search, control, or already-grounded business choices.
                continue
            public_specs.append(
                BusinessIntentSpec(
                    intent=spec.intent,
                    description=spec.description,
                    parameters=public_parameters,
                    category=spec.category,
                    source_name=spec.source_name,
                )
            )
            try:
                require_business_intent_authority_safe(
                    public_specs[-1].parameters,
                    root=f"intent[{spec.intent}].arguments",
                )
            except ValueError as exc:
                raise FrameworkAuthorityError(
                    "Agent public business schema violates reference authority"
                ) from exc
            hidden_by_intent[spec.intent] = hidden_bindings
            for hidden_value in hidden_bindings.values():
                if isinstance(hidden_value, str):
                    hidden_goal_values.add(hidden_value)
                elif isinstance(hidden_value, Sequence) and not isinstance(
                    hidden_value, (str, bytes, bytearray)
                ):
                    hidden_goal_values.update(
                        item for item in hidden_value if isinstance(item, str)
                    )
        if not public_specs:
            raise FrameworkAuthorityError(
                "Agent phase has no grounded provider-visible business intent"
            )
        public_goal = _project_goal_references(
            goal,
            aliases_by_internal=aliases_by_internal,
            hidden_internal=hidden_goal_values,
        )
        return cls(
            public_goal=public_goal,
            public_observations=public_observations,
            public_specs=tuple(public_specs),
            internal_by_alias=internal_by_alias,
            internal_specs=internal_specs,
            hidden_by_intent=hidden_by_intent,
        )

    def restore_arguments(
        self,
        intent: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            spec = self.internal_specs[intent]
        except KeyError as exc:
            raise BusinessDecisionContractError(
                "model selected a business intent outside the current phase"
            ) from exc
        restored = _restore_reference_value(
            spec.parameters,
            arguments,
            internal_by_alias=self.internal_by_alias,
            hidden_bindings=self.hidden_by_intent.get(intent, {}),
            property_path=(),
            path="arguments",
        )
        if not isinstance(restored, Mapping):
            raise BusinessDecisionContractError("business decision arguments must be an object")
        return restored


class AgentDecisionBridge:
    """Shared request, parse, compile, and route-verification boundary."""

    def request(
        self,
        contract: AgentPhaseContract,
        *,
        decision_id: str,
    ) -> LLMDecisionRequestV1:
        specs = contract.adapter.decision_schema(contract.authority)
        observations = tuple(copy.deepcopy(dict(row)) for row in contract.observations)
        projection = _PublicReferenceProjection.build(
            goal=contract.goal,
            observations=observations,
            specs=specs,
            framework_observations=contract.framework_observations,
        )
        return LLMDecisionRequestV1(
            decision_id=decision_id,
            role=contract.role,
            goal=projection.public_goal,
            phase=contract.phase_id,
            observations=projection.public_observations,
            allowed_intents=projection.public_specs,
        )

    def parse(
        self,
        raw: str,
        contract: AgentPhaseContract,
    ) -> LLMBusinessDecisionV1:
        """Return the Agent-resolved decision for non-persisting callers."""

        return self.parse_with_provenance(raw, contract).decision

    def parse_with_provenance(
        self,
        raw: str,
        contract: AgentPhaseContract,
    ) -> "BridgedBusinessDecisionV1":
        """Parse one public choice and bind its Agent-resolved arguments.

        The model provenance is created before opaque public references are
        restored.  Its digest and the separately hashed resolved arguments
        make the ownership boundary explicit without persisting provider text.
        """

        specs = contract.adapter.decision_schema(contract.authority)
        projection = _PublicReferenceProjection.build(
            goal=contract.goal,
            observations=contract.observations,
            specs=specs,
            framework_observations=contract.framework_observations,
        )
        try:
            public_decision = LLMBusinessDecisionV1.parse(
                raw,
                allowed_intents=projection.public_specs,
            )
            _reject_internal_reference_keys(public_decision.arguments)
            public_specs = {row.intent: row for row in projection.public_specs}
            model_choice = ModelBusinessChoiceV1.from_validated_decision(
                public_decision,
                spec=public_specs[public_decision.intent],
            )
            resolved_decision = LLMBusinessDecisionV1(
                intent=public_decision.intent,
                arguments=projection.restore_arguments(
                    public_decision.intent,
                    public_decision.arguments,
                ),
                message=public_decision.message,
            )
            resolved_chars, resolved_sha256 = canonical_business_value_digest(
                resolved_decision.arguments
            )
            binding_sha256 = _argument_binding_sha256(
                intent=resolved_decision.intent,
                model_arguments_sha256=model_choice.arguments_sha256,
                resolved_arguments_sha256=resolved_sha256,
            )
            return BridgedBusinessDecisionV1(
                decision=resolved_decision,
                model_choice=model_choice,
                resolved_arguments_chars=resolved_chars,
                resolved_arguments_sha256=resolved_sha256,
                argument_binding_sha256=binding_sha256,
            )
        except BusinessDecisionContractError as exc:
            raise ModelBusinessDecisionError(str(exc)) from exc

    def compile(
        self,
        contract: AgentPhaseContract,
        decision: LLMBusinessDecisionV1,
    ) -> CompiledPhaseDecision:
        if decision.intent not in contract.allowed_operations:
            raise ModelBusinessDecisionError(
                "model selected a business intent outside the active Agent phase"
            )
        compiled = contract.adapter.compile(contract.authority, decision)
        specs = {row.intent: row for row in contract.adapter.decision_schema(contract.authority)}
        spec = specs[decision.intent]
        if isinstance(compiled, CompiledWorldRead):
            direct_route = next(
                (row for row in contract.route_bindings if row.operation == decision.intent),
                None,
            )
            if (
                spec.category != "observe"
                or compiled.operation != decision.intent
                or compiled.source_name != spec.source_name
                or direct_route is not None
            ):
                raise DeterministicCompilerError(
                    "phase adapter compiled outside its World-read source binding"
                )
            return CompiledWorldRead(
                operation=compiled.operation,
                tool=compiled.tool,
                args=copy.deepcopy(dict(compiled.args)),
                source_name=compiled.source_name,
            )
        if isinstance(compiled, LocalControlDecision):
            direct_route = next(
                (row for row in contract.route_bindings if row.operation == decision.intent),
                None,
            )
            if (
                compiled.intent != decision.intent
                or compiled.source_name != spec.source_name
                or direct_route is not None
                or spec.category == "observe"
            ):
                raise DeterministicCompilerError(
                    "phase adapter compiled outside its local-control source binding"
                )
            return LocalControlDecision(
                intent=compiled.intent,
                reason=compiled.reason,
                source_name=compiled.source_name,
            )
        if not isinstance(compiled, CompiledBusinessAction):
            raise DeterministicCompilerError(
                "phase adapter returned an unsupported compile disposition"
            )
        if spec.category != "act":
            raise DeterministicCompilerError(
                "phase adapter compiled a non-action choice into a wire action"
            )
        binding = contract.route_registry.by_operation(compiled.operation)
        if contract.role not in binding.roles:
            raise DeterministicCompilerError(
                "compiled business operation is outside the actor role"
            )
        if (
            binding.source_name != compiled.source_name
            or binding.action_kind != compiled.action_kind
            or not _destination_matches(binding.destination, compiled.destination)
        ):
            raise DeterministicCompilerError(
                "domain adapter compiled outside its sole internal route binding"
            )
        return CompiledBusinessAction(
            operation=compiled.operation,
            action_kind=binding.action_kind,
            # Dynamic route tokens are registry identities, not wire
            # destinations.  The domain compiler has already resolved them
            # from authenticated Agent context or a model-visible participant
            # choice, and the match above proved that resolution belongs to
            # this sole binding.  Replacing it with ``@inbound_sender`` or
            # ``@argument:recipient_id`` would leak a registry placeholder onto
            # the Runtime bus and can also retarget a reply to the wrong actor.
            destination=compiled.destination,
            payload=copy.deepcopy(dict(compiled.payload)),
            source_name=binding.source_name,
        )


@dataclass(frozen=True, slots=True)
class BridgedBusinessDecisionV1:
    """One model choice plus its explicit Agent-owned argument restoration."""

    decision: LLMBusinessDecisionV1
    model_choice: ModelBusinessChoiceV1
    resolved_arguments_chars: int
    resolved_arguments_sha256: str
    argument_binding_sha256: str

    def __post_init__(self) -> None:
        chars, digest = canonical_business_value_digest(self.decision.arguments)
        if (
            self.decision.intent != self.model_choice.intent
            or isinstance(self.resolved_arguments_chars, bool)
            or self.resolved_arguments_chars != chars
            or self.resolved_arguments_sha256 != digest
            or self.argument_binding_sha256
            != _argument_binding_sha256(
                intent=self.decision.intent,
                model_arguments_sha256=self.model_choice.arguments_sha256,
                resolved_arguments_sha256=digest,
            )
        ):
            raise BusinessDecisionContractError(
                "Agent business decision argument binding is inconsistent"
            )

    def argument_binding_dict(self) -> dict[str, Any]:
        return {
            "model_arguments_sha256": self.model_choice.arguments_sha256,
            "resolved_arguments_chars": self.resolved_arguments_chars,
            "resolved_arguments_sha256": self.resolved_arguments_sha256,
            "binding_sha256": self.argument_binding_sha256,
        }


def _argument_binding_sha256(
    *,
    intent: str,
    model_arguments_sha256: str,
    resolved_arguments_sha256: str,
) -> str:
    _chars, digest = canonical_business_value_digest(
        {
            "schema_version": "cwe.agent-argument-binding.v1",
            "intent": intent,
            "model_arguments_sha256": model_arguments_sha256,
            "resolved_arguments_sha256": resolved_arguments_sha256,
        }
    )
    return digest


def _reference_field(value: str) -> tuple[str, str, str] | None:
    """Return ``(kind, normalized stem, public name)`` for an ID field."""

    normalized = value.casefold()
    if normalized.endswith(_INTERNAL_PLURAL_SUFFIX) and len(value) > len(_INTERNAL_PLURAL_SUFFIX):
        stem = value[: -len(_INTERNAL_PLURAL_SUFFIX)]
        return "plural", stem.casefold(), stem + _PUBLIC_PLURAL_SUFFIX
    if normalized.endswith(_INTERNAL_SINGULAR_SUFFIX) and len(value) > len(
        _INTERNAL_SINGULAR_SUFFIX
    ):
        stem = value[: -len(_INTERNAL_SINGULAR_SUFFIX)]
        return "singular", stem.casefold(), stem + _PUBLIC_SINGULAR_SUFFIX
    return None


def _public_alias(internal: str) -> str:
    digest = hashlib.sha256(
        _PUBLIC_ALIAS_DOMAIN.encode("utf-8") + b"\0" + internal.encode("utf-8")
    ).hexdigest()
    return _PUBLIC_ALIAS_PREFIX + digest[:_PUBLIC_ALIAS_HEX_CHARS]


def public_reference_alias_v1(internal: str) -> str:
    """Return the frozen opaque alias for one Agent-private business identity."""

    if not isinstance(internal, str) or not internal.strip():
        raise ValueError("public business reference requires a non-empty identity")
    return _public_alias(internal)


def _reference_text(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrameworkAuthorityError(f"{path} contains a malformed internal business identity")
    return value


def _add_reference_candidates(
    values: Sequence[str],
    *,
    stem: str,
    candidates_by_stem: dict[str, set[str]],
    all_internal: set[str],
) -> None:
    candidates_by_stem.setdefault(stem, set()).update(values)
    all_internal.update(values)


def _candidates_for_stem(
    stem: str,
    candidates_by_stem: Mapping[str, set[str]],
) -> tuple[tuple[str, ...], bool]:
    # Exact observations and semantically qualified task facts are cumulative
    # authority, not competing sources.  For example, after reading the first
    # ``record_id`` in a task whose frozen input exposes several
    # ``evidence_record_ids``, the exact ``record`` stem must not shrink the
    # next read surface to that first row.  Every value below was already
    # collected from an actor-visible structured reference, so unioning the
    # compatible stems preserves finite choice authority without accepting a
    # provider-authored free string.
    compatible: set[str] = set(candidates_by_stem.get(stem, ()))
    authority_seen = stem in candidates_by_stem
    for candidate_stem, values in candidates_by_stem.items():
        if candidate_stem == stem:
            continue
        shorter: str | None = None
        if candidate_stem.endswith("_" + stem):
            shorter = stem
        elif stem.endswith("_" + candidate_stem):
            shorter = candidate_stem
        shared_evidence_family = "evidence" in stem.split(
            "_"
        ) and "evidence" in candidate_stem.split("_")
        if shorter in _SEMANTIC_REFERENCE_ROOTS or shared_evidence_family:
            compatible.update(values)
            authority_seen = True
    for related_stem in _REFERENCE_STEM_RELATIONS.get(stem, frozenset()):
        if related_stem in candidates_by_stem:
            authority_seen = True
            compatible.update(candidates_by_stem[related_stem])
    return tuple(sorted(compatible)), authority_seen


def _collect_observation_references(
    value: Any,
    *,
    candidates_by_stem: dict[str, set[str]],
    all_internal: set[str],
    hidden_observed_by_name: dict[str, list[Any]],
    hidden_goal_values: set[str],
    path: str,
) -> None:
    if isinstance(value, Mapping):
        # Listing-claim snapshots use the canonical protocol field ``subject``
        # (rather than ``subject_id``) for the stable claim subject.  Treat it
        # as a business resource identity only when the surrounding record is
        # unmistakably a claim snapshot.  This keeps ordinary prose fields
        # named ``subject`` untouched while preventing claim-subject ids from
        # bypassing the shared opaque-reference projection.
        claim_subject = value.get("subject")
        if (
            isinstance(claim_subject, str)
            and claim_subject.strip()
            and isinstance(value.get("claim_id"), str)
            and (
                isinstance(value.get("listing_id"), str)
                or isinstance(value.get("versions"), Sequence)
            )
        ):
            _add_reference_candidates(
                (claim_subject,),
                stem="subject",
                candidates_by_stem=candidates_by_stem,
                all_internal=all_internal,
            )
        # Evidence records carry a typed subject.  Only the CommerceWorld
        # product-evidence kind promotes that subject to a product candidate;
        # unrelated order, actor, or shipment subjects cannot cross this
        # resource relation.
        if value.get("kind") == "commerce_product_evidence":
            product = value.get("subject_id")
            if product is not None:
                _add_reference_candidates(
                    (_reference_text(product, path=f"{path}.subject_id"),),
                    stem="product",
                    candidates_by_stem=candidates_by_stem,
                    all_internal=all_internal,
                )
        for raw_key, item in value.items():
            key = str(raw_key)
            if benchmark_internal_namespace_field_v1(key):
                continue
            reference = _reference_field(key)
            if reference is None:
                _collect_observation_references(
                    item,
                    candidates_by_stem=candidates_by_stem,
                    all_internal=all_internal,
                    hidden_observed_by_name=hidden_observed_by_name,
                    hidden_goal_values=hidden_goal_values,
                    path=f"{path}.{key}",
                )
                continue
            kind, stem, _public_name = reference
            if item is None:
                continue
            if kind == "singular":
                values = (_reference_text(item, path=f"{path}.{key}"),)
            else:
                if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)):
                    raise FrameworkAuthorityError(
                        f"{path}.{key} must contain an internal identity sequence"
                    )
                values = tuple(_reference_text(entry, path=f"{path}.{key}[]") for entry in item)
            if framework_owned_reference_field(key):
                hidden_value: Any = values[0] if kind == "singular" else values
                rows = hidden_observed_by_name.setdefault(key.casefold(), [])
                if hidden_value not in rows:
                    rows.append(hidden_value)
                hidden_goal_values.update(values)
                continue
            _add_reference_candidates(
                values,
                stem=stem,
                candidates_by_stem=candidates_by_stem,
                all_internal=all_internal,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _collect_observation_references(
                item,
                candidates_by_stem=candidates_by_stem,
                all_internal=all_internal,
                hidden_observed_by_name=hidden_observed_by_name,
                hidden_goal_values=hidden_goal_values,
                path=f"{path}[{index}]",
            )


def _collect_framework_reference_bindings(
    value: Any,
    *,
    hidden_observed_by_name: dict[str, list[Any]],
    hidden_goal_values: set[str],
    path: str,
) -> None:
    """Collect Agent-owned correlation authority without exposing business data."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            reference = _reference_field(key)
            if reference is None or not framework_owned_reference_field(key):
                _collect_framework_reference_bindings(
                    item,
                    hidden_observed_by_name=hidden_observed_by_name,
                    hidden_goal_values=hidden_goal_values,
                    path=f"{path}.{key}",
                )
                continue
            kind, _stem, _public_name = reference
            if item is None:
                continue
            if kind == "singular":
                values = (_reference_text(item, path=f"{path}.{key}"),)
                hidden_value: Any = values[0]
            else:
                if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)):
                    raise FrameworkAuthorityError(
                        f"{path}.{key} must contain an internal identity sequence"
                    )
                values = tuple(_reference_text(entry, path=f"{path}.{key}[]") for entry in item)
                hidden_value = values
            rows = hidden_observed_by_name.setdefault(key.casefold(), [])
            if hidden_value not in rows:
                rows.append(hidden_value)
            hidden_goal_values.update(values)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _collect_framework_reference_bindings(
                item,
                hidden_observed_by_name=hidden_observed_by_name,
                hidden_goal_values=hidden_goal_values,
                path=f"{path}[{index}]",
            )


def _finite_reference_enum(
    value: Any,
    *,
    path: str,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise FrameworkAuthorityError(f"{path} business reference enum is malformed")
    if not value:
        return ()
    rows = tuple(_reference_text(item, path=path) for item in value)
    if len(rows) != len(set(rows)):
        raise FrameworkAuthorityError(f"{path} business reference enum contains duplicates")
    return rows


def _schema_reference_candidates(
    schema: Mapping[str, Any],
    *,
    kind: str,
    path: str,
) -> tuple[str, ...]:
    if kind == "singular":
        schema_type = schema.get("type")
        if schema_type is not None and schema_type != "string":
            raise FrameworkAuthorityError(f"{path} singular business reference schema is not text")
        return _finite_reference_enum(schema.get("enum"), path=f"{path}.enum")
    schema_type = schema.get("type")
    if schema_type is not None and schema_type != "array":
        raise FrameworkAuthorityError(f"{path} plural business reference schema is not an array")
    items = schema.get("items")
    if not isinstance(items, Mapping):
        raise FrameworkAuthorityError(f"{path} plural business reference schema has no item schema")
    item_type = items.get("type")
    if item_type is not None and item_type != "string":
        raise FrameworkAuthorityError(f"{path} plural business reference item schema is not text")
    return _finite_reference_enum(
        items.get("enum"),
        path=f"{path}.items.enum",
    )


def _collect_schema_enum_references(
    schema: Mapping[str, Any],
    *,
    candidates_by_stem: dict[str, set[str]],
    all_internal: set[str],
    path: str,
) -> None:
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for raw_name, child in properties.items():
            name = str(raw_name)
            if not isinstance(child, Mapping):
                raise FrameworkAuthorityError(f"{path}.properties.{name} is not a schema")
            reference = _reference_field(name)
            if reference is not None and not framework_owned_reference_field(name):
                kind, stem, _public_name = reference
                values = _schema_reference_candidates(
                    child,
                    kind=kind,
                    path=f"{path}.properties.{name}",
                )
                if values:
                    _add_reference_candidates(
                        values,
                        stem=stem,
                        candidates_by_stem=candidates_by_stem,
                        all_internal=all_internal,
                    )
            _collect_schema_enum_references(
                child,
                candidates_by_stem=candidates_by_stem,
                all_internal=all_internal,
                path=f"{path}.properties.{name}",
            )
    items = schema.get("items")
    if isinstance(items, Mapping):
        _collect_schema_enum_references(
            items,
            candidates_by_stem=candidates_by_stem,
            all_internal=all_internal,
            path=f"{path}.items",
        )
    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, Sequence) and not isinstance(branches, (str, bytes, bytearray)):
            for index, branch in enumerate(branches):
                if isinstance(branch, Mapping):
                    _collect_schema_enum_references(
                        branch,
                        candidates_by_stem=candidates_by_stem,
                        all_internal=all_internal,
                        path=f"{path}.{keyword}[{index}]",
                    )
    definitions = schema.get("$defs")
    if isinstance(definitions, Mapping):
        for name, child in definitions.items():
            if isinstance(child, Mapping):
                _collect_schema_enum_references(
                    child,
                    candidates_by_stem=candidates_by_stem,
                    all_internal=all_internal,
                    path=f"{path}.$defs.{name}",
                )


def _project_observation_references(
    value: Any,
    *,
    aliases_by_internal: Mapping[str, str],
    path: str,
    semantic_path: tuple[str, ...],
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
                output[public_key] = _project_observation_references(
                    item,
                    aliases_by_internal=aliases_by_internal,
                    path=f"{path}.{public_key}",
                    semantic_path=(*semantic_path, public_key),
                )
                continue
            if benchmark_internal_namespace_field_v1(normalized):
                continue
            if normalized in _AGENT_PRIVATE_OBSERVATION_FIELDS or (
                provider_boundary_field_is_private_v1(
                    semantic_path,
                    normalized,
                )
            ):
                continue
            reference = _reference_field(key)
            if reference is not None and framework_owned_reference_field(key):
                continue
            public_name = reference[2] if reference is not None else key
            if public_name in output:
                raise FrameworkAuthorityError(
                    f"{path} has colliding internal and public reference fields"
                )
            if reference is None or item is None:
                projected = _project_observation_references(
                    item,
                    aliases_by_internal=aliases_by_internal,
                    path=f"{path}.{key}",
                    semantic_path=(*semantic_path, normalized),
                )
            elif reference[0] == "singular":
                internal = _reference_text(item, path=f"{path}.{key}")
                projected = aliases_by_internal[internal]
            else:
                if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)):
                    raise FrameworkAuthorityError(
                        f"{path}.{key} must contain an internal identity sequence"
                    )
                projected = [
                    aliases_by_internal[_reference_text(entry, path=f"{path}.{key}[]")]
                    for entry in item
                ]
            output[public_name] = projected
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _project_observation_references(
                item,
                aliases_by_internal=aliases_by_internal,
                path=f"{path}[{index}]",
                semantic_path=semantic_path,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, str) and value in aliases_by_internal:
        # The same structured identity can be repeated in a canonical record
        # under a legacy semantic field such as listing-claim ``subject``.
        # Once authority collection has classified that exact value, project
        # every exact occurrence so nested versions cannot leak it.
        return aliases_by_internal[value]
    return copy.deepcopy(value)


def _project_goal_references(
    goal: str,
    *,
    aliases_by_internal: Mapping[str, str],
    hidden_internal: Sequence[str] | frozenset[str] | set[str],
) -> str:
    """Replace exact identity tokens in otherwise free-form business text."""

    projected = goal
    for internal in sorted(aliases_by_internal, key=lambda row: (-len(row), row)):
        token = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(internal)}(?![A-Za-z0-9_])")
        projected = token.sub(aliases_by_internal[internal], projected)
    for internal in sorted(hidden_internal, key=lambda row: (-len(row), row)):
        token = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(internal)}(?![A-Za-z0-9_])")
        projected = token.sub("current-business-context", projected)
    # Free-form goals are not structured authority sources.  An actor/service
    # address may appear only here and therefore never enter the finite alias
    # map above; remove those addresses at the shared bridge boundary too.
    projected = _PRIVATE_ADDRESS_TOKEN_RE.sub(
        "current-business-context",
        projected,
    )
    return _PRIVATE_ACTION_TOKEN_RE.sub("marketplace-action", projected)


def _resolve_hidden_reference(
    schema: Mapping[str, Any],
    *,
    name: str,
    kind: str,
    hidden_observed_by_name: Mapping[str, Sequence[Any]],
    path: str,
) -> Any:
    observed = tuple(hidden_observed_by_name.get(name.casefold(), ()))
    if kind == "plural":
        explicit_const = schema.get("const")
        candidates = list(observed)
        if explicit_const is not None:
            if not isinstance(explicit_const, Sequence) or isinstance(
                explicit_const, (str, bytes, bytearray)
            ):
                raise FrameworkAuthorityError(
                    f"{path} hidden framework reference list is malformed"
                )
            constant = tuple(
                _reference_text(item, path=f"{path}.const[]") for item in explicit_const
            )
            candidates = [row for row in candidates if row == constant] or [constant]
        unique = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        if not unique:
            raise _UnavailableBusinessReferenceSurface(path)
        if len(unique) != 1:
            raise FrameworkAuthorityError(
                f"{path} hidden framework reference list is not uniquely bound"
            )
        return list(unique[0])

    explicit = _schema_reference_candidates(schema, kind=kind, path=path)
    observed_text = tuple(row for row in observed if isinstance(row, str))
    if explicit and observed_text:
        candidates = tuple(sorted(set(explicit).intersection(observed_text)))
        if not candidates:
            raise FrameworkAuthorityError(
                f"{path} hidden framework reference disagrees with observation authority"
            )
    else:
        candidates = explicit or tuple(sorted(set(observed_text)))
    if not candidates:
        raise _UnavailableBusinessReferenceSurface(path)
    if len(candidates) != 1:
        raise FrameworkAuthorityError(f"{path} hidden framework reference is not uniquely bound")
    return candidates[0]


def _project_reference_schema(
    schema: Mapping[str, Any],
    *,
    candidates_by_stem: Mapping[str, set[str]],
    aliases_by_internal: Mapping[str, str],
    hidden_observed_by_name: Mapping[str, Sequence[Any]],
    hidden_bindings: dict[tuple[str, ...], Any],
    property_path: tuple[str, ...],
    path: str,
    reference: tuple[str, str, str] | None = None,
) -> dict[str, Any]:
    chosen: tuple[str, ...] = ()
    empty_plural_authority = False
    if reference is not None:
        kind, stem, _public_name = reference
        explicit = _schema_reference_candidates(schema, kind=kind, path=path)
        observed, authority_seen = _candidates_for_stem(
            stem,
            candidates_by_stem,
        )
        chosen = explicit or observed
        if not chosen:
            if kind == "plural" and authority_seen:
                empty_plural_authority = True
            else:
                raise _UnavailableBusinessReferenceSurface(path)

    output: dict[str, Any] = {}
    for raw_key, item in schema.items():
        key = str(raw_key)
        if key == "properties" and isinstance(item, Mapping):
            public_properties: dict[str, Any] = {}
            for raw_name, child in item.items():
                name = str(raw_name)
                if not isinstance(child, Mapping):
                    raise FrameworkAuthorityError(f"{path}.properties.{name} is not a schema")
                child_reference = _reference_field(name)
                if child_reference is not None and framework_owned_reference_field(name):
                    hidden_bindings[property_path + (name,)] = _resolve_hidden_reference(
                        child,
                        name=name,
                        kind=child_reference[0],
                        hidden_observed_by_name=hidden_observed_by_name,
                        path=f"{path}.properties.{name}",
                    )
                    continue
                public_name = child_reference[2] if child_reference is not None else name
                if public_name in public_properties:
                    raise FrameworkAuthorityError(
                        f"{path} has colliding internal and public reference fields"
                    )
                public_properties[public_name] = _project_reference_schema(
                    child,
                    candidates_by_stem=candidates_by_stem,
                    aliases_by_internal=aliases_by_internal,
                    hidden_observed_by_name=hidden_observed_by_name,
                    hidden_bindings=hidden_bindings,
                    property_path=property_path + (name,),
                    path=f"{path}.properties.{name}",
                    reference=child_reference,
                )
            output[key] = public_properties
        elif (
            key == "required"
            and isinstance(item, Sequence)
            and not isinstance(item, (str, bytes, bytearray))
        ):
            output[key] = [
                (_reference_field(str(name)) or (None, None, str(name)))[2]
                for name in item
                if not framework_owned_reference_field(str(name))
            ]
        elif key == "dependentRequired" and isinstance(item, Mapping):
            projected_dependencies: dict[str, Any] = {}
            for raw_name, dependencies in item.items():
                name = str(raw_name)
                if framework_owned_reference_field(name):
                    continue
                public_name = (_reference_field(name) or (None, None, name))[2]
                if not isinstance(dependencies, Sequence) or isinstance(
                    dependencies, (str, bytes, bytearray)
                ):
                    raise FrameworkAuthorityError(f"{path}.dependentRequired is malformed")
                projected_dependencies[public_name] = [
                    (_reference_field(str(dependency)) or (None, None, str(dependency)))[2]
                    for dependency in dependencies
                    if not framework_owned_reference_field(str(dependency))
                ]
            output[key] = projected_dependencies
        elif key == "items" and isinstance(item, Mapping):
            projected_items = _project_reference_schema(
                item,
                candidates_by_stem=candidates_by_stem,
                aliases_by_internal=aliases_by_internal,
                hidden_observed_by_name=hidden_observed_by_name,
                hidden_bindings=hidden_bindings,
                property_path=property_path,
                path=f"{path}.items",
            )
            if reference is not None and reference[0] == "plural" and not empty_plural_authority:
                projected_items["enum"] = [aliases_by_internal[value] for value in chosen]
            output[key] = projected_items
        elif key == "enum" and reference is not None and reference[0] == "singular":
            output[key] = [aliases_by_internal[value] for value in chosen]
        elif (
            key in {"allOf", "anyOf", "oneOf"}
            and isinstance(item, Sequence)
            and not isinstance(item, (str, bytes, bytearray))
        ):
            output[key] = [
                _project_reference_schema(
                    branch,
                    candidates_by_stem=candidates_by_stem,
                    aliases_by_internal=aliases_by_internal,
                    hidden_observed_by_name=hidden_observed_by_name,
                    hidden_bindings=hidden_bindings,
                    property_path=property_path,
                    path=f"{path}.{key}[{index}]",
                    reference=reference,
                )
                if isinstance(branch, Mapping)
                else copy.deepcopy(branch)
                for index, branch in enumerate(item)
            ]
        elif key in {"$defs", "dependentSchemas", "patternProperties"} and isinstance(
            item, Mapping
        ):
            output[key] = {
                str(name): _project_reference_schema(
                    child,
                    candidates_by_stem=candidates_by_stem,
                    aliases_by_internal=aliases_by_internal,
                    hidden_observed_by_name=hidden_observed_by_name,
                    hidden_bindings=hidden_bindings,
                    property_path=property_path,
                    path=f"{path}.{key}.{name}",
                )
                if isinstance(child, Mapping)
                else copy.deepcopy(child)
                for name, child in item.items()
            }
        elif empty_plural_authority and key in {"minItems", "maxItems", "const"}:
            # Replaced below by the one exact legal empty choice.
            continue
        else:
            output[key] = copy.deepcopy(item)

    if reference is not None and reference[0] == "singular" and "enum" not in output:
        output["enum"] = [aliases_by_internal[value] for value in chosen]
    if reference is not None and reference[0] == "plural":
        items = output.get("items")
        if not isinstance(items, dict):
            raise FrameworkAuthorityError(
                f"{path} plural business reference schema has no item schema"
            )
        if empty_plural_authority:
            items.pop("enum", None)
            output["const"] = []
            output["maxItems"] = 0
        else:
            items["enum"] = [aliases_by_internal[value] for value in chosen]
    return output


def _restore_reference_value(
    schema: Mapping[str, Any],
    value: Any,
    *,
    internal_by_alias: Mapping[str, str],
    hidden_bindings: Mapping[tuple[str, ...], Any],
    property_path: tuple[str, ...],
    path: str,
    reference: tuple[str, str, str] | None = None,
) -> Any:
    if reference is not None:
        if reference[0] == "singular":
            if not isinstance(value, str) or value not in internal_by_alias:
                raise BusinessDecisionContractError(
                    "business decision used an unknown public business reference"
                )
            return internal_by_alias[value]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise BusinessDecisionContractError(
                "business decision public reference list is invalid"
            )
        restored_rows: list[str] = []
        for item in value:
            if not isinstance(item, str) or item not in internal_by_alias:
                raise BusinessDecisionContractError(
                    "business decision used an unknown public business reference"
                )
            restored_rows.append(internal_by_alias[item])
        return restored_rows

    if isinstance(value, Mapping):
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            return _restore_dynamic_reference_aliases(
                value,
                internal_by_alias=internal_by_alias,
                path=path,
            )
        by_public_name: dict[str, tuple[str, Mapping[str, Any], tuple[str, str, str] | None]] = {}
        for raw_name, child in properties.items():
            name = str(raw_name)
            if not isinstance(child, Mapping):
                continue
            child_reference = _reference_field(name)
            if child_reference is not None and framework_owned_reference_field(name):
                continue
            public_name = child_reference[2] if child_reference is not None else name
            if public_name in by_public_name:
                raise FrameworkAuthorityError(
                    f"{path} has colliding internal and public reference fields"
                )
            by_public_name[public_name] = (name, child, child_reference)
        output: dict[str, Any] = {}
        for public_name, item in value.items():
            binding = by_public_name.get(str(public_name))
            if binding is None:
                additional = schema.get("additionalProperties", True)
                restored_name = internal_by_alias.get(
                    str(public_name),
                    str(public_name),
                )
                if restored_name in output:
                    raise BusinessDecisionContractError(
                        "business decision dynamic reference keys collide"
                    )
                if isinstance(additional, Mapping):
                    output[restored_name] = _restore_reference_value(
                        additional,
                        item,
                        internal_by_alias=internal_by_alias,
                        hidden_bindings=hidden_bindings,
                        property_path=property_path + (restored_name,),
                        path=f"{path}.{public_name}",
                    )
                else:
                    output[restored_name] = _restore_dynamic_reference_aliases(
                        item,
                        internal_by_alias=internal_by_alias,
                        path=f"{path}.{public_name}",
                    )
                continue
            internal_name, child, child_reference = binding
            output[internal_name] = _restore_reference_value(
                child,
                item,
                internal_by_alias=internal_by_alias,
                hidden_bindings=hidden_bindings,
                property_path=property_path + (internal_name,),
                path=f"{path}.{public_name}",
                reference=child_reference,
            )
        for raw_name, child in properties.items():
            name = str(raw_name)
            if not isinstance(child, Mapping):
                continue
            child_reference = _reference_field(name)
            if child_reference is None or not framework_owned_reference_field(name):
                continue
            binding_path = property_path + (name,)
            if binding_path not in hidden_bindings:
                raise FrameworkAuthorityError(f"{path} lost a hidden framework reference binding")
            output[name] = copy.deepcopy(hidden_bindings[binding_path])
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = schema.get("items")
        if not isinstance(items, Mapping):
            return copy.deepcopy(list(value))
        return [
            _restore_reference_value(
                items,
                item,
                internal_by_alias=internal_by_alias,
                hidden_bindings=hidden_bindings,
                property_path=property_path,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    return copy.deepcopy(value)


def _restore_dynamic_reference_aliases(
    value: Any,
    *,
    internal_by_alias: Mapping[str, str],
    path: str,
) -> Any:
    """Restore only exact, already-issued aliases inside an open JSON value.

    Some business evidence is deliberately an open object whose map keys and
    leaves are business resource references.  The schema cannot name those
    dynamic keys ahead of time.  Exact alias membership is nevertheless a
    finite Agent authority, so restoration is safe without guessing suffixes
    or accepting provider-authored internal identifiers.
    """

    if isinstance(value, str):
        return internal_by_alias.get(value, value)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_name, item in value.items():
            public_name = str(raw_name)
            internal_name = internal_by_alias.get(public_name, public_name)
            if internal_name in output:
                raise BusinessDecisionContractError(f"{path} dynamic reference keys collide")
            output[internal_name] = _restore_dynamic_reference_aliases(
                item,
                internal_by_alias=internal_by_alias,
                path=f"{path}.{public_name}",
            )
        return output
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _restore_dynamic_reference_aliases(
                item,
                internal_by_alias=internal_by_alias,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    return copy.deepcopy(value)


def _reject_internal_reference_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if _reference_field(key) is not None or framework_owned_reference_field(key):
                raise BusinessDecisionContractError(
                    "model business decision used an Agent-private reference field"
                )
            _reject_internal_reference_keys(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _reject_internal_reference_keys(item)


def _destination_matches(registered: str, compiled: str) -> bool:
    if registered == compiled:
        return True
    if registered in {"@inbound_sender", "@argument:recipient_id"}:
        role = compiled.split(":", 1)[0]
        return role in {"buyer", "merchant", "consumer"}
    return False


__all__ = [
    "AgentDecisionBridge",
    "AgentPhaseContract",
    "BridgedBusinessDecisionV1",
    "CompiledBusinessAction",
    "CompiledPhaseDecision",
    "CompiledWorldRead",
    "LocalControlDecision",
    "PhaseAdapter",
    "RouteBinding",
    "RouteRegistry",
    "public_reference_alias_v1",
]
