"""Domain authorities behind the unified Agent phase/decision contract.

Each adapter keeps its domain's authority construction and business validation,
while route identity is resolved only through :class:`RouteRegistry`.  Model
requests therefore contain business choices, never destination addresses,
action namespaces, authority digests, or framework-owned identifiers.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agents.after_sales_turn_authority import (
    AfterSalesTurnAuthority,
    build_projected_after_sales_turn_authority,
    build_after_sales_turn_authority,
)
from agents.agent_phase import (
    CompiledBusinessAction,
    CompiledPhaseDecision,
    LocalControlDecision,
    PhaseAdapter,
    RouteBinding,
    RouteRegistry,
)
from agents.business_decision import BusinessIntentSpec, LLMBusinessDecisionV1
from agents.governance_turn_authority import (
    GovernanceProjectionCache,
    GovernanceTurnAuthority,
)
from agents.negotiation_turn_authority import NegotiationTurnAuthority
from agents.protocol_event_turn_authority import ProtocolEventTurnAuthority
from agents.ranked_offer_turn_authority import RankedOfferTurnAuthority
from agents.decision_errors import FrameworkAuthorityError, ModelBusinessDecisionError


@dataclass(frozen=True, slots=True)
class DomainPhaseAuthority:
    """Role-bound wrapper preventing cross-actor authority reuse."""

    role: str
    value: object

    def __post_init__(self) -> None:
        if self.role not in {"buyer", "merchant"}:
            raise FrameworkAuthorityError("domain phase role is invalid")


@dataclass(frozen=True, slots=True)
class BoundDomainPhase:
    """One already-authenticated domain adapter/authority pair.

    ``base.Agent`` may bind several candidate domains for an inbound (for
    example a ranked response that also permits negotiation).  Semantic and
    business-decision paths consume this uniform collection; they never infer
    a domain from a destination or reconstruct an adapter from a tool name.
    """

    adapter: PhaseAdapter
    authority: DomainPhaseAuthority

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, PhaseAdapter):
            raise FrameworkAuthorityError("bound domain phase adapter is invalid")
        # Fail closed before provider access if authority, decision surface,
        # or route closure are inconsistent.
        self.adapter.decision_schema(self.authority)
        self.adapter.route_bindings(self.authority)


class _AuthorityAdapter:
    """Shared PhaseAdapter mechanics; subclasses retain domain rules only."""

    def __init__(self, routes: RouteRegistry) -> None:
        self._routes = routes

    def _resolved(self, authority: object, expected: type[Any]) -> DomainPhaseAuthority:
        if not isinstance(authority, DomainPhaseAuthority) or not isinstance(
            authority.value, expected
        ):
            raise FrameworkAuthorityError("phase adapter received the wrong authority")
        return authority

    def _decision_schema(
        self,
        *,
        role: str,
        specs: tuple[BusinessIntentSpec, ...],
    ) -> tuple[BusinessIntentSpec, ...]:
        """Bind routed choices to the registry's sole private source name.

        Some domain choices are deliberately local (for example, continuing
        ranked-offer reasoning), and some compile to one of several routed
        operations (for example, approving or denying an after-sales case).
        Those choices have no one-to-one route until compilation and retain
        their authority-local source identity.
        """

        bound: list[BusinessIntentSpec] = []
        for spec in specs:
            source_name = spec.source_name
            try:
                route = self._routes.by_operation(spec.intent)
            except FrameworkAuthorityError:
                pass
            else:
                if role not in route.roles:
                    raise FrameworkAuthorityError(
                        "domain business choice is outside the actor role"
                    )
                source_name = route.source_name
            bound.append(
                BusinessIntentSpec(
                    intent=spec.intent,
                    description=_public_description(spec.description),
                    parameters=copy.deepcopy(dict(spec.parameters)),
                    category=spec.category,
                    source_name=source_name,
                )
            )
        return tuple(bound)

    def _compiled(
        self,
        *,
        role: str,
        operation: str,
        payload: Mapping[str, Any],
        route_target: str | None = None,
    ) -> CompiledBusinessAction:
        route = self._routes.by_operation(operation)
        if role not in route.roles:
            raise FrameworkAuthorityError("compiled domain choice is outside the actor role")
        if route.destination in {"@inbound_sender", "@argument:recipient_id"}:
            if not isinstance(route_target, str) or _role(route_target) not in {
                "buyer",
                "merchant",
            }:
                raise FrameworkAuthorityError(
                    "compiled domain choice has no valid dynamic route target"
                )
            destination = route_target
        else:
            if route_target is not None:
                raise FrameworkAuthorityError(
                    "compiled domain choice supplied an unexpected route target"
                )
            destination = route.destination
        return CompiledBusinessAction(
            operation=route.operation,
            action_kind=route.action_kind,
            destination=destination,
            payload=copy.deepcopy(dict(payload)),
            source_name=route.source_name,
        )

    def _route_bindings(
        self,
        *,
        role: str,
        operations: frozenset[str],
    ) -> tuple[RouteBinding, ...]:
        bindings: list[RouteBinding] = []
        for operation in sorted(operations):
            route = self._routes.by_operation(operation)
            if role not in route.roles:
                raise FrameworkAuthorityError("domain route binding is outside the actor role")
            bindings.append(route)
        return tuple(bindings)


class NegotiationPhaseAdapter(_AuthorityAdapter):
    def build_authority(self, **inputs: Any) -> DomainPhaseAuthority:
        actor_id = str(inputs.pop("actor_id"))
        inbound = inputs.pop("inbound")
        value = NegotiationTurnAuthority.from_inbound(inbound, actor_id=actor_id, **inputs)
        return DomainPhaseAuthority(_role(actor_id), value)

    def decision_schema(self, authority: object) -> tuple[BusinessIntentSpec, ...]:
        resolved = self._resolved(authority, NegotiationTurnAuthority)
        return self._decision_schema(
            role=resolved.role,
            specs=resolved.value.decision_schema(),
        )

    def route_bindings(self, authority: object) -> tuple[RouteBinding, ...]:
        resolved = self._resolved(authority, NegotiationTurnAuthority)
        # Route ownership is broader than the current provider surface. At a
        # terminal round the domain deliberately removes ``counter_offer``
        # from decision_schema, but it must still own that registered route so
        # CompositePhaseAdapter cannot reintroduce the generic counter schema.
        operations = (
            frozenset({"propose_offer"})
            if resolved.value.source == "rank_offers"
            else frozenset(
                {
                    "counter_offer",
                    "accept_negotiated_offer",
                    "reject_offer",
                }
            )
        )
        return self._route_bindings(
            role=resolved.role,
            operations=operations,
        )

    def compile(
        self, authority: object, decision: LLMBusinessDecisionV1
    ) -> CompiledBusinessAction | None:
        resolved = self._resolved(authority, NegotiationTurnAuthority)
        compiled = resolved.value.compile(decision.intent, decision.arguments)
        return self._compiled(
            role=resolved.role,
            operation=compiled.operation,
            payload=compiled.payload,
        )


class RankedOfferPhaseAdapter(_AuthorityAdapter):
    def __init__(
        self,
        routes: RouteRegistry,
        *,
        allow_accept: bool = True,
    ) -> None:
        super().__init__(routes)
        self._allow_accept = bool(allow_accept)

    def build_authority(self, **inputs: Any) -> DomainPhaseAuthority:
        actor_id = str(inputs.pop("actor_id"))
        inbound = inputs.pop("inbound")
        value = RankedOfferTurnAuthority.from_inbound(inbound, actor_id=actor_id, **inputs)
        return DomainPhaseAuthority(_role(actor_id), value)

    def decision_schema(self, authority: object) -> tuple[BusinessIntentSpec, ...]:
        resolved = self._resolved(authority, RankedOfferTurnAuthority)
        specs = resolved.value.decision_schema()
        if not self._allow_accept:
            specs = tuple(
                spec for spec in specs if spec.intent != "accept_ranked_offer"
            )
        return self._decision_schema(
            role=resolved.role,
            specs=specs,
        )

    def route_bindings(self, authority: object) -> tuple[RouteBinding, ...]:
        resolved = self._resolved(authority, RankedOfferTurnAuthority)
        operations = (
            frozenset({"accept_ranked_offer"})
            if self._allow_accept
            else frozenset()
        )
        return self._route_bindings(role=resolved.role, operations=operations)

    def compile(self, authority: object, decision: LLMBusinessDecisionV1) -> CompiledPhaseDecision:
        resolved = self._resolved(authority, RankedOfferTurnAuthority)
        compiled = resolved.value.compile(decision.intent, decision.arguments)
        if compiled.operation is None or compiled.payload is None:
            reason = decision.arguments.get("reason")
            return LocalControlDecision(
                intent=decision.intent,
                reason=(
                    str(reason).strip()
                    if isinstance(reason, str) and reason.strip()
                    else decision.message.strip()
                    if isinstance(decision.message, str) and decision.message.strip()
                    else None
                ),
                source_name=decision.intent,
            )
        return self._compiled(
            role=resolved.role,
            operation=compiled.operation,
            payload=compiled.payload,
        )


class ProtocolEventPhaseAdapter(_AuthorityAdapter):
    def build_authority(self, **inputs: Any) -> DomainPhaseAuthority:
        actor_id = str(inputs.pop("actor_id"))
        inbound = inputs.pop("inbound")
        value = ProtocolEventTurnAuthority.from_inbound(inbound, actor_id=actor_id, **inputs)
        return DomainPhaseAuthority(_role(actor_id), value)

    def decision_schema(self, authority: object) -> tuple[BusinessIntentSpec, ...]:
        resolved = self._resolved(authority, ProtocolEventTurnAuthority)
        return self._decision_schema(
            role=resolved.role,
            specs=resolved.value.decision_schema(),
        )

    def route_bindings(self, authority: object) -> tuple[RouteBinding, ...]:
        resolved = self._resolved(authority, ProtocolEventTurnAuthority)
        return self._route_bindings(
            role=resolved.role,
            operations=frozenset(spec.intent for spec in resolved.value.decision_schema()),
        )

    def compile(
        self, authority: object, decision: LLMBusinessDecisionV1
    ) -> CompiledBusinessAction | None:
        resolved = self._resolved(authority, ProtocolEventTurnAuthority)
        compiled = resolved.value.compile(decision.intent, decision.arguments)
        return self._compiled(
            role=resolved.role,
            operation=compiled.operation,
            payload=compiled.payload,
        )


class GovernancePhaseAdapter(_AuthorityAdapter):
    def build_authority(self, **inputs: Any) -> DomainPhaseAuthority:
        actor_id = str(inputs.pop("actor_id"))
        inbound = inputs.pop("inbound")
        projection_cache = inputs.pop("projection_cache", None)
        if projection_cache is not None:
            if inputs:
                raise FrameworkAuthorityError(
                    "governance projection authority cannot mix World-table inputs"
                )
            if not isinstance(projection_cache, GovernanceProjectionCache):
                raise FrameworkAuthorityError("governance projection cache has the wrong type")
            value = GovernanceTurnAuthority.from_platform_projection(
                inbound,
                actor_id=actor_id,
                cache=projection_cache,
            )
        else:
            value = GovernanceTurnAuthority.from_inbound(
                inbound,
                actor_id=actor_id,
                **inputs,
            )
        return DomainPhaseAuthority(_role(actor_id), value)

    def decision_schema(self, authority: object) -> tuple[BusinessIntentSpec, ...]:
        resolved = self._resolved(authority, GovernanceTurnAuthority)
        return self._decision_schema(
            role=resolved.role,
            specs=resolved.value.decision_schema(),
        )

    def route_bindings(self, authority: object) -> tuple[RouteBinding, ...]:
        resolved = self._resolved(authority, GovernanceTurnAuthority)
        operations = {spec.intent for spec in resolved.value.decision_schema()}
        if resolved.value.inbound_kind == "platform.governance_updated" and operations.intersection(
            {"disclose_placement", "activate_campaign"}
        ):
            # A campaign update exposes exactly one legal frontier, while a
            # public phase may declare both disclosure and activation.  Close
            # both registry routes so the inactive sibling cannot fall back to
            # its retired protocol-shaped generic schema.
            operations.update({"disclose_placement", "activate_campaign"})
        return self._route_bindings(
            role=resolved.role,
            operations=frozenset(operations),
        )

    def compile(
        self, authority: object, decision: LLMBusinessDecisionV1
    ) -> CompiledBusinessAction | None:
        resolved = self._resolved(authority, GovernanceTurnAuthority)
        compiled = resolved.value.compile(decision.intent, decision.arguments)
        return self._compiled(
            role=resolved.role,
            operation=compiled.operation,
            payload=compiled.payload,
        )


class AfterSalesPhaseAdapter(_AuthorityAdapter):
    def build_authority(self, **inputs: Any) -> DomainPhaseAuthority:
        actor_id = str(inputs.get("actor_id", ""))
        if "projections" in inputs:
            value = build_projected_after_sales_turn_authority(**inputs)
        else:
            value = build_after_sales_turn_authority(**inputs)
        return DomainPhaseAuthority(_role(actor_id), value)

    def decision_schema(self, authority: object) -> tuple[BusinessIntentSpec, ...]:
        resolved = self._resolved(authority, AfterSalesTurnAuthority)
        return self._decision_schema(
            role=resolved.role,
            specs=resolved.value.decision_schema(),
        )

    def route_bindings(self, authority: object) -> tuple[RouteBinding, ...]:
        resolved = self._resolved(authority, AfterSalesTurnAuthority)
        return self._route_bindings(
            role=resolved.role,
            operations=resolved.value.routed_operations(),
        )

    def compile(
        self, authority: object, decision: LLMBusinessDecisionV1
    ) -> CompiledBusinessAction | None:
        resolved = self._resolved(authority, AfterSalesTurnAuthority)
        compiled = resolved.value.compile(decision.intent, decision.arguments)
        return self._compiled(
            role=resolved.role,
            operation=compiled.operation,
            payload=compiled.payload,
        )


@dataclass(frozen=True, slots=True)
class SupplyPhaseAuthority:
    """Exact Agent-owned purchase authority for one supply observation."""

    buyer_id: str
    options: tuple[Mapping[str, Any], ...]

    def decision_schema(self) -> tuple[BusinessIntentSpec, ...]:
        skus = [str(row.get("sku_id")) for row in self.options]
        return (
            BusinessIntentSpec(
                intent="settle_payment",
                description="Purchase an available supply option.",
                parameters={
                    "type": "object",
                    "properties": {
                        "sku_id": {"type": "string", "enum": skus},
                        "qty": {"type": "integer", "minimum": 1},
                        "allow_partial": {"type": "boolean"},
                    },
                    "required": ["sku_id", "qty", "allow_partial"],
                    "additionalProperties": False,
                },
                category="act",
                source_name="settle_payment",
            ),
        )

    def compile(self, decision: LLMBusinessDecisionV1) -> tuple[str, Mapping[str, Any]]:
        if decision.intent != "settle_payment":
            raise FrameworkAuthorityError(
                "selected supply decision is outside the current authority"
            )
        sku_id = decision.arguments.get("sku_id")
        matches = [row for row in self.options if row.get("sku_id") == sku_id]
        if len(matches) != 1:
            raise FrameworkAuthorityError("selected supply option is not authoritative")
        option = matches[0]
        return "settle_payment", {
            "supply_authority_id": option.get("authority_id"),
            "supply_authority_digest": option.get("authority_digest"),
            "sku_id": sku_id,
            "qty": decision.arguments.get("qty"),
            "allow_partial": decision.arguments.get("allow_partial"),
        }


class SupplyPhaseAdapter(_AuthorityAdapter):
    def build_authority(self, **inputs: Any) -> DomainPhaseAuthority:
        buyer_id = str(inputs.get("buyer_id", ""))
        options = inputs.get("options")
        if _role(buyer_id) != "buyer" or not isinstance(options, (list, tuple)):
            raise FrameworkAuthorityError("supply authority inputs are invalid")
        normalized = tuple(copy.deepcopy(dict(row)) for row in options if isinstance(row, Mapping))
        if len(normalized) != len(options):
            raise FrameworkAuthorityError("supply authority options are malformed")
        return DomainPhaseAuthority("buyer", SupplyPhaseAuthority(buyer_id, normalized))

    def decision_schema(self, authority: object) -> tuple[BusinessIntentSpec, ...]:
        resolved = self._resolved(authority, SupplyPhaseAuthority)
        return self._decision_schema(
            role=resolved.role,
            specs=resolved.value.decision_schema(),
        )

    def route_bindings(self, authority: object) -> tuple[RouteBinding, ...]:
        resolved = self._resolved(authority, SupplyPhaseAuthority)
        return self._route_bindings(
            role=resolved.role,
            operations=frozenset(spec.intent for spec in resolved.value.decision_schema()),
        )

    def compile(
        self, authority: object, decision: LLMBusinessDecisionV1
    ) -> CompiledBusinessAction | None:
        resolved = self._resolved(authority, SupplyPhaseAuthority)
        operation, payload = resolved.value.compile(decision)
        return self._compiled(
            role=resolved.role,
            operation=operation,
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class SupplyReportPhaseAuthority:
    """Merchant-owned observations and route binding for one supply report."""

    merchant_id: str
    recipient_id: str
    states: tuple[Mapping[str, Any], ...]

    def decision_schema(self) -> tuple[BusinessIntentSpec, ...]:
        sku_ids = [str(row["sku_id"]) for row in self.states]
        state_schema = {
            "type": "object",
            "properties": {
                "sku_id": {"type": "string", "enum": sku_ids},
                "available_qty": {"type": "integer", "minimum": 0},
                "reserved_qty": {"type": "integer", "minimum": 0},
                "eta_day": {"type": "integer", "minimum": 0},
                "unit_price_cents": {"type": "integer", "minimum": 0},
            },
            "required": [
                "sku_id",
                "available_qty",
                "reserved_qty",
                "eta_day",
                "unit_price_cents",
            ],
            "additionalProperties": False,
        }
        return (
            BusinessIntentSpec(
                intent="send_message",
                description=(
                    "Report the observed inventory, reservation, ETA, and price "
                    "facts to the current business counterparty."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "states": {
                            "type": "array",
                            "items": state_schema,
                            "minItems": len(self.states),
                            "maxItems": len(self.states),
                        }
                    },
                    "required": ["states"],
                    "additionalProperties": False,
                },
                category="act",
                source_name="send_message",
            ),
        )

    def compile(
        self,
        decision: LLMBusinessDecisionV1,
    ) -> tuple[str, Mapping[str, Any], str]:
        if decision.intent != "send_message":
            raise ModelBusinessDecisionError(
                "selected supply report decision is outside the current authority"
            )
        raw_states = decision.arguments.get("states")
        if not isinstance(raw_states, list) or len(raw_states) != len(self.states):
            raise ModelBusinessDecisionError(
                "supply report must cover every observed SKU exactly once"
            )
        required = {
            "sku_id",
            "available_qty",
            "reserved_qty",
            "eta_day",
            "unit_price_cents",
        }
        by_sku: dict[str, dict[str, Any]] = {}
        for row in raw_states:
            if not isinstance(row, Mapping) or set(row) != required:
                raise ModelBusinessDecisionError("supply report state has invalid fields")
            sku_id = row.get("sku_id")
            if not isinstance(sku_id, str) or sku_id in by_sku:
                raise ModelBusinessDecisionError(
                    "supply report contains a duplicate or invalid SKU"
                )
            for name in required - {"sku_id"}:
                value = row.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ModelBusinessDecisionError(
                        "supply report contains an invalid numeric business fact"
                    )
            by_sku[sku_id] = copy.deepcopy(dict(row))
        authoritative_skus = [str(row["sku_id"]) for row in self.states]
        if set(by_sku) != set(authoritative_skus):
            raise ModelBusinessDecisionError(
                "supply report selected a SKU outside current observations"
            )
        authority_by_sku = {str(row["sku_id"]): row for row in self.states}
        compiled_states = [
            {
                **by_sku[sku_id],
                "merchant_id": self.merchant_id,
                "version": authority_by_sku[sku_id]["version"],
            }
            for sku_id in authoritative_skus
        ]
        return (
            "send_message",
            {
                "category": "inventory_eta_report",
                "states": compiled_states,
            },
            self.recipient_id,
        )


class SupplyReportPhaseAdapter(_AuthorityAdapter):
    def build_authority(self, **inputs: Any) -> DomainPhaseAuthority:
        merchant_id = str(inputs.get("merchant_id", ""))
        recipient_id = str(inputs.get("recipient_id", ""))
        states = inputs.get("states")
        if (
            _role(merchant_id) != "merchant"
            or _role(recipient_id) != "buyer"
            or not isinstance(states, (list, tuple))
            or not states
            or any(not isinstance(row, Mapping) for row in states)
        ):
            raise FrameworkAuthorityError("supply report authority inputs are invalid")
        normalized = tuple(copy.deepcopy(dict(row)) for row in states)
        required = {
            "sku_id",
            "merchant_id",
            "available_qty",
            "reserved_qty",
            "eta_day",
            "unit_price_cents",
            "version",
        }
        sku_ids: set[str] = set()
        for row in normalized:
            sku_id = row.get("sku_id")
            if (
                set(row) != required
                or not isinstance(sku_id, str)
                or not sku_id.strip()
                or sku_id in sku_ids
                or row.get("merchant_id") != merchant_id
            ):
                raise FrameworkAuthorityError("supply report authority state identity is malformed")
            for name in required - {"sku_id", "merchant_id"}:
                value = row.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise FrameworkAuthorityError(
                        "supply report authority state value is malformed"
                    )
            sku_ids.add(sku_id)
        return DomainPhaseAuthority(
            "merchant",
            SupplyReportPhaseAuthority(
                merchant_id=merchant_id,
                recipient_id=recipient_id,
                states=normalized,
            ),
        )

    def decision_schema(self, authority: object) -> tuple[BusinessIntentSpec, ...]:
        resolved = self._resolved(authority, SupplyReportPhaseAuthority)
        return self._decision_schema(
            role=resolved.role,
            specs=resolved.value.decision_schema(),
        )

    def route_bindings(self, authority: object) -> tuple[RouteBinding, ...]:
        resolved = self._resolved(authority, SupplyReportPhaseAuthority)
        return self._route_bindings(
            role=resolved.role,
            operations=frozenset({"send_message"}),
        )

    def compile(
        self, authority: object, decision: LLMBusinessDecisionV1
    ) -> CompiledBusinessAction | None:
        resolved = self._resolved(authority, SupplyReportPhaseAuthority)
        operation, payload, recipient_id = resolved.value.compile(decision)
        return self._compiled(
            role=resolved.role,
            operation=operation,
            payload=payload,
            route_target=recipient_id,
        )


def _role(actor_id: object) -> str:
    if not isinstance(actor_id, str):
        raise FrameworkAuthorityError("domain authority has no actor identity")
    role = actor_id.split(":", 1)[0]
    if role not in {"buyer", "merchant"}:
        raise FrameworkAuthorityError("domain authority actor role is invalid")
    return role


def _public_description(value: str) -> str:
    """Remove implementation nouns from the model-visible business surface."""

    return re.sub(
        r"\b(?:CommerceWorld|Platform|World|Runtime|VCP)\b",
        "marketplace",
        value,
        flags=re.IGNORECASE,
    )


__all__ = [
    "AfterSalesPhaseAdapter",
    "BoundDomainPhase",
    "DomainPhaseAuthority",
    "GovernancePhaseAdapter",
    "NegotiationPhaseAdapter",
    "ProtocolEventPhaseAdapter",
    "RankedOfferPhaseAdapter",
    "SupplyPhaseAdapter",
    "SupplyPhaseAuthority",
    "SupplyReportPhaseAdapter",
    "SupplyReportPhaseAuthority",
]
