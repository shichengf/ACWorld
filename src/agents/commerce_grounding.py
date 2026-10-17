"""World-derived authority for high-level merchant commerce actions.

The model may decide *what* business change to make.  It must not transcribe
listing ownership, claim lineage, or evidence access-control data into a wire
protocol action.  This module records only successful, actor-visible World
reads and authoritative Platform receipts.  The semantic compiler can then
bind those identities without importing a benchmark scenario or oracle.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from agents.decision_errors import FrameworkAuthorityError


GROUNDED_COMMERCE_AUTHORITY_V1 = "cwe.grounded-commerce-authority.v1"
GROUNDING_AUTHORITY_SNAPSHOT_V1 = "cwe.grounding-authority-snapshot.v1"
CATALOG_UPDATE_ROUTE = ("commerce.update_listing", "platform:catalog")
LISTING_CLAIM_ROUTE = ("commerce.apply_listing_claim", "platform:claims")
GROUNDING_REQUIRED_ROUTES = frozenset(
    {
        CATALOG_UPDATE_ROUTE,
        LISTING_CLAIM_ROUTE,
    }
)


class CommerceGroundingError(FrameworkAuthorityError):
    """A high-level commerce phase lacks authoritative World grounding."""


@dataclass(frozen=True, slots=True)
class _ImmutableAuthorityMapping(Mapping[str, Any]):
    """Small recursively immutable mapping used inside authority snapshots."""

    _items: tuple[tuple[str, Any], ...]

    @classmethod
    def freeze(cls, value: Mapping[str, Any]) -> _ImmutableAuthorityMapping:
        items: list[tuple[str, Any]] = []
        names: set[str] = set()
        for raw_name, item in value.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise CommerceGroundingError("grounding authority contains a malformed field name")
            if raw_name in names:
                raise CommerceGroundingError("grounding authority contains duplicate field names")
            names.add(raw_name)
            items.append((raw_name, _freeze_authority_value(item)))
        return cls(tuple(items))

    def __getitem__(self, key: str) -> Any:
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, memo: dict[int, Any]) -> _ImmutableAuthorityMapping:
        del memo
        return self


def _freeze_authority_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _ImmutableAuthorityMapping.freeze(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_authority_value(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise CommerceGroundingError("grounding authority contains a non-JSON business value")


def _thaw_authority_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _thaw_authority_value(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_authority_value(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class GroundedRouteAuthority:
    """Immutable World/Platform authority for one registered mutation route."""

    route: tuple[str, str]
    kind: str
    value: Mapping[str, Any]

    def __post_init__(self) -> None:
        route = _route(self.route)
        expected_kind = {
            CATALOG_UPDATE_ROUTE: "catalog_update",
            LISTING_CLAIM_ROUTE: "listing_claim",
        }.get(route)
        if expected_kind is None or self.kind != expected_kind:
            raise CommerceGroundingError("grounded route authority has an inconsistent route kind")
        if not isinstance(self.value, Mapping):
            raise CommerceGroundingError("grounded route authority value must be an object")
        frozen = _ImmutableAuthorityMapping.freeze(self.value)
        if (
            frozen.get("schema_version") != GROUNDED_COMMERCE_AUTHORITY_V1
            or frozen.get("kind") != self.kind
        ):
            raise CommerceGroundingError(
                "grounded route authority value has an inconsistent identity"
            )
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "value", frozen)

    def to_mutable_value(self) -> dict[str, Any]:
        """Return a detached compiler payload, never the stored authority."""

        return _thaw_authority_value(self.value)


@dataclass(frozen=True, slots=True)
class ResolvedAuthorityPrerequisites:
    """Single immutable prerequisite authority for one Agent phase.

    This object is the grounding boundary consumed by an ``AgentPhaseContract``
    or ``PhaseAdapter``.  It binds resource visibility, route prerequisites,
    concrete route authority, revisions, and authenticated source closure in
    one value, so domain code cannot assemble a contradictory collection of
    context keys.
    """

    actor_id: str
    inbound_id: str
    visible_sku_ids: tuple[str, ...]
    resource_revision: int
    current_listing_id: str | None
    decision_surface_revision: int
    decision_surface_source_msg_ids: tuple[str, ...]
    completed_claim_ids: tuple[str, ...]
    required_routes: frozenset[tuple[str, str]]
    pending_routes: frozenset[tuple[str, str]]
    route_authorities: tuple[GroundedRouteAuthority, ...]
    schema_version: str = GROUNDING_AUTHORITY_SNAPSHOT_V1

    def __post_init__(self) -> None:
        if self.schema_version != GROUNDING_AUTHORITY_SNAPSHOT_V1:
            raise CommerceGroundingError("grounding authority snapshot has an unsupported schema")
        if _text(self.actor_id) is None or _text(self.inbound_id) is None:
            raise CommerceGroundingError(
                "grounding authority snapshot has no actor or inbound identity"
            )
        visible = _unique_texts(
            self.visible_sku_ids,
            name="visible SKU",
            sort_values=True,
        )
        sources = _unique_texts(
            self.decision_surface_source_msg_ids,
            name="decision surface source message",
        )
        completed_claims = _unique_texts(
            self.completed_claim_ids,
            name="completed claim",
            sort_values=True,
        )
        required = frozenset(_route(route) for route in self.required_routes)
        pending = frozenset(_route(route) for route in self.pending_routes)
        if not required <= GROUNDING_REQUIRED_ROUTES:
            raise CommerceGroundingError(
                "grounding authority snapshot contains an unregistered route"
            )
        if not pending <= required:
            raise CommerceGroundingError("pending grounding routes exceed required routes")
        if any(
            not isinstance(revision, int) or isinstance(revision, bool) or revision < 0
            for revision in (
                self.resource_revision,
                self.decision_surface_revision,
            )
        ):
            raise CommerceGroundingError("grounding authority snapshot has an invalid revision")
        if required and _text(self.current_listing_id) is None:
            raise CommerceGroundingError("grounding authority snapshot has no active listing")
        if not required and self.current_listing_id is not None:
            raise CommerceGroundingError(
                "resource-only grounding snapshot cannot bind a mutation listing"
            )
        if any(not isinstance(row, GroundedRouteAuthority) for row in self.route_authorities):
            raise CommerceGroundingError(
                "grounding authority snapshot contains a malformed route authority"
            )
        authorities = tuple(sorted(self.route_authorities, key=lambda row: row.route))
        authority_routes = [row.route for row in authorities]
        if len(authority_routes) != len(set(authority_routes)):
            raise CommerceGroundingError(
                "grounding authority snapshot binds a route more than once"
            )
        if frozenset(authority_routes) != required - pending:
            raise CommerceGroundingError(
                "grounding authority snapshot pending state is inconsistent"
            )
        for authority in authorities:
            listing_id = authority.value.get("sku_id") or authority.value.get("listing_id")
            if listing_id != self.current_listing_id:
                raise CommerceGroundingError(
                    "grounded route authority differs from active listing lineage"
                )
        object.__setattr__(self, "visible_sku_ids", visible)
        object.__setattr__(self, "decision_surface_source_msg_ids", sources)
        object.__setattr__(self, "completed_claim_ids", completed_claims)
        object.__setattr__(self, "required_routes", required)
        object.__setattr__(self, "pending_routes", pending)
        object.__setattr__(self, "route_authorities", authorities)

    @property
    def bound_routes(self) -> frozenset[tuple[str, str]]:
        return self.required_routes - self.pending_routes

    def requires(self, route: tuple[str, str]) -> bool:
        return _route(route) in self.required_routes

    def is_pending(self, route: tuple[str, str]) -> bool:
        return _route(route) in self.pending_routes

    def authority_for(
        self,
        route: tuple[str, str],
    ) -> GroundedRouteAuthority | None:
        resolved = _route(route)
        return next(
            (row for row in self.route_authorities if row.route == resolved),
            None,
        )


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and bool(value.strip()) else None


def _unique_texts(
    values: Any,
    *,
    name: str,
    sort_values: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, frozenset)):
        raise CommerceGroundingError(f"grounding {name} values are malformed")
    resolved = tuple(values)
    if any(_text(value) is None for value in resolved):
        raise CommerceGroundingError(f"grounding {name} values are malformed")
    if len(resolved) != len(set(resolved)):
        raise CommerceGroundingError(f"grounding {name} values are ambiguous")
    normalized = tuple(str(value) for value in resolved)
    return tuple(sorted(normalized)) if sort_values else normalized


def _route(value: Any) -> tuple[str, str]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(_text(item) is None for item in value)
    ):
        raise CommerceGroundingError("grounding route is malformed")
    return str(value[0]), str(value[1])


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    raise CommerceGroundingError("World commerce grounding contains a non-JSON business value")


def _content_schema(content: Mapping[str, Any]) -> dict[str, str]:
    if not content or any(_text(name) is None for name in content):
        raise CommerceGroundingError("listing claim content is malformed")
    return {str(name): _json_type(value) for name, value in content.items()}


def _claim_observation(value: Any) -> dict[str, Any] | None:
    """Project either canonical wire or formatted dataclass output."""

    if not isinstance(value, Mapping):
        return None
    claim_id = _text(value.get("claim_id"))
    listing_id = _text(value.get("listing_id"))
    merchant_id = _text(value.get("merchant_id"))
    subject = _text(value.get("subject"))
    versions = value.get("versions")
    if (
        claim_id is None
        or listing_id is None
        or merchant_id is None
        or subject is None
        or not isinstance(versions, (list, tuple))
        or not versions
        or not isinstance(versions[-1], Mapping)
    ):
        return None
    current = versions[-1]
    state = current.get("state", value.get("state"))
    content = current.get("content")
    if state not in {"draft", "published", "corrected", "retracted"}:
        return None
    if not isinstance(content, Mapping) or not content:
        return None
    used_evidence_record_ids: list[str] = []
    for version in versions:
        if not isinstance(version, Mapping):
            return None
        evidence = version.get("evidence", ())
        if not isinstance(evidence, (list, tuple)):
            return None
        for row in evidence:
            if not isinstance(row, Mapping):
                return None
            source_id = _text(row.get("source_id"))
            if source_id is None:
                return None
            used_evidence_record_ids.append(source_id)
    if len(used_evidence_record_ids) != len(set(used_evidence_record_ids)):
        # Canonical claim history forbids reusing one source. Treat a World
        # response that already violates that invariant as unusable authority.
        return None
    return {
        "claim_id": claim_id,
        "listing_id": listing_id,
        "merchant_id": merchant_id,
        "subject": subject,
        "state": state,
        "content_schema": _content_schema(content),
        "used_evidence_record_ids": sorted(used_evidence_record_ids),
    }


@dataclass
class CommerceActionGrounding:
    """Per-Agent, replay-deterministic authority learned from CommerceWorld."""

    actor_id: str
    active_listing_id: str | None = None
    _active_inbound_id: str | None = None
    _listings: dict[str, dict[str, Any]] = field(default_factory=dict)
    _claims: dict[str, dict[str, Any]] = field(default_factory=dict)
    _evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    _outbound_listing_ids: dict[str, str] = field(default_factory=dict)
    _outbound_claim_ids: dict[str, str] = field(default_factory=dict)
    _completed_claim_ids: set[str] = field(default_factory=set)
    _resource_inbound_id: str | None = None
    _visible_sku_ids: set[str] = field(default_factory=set)
    _resource_revision: int = 0
    _decision_surface_revision: int = 0
    _decision_surface_source_msg_ids: list[str] = field(default_factory=list)

    def bind_resource_inbound(self, *, inbound_id: str) -> None:
        """Start one exact turn-local window for World-grounded choices.

        Business resource identities may be exposed to a provider only after
        the current turn has read them from World.  A later inbound message
        therefore clears the finite choice set instead of inheriting stale
        listing identities from an earlier transaction.
        """

        if _text(inbound_id) is None:
            raise CommerceGroundingError("commerce resource inbound has no identity")
        if inbound_id == self._resource_inbound_id:
            return
        if self._visible_sku_ids:
            self._visible_sku_ids.clear()
            self._resource_revision += 1
        self._resource_inbound_id = inbound_id

    def bind_inbound(
        self,
        *,
        inbound_id: str,
        in_reply_to: str | None,
        action_kind: str,
        payload: Any,
        required_routes: frozenset[tuple[str, str]],
    ) -> None:
        """Bind the phase target and consume authoritative Platform receipts."""

        if not required_routes:
            return
        if _text(inbound_id) is None:
            raise CommerceGroundingError("grounded commerce inbound has no identity")
        same_inbound = inbound_id == self._active_inbound_id
        current: str | None = None
        if isinstance(payload, Mapping):
            current = _text(payload.get("sku_id")) or _text(payload.get("listing_id"))
            if action_kind == "platform.listing_claim_updated":
                claim = _claim_observation(payload.get("claim"))
                if claim is None:
                    raise CommerceGroundingError(
                        "Platform claim receipt has no authoritative claim"
                    )
                if claim["merchant_id"] != self.actor_id:
                    raise CommerceGroundingError(
                        "Platform claim receipt owner differs from the Agent"
                    )
                expected_listing_id = self._outbound_listing_ids.get(str(in_reply_to))
                expected_claim_id = self._outbound_claim_ids.get(str(in_reply_to))
                if (
                    expected_listing_id is None
                    or claim["listing_id"] != expected_listing_id
                    or expected_claim_id is None
                    or claim["claim_id"] != expected_claim_id
                ):
                    raise CommerceGroundingError(
                        "Platform claim receipt differs from committed action lineage"
                    )
                self._store_if_changed(
                    self._claims,
                    claim["claim_id"],
                    claim,
                )
                self._completed_claim_ids.add(claim["claim_id"])
                current = claim["listing_id"]
        if not same_inbound and action_kind != "platform.listing_claim_updated":
            # A new principal or participant directive starts a fresh grounding
            # window.  Cached observations from an older turn must not silently
            # authorize a new mutation, even when it names the same listing.
            if self._listings or self._claims or self._evidence:
                self._listings.clear()
                self._claims.clear()
                self._evidence.clear()
            self._outbound_listing_ids.clear()
            self._outbound_claim_ids.clear()
            self._completed_claim_ids.clear()
            self.active_listing_id = None
        if current is None and same_inbound:
            current = self.active_listing_id
        if current is not None:
            self.active_listing_id = current
            self._active_inbound_id = inbound_id
        if self.active_listing_id is None:
            raise CommerceGroundingError("grounded commerce phase has no inbound listing target")

    def record_committed_outbound(
        self,
        *,
        msg_id: str,
        action_kind: str,
        payload: Any,
    ) -> None:
        """Bind a later Platform receipt to one Runtime-committed actor intent."""

        if action_kind not in {
            "commerce.update_listing",
            "commerce.apply_listing_claim",
        }:
            return
        if not isinstance(payload, Mapping):
            raise CommerceGroundingError("committed grounded commerce action has no payload")
        listing_id = _text(payload.get("sku_id")) or _text(payload.get("listing_id"))
        if _text(msg_id) is None or listing_id is None:
            raise CommerceGroundingError(
                "committed grounded commerce action lost its listing lineage"
            )
        existing = self._outbound_listing_ids.get(msg_id)
        if existing is not None and existing != listing_id:
            raise CommerceGroundingError(
                "committed grounded commerce message identity is ambiguous"
            )
        self._outbound_listing_ids[msg_id] = listing_id
        if action_kind == "commerce.apply_listing_claim":
            claim_id = _text(payload.get("claim_id"))
            if claim_id is None:
                raise CommerceGroundingError(
                    "committed listing claim action lost its claim lineage"
                )
            existing_claim_id = self._outbound_claim_ids.get(msg_id)
            if existing_claim_id is not None and existing_claim_id != claim_id:
                raise CommerceGroundingError("committed claim message identity is ambiguous")
            self._outbound_claim_ids[msg_id] = claim_id

    def observe_read_results(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Record real World reads and their model-surface provenance.

        Every framework-produced World result changes the next business
        request, even when it is an error or does not establish mutation
        authority.  Its authenticated response identity must therefore join
        the same monotonic closure as the decision-surface revision.  Local
        in-process reads have no message identity but still advance revision.
        """

        for row in rows:
            tool = row.get("tool")
            if not isinstance(tool, str) or not tool.startswith("world."):
                continue
            result = row.get("result")
            source_msg_id = row.get("source_msg_id")
            source = _text(source_msg_id)
            if source is None:
                self._decision_surface_revision += 1
            elif source not in self._decision_surface_source_msg_ids:
                self._decision_surface_source_msg_ids.append(source)
                self._decision_surface_revision += 1
            if tool == "world.get_listing":
                self._observe_visible_listing(result)
                self._observe_listing(result)
            elif tool == "world.search_catalog" and isinstance(result, (list, tuple)):
                for listing in result:
                    self._observe_visible_listing(listing)
            elif tool == "world.get_listing_claim":
                self._observe_claim(result)
            elif tool == "world.list_listing_claims" and isinstance(result, (list, tuple)):
                for claim in result:
                    self._observe_claim(claim)
            elif tool == "world.get_evidence_record":
                self._observe_evidence(result)

    def _observe_visible_listing(self, value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        sku_id = _text(value.get("sku_id"))
        if sku_id is not None and sku_id not in self._visible_sku_ids:
            self._visible_sku_ids.add(sku_id)
            self._resource_revision += 1

    def _snapshot(
        self,
        *,
        inbound_id: str,
        required_routes: frozenset[tuple[str, str]],
    ) -> ResolvedAuthorityPrerequisites:
        """Freeze the complete prerequisite authority for one phase."""

        if inbound_id != self._resource_inbound_id:
            raise CommerceGroundingError(
                "grounding snapshot inbound differs from its resource window"
            )
        listing_id = self.active_listing_id if required_routes else None
        if required_routes and listing_id is None:
            raise CommerceGroundingError("grounded commerce phase has no active listing")
        authorities: list[GroundedRouteAuthority] = []
        listing = self._listings.get(str(listing_id))
        if listing is not None and CATALOG_UPDATE_ROUTE in required_routes:
            authorities.append(
                GroundedRouteAuthority(
                    route=CATALOG_UPDATE_ROUTE,
                    kind="catalog_update",
                    value={
                        "schema_version": GROUNDED_COMMERCE_AUTHORITY_V1,
                        "kind": "catalog_update",
                        "sku_id": listing_id,
                        "attribute_schemas": listing["attribute_schemas"],
                    },
                )
            )
        if listing is not None and LISTING_CLAIM_ROUTE in required_routes:
            claims = [
                claim
                for claim in self._claims.values()
                if claim["listing_id"] == listing_id
                and claim["merchant_id"] == self.actor_id
                and claim["state"] != "retracted"
            ]
            if claims:
                authorities.append(
                    GroundedRouteAuthority(
                        route=LISTING_CLAIM_ROUTE,
                        kind="listing_claim",
                        value={
                            "schema_version": GROUNDED_COMMERCE_AUTHORITY_V1,
                            "kind": "listing_claim",
                            "listing_id": listing_id,
                            "claims": sorted(
                                claims,
                                key=lambda row: row["claim_id"],
                            ),
                            "evidence_records": sorted(
                                self._evidence.values(),
                                key=lambda row: row["record_id"],
                            ),
                        },
                    )
                )
        bound_routes = frozenset(row.route for row in authorities)
        return ResolvedAuthorityPrerequisites(
            actor_id=self.actor_id,
            inbound_id=inbound_id,
            visible_sku_ids=tuple(sorted(self._visible_sku_ids)),
            resource_revision=self._resource_revision,
            current_listing_id=listing_id,
            decision_surface_revision=self._decision_surface_revision,
            decision_surface_source_msg_ids=tuple(self._decision_surface_source_msg_ids),
            completed_claim_ids=tuple(sorted(self._completed_claim_ids)),
            required_routes=required_routes,
            pending_routes=required_routes - bound_routes,
            route_authorities=tuple(authorities),
        )

    def _observe_listing(self, value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        sku_id = _text(value.get("sku_id"))
        merchant_id = _text(value.get("merchant_id"))
        attributes = value.get("attributes")
        if sku_id is None or merchant_id != self.actor_id or not isinstance(attributes, Mapping):
            return
        schemas = {
            str(name): _json_type(item)
            for name, item in attributes.items()
            if _text(name) is not None
        }
        self._store_if_changed(
            self._listings,
            sku_id,
            {
                "sku_id": sku_id,
                "merchant_id": merchant_id,
                "attribute_schemas": schemas,
            },
        )

    def _observe_claim(self, value: Any) -> None:
        claim = _claim_observation(value)
        if claim is not None and claim["merchant_id"] == self.actor_id:
            self._store_if_changed(
                self._claims,
                claim["claim_id"],
                claim,
            )

    def _observe_evidence(self, value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        record_id = _text(value.get("record_id"))
        subject_id = _text(value.get("subject_id"))
        owner_id = _text(value.get("owner_id"))
        read_acl = value.get("read_acl")
        if (
            record_id is None
            or subject_id is None
            or owner_id is None
            or not isinstance(read_acl, (list, tuple))
            or any(_text(actor_id) is None for actor_id in read_acl)
            or not (owner_id == self.actor_id or self.actor_id in read_acl)
        ):
            return
        self._store_if_changed(
            self._evidence,
            record_id,
            {
                "record_id": record_id,
                "subject_id": subject_id,
                "owner_id": owner_id,
                "read_acl": tuple(str(actor_id) for actor_id in read_acl),
            },
        )

    def _store_if_changed(
        self,
        store: dict[str, dict[str, Any]],
        key: str,
        value: dict[str, Any],
    ) -> None:
        if store.get(key) != value:
            store[key] = value


@dataclass
class AuthorityPrerequisiteResolver:
    """Shared authority prerequisite resolver used by every Agent phase.

    Callers supply the authenticated inbound and the current phase's registered
    routes once.  Resource visibility, mutation grounding, revision tracking,
    and authority-source closure are resolved together; domain code no longer
    decides which scattered context keys to populate.
    """

    actor_id: str
    grounding: CommerceActionGrounding = field(init=False)

    def __post_init__(self) -> None:
        if _text(self.actor_id) is None:
            raise CommerceGroundingError("authority prerequisite resolver has no actor identity")
        self.grounding = CommerceActionGrounding(actor_id=self.actor_id)

    def resolve_phase(
        self,
        *,
        inbound_id: str,
        in_reply_to: str | None,
        action_kind: str,
        payload: Any,
        registered_routes: Sequence[tuple[str, str]],
    ) -> ResolvedAuthorityPrerequisites:
        self.grounding.bind_resource_inbound(inbound_id=inbound_id)
        routes = tuple(_route(route) for route in registered_routes)
        required = frozenset(route for route in routes if route in GROUNDING_REQUIRED_ROUTES)
        if required:
            self.grounding.bind_inbound(
                inbound_id=inbound_id,
                in_reply_to=in_reply_to,
                action_kind=action_kind,
                payload=payload,
                required_routes=required,
            )
        return self.grounding._snapshot(
            inbound_id=inbound_id,
            required_routes=required,
        )

    def observe_read_results(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.grounding.observe_read_results(rows)

    def record_committed_outbound(self, *, msg_id: str, action_kind: str, payload: Any) -> None:
        self.grounding.record_committed_outbound(
            msg_id=msg_id,
            action_kind=action_kind,
            payload=payload,
        )


__all__ = [
    "CATALOG_UPDATE_ROUTE",
    "AuthorityPrerequisiteResolver",
    "CommerceActionGrounding",
    "CommerceGroundingError",
    "GROUNDED_COMMERCE_AUTHORITY_V1",
    "GROUNDING_AUTHORITY_SNAPSHOT_V1",
    "GROUNDING_REQUIRED_ROUTES",
    "GroundedRouteAuthority",
    "LISTING_CLAIM_ROUTE",
    "ResolvedAuthorityPrerequisites",
]
