"""Canonical persistence for immutable actor-context registrations.

The live :class:`~runtime.actor_context.ActorContextResolver` receives trusted
registrations before an episode begins.  Formal evidence must preserve those
same inputs so an offline verifier can rebuild the resolver without consulting
scenario code or actor-authored payload fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from protocol.errors import SchemaError
from runtime.actor_context import (
    DEFAULT_REPORT_ACTION_KINDS,
    DEFAULT_ROOT_ACTION_KINDS,
    ActorContextError,
    ActorContextResolver,
    RegisteredActorContext,
)


ACTOR_CONTEXTS_ARTIFACT = "actor.contexts.json"
ACTOR_CONTEXTS_SCHEMA = "cwe.actor-context-registrations.v1"

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "episode_context_id",
        "root_action_kinds",
        "report_action_kinds",
        "registrations",
    }
)
_REGISTRATION_FIELDS = frozenset(
    {
        "root_msg_id",
        "root_action_kind",
        "actor_id",
        "principal_id",
        "task_id",
        "mandate_id",
        "context_id",
        "report_action_kinds",
    }
)


class ActorContextArtifactError(SchemaError):
    """The persisted actor-context registry is absent or non-canonical."""


@dataclass(frozen=True, slots=True)
class ActorContextRegistrations:
    """One immutable, replay-complete resolver configuration."""

    episode_context_id: str
    root_action_kinds: tuple[str, ...]
    report_action_kinds: tuple[str, ...]
    registrations: tuple[RegisteredActorContext, ...]
    schema_version: str = ACTOR_CONTEXTS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ACTOR_CONTEXTS_SCHEMA:
            raise ActorContextArtifactError(
                "unsupported actor-context registration schema"
            )
        _require_text(self.episode_context_id, "episode_context_id")
        _require_canonical_action_scope(
            self.root_action_kinds,
            "root_action_kinds",
        )
        _require_canonical_action_scope(
            self.report_action_kinds,
            "report_action_kinds",
        )
        if not isinstance(self.registrations, tuple):
            raise ActorContextArtifactError("registrations must be a tuple")
        root_ids = tuple(item.root_msg_id for item in self.registrations)
        if root_ids != tuple(sorted(root_ids)):
            raise ActorContextArtifactError(
                "actor-context registrations must be sorted by root_msg_id"
            )
        if len(root_ids) != len(set(root_ids)):
            raise ActorContextArtifactError(
                "actor-context registrations contain duplicate root_msg_id values"
            )
        try:
            # Reuse the live resolver's registration validation.  No envelope is
            # observed here; this only validates the frozen configuration.
            self.build_resolver()
        except ActorContextError as exc:
            raise ActorContextArtifactError(
                f"invalid actor-context registration contract: {exc}"
            ) from exc

    def build_resolver(self) -> ActorContextResolver:
        """Build a fresh resolver with exactly the persisted configuration."""

        return ActorContextResolver(
            episode_context_id=self.episode_context_id,
            actor_contexts=self.registrations,
            root_action_kinds=self.root_action_kinds,
            report_action_kinds=self.report_action_kinds,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON-compatible artifact payload."""

        return {
            "schema_version": self.schema_version,
            "episode_context_id": self.episode_context_id,
            "root_action_kinds": list(self.root_action_kinds),
            "report_action_kinds": list(self.report_action_kinds),
            "registrations": [
                {
                    "root_msg_id": item.root_msg_id,
                    "root_action_kind": item.root_action_kind,
                    "actor_id": item.actor_id,
                    "principal_id": item.principal_id,
                    "task_id": item.task_id,
                    "mandate_id": item.mandate_id,
                    "context_id": item.context_id,
                    "report_action_kinds": list(item.report_action_kinds),
                }
                for item in self.registrations
            ],
        }


def build_actor_context_registrations(
    *,
    episode_context_id: str,
    actor_contexts: Iterable[RegisteredActorContext] = (),
    root_action_kinds: Iterable[str] = DEFAULT_ROOT_ACTION_KINDS,
    report_action_kinds: Iterable[str] = DEFAULT_REPORT_ACTION_KINDS,
) -> ActorContextRegistrations:
    """Normalize trusted resolver inputs into their canonical artifact form."""

    roots = _normalized_action_scope(root_action_kinds, "root_action_kinds")
    reports = _normalized_action_scope(report_action_kinds, "report_action_kinds")
    normalized: list[RegisteredActorContext] = []
    root_ids: set[str] = set()
    for item in actor_contexts:
        if not isinstance(item, RegisteredActorContext):
            raise ActorContextArtifactError(
                "actor_contexts must contain RegisteredActorContext values"
            )
        if item.root_msg_id in root_ids:
            raise ActorContextArtifactError(
                f"duplicate actor-context root_msg_id {item.root_msg_id!r}"
            )
        root_ids.add(item.root_msg_id)
        normalized.append(
            RegisteredActorContext(
                root_msg_id=item.root_msg_id,
                root_action_kind=item.root_action_kind,
                actor_id=item.actor_id,
                principal_id=item.principal_id,
                task_id=item.task_id,
                mandate_id=item.mandate_id,
                context_id=item.context_id,
                report_action_kinds=cast(
                    tuple[Any, ...], tuple(sorted(item.report_action_kinds))
                ),
            )
        )
    normalized.sort(key=lambda item: item.root_msg_id)
    return ActorContextRegistrations(
        episode_context_id=episode_context_id,
        root_action_kinds=roots,
        report_action_kinds=reports,
        registrations=tuple(normalized),
    )


def write_actor_contexts(
    path: str | Path,
    *,
    episode_context_id: str,
    actor_contexts: Iterable[RegisteredActorContext] = (),
    root_action_kinds: Iterable[str] = DEFAULT_ROOT_ACTION_KINDS,
    report_action_kinds: Iterable[str] = DEFAULT_REPORT_ACTION_KINDS,
) -> ActorContextRegistrations:
    """Persist the immutable registry once, refusing a changed replacement."""

    artifact = build_actor_context_registrations(
        episode_context_id=episode_context_id,
        actor_contexts=actor_contexts,
        root_action_kinds=root_action_kinds,
        report_action_kinds=report_action_kinds,
    )
    encoded = _canonical_json(artifact.to_dict()).encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != encoded:
            raise ActorContextArtifactError(
                "immutable actor-context artifact already exists with different bytes"
            )
        return load_actor_contexts(target, strict=True)
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
    except FileExistsError:
        # A concurrent creator is acceptable only when it wrote the same exact
        # immutable contract.
        if target.read_bytes() != encoded:
            raise ActorContextArtifactError(
                "immutable actor-context artifact was concurrently changed"
            ) from None
    return artifact


def load_actor_contexts(
    path: str | Path,
    *,
    strict: bool = True,
) -> ActorContextRegistrations:
    """Load and validate one exact actor-context registration artifact."""

    target = Path(path)
    try:
        raw = target.read_bytes()
    except FileNotFoundError as exc:
        raise ActorContextArtifactError(
            f"missing actor-context registration artifact: {target}"
        ) from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ActorContextArtifactError(
            f"invalid actor-context registration artifact {target}: {exc}"
        ) from exc
    if not isinstance(value, dict) or frozenset(value) != _TOP_LEVEL_FIELDS:
        raise ActorContextArtifactError(
            "actor-context artifact fields differ from the exact schema"
        )
    if value.get("schema_version") != ACTOR_CONTEXTS_SCHEMA:
        raise ActorContextArtifactError(
            "unsupported actor-context registration schema"
        )
    roots = _string_list(value.get("root_action_kinds"), "root_action_kinds")
    reports = _string_list(
        value.get("report_action_kinds"),
        "report_action_kinds",
    )
    raw_registrations = value.get("registrations")
    if not isinstance(raw_registrations, list):
        raise ActorContextArtifactError("registrations must be an array")
    registrations: list[RegisteredActorContext] = []
    for position, raw_registration in enumerate(raw_registrations):
        if (
            not isinstance(raw_registration, Mapping)
            or frozenset(raw_registration) != _REGISTRATION_FIELDS
        ):
            raise ActorContextArtifactError(
                f"actor-context registration {position} fields differ from the exact schema"
            )
        scoped_reports = _string_list(
            raw_registration.get("report_action_kinds"),
            f"registrations[{position}].report_action_kinds",
        )
        try:
            registrations.append(
                RegisteredActorContext(
                    root_msg_id=_text_field(raw_registration, "root_msg_id", position),
                    root_action_kind=_text_field(
                        raw_registration, "root_action_kind", position
                    ),
                    actor_id=_text_field(raw_registration, "actor_id", position),
                    principal_id=_text_field(
                        raw_registration, "principal_id", position
                    ),
                    task_id=_text_field(raw_registration, "task_id", position),
                    mandate_id=_text_field(raw_registration, "mandate_id", position),
                    context_id=_text_field(raw_registration, "context_id", position),
                    report_action_kinds=cast(tuple[Any, ...], scoped_reports),
                )
            )
        except ActorContextError as exc:
            raise ActorContextArtifactError(
                f"invalid actor-context registration {position}: {exc}"
            ) from exc
    artifact = ActorContextRegistrations(
        episode_context_id=_text_value(
            value.get("episode_context_id"), "episode_context_id"
        ),
        root_action_kinds=roots,
        report_action_kinds=reports,
        registrations=tuple(registrations),
    )
    if strict and raw != _canonical_json(artifact.to_dict()).encode("utf-8"):
        raise ActorContextArtifactError(
            f"non-canonical actor-context artifact bytes: {target}"
        )
    return artifact


def _normalized_action_scope(values: Iterable[str], label: str) -> tuple[str, ...]:
    try:
        normalized = tuple(sorted(values))
    except TypeError as exc:
        raise ActorContextArtifactError(f"{label} must contain strings") from exc
    _require_canonical_action_scope(normalized, label)
    return normalized


def _require_canonical_action_scope(values: tuple[str, ...], label: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ActorContextArtifactError(f"{label} must be a non-empty tuple")
    for value in values:
        _require_text(value, f"{label} item")
    if values != tuple(sorted(set(values))):
        raise ActorContextArtifactError(f"{label} must be sorted and unique")


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ActorContextArtifactError(f"{label} must be an array")
    result = tuple(value)
    _require_canonical_action_scope(result, label)
    return cast(tuple[str, ...], result)


def _text_field(value: Mapping[str, Any], field: str, position: int) -> str:
    return _text_value(value.get(field), f"registrations[{position}].{field}")


def _text_value(value: Any, label: str) -> str:
    _require_text(value, label)
    return cast(str, value)


def _require_text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ActorContextArtifactError(f"{label} must be a non-empty string")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ActorContextArtifactError(
            f"actor-context artifact is not canonical JSON: {exc}"
        ) from exc


__all__ = [
    "ACTOR_CONTEXTS_ARTIFACT",
    "ACTOR_CONTEXTS_SCHEMA",
    "ActorContextArtifactError",
    "ActorContextRegistrations",
    "build_actor_context_registrations",
    "load_actor_contexts",
    "write_actor_contexts",
]
