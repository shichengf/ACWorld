"""Public registries for CommerceWorld extensions.

The core runtime depends only on these small registries, never on a concrete
research extension.  A plugin can register a scenario generator, platform
policy, world-event handler, market metric, or oracle primitive by stable name.
Registries reject accidental replacement so benchmark behavior cannot drift
silently between imports.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any, Generic, TypeVar


T = TypeVar("T")


class ExtensionRegistrationError(ValueError):
    """An extension name or component violates the registry contract."""


@dataclass(frozen=True, slots=True)
class ExtensionRecord(Generic[T]):
    """One named, versioned extension component."""

    name: str
    component: T
    version: str = "1"
    description: str = ""


class ExtensionRegistry(Generic[T]):
    """Thread-safe deterministic registry with duplicate protection."""

    def __init__(self, kind: str) -> None:
        if not kind.strip():
            raise ExtensionRegistrationError("registry kind must be non-empty")
        self.kind = kind
        self._records: dict[str, ExtensionRecord[T]] = {}
        self._lock = RLock()

    def register(
        self,
        name: str,
        component: T,
        *,
        version: str = "1",
        description: str = "",
    ) -> T:
        """Register ``component`` and return it for decorator-style use."""
        normalized = name.strip()
        if not normalized or any(char.isspace() for char in normalized):
            raise ExtensionRegistrationError(
                f"{self.kind} extension name must be non-empty and contain no whitespace"
            )
        if not version.strip():
            raise ExtensionRegistrationError("extension version must be non-empty")
        with self._lock:
            if normalized in self._records:
                raise ExtensionRegistrationError(
                    f"duplicate {self.kind} extension {normalized!r}"
                )
            self._records[normalized] = ExtensionRecord(
                normalized, component, version.strip(), description.strip()
            )
        return component

    def decorator(
        self,
        name: str,
        *,
        version: str = "1",
        description: str = "",
    ) -> Callable[[T], T]:
        """Return a decorator that registers a function or class."""

        def add(component: T) -> T:
            return self.register(
                name, component, version=version, description=description
            )

        return add

    def get(self, name: str) -> T:
        try:
            return self._records[name].component
        except KeyError as exc:
            raise KeyError(f"unknown {self.kind} extension {name!r}") from exc

    def record(self, name: str) -> ExtensionRecord[T]:
        try:
            return self._records[name]
        except KeyError as exc:
            raise KeyError(f"unknown {self.kind} extension {name!r}") from exc

    def names(self) -> tuple[str, ...]:
        """Stable names for manifests and paper artifact inventories."""
        with self._lock:
            return tuple(sorted(self._records))

    def snapshot(self) -> Mapping[str, ExtensionRecord[T]]:
        """Immutable, name-sorted view safe to serialize into a run manifest."""
        with self._lock:
            return MappingProxyType({name: self._records[name] for name in self.names()})

    def __contains__(self, name: object) -> bool:
        return name in self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


ScenarioGenerator = Callable[[int, Mapping[str, Any]], Any]
PlatformPolicyFactory = Callable[[Mapping[str, Any]], Any]
WorldEventHandler = Callable[[Any, Mapping[str, Any]], Any]
MarketMetric = Callable[..., Any]
OraclePrimitive = Callable[..., Any]

SCENARIO_GENERATORS: ExtensionRegistry[ScenarioGenerator] = ExtensionRegistry(
    "scenario_generator"
)
PLATFORM_POLICIES: ExtensionRegistry[PlatformPolicyFactory] = ExtensionRegistry(
    "platform_policy"
)
WORLD_EVENT_HANDLERS: ExtensionRegistry[WorldEventHandler] = ExtensionRegistry(
    "world_event_handler"
)
MARKET_METRICS: ExtensionRegistry[MarketMetric] = ExtensionRegistry("market_metric")
ORACLE_PRIMITIVES: ExtensionRegistry[OraclePrimitive] = ExtensionRegistry(
    "oracle_primitive"
)


def extension_manifest() -> dict[str, tuple[str, ...]]:
    """Return every registry's stable names for reproducibility artifacts."""
    return {
        "scenario_generators": SCENARIO_GENERATORS.names(),
        "platform_policies": PLATFORM_POLICIES.names(),
        "world_event_handlers": WORLD_EVENT_HANDLERS.names(),
        "market_metrics": MARKET_METRICS.names(),
        "oracle_primitives": ORACLE_PRIMITIVES.names(),
    }


__all__ = [
    "ExtensionRecord",
    "ExtensionRegistrationError",
    "ExtensionRegistry",
    "MARKET_METRICS",
    "ORACLE_PRIMITIVES",
    "PLATFORM_POLICIES",
    "SCENARIO_GENERATORS",
    "WORLD_EVENT_HANDLERS",
    "extension_manifest",
]
