"""Non-scoring diagnostics for formal CommerceWorld runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


RUN_DIAGNOSTICS_SCHEMA_V1 = "cwe.run-diagnostics.v1"


@dataclass(frozen=True, slots=True)
class RunDiagnosticsV1:
    """Operational facts that are forbidden from benchmark capability score."""

    provider_failure: str | None = None
    decision_format_failures: int = 0
    model_no_decision: bool = False
    agent_bridge_failure: str | None = None
    authority_closure_failure: str | None = None
    platform_failure: str | None = None
    world_failure: str | None = None
    replay_failure: str | None = None
    schema_version: str = RUN_DIAGNOSTICS_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != RUN_DIAGNOSTICS_SCHEMA_V1:
            raise ValueError("unsupported run diagnostics schema")
        if (
            isinstance(self.decision_format_failures, bool)
            or not isinstance(self.decision_format_failures, int)
            or self.decision_format_failures < 0
        ):
            raise ValueError("decision format failure count must be non-negative")
        if not isinstance(self.model_no_decision, bool):
            raise ValueError("model_no_decision must be boolean")
        for name in (
            "provider_failure",
            "agent_bridge_failure",
            "authority_closure_failure",
            "platform_failure",
            "world_failure",
            "replay_failure",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or null")

    @property
    def valid_for_scoring(self) -> bool:
        return not any((
            self.provider_failure,
            self.agent_bridge_failure,
            self.authority_closure_failure,
            self.platform_failure,
            self.world_failure,
            self.replay_failure,
        ))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "RunDiagnosticsV1":
        if not isinstance(value, Mapping):
            raise ValueError("run diagnostics must be an object")
        expected = {
            "schema_version",
            "provider_failure",
            "decision_format_failures",
            "model_no_decision",
            "agent_bridge_failure",
            "authority_closure_failure",
            "platform_failure",
            "world_failure",
            "replay_failure",
        }
        if set(value) != expected:
            raise ValueError("run diagnostics fields do not match v1 schema")
        return cls(
            schema_version=value["schema_version"],
            provider_failure=value["provider_failure"],
            decision_format_failures=value["decision_format_failures"],
            model_no_decision=value["model_no_decision"],
            agent_bridge_failure=value["agent_bridge_failure"],
            authority_closure_failure=value["authority_closure_failure"],
            platform_failure=value["platform_failure"],
            world_failure=value["world_failure"],
            replay_failure=value["replay_failure"],
        )

    @classmethod
    def from_infrastructure_exception(
        cls,
        error: BaseException,
    ) -> "RunDiagnosticsV1 | None":
        """Classify a failed formal attempt without persisting exception text."""

        error_type = type(error)
        module = error_type.__module__.casefold()
        name = error_type.__name__
        lowered_name = name.casefold()
        code = f"{error_type.__module__}.{name}"
        if "replay" in module or "replay" in lowered_name:
            return cls(replay_failure=code)
        if "transport" in lowered_name or "provider" in lowered_name:
            return cls(provider_failure=code)
        if module.startswith("world") or any(
            marker in lowered_name
            for marker in ("world", "rescore", "commerceworldintegrity")
        ):
            return cls(world_failure=code)
        if "platform" in module or "platform" in lowered_name:
            return cls(platform_failure=code)
        if module.startswith("agents") or any(
            marker in lowered_name
            for marker in ("agent", "tracker", "authority", "publictask")
        ):
            field = (
                "authority_closure_failure"
                if "authority" in lowered_name
                else "agent_bridge_failure"
            )
            return cls(**{field: code})
        return None


__all__ = ["RUN_DIAGNOSTICS_SCHEMA_V1", "RunDiagnosticsV1"]
