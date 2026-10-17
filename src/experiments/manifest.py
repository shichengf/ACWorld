"""Model manifest parsing for reproducible ACWorld panels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_MANIFEST_SCHEMA = "cwe.openrouter-model-manifest.v1"


@dataclass(frozen=True)
class ModelSpec:
    """One exact OpenRouter model identifier and its required request features."""

    model_id: str
    label: str
    family: str
    required_parameters: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelSpec":
        model_id = str(raw.get("id", "")).strip()
        label = str(raw.get("label", "")).strip()
        family = str(raw.get("family", "")).strip()
        parameters = raw.get("required_parameters", ())
        if not model_id or "/" not in model_id:
            raise ValueError(f"invalid OpenRouter model id: {model_id!r}")
        if not label or not family:
            raise ValueError(f"model {model_id!r} needs non-empty label and family")
        if not isinstance(parameters, list):
            raise ValueError(f"model {model_id!r} required_parameters must be a list")
        required = tuple(str(value).strip() for value in parameters)
        if any(not value for value in required) or len(required) != len(set(required)):
            raise ValueError(f"model {model_id!r} has invalid required_parameters")
        return cls(model_id=model_id, label=label, family=family,
                   required_parameters=required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "label": self.label,
            "family": self.family,
            "required_parameters": list(self.required_parameters),
        }


@dataclass(frozen=True)
class ModelManifest:
    """Immutable model panel definition checked into the repository."""

    manifest_id: str
    provider: str
    models: tuple[ModelSpec, ...]
    schema_version: str = MODEL_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_MANIFEST_SCHEMA:
            raise ValueError(
                f"unsupported model manifest schema {self.schema_version!r}; "
                f"expected {MODEL_MANIFEST_SCHEMA!r}"
            )
        if not self.manifest_id:
            raise ValueError("model manifest needs a manifest_id")
        if self.provider != "openrouter":
            raise ValueError("CommerceWorld v1 model panels require provider='openrouter'")
        if not self.models:
            raise ValueError("model manifest cannot be empty")
        ids = [model.model_id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model manifest contains duplicate model ids")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelManifest":
        models = raw.get("models")
        if not isinstance(models, list):
            raise ValueError("model manifest models must be a list")
        return cls(
            schema_version=str(raw.get("schema_version", "")),
            manifest_id=str(raw.get("manifest_id", "")).strip(),
            provider=str(raw.get("provider", "")).strip(),
            models=tuple(ModelSpec.from_dict(item) for item in models),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "provider": self.provider,
            "models": [model.to_dict() for model in self.models],
        }

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model.model_id for model in self.models)


def load_model_manifest(path: str | Path) -> ModelManifest:
    """Load and validate a model manifest without making any network request."""

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("model manifest root must be an object")
    return ModelManifest.from_dict(raw)


__all__ = [
    "MODEL_MANIFEST_SCHEMA",
    "ModelManifest",
    "ModelSpec",
    "load_model_manifest",
]
