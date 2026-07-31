"""Read-only OpenRouter model availability check for a pinned manifest."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from experiments.manifest import ModelManifest


OPENROUTER_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"


class PreflightError(RuntimeError):
    """The availability endpoint could not return a valid model catalog."""


@dataclass(frozen=True)
class ModelAvailability:
    model_id: str
    present: bool
    missing_parameters: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.present and not self.missing_parameters

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "present": self.present,
            "missing_parameters": list(self.missing_parameters),
            "ready": self.ready,
        }


@dataclass(frozen=True)
class PreflightReport:
    manifest_id: str
    endpoint: str
    models: tuple[ModelAvailability, ...]

    @property
    def ok(self) -> bool:
        return all(model.ready for model in self.models)

    @property
    def unavailable_model_ids(self) -> tuple[str, ...]:
        return tuple(model.model_id for model in self.models if not model.ready)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "endpoint": self.endpoint,
            # The models catalog exposes aggregate capability metadata.  It
            # does not prove that a concrete provider endpoint can route the
            # exact Chat Completions request used by a later paid run.
            "scope": "catalog_capability_only",
            "endpoint_route_probe_performed": False,
            "ok": self.ok,
            "models": [model.to_dict() for model in self.models],
        }


def preflight_openrouter_models(
    manifest: ModelManifest,
    *,
    endpoint: str = OPENROUTER_MODELS_ENDPOINT,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> PreflightReport:
    """GET OpenRouter's model catalog and require exact manifest IDs.

    This function never chooses aliases or fallback models.  Missing IDs and
    unsupported required request parameters are returned as explicit failures.
    It only reads the public models endpoint; it does not call chat completions.
    """

    headers = {
        "Accept": "application/json",
        "User-Agent": "CommerceWorld-KDD2027-Preflight/1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise PreflightError(f"OpenRouter models endpoint HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise PreflightError(f"OpenRouter models endpoint unavailable: {exc.reason}") from exc

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"OpenRouter models endpoint returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise PreflightError("OpenRouter models endpoint response needs a data list")

    catalog: dict[str, set[str]] = {}
    for item in payload["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        supported = item.get("supported_parameters", [])
        parameters = (
            {str(value) for value in supported}
            if isinstance(supported, list)
            else set()
        )
        catalog[item["id"]] = parameters

    checks: list[ModelAvailability] = []
    for model in manifest.models:
        supported = catalog.get(model.model_id)
        if supported is None:
            checks.append(ModelAvailability(model_id=model.model_id, present=False))
            continue
        missing = tuple(
            parameter for parameter in model.required_parameters if parameter not in supported
        )
        checks.append(ModelAvailability(
            model_id=model.model_id, present=True, missing_parameters=missing
        ))
    return PreflightReport(
        manifest_id=manifest.manifest_id,
        endpoint=endpoint,
        models=tuple(checks),
    )


__all__ = [
    "OPENROUTER_MODELS_ENDPOINT",
    "ModelAvailability",
    "PreflightError",
    "PreflightReport",
    "preflight_openrouter_models",
]
