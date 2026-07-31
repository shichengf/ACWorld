"""Deterministic, human-readable JSON serialization for evaluator artifacts.

One canonical encoder shared by the in-process and HTTP launch paths so
``world.initial.json`` / ``world.final.json`` / ``reward.json`` have a stable,
diff-friendly shape. Handles dataclasses, Enums, Decimal, NewType (str/int at
runtime), tuples, and dict keys. No ``NaN``/``Infinity``; no dataclass repr
strings; no agent private_context (the snapshot/report types simply do not
carry it).
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

WORLD_SCHEMA_VERSION = "world.v1"


def to_canonical(obj: Any) -> Any:
    """Recursively convert ``obj`` into a JSON-safe structure with deterministic
    types. Decimals become strings (exact); Enums become their value; mappings
    get string keys; dataclasses become field dicts.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float in evaluator artifact: {obj!r}")
        return obj
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, Enum):
        return to_canonical(obj.value)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Governance envelopes contain protocol payload dataclasses whose
        # canonical wire schemas are deliberately not their Python field
        # layouts.  In particular, protocol records add ``schema_version``
        # while the persistence envelope uses ``schema_id``.  Snapshot and
        # commit artifacts are replay inputs, so they must use the same strict
        # codec as SQLite and VCP instead of generic ``dataclasses.fields``.
        # Keep the import local so the evaluator serializer does not impose a
        # module-import dependency on the optional governance integration.
        try:
            from world.market_governance_persistence import (
                GovernancePolicyEnvelope,
                GovernanceRecordEnvelope,
                policy_envelope_to_wire,
                record_envelope_to_wire,
            )
        except ImportError:  # pragma: no cover - minimal evaluator installs
            pass
        else:
            if isinstance(obj, GovernancePolicyEnvelope):
                return to_canonical(policy_envelope_to_wire(obj))
            if isinstance(obj, GovernanceRecordEnvelope):
                return to_canonical(record_envelope_to_wire(obj))
        return {f.name: to_canonical(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        # sort by stringified key for stable ordering
        return {str(k): to_canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [to_canonical(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [to_canonical(x) for x in sorted(obj, key=str)]
    # last resort: a stable string (avoid leaking a dataclass repr by accident)
    return str(obj)


def dumps(obj: Any) -> str:
    """Canonical JSON string: sorted keys, 2-space indent, no NaN/Infinity."""
    return json.dumps(to_canonical(obj), sort_keys=True, indent=2, allow_nan=False)


def dump_snapshot(snapshot: Any, path: "str | Path", *, phase: str) -> None:
    """Write a world snapshot as a schema-versioned, deterministic JSON artifact.

    ``phase`` is ``"initial"`` or ``"final"``. The ``initial`` snapshot is taken
    before kickoff, so inventory is visibly pre-settlement.
    """
    payload = {
        "schema_version": WORLD_SCHEMA_VERSION,
        "phase": phase,
        "tables": to_canonical(snapshot),
    }
    Path(path).write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")


__all__ = ["WORLD_SCHEMA_VERSION", "to_canonical", "dumps", "dump_snapshot"]
