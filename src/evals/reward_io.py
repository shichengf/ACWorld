"""Deterministic ``reward.json`` serialization for the step-score report.

Same canonical encoder as the world snapshots (``evals.serialize``): stable key
ordering, no NaN/Infinity, explicit null/N/A, no dataclass repr strings, no
private_context (the report types carry only sanitized evidence refs, not raw
secrets or copied envelopes). Same shape in the HTTP and in-process paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.serialize import to_canonical


def reward_to_dict(report: Any) -> "dict[str, Any]":
    """Canonical JSON-safe dict for a :class:`~evals.step_types.StepScoreReport`.

    A ``StepScoreReport`` is a dataclass, so ``to_canonical`` always yields a
    mapping; the isinstance narrowing keeps the declared return type honest
    (``to_canonical`` is typed ``-> Any``) without an unchecked cast."""
    canonical = to_canonical(report)
    return canonical if isinstance(canonical, dict) else {}


def reward_json(report: Any) -> str:
    """Canonical JSON string (sorted keys, 2-space indent, no NaN/Infinity)."""
    return json.dumps(reward_to_dict(report), sort_keys=True, indent=2, allow_nan=False)


def dump_reward(report: Any, path: "str | Path") -> None:
    """Write ``reward.json`` deterministically."""
    Path(path).write_text(reward_json(report), encoding="utf-8")


__all__ = ["reward_to_dict", "reward_json", "dump_reward"]
