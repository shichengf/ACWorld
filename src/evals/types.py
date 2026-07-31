"""Passive data types for the evals package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricResult:
    """One named metric's value and the audit pointer that justifies it.

    ``audit_pointer`` is the index (or msg_id) of the envelope in the audit log
    that the metric was computed against. ``None`` means the metric was computed
    from final world state rather than from any single envelope.
    """
    name: str
    value: bool | int | float | None
    audit_pointer: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class ProbeReport:
    """Aggregate output of one probe run.

    The probe interface is wire-stable; the report shape is additive.
    """
    probe_id: str            # e.g. "order_ablation", "role_ablation"
    n_seeds: int
    per_seed: tuple[dict[str, Any], ...]
    summary: dict[str, Any]  # mean/std per metric
