"""Frozen ScenarioSpec eligibility gates for runtime benchmark scorers.

The episode evidence manifest proves that an artifact bundle is internally
consistent.  It does not, by itself, prove that the bundle was produced from
the fixed input for the task being scored.  This module closes that second
boundary by materializing the task's authoritative initial World with the
same production seeding path used by :mod:`episode.runner` and by matching all
declared kickoff envelopes on their canonical wire representation.

These checks are environment eligibility requirements.  They never award or
remove capability credit.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING, Any, Mapping

from episode.capability_runtime import (
    RuntimeBenchmarkIntegrityError,
    canonical_sha256,
)
from episode.scenario import (
    kickoff_envelopes,
    materialize_initial_world_tables,
)
from episode.termination import load_verified_scoreable_termination
from protocol.envelope import to_json

if TYPE_CHECKING:
    from episode.capability_runtime import RuntimeEvidenceBundleV2
    from episode.types import ScenarioSpec


def _wire_kickoffs(scenario: ScenarioSpec) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for envelope in kickoff_envelopes(scenario):
        value = json.loads(to_json(envelope))
        if not isinstance(value, dict):  # pragma: no cover - protocol invariant
            raise RuntimeBenchmarkIntegrityError(
                "frozen ScenarioSpec produced a non-object kickoff envelope"
            )
        rows.append(value)
    return tuple(rows)


def require_frozen_scenario_fixture_v2(
    evidence: RuntimeEvidenceBundleV2,
    scenario: ScenarioSpec,
    *,
    family: str,
) -> None:
    """Require one evidence bundle to match the task's frozen environment.

    The full initial World snapshot is compared exactly.  This includes empty
    authority tables, listing and product identities, ownership, inventory,
    seeded orders, shipments, and logical time.  Declared kickoff inputs are
    joined by ``msg_id`` and compared on canonical wire bytes.  Later model
    actions remain unconstrained here and are left to capability scoring.
    """

    manifest = evidence.evidence_manifest
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("scenario_id") != scenario.scenario_id
    ):
        raise RuntimeBenchmarkIntegrityError(
            f"{family} evidence is not bound to the frozen ScenarioSpec identity"
        )

    observed_tables = evidence.initial_world.get("tables")
    expected_tables = materialize_initial_world_tables(scenario)
    if observed_tables != expected_tables:
        raise RuntimeBenchmarkIntegrityError(
            f"{family} initial World snapshot is not the frozen task fixture "
            f"(expected={canonical_sha256(expected_tables)}, "
            f"observed={canonical_sha256(observed_tables)})"
        )

    expected_kickoffs = _wire_kickoffs(scenario)
    expected_ids = tuple(str(row.get("msg_id", "")) for row in expected_kickoffs)
    if any(not msg_id for msg_id in expected_ids) or len(set(expected_ids)) != len(
        expected_ids
    ):
        raise RuntimeBenchmarkIntegrityError(
            f"{family} frozen ScenarioSpec has invalid kickoff identities"
        )
    expected_id_set = set(expected_ids)
    observed_kickoffs = tuple(
        row
        for row in evidence.envelopes
        if str(row.get("msg_id", "")) in expected_id_set
    )
    observed_ids = tuple(str(row.get("msg_id", "")) for row in observed_kickoffs)
    observed_counts = Counter(observed_ids)
    expected_by_id = dict(zip(expected_ids, expected_kickoffs, strict=True))
    exact_observed_prefix = bool(observed_ids) and all(
        observed_counts[msg_id] == 1 and row == expected_by_id[msg_id]
        for msg_id, row in zip(observed_ids, observed_kickoffs, strict=True)
    ) and observed_ids == expected_ids[: len(observed_ids)]
    if not exact_observed_prefix:
        raise RuntimeBenchmarkIntegrityError(
            f"{family} Runtime kickoff inputs do not match the frozen task fixture"
        )
    if len(observed_ids) != len(expected_ids):
        try:
            termination = load_verified_scoreable_termination(evidence.episode_dir)
        except ValueError as exc:
            raise RuntimeBenchmarkIntegrityError(
                f"{family} incomplete kickoff prefix has an invalid termination binding"
            ) from exc
        if termination is None:
            raise RuntimeBenchmarkIntegrityError(
                f"{family} Runtime omitted a frozen kickoff without a scoreable abort"
            )


__all__ = [
    "require_frozen_scenario_fixture_v2",
]
