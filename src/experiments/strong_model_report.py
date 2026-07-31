"""Derive all per-model paper statistics from one completed result directory."""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping
import zlib

from experiments.results import ResultStatus


REPORT_SCHEMA = "ACWORLD_PAPER_STATS_V1"
ANALYSIS_SCHEMA = "ACWORLD_PAPER_ANALYSIS_V1"
BOOTSTRAP_SEED = 20_270_725
BOOTSTRAP_REPLICATES = 10_000
FAMILY_ORDER = tuple(f"T{index}" for index in range(1, 11))
FAMILY_LABELS = {
    "T1": "Discovery",
    "T2": "Grounding",
    "T3": "Preference",
    "T4": "Negotiation",
    "T5": "Multi-item",
    "T6": "Inventory",
    "T7": "Lifecycle",
    "T8": "Governance",
    "T9": "Adversarial",
    "T10": "Timing",
}
CATEGORIES = ("evidence", "choice", "execution", "authority")

# This diagnostic grouping is fixed independently of model results. It changes
# no benchmark score; it only assigns each unearned predicate weight to the
# paper's four explanatory categories.
PREDICATE_CATEGORY = {
    "all_source_trust_records_inspected": "evidence",
    "authoritative_record_grounding": "evidence",
    "authority_records_inspected": "evidence",
    "authorization_records_inspected": "evidence",
    "candidate_evidence_coverage": "evidence",
    "cart_evidence_coverage": "evidence",
    "complete_candidate_comparison": "evidence",
    "complete_world_grounding": "evidence",
    "difficulty_evidence_grounded": "evidence",
    "decision_target_grounding": "evidence",
    "evidence_citations": "evidence",
    "negotiation_process_grounding": "evidence",
    "placement_disclosure_coverage": "evidence",
    "requested_evidence_coverage": "evidence",
    "source_trust_records_inspected": "evidence",
    "supply_state_evidence_coverage": "evidence",
    "trusted_signals_used": "evidence",
    "unreliable_signals_discarded": "evidence",
    "verified_social_evidence_used": "evidence",
    "visible_candidate_evidence": "evidence",
    "best_feasible": "choice",
    "best_feasible_listing": "choice",
    "best_listing_within_mandate": "choice",
    "context_aware_listing_optimal": "choice",
    "false_anchor_resistance": "choice",
    "hard_constraint_preservation": "choice",
    "hard_constraints": "choice",
    "hard_constraints_dominate_soft_value": "choice",
    "hard_constraints_preserved": "choice",
    "no_zopa_exit": "choice",
    "price_quality_optimal_choice": "choice",
    "public_hard_constraints": "choice",
    "public_oracle_optimality": "choice",
    "quote_scope_and_quantities": "choice",
    "resolved_case_filtered_optimal_choice": "choice",
    "security_verified_selection": "choice",
    "sponsorship_neutral_price_quality_choice": "choice",
    "strongest_reputation_history_choice": "choice",
    "tier_or_bundle_rule_application": "choice",
    "decision_parameter_correctness": "choice",
    "trust_aware_listing_optimal": "choice",
    "updated_preference_utility_optimal": "choice",
    "verified_review_optimal_choice": "choice",
    "weighted_preference_utility_optimal": "choice",
    "zopa_agreement": "choice",
    "authorized_revisions_applied_in_order": "execution",
    "business_action_completion": "execution",
    "business_event_classification": "execution",
    "campaign_activation": "execution",
    "campaign_published": "execution",
    "charge_and_grand_total_accuracy": "execution",
    "decision_semantics": "execution",
    "decision_action_selection": "execution",
    "evaluated_operations_completed": "execution",
    "negotiation_participation": "execution",
    "offer_acceptance_completed": "execution",
    "operation_selection_coverage": "execution",
    "quote_bound_execution": "execution",
    "remediation_step_coverage": "execution",
    "settlement_decision": "execution",
    "staged_reformulation": "execution",
    "task_decision": "execution",
    "terminal_by_deadline": "execution",
    "economic_boundary": "authority",
    "truthful_semantics": "authority",
    "authenticated_plan_acceptance": "authority",
    "bounded_response_scope": "authority",
    "governing_mandate_preserved": "authority",
    "highest_authority_instruction_governs": "authority",
    "lower_conflicting_instructions_rejected": "authority",
    "no_spurious_context": "authority",
    "policy_safe_rejection": "authority",
    "policy_safe_resolution": "authority",
    "privacy_preservation": "authority",
    "private_value_withheld": "authority",
    "stale_and_duplicate_action_resistance": "authority",
    "unauthorized_revision_rejected": "authority",
    "line_price_and_subtotal_accuracy": "choice",
}


class StrongModelReportError(RuntimeError):
    """A result directory cannot produce a complete paper-facing report."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StrongModelReportError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StrongModelReportError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise StrongModelReportError(f"cannot read JSONL: {path}") from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StrongModelReportError(f"cannot read JSONL: {path}") from exc
        if not isinstance(value, dict):
            raise StrongModelReportError(f"expected JSON object in: {path}")
        rows.append(value)
    return rows


def _percentage(value: float) -> float:
    return round(100.0 * float(value), 6)


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise StrongModelReportError("cannot calculate a percentile without values")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _dominant_deficit(score: Mapping[str, Any]) -> str | None:
    if bool(score.get("strict_success")):
        return None
    masses = {category: 0.0 for category in CATEGORIES}
    checks = score.get("capability_checks")
    if not isinstance(checks, list) or not checks:
        raise StrongModelReportError("task score has no capability checks")
    for check in checks:
        if not isinstance(check, Mapping):
            raise StrongModelReportError("capability check is not an object")
        name = str(check.get("name"))
        category = PREDICATE_CATEGORY.get(name)
        if category is None:
            raise StrongModelReportError(f"unclassified score predicate: {name}")
        masses[category] += float(check["weight"]) * (1.0 - float(check["credit"]))
    maximum = max(masses.values())
    if maximum <= 1e-12:
        raise StrongModelReportError("non-full result has no unearned predicate weight")
    winners = [
        category
        for category, value in masses.items()
        if abs(value - maximum) <= 1e-9
    ]
    return winners[0] if len(winners) == 1 else "mixed"


def _load_records(result_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result_path in sorted((result_root / "runs").glob("*/result.json")):
        result = _read_json(result_path)
        if result.get("status") != ResultStatus.SUCCEEDED.value:
            continue
        run = result.get("run")
        artifacts = result.get("artifacts")
        if not isinstance(run, Mapping) or not isinstance(artifacts, Mapping):
            raise StrongModelReportError(f"malformed result: {result_path}")
        score_relative = artifacts.get("task_score_v3")
        if not isinstance(score_relative, str):
            raise StrongModelReportError(f"result has no task score: {result_path}")
        score = _read_json(result_path.parent / score_relative)
        frozen = float(score["capability_score"])
        metric = float(result.get("metrics", {}).get("capability_score"))
        if abs(frozen - metric) > 1e-9:
            raise StrongModelReportError(
                f"result and task score disagree: {run.get('task_id')}"
            )
        checks = score.get("capability_checks")
        if not isinstance(checks, list) or not checks:
            raise StrongModelReportError(
                f"task score has no checks: {run.get('task_id')}"
            )
        equal = statistics.fmean(float(check["credit"]) for check in checks)
        outcome = (
            "full"
            if bool(score["strict_success"])
            else "zero"
            if frozen <= 1e-12
            else "partial"
        )
        records.append(
            {
                "model_id": str(run["model_id"]),
                "task_id": str(run["task_id"]),
                "family": str(run["task_family"]),
                "role": str(run["evaluated_role"]),
                "score": frozen,
                "equal_score": equal,
                "checks": [
                    {
                        "name": str(check["name"]),
                        "credit": float(check["credit"]),
                        "weight": float(check["weight"]),
                    }
                    for check in checks
                ],
                "outcome": outcome,
                "deficit": _dominant_deficit(score),
                "protocol_error": (
                    result.get("failure_mode") == "protocol"
                    or result.get("stop_reason") == "model_protocol_error"
                ),
                "model_calls": int(
                    result.get("metrics", {}).get("model_call_count", 0)
                ),
                "latency_seconds": float(
                    result.get("metrics", {}).get(
                        "model_latency_seconds", 0.0
                    )
                ),
                "safety_flag": bool(score.get("model_safety_violation")),
                "privacy_flag": bool(score.get("model_privacy_violation")),
            }
        )
    return records


def _merge_task_records(
    base: list[dict[str, Any]],
    override: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    merged = {str(record["task_id"]): record for record in base}
    if len(merged) != len(base):
        raise StrongModelReportError("base results contain duplicate task IDs")
    for record in override or ():
        merged[str(record["task_id"])] = record
    return list(merged.values())


def _process_reward_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    outcome_rows: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    non_full = 0
    non_full_with_signal = 0
    for record in records:
        outcome = str(record["outcome"])
        checks = record["checks"]
        if not isinstance(checks, list) or not checks:
            raise StrongModelReportError("task score has no capability checks")
        positive = sum(float(check["credit"]) > 0.0 for check in checks)
        outcome_rows[outcome].append(
            (
                positive,
                len(checks),
                statistics.fmean(float(check["credit"]) for check in checks),
            )
        )
        if outcome != "full":
            non_full += 1
            non_full_with_signal += int(positive > 0)
    by_outcome = {}
    for outcome in ("full", "partial", "zero"):
        rows = outcome_rows[outcome]
        by_outcome[outcome] = {
            "runs": len(rows),
            "mean_positive_predicates": (
                round(statistics.fmean(row[0] for row in rows), 6)
                if rows
                else 0.0
            ),
            "mean_declared_predicates": (
                round(statistics.fmean(row[1] for row in rows), 6)
                if rows
                else 0.0
            ),
            "mean_unweighted_predicate_credit": (
                round(statistics.fmean(row[2] for row in rows), 6)
                if rows
                else 0.0
            ),
        }
    return {
        "non_full": non_full,
        "non_full_with_positive_predicate_reward": non_full_with_signal,
        "share_with_positive_predicate_reward": (
            round(non_full_with_signal / non_full, 6) if non_full else 0.0
        ),
        "by_outcome": by_outcome,
    }


def build_paper_report(
    result_root: str | Path,
    *,
    override_result_root: str | Path | None = None,
    expected_model_id: str | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Calculate every per-model value currently used by the paper."""

    root = Path(result_root)
    override = (
        _load_records(Path(override_result_root))
        if override_result_root is not None
        else None
    )
    records = _merge_task_records(_load_records(root), override)
    if len(records) != 200:
        raise StrongModelReportError(
            f"paper report requires 200 successful tasks; found {len(records)}"
        )
    task_ids = [str(record["task_id"]) for record in records]
    if len(set(task_ids)) != 200:
        raise StrongModelReportError("paper report contains duplicate task IDs")
    records.sort(key=lambda record: str(record["task_id"]))
    model_ids = {str(record["model_id"]) for record in records}
    if len(model_ids) != 1:
        raise StrongModelReportError(f"expected one model, found {sorted(model_ids)!r}")
    model_id = next(iter(model_ids))
    if expected_model_id is not None and model_id != expected_model_id:
        raise StrongModelReportError(
            f"result model is {model_id!r}, expected {expected_model_id!r}"
        )
    family_counts = Counter(str(record["family"]) for record in records)
    if family_counts != Counter({family: 20 for family in FAMILY_ORDER}):
        raise StrongModelReportError(
            f"expected 20 tasks per family, found {dict(family_counts)!r}"
        )
    role_counts = Counter(str(record["role"]) for record in records)
    if role_counts != Counter({"buyer": 116, "merchant": 84}):
        raise StrongModelReportError(
            f"expected the 116/84 role split, found {dict(role_counts)!r}"
        )

    scores = [float(record["score"]) for record in records]
    equal_scores = [float(record["equal_score"]) for record in records]
    overall = statistics.fmean(scores)
    equal_mean = statistics.fmean(equal_scores)
    outcomes = Counter(str(record["outcome"]) for record in records)
    attribution: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        deficit = record["deficit"]
        if deficit is not None:
            attribution[str(deficit)][str(record["outcome"])] += 1

    randomizer = random.Random(BOOTSTRAP_SEED)
    bootstrap_means = [
        statistics.fmean(randomizer.choice(scores) for _ in scores)
        for _ in range(bootstrap_replicates)
    ]

    families: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        rows = [record for record in records if record["family"] == family]
        family_outcomes = Counter(str(record["outcome"]) for record in rows)
        family_deficits = Counter(
            str(record["deficit"])
            for record in rows
            if record["deficit"] is not None
        )
        families[family] = {
            "label": FAMILY_LABELS[family],
            "mean_pct": _percentage(
                statistics.fmean(float(record["score"]) for record in rows)
            ),
            "full_partial_zero": [
                int(family_outcomes["full"]),
                int(family_outcomes["partial"]),
                int(family_outcomes["zero"]),
            ],
            "deficits": {
                category: int(family_deficits[category])
                for category in (*CATEGORIES, "mixed")
            },
        }

    category_rows = {
        category: {
            "total": int(
                attribution[category]["partial"] + attribution[category]["zero"]
            ),
            "partial_zero": [
                int(attribution[category]["partial"]),
                int(attribution[category]["zero"]),
            ],
        }
        for category in (*CATEGORIES, "mixed")
    }
    report = {
        "schema": REPORT_SCHEMA,
        "model_id": model_id,
        "runs": {
            "successful": len(records),
            "buyer": role_counts["buyer"],
            "merchant": role_counts["merchant"],
        },
        "mean_score_pct": {
            "overall": _percentage(overall),
            "buyer": _percentage(
                statistics.fmean(
                    float(record["score"])
                    for record in records
                    if record["role"] == "buyer"
                )
            ),
            "merchant": _percentage(
                statistics.fmean(
                    float(record["score"])
                    for record in records
                    if record["role"] == "merchant"
                )
            ),
        },
        "full_partial_zero": [
            int(outcomes["full"]),
            int(outcomes["partial"]),
            int(outcomes["zero"]),
        ],
        "families": families,
        "dominant_deficits": category_rows,
        "task_bootstrap_95_pct": [
            _percentage(_percentile(bootstrap_means, 0.025)),
            _percentage(_percentile(bootstrap_means, 0.975)),
        ],
        "equal_predicate_weight": {
            "mean_pct": _percentage(equal_mean),
            "delta_pp": round(100.0 * (equal_mean - overall), 6),
        },
        "process_rewards": _process_reward_summary(records),
        "diagnostics": {
            "protocol_errors": sum(
                bool(record["protocol_error"]) for record in records
            ),
            "model_calls": sum(int(record["model_calls"]) for record in records),
            "mean_latency_seconds": round(
                statistics.fmean(
                    float(record["latency_seconds"]) for record in records
                ),
                6,
            ),
            "safety_flagged_runs": sum(
                bool(record["safety_flag"]) for record in records
            ),
            "privacy_flagged_runs": sum(
                bool(record["privacy_flag"]) for record in records
            ),
        },
        "panel_note": (
            "Restricted-view collision counts are panel-level and must be "
            "recomputed after the final model set is chosen."
        ),
    }
    return report


def transfer_line(report: Mapping[str, Any]) -> str:
    """Return one copy-paste-safe line containing the complete report."""

    if report.get("schema") != REPORT_SCHEMA:
        raise StrongModelReportError("paper report has an unexpected schema")
    return f"{REPORT_SCHEMA} " + json.dumps(
        dict(report),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def write_paper_report(
    result_root: str | Path,
    *,
    override_result_root: str | Path | None = None,
    output_root: str | Path | None = None,
    expected_model_id: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Write readable JSON plus the one-line transfer form."""

    root = Path(result_root)
    destination = Path(output_root) if output_root is not None else root
    destination.mkdir(parents=True, exist_ok=True)
    report = build_paper_report(
        root,
        override_result_root=override_result_root,
        expected_model_id=expected_model_id,
    )
    line = transfer_line(report)
    (destination / "paper-stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "paper-stats.txt").write_text(line + "\n", encoding="utf-8")
    return report, line


def _evaluated_actor_id(
    result_path: Path,
    result: Mapping[str, Any],
) -> str:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise StrongModelReportError(f"result has no artifacts: {result_path}")
    relative = artifacts.get("execution_manifest_v2")
    if not isinstance(relative, str):
        raise StrongModelReportError(
            f"result has no execution manifest: {result_path}"
        )
    manifest = _read_json(result_path.parent / relative)
    controllers = manifest.get("controllers")
    if not isinstance(controllers, list):
        raise StrongModelReportError(
            f"execution manifest has no controllers: {result_path}"
        )
    actors = [
        str(controller["actor_id"])
        for controller in controllers
        if isinstance(controller, Mapping) and controller.get("source") == "model"
    ]
    if len(actors) != 1:
        raise StrongModelReportError(
            f"expected one model-controlled actor: {result_path}"
        )
    return actors[0]


def _vcp_action_views(
    trace_path: Path,
    actor_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    for row in _read_jsonl(trace_path):
        if row.get("agent_id") != actor_id:
            continue
        for step in row.get("steps") or ():
            if not isinstance(step, Mapping) or step.get("kind") != "semantic_action":
                continue
            data = step.get("data")
            if not isinstance(data, Mapping):
                data = {}
            compiled = data.get("compiled_vcp")
            if not isinstance(compiled, Mapping):
                compiled = {}
            actions.append(
                {
                    "business_intent": data.get("business_intent"),
                    "action_kind": compiled.get("action_kind"),
                    "payload_projection": compiled.get("payload_projection"),
                }
            )
    return (actions[-1] if actions else None), actions


def _load_analysis_records(result_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result_path in sorted((result_root / "runs").glob("*/result.json")):
        result = _read_json(result_path)
        if result.get("status") != ResultStatus.SUCCEEDED.value:
            continue
        run = result.get("run")
        artifacts = result.get("artifacts")
        metrics = result.get("metrics")
        if not isinstance(run, Mapping) or not isinstance(artifacts, Mapping):
            raise StrongModelReportError(f"malformed result: {result_path}")
        if not isinstance(metrics, Mapping):
            metrics = {}
        score_relative = artifacts.get("task_score_v3")
        replay_relative = artifacts.get("replay")
        trace_relative = artifacts.get("trace")
        world_relative = artifacts.get("world_final")
        if not all(
            isinstance(value, str)
            for value in (
                score_relative,
                replay_relative,
                trace_relative,
                world_relative,
            )
        ):
            raise StrongModelReportError(
                f"result lacks a paper-analysis artifact: {result_path}"
            )
        score = _read_json(result_path.parent / str(score_relative))
        replay = _read_json(result_path.parent / str(replay_relative))
        if not bool(replay.get("replay_ok")):
            raise StrongModelReportError(f"replay did not pass: {result_path}")
        final_world = _read_json(result_path.parent / str(world_relative))
        actor_id = _evaluated_actor_id(result_path, result)
        final_action, action_sequence = _vcp_action_views(
            result_path.parent / str(trace_relative),
            actor_id,
        )
        checks = score.get("capability_checks")
        if not isinstance(checks, list) or not checks:
            raise StrongModelReportError(
                f"task score has no capability checks: {result_path}"
            )
        records.append(
            {
                "model_id": str(run["model_id"]),
                "task_id": str(run["task_id"]),
                "family": str(run["task_family"]),
                "role": str(run["evaluated_role"]),
                "score": float(score["capability_score"]),
                "strict_success": bool(score["strict_success"]),
                "checks": [
                    {
                        "name": str(check["name"]),
                        "credit": float(check["credit"]),
                        "weight": float(check["weight"]),
                    }
                    for check in checks
                    if isinstance(check, Mapping)
                ],
                "final_world_state": final_world,
                "final_vcp_action": final_action,
                "vcp_action_sequence": action_sequence,
                "protocol_error": (
                    result.get("failure_mode") == "protocol"
                    or result.get("stop_reason") == "model_protocol_error"
                ),
                "model_calls": int(metrics.get("model_call_count", 0)),
                "latency_seconds": float(
                    metrics.get("model_latency_seconds", 0.0)
                ),
                "safety_flag": bool(score.get("model_safety_violation")),
                "privacy_flag": bool(score.get("model_privacy_violation")),
            }
        )
    return records


def build_paper_analysis(
    result_root: str | Path,
    *,
    override_result_root: str | Path | None = None,
    expected_model_id: str | None = None,
) -> dict[str, Any]:
    """Export the per-task evidence needed to recompute every paper analysis."""

    override = (
        _load_analysis_records(Path(override_result_root))
        if override_result_root is not None
        else None
    )
    records = _merge_task_records(
        _load_analysis_records(Path(result_root)),
        override,
    )
    if len(records) != 200:
        raise StrongModelReportError(
            f"paper analysis requires 200 successful tasks; found {len(records)}"
        )
    records.sort(key=lambda record: str(record["task_id"]))
    if len({str(record["task_id"]) for record in records}) != 200:
        raise StrongModelReportError("paper analysis contains duplicate task IDs")
    model_ids = {str(record["model_id"]) for record in records}
    if len(model_ids) != 1:
        raise StrongModelReportError(
            f"expected one model, found {sorted(model_ids)!r}"
        )
    model_id = next(iter(model_ids))
    if expected_model_id is not None and model_id != expected_model_id:
        raise StrongModelReportError(
            f"result model is {model_id!r}, expected {expected_model_id!r}"
        )
    family_counts = Counter(str(record["family"]) for record in records)
    if family_counts != Counter({family: 20 for family in FAMILY_ORDER}):
        raise StrongModelReportError(
            f"expected 20 tasks per family, found {dict(family_counts)!r}"
        )
    return {
        "schema": ANALYSIS_SCHEMA,
        "model_id": model_id,
        "runs": len(records),
        "fields": {
            "scores_and_checks": (
                "rebuilds means, outcomes, failure attribution, process "
                "rewards, bootstrap intervals, and equal-weight sensitivity"
            ),
            "restricted_views": (
                "rebuilds final-state, final-action, and action-sequence "
                "collision analyses across the final model panel"
            ),
            "final_world_state": (
                "the replay-verified state itself, not a release hash or "
                "source contract"
            ),
        },
        "records": records,
    }


def analysis_transfer_line(analysis: Mapping[str, Any]) -> str:
    """Encode the complete analysis export as one copy-paste-safe line."""

    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise StrongModelReportError("paper analysis has an unexpected schema")
    raw = json.dumps(
        dict(analysis),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(zlib.compress(raw, level=9)).decode("ascii")
    return f"{ANALYSIS_SCHEMA} {encoded}"


def decode_analysis_transfer_line(line: str) -> dict[str, Any]:
    """Decode a returned analysis line without reading any run artifacts."""

    prefix = f"{ANALYSIS_SCHEMA} "
    if not line.startswith(prefix):
        raise StrongModelReportError("paper-analysis line has an unexpected prefix")
    try:
        raw = zlib.decompress(base64.urlsafe_b64decode(line[len(prefix) :]))
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError, zlib.error) as exc:
        raise StrongModelReportError("cannot decode paper-analysis line") from exc
    if not isinstance(value, dict) or value.get("schema") != ANALYSIS_SCHEMA:
        raise StrongModelReportError("decoded paper analysis has an invalid schema")
    return value


def write_paper_analysis(
    result_root: str | Path,
    *,
    override_result_root: str | Path | None = None,
    output_root: str | Path | None = None,
    expected_model_id: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Write readable analysis JSON plus a compressed transfer line."""

    root = Path(result_root)
    destination = Path(output_root) if output_root is not None else root
    destination.mkdir(parents=True, exist_ok=True)
    analysis = build_paper_analysis(
        root,
        override_result_root=override_result_root,
        expected_model_id=expected_model_id,
    )
    line = analysis_transfer_line(analysis)
    (destination / "paper-analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "paper-analysis.txt").write_text(line + "\n", encoding="utf-8")
    return analysis, line


__all__ = [
    "ANALYSIS_SCHEMA",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "REPORT_SCHEMA",
    "StrongModelReportError",
    "analysis_transfer_line",
    "build_paper_analysis",
    "build_paper_report",
    "decode_analysis_transfer_line",
    "transfer_line",
    "write_paper_analysis",
    "write_paper_report",
]
