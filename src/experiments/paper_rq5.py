"""Paper-facing, deterministic analysis of the frozen RQ5 model results.

The module never calls a model and never uses an LLM judge.  It loads the
persisted experiment plan/result records, computes workload-qualified summary
statistics, and emits CSV/SVG assets whose values remain traceable to stable
run keys.  Confidence intervals are descriptive hierarchical bootstrap
intervals over the declared family and seed units; they are not repeated-model
confidence intervals or significance tests, and they do not turn the
single-rollout panel into a universal ranking.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.plan import ExperimentPlan
from experiments.results import ResultStatus, ResultStore


PAPER_RQ5_SCHEMA = "cwe.paper-rq5-results.v1"
PAPER_RQ5_BUNDLE_SCHEMA = "cwe.paper-rq5-bundle.v1"
BOOTSTRAP_METHOD = "hierarchical_family_seed_percentile_v1"
DESCRIPTIVE_INTERVAL_LABEL = "95% descriptive workload bootstrap interval"
S3_CONTRAST_LABEL = "normalized lane contrast (buyer minus merchant; not causal)"
MARKET_METRICS = (
    "market_trade_rate",
    "market_consumer_surplus",
    "market_producer_surplus",
    "market_social_welfare",
    "market_allocative_efficiency",
    "market_inventory_correctness",
    "market_exposure_fairness",
    "market_privacy_leakage_rate",
    "market_protocol_violation_rate",
)

REQUIRED_BUNDLE_ARTIFACTS = frozenset(
    {
        "RQ5_RESULTS.md",
        "buyer_family_heatmap.csv",
        "failure_summary.csv",
        "figure6_buyer_heatmap.svg",
        "figure7_lane_failure.svg",
        "main_model_summary.csv",
        "market_metric_applicability.csv",
        "market_outcomes.csv",
        "rq5-results.json",
        "run_level_results.csv",
        "s3_lane_contrast.csv",
        "self_play_outcomes.csv",
    }
)


@dataclass(frozen=True, slots=True)
class PaperResultRow:
    run_key: str
    suite: str
    model_id: str
    variant_id: str
    seed: int
    evaluated_role: str
    task_family: str | None
    score: float
    task_success: bool
    completed: bool
    truncated: bool
    failure_mode: str
    stop_reason: str | None
    market_metrics: dict[str, bool | float | int | None]


def _require_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def load_paper_rows(
    plan_path: str | Path,
    results_root: str | Path,
    *,
    contract_id: str | None = None,
) -> tuple[PaperResultRow, ...]:
    """Load every planned successful result without silently dropping a row."""

    plan = ExperimentPlan.load(plan_path)
    store = ResultStore(results_root, contract_id=contract_id)
    rows: list[PaperResultRow] = []
    for run in plan.runs:
        result = store.read(run)
        if result is None:
            raise ValueError(f"planned result is missing: {run.run_key}")
        if result.status != ResultStatus.SUCCEEDED:
            raise ValueError(f"planned result did not succeed: {run.run_key}")
        raw_score = result.metrics.get("overall_score")
        score = _require_number(raw_score, label=f"{run.run_key} overall_score")
        if score < 0.0 or score > 1.0:
            raise ValueError(f"{run.run_key} overall_score is outside [0, 1]")
        market_metrics = {
            key: result.metrics.get(key)
            for name in MARKET_METRICS
            for key in (name, f"{name}_applicable")
        }
        rows.append(PaperResultRow(
            run_key=run.run_key,
            suite=run.suite,
            model_id=run.model_id,
            variant_id=run.variant_id,
            seed=run.seed,
            evaluated_role=run.evaluated_role,
            task_family=run.task_family,
            score=score,
            task_success=result.metrics.get("task_success") is True,
            completed=result.completed is True,
            truncated=result.truncated,
            failure_mode=str(result.failure_mode or "unknown"),
            stop_reason=result.stop_reason,
            market_metrics=market_metrics,
        ))
    return tuple(rows)


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return a type-7 linearly interpolated percentile."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if probability < 0.0 or probability > 1.0:
        raise ValueError("percentile probability must be within [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _interval(replicates: Sequence[float]) -> dict[str, float]:
    return {
        "lower": _percentile(replicates, 0.025),
        "upper": _percentile(replicates, 0.975),
        "bootstrap_standard_error": statistics.pstdev(replicates),
    }


def hierarchical_macro_interval(
    family_values: Mapping[str, Sequence[float]],
    *,
    samples: int,
    random_seed: str,
) -> dict[str, Any]:
    """Compute an equal-family macro and deterministic descriptive interval."""

    if isinstance(samples, bool) or samples < 100:
        raise ValueError("bootstrap samples must be an integer >= 100")
    normalized = {
        str(family): tuple(float(value) for value in values)
        for family, values in family_values.items()
    }
    if not normalized or any(not values for values in normalized.values()):
        raise ValueError("every bootstrap family must contain at least one value")
    families = tuple(sorted(normalized))
    observed = statistics.fmean(
        statistics.fmean(normalized[family]) for family in families
    )
    rng = random.Random(random_seed)
    replicates: list[float] = []
    for _ in range(samples):
        selected_families = rng.choices(families, k=len(families))
        selected_means: list[float] = []
        for family in selected_families:
            values = normalized[family]
            seed_sample = rng.choices(values, k=len(values))
            selected_means.append(statistics.fmean(seed_sample))
        replicates.append(statistics.fmean(selected_means))
    return {
        "observed": observed,
        "family_count": len(families),
        "instance_count": sum(len(values) for values in normalized.values()),
        "interval_coverage": 0.95,
        "interval_label": DESCRIPTIVE_INTERVAL_LABEL,
        "estimand": "equal-weight macro over the declared benchmark families",
        "resampling_units": "declared task families, then fixed scenario seeds within family",
        "model_repeat_uncertainty": False,
        "method": BOOTSTRAP_METHOD,
        "samples": samples,
        **_interval(replicates),
    }


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _main_family_values(
    rows: Sequence[PaperResultRow],
    *,
    model_id: str,
    role: str,
) -> dict[str, tuple[float, ...]]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        if (
            row.suite == "main"
            and row.model_id == model_id
            and row.evaluated_role == role
            and row.task_family is not None
        ):
            buckets.setdefault(row.task_family, []).append(row.score)
    return {
        family: tuple(values)
        for family, values in sorted(buckets.items())
    }


def _s3_lane_contrast(
    rows: Sequence[PaperResultRow],
    *,
    model_id: str,
) -> dict[str, Any]:
    buyer: dict[int, float] = {}
    merchant: dict[int, float] = {}
    for row in rows:
        if row.suite != "main" or row.model_id != model_id or row.variant_id != "S3":
            continue
        if row.evaluated_role == "buyer":
            buyer[row.seed] = row.score
        elif row.evaluated_role == "merchant":
            merchant[row.seed] = row.score
    seeds = tuple(sorted(set(buyer) & set(merchant)))
    if not seeds:
        return {
            "variant_id": "S3",
            "available": False,
            "reason": "no paired buyer/merchant seeds",
        }
    differences = tuple(buyer[seed] - merchant[seed] for seed in seeds)
    return {
        "variant_id": "S3",
        "available": True,
        "seeds": list(seeds),
        "buyer_mean": statistics.fmean(buyer[seed] for seed in seeds),
        "merchant_mean": statistics.fmean(merchant[seed] for seed in seeds),
        "contrast_label": S3_CONTRAST_LABEL,
        "buyer_minus_merchant": statistics.fmean(differences),
        "per_seed_differences": list(differences),
        "difference_min": min(differences),
        "difference_max": max(differences),
        "pair_count": len(differences),
        "causal_effect": False,
        "interpretation": (
            "descriptive comparison of two differently defined benchmark lanes with "
            "different deterministic counterpart policies and role-specific objectives"
        ),
    }


def analyze_paper_rq5(
    rows: Sequence[PaperResultRow],
    *,
    bootstrap_samples: int = 10_000,
    random_seed: int = 20_270_715,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete deterministic RQ5 paper summary."""

    if not rows:
        raise ValueError("RQ5 analysis requires at least one row")
    models = _ordered_unique(
        row.model_id for row in rows if row.suite == "main"
    )
    model_summaries: list[dict[str, Any]] = []
    heatmap: list[dict[str, Any]] = []
    lane_contrasts: list[dict[str, Any]] = []
    for model_id in models:
        role_summaries: dict[str, Any] = {}
        for role in ("buyer", "merchant"):
            family_values = _main_family_values(rows, model_id=model_id, role=role)
            if not family_values:
                continue
            summary = hierarchical_macro_interval(
                family_values,
                samples=bootstrap_samples,
                random_seed=f"{random_seed}|{model_id}|{role}",
            )
            summary["family_means"] = {
                family: statistics.fmean(values)
                for family, values in family_values.items()
            }
            summary["family_seed_counts"] = {
                family: len(values) for family, values in family_values.items()
            }
            role_rows = [
                row
                for row in rows
                if row.suite == "main"
                and row.model_id == model_id
                and row.evaluated_role == role
            ]
            summary["strict_task_successes"] = sum(row.task_success for row in role_rows)
            summary["completed"] = sum(row.completed for row in role_rows)
            summary["truncated"] = sum(row.truncated for row in role_rows)
            role_summaries[role] = summary
            if role == "buyer":
                heatmap.append({
                    "model_id": model_id,
                    "family_means": dict(summary["family_means"]),
                })
        main_rows = [
            row for row in rows
            if row.suite == "main" and row.model_id == model_id
        ]
        model_summaries.append({
            "model_id": model_id,
            "coverage": len(main_rows),
            "strict_task_successes": sum(row.task_success for row in main_rows),
            "completed": sum(row.completed for row in main_rows),
            "truncated": sum(row.truncated for row in main_rows),
            "roles": role_summaries,
        })
        lane_contrasts.append({
            "model_id": model_id,
            **_s3_lane_contrast(
                rows,
                model_id=model_id,
            ),
        })

    scores = [row.score for row in rows]
    score_distribution: dict[str, int] = {}
    for score in scores:
        label = format(score, ".12g")
        score_distribution[label] = score_distribution.get(label, 0) + 1
    failure_modes: dict[str, int] = {}
    stop_reasons: dict[str, int] = {}
    for row in rows:
        failure_modes[row.failure_mode] = failure_modes.get(row.failure_mode, 0) + 1
        stop = row.stop_reason or "none"
        stop_reasons[stop] = stop_reasons.get(stop, 0) + 1

    market_rows = [
        {
            "run_key": row.run_key,
            "model_id": row.model_id,
            "variant_id": row.variant_id,
            "task_family": row.task_family,
            "evaluated_role": row.evaluated_role,
            "seed": row.seed,
            "score": row.score,
            "task_success": row.task_success,
            "completed": row.completed,
            "truncated": row.truncated,
            "failure_mode": row.failure_mode,
            "stop_reason": row.stop_reason,
            **row.market_metrics,
        }
        for row in rows if row.suite == "many_to_many"
    ]
    market_metric_applicability: dict[str, dict[str, Any]] = {}
    for metric in MARKET_METRICS:
        applicable_key = f"{metric}_applicable"
        applicable_rows = [
            row for row in market_rows if row.get(applicable_key) is True
        ]
        values_present = sum(row.get(metric) is not None for row in applicable_rows)
        market_metric_applicability[metric] = {
            "applicable": len(applicable_rows),
            "total_market_rows": len(market_rows),
            "not_applicable": len(market_rows) - len(applicable_rows),
            "values_present": values_present,
            "evidence_available": bool(applicable_rows)
            and values_present == len(applicable_rows),
            "full_panel_coverage": bool(market_rows)
            and len(applicable_rows) == len(market_rows)
            and values_present == len(market_rows),
            "headline_supported": bool(market_rows)
            and len(applicable_rows) == len(market_rows)
            and values_present == len(market_rows),
        }
    self_play_rows = [
        {
            "run_key": row.run_key,
            "model_id": row.model_id,
            "variant_id": row.variant_id,
            "seed": row.seed,
            "score": row.score,
            "task_success": row.task_success,
            "completed": row.completed,
            "truncated": row.truncated,
            "failure_mode": row.failure_mode,
            "stop_reason": row.stop_reason,
        }
        for row in rows if row.suite == "self_play"
    ]
    run_rows = [
        {
            "run_key": row.run_key,
            "suite": row.suite,
            "model_id": row.model_id,
            "variant_id": row.variant_id,
            "seed": row.seed,
            "evaluated_role": row.evaluated_role,
            "task_family": row.task_family,
            "score": row.score,
            "task_success": row.task_success,
            "completed": row.completed,
            "truncated": row.truncated,
            "failure_mode": row.failure_mode,
            "stop_reason": row.stop_reason,
        }
        for row in rows
    ]
    return {
        "schema_version": PAPER_RQ5_SCHEMA,
        "source": dict(source or {}),
        "analysis_contract": {
            "headline_judge": "deterministic_python",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_random_seed": random_seed,
            "macro_method": "equal_weight_family_means_after_seed_aggregation",
            "interval_method": BOOTSTRAP_METHOD,
            "interval_label": DESCRIPTIVE_INTERVAL_LABEL,
            "interval_interpretation": (
                "descriptive sensitivity to the declared workload composition; not a "
                "confidence interval for repeated model sampling and not a significance test"
            ),
            "s3_contrast_label": S3_CONTRAST_LABEL,
            "single_rollout": True,
        },
        "coverage": {
            "total": len(rows),
            "main": sum(row.suite == "main" for row in rows),
            "self_play": sum(row.suite == "self_play" for row in rows),
            "many_to_many": sum(row.suite == "many_to_many" for row in rows),
            "models": len(models),
        },
        "model_summaries": model_summaries,
        "buyer_heatmap": heatmap,
        "s3_lane_contrasts": lane_contrasts,
        "self_play": self_play_rows,
        "many_to_many": market_rows,
        "market_metric_applicability": market_metric_applicability,
        "run_rows": run_rows,
        "score_analysis": {
            "distribution": dict(sorted(score_distribution.items(), key=lambda item: float(item[0]))),
            "zero": sum(score == 0.0 for score in scores),
            "strictly_partial": sum(0.0 < score < 1.0 for score in scores),
            "full": sum(score == 1.0 for score in scores),
            "nonzero": sum(score > 0.0 for score in scores),
        },
        "failure_analysis": {
            "failure_modes": dict(sorted(failure_modes.items())),
            "stop_reasons": dict(sorted(stop_reasons.items())),
        },
        "claim_boundary": [
            "Lean Hybrid Core12 is not full S1-S40 model coverage.",
            "Intervals describe the declared family/seed workload and are not pairwise significance tests.",
            "The S3 buyer-minus-merchant value is a normalized lane contrast, not a causal role effect.",
            "Each identity has one rollout; the report does not claim repeated-run reliability.",
            "Self-play and 3x3 markets are bounded three-model demonstrations.",
            *(
                f"No full-panel headline is claimed for {metric}: "
                f"{counts['applicable']}/{len(market_rows)} 3x3 rows marked the metric "
                "applicable; any value is reported only with that denominator."
                for metric, counts in market_metric_applicability.items()
                if counts["applicable"] < len(market_rows)
            ),
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _short_model(model_id: str) -> str:
    value = model_id.split("/", 1)[-1]
    replacements = {
        "gpt-5.6-terra": "GPT",
        "claude-sonnet-5": "Claude",
        "gemini-3.5-flash": "Gemini",
        "deepseek-v4-pro": "DeepSeek",
        "qwen3.7-plus": "Qwen",
        "mistral-medium-3-5": "Mistral",
    }
    return replacements.get(value, value)


def _score_color(score: float) -> str:
    start = (237, 248, 245)
    end = (0, 109, 119)
    bounded = min(1.0, max(0.0, score))
    channels = tuple(round(a + (b - a) * bounded) for a, b in zip(start, end))
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def render_heatmap_svg(report: Mapping[str, Any]) -> str:
    heatmap = list(report["buyer_heatmap"])
    families = [f"T{index}" for index in range(1, 11)]
    cell_w, cell_h = 76, 48
    left, top = 160, 82
    width = left + len(families) * cell_w + 35
    height = top + len(heatmap) * cell_h + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="RQ5 buyer family heatmap">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#18323a}.title{font-size:18px;font-weight:700}'
        '.axis{font-size:12px;font-weight:700}.value{font-size:11px;font-weight:700}</style>',
        '<text class="title" x="20" y="30">Buyer family macro score</text>',
    ]
    for column, family in enumerate(families):
        x = left + column * cell_w + cell_w / 2
        parts.append(f'<text class="axis" x="{x}" y="64" text-anchor="middle">{family}</text>')
    for row_index, item in enumerate(heatmap):
        y = top + row_index * cell_h
        label = html.escape(_short_model(str(item["model_id"])))
        parts.append(
            f'<text class="axis" x="{left - 12}" y="{y + 29}" text-anchor="end">{label}</text>'
        )
        means = item["family_means"]
        for column, family in enumerate(families):
            score = float(means.get(family, 0.0))
            x = left + column * cell_w
            fill = _score_color(score)
            text_fill = "#ffffff" if score >= 0.58 else "#18323a"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" '
                f'fill="{fill}" rx="3"/>'
            )
            parts.append(
                f'<text class="value" x="{x + (cell_w - 2) / 2}" y="{y + 29}" '
                f'text-anchor="middle" style="fill:{text_fill}">{score:.3f}</text>'
            )
    parts.append(
        f'<text x="20" y="{height - 20}" font-size="11">'
        'Deterministic Python scores; declared Lean Hybrid workload; descriptive comparison.</text>'
    )
    parts.append("</svg>")
    return "".join(parts) + "\n"


def render_role_failure_svg(report: Mapping[str, Any]) -> str:
    contrasts = [
        item for item in report["s3_lane_contrasts"] if item.get("available")
    ]
    failures = report["failure_analysis"]["failure_modes"]
    width, height = 1080, 470
    left_x, plot_w = 145, 360
    right_x, right_w = 710, 300
    row_h = 49
    max_failure = max(int(value) for value in failures.values())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="S3 role gaps and failure modes">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#18323a}.title{font-size:18px;font-weight:700}'
        '.axis{font-size:12px;font-weight:700}.small{font-size:11px}</style>',
        '<text class="title" x="20" y="30">Matched S3 role scores</text>',
        '<text class="title" x="610" y="30">Deterministic failure taxonomy</text>',
        f'<line x1="{left_x}" y1="55" x2="{left_x + plot_w}" y2="55" stroke="#9fb5ba"/>',
    ]
    for tick in range(6):
        score = tick / 5
        x = left_x + plot_w * score
        parts.append(f'<text class="small" x="{x}" y="48" text-anchor="middle">{score:.1f}</text>')
    for index, item in enumerate(contrasts):
        y = 82 + index * row_h
        buyer = float(item["buyer_mean"])
        merchant = float(item["merchant_mean"])
        x_buyer = left_x + plot_w * buyer
        x_merchant = left_x + plot_w * merchant
        label = html.escape(_short_model(str(item["model_id"])))
        parts.extend([
            f'<text class="axis" x="{left_x - 12}" y="{y + 5}" text-anchor="end">{label}</text>',
            f'<line x1="{min(x_buyer, x_merchant)}" y1="{y}" x2="{max(x_buyer, x_merchant)}" '
            'y2="{y}" stroke="#7e969c" stroke-width="3"/>',
            f'<circle cx="{x_buyer}" cy="{y}" r="6" fill="#006d77"/>',
            f'<circle cx="{x_merchant}" cy="{y}" r="6" fill="#d97706"/>',
        ])
    failure_items = sorted(failures.items(), key=lambda item: (-int(item[1]), item[0]))
    for index, (name, count) in enumerate(failure_items):
        y = 82 + index * 66
        bar = right_w * int(count) / max_failure
        parts.extend([
            f'<text class="axis" x="{right_x - 12}" y="{y + 5}" text-anchor="end">{html.escape(name)}</text>',
            f'<rect x="{right_x}" y="{y - 13}" width="{bar}" height="24" fill="#006d77" rx="3"/>',
            f'<text class="axis" x="{right_x + bar + 8}" y="{y + 5}">{int(count)}</text>',
        ])
    parts.extend([
        '<circle cx="155" cy="420" r="5" fill="#006d77"/><text class="small" x="166" y="424">buyer</text>',
        '<circle cx="225" cy="420" r="5" fill="#d97706"/><text class="small" x="236" y="424">merchant</text>',
        f'<text class="small" x="610" y="424">Counts cover all '
        f'{report["coverage"]["total"]} successful formal rows.</text>',
        '<text class="small" x="20" y="452">S3 is a normalized lane contrast, not a causal role effect.</text>',
        "</svg>",
    ])
    return "".join(parts) + "\n"


def render_rq5_markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        "# RQ5 Paper Results Bundle",
        "",
        "This report is generated from the frozen experiment plan and persisted results. "
        "All scores are deterministic Python outputs. The 95% descriptive workload bootstrap "
        "intervals measure sensitivity to the declared family/seed workload; they are neither "
        "repeated-model confidence intervals nor significance tests.",
        "",
        "## Coverage",
        "",
        f"- Total: {coverage['total']}",
        f"- Main: {coverage['main']}",
        f"- Self-play: {coverage['self_play']}",
        f"- Bounded 3×3: {coverage['many_to_many']}",
        "",
        "## Main model summary",
        "",
        "| Model | Coverage | Buyer macro [95% descriptive workload interval] | Merchant macro [95% descriptive workload interval] | Strict | Completed | Truncated |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["model_summaries"]:
        buyer = item["roles"]["buyer"]
        merchant = item["roles"]["merchant"]
        lines.append(
            f"| {_short_model(item['model_id'])} | {item['coverage']} | "
            f"{buyer['observed']:.4f} [{buyer['lower']:.4f}, {buyer['upper']:.4f}] | "
            f"{merchant['observed']:.4f} [{merchant['lower']:.4f}, {merchant['upper']:.4f}] | "
            f"{item['strict_task_successes']} | {item['completed']} | {item['truncated']} |"
        )
    score = report["score_analysis"]
    lines.extend([
        "",
        "## 3×3 market metric applicability",
        "",
        "| Metric | Applicable / rows | Values present | Full-panel headline |",
        "|---|---:|---:|---:|",
    ])
    for metric, counts in report["market_metric_applicability"].items():
        lines.append(
            f"| {metric} | {counts['applicable']} / {counts['total_market_rows']} | "
            f"{counts['values_present']} | {'yes' if counts['headline_supported'] else 'no'} |"
        )
    lines.extend([
        "",
        "## S3 normalized lane contrast",
        "",
        "Buyer and merchant rows use different objectives, scorers, and deterministic "
        "counterpart policies. Their difference is reported only as a descriptive lane "
        "contrast, not a causal role effect.",
        "",
        "## Score bands",
        "",
        f"- Zero: {score['zero']}",
        f"- Strictly partial: {score['strictly_partial']}",
        f"- Full: {score['full']}",
        f"- Non-zero: {score['nonzero']}",
        "",
        "## Claim boundary",
        "",
    ])
    lines.extend(f"- {item}" for item in report["claim_boundary"])
    return "\n".join(lines) + "\n"


def write_paper_rq5_bundle(
    report: Mapping[str, Any],
    out_dir: str | Path,
    *,
    source_files: Mapping[str, str | Path] | None = None,
) -> Path:
    """Write one immutable, self-describing RQ5 paper-results directory."""

    output = Path(out_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RQ5 paper bundle: {output}")
    output.mkdir(parents=True)
    report_path = output / "rq5-results.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "RQ5_RESULTS.md").write_text(render_rq5_markdown(report), encoding="utf-8")

    main_rows: list[dict[str, object]] = []
    for item in report["model_summaries"]:
        for role, role_summary in item["roles"].items():
            main_rows.append({
                "model_id": item["model_id"],
                "role": role,
                "model_coverage": item["coverage"],
                "role_coverage": role_summary["instance_count"],
                "macro": role_summary["observed"],
                "workload_interval_95_lower": role_summary["lower"],
                "workload_interval_95_upper": role_summary["upper"],
                "family_count": role_summary["family_count"],
                "instance_count": role_summary["instance_count"],
                "role_strict_task_successes": role_summary["strict_task_successes"],
                "role_completed": role_summary["completed"],
                "role_truncated": role_summary["truncated"],
            })
    _write_csv(
        output / "main_model_summary.csv",
        main_rows,
        (
            "model_id", "role", "model_coverage", "role_coverage", "macro",
            "workload_interval_95_lower", "workload_interval_95_upper", "family_count",
            "instance_count", "role_strict_task_successes", "role_completed",
            "role_truncated",
        ),
    )
    heat_rows: list[dict[str, object]] = []
    for item in report["buyer_heatmap"]:
        heat_rows.append({"model_id": item["model_id"], **item["family_means"]})
    _write_csv(
        output / "buyer_family_heatmap.csv",
        heat_rows,
        ("model_id", *(f"T{index}" for index in range(1, 11))),
    )
    gap_rows: list[dict[str, object]] = []
    for item in report["s3_lane_contrasts"]:
        if not item.get("available"):
            continue
        gap_rows.append({
            "model_id": item["model_id"],
            "buyer_mean": item["buyer_mean"],
            "merchant_mean": item["merchant_mean"],
            "normalized_lane_contrast": item["buyer_minus_merchant"],
            "difference_min": item["difference_min"],
            "difference_max": item["difference_max"],
            "matched_scenario_seeds": item["pair_count"],
            "causal_effect": item["causal_effect"],
        })
    _write_csv(
        output / "s3_lane_contrast.csv",
        gap_rows,
        (
            "model_id", "buyer_mean", "merchant_mean", "normalized_lane_contrast",
            "difference_min", "difference_max", "matched_scenario_seeds", "causal_effect",
        ),
    )
    market_fields = (
        "run_key", "model_id", "variant_id", "task_family", "evaluated_role", "seed",
        "score", "task_success", "completed", "truncated", "failure_mode", "stop_reason",
        *(key for name in MARKET_METRICS for key in (name, f"{name}_applicable")),
    )
    _write_csv(output / "market_outcomes.csv", report["many_to_many"], market_fields)
    _write_csv(
        output / "market_metric_applicability.csv",
        [
            {"metric": metric, **counts}
            for metric, counts in report["market_metric_applicability"].items()
        ],
        (
            "metric", "applicable", "total_market_rows", "not_applicable",
            "values_present", "evidence_available", "full_panel_coverage",
            "headline_supported",
        ),
    )
    _write_csv(
        output / "self_play_outcomes.csv",
        report["self_play"],
        (
            "run_key", "model_id", "variant_id", "seed", "score", "task_success",
            "completed", "truncated", "failure_mode", "stop_reason",
        ),
    )
    failure_rows = [
        {"taxonomy": "failure_mode", "label": label, "count": count}
        for label, count in report["failure_analysis"]["failure_modes"].items()
    ] + [
        {"taxonomy": "stop_reason", "label": label, "count": count}
        for label, count in report["failure_analysis"]["stop_reasons"].items()
    ]
    _write_csv(
        output / "failure_summary.csv",
        failure_rows,
        ("taxonomy", "label", "count"),
    )
    _write_csv(
        output / "run_level_results.csv",
        report["run_rows"],
        (
            "run_key", "suite", "model_id", "variant_id", "seed", "evaluated_role",
            "task_family", "score", "task_success", "completed", "truncated",
            "failure_mode", "stop_reason",
        ),
    )
    (output / "figure6_buyer_heatmap.svg").write_text(
        render_heatmap_svg(report), encoding="utf-8"
    )
    (output / "figure7_lane_failure.svg").write_text(
        render_role_failure_svg(report), encoding="utf-8"
    )

    source_descriptors = []
    for name, raw_path in sorted((source_files or {}).items()):
        path = Path(raw_path)
        source_descriptors.append({
            "name": name,
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "verification_scope": "external source descriptor captured at generation",
        })
    artifact_paths = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "bundle-manifest.json"
    )
    generated_names = {path.name for path in artifact_paths}
    if generated_names != REQUIRED_BUNDLE_ARTIFACTS:
        raise ValueError(
            "RQ5 writer artifact contract mismatch: "
            f"missing={sorted(REQUIRED_BUNDLE_ARTIFACTS - generated_names)}, "
            f"extra={sorted(generated_names - REQUIRED_BUNDLE_ARTIFACTS)}"
        )
    manifest = {
        "schema_version": PAPER_RQ5_BUNDLE_SCHEMA,
        "required_artifacts": sorted(REQUIRED_BUNDLE_ARTIFACTS),
        "external_sources_embedded": False,
        "source_files": source_descriptors,
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        ],
    }
    manifest_path = output / "bundle-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def verify_paper_rq5_bundle(root: str | Path) -> dict[str, Any]:
    """Verify every generated artifact in a relocated RQ5 summary bundle."""

    directory = Path(root)
    manifest_path = directory / "bundle-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read RQ5 bundle manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PAPER_RQ5_BUNDLE_SCHEMA:
        raise ValueError("unsupported RQ5 bundle manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("RQ5 bundle manifest has no artifacts")
    required = manifest.get("required_artifacts")
    if required != sorted(REQUIRED_BUNDLE_ARTIFACTS):
        raise ValueError("RQ5 bundle required-artifact contract mismatch")
    described_names = [
        str(item.get("path", "")) if isinstance(item, dict) else ""
        for item in artifacts
    ]
    if len(described_names) != len(set(described_names)):
        raise ValueError("RQ5 bundle contains duplicate artifact descriptors")
    if set(described_names) != REQUIRED_BUNDLE_ARTIFACTS:
        raise ValueError("RQ5 bundle artifact descriptor set mismatch")
    actual_names = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != "bundle-manifest.json"
    }
    if actual_names != REQUIRED_BUNDLE_ARTIFACTS:
        raise ValueError("RQ5 bundle contains missing or extra artifacts")
    sources = manifest.get("source_files")
    if not isinstance(sources, list):
        raise ValueError("RQ5 external source descriptors are missing")
    source_names: set[str] = set()
    for descriptor in sources:
        if not isinstance(descriptor, dict):
            raise ValueError("RQ5 source descriptor must be an object")
        name = str(descriptor.get("name", ""))
        digest = str(descriptor.get("sha256", ""))
        if not name or name in source_names or len(digest) != 64:
            raise ValueError("RQ5 external source descriptor is invalid")
        source_names.add(name)
    verified = 0
    for descriptor in artifacts:
        if not isinstance(descriptor, dict):
            raise ValueError("RQ5 artifact descriptor must be an object")
        relative = str(descriptor.get("path", ""))
        if not relative or Path(relative).name != relative:
            raise ValueError(f"unsafe RQ5 artifact path: {relative!r}")
        path = directory / relative
        if not path.is_file():
            raise ValueError(f"RQ5 artifact is missing: {relative}")
        if path.stat().st_size != int(descriptor.get("bytes", -1)):
            raise ValueError(f"RQ5 artifact size mismatch: {relative}")
        if _sha256(path) != str(descriptor.get("sha256", "")):
            raise ValueError(f"RQ5 artifact hash mismatch: {relative}")
        verified += 1
    report = json.loads((directory / "rq5-results.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema_version") != PAPER_RQ5_SCHEMA:
        raise ValueError("RQ5 result report has an unsupported schema")
    coverage = report.get("coverage")
    run_rows = report.get("run_rows")
    if (
        not isinstance(coverage, dict)
        or not isinstance(run_rows, list)
        or len(run_rows) != int(coverage.get("total", -1))
    ):
        raise ValueError("RQ5 result coverage does not match run-level rows")
    return {
        "schema_version": "cwe.paper-rq5-bundle-verification.v1",
        "bundle_ok": True,
        "artifacts_verified": verified,
        "coverage": coverage,
        "model_calls": 0,
    }


__all__ = [
    "BOOTSTRAP_METHOD",
    "DESCRIPTIVE_INTERVAL_LABEL",
    "MARKET_METRICS",
    "PAPER_RQ5_BUNDLE_SCHEMA",
    "PAPER_RQ5_SCHEMA",
    "REQUIRED_BUNDLE_ARTIFACTS",
    "S3_CONTRAST_LABEL",
    "PaperResultRow",
    "analyze_paper_rq5",
    "hierarchical_macro_interval",
    "load_paper_rows",
    "render_heatmap_svg",
    "render_role_failure_svg",
    "render_rq5_markdown",
    "verify_paper_rq5_bundle",
    "write_paper_rq5_bundle",
]
