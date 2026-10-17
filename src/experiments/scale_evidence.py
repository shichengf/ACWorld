"""Build deterministic paper evidence from a complete scale-probe suite.

The generator is intentionally downstream of the raw-bundle verifier.  It
does not trust the summary rows in ``scale-suite.json`` until every manifest
hash has been checked and every event stream has rebuilt an independent World
database.  The resulting JSON and Markdown contain only stable measurements
from the frozen suite; verifier wall time is excluded because it belongs to
the verification host rather than the measured run.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from episode.replay import ReplayVerificationError, verify_scale_artifact_bundle
from episode.scale import (
    SCALE_PROVENANCE_SCHEMA,
    SCALE_SUITE_SCHEMA,
    scale_file_descriptor,
)


SCALE_PAPER_EVIDENCE_SCHEMA = "cwe.scale-paper-evidence.v1"
PAPER_SIZES = (10, 100, 1000)
PAPER_SEEDS = (42, 43, 44)
PAPER_TOP_K = 5

_METRICS = (
    ("registration_seconds", "seconds"),
    ("query_seconds", "seconds"),
    ("settlement_seconds", "seconds"),
    ("replay_seconds", "seconds"),
    ("peak_memory_bytes", "bytes"),
    ("database_bytes", "bytes"),
    ("event_log_bytes", "bytes"),
    ("events_recorded", "events"),
)


class ScaleEvidenceError(ValueError):
    """Raised when a verified suite cannot be summarized as paper evidence."""


def nearest_rank(values: Sequence[int | float], percentile: float) -> int | float:
    """Return the nearest-rank percentile ``x[ceil(p*n)]``.

    This definition is deliberately simple for the three-seed paper suite:
    p50 is the middle observation and p95/p99 are the maximum observation.
    """

    if not values:
        raise ValueError("nearest-rank percentile requires at least one value")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in the interval (0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _stable_number(value: int | float) -> int | float:
    if isinstance(value, int):
        return value
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0 else rounded


def summarize_values(values: Sequence[int | float]) -> dict[str, int | float]:
    """Return the frozen across-run statistics used in the paper tables."""

    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": _stable_number(math.fsum(values) / len(values)),
        "median": _stable_number(statistics.median(values)),
        "p50": _stable_number(nearest_rank(values, 0.50)),
        "p95": _stable_number(nearest_rank(values, 0.95)),
        "p99": _stable_number(nearest_rank(values, 0.99)),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ScaleEvidenceError(f"missing scale suite: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ScaleEvidenceError(f"invalid scale suite JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ScaleEvidenceError("scale suite must be a JSON object")
    return value


def _suite_path(target: str | Path) -> Path:
    path = Path(target)
    if path.is_dir():
        path = path / "scale-suite.json"
    if path.name != "scale-suite.json" or not path.is_file():
        raise ScaleEvidenceError(
            "paper scale evidence requires a complete suite with scale-suite.json"
        )
    return path


def _numeric(report: dict[str, Any], name: str) -> int | float:
    value = report.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ScaleEvidenceError(f"scale report has invalid non-negative metric {name!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ScaleEvidenceError(f"scale report has non-finite metric {name!r}")
    return value


def _event_count_from_raw_suite(
    suite_root: Path,
    manifest_entry: dict[str, Any],
) -> int:
    descriptor = manifest_entry.get("manifest")
    raw_path = descriptor.get("path") if isinstance(descriptor, dict) else None
    if not isinstance(raw_path, str):
        raise ScaleEvidenceError("suite run manifest has no path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ScaleEvidenceError("suite run manifest has an unsafe path")
    event_log = suite_root / relative.parent / "events.jsonl"
    try:
        return sum(1 for line in event_log.read_bytes().splitlines() if line.strip())
    except OSError as exc:
        raise ScaleEvidenceError(f"cannot read verified event log {event_log}") from exc


def _reports_with_event_counts(
    suite: dict[str, Any],
    suite_root: Path,
) -> list[dict[str, Any]]:
    reports = suite.get("runs")
    manifests = suite.get("run_manifests")
    if not isinstance(reports, list) or not isinstance(manifests, list):
        raise ScaleEvidenceError("scale suite has no reports or run manifests")
    if len(reports) != len(manifests):
        raise ScaleEvidenceError("scale suite report/manifest counts differ")
    normalized: list[dict[str, Any]] = []
    for report, manifest in zip(reports, manifests):
        if not isinstance(report, dict) or not isinstance(manifest, dict):
            raise ScaleEvidenceError("scale suite contains a malformed run entry")
        copied = dict(report)
        if "events_recorded" not in copied:
            copied["events_recorded"] = _event_count_from_raw_suite(suite_root, manifest)
        normalized.append(copied)
    return normalized


def _configuration(report: dict[str, Any]) -> dict[str, int]:
    raw = report.get("config")
    if not isinstance(raw, dict):
        raise ScaleEvidenceError("scale report has no config")
    names = ("buyers", "merchants", "top_k", "seed", "max_transactions_per_buyer")
    config: dict[str, int] = {}
    for name in names:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ScaleEvidenceError(f"scale config has invalid {name!r}")
        config[name] = value
    return config


def _size_summaries(reports: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        config = _configuration(report)
        grouped[(config["buyers"], config["merchants"])].append(report)

    summaries: list[dict[str, Any]] = []
    for (buyers, merchants), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: _configuration(row)["seed"])
        seeds = [_configuration(row)["seed"] for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ScaleEvidenceError(f"duplicate seed in {buyers}x{merchants} scale group")
        metrics: dict[str, Any] = {}
        for name, unit in _METRICS:
            values = [_numeric(row, name) for row in rows]
            metrics[name] = {
                "unit": unit,
                "values_by_seed": [
                    {"seed": seed, "value": _stable_number(value)}
                    for seed, value in zip(seeds, values)
                ],
                **summarize_values(values),
            }
        summaries.append(
            {
                "buyers": buyers,
                "merchants": merchants,
                "run_count": len(rows),
                "seeds": seeds,
                "candidate_graph": {
                    "materialized_edges_by_seed": [
                        {
                            "seed": seed,
                            "value": int(_numeric(row, "materialized_edges")),
                        }
                        for seed, row in zip(seeds, rows)
                    ],
                    "cartesian_pairs_reference_by_seed": [
                        {
                            "seed": seed,
                            "value": int(_numeric(row, "cartesian_pairs")),
                        }
                        for seed, row in zip(seeds, rows)
                    ],
                },
                "metrics": metrics,
            }
        )
    return summaries


def _valid_hex_digest(value: Any, lengths: Iterable[int]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in set(lengths)
        and all(character in "0123456789abcdef" for character in value)
    )


def evaluate_scale_promotion_conditions(
    suite: dict[str, Any],
    reports: Sequence[dict[str, Any]],
    *,
    verification_runs: int,
) -> list[dict[str, Any]]:
    configs = [_configuration(report) for report in reports]
    actual_matrix = {
        (
            config["buyers"],
            config["merchants"],
            config["seed"],
            config["top_k"],
            config["max_transactions_per_buyer"],
        )
        for config in configs
    }
    expected_matrix = {
        (size, size, seed, PAPER_TOP_K, 1) for size in PAPER_SIZES for seed in PAPER_SEEDS
    }
    provenance = suite.get("provenance")
    source_revision = provenance.get("source_revision") if isinstance(provenance, dict) else None
    lock = provenance.get("dependency_lock") if isinstance(provenance, dict) else None
    project = provenance.get("project_manifest") if isinstance(provenance, dict) else None
    environment = suite.get("environment")
    workload = suite.get("suite_workload")

    replay_exact = all(
        report.get("replay_ok") is True
        and report.get("state_digest") == report.get("replay_state_digest")
        and report.get("event_digest") == report.get("replay_event_digest")
        for report in reports
    )
    completed = all(
        report.get("completed_transactions") == config["buyers"]
        for report, config in zip(reports, configs)
    )
    sparse = all(
        report.get("materialized_edges") == config["buyers"] * config["top_k"]
        and report.get("cartesian_pairs") == config["buyers"] * config["merchants"]
        and isinstance(report.get("workload"), dict)
        and report["workload"].get("candidate_graph", {}).get("materializes_cartesian_matrix")
        is False
        for report, config in zip(reports, configs)
    )
    headline_rows = [report for report, config in zip(reports, configs) if config["buyers"] == 1000]
    headline_sparse = len(headline_rows) == len(PAPER_SEEDS) and all(
        report.get("materialized_edges") == 5_000 and report.get("cartesian_pairs") == 1_000_000
        for report in headline_rows
    )
    revision_ok = (
        isinstance(provenance, dict)
        and set(provenance)
        == {"schema_version", "source_revision", "dependency_lock", "project_manifest"}
        and provenance.get("schema_version") == SCALE_PROVENANCE_SCHEMA
        and isinstance(source_revision, dict)
        and set(source_revision) == {"vcs", "commit", "dirty"}
        and source_revision.get("vcs") == "git"
        and _valid_hex_digest(source_revision.get("commit"), (40, 64))
        and source_revision.get("dirty") is False
    )
    lock_ok = (
        isinstance(lock, dict)
        and set(lock) == {"path", "bytes", "sha256"}
        and lock.get("path") == "uv.lock"
        and isinstance(lock.get("bytes"), int)
        and lock["bytes"] > 0
        and _valid_hex_digest(lock.get("sha256"), (64,))
    )
    project_ok = (
        isinstance(project, dict)
        and set(project) == {"path", "bytes", "sha256"}
        and project.get("path") == "pyproject.toml"
        and isinstance(project.get("bytes"), int)
        and project["bytes"] > 0
        and _valid_hex_digest(project.get("sha256"), (64,))
    )
    environment_ok = (
        isinstance(environment, dict)
        and all(
            environment.get(name)
            for name in ("python", "implementation", "operating_system", "architecture")
        )
        and isinstance(environment.get("logical_cpus"), int)
    )
    workload_ok = (
        isinstance(workload, dict)
        and workload.get("run_count") == len(expected_matrix)
        and workload.get("all_runs_use_deterministic_scripted_agents") is True
        and workload.get("all_runs_exclude_model_inference") is True
    )

    return [
        {
            "id": "complete_verified_raw_bundle",
            "passed": verification_runs == len(expected_matrix) == len(reports),
            "evidence": f"{verification_runs} independently replayed raw runs",
        },
        {
            "id": "frozen_3x3_matrix",
            "passed": actual_matrix == expected_matrix and len(configs) == len(expected_matrix),
            "evidence": "sizes 10/100/1000 x seeds 42/43/44, top_k=5, max_tx=1",
        },
        {
            "id": "clean_source_revision",
            "passed": revision_ok,
            "evidence": "full Git commit recorded and pre-run worktree dirty=false",
        },
        {
            "id": "dependency_lock_anchored",
            "passed": lock_ok and project_ok,
            "evidence": "uv.lock and pyproject.toml byte lengths/SHA-256 recorded in suite",
        },
        {
            "id": "environment_and_workload_recorded",
            "passed": environment_ok and workload_ok,
            "evidence": "non-identifying environment card and deterministic workload matrix",
        },
        {
            "id": "exact_replay_and_full_settlement",
            "passed": replay_exact and completed,
            "evidence": "state/event digests match and every scripted buyer settles once",
        },
        {
            "id": "sparse_interaction_contract",
            "passed": sparse and headline_sparse,
            "evidence": "1000x1000 materializes 5,000 edges, not 1,000,000 pairs",
        },
    ]


def generate_scale_paper_evidence(
    bundle: str | Path,
    *,
    promote: bool = False,
) -> dict[str, Any]:
    """Verify a complete suite, aggregate it, and evaluate paper promotion."""

    suite_path = _suite_path(bundle)
    try:
        verification = verify_scale_artifact_bundle(suite_path)
    except (ReplayVerificationError, OSError, ValueError) as exc:
        raise ScaleEvidenceError(f"scale suite verification failed: {exc}") from exc
    if verification.bundle_kind != "suite" or not verification.raw_bundle_complete:
        raise ScaleEvidenceError("paper scale evidence requires a complete raw suite")

    suite = _read_json_object(suite_path)
    if suite.get("schema_version") != SCALE_SUITE_SCHEMA:
        raise ScaleEvidenceError("unsupported scale suite schema")
    reports = _reports_with_event_counts(suite, suite_path.parent)
    summaries = _size_summaries(reports)
    conditions = evaluate_scale_promotion_conditions(
        suite,
        reports,
        verification_runs=verification.runs_verified,
    )
    eligible = all(condition["passed"] for condition in conditions)
    thousand = next(
        (summary for summary in summaries if summary["buyers"] == 1000),
        None,
    )
    headline: dict[str, Any] | None = None
    if thousand is not None:
        edges = {row["value"] for row in thousand["candidate_graph"]["materialized_edges_by_seed"]}
        pairs = {
            row["value"] for row in thousand["candidate_graph"]["cartesian_pairs_reference_by_seed"]
        }
        if len(edges) == 1 and len(pairs) == 1:
            edge_count = next(iter(edges))
            pair_count = next(iter(pairs))
            headline = {
                "market": "1000x1000",
                "materialized_edges_per_run": edge_count,
                "cartesian_pairs_for_reference_only": pair_count,
                "cartesian_pairs_materialized": False,
                "materialized_fraction": _stable_number(edge_count / pair_count),
                "unmaterialized_pairs": pair_count - edge_count,
            }

    provenance = suite.get("provenance")
    return {
        "schema_version": SCALE_PAPER_EVIDENCE_SCHEMA,
        "paper_result": bool(promote and eligible),
        "promotion": {
            "requested": promote,
            "eligible": eligible,
            "conditions": conditions,
            "unmet_conditions": [
                condition["id"] for condition in conditions if not condition["passed"]
            ],
            "rule": "paper_result=true only when promotion is explicit and every gate passes",
        },
        "source_bundle": {
            "suite_manifest": scale_file_descriptor(suite_path),
            "raw_bundle_complete": True,
        },
        "verification": {
            "hash_algorithm": "sha256",
            "runs_verified": verification.runs_verified,
            "artifacts_verified": verification.artifacts_verified,
            "artifact_bytes_verified": verification.artifact_bytes_verified,
            "events_verified": verification.events_verified,
            "transactions_replayed": verification.transactions_replayed,
            "independent_replay": True,
            "verifier_wall_time_excluded_from_statistics": True,
        },
        "provenance": provenance,
        "environment": suite.get("environment"),
        "workload": suite.get("suite_workload"),
        "statistical_method": {
            "unit_of_analysis": "one frozen run at one population size and seed",
            "aggregation": "within population size across seeds",
            "seed_count_expected": 3,
            "mean": "arithmetic mean using math.fsum",
            "median": "standard sample median; even n averages the two central values",
            "percentiles": "nearest-rank x[ceil(p*n)] after ascending sort",
            "small_n_interpretation": (
                "with n=3, p50 is the middle observation and p95/p99 are the maximum; "
                "these are descriptive seed summaries, not tail-latency estimates"
            ),
        },
        "headline_sparse_market": headline,
        "size_summaries": summaries,
    }


def _markdown_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def render_scale_paper_evidence_markdown(evidence: dict[str, Any]) -> str:
    """Render the stable human-readable companion to the JSON evidence."""

    promotion = evidence["promotion"]
    provenance = evidence.get("provenance") or {}
    revision = provenance.get("source_revision") or {}
    lock = provenance.get("dependency_lock") or {}
    verification = evidence["verification"]
    status = (
        "promoted paper result"
        if evidence["paper_result"]
        else (
            "eligible; explicit promotion not requested"
            if promotion["eligible"]
            else "not eligible for paper-result promotion"
        )
    )
    lines = [
        "# CommerceWorld RQ4 Scale Evidence",
        "",
        f"**Status:** {status}.",
        "",
        "## Frozen identity",
        "",
        f"- Source revision: `{revision.get('commit')}`",
        f"- Pre-run worktree dirty: `{revision.get('dirty')}`",
        f"- Dependency lock: `{lock.get('path')}` / `{lock.get('sha256')}`",
        f"- Suite SHA-256: `{evidence['source_bundle']['suite_manifest']['sha256']}`",
        "",
        "## Raw verification",
        "",
        f"- Runs independently replayed: {verification['runs_verified']}",
        f"- Manifest-listed artifacts verified: {verification['artifacts_verified']}",
        f"- Artifact bytes verified: {verification['artifact_bytes_verified']}",
        f"- Events verified: {verification['events_verified']}",
        f"- Transactions replayed: {verification['transactions_replayed']}",
        "- Verifier wall time is excluded from the measured-run statistics.",
        "",
        "## Statistical protocol",
        "",
        (
            "For each population size, statistics are computed across the three frozen "
            "seeds. Mean is the arithmetic mean; median is the standard sample median. "
            "Percentiles use nearest rank, `x[ceil(p*n)]`, after ascending sort. With "
            "n=3, p50 is the middle value and p95/p99 are the maximum; therefore the "
            "latter are descriptive seed summaries, not tail-latency estimates."
        ),
        "",
        "## Scale summaries",
        "",
        "| Market | Seeds | Metric | Unit | Mean | Median | P50 | P95 | P99 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in evidence["size_summaries"]:
        market = f"{summary['buyers']}x{summary['merchants']}"
        seeds = ",".join(str(seed) for seed in summary["seeds"])
        for metric, values in summary["metrics"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        market,
                        seeds,
                        metric,
                        values["unit"],
                        _markdown_number(values["mean"]),
                        _markdown_number(values["median"]),
                        _markdown_number(values["p50"]),
                        _markdown_number(values["p95"]),
                        _markdown_number(values["p99"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Sparse-interaction evidence",
            "",
            "| Market | Materialized edges/run | Cartesian pairs (reference only) | Fraction |",
            "|---|---:|---:|---:|",
        ]
    )
    headline = evidence.get("headline_sparse_market")
    if headline is not None:
        lines.append(
            f"| {headline['market']} | {headline['materialized_edges_per_run']} | "
            f"{headline['cartesian_pairs_for_reference_only']} | "
            f"{_markdown_number(headline['materialized_fraction'])} |"
        )
    else:
        lines.append("| 1000x1000 | not present | not present | n/a |")

    lines.extend(
        [
            "",
            "## Paper-result promotion gates",
            "",
            "| Gate | Passed | Evidence |",
            "|---|:---:|---|",
        ]
    )
    for condition in promotion["conditions"]:
        lines.append(
            f"| `{condition['id']}` | {'yes' if condition['passed'] else 'no'} | "
            f"{condition['evidence']} |"
        )
    lines.extend(
        [
            "",
            (
                "The raw `scale-suite.json` remains an engineering artifact with "
                "`paper_result=false`. This derived evidence record is promoted only after an "
                "explicit request and successful completion of every gate above."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_scale_paper_evidence(
    evidence: dict[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write a JSON/Markdown pair without silently replacing frozen evidence."""

    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    occupied = [path for path in (json_target, markdown_target) if path.exists()]
    if occupied:
        raise FileExistsError(f"scale evidence refuses to overwrite artifacts: {occupied}")
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    with json_target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with markdown_target.open("x", encoding="utf-8") as handle:
        handle.write(render_scale_paper_evidence_markdown(evidence))
    return json_target, markdown_target


__all__ = [
    "PAPER_SEEDS",
    "PAPER_SIZES",
    "PAPER_TOP_K",
    "SCALE_PAPER_EVIDENCE_SCHEMA",
    "ScaleEvidenceError",
    "evaluate_scale_promotion_conditions",
    "generate_scale_paper_evidence",
    "nearest_rank",
    "render_scale_paper_evidence_markdown",
    "summarize_values",
    "write_scale_paper_evidence",
]
