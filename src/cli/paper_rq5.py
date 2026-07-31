"""Generate publication-facing statistics and visuals from frozen RQ5 results."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from experiments.contract import ExperimentContract, load_contract
from experiments.paper_rq5 import (
    analyze_paper_rq5,
    load_paper_rows,
    verify_paper_rq5_bundle,
    write_paper_rq5_bundle,
)
from experiments.plan import ExperimentPlan
from experiments.results import (
    INVENTORY_SCHEMA,
    ResultStatus,
    ResultStore,
    RunResult,
)


RESULT_SET_SCHEMA = "cwe.paper-rq5-result-set.v1"
PROVENANCE_SCHEMA = "cwe.paper-rq5-provenance.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acworld-paper-rq5", description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--formal-summary", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path, metavar="BUNDLE")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_270_715)
    parser.add_argument("--expected-total", type=int, default=231)
    parser.add_argument("--expected-main", type=int, default=216)
    parser.add_argument("--expected-self-play", type=int, default=3)
    parser.add_argument("--expected-many-to-many", type=int, default=12)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit development output from a dirty worktree; never use for paper evidence",
    )
    return parser


def _git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _read_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return raw


def _sha256_file(path: Path) -> str:
    """Use the contract command's byte-for-byte plan/file digest semantics."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _portable_source_label(path: Path, *, repo_root: Path, logical_name: str) -> str:
    """Return a relocatable label without disclosing a host absolute path."""

    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return f"external/{logical_name}/{resolved.name}"


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_inventory(
    inventory_path: Path,
    *,
    plan: ExperimentPlan,
    contract: ExperimentContract,
    results: Mapping[str, RunResult],
) -> dict[str, Any]:
    inventory = _read_object(inventory_path)
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("inventory has an unsupported schema")
    if inventory.get("manifest_id") != plan.manifest_id:
        raise ValueError("inventory manifest_id does not match the plan")
    if inventory.get("contract_id") != contract.contract_id:
        raise ValueError("inventory contract_id does not match the experiment contract")
    if inventory.get("planned_runs") != len(plan.runs):
        raise ValueError("inventory planned_runs does not match the plan")

    raw_entries = inventory.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("inventory entries must be a list")
    entries: dict[str, Mapping[str, Any]] = {}
    for raw_entry in raw_entries:
        entry = _require_mapping(raw_entry, label="inventory entry")
        run_key = str(entry.get("run_key", ""))
        if not run_key or run_key in entries:
            raise ValueError("inventory contains an empty or duplicate run_key")
        entries[run_key] = entry
    planned_keys = {run.run_key for run in plan.runs}
    if set(entries) != planned_keys:
        raise ValueError("inventory run keys do not exactly match the plan")

    for run_key, entry in entries.items():
        result = results[run_key]
        if entry.get("status") != ResultStatus.SUCCEEDED.value:
            raise ValueError(f"inventory result is not succeeded: {run_key}")
        if entry.get("truncated") is not result.truncated:
            raise ValueError(f"inventory truncated flag mismatch: {run_key}")
        resource_limited = result.metrics.get("resource_limited") is True
        if entry.get("resource_limited") is not resource_limited:
            raise ValueError(f"inventory resource_limited flag mismatch: {run_key}")
        if entry.get("attempt_count") != result.attempt_count:
            raise ValueError(f"inventory attempt_count mismatch: {run_key}")

    expected_counts = {
        "pending": 0,
        "succeeded": len(plan.runs),
        "failed": 0,
        "invalid": 0,
    }
    if inventory.get("counts") != expected_counts:
        raise ValueError(
            f"inventory counts mismatch: expected={expected_counts}, "
            f"observed={inventory.get('counts')!r}"
        )
    expected_outcomes = {
        "truncated": sum(result.truncated for result in results.values()),
        "resource_limited": sum(
            result.metrics.get("resource_limited") is True
            for result in results.values()
        ),
    }
    if inventory.get("outcome_counts") != expected_outcomes:
        raise ValueError("inventory outcome_counts do not match result records")
    return inventory


def _load_result_set(
    plan: ExperimentPlan,
    results_root: Path,
    *,
    contract: ExperimentContract,
    plan_sha256: str,
) -> tuple[dict[str, RunResult], dict[str, Any]]:
    """Validate and digest exactly one successful ``result.json`` per planned run."""

    store = ResultStore(results_root, contract_id=contract.contract_id)
    runs_root = results_root / "runs"
    if not runs_root.is_dir():
        raise ValueError("results root has no runs directory")
    discovered: dict[str, Path] = {}
    for path in sorted(runs_root.rglob("result.json")):
        relative = path.relative_to(runs_root)
        if len(relative.parts) != 2 or relative.name != "result.json":
            raise ValueError(f"unexpected result.json location: {relative.as_posix()}")
        run_key = relative.parts[0]
        if run_key in discovered:
            raise ValueError(f"duplicate result.json for run: {run_key}")
        discovered[run_key] = path
    planned = {run.run_key: run for run in plan.runs}
    if set(discovered) != set(planned):
        missing = sorted(set(planned) - set(discovered))
        extra = sorted(set(discovered) - set(planned))
        raise ValueError(
            f"result.json run keys do not exactly match the plan: "
            f"missing={missing}, extra={extra}"
        )

    results: dict[str, RunResult] = {}
    files: list[dict[str, Any]] = []
    for run_key in sorted(planned):
        result = store.read(planned[run_key])
        if result is None:
            raise ValueError(f"planned result is missing: {run_key}")
        if result.status != ResultStatus.SUCCEEDED:
            raise ValueError(f"planned result did not succeed: {run_key}")
        path = discovered[run_key]
        results[run_key] = result
        files.append({
            "run_key": run_key,
            "path": path.relative_to(results_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    digest_payload = {
        "schema_version": RESULT_SET_SCHEMA,
        "contract_id": contract.contract_id,
        "plan_sha256": plan_sha256,
        "files": files,
    }
    return results, {
        **digest_payload,
        "count": len(files),
        "results_sha256": _canonical_digest(digest_payload),
    }


def _scope_from_plan(plan: ExperimentPlan) -> dict[str, int]:
    counts = Counter(run.suite for run in plan.runs)
    market_runs = [run for run in plan.runs if run.suite == "many_to_many"]
    if any((run.buyers, run.merchants) != (3, 3) for run in market_runs):
        raise ValueError("formal many_to_many scope contains a non-3x3 run")
    rollout_groups: Counter[tuple[object, ...]] = Counter(
        (
            run.suite,
            run.model_id,
            run.variant_id,
            run.seed,
            run.evaluated_role,
            run.buyers,
            run.merchants,
        )
        for run in plan.runs
    )
    rollout_counts = set(rollout_groups.values())
    if len(rollout_counts) != 1:
        raise ValueError("plan does not have a uniform rollout count per identity")
    return {
        "planned": len(plan.runs),
        "main": counts["main"],
        "self_play": counts["self_play"],
        "many_to_many_3x3": counts["many_to_many"],
        "rollouts_per_identity": next(iter(rollout_counts), 0),
    }


def _outcomes_from_results(results: Mapping[str, RunResult]) -> dict[str, int]:
    return {
        "succeeded": len(results),
        "failed": 0,
        "pending": 0,
        "invalid": 0,
        "completed": sum(result.completed is True for result in results.values()),
        "strict_task_successes": sum(
            result.metrics.get("task_success") is True for result in results.values()
        ),
        "truncated": sum(result.truncated for result in results.values()),
        "resource_limited": sum(
            result.metrics.get("resource_limited") is True
            for result in results.values()
        ),
    }


def _require_subset(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} mismatch: {mismatches}")


def _validate_formal_summary(
    summary_path: Path,
    *,
    plan: ExperimentPlan,
    contract: ExperimentContract,
    plan_sha256: str,
    outcomes: Mapping[str, int],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    summary = _read_object(summary_path)
    if summary.get("schema_version") != "cwe.formal-lean-hybrid-summary.v1":
        raise ValueError("formal summary has an unsupported schema")
    if summary.get("profile") != plan.profile:
        raise ValueError("formal summary profile does not match the plan")
    if summary.get("paper_result_source") is not True:
        raise ValueError("formal summary is not marked as a paper result source")

    model_inputs = [item for item in contract.inputs if item.category == "model_manifest"]
    if len(model_inputs) != 1:
        raise ValueError("contract must freeze exactly one model manifest")
    frozen_expected: dict[str, Any] = {
        "contract_id": contract.contract_id,
        "git_commit": contract.git.commit,
        "git_dirty": contract.git.dirty,
        "plan_sha256": plan_sha256,
        "model_manifest_sha256": model_inputs[0].sha256,
    }
    for key in ("max_model_calls_per_run", "timeout_seconds", "response_format"):
        if key in contract.request_config:
            frozen_expected[key] = contract.request_config[key]
    _require_subset(
        _require_mapping(summary.get("frozen_contract"), label="frozen_contract"),
        frozen_expected,
        label="formal summary frozen_contract",
    )
    _require_subset(
        _require_mapping(summary.get("scope"), label="scope"),
        _scope_from_plan(plan),
        label="formal summary scope",
    )
    _require_subset(
        _require_mapping(summary.get("outcomes"), label="outcomes"),
        outcomes,
        label="formal summary outcomes",
    )
    local_hashes = _require_mapping(
        summary.get("local_artifact_hashes"), label="local_artifact_hashes"
    )
    _require_subset(local_hashes, source_hashes, label="formal summary source hashes")
    return summary


def _validate_rescore(
    path: Path,
    *,
    plan: ExperimentPlan,
    results_root: Path,
    results: Mapping[str, RunResult],
) -> dict[str, Any]:
    raw = _read_object(path)
    if raw.get("schema_version") != "cwe.progress-rescore-summary.v1":
        raise ValueError("final rescore has an unsupported schema")
    count = len(plan.runs)
    _require_subset(
        raw,
        {
            "selected_runs": count,
            "rescored_runs": count,
            "skipped_runs": 0,
            "skipped": [],
            "write": False,
        },
        label="final rescore summary",
    )
    raw_runs = raw.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError("final rescore runs must be a list")
    by_key: dict[str, Mapping[str, Any]] = {}
    for item in raw_runs:
        record = _require_mapping(item, label="final rescore run")
        run_key = str(record.get("run_key", ""))
        if not run_key or run_key in by_key:
            raise ValueError("final rescore contains an empty or duplicate run_key")
        by_key[run_key] = record
    if set(by_key) != set(results):
        raise ValueError("final rescore run keys do not exactly match the result set")

    plan_by_key = {run.run_key: run for run in plan.runs}
    audits_verified = 0
    for run_key, result in results.items():
        record = by_key[run_key]
        run = plan_by_key[run_key]
        _require_subset(
            record,
            {
                "model_id": run.model_id,
                "variant_id": run.variant_id,
                "evaluated_role": run.evaluated_role,
                "overall_score": result.metrics.get("overall_score"),
            },
            label=f"final rescore run {run_key}",
        )
        audit_relative = result.artifacts.get("audit")
        if not audit_relative:
            raise ValueError(f"result has no audit artifact for rescore: {run_key}")
        run_dir = (results_root / "runs" / run_key).resolve(strict=True)
        audit_path = (run_dir / audit_relative).resolve(strict=True)
        try:
            audit_path.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError(f"unsafe audit artifact path: {run_key}") from exc
        if record.get("audit_sha256") != _sha256_file(audit_path):
            raise ValueError(f"final rescore audit hash mismatch: {run_key}")
        audits_verified += 1
    return {
        "available": True,
        "verification_scope": "all planned run keys, stored scores, and audit hashes",
        "runs_verified": len(results),
        "audits_verified": audits_verified,
        "sha256": _sha256_file(path),
    }


def _validate_replay(
    path: Path,
    *,
    contract: ExperimentContract,
    formal_summary: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _read_object(path)
    _require_subset(
        raw,
        {
            "schema_version": "cwe.formal-replay-summary.v1",
            "contract_id": contract.contract_id,
            "strict": True,
            "replay_ok": True,
            "failed_replays": 0,
        },
        label="final replay summary",
    )
    verification = _require_mapping(
        formal_summary.get("verification"), label="formal summary verification"
    )
    _require_subset(
        verification,
        {
            "strict_replay_ok": raw.get("replay_ok"),
            "attempts_replayed": raw.get("attempts_replayed"),
            "events_verified": raw.get("events_verified"),
            "transactions_replayed": raw.get("transactions_replayed"),
            "failed_replays": raw.get("failed_replays"),
        },
        label="formal summary replay verification",
    )
    return {
        "available": True,
        "verification_scope": "aggregate contract-level replay summary",
        "run_level_attribution": False,
        "boundary": (
            "the replay summary contains no run-key list, so it supports aggregate "
            "contract integrity rather than per-result replay attribution"
        ),
        "attempts_replayed": raw.get("attempts_replayed"),
        "events_verified": raw.get("events_verified"),
        "transactions_replayed": raw.get("transactions_replayed"),
        "sha256": _sha256_file(path),
    }


def _cross_checks(
    contract_path: Path,
    *,
    plan: ExperimentPlan,
    contract: ExperimentContract,
    results_root: Path,
    results: Mapping[str, RunResult],
    formal_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Path]]:
    directory = contract_path.parent
    rescore_path = directory / "final-rescore.json"
    replay_path = directory / "final-replay-summary.json"
    source_files: dict[str, Path] = {}
    if rescore_path.is_file():
        rescore = _validate_rescore(
            rescore_path,
            plan=plan,
            results_root=results_root,
            results=results,
        )
        source_files["final_rescore"] = rescore_path
    else:
        rescore = {
            "available": False,
            "boundary": "no final-rescore.json was supplied beside the contract",
        }
    if replay_path.is_file():
        replay = _validate_replay(
            replay_path,
            contract=contract,
            formal_summary=formal_summary,
        )
        source_files["final_replay_summary"] = replay_path
    else:
        replay = {
            "available": False,
            "boundary": "no final-replay-summary.json was supplied beside the contract",
        }
    return {"rescore": rescore, "replay": replay}, source_files


def _assert_portable_provenance(value: object) -> None:
    """Defence in depth: provenance strings may not contain absolute paths."""

    if isinstance(value, Mapping):
        for item in value.values():
            _assert_portable_provenance(item)
    elif isinstance(value, list):
        for item in value:
            _assert_portable_provenance(item)
    elif isinstance(value, str) and Path(value).is_absolute():
        raise ValueError("RQ5 provenance contains an absolute path")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verify is not None:
        print(json.dumps(verify_paper_rq5_bundle(args.verify), sort_keys=True))
        return 0
    required = {
        "--plan": args.plan,
        "--results": args.results,
        "--contract": args.contract,
        "--inventory": args.inventory,
        "--formal-summary": args.formal_summary,
        "--out": args.out,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"generation requires: {', '.join(missing)}")
    assert args.plan is not None
    assert args.results is not None
    assert args.contract is not None
    assert args.inventory is not None
    assert args.formal_summary is not None
    assert args.out is not None
    revision = _git("rev-parse", "HEAD")
    repo_root = Path(_git("rev-parse", "--show-toplevel")).resolve(strict=True)
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        raise RuntimeError("paper RQ5 bundle requires a clean Git worktree")

    plan_path = args.plan.resolve(strict=True)
    results_root = args.results.resolve(strict=True)
    contract_path = args.contract.resolve(strict=True)
    inventory_path = args.inventory.resolve(strict=True)
    summary_path = args.formal_summary.resolve(strict=True)
    plan = ExperimentPlan.load(plan_path)
    contract = load_contract(contract_path)
    plan_sha256 = _sha256_file(plan_path)
    if plan_sha256 != contract.plan_sha256:
        raise ValueError(
            "experiment plan sha256 does not match the frozen contract: "
            f"recorded={contract.plan_sha256}, observed={plan_sha256}"
        )
    results, result_set = _load_result_set(
        plan,
        results_root,
        contract=contract,
        plan_sha256=plan_sha256,
    )
    _validate_inventory(
        inventory_path,
        plan=plan,
        contract=contract,
        results=results,
    )
    outcomes = _outcomes_from_results(results)
    source_hashes = {
        "experiment_contract_json": _sha256_file(contract_path),
        "plan_json": plan_sha256,
        "final_inventory_json": _sha256_file(inventory_path),
    }
    formal_summary = _validate_formal_summary(
        summary_path,
        plan=plan,
        contract=contract,
        plan_sha256=plan_sha256,
        outcomes=outcomes,
        source_hashes=source_hashes,
    )
    cross_checks, optional_sources = _cross_checks(
        contract_path,
        plan=plan,
        contract=contract,
        results_root=results_root,
        results=results,
        formal_summary=formal_summary,
    )
    local_hashes = _require_mapping(
        formal_summary.get("local_artifact_hashes"), label="local_artifact_hashes"
    )
    optional_expected = {
        "final_rescore_json": cross_checks["rescore"].get("sha256"),
        "final_replay_summary_json": cross_checks["replay"].get("sha256"),
    }
    _require_subset(
        local_hashes,
        {key: value for key, value in optional_expected.items() if value is not None},
        label="formal summary verification artifact hashes",
    )

    rows = load_paper_rows(plan_path, results_root, contract_id=contract.contract_id)
    source = {
        "schema_version": PROVENANCE_SCHEMA,
        "analysis_git_revision": revision,
        "analysis_git_dirty": dirty,
        "experiment_git_revision": contract.git.commit,
        "experiment_git_dirty": contract.git.dirty,
        "contract": {
            "contract_id": contract.contract_id,
            "plan_sha256": contract.plan_sha256,
            "path": _portable_source_label(
                contract_path, repo_root=repo_root, logical_name="contract"
            ),
            "sha256": _sha256_file(contract_path),
        },
        "plan": {
            "path": _portable_source_label(
                plan_path, repo_root=repo_root, logical_name="plan"
            ),
            "manifest_id": plan.manifest_id,
            "profile": plan.profile,
            "planned_runs": len(plan.runs),
            "sha256": plan_sha256,
        },
        "inventory": {
            "path": _portable_source_label(
                inventory_path, repo_root=repo_root, logical_name="inventory"
            ),
            "sha256": _sha256_file(inventory_path),
        },
        "formal_summary": {
            "path": _portable_source_label(
                summary_path, repo_root=repo_root, logical_name="formal_summary"
            ),
            "sha256": _sha256_file(summary_path),
        },
        "results": {
            "root": _portable_source_label(
                results_root, repo_root=repo_root, logical_name="results"
            ),
            **result_set,
        },
        "cross_checks": cross_checks,
        "model_calls": 0,
    }
    _assert_portable_provenance(source)
    report = analyze_paper_rq5(
        rows,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.bootstrap_seed,
        source=source,
    )
    coverage = report["coverage"]
    expected = {
        "total": args.expected_total,
        "main": args.expected_main,
        "self_play": args.expected_self_play,
        "many_to_many": args.expected_many_to_many,
    }
    observed = {name: int(coverage[name]) for name in expected}
    if observed != expected:
        raise ValueError(f"RQ5 coverage mismatch: expected={expected}, observed={observed}")
    source_files = {
        "experiment_contract": contract_path,
        "formal_summary": summary_path,
        "inventory": inventory_path,
        "plan": plan_path,
        **optional_sources,
    }
    manifest = write_paper_rq5_bundle(
        report,
        args.out,
        source_files=source_files,
    )
    print(json.dumps({
        "schema_version": "cwe.paper-rq5-cli-result.v1",
        "bundle": str(args.out),
        "manifest": str(manifest),
        "coverage": coverage,
        "git_revision": revision,
        "git_dirty": dirty,
        "results_sha256": result_set["results_sha256"],
        "model_calls": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
