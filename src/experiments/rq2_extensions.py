"""Build and verify the deterministic RQ2 extension case-study evidence bundle.

The suite measures execution and validation time, not unobserved human authoring
time.  It makes no model calls.  Three public extension surfaces are exercised:

* the pet-supplies scenario generator through the scenario CLI and dry-run;
* the round-robin ranking policy in a matched 3x3 PlatformService comparison;
* the flash-restock event/oracle through in-process and HTTP Episode paths.

Run from a clean frozen revision with::

    python -m experiments.rq2_extensions run --out reports/experiments/rq2-extensions
    python -m experiments.rq2_extensions verify --out reports/experiments/rq2-extensions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any


RQ2_EVIDENCE_SCHEMA = "cwe.rq2-extension-evidence.v1"
RQ2_VERIFICATION_SCHEMA = "cwe.rq2-extension-verification.v1"
CORE_MANIFEST_SCHEMA = "cwe.rq2-core-boundary.v1"
SOURCE_INVENTORY_SCHEMA = "cwe.rq2-extension-source-inventory.v1"
DEFAULT_SEEDS: tuple[int, ...] = (42, 1337, 2024)
SUMMARY_FILENAME = "summary.json"

# This is intentionally broader than the minimum paper definition.  The whole
# Episode package is frozen alongside protocol, router/runtime, World, and the
# platform implementation so a case study cannot hide a scheduler change.
CORE_BOUNDARY_PATHS: tuple[str, ...] = (
    "src/protocol",
    "src/runtime",
    "src/world",
    "src/episode",
    "src/agents/platform.py",
)

CASE_SOURCE_FILES: dict[str, dict[str, tuple[str, ...]]] = {
    "pet_supplies_domain": {
        "implementation": ("src/cwe_examples/pet_supplies_domain.py",),
        "tests": (
            "tests/unit/episode/test_extension_examples.py",
            "tests/unit/experiments/test_rq2_extension_evidence.py",
        ),
    },
    "round_robin_ranking": {
        "implementation": ("src/cwe_examples/round_robin_ranking.py",),
        "tests": (
            "tests/unit/episode/test_extension_examples.py",
            "tests/unit/experiments/test_rq2_extension_evidence.py",
        ),
    },
    "flash_restock_event_oracle": {
        "implementation": ("src/cwe_examples/flash_restock_event.py",),
        "execution_substrate": (
            "src/experiments/environment_study.py",
            "src/experiments/multiagent_preflight.py",
            "src/experiments/scripted_channel.py",
        ),
        "tests": (
            "tests/unit/episode/test_extension_runtime.py",
            "tests/integration/test_extension_episode_parity.py",
            "tests/unit/experiments/test_rq2_extension_evidence.py",
        ),
    },
}


class RQ2EvidenceError(RuntimeError):
    """The extension evidence cannot be produced or independently verified."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RQ2EvidenceError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RQ2EvidenceError(f"JSON artifact must be an object: {path}")
    return payload


def _sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _value_digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(body.encode("utf-8"))


def _duration(started: float) -> float:
    return round(perf_counter() - started, 9)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RQ2EvidenceError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def _resolve_ref(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()


def _reference_manifest(repo: Path, revision: str) -> dict[str, Any]:
    output = _git(
        repo,
        "ls-tree",
        "-r",
        "--full-tree",
        revision,
        "--",
        *CORE_BOUNDARY_PATHS,
    )
    files: list[dict[str, str]] = []
    for raw in output.splitlines():
        metadata, separator, path = raw.partition("\t")
        fields = metadata.split()
        if not separator or len(fields) != 3:
            raise RQ2EvidenceError(f"unexpected git ls-tree row: {raw!r}")
        files.append({"path": path, "git_blob": fields[2]})
    files.sort(key=lambda item: item["path"])
    return {
        "files": files,
        "file_count": len(files),
        "manifest_sha256": _value_digest(files),
    }


def _worktree_manifest(repo: Path) -> dict[str, Any]:
    output = _git(
        repo,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        *CORE_BOUNDARY_PATHS,
    )
    files: list[dict[str, Any]] = []
    for raw_path in sorted(set(output.splitlines())):
        path = repo / raw_path
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append(
            {
                "path": raw_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "files": files,
        "file_count": len(files),
        "manifest_sha256": _value_digest(files),
    }


def _diff_stat(repo: Path, baseline: str, evaluated: str) -> dict[str, Any]:
    names = sorted(
        filter(
            None,
            _git(
                repo,
                "diff",
                "--name-only",
                baseline,
                evaluated,
                "--",
                *CORE_BOUNDARY_PATHS,
            ).splitlines(),
        )
    )
    added = 0
    deleted = 0
    binary_files: list[str] = []
    for row in _git(
        repo,
        "diff",
        "--numstat",
        baseline,
        evaluated,
        "--",
        *CORE_BOUNDARY_PATHS,
    ).splitlines():
        raw_added, raw_deleted, path = row.split("\t", 2)
        if raw_added == "-" or raw_deleted == "-":
            binary_files.append(path)
        else:
            added += int(raw_added)
            deleted += int(raw_deleted)
    return {
        "changed_files": names,
        "changed_file_count": len(names),
        "lines_added": added,
        "lines_deleted": deleted,
        "binary_files": sorted(binary_files),
    }


def _worktree_diff(repo: Path, revision: str) -> dict[str, Any]:
    tracked = set(
        filter(
            None,
            _git(
                repo,
                "diff",
                "--name-only",
                revision,
                "--",
                *CORE_BOUNDARY_PATHS,
            ).splitlines(),
        )
    )
    untracked = set(
        filter(
            None,
            _git(
                repo,
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                *CORE_BOUNDARY_PATHS,
            ).splitlines(),
        )
    )
    changed = sorted(tracked | untracked)
    return {
        "changed_files": changed,
        "changed_file_count": len(changed),
        "tracked_changed_files": sorted(tracked),
        "untracked_files": sorted(untracked),
    }


def _manifest_changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    before_files = {row["path"]: row["sha256"] for row in before["files"]}
    after_files = {row["path"]: row["sha256"] for row in after["files"]}
    return sorted(
        path
        for path in before_files.keys() | after_files.keys()
        if before_files.get(path) != after_files.get(path)
    )


def _core_evidence(
    repo: Path,
    *,
    baseline_ref: str,
    evaluated_ref: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    baseline = _resolve_ref(repo, baseline_ref)
    evaluated = _resolve_ref(repo, evaluated_ref)
    historical = _diff_stat(repo, baseline, evaluated)
    worktree = _worktree_diff(repo, evaluated)
    execution_changes = _manifest_changes(before, after)
    headline_changes = sorted(
        set(historical["changed_files"]) | set(worktree["changed_files"]) | set(execution_changes)
    )
    return {
        "schema_version": CORE_MANIFEST_SCHEMA,
        "boundary_paths": list(CORE_BOUNDARY_PATHS),
        "baseline_ref": baseline_ref,
        "baseline_revision": baseline,
        "evaluated_ref": evaluated_ref,
        "evaluated_revision": evaluated,
        "baseline_equals_evaluated": baseline == evaluated,
        "baseline_manifest": _reference_manifest(repo, baseline),
        "evaluated_manifest": _reference_manifest(repo, evaluated),
        "worktree_before_execution": before,
        "worktree_after_execution": after,
        "baseline_to_evaluated": historical,
        "evaluated_to_worktree": worktree,
        "execution_before_to_after": {
            "changed_files": execution_changes,
            "changed_file_count": len(execution_changes),
        },
        "headline_core_changed_files": headline_changes,
        "headline_core_changed_file_count": len(headline_changes),
        "verification_method": {
            "historical_diff": (
                "git diff --name-only <baseline_revision> <evaluated_revision> -- "
                + " ".join(CORE_BOUNDARY_PATHS)
            ),
            "worktree_diff": (
                "git diff --name-only <evaluated_revision> -- " + " ".join(CORE_BOUNDARY_PATHS)
            ),
            "runtime_integrity": "compare SHA-256 worktree manifests before and after suite",
            "interpretation": (
                "zero means no frozen-core source diff across the supplied refs, no "
                "evaluated-revision core dirt, and no core mutation during execution"
            ),
        },
        "authoring_history_claimed": False,
    }


def _file_metrics(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    if not path.is_file():
        raise RQ2EvidenceError(f"declared extension evidence file is missing: {relative}")
    lines = path.read_text(encoding="utf-8").splitlines()
    nonblank = [line for line in lines if line.strip()]
    source = [line for line in nonblank if not line.lstrip().startswith("#")]
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "physical_lines": len(lines),
        "nonblank_lines": len(nonblank),
        "source_lines": len(source),
        "sha256": _sha256_file(path),
    }


def _source_inventory(repo: Path) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    unique_files: set[str] = set()
    for case_name, groups in CASE_SOURCE_FILES.items():
        case: dict[str, Any] = {}
        for group, paths in groups.items():
            metrics = [_file_metrics(repo, path) for path in paths]
            unique_files.update(paths)
            case[group] = {
                "files": metrics,
                "file_count": len(metrics),
                "physical_lines": sum(item["physical_lines"] for item in metrics),
                "source_lines": sum(item["source_lines"] for item in metrics),
            }
        cases[case_name] = case
    return {
        "schema_version": SOURCE_INVENTORY_SCHEMA,
        "line_count_definition": {
            "physical_lines": "Python str.splitlines count",
            "source_lines": "nonblank lines excluding lines whose first token is #",
        },
        "cases": cases,
        "unique_files": [_file_metrics(repo, path) for path in sorted(unique_files)],
    }


def _run_json_command(repo: Path, arguments: list[str]) -> tuple[dict[str, Any], float]:
    env = os.environ.copy()
    python_path = [str(repo / "src"), str(repo)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    started = perf_counter()
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    elapsed = _duration(started)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RQ2EvidenceError(f"command {' '.join(arguments)} failed: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RQ2EvidenceError(f"command {' '.join(arguments)} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RQ2EvidenceError("scenario CLI output must be a JSON object")
    return payload, elapsed


def _normalize_scenario_cli(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(payload)
    generated = normalized.get("generated")
    if isinstance(generated, str):
        normalized["generated"] = Path(generated).name
    scenarios = normalized.get("scenarios")
    if isinstance(scenarios, list):
        for row in scenarios:
            if isinstance(row, dict) and isinstance(row.get("path"), str):
                row["path"] = Path(row["path"]).name
    return normalized


def _domain_discovery(path: Path) -> dict[str, Any]:
    from agents.platform import PlatformService
    from episode.scenario import from_yaml, population_for_scenario, seed_world
    from protocol.envelope import Envelope
    from world import World

    spec = from_yaml(path)
    world = World()
    seed_world(world, spec)
    population = population_for_scenario(spec)
    platform = PlatformService(
        world=world,
        max_search_results=int(population.matching["top_k"]),
    )
    buyer_id = population.buyers[0].buyer_id
    response = platform.handle(
        Envelope(
            msg_id=f"rq2-domain-{spec.seed}",
            ts="2026-07-15T00:00:00Z",
            from_=buyer_id,
            to="platform:aggregator",
            in_reply_to=None,
            idempotency_key=f"rq2-domain-{spec.seed}",
            action={"kind": "commerce.search", "payload": {"query": "pet bed", "limit": 5}},
        )
    )
    if not isinstance(response, Envelope):
        raise RQ2EvidenceError("pet-supplies discovery did not return one Envelope")
    candidates = response.action["payload"]["candidates"]
    snapshot = world.snapshot()
    listings = {str(item.sku_id): item for item in snapshot.catalog}
    inventory = {str(key): value for key, value in snapshot.inventory.items()}
    feasible_skus = []
    for candidate in candidates:
        sku = str(candidate["sku_id"])
        listing = listings[sku]
        row = inventory[sku]
        if (
            listing.attributes.get("washable") is True
            and int(candidate["unit_price"]) <= 6000
            and int(getattr(row, "qty_available", 0)) > 0
        ):
            feasible_skus.append(sku)
    return {
        "scenario_id": spec.scenario_id,
        "buyers": len(population.buyers),
        "merchants": len(population.merchants),
        "listing_count": len(snapshot.catalog),
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible_skus),
        "feasible_skus": sorted(feasible_skus),
        "public_execution_path": "ScenarioSpec→seed_world→PlatformService.handle",
    }


def run_product_domain_case(repo: Path, case_root: Path, seed: int) -> dict[str, Any]:
    """Generate feasible/infeasible pet markets and validate both through the CLI."""

    seed_root = case_root / f"seed-{seed}"
    seed_root.mkdir(parents=True, exist_ok=False)
    variants: dict[str, Any] = {}
    generation_duration = 0.0
    validation_duration = 0.0
    execution_duration = 0.0
    for label, feasible in (("feasible", True), ("infeasible", False)):
        scenario_path = seed_root / f"{label}.yaml"
        params = json.dumps(
            {"buyers": 3, "merchants": 3, "feasible": feasible},
            sort_keys=True,
            separators=(",", ":"),
        )
        generated, generate_seconds = _run_json_command(
            repo,
            [
                "-m",
                "cli.scenario",
                "generate",
                "--plugin",
                "cwe_examples.pet_supplies_domain",
                "--generator",
                "example.pet-supplies",
                "--seed",
                str(seed),
                "--params",
                params,
                "--out",
                str(scenario_path),
            ],
        )
        dry_run, validate_seconds = _run_json_command(
            repo, ["-m", "cli.scenario", "dry-run", str(scenario_path)]
        )
        started = perf_counter()
        discovery = _domain_discovery(scenario_path)
        discover_seconds = _duration(started)
        dry_row = dry_run["scenarios"][0]
        expected_feasible = feasible
        observed_feasible = discovery["feasible_candidate_count"] > 0
        variants[label] = {
            "parameters": {"buyers": 3, "merchants": 3, "feasible": feasible},
            "scenario_path": str(scenario_path.relative_to(case_root.parent.parent)),
            "scenario_sha256": _sha256_file(scenario_path),
            "generation": _normalize_scenario_cli(generated),
            "dry_run": _normalize_scenario_cli(dry_run),
            "discovery": discovery,
            "expected_feasible": expected_feasible,
            "observed_feasible": observed_feasible,
            "passed": (
                dry_row.get("status") == "dry_run_ok" and observed_feasible == expected_feasible
            ),
            "generation_duration_seconds": generate_seconds,
            "validation_duration_seconds": validate_seconds,
            "execution_duration_seconds": discover_seconds,
        }
        generation_duration += generate_seconds
        validation_duration += validate_seconds
        execution_duration += discover_seconds
    result = {
        "case_id": "pet_supplies_domain",
        "seed": seed,
        "variants": variants,
        "passed": all(row["passed"] for row in variants.values()),
        "generation_duration_seconds": round(generation_duration, 9),
        "validation_duration_seconds": round(validation_duration, 9),
        "execution_duration_seconds": round(execution_duration, 9),
        "network_calls": 0,
    }
    _write_json(seed_root / "result.json", result)
    return result


def _commerce_state_digest(world: Any) -> str:
    """Digest operational commerce state while excluding search evidence.

    A Platform search deliberately appends a World-owned ``search_sessions``
    record so discovery can be audited and replayed.  That append is expected
    for both ranking policies and must not be described as a mutation of the
    catalog, inventory, orders, ledger, or other commerce state whose
    preservation this extension case tests.
    """

    from evals.serialize import to_canonical

    state = to_canonical(world.snapshot())
    state.pop("search_sessions", None)
    return _value_digest(state)


def _ranking_world(seed: int) -> Any:
    from world import AgentId, InventoryRow, Listing, Money, SkuId, World

    world = World()
    listings = []
    inventory = {}
    for merchant_index, prefix in enumerate(("a", "b", "c"), start=1):
        merchant_id = AgentId(f"merchant:m{merchant_index}")
        for listing_index in range(3):
            sku = SkuId(f"{prefix}-{seed}-{listing_index}")
            listing = Listing(
                sku_id=sku,
                category="pet-supplies",
                name=f"Pet bed {merchant_index}-{listing_index}",
                attributes={"washable": True},
                list_price=Money(Decimal(str(40 + merchant_index + listing_index))),
                merchant_id=merchant_id,
                product_id="pet-bed",
            )
            listings.append(listing)
            inventory[sku] = InventoryRow(sku, merchant_id, 2)
    world.apply({"catalog": tuple(listings), "inventory": inventory})
    return world


def _ranking_metrics(rankings: list[list[str]]) -> dict[str, Any]:
    exposures = Counter(merchant for ranking in rankings for merchant in ranking)
    total = sum(exposures.values())
    hhi = Fraction(sum(count * count for count in exposures.values()), total * total)
    unique_counts = [len(set(ranking)) for ranking in rankings]
    maximum = max(exposures.values())
    minimum = min(exposures.values())
    return {
        "exposures": dict(sorted(exposures.items())),
        "total_exposures": total,
        "unique_merchants_per_query": unique_counts,
        "mean_unique_merchants": str(Fraction(sum(unique_counts), len(unique_counts))),
        "exposure_hhi": {
            "fraction": f"{hhi.numerator}/{hhi.denominator}",
            "decimal": f"{float(hhi):.6f}",
        },
        "exposure_gap": maximum - minimum,
    }


def _execute_ranking(seed: int, *, round_robin: bool) -> dict[str, Any]:
    from agents.platform import PlatformService
    from protocol.envelope import Envelope

    if round_robin:
        import cwe_examples.round_robin_ranking  # noqa: F401

    world = _ranking_world(seed)
    before = _commerce_state_digest(world)
    policy = (
        {"name": "example.round-robin-ranking", "config": {"limit": 3}} if round_robin else None
    )
    platform = PlatformService(world=world, platform_policy=policy)
    rankings: list[list[str]] = []
    ranked_skus: list[list[str]] = []
    for buyer_index in range(1, 4):
        response = platform.handle(
            Envelope(
                msg_id=f"rq2-ranking-{seed}-{buyer_index}",
                ts="2026-07-15T00:00:00Z",
                from_=f"buyer:b{buyer_index}",
                to="platform:aggregator",
                in_reply_to=None,
                idempotency_key=f"rq2-ranking-{seed}-{buyer_index}",
                action={"kind": "commerce.search", "payload": {"query": "", "limit": 3}},
            )
        )
        if not isinstance(response, Envelope):
            raise RQ2EvidenceError("ranking comparison did not return one Envelope")
        candidates = response.action["payload"]["candidates"]
        rankings.append([str(row["merchant_id"]) for row in candidates])
        ranked_skus.append([str(row["sku_id"]) for row in candidates])
    after = _commerce_state_digest(world)
    return {
        "policy": "example.round-robin-ranking" if round_robin else "builtin_baseline",
        "ranked_merchants": rankings,
        "ranked_skus": ranked_skus,
        "metrics": _ranking_metrics(rankings),
        "commerce_state_before_sha256": before,
        "commerce_state_after_sha256": after,
        "commerce_state_unchanged": before == after,
    }


def run_ranking_case(case_root: Path, seed: int) -> dict[str, Any]:
    """Run one matched 3-buyer x 3-merchant baseline/policy comparison."""

    started = perf_counter()
    baseline = _execute_ranking(seed, round_robin=False)
    round_robin = _execute_ranking(seed, round_robin=True)
    baseline_unique = Fraction(baseline["metrics"]["mean_unique_merchants"])
    policy_unique = Fraction(round_robin["metrics"]["mean_unique_merchants"])
    baseline_hhi = Fraction(baseline["metrics"]["exposure_hhi"]["fraction"])
    policy_hhi = Fraction(round_robin["metrics"]["exposure_hhi"]["fraction"])
    matched = (
        baseline["commerce_state_before_sha256"]
        == round_robin["commerce_state_before_sha256"]
    )
    passed = bool(
        matched
        and baseline["commerce_state_unchanged"]
        and round_robin["commerce_state_unchanged"]
        and policy_unique > baseline_unique
        and policy_hhi < baseline_hhi
    )
    result = {
        "case_id": "round_robin_ranking",
        "seed": seed,
        "market": {"buyers": 3, "merchants": 3, "listings_per_merchant": 3, "top_k": 3},
        "matched_input_state": matched,
        "baseline": baseline,
        "round_robin": round_robin,
        "comparison": {
            "mean_unique_merchant_delta": str(policy_unique - baseline_unique),
            "exposure_hhi_delta": str(policy_hhi - baseline_hhi),
            "round_robin_increases_diversity": policy_unique > baseline_unique,
            "round_robin_reduces_concentration": policy_hhi < baseline_hhi,
        },
        "passed": passed,
        "execution_duration_seconds": _duration(started),
        "network_calls": 0,
        "public_execution_path": "PlatformService→registered PLATFORM_POLICIES callable",
    }
    path = case_root / f"seed-{seed}.json"
    _write_json(path, result)
    return result


def _event_scenario(seed: int) -> Any:
    from dataclasses import replace

    from episode.types import ExtensionEvaluationSpec, WorldEventSpec
    from experiments.multiagent_preflight import build_multiagent_scenario, sku_id

    base = build_multiagent_scenario(2)
    target_sku = sku_id(1)
    early_quantity = 1 + seed % 2
    return replace(
        base,
        scenario_id=f"s1_rq2_flash_restock_{seed}",
        seed=seed,
        # Deliberately declared out of order; the extension runtime must apply
        # logical_time first and event_id second.
        world_events=(
            WorldEventSpec(
                "restock-late",
                "example.flash-restock",
                {"sku_id": target_sku, "quantity": 2},
                2,
            ),
            WorldEventSpec(
                "restock-early",
                "example.flash-restock",
                {"sku_id": target_sku, "quantity": early_quantity},
                1,
            ),
        ),
        extension_evaluations=(
            ExtensionEvaluationSpec(
                "early-delta-correct",
                "oracle_primitive",
                "example.flash-restock-delta",
                {"event_id": "restock-early"},
            ),
            ExtensionEvaluationSpec(
                "late-delta-correct",
                "oracle_primitive",
                "example.flash-restock-delta",
                {"event_id": "restock-late"},
            ),
        ),
    )


def run_event_case(case_root: Path, seed: int) -> dict[str, Any]:
    """Run restock/oracle through both official transports and strict replay."""

    import cwe_examples.flash_restock_event  # noqa: F401
    from episode.replay import verify_episode_replay
    from episode.extensions import ORACLE_PRIMITIVES
    from episode.scenario import population_for_scenario
    from experiments.environment_study import network_disabled
    from experiments.multiagent_preflight import (
        make_multiagent_factory,
        run_multiagent_episode,
        sku_id,
    )
    from runtime.tracker_evidence import verify_all_active_actor_tracker_evidence

    seed_root = case_root / f"seed-{seed}"
    spec = _event_scenario(seed)
    factories = {
        transport: make_multiagent_factory(2)
        for transport in ("in_process", "http_vcp")
    }
    durations: dict[str, float] = {}
    with network_disabled() as network_guard:
        started = perf_counter()
        inprocess_run = run_multiagent_episode(
            scenario=spec,
            factory=factories["in_process"],
            out_root=seed_root / "inprocess",
            transport="in_process",
        )
        durations["in_process"] = _duration(started)

        started = perf_counter()
        http_run = run_multiagent_episode(
            scenario=spec,
            factory=factories["http_vcp"],
            out_root=seed_root / "http",
            transport="http_vcp",
        )
        durations["http_vcp"] = _duration(started)

    inprocess = Path(inprocess_run.evidence.episode_dir)
    http = Path(http_run.evidence.episode_dir)
    population = population_for_scenario(spec)
    buyer_id = population.buyers[0].buyer_id
    tracker_verdicts = []
    for run, factory in (
        (inprocess_run, factories["in_process"]),
        (http_run, factories["http_vcp"]),
    ):
        verdict = verify_all_active_actor_tracker_evidence(
            run.evidence,
            declared_actor_ids=factory.actor_ids(),
            evaluated_actor_id=buyer_id,
            evaluated_actor_strict=True,
        )
        tracker_verdicts.append(verdict)

    inprocess_artifact = _read_json(inprocess / "extensions.json")
    http_artifact = _read_json(http / "extensions.json")
    parity = inprocess_artifact == http_artifact
    event_order = [event["event_id"] for event in inprocess_artifact["world_events"]]
    evaluations = inprocess_artifact["evaluations"]
    ideal_oracles_pass = all(row["result"] is True for row in evaluations)
    oracle = ORACLE_PRIMITIVES.get("example.flash-restock-delta")
    mutations = []
    for event in inprocess_artifact["world_events"]:
        output = event["result"]
        mutation_result = oracle(
            before=int(output["before"]),
            after=int(output["after"]),
            quantity=int(output["quantity"]) + 1,
        )
        mutations.append(
            {
                "event_id": event["event_id"],
                "mutation": "declared_quantity_plus_one",
                "oracle_result": mutation_result,
                "detected": mutation_result is False,
            }
        )
    inprocess_replay = verify_episode_replay(inprocess, strict=True).to_dict()
    http_replay = verify_episode_replay(http, strict=True).to_dict()
    inprocess_replay["target"] = f"$OUT/event-oracle/seed-{seed}/inprocess"
    http_replay["target"] = f"$OUT/event-oracle/seed-{seed}/http"
    final_tables = inprocess_run.evidence.final_world.get("tables")
    if not isinstance(final_tables, dict):
        raise RQ2EvidenceError("extension Episode final World has no tables")
    inventory = final_tables.get("inventory")
    if not isinstance(inventory, dict):
        raise RQ2EvidenceError("extension Episode final World has no inventory")
    target_sku = sku_id(1)
    inventory_row = inventory.get(target_sku)
    if not isinstance(inventory_row, dict):
        raise RQ2EvidenceError("extension Episode final World lost the test listing")
    final_quantity = int(inventory_row["qty_available"])
    event_results = [event["result"] for event in inprocess_artifact["world_events"]]
    expected_quantity = int(event_results[-1]["after"])
    event_chain_ok = all(
        int(result["after"]) == int(result["before"]) + int(result["quantity"])
        for result in event_results
    ) and all(
        int(current["after"]) == int(next_result["before"])
        for current, next_result in pairwise(event_results)
    )
    episode_completed = all(
        not (Path(run.evidence.episode_dir) / "termination.json").exists()
        for run in (inprocess_run, http_run)
    )
    tracker_ok = all(verdict.verified for verdict in tracker_verdicts)
    zero_provider_calls = all(
        factory.provider_calls == 0 for factory in factories.values()
    )
    final_world_parity = inprocess_run.evidence.final_world == http_run.evidence.final_world
    network_guard_result = network_guard.to_dict()
    egress_free = (
        network_guard_result["blocked_connect_count"] == 0
        and network_guard_result["egress_free"] is True
    )
    passed = bool(
        parity
        and final_world_parity
        and event_order == ["restock-early", "restock-late"]
        and ideal_oracles_pass
        and all(item["detected"] for item in mutations)
        and inprocess_replay["replay_ok"]
        and http_replay["replay_ok"]
        and final_quantity == expected_quantity
        and event_chain_ok
        and episode_completed
        and tracker_ok
        and zero_provider_calls
        and egress_free
    )
    result = {
        "case_id": "flash_restock_event_oracle",
        "seed": seed,
        "event_order": event_order,
        "deterministic_event_order_ok": event_order == ["restock-early", "restock-late"],
        "ideal_oracles_pass": ideal_oracles_pass,
        "mutations": mutations,
        "mutation_detection_ok": all(item["detected"] for item in mutations),
        "inprocess_http_extensions_parity": parity,
        "inprocess_http_final_world_parity": final_world_parity,
        "inprocess_extensions_sha256": _sha256_file(inprocess / "extensions.json"),
        "http_extensions_sha256": _sha256_file(http / "extensions.json"),
        "inprocess_replay": inprocess_replay,
        "http_replay": http_replay,
        "episode_completed_without_termination": episode_completed,
        "tracker_causal_closure": tracker_ok,
        "scripted_business_decisions": sum(
            len(factory.channel_for(actor_id).decision_log)
            for factory in factories.values()
            for actor_id in factory.actor_ids()
        ),
        "provider_calls": 0,
        "network_egress_disabled": egress_free,
        "final_inventory_quantity": final_quantity,
        "expected_inventory_quantity": expected_quantity,
        "restock_event_chain_valid": event_chain_ok,
        "inprocess_execution_duration_seconds": durations["in_process"],
        "http_execution_duration_seconds": durations["http_vcp"],
        "execution_duration_seconds": round(sum(durations.values()), 9),
        "network_calls": 0,
        "http_transport": "FastAPI TestClient over the official /vcp path",
        "passed": passed,
    }
    _write_json(seed_root / "result.json", result)
    return result


def _artifact_descriptors(root: Path) -> list[dict[str, Any]]:
    excluded = {SUMMARY_FILENAME}
    descriptors = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        descriptors.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return descriptors


def _aggregate_case(
    case_id: str,
    rows: list[dict[str, Any]],
    *,
    source_inventory: dict[str, Any],
    transport_evidence: dict[str, Any],
) -> dict[str, Any]:
    duration_fields = sorted(
        {key for row in rows for key in row if key.endswith("duration_seconds")}
    )
    return {
        "case_id": case_id,
        "seeds": [row["seed"] for row in rows],
        "runs": rows,
        "run_count": len(rows),
        "passed_runs": sum(bool(row["passed"]) for row in rows),
        "passed": all(bool(row["passed"]) for row in rows),
        "durations_seconds": {
            key: round(sum(float(row.get(key, 0.0)) for row in rows), 9) for key in duration_fields
        },
        "source_inventory": source_inventory,
        "transport_or_equivalent_evidence": transport_evidence,
        "network_calls": 0,
        "llm_judge": False,
    }


def write_rq2_extension_evidence(
    out_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    core_baseline_ref: str = "HEAD",
    evaluated_ref: str = "HEAD",
) -> dict[str, Any]:
    """Write a complete RQ2 bundle without making any model/network call."""

    repo = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    root = Path(out_root).resolve()
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if len(normalized_seeds) < 3 or len(set(normalized_seeds)) != len(normalized_seeds):
        raise RQ2EvidenceError("RQ2 evidence requires at least three distinct seeds")
    if root.exists():
        raise RQ2EvidenceError(f"refusing to overwrite existing RQ2 output: {root}")
    root.mkdir(parents=True, exist_ok=False)

    suite_started = perf_counter()
    core_before = _worktree_manifest(repo)
    inventory = _source_inventory(repo)
    _write_json(root / "extension-source-inventory.json", inventory)

    domain_rows = [
        run_product_domain_case(repo, root / "product-domain", seed) for seed in normalized_seeds
    ]
    ranking_rows = [run_ranking_case(root / "ranking", seed) for seed in normalized_seeds]
    event_rows = [run_event_case(root / "event-oracle", seed) for seed in normalized_seeds]
    core_after = _worktree_manifest(repo)
    core = _core_evidence(
        repo,
        baseline_ref=core_baseline_ref,
        evaluated_ref=evaluated_ref,
        before=core_before,
        after=core_after,
    )
    _write_json(root / "core-boundary-manifest.json", core)

    cases = {
        "pet_supplies_domain": _aggregate_case(
            "pet_supplies_domain",
            domain_rows,
            source_inventory=inventory["cases"]["pet_supplies_domain"],
            transport_evidence={
                "kind": "official_scenario_cli_generation_and_dry_run",
                "execution_path": "cwe-scenario generate→from_yaml→seed_world→Router validation",
                "existing_regression_test": (
                    "tests/unit/cli/test_scenario_cli.py::test_init_validate_and_structural_dry_run"
                ),
            },
        ),
        "round_robin_ranking": _aggregate_case(
            "round_robin_ranking",
            ranking_rows,
            source_inventory=inventory["cases"]["round_robin_ranking"],
            transport_evidence={
                "kind": "official_platform_service_matched_execution",
                "execution_path": "PlatformService.handle→AggregatorPolicy→registered policy",
                "existing_regression_test": (
                    "tests/unit/episode/test_extension_examples.py::"
                    "test_registered_ranking_policy_runs_through_platform_without_core_changes"
                ),
            },
        ),
        "flash_restock_event_oracle": _aggregate_case(
            "flash_restock_event_oracle",
            event_rows,
            source_inventory=inventory["cases"]["flash_restock_event_oracle"],
            transport_evidence={
                "kind": "measured_inprocess_http_artifact_parity",
                "artifact": "extensions.json",
                "existing_regression_test": (
                    "tests/integration/test_extension_episode_parity.py::"
                    "test_inprocess_and_http_write_same_extension_evidence"
                ),
            },
        ),
    }
    core_zero = core["headline_core_changed_file_count"] == 0
    all_cases = all(case["passed"] for case in cases.values())
    pre_verification_artifacts = _artifact_descriptors(root)
    summary: dict[str, Any] = {
        "schema_version": RQ2_EVIDENCE_SCHEMA,
        "evaluated_revision": core["evaluated_revision"],
        "core_baseline_revision": core["baseline_revision"],
        "seeds": list(normalized_seeds),
        "cases": cases,
        "case_count": len(cases),
        "network_calls": 0,
        "model_inference": False,
        "llm_judge": False,
        "timing_scope": {
            "authoring_time_reported": False,
            "reason": "historical human authoring time was not prospectively observed",
            "reported_measurements": [
                "automated_generation_duration_seconds",
                "automated_validation_duration_seconds",
                "automated_execution_duration_seconds",
            ],
        },
        "core_boundary": {
            "path": "core-boundary-manifest.json",
            "sha256": _sha256_file(root / "core-boundary-manifest.json"),
            "manifest_sha256": core["evaluated_manifest"]["manifest_sha256"],
            "changed_files": core["headline_core_changed_files"],
            "changed_file_count": core["headline_core_changed_file_count"],
            "method": core["verification_method"],
        },
        "source_inventory": {
            "path": "extension-source-inventory.json",
            "sha256": _sha256_file(root / "extension-source-inventory.json"),
        },
        "acceptance": {
            "core_changed_files_zero": core_zero,
            "all_case_studies_passed": all_cases,
            "accepted": core_zero and all_cases,
        },
        "suite_execution_duration_seconds": _duration(suite_started),
        "artifacts": pre_verification_artifacts,
    }
    verification = _validate_summary(summary, root, check_declared_artifacts=True)
    _write_json(root / "verification.json", verification)
    summary["artifacts"] = _artifact_descriptors(root)
    summary["artifact_count"] = len(summary["artifacts"])
    _write_json(root / SUMMARY_FILENAME, summary)
    return summary


def _validate_summary(
    summary: dict[str, Any],
    root: Path,
    *,
    check_declared_artifacts: bool,
) -> dict[str, Any]:
    if summary.get("schema_version") != RQ2_EVIDENCE_SCHEMA:
        raise RQ2EvidenceError("unsupported RQ2 evidence schema")
    seeds = summary.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise RQ2EvidenceError("RQ2 summary must contain at least three distinct seeds")
    cases = summary.get("cases")
    expected_cases = {
        "pet_supplies_domain",
        "round_robin_ranking",
        "flash_restock_event_oracle",
    }
    if not isinstance(cases, dict) or set(cases) != expected_cases:
        raise RQ2EvidenceError("RQ2 summary case set is incomplete")
    for case_name, case in cases.items():
        if not isinstance(case, dict) or case.get("passed") is not True:
            raise RQ2EvidenceError(f"RQ2 case did not pass: {case_name}")
        if case.get("run_count") != len(seeds) or case.get("passed_runs") != len(seeds):
            raise RQ2EvidenceError(f"RQ2 case seed coverage mismatch: {case_name}")
    core = summary.get("core_boundary")
    if not isinstance(core, dict) or not isinstance(core.get("changed_file_count"), int):
        raise RQ2EvidenceError("RQ2 core-boundary result is missing")
    if summary.get("network_calls") != 0 or summary.get("llm_judge") is not False:
        raise RQ2EvidenceError("RQ2 suite must remain deterministic and model-free")
    descriptors = summary.get("artifacts")
    if check_declared_artifacts:
        if not isinstance(descriptors, list):
            raise RQ2EvidenceError("RQ2 artifact descriptor list is missing")
        seen: set[str] = set()
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                raise RQ2EvidenceError("RQ2 artifact descriptor must be an object")
            relative = str(descriptor.get("path", ""))
            if (
                not relative
                or relative in seen
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise RQ2EvidenceError(f"invalid RQ2 artifact path: {relative!r}")
            seen.add(relative)
            path = root / relative
            if not path.is_file():
                raise RQ2EvidenceError(f"RQ2 artifact is missing: {relative}")
            if descriptor.get("sha256") != _sha256_file(path):
                raise RQ2EvidenceError(f"RQ2 artifact hash mismatch: {relative}")
            if descriptor.get("bytes") != path.stat().st_size:
                raise RQ2EvidenceError(f"RQ2 artifact byte count mismatch: {relative}")
    return {
        "schema_version": RQ2_VERIFICATION_SCHEMA,
        "artifact_integrity_ok": True,
        "acceptance_ok": (
            summary["acceptance"]["accepted"] is True and core["changed_file_count"] == 0
        ),
        "checked_artifacts": len(descriptors) if isinstance(descriptors, list) else 0,
        "checked_cases": len(cases),
        "checked_seeds": len(seeds),
        "core_changed_file_count": core["changed_file_count"],
        "network_calls": 0,
    }


def verify_rq2_extension_evidence(out_root: str | Path) -> dict[str, Any]:
    """Verify hashes, coverage, case outcomes, and the zero-core-change gate."""

    root = Path(out_root).resolve()
    summary = _read_json(root / SUMMARY_FILENAME)
    if summary.get("artifact_count") != len(summary.get("artifacts", [])):
        raise RQ2EvidenceError("RQ2 artifact count does not match descriptors")
    return _validate_summary(summary, root, check_declared_artifacts=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.rq2_extensions")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="write a new deterministic RQ2 evidence bundle")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    run.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    run.add_argument("--core-baseline-ref", default="HEAD")
    run.add_argument("--evaluated-ref", default="HEAD")
    verify = commands.add_parser("verify", help="verify an existing RQ2 evidence bundle")
    verify.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            summary = write_rq2_extension_evidence(
                args.out,
                repo_root=args.repo,
                seeds=args.seeds,
                core_baseline_ref=args.core_baseline_ref,
                evaluated_ref=args.evaluated_ref,
            )
            payload = {
                "summary": str(Path(args.out) / SUMMARY_FILENAME),
                "accepted": summary["acceptance"]["accepted"],
                "cases": summary["case_count"],
                "seeds": len(summary["seeds"]),
                "network_calls": 0,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload["accepted"] else 1
        verification = verify_rq2_extension_evidence(args.out)
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 0 if verification["acceptance_ok"] else 1
    except RQ2EvidenceError as exc:
        print(
            json.dumps({"error": str(exc), "schema_version": "cwe.rq2-error.v1"}), file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CORE_BOUNDARY_PATHS",
    "DEFAULT_SEEDS",
    "RQ2_EVIDENCE_SCHEMA",
    "RQ2EvidenceError",
    "run_event_case",
    "run_product_domain_case",
    "run_ranking_case",
    "verify_rq2_extension_evidence",
    "write_rq2_extension_evidence",
]
