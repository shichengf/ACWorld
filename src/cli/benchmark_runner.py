"""Unified entry point for the 200-task, 60-task, and 260-task runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from cli.large_catalog import main as large_catalog_main
from cli.strong_models import main as core_benchmark_main
from experiments.openrouter_runtime import MAX_SUPPORTED_WORKERS
from experiments.benchmark_plan import (
    PAPER_MODELS_V2,
    SUPPORTED_BENCHMARK_MODELS_V2,
)


def _worker_count(value: str) -> int:
    try:
        workers = int(value)
    except ValueError:
        workers = 0
    if not 1 <= workers <= MAX_SUPPORTED_WORKERS:
        raise argparse.ArgumentTypeError(
            f"--workers must be a whole number from 1 to {MAX_SUPPORTED_WORKERS}. "
            f"got {value!r}"
        )
    return workers


DEFAULT_CORE_OUTPUT = Path("output/benchmark")
DEFAULT_LARGE_OUTPUT = Path("output/large-catalog")
DEFAULT_CATALOG_DB = DEFAULT_LARGE_OUTPUT / "catalog.sqlite"
DEFAULT_LARGE_TASKS = DEFAULT_LARGE_OUTPUT / "tasks.json"


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./run_benchmark.sh run",
        description=(
            "Run the original 200 tasks, the 60-task large-catalog suite, "
            "or both suites for 260 tasks."
        ),
    )
    parser.add_argument(
        "--tasks",
        choices=("200", "60", "260"),
        default="200",
        help="benchmark size; omitted means the original 200 tasks",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        choices=(*SUPPORTED_BENCHMARK_MODELS_V2, "all"),
        help="exact model ID; repeat the option or use 'all' for the paper panel",
    )
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="raw CSV directory; needed only when the large catalog is not prepared",
    )
    parser.add_argument(
        "--workers",
        type=_worker_count,
        default=2,
        help=(
            f"concurrent tasks per model, 1 to {MAX_SUPPORTED_WORKERS} "
            "(default 2). Tasks are scored in isolation, so this does not "
            "change any task's score, but a higher rate of provider calls is "
            "more likely to be throttled upstream and a throttled episode is "
            "dropped rather than scored"
        ),
    )
    parser.add_argument("--max-cost-usd", type=float, default=10.0)
    parser.add_argument("--skip-local-smoke", action="store_true")
    parser.add_argument(
        "--canary-only",
        action="store_true",
        help="run only the three large-catalog canaries; requires --tasks 60",
    )
    parser.add_argument(
        "--output-root",
        "--core-output-root",
        dest="core_output_root",
        type=Path,
        default=DEFAULT_CORE_OUTPUT,
        help="output directory for the original 200 tasks",
    )
    parser.add_argument(
        "--large-output-root",
        type=Path,
        default=DEFAULT_LARGE_OUTPUT,
    )
    parser.add_argument("--catalog-db", type=Path, default=DEFAULT_CATALOG_DB)
    parser.add_argument(
        "--large-task-file",
        type=Path,
        default=DEFAULT_LARGE_TASKS,
    )
    return parser


def _selected_models(values: Sequence[str]) -> tuple[str, ...]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("'all' cannot be combined with another --model")
        return PAPER_MODELS_V2
    selected = tuple(values)
    if len(selected) != len(set(selected)):
        raise ValueError("model selection contains duplicates")
    return selected


def _large_file_args(args: argparse.Namespace) -> list[str]:
    return [
        "--catalog-db",
        str(args.catalog_db),
        "--tasks",
        str(args.large_task_file),
        "--output-root",
        str(args.large_output_root),
    ]


def _prepare_and_validate_large(args: argparse.Namespace) -> None:
    if not args.catalog_db.is_file():
        if args.data_root is None:
            large_catalog_main(["download", *_large_file_args(args)])
        else:
            large_catalog_main(
                [
                    "prepare",
                    "--data-root",
                    str(args.data_root),
                    *_large_file_args(args),
                ]
            )
    large_catalog_main(["validate", *_large_file_args(args)])


def _run_core(args: argparse.Namespace) -> int:
    command = [
        "run",
        *[
            value
            for model_id in args.model
            for value in ("--model", model_id)
        ],
        "--api-key-file",
        str(args.api_key_file),
        "--output-root",
        str(args.core_output_root),
        "--workers",
        str(args.workers),
    ]
    if args.skip_local_smoke:
        command.append("--skip-local-smoke")
    return core_benchmark_main(command)


def _run_large(args: argparse.Namespace) -> None:
    command = [
        "run",
        *[
            value
            for model_id in args.model
            for value in ("--model", model_id)
        ],
        "--api-key-file",
        str(args.api_key_file),
        "--max-workers",
        str(args.workers),
        "--max-cost-usd",
        str(args.max_cost_usd),
        *_large_file_args(args),
    ]
    if args.canary_only:
        command.append("--canary-only")
    large_catalog_main(command)
    large_catalog_main(["summary", *_large_file_args(args)])


def _run(argv: Sequence[str]) -> int:
    args = _run_parser().parse_args(argv)
    selected = _selected_models(args.model)
    if args.tasks in {"60", "260"}:
        unsupported = tuple(
            model_id for model_id in selected if model_id not in PAPER_MODELS_V2
        )
        if unsupported:
            raise ValueError(
                "the large-catalog suite is limited to the ten paper models: "
                f"{unsupported!r}"
            )
        _prepare_and_validate_large(args)
    if args.canary_only and args.tasks != "60":
        raise ValueError("--canary-only is available only with --tasks 60")
    if args.tasks in {"200", "260"}:
        status = _run_core(args)
        if status:
            return status
    if args.tasks in {"60", "260"}:
        _run_large(args)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] != "run":
        return core_benchmark_main(values)
    try:
        return _run(values[1:])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "interrupted: completed tasks are saved; rerun the same command to resume",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
