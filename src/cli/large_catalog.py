"""Command-line entry point for the large-catalog stress suite."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any, Sequence

from experiments.openrouter_runtime import MAX_SUPPORTED_WORKERS
from large_catalog.database import CatalogDatabase, prepare_catalog
from large_catalog.download import download_prepared_catalog
from large_catalog.models import SearchRequest
from large_catalog.runner import (
    LARGE_CATALOG_MODELS,
    run_model_tasks,
    run_reference_tasks,
    write_combined_summary,
    write_prompt_audit,
)
from large_catalog.tasks import (
    build_tasks,
    write_oracle_reports,
    write_tasks,
)
from large_catalog.validation import validate_suite


DEFAULT_OUTPUT_ROOT = Path("output/large-catalog")
DEFAULT_CATALOG_DB = DEFAULT_OUTPUT_ROOT / "catalog.sqlite"
DEFAULT_TASKS = DEFAULT_OUTPUT_ROOT / "tasks.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acworld-large-catalog",
        description="Prepare, validate, and run the ACWorld large-catalog stress suite.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build the catalog and sixty tasks.")
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--catalog-db", default=str(DEFAULT_CATALOG_DB))
    prepare.add_argument("--tasks", default=str(DEFAULT_TASKS))
    prepare.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    download = subparsers.add_parser(
        "download",
        help="Download the prepared public catalog package.",
    )
    _common_files(download)
    download.add_argument(
        "--asset-url",
        action="append",
        help="override data asset URL; repeat for an ordered multipart archive",
    )

    validate = subparsers.add_parser("validate", help="Run focused offline validation.")
    _common_files(validate)

    reference = subparsers.add_parser("reference", help="Run the model-free reference policy.")
    _common_files(reference)

    run = subparsers.add_parser("run", help="Run one or all paper models through OpenRouter.")
    _common_files(run)
    run.add_argument(
        "--model",
        action="append",
        required=True,
        choices=(*LARGE_CATALOG_MODELS, "all"),
        help="exact paper model ID; repeat the option or use 'all'",
    )
    run.add_argument("--api-key-file", required=True)
    run.add_argument(
        "--max-workers",
        type=int,
        choices=range(1, MAX_SUPPORTED_WORKERS + 1),
        metavar=f"{{1..{MAX_SUPPORTED_WORKERS}}}",
        default=2,
        help=(
            "concurrent tasks. Tasks are scored in isolation, so this does not "
            "change any task's score; a higher rate of provider calls is more "
            "likely to be throttled upstream and a throttled task is dropped"
        ),
    )
    run.add_argument("--max-cost-usd", type=float, default=10.0)
    run.add_argument("--canary-only", action="store_true")

    summary = subparsers.add_parser("summary", help="Rebuild the combined model summary.")
    _common_files(summary)

    audit = subparsers.add_parser("prompts", help="Export all English and Chinese prompts.")
    _common_files(audit)
    audit.add_argument(
        "--destination",
        default=str(DEFAULT_OUTPUT_ROOT / "prompt-audit.csv"),
    )
    return parser


def _common_files(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog-db", default=str(DEFAULT_CATALOG_DB))
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))


def _peak_memory_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    return int(raw if raw > 10_000_000 else raw * 1024)


def _read_key(path: str | Path) -> str:
    source = Path(path).expanduser()
    if not source.is_file():
        raise SystemExit(f"API key file does not exist: {source}")
    value = source.read_text(encoding="utf-8").strip()
    if not value or any(character.isspace() for character in value):
        raise SystemExit("API key file must contain exactly one non-empty token")
    return value


def _load_or_build_tasks(catalog_db: str | Path, task_path: str | Path):
    path = Path(task_path)
    tasks = build_tasks(catalog_db)
    write_tasks(tasks, path)
    return tasks


def _query_latency_ms(
    database: CatalogDatabase,
    tasks: Sequence[Any],
) -> tuple[float, float]:
    samples: list[float] = []
    seen: set[tuple[str, str]] = set()
    for task in tasks:
        context = task.public_context
        if "query" not in context:
            continue
        query = str(context["query"])
        filters = dict(context.get("filters", {}))
        key = (query, json.dumps(filters, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        started = time.perf_counter()
        database.search(
            SearchRequest(
                query=query,
                filters=filters,
                sort="price_asc",
                page_size=20,
            )
        )
        samples.append((time.perf_counter() - started) * 1_000)
    ordered = sorted(samples)
    if not ordered:
        return 0.0, 0.0
    p50 = ordered[(len(ordered) - 1) // 2]
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return p50, p95


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "download":
        payload = download_prepared_catalog(
            output_root=args.output_root,
            catalog_db=args.catalog_db,
            task_path=args.tasks,
            asset_urls=tuple(args.asset_url) if args.asset_url else None,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "prepare":
        report = prepare_catalog(args.data_root, args.catalog_db)
        tasks = build_tasks(args.catalog_db)
        write_tasks(tasks, args.tasks)
        write_oracle_reports(tasks, Path(args.output_root) / "oracle-reports")
        with CatalogDatabase(args.catalog_db) as database:
            catalog_summary = database.summary()
            query_p50_ms, query_p95_ms = _query_latency_ms(database, tasks)
        payload: dict[str, Any] = {
            **report.to_dict(),
            **catalog_summary,
            "tasks": len(tasks),
            "peak_memory_bytes": _peak_memory_bytes(),
            "catalog_query_p50_ms": query_p50_ms,
            "catalog_query_p95_ms": query_p95_ms,
        }
        path = Path(args.output_root) / "catalog-summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    tasks = _load_or_build_tasks(args.catalog_db, args.tasks)
    if args.command == "validate":
        report = validate_suite(args.catalog_db, tasks)
        payload = report.to_dict()
        path = Path(args.output_root) / "validation-report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not report.passed:
            raise SystemExit(1)
    elif args.command == "reference":
        payload = run_reference_tasks(
            database_path=args.catalog_db,
            tasks=tasks,
            output_root=args.output_root,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "run":
        api_key = _read_key(args.api_key_file)
        if "all" in args.model:
            if len(args.model) != 1:
                raise SystemExit("'all' cannot be combined with another --model")
            model_ids = LARGE_CATALOG_MODELS
        else:
            if len(args.model) != len(set(args.model)):
                raise SystemExit("model selection contains duplicates")
            model_ids = tuple(args.model)
        payload = {}
        for model_id in model_ids:
            payload[model_id] = run_model_tasks(
                model_id=model_id,
                api_key=api_key,
                database_path=args.catalog_db,
                tasks=tasks,
                output_root=args.output_root,
                max_workers=args.max_workers,
                max_cost_usd=args.max_cost_usd,
                canary_only=args.canary_only,
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "summary":
        print(
            json.dumps(
                write_combined_summary(tasks=tasks, output_root=args.output_root),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "prompts":
        write_prompt_audit(tasks, args.destination)
        print(args.destination)


if __name__ == "__main__":
    main()
