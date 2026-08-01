"""One-entry OpenRouter runner for the ACWorld benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from experiments.benchmark_plan import (
    PAPER_MODELS_V2,
    SUPPORTED_BENCHMARK_MODELS_V2,
    build_strong_model_benchmark_plan,
)
from experiments.strong_model_runner import (
    DEFAULT_MAX_WORKERS,
    MAX_WORKERS_V2,
    DEFAULT_MULTI_ITEM_OUTPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MULTI_ITEM_RERUN_MODELS,
    MULTI_ITEM_TASK_IDS,
    STRONG_MODEL_CANDIDATES,
    StrongModelRunError,
    check_openrouter_catalog,
    read_api_key,
    run_openrouter_model,
    run_openrouter_smoke,
    run_openrouter_multi_item,
    run_reference_smoke,
    safe_model_slug,
    select_plan_tasks,
    write_multi_item_report,
    write_result_summary,
)
from experiments.strong_model_report import (
    StrongModelReportError,
    write_paper_analysis,
    write_paper_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acworld-benchmark-runner",
        description=__doc__,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list the accepted model endpoints")

    check = commands.add_parser(
        "check",
        help="check exact model IDs in OpenRouter's public catalog; no inference",
    )
    _add_model_selection(check, SUPPORTED_BENCHMARK_MODELS_V2)

    smoke = commands.add_parser(
        "smoke",
        help="run one Buyer and one Merchant reference task locally; no API key",
    )
    smoke.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "_local_smoke",
    )

    run = commands.add_parser(
        "run",
        help="run or resume selected models on all 200 benchmark tasks",
    )
    _add_model_selection(run, SUPPORTED_BENCHMARK_MODELS_V2)
    run.add_argument("--api-key-file", type=Path, required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument(
        "--workers",
        type=int,
        choices=range(1, MAX_WORKERS_V2 + 1),
        metavar=f"{{1..{MAX_WORKERS_V2}}}",
        default=DEFAULT_MAX_WORKERS,
    )
    run.add_argument(
        "--skip-local-smoke",
        action="store_true",
        help="skip the automatic no-cost Buyer/Merchant reference smoke",
    )

    live_smoke = commands.add_parser(
        "live-smoke",
        help="run one real OpenRouter task through the full benchmark path",
    )
    _add_model_selection(live_smoke, SUPPORTED_BENCHMARK_MODELS_V2)
    live_smoke.add_argument("--api-key-file", type=Path, required=True)
    live_smoke.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "_live_smoke",
    )
    live_smoke.add_argument(
        "--skip-local-smoke",
        action="store_true",
        help="skip the automatic no-cost Buyer/Merchant reference smoke",
    )

    multi_item = commands.add_parser(
        "rerun-multi-item",
        help="run only the 20 corrected Multi-item tasks",
    )
    _add_model_selection(multi_item, MULTI_ITEM_RERUN_MODELS)
    multi_item.add_argument("--api-key-file", type=Path, required=True)
    multi_item.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_MULTI_ITEM_OUTPUT_ROOT,
    )
    multi_item.add_argument(
        "--workers",
        type=int,
        choices=range(1, MAX_WORKERS_V2 + 1),
        metavar=f"{{1..{MAX_WORKERS_V2}}}",
        default=DEFAULT_MAX_WORKERS,
    )
    multi_item.add_argument(
        "--skip-local-smoke",
        action="store_true",
        help="skip the automatic no-cost Buyer/Merchant reference smoke",
    )
    multi_item.add_argument(
        "--one-task-smoke",
        action="store_true",
        help="submit only CWV2-T05-01 to verify the live path; not a final rerun",
    )

    status = commands.add_parser(
        "status",
        help="refresh CSV/JSON summaries without contacting a provider",
    )
    _add_model_selection(status, SUPPORTED_BENCHMARK_MODELS_V2)
    status.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    paper_report = commands.add_parser(
        "paper-report",
        help="print one copy-paste line with all per-model paper statistics",
    )
    _add_model_selection(paper_report, SUPPORTED_BENCHMARK_MODELS_V2)
    paper_report.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    paper_report.add_argument(
        "--t5-output-root",
        type=Path,
        help="optional corrected T5 root to overlay on the existing 200 runs",
    )

    paper_analysis = commands.add_parser(
        "paper-analysis",
        help="export every per-task field needed to recompute paper analyses",
    )
    _add_model_selection(paper_analysis, SUPPORTED_BENCHMARK_MODELS_V2)
    paper_analysis.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    paper_analysis.add_argument(
        "--t5-output-root",
        type=Path,
        help="optional corrected T5 root to overlay on the existing 200 runs",
    )

    multi_item_report = commands.add_parser(
        "multi-item-report",
        help="rebuild the copy-paste report for an existing T5 rerun",
    )
    _add_model_selection(multi_item_report, MULTI_ITEM_RERUN_MODELS)
    multi_item_report.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_MULTI_ITEM_OUTPUT_ROOT,
    )
    return parser


def _add_model_selection(
    parser: argparse.ArgumentParser,
    allowed_models: Sequence[str],
) -> None:
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        choices=(*allowed_models, "all"),
        help="exact OpenRouter model ID; repeat the option or use 'all'",
    )


def _selected_models(
    values: Sequence[str],
    *,
    allowed_models: Sequence[str],
    all_models: Sequence[str],
) -> tuple[str, ...]:
    if "all" in values:
        if len(values) != 1:
            raise StrongModelRunError("'all' cannot be combined with another --model")
        return tuple(all_models)
    selected = tuple(values)
    if len(selected) != len(set(selected)):
        raise StrongModelRunError("model selection contains duplicates")
    if any(model_id not in allowed_models for model_id in selected):
        raise StrongModelRunError("model selection is not allowed for this command")
    return selected


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_command(args: argparse.Namespace) -> int:
    if args.command == "list":
        _print_json(
            [
                {
                    "model_id": candidate.model_id,
                    "label": candidate.label,
                    "note": candidate.note,
                }
                for candidate in STRONG_MODEL_CANDIDATES
            ]
        )
        return 0

    if args.command == "smoke":
        summary, report = run_reference_smoke(args.output_root)
        _print_json(
            {
                "smoke_ok": True,
                "execution": asdict(summary),
                "results": report,
                "output_root": str(args.output_root.resolve()),
            }
        )
        return 0

    allowed_models = (
        MULTI_ITEM_RERUN_MODELS
        if args.command in {"rerun-multi-item", "multi-item-report"}
        else SUPPORTED_BENCHMARK_MODELS_V2
    )
    all_models = (
        MULTI_ITEM_RERUN_MODELS
        if args.command in {"rerun-multi-item", "multi-item-report"}
        else PAPER_MODELS_V2
    )
    models = _selected_models(
        args.model,
        allowed_models=allowed_models,
        all_models=all_models,
    )
    if args.command == "check":
        checked = check_openrouter_catalog(models)
        _print_json({"catalog_ok": True, "models": checked})
        return 0

    if args.command == "status":
        reports = {}
        for model_id in models:
            plan = build_strong_model_benchmark_plan((model_id,))
            root = args.output_root / safe_model_slug(model_id)
            reports[model_id] = write_result_summary(plan, root)
        _print_json(reports)
        return 0

    if args.command == "paper-report":
        for model_id in models:
            root = args.output_root / safe_model_slug(model_id)
            override_root = (
                args.t5_output_root / safe_model_slug(model_id)
                if args.t5_output_root is not None
                else None
            )
            destination = override_root if override_root is not None else root
            _report, line = write_paper_report(
                root,
                override_result_root=override_root,
                output_root=destination,
                expected_model_id=model_id,
            )
            print(line)
        return 0

    if args.command == "paper-analysis":
        for model_id in models:
            root = args.output_root / safe_model_slug(model_id)
            override_root = (
                args.t5_output_root / safe_model_slug(model_id)
                if args.t5_output_root is not None
                else None
            )
            destination = override_root if override_root is not None else root
            _analysis, line = write_paper_analysis(
                root,
                override_result_root=override_root,
                output_root=destination,
                expected_model_id=model_id,
            )
            print(line)
        return 0

    if args.command == "multi-item-report":
        for model_id in models:
            full_plan = build_strong_model_benchmark_plan((model_id,))
            plan = select_plan_tasks(full_plan, MULTI_ITEM_TASK_IDS)
            root = args.output_root / safe_model_slug(model_id)
            _report, line = write_multi_item_report(plan, root)
            print(line)
        return 0

    if args.command not in {"run", "live-smoke", "rerun-multi-item"}:
        raise StrongModelRunError(f"unsupported command: {args.command!r}")

    # All no-cost validation happens before the API key is read.
    checked = check_openrouter_catalog(models)
    if not args.skip_local_smoke:
        run_reference_smoke(args.output_root / "_local_smoke")
    api_key = read_api_key(args.api_key_file)

    reports = {}
    for model_id in models:
        print(f"Running or resuming {model_id} ...", flush=True)
        if args.command == "live-smoke":
            reports[model_id] = run_openrouter_smoke(
                model_id,
                api_key=api_key,
                output_root=args.output_root,
            )
        elif args.command == "rerun-multi-item":
            report, line = run_openrouter_multi_item(
                model_id,
                api_key=api_key,
                output_root=args.output_root,
                max_workers=args.workers,
                one_task_smoke=args.one_task_smoke,
            )
            reports[model_id] = report
            print(line)
        else:
            reports[model_id] = run_openrouter_model(
                model_id,
                api_key=api_key,
                output_root=args.output_root,
                max_workers=args.workers,
            )
    _print_json(
        {
            "catalog": checked,
            "results": reports,
            "output_root": str(args.output_root.resolve()),
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_command(args)
    except (StrongModelRunError, StrongModelReportError) as exc:
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
