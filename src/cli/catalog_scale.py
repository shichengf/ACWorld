"""Plan, checkpoint, resume, and verify the offline real-catalog scale axis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.catalog_scale import (
    CATALOG_SCALE_VERIFICATION_SCHEMA,
    CatalogScaleVerificationError,
    build_catalog_scale_plan,
    load_catalog_scale_plan,
    run_catalog_scale_probe,
    validate_catalog_scale_plan,
    verify_catalog_scale_report,
    write_catalog_scale_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acworld-catalog-scale", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="write the canonical 36-cell plan")
    plan.add_argument("--out", type=Path, required=True)

    run = commands.add_parser("run", help="run or resume deterministic probes")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--max-probes",
        type=int,
        help="checkpoint after at most this many new dataset probes",
    )

    verify = commands.add_parser(
        "verify", help="independently recompute a complete report"
    )
    verify.add_argument("--report", type=Path, required=True)
    return parser


def _emit(value: object, *, stream: object | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    print(json.dumps(value, sort_keys=True), file=stream)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            plan = build_catalog_scale_plan()
            validate_catalog_scale_plan(plan)
            write_catalog_scale_plan(plan, args.out)
            _emit({
                "schema_version": CATALOG_SCALE_VERIFICATION_SCHEMA,
                "kind": "plan_written",
                "path": str(args.out),
                "plan_sha256": plan["plan_sha256"],
                "dataset_probes": len(plan["dataset_probes"]),
                "cells": len(plan["cells"]),
                "modifies_main_model_plan": False,
            })
            return 0

        if args.command == "run":
            plan = load_catalog_scale_plan(args.plan)
            report = run_catalog_scale_probe(
                plan,
                args.out,
                resume=args.resume,
                max_probes=args.max_probes,
            )
            _emit({
                "schema_version": CATALOG_SCALE_VERIFICATION_SCHEMA,
                "kind": "probe_checkpoint",
                "path": str(args.out),
                "report_sha256": report["report_sha256"],
                **report["progress"],
                "model_inference": False,
                "paper_result": False,
            })
            return 0

        verification = verify_catalog_scale_report(args.report)
        _emit(verification)
        return 0
    except (
        CatalogScaleVerificationError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        _emit({
            "schema_version": CATALOG_SCALE_VERIFICATION_SCHEMA,
            "verified": False,
            "paper_result": False,
            "error": str(exc),
        }, stream=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
