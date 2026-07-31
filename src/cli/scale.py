"""Run deterministic 10×10, 100×100, or 1000×1000 sparse market probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from episode.replay import ReplayVerificationError, verify_scale_artifact_bundle
from episode.scale import (
    ScaleConfig,
    build_scale_suite_manifest,
    collect_scale_provenance,
    run_scale_probe,
    scale_file_descriptor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acworld-scale-probe", description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("reports/scale"))
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="BUNDLE",
        help=(
            "verify a frozen external run/suite bundle without writing to it; "
            "summary-only reports are rejected"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify is not None:
        try:
            verification = verify_scale_artifact_bundle(args.verify)
        except (ReplayVerificationError, OSError, ValueError) as exc:
            print(
                json.dumps({
                    "schema_version": "cwe.scale-bundle-verification.v1",
                    "raw_bundle_complete": False,
                    "paper_result": False,
                    "error": str(exc),
                }),
                file=sys.stderr,
            )
            return 1
        print(json.dumps(verification.to_dict(), sort_keys=True))
        return 0

    sizes = sorted(set(args.sizes))
    seeds = sorted(set(args.seeds))
    configs = [
        ScaleConfig(size, size, top_k=args.top_k, seed=seed)
        for size in sizes
        for seed in seeds
    ]
    suite_path = args.out / "scale-suite.json"
    planned = [suite_path]
    for config in configs:
        root = args.out / f"{config.buyers}x{config.merchants}" / f"seed-{config.seed}"
        planned.extend(
            root / name
            for name in (
                "world.sqlite3",
                "world.replay.sqlite3",
                "events.jsonl",
                "scale-report.json",
                "run-manifest.json",
            )
        )
    occupied = [path for path in planned if path.exists()]
    if occupied:
        raise FileExistsError(f"scale suite refuses to overwrite artifacts: {occupied}")

    results = []
    provenance = collect_scale_provenance()
    for config in configs:
        size = config.buyers
        seed = config.seed
        out = args.out / f"{size}x{size}" / f"seed-{seed}"
        results.append(run_scale_probe(config, out))
    args.out.mkdir(parents=True, exist_ok=True)
    suite = build_scale_suite_manifest(args.out, results, provenance=provenance)
    with suite_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(suite, indent=2) + "\n")
    verification = verify_scale_artifact_bundle(suite_path)
    print(json.dumps({
        "report": str(suite_path),
        "runs": len(results),
        "paper_result": False,
        "raw_bundle_complete": verification.raw_bundle_complete,
        "suite_descriptor": scale_file_descriptor(suite_path, relative_to=args.out),
        "verification": verification.to_dict(),
        "artifact_boundary": (
            "Retain the complete directory in an external artifact archive; "
            "the suite summary alone is not independently verifiable."
        ),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
