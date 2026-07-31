"""Verify and aggregate a frozen RQ4 scale suite into paper evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.scale_evidence import (
    ScaleEvidenceError,
    generate_scale_paper_evidence,
    write_scale_paper_evidence,
)
from episode.scale import scale_file_descriptor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acworld-scale-evidence", description=__doc__)
    parser.add_argument("bundle", type=Path, help="complete scale suite directory or manifest")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="set paper_result=true only if every frozen-evidence gate passes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evidence = generate_scale_paper_evidence(args.bundle, promote=args.promote)
        json_path, markdown_path = write_scale_paper_evidence(
            evidence,
            args.json_out,
            args.markdown_out,
        )
    except (FileExistsError, OSError, ScaleEvidenceError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "cwe.scale-paper-evidence-cli.v1",
                    "paper_result": False,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "paper_result": evidence["paper_result"],
                "promotion_eligible": evidence["promotion"]["eligible"],
                "unmet_conditions": evidence["promotion"]["unmet_conditions"],
                "json": scale_file_descriptor(json_path),
                "markdown": scale_file_descriptor(markdown_path),
            },
            sort_keys=True,
        )
    )
    return 0 if not args.promote or evidence["paper_result"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
