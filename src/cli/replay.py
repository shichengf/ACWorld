"""``acworld-replay``: strict offline verification of episode and scale runs."""

from __future__ import annotations

import argparse
import json
import sys

from episode.replay import ReplayVerificationError, verify_replay_target


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for ``acworld-replay``."""
    parser = argparse.ArgumentParser(
        prog="acworld-replay",
        description=(
            "Verify canonical audit bytes and reconstruct episode state from ordered "
            "transaction diffs, or rebuild a scale probe from its event stream."
        ),
    )
    parser.add_argument(
        "target",
        help="episode/scale directory, batch root, audit.jsonl, or events.jsonl",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require byte-canonical audit records (default: true)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Return zero only when every resolved run verifies exactly."""

    args = build_parser().parse_args(argv)
    try:
        results = verify_replay_target(args.target, strict=args.strict)
    except ReplayVerificationError as exc:
        print(json.dumps({"replay_ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    payload = {
        "replay_ok": True,
        "runs_verified": len(results),
        "results": [result.to_dict() for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
