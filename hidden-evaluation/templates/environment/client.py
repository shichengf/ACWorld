"""CLI transport for the trusted ACWorld service. Uses only the standard library."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request


SERVICE_URL = "http://acworld:8765"
MAX_DECISION_BYTES = 1024 * 1024


def reject_constant(value: str):
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("observe", help="Read the current business request")
    submit = commands.add_parser("submit", help="Submit a raw decision object")
    submit.add_argument("--decision", type=Path, required=True)
    submit.add_argument("--request-id", required=True, help="request_id from the current observation")
    args = parser.parse_args()

    payload = None
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    try:
        if args.command == "submit":
            if not args.request_id.strip():
                raise ValueError("Request ID must not be empty")
            headers["X-ACWorld-Request-ID"] = args.request_id
            with args.decision.open("rb") as stream:
                raw = stream.read(MAX_DECISION_BYTES + 1)
            if len(raw) > MAX_DECISION_BYTES:
                raise ValueError("Decision file exceeds 1 MiB")
            decision = json.loads(raw, parse_constant=reject_constant)
            if not isinstance(decision, dict):
                raise ValueError("Decision must be a JSON object")
            payload = json.dumps(decision, allow_nan=False).encode("utf-8")

        request = urllib.request.Request(
            SERVICE_URL + "/" + args.command,
            data=payload,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        # A fixed internal destination must not be routed through model API proxies.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=180) as response:
            result = json.load(response, parse_constant=reject_constant)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"ACWorld request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
