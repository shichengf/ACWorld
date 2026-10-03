"""Validate trusted sidecar evidence and expose the unchanged ACWorld scores."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys


RESULT_PATH = Path("/acworld-evidence/result.json")
EXPECTED_PATH = Path("/tests/expected.json")
LOG_DIR = Path("/logs/verifier")
SCHEMA = "acworld.harbor-result.v1"
BASELINE_COMMIT = "169602300bb0b9c51c69c0c8afa54c0756a3f7d3"
BINDING_FIELDS = (
    "task_id",
    "task_set_version",
    "task_manifest_sha256",
    "baseline_commit",
)


def reject_constant(value: str):
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def validate_result(result, expected) -> dict[str, float | int]:
    if not isinstance(result, dict) or not isinstance(expected, dict):
        raise ValueError("Result and expected binding must be JSON objects")
    if result.get("schema") != SCHEMA:
        raise ValueError("Unexpected result schema")
    if result.get("finalized") is not True:
        raise ValueError("Trusted result was not finalized")
    if expected.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("Verifier expected binding has the wrong baseline commit")
    for field in BINDING_FIELDS:
        if not isinstance(expected.get(field), str) or not expected[field]:
            raise ValueError(f"Verifier expected binding is missing {field}")
        if result.get(field) != expected[field]:
            raise ValueError(f"Trusted result binding differs: {field}")
    digest = expected["task_manifest_sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Verifier expected task manifest hash is invalid")
    if result.get("status") != "scored":
        raise ValueError(f"ACWorld run was not scored: {result.get('status')!r}")
    if not isinstance(result.get("scorer_version"), str) or not result["scorer_version"]:
        raise ValueError("Missing original scorer_version")
    if "scorer_version" in expected and result["scorer_version"] != expected["scorer_version"]:
        raise ValueError("Trusted result binding differs: scorer_version")
    score = result.get("capability_score")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid original capability_score")
    strict_success = result.get("strict_success")
    if type(strict_success) is not bool:
        raise ValueError("Invalid original strict_success")
    count = result.get("submitted_decisions")
    if type(count) is not int or count < 0:
        raise ValueError("Invalid submitted_decisions count")
    if "interruption_reason" not in result or result["interruption_reason"] is not None:
        raise ValueError("A scored result must have no interruption reason")
    return {"reward": int(strict_success), "capability_score": score}


def save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Never leave an earlier reward beside a failed verification attempt.
    for name in ("reward.json", "reward.txt"):
        (LOG_DIR / name).unlink(missing_ok=True)
    try:
        result = json.loads(RESULT_PATH.read_text(), parse_constant=reject_constant)
        expected = json.loads(EXPECTED_PATH.read_text(), parse_constant=reject_constant)
        rewards = validate_result(result, expected)
    except (OSError, ValueError, TypeError) as exc:
        message = f"Trusted ACWorld evidence could not be verified: {exc}"
        save(LOG_DIR / "verification.json", {"status": "failed", "error": message})
        print(message, file=sys.stderr)
        return 1
    save(LOG_DIR / "reward.json", rewards)
    save(LOG_DIR / "verification.json", {"status": "scored", "task_id": result["task_id"]})
    print(json.dumps(rewards, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
