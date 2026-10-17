"""``acworld-run`` console entry point — run scenarios end to end and seal evidence.

Usage:
    uv run acworld-run --scenarios all --seed 42 --limit 3 \
      --allow-paid-inference                         # live (OpenRouter)
    uv run acworld-run --scenarios scenarios/s1_direct_purchase_42.yaml \
      --limit 1 --allow-paid-inference

Live runs need an OpenRouter key and spend API tokens. The key is read from
``--api-key-file``, then ``OPENROUTER_API_KEY``, then the repo's
``openrouter_APIkey.txt``. Replay-verifiable per-episode artifacts land under
``--out``; benchmark scores are produced separately.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for ``acworld-run``."""
    from episode.benchmark import BenchmarkTrack, CatalogScale, Difficulty, TaskFamily

    p = argparse.ArgumentParser(prog="acworld-run", description=__doc__)
    p.add_argument("--scenarios", default="all",
                   help="'all' (the bundled corpus) or a path to a YAML file/dir")
    p.add_argument("--seed", type=int, default=None,
                   help="only run scenarios whose seed matches (e.g. 42)")
    p.add_argument("--out", type=Path, default=Path("reports"), help="report output root")
    p.add_argument("--model", default="anthropic/claude-sonnet-5", help="OpenRouter model id")
    p.add_argument("--api-key-file", type=Path, default=None,
                   help="file holding the OpenRouter key (overrides OPENROUTER_API_KEY)")
    p.add_argument("--mode", choices=["E0", "E1"], default="E0")
    p.add_argument("--limit", type=int, default=None,
                   help="required positive cap on paid scenario episodes")
    p.add_argument(
        "--allow-paid-inference",
        action="store_true",
        help="explicitly acknowledge that this legacy live runner can incur charges",
    )
    p.add_argument("--task-family", choices=[f.value for f in TaskFamily], default=None,
                   help="only run one paper-facing task family (T1..T10)")
    p.add_argument("--track", choices=[t.value for t in BenchmarkTrack], default=None,
                   help="agent, platform_diagnostic, or conformance")
    p.add_argument("--difficulty", choices=[d.value for d in Difficulty], default=None)
    p.add_argument("--catalog-scale", choices=[s.value for s in CatalogScale], default=None)
    p.add_argument(
        "--plugin",
        action="append",
        default=[],
        help="import an extension module before loading scenarios (repeatable)",
    )
    return p


def _resolve_scenarios(arg: str) -> list[Any]:
    from episode.scenario import from_yaml, load_all

    if arg == "all":
        return list(load_all(_REPO / "scenarios"))
    path = Path(arg)
    return list(load_all(path)) if path.is_dir() else [from_yaml(path)]


def _live_channels(model: str, api_key_file: "Path | None" = None) -> Any:
    from agents.inference import OpenAIChannel

    key = ""
    if api_key_file is not None:
        if not api_key_file.is_file():
            raise SystemExit(f"API key file is missing: {api_key_file}")
        key = api_key_file.read_text(encoding="utf-8").strip()
    if not key:
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        keyfile = _REPO / "openrouter_APIkey.txt"
        key = keyfile.read_text(encoding="utf-8").strip() if keyfile.is_file() else ""
    if not key:
        raise SystemExit(
            "no OpenRouter key (pass --api-key-file, set OPENROUTER_API_KEY, "
            "or add openrouter_APIkey.txt)"
        )
    base = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    timeout = float(os.environ.get("CWE_TIMEOUT", "120"))

    # ``episode.scenario`` builds one actor-specific channel per agent and calls
    # this factory as ``factory(agent_id, role)``.
    def make(_agent_id: str, _role: str) -> Any:
        return OpenAIChannel(base_url=base, api_key=key, model=model, timeout=timeout)

    return make


def main(argv: "list[str] | None" = None) -> int:
    """Entry point. Returns the process exit code (0 = success)."""
    args = build_parser().parse_args(argv)
    # Keep this guard before plugin imports, key reads, scenario loading, or
    # channel construction. The formal six-model runner has stricter manifest
    # and availability preflight; this legacy exploratory entry point must at
    # least require an explicit acknowledgement and a hard episode ceiling.
    if not args.allow_paid_inference:
        print(
            "refusing live execution: pass --allow-paid-inference after reviewing cost",
            file=sys.stderr,
        )
        return 2
    if args.limit is None or args.limit <= 0:
        print(
            "refusing unbounded live execution: pass a positive --limit",
            file=sys.stderr,
        )
        return 2
    for module in args.plugin:
        importlib.import_module(module)
    scenarios = _resolve_scenarios(args.scenarios)
    if args.seed is not None:
        scenarios = [s for s in scenarios if s.seed == args.seed]
    if args.task_family is not None:
        scenarios = [s for s in scenarios if s.benchmark is not None
                     and s.benchmark.task_family is not None
                     and s.benchmark.task_family.value == args.task_family]
    if args.track is not None:
        scenarios = [s for s in scenarios if s.benchmark is not None
                     and s.benchmark.track.value == args.track]
    if args.difficulty is not None:
        scenarios = [s for s in scenarios if s.benchmark is not None
                     and s.benchmark.difficulty.value == args.difficulty]
    if args.catalog_scale is not None:
        scenarios = [s for s in scenarios if s.benchmark is not None
                     and s.benchmark.catalog_scale.value == args.catalog_scale]
    scenarios = scenarios[: args.limit]
    if not scenarios:
        print("no scenarios matched", file=sys.stderr)
        return 1

    from episode.runner import EpisodeBatch

    run_id = f"{args.mode}-seed-{args.seed if args.seed is not None else 'all'}"
    print(f"running {len(scenarios)} scenario(s) live [{args.model}] -> {args.out / run_id}",
          file=sys.stderr)
    batch = EpisodeBatch(scenarios=scenarios,
                         channels=_live_channels(args.model, args.api_key_file),
                         out_root=args.out / run_id, mode=args.mode)
    _summary(batch.run())
    return 0


def _summary(results: "list[tuple[str, Any]]") -> None:
    print("\n=== run summary ===")
    for sid, _outcome in results:
        print(f"  {sid:34} evidence sealed")
    print(f"  completed episodes: {len(results)}")


if __name__ == "__main__":
    raise SystemExit(main())
