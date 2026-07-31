"""Create, validate, generate, and structurally dry-run CommerceWorld scenarios."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


_TEMPLATE = """# CommerceWorld many-to-many scenario (money in population is cents)
scenario_id: s18_custom_market_42
seed: 42
initial_state:
  catalog:
    - sku_id: merchant:m1:sku1
      product_id: product:sku1
      merchant_id: merchant:m1
      category: general
      name: Example product
      list_price: 50
      inventory: 2
      attributes: { in_stock: true }
buyer_goal: { product_type: example, max_budget: 75, quantity: 1, constraints: [] }
merchant_policy: { floor_price: 35, refund_policy: 7_day_return }
allowed_actions: [search, propose_offer, counter_offer, accept_offer, create_order, settle]
success_oracle: { product_match: true, final_price_lte: 75 }
population:
  buyers:
    - buyer_id: buyer:b1
      persona: { name: Buyer 1 }
      mandate:
        mandate_id: custom:b1
        goal: example
        quantity: 1
        hard_constraints: { budget: 7500, must_have: [] }
        soft_constraints: []
        soft_preferences: { style: [], avoid: [] }
        authority: { can_buy_without_confirmation: true, must_not_share_with_merchant: [budget] }
        intent_expiry: 2099-01-01T00:00:00Z
  merchants:
    - merchant_id: merchant:m1
      persona: { name: Merchant 1 }
      policy: { floor_price: 3500, refund_policy: 7_day_return }
      catalog_scope: [merchant:m1:sku1]
  matching: { top_k: 5 }
  execution: { max_transactions_per_buyer: 1 }
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acworld-scenario", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="write a many-to-many scenario template")
    init.add_argument("path", type=Path)
    init.add_argument("--force", action="store_true")

    validate = commands.add_parser("validate", help="parse and validate YAML scenarios")
    validate.add_argument("paths", type=Path, nargs="+")

    generate = commands.add_parser("generate", help="run a registered generator plugin")
    generate.add_argument("--plugin", required=True, help="Python module that registers generator")
    generate.add_argument("--generator", required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--params", default="{}", help="JSON object")
    generate.add_argument("--out", type=Path, required=True)

    dry = commands.add_parser(
        "dry-run", help="hydrate, seed, and validate initial events without model inference"
    )
    dry.add_argument("paths", type=Path, nargs="+")
    return parser


def _scenario_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(path.rglob("*.yaml")))
        else:
            out.append(path)
    return sorted(set(out), key=lambda item: item.as_posix())


def _summary(path: Path, *, dry_run: bool) -> dict[str, Any]:
    from episode.scenario import (
        build_secret_registry,
        from_yaml,
        kickoff_envelopes,
        population_for_scenario,
        seed_world,
    )
    from protocol.envelope import validate
    from runtime.router import Router
    from world.state import World

    spec = from_yaml(path)
    population = population_for_scenario(spec)
    result: dict[str, Any] = {
        "path": str(path),
        "scenario_id": spec.scenario_id,
        "buyers": len(population.buyers),
        "merchants": len(population.merchants),
        "top_k": population.matching["top_k"],
        "initial_events": len(kickoff_envelopes(spec)),
        "status": "valid",
    }
    if dry_run:
        world = World()
        seed_world(world, spec)
        router = Router(build_secret_registry(spec))
        events = kickoff_envelopes(spec)
        for event in events:
            validate(event)
            router.check_outbound(event, event.from_)
            router.check_payload(event, event.from_)
        snapshot = world.snapshot()
        result.update(
            {
                "listings": len(snapshot.catalog),
                "inventory_rows": len(snapshot.inventory),
                "secrets": len(build_secret_registry(spec).money),
                "status": "dry_run_ok",
            }
        )
    return result


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _yaml_scalar_sequence(value: Any) -> str | None:
    """Render a flat sequence in the flow form supported by the YAML loader."""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if any(isinstance(item, (Mapping, list, tuple)) for item in value):
        return None
    return "[" + ", ".join(_yaml_scalar(item) for item in value) + "]"


def _dump_yaml(value: Any, *, indent: int = 0) -> list[str]:
    """Emit the dependency-free YAML subset accepted by ``episode.scenario``."""
    pad = " " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, child in value.items():
            scalar_sequence = _yaml_scalar_sequence(child)
            if scalar_sequence is not None:
                lines.append(f"{pad}{key}: {scalar_sequence}")
            elif isinstance(child, (Mapping, list, tuple)) and child:
                lines.append(f"{pad}{key}:")
                lines.extend(_dump_yaml(child, indent=indent + 2))
            elif isinstance(child, (list, tuple)):
                lines.append(f"{pad}{key}: []")
            elif isinstance(child, Mapping):
                lines.append(f"{pad}{key}: {{}}")
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(child)}")
        return lines
    if isinstance(value, (list, tuple)):
        lines = []
        for child in value:
            if isinstance(child, Mapping) and child:
                items = list(child.items())
                key, first = items[0]
                if isinstance(first, (Mapping, list, tuple)) and first:
                    lines.append(f"{pad}- {key}:")
                    lines.extend(_dump_yaml(first, indent=indent + 4))
                else:
                    lines.append(f"{pad}- {key}: {_yaml_scalar(first)}")
                for key, rest in items[1:]:
                    scalar_sequence = _yaml_scalar_sequence(rest)
                    if scalar_sequence is not None:
                        lines.append(f"{pad}  {key}: {scalar_sequence}")
                    elif isinstance(rest, (Mapping, list, tuple)) and rest:
                        lines.append(f"{pad}  {key}:")
                        lines.extend(_dump_yaml(rest, indent=indent + 4))
                    elif isinstance(rest, (list, tuple)):
                        lines.append(f"{pad}  {key}: []")
                    elif isinstance(rest, Mapping):
                        lines.append(f"{pad}  {key}: {{}}")
                    else:
                        lines.append(f"{pad}  {key}: {_yaml_scalar(rest)}")
            else:
                lines.append(f"{pad}- {_yaml_scalar(child)}")
        return lines
    return [f"{pad}{_yaml_scalar(value)}"]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        if args.path.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite existing file: {args.path}")
        args.path.parent.mkdir(parents=True, exist_ok=True)
        args.path.write_text(_TEMPLATE, encoding="utf-8")
        print(json.dumps({"created": str(args.path)}))
        return 0

    if args.command in {"validate", "dry-run"}:
        paths = _scenario_paths(args.paths)
        reports = [_summary(path, dry_run=args.command == "dry-run") for path in paths]
        print(json.dumps({"scenarios": reports}, indent=2))
        return 0

    if args.command == "generate":
        importlib.import_module(args.plugin)
        from episode.extensions import SCENARIO_GENERATORS

        params = json.loads(args.params)
        if not isinstance(params, dict):
            raise SystemExit("--params must decode to a JSON object")
        generated = SCENARIO_GENERATORS.get(args.generator)(args.seed, params)
        if isinstance(generated, str):
            body = generated.rstrip() + "\n"
        else:
            if is_dataclass(generated) and not isinstance(generated, type):
                generated = asdict(generated)
            if not isinstance(generated, Mapping):
                raise SystemExit("generator must return YAML text, a mapping, or a dataclass")
            body = "\n".join(_dump_yaml(generated)) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body, encoding="utf-8")
        _summary(args.out, dry_run=False)
        print(json.dumps({"generated": str(args.out), "generator": args.generator}))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
