"""``ScenarioSpec`` hydration helpers and the v0.1 YAML loader.

The project ships with **zero runtime dependencies** (``pyproject.toml`` —
``dependencies = []``), so this loader cannot use PyYAML. Instead it parses
the *constrained YAML subset* the v0.1 scenario generators emit
(``scenarios/generators/``): block mappings, block sequences of mappings,
inline flow lists ``[a, b]`` and flow maps ``{ k: v }``, int/bool/str scalars,
``#`` comments. That is the shape the scenario generators emit — no anchors,
no multi-line flow, no quoting beyond optional simple quotes.

When the dropped ``src/scenarios/`` package eventually returns, a richer
loader will satisfy ``episode.interfaces.ScenarioLoader`` and replace this
default without touching ``Episode``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from episode.errors import ScenarioInvalid
from episode.benchmark import metadata_for_scenario
from episode.types import (
    BuyerSpec,
    ControlServiceSpec,
    ExtensionEvaluationSpec,
    MerchantSpec,
    PopulationSpec,
    ScenarioSpec,
    WorldEventSpec,
)
from protocol.actions import ActionKind

if TYPE_CHECKING:
    from agents.base import Agent
    from protocol.envelope import Envelope


# --- minimal dependency-free YAML-subset parser -----------------------

_INT_RE = re.compile(r"-?\d+$")
_FLOAT_RE = re.compile(r"-?\d+\.\d+$")


def _strip_inline_comment(s: str) -> str:
    """Drop a trailing ``# …`` comment that is not inside a flow ``[]``/``{}``."""
    depth = 0
    for i, c in enumerate(s):
        if c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
        elif c == "#" and depth == 0 and (i == 0 or s[i - 1] in " \t"):
            return s[:i].rstrip()
    return s.rstrip()


def _split_flow(s: str) -> list[str]:
    """Split top-level commas, respecting nested flows and quoted scalars."""
    parts: list[str] = []
    depth = 0
    cur = ""
    quote: str | None = None
    escaped = False
    for c in s:
        if quote is not None:
            cur += c
            if escaped:
                escaped = False
            elif c == "\\" and quote == '"':
                escaped = True
            elif c == quote:
                quote = None
            continue
        if c in ("'", '"'):
            quote = c
            cur += c
            continue
        if c in "[{":
            depth += 1
            cur += c
        elif c in "]}":
            depth -= 1
            cur += c
        elif c == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        parts.append(cur)
    return parts


def _scalar(tok: str) -> Any:
    t = tok.strip()
    if t == "" or t in ("null", "~"):
        return None
    if t == "true":
        return True
    if t == "false":
        return False
    if _INT_RE.match(t):
        return int(t)
    if _FLOAT_RE.match(t):
        return float(t)
    if t[0] == "[" and t[-1] == "]":
        inner = t[1:-1].strip()
        return [_scalar(x) for x in _split_flow(inner)] if inner else []
    if t[0] == "{" and t[-1] == "}":
        inner = t[1:-1].strip()
        d: dict[str, Any] = {}
        for part in _split_flow(inner) if inner else []:
            k, _, v = part.partition(":")
            d[k.strip()] = _scalar(v)
        return d
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        if t[0] == '"':
            try:
                return json.loads(t)
            except json.JSONDecodeError as exc:
                raise ScenarioInvalid(f"invalid quoted scalar: {t!r}") from exc
        return t[1:-1]
    return t


def _prep(text: str) -> list[tuple[int, str]]:
    """Return ``(indent, content)`` for each significant line."""
    out: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        content = _strip_inline_comment(raw).rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        out.append((indent, content.strip()))
    return out


def _parse(lines: list[tuple[int, str]], i: int, indent: int) -> tuple[Any, int]:
    if i < len(lines) and lines[i][1].startswith("- "):
        return _parse_seq(lines, i, indent)
    return _parse_map(lines, i, indent)


def _parse_map(lines: list[tuple[int, str]], i: int, indent: int) -> tuple[dict[str, Any], int]:
    d: dict[str, Any] = {}
    while i < len(lines):
        ind, content = lines[i]
        if ind != indent or content.startswith("- "):
            break
        key, sep, rest = content.partition(":")
        if not sep:
            raise ScenarioInvalid(f"expected 'key: value', got {content!r}")
        key = key.strip()
        rest = rest.strip()
        if rest == "":
            if i + 1 < len(lines) and lines[i + 1][0] > indent:
                val, i = _parse(lines, i + 1, lines[i + 1][0])
            else:
                val, i = None, i + 1
            d[key] = val
        else:
            d[key] = _scalar(rest)
            i += 1
    return d, i


def _parse_seq(lines: list[tuple[int, str]], i: int, indent: int) -> tuple[list[Any], int]:
    seq: list[Any] = []
    while i < len(lines):
        ind, content = lines[i]
        if ind != indent or not content.startswith("- "):
            break
        sub: list[tuple[int, str]] = [(0, content[2:].strip())]
        i += 1
        while i < len(lines) and lines[i][0] >= indent + 2:
            sub.append((lines[i][0] - (indent + 2), lines[i][1]))
            i += 1
        val, _ = _parse(sub, 0, 0)
        seq.append(val)
    return seq, i


def _loads(text: str) -> Any:
    lines = _prep(text)
    if not lines:
        return {}
    value, _ = _parse(lines, 0, lines[0][0])
    return value


# --- ScenarioSpec hydration -------------------------------------------

#: The SINGLE source of truth mapping a scenario's task-relevant bare action
#: verb to its protocol ``ActionKind``. Scenarios are hand-authored with friendly bare verbs
#: (``search``, ``settle``, …); the protocol uses namespaced wire strings
#: (``commerce.search``, ``platform.settle_payment``, …) — and the two do NOT
#: derive from each other cleanly (``settle`` ≠ the suffix of
#: ``platform.settle_payment``). Rather than run two mechanisms (suffix
#: derivation + an exception map) that drift, this one explicit table is the
#: only place the correspondence lives. ``allowed_actions`` is oracle metadata,
#: not a Runtime permission boundary. ``from_yaml`` validates ``allowed_actions``
#: against its keys; ``test_load_all_scenarios`` asserts every verb in every
#: shipped scenario is present here, so any drift fails a test, not a live run.
_SCENARIO_VERBS: dict[str, ActionKind] = {
    "search": ActionKind.SEARCH,
    "get_sku": ActionKind.GET_SKU,
    "propose_offer": ActionKind.PROPOSE_OFFER,
    "counter_offer": ActionKind.COUNTER_OFFER,
    "accept_offer": ActionKind.ACCEPT_OFFER,
    "reject_offer": ActionKind.REJECT_OFFER,
    "create_order": ActionKind.CREATE_ORDER,
    "settle": ActionKind.SETTLE,
    "dispatch": ActionKind.COMMERCE_DISPATCH,
    "request_return": ActionKind.REQUEST_RETURN,
    "request_exchange": ActionKind.COMMERCE_REQUEST_EXCHANGE,
    "issue_refund": ActionKind.COMMERCE_ISSUE_REFUND,
    "send_message": ActionKind.COMMERCE_SEND_MESSAGE,
    "open_dispute": ActionKind.PLATFORM_OPEN_DISPUTE,
    "read_supply_state": ActionKind.COMMERCE_READ_SUPPLY_STATE,
    "update_supply": ActionKind.COMMERCE_UPDATE_SUPPLY,
    "allocate_fulfillment": ActionKind.COMMERCE_ALLOCATE_FULFILLMENT,
    "read_shipment": ActionKind.COMMERCE_READ_SHIPMENT,
    "resolve_shipment": ActionKind.COMMERCE_RESOLVE_SHIPMENT,
}
_REQUIRED = {
    "scenario_id": str,
    "seed": int,
    "initial_state": dict,
    "buyer_goal": dict,
    "merchant_policy": dict,
    "allowed_actions": list,
    "success_oracle": dict,
}


def _population_from_raw(raw: Any, *, source: Path) -> "PopulationSpec | None":
    """Validate and hydrate the optional many-to-many ``population`` block.

    Population monetary values are already agent-native integer minor units;
    only the legacy ``buyer_goal`` / ``merchant_policy`` path performs the
    historical dollars-to-cents conversion.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ScenarioInvalid(f"{source}: field 'population' must be a mapping")
    buyer_rows = raw.get("buyers")
    merchant_rows = raw.get("merchants")
    if not isinstance(buyer_rows, list) or not buyer_rows:
        raise ScenarioInvalid(f"{source}: population.buyers must be a non-empty list")
    if not isinstance(merchant_rows, list) or not merchant_rows:
        raise ScenarioInvalid(f"{source}: population.merchants must be a non-empty list")

    buyers: list[BuyerSpec] = []
    merchants: list[MerchantSpec] = []
    ids: set[str] = set()

    for i, row in enumerate(buyer_rows):
        where = f"population.buyers[{i}]"
        if not isinstance(row, dict):
            raise ScenarioInvalid(f"{source}: {where} must be a mapping")
        buyer_id = str(row.get("buyer_id", ""))
        if buyer_id != "buyer" and not buyer_id.startswith("buyer:"):
            raise ScenarioInvalid(f"{source}: {where}.buyer_id must use the buyer namespace")
        if buyer_id in ids:
            raise ScenarioInvalid(f"{source}: duplicate participant id {buyer_id!r}")
        if not isinstance(row.get("mandate"), dict):
            raise ScenarioInvalid(f"{source}: {where}.mandate must be a mapping")
        persona = row.get("persona", {"name": buyer_id})
        initial = row.get("initial_state", {})
        if not isinstance(persona, dict) or not isinstance(initial, dict):
            raise ScenarioInvalid(f"{source}: {where} persona/initial_state must be mappings")
        buyers.append(BuyerSpec(buyer_id, dict(persona), dict(row["mandate"]), dict(initial)))
        ids.add(buyer_id)

    for i, row in enumerate(merchant_rows):
        where = f"population.merchants[{i}]"
        if not isinstance(row, dict):
            raise ScenarioInvalid(f"{source}: {where} must be a mapping")
        merchant_id = str(row.get("merchant_id", ""))
        if merchant_id != "merchant" and not merchant_id.startswith("merchant:"):
            raise ScenarioInvalid(
                f"{source}: {where}.merchant_id must use the merchant namespace"
            )
        if merchant_id in ids:
            raise ScenarioInvalid(f"{source}: duplicate participant id {merchant_id!r}")
        if not isinstance(row.get("policy"), dict):
            raise ScenarioInvalid(f"{source}: {where}.policy must be a mapping")
        persona = row.get("persona", {"name": merchant_id})
        initial = row.get("initial_state", {})
        scope = row.get("catalog_scope", [])
        if not isinstance(persona, dict) or not isinstance(initial, dict):
            raise ScenarioInvalid(f"{source}: {where} persona/initial_state must be mappings")
        if not isinstance(scope, list) or not all(isinstance(x, str) for x in scope):
            raise ScenarioInvalid(f"{source}: {where}.catalog_scope must be a list of strings")
        merchants.append(MerchantSpec(
            merchant_id, dict(persona), dict(row["policy"]), tuple(scope), dict(initial)
        ))
        ids.add(merchant_id)

    events = raw.get("initial_events", [])
    matching = raw.get("matching", {"top_k": 5})
    execution = raw.get("execution", {"max_transactions_per_buyer": 1})
    if not isinstance(events, list) or not all(isinstance(x, dict) for x in events):
        raise ScenarioInvalid(f"{source}: population.initial_events must be a list of mappings")
    if not isinstance(matching, dict) or not isinstance(execution, dict):
        raise ScenarioInvalid(f"{source}: population matching/execution must be mappings")
    top_k = matching.get("top_k", 5)
    max_txn = execution.get("max_transactions_per_buyer", 1)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ScenarioInvalid(f"{source}: population.matching.top_k must be a positive integer")
    if isinstance(max_txn, bool) or not isinstance(max_txn, int) or max_txn <= 0:
        raise ScenarioInvalid(
            f"{source}: population.execution.max_transactions_per_buyer must be positive"
        )
    matching = {**matching, "top_k": top_k}
    execution = {**execution, "max_transactions_per_buyer": max_txn}
    return PopulationSpec(
        buyers=tuple(sorted(buyers, key=lambda item: item.buyer_id)),
        merchants=tuple(sorted(merchants, key=lambda item: item.merchant_id)),
        initial_events=tuple(dict(item) for item in events),
        matching=matching,
        execution=execution,
    )


def _extensions_from_raw(
    raw_events: Any,
    raw_evaluations: Any,
    *,
    source: Path,
) -> tuple[tuple[WorldEventSpec, ...], tuple[ExtensionEvaluationSpec, ...]]:
    """Validate optional declarative registered-extension invocations."""
    if raw_events is None:
        raw_events = []
    if raw_evaluations is None:
        raw_evaluations = []
    if not isinstance(raw_events, list):
        raise ScenarioInvalid(f"{source}: world_events must be a list of mappings")
    if not isinstance(raw_evaluations, list):
        raise ScenarioInvalid(
            f"{source}: extension_evaluations must be a list of mappings"
        )

    events: list[WorldEventSpec] = []
    event_ids: set[str] = set()
    for index, row in enumerate(raw_events):
        where = f"world_events[{index}]"
        if not isinstance(row, dict):
            raise ScenarioInvalid(f"{source}: {where} must be a mapping")
        event_id = str(row.get("event_id", "")).strip()
        handler = str(row.get("handler", "")).strip()
        payload = row.get("payload", {})
        logical_time = row.get("logical_time", 0)
        if not event_id or not handler:
            raise ScenarioInvalid(
                f"{source}: {where}.event_id and handler must be non-empty"
            )
        if event_id in event_ids:
            raise ScenarioInvalid(f"{source}: duplicate world event id {event_id!r}")
        if not isinstance(payload, dict):
            raise ScenarioInvalid(f"{source}: {where}.payload must be a mapping")
        if (
            isinstance(logical_time, bool)
            or not isinstance(logical_time, int)
            or logical_time < 0
        ):
            raise ScenarioInvalid(
                f"{source}: {where}.logical_time must be a non-negative integer"
            )
        events.append(WorldEventSpec(event_id, handler, dict(payload), logical_time))
        event_ids.add(event_id)

    evaluations: list[ExtensionEvaluationSpec] = []
    evaluation_ids: set[str] = set()
    allowed_kinds = {"market_metric", "oracle_primitive"}
    for index, row in enumerate(raw_evaluations):
        where = f"extension_evaluations[{index}]"
        if not isinstance(row, dict):
            raise ScenarioInvalid(f"{source}: {where} must be a mapping")
        evaluation_id = str(row.get("evaluation_id", "")).strip()
        kind = str(row.get("kind", "")).strip()
        name = str(row.get("name", "")).strip()
        arguments = row.get("arguments", {})
        if not evaluation_id or not name:
            raise ScenarioInvalid(
                f"{source}: {where}.evaluation_id and name must be non-empty"
            )
        if evaluation_id in evaluation_ids:
            raise ScenarioInvalid(
                f"{source}: duplicate extension evaluation id {evaluation_id!r}"
            )
        if kind not in allowed_kinds:
            raise ScenarioInvalid(
                f"{source}: {where}.kind must be one of {sorted(allowed_kinds)}"
            )
        if not isinstance(arguments, dict):
            raise ScenarioInvalid(f"{source}: {where}.arguments must be a mapping")
        evaluations.append(ExtensionEvaluationSpec(
            evaluation_id=evaluation_id,
            kind=cast(Literal["market_metric", "oracle_primitive"], kind),
            name=name,
            arguments=dict(arguments),
        ))
        evaluation_ids.add(evaluation_id)

    return tuple(events), tuple(evaluations)


def _control_services_from_raw(
    raw: Any, *, source: Path
) -> tuple[ControlServiceSpec, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ScenarioInvalid(f"{source}: control_services must be a list of mappings")
    services: list[ControlServiceSpec] = []
    seen: set[str] = set()
    for index, row in enumerate(raw):
        where = f"control_services[{index}]"
        if not isinstance(row, dict) or set(row) - {"service_id", "kind", "config"}:
            raise ScenarioInvalid(
                f"{source}: {where} must contain service_id, kind, and optional config"
            )
        service_id = row.get("service_id")
        kind = row.get("kind")
        config = row.get("config", {})
        if not isinstance(service_id, str) or not service_id.startswith("runtime:"):
            raise ScenarioInvalid(f"{source}: {where}.service_id must be runtime:* text")
        if not isinstance(kind, str) or not kind.strip():
            raise ScenarioInvalid(f"{source}: {where}.kind must be non-empty text")
        if not isinstance(config, dict):
            raise ScenarioInvalid(f"{source}: {where}.config must be a mapping")
        if service_id in seen:
            raise ScenarioInvalid(f"{source}: duplicate control service {service_id!r}")
        services.append(ControlServiceSpec(service_id, kind, dict(config)))
        seen.add(service_id)
    return tuple(services)


def from_yaml(path: Path) -> ScenarioSpec:
    """Parse one YAML file into a ``ScenarioSpec``.

    Raises:
        ScenarioInvalid: required field missing, wrong type, or unknown action
            in ``allowed_actions``.
    """
    try:
        raw = _loads(Path(path).read_text(encoding="utf-8"))
    except ScenarioInvalid:
        raise
    except Exception as e:  # noqa: BLE001 — surface any parse failure uniformly
        raise ScenarioInvalid(f"{path}: could not parse ({e})") from e

    if not isinstance(raw, dict):
        raise ScenarioInvalid(f"{path}: top level must be a mapping")

    for key, typ in _REQUIRED.items():
        if key not in raw:
            raise ScenarioInvalid(f"{path}: missing required field {key!r}")
        if not isinstance(raw[key], typ):
            raise ScenarioInvalid(
                f"{path}: field {key!r} must be {typ.__name__}, got "
                f"{type(raw[key]).__name__}"
            )

    bad = [a for a in raw["allowed_actions"] if a not in _SCENARIO_VERBS]
    if bad:
        raise ScenarioInvalid(
            f"{path}: unknown action verb(s) in allowed_actions: {bad} "
            f"(not in episode.scenario._SCENARIO_VERBS)"
        )

    try:
        benchmark = metadata_for_scenario(raw["scenario_id"], raw.get("benchmark"))
    except ValueError as exc:
        raise ScenarioInvalid(f"{path}: {exc}") from exc
    population = _population_from_raw(raw.get("population"), source=Path(path))
    world_events, extension_evaluations = _extensions_from_raw(
        raw.get("world_events"), raw.get("extension_evaluations"), source=Path(path)
    )
    control_services = _control_services_from_raw(
        raw.get("control_services"), source=Path(path)
    )

    return ScenarioSpec(
        scenario_id=raw["scenario_id"],
        seed=raw["seed"],
        initial_state=raw["initial_state"],
        buyer_goal=raw["buyer_goal"],
        merchant_policy=raw["merchant_policy"],
        allowed_actions=tuple(raw["allowed_actions"]),
        success_oracle=raw["success_oracle"],
        platform_policy=raw.get("platform_policy"),
        persona_axes=raw.get("persona_axes"),
        benchmark=benchmark,
        population=population,
        world_events=world_events,
        extension_evaluations=extension_evaluations,
        control_services=control_services,
    )


def load_all(root: Path) -> Iterable[ScenarioSpec]:
    """Recursively load every YAML under ``root`` in deterministic order.

    Order is sorted by relative path so replay is reproducible.
    """
    root = Path(root)
    for p in sorted(root.rglob("*.yaml"), key=lambda q: q.relative_to(root).as_posix()):
        yield from_yaml(p)


# --- episode-runner seams (other lanes) -------------------------------

#: Merchant id used for every seeded scenario listing (the corpus is single-
#: merchant). Buyer-side agent id is always ``"buyer"``.
_MERCHANT_ID = "merchant:m1"
_SHIP_RE = re.compile(r"shipping_within_(\d+)_days")


def _cents(dollars: Any) -> int:
    """Dollars (the corpus unit) -> integer minor units (the buyer/scorer unit)."""
    from decimal import Decimal

    return int(Decimal(str(dollars)) * 100)


#: A merchant ``refund_policy`` that grants returns must be an explicit positive
#: form (``N_day_return``, or a known positive token). Default-deny: anything
#: else (``no_returns``, empty/None, or an unrecognized string) is NOT returnable.
_RETURN_POLICY_RE = re.compile(r"^\d+_day_return$")
_RETURNABLE_TOKENS = frozenset({"returnable", "returns_accepted", "free_returns"})
_NON_RETURNABLE_TOKENS = frozenset({"no_returns", "no_return", "none", "false", "final_sale"})


def is_returnable(refund_policy: Any) -> bool:
    """Return ``True`` iff ``refund_policy`` grants returns — the SINGLE shared
    interpretation for loaders/seeding/scorers (so they never drift).

    Explicit + default-deny, replacing the old ``"return" in policy.lower()``
    substring bug that classified ``no_returns`` (which contains "return") as
    returnable. Recognized forms: ``N_day_return`` (e.g. ``7_day_return``) and a
    small positive-token allowlist -> True; ``no_returns`` / empty / None /
    unrecognized -> False.
    """
    if not refund_policy:
        return False
    p = str(refund_policy).strip().lower()
    if p in _NON_RETURNABLE_TOKENS:
        return False
    if _RETURN_POLICY_RE.match(p) or p in _RETURNABLE_TOKENS:
        return True
    return False


def _legacy_buyer_mandate(spec: ScenarioSpec) -> "dict[str, Any]":
    """Map the (old-schema) ``buyer_goal`` to the rich buyer mandate.

    ``max_budget`` (dollars) -> ``hard_constraints.budget`` (cents).
    ``constraints`` -> ``must_have`` features, except ``shipping_within_N_days``
    which becomes ``delivery_days``. The corpus carries no ``soft_constraints``/
    friends, so those stay empty (the oracle then marks soft/friend N/A — these
    scenarios test base flow, not the social contribution).
    """
    g = spec.buyer_goal
    must_have: list[str] = []
    delivery_days: int | None = None
    for c in g.get("constraints", []):
        m = _SHIP_RE.fullmatch(str(c))
        if m:
            delivery_days = int(m.group(1))
        else:
            must_have.append(str(c))
    hard: dict[str, Any] = {"budget": _cents(g["max_budget"]), "must_have": must_have}
    if delivery_days is not None:
        hard["delivery_days"] = delivery_days
    return {
        "mandate_id": spec.scenario_id,
        "goal": str(g.get("product_type", "")),
        "quantity": int(g.get("quantity", 1)),
        "return_after_purchase": bool(g.get("return_after_purchase", False)),
        "hard_constraints": hard,
        "soft_constraints": g.get("soft_constraints", []),
        "soft_preferences": g.get("soft_preferences", {"style": [], "avoid": []}),
        "authority": {"can_buy_without_confirmation": True,
                      "must_not_share_with_merchant": ["budget"]},
        "intent_expiry": "2099-12-31T00:00:00Z",
    }


def buyer_mandate(spec: ScenarioSpec) -> "dict[str, Any]":
    """Return the deterministic primary buyer mandate.

    The legacy 1x1 schema is normalized from dollars to cents. Explicit
    populations already carry agent-native mandates; the lexicographically
    first buyer is the primary lane used by backward-compatible single-lane
    scorers. Callers that score a many-participant market must iterate the
    population or use the explicit market-metric APIs; this helper does not do
    that aggregation itself.
    """
    if spec.population is not None:
        buyer = min(spec.population.buyers, key=lambda item: item.buyer_id)
        return dict(buyer.mandate)
    return _legacy_buyer_mandate(spec)


def merchant_floor_cents(spec: ScenarioSpec) -> int:
    """Primary merchant floor in cents for the legacy single-lane scorer."""
    if spec.population is not None:
        merchant = min(spec.population.merchants, key=lambda item: item.merchant_id)
        floor = merchant.policy.get("floor_price", 0)
        return int(floor) if floor else 0
    return _cents(spec.merchant_policy.get("floor_price", 0))


def population_for_scenario(spec: ScenarioSpec) -> PopulationSpec:
    """Hydrate a scenario into the canonical many-to-many participant shape.

    Old scenarios remain represented on disk exactly as before. They are mapped
    lazily to one buyer and one merchant so existing YAML, hashes, scorer inputs,
    and command-line behavior remain stable.
    """
    if spec.population is not None:
        return PopulationSpec(
            buyers=tuple(sorted(spec.population.buyers, key=lambda item: item.buyer_id)),
            merchants=tuple(
                sorted(spec.population.merchants, key=lambda item: item.merchant_id)
            ),
            initial_events=tuple(spec.population.initial_events),
            matching={"top_k": 5, **spec.population.matching},
            execution={
                "max_transactions_per_buyer": 1,
                **spec.population.execution,
            },
        )
    mp = spec.merchant_policy
    return PopulationSpec(
        buyers=(BuyerSpec(
            buyer_id="buyer",
            persona={"name": "Buyer"},
            mandate=_legacy_buyer_mandate(spec),
        ),),
        merchants=(MerchantSpec(
            merchant_id=_MERCHANT_ID,
            persona={"name": "Merchant"},
            policy={
                "floor_price": _cents(mp.get("floor_price", 0)),
                "margin_target_bps": 1500,
                "max_negotiation_rounds": 3,
                "refund_policy": mp.get("refund_policy"),
                "claim_aggressiveness": mp.get("claim_aggressiveness"),
            },
        ),),
        matching={"top_k": 5},
        execution={"max_transactions_per_buyer": 1},
    )


def build_secret_registry(spec: ScenarioSpec) -> "Any":
    """Build the runtime :class:`~runtime.privacy.SecretRegistry` from the scenario
    ANSWER KEY (never from agent memory).

    Registers exactly the two scoring-relevant monetary secrets, scoped to their
    counterparty boundary:

    * buyer ``max_budget`` (``hard_constraints.budget``) — must not reach the merchant;
    * merchant ``floor_price`` (``merchant_policy.floor_price``) — must not reach the buyer.

    Booleans, round counts, urgency flags, quantities are deliberately NOT
    registered. Amounts that are zero/absent are skipped.
    """
    from runtime.privacy import MoneySecret, SecretRegistry

    secrets: list[Any] = []
    population = population_for_scenario(spec)
    legacy_population = spec.population is None
    for buyer in population.buyers:
        budget = (buyer.mandate.get("hard_constraints") or {}).get("budget")
        if budget:
            principal_id = (
                "consumer:persona"
                if legacy_population
                else f"consumer:{buyer.buyer_id.split(':', 1)[-1]}"
            )
            allowed_actor_ids = {principal_id}
            if not legacy_population:
                # World is the authoritative store for actor-scoped mandate
                # revisions.  Returning an owner's own secret through that
                # internal read route is not a disclosure to another actor.
                allowed_actor_ids.add("world")
            secrets.append(MoneySecret(
                owner_id=buyer.buyer_id,
                name="max_budget",
                amount_cents=int(budget),
                counterparty_roles=frozenset({"merchant"}),
                allowed_actor_ids=frozenset(allowed_actor_ids),
            ))
    for merchant in population.merchants:
        floor = merchant.policy.get("floor_price", 0)
        if floor:
            secrets.append(MoneySecret(
                owner_id=merchant.merchant_id,
                name="floor_price",
                amount_cents=int(floor),
                counterparty_roles=frozenset({"buyer"}),
                allowed_actor_ids=(
                    frozenset({"world"})
                    if not legacy_population
                    else frozenset()
                ),
            ))
    return SecretRegistry(money=tuple(secrets), actor_scoped=spec.population is not None)


def expected_in_cents(spec: ScenarioSpec) -> "dict[str, Any]":
    """The scenario's ``success_oracle`` with price thresholds normalized to cents.

    Same schema as the YAML block — every dollar-denominated price threshold is
    scaled (the scorer is cents-native). ``final_price_gte`` is included here so
    a lower-bound, when a scenario declares one (e.g. the s3 negotiation family),
    is normalized identically to ``final_price_lte``; omitting it would leave a
    dollars value to be compared against a cents settle price. (As of this patch
    the legacy ``_check_outcome`` reads only ``final_price_lte``; normalizing the
    lower bound keeps the answer key correct for the future per-field scorer.)
    """
    out = dict(spec.success_oracle)
    for key in ("final_price_lte", "final_price_gte", "merchant_margin_gte"):
        if key in out:
            out[key] = _cents(out[key])
    return out


def seed_world(world: Any, spec: ScenarioSpec) -> None:
    """Convert the scenario's minimal ``initial_state`` into World rows and apply.

    Catalog dicts are minimal (``sku_id``/``list_price``/``inventory``/
    ``attributes``); ``name``/``category``/``merchant_id`` are synthesized.
    ``name=sku_id`` is safe because the corpus matches constraints against
    *attributes*, not product names. Forward-compat: ``friendships``/``reviews``
    (for future social YAMLs) seed straight through.
    """
    from decimal import Decimal

    from episode.benchmark import CatalogSource
    from world.types import (
        AgentId,
        InventoryRow,
        Listing,
        Money,
        Order,
        OrderId,
        OrderState,
        OrderTimeline,
        Receipt,
        Shipment,
        ShipmentId,
        ShipmentResolution,
        ShipmentStatus,
        ShipmentStatusEvent,
        SkuId,
        TxnId,
    )

    benchmark = spec.benchmark
    if benchmark is not None and benchmark.catalog_source == CatalogSource.REAL_CSV:
        if spec.initial_state.get("catalog"):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: real_csv catalog_source requires an empty inline catalog"
            )
        from agents.merchant_data_csv import KNOWN_MERCHANTS
        from episode.seed_catalog import seed_world_catalog

        seed_world_catalog(
            world,
            merchants=benchmark.catalog_merchants or KNOWN_MERCHANTS,
            seed=spec.seed,
            in_stock_only=benchmark.in_stock_only,
            catalog_scale=benchmark.catalog_scale,
        )
        _seed_order_settlement_setup(world, spec)
        _seed_pricing_policy_fixtures(world, spec)
        _seed_match_authorizations(world, spec)
        _seed_evidence_contract_tables(world, spec)
        _seed_after_sales_setup(world, spec)
        _seed_market_governance_setup(world, spec)
        _seed_social_tables(world, spec)
        return

    population = population_for_scenario(spec)
    merchant_by_id = {item.merchant_id: item for item in population.merchants}

    catalog: list[Any] = []
    inventory: dict[Any, Any] = {}
    for row in spec.initial_state.get("catalog", []):
        sku = str(row["sku_id"])
        merchant_id = str(row.get("merchant_id") or population.merchants[0].merchant_id)
        merchant = merchant_by_id.get(merchant_id)
        if merchant is None:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: listing {sku!r} references undeclared merchant "
                f"{merchant_id!r}"
            )
        if SkuId(sku) in inventory:
            raise ScenarioInvalid(f"{spec.scenario_id}: duplicate sku_id {sku!r}")
        refund_policy = merchant.policy.get("refund_policy")
        returnable = is_returnable(refund_policy)
        attrs = dict(row.get("attributes", {}))
        attrs.setdefault("returnable", returnable)
        if refund_policy is not None:
            attrs.setdefault("refund_policy", str(refund_policy))
        catalog.append(Listing(
            sku_id=SkuId(sku),
            category=str(row.get("category", "general")),
            name=str(row.get("name", sku)),
            attributes=attrs,
            list_price=Money(amount=Decimal(str(row["list_price"]))),
            merchant_id=AgentId(merchant_id),
            product_id=str(row.get("product_id") or sku),
        ))
        inventory[SkuId(sku)] = InventoryRow(
            sku_id=SkuId(sku), merchant_id=AgentId(merchant_id),
            qty_available=int(row.get("inventory", 0)),
            qty_reserved=int(row.get("qty_reserved", 0)),
            eta_day=int(row.get("eta_day", 0)),
            version=int(row.get("version", 1)),
        )
        if (
            inventory[SkuId(sku)].qty_available < 0
            or inventory[SkuId(sku)].qty_reserved < 0
            or inventory[SkuId(sku)].qty_reserved
            > inventory[SkuId(sku)].qty_available
        ):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: listing {sku!r} has invalid inventory capacity"
            )
    seeded_skus = {str(item.sku_id) for item in catalog}
    for merchant in population.merchants:
        missing = sorted(set(merchant.catalog_scope) - seeded_skus)
        if missing:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {merchant.merchant_id} catalog_scope references "
                f"unknown sku_id(s): {missing}"
            )
        wrongly_owned = sorted(
            sku for sku in merchant.catalog_scope
            if next(str(item.merchant_id) for item in catalog if str(item.sku_id) == sku)
            != merchant.merchant_id
        )
        if wrongly_owned:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {merchant.merchant_id} catalog_scope contains "
                f"listing(s) owned by another merchant: {wrongly_owned}"
            )
    buyer_ids = {buyer.buyer_id for buyer in population.buyers}
    merchant_ids = {merchant.merchant_id for merchant in population.merchants}
    listings = {str(item.sku_id): item for item in catalog}
    orders: list[Order] = []
    orders_by_id: dict[str, Order] = {}
    for index, row in enumerate(spec.initial_state.get("orders", []) or []):
        where = f"initial_state.orders[{index}]"
        if not isinstance(row, dict):
            raise ScenarioInvalid(f"{spec.scenario_id}: {where} must be a mapping")
        try:
            order = Order(
                order_id=OrderId(str(row["order_id"])),
                buyer_id=AgentId(str(row["buyer_id"])),
                merchant_id=AgentId(str(row["merchant_id"])),
                sku_id=SkuId(str(row["sku_id"])),
                qty=int(row["qty"]),
                agreed_price=Money(
                    Decimal(str(row["agreed_price"])),
                    str(row.get("currency", "USD")),
                ),
                state=OrderState(str(row["state"])),
                request_order=int(row.get("request_order", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: invalid {where}: {exc}"
            ) from exc
        order_id = str(order.order_id)
        if order_id in orders_by_id:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: duplicate seeded order_id {order_id!r}"
            )
        listing = listings.get(str(order.sku_id))
        if order.qty <= 0:
            raise ScenarioInvalid(f"{spec.scenario_id}: {where}.qty must be positive")
        if str(order.buyer_id) not in buyer_ids:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} references undeclared buyer "
                f"{str(order.buyer_id)!r}"
            )
        if str(order.merchant_id) not in merchant_ids:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} references undeclared merchant "
                f"{str(order.merchant_id)!r}"
            )
        if listing is None or str(listing.merchant_id) != str(order.merchant_id):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} sku ownership does not match its merchant"
            )
        orders.append(order)
        orders_by_id[order_id] = order

    ledger: list[Receipt] = []
    txn_ids: set[str] = set()
    for index, row in enumerate(spec.initial_state.get("ledger", []) or []):
        where = f"initial_state.ledger[{index}]"
        if not isinstance(row, dict):
            raise ScenarioInvalid(f"{spec.scenario_id}: {where} must be a mapping")
        try:
            effect = str(row.get("effect", "charge"))
            if effect not in {"charge", "refund"}:
                raise ValueError("effect must be charge or refund")
            receipt = Receipt(
                txn_id=TxnId(str(row["txn_id"])),
                ts=str(row.get("ts", "authoritative-seed")),
                order_id=OrderId(str(row["order_id"])),
                buyer_id=AgentId(str(row["buyer_id"])),
                merchant_id=AgentId(str(row["merchant_id"])),
                sku_id=SkuId(str(row["sku_id"])),
                qty=int(row["qty"]),
                price=Money(
                    Decimal(str(row["price"])),
                    str(row.get("currency", "USD")),
                ),
                idempotency_key=str(row["idempotency_key"]),
                effect=cast(Any, effect),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: invalid {where}: {exc}"
            ) from exc
        txn_id = str(receipt.txn_id)
        if txn_id in txn_ids:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: duplicate seeded txn_id {txn_id!r}"
            )
        order = orders_by_id.get(str(receipt.order_id))
        if order is None or (
            str(receipt.buyer_id),
            str(receipt.merchant_id),
            str(receipt.sku_id),
            receipt.qty,
            receipt.price,
        ) != (
            str(order.buyer_id),
            str(order.merchant_id),
            str(order.sku_id),
            order.qty,
            order.agreed_price,
        ):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} does not exactly match its seeded order"
            )
        ledger.append(receipt)
        txn_ids.add(txn_id)

    logical_time = spec.initial_state.get("logical_time", 0)
    if (
        isinstance(logical_time, bool)
        or not isinstance(logical_time, int)
        or logical_time < 0
    ):
        raise ScenarioInvalid(
            f"{spec.scenario_id}: initial_state.logical_time must be a non-negative integer"
        )

    timelines: list[OrderTimeline] = []
    timeline_order_ids: set[str] = set()
    tick_fields = (
        "settled_at_tick",
        "dispatched_at_tick",
        "return_authorized_at_tick",
        "returned_at_tick",
        "refunded_at_tick",
    )
    for index, row in enumerate(spec.initial_state.get("order_timelines", []) or []):
        where = f"initial_state.order_timelines[{index}]"
        if not isinstance(row, dict):
            raise ScenarioInvalid(f"{spec.scenario_id}: {where} must be a mapping")
        order = orders_by_id.get(str(row.get("order_id", "")))
        if order is None:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} references an unknown seeded order"
            )
        order_id = str(order.order_id)
        if order_id in timeline_order_ids:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: duplicate timeline for order {order_id!r}"
            )
        if (
            str(row.get("buyer_id", "")) != str(order.buyer_id)
            or str(row.get("merchant_id", "")) != str(order.merchant_id)
        ):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} parties do not match its seeded order"
            )
        ticks: dict[str, int | None] = {}
        for field in tick_fields:
            value = row.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: {where}.{field} must be a non-negative integer"
                )
            if value is not None and value > logical_time:
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: {where}.{field} exceeds logical_time"
                )
            ticks[field] = value
        window = row.get("return_window_ticks")
        if window is not None and (
            isinstance(window, bool) or not isinstance(window, int) or window <= 0
        ):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where}.return_window_ticks must be positive"
            )
        listing = listings[str(order.sku_id)]
        captured = listing.attributes.get("return_window_ticks")
        if captured is not None and window != captured:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} does not match the listing return window"
            )
        chronology = (
            ("settled_at_tick", "dispatched_at_tick"),
            ("dispatched_at_tick", "return_authorized_at_tick"),
            ("dispatched_at_tick", "returned_at_tick"),
            ("return_authorized_at_tick", "refunded_at_tick"),
            ("returned_at_tick", "refunded_at_tick"),
        )
        if any(
            ticks[before] is not None
            and ticks[after] is not None
            and ticks[before] > ticks[after]
            for before, after in chronology
        ):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} lifecycle ticks are not monotonic"
            )
        timeline = OrderTimeline(
            order_id=order.order_id,
            buyer_id=order.buyer_id,
            merchant_id=order.merchant_id,
            settled_at_tick=ticks["settled_at_tick"],
            dispatched_at_tick=ticks["dispatched_at_tick"],
            return_window_ticks=window,
            return_authorized_at_tick=ticks["return_authorized_at_tick"],
            returned_at_tick=ticks["returned_at_tick"],
            refunded_at_tick=ticks["refunded_at_tick"],
        )
        timelines.append(timeline)
        timeline_order_ids.add(order_id)

    shipments: list[Shipment] = []
    shipment_ids: set[str] = set()
    shipment_orders: set[str] = set()
    for index, row in enumerate(spec.initial_state.get("shipments", []) or []):
        where = f"initial_state.shipments[{index}]"
        if not isinstance(row, dict):
            raise ScenarioInvalid(f"{spec.scenario_id}: {where} must be a mapping")
        order = orders_by_id.get(str(row.get("order_id", "")))
        if order is None:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} references an unknown order"
            )
        try:
            history_raw = row.get("status_history", [])
            if not isinstance(history_raw, list) or not history_raw:
                raise ValueError("status_history must be a non-empty list")
            history = tuple(
                ShipmentStatusEvent(
                    event_id=str(event["event_id"]),
                    status=ShipmentStatus(str(event["status"])),
                    logical_time=int(event["logical_time"]),
                )
                for event in history_raw
            )
            resolution_raw = row.get("resolution")
            replacement_raw = row.get("replacement_sku_id")
            shipment = Shipment(
                shipment_id=ShipmentId(str(row["shipment_id"])),
                order_id=order.order_id,
                buyer_id=AgentId(str(row.get("buyer_id", order.buyer_id))),
                merchant_id=AgentId(str(row.get("merchant_id", order.merchant_id))),
                original_sku_id=SkuId(
                    str(row.get("original_sku_id", order.sku_id))
                ),
                status=ShipmentStatus(str(row["status"])),
                status_history=history,
                resolution=(
                    None
                    if resolution_raw is None
                    else ShipmentResolution(str(resolution_raw))
                ),
                replacement_sku_id=(
                    None
                    if replacement_raw is None
                    else SkuId(str(replacement_raw))
                ),
                version=int(row.get("version", 1)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: invalid {where}: {exc}"
            ) from exc
        shipment_id = str(shipment.shipment_id)
        order_id = str(shipment.order_id)
        if shipment_id in shipment_ids or order_id in shipment_orders:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: duplicate shipment id or order in {where}"
            )
        if (
            shipment.buyer_id != order.buyer_id
            or shipment.merchant_id != order.merchant_id
            or shipment.original_sku_id != order.sku_id
            or order.state != OrderState.DISPATCHED
            or shipment.version != len(shipment.status_history)
            or shipment.status_history[-1].status != shipment.status
            or any(not event.event_id for event in shipment.status_history)
            or len({event.event_id for event in shipment.status_history})
            != len(shipment.status_history)
            or any(
                event.logical_time < 0 or event.logical_time > logical_time
                for event in shipment.status_history
            )
            or any(
                left.logical_time > right.logical_time
                for left, right in zip(
                    shipment.status_history, shipment.status_history[1:]
                )
            )
        ):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} violates shipment/order/history invariants"
            )
        shipments.append(shipment)
        shipment_ids.add(shipment_id)
        shipment_orders.add(order_id)

    reserved_by_sku: dict[SkuId, int] = {}
    for order in orders:
        if order.state in {
            OrderState.PARTIALLY_SETTLED,
            OrderState.SETTLED,
            OrderState.DISPATCHED,
            OrderState.RETURNED,
            OrderState.REFUNDED,
        }:
            if not any(str(item.order_id) == str(order.order_id) for item in ledger):
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: paid seeded order {order.order_id!r} has no ledger"
                )
        # Only paid inventory-bearing states require a seeded reservation.
        # PROPOSED/ACCEPTED are non-financial order intents and BACKORDERED is
        # explicitly zero-fill unless a FulfillmentAllocation says otherwise;
        # forcing those states to reserve stock made a pre-dispatch cancellation
        # fixture consume inventory before payment.
        if order.state in {
            OrderState.PARTIALLY_SETTLED,
            OrderState.SETTLED,
            OrderState.DISPATCHED,
        }:
            reserved_by_sku[order.sku_id] = (
                reserved_by_sku.get(order.sku_id, 0) + order.qty
            )
        listing = listings[str(order.sku_id)]
        if (
            listing.attributes.get("return_window_ticks") is not None
            and str(order.order_id) not in timeline_order_ids
        ):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: explicitly windowed order {order.order_id!r} "
                "requires authoritative timeline evidence"
            )
    for sku, required in reserved_by_sku.items():
        inv = inventory.get(sku)
        if inv is not None and inv.qty_reserved < required:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: inventory reservation is below seeded paid order qty"
            )

    reputation = _seeded_reputation(spec)
    state: dict[str, Any] = {"catalog": catalog, "inventory": inventory}
    # Absence means "leave the persistent E1 table alone"; an explicitly empty
    # block means "seed this table empty". This preserves legacy E1 behavior.
    for key, value in (
        ("orders", orders),
        ("ledger", ledger),
        ("order_timelines", timelines),
        ("reputation", reputation),
        ("logical_time", logical_time),
        ("shipments", shipments),
    ):
        if key in spec.initial_state:
            state[key] = value
    world.apply(state)
    _seed_order_settlement_setup(world, spec)
    _seed_pricing_policy_fixtures(world, spec)
    _seed_match_authorizations(world, spec)
    _seed_evidence_contract_tables(world, spec)
    _seed_after_sales_setup(world, spec)
    _seed_market_governance_setup(world, spec)
    _seed_social_tables(world, spec)


def materialize_initial_world_tables(spec: ScenarioSpec) -> dict[str, Any]:
    """Materialize the authoritative pre-kickoff tables for ``spec``.

    This is the read-only production fixture seam used when a caller needs to
    compare a stored initial snapshot with the state that :class:`Episode`
    would create.  It intentionally reuses the same World implementation,
    scenario seeding, registered extension events, and canonical serializer as
    both Episode transports.  No policy action is executed here.
    """

    from episode.extension_runtime import apply_registered_world_events
    from evals.serialize import to_canonical
    from world import World

    world = World()
    seed_world(world, spec)
    apply_registered_world_events(world, spec)
    tables = to_canonical(world.snapshot())
    if not isinstance(tables, dict):  # pragma: no cover - serializer invariant
        raise ScenarioInvalid(
            f"{spec.scenario_id}: materialized World snapshot must be a mapping"
        )
    return tables


def _seed_order_settlement_setup(world: Any, spec: ScenarioSpec) -> None:
    """Bootstrap paid orders through the normal authoritative World lifecycle.

    A scenario may name an already seeded order plus opaque transaction and
    idempotency identities.  CommerceWorld reads the authoritative order and
    derives every commercial fact in the receipt before calling
    :meth:`World.settle_order`.  The setup therefore produces the same atomic
    order, inventory, ledger, timeline, payment, clock, and replay records as
    a live settlement.  It is not a loader for those tables.
    """

    raw = spec.initial_state.get("order_settlement_setup")
    if raw is None:
        return
    where = "initial_state.order_settlement_setup"
    if not isinstance(raw, list):
        raise ScenarioInvalid(f"{spec.scenario_id}: {where} must be an array")

    from world.types import Order, OrderState, Receipt, TxnId

    validated: list[tuple[dict[str, str], Order]] = []
    seen_orders: set[str] = set()
    seen_txns: set[str] = set()
    seen_keys: set[str] = set()
    exact_fields = {"order_id", "txn_id", "idempotency_key"}
    for index, value in enumerate(raw):
        row_where = f"{where}[{index}]"
        if not isinstance(value, Mapping) or set(value) != exact_fields:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {row_where} fields must be exactly: "
                + ", ".join(sorted(exact_fields))
            )
        row: dict[str, str] = {}
        for field in sorted(exact_fields):
            item = value.get(field)
            if not isinstance(item, str) or not item.strip():
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: {row_where}.{field} must be non-empty"
                )
            row[field] = item
        if row["order_id"] in seen_orders:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: duplicate setup order {row['order_id']!r}"
            )
        if row["txn_id"] in seen_txns:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: duplicate setup transaction {row['txn_id']!r}"
            )
        if row["idempotency_key"] in seen_keys:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: duplicate settlement setup key "
                f"{row['idempotency_key']!r}"
            )
        order = world.read("orders", row["order_id"], caller="platform:psp")
        if not isinstance(order, Order):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {row_where} references unknown order "
                f"{row['order_id']!r}"
            )
        if order.state not in {OrderState.PROPOSED, OrderState.ACCEPTED}:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {row_where} order must start proposed or accepted"
            )
        seen_orders.add(row["order_id"])
        seen_txns.add(row["txn_id"])
        seen_keys.add(row["idempotency_key"])
        validated.append((row, order))

    try:
        for row, order in validated:
            receipt = Receipt(
                txn_id=TxnId(row["txn_id"]),
                ts=f"world-settlement-setup:{world.logical_time + 1}",
                order_id=order.order_id,
                buyer_id=order.buyer_id,
                merchant_id=order.merchant_id,
                sku_id=order.sku_id,
                qty=order.qty,
                price=order.agreed_price,
                idempotency_key=row["idempotency_key"],
                effect="charge",
            )
            world.settle_order(
                order=order,
                receipt=receipt,
                by_role="platform:psp",
                idempotency_key=row["idempotency_key"],
            )
    except ScenarioInvalid:
        raise
    except Exception as exc:
        raise ScenarioInvalid(
            f"{spec.scenario_id}: invalid {where}: {exc}"
        ) from exc


def _seed_after_sales_setup(world: Any, spec: ScenarioSpec) -> None:
    """Hydrate typed after-sales prerequisites through authoritative World APIs.

    The scenario is allowed to describe only compact policy, payment, and
    packing intents.  Service identity, transaction parties, versions,
    digests, logical time, and outcomes are derived by CommerceWorld.  This is
    deliberately not a generic after-sales record loader: return, refund,
    exchange, dispute, ruling, and reconciliation records must be produced by
    the real Runtime -> Platform -> World execution path.
    """

    raw = spec.initial_state.get("after_sales_setup")
    if raw is None:
        return
    where = "initial_state.after_sales_setup"
    expected_sections = {
        "policies",
        "payment_transitions",
        "packing_transitions",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_sections:
        raise ScenarioInvalid(
            f"{spec.scenario_id}: {where} fields must be exactly: "
            + ", ".join(sorted(expected_sections))
        )

    sections: dict[str, tuple[dict[str, Any], ...]] = {}
    section_fields = {
        "policies": {"merchant_id", "idempotency_key", "intent"},
        "payment_transitions": {"idempotency_key", "intent"},
        "packing_transitions": {"idempotency_key", "intent"},
    }
    for section, exact_fields in section_fields.items():
        rows = raw[section]
        if not isinstance(rows, list):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where}.{section} must be an array"
            )
        validated: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            row_where = f"{where}.{section}[{index}]"
            if not isinstance(row, Mapping) or set(row) != exact_fields:
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: {row_where} fields must be exactly: "
                    + ", ".join(sorted(exact_fields))
                )
            idempotency_key = row.get("idempotency_key")
            intent = row.get("intent")
            if not isinstance(idempotency_key, str) or not idempotency_key:
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: {row_where}.idempotency_key must be non-empty"
                )
            if not isinstance(intent, Mapping):
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: {row_where}.intent must be an object"
                )
            if section == "policies":
                merchant_id = row.get("merchant_id")
                if not isinstance(merchant_id, str) or not merchant_id:
                    raise ScenarioInvalid(
                        f"{spec.scenario_id}: {row_where}.merchant_id must be non-empty"
                    )
            validated.append(dict(row))
        sections[section] = tuple(validated)

    population = population_for_scenario(spec)
    declared_merchants = {merchant.merchant_id for merchant in population.merchants}
    resolved_payments: list[tuple[dict[str, Any], Any]] = []
    resolved_packings: list[tuple[dict[str, Any], Any]] = []

    # Resolve every order and principal before the first setup write.  A bad
    # reference therefore cannot leave a partially hydrated scenario behind.
    for section, target in (
        ("payment_transitions", resolved_payments),
        ("packing_transitions", resolved_packings),
    ):
        for index, row in enumerate(sections[section]):
            intent = cast(dict[str, Any], row["intent"])
            order_id = intent.get("order_id")
            if not isinstance(order_id, str) or not order_id:
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: {where}.{section}[{index}].intent "
                    "requires a non-empty order_id"
                )
            order = world.read("orders", order_id, caller="platform:setup")
            if order is None:
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: {where}.{section}[{index}] references "
                    f"unknown order {order_id!r}"
                )
            target.append((row, order))

    try:
        for row in sections["policies"]:
            merchant_id = cast(str, row["merchant_id"])
            if merchant_id not in declared_merchants:
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: {where}.policies references undeclared "
                    f"merchant {merchant_id!r}"
                )
            world.publish_after_sales_policy(
                cast(Mapping[str, Any], row["intent"]),
                by_actor="platform:policy",
                original_actor=merchant_id,
                idempotency_key=cast(str, row["idempotency_key"]),
            )

        for row, order in resolved_payments:
            intent = cast(Mapping[str, Any], row["intent"])
            op = intent.get("op")
            if op == "authorize":
                world.authorize_payment(
                    intent,
                    by_actor="platform:psp",
                    original_actor=str(order.buyer_id),
                    idempotency_key=cast(str, row["idempotency_key"]),
                )
            elif op == "capture":
                world.capture_payment(
                    intent,
                    by_actor="platform:psp",
                    original_actor=str(order.buyer_id),
                    idempotency_key=cast(str, row["idempotency_key"]),
                )
            else:
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: payment setup op must be authorize or capture"
                )

        for row, order in resolved_packings:
            world.apply_packing_intent(
                cast(Mapping[str, Any], row["intent"]),
                by_actor="platform:fulfillment",
                original_actor=str(order.merchant_id),
                idempotency_key=cast(str, row["idempotency_key"]),
            )
    except ScenarioInvalid:
        raise
    except Exception as exc:
        raise ScenarioInvalid(
            f"{spec.scenario_id}: invalid {where}: {exc}"
        ) from exc


def _seed_market_governance_setup(world: Any, spec: ScenarioSpec) -> None:
    """Publish trusted governance policy inputs through World public methods.

    This setup surface is deliberately policy-only.  Campaigns, reviews,
    aggregates, signals, cases, decisions, reputation events, remediation
    plans, and ranking contexts are execution outcomes and cannot be loaded
    here.  Trusted service identity is derived from the policy kind rather
    than accepted from the scenario.
    """

    raw = spec.initial_state.get("market_governance_setup")
    if raw is None:
        return
    where = "initial_state.market_governance_setup"
    if not isinstance(raw, Mapping) or set(raw) != {"policies"}:
        raise ScenarioInvalid(
            f"{spec.scenario_id}: {where} fields must be exactly: policies"
        )
    policy_rows = raw.get("policies")
    if not isinstance(policy_rows, list):
        raise ScenarioInvalid(
            f"{spec.scenario_id}: {where}.policies must be an array"
        )

    from world.market_governance_world import governance_policy_authority

    from protocol.evidence_records import coerce_evidence_record

    declared_buyers = {
        buyer.buyer_id for buyer in population_for_scenario(spec).buyers
    }
    raw_evidence = spec.initial_state.get("evidence_records", ())
    evidence_rows = (
        list(raw_evidence.values())
        if isinstance(raw_evidence, Mapping)
        else list(raw_evidence or ())
    )
    historical_reviewers: set[str] = set()
    for index, row in enumerate(evidence_rows):
        try:
            record = coerce_evidence_record(row)
        except Exception as exc:  # noqa: BLE001 - normalize public loader error
            raise ScenarioInvalid(
                f"{spec.scenario_id}: invalid initial_state.evidence_records[{index}]: "
                f"{exc}"
            ) from exc
        if (
            record.kind == "review_observation"
            and record.owner_id == record.subject_id
            and record.owner_id.startswith("buyer:")
            and "platform:reviews" in record.read_acl
        ):
            historical_reviewers.add(record.owner_id)
    prepared: list[tuple[dict[str, Any], str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(policy_rows):
        row_where = f"{where}.policies[{index}]"
        if not isinstance(value, Mapping) or set(value) != {
            "idempotency_key",
            "intent",
        }:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {row_where} fields must be exactly: "
                "idempotency_key, intent"
            )
        key = value.get("idempotency_key")
        intent = value.get("intent")
        if not isinstance(key, str) or not key.strip():
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {row_where}.idempotency_key must be non-empty"
            )
        if not isinstance(intent, Mapping) or any(
            not isinstance(field, str) for field in intent
        ):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {row_where}.intent must be an object with text keys"
            )
        kind = intent.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {row_where}.intent.kind must be non-empty"
            )
        try:
            authority = governance_policy_authority(kind)
        except Exception as exc:  # noqa: BLE001 - normalize public loader error
            raise ScenarioInvalid(
                f"{spec.scenario_id}: invalid {row_where}.intent.kind: {exc}"
            ) from exc
        if kind == "review_account_binding":
            reviewer_id = intent.get("reviewer_id")
            if (
                reviewer_id not in declared_buyers
                and reviewer_id not in historical_reviewers
            ):
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: {row_where} reviewer must be an active buyer "
                    "or a bound historical review-observation owner"
                )
        identity = (authority, key)
        if identity in seen:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: duplicate governance setup key {key!r} "
                f"for {authority}"
            )
        seen.add(identity)
        prepared.append((dict(intent), authority, key))

    try:
        for intent, authority, key in prepared:
            world.publish_governance_policy(
                intent,
                by_actor=authority,
                original_actor=authority,
                idempotency_key=key,
            )
    except Exception as exc:  # noqa: BLE001 - normalize public loader error
        raise ScenarioInvalid(
            f"{spec.scenario_id}: invalid {where}: {exc}"
        ) from exc


def _seed_pricing_policy_fixtures(world: Any, spec: ScenarioSpec) -> None:
    """Publish compact scenario policies through the normal World core API.

    The fixture may choose the authenticated merchant and idempotency key, but
    it cannot provide owner ids, revisions, logical ticks, predecessors, or
    digests. ``World.publish_pricing_policy`` derives and validates all of
    those fields against the catalog that was hydrated immediately before it.
    """

    if "pricing_policy_fixtures" not in spec.initial_state:
        return
    raw = spec.initial_state["pricing_policy_fixtures"]
    if not isinstance(raw, list):
        raise ScenarioInvalid(
            f"{spec.scenario_id}: initial_state.pricing_policy_fixtures must be a list"
        )
    seen: set[tuple[str, str]] = set()
    for index, fixture in enumerate(raw):
        where = f"initial_state.pricing_policy_fixtures[{index}]"
        if not isinstance(fixture, dict) or set(fixture) != {
            "merchant_id",
            "idempotency_key",
            "intent",
        }:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} requires exactly merchant_id, "
                "idempotency_key, and intent"
            )
        merchant_id = fixture["merchant_id"]
        key = fixture["idempotency_key"]
        intent = fixture["intent"]
        if (
            not isinstance(merchant_id, str)
            or not merchant_id.startswith("merchant:")
            or not isinstance(key, str)
            or not key.strip()
            or not isinstance(intent, dict)
        ):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} has invalid actor, key, or intent"
            )
        actor_key = (merchant_id, key)
        if actor_key in seen:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: duplicate pricing policy fixture key "
                f"{actor_key!r}"
            )
        try:
            world.publish_pricing_policy(
                intent,
                by_actor="platform:pricing",
                original_actor=merchant_id,
                idempotency_key=key,
            )
        except Exception as exc:  # noqa: BLE001 - normalize scenario boundary
            raise ScenarioInvalid(
                f"{spec.scenario_id}: invalid {where}: {exc}"
            ) from exc
        seen.add(actor_key)


def _seed_match_authorizations(world: Any, spec: ScenarioSpec) -> None:
    """Issue compact authorization fixtures through the core matching path.

    The fixture is intentionally unable to provide parties, price, quantity,
    revisions, logical time, certificate identity, or digests.  The World core
    derives those facts from already seeded authoritative rows and persists the
    complete session, offer, acceptance, and certificate chain.
    """

    from world.match_authorizations import issue_order_match_authorization

    if "match_authorizations" not in spec.initial_state:
        return
    raw = spec.initial_state["match_authorizations"]
    if not isinstance(raw, list):
        raise ScenarioInvalid(
            f"{spec.scenario_id}: initial_state.match_authorizations must be a list"
        )
    seen: set[str] = set()
    for index, fixture in enumerate(raw):
        if not isinstance(fixture, dict):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: initial_state.match_authorizations[{index}] "
                "must be a mapping"
            )
        authorization_id = fixture.get("authorization_id")
        if isinstance(authorization_id, str) and authorization_id in seen:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: duplicate match authorization id "
                f"{authorization_id!r}"
            )
        try:
            chain = issue_order_match_authorization(world, fixture)
        except Exception as exc:  # noqa: BLE001 - normalize the scenario boundary
            raise ScenarioInvalid(
                f"{spec.scenario_id}: invalid "
                f"initial_state.match_authorizations[{index}]: {exc}"
            ) from exc
        if chain.fixture.authorization_id in seen:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: duplicate match authorization id "
                f"{chain.fixture.authorization_id!r}"
            )
        seen.add(chain.fixture.authorization_id)


def _seed_evidence_contract_tables(world: Any, spec: ScenarioSpec) -> None:
    """Hydrate durable authority rows declared by a scenario.

    These records are CommerceWorld state, not benchmark-side fixtures.  They
    therefore enter through ``World.apply`` for both memory and database
    backends, where the normal cross-table ownership, digest, ACL, history, and
    logical-time validation runs.  Supplying an explicit empty block clears
    that table, matching the rest of the scenario seed contract.
    """

    from protocol.evidence_records import (
        coerce_evidence_record,
        coerce_mandate_revision,
    )
    from protocol.listing_claims import coerce_listing_claim
    from world.evidence_contracts import coerce_mandate_authority

    coercers = {
        "evidence_records": coerce_evidence_record,
        "mandate_authorities": coerce_mandate_authority,
        "mandate_revisions": coerce_mandate_revision,
        "listing_claims": coerce_listing_claim,
    }
    state: dict[str, list[Any]] = {}
    for table, coerce in coercers.items():
        if table not in spec.initial_state:
            continue
        raw = spec.initial_state[table]
        if raw is None:
            rows: list[Any] = []
        elif isinstance(raw, dict):
            rows = list(raw.values())
        elif isinstance(raw, list):
            rows = list(raw)
        else:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: initial_state.{table} must be a list or mapping"
            )
        hydrated: list[Any] = []
        for index, row in enumerate(rows):
            try:
                hydrated.append(coerce(row))
            except Exception as exc:  # noqa: BLE001 - normalize public loader error
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: invalid initial_state.{table}[{index}]: "
                    f"{exc}"
                ) from exc
        state[table] = hydrated
    if state:
        world.apply(state)


def _seeded_reputation(spec: ScenarioSpec) -> list[Any]:
    """Hydrate optional authoritative reputation rows from scenario state."""
    from world.types import AgentId, ReputationScore

    raw = spec.initial_state.get("reputation", []) or []
    rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
    known_merchants = {
        merchant.merchant_id for merchant in population_for_scenario(spec).merchants
    }
    output: list[ReputationScore] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        where = f"initial_state.reputation[{index}]"
        if not isinstance(row, dict):
            raise ScenarioInvalid(f"{spec.scenario_id}: {where} must be a mapping")
        merchant_id = str(row.get("merchant_id", ""))
        if merchant_id not in known_merchants:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} names unknown merchant {merchant_id!r}"
            )
        if merchant_id in seen:
            raise ScenarioInvalid(
                f"{spec.scenario_id}: duplicate reputation row for {merchant_id!r}"
            )
        rolling_avg = row.get("rolling_avg")
        n_settled = row.get("n_settled")
        n_disputed = row.get("n_disputed")
        if (
            isinstance(rolling_avg, bool)
            or not isinstance(rolling_avg, (int, float))
            or not 0 <= float(rolling_avg) <= 5
        ):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where}.rolling_avg must be in [0, 5]"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (n_settled, n_disputed)
        ):
            raise ScenarioInvalid(
                f"{spec.scenario_id}: {where} counts must be non-negative integers"
            )
        output.append(ReputationScore(
            merchant_id=AgentId(merchant_id),
            rolling_avg=float(rolling_avg),
            n_settled=int(n_settled),
            n_disputed=int(n_disputed),
        ))
        seen.add(merchant_id)
    return output


def _seed_social_tables(world: Any, spec: ScenarioSpec) -> None:
    """Seed optional authoritative social rows without replacing the catalog."""

    from world.types import AgentId, Friendship, Review, ReviewId, SkuId

    state: dict[str, Any] = {}
    if spec.initial_state.get("friendships"):
        state["friendships"] = [
            Friendship(buyer_id=AgentId(f["buyer_id"]),
                       friends=tuple(AgentId(x) for x in f["friends"]))
            for f in spec.initial_state["friendships"]
        ]
    if spec.initial_state.get("reviews"):
        state["reviews"] = [
            Review(ReviewId(r["review_id"]), AgentId(r["reviewer_id"]), SkuId(r["sku_id"]),
                   AgentId(r["merchant_id"]), int(r["rating"]), str(r.get("text", "")))
            for r in spec.initial_state["reviews"]
        ]
    if state:
        world.apply(state)


def _channel_from_factory(factory: Any, *, agent_id: str, role: str) -> Any:
    """Construct one actor-specific typed inference channel."""

    return factory(agent_id, role)


def _memory_from_initial_state(initial_state: dict[str, Any]) -> Any:
    """Build one actor-private memory store from declarative bucket mappings."""
    from agents.memory import InMemoryStore
    from agents.types import MemoryType

    memory = InMemoryStore()
    for raw_bucket, rows in sorted(initial_state.items(), key=lambda item: str(item[0])):
        try:
            bucket = MemoryType(str(raw_bucket).lower())
        except ValueError:
            try:
                bucket = MemoryType[str(raw_bucket).upper()]
            except KeyError as exc:
                raise ScenarioInvalid(
                    f"unknown actor initial_state memory bucket {raw_bucket!r}"
                ) from exc
        if not isinstance(rows, dict):
            raise ScenarioInvalid(
                f"actor initial_state bucket {raw_bucket!r} must be a mapping"
            )
        for key, value in sorted(rows.items(), key=lambda item: str(item[0])):
            memory.write(bucket, str(key), value)
    return memory


def build_agents(
    spec: ScenarioSpec,
    *,
    channels: Any,
    strict_skill_selection: bool = False,
) -> "list[Agent]":
    """Construct every buyer and merchant in deterministic id order.

    The platform is a service (built by the runner alongside the Runtime), not an
    Agent, so it is not returned here. Callers inject ``channels`` as an exact
    ``(agent_id, role) -> typed business-decision channel`` factory.
    """
    from agents.buyer import make_buyer_agent
    from agents.merchant import make_merchant_agent
    from agents.skill_selector_buyer import BuyerSkillSelector
    from agents.skill_selector_merchant import MerchantSkillSelector
    from agents.types import AgentInputs
    from episode.actor_evidence import actor_contexts_for_scenario

    population = population_for_scenario(spec)
    actor_contexts = actor_contexts_for_scenario(spec)
    report_roots_by_actor = {
        actor_id: frozenset(
            context.root_msg_id
            for context in actor_contexts
            if context.actor_id == actor_id
        )
        for actor_id in (
            *(buyer.buyer_id for buyer in population.buyers),
            *(merchant.merchant_id for merchant in population.merchants),
        )
    }
    root_principals_by_actor = {
        actor_id: {
            context.root_msg_id: context.principal_id
            for context in actor_contexts
            if context.actor_id == actor_id
        }
        for actor_id in (
            *(buyer.buyer_id for buyer in population.buyers),
            *(merchant.merchant_id for merchant in population.merchants),
        )
    }
    agents: list[Agent] = []
    for buyer in population.buyers:
        channel = _channel_from_factory(channels, agent_id=buyer.buyer_id, role="buyer")
        agents.append(make_buyer_agent(
            inputs=AgentInputs(persona=dict(buyer.persona), mandate=dict(buyer.mandate)),
            channel=channel,
            memory=_memory_from_initial_state(buyer.initial_state),
            agent_id=buyer.buyer_id,
            selector=BuyerSkillSelector(strict=strict_skill_selection),
            actor_report_root_msg_ids=report_roots_by_actor[buyer.buyer_id],
            semantic_search_limit=int(population.matching["top_k"]),
            semantic_root_principals=root_principals_by_actor[buyer.buyer_id],
        ))
    for merchant in population.merchants:
        channel = _channel_from_factory(
            channels,
            agent_id=merchant.merchant_id,
            role="merchant",
        )
        agents.append(make_merchant_agent(
            inputs=AgentInputs(persona=dict(merchant.persona), policy=dict(merchant.policy)),
            channel=channel,
            memory=_memory_from_initial_state(merchant.initial_state),
            agent_id=merchant.merchant_id,
            selector=MerchantSkillSelector(strict=strict_skill_selection),
            actor_report_root_msg_ids=report_roots_by_actor[merchant.merchant_id],
            semantic_root_principals=root_principals_by_actor[merchant.merchant_id],
        ))
    return agents


def kickoff_envelopes(spec: ScenarioSpec) -> "tuple[Envelope, ...]":
    """Return deterministic initial events for all population participants."""
    from protocol.envelope import Envelope

    population = population_for_scenario(spec)
    if population.initial_events:
        events: list[Envelope] = []
        for ordinal, raw in enumerate(population.initial_events):
            action = raw.get("action")
            if not isinstance(action, dict) or "kind" not in action:
                raise ScenarioInvalid(
                    f"{spec.scenario_id}: initial_events[{ordinal}].action must contain kind"
                )
            events.append(Envelope(
                msg_id=str(raw.get("msg_id", f"kickoff:{spec.scenario_id}:{ordinal}")),
                ts=str(raw.get("ts", "2026-06-04T12:00:00Z")),
                from_=str(raw.get("from", raw.get("from_", ""))),
                to=str(raw.get("to", "")),
                in_reply_to=(
                    str(raw["in_reply_to"]) if raw.get("in_reply_to") is not None else None
                ),
                idempotency_key=str(
                    raw.get("idempotency_key", f"kickoff:{spec.scenario_id}:{ordinal}")
                ),
                action=dict(action),
            ))
        return tuple(events)

    events = []
    legacy = spec.population is None
    for buyer in population.buyers:
        suffix = buyer.buyer_id.split(":", 1)[-1]
        principal = "consumer:persona" if legacy else f"consumer:{suffix}"
        marker = spec.scenario_id if legacy else f"{spec.scenario_id}:{buyer.buyer_id}"
        events.append(Envelope(
            msg_id=f"kickoff:{marker}",
            ts="2026-06-04T12:00:00Z",
            from_=principal,
            to=buyer.buyer_id,
            in_reply_to=None,
            idempotency_key=f"kickoff:{marker}",
            action={
                "kind": "delegate.create_purchase_mandate",
                "payload": dict(buyer.mandate),
            },
        ))
    return tuple(events)


def kickoff_envelope(spec: ScenarioSpec) -> "Envelope":
    """Backward-compatible singular kickoff for legacy 1x1 callers."""
    events = kickoff_envelopes(spec)
    if len(events) != 1:
        raise ScenarioInvalid(
            f"{spec.scenario_id}: expected one kickoff, population defines {len(events)}; "
            "use kickoff_envelopes()"
        )
    return events[0]
