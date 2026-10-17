"""Deterministic predicate scoring for large-catalog trajectories."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from large_catalog.models import LargeCatalogTask, ProcessReward, TaskResult


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalize(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.split())
    return value


_MISSING_FIELD_PATTERN = re.compile(
    r"^(?:"
    r"n a|na|none|null|unknown|unspecified|unavailable|"
    r"not (?:provided|available|specified|listed|stated|included|given|shown)|"
    r"(?:the )?[a-z0-9_ ]+ (?:is |are )?"
    r"not (?:provided|available|specified|listed|stated|included|given|shown)"
    r")"
    r"(?: in (?:the )?(?:current )?(?:listing )?record)?$"
)


def _is_missing_field_value(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return not normalized or _MISSING_FIELD_PATTERN.fullmatch(normalized) is not None


def _field_value_equal(actual: Any, expected: Any) -> bool:
    """Ignore punctuation-only edits while preserving words, values, and order."""

    if _is_missing_field_value(expected):
        return _is_missing_field_value(actual)
    if isinstance(actual, str) and isinstance(expected, str):
        actual_tokens = re.findall(r"\w+", actual.casefold())
        expected_tokens = re.findall(r"\w+", expected.casefold())
        return actual_tokens == expected_tokens
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _field_value_equal(actual[key], expected[key]) for key in expected
        )
    if (
        isinstance(actual, Sequence)
        and not isinstance(actual, (str, bytes))
        and isinstance(expected, Sequence)
        and not isinstance(expected, (str, bytes))
    ):
        return len(actual) == len(expected) and all(
            _field_value_equal(left, right)
            for left, right in zip(actual, expected)
        )
    return _normalize(actual) == _normalize(expected)


def _answer_content_matches(
    actual_fields: Any,
    expected_fields: Any,
) -> bool:
    return (
        isinstance(actual_fields, Mapping)
        and isinstance(expected_fields, Mapping)
        and set(actual_fields) == set(expected_fields)
        and _field_value_equal(actual_fields, expected_fields)
    )


def _answer_omits_unsupported_content(
    actual_fields: Any,
    expected_fields: Any,
) -> bool:
    if not isinstance(actual_fields, Mapping) or not isinstance(
        expected_fields, Mapping
    ):
        return False
    if not set(actual_fields).issubset(expected_fields):
        return False
    return all(
        _field_value_equal(actual, expected_fields[key])
        for key, actual in actual_fields.items()
    )


def _actual_result(observation: Mapping[str, Any]) -> Any:
    result = observation.get("result")
    if isinstance(result, Mapping) and "result" in result:
        return result["result"]
    return result


def _observed_refs(run: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    for observation in run.get("observations", []):
        if observation.get("intent") != "observe_listing":
            continue
        result = observation.get("result")
        if isinstance(result, Mapping) and "result" in result:
            result = result["result"]
        if isinstance(result, Mapping) and result.get("found"):
            listing = result.get("listing")
            if isinstance(listing, Mapping):
                refs.add(str(listing.get("listing_ref", "")))
    return refs


def _observed_records(run: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for observation in run.get("observations", []):
        if observation.get("intent") != "observe_listing":
            continue
        result = _actual_result(observation)
        if not isinstance(result, Mapping) or not result.get("found"):
            continue
        listing = result.get("listing")
        if isinstance(listing, Mapping):
            records[str(listing.get("listing_ref", ""))] = listing
    return records


def _claim_matches(
    task: LargeCatalogTask,
    run: Mapping[str, Any],
    actual: Any,
) -> bool:
    """Accept only extra claim fields that the observed records verify."""

    if not isinstance(actual, Mapping):
        return False
    expected = task.oracle.get("claim")
    if not isinstance(expected, Mapping):
        return False
    if any(
        key not in actual or _normalize(actual[key]) != _normalize(value)
        for key, value in expected.items()
    ):
        return False
    extras = set(actual) - set(expected)
    if not extras:
        return True

    kind = str(expected.get("type", ""))
    allowed_extras = {
        "lower_price": {
            "comparison_price_minor",
            "comparison_ref",
            "currency",
            "object_ref",
            "subject_price_minor",
        },
        "both_in_stock": {"value"},
        "price_difference_at_most": {
            "actual_difference_minor",
            "currency",
            "price_difference_minor",
        },
    }.get(kind, set())
    if not extras.issubset(allowed_extras):
        return False

    comparison = task.oracle.get("comparison")
    if not isinstance(comparison, Mapping):
        return False
    records = _observed_records(run)
    listing_refs = tuple(str(ref) for ref in comparison.get("listing_refs", ()))
    if any(ref not in records for ref in listing_refs):
        return False

    subject_ref = str(expected.get("subject_ref", ""))
    comparison_refs = [ref for ref in listing_refs if ref != subject_ref]
    checks: dict[str, bool] = {
        "actual_difference_minor": actual.get("actual_difference_minor")
        == comparison.get("price_difference_minor"),
        "value": actual.get("value") == comparison.get("both_in_stock"),
    }
    if subject_ref in records:
        checks["subject_price_minor"] = (
            actual.get("subject_price_minor")
            == records[subject_ref].get("price_minor")
        )
    if len(comparison_refs) == 1:
        comparison_ref = comparison_refs[0]
        checks["comparison_ref"] = actual.get("comparison_ref") == comparison_ref
        checks["object_ref"] = actual.get("object_ref") == comparison_ref
        checks["comparison_price_minor"] = (
            actual.get("comparison_price_minor")
            == records[comparison_ref].get("price_minor")
        )
    checks["price_difference_minor"] = (
        actual.get("price_difference_minor")
        == comparison.get("price_difference_minor")
    )
    currencies = {records[ref].get("currency") for ref in listing_refs}
    checks["currency"] = len(currencies) == 1 and actual.get("currency") in currencies
    return all(checks.get(key, False) for key in extras)


def _listing_refs_in(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, Mapping):
        listing_ref = value.get("listing_ref")
        if isinstance(listing_ref, str) and listing_ref:
            refs.add(listing_ref)
        for item in value.values():
            refs.update(_listing_refs_in(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            refs.update(_listing_refs_in(item))
    return refs


def _summary_refs(observations: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return listing IDs exposed by authoritative search or cart summaries."""

    refs: set[str] = set()
    for observation in observations:
        refs.update(_listing_refs_in(observation.get("result")))
    return refs


def _requires_full_listing_record(oracle_kind: str) -> bool:
    return oracle_kind in {"answer", "comparison", "claim", "quote"}


def _resolve_evidence_refs(
    run: Mapping[str, Any],
    evidence_refs: set[str],
) -> set[str]:
    """Resolve both listing IDs and runtime-issued observation IDs."""

    resolved = set(evidence_refs)
    for observation in run.get("observations", []):
        observation_ref = observation.get("observation_ref")
        if observation_ref in evidence_refs:
            resolved.update(_listing_refs_in(observation.get("result")))
    return resolved


def _search_requirements(value: Any) -> Any:
    """Compare the fields that affect search, excluding display-only labels."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return _normalize(value)
    return [
        {
            "query": str(item.get("query", "")),
            "filters": _normalize(item.get("filters", {})),
        }
        for item in value
        if isinstance(item, Mapping)
    ]


def _valid_rejection_reason(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    return normalized.startswith(("no_", "quoted_")) or normalized in {
        "budget_exceeded",
        "exceeds_budget",
        "over_budget",
        "unsatisfiable_constraints",
    }


def _searches(run: Mapping[str, Any], intent: str) -> list[Mapping[str, Any]]:
    return [
        row
        for row in run.get("observations", [])
        if row.get("intent") == intent
    ]


def _final_refs(intent: str | None, arguments: Mapping[str, Any]) -> tuple[str, ...]:
    if intent == "select_listing":
        return (str(arguments.get("listing_ref", "")),)
    lines = arguments.get("lines")
    if isinstance(lines, Sequence) and not isinstance(lines, (str, bytes)):
        return tuple(
            str(line.get("listing_ref", ""))
            for line in lines
            if isinstance(line, Mapping)
        )
    refs = arguments.get("listing_refs")
    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        return tuple(str(ref) for ref in refs)
    if "listing_ref" in arguments:
        return (str(arguments["listing_ref"]),)
    return ()


def _accepted_refs(task: LargeCatalogTask) -> tuple[str, ...]:
    return tuple(str(ref) for ref in task.oracle.get("accepted_refs", ()))


def _accepted_carts(task: LargeCatalogTask) -> set[tuple[str, ...]]:
    return {
        tuple(str(ref) for ref in cart)
        for cart in task.oracle.get("accepted_carts", ())
    }


def score_run(
    task: LargeCatalogTask,
    run: Mapping[str, Any],
    *,
    model_id: str,
) -> TaskResult:
    """Score one completed or failed run without any model call."""

    final_intent = run.get("final_intent")
    raw_arguments = run.get("final_arguments")
    arguments = dict(raw_arguments) if isinstance(raw_arguments, Mapping) else {}
    oracle_kind = str(task.oracle.get("kind", ""))
    observed = _observed_refs(run)
    final_refs = _final_refs(
        str(final_intent) if final_intent is not None else None,
        arguments,
    )
    evidence_refs = {
        str(ref)
        for ref in arguments.get("evidence_refs", ())
        if isinstance(ref, str)
    }
    resolved_evidence_refs = _resolve_evidence_refs(run, evidence_refs)
    accepted_refs = set(_accepted_refs(task))
    accepted_carts = _accepted_carts(task)
    expected_reject = oracle_kind in {"selection", "cart"} and not (
        accepted_refs or accepted_carts
    )
    if oracle_kind == "quote_decision":
        expected_intent = str(task.oracle["expected_action"])
    elif oracle_kind == "claim":
        expected_intent = str(task.oracle["expected_action"])
    elif expected_reject:
        expected_intent = "reject_purchase"
    elif oracle_kind == "answer":
        expected_intent = "submit_product_answer"
    elif oracle_kind == "comparison":
        expected_intent = "submit_comparison"
    elif oracle_kind == "quote":
        expected_intent = "submit_quote"
    elif oracle_kind == "cart":
        expected_intent = "submit_cart"
    else:
        expected_intent = "select_listing"

    catalog_searches = _searches(run, "search_catalog")
    cart_searches = _searches(run, "search_cart_candidates")
    matching_catalog_searches: list[Mapping[str, Any]] = []
    matching_cart_searches: list[Mapping[str, Any]] = []
    if "requirements" in task.public_context:
        target_requirements = _search_requirements(task.public_context["requirements"])
        target_constraints = _normalize(task.public_context.get("constraints", {}))
        matching_cart_searches = [
            row
            for row in cart_searches
            if _search_requirements(row.get("arguments", {}).get("requirements"))
            == target_requirements
            and _normalize(row.get("arguments", {}).get("constraints", {}))
            == target_constraints
        ]
        search_preserves = bool(matching_cart_searches)
        search_supports = bool(matching_cart_searches)
    elif (
        "listing_refs" in task.public_context or "quote_lines" in task.public_context
    ) and not catalog_searches:
        # Named-record tasks begin with public references and need no discovery.
        search_preserves = True
        search_supports = True
    else:
        target_query = str(task.public_context.get("query", ""))
        raw_target_filters = dict(task.public_context.get("filters", {}))
        target_filters = _normalize(raw_target_filters)
        base_searches = [
            row
            for row in catalog_searches
            if str(row.get("arguments", {}).get("query", "")) == target_query
            and _normalize(row.get("arguments", {}).get("filters", {}))
            == target_filters
        ]
        matching_catalog_searches = list(base_searches)
        preference = dict(
            task.public_context.get("preference")
            or task.public_context.get("latest_preference")
            or {}
        )
        preference_kind = str(preference.get("kind", ""))
        if preference_kind in {
            "category_then_price",
            "feature_then_price",
            "merchant_then_price",
        }:
            preferred_filters = dict(raw_target_filters)
            if preference_kind == "category_then_price":
                preferred_filters["category"] = str(preference.get("category", ""))
            elif preference_kind == "feature_then_price":
                preferred_filters["required_features"] = [
                    str(preference.get("feature", ""))
                ]
            else:
                preferred_filters["merchant"] = str(preference.get("merchant", ""))
            preferred_searches = [
                row
                for row in catalog_searches
                if str(row.get("arguments", {}).get("query", "")) == target_query
                and _normalize(row.get("arguments", {}).get("filters", {}))
                == _normalize(preferred_filters)
            ]
            if any(_page_items(row) for row in preferred_searches):
                matching_catalog_searches.extend(preferred_searches)
        search_preserves = bool(matching_catalog_searches)
        search_supports = any(
            row.get("arguments", {}).get("sort", "price_asc") == "price_asc"
            for row in matching_catalog_searches
        )

    summary_refs = _summary_refs([*catalog_searches, *cart_searches])
    visible_decisive_refs = observed
    if not _requires_full_listing_record(oracle_kind):
        visible_decisive_refs = observed | summary_refs
    decisive = set(final_refs)
    if expected_reject:
        decisive_records = search_supports
        evidence_matches = search_supports
    else:
        decisive_records = bool(decisive) and decisive.issubset(visible_decisive_refs)
        evidence_matches = bool(decisive) and decisive.issubset(
            resolved_evidence_refs
        )

    hard_ok = False
    preference_ok = False
    oracle_ok = False
    structured_ok = False
    unsupported_omitted = False
    answer_targets_requested_listing = False
    answer_field_checks: dict[str, bool] = {}
    if oracle_kind == "selection":
        chosen = final_refs[0] if len(final_refs) == 1 else ""
        oracle_ok = final_intent == expected_intent and (
            (expected_reject and final_intent == "reject_purchase")
            or chosen in accepted_refs
        )
        hard_ok = oracle_ok or (
            chosen
            and chosen
            in {
                str(item.get("listing_ref"))
                for row in matching_catalog_searches
                for item in _page_items(row)
            }
        )
        preference_ok = oracle_ok
    elif oracle_kind == "cart":
        cart = tuple(final_refs)
        oracle_ok = final_intent == expected_intent and (
            (expected_reject and final_intent == "reject_purchase")
            or cart in accepted_carts
        )
        visible_carts = {
            tuple(
                str(line.get("listing_ref", ""))
                for line in item.get("lines", ())
                if isinstance(line, Mapping)
            )
            for row in matching_cart_searches
            for item in _page_items(row)
        }
        hard_ok = oracle_ok or cart in visible_carts
        preference_ok = oracle_ok
    elif oracle_kind == "quote_decision":
        quote_rows = _searches(run, "request_cart_quote")
        quote: Mapping[str, Any] | None = None
        if quote_rows:
            result = quote_rows[-1].get("result")
            if isinstance(result, Mapping) and "world_result" in result:
                world_result = result["world_result"]
                if isinstance(world_result, Mapping) and isinstance(
                    world_result.get("quote"), Mapping
                ):
                    quote = world_result["quote"]
        quote_cart = (
            tuple(str(line.get("listing_ref", "")) for line in quote.get("lines", []))
            if quote is not None
            else ()
        )
        quote_ok = (
            quote is not None
            and quote_cart in accepted_carts
            and int(quote.get("total_minor", -1))
            == int(task.oracle["quote_total_minor"])
        )
        oracle_ok = final_intent == expected_intent and quote_ok
        hard_ok = quote_ok
        preference_ok = oracle_ok
        if final_intent in {"checkout_quote", "decline_quote"}:
            decisive_records = True
            quote_evidence: set[str] = set()
            if quote_rows:
                quote_evidence = {
                    str(ref)
                    for ref in quote_rows[-1]
                    .get("arguments", {})
                    .get("evidence_refs", ())
                    if isinstance(ref, str)
                }
            evidence_matches = quote_ok and set(quote_cart).issubset(
                _resolve_evidence_refs(run, quote_evidence)
            )
    elif oracle_kind == "answer":
        actual_fields = arguments.get("fields")
        expected_fields = task.oracle["expected_fields"]
        answer_targets_requested_listing = (
            final_intent == expected_intent
            and str(arguments.get("listing_ref", ""))
            == str(task.oracle["listing_ref"])
        )
        if isinstance(expected_fields, Mapping):
            answer_field_checks = {
                f"answer_field_correct.{field}": (
                    isinstance(actual_fields, Mapping)
                    and field in actual_fields
                    and _field_value_equal(actual_fields[field], expected)
                )
                for field, expected in expected_fields.items()
            }
        structured_ok = _answer_content_matches(actual_fields, expected_fields)
        unsupported_omitted = _answer_omits_unsupported_content(
            actual_fields,
            expected_fields,
        )
        oracle_ok = answer_targets_requested_listing and structured_ok
    elif oracle_kind == "comparison":
        expected = {
            key: value
            for key, value in task.oracle.items()
            if key
            in {
                "lower_price_ref",
                "same_category",
                "both_in_stock",
                "price_difference_minor",
                "same_name",
                "same_variant",
            }
        }
        results = arguments.get("results", {})
        structured_ok = (
            final_intent == expected_intent
            and all(_normalize(results.get(key)) == _normalize(value) for key, value in expected.items())
        )
        unsupported_omitted = structured_ok
        oracle_ok = structured_ok
    elif oracle_kind == "claim":
        structured_ok = final_intent == expected_intent
        if final_intent == "publish_comparison_claim":
            structured_ok = structured_ok and _claim_matches(
                task,
                run,
                arguments.get("claim", {}),
            )
        unsupported_omitted = structured_ok
        oracle_ok = structured_ok
    elif oracle_kind == "quote":
        expected = {
            key: task.oracle[key]
            for key in (
                "lines",
                "subtotal_minor",
                "discount_minor",
                "fee_minor",
                "total_minor",
            )
        }
        actual = {
            key: arguments.get(key)
            for key in (
                "lines",
                "subtotal_minor",
                "discount_minor",
                "fee_minor",
                "total_minor",
            )
        }
        structured_ok = final_intent == expected_intent and _normalize(actual) == _normalize(expected)
        unsupported_omitted = structured_ok
        oracle_ok = structured_ok

    platform_accepts = any(
        row.get("stage") == "validation"
        and row.get("intent") == final_intent
        and row.get("accepted") is True
        for row in run.get("trace", [])
    )
    effects = [
        row
        for row in run.get("effects", [])
        if row.get("intent") == final_intent
    ]
    world_matches_decision = bool(effects)
    world_matches_oracle = world_matches_decision and oracle_ok

    checks = {
        "search_preserves_constraints": search_preserves,
        "search_supports_global_judgment": search_supports,
        "decisive_records_observed": decisive_records,
        "evidence_matches_decision": evidence_matches,
        "hard_constraints_satisfied": hard_ok,
        "preference_logic_correct": preference_ok,
        "infeasibility_correct": expected_reject and final_intent == "reject_purchase",
        "rejection_reason_correct": expected_reject
        and _valid_rejection_reason(arguments.get("reason_code", "")),
        "structured_content_correct": structured_ok,
        "answer_targets_requested_listing": answer_targets_requested_listing,
        "unsupported_content_omitted": unsupported_omitted,
        "decision_matches_oracle": oracle_ok,
        "platform_accepts_valid_action": platform_accepts,
        "world_effect_matches_decision": world_matches_decision,
        "world_effect_matches_oracle": world_matches_oracle,
    }
    checks.update(answer_field_checks)
    rewards = tuple(
        ProcessReward(
            stage=predicate.stage,
            predicate=predicate.predicate_id,
            points=int(bool(checks.get(predicate.predicate_id, False))),
            maximum=1,
            event_ref=_event_for_stage(run, predicate.stage),
        )
        for predicate in task.predicates
    )
    points = sum(row.points for row in rewards)
    maximum = sum(row.maximum for row in rewards)
    score = points / maximum if maximum else 0.0
    strict = all(
        reward.points == reward.maximum
        for predicate, reward in zip(task.predicates, rewards)
        if predicate.mandatory
    )
    return TaskResult(
        task_id=task.task_id,
        model_id=model_id,
        role=task.role,
        score=score,
        strict_success=strict,
        terminal=str(run.get("terminal", "unknown")),
        process_rewards=rewards,
        trace=tuple(run.get("trace", ())),
        model_calls=int(run.get("model_calls", 0)),
        latency_seconds=float(run.get("latency_seconds", 0.0)),
        error=(
            None
            if run.get("terminal") == "completed"
            else str(run.get("protocol_error") or run.get("terminal"))
        ),
    )


def _page_items(observation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = observation.get("result")
    if isinstance(result, Mapping) and "result" in result:
        result = result["result"]
    if not isinstance(result, Mapping):
        return []
    items = result.get("items", [])
    return [item for item in items if isinstance(item, Mapping)]


def _event_for_stage(run: Mapping[str, Any], stage: str) -> str | None:
    if stage == "search":
        for row in reversed(run.get("trace", [])):
            if row.get("stage") == "decision" and row.get("intent") in {
                "search_catalog",
                "search_cart_candidates",
            }:
                value = row.get("event_ref")
                return str(value) if value is not None else None
    for row in reversed(run.get("trace", [])):
        if row.get("stage") == stage:
            value = row.get("event_ref")
            return str(value) if value is not None else None
    if stage == "evidence":
        for row in reversed(run.get("trace", [])):
            if row.get("stage") == "decision":
                value = row.get("event_ref")
                return str(value) if value is not None else None
    return None
