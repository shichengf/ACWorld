"""Deterministic reference and error policies used before paid evaluation."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from large_catalog.models import LargeCatalogTask
from large_catalog.reference import ReferencePolicy


ERROR_POLICY_NAMES = (
    "first_result",
    "constraint_blind_cheapest",
    "stock_blind_cheapest",
    "unconditional_preference",
    "premature_fallback",
    "merged_variants",
    "greedy_cart",
    "partial_cart",
    "evidence_free",
    "over_budget_checkout",
    "premature_reject",
    "force_action",
)


class FixedErrorPolicy:
    """Inject one deterministic mistake while preserving the normal runtime path."""

    def __init__(self, task: LargeCatalogTask, mode: str) -> None:
        if mode not in ERROR_POLICY_NAMES:
            raise ValueError(f"unknown fixed error policy: {mode}")
        self.task = task
        self.mode = mode
        self.reference = ReferencePolicy()
        self.calls = 0
        self._stopped = False

    def decide(
        self,
        task: LargeCatalogTask,
        observations: Sequence[Mapping[str, Any]],
        allowed_intents: Sequence[str],
        *,
        decision_id: str,
    ) -> Mapping[str, Any]:
        self.calls += 1
        if self.mode == "premature_reject" and not observations:
            if "reject_purchase" in allowed_intents:
                return {
                    "intent": "reject_purchase",
                    "arguments": {"reason_code": "gave_up_early", "evidence_refs": []},
                }
            return {"intent": "not_an_allowed_intent", "arguments": {}}
        if self.mode == "evidence_free" and not observations:
            replacement = dict(_oracle_decision(task))
            replacement_arguments = dict(replacement.get("arguments", {}))
            replacement_arguments["evidence_refs"] = []
            return {
                "intent": replacement["intent"],
                "arguments": replacement_arguments,
            }
        decision = dict(
            self.reference.decide(
                task,
                observations,
                allowed_intents,
                decision_id=decision_id,
            )
        )
        intent = str(decision["intent"])
        arguments = dict(decision["arguments"])
        if self.mode == "evidence_free" and intent == "observe_listing":
            replacement = dict(_oracle_decision(task))
            replacement_arguments = dict(replacement.get("arguments", {}))
            replacement_arguments["evidence_refs"] = []
            return {"intent": replacement["intent"], "arguments": replacement_arguments}

        if intent == "search_catalog" and self.mode in {
            "first_result",
            "constraint_blind_cheapest",
            "stock_blind_cheapest",
        }:
            filters = dict(arguments.get("filters", {}))
            if self.mode == "first_result":
                arguments["sort"] = "name_asc"
            elif self.mode == "constraint_blind_cheapest":
                filters = {}
            else:
                filters.pop("in_stock", None)
            arguments["filters"] = filters
            return {"intent": intent, "arguments": arguments}

        if intent == "observe_listing" and self.mode in {
            "first_result",
            "constraint_blind_cheapest",
            "stock_blind_cheapest",
            "premature_fallback",
        }:
            alternative = _first_visible_ref(observations)
            if alternative:
                arguments["listing_ref"] = alternative
            return {"intent": intent, "arguments": arguments}

        if intent == "request_cart_quote" and self.mode in {
            "greedy_cart",
            "partial_cart",
            "stock_blind_cheapest",
            "evidence_free",
        }:
            if self.mode == "evidence_free":
                arguments["evidence_refs"] = []
            lines = list(arguments.get("lines", []))
            if lines and self.mode in {"greedy_cart", "partial_cart"}:
                arguments["lines"] = (
                    list(reversed(lines)) if self.mode == "greedy_cart" else lines[:-1]
                )
            if self.mode == "stock_blind_cheapest":
                arguments["evidence_refs"] = []
            return {"intent": intent, "arguments": arguments}

        if intent in {
            "select_listing",
            "reject_purchase",
            "submit_product_answer",
            "submit_comparison",
            "submit_cart",
            "checkout_quote",
            "decline_quote",
            "publish_comparison_claim",
            "decline_comparison_claim",
            "submit_quote",
        }:
            if self.mode == "evidence_free":
                arguments["evidence_refs"] = []
            return self._mutate_final(intent, arguments, observations)
        return {"intent": intent, "arguments": arguments}

    def _mutate_final(
        self,
        intent: str,
        arguments: dict[str, Any],
        observations: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        mode = self.mode
        if (
            mode == "first_result"
            and self.task.scenario == "D4"
            and intent in {"checkout_quote", "decline_quote"}
        ):
            arguments["quote_ref"] = "quote:unknown"
        if mode == "constraint_blind_cheapest" and intent != "select_listing":
            if intent == "submit_product_answer":
                fields = dict(arguments.get("fields", {}))
                if fields:
                    fields.pop(next(iter(fields)))
                arguments["fields"] = fields
            elif intent == "submit_comparison":
                results = dict(arguments.get("results", {}))
                if results:
                    results.pop(next(iter(results)))
                arguments["results"] = results
            elif intent == "submit_quote":
                arguments["subtotal_minor"] = int(arguments.get("subtotal_minor", 0)) + 1
            elif intent in {"publish_comparison_claim", "decline_comparison_claim"}:
                arguments["evidence_refs"] = []
        elif mode == "stock_blind_cheapest" and intent != "select_listing":
            arguments["evidence_refs"] = ["merchant:wrong:evidence"]
        if mode in {
            "first_result",
            "constraint_blind_cheapest",
            "stock_blind_cheapest",
            "premature_fallback",
        } and intent == "select_listing":
            alternative = _first_visible_ref(observations)
            if alternative:
                arguments["listing_ref"] = alternative
                arguments["evidence_refs"] = [alternative]
        elif mode == "first_result" and intent == "submit_cart":
            lines = list(arguments.get("lines", []))
            arguments["lines"] = list(reversed(lines))
        elif mode == "unconditional_preference":
            if intent == "select_listing":
                visible = _all_visible_refs(observations)
                accepted = set(self.task.oracle.get("accepted_refs", ()))
                alternative = next(
                    (ref for ref in reversed(visible) if ref not in accepted),
                    None,
                )
                if alternative:
                    arguments["listing_ref"] = alternative
                    arguments["evidence_refs"] = [alternative]
            elif intent == "reject_purchase":
                return {"intent": "not_an_allowed_intent", "arguments": {}}
        elif mode == "merged_variants" and intent == "submit_comparison":
            results = dict(arguments.get("results", {}))
            results["same_variant"] = True
            results["variants"] = ["same", "same"]
            arguments["results"] = results
        elif mode == "merged_variants" and intent == "submit_product_answer":
            fields = dict(arguments.get("fields", {}))
            if fields:
                first = next(iter(fields))
                fields[first] = _wrong_value(fields[first])
            arguments["fields"] = fields
        elif mode == "merged_variants" and intent in {
            "publish_comparison_claim",
            "decline_comparison_claim",
        }:
            opposite = (
                "decline_comparison_claim"
                if intent == "publish_comparison_claim"
                else "publish_comparison_claim"
            )
            replacement: dict[str, Any] = {
                "listing_refs": arguments.get("listing_refs", []),
                "evidence_refs": arguments.get("evidence_refs", []),
            }
            if opposite == "publish_comparison_claim":
                replacement["claim"] = dict(
                    self.task.public_context.get("requested_claim", {})
                )
            else:
                replacement["reason_code"] = "unsupported"
            return {"intent": opposite, "arguments": replacement}
        elif mode in {"greedy_cart", "partial_cart"} and intent == "submit_cart":
            lines = list(arguments.get("lines", []))
            if lines:
                arguments["lines"] = lines[:-1] if mode == "partial_cart" else list(reversed(lines))
        elif mode == "over_budget_checkout":
            if intent == "decline_quote":
                return {
                    "intent": "checkout_quote",
                    "arguments": {"quote_ref": arguments.get("quote_ref", "")},
                }
            if intent == "submit_quote":
                arguments["total_minor"] = int(arguments.get("total_minor", 0)) + 1
        elif mode == "force_action":
            if intent == "decline_comparison_claim":
                return {
                    "intent": "publish_comparison_claim",
                    "arguments": {
                        "listing_refs": arguments.get("listing_refs", []),
                        "claim": dict(self.task.public_context.get("requested_claim", {})),
                        "evidence_refs": arguments.get("evidence_refs", []),
                    },
                }
            if intent == "reject_purchase":
                return {"intent": "not_an_allowed_intent", "arguments": {}}
            if intent == "submit_product_answer":
                fields = dict(arguments.get("fields", {}))
                if fields:
                    first = next(iter(fields))
                    fields[first] = _wrong_value(fields[first])
                arguments["fields"] = fields
                arguments["listing_ref"] = "merchant:unknown:row:0"
            elif intent == "submit_comparison":
                results = dict(arguments.get("results", {}))
                if results:
                    first = next(iter(results))
                    results[first] = _wrong_value(results[first])
                arguments["results"] = results
            elif intent == "submit_quote":
                arguments["total_minor"] = int(arguments.get("total_minor", 0)) + 1
            elif intent == "select_listing":
                arguments["listing_ref"] = "merchant:unknown:row:0"
        if mode == "premature_fallback" and intent == "reject_purchase":
            arguments["reason_code"] = "preferred_category_missing"
        return {"intent": intent, "arguments": arguments}


def _oracle_decision(task: LargeCatalogTask) -> Mapping[str, Any]:
    """Private validation policy used only to create evidence-free mutations."""

    kind = str(task.oracle.get("kind", ""))
    if kind == "selection":
        refs = list(task.oracle.get("accepted_refs", []))
        if not refs:
            return {
                "intent": "reject_purchase",
                "arguments": {"reason_code": "no_eligible_listing", "evidence_refs": []},
            }
        return {
            "intent": "select_listing",
            "arguments": {"listing_ref": refs[0], "evidence_refs": []},
        }
    if kind == "cart":
        carts = list(task.oracle.get("accepted_carts", []))
        if not carts:
            return {
                "intent": "reject_purchase",
                "arguments": {"reason_code": "no_feasible_cart", "evidence_refs": []},
            }
        return {
            "intent": "submit_cart",
            "arguments": {
                "lines": [
                    {"listing_ref": ref, "quantity": 1} for ref in carts[0]
                ],
                "evidence_refs": [],
            },
        }
    if kind == "answer":
        return {
            "intent": "submit_product_answer",
            "arguments": {
                "listing_ref": task.oracle["listing_ref"],
                "fields": copy.deepcopy(task.oracle["expected_fields"]),
                "evidence_refs": [],
            },
        }
    if kind == "comparison":
        keys = {
            "lower_price_ref",
            "same_category",
            "both_in_stock",
            "price_difference_minor",
            "same_name",
            "same_variant",
        }
        return {
            "intent": "submit_comparison",
            "arguments": {
                "listing_refs": list(task.oracle["listing_refs"]),
                "results": {
                    key: copy.deepcopy(value)
                    for key, value in task.oracle.items()
                    if key in keys
                },
                "evidence_refs": [],
            },
        }
    if kind == "claim":
        intent = str(task.oracle["expected_action"])
        arguments = {
            "listing_refs": list(task.public_context["listing_refs"]),
            "evidence_refs": [],
        }
        if intent == "publish_comparison_claim":
            arguments["claim"] = copy.deepcopy(task.oracle["claim"])
        else:
            arguments["reason_code"] = "catalog_evidence_does_not_support_claim"
        return {"intent": intent, "arguments": arguments}
    if kind == "quote":
        return {
            "intent": "submit_quote",
            "arguments": {
                key: copy.deepcopy(task.oracle[key])
                for key in (
                    "lines",
                    "subtotal_minor",
                    "discount_minor",
                    "fee_minor",
                    "total_minor",
                )
            }
            | {"evidence_refs": []},
        }
    # Quote decisions need the quote reference, so a direct evidence-free
    # action is intentionally invalid and scores as a protocol stop.
    return {"intent": "not_an_allowed_intent", "arguments": {}}


def _all_visible_refs(observations: Sequence[Mapping[str, Any]]) -> list[str]:
    refs: list[str] = []
    for observation in observations:
        result = observation.get("result")
        if isinstance(result, Mapping) and "result" in result:
            result = result["result"]
        if not isinstance(result, Mapping):
            continue
        for item in result.get("items", []):
            if isinstance(item, Mapping) and "listing_ref" in item:
                refs.append(str(item["listing_ref"]))
    return refs


def _first_visible_ref(observations: Sequence[Mapping[str, Any]]) -> str | None:
    refs = _all_visible_refs(observations)
    return refs[0] if refs else None


def _wrong_value(value: Any) -> Any:
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if value is None:
        return "unsupported"
    if isinstance(value, str):
        return value + " (unverified)"
    return None
