"""Model-visible reference policy for the large-catalog suite."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from large_catalog.models import LargeCatalogTask


class ReferencePolicyError(RuntimeError):
    """The public tools did not expose enough information to solve a task."""


def _actual_result(observation: Mapping[str, Any]) -> Any:
    outer = observation.get("result")
    if isinstance(outer, Mapping) and "result" in outer:
        return outer["result"]
    return outer


def _observations_for(
    observations: Sequence[Mapping[str, Any]],
    intent: str,
) -> list[Mapping[str, Any]]:
    return [row for row in observations if row.get("intent") == intent]


class ReferencePolicy:
    """Solve tasks using only task-visible context and returned observations."""

    calls = 0

    def decide(
        self,
        task: LargeCatalogTask,
        observations: Sequence[Mapping[str, Any]],
        allowed_intents: Sequence[str],
        *,
        decision_id: str,
    ) -> Mapping[str, Any]:
        del allowed_intents, decision_id
        self.calls += 1
        context = dict(task.public_context)
        if "requirements" in context:
            return self._cart(task, context, observations)
        if "quote_lines" in context:
            return self._merchant_quote(context, observations)
        refs = tuple(str(ref) for ref in context.get("listing_refs", ()))
        if refs:
            missing = [
                ref
                for ref in refs
                if ref not in self._observed_listing_records(observations)
            ]
            if missing:
                return {
                    "intent": "observe_listing",
                    "arguments": {"listing_ref": missing[0]},
                }
            if "requested_fields" in context:
                return self._answer(context, observations)
            if "requested_claim" in context:
                return self._claim(context, observations)
            return self._comparison(task, refs, observations)
        return self._selection(task, context, observations)

    def _selection(
        self,
        task: LargeCatalogTask,
        context: Mapping[str, Any],
        observations: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        query = str(context["query"])
        filters = dict(context["filters"])
        preference = dict(
            context.get("preference") or context.get("latest_preference") or {}
        )
        searches = _observations_for(observations, "search_catalog")

        desired_filters = dict(filters)
        preference_kind = preference.get("kind")
        if preference_kind in {"category_then_price", "category_premium"}:
            desired_filters["category"] = str(preference.get("category", ""))
        elif preference_kind == "feature_then_price":
            desired_filters["required_features"] = [str(preference.get("feature", ""))]
        elif preference_kind == "merchant_then_price":
            desired_filters["merchant"] = str(preference.get("merchant", ""))

        preferred_page = self._search_page(searches, query, desired_filters)
        general_page = self._search_page(searches, query, filters)
        if preference_kind and desired_filters != filters and preferred_page is None:
            return self._search(query, desired_filters)
        if general_page is None:
            return self._search(query, filters)

        preferred_items = [] if preferred_page is None else preferred_page.get("items", [])
        general_items = general_page.get("items", [])
        selected: Mapping[str, Any] | None = None
        if preference_kind == "category_premium" and preferred_items and general_items:
            premium = int(preference.get("premium_minor", 0))
            if int(preferred_items[0]["price_minor"]) <= int(general_items[0]["price_minor"]) + premium:
                selected = preferred_items[0]
            else:
                selected = general_items[0]
        elif preferred_items:
            selected = preferred_items[0]
        elif general_items:
            selected = general_items[0]
        if selected is None:
            return {
                "intent": "reject_purchase",
                "arguments": {
                    "reason_code": "no_eligible_listing",
                    "evidence_refs": [],
                },
            }
        ref = str(selected["listing_ref"])
        return {
            "intent": "select_listing",
            "arguments": {"listing_ref": ref, "evidence_refs": [ref]},
        }

    @staticmethod
    def _search(query: str, filters: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "intent": "search_catalog",
            "arguments": {
                "query": query,
                "filters": dict(filters),
                "sort": "price_asc",
                "cursor": None,
                "page_size": 20,
            },
        }

    @staticmethod
    def _search_page(
        searches: Sequence[Mapping[str, Any]],
        query: str,
        filters: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        for row in searches:
            if row.get("arguments", {}).get("query") != query:
                continue
            if dict(row.get("arguments", {}).get("filters", {})) != dict(filters):
                continue
            result = _actual_result(row)
            if isinstance(result, Mapping):
                return result
        return None

    def _answer(
        self,
        context: Mapping[str, Any],
        observations: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        ref = str(context["listing_refs"][0])
        record = self._observed_listing_records(observations)[ref]
        fields = {
            field: record.get(field) if record.get(field) not in {"", None} else None
            for field in context["requested_fields"]
        }
        return {
            "intent": "submit_product_answer",
            "arguments": {
                "listing_ref": ref,
                "fields": fields,
                "evidence_refs": [ref],
            },
        }

    def _comparison(
        self,
        task: LargeCatalogTask,
        refs: Sequence[str],
        observations: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        records = self._observed_listing_records(observations)
        left, right = (records[ref] for ref in refs)
        results = {
            "lower_price_ref": (
                refs[0]
                if int(left["price_minor"]) < int(right["price_minor"])
                else refs[1]
                if int(right["price_minor"]) < int(left["price_minor"])
                else None
            ),
            "same_category": str(left["category"]).casefold()
            == str(right["category"]).casefold(),
            "both_in_stock": bool(left["in_stock"]) and bool(right["in_stock"]),
            "price_difference_minor": abs(
                int(left["price_minor"]) - int(right["price_minor"])
            ),
            "same_name": str(left["name"]).casefold() == str(right["name"]).casefold(),
            "same_variant": str(left["variant"]).casefold()
            == str(right["variant"]).casefold(),
        }
        if task.scenario == "B3":
            results["variants"] = [left["variant"], right["variant"]]
            results["prices_minor"] = [left["price_minor"], right["price_minor"]]
        return {
            "intent": "submit_comparison",
            "arguments": {
                "listing_refs": list(refs),
                "results": results,
                "evidence_refs": list(refs),
            },
        }

    def _claim(
        self,
        context: Mapping[str, Any],
        observations: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        refs = tuple(str(ref) for ref in context["listing_refs"])
        left, right = (self._observed_listing_records(observations)[ref] for ref in refs)
        claim = dict(context["requested_claim"])
        kind = claim.get("type")
        supported = False
        if kind == "lower_price":
            subject = str(claim.get("subject_ref", ""))
            prices = {refs[0]: int(left["price_minor"]), refs[1]: int(right["price_minor"])}
            supported = subject in prices and prices[subject] == min(prices.values())
        elif kind == "both_in_stock":
            supported = bool(left["in_stock"]) and bool(right["in_stock"])
        elif kind == "price_difference_at_most":
            supported = abs(int(left["price_minor"]) - int(right["price_minor"])) <= int(
                claim.get("threshold_minor", -1)
            )
        if supported:
            return {
                "intent": "publish_comparison_claim",
                "arguments": {
                    "listing_refs": list(refs),
                    "claim": claim,
                    "evidence_refs": list(refs),
                },
            }
        return {
            "intent": "decline_comparison_claim",
            "arguments": {
                "listing_refs": list(refs),
                "reason_code": "catalog_evidence_does_not_support_claim",
                "evidence_refs": list(refs),
            },
        }

    def _cart(
        self,
        task: LargeCatalogTask,
        context: Mapping[str, Any],
        observations: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        searches = _observations_for(observations, "search_cart_candidates")
        if not searches:
            return {
                "intent": "search_cart_candidates",
                "arguments": {
                    "requirements": list(context["requirements"]),
                    "constraints": dict(context.get("constraints", {})),
                    "cursor": None,
                    "page_size": 20,
                },
            }
        page = _actual_result(searches[-1])
        items = page.get("items", []) if isinstance(page, Mapping) else []
        if not items:
            return {
                "intent": "reject_purchase",
                "arguments": {
                    "reason_code": "no_feasible_cart",
                    "evidence_refs": [],
                },
            }
        lines = [
            {"listing_ref": str(line["listing_ref"]), "quantity": 1}
            for line in items[0]["lines"]
        ]
        evidence = [line["listing_ref"] for line in lines]
        if task.scenario == "D4":
            quotes = _observations_for(observations, "request_cart_quote")
            if not quotes:
                return {
                    "intent": "request_cart_quote",
                    "arguments": {"lines": lines, "evidence_refs": evidence},
                }
            quote_effect = _actual_result(quotes[-1])
            effect = quote_effect
            if isinstance(effect, Mapping) and "world_result" in effect:
                effect = effect["world_result"]
            quote = effect.get("quote") if isinstance(effect, Mapping) else None
            if not isinstance(quote, Mapping):
                raise ReferencePolicyError("quote request returned no authoritative quote")
            if int(quote["total_minor"]) <= int(context["budget_minor"]):
                return {
                    "intent": "checkout_quote",
                    "arguments": {"quote_ref": quote["quote_ref"]},
                }
            return {
                "intent": "decline_quote",
                "arguments": {
                    "quote_ref": quote["quote_ref"],
                    "reason_code": "quoted_total_exceeds_budget",
                },
            }
        return {
            "intent": "submit_cart",
            "arguments": {"lines": lines, "evidence_refs": evidence},
        }

    def _merchant_quote(
        self,
        context: Mapping[str, Any],
        observations: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        refs = [str(line["listing_ref"]) for line in context["quote_lines"]]
        observed = self._observed_listing_records(observations)
        for ref in refs:
            if ref not in observed:
                return {"intent": "observe_listing", "arguments": {"listing_ref": ref}}
        lines = []
        for requested in context["quote_lines"]:
            ref = str(requested["listing_ref"])
            quantity = int(requested["quantity"])
            unit = int(observed[ref]["price_minor"])
            lines.append(
                {
                    "listing_ref": ref,
                    "quantity": quantity,
                    "unit_price_minor": unit,
                    "line_total_minor": unit * quantity,
                }
            )
        subtotal = sum(line["line_total_minor"] for line in lines)
        discount = int(context["discount_minor"])
        fee = int(context["fee_minor"])
        return {
            "intent": "submit_quote",
            "arguments": {
                "lines": lines,
                "subtotal_minor": subtotal,
                "discount_minor": discount,
                "fee_minor": fee,
                "total_minor": subtotal - discount + fee,
                "evidence_refs": refs,
            },
        }

    @staticmethod
    def _observed_listing_records(
        observations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        rows: dict[str, Mapping[str, Any]] = {}
        for observation in _observations_for(observations, "observe_listing"):
            actual = _actual_result(observation)
            if not isinstance(actual, Mapping) or not actual.get("found"):
                continue
            listing = actual.get("listing")
            if isinstance(listing, Mapping):
                rows[str(listing["listing_ref"])] = listing
        return rows
