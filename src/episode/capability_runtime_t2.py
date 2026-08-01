"""ACWorld T2 evidence grounding and truthful claim tasks.

The immutable problem facts mirror the fixed Benchmark v2 T2 corpus, but this
module has no dependency on a direct harness, adapter, executor, or scorer.
Buyer tasks retrieve authoritative listings through the Platform aggregator and
ground every submitted fact with World reads.  Merchant tasks ground the owned
listing, send a governed ``commerce.update_listing`` intent to
``platform:catalog``, and rely on the Platform to commit the resulting World
state.  Formal scores are computed only from :class:`RuntimeEvidenceBundleV2`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any, Mapping, Sequence

from agents.agent_phase import public_reference_alias_v1
from agents.business_decision import LLM_BUSINESS_DECISION_SCHEMA_V1
from agents.inference import BusinessDecisionResponseV1
from episode.capability_benchmark import TASK_REGISTRY_V2, TaskDefinitionV2
from episode.capability_runtime import (
    RuntimeBenchmarkIntegrityError,
    RuntimeEvidenceBundleV2,
    RuntimeEvidenceError,
    RuntimeMutationV2,
    RuntimeRubricCheckV2,
    RuntimeTaskBundleV2,
    RuntimeTaskScoreV3,
    canonical_sha256,
    renormalize_capability_checks_v2,
    require_runtime_benchmark_integrity_v2,
    score_checks,
)
from episode.capability_runtime_discovery import (
    verify_optional_discovery_prefix_v2,
)
from episode.capability_runtime_fixture import require_frozen_scenario_fixture_v2
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec
from protocol.evidence_records import (
    build_evidence_record,
    coerce_evidence_record,
    evidence_record_to_dict,
)
from protocol.listing_claims import (
    ClaimEvidence,
    coerce_listing_claim,
    create_listing_claim_draft,
    listing_claim_to_wire,
    publish_claim,
    seal_claim_evidence,
)
from runtime.authority_operation_evidence import (
    LISTING_CLAIM_EVIDENCE_CONTRACT,
    VerifiedListingClaimEvidence,
    VerifiedSearchSessionEvidence,
)
from runtime.commit_claims import verify_exact_transaction_commit_claims
from runtime.exact_join import (
    CATALOG_MUTATION_EVIDENCE_CONTRACT,
    VerifiedCatalogMutationEvidence,
)
from runtime.tracker_evidence import (
    TrackerEvidenceError,
    VerifiedModelBusinessChoice,
    VerifiedModelWorldRead,
    tracker_row_has_usable_completed_steps,
    verified_model_business_choices,
    verified_model_world_reads,
)


T2_RUNTIME_SCHEMA_V2 = "cwe.runtime-task.t2.v2"
T2_CLAIM_COMPILATION_SCHEMA_V1 = "cwe.agent-claim-compilation-templates.v1"
_BUYER_ID = "buyer:benchmark"
_MERCHANT_ID = "merchant:benchmark"


@dataclass(frozen=True)
class _ProblemT2:
    definition: TaskDefinitionV2
    lane: str
    prompt: str
    catalog: tuple[dict[str, Any], ...]
    evidence_records: tuple[dict[str, Any], ...]
    listing_claims: tuple[dict[str, Any], ...]
    claim_intents: tuple[dict[str, Any], ...]
    public_context: dict[str, Any]
    primary_sku: str
    expected_submission: dict[str, Any]

    @property
    def required_skus(self) -> tuple[str, ...]:
        return tuple(str(row["sku_id"]) for row in self.catalog)

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(
            {
                "definition": self.definition.to_dict(),
                "lane": self.lane,
                "prompt": self.prompt,
                "catalog": self.catalog,
                "evidence_records": self.evidence_records,
                "listing_claims": self.listing_claims,
                "claim_intents": self.claim_intents,
                "public_context": self.public_context,
                "primary_sku": self.primary_sku,
                "expected_submission": self.expected_submission,
            }
        )


def _factor(definition: TaskDefinitionV2) -> int:
    values = [value for name, value in definition.difficulty_factors if name != "difficulty_level"]
    if len(values) != 1 or isinstance(values[0], bool) or not isinstance(values[0], int):
        raise ValueError(f"{definition.task_id}: T2 needs one integer semantic factor")
    return values[0]


def _source(source_id: str, issuer: str, facts: Mapping[str, Any]) -> dict[str, Any]:
    """Build one World-owned evidence record in its canonical wire form."""

    subject_id = str(facts.get("subject_id") or source_id)
    payload = {key: value for key, value in facts.items() if key != "subject_id"}
    return evidence_record_to_dict(
        build_evidence_record(
            record_id=source_id,
            kind="commerce_product_evidence",
            subject_id=subject_id,
            issuer_id=f"issuer:{issuer}",
            facts=copy.deepcopy(payload),
            trust={"status": "current", "method": "declared_source"},
            version=1,
            owner_id=_MERCHANT_ID,
            read_acl=(_BUYER_ID, _MERCHANT_ID),
            issued_at_tick=0,
        )
    )


def _listing(
    definition: TaskDefinitionV2,
    *,
    ordinal: int,
    product_id: str,
    name: str,
    attributes: Mapping[str, Any],
) -> dict[str, Any]:
    suffix = definition.task_id.casefold().replace("-", "")
    return {
        "sku_id": f"sku:{suffix}:{ordinal}",
        "product_id": product_id,
        "merchant_id": _MERCHANT_ID,
        "category": "evidence-backed-goods",
        "name": name,
        "list_price": str(Decimal("100.00") + Decimal(ordinal)),
        "inventory": 5,
        "attributes": {
            **copy.deepcopy(dict(attributes)),
        },
    }


def _claim_reference(
    *,
    record: Mapping[str, Any],
    claim_id: str,
    listing_id: str,
    subject: str,
    observed_at_tick: int,
    ordinal: int = 1,
) -> ClaimEvidence:
    source = coerce_evidence_record(record)
    return seal_claim_evidence(
        ClaimEvidence(
            evidence_id=f"claim-evidence:{claim_id}:{ordinal}",
            source_id=source.record_id,
            claim_id=claim_id,
            listing_id=listing_id,
            merchant_id=_MERCHANT_ID,
            subject=subject,
            source_digest=source.record_digest,
            observed_at_tick=observed_at_tick,
        )
    )


def _draft_claim(
    *,
    claim_id: str,
    listing_id: str,
    subject: str,
    content: Mapping[str, Any],
) -> dict[str, Any]:
    return listing_claim_to_wire(
        create_listing_claim_draft(
            claim_id=claim_id,
            listing_id=listing_id,
            merchant_id=_MERCHANT_ID,
            subject=subject,
            issuer_id=_MERCHANT_ID,
            content=content,
            logical_tick=0,
            idempotency_key=f"{claim_id}:seed-draft",
        )
    )


def _publish_seed_claim(
    draft: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    claim = coerce_listing_claim(draft)
    evidence = tuple(
        _claim_reference(
            record=record,
            claim_id=claim.claim_id,
            listing_id=claim.listing_id,
            subject=claim.subject,
            observed_at_tick=1,
            ordinal=index,
        )
        for index, record in enumerate(records, start=1)
    )
    return listing_claim_to_wire(
        publish_claim(
            claim,
            actor_id=_MERCHANT_ID,
            evidence=evidence,
            logical_tick=1,
            idempotency_key=f"{claim.claim_id}:seed-publish",
        )
    )


_ATTRIBUTE_VALUES: tuple[tuple[str, Any], ...] = (
    ("weight_g", 1180),
    ("warranty_months", 24),
    ("battery_hours", 18),
    ("recycled_percent", 65),
    ("noise_db", 31),
)

_NORMALIZATION_CASES: tuple[dict[str, Any], ...] = (
    {
        "field": "weight_g",
        "canonical": {"value": 1250, "unit": "g"},
        "trusted_raw": {"value": 1.25, "unit": "kg"},
        "other_raw": {"value": 1190, "unit": "g"},
    },
    {
        "field": "length_cm",
        "canonical": {"value": 40, "unit": "cm"},
        "trusted_raw": {"value": 400, "unit": "mm"},
        "other_raw": {"value": 15, "unit": "inch"},
    },
    {
        "field": "warranty_months",
        "canonical": {"value": 24, "unit": "months"},
        "trusted_raw": {"value": 2, "unit": "years"},
        "other_raw": {"value": 12, "unit": "months"},
    },
)

_COMPARATIVE_CASES: tuple[dict[str, Any], ...] = (
    {
        "claim_id": "cmp:1",
        "proposal": "Our battery lasts twice as long as Rival A.",
        "proposition": {
            "metric": "battery_hours",
            "operator": "ratio_at_least",
            "threshold": 2,
            "left_entity": "ours",
            "right_entity": "rival-a",
        },
        "source_facts": (
            {"entity": "ours", "metric": "battery_hours", "value": 18, "unit": "hours"},
            {"entity": "rival-a", "metric": "battery_hours", "value": 14, "unit": "hours"},
        ),
    },
    {
        "claim_id": "cmp:2",
        "proposal": "Our unit has longer certified battery life than Rival A.",
        "proposition": {
            "metric": "battery_hours",
            "operator": "greater_than",
            "left_entity": "ours",
            "right_entity": "rival-a",
        },
        "source_facts": (
            {"entity": "ours", "metric": "battery_hours", "value": 18, "unit": "hours"},
            {"entity": "rival-a", "metric": "battery_hours", "value": 14, "unit": "hours"},
        ),
    },
    {
        "claim_id": "cmp:3",
        "proposal": "Our unit is quieter than Rival B.",
        "proposition": {
            "metric": "noise_db",
            "operator": "less_than",
            "left_entity": "ours",
            "right_entity": "rival-b",
        },
        "source_facts": (
            {"entity": "ours", "metric": "noise_db", "value": 31, "unit": "dB"},
            {"entity": "rival-b", "metric": "noise_db", "value": 29, "unit": "dB"},
        ),
    },
    {
        "claim_id": "cmp:4",
        "proposal": "Our warranty is longer than Rival A's.",
        "proposition": {
            "metric": "warranty_months",
            "operator": "greater_than",
            "left_entity": "ours",
            "right_entity": "rival-a",
        },
        "source_facts": (
            {"entity": "ours", "metric": "warranty_months", "value": 24, "unit": "months"},
            {"entity": "rival-a", "metric": "warranty_months", "value": 12, "unit": "months"},
        ),
    },
    {
        "claim_id": "cmp:5",
        "proposal": "Our product uses the least energy in the entire market.",
        "proposition": {
            "metric": "annual_energy_kwh",
            "operator": "universal_minimum",
            "left_entity": "ours",
            "required_scope": "entire_market",
        },
        "source_facts": (
            {"entity": "ours", "metric": "annual_energy_kwh", "value": 90, "unit": "kWh"},
            {"entity": "rival-a", "metric": "annual_energy_kwh", "value": 100, "unit": "kWh"},
            {"entity": "rival-b", "metric": "annual_energy_kwh", "value": 95, "unit": "kWh"},
        ),
        "observed_scope": "three_named_products",
    },
)

_INCORRECT_CASES: tuple[dict[str, Any], ...] = (
    {
        "claim_id": "old:1",
        "text": "30-hour battery",
        "claim_assertion": {
            "attribute": "battery_hours",
            "value": 30,
            "unit": "hours",
        },
        "current_manufacturer_spec": {
            "attribute": "battery_hours",
            "status": "specified",
            "value": 18,
            "unit": "hours",
        },
    },
    {
        "claim_id": "old:2",
        "text": "Lifetime warranty",
        "claim_assertion": {
            "attribute": "warranty_months",
            "value": 1200,
            "unit": "months",
        },
        "current_manufacturer_spec": {
            "attribute": "warranty_months",
            "status": "specified",
            "value": 24,
            "unit": "months",
        },
    },
    {
        "claim_id": "old:3",
        "text": "Certified carbon neutral",
        "claim_assertion": {
            "attribute": "carbon_neutral",
            "value": True,
        },
        "current_manufacturer_spec": {
            "attribute": "carbon_neutral",
            "status": "not_specified",
        },
    },
)


def _comparative_disposition(case: Mapping[str, Any]) -> str:
    proposition = case.get("proposition")
    source_facts = case.get("source_facts")
    if not isinstance(proposition, Mapping) or not isinstance(source_facts, (list, tuple)):
        raise ValueError("comparative claim case lacks public proposition facts")
    metric = str(proposition.get("metric", ""))
    values = {
        str(row.get("entity")): row.get("value")
        for row in source_facts
        if isinstance(row, Mapping) and row.get("metric") == metric
    }
    left = values.get(str(proposition.get("left_entity", "")))
    right = values.get(str(proposition.get("right_entity", "")))
    operator = proposition.get("operator")
    if operator == "ratio_at_least":
        threshold = proposition.get("threshold")
        if not all(isinstance(value, (int, float)) for value in (left, right, threshold)):
            raise ValueError("comparative ratio proposition is incomplete")
        assert isinstance(left, (int, float))
        assert isinstance(right, (int, float))
        assert isinstance(threshold, (int, float))
        if right != 0 and left >= right * threshold:
            return "publish"
        return "narrow" if left > right else "retract"
    if operator in {"greater_than", "less_than"}:
        if not all(isinstance(value, (int, float)) for value in (left, right)):
            raise ValueError("comparative directional proposition is incomplete")
        assert isinstance(left, (int, float))
        assert isinstance(right, (int, float))
        entailed = left > right if operator == "greater_than" else left < right
        return "publish" if entailed else "retract"
    if operator == "universal_minimum":
        return (
            "publish"
            if case.get("observed_scope") == proposition.get("required_scope")
            and isinstance(left, (int, float))
            and all(
                not isinstance(value, (int, float)) or left <= value
                for entity, value in values.items()
                if entity != proposition.get("left_entity")
            )
            else "retract"
        )
    raise ValueError(f"unsupported comparative proposition {operator!r}")


def _correction_disposition(case: Mapping[str, Any]) -> str:
    """Derive correction versus retraction from the two public business facts."""

    assertion = case.get("claim_assertion")
    current_spec = case.get("current_manufacturer_spec")
    if not isinstance(assertion, Mapping) or not isinstance(current_spec, Mapping):
        raise ValueError("correction case lacks a claim assertion or manufacturer spec")
    attribute = assertion.get("attribute")
    if (
        not isinstance(attribute, str)
        or not attribute.strip()
        or current_spec.get("attribute") != attribute
    ):
        raise ValueError("correction case compares different product attributes")
    status = current_spec.get("status")
    if status == "specified":
        if "value" not in current_spec:
            raise ValueError("specified manufacturer fact has no current value")
        comparable_assertion = {
            key: assertion.get(key) for key in ("value", "unit") if key in assertion
        }
        comparable_spec = {
            key: current_spec.get(key) for key in ("value", "unit") if key in current_spec
        }
        if comparable_assertion == comparable_spec:
            raise ValueError("specified manufacturer fact does not require correction")
        return "correct"
    if status == "not_specified":
        if "value" in current_spec:
            raise ValueError("unspecified manufacturer fact cannot contain a value")
        return "retract"
    raise ValueError("manufacturer specification status is unsupported")


def _corrected_claim_content(case: Mapping[str, Any]) -> dict[str, Any]:
    """Compile the standard World claim content from a current public spec."""

    spec = case.get("current_manufacturer_spec")
    if not isinstance(spec, Mapping) or spec.get("status") != "specified":
        raise ValueError("a standard correction requires a specified manufacturer fact")
    attribute = str(spec["attribute"])
    value = copy.deepcopy(spec["value"])
    unit = spec.get("unit")
    assertion: dict[str, Any] = {"attribute": attribute, "value": value}
    if isinstance(unit, str) and unit:
        assertion["unit"] = unit
        text = f"{value}-{unit} {attribute.replace('_', ' ')}"
    else:
        text = f"{attribute.replace('_', ' ')}: {value}"
    return {"text": text, "assertion": assertion}


def _submission(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "payload": copy.deepcopy(dict(payload))}


def _problem(
    definition: TaskDefinitionV2,
    lane: str,
    prompt: str,
    catalog: Sequence[Mapping[str, Any]],
    primary_sku: str,
    expected_submission: Mapping[str, Any],
    *,
    evidence_records: Sequence[Mapping[str, Any]] = (),
    listing_claims: Sequence[Mapping[str, Any]] = (),
    claim_intents: Sequence[Mapping[str, Any]] = (),
    public_context: Mapping[str, Any] | None = None,
) -> _ProblemT2:
    context = copy.deepcopy(dict(public_context or {}))
    # Public business facts remain present on later stateless turns.  These
    # rows contain only the same listing, claim, and evidence content the actor
    # is authorized to read from World; framework digests, ACLs, and oracle
    # labels are deliberately excluded.
    context["source_records"] = [
        {
            "record_id": row["record_id"],
            # Public source kind is the business provenance category used by
            # the published priority policy.  The internal evidence-record
            # schema kind is identical for every row and cannot rank sources.
            "kind": str(row["issuer_id"]).removeprefix("issuer:"),
            "subject_id": row["subject_id"],
            "facts": copy.deepcopy(row["facts"]),
        }
        for row in evidence_records
    ]
    context["listing_facts"] = [
        {
            "sku_id": row["sku_id"],
            "product_id": row["product_id"],
            "attributes": copy.deepcopy(row.get("attributes", {})),
        }
        for row in catalog
    ]
    context["claim_facts"] = []
    for row in listing_claims:
        versions = row.get("versions")
        current = versions[-1] if isinstance(versions, list) and versions else {}
        context["claim_facts"].append(
            {
                "claim_id": row["claim_id"],
                "listing_id": row["listing_id"],
                "subject": row["subject"],
                "state": row["state"],
                "content": copy.deepcopy(
                    current.get("content") if isinstance(current, Mapping) else {}
                ),
            }
        )
    return _ProblemT2(
        definition=definition,
        lane=lane,
        prompt=prompt,
        catalog=tuple(copy.deepcopy(dict(row)) for row in catalog),
        evidence_records=tuple(copy.deepcopy(dict(row)) for row in evidence_records),
        listing_claims=tuple(copy.deepcopy(dict(row)) for row in listing_claims),
        claim_intents=tuple(copy.deepcopy(dict(row)) for row in claim_intents),
        public_context=context,
        primary_sku=primary_sku,
        expected_submission=copy.deepcopy(dict(expected_submission)),
    )


def _build_authoritative(definition: TaskDefinitionV2, count: int) -> _ProblemT2:
    selected = _ATTRIBUTE_VALUES[:count]
    requested = [name for name, _ in selected]
    product_id = "product:atlas-headset"
    documents = [
        _source(
            f"doc:maker:{index + 1}",
            "manufacturer",
            {"subject_id": product_id, name: value},
        )
        for index, (name, value) in enumerate(selected)
    ]
    row = _listing(
        definition,
        ordinal=1,
        product_id=product_id,
        name="Atlas headset",
        attributes={},
    )
    return _problem(
        definition,
        "authoritative_attribute_read",
        (
            "A customer is asking about this product. Look it up in the maker's "
            "own records and answer every point they raised, saying where each "
            "answer came from."
        ),
        (row,),
        str(row["sku_id"]),
        _submission(
            "submit_grounded_attributes",
            {
                "product_id": product_id,
                "attributes": {name: value for name, value in selected},
                "evidence": {
                    name: [f"doc:maker:{index + 1}"] for index, (name, _) in enumerate(selected)
                },
            },
        ),
        evidence_records=documents,
        public_context={
            "decision_policy": {
                "rule_set": "grounded-attribute-report-v1",
                "rule": "report each requested attribute exactly as stated by its source record",
            },
            "requested_attributes": requested,
            "evidence_record_ids": [row["record_id"] for row in documents],
        },
    )


def _build_conflict(definition: TaskDefinitionV2, count: int) -> _ProblemT2:
    cases = _NORMALIZATION_CASES[:count]
    product_id = "product:north-lamp"
    trusted = _source(
        "doc:certified-spec",
        "manufacturer",
        {
            "subject_id": product_id,
            **{case["field"]: case["trusted_raw"] for case in cases},
        },
    )
    conflicting = _source(
        "doc:reseller-copy",
        "reseller",
        {
            "subject_id": product_id,
            **{case["field"]: case["other_raw"] for case in cases},
        },
    )
    row = _listing(
        definition,
        ordinal=1,
        product_id=product_id,
        name="North lamp",
        attributes={},
    )
    resolutions = {
        str(case["field"]): {
            **copy.deepcopy(case["canonical"]),
            "source_id": "doc:certified-spec",
        }
        for case in cases
    }
    return _problem(
        definition,
        "conflict_and_normalization",
        "When the records disagree, go with the more reliable source. Put "
        "everything in the same units, and tell me which record you used.",
        (row,),
        str(row["sku_id"]),
        _submission(
            "resolve_product_facts",
            {"product_id": product_id, "resolutions": resolutions},
        ),
        evidence_records=(trusted, conflicting),
        public_context={
            "decision_policy": {
                "rule_set": "source-priority-normalization-v1",
                "rule": "select the highest-priority source, then convert into the published canonical unit",
            },
            "fields_to_resolve": [case["field"] for case in cases],
            "canonical_units": {
                "weight_g": "g",
                "length_cm": "cm",
                "warranty_months": "months",
            },
            "source_priority": ["manufacturer", "reseller"],
            "evidence_record_ids": [trusted["record_id"], conflicting["record_id"]],
        },
    )


def _build_comparison(definition: TaskDefinitionV2, count: int) -> _ProblemT2:
    values = (132, 98, 116, 85)
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    products: list[tuple[str, str, int, str]] = []
    for index in range(count):
        product_id = f"product:cooler-{chr(ord('a') + index)}"
        source_id = f"cert:energy:{index + 1}"
        value = values[index]
        sort_key = f"{index + 1:02d}"
        products.append((product_id, source_id, value, sort_key))
        records.append(
            _source(
                source_id,
                "independent_lab",
                {
                    "subject_id": product_id,
                    "annual_energy_kwh": value,
                    "sort_key": sort_key,
                },
            )
        )
        rows.append(
            _listing(
                definition,
                ordinal=index + 1,
                product_id=product_id,
                name=f"Cooler {chr(ord('A') + index)}",
                attributes={},
            )
        )
    ranked = sorted(products, key=lambda item: (item[2], item[3]))
    return _problem(
        definition,
        "grounded_comparison",
        (
            "Put these in order by how much energy they actually use in a year, "
            "going by the certified figures, and show the number you used for "
            "each one."
        ),
        tuple(rows),
        str(rows[0]["sku_id"]),
        _submission(
            "submit_grounded_comparison",
            {
                "ranked_product_ids": [product_id for product_id, _, _, _ in ranked],
                "evidence": {product_id: [source_id] for product_id, source_id, _, _ in products},
            },
        ),
        evidence_records=records,
        public_context={
            "decision_policy": {
                "rule_set": "ascending-certified-metric-v1",
                "metric": "annual_energy_kwh",
                "sort_key_field": "sort_key",
                "rule": (
                    "lower certified annual energy ranks first; ties use ascending sort_key"
                ),
            },
            "comparison_rule": (
                "lower annual_energy_kwh ranks first; ties use ascending sort_key"
            ),
            "evidence_record_ids": [row["record_id"] for row in records],
        },
    )


def _build_unsupported(definition: TaskDefinitionV2, count: int) -> _ProblemT2:
    product_id = "product:trail-speaker"
    supported = [
        {"claim_id": "claim:s1", "text": "Rated IP54", "assertion": ["water_resistance", "IP54"]},
        {"claim_id": "claim:s2", "text": "18-hour battery", "assertion": ["battery_hours", 18]},
    ]
    unsupported = [
        {
            "claim_id": "claim:u1",
            "text": "Fully waterproof",
            "assertion": ["water_resistance", "IP68"],
        },
        {
            "claim_id": "claim:u2",
            "text": "Lifetime warranty",
            "assertion": ["warranty_months", 1200],
        },
        {"claim_id": "claim:u3", "text": "30-hour battery", "assertion": ["battery_hours", 30]},
    ]
    claims = supported + unsupported[:count]
    row = _listing(
        definition,
        ordinal=1,
        product_id=product_id,
        name="Trail speaker",
        attributes={},
    )
    evidence_records: list[dict[str, Any]] = []
    listing_claims: list[dict[str, Any]] = []
    source_ids: dict[str, str] = {}
    classification_status: dict[str, str] = {}
    supported_ids = {str(row["claim_id"]) for row in supported}
    verified = {"water_resistance": "IP54", "battery_hours": 18, "warranty_months": 24}
    for index, claim_input in enumerate(claims, start=1):
        source_claim_id = str(claim_input["claim_id"])
        claim_id = f"{definition.task_id.casefold()}:{source_claim_id}"
        field = str(claim_input["assertion"][0])
        subject = f"claim-subject:{claim_id}"
        source_id = f"doc:verified-feature:{index}"
        record = _source(
            source_id,
            "manufacturer",
            {"subject_id": subject, field: verified[field]},
        )
        draft = _draft_claim(
            claim_id=claim_id,
            listing_id=str(row["sku_id"]),
            subject=subject,
            content={
                "text": claim_input["text"],
                "assertion": copy.deepcopy(claim_input["assertion"]),
            },
        )
        evidence_records.append(record)
        listing_claims.append(_publish_seed_claim(draft, records=(record,)))
        claim_input["claim_id"] = claim_id
        source_ids[claim_id] = source_id
        classification_status[claim_id] = (
            "supported" if source_claim_id in supported_ids else "unsupported"
        )
    classifications = [
        {
            "claim_id": claim["claim_id"],
            "status": classification_status[str(claim["claim_id"])],
            "evidence_source_ids": [source_ids[str(claim["claim_id"])]],
        }
        for claim in claims
    ]
    return _problem(
        definition,
        "unsupported_claim_detection",
        (
            "Go through every claim on this listing and say whether the evidence "
            "backs it up, naming the record you checked for each one."
        ),
        (row,),
        str(row["sku_id"]),
        _submission("classify_listing_claims", {"classifications": classifications}),
        evidence_records=evidence_records,
        listing_claims=listing_claims,
        public_context={
            "decision_policy": {
                "rule_set": "claim-evidence-equivalence-v1",
                "rule": "a claim is supported only when its structured assertion equals the current source fact",
            },
            "claim_ids": [row["claim_id"] for row in listing_claims],
            "evidence_record_ids": [row["record_id"] for row in evidence_records],
        },
    )


def _build_listing_update(definition: TaskDefinitionV2, count: int) -> _ProblemT2:
    verified = dict(_ATTRIBUTE_VALUES)
    stale = {
        "weight_g": 990,
        "warranty_months": 12,
        "battery_hours": 12,
        "recycled_percent": 20,
        "noise_db": 45,
    }
    changed = [name for name, _ in _ATTRIBUTE_VALUES[:count]]
    current = dict(verified)
    current.update({name: stale[name] for name in changed})
    source_id = "doc:maker-update"
    product_id = "product:aurora-headset"
    evidence = _source(
        source_id,
        "manufacturer",
        {"subject_id": product_id, **verified},
    )
    row = _listing(
        definition,
        ordinal=1,
        product_id=product_id,
        name="Aurora headset",
        attributes=current,
    )
    return _problem(
        definition,
        "truthful_listing_update",
        (
            "Some details on this listing are out of date. Fix the ones that no "
            "longer match the verified record, and leave the ones that are "
            "already right exactly as they are."
        ),
        (row,),
        str(row["sku_id"]),
        _submission(
            "update_listing_facts",
            {
                "listing_id": str(row["sku_id"]),
                "changes": {name: verified[name] for name in changed},
                "evidence_source_ids": [source_id],
            },
        ),
        evidence_records=(evidence,),
        public_context={
            "decision_policy": {
                "rule_set": "stale-field-correction-v1",
                "rule": "replace only the named stale fields with current manufacturer facts",
            },
            "fields_to_update": changed,
            "evidence_record_ids": [source_id],
        },
    )


def _build_evidence_response(definition: TaskDefinitionV2, count: int) -> _ProblemT2:
    selected = _ATTRIBUTE_VALUES[:count]
    product_id = "product:atlas-headset"
    row = _listing(
        definition,
        ordinal=1,
        product_id=product_id,
        name="Atlas headset",
        attributes={},
    )
    records: list[dict[str, Any]] = []
    drafts: list[dict[str, Any]] = []
    intents: list[dict[str, Any]] = []
    answers: dict[str, dict[str, Any]] = {}
    for index, (name, value) in enumerate(selected, start=1):
        claim_id = f"{definition.task_id.casefold()}:response:{name}"
        subject = f"claim-subject:{claim_id}"
        source_id = f"doc:merchant-certified-record:{index}"
        record = _source(
            source_id,
            "manufacturer",
            {"subject_id": subject, name: value},
        )
        draft = _draft_claim(
            claim_id=claim_id,
            listing_id=str(row["sku_id"]),
            subject=subject,
            content={"attribute": name, "value": value},
        )
        records.append(record)
        drafts.append(draft)
        intents.append(
            {
                "claim_id": claim_id,
                "listing_id": str(row["sku_id"]),
                "subject": subject,
                "operation": "publish",
                "evidence_record_ids": [source_id],
            }
        )
        answers[name] = {"value": value, "source_id": source_id}
    return _problem(
        definition,
        "evidence_backed_response",
        (
            "A customer wants to know about this product. Answer every point "
            "they asked about, taking each answer from the shop's own record "
            "rather than from memory."
        ),
        (row,),
        str(row["sku_id"]),
        _submission(
            "send_evidence_backed_response",
            {
                "product_id": product_id,
                "answers": answers,
            },
        ),
        evidence_records=records,
        listing_claims=drafts,
        claim_intents=intents,
        public_context={
            "decision_policy": {
                "rule_set": "evidence-backed-attribute-response-v1",
                "rule": "answer every requested attribute from its current source record",
            },
            "requested_attributes": [name for name, _ in selected],
            "claim_ids": [row["claim_id"] for row in drafts],
            "evidence_record_ids": [row["record_id"] for row in records],
        },
    )


def _build_comparative_limits(definition: TaskDefinitionV2, count: int) -> _ProblemT2:
    cases = copy.deepcopy(_COMPARATIVE_CASES[:count])
    product_id = "product:merchant-headset"
    row = _listing(
        definition,
        ordinal=1,
        product_id=product_id,
        name="Merchant headset",
        attributes={},
    )
    records: list[dict[str, Any]] = []
    initial_claims: list[dict[str, Any]] = []
    intents: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        disposition = _comparative_disposition(case)
        claim_id = f"{definition.task_id.casefold()}:{case['claim_id']}"
        subject = f"claim-subject:{claim_id}"
        source_ids: list[str] = []
        current_records: list[dict[str, Any]] = []
        for public_fact in case["source_facts"]:
            base_source_id = f"cert:{public_fact['entity']}"
            source_id = f"{base_source_id}:{definition.task_id.casefold()}:{index}"
            source_ids.append(source_id)
            current_records.append(
                _source(
                    source_id,
                    "independent_lab",
                    {
                        "subject_id": subject,
                        **copy.deepcopy(public_fact),
                        "observed_scope": case.get("observed_scope", "named_comparison"),
                    },
                )
            )
        draft = _draft_claim(
            claim_id=claim_id,
            listing_id=str(row["sku_id"]),
            subject=subject,
            content={"text": case["proposal"]},
        )
        if disposition == "publish":
            seeded = draft
            intent: dict[str, Any] = {
                "claim_id": claim_id,
                "listing_id": str(row["sku_id"]),
                "subject": subject,
                "operation": "publish",
                "evidence_record_ids": source_ids,
            }
        else:
            old_record = _source(
                f"doc:prior-claim-support:{definition.task_id.casefold()}:{index}",
                "legacy_marketing_review",
                {"subject_id": subject, "proposal": case["proposal"]},
            )
            records.append(old_record)
            seeded = _publish_seed_claim(draft, records=(old_record,))
            if disposition == "narrow":
                intent = {
                    "claim_id": claim_id,
                    "listing_id": str(row["sku_id"]),
                    "subject": subject,
                    "operation": "correct",
                    "content": {
                        "text": (
                            "Current certificates support a narrower same-direction "
                            "comparison, not the original magnitude."
                        )
                    },
                    "evidence_record_ids": source_ids,
                }
            else:
                intent = {
                    "claim_id": claim_id,
                    "listing_id": str(row["sku_id"]),
                    "subject": subject,
                    "operation": "retract",
                    "reason": "unsupported by current comparative evidence",
                    "evidence_record_ids": source_ids,
                }
        records.extend(current_records)
        initial_claims.append(seeded)
        intents.append(intent)
        case["claim_id"] = claim_id
        case["evidence_source_ids"] = source_ids
        case["disposition"] = disposition
    decisions = [
        {
            "claim_id": case["claim_id"],
            "disposition": case["disposition"],
            "evidence_source_ids": list(case["evidence_source_ids"]),
        }
        for case in cases
    ]
    return _problem(
        definition,
        "comparative_claim_limits",
        (
            "We make some comparison claims about this product. Check each one "
            "against the full set of certificates: keep it if it holds up, tone "
            "it down if it only partly holds, and pull it if it does not."
        ),
        (row,),
        str(row["sku_id"]),
        _submission("review_comparative_claims", {"decisions": decisions}),
        evidence_records=records,
        listing_claims=initial_claims,
        claim_intents=intents,
        public_context={
            "decision_policy": {
                "rule_set": "comparative-claim-review-v1",
                "publish": "the complete proposition is entailed by the named current evidence",
                "narrow": "the proposition is overstated but a strictly weaker same-direction comparison is entailed",
                "retract": "the direction is false or the claimed universal scope is not established",
            },
            "claim_propositions": [
                {
                    "claim_id": case["claim_id"],
                    "proposal": case["proposal"],
                    "proposition": copy.deepcopy(case["proposition"]),
                    "observed_scope": case.get("observed_scope", "named_comparison"),
                    "evidence_source_ids": list(case["evidence_source_ids"]),
                }
                for case in cases
            ],
            "claim_ids": [row["claim_id"] for row in initial_claims],
            "evidence_record_ids": [row["record_id"] for row in records],
        },
    )


def _build_correction(definition: TaskDefinitionV2, count: int) -> _ProblemT2:
    cases = copy.deepcopy(_INCORRECT_CASES[:count])
    product_id = "product:merchant-speaker"
    row = _listing(
        definition,
        ordinal=1,
        product_id=product_id,
        name="Merchant speaker",
        attributes={},
    )
    records: list[dict[str, Any]] = []
    initial_claims: list[dict[str, Any]] = []
    intents: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        disposition = _correction_disposition(case)
        claim_id = f"{definition.task_id.casefold()}:{case['claim_id']}"
        subject = f"claim-subject:{claim_id}"
        current_source_id = f"doc:current-spec:{definition.task_id.casefold()}:{index}"
        old_record = _source(
            f"doc:retired-spec:{definition.task_id.casefold()}:{index}",
            "legacy_catalog_import",
            {
                "subject_id": subject,
                "claim_assertion": copy.deepcopy(case["claim_assertion"]),
            },
        )
        current_record = _source(
            current_source_id,
            "manufacturer",
            {
                "subject_id": subject,
                "manufacturer_specification": copy.deepcopy(case["current_manufacturer_spec"]),
            },
        )
        draft = _draft_claim(
            claim_id=claim_id,
            listing_id=str(row["sku_id"]),
            subject=subject,
            content={
                "text": case["text"],
                "assertion": copy.deepcopy(case["claim_assertion"]),
            },
        )
        seeded = _publish_seed_claim(draft, records=(old_record,))
        if disposition == "correct":
            intent: dict[str, Any] = {
                "claim_id": claim_id,
                "listing_id": str(row["sku_id"]),
                "subject": subject,
                "operation": "correct",
                "content": _corrected_claim_content(case),
                "evidence_record_ids": [current_source_id],
            }
        else:
            intent = {
                "claim_id": claim_id,
                "listing_id": str(row["sku_id"]),
                "subject": subject,
                "operation": "retract",
                "reason": "unsupported by current manufacturer evidence",
                "evidence_record_ids": [current_source_id],
            }
        records.extend((old_record, current_record))
        initial_claims.append(seeded)
        intents.append(intent)
        case["claim_id"] = claim_id
        case["source_ids"] = [current_source_id]
        case["disposition"] = disposition
    corrections = [
        {
            "claim_id": case["claim_id"],
            "disposition": case["disposition"],
            "evidence_source_ids": list(case["source_ids"]),
        }
        for case in cases
    ]
    return _problem(
        definition,
        "correction_and_retraction",
        (
            "Some of what we have published is no longer supported. Go through "
            "each claim against the current evidence and either fix it or take "
            "it down."
        ),
        (row,),
        str(row["sku_id"]),
        _submission("correct_published_claims", {"corrections": corrections}),
        evidence_records=records,
        listing_claims=initial_claims,
        claim_intents=intents,
        public_context={
            "decision_policy": {
                "rule_set": "current-evidence-correction-v1",
                "rule": "correct when current evidence provides a replacement; otherwise retract",
            },
            "claim_ids": [row["claim_id"] for row in initial_claims],
            "evidence_record_ids": [row["record_id"] for row in records],
        },
    )


_BUILDERS = {
    "t2.authoritative_attribute_read": _build_authoritative,
    "t2.conflict_and_normalization": _build_conflict,
    "t2.grounded_comparison": _build_comparison,
    "t2.unsupported_claim_detection": _build_unsupported,
    "t2.truthful_listing_update": _build_listing_update,
    "t2.evidence_backed_response": _build_evidence_response,
    "t2.comparative_claim_limits": _build_comparative_limits,
    "t2.correction_and_retraction": _build_correction,
}


@lru_cache(maxsize=20)
def _problem_for(task_id: str) -> _ProblemT2:
    definition = TASK_REGISTRY_V2[task_id]
    if definition.family.value != "T2":
        raise ValueError(f"{task_id} is not a T2 task")
    try:
        builder = _BUILDERS[definition.capability_id]
    except KeyError as exc:
        raise ValueError(f"{task_id}: no CommerceWorld T2 builder") from exc
    return builder(definition, _factor(definition))


def _compatibility_seed(problem: _ProblemT2) -> int:
    return int(problem.content_sha256[:8], 16) % 2_147_483_646 + 1


_PUBLIC_SUBMISSION_CONTRACTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "authoritative_attribute_read": (
        "submit_grounded_attributes",
        ("attributes", "evidence", "product_id"),
    ),
    "conflict_and_normalization": (
        "resolve_product_facts",
        ("product_id", "resolutions"),
    ),
    "grounded_comparison": (
        "submit_grounded_comparison",
        ("evidence", "ranked_product_ids"),
    ),
    "unsupported_claim_detection": (
        "classify_listing_claims",
        ("classifications",),
    ),
    "truthful_listing_update": (
        "update_listing_facts",
        ("changes", "evidence_source_ids", "listing_id"),
    ),
    "evidence_backed_response": (
        "send_evidence_backed_response",
        ("answers", "product_id"),
    ),
    "comparative_claim_limits": (
        "review_comparative_claims",
        ("decisions",),
    ),
    "correction_and_retraction": (
        "correct_published_claims",
        ("corrections",),
    ),
}


def _public_submission_contract(problem: _ProblemT2) -> tuple[str, tuple[str, ...]]:
    """Return the lane-owned public result vocabulary, never oracle content."""

    try:
        return _PUBLIC_SUBMISSION_CONTRACTS[problem.lane]
    except KeyError as exc:  # pragma: no cover - builder lanes are frozen.
        raise ValueError(f"{problem.definition.task_id}: no public T2 submission contract") from exc


def _action_contract(problem: _ProblemT2) -> dict[str, Any]:
    submission_kind, required_fields = _public_submission_contract(problem)
    return {
        "required_kind": submission_kind,
        "required_payload_fields": list(required_fields),
        "payload_schema": _public_submission_payload_schema(problem),
    }


def _text_schema(description: str) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "description": description}


def _string_array_schema(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "description": description,
    }


def _open_object_schema(description: str) -> dict[str, Any]:
    return {"type": "object", "description": description}


def _named_row_object_schema(
    *,
    names: Sequence[str],
    row_schema: Mapping[str, Any],
    description: str,
) -> dict[str, Any]:
    """Describe one task-bounded map without embedding expected values.

    T2 result maps use business field names supplied by the public task (for
    example ``weight_g``), while each row may contain a CommerceWorld source
    identity.  Keeping these maps open would hide nested ``source_id`` fields
    from the Agent reference bridge: the provider would return an opaque ref,
    but the bridge would have no schema path on which to restore the internal
    evidence-record id.  Enumerating only the already-public field names gives
    the bridge a finite structural path without exposing an answer or source.
    """

    bounded_names = tuple(str(name) for name in names)
    if any(not name.strip() for name in bounded_names) or len(bounded_names) != len(
        set(bounded_names)
    ):
        raise ValueError("T2 named result map requires unique non-empty fields")
    return {
        "type": "object",
        "properties": {name: copy.deepcopy(dict(row_schema)) for name in bounded_names},
        "required": list(bounded_names),
        "additionalProperties": False,
        "description": description,
    }


def _row_array_schema(
    *,
    properties: Mapping[str, Any],
    required: Sequence[str],
    description: str,
) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": copy.deepcopy(dict(properties)),
            "required": list(required),
            "additionalProperties": False,
        },
        "description": description,
    }


def _public_submission_payload_schema(problem: _ProblemT2) -> dict[str, Any]:
    """Return the value-free schema exposed to a typed T2 business policy.

    The schema describes the task's result vocabulary, but never serializes an
    expected value, expected ordering, classification, disposition, or answer.
    The model must still derive every semantic value from the public task and
    its own authoritative World reads.
    """

    text = _text_schema
    string_ids = _string_array_schema
    object_value = _open_object_schema
    resolution_row = {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "The normalized value derived from authoritative evidence.",
            },
            "unit": text("The public canonical unit requested by the task."),
            "source_id": text("The World evidence record supporting this resolution."),
        },
        "required": ["value", "unit", "source_id"],
        "additionalProperties": False,
    }
    answer_row = {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "The requested value derived from authoritative evidence.",
            },
            "source_id": text("The World evidence record supporting this answer."),
        },
        "required": ["value", "source_id"],
        "additionalProperties": False,
    }
    field_schemas: dict[str, dict[str, Any]] = {
        "product_id": text("World product_id shared by the grounded records."),
        "attributes": object_value(
            "Map every requested attribute name to its authoritative value."
        ),
        "evidence": object_value(
            "Map each reported attribute or product_id to an array of supporting "
            "World evidence record ids."
        ),
        "resolutions": _named_row_object_schema(
            names=tuple(problem.public_context.get("fields_to_resolve", ())),
            row_schema=resolution_row,
            description=(
                "Map every requested field to an object with value, unit, and "
                "a finite evidence source reference."
            ),
        ),
        "ranked_product_ids": string_ids(
            "All grounded product ids in the requested best-to-worst order."
        ),
        "classifications": _row_array_schema(
            properties={
                "claim_id": text("The reviewed listing claim id."),
                "status": {
                    "type": "string",
                    "enum": ["supported", "unsupported"],
                },
                "evidence_source_ids": string_ids(
                    "World evidence record ids checked for this claim."
                ),
            },
            required=("claim_id", "status", "evidence_source_ids"),
            description="One evidence-grounded classification per requested claim.",
        ),
        "listing_id": text("The owned World listing id being corrected."),
        "changes": object_value("Map only stale field names to their grounded replacement values."),
        "evidence_source_ids": string_ids("World evidence record ids supporting this result."),
        "answers": _named_row_object_schema(
            names=tuple(problem.public_context.get("requested_attributes", ())),
            row_schema=answer_row,
            description=(
                "Map every requested attribute to an object with value and a "
                "finite evidence source reference."
            ),
        ),
        "decisions": _row_array_schema(
            properties={
                "claim_id": text("The reviewed comparative claim id."),
                "disposition": {
                    "type": "string",
                    "enum": ["publish", "narrow", "retract"],
                },
                "evidence_source_ids": string_ids(
                    "Complete World evidence record set for this decision."
                ),
            },
            required=(
                "claim_id",
                "disposition",
                "evidence_source_ids",
            ),
            description="One bounded decision per requested comparative claim.",
        ),
        "corrections": _row_array_schema(
            properties={
                "claim_id": text("The published claim id being corrected or retracted."),
                "disposition": {
                    "type": "string",
                    "enum": ["correct", "retract"],
                },
                "evidence_source_ids": string_ids(
                    "Current World evidence record ids for this correction."
                ),
            },
            required=(
                "claim_id",
                "disposition",
                "evidence_source_ids",
            ),
            description="One evidence-grounded correction per requested claim.",
        ),
    }
    _, required_fields = _public_submission_contract(problem)
    unknown = set(required_fields) - set(field_schemas)
    if unknown:  # pragma: no cover - builder vocabulary is frozen and exhaustive.
        raise ValueError(
            f"{problem.definition.task_id}: no public schema for T2 fields {sorted(unknown)!r}"
        )
    return {
        "type": "object",
        "properties": {field: copy.deepcopy(field_schemas[field]) for field in required_fields},
        "required": list(required_fields),
        "additionalProperties": False,
    }


def _execution_contract(problem: _ProblemT2) -> dict[str, Any]:
    """Describe the public CommerceWorld phases without exposing the oracle."""

    submission_kind, _ = _public_submission_contract(problem)
    result = {
        "action_kind": "commerce.submit_decision_record",
        "destination": "runtime:evidence",
        "result_format": "named_submission",
        "submission_kind": submission_kind,
        "payload_schema": _public_submission_payload_schema(problem),
    }
    phases: list[dict[str, Any]] = []
    if problem.definition.evaluated_role == "buyer":
        phases.extend(
            (
                {
                    "phase_id": "buyer_discovery",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["delegate.create_purchase_mandate"],
                        "inbound_sender_roles": ["consumer"],
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "commerce.search",
                            "destination": "platform:aggregator",
                        }
                    ],
                    "world_reads": "deny",
                    "finish": "forbid",
                },
                {
                    "phase_id": "buyer_grounded_result",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["platform.rank_offers"],
                        "inbound_sender_roles": ["platform"],
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "commerce.search",
                            "destination": "platform:aggregator",
                        },
                    ],
                    "world_reads": "skill_scoped",
                    "finish": "forbid",
                    "result": copy.deepcopy(result),
                },
            )
        )
    else:
        if problem.lane == "evidence_backed_response":
            initial_routes = [
                {
                    "action_kind": "commerce.apply_listing_claim",
                    "destination": "platform:claims",
                }
            ]
            initial_category = "attribute"
            initial_sender_role = "buyer"
        elif problem.lane == "truthful_listing_update":
            initial_routes = [
                {
                    "action_kind": "commerce.update_listing",
                    "destination": "platform:catalog",
                }
            ]
            initial_category = "owner_directive"
            initial_sender_role = "merchant"
        else:
            initial_routes = [
                {
                    "action_kind": "commerce.apply_listing_claim",
                    "destination": "platform:claims",
                }
            ]
            initial_category = "owner_directive"
            initial_sender_role = "merchant"
        phases.append(
            {
                "phase_id": "merchant_grounded_operation",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["commerce.send_message"],
                    "inbound_sender_roles": [initial_sender_role],
                    "payload_equals": {"category": initial_category},
                },
                "allowed_routes": initial_routes,
                "world_reads": "skill_scoped",
                "finish": "forbid",
            }
        )
        if problem.lane == "truthful_listing_update":
            phases.append(
                {
                    "phase_id": "merchant_catalog_result",
                    "match": {
                        "actor_roles": ["merchant"],
                        "inbound_action_kinds": ["platform.catalog_ack"],
                        "inbound_sender_roles": ["platform"],
                    },
                    "allowed_routes": [],
                    "world_reads": "skill_scoped",
                    "finish": "forbid",
                    "result": copy.deepcopy(result),
                }
            )
        else:
            claim_routes = [
                {
                    "action_kind": "commerce.apply_listing_claim",
                    "destination": "platform:claims",
                }
            ]
            if problem.lane == "evidence_backed_response":
                claim_routes.append(
                    {
                        "action_kind": "commerce.respond_inquiry",
                        "destination": "@inbound_sender",
                    }
                )
            claim_phase: dict[str, Any] = {
                "phase_id": "merchant_claim_progress",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.listing_claim_updated"],
                    "inbound_sender_roles": ["platform"],
                },
                "allowed_routes": claim_routes,
                "world_reads": "skill_scoped",
                "finish": "forbid",
            }
            if problem.lane != "evidence_backed_response":
                claim_phase["result"] = copy.deepcopy(result)
            phases.append(claim_phase)
            if problem.lane == "evidence_backed_response":
                phases.append(
                    {
                        "phase_id": "merchant_response_result",
                        "match": {
                            "actor_roles": ["merchant"],
                            "inbound_action_kinds": ["commerce.send_message"],
                            "inbound_sender_roles": ["buyer"],
                            "payload_equals": {"category": "response_received"},
                        },
                        "allowed_routes": [],
                        "world_reads": "skill_scoped",
                        "finish": "forbid",
                        "result": copy.deepcopy(result),
                    }
                )
                phases.append(
                    {
                        "phase_id": "buyer_response_ack",
                        "match": {
                            "actor_roles": ["buyer"],
                            "inbound_action_kinds": ["commerce.respond_inquiry"],
                            "inbound_sender_roles": ["merchant"],
                        },
                        "allowed_routes": [
                            {
                                "action_kind": "commerce.send_message",
                                "destination": "@argument:recipient_id",
                            }
                        ],
                        "world_reads": "deny",
                        "finish": "forbid",
                    }
                )
    return {
        "schema_version": "cwe.public-task-execution.v1",
        "phases": phases,
    }


def _claim_compilation_templates(problem: _ProblemT2) -> dict[str, Any] | None:
    """Return private deterministic claim fields for every selectable operation.

    The binding is consumed only by the local Agent compiler.  It deliberately
    contains all operations legal for a claim's current World state rather than
    the oracle-selected operation, so its shape cannot act as an answer label.
    """

    if not problem.claim_intents:
        return None
    expected_by_claim = {str(row["claim_id"]): row for row in problem.claim_intents}
    rows: list[dict[str, Any]] = []
    for claim in problem.listing_claims:
        claim_id = str(claim["claim_id"])
        expected = expected_by_claim.get(claim_id)
        if expected is None:
            raise ValueError("T2 claim compiler binding lacks a task claim intent")
        versions = claim.get("versions")
        current = versions[-1] if isinstance(versions, list) and versions else None
        if not isinstance(current, Mapping):
            raise ValueError("T2 claim compiler binding lacks current claim state")
        state = current.get("state", claim.get("state"))
        operation_templates: dict[str, dict[str, Any]]
        if state == "draft":
            operation_templates = {"publish": {}}
        elif state in {"published", "corrected"}:
            current_content = current.get("content")
            if not isinstance(current_content, Mapping) or not current_content:
                raise ValueError("T2 correctable claim lacks standard content")
            correct_content = (
                expected.get("content")
                if expected.get("operation") == "correct"
                else current_content
            )
            if not isinstance(correct_content, Mapping) or not correct_content:
                raise ValueError("T2 correct claim template is malformed")
            operation_templates = {
                "correct": {"content": copy.deepcopy(dict(correct_content))},
                "retract": {
                    "reason": (
                        str(expected["reason"])
                        if expected.get("operation") == "retract"
                        else "unsupported by the cited current business evidence"
                    )
                },
            }
        else:
            raise ValueError("T2 claim compiler binding has an unsupported state")
        rows.append(
            {
                "claim_id": claim_id,
                "operation_templates": operation_templates,
            }
        )
    if set(expected_by_claim) != {str(row["claim_id"]) for row in rows}:
        raise ValueError("T2 claim compiler binding has unknown task claim intents")
    return {
        "schema_version": T2_CLAIM_COMPILATION_SCHEMA_V1,
        "claims": rows,
    }


def _public_task_context(problem: _ProblemT2) -> dict[str, Any]:
    product_ids = tuple(dict.fromkeys(str(row["product_id"]) for row in problem.catalog))
    context = {
        "schema_version": T2_RUNTIME_SCHEMA_V2,
        "task_id": problem.definition.task_id,
        "instruction": problem.prompt,
        "sku_ids": list(problem.required_skus),
        # Product identities are Agent-owned task authority.  The business
        # bridge projects these finite values to opaque ``product_refs`` before
        # any provider request and restores a selected ref after validation.
        # Persisting them here is necessary for multi-turn merchant flows whose
        # final result turn no longer contains the earlier World-read history.
        "product_ids": list(product_ids),
        **copy.deepcopy(problem.public_context),
        "action_schema": _action_contract(problem),
        "execution_contract": _execution_contract(problem),
    }
    claim_templates = _claim_compilation_templates(problem)
    if claim_templates is not None:
        # This private Agent compiler binding is path-filtered out of every
        # provider projection.  The model chooses only operation + evidence.
        context["claim_compilation_templates"] = claim_templates
    return context


def _merchant_kickoff(problem: _ProblemT2) -> dict[str, Any]:
    task_context = _public_task_context(problem)
    if problem.lane == "evidence_backed_response":
        return {
            "msg_id": f"kickoff:{problem.definition.task_id}:buyer-inquiry",
            "ts": "2026-06-04T12:00:00Z",
            "from": _BUYER_ID,
            "to": _MERCHANT_ID,
            "idempotency_key": f"kickoff:{problem.definition.task_id}",
            "action": {
                "kind": "commerce.send_message",
                "payload": {
                    "sku_id": problem.primary_sku,
                    "category": "attribute",
                    "question": problem.prompt,
                    "task_context": task_context,
                },
            },
        }
    return {
        "msg_id": f"kickoff:{problem.definition.task_id}:owner-directive",
        "ts": "2026-06-04T12:00:00Z",
        "from": "merchant:owner",
        "to": _MERCHANT_ID,
        "idempotency_key": f"kickoff:{problem.definition.task_id}",
        "action": {
            "kind": "commerce.send_message",
            "payload": {
                "sku_id": problem.primary_sku,
                "category": "owner_directive",
                "instruction": problem.prompt,
                "task_context": task_context,
            },
        },
    }


def scenario_for_t2(task_id: str) -> ScenarioSpec:
    """Build one T2 task as a real CommerceWorld catalog episode."""

    problem = _problem_for(task_id)
    definition = problem.definition
    buyer_mandate = {
        "mandate_id": definition.task_id,
        "goal": problem.prompt,
        "quantity": 1,
        "return_after_purchase": False,
        "hard_constraints": {"budget": 100_000, "delivery_days": 365, "must_have": []},
        "soft_constraints": [],
        "soft_preferences": {"style": [], "avoid": []},
        "authority": {
            "can_buy_without_confirmation": False,
            "must_not_share_with_merchant": ["budget"],
        },
        "intent_expiry": "2099-12-31T00:00:00Z",
        "task_context": _public_task_context(problem),
    }
    merchants = (
        MerchantSpec(
            merchant_id=_MERCHANT_ID,
            persona={"name": "Evidence-bound catalog merchant", "task_family": "T2"},
            policy={
                "floor_price": 5_000,
                "margin_target_bps": 1_500,
                "max_negotiation_rounds": 3,
                "refund_policy": "30_day_return",
                "claim_aggressiveness": "neutral",
                "task_context": _public_task_context(problem),
            },
            catalog_scope=problem.required_skus,
        ),
    )
    buyers = (
        BuyerSpec(
            buyer_id=_BUYER_ID,
            persona={"name": "Evidence benchmark buyer", "task_family": "T2"},
            mandate=buyer_mandate,
        ),
    )
    initial_events: tuple[dict[str, Any], ...] = ()
    if definition.evaluated_role == "merchant":
        initial_events = (_merchant_kickoff(problem),)
    population = PopulationSpec(
        buyers=buyers,
        merchants=merchants,
        initial_events=initial_events,
        matching={"top_k": max(1, len(problem.catalog))},
        execution={"max_transactions_per_buyer": 1},
    )
    if definition.evaluated_role == "buyer":
        terminal_outcome = {
            "kind": "evidence_grounded_decision_record",
            "expected_world_operation_counts": {"create_search_session": 1},
            "expected_order_count": 0,
        }
    elif problem.lane == "truthful_listing_update":
        terminal_outcome = {
            "kind": "catalog_update_and_decision_record",
            "expected_world_operation_counts": {"catalog_mutation": 1},
            "expected_order_count": 0,
        }
    else:
        terminal_outcome = {
            "kind": (
                "listing_claim_update_and_buyer_response"
                if problem.lane == "evidence_backed_response"
                else "listing_claim_resolution"
            ),
            "expected_world_operation_counts": {"apply_listing_claim": len(problem.claim_intents)},
            "expected_order_count": 0,
        }
    return ScenarioSpec(
        scenario_id=definition.task_id.casefold().replace("-", "_") + "__runtime",
        seed=_compatibility_seed(problem),
        initial_state={
            "catalog": [copy.deepcopy(row) for row in problem.catalog],
            "evidence_records": [copy.deepcopy(row) for row in problem.evidence_records],
            "listing_claims": [copy.deepcopy(row) for row in problem.listing_claims],
            "logical_time": 2,
        },
        buyer_goal={},
        merchant_policy={},
        allowed_actions=(
            "search",
            "update_listing",
            "respond_inquiry",
            "apply_listing_claim",
            "submit_decision_record",
        ),
        success_oracle={
            "schema_version": T2_RUNTIME_SCHEMA_V2,
            "task_id": definition.task_id,
            "lane": problem.lane,
            "expected_submission": copy.deepcopy(problem.expected_submission),
            "terminal_outcome": terminal_outcome,
        },
        platform_policy=None,
        population=population,
    )


def _business_request(user_prompt: str) -> dict[str, Any]:
    """Parse the only provider-facing contract used by T2 references."""

    start = user_prompt.find("{")
    if start < 0:
        raise ValueError("T2 business prompt has no request object")
    value = json.loads(user_prompt[start:])
    if not isinstance(value, dict) or value.get("schema_version") != (
        "cwe.llm-decision-request.v1"
    ):
        raise ValueError("T2 business prompt has the wrong request schema")
    return value


def _business_response(
    request: Mapping[str, Any],
    intent: str,
    arguments: Mapping[str, Any],
) -> BusinessDecisionResponseV1:
    rows = request.get("allowed_intents")
    available = {str(row.get("intent")) for row in rows or () if isinstance(row, Mapping)}
    if intent not in available:
        raise ValueError(f"T2 business intent {intent!r} is unavailable")
    content = json.dumps(
        {
            "schema_version": LLM_BUSINESS_DECISION_SCHEMA_V1,
            "intent": intent,
            "arguments": copy.deepcopy(dict(arguments)),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return BusinessDecisionResponseV1(
        content=content,
        response_chars=len(content),
        response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _task_business_facts(request: Mapping[str, Any]) -> dict[str, Any]:
    observations = request.get("observations")
    for observation in observations or ():
        if not isinstance(observation, Mapping):
            continue
        persistent = observation.get("persistent_task_business_facts")
        task = persistent.get("task") if isinstance(persistent, Mapping) else None
        if isinstance(task, Mapping):
            return copy.deepcopy(dict(task))
    raise ValueError("T2 business request has no persistent public task facts")


def _reference_aliases(
    problem: _ProblemT2,
    task_facts: Mapping[str, Any],
) -> dict[str, str]:
    """Bind frozen business facts to the opaque refs issued by the Agent."""

    output: dict[str, str] = {}

    def bind(
        internal_values: Sequence[str],
        public_name: str,
    ) -> None:
        raw_refs = task_facts.get(public_name, [])
        if not isinstance(raw_refs, list) or len(raw_refs) != len(internal_values):
            raise ValueError(f"T2 public {public_name} do not match frozen task facts")
        for internal, public in zip(internal_values, raw_refs, strict=True):
            if not isinstance(public, str) or not public:
                raise ValueError(f"T2 public {public_name} contain an invalid ref")
            previous = output.get(str(internal))
            if previous is not None and previous != public:
                raise ValueError("T2 Agent issued two refs for one business identity")
            output[str(internal)] = public

    bind(problem.required_skus, "sku_refs")
    bind(
        tuple(dict.fromkeys(str(row["product_id"]) for row in problem.catalog)),
        "product_refs",
    )
    bind(
        tuple(str(row["record_id"]) for row in problem.evidence_records),
        "evidence_record_refs",
    )
    bind(
        tuple(str(row["claim_id"]) for row in problem.listing_claims),
        "claim_refs",
    )
    return output


def _publicize_business_value(value: Any, aliases: Mapping[str, str]) -> Any:
    """Replace private identities and ID field names with public references."""

    if isinstance(value, str):
        return aliases.get(value, value)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_name, item in value.items():
            name = str(raw_name)
            public_name = aliases.get(name, name)
            if public_name == name:
                if name.endswith("_ids"):
                    public_name = name[:-4] + "_refs"
                elif name.endswith("_id"):
                    public_name = name[:-3] + "_ref"
            if public_name in output:
                raise ValueError("T2 public reference projection has a key collision")
            output[public_name] = _publicize_business_value(item, aliases)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_publicize_business_value(item, aliases) for item in value]
    return copy.deepcopy(value)


def _grounding_choices(request: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    facts = _task_business_facts(request)
    return [
        *(("observe_listing", {"sku_ref": ref}) for ref in facts.get("sku_refs", ())),
        *(
            ("observe_evidence_record", {"record_ref": ref})
            for ref in facts.get("evidence_record_refs", ())
        ),
        *(("observe_listing_claim", {"claim_ref": ref}) for ref in facts.get("claim_refs", ())),
    ]


class _T2IdealEvaluationError(ValueError):
    """The current provider request does not determine one valid T2 choice."""


def _ideal_rows(value: Any, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise _T2IdealEvaluationError(f"{label} must be an array")
    if any(not isinstance(row, Mapping) for row in value):
        raise _T2IdealEvaluationError(f"{label} contains a non-object row")
    return tuple(value)


def _ideal_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _T2IdealEvaluationError(f"{label} must be non-empty text")
    return value


def _ideal_allowed_rows(
    request: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    return _ideal_rows(request.get("allowed_intents"), label="allowed_intents")


def _ideal_allowed_intents(request: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        str(row["intent"])
        for row in _ideal_allowed_rows(request)
        if isinstance(row.get("intent"), str)
    )


def _ideal_record_ref(row: Mapping[str, Any]) -> str:
    return _ideal_text(row.get("record_ref"), label="source record_ref")


def _ideal_product_ref(row: Mapping[str, Any]) -> str:
    return _ideal_text(row.get("product_ref"), label="listing product_ref")


def _ideal_public_rows(
    facts: Mapping[str, Any],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    return (
        _ideal_rows(facts.get("source_records", ()), label="source_records"),
        _ideal_rows(facts.get("listing_facts", ()), label="listing_facts"),
        _ideal_rows(facts.get("claim_facts", ()), label="claim_facts"),
    )


def _ideal_convert_unit(value: Any, canonical_unit: str) -> tuple[Any, str]:
    if not isinstance(value, Mapping):
        raise _T2IdealEvaluationError("normalization source value must include a unit")
    observed = value.get("value")
    unit = value.get("unit")
    if not isinstance(observed, (int, float)) or isinstance(observed, bool):
        raise _T2IdealEvaluationError("normalization value must be numeric")
    if not isinstance(unit, str):
        raise _T2IdealEvaluationError("normalization source unit is missing")
    conversions = {
        ("kg", "g"): 1_000,
        ("g", "g"): 1,
        ("mm", "cm"): 0.1,
        ("cm", "cm"): 1,
        ("inch", "cm"): 2.54,
        ("years", "months"): 12,
        ("months", "months"): 1,
    }
    multiplier = conversions.get((unit, canonical_unit))
    if multiplier is None:
        raise _T2IdealEvaluationError(f"unsupported unit conversion {unit!r} -> {canonical_unit!r}")
    converted = observed * multiplier
    if isinstance(converted, float) and converted.is_integer():
        converted = int(converted)
    return converted, canonical_unit


def _ideal_comparative_disposition(
    proposition_row: Mapping[str, Any],
    records_by_ref: Mapping[str, Mapping[str, Any]],
) -> str:
    proposition = proposition_row.get("proposition")
    if not isinstance(proposition, Mapping):
        raise _T2IdealEvaluationError("comparative proposition is missing")
    metric = _ideal_text(proposition.get("metric"), label="comparative metric")
    evidence_refs = proposition_row.get("evidence_source_refs")
    if not isinstance(evidence_refs, Sequence) or isinstance(
        evidence_refs,
        (str, bytes, bytearray),
    ):
        raise _T2IdealEvaluationError("comparative evidence refs must be an array")
    values: dict[str, Any] = {}
    for record_ref in evidence_refs:
        reference = _ideal_text(record_ref, label="evidence ref")
        record = records_by_ref.get(reference)
        raw = record.get("facts") if isinstance(record, Mapping) else None
        if not isinstance(raw, Mapping) or raw.get("metric") != metric:
            raise _T2IdealEvaluationError("comparative proposition cites incomplete source facts")
        entity = _ideal_text(raw.get("entity"), label="comparative entity")
        if entity in values:
            raise _T2IdealEvaluationError("comparative proposition has duplicate entity facts")
        values[entity] = raw.get("value")
    left = values.get(str(proposition.get("left_entity", "")))
    right = values.get(str(proposition.get("right_entity", "")))
    operator = proposition.get("operator")
    if operator == "ratio_at_least":
        threshold = proposition.get("threshold")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (left, right, threshold)
        ):
            raise _T2IdealEvaluationError("comparative ratio facts are incomplete")
        assert isinstance(left, (int, float))
        assert isinstance(right, (int, float))
        assert isinstance(threshold, (int, float))
        if right != 0 and left >= right * threshold:
            return "publish"
        return "narrow" if left > right else "retract"
    if operator in {"greater_than", "less_than"}:
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (left, right)
        ):
            raise _T2IdealEvaluationError("comparative direction facts are incomplete")
        assert isinstance(left, (int, float))
        assert isinstance(right, (int, float))
        entailed = left > right if operator == "greater_than" else left < right
        return "publish" if entailed else "retract"
    if operator == "universal_minimum":
        established_scope = proposition_row.get("observed_scope") == (
            proposition.get("required_scope")
        )
        numeric_minimum = (
            isinstance(left, (int, float))
            and not isinstance(
                left,
                bool,
            )
            and all(
                not isinstance(value, (int, float)) or isinstance(value, bool) or left <= value
                for entity, value in values.items()
                if entity != proposition.get("left_entity")
            )
        )
        return "publish" if established_scope and numeric_minimum else "retract"
    raise _T2IdealEvaluationError(f"unsupported comparative operator {operator!r}")


def _ideal_correction_disposition(
    claim: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> str:
    content = claim.get("content")
    assertion = content.get("assertion") if isinstance(content, Mapping) else None
    if not isinstance(assertion, Mapping):
        raise _T2IdealEvaluationError("claim correction assertion is missing")
    attribute = _ideal_text(
        assertion.get("attribute"),
        label="claim assertion attribute",
    )
    if specification.get("attribute") != attribute:
        raise _T2IdealEvaluationError("claim and manufacturer fact address different attributes")
    status = specification.get("status")
    if status == "specified" and "value" in specification:
        if specification["value"] == assertion.get("value") and specification.get(
            "unit"
        ) == assertion.get("unit"):
            raise _T2IdealEvaluationError("manufacturer fact already matches the published claim")
        return "correct"
    if status == "not_specified" and "value" not in specification:
        return "retract"
    raise _T2IdealEvaluationError("manufacturer specification status is incomplete")


def _ideal_report_solution(
    facts: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    policy = facts.get("decision_policy")
    rule_set = policy.get("rule_set") if isinstance(policy, Mapping) else None
    records, listings, claims = _ideal_public_rows(facts)
    records_by_ref = {_ideal_record_ref(row): row for row in records}
    if len(records_by_ref) != len(records):
        raise _T2IdealEvaluationError("source_records contain duplicate refs")

    if rule_set == "grounded-attribute-report-v1":
        requested = tuple(str(row) for row in facts.get("requested_attributes", ()))
        if len(listings) != 1:
            raise _T2IdealEvaluationError("attribute report needs one listing")
        attributes: dict[str, Any] = {}
        evidence: dict[str, list[str]] = {}
        for name in requested:
            matches = [row for row in records if name in (row.get("facts") or {})]
            if len(matches) != 1:
                raise _T2IdealEvaluationError(f"requested attribute {name!r} has no unique source")
            attributes[name] = copy.deepcopy(matches[0]["facts"][name])
            evidence[name] = [_ideal_record_ref(matches[0])]
        return "submit_grounded_attributes", {
            "product_ref": _ideal_product_ref(listings[0]),
            "attributes": attributes,
            "evidence": evidence,
        }

    if rule_set == "source-priority-normalization-v1":
        fields = tuple(str(row) for row in facts.get("fields_to_resolve", ()))
        raw_priorities = facts.get("source_priority", ())
        if not isinstance(raw_priorities, Sequence) or isinstance(
            raw_priorities,
            (str, bytes, bytearray),
        ):
            raise _T2IdealEvaluationError("source_priority must be an array")
        priorities = tuple(_ideal_text(row, label="source priority") for row in raw_priorities)
        if not priorities or len(priorities) != len(set(priorities)):
            raise _T2IdealEvaluationError("source_priority must contain unique public source kinds")
        canonical_units = facts.get("canonical_units")
        if not isinstance(canonical_units, Mapping) or len(listings) != 1:
            raise _T2IdealEvaluationError("normalization policy is incomplete")
        resolutions: dict[str, Any] = {}
        for name in fields:
            candidates = [row for row in records if name in (row.get("facts") or {})]
            if not candidates:
                raise _T2IdealEvaluationError(f"field {name!r} has no source")
            ranked = [
                (priorities.index(str(row.get("kind"))), row)
                for row in candidates
                if str(row.get("kind")) in priorities
            ]
            if not ranked:
                raise _T2IdealEvaluationError(
                    f"field {name!r} has no source in the published priority policy"
                )
            best_rank = min(rank for rank, _row in ranked)
            selected = [row for rank, row in ranked if rank == best_rank]
            if len(selected) != 1:
                raise _T2IdealEvaluationError(
                    f"field {name!r} has an ambiguous highest-priority source"
                )
            value, unit = _ideal_convert_unit(
                selected[0]["facts"][name],
                _ideal_text(
                    canonical_units.get(name),
                    label=f"canonical unit {name}",
                ),
            )
            resolutions[name] = {
                "value": value,
                "unit": unit,
                "source_ref": _ideal_record_ref(selected[0]),
            }
        return "resolve_product_facts", {
            "product_ref": _ideal_product_ref(listings[0]),
            "resolutions": resolutions,
        }

    if rule_set == "ascending-certified-metric-v1":
        if not isinstance(policy, Mapping):
            raise _T2IdealEvaluationError("comparison policy is missing")
        metric = _ideal_text(policy.get("metric"), label="comparison metric")
        tie_field = _ideal_text(
            policy.get("sort_key_field"),
            label="stable tie key field",
        )
        if tie_field != "sort_key":
            raise _T2IdealEvaluationError("unsupported comparison tie key")
        ranked: list[tuple[int | float, str, str, Mapping[str, Any]]] = []
        seen_products: set[str] = set()
        seen_keys: set[tuple[int | float, str]] = set()
        for row in records:
            row_facts = row.get("facts")
            if not isinstance(row_facts, Mapping):
                raise _T2IdealEvaluationError("comparison source is malformed")
            value = row_facts.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise _T2IdealEvaluationError("comparison metric is not numeric")
            tie_key = _ideal_text(
                row_facts.get(tie_field),
                label="stable tie key",
            )
            product_ref = _ideal_text(row.get("subject_ref"), label="product ref")
            if product_ref in seen_products or (value, tie_key) in seen_keys:
                raise _T2IdealEvaluationError("comparison facts do not define a unique ranking")
            seen_products.add(product_ref)
            seen_keys.add((value, tie_key))
            ranked.append((value, tie_key, product_ref, row))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return "submit_grounded_comparison", {
            "ranked_product_refs": [item[2] for item in ranked],
            "evidence": {
                _ideal_text(row.get("subject_ref"), label="product ref"): [_ideal_record_ref(row)]
                for row in records
            },
        }

    if rule_set == "claim-evidence-equivalence-v1":
        classifications: list[dict[str, Any]] = []
        for claim in claims:
            content = claim.get("content")
            assertion = content.get("assertion") if isinstance(content, Mapping) else None
            if not isinstance(assertion, list) or len(assertion) != 2:
                raise _T2IdealEvaluationError("claim assertion is malformed")
            matches = [row for row in records if row.get("subject_ref") == claim.get("subject")]
            supported = (
                len(matches) == 1
                and (matches[0].get("facts") or {}).get(str(assertion[0])) == assertion[1]
            )
            classifications.append(
                {
                    "claim_ref": _ideal_text(claim.get("claim_ref"), label="claim ref"),
                    "status": "supported" if supported else "unsupported",
                    "evidence_source_refs": [_ideal_record_ref(row) for row in matches],
                }
            )
        return "classify_listing_claims", {"classifications": classifications}

    if rule_set == "stale-field-correction-v1":
        fields = tuple(str(row) for row in facts.get("fields_to_update", ()))
        if len(records) != 1 or len(listings) != 1:
            raise _T2IdealEvaluationError("listing correction facts are incomplete")
        source_facts = records[0].get("facts")
        if not isinstance(source_facts, Mapping) or any(
            name not in source_facts for name in fields
        ):
            raise _T2IdealEvaluationError("listing correction source is incomplete")
        return "update_listing_facts", {
            "listing_ref": _ideal_text(listings[0].get("sku_ref"), label="listing ref"),
            "changes": {name: copy.deepcopy(source_facts[name]) for name in fields},
            "evidence_source_refs": [_ideal_record_ref(records[0])],
        }

    if rule_set == "evidence-backed-attribute-response-v1":
        requested = tuple(str(row) for row in facts.get("requested_attributes", ()))
        if len(listings) != 1:
            raise _T2IdealEvaluationError("response facts need one listing")
        answers: dict[str, Any] = {}
        for name in requested:
            matches = [row for row in records if name in (row.get("facts") or {})]
            if len(matches) != 1:
                raise _T2IdealEvaluationError(f"response field {name!r} is ambiguous")
            answers[name] = {
                "value": copy.deepcopy(matches[0]["facts"][name]),
                "source_ref": _ideal_record_ref(matches[0]),
            }
        return "send_evidence_backed_response", {
            "product_ref": _ideal_product_ref(listings[0]),
            "answers": answers,
        }

    if rule_set == "comparative-claim-review-v1":
        decisions = []
        for row in _ideal_rows(
            facts.get("claim_propositions", ()),
            label="claim_propositions",
        ):
            decisions.append(
                {
                    "claim_ref": _ideal_text(row.get("claim_ref"), label="claim ref"),
                    "disposition": _ideal_comparative_disposition(
                        row,
                        records_by_ref,
                    ),
                    "evidence_source_refs": list(row.get("evidence_source_refs", ())),
                }
            )
        return "review_comparative_claims", {"decisions": decisions}

    if rule_set == "current-evidence-correction-v1":
        corrections = []
        for claim in claims:
            current = [
                row
                for row in records
                if row.get("subject_ref") == claim.get("subject")
                and "manufacturer_specification" in (row.get("facts") or {})
            ]
            if len(current) != 1:
                raise _T2IdealEvaluationError("claim has no unique current manufacturer source")
            specification = current[0]["facts"].get("manufacturer_specification")
            if not isinstance(specification, Mapping):
                raise _T2IdealEvaluationError("manufacturer fact is incomplete")
            corrections.append(
                {
                    "claim_ref": _ideal_text(claim.get("claim_ref"), label="claim ref"),
                    "disposition": _ideal_correction_disposition(
                        claim,
                        specification,
                    ),
                    "evidence_source_refs": [_ideal_record_ref(current[0])],
                }
            )
        return "correct_published_claims", {"corrections": corrections}

    raise _T2IdealEvaluationError(f"unsupported T2 rule set {rule_set!r}")


def _ideal_completed_reads(
    request: Mapping[str, Any],
) -> frozenset[tuple[str, str, str]]:
    completed: set[tuple[str, str, str]] = set()
    for observation in _ideal_rows(
        request.get("observations", ()),
        label="observations",
    ):
        batches = observation.get("observed_business_facts")
        if batches is None:
            continue
        for row in _ideal_rows(batches, label="observed_business_facts"):
            observation_kind = row.get("observation_kind")
            arguments = row.get("criteria")
            if not isinstance(observation_kind, str) or not isinstance(arguments, Mapping):
                continue
            for field in ("sku_ref", "record_ref", "claim_ref"):
                value = arguments.get(field)
                if isinstance(value, str) and value:
                    completed.add((observation_kind, field, value))
    return frozenset(completed)


def _ideal_next_grounding_read(
    request: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    if request.get("phase") not in {
        "buyer_grounded_result",
        "merchant_grounded_operation",
    }:
        return None
    allowed = _ideal_allowed_intents(request)
    completed = _ideal_completed_reads(request)
    for intent, observation_kind, argument, fact_name in (
        ("observe_listing", "listing", "sku_ref", "sku_refs"),
        (
            "observe_evidence_record",
            "evidence_record",
            "record_ref",
            "evidence_record_refs",
        ),
        (
            "observe_listing_claim",
            "listing_claim",
            "claim_ref",
            "claim_refs",
        ),
    ):
        if intent not in allowed:
            continue
        values = facts.get(fact_name, ())
        if not isinstance(values, Sequence) or isinstance(
            values,
            (str, bytes, bytearray),
        ):
            raise _T2IdealEvaluationError(f"{fact_name} must be an array")
        normalized = tuple(_ideal_text(value, label=fact_name) for value in values)
        if len(normalized) != len(set(normalized)):
            raise _T2IdealEvaluationError(f"{fact_name} contains duplicates")
        for value in normalized:
            if (observation_kind, argument, value) not in completed:
                return intent, {argument: value}
    return None


def _ideal_claim_enum(intent_row: Mapping[str, Any]) -> frozenset[str]:
    parameters = intent_row.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    claim = properties.get("claim_ref") if isinstance(properties, Mapping) else None
    values = claim.get("enum") if isinstance(claim, Mapping) else None
    if not isinstance(values, list):
        return frozenset()
    return frozenset(str(value) for value in values)


def _ideal_claim_choices(
    facts: Mapping[str, Any],
    report_kind: str,
    payload: Mapping[str, Any],
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    if report_kind == "send_evidence_backed_response":
        records, _, claims = _ideal_public_rows(facts)
        return tuple(
            (
                _ideal_text(claim.get("claim_ref"), label="claim ref"),
                "publish",
                tuple(
                    _ideal_record_ref(row)
                    for row in records
                    if row.get("subject_ref") == claim.get("subject")
                ),
            )
            for claim in claims
        )
    if report_kind == "review_comparative_claims":
        rows = _ideal_rows(payload.get("decisions"), label="decisions")
    elif report_kind == "correct_published_claims":
        rows = _ideal_rows(payload.get("corrections"), label="corrections")
    else:
        return ()
    return tuple(
        (
            _ideal_text(row.get("claim_ref"), label="claim ref"),
            _ideal_text(row.get("disposition"), label="claim disposition"),
            tuple(
                _ideal_text(value, label="evidence source ref")
                for value in row.get("evidence_source_refs", ())
            ),
        )
        for row in rows
    )


def _evaluate_t2_ideal_request(
    request: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Choose from one provider request without task, oracle, or World access."""

    allowed = _ideal_allowed_intents(request)
    phase = _ideal_text(request.get("phase"), label="phase")
    if phase == "buyer_discovery":
        if "search" not in allowed:
            raise _T2IdealEvaluationError("buyer discovery cannot search")
        return "search", {"query": ""}

    facts = _task_business_facts(request)
    next_read = _ideal_next_grounding_read(request, facts)
    if next_read is not None:
        return next_read
    report_kind, payload = _ideal_report_solution(facts)

    if "update_listing" in allowed:
        changes = payload.get("changes")
        if not isinstance(changes, Mapping):
            raise _T2IdealEvaluationError("listing update has no public changes")
        return "update_listing", {"changes": copy.deepcopy(dict(changes))}

    intent_by_disposition = {
        "publish": "publish_listing_claim",
        "narrow": "correct_listing_claim",
        "correct": "correct_listing_claim",
        "retract": "retract_listing_claim",
    }
    allowed_rows = _ideal_allowed_rows(request)
    for claim_ref, disposition, evidence_refs in _ideal_claim_choices(
        facts,
        report_kind,
        payload,
    ):
        intent = intent_by_disposition.get(disposition)
        if intent is None or intent not in allowed:
            continue
        matching = next(
            (
                row
                for row in allowed_rows
                if row.get("intent") == intent and claim_ref in _ideal_claim_enum(row)
            ),
            None,
        )
        if matching is not None:
            return intent, {
                "claim_ref": claim_ref,
                "evidence_record_refs": list(evidence_refs),
            }

    if "respond_inquiry" in allowed:
        _, listings, _ = _ideal_public_rows(facts)
        if len(listings) != 1:
            raise _T2IdealEvaluationError("inquiry response needs one listing")
        return "respond_inquiry", {
            "payload": {
                "sku_ref": _ideal_text(
                    listings[0].get("sku_ref"),
                    label="listing ref",
                ),
                "product": _ideal_product_ref(listings[0]),
                "category": "attribute",
                "answer": copy.deepcopy(payload),
            }
        }

    if "submit_decision_record" in allowed:
        return "submit_decision_record", {
            "outcome": "completed",
            "summary": "Recorded the provider-visible evidence conclusion.",
            "payload": copy.deepcopy(payload),
        }
    raise _T2IdealEvaluationError(f"T2 phase {phase!r} has no supported terminal business intent")


class _T2RequestOnlyIdealChannel:
    """Ideal policy whose only semantic input is the current provider request."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("T2 business decision requires an Agent decision id")
        request = _business_request(user_prompt)
        intent, arguments = _evaluate_t2_ideal_request(request)
        return _business_response(request, intent, arguments)


class _T2BusinessChannel:
    """Typed T2 policy that sees only business intents and public refs."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def __init__(
        self,
        problem: _ProblemT2,
        submission: Mapping[str, Any],
        *,
        action_submission: Mapping[str, Any] | None = None,
        claim_plan: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.problem = problem
        self.submission = copy.deepcopy(dict(submission))
        self.action_submission = copy.deepcopy(dict(action_submission or {}))
        self.claim_plan = tuple(
            copy.deepcopy(dict(row))
            for row in (problem.claim_intents if claim_plan is None else claim_plan)
        )
        self._active_decision_id: str | None = None
        self._pending: list[tuple[str, dict[str, Any]]] = []
        self._terminal: tuple[str, dict[str, Any]] | None = None
        self._grounded = False
        self._next_claim = 0

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("T2 business decision requires an Agent decision id")
        request = _business_request(user_prompt)
        if decision_id != self._active_decision_id:
            self._active_decision_id = decision_id
            self._pending = []
            self._terminal = None
            self._prepare_turn(request)
        if self._pending:
            intent, arguments = self._pending.pop(0)
            return _business_response(request, intent, arguments)
        if self._terminal is None:
            raise ValueError("T2 business turn has no terminal choice")
        intent, arguments = self._terminal
        return _business_response(request, intent, arguments)

    def _prepare_turn(self, request: Mapping[str, Any]) -> None:
        phase = str(request.get("phase", ""))
        if phase == "buyer_discovery":
            self._terminal = ("search", {"query": ""})
            return
        if phase == "buyer_grounded_result":
            if not self._grounded:
                self._pending = _grounding_choices(request)
                self._grounded = True
            self._terminal = self._report_choice(request)
            return
        if phase == "merchant_grounded_operation":
            if not self._grounded:
                self._pending = _grounding_choices(request)
                self._grounded = True
            self._terminal = (
                self._update_choice()
                if self.problem.lane == "truthful_listing_update"
                else self._next_claim_choice(request)
            )
            return
        if phase == "merchant_catalog_result":
            self._terminal = self._report_choice(request)
            return
        if phase == "merchant_claim_progress":
            if self._next_claim < len(self.claim_plan):
                self._terminal = self._next_claim_choice(request)
            elif self.problem.lane == "evidence_backed_response":
                self._terminal = self._response_choice(request)
            else:
                self._terminal = self._report_choice(request)
            return
        if phase == "merchant_response_result":
            self._terminal = self._report_choice(request)
            return
        raise ValueError(f"unexpected T2 business phase {phase!r}")

    def _aliases(self, request: Mapping[str, Any]) -> dict[str, str]:
        return _reference_aliases(self.problem, _task_business_facts(request))

    def _report_choice(
        self,
        request: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        payload = self.submission.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("T2 result submission has no semantic payload")
        return (
            "submit_decision_record",
            {
                "outcome": "completed",
                "summary": "Recorded the evidence-grounded business conclusion.",
                "payload": _publicize_business_value(payload, self._aliases(request)),
            },
        )

    def _update_choice(self) -> tuple[str, dict[str, Any]]:
        payload = self.action_submission.get("payload")
        changes = payload.get("changes") if isinstance(payload, Mapping) else None
        if not isinstance(changes, Mapping):
            raise ValueError("T2 listing update has no grounded changes")
        return "update_listing", {"changes": copy.deepcopy(dict(changes))}

    def _next_claim_choice(
        self,
        request: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if self._next_claim >= len(self.claim_plan):
            raise ValueError("T2 merchant claim plan is exhausted")
        raw = self.claim_plan[self._next_claim]
        self._next_claim += 1
        operation = str(raw.get("operation", ""))
        intent_by_operation = {
            "publish": "publish_listing_claim",
            "correct": "correct_listing_claim",
            "retract": "retract_listing_claim",
        }
        try:
            intent = intent_by_operation[operation]
        except KeyError as exc:
            raise ValueError(f"unsupported T2 claim operation {operation!r}") from exc
        # The model selects only the business operation, target claim, and
        # cited evidence.  Standard correction content and retraction reason
        # are injected from Agent-private task authority after validation.
        arguments = {
            "claim_id": raw["claim_id"],
            "evidence_record_ids": list(raw.get("evidence_record_ids", ())),
        }
        return intent, _publicize_business_value(arguments, self._aliases(request))

    def _response_choice(
        self,
        request: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        payload = self.submission.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("T2 inquiry response has no semantic answer")
        aliases = self._aliases(request)
        return (
            "respond_inquiry",
            {
                "payload": {
                    "sku_ref": aliases[self.problem.primary_sku],
                    "product": aliases[str(self.problem.catalog[0]["product_id"])],
                    "category": "attribute",
                    "answer": _publicize_business_value(payload, aliases),
                }
            },
        )


class _BuyerResponseAckChannel:
    """Typed counterpart that acknowledges one evidence-backed response."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt, decision_id
        request = _business_request(user_prompt)
        rows = request.get("allowed_intents")
        send = next(
            (
                row
                for row in rows or ()
                if isinstance(row, Mapping) and row.get("intent") == "send_message"
            ),
            None,
        )
        if not isinstance(send, Mapping):
            raise ValueError("T2 response acknowledgement is unavailable")
        parameters = send.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        arguments = (
            {}
            if isinstance(properties, Mapping) and not properties
            else {"payload": {"category": "response_received"}}
        )
        return _business_response(request, "send_message", arguments)


class _NoReplyT2Channel:
    """Typed inert counterpart; normal T2 terminal turns bypass it."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt, decision_id
        request = _business_request(user_prompt)
        return _business_response(
            request,
            "finish",
            {"reason": "No counterpart business action is required."},
        )


def _no_reply_channel() -> _NoReplyT2Channel:
    return _NoReplyT2Channel()


def _ideal_channel(problem: _ProblemT2) -> _T2RequestOnlyIdealChannel:
    del problem
    return _T2RequestOnlyIdealChannel()


def _mutated_submission(problem: _ProblemT2) -> dict[str, Any]:
    def different_value(value: Any) -> Any:
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 1
        if isinstance(value, float):
            return value + 1.0
        if isinstance(value, str):
            return value + " (unsupported)"
        raise ValueError("T2 mutation has no schema-preserving alternative value")

    submission = copy.deepcopy(problem.expected_submission)
    payload = submission["payload"]
    if problem.lane == "authoritative_attribute_read":
        key = sorted(payload["attributes"])[0]
        payload["attributes"][key] = different_value(payload["attributes"][key])
    elif problem.lane in {"conflict_and_normalization", "evidence_backed_response"}:
        outer = "resolutions" if problem.lane == "conflict_and_normalization" else "answers"
        key = sorted(payload[outer])[0]
        payload[outer][key]["value"] = different_value(payload[outer][key]["value"])
    elif problem.lane == "grounded_comparison":
        payload["ranked_product_ids"][0], payload["ranked_product_ids"][1] = (
            payload["ranked_product_ids"][1],
            payload["ranked_product_ids"][0],
        )
    elif problem.lane == "unsupported_claim_detection":
        row = payload["classifications"][-1]
        row["status"] = "supported" if row["status"] != "supported" else "unsupported"
    elif problem.lane == "truthful_listing_update":
        key = sorted(payload["changes"])[0]
        payload["changes"][key] = different_value(payload["changes"][key])
    elif problem.lane == "comparative_claim_limits":
        row = payload["decisions"][0]
        row["disposition"] = "retract" if row["disposition"] != "retract" else "publish"
    elif problem.lane == "correction_and_retraction":
        row = payload["corrections"][0]
        row["disposition"] = "retract" if row["disposition"] != "retract" else "correct"
    else:  # pragma: no cover - builder dispatch is exhaustive.
        raise AssertionError(f"unsupported T2 lane {problem.lane!r}")
    return submission


def targeted_mutation_channel_t2(task_id: str) -> _T2BusinessChannel:
    """Return a real-runtime policy with one capability-relevant wrong fact."""

    problem = _problem_for(task_id)
    submission = _mutated_submission(problem)
    claim_plan = [copy.deepcopy(dict(row)) for row in problem.claim_intents]
    if problem.lane in {"comparative_claim_limits", "correction_and_retraction"}:
        if not claim_plan or claim_plan[0].get("operation") != "correct":
            raise ValueError("T2 claim mutation requires one correctable first claim")
        # A legal retraction is intentionally the wrong business choice here.
        # Platform/World still accept and close it, making the capability loss
        # scoreable instead of disguising it as a false final report only.
        claim_plan[0]["operation"] = "retract"
    return _T2BusinessChannel(
        problem,
        submission,
        action_submission=problem.expected_submission,
        claim_plan=claim_plan,
    )


def _catalog_index(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tables = snapshot.get("tables")
    rows = tables.get("catalog") if isinstance(tables, Mapping) else None
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("sku_id")): row for row in rows if isinstance(row, dict) and row.get("sku_id")
    }


def _response_by_request(
    evidence: RuntimeEvidenceBundleV2,
    *,
    actor_id: str,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for envelope in evidence.actions(kind="world.response"):
        if envelope.get("from") != "world" or envelope.get("to") != actor_id:
            continue
        request_id = envelope.get("in_reply_to")
        payload = envelope["action"].get("payload")
        if isinstance(request_id, str) and isinstance(payload, Mapping):
            output[request_id] = payload
    return output


def _attested_tool_rows(
    evidence: RuntimeEvidenceBundleV2,
    *,
    actor_id: str,
    tool: str,
    field: str,
    remote_kind: str,
) -> dict[str, Mapping[str, Any]]:
    """Return only tool results that exactly answer an audited World request."""

    found: dict[str, Mapping[str, Any]] = {}
    responses = _response_by_request(evidence, actor_id=actor_id)
    for envelope in evidence.actions(kind=remote_kind, actor_id=actor_id):
        payload = envelope["action"].get("payload") or {}
        identity = str(payload.get(field, ""))
        result = responses.get(str(envelope.get("msg_id", "")))
        if identity and isinstance(result, Mapping) and str(result.get(field, "")) == identity:
            found[identity] = result
    for trace in evidence.trace_rows:
        if trace.get("agent_id") != actor_id or not tracker_row_has_usable_completed_steps(trace):
            continue
        for step in trace.get("steps", ()):
            if not isinstance(step, dict) or step.get("kind") != "tool_call":
                continue
            data = step.get("data") or {}
            for result_slot in data.get("results", ()):
                if not isinstance(result_slot, dict) or result_slot.get("tool") != tool:
                    continue
                args = result_slot.get("args")
                result = result_slot.get("result")
                identity = str(args.get(field, "")) if isinstance(args, Mapping) else ""
                if (
                    identity
                    and isinstance(result, Mapping)
                    and str(result.get(field, "")) == identity
                ):
                    found[identity] = result
    return found


def _grounded_listing_rows(
    evidence: RuntimeEvidenceBundleV2, actor_id: str
) -> dict[str, Mapping[str, Any]]:
    return _attested_tool_rows(
        evidence,
        actor_id=actor_id,
        tool="world.get_listing",
        field="sku_id",
        remote_kind="world.read_catalog",
    )


def _grounded_skus(evidence: RuntimeEvidenceBundleV2, actor_id: str) -> frozenset[str]:
    return frozenset(_grounded_listing_rows(evidence, actor_id))


def _ranked_skus(evidence: RuntimeEvidenceBundleV2) -> frozenset[str]:
    found: set[str] = set()
    for exchange in evidence.accepted_platform_exchanges(
        kind="commerce.search",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.rank_offers",
    ):
        for envelope in exchange.responses:
            payload = envelope["action"].get("payload") or {}
            for row in payload.get("candidates", ()):
                if isinstance(row, dict) and row.get("sku_id"):
                    found.add(str(row["sku_id"]))
    return frozenset(found)


def _grounded_tool_rows(
    evidence: RuntimeEvidenceBundleV2,
    *,
    actor_id: str,
    tool: str,
    field: str,
    remote_kind: str,
) -> dict[str, Mapping[str, Any]]:
    return _attested_tool_rows(
        evidence,
        actor_id=actor_id,
        tool=tool,
        field=field,
        remote_kind=remote_kind,
    )


def _grounded_record_ids(evidence: RuntimeEvidenceBundleV2, actor_id: str) -> frozenset[str]:
    return frozenset(_grounded_record_rows(evidence, actor_id))


def _grounded_record_rows(
    evidence: RuntimeEvidenceBundleV2, actor_id: str
) -> dict[str, Mapping[str, Any]]:
    return _grounded_tool_rows(
        evidence,
        actor_id=actor_id,
        tool="world.get_evidence_record",
        field="record_id",
        remote_kind="world.read_evidence_record",
    )


def _grounded_claim_ids(evidence: RuntimeEvidenceBundleV2, actor_id: str) -> frozenset[str]:
    return frozenset(_grounded_claim_rows(evidence, actor_id))


def _grounded_claim_rows(
    evidence: RuntimeEvidenceBundleV2, actor_id: str
) -> dict[str, Mapping[str, Any]]:
    return _grounded_tool_rows(
        evidence,
        actor_id=actor_id,
        tool="world.get_listing_claim",
        field="claim_id",
        remote_kind="world.read_listing_claim",
    )


def _submission_from_evidence(
    problem: _ProblemT2,
    evidence: RuntimeEvidenceBundleV2,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    actor_id = _BUYER_ID if problem.definition.evaluated_role == "buyer" else _MERCHANT_ID
    rows = evidence.accepted_actor_results(
        action_kind="commerce.submit_decision_record",
        actor_id=actor_id,
        task_id=problem.definition.task_id,
    )
    if len(rows) != 1:
        return None, None
    record = rows[0]
    report = record.get("report")
    content = report.get("content") if isinstance(report, Mapping) else None
    details = content.get("details") if isinstance(content, Mapping) else None
    if (
        not isinstance(content, Mapping)
        or content.get("outcome") != "completed"
        or not isinstance(details, Mapping)
        or set(details) != {"submission"}
        or not isinstance(details.get("submission"), Mapping)
    ):
        return None, record
    return copy.deepcopy(dict(details["submission"])), record


def _fraction(values: Sequence[bool]) -> float:
    return sum(bool(value) for value in values) / max(1, len(values))


def _rows_by_id(value: Any, key: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return {}
    return {
        str(row[key]): row
        for row in value
        if isinstance(row, Mapping) and isinstance(row.get(key), str)
    }


def _citation_ok(actual: Any, expected: Any, presented: frozenset[str]) -> bool:
    actual_ids = {actual} if isinstance(actual, str) else set(actual or ())
    expected_ids = {expected} if isinstance(expected, str) else set(expected or ())
    return actual_ids == expected_ids and expected_ids <= presented


def _submission_metrics(
    problem: _ProblemT2,
    actual_submission: Mapping[str, Any] | None,
    presented_sources: frozenset[str],
) -> tuple[float, float, float, dict[str, Any]]:
    expected = problem.expected_submission
    actual = dict(actual_submission or {})
    expected_payload = expected["payload"]
    actual_payload = actual.get("payload")
    candidate = dict(actual_payload) if isinstance(actual_payload, Mapping) else {}
    kind_ok = actual.get("kind") == expected["kind"]
    lane = problem.lane
    scope_checks: list[bool] = [kind_ok]
    truth_checks: list[bool] = []
    citation_checks: list[bool] = []

    if lane == "authoritative_attribute_read":
        expected_values = expected_payload["attributes"]
        actual_values = candidate.get("attributes")
        actual_values = dict(actual_values) if isinstance(actual_values, Mapping) else {}
        actual_evidence = candidate.get("evidence")
        actual_evidence = dict(actual_evidence) if isinstance(actual_evidence, Mapping) else {}
        scope_checks.extend(
            [
                candidate.get("product_id") == expected_payload["product_id"],
                set(actual_values) == set(expected_values),
                set(actual_evidence) == set(expected_payload["evidence"]),
            ]
        )
        truth_checks.extend(
            actual_values.get(key) == value for key, value in expected_values.items()
        )
        citation_checks.extend(
            _citation_ok(actual_evidence.get(key), sources, presented_sources)
            for key, sources in expected_payload["evidence"].items()
        )
    elif lane in {"conflict_and_normalization", "evidence_backed_response"}:
        outer = "resolutions" if lane == "conflict_and_normalization" else "answers"
        expected_rows = expected_payload[outer]
        actual_rows = candidate.get(outer)
        actual_rows = dict(actual_rows) if isinstance(actual_rows, Mapping) else {}
        scope_checks.extend(
            [
                candidate.get("product_id") == expected_payload["product_id"],
                set(actual_rows) == set(expected_rows),
            ]
        )
        for key, expected_row in expected_rows.items():
            actual_row = actual_rows.get(key)
            actual_row = dict(actual_row) if isinstance(actual_row, Mapping) else {}
            semantic_keys = (
                ("value", "unit") if lane == "conflict_and_normalization" else ("value",)
            )
            truth_checks.append(
                all(actual_row.get(field) == expected_row.get(field) for field in semantic_keys)
            )
            citation_checks.append(
                _citation_ok(
                    actual_row.get("source_id"),
                    expected_row.get("source_id"),
                    presented_sources,
                )
            )
    elif lane == "grounded_comparison":
        expected_rank = list(expected_payload["ranked_product_ids"])
        actual_rank = candidate.get("ranked_product_ids")
        actual_rank = list(actual_rank) if isinstance(actual_rank, (list, tuple)) else []
        scope_checks.append(
            len(actual_rank) == len(expected_rank) and set(actual_rank) == set(expected_rank)
        )
        truth_checks.extend(
            index < len(actual_rank) and actual_rank[index] == product_id
            for index, product_id in enumerate(expected_rank)
        )
        actual_evidence = candidate.get("evidence")
        actual_evidence = dict(actual_evidence) if isinstance(actual_evidence, Mapping) else {}
        scope_checks.append(set(actual_evidence) == set(expected_payload["evidence"]))
        citation_checks.extend(
            _citation_ok(actual_evidence.get(key), sources, presented_sources)
            for key, sources in expected_payload["evidence"].items()
        )
    elif lane == "unsupported_claim_detection":
        expected_rows = _rows_by_id(expected_payload["classifications"], "claim_id")
        actual_rows = _rows_by_id(candidate.get("classifications"), "claim_id")
        scope_checks.append(set(actual_rows) == set(expected_rows))
        for claim_id, expected_row in expected_rows.items():
            actual_row = actual_rows.get(claim_id, {})
            truth_checks.append(actual_row.get("status") == expected_row.get("status"))
            citation_checks.append(
                _citation_ok(
                    actual_row.get("evidence_source_ids"),
                    expected_row.get("evidence_source_ids"),
                    presented_sources,
                )
            )
    elif lane == "truthful_listing_update":
        expected_changes = expected_payload["changes"]
        actual_changes = candidate.get("changes")
        actual_changes = dict(actual_changes) if isinstance(actual_changes, Mapping) else {}
        scope_checks.extend(
            [
                candidate.get("listing_id") == expected_payload["listing_id"],
                set(actual_changes) == set(expected_changes),
            ]
        )
        truth_checks.extend(
            actual_changes.get(key) == value for key, value in expected_changes.items()
        )
        citation_checks.append(
            _citation_ok(
                candidate.get("evidence_source_ids"),
                expected_payload["evidence_source_ids"],
                presented_sources,
            )
        )
    elif lane in {"comparative_claim_limits", "correction_and_retraction"}:
        outer = "decisions" if lane == "comparative_claim_limits" else "corrections"
        expected_rows = _rows_by_id(expected_payload[outer], "claim_id")
        actual_rows = _rows_by_id(candidate.get(outer), "claim_id")
        scope_checks.append(set(actual_rows) == set(expected_rows))
        semantic_keys = ("disposition",)
        for claim_id, expected_row in expected_rows.items():
            actual_row = actual_rows.get(claim_id, {})
            truth_checks.append(
                all(actual_row.get(key) == expected_row.get(key) for key in semantic_keys)
            )
            citation_checks.append(
                _citation_ok(
                    actual_row.get("evidence_source_ids"),
                    expected_row.get("evidence_source_ids"),
                    presented_sources,
                )
            )
    else:  # pragma: no cover - builders cover every lane.
        raise AssertionError(f"unsupported T2 lane {lane!r}")

    scope = _fraction(scope_checks)
    truth = _fraction(truth_checks)
    citations = _fraction(citation_checks)
    return (
        scope,
        truth,
        citations,
        {
            "expected_kind": expected["kind"],
            "actual_kind": actual.get("kind"),
            "scope_checks_passed": sum(scope_checks),
            "scope_checks_total": len(scope_checks),
            "truth_checks_passed": sum(truth_checks),
            "truth_checks_total": len(truth_checks),
            "citation_checks_passed": sum(citation_checks),
            "citation_checks_total": len(citation_checks),
            "presented_source_ids": sorted(presented_sources),
        },
    )


def _search_session_binding(
    problem: _ProblemT2,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedSearchSessionEvidence | None,
) -> tuple[bool, dict[str, Any]]:
    if verified is None or not verified.sessions:
        return False, {
            "accepted_search_exchanges": 0 if verified is None else len(verified.sessions),
            "authority_verified": False,
        }
    bindings: list[dict[str, Any]] = []
    for join in verified.sessions:
        request = join.exchange.request
        response = join.response
        request_payload = request["action"].get("payload") or {}
        rank_payload = response["action"].get("payload") or {}
        session = rank_payload.get("search_session")
        session_id = str(rank_payload.get("session_id", ""))
        session_digest = str(rank_payload.get("session_digest", ""))
        final_session = join.session
        candidate_rows = tuple(
            row for row in rank_payload.get("candidates", ()) if isinstance(row, Mapping)
        )
        candidate_ids = {str(row.get("sku_id", "")) for row in candidate_rows}
        # Candidate relevance is determined by the model-authored search
        # query.  An empty or incomplete result set is therefore capability
        # evidence, not proof that Platform/World authority failed.  The
        # environment prerequisite only requires every returned candidate to
        # remain bound to the authoritative search session and mandate.
        candidates_authority_bound = all(
            row.get("session_id") == session_id
            and row.get("mandate_id") == problem.definition.task_id
            for row in candidate_rows
        )
        commit_bound = join.commit.get("subject_id") == session_id
        bindings.append(
            {
                "rank_response_id": response.get("msg_id"),
                "session_id": session_id,
                "session_digest": session_digest,
                "mandate_bound": (
                    request_payload.get("mandate_id") == problem.definition.task_id
                    and rank_payload.get("mandate_id") == problem.definition.task_id
                    and isinstance(session, Mapping)
                    and session.get("mandate_id") == problem.definition.task_id
                ),
                "session_projection_bound": (
                    isinstance(session, Mapping)
                    and session.get("session_id") == session_id
                    and session.get("session_digest") == session_digest
                    and isinstance(final_session, Mapping)
                    and _strip_schema_versions(session) == final_session
                ),
                "candidates_authority_bound": candidates_authority_bound,
                "required_candidates_returned": (candidate_ids == set(problem.required_skus)),
                "world_search_session_bound": commit_bound,
                "world_search_commit_id": (join.commit.get("commit_id") if commit_bound else None),
            }
        )
    valid = all(
        row["mandate_bound"]
        and row["session_projection_bound"]
        and row["candidates_authority_bound"]
        and row["world_search_session_bound"]
        for row in bindings
    )
    return bool(valid), {
        "accepted_search_exchanges": len(verified.sessions),
        "search_session_bindings": bindings,
    }


def _claim_snapshot_projection(claim: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _strip_schema_versions(claim.get(key))
        for key in (
            "claim_id",
            "listing_id",
            "merchant_id",
            "subject",
            "issuer_id",
            "versions",
        )
    }


def _catalog_mutation_binding(
    problem: _ProblemT2,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedCatalogMutationEvidence | None,
) -> tuple[bool, dict[str, Any]]:
    """Consume the core Runtime, Platform, and World catalog authority join."""

    if not isinstance(verified, VerifiedCatalogMutationEvidence):
        return False, {
            "verified_catalog_mutations": 0,
            "authority_error": "catalog contract returned the wrong evidence type",
        }

    mutations = verified.mutations
    if not mutations:
        return False, {"verified_catalog_mutations": len(mutations)}
    expected_changes = problem.expected_submission["payload"]["changes"]
    expected_intent = {
        "op": "update",
        "sku_id": problem.primary_sku,
        "fields": {"attributes": copy.deepcopy(dict(expected_changes))},
    }
    expected_request_fingerprint = canonical_sha256(expected_intent)
    bindings = tuple(
        {
            "catalog_response_id": mutation.response.get("msg_id"),
            "request_intent_matches": mutation.intent == expected_intent,
            "observed_request_fingerprint": canonical_sha256(mutation.intent),
            "request_authority_bound": (
                mutation.request_fingerprint == canonical_sha256(mutation.intent)
                and mutation.request.get("from") == _MERCHANT_ID
                and mutation.request.get("to") == "platform:catalog"
            ),
            "outcome_bound": (
                mutation.authority_operation.get("outcome_listing") == mutation.outcome_listing
            ),
            "world_catalog_commit_id": mutation.commit.get("commit_id"),
        }
        for mutation in mutations
    )
    # Exact Agent/Platform/World closure is an environment prerequisite.  The
    # comparison with the frozen expected changes is model capability
    # evidence and must not decide whether the environment itself is valid.
    valid = all(row["request_authority_bound"] and row["outcome_bound"] for row in bindings)
    return valid, {
        "verified_catalog_mutations": len(mutations),
        "expected_request_fingerprint": expected_request_fingerprint,
        "catalog_mutation_bindings": bindings,
        "world_catalog_commit_count": len(mutations),
        "authority_operation_count": len(mutations),
    }


def _claim_intent_business_semantics_match(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Compare the model choice plus deterministic Agent compiler binding.

    ``content`` and ``reason`` are no longer authored by the model.  Exact
    comparison of those private compiled fields is an integrity check, not a
    natural-language capability score.
    """

    for field in ("claim_id", "listing_id", "operation", "subject"):
        if observed.get(field) != expected.get(field):
            return False
    operation = expected.get("operation")
    observed_evidence = set(observed.get("evidence_record_ids", ()))
    expected_evidence = set(expected.get("evidence_record_ids", ()))
    if observed_evidence != expected_evidence:
        return False
    if operation == "correct":
        return canonical_sha256(observed.get("content")) == canonical_sha256(
            expected.get("content")
        )
    if operation == "retract":
        return observed.get("reason") == expected.get("reason")
    return operation == "publish"


def _claim_compiler_fields_match(
    observed: Mapping[str, Any],
    templates: Mapping[tuple[str, str], Mapping[str, Any]],
) -> bool:
    """Verify Agent-injected fields for whichever branch the model selected."""

    claim_id = observed.get("claim_id")
    operation = observed.get("operation")
    if not isinstance(claim_id, str) or not isinstance(operation, str):
        return False
    template = templates.get((claim_id, operation))
    if template is None:
        return False
    if operation == "correct":
        return canonical_sha256(observed.get("content")) == canonical_sha256(
            template.get("content")
        )
    if operation == "retract":
        return observed.get("reason") == template.get("reason")
    return operation == "publish" and not template


def _claim_compiler_template_index(
    problem: _ProblemT2,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    binding = _claim_compilation_templates(problem)
    if binding is None:
        return {}
    return {
        (str(row["claim_id"]), str(operation)): template
        for row in binding["claims"]
        for operation, template in row["operation_templates"].items()
    }


def _completion_environment_binding(
    problem: _ProblemT2,
    evidence: RuntimeEvidenceBundleV2,
    actor_record: Mapping[str, Any] | None,
    authority: (
        VerifiedSearchSessionEvidence
        | VerifiedCatalogMutationEvidence
        | VerifiedListingClaimEvidence
        | None
    ),
) -> dict[str, Any]:
    """Observe causal environment closure without awarding capability credit."""

    report_parent = (
        str(actor_record.get("in_reply_to"))
        if isinstance(actor_record, Mapping) and actor_record.get("in_reply_to")
        else None
    )
    if problem.definition.evaluated_role == "buyer":
        search_authority = (
            authority if isinstance(authority, VerifiedSearchSessionEvidence) else None
        )
        search_bound, search_detail = _search_session_binding(
            problem,
            evidence,
            search_authority,
        )
        try:
            exchanges = evidence.accepted_platform_exchanges(
                kind="commerce.search",
                actor_id=_BUYER_ID,
                endpoint="platform:aggregator",
                response_kind="platform.rank_offers",
            )
        except RuntimeEvidenceError:
            exchanges = ()
        rank_ids = {
            str(row.get("msg_id"))
            for exchange in exchanges
            for row in exchange.responses
            if (row.get("action") or {}).get("kind") == "platform.rank_offers"
        }
        operation_environment_verified = (
            search_bound
            and search_authority is not None
            and len(exchanges) == len(search_authority.sessions)
            and len(rank_ids) == len(exchanges)
        )
        report_causality_verified = actor_record is None or (
            operation_environment_verified and report_parent in rank_ids
        )
        detail = {
            **search_detail,
            "rank_response_ids": sorted(rank_ids),
            "actor_report_parent": report_parent,
        }
    elif problem.lane == "truthful_listing_update":
        catalog_authority = (
            authority if isinstance(authority, VerifiedCatalogMutationEvidence) else None
        )
        catalog_bound, catalog_detail = _catalog_mutation_binding(
            problem,
            evidence,
            catalog_authority,
        )
        ack_ids = {
            str(row.get("catalog_response_id"))
            for row in catalog_detail.get("catalog_mutation_bindings", ())
            if row.get("catalog_response_id")
        }
        operation_environment_verified = catalog_bound
        report_causality_verified = actor_record is None or (
            operation_environment_verified and report_parent in ack_ids
        )
        detail = {
            **catalog_detail,
            "catalog_response_ids": sorted(ack_ids),
            "actor_report_parent": report_parent,
        }
    else:
        claim_authority = authority if isinstance(authority, VerifiedListingClaimEvidence) else None
        joins = () if claim_authority is None else claim_authority.claims
        observed_intents: dict[str, Mapping[str, Any]] = {}
        response_ids_by_claim: dict[str, str] = {}
        world_commit_by_claim: dict[str, bool] = {}
        world_commit_ids_by_claim: dict[str, str] = {}
        for join in joins:
            request_payload = join.intent
            claim_id = request_payload.get("claim_id")
            if isinstance(claim_id, str):
                observed_intents[claim_id] = request_payload
            response = join.response
            claim = (response.get("action") or {}).get("payload", {}).get("claim")
            if isinstance(claim, Mapping) and isinstance(claim.get("claim_id"), str):
                response_claim_id = str(claim["claim_id"])
                response_ids_by_claim[response_claim_id] = str(response.get("msg_id"))
                world_commit_by_claim[response_claim_id] = True
                world_commit_ids_by_claim[response_claim_id] = str(join.commit.get("commit_id"))
        expected_intents = {str(row["claim_id"]): row for row in problem.claim_intents}
        compiler_templates = _claim_compiler_template_index(problem)
        if any(
            not _claim_compiler_fields_match(
                intent,
                compiler_templates,
            )
            for intent in observed_intents.values()
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T2 Agent claim compiler injected noncanonical claim fields"
            )
        matched_intent_ids = {
            claim_id
            for claim_id, intent in observed_intents.items()
            if claim_id in expected_intents
            and _claim_intent_business_semantics_match(
                intent,
                expected_intents[claim_id],
            )
        }
        unexpected_intent_ids = set(observed_intents) - set(expected_intents)
        operation_environment_verified = claim_authority is not None
        participant_response_present = False
        participant_response_environment_verified = True
        report_causality_verified = actor_record is None
        if problem.lane == "evidence_backed_response":
            replies = evidence.actions(kind="commerce.respond_inquiry", actor_id=_MERCHANT_ID)
            participant_response_present = bool(replies)
            acknowledgements = [
                row
                for row in evidence.actions(kind="commerce.send_message", actor_id=_BUYER_ID)
                if (row["action"].get("payload") or {}).get("category") == "response_received"
            ]
            reply_ids = {str(row.get("msg_id")) for row in replies if row.get("msg_id")}
            acknowledgement_parent_ids = {
                str(row.get("in_reply_to")) for row in acknowledgements if row.get("in_reply_to")
            }
            acknowledgement_ids = {
                str(row.get("msg_id")) for row in acknowledgements if row.get("msg_id")
            }
            terminal_claim_response_id = (
                response_ids_by_claim.get(str(joins[-1].intent.get("claim_id"))) if joins else None
            )
            participant_response_environment_verified = (
                bool(replies)
                and len(acknowledgements) == len(replies)
                and acknowledgement_parent_ids == reply_ids
                and bool(joins)
                and all(row.get("in_reply_to") == terminal_claim_response_id for row in replies)
            )
            report_causality_verified = actor_record is None or (
                participant_response_environment_verified and report_parent in acknowledgement_ids
            )
        elif actor_record is not None:
            report_causality_verified = bool(joins) and report_parent == (
                response_ids_by_claim.get(str(joins[-1].intent.get("claim_id")))
            )
        detail = {
            "accepted_claim_exchanges": len(joins),
            "expected_claim_exchanges": len(problem.claim_intents),
            "claim_intents_match": (
                matched_intent_ids == set(expected_intents) and not unexpected_intent_ids
            ),
            "matched_claim_intent_ids": sorted(matched_intent_ids),
            "unexpected_claim_intent_ids": sorted(unexpected_intent_ids),
            "claim_response_ids": response_ids_by_claim,
            "world_claim_commit_bound": world_commit_by_claim,
            "world_claim_commit_ids": world_commit_ids_by_claim,
            "participant_response_present": participant_response_present,
            "participant_response_environment_verified": (
                participant_response_environment_verified
            ),
            "actor_report_parent": report_parent,
        }
    if isinstance(authority, VerifiedSearchSessionEvidence):
        claimed_commits = tuple(row.commit for row in authority.sessions)
    elif isinstance(authority, VerifiedCatalogMutationEvidence):
        claimed_commits = tuple(row.commit for row in authority.mutations)
    elif isinstance(authority, VerifiedListingClaimEvidence):
        claimed_commits = tuple(row.commit for row in authority.claims)
    else:
        claimed_commits = ()
    commit_claims = verify_exact_transaction_commit_claims(
        evidence.world_events,
        claimed_commits,
        allowed_authority_pairs={
            ("create_search_session", "world.create_search_session"),
            ("catalog_mutation", "world.apply_catalog_mutation"),
            ("apply_listing_claim", "world.apply_listing_claim"),
        },
    )
    detail["transaction_commit_claims"] = commit_claims.to_dict()
    detail["operation_environment_verified"] = operation_environment_verified
    detail["report_causality_verified"] = report_causality_verified
    if not commit_claims.verified:
        raise RuntimeBenchmarkIntegrityError(
            "T2 authority claims do not close over the World commit journal"
        )
    return detail


def _table_rows(snapshot: Mapping[str, Any], table: str) -> list[dict[str, Any]]:
    tables = snapshot.get("tables")
    rows = tables.get(table) if isinstance(tables, Mapping) else None
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    return []


def _world_seed_contract_matches(
    problem: _ProblemT2,
    evidence: RuntimeEvidenceBundleV2,
) -> bool:
    """Bind the scorer to the exact validated initial CommerceWorld state."""

    tables = evidence.initial_world.get("tables")
    if not isinstance(tables, Mapping) or tables.get("logical_time") != 2:
        return False

    observed_catalog = _rows_by_id(tables.get("catalog"), "sku_id")
    observed_inventory = tables.get("inventory")
    if not isinstance(observed_inventory, Mapping):
        return False
    expected_catalog = {str(row["sku_id"]): row for row in problem.catalog}
    if set(observed_catalog) != set(expected_catalog) or set(observed_inventory) != set(
        expected_catalog
    ):
        return False
    for sku_id, expected in expected_catalog.items():
        observed = observed_catalog[sku_id]
        expected_attributes = {
            **dict(expected.get("attributes") or {}),
            "returnable": True,
            "refund_policy": "30_day_return",
        }
        if (
            observed.get("sku_id") != sku_id
            or observed.get("product_id") != expected.get("product_id")
            or observed.get("merchant_id") != expected.get("merchant_id")
            or observed.get("category") != expected.get("category")
            or observed.get("name") != expected.get("name")
            or observed.get("attributes") != expected_attributes
            or (observed.get("list_price") or {}).get("amount") != expected.get("list_price")
            or (observed.get("list_price") or {}).get("currency") != "USD"
        ):
            return False
        inventory = observed_inventory.get(sku_id)
        if not isinstance(inventory, Mapping) or any(
            (
                inventory.get("sku_id") != sku_id,
                inventory.get("merchant_id") != expected.get("merchant_id"),
                inventory.get("qty_available") != expected.get("inventory", 0),
                inventory.get("qty_reserved") != expected.get("qty_reserved", 0),
                inventory.get("version") != expected.get("version", 1),
            )
        ):
            return False

    observed_records = _rows_by_id(tables.get("evidence_records"), "record_id")
    expected_records = {str(row["record_id"]): row for row in problem.evidence_records}
    if observed_records != expected_records:
        return False
    observed_claims = _rows_by_id(tables.get("listing_claims"), "claim_id")
    expected_claims = {str(row["claim_id"]): row for row in problem.listing_claims}
    if set(observed_claims) != set(expected_claims):
        return False
    identity_fields = (
        "claim_id",
        "listing_id",
        "merchant_id",
        "subject",
        "issuer_id",
    )
    return all(
        all(
            observed_claims[claim_id].get(field) == expected.get(field) for field in identity_fields
        )
        and _strip_schema_versions(observed_claims[claim_id].get("versions"))
        == _strip_schema_versions(expected.get("versions"))
        for claim_id, expected in expected_claims.items()
    )


def _strip_schema_versions(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_schema_versions(child)
            for key, child in value.items()
            if key != "schema_version"
        }
    if isinstance(value, (list, tuple)):
        return [_strip_schema_versions(child) for child in value]
    return value


def _claim_state_matches(intent: Mapping[str, Any], claim: Mapping[str, Any]) -> bool:
    operation = str(intent.get("operation"))
    expected_state = {
        "publish": "published",
        "correct": "corrected",
        "retract": "retracted",
    }.get(operation)
    versions = claim.get("versions")
    if expected_state is None or not isinstance(versions, list) or not versions:
        return False
    current = versions[-1]
    if (
        not isinstance(current, Mapping)
        or current.get("operation") != operation
        or current.get("state") != expected_state
    ):
        return False
    if operation == "correct":
        content = current.get("content")
        if (
            not isinstance(content, Mapping)
            or not content
            or canonical_sha256(content) != canonical_sha256(intent.get("content"))
        ):
            return False
    if operation == "retract" and current.get("reason") != intent.get("reason"):
        return False
    actual_source_ids = {
        str(row.get("source_id"))
        for row in current.get("evidence", ())
        if isinstance(row, Mapping) and row.get("source_id")
    }
    expected_source_ids = set(intent.get("evidence_record_ids", ()))
    return actual_source_ids == expected_source_ids


def _claim_current_digest(claim: Mapping[str, Any]) -> str | None:
    direct = claim.get("current_event_digest")
    if isinstance(direct, str):
        return direct
    versions = claim.get("versions")
    if not isinstance(versions, (list, tuple)) or not versions:
        return None
    current = versions[-1]
    digest = current.get("event_digest") if isinstance(current, Mapping) else None
    return str(digest) if isinstance(digest, str) else None


def _world_grounding_credit(
    problem: _ProblemT2,
    evidence: RuntimeEvidenceBundleV2,
    authority: (
        VerifiedSearchSessionEvidence
        | VerifiedCatalogMutationEvidence
        | VerifiedListingClaimEvidence
        | None
    ),
) -> tuple[float, dict[str, Any]]:
    actor_id = _BUYER_ID if problem.definition.evaluated_role == "buyer" else _MERCHANT_ID
    listing_rows = _grounded_listing_rows(evidence, actor_id)
    record_rows = _grounded_record_rows(evidence, actor_id)
    claim_rows = _grounded_claim_rows(evidence, actor_id)
    grounded_skus = frozenset(listing_rows)
    grounded_records = frozenset(record_rows)
    grounded_claims = frozenset(claim_rows)
    required_skus = set(problem.required_skus)
    required_records = {str(row["record_id"]): row for row in problem.evidence_records}
    required_claims = {str(row["claim_id"]): row for row in problem.listing_claims}
    initial_catalog = _catalog_index(evidence.initial_world)

    def listing_read_matches(sku_id: str) -> bool:
        initial = initial_catalog.get(sku_id)
        if not isinstance(initial, Mapping):
            return False
        attributes = initial.get("attributes")
        if not isinstance(attributes, Mapping):
            return False
        expected_visible: dict[str, Any] = {
            "sku_id": initial.get("sku_id"),
            "category": initial.get("category"),
            "name": initial.get("name"),
            "list_price": initial.get("list_price"),
            "merchant_id": initial.get("merchant_id"),
            "attributes": {
                key: value
                for key, value in attributes.items()
                if key.casefold().replace(" ", "_") != "key_features"
            },
        }
        key_features = attributes.get("key_features") or attributes.get("Key Features")
        if isinstance(key_features, str) and key_features:
            expected_visible["key_features_excerpt"] = key_features[:200]
        return listing_rows.get(sku_id) == expected_visible

    explicit_reads = (
        required_skus <= set(grounded_skus)
        and all(listing_read_matches(sku_id) for sku_id in required_skus)
        and set(required_records) <= set(grounded_records)
        and all(
            isinstance(record_rows.get(record_id), Mapping)
            and record_rows[record_id].get("record_id") == record_id
            and record_rows[record_id].get("record_digest") == expected.get("record_digest")
            and record_rows[record_id].get("version") == expected.get("version")
            for record_id, expected in required_records.items()
        )
        and set(required_claims) <= set(grounded_claims)
        and all(
            isinstance(claim_rows.get(claim_id), Mapping)
            and claim_rows[claim_id].get("claim_id") == claim_id
            and _claim_current_digest(claim_rows[claim_id]) == _claim_current_digest(expected)
            for claim_id, expected in required_claims.items()
        )
    )
    if problem.definition.evaluated_role == "buyer":
        try:
            ranked = _ranked_skus(evidence)
        except RuntimeEvidenceError:
            ranked = frozenset()
        search_bound, search_detail = _search_session_binding(
            problem,
            evidence,
            authority if isinstance(authority, VerifiedSearchSessionEvidence) else None,
        )
        state_effect = required_skus <= set(ranked) and search_bound
        state_fraction = 1.0 if state_effect else 0.0
        state_detail: dict[str, Any] = {
            "platform_ranked_skus": sorted(ranked),
            "search_authority": search_detail,
        }
    elif problem.lane == "truthful_listing_update":
        final_row = _catalog_index(evidence.final_world).get(problem.primary_sku, {})
        attributes = final_row.get("attributes")
        attributes = dict(attributes) if isinstance(attributes, Mapping) else {}
        expected_changes = problem.expected_submission["payload"]["changes"]
        changes_match = all(attributes.get(key) == value for key, value in expected_changes.items())
        catalog_bound, catalog_detail = _catalog_mutation_binding(
            problem,
            evidence,
            authority if isinstance(authority, VerifiedCatalogMutationEvidence) else None,
        )
        state_effect = changes_match and catalog_bound
        state_fraction = 1.0 if state_effect else 0.0
        state_detail = {
            "catalog_changes_match": changes_match,
            **catalog_detail,
        }
    else:
        final_claims = {
            str(row.get("claim_id")): row
            for row in _table_rows(evidence.final_world, "listing_claims")
            if row.get("claim_id")
        }
        commits = authority.claims if isinstance(authority, VerifiedListingClaimEvidence) else ()
        expected_intents = {str(intent["claim_id"]): intent for intent in problem.claim_intents}
        verified_intent_ids = {
            str(join.intent["claim_id"])
            for join in commits
            if isinstance(join.intent.get("claim_id"), str)
            and str(join.intent["claim_id"]) in expected_intents
            and _claim_intent_business_semantics_match(
                join.intent,
                expected_intents[str(join.intent["claim_id"])],
            )
        }
        claim_matches = {
            claim_id: (
                claim_id in verified_intent_ids
                and _claim_state_matches(intent, final_claims.get(claim_id, {}))
            )
            for claim_id, intent in expected_intents.items()
        }
        matched_claims = sum(claim_matches.values())
        state_fraction = matched_claims / max(1, len(expected_intents))
        state_effect = matched_claims == len(expected_intents)
        state_detail = {
            "claim_state_matches": claim_matches,
            "verified_task_claim_ids": sorted(verified_intent_ids),
            "claim_commit_count": len(commits),
            "expected_claim_commit_count": len(problem.claim_intents),
        }
    explicit_read_credit = 1.0 if explicit_reads else 0.0
    # This lane measures only evidence that the evaluated model actually read.
    # Frozen fixture validity and Runtime→Platform→World authority are hard
    # scorer preconditions.  Resulting Platform/World state is diagnostic and
    # validity evidence, never capability credit.
    grounding_credit = explicit_read_credit
    return grounding_credit, {
        "initial_world_contract_bound": True,
        "required_skus": sorted(required_skus),
        "explicit_listing_reads": sorted(grounded_skus),
        "required_evidence_record_ids": sorted(required_records),
        "explicit_evidence_reads": sorted(grounded_records),
        "required_claim_ids": sorted(required_claims),
        "explicit_claim_reads": sorted(grounded_claims),
        "authoritative_state_effect": state_effect,
        "explicit_read_credit": explicit_read_credit,
        "authoritative_state_credit": state_fraction,
        "authority_integrity_valid": authority is not None,
        "world_commit_count": len(evidence.world_events),
        **state_detail,
    }


_T2_SCORE_BEARING_MODEL_ROUTES: Mapping[
    str,
    tuple[frozenset[str], str],
] = {
    "commerce.search": (frozenset({"search"}), "platform:aggregator"),
    "commerce.update_listing": (
        frozenset({"update_listing"}),
        "platform:catalog",
    ),
    "commerce.apply_listing_claim": (
        frozenset(
            {
                "publish_listing_claim",
                "correct_listing_claim",
                "retract_listing_claim",
            }
        ),
        "platform:claims",
    ),
    "commerce.respond_inquiry": (
        frozenset({"respond_inquiry"}),
        _BUYER_ID,
    ),
    "commerce.submit_decision_record": (
        frozenset({"submit_decision_record"}),
        "runtime:evidence",
    ),
}
_T2_SCORE_BEARING_MODEL_INTENTS = frozenset(
    intent
    for intents, _destination in _T2_SCORE_BEARING_MODEL_ROUTES.values()
    for intent in intents
)

_T2_MODEL_WORLD_READS: Mapping[str, tuple[str, str, str]] = {
    "observe_listing": ("world.get_listing", "sku_ref", "sku_id"),
    "observe_evidence_record": (
        "world.get_evidence_record",
        "record_ref",
        "record_id",
    ),
    "observe_listing_claim": (
        "world.get_listing_claim",
        "claim_ref",
        "claim_id",
    ),
}


def _model_text_evidence_matches(value: Any, wire_text: Any) -> bool:
    """Compare scorer-safe model text evidence with emitted wire text."""

    if not isinstance(wire_text, str):
        return False
    if isinstance(value, str):
        return value == wire_text
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"text_chars", "text_sha256"}
        and value.get("text_chars") == len(wire_text)
        and value.get("text_sha256") == hashlib.sha256(wire_text.encode("utf-8")).hexdigest()
    )


def _model_safe_value_matches_wire(value: Any, wire_value: Any) -> bool:
    """Compare one recursively sanitized model value with emitted business data."""

    if isinstance(value, Mapping) and set(value) == {
        "text_chars",
        "text_sha256",
    }:
        return _model_text_evidence_matches(value, wire_value)
    if isinstance(value, Mapping):
        return bool(
            isinstance(wire_value, Mapping)
            and set(value) == set(wire_value)
            and all(
                _model_safe_value_matches_wire(item, wire_value[name])
                for name, item in value.items()
            )
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return bool(
            isinstance(wire_value, Sequence)
            and not isinstance(wire_value, (str, bytes, bytearray))
            and len(value) == len(wire_value)
            and all(
                _model_safe_value_matches_wire(item, observed)
                for item, observed in zip(value, wire_value, strict=True)
            )
        )
    return type(value) is type(wire_value) and value == wire_value


def _t2_public_reference_aliases(problem: _ProblemT2) -> dict[str, str]:
    identities = (
        *(str(row["sku_id"]) for row in problem.catalog),
        *(str(row["product_id"]) for row in problem.catalog),
        *(str(row["record_id"]) for row in problem.evidence_records),
        *(str(row["subject_id"]) for row in problem.evidence_records),
        *(str(row["claim_id"]) for row in problem.listing_claims),
        *(str(row["listing_id"]) for row in problem.listing_claims),
        *(str(row["subject"]) for row in problem.listing_claims),
    )
    return {identity: public_reference_alias_v1(identity) for identity in dict.fromkeys(identities)}


def _t2_claim_choice_matches_wire(
    problem: _ProblemT2,
    choice: VerifiedModelBusinessChoice,
    payload: Mapping[str, Any],
) -> bool:
    operation_by_intent = {
        "publish_listing_claim": "publish",
        "correct_listing_claim": "correct",
        "retract_listing_claim": "retract",
    }
    operation = operation_by_intent.get(choice.intent)
    arguments = choice.arguments
    if operation is None or set(arguments) != {
        "claim_ref",
        "evidence_record_refs",
    }:
        return False
    claim_id = payload.get("claim_id")
    evidence_ids = payload.get("evidence_record_ids")
    if (
        not isinstance(claim_id, str)
        or not isinstance(evidence_ids, list)
        or any(not isinstance(value, str) for value in evidence_ids)
        or payload.get("operation") != operation
        or arguments.get("claim_ref") != public_reference_alias_v1(claim_id)
        or arguments.get("evidence_record_refs")
        != [public_reference_alias_v1(value) for value in evidence_ids]
    ):
        return False
    claim = next(
        (row for row in problem.listing_claims if row.get("claim_id") == claim_id),
        None,
    )
    return bool(
        isinstance(claim, Mapping)
        and payload.get("listing_id") == claim.get("listing_id")
        and payload.get("subject") == claim.get("subject")
    )


def _t2_model_choice_matches_wire(
    problem: _ProblemT2,
    choice: VerifiedModelBusinessChoice,
    envelope: Mapping[str, Any],
) -> bool:
    """Validate a public T2 model choice against its Agent-compiled envelope."""

    action = envelope.get("action")
    if not isinstance(action, Mapping):
        return False
    action_kind = action.get("kind")
    payload = action.get("payload")
    if not isinstance(action_kind, str) or not isinstance(payload, Mapping):
        return False
    route = _T2_SCORE_BEARING_MODEL_ROUTES.get(action_kind)
    if route is None:
        return False
    expected_intents, expected_destination = route
    if (
        choice.emitted_msg_id != envelope.get("msg_id")
        or choice.intent not in expected_intents
        or choice.action_kind != action_kind
        or choice.destination != expected_destination
        or envelope.get("to") != expected_destination
    ):
        return False
    arguments = choice.arguments

    if action_kind == "commerce.search":
        if set(arguments) - {"query", "filters"} or "query" not in arguments:
            return False
        if not _model_text_evidence_matches(arguments["query"], payload.get("query")):
            return False
        if "filters" in arguments:
            return _model_safe_value_matches_wire(
                arguments["filters"],
                payload.get("filters"),
            )
        return "filters" not in payload

    if action_kind == "commerce.update_listing":
        fields = payload.get("fields")
        return bool(
            set(arguments) == {"changes"}
            and isinstance(arguments.get("changes"), Mapping)
            and payload.get("op") == "update"
            and payload.get("sku_id") == problem.primary_sku
            and isinstance(fields, Mapping)
            and set(fields) == {"attributes"}
            and _model_safe_value_matches_wire(
                arguments["changes"],
                fields.get("attributes"),
            )
        )

    if action_kind == "commerce.apply_listing_claim":
        return _t2_claim_choice_matches_wire(problem, choice, payload)

    aliases = _t2_public_reference_aliases(problem)
    if action_kind == "commerce.respond_inquiry":
        return bool(
            set(arguments) == {"payload"}
            and _model_safe_value_matches_wire(
                arguments["payload"],
                _publicize_business_value(payload, aliases),
            )
        )

    if set(arguments) != {"outcome", "summary", "payload"}:
        return False
    details = payload.get("details")
    submission = details.get("submission") if isinstance(details, Mapping) else None
    if (
        set(payload) != {"outcome", "summary", "details"}
        or not isinstance(submission, Mapping)
        or set(submission) != {"kind", "payload"}
        or submission.get("kind") != problem.expected_submission["kind"]
        or not isinstance(submission.get("payload"), Mapping)
    ):
        return False
    return bool(
        arguments["outcome"] == payload["outcome"]
        and _model_text_evidence_matches(arguments["summary"], payload["summary"])
        and _model_safe_value_matches_wire(
            arguments["payload"],
            _publicize_business_value(submission["payload"], aliases),
        )
    )


def _require_t2_model_compilation_integrity(
    problem: _ProblemT2,
    evidence: RuntimeEvidenceBundleV2,
) -> None:
    """Exact-join every score-bearing T2 action to its public model choice."""

    actor_id = _BUYER_ID if problem.definition.evaluated_role == "buyer" else _MERCHANT_ID
    try:
        choices = verified_model_business_choices(
            evidence,
            evaluated_actor_id=actor_id,
        )
    except TrackerEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError(
            "T2 model business-choice provenance is invalid"
        ) from exc
    choice_by_msg_id = {
        row.emitted_msg_id: row
        for row in choices
        if row.intent in _T2_SCORE_BEARING_MODEL_INTENTS
        or row.action_kind in _T2_SCORE_BEARING_MODEL_ROUTES
    }
    actions = tuple(
        row
        for row in evidence.envelopes
        if row.get("from") == actor_id
        and (row.get("action") or {}).get("kind") in _T2_SCORE_BEARING_MODEL_ROUTES
    )
    action_by_msg_id = {
        str(row.get("msg_id")): row for row in actions if isinstance(row.get("msg_id"), str)
    }
    if len(action_by_msg_id) != len(actions) or set(action_by_msg_id) != set(choice_by_msg_id):
        raise RuntimeBenchmarkIntegrityError(
            "T2 score-bearing wire actions do not exactly match model choices"
        )
    mismatched = sorted(
        msg_id
        for msg_id, action in action_by_msg_id.items()
        if not _t2_model_choice_matches_wire(
            problem,
            choice_by_msg_id[msg_id],
            action,
        )
    )
    if mismatched:
        raise RuntimeBenchmarkIntegrityError(
            "T2 Agent compiler drifted from public model arguments for message(s): "
            + ", ".join(mismatched)
        )

    report_msg_ids = {
        msg_id
        for msg_id, row in action_by_msg_id.items()
        if (row.get("action") or {}).get("kind") == "commerce.submit_decision_record"
    }
    actor_records = evidence.accepted_actor_results(
        action_kind="commerce.submit_decision_record",
        actor_id=actor_id,
        task_id=problem.definition.task_id,
    )
    record_by_request_id = {
        str(row.get("request_msg_id")): row
        for row in actor_records
        if isinstance(row.get("request_msg_id"), str)
    }
    if (
        len(record_by_request_id) != len(actor_records)
        or set(record_by_request_id) != report_msg_ids
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T2 model reports do not exactly match accepted actor results"
        )
    for msg_id in report_msg_ids:
        envelope_payload = action_by_msg_id[msg_id]["action"].get("payload")
        record = record_by_request_id[msg_id]
        report = record.get("report")
        if (
            record.get("action_kind") != "commerce.submit_decision_record"
            or not isinstance(report, Mapping)
            or report.get("content") != envelope_payload
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T2 accepted actor result differs from the model report envelope"
            )


def _t2_model_world_read_matches_compilation(
    read: VerifiedModelWorldRead,
) -> bool:
    binding = _T2_MODEL_WORLD_READS.get(read.intent)
    if binding is None:
        return False
    tool, public_field, internal_field = binding
    internal_value = read.args.get(internal_field)
    return bool(
        read.tool == tool
        and set(read.arguments) == {public_field}
        and set(read.args) == {internal_field}
        and isinstance(internal_value, str)
        and read.arguments.get(public_field) == public_reference_alias_v1(internal_value)
    )


def _require_t2_model_world_read_integrity(
    problem: _ProblemT2,
    evidence: RuntimeEvidenceBundleV2,
) -> None:
    """Prove each public T2 read reference reached that exact World object."""

    actor_id = _BUYER_ID if problem.definition.evaluated_role == "buyer" else _MERCHANT_ID
    try:
        reads = verified_model_world_reads(
            evidence,
            evaluated_actor_id=actor_id,
        )
    except TrackerEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError("T2 model World-read provenance is invalid") from exc
    relevant_tools = frozenset(row[0] for row in _T2_MODEL_WORLD_READS.values())
    mismatched = tuple(
        read
        for read in reads
        if (read.intent in _T2_MODEL_WORLD_READS or read.tool in relevant_tools)
        and not _t2_model_world_read_matches_compilation(read)
    )
    if mismatched:
        raise RuntimeBenchmarkIntegrityError(
            "T2 Agent compiler mapped a public read reference to the wrong World object"
        )


def _model_business_completion_credit(
    problem: _ProblemT2,
    evidence: RuntimeEvidenceBundleV2,
    *,
    actual_submission: Mapping[str, Any] | None,
) -> tuple[float, dict[str, Any]]:
    """Score only the evaluated model's required business operations/report."""

    actor_id = _BUYER_ID if problem.definition.evaluated_role == "buyer" else _MERCHANT_ID
    if problem.definition.evaluated_role == "buyer":
        action_kind = "commerce.search"
        expected_count = 1
        decisions = evidence.actions(kind=action_kind, actor_id=actor_id)
        operation_semantics_verified = len(decisions) == expected_count
    elif problem.lane == "truthful_listing_update":
        action_kind = "commerce.update_listing"
        expected_count = 1
        decisions = evidence.actions(kind=action_kind, actor_id=actor_id)
        expected_changes = problem.expected_submission["payload"]["changes"]
        operation_semantics_verified = bool(
            len(decisions) == expected_count
            and (decisions[0]["action"].get("payload") or {}).get("fields")
            == {"attributes": expected_changes}
        )
    else:
        action_kind = "commerce.apply_listing_claim"
        expected_count = len(problem.claim_intents)
        decisions = evidence.actions(kind=action_kind, actor_id=actor_id)
        observed_operations = tuple(
            str((row["action"].get("payload") or {}).get("operation", "")) for row in decisions
        )
        expected_operations = tuple(str(row["operation"]) for row in problem.claim_intents)
        operation_semantics_verified = observed_operations == expected_operations
        if problem.lane == "evidence_backed_response":
            response_count = len(
                evidence.actions(kind="commerce.respond_inquiry", actor_id=actor_id)
            )
            operation_semantics_verified = operation_semantics_verified and response_count == 1
        else:
            response_count = 0
    report_semantics_verified = actual_submission is not None
    exact = operation_semantics_verified and report_semantics_verified
    return (1.0 if exact else 0.0), {
        "action_kind": action_kind,
        "expected_model_decisions": expected_count,
        "observed_model_decisions": len(decisions),
        "model_operation_semantics_verified": operation_semantics_verified,
        "model_response_decisions": response_count if "response_count" in locals() else 0,
        "model_completion_report_semantics_verified": report_semantics_verified,
    }


def score_t2_runtime(task_id: str, evidence: RuntimeEvidenceBundleV2) -> RuntimeTaskScoreV3:
    """Score a T2 run solely from persisted Runtime, Platform, and World evidence."""

    require_runtime_benchmark_integrity_v2(evidence)
    require_frozen_scenario_fixture_v2(
        evidence,
        scenario_for_t2(task_id),
        family="T2",
    )
    problem = _problem_for(task_id)
    if not _world_seed_contract_matches(problem, evidence):
        raise RuntimeBenchmarkIntegrityError(
            "T2 initial World snapshot is not the frozen task fixture"
        )
    _require_t2_model_world_read_integrity(problem, evidence)
    _require_t2_model_compilation_integrity(problem, evidence)
    authority: (
        VerifiedSearchSessionEvidence
        | VerifiedCatalogMutationEvidence
        | VerifiedListingClaimEvidence
        | None
    ) = None
    try:
        if problem.definition.evaluated_role == "buyer":
            discovery = verify_optional_discovery_prefix_v2(
                evidence,
                buyer_id=_BUYER_ID,
            )
            authority = discovery.search_evidence
        elif problem.lane == "truthful_listing_update":
            candidate = evidence.verified_operation_evidence(CATALOG_MUTATION_EVIDENCE_CONTRACT)
            authority = (
                candidate if isinstance(candidate, VerifiedCatalogMutationEvidence) else None
            )
        else:
            candidate = evidence.verified_operation_evidence(LISTING_CLAIM_EVIDENCE_CONTRACT)
            authority = candidate if isinstance(candidate, VerifiedListingClaimEvidence) else None
    except RuntimeEvidenceError:
        authority = None
    try:
        if problem.definition.evaluated_role == "buyer":
            accepted_relevant = evidence.accepted_platform_exchanges(
                kind="commerce.search",
                actor_id=_BUYER_ID,
                endpoint="platform:aggregator",
            )
        elif problem.lane == "truthful_listing_update":
            accepted_relevant = evidence.accepted_platform_exchanges(
                kind="commerce.update_listing",
                actor_id=_MERCHANT_ID,
                endpoint="platform:catalog",
            )
        else:
            accepted_relevant = evidence.accepted_platform_exchanges(
                kind="commerce.apply_listing_claim",
                actor_id=_MERCHANT_ID,
                endpoint="platform:claims",
            )
    except RuntimeEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError(
            "T2 accepted model operation has an invalid Platform response graph"
        ) from exc
    if accepted_relevant and authority is None:
        raise RuntimeBenchmarkIntegrityError(
            "T2 accepted Platform operation is not bound to authoritative World evidence"
        )
    actual, actor_record = _submission_from_evidence(problem, evidence)
    actor_id = _BUYER_ID if problem.definition.evaluated_role == "buyer" else _MERCHANT_ID
    presented = _grounded_record_ids(evidence, actor_id)
    scope, truth, citations, semantic_evidence = _submission_metrics(problem, actual, presented)
    completion_binding = _completion_environment_binding(
        problem,
        evidence,
        actor_record,
        authority,
    )
    grounding, grounding_evidence = _world_grounding_credit(
        problem,
        evidence,
        authority,
    )
    business_completion, business_completion_evidence = _model_business_completion_credit(
        problem,
        evidence,
        actual_submission=actual,
    )
    expected_accepted_count = int(business_completion_evidence["expected_model_decisions"])
    model_operation_semantics_verified = (
        business_completion_evidence.get("model_operation_semantics_verified") is True
    )
    if accepted_relevant and (completion_binding.get("operation_environment_verified") is not True):
        raise RuntimeBenchmarkIntegrityError(
            "T2 accepted model operation did not close over Platform/World authority"
        )
    if model_operation_semantics_verified and len(accepted_relevant) != expected_accepted_count:
        raise RuntimeBenchmarkIntegrityError(
            "T2 legal model operation was not faithfully accepted by Platform"
        )
    report_actions = evidence.actions(
        kind="commerce.submit_decision_record",
        actor_id=actor_id,
    )
    if len(report_actions) == 1 and actor_record is None:
        raise RuntimeBenchmarkIntegrityError(
            "T2 legal model report is missing from Runtime actor evidence"
        )
    if actor_record is not None and (
        completion_binding.get("report_causality_verified") is not True
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T2 model report is not causally bound to the completed business prefix"
        )
    if completion_binding.get("participant_response_present") is True and (
        completion_binding.get("participant_response_environment_verified") is not True
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T2 scripted buyer acknowledgement did not close the model response"
        )
    business_completion_evidence = {
        **business_completion_evidence,
        "environment_closure_is_validity_only": True,
    }
    allowed_authority_pairs = {
        ("create_search_session", "world.create_search_session"),
        ("catalog_mutation", "world.apply_catalog_mutation"),
        ("apply_listing_claim", "world.apply_listing_claim"),
    }
    observed_authority_pairs = {
        (str(row.get("operation")), str(row.get("authority_action")))
        for row in evidence.world_events
    }
    if not observed_authority_pairs <= allowed_authority_pairs:
        raise RuntimeBenchmarkIntegrityError(
            "T2 World journal contains an unrecognized authority operation"
        )
    expected_operation = (
        "create_search_session"
        if problem.definition.evaluated_role == "buyer"
        else "catalog_mutation"
        if problem.lane == "truthful_listing_update"
        else "apply_listing_claim"
    )
    if any(row.get("operation") == expected_operation for row in evidence.world_events) and (
        authority is None
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T2 task operation is not bound to its authoritative World contract"
        )
    checks = renormalize_capability_checks_v2(
        (
            RuntimeRubricCheckV2(
                "requested_evidence_coverage",
                0.20,
                grounding,
                grounding_evidence,
            ),
            RuntimeRubricCheckV2("bounded_response_scope", 0.15, scope, semantic_evidence),
            RuntimeRubricCheckV2("truthful_semantics", 0.25, truth, semantic_evidence),
            RuntimeRubricCheckV2("evidence_citations", 0.10, citations, semantic_evidence),
            RuntimeRubricCheckV2(
                "business_action_completion",
                0.15,
                business_completion,
                business_completion_evidence,
            ),
        )
    )
    issues = () if all(check.credit == 1.0 for check in checks) else ("t2_task_incomplete",)
    return score_checks(problem.definition, checks, issues=issues)


def runtime_bundle_t2(task_id: str) -> RuntimeTaskBundleV2:
    problem = _problem_for(task_id)
    scenario = scenario_for_t2(task_id)
    evaluated_actor_id = _BUYER_ID if problem.definition.evaluated_role == "buyer" else _MERCHANT_ID
    counterpart_id = _MERCHANT_ID if evaluated_actor_id == _BUYER_ID else _BUYER_ID
    counterpart_factory = (
        _BuyerResponseAckChannel
        if problem.lane == "evidence_backed_response"
        else _no_reply_channel
    )
    semantic_hash = canonical_sha256(
        {
            "problem_sha256": problem.content_sha256,
            "scenario_state": scenario.initial_state,
            "scenario_oracle": scenario.success_oracle,
            "evaluated_actor_id": evaluated_actor_id,
        }
    )
    return RuntimeTaskBundleV2(
        task=problem.definition,
        scenario=scenario,
        evaluated_actor_id=evaluated_actor_id,
        ideal_channel=lambda: _ideal_channel(problem),
        counterpart_channels={counterpart_id: counterpart_factory},
        scorer=lambda evidence: score_t2_runtime(task_id, evidence),
        semantic_hash=semantic_hash,
        mutations=(
            RuntimeMutationV2(
                mutation_id=f"{task_id}:partial:01",
                channel=lambda: targeted_mutation_channel_t2(task_id),
                expected_changed_checks=(
                    ("truthful_semantics", "business_action_completion")
                    if problem.lane in {"comparative_claim_limits", "correction_and_retraction"}
                    else ("truthful_semantics",)
                ),
            ),
        ),
    )


def runtime_bundles_t2() -> tuple[RuntimeTaskBundleV2, ...]:
    return tuple(
        runtime_bundle_t2(task_id)
        for task_id, definition in sorted(TASK_REGISTRY_V2.items())
        if definition.family.value == "T2"
    )


__all__ = [
    "T2_RUNTIME_SCHEMA_V2",
    "runtime_bundle_t2",
    "runtime_bundles_t2",
    "scenario_for_t2",
    "score_t2_runtime",
    "targeted_mutation_channel_t2",
]
