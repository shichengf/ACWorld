"""Normalize T3 decision facts from provider-visible World-read observations."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any


def _rows(value: Any, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an array")
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{label} contains a non-object row")
    return tuple(value)


def _values(value: Any, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an array")
    return tuple(value)


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _ref(value: Mapping[str, Any], stem: str, *, label: str) -> str:
    return _text(
        value.get(f"{stem}_ref", value.get(f"{stem}_id")),
        label=label,
    )


def _observed_rows(
    request: Mapping[str, Any],
    observation_kind: str,
) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    for observation in _rows(request.get("observations", ()), label="observations"):
        raw = observation.get("observed_business_facts")
        if raw is None:
            continue
        for row in _rows(raw, label="observed_business_facts"):
            if row.get("observation_kind") == observation_kind:
                output.append(row)
    return tuple(output)


def grounded_t3_facts_v1(
    request: Mapping[str, Any],
    persistent_facts: Mapping[str, Any],
) -> dict[str, Any]:
    """Return T3 choice facts grounded in the current request's World reads.

    Evidence/reporting phases may include frozen value rows. Selection phases
    omit those duplicates and are reconstructed here from model-visible World
    observations plus value-free reference lists.
    """

    output = copy.deepcopy(dict(persistent_facts))
    if output.get("candidate_facts") is not None:
        return output

    candidate_refs = tuple(
        _text(value, label="candidate listing ref")
        for value in _values(
            output.get("candidate_listing_refs", ()),
            label="candidate_listing_refs",
        )
    )
    listing_by_ref: dict[str, Mapping[str, Any]] = {}
    for observation in _observed_rows(request, "listing"):
        criteria = observation.get("criteria")
        facts = observation.get("facts")
        if not isinstance(criteria, Mapping) or not isinstance(facts, Mapping):
            raise ValueError("T3 listing observation is incomplete")
        listing_ref = _ref(criteria, "sku", label="listing observation sku ref")
        if listing_ref in listing_by_ref:
            raise ValueError("T3 listing was observed more than once")
        listing_by_ref[listing_ref] = facts
    if set(listing_by_ref) != set(candidate_refs):
        raise ValueError("T3 selection lacks the complete grounded candidate set")

    candidates: list[dict[str, Any]] = []
    for index, listing_ref in enumerate(candidate_refs):
        facts = listing_by_ref[listing_ref]
        attributes = facts.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError("T3 grounded listing has no attributes")
        hard_attributes = attributes.get("hard_attributes", attributes)
        preference_values = attributes.get("preference_values")
        if not isinstance(hard_attributes, Mapping) or not isinstance(
            preference_values,
            Mapping,
        ):
            raise ValueError("T3 grounded listing lacks decision attributes")
        candidates.append(
            {
                "listing_ref": listing_ref,
                "hard_attributes": copy.deepcopy(dict(hard_attributes)),
                "preference_values": copy.deepcopy(dict(preference_values)),
                "sort_key": f"{index:04d}",
            }
        )

    record_refs = tuple(
        _text(value, label="available record ref")
        for value in _values(
            output.get("available_record_refs", ()),
            label="available_record_refs",
        )
    )
    records: list[dict[str, Any]] = []
    record_by_subject: dict[str, str] = {}
    observed_record_refs: set[str] = set()
    for observation in _observed_rows(request, "evidence_record"):
        criteria = observation.get("criteria")
        facts = observation.get("facts")
        if not isinstance(criteria, Mapping) or not isinstance(facts, Mapping):
            raise ValueError("T3 evidence-record observation is incomplete")
        record_ref = _ref(criteria, "record", label="evidence record ref")
        attributes = facts.get("facts", facts.get("attributes"))
        if not isinstance(attributes, Mapping):
            raise ValueError("T3 evidence record has no public attributes")
        if record_ref in observed_record_refs:
            raise ValueError("T3 evidence record was observed more than once")
        observed_record_refs.add(record_ref)
        row: dict[str, Any] = {
            "record_ref": record_ref,
            "attributes": copy.deepcopy(dict(attributes)),
        }
        subject = facts.get("subject_ref", facts.get("subject_id"))
        if isinstance(subject, str) and subject:
            row["subject_ref"] = subject
            record_by_subject[subject] = record_ref
        records.append(row)
    if observed_record_refs != set(record_refs):
        raise ValueError("T3 selection lacks the complete grounded evidence-record set")

    revisions: list[Mapping[str, Any]] = []
    for observation in _observed_rows(request, "mandate_revisions"):
        revisions.extend(
            _rows(observation.get("facts"), label="mandate revisions")
        )
    revisions.sort(
        key=lambda row: (
            int(row.get("logical_tick", 0)),
            str(row.get("principal_ref", row.get("principal_id", ""))),
        )
    )

    updates: list[dict[str, Any]] = []
    instructions: list[dict[str, Any]] = []
    for index, revision in enumerate(revisions, start=1):
        changes = revision.get("changes")
        if not isinstance(changes, Mapping):
            continue
        preferences = changes.get("preferences")
        if isinstance(preferences, Mapping):
            update_ref = _ref(
                preferences,
                "source_update",
                label="mandate preference update ref",
            )
            weights = preferences.get("weights")
            if not isinstance(weights, Mapping):
                raise ValueError("T3 mandate preference update has no weights")
            updates.append(
                {
                    "update_ref": update_ref,
                    "preference_weights": copy.deepcopy(dict(weights)),
                    "update_sequence": int(revision.get("logical_tick", index)),
                    "sort_key": f"{index:04d}",
                    "authorized": True,
                }
            )
        hard_constraints = changes.get("hard_constraints")
        if isinstance(hard_constraints, Mapping):
            instruction_ref = _ref(
                hard_constraints,
                "source_instruction",
                label="mandate authority instruction ref",
            )
            constraint = hard_constraints.get("constraint")
            if not isinstance(constraint, Mapping):
                raise ValueError("T3 mandate authority instruction has no constraint")
            instructions.append(
                {
                    "instruction_ref": instruction_ref,
                    "constraint": copy.deepcopy(dict(constraint)),
                    "sort_key": f"{index:04d}",
                    "authorized": True,
                }
            )

    social_signals: list[dict[str, Any]] = []
    observed_signal_refs: set[str] = set()
    for observation in _observed_rows(request, "review_evidence"):
        facts = observation.get("facts")
        if not isinstance(facts, Mapping):
            raise ValueError("T3 review observation is incomplete")
        for review in _rows(facts.get("reviews", ()), label="review evidence"):
            signal_ref = _ref(review, "review", label="review signal ref")
            listing_ref = _ref(review, "sku", label="review listing ref")
            reviewer_ref = _ref(review, "reviewer", label="reviewer ref")
            trust_record_ref = record_by_subject.get(reviewer_ref)
            if trust_record_ref is None:
                raise ValueError("T3 review has no grounded source-trust record")
            if signal_ref in observed_signal_refs:
                raise ValueError("T3 social signal was observed more than once")
            observed_signal_refs.add(signal_ref)
            social_signals.append(
                {
                    "signal_ref": signal_ref,
                    "listing_ref": listing_ref,
                    "advisor_ref": reviewer_ref,
                    "trust_record_ref": trust_record_ref,
                    "rating": review.get("rating"),
                }
            )
    expected_signal_refs = {
        _text(value, label="social signal ref")
        for value in _values(
            output.get("social_signal_refs", ()),
            label="social_signal_refs",
        )
    }
    if observed_signal_refs != expected_signal_refs:
        raise ValueError("T3 selection lacks the complete grounded social-signal set")

    output["candidate_facts"] = candidates
    output["source_records"] = records
    output["preference_updates"] = updates
    output["authority_instructions"] = instructions
    output["social_signals"] = social_signals
    return output


__all__ = ["grounded_t3_facts_v1"]
