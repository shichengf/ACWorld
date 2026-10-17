"""Pure pre-commit contract for benchmark actor terminal actions.

An Agent turn may emit at most one terminal envelope.  Runtime must reject a
malformed model-authored action *before* Tracker records that envelope as a
successful emission.  This module is the shared, side-effect-free boundary for
the actor actions used by the fixed 200-task benchmark.

The contract deliberately does not consult Platform or World state.  It checks
only the canonical destination and the model-authored payload shape.  Market
rules, ownership, inventory, lifecycle state, and idempotency remain the
responsibility of Platform and World after the envelope has been audited.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Literal, Mapping

from protocol.actor_result import (
    ActorReportContent,
    ActorResultReport,
    ActorResultSchemaError,
    coerce_actor_report_submission,
    coerce_actor_result_report,
)
from protocol.envelope import Envelope
from protocol.errors import SchemaError


DELEGATE_REPORT_RESULT = "delegate.report_result"
COMMERCE_SUBMIT_DECISION_RECORD = "commerce.submit_decision_record"
_ACTOR_RESULT_ACTIONS = frozenset(
    {DELEGATE_REPORT_RESULT, COMMERCE_SUBMIT_DECISION_RECORD}
)
_KNOWN_PLATFORM_ENDPOINTS = frozenset(
    {
        "platform:ads",
        "platform:adjudicator",
        "platform:after-sales",
        "platform:aggregator",
        "platform:catalog",
        "platform:checkout",
        "platform:claims",
        "platform:events",
        "platform:evidence",
        "platform:fulfillment",
        "platform:governance",
        "platform:mandate",
        "platform:negotiation",
        "platform:psp",
        "platform:remediation",
        "platform:reputation",
        "platform:supply",
    }
)


class ActorTerminalContractError(SchemaError):
    """A model-authored terminal action violates the benchmark wire contract."""


@dataclass(frozen=True, slots=True)
class ActorTerminalActionPreview:
    """Validated, non-authoritative preview of one actor terminal action.

    ``actor_result`` is populated only for ``runtime:evidence`` submissions.
    It proves that the compact or sealed report can be coerced without writing
    the Runtime evidence journal.  Runtime still supplies and verifies the
    authoritative actor/principal/task binding when it accepts the envelope.
    """

    action_kind: str
    destination: str
    payload_form: Literal["mapping", "actor_report_submission", "actor_result_report"]
    actor_result: ActorReportContent | ActorResultReport | None = None


@dataclass(frozen=True, slots=True)
class _PayloadSpec:
    required: Mapping[str, Callable[[Any, str], None]]
    optional: Mapping[str, Callable[[Any, str], None]]

    def validate(self, value: Any, *, action_kind: str) -> None:
        payload = _mapping(value, action_kind)
        keys = frozenset(payload)
        required = frozenset(self.required)
        allowed = required | frozenset(self.optional)
        missing = required - keys
        unknown = keys - allowed
        if missing:
            raise ActorTerminalContractError(
                f"{action_kind} payload is missing required fields: "
                + ", ".join(sorted(missing))
            )
        if unknown:
            raise ActorTerminalContractError(
                f"{action_kind} payload has unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        for name, validator in self.required.items():
            validator(payload[name], f"{action_kind}.{name}")
        for name, validator in self.optional.items():
            if name in payload:
                validator(payload[name], f"{action_kind}.{name}")


def _spec(
    required: Mapping[str, Callable[[Any, str], None]],
    optional: Mapping[str, Callable[[Any, str], None]] | None = None,
) -> _PayloadSpec:
    return _PayloadSpec(required=dict(required), optional=dict(optional or {}))


def _text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ActorTerminalContractError(f"{field} must be non-empty text")


def _string(value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise ActorTerminalContractError(f"{field} must be a string")


def _integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActorTerminalContractError(f"{field} must be an integer")


def _nonnegative_integer(value: Any, field: str) -> None:
    _integer(value, field)
    if value < 0:
        raise ActorTerminalContractError(f"{field} must be non-negative")


def _positive_integer(value: Any, field: str) -> None:
    _integer(value, field)
    if value <= 0:
        raise ActorTerminalContractError(f"{field} must be positive")


def _rating(value: Any, field: str) -> None:
    _integer(value, field)
    if value < 1 or value > 5:
        raise ActorTerminalContractError(f"{field} must be between 1 and 5")


def _boolean(value: Any, field: str) -> None:
    if not isinstance(value, bool):
        raise ActorTerminalContractError(f"{field} must be a boolean")


def _object(value: Any, field: str) -> None:
    if not isinstance(value, Mapping):
        raise ActorTerminalContractError(f"{field} must be an object")


def _text_list(value: Any, field: str) -> None:
    if not isinstance(value, (list, tuple)):
        raise ActorTerminalContractError(f"{field} must be a list")
    for item in value:
        _text(item, f"{field}[]")


def _object_list(value: Any, field: str) -> None:
    if not isinstance(value, (list, tuple)):
        raise ActorTerminalContractError(f"{field} must be a list")
    for item in value:
        _object(item, f"{field}[]")


def _nonempty_unique_text_list(value: Any, field: str) -> None:
    _text_list(value, field)
    if not value:
        raise ActorTerminalContractError(f"{field} must not be empty")
    if len(value) != len(set(value)):
        raise ActorTerminalContractError(f"{field} must not contain duplicates")


def _bounded_supply_sku_ids(value: Any, field: str) -> None:
    _nonempty_unique_text_list(value, field)
    if len(value) > 64:
        raise ActorTerminalContractError(f"{field} must contain at most 64 values")


#: What an actor may state about its own observation. Every other evidence
#: field is Platform authority and is rejected here, not merely ignored.
_EVIDENCE_OBSERVATION_REQUIRED = frozenset({"kind", "subject_id", "facts"})
_EVIDENCE_OBSERVATION_OPTIONAL = frozenset({"trust"})


def _evidence_observation(value: Any, field: str) -> None:
    if not isinstance(value, dict):
        raise ActorTerminalContractError(f"{field} must be an object")
    keys = frozenset(value)
    missing = _EVIDENCE_OBSERVATION_REQUIRED - keys
    if missing:
        raise ActorTerminalContractError(
            f"{field} is missing {', '.join(sorted(missing))}"
        )
    unknown = keys - _EVIDENCE_OBSERVATION_REQUIRED - _EVIDENCE_OBSERVATION_OPTIONAL
    if unknown:
        raise ActorTerminalContractError(
            f"{field} carries Platform-owned fields: {', '.join(sorted(unknown))}"
        )
    _text(value["kind"], f"{field}.kind")
    _text(value["subject_id"], f"{field}.subject_id")
    for name in ("facts", "trust"):
        if name in value and not isinstance(value[name], dict):
            raise ActorTerminalContractError(f"{field}.{name} must be an object")


_CATALOG_PUBLISH_FIELDS = frozenset(
    {
        "list_price",
        "attributes",
        "permitted_claims",
        "must_not_claim",
        "inventory",
        "status",
        "category",
        "name",
        "currency",
        "product_id",
    }
)
_CATALOG_UPDATE_FIELDS = frozenset(
    {
        "attributes",
        "permitted_claims",
        "must_not_claim",
        "status",
        "category",
        "name",
        "product_id",
    }
)


def _catalog_fields(value: Any, field: str) -> None:
    """Validate the public value types shared by listing mutation operations."""

    row = _mapping(value, field)
    unknown = frozenset(row) - _CATALOG_PUBLISH_FIELDS
    if unknown:
        raise ActorTerminalContractError(
            f"{field} has unsupported catalog fields: "
            + ", ".join(sorted(unknown))
        )
    if "list_price" in row:
        _positive_integer(row["list_price"], f"{field}.list_price")
    if "inventory" in row:
        _nonnegative_integer(row["inventory"], f"{field}.inventory")
        if row["inventory"] != 0:
            raise ActorTerminalContractError(
                f"{field}.inventory must be zero; stock arrives through supply"
            )
    if "attributes" in row:
        attributes = _mapping(row["attributes"], f"{field}.attributes")
        if any(not key.strip() for key in attributes):
            raise ActorTerminalContractError(
                f"{field}.attributes keys must be non-empty text"
            )
    for name in ("permitted_claims", "must_not_claim"):
        if name in row:
            _text_list(row[name], f"{field}.{name}")
    for name in ("status", "category", "name", "currency", "product_id"):
        if name in row:
            _string(row[name], f"{field}.{name}")


def _validate_catalog_listing_update(payload: Any, *, action_kind: str) -> None:
    """Enforce the operation-dependent catalog shape before Platform dispatch."""

    spec = _PLATFORM_SPECS[(action_kind, "platform:catalog")]
    spec.validate(payload, action_kind=action_kind)
    row = _mapping(payload, action_kind)
    op = str(row["op"])
    fields = _mapping(row["fields"], f"{action_kind}.fields")
    allowed = (
        _CATALOG_PUBLISH_FIELDS
        if op == "publish"
        else _CATALOG_UPDATE_FIELDS
        if op == "update"
        else frozenset()
    )
    unsupported = frozenset(fields) - allowed
    if unsupported:
        raise ActorTerminalContractError(
            f"{action_kind}.{op} has unsupported fields: "
            + ", ".join(sorted(unsupported))
        )
    if op == "publish" and "list_price" not in fields:
        raise ActorTerminalContractError(
            f"{action_kind}.publish requires fields.list_price"
        )


def _cart_lines(value: Any, field: str) -> None:
    if not isinstance(value, (list, tuple)) or not value:
        raise ActorTerminalContractError(f"{field} must be a non-empty list")
    for line in value:
        row = _mapping(line, field)
        if frozenset(row) != frozenset({"sku_id", "qty"}):
            raise ActorTerminalContractError(
                f"{field} entries require exactly sku_id and qty"
            )
        _text(row["sku_id"], f"{field}[].sku_id")
        _positive_integer(row["qty"], f"{field}[].qty")


def _enum(*values: str) -> Callable[[Any, str], None]:
    allowed = frozenset(values)

    def validate(value: Any, field: str) -> None:
        if not isinstance(value, str) or value not in allowed:
            raise ActorTerminalContractError(f"{field} has an unsupported value")

    return validate


def _money(value: Any, field: str) -> None:
    row = _mapping(value, field)
    if frozenset(row) != frozenset({"amount", "currency"}):
        raise ActorTerminalContractError(
            f"{field} requires exactly amount and currency"
        )
    if not isinstance(row["amount"], (str, int)) or isinstance(row["amount"], bool):
        raise ActorTerminalContractError(f"{field}.amount must be text or an integer")
    try:
        amount = Decimal(str(row["amount"]))
    except (InvalidOperation, ValueError) as exc:
        raise ActorTerminalContractError(
            f"{field}.amount must be a parseable decimal"
        ) from exc
    if not amount.is_finite():
        raise ActorTerminalContractError(f"{field}.amount must be finite")
    _text(row["currency"], f"{field}.currency")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ActorTerminalContractError(f"{label} must be an object with string keys")
    return value


_TYPED_SEARCH_NUMERIC_FILTERS = frozenset(
    {
        "shipping_days_max",
        "warranty_months_min",
        "return_days_min",
        "energy_score_min",
    }
)
_TYPED_SEARCH_FILTERS = _TYPED_SEARCH_NUMERIC_FILTERS | frozenset(
    {"required_features"}
)


def _typed_search_filters(value: Any, field: str) -> None:
    row = _mapping(value, field)
    unknown = frozenset(row) - _TYPED_SEARCH_FILTERS
    if unknown:
        raise ActorTerminalContractError(
            f"{field} has unsupported filters: " + ", ".join(sorted(unknown))
        )
    for name in _TYPED_SEARCH_NUMERIC_FILTERS:
        if name in row:
            _nonnegative_integer(row[name], f"{field}.{name}")
    if "required_features" in row:
        features = row["required_features"]
        _text_list(features, f"{field}.required_features")
        if len(features) != len(set(features)):
            raise ActorTerminalContractError(
                f"{field}.required_features must not contain duplicates"
            )


def _validate_search(payload: Any, *, action_kind: str) -> None:
    spec = _PLATFORM_SPECS[(action_kind, "platform:aggregator")]
    spec.validate(payload, action_kind=action_kind)
    row = _mapping(payload, action_kind)
    # An explicit empty map is the retained no-filter wire form used by
    # existing clients.  Non-empty typed constraints require the versioned
    # contract so Platform cannot silently ignore them.
    if row.get("filters") and "filter_contract" not in row:
        raise ActorTerminalContractError(
            f"{action_kind}.filters requires filter_contract"
        )


def _validate_listing_claim(payload: Any, *, action_kind: str) -> None:
    spec = _PLATFORM_SPECS[(action_kind, "platform:claims")]
    spec.validate(payload, action_kind=action_kind)
    row = _mapping(payload, action_kind)
    base = frozenset({"claim_id", "listing_id", "operation", "subject"})
    operation = str(row["operation"])
    required = (
        base | frozenset({"evidence_record_ids"})
        if operation == "publish"
        else base | frozenset({"content", "evidence_record_ids"})
        if operation == "correct"
        else base | frozenset({"reason"})
    )
    allowed = (
        required | frozenset({"evidence_record_ids"})
        if operation == "retract"
        else required
    )
    keys = frozenset(row)
    if not required.issubset(keys) or not keys.issubset(allowed):
        raise ActorTerminalContractError(
            f"{action_kind}.{operation} has an invalid field combination"
        )
    if "content" in row and not row["content"]:
        raise ActorTerminalContractError(
            f"{action_kind}.{operation}.content must not be empty"
        )
    evidence_ids = row.get("evidence_record_ids", [])
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ActorTerminalContractError(
            f"{action_kind}.{operation}.evidence_record_ids must not contain duplicates"
        )
    if operation in {"publish", "correct"} and not evidence_ids:
        raise ActorTerminalContractError(
            f"{action_kind}.{operation} requires evidence_record_ids"
        )


def _validate_supply_update(payload: Any, *, action_kind: str) -> None:
    spec = _PLATFORM_SPECS[(action_kind, "platform:supply")]
    spec.validate(payload, action_kind=action_kind)
    row = _mapping(payload, action_kind)
    if (
        row.get("qty_delta", 0) == 0
        and row.get("eta_day") is None
        and row.get("unit_price_cents") is None
    ):
        raise ActorTerminalContractError(
            f"{action_kind} must change quantity, ETA, or price"
        )


def _validate_allocation(payload: Any, *, action_kind: str) -> None:
    spec = _PLATFORM_SPECS[(action_kind, "platform:fulfillment")]
    spec.validate(payload, action_kind=action_kind)
    row = _mapping(payload, action_kind)
    _nonempty_unique_text_list(
        row["priority_order_ids"],
        f"{action_kind}.priority_order_ids",
    )


def _validate_supply_read(payload: Any, *, action_kind: str) -> None:
    spec = _PLATFORM_SPECS[(action_kind, "platform:supply")]
    spec.validate(payload, action_kind=action_kind)
    row = _mapping(payload, action_kind)
    _bounded_supply_sku_ids(row["sku_ids"], f"{action_kind}.sku_ids")


_NEGOTIATION_REQUIRED = {
    "negotiation_id": _text,
    "offer_id": _text,
    "sku_id": _text,
    "counterparty_id": _text,
}
_NEGOTIATION_OPTIONAL = {
    "unit_price": _nonnegative_integer,
    "qty": _positive_integer,
    "fulfillment": _object,
    "round_no": _nonnegative_integer,
    "reason_code": _text,
    "reason": _text,
    "round_limit": _positive_integer,
    "probe_index": _nonnegative_integer,
    "probe_type": _text,
    "question": _text,
    "anchor_claim": _text,
    "claimed_reference_price_cents": _nonnegative_integer,
    "evidence_source": _text,
}

_AFTER_SALES_SPECS: dict[str, _PayloadSpec] = {
    "commerce.cancel_paid_order": _spec({"order_id": _text, "reason": _text}),
    "commerce.request_return": _spec(
        {
            "order_id": _text,
            "requested_qty": _positive_integer,
            "reason": _text,
            "evidence_ids": _text_list,
        }
    ),
    "commerce.authorize_return": _spec(
        {"order_id": _text, "request_id": _text, "reason": _text}
    ),
    "commerce.deny_return": _spec(
        {"order_id": _text, "request_id": _text, "reason": _text}
    ),
    "commerce.receive_return": _spec(
        {
            "order_id": _text,
            "request_id": _text,
            "authorization_id": _text,
            "received_qty": _positive_integer,
            "condition": _text,
        }
    ),
    "commerce.open_refund_case": _spec({"order_id": _text, "reason": _text}),
    "commerce.approve_refund": _spec(
        {"order_id": _text, "case_id": _text, "reason": _text}
    ),
    "commerce.deny_refund": _spec(
        {"order_id": _text, "case_id": _text, "reason": _text}
    ),
    "commerce.request_exchange": _spec(
        {"order_id": _text, "replacement_sku_id": _text, "reason": _text}
    ),
    "commerce.authorize_exchange": _spec(
        {"order_id": _text, "case_id": _text, "reason": _text}
    ),
    "commerce.deny_exchange": _spec(
        {"order_id": _text, "case_id": _text, "reason": _text}
    ),
    "commerce.complete_exchange": _spec(
        {"order_id": _text, "case_id": _text, "reason": _text}
    ),
    "commerce.open_dispute": _spec({"order_id": _text, "reason": _text}),
    "commerce.submit_dispute_evidence": _spec(
        {"order_id": _text, "dispute_id": _text, "evidence_id": _text}
    ),
    "commerce.respond_to_dispute": _spec(
        {
            "order_id": _text,
            "dispute_id": _text,
            "position": _text,
            "evidence_ids": _text_list,
        }
    ),
    "commerce.request_ledger_reconciliation": _spec(
        {"order_id": _text, "reason": _text}
    ),
    "commerce.read_payment_history": _spec({"order_id": _text}),
    "commerce.read_ledger_history": _spec({"order_id": _text}),
    "commerce.read_packing_history": _spec({"order_id": _text}),
    "commerce.read_after_sales_history": _spec({"order_id": _text}),
    "commerce.read_after_sales_policy": _spec({"merchant_id": _text}),
}

_PLATFORM_SPECS: dict[tuple[str, str], _PayloadSpec] = {
    ("commerce.search", "platform:aggregator"): _spec(
        {"query": _string, "limit": _positive_integer},
        {
            "benchmark_task_id": _text,
            "mandate_id": _text,
            "filter_contract": _enum("typed_constraints.v1"),
            "filters": _typed_search_filters,
            "hard_constraint_ids": _text_list,
            "hard_constraints": _object_list,
        },
    ),
    ("commerce.accept_remediation_plan", "platform:remediation"): _spec(
        {"plan_id": _text}
    ),
    ("commerce.complete_remediation_step", "platform:remediation"): _spec(
        {"plan_id": _text, "step_id": _text}
    ),
    ("commerce.publish_campaign", "platform:ads"): _spec({"campaign_id": _text}),
    ("commerce.activate_campaign", "platform:ads"): _spec({"campaign_id": _text}),
    ("commerce.disclose_placement", "platform:ads"): _spec(
        {"campaign_id": _text, "placement_id": _text, "disclosure_text": _text}
    ),
    ("commerce.reject_coordination", "platform:governance"): _spec(
        {"case_id": _text}
    ),
    ("commerce.reject_review_manipulation", "platform:governance"): _spec(
        {"case_id": _text}
    ),
    ("commerce.read_governance_history", "platform:governance"): _spec(
        {"record_kind": _text, "stable_id": _text}
    ),
    ("commerce.submit_review", "platform:reviews"): _spec(
        {"sku_id": _text, "rating": _rating},
        {"review_text": _string},
    ),
    # An actor states an observation. Issuer, ownership, read authority,
    # logical time, and the digest are sealed by platform:evidence, so the
    # actor-facing payload deliberately cannot carry a complete record.
    ("commerce.publish_evidence_record", "platform:evidence"): _spec(
        {"observation": _evidence_observation}
    ),
    ("commerce.allocate_fulfillment", "platform:fulfillment"): _spec(
        {
            "allocation_id": _text,
            "sku_id": _text,
            "priority_order_ids": _text_list,
        }
    ),
    ("commerce.read_shipment", "platform:fulfillment"): _spec(
        {"shipment_id": _text}
    ),
    ("commerce.resolve_shipment", "platform:fulfillment"): _spec(
        {"shipment_id": _text, "resolution": _enum("wait", "replacement", "refund")},
        {"replacement_sku_id": _text},
    ),
    ("commerce.apply_listing_claim", "platform:claims"): _spec(
        {
            "claim_id": _text,
            "listing_id": _text,
            "operation": _enum("publish", "correct", "retract"),
            "subject": _text,
        },
        {
            "content": _object,
            "reason": _text,
            "evidence_record_ids": _text_list,
        },
    ),
    ("commerce.get_sku", "platform:catalog"): _spec(
        {"request_id": _text, "sku_id": _text}
    ),
    ("commerce.update_listing", "platform:catalog"): _spec(
        {
            "op": _enum("publish", "update", "delist"),
            "sku_id": _text,
            "fields": _catalog_fields,
        },
        {"verification_source_id": _text},
    ),
    ("commerce.adjust_price", "platform:catalog"): _spec(
        {"sku_id": _text, "list_price": _positive_integer}
    ),
    ("commerce.read_supply_state", "platform:supply"): _spec(
        {"sku_ids": _text_list}
    ),
    ("commerce.update_supply", "platform:supply"): _spec(
        {"sku_id": _text},
        {
            "qty_delta": _integer,
            "eta_day": _integer,
            "unit_price_cents": _nonnegative_integer,
            "expected_version": _nonnegative_integer,
        },
    ),
    ("commerce.process_protocol_event", "platform:events"): _spec(
        {"event_id": _text, "reason": _text}
    ),
    ("commerce.acknowledge_protocol_event", "platform:events"): _spec(
        {"event_id": _text, "reason": _text}
    ),
    ("commerce.reject_protocol_event", "platform:events"): _spec(
        {"event_id": _text, "reason": _text}
    ),
    ("platform.checkout_cart", "platform:checkout"): _spec({"quote_id": _text}),
    ("platform.settle_payment", "platform:psp"): _spec(
        {
            "order_id": _text,
            "buyer_id": _text,
            "merchant_id": _text,
            "sku_id": _text,
            "qty": _positive_integer,
            "agreed_price": _money,
        },
        {
            "allow_partial": _boolean,
            "negotiation_id": _text,
            "payment_rail": _text,
            "cert_id": _text,
        },
    ),
    # Retained immediate-refund compatibility surface.  Platform recognizes
    # exactly these two historical shapes and normalizes them through the
    # authoritative after-sales intent contract before committing to World.
    ("commerce.request_return", "platform:psp"): _spec(
        {"order_id": _text},
        {"reason": _text},
    ),
}

for _kind, _payload_spec in _AFTER_SALES_SPECS.items():
    _PLATFORM_SPECS[(_kind, "platform:after-sales")] = _payload_spec

for _kind in ("commerce.propose_offer", "commerce.counter_offer"):
    _PLATFORM_SPECS[(_kind, "platform:negotiation")] = _spec(
        {**_NEGOTIATION_REQUIRED, "unit_price": _nonnegative_integer},
        _NEGOTIATION_OPTIONAL,
    )
_PLATFORM_SPECS[("commerce.accept_offer", "platform:negotiation")] = _spec(
    {**_NEGOTIATION_REQUIRED, "unit_price": _nonnegative_integer},
    _NEGOTIATION_OPTIONAL,
)
_PLATFORM_SPECS[("commerce.reject_offer", "platform:negotiation")] = _spec(
    _NEGOTIATION_REQUIRED,
    _NEGOTIATION_OPTIONAL,
)
_PLATFORM_SPECS[("commerce.withdraw_offer", "platform:negotiation")] = _spec(
    _NEGOTIATION_REQUIRED,
    _NEGOTIATION_OPTIONAL,
)


def _validate_acceptance(payload: Any, *, action_kind: str) -> None:
    row = _mapping(payload, action_kind)
    minimal = frozenset({"mandate_id", "offer_id"})
    certified = frozenset(
        {
            "mandate_id",
            "offer_id",
            "session_id",
            "session_digest",
            "offer_digest",
            "sku_id",
            "merchant_id",
            "qty",
            "unit_price_cents",
            "currency",
            "catalog_revision",
            "inventory_revision",
            "order_id",
        }
    )
    if frozenset(row) == minimal:
        _text(row["mandate_id"], f"{action_kind}.mandate_id")
        _text(row["offer_id"], f"{action_kind}.offer_id")
        return
    if frozenset(row) != certified:
        raise ActorTerminalContractError(
            f"{action_kind} payload must use the compact or certified acceptance shape"
        )
    for field in certified - {"qty", "unit_price_cents", "catalog_revision", "inventory_revision"}:
        _text(row[field], f"{action_kind}.{field}")
    _positive_integer(row["qty"], f"{action_kind}.qty")
    _nonnegative_integer(row["unit_price_cents"], f"{action_kind}.unit_price_cents")
    _nonnegative_integer(row["catalog_revision"], f"{action_kind}.catalog_revision")
    _nonnegative_integer(row["inventory_revision"], f"{action_kind}.inventory_revision")


def _validate_cart_request(
    payload: Any,
    *,
    action_kind: str,
    actor_role: str,
) -> None:
    row = _mapping(payload, action_kind)
    if action_kind == "commerce.request_cart_quote" and actor_role == "merchant":
        _spec(
            {"request_id": _text},
            {"quote_ttl_ticks": _positive_integer},
        ).validate(row, action_kind=action_kind)
        return
    if actor_role != "buyer":
        raise ActorTerminalContractError(
            f"{action_kind} has no cart request shape for {actor_role}"
        )
    ttl_name = (
        "request_ttl_ticks"
        if action_kind == "commerce.create_cart_quote_request"
        else "quote_ttl_ticks"
    )
    _spec(
        {
            "market_id": _text,
            "mandate_id": _text,
            "lines": _cart_lines,
            "fill_policy": _enum("all_or_none", "allow_partial"),
            "backorder_policy": _enum("reject", "allow"),
        },
        {ttl_name: _positive_integer},
    ).validate(row, action_kind=action_kind)


def _validate_actor_result_payload(
    action_kind: str,
    payload: Any,
) -> tuple[
    Literal["actor_report_submission", "actor_result_report"],
    ActorReportContent | ActorResultReport,
]:
    row = _mapping(payload, f"{action_kind} payload")
    try:
        if frozenset(row) == frozenset({"outcome", "summary", "details"}):
            return "actor_report_submission", coerce_actor_report_submission(row)
        report = coerce_actor_result_report(row)
    except ActorResultSchemaError as exc:
        raise ActorTerminalContractError(
            "actor result payload violates the actor terminal contract"
        ) from exc
    if report.action_kind != action_kind:
        raise ActorTerminalContractError(
            "sealed actor result action_kind differs from its envelope"
        )
    return "actor_result_report", report


_BENCHMARK_PLATFORM_ACTION_KINDS = frozenset(
    kind for kind, _endpoint in _PLATFORM_SPECS
) | frozenset(
    {
        "commerce.accept_offer",
        "commerce.create_cart_quote_request",
        "commerce.request_cart_quote",
    }
)


def _validate_settlement(payload: Any, *, action_kind: str) -> None:
    """Validate the current and retained typed settlement request forms."""

    row = _mapping(payload, action_kind)
    keys = frozenset(row)
    if keys == frozenset({"cert_id"}):
        _text(row["cert_id"], f"{action_kind}.cert_id")
        return
    supply_required = frozenset({
        "supply_authority_id",
        "supply_authority_digest",
        "sku_id",
        "qty",
    })
    if keys in {
        supply_required,
        supply_required | {"allow_partial"},
    }:
        _text(row["supply_authority_id"], f"{action_kind}.supply_authority_id")
        _text(
            row["supply_authority_digest"],
            f"{action_kind}.supply_authority_digest",
        )
        _text(row["sku_id"], f"{action_kind}.sku_id")
        _positive_integer(row["qty"], f"{action_kind}.qty")
        if "allow_partial" in row:
            _boolean(row["allow_partial"], f"{action_kind}.allow_partial")
        return
    if not keys:
        # Retained negative conformance shape.  Platform rejects it without a
        # World effect; keeping it here preserves that explicit policy test.
        return
    _PLATFORM_SPECS[(action_kind, "platform:psp")].validate(
        row, action_kind=action_kind
    )


def preview_actor_terminal_action(env: Envelope) -> ActorTerminalActionPreview:
    """Validate and preview one buyer/merchant terminal output.

    Unknown Platform routes, the bare ``platform`` address, every actor emit to
    World, and non-report Runtime destinations fail closed.  Participant and
    consumer messages used by the benchmark remain role-routed because their
    concrete identity is scenario-defined; Runtime separately verifies that a
    participant identity is registered.
    """

    role = env.from_.split(":", 1)[0]
    if role not in {"buyer", "merchant"}:
        raise ActorTerminalContractError(
            "actor terminal contract accepts only buyer or merchant outputs"
        )
    kind = str(env.action.get("kind", ""))
    payload = env.action.get("payload")
    destination_role = env.to.split(":", 1)[0]

    if destination_role == "world":
        raise ActorTerminalContractError(
            "buyer and merchant terminal actions may not target World"
        )
    if env.to == "platform":
        raise ActorTerminalContractError(
            "actor terminal actions require a typed Platform endpoint"
        )
    if destination_role == "platform":
        if (kind, env.to) == ("commerce.update_listing", "platform:catalog"):
            _validate_catalog_listing_update(payload, action_kind=kind)
        elif (kind, env.to) == ("commerce.search", "platform:aggregator"):
            _validate_search(payload, action_kind=kind)
        elif (kind, env.to) == ("commerce.accept_offer", "platform:aggregator"):
            _validate_acceptance(payload, action_kind=kind)
        elif (kind, env.to) == ("commerce.apply_listing_claim", "platform:claims"):
            _validate_listing_claim(payload, action_kind=kind)
        elif (kind, env.to) == ("commerce.update_supply", "platform:supply"):
            _validate_supply_update(payload, action_kind=kind)
        elif (kind, env.to) == (
            "commerce.allocate_fulfillment",
            "platform:fulfillment",
        ):
            _validate_allocation(payload, action_kind=kind)
        elif (kind, env.to) == ("commerce.read_supply_state", "platform:supply"):
            _validate_supply_read(payload, action_kind=kind)
        elif (kind, env.to) in {
            ("commerce.create_cart_quote_request", "platform:checkout"),
            ("commerce.request_cart_quote", "platform:checkout"),
        }:
            _validate_cart_request(
                payload,
                action_kind=kind,
                actor_role=role,
            )
        elif (kind, env.to) == ("platform.settle_payment", "platform:psp"):
            _validate_settlement(payload, action_kind=kind)
        else:
            spec = _PLATFORM_SPECS.get((kind, env.to))
            if spec is None:
                if kind in _BENCHMARK_PLATFORM_ACTION_KINDS:
                    raise ActorTerminalContractError(
                        "actor terminal action uses a non-canonical benchmark Platform route"
                    )
                if env.to not in _KNOWN_PLATFORM_ENDPOINTS:
                    raise ActorTerminalContractError(
                        "actor terminal action targets an unknown Platform endpoint"
                    )
                # This minimal contract intentionally covers only the fixed
                # 200-task action set.  Other established Platform endpoints
                # retain Router and Platform validation, while their payload
                # still must be an object at this shared wire boundary.
                _mapping(payload, kind)
                return ActorTerminalActionPreview(kind, env.to, "mapping")
            spec.validate(payload, action_kind=kind)
        return ActorTerminalActionPreview(kind, env.to, "mapping")

    if destination_role == "runtime":
        if env.to != "runtime:evidence" or kind not in _ACTOR_RESULT_ACTIONS:
            raise ActorTerminalContractError(
                "actor terminal Runtime actions must be result reports to runtime:evidence"
            )
        payload_form, report = _validate_actor_result_payload(kind, payload)
        return ActorTerminalActionPreview(kind, env.to, payload_form, report)

    if destination_role in {"buyer", "merchant", "consumer"}:
        _mapping(payload, kind)
    else:
        raise ActorTerminalContractError(
            "actor terminal action has no canonical benchmark participant route"
        )
    return ActorTerminalActionPreview(kind, env.to, "mapping")


def validate_actor_terminal_action(env: Envelope) -> None:
    """Validate one actor terminal action without producing side effects."""

    preview_actor_terminal_action(env)


BENCHMARK_PLATFORM_ACTOR_ROUTES = frozenset(_PLATFORM_SPECS) | frozenset(
    {
        ("commerce.accept_offer", "platform:aggregator"),
        ("commerce.create_cart_quote_request", "platform:checkout"),
        ("commerce.request_cart_quote", "platform:checkout"),
    }
)


__all__ = [
    "ActorTerminalActionPreview",
    "ActorTerminalContractError",
    "BENCHMARK_PLATFORM_ACTOR_ROUTES",
    "preview_actor_terminal_action",
    "validate_actor_terminal_action",
]
