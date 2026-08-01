"""ACWorld order lifecycle and after sales tasks for family T7.

Every request is mediated by the first-class after-sales Platform service and
committed through authoritative World payment, packing, policy, and
after-sales tables.  The task adapter contains fixed task materialization,
scripted reference policies, and deterministic evidence scoring only.  It has
no benchmark-local state machine, fabricated Platform response, or direct
World mutation path.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from agents.agent_phase import public_reference_alias_v1
from agents.business_decision import LLM_BUSINESS_DECISION_SCHEMA_V1
from agents.inference import BusinessDecisionResponseV1, InferenceChannel
from episode.capability_benchmark import (
    TASK_REGISTRY_V2,
    TaskDefinitionV2,
    is_hardened_task_v2,
)
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
from episode.capability_runtime_authority import (
    runtime_evidence_core_ownership_violations_v2,
)
from episode.capability_runtime_fixture import require_frozen_scenario_fixture_v2
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec
from protocol.evidence_records import build_evidence_record, evidence_record_to_dict
from runtime.after_sales_evidence import (
    AFTER_SALES_EVIDENCE_CONTRACT,
    VerifiedAfterSalesEvidence,
    VerifiedAfterSalesRequest,
    VerifiedAfterSalesServiceOperation,
)
from world.payment_fulfillment import authoritative_payment_receipt_digest
from world.types import AgentId, Money, OrderId, Receipt, SkuId, TxnId


T7_RUNTIME_SCHEMA_V2 = "cwe.runtime-task.t7.v3"
_BUYER_ID = "buyer:t7-benchmark"
_MERCHANT_ID = "merchant:t7-benchmark"

_T7_TASK_IDS = tuple(
    task_id for task_id, definition in TASK_REGISTRY_V2.items() if definition.family.value == "T7"
)

T7_RUNTIME_READY_TASK_IDS = _T7_TASK_IDS
T7_RUNTIME_PENDING_TASK_IDS: tuple[str, ...] = ()


Scalar = str | int


@dataclass(frozen=True, slots=True)
class _ProductT7:
    sku_id: str
    product_id: str
    merchant_id: str
    name: str
    list_price: str
    inventory: int
    product_facts: tuple[tuple[str, Scalar], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku_id": self.sku_id,
            "product_id": self.product_id,
            "merchant_id": self.merchant_id,
            "name": self.name,
            "list_price": self.list_price,
            "inventory": self.inventory,
            "product_facts": dict(self.product_facts),
        }


@dataclass(frozen=True, slots=True)
class _CaseT7:
    definition: TaskDefinitionV2
    lane: str
    axis_name: str
    axis_value: Scalar
    order_id: str
    original_sku_id: str
    products: tuple[_ProductT7, ...]
    operation_sequence: tuple[str, ...]
    evaluated_operations: tuple[str, ...]
    initial_payment_stage: str
    initial_packing_stage: str | None
    allowed_return_conditions: tuple[str, ...] = ()
    # What the inspections actually say the returned item is like.  This is
    # separate from the policy on purpose: when the two disagree the merchant
    # is supposed to refuse, and a lane where they can never disagree is a
    # lane where "approve" is right by construction.
    attested_return_conditions: tuple[str, ...] = ()
    replacement_requirements: tuple[tuple[str, Scalar], ...] = ()
    # Which replacement the buyer actually asked for on a merchant-evaluated
    # exchange.  ``None`` means the buyer picks the eligible one at run time.
    requested_replacement_sku_id: str | None = None
    trusted_evidence_facts: tuple[tuple[str, str], ...] = ()
    ledger_source_count: int = 0
    prerequisite_causal_layers: tuple[str, ...] = ()

    @property
    def evaluated_actor_id(self) -> str:
        return _BUYER_ID if self.definition.evaluated_role == "buyer" else _MERCHANT_ID

    @property
    def entitled(self) -> bool:
        """Say whether the customer is owed what they have asked for.

        Scorer-side only.  Nothing derived from this reaches the model: the
        facts that decide it -- the published returns policy against the
        inspections, the published replacement rules against the item the
        buyer named -- are all visible, and working out which way they point
        is the task.
        """

        if self.lane == "merchant_return_authorization":
            accepted = _accepted_return_conditions(self)
            return self.attested_return_conditions[0] in accepted
        if self.lane == "merchant_exchange" and self.requested_replacement_sku_id is not None:
            return self.requested_replacement_sku_id == _expected_replacement_sku(self)
        return True

    @property
    def order_ids(self) -> tuple[str, ...]:
        if self.ledger_source_count:
            return tuple(
                f"{self.order_id}:{index}" for index in range(1, self.ledger_source_count + 1)
            )
        return (self.order_id,)

    @property
    def shipment_id(self) -> str | None:
        if self.initial_packing_stage == "handed_off":
            return f"shipment:{_slug(self.definition)}"
        return None

    @property
    def dispute_evidence_counts(self) -> tuple[int, int]:
        if self.axis_name != "evidence_item_count":
            return (0, 0)
        evaluated_count = int(self.axis_value)
        if self.lane == "buyer_dispute":
            return (evaluated_count, 1)
        if self.lane == "merchant_dispute":
            return (1, evaluated_count)
        raise ValueError("evidence_item_count is valid only for a dispute lane")

    @property
    def expected_ruling_operation(self) -> str | None:
        filer_count, respondent_count = self.dispute_evidence_counts
        if not filer_count and not respondent_count:
            return None
        if filer_count > respondent_count:
            return "rule_for_filer"
        if respondent_count > filer_count:
            return "rule_for_respondent"
        return "rule_split"

    @property
    def counterpart_operations(self) -> tuple[str, ...]:
        operations = _COUNTERPART_OPERATIONS[self.lane]
        if self.axis_name == "evidence_item_count" and self.lane == "merchant_dispute":
            filer_count, _ = self.dispute_evidence_counts
            return _repeat_operation(
                operations,
                "submit_dispute_evidence",
                filer_count,
            )
        return operations

    @property
    def service_operations(self) -> tuple[str, ...]:
        if self.lane == "merchant_ledger_close":
            return ("complete_ledger_reconciliation",) * self.ledger_source_count
        if self.expected_ruling_operation is not None:
            return (self.expected_ruling_operation,)
        return _SERVICE_OPERATIONS[self.lane]

    @property
    def semantic_contract(self) -> dict[str, Any]:
        return {
            "schema_version": T7_RUNTIME_SCHEMA_V2,
            "definition": self.definition.to_dict(),
            **(
                {"evaluation_profile": "hard-tier-step-attribution"}
                if is_hardened_task_v2(self.definition)
                else {}
            ),
            "lane": self.lane,
            "difficulty": {self.axis_name: self.axis_value},
            "order_id": self.order_id,
            "order_ids": self.order_ids,
            "shipment_id": self.shipment_id,
            "original_sku_id": self.original_sku_id,
            "products": tuple(row.to_dict() for row in self.products),
            "operation_sequence": self.operation_sequence,
            "evaluated_operations": self.evaluated_operations,
            "counterpart_operations": self.counterpart_operations,
            "service_operations": self.service_operations,
            "initial_payment_stage": self.initial_payment_stage,
            "initial_packing_stage": self.initial_packing_stage,
            "after_sales_policy": {
                "allowed_return_conditions": self.allowed_return_conditions,
            },
            "attested_return_conditions": self.attested_return_conditions,
            "requested_replacement_sku_id": self.requested_replacement_sku_id,
            "entitled": self.entitled,
            "replacement_requirements": dict(self.replacement_requirements),
            "trusted_evidence_facts": self.trusted_evidence_facts,
            "dispute_evidence_counts": {
                "filer": self.dispute_evidence_counts[0],
                "respondent": self.dispute_evidence_counts[1],
            },
            "ledger_source_count": self.ledger_source_count,
            "prerequisite_causal_layers": self.prerequisite_causal_layers,
            "runtime_ready": True,
            "authority_path": (
                "actor envelope",
                "platform:after-sales validation",
                "World compact intent",
                "typed World rows and atomic commit",
                AFTER_SALES_EVIDENCE_CONTRACT,
                "deterministic Python score",
            ),
        }


def _axis(definition: TaskDefinitionV2) -> tuple[str, Scalar]:
    values = [
        (name, value) for name, value in definition.difficulty_factors if name != "difficulty_level"
    ]
    if len(values) != 1:
        raise ValueError(f"{definition.task_id}: T7 needs exactly one semantic axis")
    name, value = values[0]
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{definition.task_id}: unsupported T7 difficulty value")
    return name, value


def _slug(definition: TaskDefinitionV2) -> str:
    return definition.task_id.casefold().replace("-", "")


def _sku_id(definition: TaskDefinitionV2, token: str) -> str:
    return f"merchant:t7:{_slug(definition)}:{token}"


def _product(
    definition: TaskDefinitionV2,
    token: str,
    *,
    inventory: int = 10,
    facts: Mapping[str, Scalar] | None = None,
) -> _ProductT7:
    sku_id = _sku_id(definition, token)
    return _ProductT7(
        sku_id=sku_id,
        product_id=f"product:{_slug(definition)}",
        merchant_id=_MERCHANT_ID,
        name=f"T7 lifecycle item {token}",
        list_price="115.00",
        inventory=inventory,
        product_facts=tuple(sorted((facts or {}).items())),
    )


def _requirements(count: int) -> tuple[tuple[str, Scalar], ...]:
    return (
        ("color", "black"),
        ("size", "large"),
        ("warranty_months", 24),
    )[:count]


def _return_evidence(count: int) -> tuple[tuple[str, str], ...]:
    reports = (
        ("inspection_report_1", "authenticated inspection confirms item identity"),
        ("inspection_report_2", "authenticated inspection confirms item condition"),
        ("inspection_report_3", "authenticated inspection confirms returned quantity"),
    )
    return reports[:count]


def _repeat_operation(operations: tuple[str, ...], operation: str, count: int) -> tuple[str, ...]:
    expanded: list[str] = []
    for item in operations:
        expanded.extend((item,) * count if item == operation else (item,))
    return tuple(expanded)


_OPERATIONS: Mapping[str, tuple[str, ...]] = {
    "buyer_cancel": ("cancel_paid_order",),
    "merchant_cancel": ("cancel_paid_order",),
    "buyer_return_refund": (
        "request_return",
        "authorize_return",
        "receive_return",
        "open_refund_case",
        "approve_refund",
    ),
    "merchant_return_authorization": ("request_return", "authorize_return"),
    "merchant_refund": (
        "request_return",
        "authorize_return",
        "receive_return",
        "open_refund_case",
        "approve_refund",
    ),
    "buyer_exchange": (
        "request_return",
        "authorize_return",
        "receive_return",
        "request_exchange",
        "authorize_exchange",
        "complete_exchange",
    ),
    "merchant_exchange": (
        "request_return",
        "authorize_return",
        "receive_return",
        "request_exchange",
        "authorize_exchange",
        "complete_exchange",
    ),
    "buyer_dispute": (
        "open_dispute",
        "submit_dispute_evidence",
        "respond_to_dispute",
        "rule_split",
        "open_refund_case",
        "approve_refund",
    ),
    "merchant_dispute": (
        "open_dispute",
        "submit_dispute_evidence",
        "respond_to_dispute",
        "rule_split",
        "open_refund_case",
        "approve_refund",
    ),
    "merchant_ledger_close": (
        "request_ledger_reconciliation",
        "complete_ledger_reconciliation",
    ),
}


_EVALUATED_OPERATIONS: Mapping[str, tuple[str, ...]] = {
    "buyer_cancel": ("cancel_paid_order",),
    "merchant_cancel": ("cancel_paid_order",),
    # The evaluated buyer owns the return request and opens the refund only
    # after the deterministic merchant has authorized and received the item.
    # Physical receipt is a merchant attestation, never a buyer self-report.
    "buyer_return_refund": ("request_return", "open_refund_case"),
    "merchant_return_authorization": ("authorize_return",),
    # The evaluated merchant performs each legal merchant decision.  Payment
    # state changes and ledger effects are derived atomically by World; the old
    # direct-simulator ``refund_step`` action is intentionally not reproduced.
    "merchant_refund": (
        "authorize_return",
        "receive_return",
        "approve_refund",
    ),
    "buyer_exchange": ("request_return", "request_exchange"),
    "merchant_exchange": (
        "authorize_return",
        "receive_return",
        "authorize_exchange",
        "complete_exchange",
    ),
    "buyer_dispute": (
        "open_dispute",
        "submit_dispute_evidence",
        "open_refund_case",
    ),
    "merchant_dispute": ("respond_to_dispute", "approve_refund"),
    "merchant_ledger_close": ("request_ledger_reconciliation",),
}

_COUNTERPART_OPERATIONS: Mapping[str, tuple[str, ...]] = {
    "buyer_cancel": (),
    "merchant_cancel": (),
    "buyer_return_refund": (
        "authorize_return",
        "receive_return",
        "approve_refund",
    ),
    "merchant_return_authorization": ("request_return",),
    "merchant_refund": ("request_return", "open_refund_case"),
    "buyer_exchange": (
        "authorize_return",
        "receive_return",
        "authorize_exchange",
        "complete_exchange",
    ),
    "merchant_exchange": ("request_return", "request_exchange"),
    "buyer_dispute": ("respond_to_dispute", "approve_refund"),
    "merchant_dispute": (
        "open_dispute",
        "submit_dispute_evidence",
        "open_refund_case",
    ),
    "merchant_ledger_close": (),
}

_SERVICE_OPERATIONS: Mapping[str, tuple[str, ...]] = {
    "buyer_cancel": (),
    "merchant_cancel": (),
    "buyer_return_refund": (),
    "merchant_return_authorization": (),
    "merchant_refund": (),
    "buyer_exchange": (),
    "merchant_exchange": (),
    "buyer_dispute": ("rule_split",),
    "merchant_dispute": ("rule_split",),
    "merchant_ledger_close": ("complete_ledger_reconciliation",),
}

# What the customer or the shop owner says.  Each one still names every step
# the evaluated actor has to take, because these are the task, not decoration.
_INSTRUCTIONS: Mapping[str, str] = {
    "buyer_cancel": (
        "I want to call off this order. It is already paid for, so check where "
        "it has actually got to and cancel it from there."
    ),
    "merchant_cancel": (
        "A customer wants to call off an order they have already paid for. "
        "Check that they are entitled to, then put the cancellation through."
    ),
    "buyer_return_refund": (
        "I want to send this back and get my money back. Ask them to take the "
        "return first, and only put in for the refund once the item is back "
        "with them."
    ),
    "merchant_return_authorization": (
        "A customer has asked to send something back. Look at what the "
        "inspection says the item is like, check it against what our returns "
        "policy actually takes back, and either take the return or turn it "
        "down. Do not take back something we do not accept."
    ),
    "merchant_refund": (
        "A customer is sending something back for a refund. Go through what "
        "has actually happened on this order, approve the return, log the item "
        "as received, then pay back exactly what is owed."
    ),
    "buyer_exchange": (
        "I want this swapped for a different one. Arrange to send the original "
        "back, and make sure the replacement you ask for actually meets what I "
        "asked for in the first place."
    ),
    "merchant_exchange": (
        "A customer wants to swap an item. Take the return, log the original "
        "as received, then look at which replacement they have asked for. If "
        "it is one they are entitled to, approve it and send it out. If it is "
        "not, turn the swap down — do not send out something that does not "
        "meet what we said."
    ),
    "buyer_dispute": (
        "This order went wrong and I want it raised formally. Open the case, "
        "then put in every piece of proof of mine that someone independent "
        "actually checked. Leave out anything that is only my own say-so — it "
        "will not hold up and it weakens the rest. Then put in for the refund."
    ),
    "merchant_dispute": (
        "A customer has raised a case against us. Reply with every piece of "
        "proof of ours that someone independent actually checked, and leave "
        "out anything that is only our own word for it. Then do whatever the "
        "ruling says, even if it goes against us."
    ),
    "merchant_ledger_close": (
        "Close the books on this. Ask for a proper reconciliation covering "
        "every order that was affected."
    ),
}


@lru_cache(maxsize=None)
def _case_for_t7(task_id: str) -> _CaseT7:
    try:
        definition = TASK_REGISTRY_V2[task_id]
    except KeyError as exc:
        raise KeyError(f"unknown Benchmark v2 task {task_id!r}") from exc
    if definition.family.value != "T7":
        raise ValueError(f"{task_id} is not a T7 task")
    axis_name, axis_value = _axis(definition)
    lane = definition.capability_id.removeprefix("t7.")
    if lane not in _OPERATIONS:
        raise ValueError(f"{task_id}: unsupported T7 lane {lane!r}")

    original = _product(definition, "original", facts={"material": "alloy"})
    products: list[_ProductT7] = [original]
    conditions: tuple[str, ...] = ()
    attested: tuple[str, ...] = ()
    requirements: tuple[tuple[str, Scalar], ...] = ()
    requested_replacement: str | None = None
    evidence: tuple[tuple[str, str], ...] = ()
    ledger_count = 0
    prerequisite_records: tuple[str, ...] = ()

    if axis_name == "return_condition_count":
        if not isinstance(axis_value, int):
            raise ValueError(f"{task_id}: return condition count must be integer")
        conditions = ("new", "opened", "damaged")[:axis_value]
        # One item comes back in one state.  The axis counts how many separate
        # inspections were run on it, not how many states it is in at once.
        #
        # On the lane where the merchant decides, both tiers are handed the
        # same finding -- the item is damaged -- and differ only in whether
        # the shop's published policy takes damaged goods back.  So the
        # decision cannot be read off the finding; the two have to be compared.
        attested = (
            ("damaged",)
            if lane == "merchant_return_authorization"
            else (conditions[-1],)
        )
        evidence = _return_evidence(axis_value)
    elif axis_name == "replacement_constraint_count":
        if not isinstance(axis_value, int):
            raise ValueError(f"{task_id}: replacement constraint count must be integer")
        requirements = _requirements(axis_value)
        # Exactly one item is both eligible and available.  Every other item
        # fails on a single point and advertises none of it.  A near miss is
        # only included once the requirement it breaks is actually in force,
        # otherwise it would qualify too and the replacement would stop being
        # unique.
        eligible_facts = {"color": "black", "size": "large", "warranty_months": 24}
        products.append(
            _product(definition, "replacement-valid", inventory=5, facts=eligible_facts)
        )
        # Deliberately no out-of-stock look-alike among these.  Such an item
        # would read as the obvious trap for a model that checks the stated
        # requirements but not availability, and it would never catch one: the
        # environment drops out-of-stock SKUs from the exchange enum, so it
        # could not be selected in the first place.  A candidate that cannot be
        # chosen is not a distractor.
        products.append(
            _product(
                definition,
                "replacement-invalid",
                inventory=5,
                facts={**eligible_facts, "color": "white"},
            )
        )
        if axis_value >= 2:
            products.append(
                _product(
                    definition,
                    "replacement-wrong-size",
                    inventory=5,
                    facts={**eligible_facts, "size": "medium"},
                )
            )
        if axis_value >= 3:
            # Short by a single month, so the warranty has to be read rather
            # than glanced at.
            products.append(
                _product(
                    definition,
                    "replacement-short-warranty",
                    inventory=5,
                    facts={**eligible_facts, "warranty_months": 23},
                )
            )
        if lane == "merchant_exchange":
            # The buyer names the item they want and the shop rules on it.
            #
            # At the wider tier they name one that breaks a requirement they
            # were told about, and the shop is supposed to turn it down rather
            # than wave it through.  The break is deliberately the quiet one:
            # right colour, right size, warranty short by a single month.  A
            # wrong colour can be caught at a glance; this has to be compared.
            #
            # An out-of-stock item would be sharper still -- right on every
            # stated point, none on the shelf -- but it cannot be reached: the
            # environment drops out-of-stock SKUs from the exchange enum, so
            # nobody can ask for one in the first place.
            requested_replacement = _sku_id(
                definition,
                "replacement-short-warranty" if axis_value >= 3 else "replacement-valid",
            )
    elif axis_name == "evidence_item_count":
        if not isinstance(axis_value, int):
            raise ValueError(f"{task_id}: evidence count must be integer")
        filer_count = axis_value if lane == "buyer_dispute" else 1
        respondent_count = axis_value if lane == "merchant_dispute" else 1
        evidence = tuple(
            (
                f"{side}_carrier_exception_{index}",
                f"verified {side} delivery exception observation {index}",
            )
            for side, count in (
                ("filer", filer_count),
                ("respondent", respondent_count),
            )
            for index in range(1, count + 1)
        )
    elif axis_name == "affected_ledger_count":
        if not isinstance(axis_value, int):
            raise ValueError(f"{task_id}: ledger count must be integer")
        ledger_count = axis_value
    elif axis_name == "payment_state_depth":
        if axis_value not in {2, 4}:
            raise ValueError(f"{task_id}: payment causal depth must be 2 or 4")
        # ``payment_state_depth`` is retained as the published task axis.  In
        # the real World it denotes caller-visible payment-and-fulfillment
        # prerequisites, not synthetic agent-authored payment steps.
        prerequisite_records = (
            "payment_state_history",
            "exact_charge_receipt",
            *(("packing_record_history", "shipment_fulfillment_state") if axis_value == 4 else ()),
        )
    elif axis_name != "order_stage":
        raise ValueError(f"{task_id}: unsupported T7 axis {axis_name!r}")

    packed = axis_name == "order_stage" and axis_value == "packed"
    causal_fulfillment = axis_name == "payment_state_depth" and axis_value == 4
    operation_sequence = _OPERATIONS[lane]
    evaluated_operations = _EVALUATED_OPERATIONS[lane]
    if axis_name == "evidence_item_count":
        filer_count = int(axis_value) if lane == "buyer_dispute" else 1
        respondent_count = int(axis_value) if lane == "merchant_dispute" else 1
        operation_sequence = _repeat_operation(
            operation_sequence,
            "submit_dispute_evidence",
            filer_count,
        )
        expected_ruling = (
            "rule_for_filer"
            if filer_count > respondent_count
            else "rule_for_respondent"
            if respondent_count > filer_count
            else "rule_split"
        )
        operation_sequence = tuple(
            expected_ruling if operation.startswith("rule_") else operation
            for operation in operation_sequence
        )
        # A ruling for the respondent authorizes no refund.  The evaluated
        # merchant must deny the resulting zero-amount case; approving it would
        # manufacture a refund ledger effect and misstate the payment/order
        # lifecycle.  Split and filer rulings continue through approval because
        # their authoritative refund amount is positive.
        if expected_ruling == "rule_for_respondent":
            operation_sequence = tuple(
                "deny_refund" if operation == "approve_refund" else operation
                for operation in operation_sequence
            )
        if lane == "buyer_dispute":
            evaluated_operations = _repeat_operation(
                evaluated_operations,
                "submit_dispute_evidence",
                int(axis_value),
            )
        elif lane == "merchant_dispute" and expected_ruling == "rule_for_respondent":
            evaluated_operations = tuple(
                "deny_refund" if operation == "approve_refund" else operation
                for operation in evaluated_operations
            )
    if ledger_count:
        operation_sequence = (
            "request_ledger_reconciliation",
            "complete_ledger_reconciliation",
        ) * ledger_count
        evaluated_operations = ("request_ledger_reconciliation",) * ledger_count

    case = _CaseT7(
        definition=definition,
        lane=lane,
        axis_name=axis_name,
        axis_value=axis_value,
        order_id=f"order:{_slug(definition)}",
        original_sku_id=original.sku_id,
        products=tuple(products),
        operation_sequence=operation_sequence,
        evaluated_operations=evaluated_operations,
        initial_payment_stage="authorized" if axis_value == "authorized" else "captured",
        initial_packing_stage=(
            "packed" if packed else "handed_off" if causal_fulfillment else None
        ),
        allowed_return_conditions=conditions,
        attested_return_conditions=attested,
        replacement_requirements=requirements,
        requested_replacement_sku_id=requested_replacement,
        trusted_evidence_facts=evidence,
        ledger_source_count=ledger_count,
        prerequisite_causal_layers=prerequisite_records,
    )
    if case.entitled:
        return case
    # A request policy does not cover ends at the refusal.  Nothing after it is
    # legitimate work, so neither the lifecycle nor the evaluated operations
    # continue past that point.
    granting, denial = (
        ("authorize_return", "deny_return")
        if lane == "merchant_return_authorization"
        else ("authorize_exchange", "deny_exchange")
    )
    completed = operation_sequence[: operation_sequence.index(granting)]
    return replace(
        case,
        operation_sequence=(*completed, denial),
        evaluated_operations=tuple(
            denial if operation == granting else operation
            for operation in evaluated_operations
            if operation in completed or operation == granting
        ),
    )


def t7_runtime_content(task_id: str) -> dict[str, Any]:
    """Return the stable, CommerceWorld-backed task contract."""

    return copy.deepcopy(_case_for_t7(task_id).semantic_contract)


def _workflow_policy(
    case: _CaseT7,
    *,
    read_prerequisites: Sequence[str],
) -> dict[str, Any]:
    """Publish the business rules without serializing the answer.

    Every tier gets this form.  The baseline tiers used to publish
    ``required_operation_sequence`` -- the literal list of operations the
    scorer expects -- which made the disposition a transcription exercise: a
    model could execute a return, a denial, or a refund without ever deciding
    that it was the right one.  What a real after-sales operator is given is
    the policy and the request, so that is what is published here.
    """

    rules = {
        "buyer_cancel": (
            "Ground the current order stage before requesting cancellation.",
            "Do not claim that a rejected cancellation changed commercial state.",
        ),
        "merchant_cancel": (
            "Resolve cancellation under the current payment and packing state.",
            "Only accepted operations may change the order or ledger.",
        ),
        "buyer_return_refund": (
            "Cite every authenticated inspection needed by the return policy.",
            "Open a refund only after the return has been received.",
        ),
        "merchant_return_authorization": (
            "Authorize only a request whose inspected condition is one the "
            "published return policy accepts; otherwise deny it.",
            "A denial must leave commercial state unchanged.",
        ),
        "merchant_refund": (
            "Read every declared payment and fulfillment prerequisite.",
            "Receive the return before approving the exact refundable amount.",
        ),
        "buyer_exchange": (
            "Request a return before requesting an exchange.",
            "Choose an in-stock replacement satisfying every declared constraint.",
        ),
        "merchant_exchange": (
            "Validate and receive the return before deciding the exchange.",
            "Authorize the exchange only when the replacement the buyer asked "
            "for meets every stated requirement and is in stock; otherwise "
            "deny it.",
        ),
        "buyer_dispute": (
            "Open the dispute before submitting evidence.",
            "Submit every verified filer record and nothing that was not "
            "independently verified.",
        ),
        "merchant_dispute": (
            "Respond with every verified respondent record and nothing that "
            "was not independently verified.",
            "Follow the authoritative ruling when approving or denying a refund.",
        ),
        "merchant_ledger_close": (
            "Request reconciliation once for every affected order.",
            "Do not manufacture ledger effects during reconciliation.",
        ),
    }[case.lane]
    return {
        "rule_set": "after-sales-evaluated-workflow.v2",
        # Names which after-sales workflow this is.  The shared provider-view
        # solver dispatches on this marker so that the wording of
        # ``business_goal`` stays free to read like a real customer or
        # merchant message.
        "capability": case.lane,
        "business_goal": _INSTRUCTIONS[case.lane],
        "business_rules": list(rules),
        "required_read_prerequisites": list(read_prerequisites),
        "ordering": "satisfy_prerequisites_before_dependent_operations",
        "unavailable_next_operation": "wait_without_advancing",
    }


def _public_contract(case: _CaseT7) -> dict[str, Any]:
    """Return only task inputs that the evaluated actor may legitimately see."""

    evidence_rows = _scenario_evidence_rows(case)
    evidence_ids = [row["record_id"] for row in evidence_rows]
    # Publish each record with how it was checked, and stop publishing the
    # pre-filtered "these are the ones to file" lists.  Those lists were the
    # answer: a model could file exactly them without ever looking at whether
    # a record stands up.
    evidence_records = [
        {
            "evidence_id": row["record_id"],
            "kind": row["kind"],
            "issued_by": row["issuer_id"],
            "side": row["facts"].get("side"),
            "verification_method": row.get("trust", {}).get("verification_method"),
            "verified": row.get("trust", {}).get("verified"),
        }
        for row in evidence_rows
    ]
    read_prerequisites: list[str] = []
    if case.lane == "merchant_refund":
        read_prerequisites.extend(("read_payment_history", "read_ledger_history"))
        if case.axis_value == 4:
            read_prerequisites.extend(("read_packing_history", "read_shipment"))
    elif case.lane == "merchant_exchange":
        # The shop can look its own stock up, so it is told which item was
        # asked for and nothing else about it.  Whether that item meets what
        # the buyer was promised, and whether any of it is on the shelf, has
        # to be read rather than handed over.
        read_prerequisites.extend(("observe_listing", "observe_stock_availability"))
    workflow_policy = _workflow_policy(case, read_prerequisites=read_prerequisites)
    # Rotate the published candidate order so the eligible replacement is not
    # always the first row.  Position is presentation only -- the reference
    # solver and the scorer both identify the replacement by its requirements
    # and its stock -- but publishing it first every time let a model take row
    # one without comparing anything, which made every added near miss inert.
    replacements = [row for row in case.products if row.sku_id != case.original_sku_id]
    if replacements:
        offset = int(case.definition.task_id.rsplit("-", 1)[1]) % len(replacements)
        replacements = replacements[offset:] + replacements[:offset]
    return {
        "schema_version": T7_RUNTIME_SCHEMA_V2,
        "task_id": case.definition.task_id,
        "instruction": _INSTRUCTIONS[case.lane],
        "evaluated_role": case.definition.evaluated_role,
        "capability": case.lane,
        "difficulty": {case.axis_name: case.axis_value},
        # Ledger-difficulty cases materialize several real World orders from a
        # stable internal stem.  Never expose that non-existent stem as a
        # selectable business reference; the singular public reference is the
        # first real order and ``order_ids`` remains the complete authority.
        "order_id": case.order_ids[0],
        "order_ids": list(case.order_ids),
        "shipment_id": case.shipment_id,
        "original_sku_id": case.original_sku_id,
        "replacement_sku_ids": [row.sku_id for row in replacements],
        # A buyer sees what a buyer sees: the catalogue entry, already loaded.
        # The shop is a different matter -- it can look its own stock up, so it
        # gets the list of items and has to go and read them.  Handing the shop
        # a table of attributes and stock levels is what made "approve" a
        # one-glance answer.
        "replacement_candidates": [
            {
                "sku_id": row.sku_id,
                "available_qty": row.inventory,
                "product_facts": dict(row.product_facts),
            }
            for row in replacements
        ]
        if case.definition.evaluated_role == "buyer"
        else [],
        "replacement_requirements": dict(case.replacement_requirements),
        # The two things an after-sales desk has in front of it: the shop's own
        # published policy, and the request it has been handed.  Whether they
        # agree is the question the evaluated merchant has to answer, so both
        # sides are visible and neither states the answer.  Only the parts that
        # bear on this lane are published -- an empty "we accept nothing" list
        # would read as a reason to refuse everything.
        "return_policy": {"accepted_conditions": _accepted_return_conditions(case)},
        "pending_request": {
            key: value
            for key, value in (
                (
                    "inspected_condition",
                    case.attested_return_conditions[0]
                    if case.attested_return_conditions
                    else None,
                ),
                ("requested_replacement_sku_id", case.requested_replacement_sku_id),
            )
            if value
        },
        "readable_evidence_ids": evidence_ids,
        "evidence_records": evidence_records,
        # T7 measures execution of a published after-sales workflow.  The
        # evaluated model must not infer a scorer-private sequence from a lane
        # label, and the Agent must not supply it from hidden channel state.
        # Operations are business-policy vocabulary; routes and wire actions
        # remain exclusively Agent-owned.
        "workflow_policy": workflow_policy,
        "authority": {
            "after_sales": "actor -> platform:after-sales -> World",
            "commercial_state": "World only",
            "scoring": AFTER_SALES_EVIDENCE_CONTRACT,
        },
    }


def _public_task_context(case: _CaseT7) -> dict[str, Any]:
    """Expose public after-sales capabilities, never the required outcome.

    T7 can require several actor turns with the same Platform response kind.
    The public contract therefore exposes a lane-level set of legitimate
    lifecycle choices.  It deliberately includes policy-valid alternatives,
    such as approve and deny, so an incorrect model decision is scored as an
    incorrect decision instead of being misclassified as infrastructure
    failure.  No expected operation sequence or ruling is consulted here.
    """

    def route(action_kind: str, destination: str = "platform:after-sales") -> dict[str, str]:
        return {"action_kind": action_kind, "destination": destination}

    policy_reads = (
        route("commerce.read_after_sales_policy"),
        route("commerce.read_after_sales_history"),
    )
    evaluated_routes: dict[str, tuple[dict[str, str], ...]] = {
        "buyer_cancel": (
            route("commerce.cancel_paid_order"),
            route("commerce.read_payment_history"),
            route("commerce.read_packing_history"),
            *policy_reads,
        ),
        "buyer_return_refund": (
            route("commerce.request_return"),
            route("commerce.open_refund_case"),
            *policy_reads,
        ),
        "buyer_exchange": (
            route("commerce.request_return"),
            route("commerce.request_exchange"),
            *policy_reads,
        ),
        "buyer_dispute": (
            route("commerce.open_dispute"),
            route("commerce.submit_dispute_evidence"),
            route("commerce.open_refund_case"),
            *policy_reads,
        ),
        "merchant_cancel": (
            route("commerce.cancel_paid_order"),
            route("commerce.read_payment_history"),
            route("commerce.read_packing_history"),
            *policy_reads,
        ),
        "merchant_return_authorization": (
            route("commerce.authorize_return"),
            route("commerce.deny_return"),
            *policy_reads,
        ),
        "merchant_refund": (
            route("commerce.read_payment_history"),
            route("commerce.read_ledger_history"),
            route("commerce.read_packing_history"),
            route("commerce.read_shipment", "platform:fulfillment"),
            route("commerce.authorize_return"),
            route("commerce.deny_return"),
            route("commerce.receive_return"),
            route("commerce.approve_refund"),
            route("commerce.deny_refund"),
            *policy_reads,
        ),
        "merchant_exchange": (
            route("commerce.authorize_return"),
            route("commerce.deny_return"),
            route("commerce.receive_return"),
            route("commerce.authorize_exchange"),
            route("commerce.deny_exchange"),
            route("commerce.complete_exchange"),
            *policy_reads,
        ),
        "merchant_dispute": (
            route("commerce.respond_to_dispute"),
            route("commerce.submit_dispute_evidence"),
            route("commerce.approve_refund"),
            route("commerce.deny_refund"),
            *policy_reads,
        ),
        "merchant_ledger_close": (
            route("commerce.request_ledger_reconciliation"),
            route("commerce.read_ledger_history"),
            route("commerce.read_after_sales_history"),
        ),
    }
    counterpart_routes: dict[str, tuple[str, tuple[dict[str, str], ...]]] = {
        "buyer_return_refund": (
            "merchant",
            (
                route("commerce.authorize_return"),
                route("commerce.deny_return"),
                route("commerce.receive_return"),
                route("commerce.approve_refund"),
                route("commerce.deny_refund"),
                *policy_reads,
            ),
        ),
        "buyer_exchange": (
            "merchant",
            (
                route("commerce.authorize_return"),
                route("commerce.deny_return"),
                route("commerce.receive_return"),
                route("commerce.authorize_exchange"),
                route("commerce.deny_exchange"),
                route("commerce.complete_exchange"),
                *policy_reads,
            ),
        ),
        "buyer_dispute": (
            "merchant",
            (
                route("commerce.respond_to_dispute"),
                route("commerce.submit_dispute_evidence"),
                route("commerce.approve_refund"),
                route("commerce.deny_refund"),
                *policy_reads,
            ),
        ),
        "merchant_return_authorization": (
            "buyer",
            (route("commerce.request_return"), *policy_reads),
        ),
        "merchant_refund": (
            "buyer",
            (
                route("commerce.request_return"),
                route("commerce.open_refund_case"),
                *policy_reads,
            ),
        ),
        "merchant_exchange": (
            "buyer",
            (
                route("commerce.request_return"),
                route("commerce.request_exchange"),
                *policy_reads,
            ),
        ),
        "merchant_dispute": (
            "buyer",
            (
                route("commerce.open_dispute"),
                route("commerce.submit_dispute_evidence"),
                route("commerce.open_refund_case"),
                *policy_reads,
            ),
        ),
    }

    routes_by_role = {
        case.definition.evaluated_role: evaluated_routes[case.lane],
    }
    counterpart = counterpart_routes.get(case.lane)
    if counterpart is not None:
        counterpart_role, routes = counterpart
        routes_by_role[counterpart_role] = routes

    # Public after-sales acknowledgements authenticate the operation that just
    # committed.  That protocol state, together with the public capability
    # lane, is enough to distinguish a required continuation from a true
    # terminal.  These sets describe lifecycle grammar, not the scorer's ideal
    # operation sequence or ruling.
    operation_phases: dict[
        str,
        dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    ] = {
        "buyer_cancel": {
            "buyer": ((), ("cancel_paid_order",)),
        },
        "buyer_return_refund": {
            "buyer": (("request_return", "open_refund_case"), ()),
            "merchant": (
                ("authorize_return", "receive_return"),
                ("deny_return", "approve_refund", "deny_refund"),
            ),
        },
        "buyer_exchange": {
            "buyer": (("request_return", "request_exchange"), ()),
            "merchant": (
                ("authorize_return", "receive_return", "authorize_exchange"),
                ("deny_return", "deny_exchange", "complete_exchange"),
            ),
        },
        "buyer_dispute": {
            "buyer": (
                (
                    "open_dispute",
                    "submit_dispute_evidence",
                    "open_refund_case",
                ),
                (),
            ),
            "merchant": (
                ("respond_to_dispute", "submit_dispute_evidence"),
                ("approve_refund", "deny_refund"),
            ),
        },
        "merchant_cancel": {
            "merchant": ((), ("cancel_paid_order",)),
        },
        "merchant_return_authorization": {
            "buyer": (("request_return",), ()),
            "merchant": ((), ("authorize_return", "deny_return")),
        },
        "merchant_refund": {
            "buyer": (("request_return", "open_refund_case"), ()),
            "merchant": (
                ("authorize_return", "receive_return"),
                ("deny_return", "approve_refund", "deny_refund"),
            ),
        },
        "merchant_exchange": {
            "buyer": (("request_return", "request_exchange"), ()),
            "merchant": (
                ("authorize_return", "receive_return", "authorize_exchange"),
                ("deny_return", "deny_exchange", "complete_exchange"),
            ),
        },
        "merchant_dispute": {
            "buyer": (
                (
                    "open_dispute",
                    "submit_dispute_evidence",
                    "open_refund_case",
                ),
                (),
            ),
            "merchant": (
                ("respond_to_dispute", "submit_dispute_evidence"),
                ("approve_refund", "deny_refund"),
            ),
        },
        # Ledger acknowledgements additionally need their public order id to
        # distinguish another required request from the final committed one.
        "merchant_ledger_close": {"merchant": ((), ())},
    }

    # Participant hand-offs are Agent protocol work.  Each listed operation
    # must first be accepted by Platform; its exact acknowledgement then binds
    # one deterministic message to the order's authenticated counterparty.
    # Dispute evidence is the sole multi-write gate: the model may submit the
    # public evidence references in any order, and the hand-off occurs only
    # after every required reference has an accepted acknowledgement.
    automatic_handoffs: dict[str, dict[str, frozenset[str]]] = {
        "buyer_return_refund": {
            "buyer": frozenset({"request_return", "open_refund_case"}),
            "merchant": frozenset({"receive_return"}),
        },
        "merchant_return_authorization": {
            "buyer": frozenset({"request_return"}),
            "merchant": frozenset(),
        },
        "merchant_refund": {
            "buyer": frozenset({"request_return", "open_refund_case"}),
            "merchant": frozenset({"receive_return"}),
        },
        "buyer_exchange": {
            "buyer": frozenset({"request_return", "request_exchange"}),
            "merchant": frozenset({"receive_return"}),
        },
        "merchant_exchange": {
            "buyer": frozenset({"request_return", "request_exchange"}),
            "merchant": frozenset({"receive_return"}),
        },
        "buyer_dispute": {
            "buyer": frozenset({"submit_dispute_evidence", "open_refund_case"}),
            "merchant": frozenset({"respond_to_dispute"}),
        },
        "merchant_dispute": {
            "buyer": frozenset({"submit_dispute_evidence", "open_refund_case"}),
            "merchant": frozenset({"respond_to_dispute"}),
        },
    }

    phases: list[dict[str, Any]] = []
    for role, routes in routes_by_role.items():
        sender_role = "merchant" if role == "buyer" else "buyer"
        phases.append(
            {
                "phase_id": f"{role}_after_sales_message",
                "match": {
                    "actor_roles": [role],
                    "inbound_action_kinds": ["commerce.send_message"],
                    "inbound_sender_roles": [sender_role],
                },
                "allowed_routes": list(routes),
                "world_reads": "skill_scoped",
                "finish": "allow_wait",
            }
        )
        phases.append(
            {
                "phase_id": f"{role}_after_sales_snapshot",
                "match": {
                    "actor_roles": [role],
                    "inbound_action_kinds": ["platform.after_sales_snapshot"],
                    "inbound_sender_roles": ["platform"],
                },
                "allowed_routes": list(routes),
                "world_reads": "skill_scoped",
                # A read is a legal information-gathering choice.  Stopping
                # after it is a scoreable model omission; deterministic peers
                # must likewise be able to wait when their next write is not
                # yet authorized by the observed World state.
                "finish": "allow_wait",
            }
        )
        actionable, terminal = operation_phases[case.lane][role]
        for operation in actionable:
            automatic = operation in automatic_handoffs.get(case.lane, {}).get(
                role,
                frozenset(),
            )
            gated = automatic and operation == "submit_dispute_evidence"
            continuation: dict[str, Any] | None = None
            if automatic:
                continuation = {
                    "action_kind": "commerce.send_message",
                    "destination": "@bound_counterparty",
                    "payload": {"category": "after_sales"},
                }
                if gated:
                    continuation["accepted_reference_gate"] = {
                        "reference_field": "evidence_id",
                        "required_values": list(_verified_evidence_ids(case, "filer")),
                    }
            phase: dict[str, Any] = {
                "phase_id": f"{role}_{operation}_progress",
                "match": {
                    "actor_roles": [role],
                    "inbound_action_kinds": ["platform.after_sales_updated"],
                    "inbound_sender_roles": ["platform"],
                    "payload_equals": {"operation": operation},
                },
                "allowed_routes": list(routes) if not automatic or gated else [],
                "world_reads": "skill_scoped" if not automatic or gated else "deny",
                "finish": "allow_wait" if not automatic or gated else "framework_continue",
            }
            if continuation is not None:
                phase["framework_continuation"] = continuation
            phases.append(
                phase
            )
        for operation in terminal:
            phases.append(
                {
                    "phase_id": f"{role}_{operation}_terminal",
                    "match": {
                        "actor_roles": [role],
                        "inbound_action_kinds": ["platform.after_sales_updated"],
                        "inbound_sender_roles": ["platform"],
                        "payload_equals": {"operation": operation},
                    },
                    "allowed_routes": [],
                    "world_reads": "deny",
                    "finish": "framework_terminal",
                }
            )
        if any(row["action_kind"] == "commerce.read_shipment" for row in routes):
            phases.append(
                {
                    "phase_id": f"{role}_delivered_shipment_progress",
                    "match": {
                        "actor_roles": [role],
                        "inbound_action_kinds": ["platform.shipment_state"],
                        "inbound_sender_roles": ["platform"],
                        "payload_equals": {"shipment": {"status": "delivered"}},
                    },
                    "allowed_routes": list(routes),
                    "world_reads": "skill_scoped",
                    "finish": "forbid",
                }
            )

    if case.lane == "merchant_ledger_close":
        merchant_routes = list(routes_by_role["merchant"])
        for index, order_id in enumerate(case.order_ids):
            terminal = index == len(case.order_ids) - 1
            phases.append(
                {
                    "phase_id": (
                        "merchant_ledger_reconciliation_terminal"
                        if terminal
                        else f"merchant_ledger_reconciliation_{index + 1}_progress"
                    ),
                    "match": {
                        "actor_roles": ["merchant"],
                        "inbound_action_kinds": ["platform.after_sales_updated"],
                        "inbound_sender_roles": ["platform"],
                        "payload_equals": {
                            "operation": "request_ledger_reconciliation",
                            "order_id": order_id,
                        },
                    },
                    "allowed_routes": [] if terminal else merchant_routes,
                    "world_reads": "deny" if terminal else "skill_scoped",
                    "finish": "framework_terminal" if terminal else "forbid",
                }
            )

    return {
        "schema_version": T7_RUNTIME_SCHEMA_V2,
        "task_id": case.definition.task_id,
        "capability": case.lane,
        "evaluated_role": case.definition.evaluated_role,
        "execution_contract": {
            "schema_version": "cwe.public-task-execution.v1",
            "phases": phases,
        },
    }


def _scenario_evidence_rows(case: _CaseT7) -> list[dict[str, Any]]:
    rows = []
    acl = (
        "platform:after-sales",
        "platform:adjudicator",
        _BUYER_ID,
        _MERCHANT_ID,
    )
    if case.axis_name == "return_condition_count":
        # Several inspections of the same item: each looks at something
        # different and they agree on what state it came back in.
        condition = case.attested_return_conditions[0]
        scopes = ("item identity", "item condition", "returned quantity")
        for index in range(1, int(case.axis_value) + 1):
            record = build_evidence_record(
                record_id=f"evidence:{_slug(case.definition)}:return:{index}",
                kind="inspection_report",
                subject_id=case.order_id,
                issuer_id="inspector:trusted",
                facts={
                    "condition": condition,
                    "inspected": scopes[index - 1],
                    "inspection_reference": (f"inspection:{_slug(case.definition)}:{index}"),
                },
                trust={
                    "verification_method": "authenticated_inspection",
                    "verified": True,
                },
                version=1,
                owner_id=_BUYER_ID,
                read_acl=acl,
                issued_at_tick=2,
            )
            rows.append(evidence_record_to_dict(record))
    elif case.axis_name == "evidence_item_count":
        filer_count, respondent_count = case.dispute_evidence_counts
        for side, count, owner_id in (
            ("filer", filer_count, _BUYER_ID),
            ("respondent", respondent_count, _MERCHANT_ID),
        ):
            for index in range(1, count + 1):
                record = build_evidence_record(
                    record_id=(f"evidence:{_slug(case.definition)}:{side}:{index}"),
                    kind="carrier_scan",
                    subject_id=case.order_id,
                    issuer_id="carrier:trusted",
                    facts={
                        "side": side,
                        "scan_code": "delivery_exception",
                        "carrier_reference": (f"carrier:{_slug(case.definition)}:{side}:{index}"),
                    },
                    trust={
                        "verification_method": "carrier_api",
                        "verified": True,
                    },
                    version=1,
                    owner_id=owner_id,
                    read_acl=acl,
                    issued_at_tick=2,
                )
                rows.append(evidence_record_to_dict(record))
            # One record on the same order, owned by the same party, that was
            # never actually verified.  World will accept it, so nothing stops
            # a model from filing it; the only reason not to is that it does
            # not stand up.  Without a row like this the difficulty axis has
            # nothing to filter -- World already refuses evidence from another
            # order or from the other party, so every remaining record was
            # worth filing by construction.
            record = build_evidence_record(
                record_id=f"evidence:{_slug(case.definition)}:{side}:unverified",
                kind="customer_photo",
                subject_id=case.order_id,
                issuer_id=owner_id,
                facts={
                    "side": side,
                    "scan_code": "self_reported_condition",
                    "carrier_reference": "",
                },
                trust={
                    "verification_method": "self_declared",
                    "verified": False,
                },
                version=1,
                owner_id=owner_id,
                read_acl=acl,
                issued_at_tick=2,
            )
            rows.append(evidence_record_to_dict(record))
    return rows


def _accepted_return_conditions(case: _CaseT7) -> list[str]:
    """Return the conditions the shop takes back, as World enforces them."""

    return list(case.allowed_return_conditions or ("new", "opened", "damaged"))


def _policy(case: _CaseT7) -> dict[str, Any]:
    allowed_conditions = tuple(_accepted_return_conditions(case))
    return {
        "policy_id": f"policy:{_slug(case.definition)}",
        "return_window_ticks": 30,
        "max_refund_bps": 10_000,
        "split_refund_bps": 5_000,
        "owner_paid_cancel_allowed": True,
        "merchant_paid_cancel_allowed": True,
        "allowed_return_conditions": list(allowed_conditions),
        "return_authorizer_ids": [_MERCHANT_ID],
        "return_receiver_ids": [_MERCHANT_ID],
        "refund_decider_ids": [_MERCHANT_ID],
        "exchange_authorizer_ids": [_MERCHANT_ID],
        "adjudicator_ids": ["platform:adjudicator"],
        "evidence_service_ids": ["platform:evidence"],
        "ledger_requester_ids": [_BUYER_ID, _MERCHANT_ID],
        "ledger_reconciler_ids": ["platform:accounting"],
    }


def _ledger_rows(case: _CaseT7) -> list[dict[str, Any]]:
    if case.initial_payment_stage == "authorized":
        return []
    return [
        {
            "txn_id": f"txn:{_slug(case.definition)}:charge:{index}",
            "order_id": order_id,
            "buyer_id": _BUYER_ID,
            "merchant_id": _MERCHANT_ID,
            "sku_id": case.original_sku_id,
            "qty": 1,
            "price": "115.00",
            "currency": "USD",
            "idempotency_key": f"seed:{_slug(case.definition)}:charge:{index}",
            "effect": "charge",
        }
        for index, order_id in enumerate(case.order_ids, start=1)
    ]


def _prepared_scenario_for_t7(task_id: str) -> ScenarioSpec:
    """Build a World-valid T7 scenario without declaring formal readiness.

    The public ``scenario_for_t7`` stays fail closed until the VCP and HTTP
    gates pass.  This private builder lets integration tests validate every
    task's authoritative setup while those shared routes are being completed.
    """

    case = _case_for_t7(task_id)
    authorized = case.initial_payment_stage == "authorized"
    order_state = (
        "accepted"
        if authorized
        else (
            "settled"
            if case.axis_name in {"order_stage", "affected_ledger_count"}
            else "dispatched"
        )
    )
    catalog = []
    for product in case.products:
        catalog.append(
            {
                **product.to_dict(),
                "category": "benchmark-t7",
                "qty_reserved": (
                    0
                    if authorized or product.sku_id != case.original_sku_id
                    else len(case.order_ids)
                ),
                "attributes": dict(product.product_facts),
            }
        )
        catalog[-1].pop("product_facts")

    initial_state: dict[str, Any] = {
        "catalog": catalog,
        "orders": [
            {
                "order_id": order_id,
                "buyer_id": _BUYER_ID,
                "merchant_id": _MERCHANT_ID,
                "sku_id": case.original_sku_id,
                "qty": 1,
                "agreed_price": "115.00",
                "currency": "USD",
                "state": order_state,
            }
            for order_id in case.order_ids
        ],
        "ledger": _ledger_rows(case),
        "logical_time": 3,
        "evidence_records": _scenario_evidence_rows(case),
        "after_sales_setup": {
            "policies": [
                {
                    "merchant_id": _MERCHANT_ID,
                    "idempotency_key": f"setup:{_slug(case.definition)}:policy",
                    "intent": _policy(case),
                }
            ],
            "payment_transitions": [
                {
                    "idempotency_key": (f"setup:{_slug(case.definition)}:payment:{index}"),
                    "intent": {
                        "op": "authorize" if authorized else "capture",
                        "order_id": order_id,
                    },
                }
                for index, order_id in enumerate(case.order_ids, start=1)
            ],
            "packing_transitions": (
                [
                    {
                        "idempotency_key": f"setup:{_slug(case.definition)}:packing:create",
                        "intent": {"op": "create", "order_id": case.order_id},
                    },
                    {
                        "idempotency_key": f"setup:{_slug(case.definition)}:packing:pack",
                        "intent": {"op": "pack", "order_id": case.order_id},
                    },
                    *(
                        [
                            {
                                "idempotency_key": (
                                    f"setup:{_slug(case.definition)}:packing:handoff"
                                ),
                                "intent": {
                                    "op": "hand_off",
                                    "order_id": case.order_id,
                                },
                            }
                        ]
                        if case.initial_packing_stage == "handed_off"
                        else []
                    ),
                ]
                if case.initial_packing_stage in {"packed", "handed_off"}
                else []
            ),
        },
    }
    if order_state == "dispatched":
        initial_state["order_timelines"] = [
            {
                "order_id": case.order_id,
                "buyer_id": _BUYER_ID,
                "merchant_id": _MERCHANT_ID,
                "settled_at_tick": 0,
                "dispatched_at_tick": 1,
                "return_window_ticks": None,
                "return_authorized_at_tick": None,
                "returned_at_tick": None,
                "refunded_at_tick": None,
            }
        ]
        initial_state["shipments"] = [
            {
                "shipment_id": f"shipment:{_slug(case.definition)}",
                "order_id": case.order_id,
                "buyer_id": _BUYER_ID,
                "merchant_id": _MERCHANT_ID,
                "original_sku_id": case.original_sku_id,
                "status": "delivered",
                "status_history": [
                    {
                        "event_id": f"shipment-event:{_slug(case.definition)}",
                        "status": "delivered",
                        "logical_time": 2,
                    }
                ],
                "version": 1,
            }
        ]

    contract = _public_contract(case)
    task_context = _public_task_context(case)
    # Buyer-evaluated tasks begin with an authenticated merchant directive.
    # Merchant-evaluated tasks begin by directing the scripted buyer, whose
    # first accepted business operation is then handed to the merchant by the
    # Agent-owned bound-counterparty continuation.  The scenario root is
    # environment setup; it is never represented as a model-authored action.
    if case.lane in {"merchant_cancel", "merchant_ledger_close"}:
        kickoff_sender = _BUYER_ID
        kickoff_recipient = _MERCHANT_ID
    else:
        kickoff_sender = _MERCHANT_ID
        kickoff_recipient = _BUYER_ID
    population = PopulationSpec(
        buyers=(
            BuyerSpec(
                _BUYER_ID,
                {"name": "T7 benchmark buyer", "task_family": "T7"},
                {
                    "mandate_id": f"mandate:{case.definition.task_id}",
                    "goal": _INSTRUCTIONS[case.lane],
                    "quantity": 1,
                    "hard_constraints": {
                        "budget": 11_500,
                        "must_have": dict(case.replacement_requirements),
                    },
                    "soft_constraints": [],
                    "soft_preferences": {"style": [], "avoid": []},
                    "authority": {
                        "can_buy_without_confirmation": True,
                        "must_not_share_with_merchant": ["budget"],
                    },
                    "intent_expiry": "2099-01-01T00:00:00Z",
                    "task_context": task_context,
                    "benchmark_contract": contract,
                },
            ),
        ),
        merchants=(
            MerchantSpec(
                _MERCHANT_ID,
                {"name": "T7 benchmark merchant", "task_family": "T7"},
                {
                    "floor_price": 10_000,
                    "task_context": task_context,
                    "benchmark_contract": contract,
                },
                tuple(row.sku_id for row in case.products),
            ),
        ),
        initial_events=(
            {
                "from": kickoff_sender,
                "to": kickoff_recipient,
                "idempotency_key": f"kickoff:{_slug(case.definition)}",
                "action": {
                    "kind": "commerce.send_message",
                    "payload": {
                        "category": "after_sales",
                        "task_id": case.definition.task_id,
                        "order_id": case.order_ids[0],
                        "order_ids": list(case.order_ids),
                        "instruction": _INSTRUCTIONS[case.lane],
                        "benchmark_contract": contract,
                    },
                },
            },
        ),
        matching={"top_k": len(case.products)},
        execution={"max_transactions_per_buyer": 1},
    )
    return ScenarioSpec(
        scenario_id=f"{_slug(case.definition)}__runtime",
        seed=int(case.definition.canonical_hash[:8], 16) % 2_147_483_646 + 1,
        initial_state=initial_state,
        buyer_goal={},
        merchant_policy={},
        allowed_actions=(),
        success_oracle={
            "schema_version": T7_RUNTIME_SCHEMA_V2,
            "task_id": case.definition.task_id,
            "lane": case.lane,
        },
        population=population,
    )


def _business_request(user_prompt: str) -> dict[str, Any]:
    """Decode the exact provider-neutral request emitted by our Agent."""

    start = user_prompt.find("{")
    if start < 0:
        raise ValueError("Agent business prompt contains no request object")
    value = json.loads(user_prompt[start:])
    if not isinstance(value, dict):
        raise ValueError("Agent business request must be an object")
    return value


class _T7IntentUnavailable(ValueError):
    """The next reviewed business choice is absent from current authority."""


def _allowed_intent(
    request: Mapping[str, Any],
    intent: str,
) -> Mapping[str, Any]:
    rows = request.get("allowed_intents")
    if not isinstance(rows, list):
        raise ValueError("Agent business request has no allowed intents")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("intent") == intent]
    if not matches:
        raise _T7IntentUnavailable(
            f"business intent {intent!r} is not currently available"
        )
    if len(matches) != 1:
        raise ValueError(f"business intent {intent!r} is not uniquely available")
    return matches[0]


def _intent_properties(
    request: Mapping[str, Any],
    intent: str,
) -> Mapping[str, Any]:
    parameters = _allowed_intent(request, intent).get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    if not isinstance(properties, Mapping):
        raise ValueError(f"business intent {intent!r} has no property schema")
    return properties


def _enum_values(
    request: Mapping[str, Any],
    intent: str,
    field: str,
    *,
    array: bool = False,
) -> tuple[str, ...]:
    schema = _intent_properties(request, intent).get(field)
    if not isinstance(schema, Mapping):
        return ()
    source = schema.get("items") if array else schema
    raw = source.get("enum") if isinstance(source, Mapping) else None
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(value, str) or not value for value in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValueError(f"{intent}.{field} has no finite public reference enum")
    return tuple(raw)


def _walk_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        rows.append(value)
        for item in value.values():
            rows.extend(_walk_mappings(item))
    elif isinstance(value, list):
        for item in value:
            rows.extend(_walk_mappings(item))
    return tuple(rows)


def _filed_evidence_refs(request: Mapping[str, Any], side: str) -> tuple[str, ...]:
    """Return this side's verified records from the published evidence list."""

    return tuple(
        reference
        for row in _walk_mappings(request.get("observations"))
        for record in (row.get("evidence_records") if isinstance(row.get("evidence_records"), list) else ())
        if isinstance(record, Mapping)
        and record.get("side") == side
        and record.get("verified") is True
        and isinstance(reference := record.get("evidence_ref"), str)
        and reference
    )


def _replacement_ref(request: Mapping[str, Any]) -> str:
    """Choose the public replacement ref from visible facts and requirements."""

    allowed = _enum_values(request, "request_exchange", "replacement_sku_ref")
    requirements: Mapping[str, Any] | None = None
    candidates: list[Mapping[str, Any]] = []
    for row in _walk_mappings(request.get("observations")):
        raw_requirements = row.get("replacement_requirements")
        if isinstance(raw_requirements, Mapping):
            requirements = raw_requirements
        raw_candidates = row.get("replacement_candidates")
        if isinstance(raw_candidates, list):
            candidates.extend(item for item in raw_candidates if isinstance(item, Mapping))
    if requirements is None:
        raise ValueError("business observations omit replacement requirements")
    matching = []
    for candidate in candidates:
        reference = candidate.get("sku_ref")
        facts = candidate.get("product_facts")
        available = candidate.get("available_qty")
        if (
            isinstance(reference, str)
            and reference in allowed
            and isinstance(facts, Mapping)
            and isinstance(available, int)
            and not isinstance(available, bool)
            and available > 0
            and all(facts.get(key) == value for key, value in requirements.items())
        ):
            matching.append(reference)
    if len(set(matching)) != 1:
        raise ValueError("visible business facts do not identify one replacement")
    return matching[0]


_T7Decision = Callable[[Mapping[str, Any]], tuple[str, Mapping[str, Any]]]


def _choice(
    intent: str,
    arguments: Mapping[str, Any] | Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> _T7Decision:
    def decide(request: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
        _allowed_intent(request, intent)
        resolved = arguments(request) if callable(arguments) else arguments
        return intent, copy.deepcopy(dict(resolved))

    return decide


def _business_response(intent: str, arguments: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "schema_version": LLM_BUSINESS_DECISION_SCHEMA_V1,
            "intent": intent,
            "arguments": dict(arguments),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class _ScriptedT7Channel:
    """Reviewed T7 policy over the typed provider-neutral business seam."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def __init__(
        self,
        decisions: tuple[_T7Decision, ...],
        *,
        actor_role: str,
        wait_on_unavailable: bool = False,
    ) -> None:
        if actor_role not in {"buyer", "merchant"}:
            raise ValueError("T7 business channel actor role is invalid")
        if not isinstance(wait_on_unavailable, bool):
            raise ValueError("T7 wait-on-unavailable policy must be boolean")
        self._decisions = decisions
        self._actor_role = actor_role
        self._wait_on_unavailable = wait_on_unavailable
        self._cursor = 0

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("T7 business channel requires Agent decision evidence")
        request = _business_request(user_prompt)
        if request.get("role") != self._actor_role:
            raise ValueError("T7 business request crossed actor roles")
        if self._cursor < len(self._decisions):
            try:
                intent, arguments = self._decisions[self._cursor](request)
            except _T7IntentUnavailable:
                if not self._wait_on_unavailable:
                    raise
                _allowed_intent(request, "finish")
                intent = "finish"
                arguments = {
                    "reason": "No authoritative counterpart action is currently available."
                }
            else:
                self._cursor += 1
        else:
            _allowed_intent(request, "finish")
            intent = "finish"
            arguments = {"reason": "No further business action was selected."}
        content = _business_response(intent, arguments)
        return BusinessDecisionResponseV1(
            content=content,
            response_chars=len(content),
            response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )


def _evidence_ids(case: _CaseT7, side: str | None = None) -> tuple[str, ...]:
    """Return every readable record, including the ones not worth filing."""

    rows = _scenario_evidence_rows(case)
    return tuple(
        row["record_id"] for row in rows if side is None or row["facts"].get("side") == side
    )


def _verified_evidence_ids(case: _CaseT7, side: str | None = None) -> tuple[str, ...]:
    """Return the records that actually stand up: the ones to file."""

    rows = _scenario_evidence_rows(case)
    return tuple(
        row["record_id"]
        for row in rows
        if (side is None or row["facts"].get("side") == side)
        and row.get("trust", {}).get("verified") is True
    )


def _expected_replacement_sku(case: _CaseT7) -> str:
    """Return the scorer-only expected SKU; model policies never call this."""

    required = dict(case.replacement_requirements)
    matches = [
        product.sku_id
        for product in case.products
        if product.sku_id != case.original_sku_id
        and all(dict(product.product_facts).get(key) == value for key, value in required.items())
        and product.inventory > 0
    ]
    if len(matches) != 1:
        raise ValueError(f"{case.definition.task_id}: replacement is not unique")
    return matches[0]


def _return_condition(case: _CaseT7) -> str:
    """Return the condition to log on receipt: what the inspections attest."""

    return case.attested_return_conditions[0] if case.attested_return_conditions else "new"


def _buyer_request_return(_case: _CaseT7) -> _T7Decision:
    def arguments(request: Mapping[str, Any]) -> Mapping[str, Any]:
        values: dict[str, Any] = {
            "requested_qty": 1,
            "reason": "Request a policy-covered return.",
        }
        evidence = _enum_values(
            request,
            "request_return",
            "evidence_refs",
            array=True,
        )
        if evidence:
            values["evidence_refs"] = list(evidence)
        return values

    return _choice("request_return", arguments)


def _merchant_authorize_return(_case: _CaseT7) -> _T7Decision:
    return _choice(
        "decide_return",
        {
            "decision": "approve",
            "reason": "Trusted evidence satisfies the return policy.",
        },
    )


def _merchant_deny_return(_case: _CaseT7) -> _T7Decision:
    return _choice(
        "decide_return",
        {
            "decision": "deny",
            "reason": "The merchant declines the requested return.",
        },
    )


def _merchant_receive_return(case: _CaseT7) -> _T7Decision:
    return _choice(
        "receive_return",
        {
            "received_qty": 1,
            "condition": _return_condition(case),
        },
    )


def _buyer_open_refund(_case: _CaseT7) -> _T7Decision:
    return _choice(
        "open_refund_case",
        {"reason": "The authoritative return was received."},
    )


def _merchant_approve_refund(_case: _CaseT7) -> _T7Decision:
    return _choice(
        "decide_refund",
        {
            "decision": "approve",
            "reason": "Causal records support the exact refund.",
        },
    )


def _merchant_deny_refund(_case: _CaseT7) -> _T7Decision:
    return _choice(
        "decide_refund",
        {
            "decision": "deny",
            "reason": "The authoritative ruling supports no refund.",
        },
    )


def _merchant_resolve_dispute_refund(case: _CaseT7) -> _T7Decision:
    if case.expected_ruling_operation == "rule_for_respondent":
        return _merchant_deny_refund(case)
    return _merchant_approve_refund(case)


def _merchant_resolve_observed_dispute_refund(_case: _CaseT7) -> _T7Decision:
    """Resolve the reference refund from a public ruling read, never the oracle."""

    def arguments(request: Mapping[str, Any]) -> Mapping[str, Any]:
        ruling_values = [
            row.get("value")
            for row in _walk_mappings(request.get("observations"))
            if row.get("table") == "after_sales_rulings"
            and isinstance(row.get("value"), Mapping)
        ]
        if len(ruling_values) != 1:
            raise ValueError(
                "reference refund policy requires one public authoritative ruling"
            )
        refund_amount = ruling_values[0].get("refund_amount")
        if (
            isinstance(refund_amount, bool)
            or not isinstance(refund_amount, int)
            or refund_amount < 0
        ):
            raise ValueError("public authoritative ruling has an invalid refund amount")
        decision = "approve" if refund_amount > 0 else "deny"
        if decision not in _enum_values(request, "decide_refund", "decision"):
            raise ValueError("authoritative dispute resolution is not currently available")
        return {
            "decision": decision,
            "reason": "Follow the public authoritative dispute ruling.",
        }

    return _choice("decide_refund", arguments)


def _buyer_request_exchange(case: _CaseT7) -> _T7Decision:
    named = case.requested_replacement_sku_id
    if named is None:
        return _choice(
            "request_exchange",
            lambda request: {
                "replacement_sku_ref": _replacement_ref(request),
                "reason": "The replacement satisfies every stated requirement.",
            },
        )

    # On a merchant-evaluated exchange the buyer is the environment, and which
    # item they ask for is a fixture fact rather than a judgement: it is what
    # the evaluated merchant then has to rule on.
    reference = public_reference_alias_v1(named)

    def arguments(request: Mapping[str, Any]) -> Mapping[str, Any]:
        if reference not in _enum_values(request, "request_exchange", "replacement_sku_ref"):
            raise ValueError("the requested replacement is not currently offerable")
        return {
            "replacement_sku_ref": reference,
            "reason": "This is the replacement the buyer wants.",
        }

    return _choice("request_exchange", arguments)


def _merchant_authorize_exchange(_case: _CaseT7) -> _T7Decision:
    return _choice(
        "decide_exchange",
        {
            "decision": "approve",
            "reason": "Grounded replacement inventory is available.",
        },
    )


def _merchant_deny_exchange(_case: _CaseT7) -> _T7Decision:
    return _choice(
        "decide_exchange",
        {
            "decision": "deny",
            "reason": "The requested replacement does not satisfy the stated requirements.",
        },
    )


def _merchant_complete_exchange(_case: _CaseT7) -> _T7Decision:
    return _choice(
        "complete_exchange",
        {"reason": "Commit the authorized replacement."},
    )


def _open_dispute(_case: _CaseT7) -> _T7Decision:
    return _choice(
        "open_dispute",
        {"reason": "Trusted delivery evidence conflicts."},
    )


def _submit_next_dispute_evidence(_case: _CaseT7) -> _T7Decision:
    def arguments(request: Mapping[str, Any]) -> Mapping[str, Any]:
        # Still submittable and worth submitting.  A record nobody verified
        # stays behind, which is the point of the axis.
        offerable = set(
            _enum_values(request, "submit_dispute_evidence", "evidence_ref")
        )
        filed = [
            reference
            for reference in _filed_evidence_refs(request, "filer")
            if reference in offerable
        ]
        if not filed:
            raise ValueError("no verified filer record remains to submit")
        return {"evidence_ref": filed[0]}

    return _choice("submit_dispute_evidence", arguments)


def _respond_to_dispute(_case: _CaseT7) -> _T7Decision:
    def arguments(request: Mapping[str, Any]) -> Mapping[str, Any]:
        values: dict[str, Any] = {"position": "contest"}
        allowed = _enum_values(
            request,
            "respond_to_dispute",
            "evidence_refs",
            array=True,
        )
        respondent_refs = _filed_evidence_refs(request, "respondent")
        selected = tuple(value for value in allowed if value in respondent_refs)
        if selected:
            values["evidence_refs"] = list(selected)
        return values

    return _choice("respond_to_dispute", arguments)


def _read_current(intent: str, reference_field: str | None = None) -> _T7Decision:
    def arguments(request: Mapping[str, Any]) -> Mapping[str, Any]:
        field = reference_field
        if field is None and intent in {
            "read_payment_history",
            "read_ledger_history",
            "read_packing_history",
            "read_after_sales_history",
        }:
            field = "order_ref"
        if field is None:
            return {}
        references = _enum_values(request, intent, field)
        return {field: references[0]} if references else {}

    return _choice(intent, arguments)


def _read_current_reference(
    intent: str,
    field: str,
    reference: str,
) -> _T7Decision:
    """Read one named business reference rather than whatever comes first."""

    def arguments(request: Mapping[str, Any]) -> Mapping[str, Any]:
        if reference not in _enum_values(request, intent, field):
            raise ValueError(f"{intent} cannot currently read {field}")
        values: dict[str, Any] = {field: reference}
        if intent == "observe_stock_availability":
            values["qty"] = 1
        return values

    return _choice(intent, arguments)


def _read_current_when_available(
    intent: str,
    *,
    required_intent: str,
) -> _T7Decision:
    """Read only after the next deterministic write has public authority."""

    read = _read_current(intent)

    def decide(request: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
        _allowed_intent(request, required_intent)
        return read(request)

    return decide


def _cancel_paid_order(_case: _CaseT7) -> _T7Decision:
    return _choice("cancel_paid_order", {"reason": "Cancel before dispatch."})


def _request_ledger_reconciliation(_case: _CaseT7) -> _T7Decision:
    return _choice(
        "request_ledger_reconciliation",
        {"reason": "Reconcile the exact order ledger."},
    )


def _evaluated_steps(case: _CaseT7, *, mutated: bool) -> tuple[_T7Decision, ...]:
    lane = case.lane
    if lane in {"buyer_cancel", "merchant_cancel"}:
        if mutated:
            return (_read_current("read_payment_history"),)
        return (_cancel_paid_order(case),)
    if lane == "buyer_return_refund":
        decisions = (_buyer_request_return(case), _buyer_open_refund(case))
        return decisions[:1] if mutated else decisions
    if lane == "merchant_return_authorization":
        if mutated:
            # The mutation is always the wrong call on this request, whichever
            # way the right one goes.
            return (
                (_merchant_deny_return(case),)
                if case.entitled
                else (_merchant_authorize_return(case),)
            )
        return (
            (_merchant_authorize_return(case),)
            if case.entitled
            else (_merchant_deny_return(case),)
        )
    if lane == "merchant_refund":
        reads: list[_T7Decision] = [
            _read_current("read_payment_history"),
            _read_current("read_ledger_history"),
        ]
        if case.axis_value == 4:
            reads.extend(
                (
                    _read_current("read_packing_history"),
                    _read_current("read_shipment", "shipment_ref"),
                )
            )
        decisions = (
            *reads,
            _merchant_authorize_return(case),
            _merchant_receive_return(case),
            _merchant_approve_refund(case),
        )
        return decisions[:-1] if mutated else decisions
    if lane == "buyer_exchange":
        decisions = (_buyer_request_return(case), _buyer_request_exchange(case))
        return decisions[:1] if mutated else decisions
    if lane == "merchant_exchange":
        requested = public_reference_alias_v1(str(case.requested_replacement_sku_id))
        opening = (
            _read_current_reference("observe_listing", "sku_ref", requested),
            _read_current_reference("observe_stock_availability", "sku_ref", requested),
            _merchant_authorize_return(case),
            _merchant_receive_return(case),
        )
        if not case.entitled:
            # The replacement the buyer named breaks a stated requirement.  The
            # mutation is the rubber stamp: approve it anyway.
            return (
                (*opening, _merchant_authorize_exchange(case))
                if mutated
                else (*opening, _merchant_deny_exchange(case))
            )
        decisions = (
            *opening,
            _merchant_authorize_exchange(case),
            _merchant_complete_exchange(case),
        )
        return decisions[:-1] if mutated else decisions
    if lane == "buyer_dispute":
        decisions: list[_T7Decision] = [_open_dispute(case)]
        decisions.extend(_submit_next_dispute_evidence(case) for _ in _verified_evidence_ids(case, "filer"))
        decisions.extend(
            (
                _buyer_open_refund(case),
            )
        )
        return tuple(decisions[:-1] if mutated else decisions)
    if lane == "merchant_dispute":
        decisions = (
            _respond_to_dispute(case),
            _merchant_resolve_dispute_refund(case),
        )
        return decisions[:-1] if mutated else decisions
    if lane == "merchant_ledger_close":
        decisions = tuple(_request_ledger_reconciliation(case) for _ in case.order_ids)
        if mutated:
            # Select a real, provider-visible business alternative instead of
            # exhausting the scripted channel while the next reconciliation
            # phase forbids ``finish``.  The history read is allowed by the
            # public phase contract; stopping from its snapshot is therefore
            # a scoreable model omission with one reconciliation still
            # outstanding, not a harness-only ``_T7IntentUnavailable``.
            return (*decisions[:-1], _read_current("read_after_sales_history"))
        return decisions
    raise ValueError(f"unsupported T7 evaluated lane {lane!r}")


def _counterpart_steps(case: _CaseT7) -> tuple[_T7Decision, ...]:
    lane = case.lane
    if lane in {"buyer_cancel", "merchant_cancel", "merchant_ledger_close"}:
        return ()
    if lane == "buyer_return_refund":
        return (
            _merchant_authorize_return(case),
            _merchant_receive_return(case),
            _merchant_approve_refund(case),
        )
    if lane in {"merchant_return_authorization", "merchant_refund"}:
        decisions: list[_T7Decision] = [
            _buyer_request_return(case),
        ]
        if lane == "merchant_refund":
            decisions.append(_buyer_open_refund(case))
        return tuple(decisions)
    if lane == "buyer_exchange":
        return (
            _merchant_authorize_return(case),
            _merchant_receive_return(case),
            _merchant_authorize_exchange(case),
            _merchant_complete_exchange(case),
        )
    if lane == "merchant_exchange":
        return (
            _buyer_request_return(case),
            _buyer_request_exchange(case),
        )
    if lane == "buyer_dispute":
        return (
            _respond_to_dispute(case),
            _read_current_when_available(
                "read_after_sales_history",
                required_intent="decide_refund",
            ),
            _merchant_resolve_observed_dispute_refund(case),
        )
    if lane == "merchant_dispute":
        decisions: list[_T7Decision] = [_open_dispute(case)]
        decisions.extend(_submit_next_dispute_evidence(case) for _ in _verified_evidence_ids(case, "filer"))
        decisions.extend(
            (
                _buyer_open_refund(case),
            )
        )
        return tuple(decisions)
    raise ValueError(f"unsupported T7 counterpart lane {lane!r}")


def _mutation_changed_checks(case: _CaseT7) -> tuple[str, ...]:
    checks = ["evaluated_operations_completed"]
    if is_hardened_task_v2(case.definition):
        checks.append("operation_selection_coverage")
    if case.lane in {
        "buyer_exchange",
        "merchant_return_authorization",
        "merchant_ledger_close",
    } or (case.lane == "merchant_exchange" and not case.entitled):
        # On a refused exchange the mutation is the rubber stamp, so the lane's
        # decision evidence moves with it; on an allowed one the mutation only
        # stops short of shipping, which the decision check does not see.
        checks.insert(1, "difficulty_evidence_grounded")
    # Every mutation stops the lane short of, or diverts it away from, the
    # disposition the order was owed, so the commercial end state moves too.
    checks.append("business_outcome_completed")
    return tuple(checks)


def _ideal_channel_t7(task_id: str) -> InferenceChannel:
    case = _case_for_t7(task_id)
    return _ScriptedT7Channel(
        _evaluated_steps(case, mutated=False),
        actor_role=case.definition.evaluated_role,
    )


def _counterpart_channel_t7(task_id: str) -> InferenceChannel:
    case = _case_for_t7(task_id)
    return _ScriptedT7Channel(
        _counterpart_steps(case),
        actor_role=("merchant" if case.definition.evaluated_role == "buyer" else "buyer"),
        wait_on_unavailable=True,
    )


def _check(
    name: str,
    weight: float,
    credit: float,
    evidence: Mapping[str, Any],
) -> RuntimeRubricCheckV2:
    return RuntimeRubricCheckV2(
        name=name,
        weight=weight,
        credit=max(0.0, min(1.0, credit)),
        evidence=copy.deepcopy(dict(evidence)),
    )


def _sequence_match_count(expected: tuple[str, ...], observed: tuple[str, ...]) -> int:
    """Count the expected sequence as an ordered subsequence of observations."""

    cursor = 0
    for value in observed:
        if cursor < len(expected) and value == expected[cursor]:
            cursor += 1
    return cursor


def _sequence_credit(expected: tuple[str, ...], observed: tuple[str, ...]) -> float:
    if not expected:
        return 1.0 if not observed else 0.99
    matched = _sequence_match_count(expected, observed)
    credit = matched / len(expected)
    # Completing the required subsequence does not make an altered sequence a
    # strict success.  Keep almost-full partial credit for harmless extra work.
    if matched == len(expected) and observed != expected:
        return min(credit, 0.99)
    return credit


def _counterpart_trigger_groups(case: _CaseT7) -> tuple[tuple[str, ...], ...]:
    """Return deterministic counterpart work caused by accepted hand-offs."""

    if case.lane == "buyer_return_refund":
        return (
            ("authorize_return", "receive_return"),
            ("approve_refund",),
        )
    if case.lane == "merchant_return_authorization":
        return (("request_return",),)
    if case.lane == "merchant_refund":
        return (("request_return",), ("open_refund_case",))
    if case.lane == "buyer_exchange":
        return (
            ("authorize_return", "receive_return"),
            ("authorize_exchange", "complete_exchange"),
        )
    if case.lane == "merchant_exchange":
        return (("request_return",), ("request_exchange",))
    if case.lane == "buyer_dispute":
        # The second group is bound from the verified adjudication service in
        # ``_verified_counterpart_trigger_groups``.  Keeping only its slot here
        # lets prerequisite counting stay value-free.
        return (("respond_to_dispute",), ())
    if case.lane == "merchant_dispute":
        filer_evidence = ("submit_dispute_evidence",) * len(_verified_evidence_ids(case, "filer"))
        return (("open_dispute", *filer_evidence), ("open_refund_case",))
    return ()


def _counterpart_trigger_prerequisites(
    case: _CaseT7,
) -> tuple[tuple[str, ...], ...]:
    """Return accepted evaluated-operation prefixes that trigger peer work."""

    if case.lane in {"buyer_return_refund", "buyer_exchange"}:
        return (
            ("request_return",),
            ("request_return", case.evaluated_operations[-1]),
        )
    if case.lane == "buyer_dispute":
        submitted = ("submit_dispute_evidence",) * len(_verified_evidence_ids(case, "filer"))
        return (
            ("open_dispute", *submitted),
            ("open_dispute", *submitted, "open_refund_case"),
        )
    if case.lane == "merchant_return_authorization":
        return ((),)
    if case.lane in {"merchant_refund", "merchant_exchange"}:
        return ((), ("authorize_return", "receive_return"))
    if case.lane == "merchant_dispute":
        return ((), ("respond_to_dispute",))
    return ()


def _accepted_counterpart_trigger_count(
    case: _CaseT7,
    actor_requests: tuple[VerifiedAfterSalesRequest, ...],
) -> int:
    """Count peer work made necessary by accepted evaluated operations.

    Empty first prerequisites occur only on merchant-evaluated lanes.  Their
    scripted buyer is started by the authenticated scenario root, not by an
    evaluated-model control action, so that initial peer group is environment
    work and is always required.
    """

    observed = tuple(row.operation for row in actor_requests)
    count = 0
    for prerequisite in _counterpart_trigger_prerequisites(case):
        if _sequence_match_count(prerequisite, observed) != len(prerequisite):
            break
        count += 1
    return count


def _required_handoff_requests(
    case: _CaseT7,
    verified: VerifiedAfterSalesEvidence,
) -> tuple[VerifiedAfterSalesRequest, ...]:
    """Return accepted operations that require an Agent-owned peer hand-off."""

    requests = tuple(
        sorted(verified.requests, key=lambda row: row.exchange.request_position)
    )

    def selected(actor_id: str, operation: str) -> tuple[VerifiedAfterSalesRequest, ...]:
        return tuple(
            row
            for row in requests
            if row.exchange.decision.get("actor_id") == actor_id
            and row.operation == operation
        )

    required: list[VerifiedAfterSalesRequest] = []
    if case.lane in {
        "buyer_return_refund",
        "merchant_return_authorization",
        "merchant_refund",
    }:
        required.extend(selected(_BUYER_ID, "request_return"))
        if case.lane != "merchant_return_authorization":
            required.extend(selected(_MERCHANT_ID, "receive_return"))
            required.extend(selected(_BUYER_ID, "open_refund_case"))
    elif case.lane in {"buyer_exchange", "merchant_exchange"}:
        required.extend(selected(_BUYER_ID, "request_return"))
        required.extend(selected(_MERCHANT_ID, "receive_return"))
        required.extend(selected(_BUYER_ID, "request_exchange"))
    elif case.lane in {"buyer_dispute", "merchant_dispute"}:
        submissions = selected(_BUYER_ID, "submit_dispute_evidence")
        submitted_ids = tuple(
            record.record_id for row in submissions for record in row.evidence_records
        )
        expected_ids = _verified_evidence_ids(case, "filer")
        if set(submitted_ids) == set(expected_ids) and len(submitted_ids) == len(expected_ids):
            required.append(submissions[-1])
        required.extend(selected(_MERCHANT_ID, "respond_to_dispute"))
        required.extend(selected(_BUYER_ID, "open_refund_case"))
    return tuple(sorted(required, key=lambda row: row.exchange.request_position))


def _require_bound_counterparty_handoff_closure(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedAfterSalesEvidence,
) -> None:
    """Require one exact Agent hand-off for every accepted trigger operation."""

    expected = _required_handoff_requests(case, verified)
    observed = tuple(
        (position, row)
        for position, row in enumerate(evidence.envelopes)
        if row.get("in_reply_to") is not None
        and isinstance(row.get("action"), Mapping)
        and row["action"].get("kind") == "commerce.send_message"
        and isinstance(row["action"].get("payload"), Mapping)
        and row["action"]["payload"].get("category") == "after_sales"
    )
    matched_positions: list[int] = []
    matched_messages: list[Mapping[str, Any]] = []
    for request in expected:
        response_id = request.response.get("msg_id")
        actor_id = request.exchange.decision.get("actor_id")
        recipient_id = _MERCHANT_ID if actor_id == _BUYER_ID else _BUYER_ID
        matches = [
            (position, row)
            for position, row in observed
            if row.get("from") == actor_id
            and row.get("to") == recipient_id
            and row.get("in_reply_to") == response_id
        ]
        if len(matches) != 1:
            raise RuntimeBenchmarkIntegrityError(
                "T7 Agent failed to emit one exact bound-counterparty continuation"
            )
        position, message = matches[0]
        payload = message["action"]["payload"]
        if (
            payload.get("order_id") != case.order_id
            or position <= request.exchange.request_position
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T7 bound-counterparty continuation crossed accepted order lineage"
            )
        response_action = request.response.get("action")
        response_payload = (
            response_action.get("payload")
            if isinstance(response_action, Mapping)
            else None
        )
        references = (
            response_payload.get("references")
            if isinstance(response_payload, Mapping)
            else None
        )
        if not isinstance(references, Mapping):
            raise RuntimeBenchmarkIntegrityError(
                "T7 accepted hand-off trigger has no authoritative references"
            )
        correlation: dict[str, Any]
        reference_field = {
            "request_return": "request_id",
            "submit_dispute_evidence": "dispute_id",
            "open_refund_case": "case_id",
            "request_exchange": "case_id",
        }.get(request.operation)
        if reference_field is not None:
            reference_value = references.get(reference_field)
            if not isinstance(reference_value, str) or not reference_value:
                raise RuntimeBenchmarkIntegrityError(
                    "T7 accepted hand-off trigger has a malformed correlation reference"
                )
            correlation = {reference_field: reference_value}
        elif request.operation == "receive_return":
            correlation = {"event": "return_received"}
        elif request.operation == "respond_to_dispute":
            correlation = {"event": "ruling_committed"}
        else:
            raise RuntimeBenchmarkIntegrityError(
                "T7 accepted operation has no deterministic hand-off payload"
            )
        expected_payload = {
            "category": "after_sales",
            "order_id": case.order_id,
            **correlation,
        }
        if dict(payload) != expected_payload:
            raise RuntimeBenchmarkIntegrityError(
                "T7 bound-counterparty payload does not match accepted response correlation"
            )
        matched_positions.append(position)
        matched_messages.append(message)
    if len(observed) != len(expected) or len(set(matched_positions)) != len(expected):
        raise RuntimeBenchmarkIntegrityError(
            "T7 run contains an untriggered or duplicated bound-counterparty continuation"
        )

    counterpart_id = _MERCHANT_ID if case.evaluated_actor_id == _BUYER_ID else _BUYER_ID
    counterpart_reads = tuple(
        row for row in verified.reads if row.actor_id == counterpart_id
    )
    expected_read_resources = (
        ("after_sales_history",)
        if case.lane == "buyer_dispute"
        and any(row.operation == "open_refund_case" for row in verified.requests)
        else ()
    )
    if tuple(row.resource for row in counterpart_reads) != expected_read_resources:
        raise RuntimeBenchmarkIntegrityError(
            "T7 scripted counterpart read closure differs from accepted hand-offs"
        )
    if counterpart_reads:
        refund_messages = [
            message
            for request, message in zip(expected, matched_messages, strict=True)
            if request.operation == "open_refund_case"
        ]
        if len(refund_messages) != 1 or (
            counterpart_reads[0].exchange.request.get("in_reply_to")
            != refund_messages[0].get("msg_id")
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T7 scripted counterpart ruling read has no exact hand-off lineage"
            )


def _verified_counterpart_trigger_groups(
    case: _CaseT7,
    verified: VerifiedAfterSalesEvidence,
    *,
    trigger_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Bind deterministic dispute resolution to the actual service ruling."""

    groups = _counterpart_trigger_groups(case)
    if case.lane != "buyer_dispute" or trigger_count < 2:
        return groups
    rulings = tuple(
        row.operation
        for row in verified.service_operations
        if row.operation in {"rule_for_filer", "rule_split", "rule_for_respondent"}
        and row.commit.get("subject_id") == case.order_id
    )
    if len(rulings) != 1:
        raise RuntimeBenchmarkIntegrityError(
            "T7 model-triggered refund has no unique authoritative dispute ruling"
        )
    resolution = "deny_refund" if rulings[0] == "rule_for_respondent" else "approve_refund"
    return (groups[0], (resolution,))


def _evaluated_actor_attempt_sequence(
    actor_id: str,
    verified: VerifiedAfterSalesEvidence,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    attempts: list[tuple[int, str]] = [
        (row.exchange.request_position, row.operation)
        for row in verified.requests
        if row.exchange.decision.get("actor_id") == actor_id
    ]
    rejected: list[str] = []
    for row in verified.rejected_requests:
        if row.exchange.decision.get("actor_id") != actor_id:
            continue
        operation = row.operation if isinstance(row.operation, str) else "unknown"
        label = f"rejected:{operation}"
        attempts.append((row.exchange.request_position, label))
        rejected.append(label)
    attempts.sort(key=lambda item: item[0])
    return tuple(value for _, value in attempts), tuple(rejected)


def _require_triggered_environment_closure(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedAfterSalesEvidence,
) -> None:
    """Require only environment work causally triggered by model-authored intents.

    Accepted after-sales requests are model choices; participant hand-offs are
    Agent work.  Once an accepted request triggers deterministic peer/service
    work, failure of that work invalidates the run.  Future work that the model
    never triggered remains a capability omission.
    """

    actor_id = case.evaluated_actor_id
    counterpart_id = _MERCHANT_ID if actor_id == _BUYER_ID else _BUYER_ID
    actor_requests = verified.requests_for_actor(actor_id)
    counterpart_requests = verified.requests_for_actor(counterpart_id)
    counterpart_rejections = tuple(
        row
        for row in verified.rejected_requests
        if row.exchange.decision.get("actor_id") == counterpart_id
    )
    if counterpart_rejections:
        raise RuntimeBenchmarkIntegrityError(
            "T7 scripted counterpart produced a rejected Platform request"
        )
    counterpart_operations = tuple(
        row.operation for row in counterpart_requests
    )
    trigger_count = _accepted_counterpart_trigger_count(case, actor_requests)
    trigger_groups = _verified_counterpart_trigger_groups(
        case,
        verified,
        trigger_count=trigger_count,
    )
    required_counterpart = tuple(
        operation
        for group in trigger_groups[:trigger_count]
        for operation in group
    )
    if counterpart_operations != required_counterpart:
        raise RuntimeBenchmarkIntegrityError(
            "T7 scripted counterpart failed to close model-triggered work: "
            f"required={list(required_counterpart)!r}, "
            f"observed={list(counterpart_operations)!r}"
        )
    matched_counterpart = list(counterpart_requests)

    return_evidence_ids = _evidence_ids(case) if case.axis_name == "return_condition_count" else ()
    filer_evidence_ids = _verified_evidence_ids(case, "filer")
    respondent_evidence_ids = _verified_evidence_ids(case, "respondent")
    observed_filer_ids = tuple(
        record.record_id
        for row in matched_counterpart
        if row.operation == "submit_dispute_evidence"
        for record in row.evidence_records
    )
    semantics_faithful = all(
        (
            row.operation != "request_return"
            or tuple(record.record_id for record in row.evidence_records)
            == return_evidence_ids
        )
        and (
            row.operation != "request_exchange"
            # On a merchant-evaluated exchange the buyer asks for whatever the
            # fixture says they asked for, which is not always something they
            # are entitled to -- that is the merchant's problem to catch.
            or row.intent.get("replacement_sku_id")
            == (case.requested_replacement_sku_id or _expected_replacement_sku(case))
        )
        and (
            row.operation != "respond_to_dispute"
            or (
                row.intent.get("position") == "contest"
                and tuple(record.record_id for record in row.evidence_records)
                == respondent_evidence_ids
            )
        )
        for row in matched_counterpart
    ) and observed_filer_ids == (
        filer_evidence_ids
        if any(row.operation == "submit_dispute_evidence" for row in matched_counterpart)
        else ()
    )
    if not semantics_faithful:
        raise RuntimeBenchmarkIntegrityError(
            "T7 scripted counterpart completed model-triggered work with unfaithful "
            "business evidence or replacement semantics"
        )
    _require_bound_counterparty_handoff_closure(case, evidence, verified)

    observed_services = tuple(
        (row.operation, str(row.commit.get("subject_id", "")))
        for row in verified.service_operations
    )
    if case.lane == "merchant_ledger_close":
        required_services = tuple(
            ("complete_ledger_reconciliation", str(row.intent.get("order_id", "")))
            for row in actor_requests
            if row.operation == "request_ledger_reconciliation"
        )
        if observed_services != required_services:
            raise RuntimeBenchmarkIntegrityError(
                "T7 accounting service failed to close model-triggered ledger work: "
                f"required={list(required_services)!r}, "
                f"observed={list(observed_services)!r}"
            )
        return

    if case.lane in {"buyer_dispute", "merchant_dispute"}:
        ruling_operations = {"rule_for_filer", "rule_split", "rule_for_respondent"}
        response_accepted = "respond_to_dispute" in (
            counterpart_operations if case.lane == "buyer_dispute" else tuple(
                row.operation for row in actor_requests
            )
        )
        ruling_services = tuple(
            row
            for row in observed_services
            if row[0] in ruling_operations and row[1] == case.order_id
        )
        if (response_accepted and len(ruling_services) != 1) or (
            not response_accepted and ruling_services
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T7 adjudication service failed to close the accepted dispute response"
            )
        if len(ruling_services) != len(observed_services):
            raise RuntimeBenchmarkIntegrityError(
                "T7 dispute run contains an untriggered service operation"
            )
        return

    if observed_services:
        raise RuntimeBenchmarkIntegrityError(
            "T7 run contains a service operation with no model-authored trigger"
        )


def _world_rows(
    evidence: RuntimeEvidenceBundleV2,
    table: str,
    *,
    initial: bool,
) -> tuple[dict[str, Any], ...]:
    snapshot = evidence.initial_world if initial else evidence.final_world
    raw = snapshot.get("tables", {}).get(table, [])
    if not isinstance(raw, list):
        raise RuntimeEvidenceError(f"World table {table!r} must be an array")
    if not all(isinstance(row, Mapping) for row in raw):
        raise RuntimeEvidenceError(f"World table {table!r} contains an invalid row")
    return tuple(dict(row) for row in raw)


def _world_mapping(
    evidence: RuntimeEvidenceBundleV2,
    table: str,
    *,
    initial: bool,
) -> dict[str, dict[str, Any]]:
    snapshot = evidence.initial_world if initial else evidence.final_world
    raw = snapshot.get("tables", {}).get(table, {})
    if not isinstance(raw, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Mapping) for key, value in raw.items()
    ):
        raise RuntimeEvidenceError(f"World table {table!r} must be an object of rows")
    return {key: dict(value) for key, value in raw.items()}


def _unique_rows(
    rows: tuple[dict[str, Any], ...],
    *,
    key: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in indexed:
            raise RuntimeEvidenceError(f"{label} has a missing or duplicate {key}")
        indexed[value] = row
    return indexed


def _orders(
    evidence: RuntimeEvidenceBundleV2,
    *,
    initial: bool,
) -> dict[str, dict[str, Any]]:
    return _unique_rows(
        _world_rows(evidence, "orders", initial=initial),
        key="order_id",
        label="World orders",
    )


def _payment_history(
    evidence: RuntimeEvidenceBundleV2,
    order_id: str,
    *,
    initial: bool,
) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        row
        for row in _world_rows(evidence, "payment_states", initial=initial)
        if row.get("order_id") == order_id
    )
    try:
        ordered = tuple(sorted(rows, key=lambda row: int(row["version"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeEvidenceError("World payment history has an invalid version") from exc
    if not ordered or tuple(int(row["version"]) for row in ordered) != tuple(
        range(1, len(ordered) + 1)
    ):
        raise RuntimeEvidenceError("World payment history is missing a version")
    return ordered


def _packing_history(
    evidence: RuntimeEvidenceBundleV2,
    order_id: str,
    *,
    initial: bool,
) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        row
        for row in _world_rows(evidence, "packing_records", initial=initial)
        if row.get("order_id") == order_id
    )
    try:
        ordered = tuple(sorted(rows, key=lambda row: int(row["version"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeEvidenceError("World packing history has an invalid version") from exc
    if ordered and tuple(int(row["version"]) for row in ordered) != tuple(
        range(1, len(ordered) + 1)
    ):
        raise RuntimeEvidenceError("World packing history is missing a version")
    return ordered


def _receipt_from_row(row: Mapping[str, Any]) -> Receipt:
    price = row.get("price")
    if not isinstance(price, Mapping):
        raise RuntimeEvidenceError("World ledger receipt has no Money value")
    amount = price.get("amount")
    if not isinstance(amount, str):
        raise RuntimeEvidenceError("World ledger receipt amount must be exact text")
    try:
        decimal_amount = Decimal(amount)
    except (InvalidOperation, ValueError) as exc:
        raise RuntimeEvidenceError("World ledger receipt amount is invalid") from exc
    qty = row.get("qty")
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        raise RuntimeEvidenceError("World ledger receipt quantity is invalid")
    effect = row.get("effect")
    if effect not in {"charge", "refund"}:
        raise RuntimeEvidenceError("World ledger receipt effect is invalid")
    required_text = (
        "txn_id",
        "ts",
        "order_id",
        "buyer_id",
        "merchant_id",
        "sku_id",
        "idempotency_key",
    )
    if any(not isinstance(row.get(name), str) or not row.get(name) for name in required_text):
        raise RuntimeEvidenceError("World ledger receipt identity is incomplete")
    currency = price.get("currency")
    if not isinstance(currency, str) or not currency:
        raise RuntimeEvidenceError("World ledger receipt currency is invalid")
    return Receipt(
        txn_id=TxnId(str(row["txn_id"])),
        ts=str(row["ts"]),
        order_id=OrderId(str(row["order_id"])),
        buyer_id=AgentId(str(row["buyer_id"])),
        merchant_id=AgentId(str(row["merchant_id"])),
        sku_id=SkuId(str(row["sku_id"])),
        qty=qty,
        price=Money(decimal_amount, currency),
        idempotency_key=str(row["idempotency_key"]),
        effect=effect,
    )


def _receipt_minor_units(receipt: Receipt) -> int:
    minor = receipt.price.amount * receipt.qty * Decimal(100)
    if not minor.is_finite() or minor < 0 or minor != minor.to_integral_value():
        raise RuntimeEvidenceError("World ledger receipt is not exact minor-unit money")
    return int(minor)


def _new_refund_receipt(
    evidence: RuntimeEvidenceBundleV2,
    order_id: str,
) -> tuple[Receipt | None, bool]:
    initial = _world_rows(evidence, "ledger", initial=True)
    final = _world_rows(evidence, "ledger", initial=False)
    prefix_preserved = final[: len(initial)] == initial
    new_rows = final[len(initial) :] if prefix_preserved else ()
    matching = tuple(
        _receipt_from_row(row)
        for row in new_rows
        if row.get("order_id") == order_id and row.get("effect") == "refund"
    )
    return (matching[0] if len(matching) == 1 else None), (
        prefix_preserved and len(new_rows) == 1 and len(matching) == 1
    )


def _after_sales_values(
    evidence: RuntimeEvidenceBundleV2,
    table: str,
) -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    for wrapper in _world_rows(evidence, "after_sales_records", initial=False):
        if wrapper.get("table") != table:
            continue
        value = wrapper.get("value")
        if not isinstance(value, Mapping):
            raise RuntimeEvidenceError(f"after-sales table {table!r} has an invalid row")
        values.append(dict(value))
    return tuple(values)


def _request_for(
    verified: VerifiedAfterSalesEvidence,
    operation: str,
    order_id: str,
) -> VerifiedAfterSalesRequest | None:
    matches = tuple(
        row
        for row in verified.requests
        if row.operation == operation and row.commit.get("subject_id") == order_id
    )
    return matches[0] if len(matches) == 1 else None


def _service_for(
    verified: VerifiedAfterSalesEvidence,
    operation: str,
    order_id: str,
) -> VerifiedAfterSalesServiceOperation | None:
    matches = tuple(
        row
        for row in verified.service_operations
        if row.operation == operation and row.commit.get("subject_id") == order_id
    )
    return matches[0] if len(matches) == 1 else None


def _commit_tables(commit: Mapping[str, Any] | None) -> tuple[str, ...]:
    if commit is None:
        return ()
    writes = commit.get("table_writes")
    if not isinstance(writes, list) or not all(
        isinstance(row, Mapping) and isinstance(row.get("table"), str) for row in writes
    ):
        raise RuntimeEvidenceError("World commit has an invalid table write set")
    return tuple(str(row["table"]) for row in writes)


def _expected_operation_commit_tables(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    row: VerifiedAfterSalesRequest | VerifiedAfterSalesServiceOperation,
    *,
    creates_binding: bool,
) -> tuple[str, ...]:
    operation = row.operation
    base = (
        *("after_sales_records" for _ in range(1 + int(creates_binding))),
        "authority_operations",
        "logical_time",
    )
    if operation == "cancel_paid_order":
        return (
            "orders",
            "inventory",
            "payment_states",
            *(("ledger",) if case.initial_payment_stage == "captured" else ()),
            *(("packing_records",) if case.initial_packing_stage == "packed" else ()),
            *base,
        )
    if operation == "receive_return":
        return ("orders", "order_timelines", *base)
    if operation == "approve_refund":
        value = _result_value(row)
        amount = None if value is None else value.get("approved_amount")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise RuntimeEvidenceError("accepted refund has an invalid approved amount")
        if amount == 0:
            return base
        payments = _payment_history(
            evidence,
            str(row.commit.get("subject_id", "")),
            initial=True,
        )
        if not payments:
            raise RuntimeEvidenceError("accepted refund has no initial payment authority")
        captured = payments[-1].get("captured_amount")
        if isinstance(captured, bool) or not isinstance(captured, int) or captured < amount:
            raise RuntimeEvidenceError("accepted refund exceeds initial captured authority")
        full = amount == captured
        return (
            *(("orders",) if full else ()),
            "ledger",
            "payment_states",
            *(("inventory",) if case.lane in {"buyer_return_refund", "merchant_refund"} else ()),
            *(("order_timelines",) if full else ()),
            *base,
        )
    if operation == "complete_exchange":
        value = _result_value(row)
        replacement = None if value is None else value.get("replacement_sku_id")
        if not isinstance(replacement, str) or not replacement:
            raise RuntimeEvidenceError("accepted exchange has no replacement identity")
        inventory_writes = 1 if replacement == case.original_sku_id else 2
        return (
            "orders",
            "orders",
            *("inventory" for _ in range(inventory_writes)),
            "exchanges",
            *base,
        )
    if operation == "open_dispute":
        return ("disputes", *base)
    if operation in {"submit_dispute_evidence", "respond_to_dispute"}:
        return ("disputes", "after_sales_records", *base)
    if operation in {"rule_for_filer", "rule_split", "rule_for_respondent"}:
        return ("rulings", "disputes", "after_sales_records", *base)
    if operation == "complete_ledger_reconciliation":
        value = _result_value(row)
        sources = None if value is None else value.get("source_txn_ids")
        if not isinstance(sources, (list, tuple)) or not sources:
            raise RuntimeEvidenceError("ledger reconciliation has no typed sources")
        return (
            *("after_sales_records" for _ in range(len(sources))),
            *base,
        )
    if operation in {
        "request_return",
        "authorize_return",
        "deny_return",
        "open_refund_case",
        "deny_refund",
        "request_exchange",
        "authorize_exchange",
        "deny_exchange",
        "request_ledger_reconciliation",
    }:
        return base
    raise RuntimeEvidenceError(f"T7 has no write-set contract for {operation!r}")


def _require_return_receipt_commit_semantics(row: VerifiedAfterSalesRequest) -> None:
    writes = row.commit.get("table_writes")
    if not isinstance(writes, list):
        raise RuntimeEvidenceError("return receipt commit has no table writes")
    order_writes = [item for item in writes if item.get("table") == "orders"]
    timeline_writes = [item for item in writes if item.get("table") == "order_timelines"]
    if len(order_writes) != 1 or len(timeline_writes) != 1:
        raise RuntimeEvidenceError("return receipt commit has no exact physical transition")
    order_before = order_writes[0].get("before")
    order_after = order_writes[0].get("after")
    timeline_before = timeline_writes[0].get("before")
    timeline_after = timeline_writes[0].get("after")
    if not all(
        isinstance(value, Mapping)
        for value in (order_before, order_after, timeline_before, timeline_after)
    ):
        raise RuntimeEvidenceError("return receipt physical transition is malformed")
    expected_order = dict(order_before)
    expected_order["state"] = "returned"
    returned_tick = timeline_after.get("returned_at_tick")
    if (
        order_after != expected_order
        or isinstance(returned_tick, bool)
        or not isinstance(returned_tick, int)
    ):
        raise RuntimeEvidenceError("return receipt did not mark the exact order returned")
    expected_timeline = dict(timeline_before)
    expected_timeline["returned_at_tick"] = returned_tick
    if timeline_after != expected_timeline:
        raise RuntimeEvidenceError("return receipt did not update the exact return timeline")


def _require_accepted_operation_write_sets(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedAfterSalesEvidence,
) -> None:
    rows = sorted(
        (*verified.requests, *verified.service_operations),
        key=lambda row: int(row.commit.get("sequence", -1)),
    )
    seen_commits: set[str] = set()
    bound_orders: set[str] = set()
    for row in rows:
        commit_id = str(row.commit.get("commit_id", ""))
        order_id = str(row.commit.get("subject_id", ""))
        if not commit_id or not order_id:
            raise RuntimeEvidenceError("accepted after-sales commit has no stable identity")
        if commit_id in seen_commits:
            continue
        creates_binding = order_id not in bound_orders
        expected = _expected_operation_commit_tables(
            case,
            evidence,
            row,
            creates_binding=creates_binding,
        )
        observed = _commit_tables(row.commit)
        if Counter(observed) != Counter(expected):
            raise RuntimeEvidenceError(
                "accepted after-sales operation has an unfaithful World write set: "
                f"operation={row.operation!r}, expected={list(expected)!r}, "
                f"observed={list(observed)!r}"
            )
        if isinstance(row, VerifiedAfterSalesRequest) and row.operation == "receive_return":
            _require_return_receipt_commit_semantics(row)
        seen_commits.add(commit_id)
        bound_orders.add(order_id)


def _same_physical_tables(
    evidence: RuntimeEvidenceBundleV2,
    tables: tuple[str, ...],
) -> bool:
    initial_tables = evidence.initial_world.get("tables", {})
    final_tables = evidence.final_world.get("tables", {})
    if not isinstance(initial_tables, Mapping) or not isinstance(final_tables, Mapping):
        raise RuntimeEvidenceError("World snapshots have invalid table objects")
    return all(initial_tables.get(table, []) == final_tables.get(table, []) for table in tables)


def _expected_inventory(
    initial: Mapping[str, Mapping[str, Any]],
    deltas: Mapping[str, int],
) -> dict[str, dict[str, Any]] | None:
    expected = copy.deepcopy({key: dict(value) for key, value in initial.items()})
    for sku_id, reserved_delta in deltas.items():
        row = expected.get(sku_id)
        if row is None:
            return None
        reserved = row.get("qty_reserved")
        version = row.get("version")
        if (
            isinstance(reserved, bool)
            or not isinstance(reserved, int)
            or isinstance(version, bool)
            or not isinstance(version, int)
        ):
            return None
        row["qty_reserved"] = reserved + reserved_delta
        row["version"] = version + 1
    return expected


def _payment_resolution_matches(
    initial: tuple[dict[str, Any], ...],
    final: tuple[dict[str, Any], ...],
    *,
    state: str,
    refunded_amount: int,
    refund_receipt: Receipt | None,
) -> bool:
    if len(final) != len(initial) + 1 or final[:-1] != initial:
        return False
    previous = initial[-1]
    current = final[-1]
    identity_fields = (
        "payment_id",
        "order_id",
        "owner_id",
        "merchant_id",
        "sku_id",
        "qty",
        "amount",
        "currency",
    )
    if any(current.get(name) != previous.get(name) for name in identity_fields):
        return False
    if (
        current.get("state") != state
        or current.get("version") != int(previous.get("version", 0)) + 1
        or current.get("previous_digest") != previous.get("record_digest")
        or current.get("captured_amount") != previous.get("captured_amount")
        or current.get("refunded_amount") != refunded_amount
    ):
        return False
    if refund_receipt is None:
        return (
            state == "voided"
            and current.get("captured_amount") == 0
            and current.get("refunded_amount") == 0
            and current.get("capture_receipt_digest") is None
            and current.get("ledger_receipt_digest") is None
        )
    digest = authoritative_payment_receipt_digest(refund_receipt)
    return (
        current.get("capture_receipt_digest") == previous.get("capture_receipt_digest")
        and current.get("ledger_receipt_digest") == digest
    )


def _append_clause(
    clauses: dict[str, bool],
    name: str,
    value: object,
) -> None:
    if name in clauses:
        raise RuntimeEvidenceError(f"duplicate T7 outcome clause {name!r}")
    clauses[name] = bool(value)


def _shipment_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "shipment_id",
        "order_id",
        "buyer_id",
        "merchant_id",
        "original_sku_id",
        "status",
        "resolution",
        "replacement_sku_id",
        "version",
    )
    if any(field not in row for field in fields):
        raise RuntimeEvidenceError("authoritative shipment row is incomplete")
    raw_history = row.get("status_history")
    if not isinstance(raw_history, list) or not all(
        isinstance(value, Mapping) for value in raw_history
    ):
        raise RuntimeEvidenceError("authoritative shipment history is invalid")
    history: list[dict[str, Any]] = []
    for value in raw_history:
        if any(field not in value for field in ("event_id", "status", "logical_time")):
            raise RuntimeEvidenceError("authoritative shipment event is incomplete")
        history.append(
            {
                "event_id": value["event_id"],
                "status": value["status"],
                "logical_time": value["logical_time"],
            }
        )
    projected = {field: row[field] for field in fields}
    projected["status_history"] = history
    return projected


def _shipment_replacement_options_projection(
    evidence: RuntimeEvidenceBundleV2,
    shipment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Replay the Platform's bounded replacement projection from World state."""

    inventory = _world_mapping(evidence, "inventory", initial=True)
    options: list[dict[str, Any]] = []
    for listing in _world_rows(evidence, "catalog", initial=True):
        sku_id = listing.get("sku_id")
        if (
            not isinstance(sku_id, str)
            or sku_id == shipment.get("original_sku_id")
            or listing.get("merchant_id") != shipment.get("merchant_id")
        ):
            continue
        stock = inventory.get(sku_id)
        price = listing.get("list_price")
        if not isinstance(stock, Mapping) or not isinstance(price, Mapping):
            raise RuntimeEvidenceError(
                "replacement projection has malformed catalog or inventory authority"
            )
        available = stock.get("qty_available")
        reserved = stock.get("qty_reserved")
        version = stock.get("version")
        if (
            isinstance(available, bool)
            or not isinstance(available, int)
            or isinstance(reserved, bool)
            or not isinstance(reserved, int)
            or isinstance(version, bool)
            or not isinstance(version, int)
        ):
            raise RuntimeEvidenceError("replacement inventory projection is invalid")
        available_qty = available - reserved
        if available_qty <= 0:
            continue
        try:
            unit_price_cents = int(
                (Decimal(str(price["amount"])) * Decimal(100)).to_integral_exact()
            )
            currency = str(price["currency"])
        except (KeyError, InvalidOperation, ValueError) as exc:
            raise RuntimeEvidenceError("replacement catalog price projection is invalid") from exc
        options.append(
            {
                "sku_id": sku_id,
                "merchant_id": listing.get("merchant_id"),
                "available_qty": available_qty,
                "unit_price_cents": unit_price_cents,
                "currency": currency,
                "supply_version": version,
            }
        )
    return sorted(options, key=lambda row: str(row["sku_id"]))


def _verified_shipment_read(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
) -> dict[str, Any]:
    """Validate shipment-read transport while scoring the actor's read choice.

    A missing, rejected, wrong-target, or repeated read is model behavior.  It
    must remain scoreable.  Once Platform accepts a read, however, its response
    must be an exact caller-scoped projection of the requested World row.
    """

    if case.shipment_id is None:
        return {"required": False, "verified": True}
    exchanges = evidence.platform_exchanges(
        kind="commerce.read_shipment",
        actor_id=case.evaluated_actor_id,
        endpoint="platform:fulfillment",
    )
    if not exchanges:
        return {
            "required": True,
            "verified": False,
            "shipment_id": case.shipment_id,
            "read_attempt_count": 0,
            "accepted_read_count": 0,
        }

    accepted: list[tuple[Any, str, dict[str, Any]]] = []
    requested_ids: list[str | None] = []
    for exchange in exchanges:
        action = exchange.request.get("action")
        payload = action.get("payload") if isinstance(action, Mapping) else None
        requested_id = (
            str(payload["shipment_id"])
            if isinstance(payload, Mapping)
            and isinstance(payload.get("shipment_id"), str)
            and payload["shipment_id"]
            else None
        )
        requested_ids.append(requested_id)
        if exchange.decision.get("decision") != "accepted":
            continue
        if requested_id is None:
            raise RuntimeEvidenceError("accepted shipment read has no shipment identity")
        if len(exchange.responses) != 1:
            raise RuntimeEvidenceError("accepted shipment read needs one exact Platform response")
        response = exchange.responses[0]
        response_action = response.get("action")
        response_payload = (
            response_action.get("payload") if isinstance(response_action, Mapping) else None
        )
        if (
            response.get("from") != "platform:fulfillment"
            or response.get("to") != case.evaluated_actor_id
            or response.get("in_reply_to") != exchange.request.get("msg_id")
            or response.get("idempotency_key") != exchange.request.get("idempotency_key")
            or not isinstance(response_action, Mapping)
            or response_action.get("kind") != "platform.shipment_state"
        ):
            raise RuntimeEvidenceError("shipment response envelope identity changed")
        rows = tuple(
            row
            for row in _world_rows(evidence, "shipments", initial=True)
            if row.get("shipment_id") == requested_id
        )
        if len(rows) != 1:
            raise RuntimeEvidenceError(
                "accepted shipment read does not resolve one initial World row"
            )
        expected = _shipment_projection(rows[0])
        expected_options = _shipment_replacement_options_projection(
            evidence,
            expected,
        )
        if not isinstance(response_payload, Mapping) or set(response_payload) != {
            "shipment",
            "replacement_options",
        }:
            raise RuntimeEvidenceError(
                "shipment response differs from the replayed caller-visible World projection"
            )
        raw_options = response_payload.get("replacement_options")
        if (
            response_payload.get("shipment") != expected
            or not isinstance(raw_options, list)
            or any(not isinstance(row, Mapping) for row in raw_options)
            or sorted(raw_options, key=lambda row: str(row.get("sku_id"))) != expected_options
        ):
            raise RuntimeEvidenceError(
                "shipment response differs from the replayed caller-visible World projection"
            )
        if case.evaluated_actor_id not in {
            expected["buyer_id"],
            expected["merchant_id"],
        }:
            raise RuntimeEvidenceError("shipment response was not scoped to an order party")
        accepted.append((exchange, requested_id, expected))

    target_read = bool(
        len(exchanges) == 1 and len(accepted) == 1 and accepted[0][1] == case.shipment_id
    )
    return {
        "required": True,
        "verified": target_read,
        "shipment_id": case.shipment_id,
        "requested_shipment_ids": requested_ids,
        "read_attempt_count": len(exchanges),
        "accepted_read_count": len(accepted),
        "request_msg_id": (accepted[0][0].request.get("msg_id") if len(accepted) == 1 else None),
        "decision_id": (accepted[0][0].decision.get("decision_id") if len(accepted) == 1 else None),
        "status": accepted[0][2]["status"] if len(accepted) == 1 else None,
        "version": accepted[0][2]["version"] if len(accepted) == 1 else None,
    }


def _require_order_stage_fixture(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
) -> None:
    """Make T7 order-stage difficulty a hard environment precondition."""

    if case.axis_name != "order_stage":
        return
    try:
        payments = _payment_history(evidence, case.order_id, initial=True)
        packings = _packing_history(evidence, case.order_id, initial=True)
    except (RuntimeEvidenceError, ValueError, TypeError) as exc:
        raise RuntimeBenchmarkIntegrityError(
            "T7 initial order-stage fixture cannot be derived from World evidence"
        ) from exc
    expected_payment_states = (case.initial_payment_stage,)
    expected_packing_states = (
        ("created", "packed") if case.initial_packing_stage == "packed" else ()
    )
    observed_payment_states = tuple(str(row.get("state", "")) for row in payments)
    observed_packing_states = tuple(str(row.get("state", "")) for row in packings)
    if (
        observed_payment_states != expected_payment_states
        or observed_packing_states != expected_packing_states
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T7 initial order-stage fixture does not match the frozen task"
        )


def _difficulty_evidence(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedAfterSalesEvidence,
    *,
    actor_operations: tuple[str, ...],
    shipment_read: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    axis = case.axis_name
    actor_id = case.evaluated_actor_id
    actor_requests = verified.requests_for_actor(actor_id)
    if axis == "order_stage":
        raise RuntimeEvidenceError(
            "order-stage state is an environment precondition, not a capability score"
        )

    if axis == "return_condition_count":
        expected_ids = set(_evidence_ids(case))
        if not expected_ids:
            raise RuntimeEvidenceError("return-condition difficulty has no evidence items")
        if case.definition.evaluated_role == "buyer":
            requests = [row for row in actor_requests if row.operation == "request_return"]
            cited_ids = tuple(
                record.record_id for row in requests for record in row.evidence_records
            )
            cited_set = set(cited_ids)
            credit = len(expected_ids & cited_set) / len(expected_ids)
            if credit == 1.0 and (
                len(requests) != 1
                or cited_set != expected_ids
                or len(cited_ids) != len(expected_ids)
            ):
                credit = 0.99
            return credit, {
                "axis": axis,
                "evaluated_actor_id": actor_id,
                "expected_evidence_ids": sorted(expected_ids),
                "cited_evidence_ids": list(cited_ids),
                "request_count": len(requests),
            }

        decisions = tuple(
            str(row.intent.get("op", ""))
            for row in actor_requests
            if row.operation in {"authorize_return", "deny_return"}
        )
        expected_decisions = (
            ("authorize_return",) if case.entitled else ("deny_return",)
        )
        return float(decisions == expected_decisions), {
            "axis": axis,
            "evaluated_actor_id": actor_id,
            "accepted_return_conditions": list(case.allowed_return_conditions),
            "inspected_return_conditions": list(case.attested_return_conditions),
            "expected_decisions": list(expected_decisions),
            "observed_decisions": list(decisions),
        }

    if axis == "replacement_constraint_count":
        expected_sku = _expected_replacement_sku(case)
        if case.definition.evaluated_role == "buyer":
            requests = [row for row in actor_requests if row.operation == "request_exchange"]
            selected = tuple(str(row.intent.get("replacement_sku_id", "")) for row in requests)
            return float(selected == (expected_sku,)), {
                "axis": axis,
                "evaluated_actor_id": actor_id,
                "expected_replacement_sku": expected_sku,
                "selected_replacement_skus": list(selected),
            }

        decisions = tuple(
            str(row.intent.get("op", ""))
            for row in actor_requests
            if row.operation in {"authorize_exchange", "deny_exchange"}
        )
        expected_decisions = (
            ("authorize_exchange",) if case.entitled else ("deny_exchange",)
        )
        return float(decisions == expected_decisions), {
            "axis": axis,
            "evaluated_actor_id": actor_id,
            "eligible_replacement_sku": expected_sku,
            "requested_replacement_sku": case.requested_replacement_sku_id,
            "expected_decisions": list(expected_decisions),
            "observed_decisions": list(decisions),
        }

    if axis == "evidence_item_count":
        side = "filer" if case.definition.evaluated_role == "buyer" else "respondent"
        expected_ids = set(_verified_evidence_ids(case, side))
        accepted = [
            row
            for row in actor_requests
            if row.operation
            == ("submit_dispute_evidence" if side == "filer" else "respond_to_dispute")
        ]
        observed_ids = tuple(
            record.record_id for row in accepted for record in row.evidence_records
        )
        observed_set = set(observed_ids)
        # Recall and precision.  World already refuses a record from another
        # order or from the other party, so the only way to file something
        # that does not stand up is to file this side's unverified record --
        # and doing that is a real filing error, not a harmless extra.  The
        # penalty is scaled by how much of the filing was junk rather than
        # zeroing an otherwise complete submission.
        filed_junk = observed_set - expected_ids
        recall = len(expected_ids & observed_set) / len(expected_ids)
        precision = len(expected_ids & observed_set) / max(len(observed_set), 1)
        credit = recall * precision
        if credit == 1.0 and len(observed_ids) != len(expected_ids):
            credit = 0.99
        return credit, {
            "axis": axis,
            "side": side,
            "expected_evidence_ids": sorted(expected_ids),
            "handled_evidence_ids": list(observed_ids),
            "unsupported_evidence_ids": sorted(filed_junk),
            "recall": recall,
            "precision": precision,
        }

    if axis == "payment_state_depth":
        expected_reads = (
            "payment_history",
            "ledger_history",
            *(("packing_history",) if case.axis_value == 4 else ()),
        )
        actor_reads = tuple(row.resource for row in verified.reads if row.actor_id == actor_id)
        nonempty = sum(
            1
            for resource in expected_reads
            if any(
                row.actor_id == actor_id and row.resource == resource and bool(row.records)
                for row in verified.reads
            )
        )
        read_credit = min(
            _sequence_credit(expected_reads, actor_reads),
            nonempty / len(expected_reads),
        )
        shipment_credit = (
            float(shipment_read.get("verified") is True) if case.axis_value == 4 else 0.0
        )
        layers = len(expected_reads) + (1 if case.axis_value == 4 else 0)
        credit = (read_credit * len(expected_reads) + shipment_credit) / layers
        return credit, {
            "axis": axis,
            "expected_after_sales_reads": list(expected_reads),
            "observed_after_sales_reads": list(actor_reads),
            "nonempty_read_count": nonempty,
            "shipment_read": dict(shipment_read),
        }

    if axis == "affected_ledger_count":
        requested = tuple(
            str(row.intent.get("order_id", ""))
            for row in actor_requests
            if row.operation == "request_ledger_reconciliation"
        )
        expected = case.order_ids
        if not expected:
            raise RuntimeEvidenceError("ledger difficulty has no expected orders")
        credit = _sequence_credit(expected, requested)
        return credit, {
            "axis": axis,
            "evaluated_actor_id": actor_id,
            "expected_order_ids": list(expected),
            "requested_order_ids": list(requested),
        }
    raise RuntimeEvidenceError(f"unsupported T7 difficulty axis {axis!r}")


def _result_value(
    row: VerifiedAfterSalesRequest | VerifiedAfterSalesServiceOperation | None,
) -> dict[str, Any] | None:
    if row is None:
        return None
    value = row.result_record.get("value")
    if not isinstance(value, Mapping):
        raise RuntimeEvidenceError("verified after-sales result has no value")
    return dict(value)


def _refund_receipt_matches(
    case: _CaseT7,
    receipt: Receipt | None,
    expected_amount: int,
) -> bool:
    return bool(
        receipt is not None
        and str(receipt.order_id) == case.order_id
        and str(receipt.buyer_id) == _BUYER_ID
        and str(receipt.merchant_id) == _MERCHANT_ID
        and str(receipt.sku_id) == case.original_sku_id
        and receipt.qty == 1
        and receipt.price.currency == "USD"
        and receipt.effect == "refund"
        and _receipt_minor_units(receipt) == expected_amount
    )


def _timeline_transition_matches(
    evidence: RuntimeEvidenceBundleV2,
    order_id: str,
    *,
    returned: bool,
    refunded: bool,
) -> bool:
    initial_rows = _unique_rows(
        _world_rows(evidence, "order_timelines", initial=True),
        key="order_id",
        label="initial order timelines",
    )
    final_rows = _unique_rows(
        _world_rows(evidence, "order_timelines", initial=False),
        key="order_id",
        label="final order timelines",
    )
    if set(initial_rows) != set(final_rows) or order_id not in initial_rows:
        return False
    expected = copy.deepcopy(initial_rows)
    actual = final_rows[order_id]
    returned_tick = actual.get("returned_at_tick")
    refunded_tick = actual.get("refunded_at_tick")
    if returned:
        if isinstance(returned_tick, bool) or not isinstance(returned_tick, int):
            return False
        expected[order_id]["returned_at_tick"] = returned_tick
    if refunded:
        if isinstance(refunded_tick, bool) or not isinstance(refunded_tick, int):
            return False
        expected[order_id]["refunded_at_tick"] = refunded_tick
    if returned and refunded and returned_tick > refunded_tick:
        return False
    return final_rows == expected


def _cancel_outcome(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedAfterSalesEvidence,
    clauses: dict[str, bool],
    detail: dict[str, Any],
) -> None:
    terminal = _request_for(verified, "cancel_paid_order", case.order_id)
    initial_orders = _orders(evidence, initial=True)
    final_orders = _orders(evidence, initial=False)
    initial_order = initial_orders.get(case.order_id)
    expected_orders = copy.deepcopy(initial_orders)
    if case.order_id in expected_orders:
        expected_orders[case.order_id]["state"] = "cancelled"
    initial_inventory = _world_mapping(evidence, "inventory", initial=True)
    final_inventory = _world_mapping(evidence, "inventory", initial=False)
    expected_inventory = _expected_inventory(
        initial_inventory,
        {case.original_sku_id: -1},
    )
    initial_payments = _payment_history(evidence, case.order_id, initial=True)
    final_payments = _payment_history(evidence, case.order_id, initial=False)

    _append_clause(clauses, "cancel_commit_exists", terminal is not None)
    _append_clause(clauses, "order_cancelled_exactly", final_orders == expected_orders)
    _append_clause(
        clauses,
        "inventory_reservation_released_exactly",
        expected_inventory is not None and final_inventory == expected_inventory,
    )
    if case.initial_payment_stage == "authorized":
        expected_tables = (
            "orders",
            "inventory",
            "payment_states",
            "after_sales_records",
            "after_sales_records",
            "authority_operations",
            "logical_time",
        )
        _append_clause(
            clauses,
            "authorized_payment_voided_exactly",
            _payment_resolution_matches(
                initial_payments,
                final_payments,
                state="voided",
                refunded_amount=0,
                refund_receipt=None,
            ),
        )
        _append_clause(
            clauses,
            "no_refund_ledger_for_void",
            _world_rows(evidence, "ledger", initial=True)
            == _world_rows(evidence, "ledger", initial=False),
        )
        _append_clause(
            clauses,
            "no_packing_side_effect_for_void",
            _world_rows(evidence, "packing_records", initial=True)
            == _world_rows(evidence, "packing_records", initial=False),
        )
    else:
        expected_tables = (
            "orders",
            "inventory",
            "ledger",
            "payment_states",
            "packing_records",
            "after_sales_records",
            "after_sales_records",
            "authority_operations",
            "logical_time",
        )
        receipt, one_new_receipt = _new_refund_receipt(evidence, case.order_id)
        expected_amount = int(initial_payments[-1].get("captured_amount", -1))
        initial_packing = _packing_history(evidence, case.order_id, initial=True)
        final_packing = _packing_history(evidence, case.order_id, initial=False)
        packing_matches = bool(
            len(final_packing) == len(initial_packing) + 1
            and final_packing[:-1] == initial_packing
            and initial_packing
            and final_packing[-1].get("state") == "cancelled"
            and final_packing[-1].get("version") == int(initial_packing[-1].get("version", 0)) + 1
            and final_packing[-1].get("previous_digest") == initial_packing[-1].get("record_digest")
            and final_packing[-1].get("payment_digest") == final_payments[-1].get("record_digest")
        )
        _append_clause(
            clauses,
            "captured_refund_receipt_exact",
            one_new_receipt and _refund_receipt_matches(case, receipt, expected_amount),
        )
        _append_clause(
            clauses,
            "captured_payment_refunded_exactly",
            _payment_resolution_matches(
                initial_payments,
                final_payments,
                state="refunded",
                refunded_amount=expected_amount,
                refund_receipt=receipt,
            ),
        )
        _append_clause(clauses, "packing_cancelled_with_payment_link", packing_matches)
    observed_tables = _commit_tables(None if terminal is None else terminal.commit)
    _append_clause(
        clauses,
        "cancel_commit_exact_write_set",
        Counter(observed_tables) == Counter(expected_tables),
    )
    detail.update(
        {
            "outcome_kind": "paid_order_cancellation",
            "initial_order_state": None if initial_order is None else initial_order.get("state"),
            "final_order_state": (
                None
                if case.order_id not in final_orders
                else final_orders[case.order_id].get("state")
            ),
            "expected_commit_tables": list(expected_tables),
            "observed_commit_tables": list(observed_tables),
        }
    )


def _return_authorization_outcome(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedAfterSalesEvidence,
    clauses: dict[str, bool],
    detail: dict[str, Any],
) -> None:
    # A correct refusal is a completed disposition, not a failure to reach one.
    # The lane's terminal operation is therefore whichever decision the request
    # was actually owed.
    granted = case.entitled
    operation = "authorize_return" if granted else "deny_return"
    terminal = _request_for(verified, operation, case.order_id)
    value = _result_value(terminal)
    initial_payment = _payment_history(evidence, case.order_id, initial=True)[-1]
    expected_tables = (
        "after_sales_records",
        "authority_operations",
        "logical_time",
    )
    observed_tables = _commit_tables(None if terminal is None else terminal.commit)
    _append_clause(clauses, "return_authorization_commit_exists", terminal is not None)
    _append_clause(
        clauses,
        "return_authorization_value_exact",
        value is not None
        and value.get("outcome") == ("authorized" if granted else "denied")
        and value.get("actor_id") == _MERCHANT_ID
        and (
            not granted
            or (
                value.get("authorized_qty") == 1
                and value.get("authorized_amount") == initial_payment.get("captured_amount")
            )
        )
        and isinstance(value.get("binding"), Mapping)
        and value["binding"].get("order_id") == case.order_id,
    )
    _append_clause(
        clauses,
        "authorization_has_no_premature_commercial_effect",
        _same_physical_tables(
            evidence,
            (
                "orders",
                "inventory",
                "payment_states",
                "ledger",
                "order_timelines",
                "packing_records",
                "exchanges",
            ),
        ),
    )
    _append_clause(
        clauses,
        "authorization_commit_exact_write_set",
        Counter(observed_tables) == Counter(expected_tables),
    )
    detail.update(
        {
            "outcome_kind": "return_authorization",
            "expected_commit_tables": list(expected_tables),
            "observed_commit_tables": list(observed_tables),
        }
    )


def _full_return_refund_outcome(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedAfterSalesEvidence,
    clauses: dict[str, bool],
    detail: dict[str, Any],
) -> None:
    terminal = _request_for(verified, "approve_refund", case.order_id)
    value = _result_value(terminal)
    initial_orders = _orders(evidence, initial=True)
    final_orders = _orders(evidence, initial=False)
    expected_orders = copy.deepcopy(initial_orders)
    if case.order_id in expected_orders:
        expected_orders[case.order_id]["state"] = "refunded"
    initial_inventory = _world_mapping(evidence, "inventory", initial=True)
    expected_inventory = _expected_inventory(
        initial_inventory,
        {case.original_sku_id: -1},
    )
    final_inventory = _world_mapping(evidence, "inventory", initial=False)
    initial_payments = _payment_history(evidence, case.order_id, initial=True)
    final_payments = _payment_history(evidence, case.order_id, initial=False)
    expected_amount = int(initial_payments[-1].get("captured_amount", -1))
    receipt, one_new_receipt = _new_refund_receipt(evidence, case.order_id)
    expected_tables = (
        "orders",
        "inventory",
        "ledger",
        "payment_states",
        "order_timelines",
        "after_sales_records",
        "authority_operations",
        "logical_time",
    )
    observed_tables = _commit_tables(None if terminal is None else terminal.commit)

    _append_clause(clauses, "refund_commit_exists", terminal is not None)
    _append_clause(
        clauses,
        "refund_decision_exact",
        value is not None
        and value.get("outcome") == "approved"
        and value.get("approved_amount") == expected_amount
        and isinstance(value.get("binding"), Mapping)
        and value["binding"].get("order_id") == case.order_id,
    )
    _append_clause(clauses, "order_refunded_exactly", final_orders == expected_orders)
    _append_clause(
        clauses,
        "returned_inventory_released_exactly",
        expected_inventory is not None and final_inventory == expected_inventory,
    )
    _append_clause(
        clauses,
        "full_refund_receipt_exact",
        one_new_receipt and _refund_receipt_matches(case, receipt, expected_amount),
    )
    _append_clause(
        clauses,
        "full_payment_refund_exact",
        _payment_resolution_matches(
            initial_payments,
            final_payments,
            state="refunded",
            refunded_amount=expected_amount,
            refund_receipt=receipt,
        ),
    )
    _append_clause(
        clauses,
        "return_and_refund_timeline_exact",
        _timeline_transition_matches(
            evidence,
            case.order_id,
            returned=True,
            refunded=True,
        ),
    )
    _append_clause(
        clauses,
        "refund_commit_exact_write_set",
        Counter(observed_tables) == Counter(expected_tables),
    )
    _append_clause(
        clauses,
        "refund_has_no_exchange_or_packing_side_effect",
        _same_physical_tables(evidence, ("packing_records", "exchanges")),
    )
    detail.update(
        {
            "outcome_kind": "returned_item_full_refund",
            "expected_refund_amount": expected_amount,
            "expected_commit_tables": list(expected_tables),
            "observed_commit_tables": list(observed_tables),
        }
    )


def _denied_exchange_outcome(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedAfterSalesEvidence,
    clauses: dict[str, bool],
    detail: dict[str, Any],
) -> None:
    """Score a refused swap: the case closes and no replacement ships."""

    terminal = _request_for(verified, "deny_exchange", case.order_id)
    value = _result_value(terminal)
    initial_exchanges = _world_rows(evidence, "exchanges", initial=True)
    final_exchanges = _world_rows(evidence, "exchanges", initial=False)
    _append_clause(clauses, "exchange_completion_commit_exists", terminal is not None)
    _append_clause(
        clauses,
        "exchange_case_closed_exactly",
        value is not None
        and value.get("state") == "denied"
        and value.get("replacement_sku_id") == case.requested_replacement_sku_id
        and value.get("completion_order_digest") is None,
    )
    # The order came back and stays back.  What must not happen is a second
    # order, a second SKU leaving stock, or money moving.
    _append_clause(clauses, "no_replacement_order_created", final_exchanges == initial_exchanges)
    _append_clause(
        clauses,
        "denied_exchange_has_no_payment_or_refund",
        _same_physical_tables(evidence, ("payment_states", "ledger")),
    )
    _append_clause(
        clauses,
        "denied_exchange_return_timeline_exact",
        _timeline_transition_matches(
            evidence,
            case.order_id,
            returned=True,
            refunded=False,
        ),
    )
    detail.update({"outcome_kind": "refused_replacement_exchange"})


def _exchange_outcome(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedAfterSalesEvidence,
    clauses: dict[str, bool],
    detail: dict[str, Any],
) -> None:
    request = _request_for(verified, "request_exchange", case.order_id)
    task_expected_sku = _expected_replacement_sku(case)
    if not case.entitled:
        _denied_exchange_outcome(case, evidence, verified, clauses, detail)
        detail["task_expected_replacement_sku"] = task_expected_sku
        detail["requested_replacement_sku"] = case.requested_replacement_sku_id
        return
    terminal = _request_for(verified, "complete_exchange", case.order_id)
    value = _result_value(terminal)
    requested_sku = None if request is None else request.intent.get("replacement_sku_id")
    actual_sku = requested_sku if isinstance(requested_sku, str) and requested_sku else None
    initial_orders = _orders(evidence, initial=True)
    final_orders = _orders(evidence, initial=False)
    initial_exchanges = _world_rows(evidence, "exchanges", initial=True)
    final_exchanges = _world_rows(evidence, "exchanges", initial=False)
    new_exchanges = (
        final_exchanges[len(initial_exchanges) :]
        if final_exchanges[: len(initial_exchanges)] == initial_exchanges
        else ()
    )
    exchange = new_exchanges[0] if len(new_exchanges) == 1 else None
    expected_orders = copy.deepcopy(initial_orders)
    if case.order_id in expected_orders:
        expected_orders[case.order_id]["state"] = "exchanged"
    if exchange is not None and case.order_id in initial_orders:
        replacement_order = copy.deepcopy(initial_orders[case.order_id])
        replacement_order["order_id"] = exchange.get("replacement_order_id")
        replacement_order["sku_id"] = actual_sku
        replacement_order["state"] = "settled"
        if isinstance(replacement_order.get("order_id"), str):
            expected_orders[str(replacement_order["order_id"])] = replacement_order
    initial_inventory = _world_mapping(evidence, "inventory", initial=True)
    expected_inventory = (
        None
        if actual_sku is None
        else _expected_inventory(
            initial_inventory,
            {
                case.original_sku_id: -1,
                actual_sku: 1,
            },
        )
    )
    expected_tables = (
        "orders",
        "orders",
        "inventory",
        "inventory",
        "exchanges",
        "after_sales_records",
        "authority_operations",
        "logical_time",
    )
    observed_tables = _commit_tables(None if terminal is None else terminal.commit)

    _append_clause(clauses, "exchange_completion_commit_exists", terminal is not None)
    _append_clause(
        clauses,
        "exchange_case_completed_exactly",
        value is not None
        and actual_sku is not None
        and value.get("state") == "completed"
        and value.get("replacement_sku_id") == actual_sku
        and isinstance(value.get("completion_order_digest"), str)
        and bool(value.get("completion_order_digest")),
    )
    _append_clause(
        clauses,
        "exchange_link_exact",
        exchange is not None
        and len(final_exchanges) == len(initial_exchanges) + 1
        and exchange.get("exchange_id") == (None if value is None else value.get("case_id"))
        and exchange.get("original_order_id") == case.order_id
        and exchange.get("original_sku_id") == case.original_sku_id
        and exchange.get("replacement_sku_id") == actual_sku
        and exchange.get("buyer_id") == _BUYER_ID
        and exchange.get("merchant_id") == _MERCHANT_ID
        and exchange.get("qty") == 1,
    )
    _append_clause(clauses, "exchange_orders_exact", final_orders == expected_orders)
    _append_clause(
        clauses,
        "exchange_inventory_exact",
        expected_inventory is not None
        and _world_mapping(evidence, "inventory", initial=False) == expected_inventory,
    )
    _append_clause(
        clauses,
        "exchange_has_no_second_payment_or_refund",
        _same_physical_tables(evidence, ("payment_states", "ledger")),
    )
    _append_clause(
        clauses,
        "exchange_return_timeline_exact",
        _timeline_transition_matches(
            evidence,
            case.order_id,
            returned=True,
            refunded=False,
        ),
    )
    _append_clause(
        clauses,
        "exchange_commit_exact_write_set",
        Counter(observed_tables) == Counter(expected_tables),
    )
    detail.update(
        {
            "outcome_kind": "replacement_exchange",
            "task_expected_replacement_sku": task_expected_sku,
            "observed_requested_replacement_sku": actual_sku,
            "replacement_order_id": (
                None if exchange is None else exchange.get("replacement_order_id")
            ),
            "expected_commit_tables": list(expected_tables),
            "observed_commit_tables": list(observed_tables),
        }
    )


def _dispute_outcome(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedAfterSalesEvidence,
    clauses: dict[str, bool],
    detail: dict[str, Any],
) -> None:
    task_expected_ruling = case.expected_ruling_operation
    if task_expected_ruling is None:
        raise RuntimeEvidenceError("dispute task has no expected ruling")

    ruling_semantics = {
        "rule_for_filer": ("claim_upheld", _BUYER_ID),
        "rule_split": ("split", _BUYER_ID),
        "rule_for_respondent": ("claim_denied", _MERCHANT_ID),
    }
    ruling_candidates = tuple(
        row
        for row in verified.service_operations
        if row.operation in ruling_semantics
        and row.commit.get("subject_id") == case.order_id
    )
    if len(ruling_candidates) > 1:
        raise RuntimeEvidenceError("dispute has multiple accepted adjudication results")
    ruling = ruling_candidates[0] if ruling_candidates else None
    actual_ruling_operation = None if ruling is None else ruling.operation
    ruling_value = _result_value(ruling)

    resolution_candidates = tuple(
        row
        for row in verified.requests
        if row.operation in {"approve_refund", "deny_refund"}
        and row.commit.get("subject_id") == case.order_id
    )
    if len(resolution_candidates) > 1:
        raise RuntimeEvidenceError("dispute has multiple accepted refund decisions")
    resolution = resolution_candidates[0] if resolution_candidates else None
    actual_resolution_operation = None if resolution is None else resolution.operation
    resolution_value = _result_value(resolution)

    initial_payments = _payment_history(evidence, case.order_id, initial=True)
    final_payments = _payment_history(evidence, case.order_id, initial=False)
    captured_amount = int(initial_payments[-1].get("captured_amount", -1))
    ruling_refund_amount = {
        "rule_for_filer": captured_amount,
        "rule_split": captured_amount * 5_000 // 10_000,
        "rule_for_respondent": 0,
    }.get(actual_ruling_operation)
    actual_ruling_semantics = ruling_semantics.get(actual_ruling_operation)
    resolution_approved = actual_resolution_operation == "approve_refund"
    committed_refund_amount = (
        ruling_refund_amount
        if resolution_approved and ruling_refund_amount is not None
        else 0
    )
    ruling_tables = (
        "rulings",
        "disputes",
        "after_sales_records",
        "after_sales_records",
        "authority_operations",
        "logical_time",
    )
    observed_ruling_tables = _commit_tables(None if ruling is None else ruling.commit)
    observed_resolution_tables = _commit_tables(None if resolution is None else resolution.commit)

    _append_clause(clauses, "trusted_ruling_commit_exists", ruling is not None)
    _append_clause(
        clauses,
        "trusted_ruling_value_exact",
        ruling_value is not None
        and actual_ruling_semantics is not None
        and ruling_value.get("outcome") == actual_ruling_semantics[0]
        and ruling_value.get("winner_id") == actual_ruling_semantics[1]
        and ruling_value.get("refund_amount") == ruling_refund_amount
        and isinstance(ruling_value.get("binding"), Mapping)
        and ruling_value["binding"].get("order_id") == case.order_id,
    )
    _append_clause(
        clauses,
        "trusted_ruling_commit_exact_write_set",
        Counter(observed_ruling_tables) == Counter(ruling_tables),
    )
    _append_clause(clauses, "dispute_resolution_commit_exists", resolution is not None)
    _append_clause(
        clauses,
        "dispute_refund_decision_exact",
        resolution_value is not None
        and actual_ruling_operation is not None
        and actual_resolution_operation in {"approve_refund", "deny_refund"}
        and resolution_value.get("outcome")
        == ("approved" if resolution_approved else "denied")
        and resolution_value.get("approved_amount") == committed_refund_amount
        and isinstance(resolution_value.get("binding"), Mapping)
        and resolution_value["binding"].get("order_id") == case.order_id,
    )

    if not resolution_approved or committed_refund_amount == 0:
        resolution_tables = (
            "after_sales_records",
            "authority_operations",
            "logical_time",
        )
        _append_clause(
            clauses,
            "zero_effect_resolution_has_no_commercial_side_effect",
            _same_physical_tables(
                evidence,
                (
                    "orders",
                    "inventory",
                    "payment_states",
                    "ledger",
                    "order_timelines",
                    "packing_records",
                    "exchanges",
                ),
            ),
        )
    else:
        receipt, one_new_receipt = _new_refund_receipt(evidence, case.order_id)
        full_refund = committed_refund_amount == captured_amount
        resolution_tables = (
            *(("orders",) if full_refund else ()),
            "ledger",
            "payment_states",
            *(("order_timelines",) if full_refund else ()),
            "after_sales_records",
            "authority_operations",
            "logical_time",
        )
        _append_clause(
            clauses,
            "dispute_refund_receipt_exact",
            one_new_receipt
            and _refund_receipt_matches(case, receipt, committed_refund_amount),
        )
        _append_clause(
            clauses,
            "dispute_payment_resolution_exact",
            _payment_resolution_matches(
                initial_payments,
                final_payments,
                state="refunded" if full_refund else "partially_refunded",
                refunded_amount=committed_refund_amount,
                refund_receipt=receipt,
            ),
        )
        initial_orders = _orders(evidence, initial=True)
        expected_orders = copy.deepcopy(initial_orders)
        if full_refund:
            expected_orders[case.order_id]["state"] = "refunded"
        _append_clause(
            clauses,
            "dispute_order_state_exact",
            _orders(evidence, initial=False) == expected_orders,
        )
        _append_clause(
            clauses,
            "dispute_never_restocks_inventory",
            _world_mapping(evidence, "inventory", initial=True)
            == _world_mapping(evidence, "inventory", initial=False),
        )
        _append_clause(
            clauses,
            "dispute_refund_timeline_exact",
            _timeline_transition_matches(
                evidence,
                case.order_id,
                returned=False,
                refunded=full_refund,
            ),
        )
    _append_clause(
        clauses,
        "dispute_resolution_commit_exact_write_set",
        Counter(observed_resolution_tables) == Counter(resolution_tables),
    )
    detail.update(
        {
            "outcome_kind": "dispute_ruling_and_resolution",
            "task_expected_ruling_operation": task_expected_ruling,
            "observed_ruling_operation": actual_ruling_operation,
            "ruling_refund_amount": ruling_refund_amount,
            "observed_resolution_operation": actual_resolution_operation,
            "committed_refund_amount": committed_refund_amount,
            "expected_ruling_commit_tables": list(ruling_tables),
            "observed_ruling_commit_tables": list(observed_ruling_tables),
            "expected_resolution_commit_tables": list(resolution_tables),
            "observed_resolution_commit_tables": list(observed_resolution_tables),
        }
    )


def _ledger_reconciliation_outcome(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedAfterSalesEvidence,
    clauses: dict[str, bool],
    detail: dict[str, Any],
) -> None:
    final_ledger = _world_rows(evidence, "ledger", initial=False)
    observed_tables: dict[str, list[str]] = {}
    exact_results = True
    exact_commits = True
    accepted_service_integrity: dict[str, bool] = {}
    found_orders: list[str] = []
    for order_id in case.order_ids:
        service = _service_for(
            verified,
            "complete_ledger_reconciliation",
            order_id,
        )
        if service is None:
            exact_results = False
            exact_commits = False
            observed_tables[order_id] = []
            continue
        value = _result_value(service)
        receipts = tuple(
            _receipt_from_row(row) for row in final_ledger if row.get("order_id") == order_id
        )
        txn_ids = tuple(sorted(str(row.txn_id) for row in receipts))
        gross = sum(_receipt_minor_units(row) for row in receipts if row.effect == "charge")
        refunded = sum(_receipt_minor_units(row) for row in receipts if row.effect == "refund")
        currencies = {row.price.currency for row in receipts}
        result_exact = bool(
            value is not None
            and isinstance(value.get("binding"), Mapping)
            and value["binding"].get("order_id") == order_id
            and tuple(sorted(value.get("source_txn_ids", ()))) == txn_ids
            and value.get("gross_amount") == gross
            and value.get("refund_amount") == refunded
            and value.get("net_amount") == gross - refunded
            and len(currencies) == 1
            and value.get("currency") in currencies
        )
        exact_results = exact_results and result_exact
        expected_tables = (
            *("after_sales_records" for _ in range(len(receipts) + 1)),
            "authority_operations",
            "logical_time",
        )
        tables = _commit_tables(service.commit)
        observed_tables[order_id] = list(tables)
        commit_exact = Counter(tables) == Counter(expected_tables)
        exact_commits = exact_commits and commit_exact
        accepted_service_integrity[order_id] = result_exact and commit_exact
        found_orders.append(order_id)

    result_values = _after_sales_values(evidence, "ledger_reconciliation_results")
    _append_clause(
        clauses,
        "one_reconciliation_service_result_per_order",
        tuple(found_orders) == case.order_ids and len(result_values) == case.ledger_source_count,
    )
    _append_clause(clauses, "reconciliation_sources_and_totals_exact", exact_results)
    _append_clause(clauses, "reconciliation_commits_exact_write_sets", exact_commits)
    _append_clause(
        clauses,
        "reconciliation_has_no_commercial_side_effect",
        _same_physical_tables(
            evidence,
            (
                "orders",
                "inventory",
                "payment_states",
                "ledger",
                "order_timelines",
                "packing_records",
                "exchanges",
            ),
        ),
    )
    detail.update(
        {
            "outcome_kind": "ledger_reconciliation",
            "expected_order_ids": list(case.order_ids),
            "observed_service_order_ids": found_orders,
            "observed_commit_tables": observed_tables,
            "accepted_service_integrity": accepted_service_integrity,
        }
    )


def _commercial_outcome_evidence(
    case: _CaseT7,
    evidence: RuntimeEvidenceBundleV2,
    verified: VerifiedAfterSalesEvidence,
) -> tuple[float, dict[str, Any]]:
    """Join the terminal workflow to exact replayed commercial state.

    Operation names are routing evidence only.  Credit here comes from exact
    World rows, append-only histories, receipt digests, and the terminal
    transaction's complete table write set.
    """

    clauses: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    if case.lane in {"buyer_cancel", "merchant_cancel"}:
        _cancel_outcome(case, evidence, verified, clauses, detail)
    elif case.lane in {"buyer_return_refund", "merchant_refund"}:
        _full_return_refund_outcome(case, evidence, verified, clauses, detail)
    elif case.lane == "merchant_return_authorization":
        _return_authorization_outcome(case, evidence, verified, clauses, detail)
    elif case.lane in {"buyer_exchange", "merchant_exchange"}:
        _exchange_outcome(case, evidence, verified, clauses, detail)
    elif case.lane in {"buyer_dispute", "merchant_dispute"}:
        _dispute_outcome(case, evidence, verified, clauses, detail)
    elif case.lane == "merchant_ledger_close":
        _ledger_reconciliation_outcome(case, evidence, verified, clauses, detail)
    else:
        raise RuntimeEvidenceError(f"unsupported T7 lane {case.lane!r}")
    if not clauses:
        raise RuntimeEvidenceError("T7 commercial outcome has no exact clauses")
    # The rubric measures whether the agent completed the intended business
    # outcome.  Exact Platform/World joins and commit write sets are hard
    # environment gates and therefore never contribute fractional points.
    if case.lane in {"buyer_cancel", "merchant_cancel"}:
        completed = clauses["cancel_commit_exists"]
        integrity_names = tuple(name for name in clauses if name != "cancel_commit_exists")
    elif case.lane == "merchant_return_authorization":
        completed = clauses["return_authorization_commit_exists"]
        integrity_names = tuple(
            name for name in clauses if name != "return_authorization_commit_exists"
        )
    elif case.lane in {"buyer_return_refund", "merchant_refund"}:
        completed = clauses["refund_commit_exists"]
        integrity_names = tuple(name for name in clauses if name != "refund_commit_exists")
    elif case.lane in {"buyer_exchange", "merchant_exchange"}:
        completed = clauses["exchange_completion_commit_exists"]
        integrity_names = tuple(
            name for name in clauses if name != "exchange_completion_commit_exists"
        )
    elif case.lane in {"buyer_dispute", "merchant_dispute"}:
        completed = bool(
            clauses["trusted_ruling_commit_exists"]
            and clauses["dispute_resolution_commit_exists"]
            and clauses["dispute_refund_decision_exact"]
        )
        integrity_names = tuple(
            name
            for name in clauses
            if name
            not in {
                "trusted_ruling_commit_exists",
                "dispute_resolution_commit_exists",
            }
        )
        if clauses["trusted_ruling_commit_exists"]:
            failed_ruling = tuple(
                name
                for name in (
                    "trusted_ruling_value_exact",
                    "trusted_ruling_commit_exact_write_set",
                )
                if not clauses[name]
            )
            if failed_ruling:
                raise RuntimeBenchmarkIntegrityError(
                    "T7 accepted dispute ruling is not faithfully committed: "
                    + ", ".join(failed_ruling)
                )
        if clauses["dispute_resolution_commit_exists"]:
            failed_resolution = tuple(
                name
                for name in integrity_names
                if not name.startswith("trusted_ruling_") and not clauses[name]
            )
            if failed_resolution:
                raise RuntimeBenchmarkIntegrityError(
                    "T7 accepted dispute resolution is not faithfully committed: "
                    + ", ".join(failed_resolution)
                )
    elif case.lane == "merchant_ledger_close":
        completed = clauses["one_reconciliation_service_result_per_order"]
        accepted_integrity = detail.get("accepted_service_integrity", {})
        if isinstance(accepted_integrity, Mapping):
            failed_services = sorted(
                str(order_id) for order_id, passed in accepted_integrity.items() if not passed
            )
            if failed_services:
                raise RuntimeBenchmarkIntegrityError(
                    "T7 accepted ledger reconciliation has an unfaithful result "
                    "binding or World write set: " + ", ".join(failed_services)
                )
        integrity_names = ("reconciliation_has_no_commercial_side_effect",)
    else:  # pragma: no cover - lane dispatch above is exhaustive
        raise RuntimeEvidenceError(f"unsupported T7 lane {case.lane!r}")

    if completed:
        failed_integrity = tuple(name for name in integrity_names if not clauses[name])
        if failed_integrity:
            raise RuntimeBenchmarkIntegrityError(
                "T7 accepted terminal operation is not faithfully committed by "
                "Platform and World: " + ", ".join(failed_integrity)
            )

    detail["clause_results"] = dict(clauses)
    detail["passed_clause_count"] = sum(clauses.values())
    detail["total_clause_count"] = len(clauses)
    detail["business_outcome_completed"] = completed
    detail["environment_integrity_checks"] = list(integrity_names)
    return float(completed), detail


def score_t7_runtime(
    task_id: str,
    evidence: RuntimeEvidenceBundleV2,
) -> RuntimeTaskScoreV3:
    """Score T7 only from replay-verified Runtime, Platform, and World evidence."""

    require_runtime_benchmark_integrity_v2(evidence)
    final_tables = evidence.final_world.get("tables")
    if not isinstance(final_tables, Mapping) or canonical_sha256(final_tables) != (
        evidence.final_digest
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T7 final World snapshot does not match its captured evidence digest"
        )
    previous_commit_sha256: str | None = None
    for commit in evidence.world_events:
        observed_commit_sha256 = commit.get("commit_sha256")
        digest_payload = dict(commit)
        digest_payload.pop("commit_sha256", None)
        if (
            commit.get("previous_commit_sha256") != previous_commit_sha256
            or not isinstance(observed_commit_sha256, str)
            or canonical_sha256(digest_payload) != observed_commit_sha256
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T7 World commit journal does not match its captured hash chain"
            )
        previous_commit_sha256 = observed_commit_sha256
    case = _case_for_t7(task_id)
    require_frozen_scenario_fixture_v2(
        evidence,
        _prepared_scenario_for_t7(task_id),
        family="T7",
    )
    _require_order_stage_fixture(case, evidence)
    ownership_violations = runtime_evidence_core_ownership_violations_v2(evidence)
    if ownership_violations:
        raise RuntimeBenchmarkIntegrityError(
            "CommerceWorld core ownership violation: " + "; ".join(ownership_violations)
        )
    has_after_sales_activity = any(
        row.get("platform_endpoint") == "platform:after-sales"
        for row in evidence.platform_decisions
    ) or bool(evidence.world_events)
    try:
        candidate = evidence.verified_operation_evidence(
            AFTER_SALES_EVIDENCE_CONTRACT,
            options={
                "allowed_order_ids": list(case.order_ids),
                "allow_rejected": True,
            },
        )
        if not isinstance(candidate, VerifiedAfterSalesEvidence):
            raise RuntimeEvidenceError(
                "after-sales evidence contract returned an unexpected result"
            )
    except RuntimeEvidenceError as exc:
        if has_after_sales_activity:
            raise RuntimeBenchmarkIntegrityError(
                "T7 after-sales authority evidence is structurally invalid"
            ) from exc
        candidate = VerifiedAfterSalesEvidence(requests=())
    verified = candidate
    try:
        shipment_read = _verified_shipment_read(case, evidence)
    except RuntimeEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError(
            "T7 shipment-read evidence is structurally invalid"
        ) from exc

    actor_id = case.evaluated_actor_id
    actor_operations = tuple(
        row.operation
        for row in verified.requests
        if row.exchange.decision.get("actor_id") == actor_id
    )
    claimed_commit_ids = {
        str(row.commit.get("commit_id"))
        for row in (*verified.requests, *verified.service_operations)
    }
    transaction_commits = tuple(
        row for row in evidence.world_events if row.get("commit_kind") == "transaction"
    )
    unexpected_commits = tuple(
        str(row.get("commit_id"))
        for row in transaction_commits
        if str(row.get("commit_id")) not in claimed_commit_ids
    )
    if unexpected_commits:
        raise RuntimeBenchmarkIntegrityError(
            "T7 after-sales evidence leaves transaction commits outside the authority graph: "
            + ", ".join(unexpected_commits)
        )
    try:
        _require_accepted_operation_write_sets(case, evidence, verified)
    except (RuntimeEvidenceError, ValueError, TypeError) as exc:
        raise RuntimeBenchmarkIntegrityError(
            "T7 accepted operation is not faithfully committed by World"
        ) from exc
    _require_triggered_environment_closure(case, evidence, verified)
    actor_attempts, rejected_attempts = _evaluated_actor_attempt_sequence(
        actor_id,
        verified,
    )
    operation_credit = _sequence_credit(case.evaluated_operations, actor_attempts)
    expected_operation_counts = Counter(case.evaluated_operations)
    observed_operation_counts = Counter(actor_attempts)
    selected_operation_count = sum(
        min(expected_count, observed_operation_counts[operation])
        for operation, expected_count in expected_operation_counts.items()
    )
    operation_selection_credit = selected_operation_count / max(
        len(case.evaluated_operations),
        1,
    )
    difficulty: RuntimeRubricCheckV2 | None = None
    if case.axis_name != "order_stage":
        try:
            difficulty_credit, difficulty_detail = _difficulty_evidence(
                case,
                evidence,
                verified,
                actor_operations=actor_operations,
                shipment_read=shipment_read,
            )
        except (RuntimeEvidenceError, ValueError, TypeError) as exc:
            raise RuntimeBenchmarkIntegrityError(
                "T7 difficulty evidence cannot be derived from authoritative World state"
            ) from exc
        difficulty = _check(
            "difficulty_evidence_grounded",
            0.35,
            difficulty_credit,
            difficulty_detail,
        )
    try:
        outcome_credit, outcome_detail = _commercial_outcome_evidence(
            case,
            evidence,
            verified,
        )
    except (RuntimeEvidenceError, ValueError, TypeError, InvalidOperation) as exc:
        raise RuntimeBenchmarkIntegrityError(
            "T7 terminal commercial evidence is structurally invalid"
        ) from exc
    # The exact World joins inside the outcome remain hard environment gates --
    # they raise rather than deduct.  What is scored is whether the lifecycle
    # actually reached the disposition the order was entitled to: money moved
    # or did not move, stock returned or did not.  This used to be computed and
    # then thrown away, which left the two cancellation lanes with no check
    # beyond "did the actor name the right operation".

    # What the family is for is the disposition, not the typing.  Naming the
    # operations in the right order is necessary but it is the cheap part, so
    # it is the minority of the weight; the majority sits on the two checks
    # that can only be satisfied by having judged the case correctly -- the
    # lane's own decision evidence and the commercial state it left behind.
    hard_tier = is_hardened_task_v2(case.definition)
    graded_decision = difficulty is not None
    outcome_weight = 0.35 if graded_decision else 0.50
    if hard_tier:
        operation_weight = 0.20 if graded_decision else 0.35
        selection_weight = 0.10 if graded_decision else 0.15
    else:
        operation_weight = 0.30 if graded_decision else 0.50
        selection_weight = 0.0
    checks: list[RuntimeRubricCheckV2] = [
        _check(
            "evaluated_operations_completed",
            operation_weight,
            operation_credit,
            {
                "expected": list(case.evaluated_operations),
                "observed": list(actor_operations),
                "attempt_sequence": list(actor_attempts),
                "rejected_attempts": list(rejected_attempts),
                "matched": _sequence_match_count(
                    case.evaluated_operations,
                    actor_attempts,
                ),
                "operation_credit": operation_credit,
            },
        )
    ]
    if hard_tier:
        checks.append(
            _check(
                "operation_selection_coverage",
                selection_weight,
                operation_selection_credit,
                {
                    "expected_operation_counts": dict(expected_operation_counts),
                    "observed_operation_counts": dict(observed_operation_counts),
                    "matched_operation_count": selected_operation_count,
                    "expected_operation_count": len(case.evaluated_operations),
                },
            )
        )
    if difficulty is not None:
        checks.append(difficulty)
    checks.append(
        _check(
            "business_outcome_completed",
            outcome_weight,
            outcome_credit,
            outcome_detail,
        )
    )
    normalized_checks = renormalize_capability_checks_v2(tuple(checks))
    issues = (
        ()
        if all(row.credit == 1.0 for row in normalized_checks)
        else ("t7_runtime_lifecycle_contract_incomplete",)
    )
    return score_checks(case.definition, normalized_checks, issues=issues)


def _prepared_runtime_bundle_t7(task_id: str) -> RuntimeTaskBundleV2:
    """Build the complete bundle privately while the public family stays closed."""

    case = _case_for_t7(task_id)
    scenario = _prepared_scenario_for_t7(task_id)
    counterpart_id = _MERCHANT_ID if case.evaluated_actor_id == _BUYER_ID else _BUYER_ID
    semantic_hash = canonical_sha256(
        {
            "content": case.semantic_contract,
            "scenario_state": scenario.initial_state,
            "scenario_oracle": scenario.success_oracle,
        }
    )
    return RuntimeTaskBundleV2(
        task=case.definition,
        scenario=scenario,
        evaluated_actor_id=case.evaluated_actor_id,
        ideal_channel=lambda: _ideal_channel_t7(task_id),
        counterpart_channels={
            counterpart_id: lambda: _counterpart_channel_t7(task_id),
        },
        scorer=lambda runtime_evidence: score_t7_runtime(task_id, runtime_evidence),
        semantic_hash=semantic_hash,
        mutations=(
            RuntimeMutationV2(
                mutation_id=f"{task_id}:partial:01",
                channel=lambda: _ScriptedT7Channel(
                    _evaluated_steps(case, mutated=True),
                    actor_role=case.definition.evaluated_role,
                ),
                expected_changed_checks=_mutation_changed_checks(case),
            ),
        ),
    )


def scenario_for_t7(task_id: str) -> ScenarioSpec:
    """Build one validated T7 scenario for the real CommerceWorld runtime."""

    return _prepared_scenario_for_t7(task_id)


def runtime_bundle_t7(task_id: str) -> RuntimeTaskBundleV2:
    """Bind one T7 task to Episode, Platform, World, replay, and scoring."""

    return _prepared_runtime_bundle_t7(task_id)


def targeted_mutation_channel_t7(task_id: str) -> InferenceChannel:
    case = _case_for_t7(task_id)
    return _ScriptedT7Channel(
        _evaluated_steps(case, mutated=True),
        actor_role=case.definition.evaluated_role,
    )


def runtime_bundles_t7_ready() -> tuple[RuntimeTaskBundleV2, ...]:
    return tuple(runtime_bundle_t7(task_id) for task_id in T7_RUNTIME_READY_TASK_IDS)


def t7_runtime_semantic_hash(task_id: str) -> str:
    return canonical_sha256(_case_for_t7(task_id).semantic_contract)


__all__ = [
    "T7_RUNTIME_PENDING_TASK_IDS",
    "T7_RUNTIME_READY_TASK_IDS",
    "T7_RUNTIME_SCHEMA_V2",
    "runtime_bundle_t7",
    "runtime_bundles_t7_ready",
    "scenario_for_t7",
    "t7_runtime_content",
    "t7_runtime_semantic_hash",
    "targeted_mutation_channel_t7",
]
