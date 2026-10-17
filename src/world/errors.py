"""Exception hierarchy for the world package.

All world-mutation paths raise a subclass of ``WorldError``. Reads return
``None`` for missing rows rather than raising.
"""

from __future__ import annotations


class WorldError(Exception):
    """Base error for the world package."""


class TableNotFound(WorldError):
    """A read or write referenced a table that does not exist."""


class WriteNotAuthorized(WorldError):
    """A ``World.write`` was called with a ``by_action`` not permitted on this table.

    This is the structural prevention of "merchant updates its own reputation"
    described in WORLD_CLASSES.md §2.1.
    """


class InsufficientFunds(WorldError):
    """The buyer ledger balance is below the order total at settle time."""


class OutOfStock(WorldError):
    """Inventory for the requested SKU is below the requested quantity."""


class IdempotencyReplay(WorldError):
    """A second write reused an idempotency key; the world returns the original outcome.

    Note: this is **not** an error in the failure sense — it is signalled so the
    runtime can short-circuit and replay the recorded receipt rather than re-write.
    """


class IdempotencyConflict(WorldError):
    """An idempotency key was reused for a different operation or payload.

    Exact retries replay their original receipt. Reusing the same scoped key
    for another order, receipt, or transaction kind is rejected before any
    world mutation instead of returning an unrelated prior outcome.
    """


class CatalogMutationRejected(WorldError):
    """A valid catalog intent conflicts with authoritative catalog state.

    The exception message is a stable reason code.  It is separate from
    malformed schema and actor authorization failures so in-process and HTTP
    callers can preserve the same failure classification while World remains
    the owner of the catalog state machine.
    """


class AfterSalesReferenceRejected(WorldError):
    """An actor selected an unusable authoritative after-sales reference.

    The exception carries only a stable reason code.  It deliberately excludes
    actor-supplied order and evidence identifiers so the same rejection can be
    journaled across in-process and HTTP topologies without disclosing another
    actor's state.  Internal context, persistence, and validation defects must
    continue to use their existing error types instead of this class.
    """

    ALLOWED_REASON_CODES = frozenset(
        {
            "after_sales_order_not_found",
            "after_sales_evidence_not_found",
            "after_sales_evidence_stale",
            "after_sales_evidence_identity_mismatch",
            "after_sales_evidence_not_authorized",
            "after_sales_evidence_not_usable",
        }
    )

    def __init__(self, reason_code: str) -> None:
        if reason_code not in self.ALLOWED_REASON_CODES:
            raise ValueError("unsupported after-sales reference rejection reason")
        self.reason_code = reason_code
        super().__init__(reason_code)


class PartitionRead(WorldError):
    """A caller attempted to read a row outside its partition."""


class NoMerchantConsent(WorldError):
    """Settle attempted at a price the named merchant never agreed to.

    PSP requires a deterministic consent trail before writing a settled
    Order to World. Either the agreed_price matches the listing's
    list_price (aggregator/match-cert path implies consent), OR the audit
    log contains a ``commerce.accept_offer`` / ``commerce.counter_offer``
    envelope FROM the merchant for the same sku at the same price.

    This is the hard guard against an LLM buyer fabricating an
    agreed_price the merchant never agreed to. Raised before any world
    write so the simulation never enters an inconsistent state.
    """


class OrderNotSettleable(WorldError):
    """Settle attempted on an order whose state is not settleable.

    Only orders on the explicit settleable allow-list (PROPOSED / ACCEPTED) may
    settle. A CANCELLED order — or any state not on that allow-list — is refused
    before any world write (no inventory reserved, no ledger entry), so a
    cancelled order is never silently settled. Default-deny: an OrderState added
    later without classifying it as settleable is refused, not settled.
    """


class OrderNotRefundable(WorldError):
    """Refund attempted on an order that is not in a refundable state.

    Allowlist + default-deny (mirrors :class:`OrderNotSettleable`): only a
    SETTLED / DISPATCHED / RETURNED order can be refunded — you cannot refund
    what was never paid (PROPOSED / ACCEPTED) nor a CANCELLED order. An
    already-REFUNDED order is an idempotent replay, not an error.
    """


class ReturnWindowClosed(OrderNotRefundable):
    """A return authorization event would fall after its captured deadline.

    The deadline is computed exclusively from World-owned logical ticks.  A
    rejected request has zero state effect, including no clock advance.
    """


class LogicalTimeError(WorldError):
    """A caller attempted to forge, regress, or otherwise misuse World time."""


class LifecycleAuthorizationError(WorldError):
    """An actor attempted a lifecycle mutation for another party's order.

    Order transitions validate the full actor id, not only its role prefix.
    Platform services remain privileged coordinators, while a buyer or merchant
    may mutate only orders to which that exact actor is a party.
    """


class InvalidOrderTransition(WorldError):
    """An order lifecycle transition was attempted from an illegal state.

    Every transition uses an explicit allow-list and validates it before the
    order row is written. Illegal transitions therefore have zero world-state
    effect.
    """


class DisputeNotActionable(WorldError):
    """A dispute or ruling is invalid, conflicting, or no longer actionable.

    This includes disputes for unpaid orders, forged party identities,
    duplicate disputes for one order, and conflicting or unauthorized rulings.
    """


class OrderIdentityMismatch(WorldError):
    """Order, inventory, and receipt identities disagree.

    This defense-in-depth check prevents one merchant's SKU or receipt from
    being charged to a different buyer/merchant even when a caller bypasses the
    platform's consent preflight and invokes the World service directly.
    """


class FulfillmentNotActionable(WorldError):
    """A partial-fill/backorder allocation is invalid or conflicts with history.

    Allocation is single-shot per order. Exact retries replay the persisted
    allocation; changing quantities, parties, receipt, or idempotency identity
    is rejected before inventory, order, or ledger mutation.
    """


class ShipmentNotActionable(WorldError):
    """A shipment status event or party resolution is invalid.

    History is append-only and versioned. Invalid transitions, forged parties,
    conflicting event ids, and infeasible resolutions have zero state effect.
    """


class ExchangeNotActionable(WorldError):
    """A replacement exchange is invalid, conflicting, or unauthorized.

    Exchanges are like-for-like and may only consume returned inventory once.
    They never create ledger rows; price differences require an explicit
    refund/settlement flow instead of this primitive.
    """
