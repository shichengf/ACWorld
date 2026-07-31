"""Protocol package — the wire envelope, action enum, partition table,
pure pricing math, and VCP payload constructors.

Wire-stable: fields can be added, never removed or renamed.

Submodules:

* :mod:`protocol.actions`   — ``ActionKind`` enum + ``PARTITION_ALLOW``.
* :mod:`protocol.envelope`  — wire envelope (de)serialization + validation.
* :mod:`protocol.errors`    — protocol-level exception hierarchy.
* :mod:`protocol.interfaces` — signer / validator interfaces.
* :mod:`protocol.pricing`   — pure pricing math (counters, bands, biases,
  markdown/markup steps). Lane-agnostic.
* :mod:`protocol.envelopes` — pure VCP payload constructors (grounded
  offers, inquiry responses) + ``assert_no_private_leak``. Lane-agnostic.
"""

from __future__ import annotations

from protocol.actions import (
    PARTITION_ALLOW,
    ActionKind,
    is_deferred,
    is_send_allowed,
)
from protocol.envelope import Envelope, from_json, to_json, validate
from protocol.envelopes import (
    ClaimNotPermitted,
    EnvelopeShapeError,
    MandateLeak,
    assert_no_private_leak,
    build_grounded_offer,
    build_inquiry_response,
)
from protocol.errors import (
    IdempotencyConflict,
    PartitionViolation,
    ProtocolError,
    SchemaError,
    UnknownActionKind,
)
from protocol.interfaces import EnvelopeSigner, EnvelopeValidator
from protocol.schemas import (
    AdjustPricePayload,
    ReceiveShipmentPayload,
    UpdateListingFieldsDelist,
    UpdateListingFieldsPublish,
    UpdateListingFieldsUpdate,
    UpdateListingPayload,
    WorldApplyCatalogMutationPayload,
)
from protocol.pricing import (
    PEER_PRICING_STRATEGIES,
    aged_stock_discount,
    apply_pricing_bias,
    apply_reputation_bias,
    availability_band,
    compute_concession_counter,
    compute_counter,
    compute_min_offer_step,
    compute_patience_loss,
    demand_step_markup,
    extract_list_prices_minor,
    filter_similar_peers,
    peer_price_stats,
    peer_similarity_score,
    recommend_list_price_from_peers,
    reputation_band,
)

__all__ = [
    "ActionKind",
    "AdjustPricePayload",
    "ClaimNotPermitted",
    "Envelope",
    "EnvelopeShapeError",
    "EnvelopeSigner",
    "EnvelopeValidator",
    "IdempotencyConflict",
    "MandateLeak",
    "PARTITION_ALLOW",
    "PEER_PRICING_STRATEGIES",
    "PartitionViolation",
    "ProtocolError",
    "ReceiveShipmentPayload",
    "SchemaError",
    "UnknownActionKind",
    "UpdateListingFieldsDelist",
    "UpdateListingFieldsPublish",
    "UpdateListingFieldsUpdate",
    "UpdateListingPayload",
    "WorldApplyCatalogMutationPayload",
    "aged_stock_discount",
    "apply_pricing_bias",
    "apply_reputation_bias",
    "assert_no_private_leak",
    "availability_band",
    "build_grounded_offer",
    "build_inquiry_response",
    "compute_concession_counter",
    "compute_counter",
    "compute_min_offer_step",
    "compute_patience_loss",
    "demand_step_markup",
    "extract_list_prices_minor",
    "filter_similar_peers",
    "from_json",
    "is_deferred",
    "is_send_allowed",
    "peer_price_stats",
    "peer_similarity_score",
    "recommend_list_price_from_peers",
    "reputation_band",
    "to_json",
    "validate",
]
