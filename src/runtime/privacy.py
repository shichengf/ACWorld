"""Path-aware, registry-driven privacy policy (v1).

The runtime privacy boundary: a registered monetary secret (a buyer's max
budget, a merchant's floor price) must never cross to a *counterparty*, and a
party must never disclose a secret it should not possess. This module is the
single authoritative detector the :class:`~runtime.router.Router` consults on
every send.

**v1 is a DETERMINISTIC, EXACT-PATH detector.** It does NOT attempt arbitrary
paraphrase or word-form detection ("a bit under eighty bucks", "eighty
dollars"); that is future work (a v2 semantic judge). What it DOES detect:

1. an explicit private *field name* (``max_budget`` / ``floor_price`` / …) at ANY
   mapping key, at any depth, INCLUDING container-valued keys
   (``{"floor_price": {...}}``), when the envelope crosses the owner's
   counterparty boundary;
2. a registered secret *amount* disclosed at any payload path that is NOT a
   canonical transaction-price path for that exact ActionKind — a cents integer,
   or a Decimal-exact money figure in free text (``$80.50`` / ``80.50 USD`` /
   ``8050 cents``). A top-level offer ``reason`` that merely repeats that same
   payload's canonical ``unit_price`` is public-price narration, not evidence
   that the sender knew an equal-valued hidden oracle;
3. cross-party leakage — a sender disclosing *another* party's registered secret,
   including forwarding it between two actors that are both unrelated to its owner.

It is NOT coupled to agent ``PRIVATE_UTILITY`` memory: secrets are registered
explicitly from the scenario answer key, so the policy can never collide an
incidental memory value (a boolean ``True``, ``max_negotiation_rounds=3``)
with a legitimate payload integer (``qty=1``, ``qty=3``).

A value equal to a secret is *exempt* only when it appears at a COMPLETE
canonical transaction-price path for the exact ActionKind (e.g. an offer's
``unit_price``, a settle's ``agreed_price.amount``, a ranked candidate's
``candidates.*.unit_price``). A leaf merely *named* ``unit_price``/``amount``
under an unrelated container (``metadata.unit_price``, ``notes.amount``) is NOT
exempt.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

# --- registered secrets ------------------------------------------------


@dataclass(frozen=True)
class MoneySecret:
    """One monetary secret to keep off a counterparty wire.

    ``amount_cents`` is the canonical internal unit (integer minor units), so
    detection never relies on float equality. ``counterparty_roles`` are the
    side prefixes this amount must not be disclosed to OR from (the
    buyer<->merchant boundary in v1)."""

    owner_id: str
    name: str
    amount_cents: int
    counterparty_roles: frozenset[str]
    # Actor ids explicitly trusted with the secret (normally the owning human
    # principal). The owner itself is always trusted implicitly.
    allowed_actor_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SecretRegistry:
    """The explicit set of scoring-relevant secrets for one episode.

    Built from the scenario answer key (NOT from agent memory). Only monetary
    secrets are registered in v1 — never booleans, round counts, quantities,
    urgency flags, or whole memory buckets."""

    money: tuple[MoneySecret, ...] = ()
    # In a many-participant market, equal numeric budgets/floors are common.
    # Actor scoping permits an equal-valued amount on a route that is trusted
    # for *one* matching owner (for example buyer:b2 -> consumer:b2), while
    # still rejecting third-party forwarding on every untrusted route.
    actor_scoped: bool = False

    def is_empty(self) -> bool:
        """True when no monetary secrets are registered. NOTE: explicit private
        FIELD-NAME detection still runs on an empty registry (it needs no
        registered amount), so :meth:`Router.check_payload` must not early-return
        on this."""
        return not self.money


@dataclass(frozen=True)
class LeakFinding:
    """A sanitized description of a detected leak. Carries NO raw secret value
    and NO copied payload text — only enough to audit the violation.

    ``secret_owner`` is the canonical owner id when a registered secret matched
    (e.g. ``merchant:m1``); for a name-only match it is the owner side
    (``buyer``/``merchant``). ``owner_role`` is always the side prefix."""

    secret_owner: str
    secret_name: str
    field_path: str
    reason: str
    owner_role: str = ""


# --- policy tables -----------------------------------------------------

#: Private field NAMES mapped to the OWNING side. Verified against the skills
#: (mandate-parsing, price-discovery, pricing-negotiate, private-utility-guard):
#: a field here lives in a PRIVATE_UTILITY bucket and must not cross to the
#: owner's counterparty. ``min_acceptable_price`` is the SELLER-side dual of the
#: buyer budget (VIBE_COMMERCE §4.2), hence merchant. ``target_band`` and
#: ``auto_accept_threshold`` are deliberately ABSENT: the skills write them to
#: PREFERENCE (a public negotiation band / threshold), not PRIVATE_UTILITY, so
#: classifying them as secrets would flag legitimate fields.
_PRIVATE_FIELD_OWNER_SIDE: dict[str, str] = {
    "max_budget": "buyer",
    "budget": "buyer",
    "maximum_unit_price_cents": "buyer",
    "floor_price": "merchant",
    "minimum_unit_price_cents": "merchant",
    "walk_away_price": "merchant",
    "min_acceptable_price": "merchant",
    "urgency_to_sell": "merchant",
}
#: Generic private markers whose owning side is the SENDER (no fixed side).
_GENERIC_PRIVATE_FIELDS = frozenset({"private_utility", "must_not_share_list"})

#: Counterparty boundary for a given owner side (v1 two-sided market).
_COUNTERPARTY: dict[str, frozenset[str]] = {
    "buyer": frozenset({"merchant"}),
    "merchant": frozenset({"buyer"}),
}

#: EXACT canonical transaction-price payload paths per ActionKind, derived from
#: the real constructors/handlers/tests (NOT key names). ``"*"`` matches only a
#: LIST index, never an arbitrary mapping segment. A registered amount at one of
#: these complete paths is a legitimate price, not a leak. Kinds that carry no
#: on-wire price (reject_offer, create_match_certificate, settlement_receipt,
#: issue_refund, create_order, authorize_payment, bare settle) are intentionally
#: absent — they get no price exemption.
#:
#: settle_payment accepts ``agreed_price`` OR ``price``, each either a scalar
#: cents value or a Money dict (``{amount, currency}``); both shapes are listed.
_PRICE_PATHS_BY_KIND: dict[str, tuple[tuple[str, ...], ...]] = {
    "commerce.propose_offer": (("unit_price",),),
    "commerce.counter_offer": (("unit_price",),),
    "commerce.accept_offer": (("unit_price",), ("unit_price_cents",)),
    "platform.rank_offers": (
        ("candidates", "*", "unit_price"),
        ("candidates", "*", "unit_price_cents"),
        ("search_session", "offers", "*", "unit_price_cents"),
    ),
    "platform.create_match_certificate": (("unit_price_cents",),),
    "platform.settle_payment": (
        ("agreed_price",),
        ("agreed_price", "amount"),
        ("price",),
        ("price", "amount"),
    ),
    # Actor-scoped after-sales reads expose only authoritative World records
    # after the Platform has proved that the recipient is an order party.  A
    # captured payment amount and an immutable ledger receipt price are
    # therefore transaction prices, even when they happen to equal the
    # buyer's mandate ceiling.  Keep the exemption at the two exact response
    # paths so arbitrary metadata in the same envelope remains protected.
    "platform.after_sales_snapshot": (
        ("records", "*", "amount"),
        ("records", "*", "captured_amount"),
        ("records", "*", "refunded_amount"),
        ("records", "*", "price", "amount"),
        # ``after_sales_history`` is the heterogeneous World projection
        # ``{table, key, value}``.  These are the exact monetary leaves of its
        # strict persisted record codecs; arbitrary facts/metadata beneath the
        # same wrapper deliberately receive no exemption.
        ("records", "*", "value", "binding", "amount"),
        ("records", "*", "value", "amount"),
        ("records", "*", "value", "requested_amount"),
        ("records", "*", "value", "authorized_amount"),
        ("records", "*", "value", "approved_amount"),
        ("records", "*", "value", "refund_amount"),
        ("records", "*", "value", "gross_amount"),
        ("records", "*", "value", "net_amount"),
    ),
    # Re-entrant World reads serialize authoritative public listings and
    # caller-scoped orders into a generic world.response.  A public list price
    # or an order's agreed transaction price may numerically equal an
    # unrelated actor's private budget or floor.  Exempt only these exact
    # typed money paths; arbitrary facts and metadata in the same response
    # remain protected.
    "world.response": (
        ("list_price", "amount"),
        ("*", "list_price", "amount"),
        ("agreed_price", "amount"),
        ("price", "amount"),
    ),
}

#: Exact typed scalar paths whose units are explicitly non-monetary.  The
#: amount detector otherwise treats every structured integer as cents.  That is
#: intentionally conservative for free-form payloads, but it must not confuse
#: a public evidence confidence (for example 9000 basis points) with another
#: actor's numerically equal 9000-cent budget.  Exemptions stay action- and
#: path-specific; arbitrary evidence facts and metadata receive no exemption.
_NON_MONEY_PATHS_BY_KIND: dict[str, tuple[tuple[str, ...], ...]] = {
    # The public after-sales policy uses integer revisions, ticks, windows,
    # and basis points.  They are not currency even when one happens to equal
    # a registered private price in cents.  Keep this allowlist at the exact
    # flat policy projection paths; nested facts/metadata remain protected.
    "platform.after_sales_snapshot": (
        ("records", "*", "revision"),
        ("records", "*", "effective_tick"),
        ("records", "*", "return_window_ticks"),
        ("records", "*", "max_refund_bps"),
        ("records", "*", "split_refund_bps"),
    ),
    # ``world.read_evidence_record`` returns the authorized record itself in a
    # generic ``world.response`` envelope.  These three fields keep their
    # evidence-record units after serialization.  In particular,
    # ``confidence_bps=9500`` is 95 percent confidence, not a 9500-cent price
    # that happens to equal an actor's private budget.  Keep the exemption
    # exact.  Arbitrary evidence facts and metadata remain protected.
    "world.response": (
        ("issued_at_tick",),
        ("trust", "confidence_bps"),
        ("version",),
    ),
    "commerce.publish_evidence_record": (
        ("record", "issued_at_tick"),
        ("record", "trust", "confidence_bps"),
        ("record", "version"),
    ),
    "platform.publish_evidence_record": (
        ("record", "issued_at_tick"),
        ("record", "trust", "confidence_bps"),
        ("record", "version"),
    ),
    "platform.evidence_record_persisted": (
        ("record", "issued_at_tick"),
        ("record", "trust", "confidence_bps"),
        ("record", "version"),
    ),
    "platform.publish_governance_policy": (
        ("policy_intent", "compliance_violation_bps"),
        ("policy_intent", "disputed_order_bps"),
        ("policy_intent", "effective_tick"),
        ("policy_intent", "fulfilled_order_bps"),
        ("policy_intent", "refund_bps"),
        ("policy_intent", "remediation_verified_bps"),
        ("policy_intent", "policy_version"),
    ),
}


def _normalize_key(key: str) -> str:
    """Conservative key normalization: case + common separators. NOT fuzzy /
    semantic — only folds ``Floor-Price``/``floor price`` -> ``floor_price``."""
    return str(key).strip().casefold().replace("-", "_").replace(" ", "_")


# --- deterministic money parsing (Decimal / cents-exact) ---------------

_DOLLAR_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")
_USD_RE = re.compile(r"\b([\d,]+(?:\.\d{1,2})?)\s*(?:usd|dollars?)\b")
_DECIMAL_RE = re.compile(r"\b(\d[\d,]*\.\d{2})\b")
_CENTS_RE = re.compile(r"\b(\d[\d,]*)\s*cents?\b")
# Exact lexical markers for an explicit private-value declaration.  This is
# intentionally narrower than a semantic privacy classifier: it distinguishes
# ``I offer $80`` from ``my maximum budget is $80`` while remaining fully
# deterministic and auditable.
_EXPLICIT_PRIVATE_VALUE_CLAIM_RE = re.compile(
    r"\b(?:"
    r"(?:my|our)\s+(?:exact\s+)?(?:"
    r"maximum(?:\s+(?:budget|spend|price))?|minimum(?:\s+price)?|"
    r"budget|floor|private\s+(?:boundary|value)|reservation\s+price|"
    r"walk[- ]away(?:\s+price)?|spending\s+limit"
    r")|(?:"
    r"maximum(?:\s+(?:budget|spend|price))?|minimum(?:\s+price)?|"
    r"budget|floor|private\s+(?:boundary|value)|reservation\s+price|"
    r"walk[- ]away(?:\s+price)?|spending\s+limit"
    r")\s*(?:is|equals?|:)"
    r"|(?:lowest|highest)\s+(?:i|we)\s+(?:will|can)\s+"
    r"(?:go|pay|spend)"
    r"|(?:is|equals?)\s+(?:my|our|the)\s+(?:"
    r"budget|floor|maximum|minimum|private\s+(?:boundary|value)|"
    r"reservation\s+price|walk[- ]away(?:\s+price)?|spending\s+limit"
    r")"
    r")",
    flags=re.IGNORECASE,
)

_STRUCTURED_MONEY_VALUE_KEYS = frozenset(
    {
        "agreed_price",
        "amount",
        "fee",
        "list_price",
        "price",
        "subtotal",
        "total",
        "unit_price",
    }
)


def _dollars_to_cents(s: str) -> "int | None":
    try:
        return int(Decimal(s.replace(",", "")) * 100)
    except (InvalidOperation, ValueError):
        return None


def _plain_to_cents(s: str) -> "int | None":
    try:
        return int(s.replace(",", ""))
    except ValueError:
        return None


def _money_cents_in_text(text: str) -> "set[int]":
    """Every money figure in ``text`` as integer cents, requiring a money or
    explicit-cents qualifier (``$``, ``USD``/``dollars``, an ``NN.NN`` decimal,
    or ``N cents``). A bare integer (``order-8050``) yields nothing, so an id
    that merely contains the digits is not a false positive."""
    low = text.casefold()
    out: set[int] = set()
    for m in _DOLLAR_RE.finditer(text):
        c = _dollars_to_cents(m.group(1))
        if c is not None:
            out.add(c)
    for m in _USD_RE.finditer(low):
        c = _dollars_to_cents(m.group(1))
        if c is not None:
            out.add(c)
    for m in _DECIMAL_RE.finditer(text):
        c = _dollars_to_cents(m.group(1))
        if c is not None:
            out.add(c)
    for m in _CENTS_RE.finditer(low):
        c = _plain_to_cents(m.group(1))
        if c is not None:
            out.add(c)
    return out


def _amount_matches(value: Any, amount_cents: int) -> bool:
    """True iff ``value`` discloses ``amount_cents``.

    * bool: never (booleans are not integers here);
    * int: exact cents equality (a structured ``8050`` matches 8050 cents);
    * float: exact equality, NEVER truncated (``80.5`` does not match 8050);
    * str: a Decimal-exact money figure equal to the secret (``$80.50``); a bare
      digit run with no money qualifier does not match.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == amount_cents
    if isinstance(value, float):
        return float(amount_cents) == value
    if isinstance(value, str):
        return amount_cents in _money_cents_in_text(value)
    return False


# --- payload traversal -------------------------------------------------


def _role(address: str) -> str:
    return address.split(":", 1)[0]


def _walk_keys(value: Any, path: "tuple[str, ...]" = ()) -> "list[tuple[tuple[str, ...], str]]":
    """Yield ``(path, key)`` for EVERY mapping key at any depth, including keys
    whose value is a container (so ``{"floor_price": {...}}`` is detected)."""
    out: "list[tuple[tuple[str, ...], str]]" = []
    if isinstance(value, dict):
        for k, v in value.items():
            kp = path + (str(k),)
            out.append((kp, str(k)))
            out.extend(_walk_keys(v, kp))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for i, v in enumerate(value):
            out.extend(_walk_keys(v, path + (str(i),)))
    return out


def _walk_scalars(value: Any, path: "tuple[str, ...]" = ()) -> "list[tuple[tuple[str, ...], Any]]":
    """Yield ``(path, value)`` for every scalar leaf."""
    out: "list[tuple[tuple[str, ...], Any]]" = []
    if isinstance(value, dict):
        for k, v in value.items():
            out.extend(_walk_scalars(v, path + (str(k),)))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for i, v in enumerate(value):
            out.extend(_walk_scalars(v, path + (str(i),)))
    else:
        out.append((path, value))
    return out


def _crosses(counterparty_roles: "frozenset[str]", sender_role: str, recipient_role: str) -> bool:
    """True iff the envelope touches the secret's counterparty (as sender or
    recipient)."""
    return sender_role in counterparty_roles or recipient_role in counterparty_roles


def _secret_crosses(
    secret: MoneySecret,
    *,
    sender_id: str,
    recipient_id: str,
    actor_scoped: bool,
) -> bool:
    """Whether this route is inside ``secret``'s enforceable boundary.

    Role-wide mode preserves the v1 single-market behavior. Actor-scoped mode
    checks owner-to-peer, peer-to-owner, and peer-to-third-party forwarding.
    A route containing only the owner and explicitly trusted principals is
    exempt. Equal-value collision handling is performed across all matching
    secrets in :func:`find_leak` before this predicate is called.
    """
    if not actor_scoped:
        return _crosses(secret.counterparty_roles, _role(sender_id), _role(recipient_id))
    if _route_is_trusted_for_secret(secret, sender_id=sender_id, recipient_id=recipient_id):
        return False
    # A party that is not trusted for this secret must not possess and forward
    # its exact value merely because the owner is absent from the route.
    return True


def _route_is_trusted_for_secret(
    secret: MoneySecret,
    *,
    sender_id: str,
    recipient_id: str,
) -> bool:
    trusted = {secret.owner_id, *secret.allowed_actor_ids}
    return {sender_id, recipient_id}.issubset(trusted)


def _path_matches(allowed: "tuple[str, ...]", actual: "tuple[str, ...]") -> bool:
    if len(allowed) != len(actual):
        return False
    for a, x in zip(allowed, actual):
        if a == "*":
            if not x.isdigit():  # wildcard matches ONLY a list index
                return False
        elif a != x:
            return False
    return True


def _is_txn_price_path(kind: str, path: "tuple[str, ...]") -> bool:
    """True iff ``path`` exactly matches a canonical transaction-price path for
    ``kind``. Exact and complete — a leaf merely named ``amount``/``unit_price``
    under an unrelated container is NOT a price path."""
    return any(_path_matches(allowed, path) for allowed in _PRICE_PATHS_BY_KIND.get(kind, ()))


def _is_non_money_path(kind: str, path: "tuple[str, ...]") -> bool:
    """Whether a schema-defined scalar path has non-currency units."""

    return any(_path_matches(allowed, path) for allowed in _NON_MONEY_PATHS_BY_KIND.get(kind, ()))


def sanitize_field_path_for_audit(field_path: str) -> str:
    """Return a useful path label without copying arbitrary payload keys.

    Payload mapping keys are model-controlled content.  Known protocol/privacy
    field names are safe identifiers; unknown names are represented only by
    length and a one-way digest.  List positions remain numeric so an operator
    can still locate the blocked field without the security sidecar becoming a
    second channel for rejected text.
    """

    if field_path == "<payload>" or not field_path:
        return "<payload>"
    safe_segments = {
        *_PRIVATE_FIELD_OWNER_SIDE,
        *_GENERIC_PRIVATE_FIELDS,
        "message",
        "disclosed_value",
    }
    for paths in (*_PRICE_PATHS_BY_KIND.values(), *_NON_MONEY_PATHS_BY_KIND.values()):
        for path in paths:
            safe_segments.update(
                segment for segment in path if segment != "*" and not segment.isdigit()
            )
    sanitized: list[str] = []
    for raw in field_path.split("."):
        if raw.isdigit():
            sanitized.append(f"[{raw}]")
            continue
        normalized = _normalize_key(raw)
        if normalized in safe_segments:
            sanitized.append(normalized)
            continue
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        sanitized.append(f"<key chars={len(raw)} sha256={digest}>")
    return ".".join(sanitized)


def find_leak(env: Any, registry: SecretRegistry) -> "LeakFinding | None":
    """Return a sanitized :class:`LeakFinding` if ``env`` would disclose a
    registered secret across a counterparty boundary, else ``None``.

    No wholesale ``delegate.*`` exemption: the valid mandate route is
    consumer->buyer (PARTITION_ALLOW), which never crosses the buyer<->merchant
    boundary, so the budget in a mandate is naturally not flagged; a malformed
    delegate route fails ``check_outbound`` (partition) before this runs.
    """
    action = env.action if isinstance(env.action, dict) else {}
    kind = str(action.get("kind", ""))
    payload = action.get("payload")
    sender_role = _role(env.from_)
    recipient_role = _role(env.to)

    # (1) Explicit private field NAME (any depth, container-valued included).
    for path, key in _walk_keys(payload):
        norm = _normalize_key(key)
        owner_side = _PRIVATE_FIELD_OWNER_SIDE.get(norm)
        if owner_side is None and norm in _GENERIC_PRIVATE_FIELDS:
            owner_side = sender_role  # generic marker -> the sender's own side
        if owner_side is None:
            continue
        crosses_roles = _crosses(
            _COUNTERPARTY.get(owner_side, frozenset()), sender_role, recipient_role
        )
        crosses_same_side_actor = (
            sender_role == owner_side and recipient_role == owner_side and env.from_ != env.to
        )
        if crosses_roles or crosses_same_side_actor:
            return LeakFinding(
                secret_owner=owner_side,
                owner_role=owner_side,
                secret_name=norm,
                field_path=".".join(path) or "<payload>",
                reason="explicit_private_field",
            )

    # (2) Registered secret AMOUNT at a NON-transaction-price path, crossing the
    #     secret's counterparty boundary (covers cross-party disclosure too).
    scalars = _walk_scalars(payload)
    for path, value in scalars:
        if _is_txn_price_path(kind, path) or _is_non_money_path(kind, path):
            continue
        # Natural-language price narration is not proof of private-value
        # disclosure merely because the number happens to equal an unseen
        # boundary.  Require an explicit lexical private-limit declaration.
        # Structured amount fields remain strict even when encoded as strings.
        if (
            isinstance(value, str)
            and (not path or _normalize_key(path[-1]) not in _STRUCTURED_MONEY_VALUE_KEYS)
            and _EXPLICIT_PRIVATE_VALUE_CLAIM_RE.search(value) is None
        ):
            continue
        matching = tuple(
            secret for secret in registry.money if _amount_matches(value, secret.amount_cents)
        )
        if not matching:
            continue
        # When two actors have the same amount, an owner-to-own-principal route
        # is legitimate for the route-local secret and numerically
        # indistinguishable from the other one. Prefer that concrete trusted
        # provenance. No such exemption exists on a peer/third-party route.
        if registry.actor_scoped and any(
            _route_is_trusted_for_secret(
                secret,
                sender_id=env.from_,
                recipient_id=env.to,
            )
            for secret in matching
        ):
            continue
        for secret in matching:
            if _secret_crosses(
                secret,
                sender_id=env.from_,
                recipient_id=env.to,
                actor_scoped=registry.actor_scoped,
            ):
                return LeakFinding(
                    secret_owner=secret.owner_id,
                    owner_role=_role(secret.owner_id),
                    secret_name=secret.name,
                    field_path=".".join(path) or "<payload>",
                    reason="secret_amount_disclosed",
                )
    return None


__all__ = [
    "LeakFinding",
    "MoneySecret",
    "SecretRegistry",
    "find_leak",
    "sanitize_field_path_for_audit",
]
