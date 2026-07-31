---
name: claim-truthfulness
description: A pre-step for catalog-serve / pricing-negotiate that filters the offer's claim set down to the subset grounded in the listing's attributes. "permitted_claims" is the LEGAL envelope (what the mandate authorizes); this skill enforces the TRUTHFULNESS contract (what the SKU actually supports). The merchant never asserts a claim the filter excluded. Emits no envelope of its own; the offer-building skill consumes the filtered claims.
when_to_use: Use as a pre-step before any envelope whose payload carries a claims array (commerce.respond_inquiry, commerce.counter_offer / propose_offer with a GroundedOffer). Call filter_truthful_claims(mandate.permitted_claims, listing.attributes, alias_map=?) and use the returned subset in the GroundedOffer's claims field. Do NOT use to gate inquiries unrelated to offer claims (inquiry-handle's answer field is its own surface).
group: trust
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_catalog
  - memory.read
context: main
---

# Claim Truthfulness

## Native function guidance

Treat a permitted claim as a legal ceiling, not proof that it is true. Check
the authoritative listing, claim records, and evidence before asserting,
applying, correcting, or retaining a public claim. Use only facts supported for
this exact listing and merchant. Never invent evidence, generalize from another
product, or expose private policy. If a claim is unsupported, use the supplied
listing correction path when authorized or omit it. Make at most one terminal
catalog action after the required reads.

## Detailed workflow

The second TRUST-group skill on the sell side, paired with
``private-utility-guard``: where the guard blocks **private** values
from leaking onto the wire, this skill blocks **false** claims —
ones the merchant is *permitted* to make but whose attribute backing
the listing does not actually have.

The distinction matters for the platform's claim-verification path
(``platform.verify_claims`` per VCP_SPEC §6). The merchant can be
penalized for unsupported claims even if those claims are inside its
permitted set. The legal envelope is one bound; truth is the other.

## Procedure

1. Read state:
   - ``mand = read_offer_mandate(product)`` — ``permitted_claims`` is the
     legal envelope; ``must_not_claim`` is the legal *anti*-envelope.
   - ``lst = get_listing(product)`` — ``attributes`` is the factual
     ground.
2. Compute:
   ``grounded = filter_truthful_claims(mand.permitted_claims,
                                       lst.attributes,
                                       alias_map=<scenario hints>)``
   The helper returns the subset of ``permitted_claims`` whose
   normalized key (or alias) maps to a truthy attribute value.
3. Hand the ``grounded`` tuple to the offer-building skill
   (``catalog-serve`` / ``pricing-negotiate``) — it goes into the
   ``GroundedOffer.claims`` field, **as-is**. Do not add a claim back
   in just because the LLM thinks it ought to be true.
4. (Optional) If ``grounded`` is empty, the offer can ship with
   ``claims: []``. Buyers see no claim instead of a false one.

## Never disclose

- The *reason* the filter excluded a claim. Saying "we don't claim
  self-cleaning because our SKU isn't" surfaces a competitive weakness
  in a way the buyer doesn't get on a competitor offer that simply
  doesn't list the claim. The wire should look identical: the claim
  isn't there.
- Internal attribute fields that are not part of the public catalog
  (cost basis, supplier identity, internal sku tags).

## Bounds and ethics

- Truthfulness is a one-way ratchet: this skill can only *remove*
  claims, never add them. If a claim is grounded but not permitted,
  ``must_not_claim`` (mandate-side) still excludes it.
- The alias map is scenario configuration, not LLM authority. The
  agent does not invent a new alias on the fly to ground a claim that
  has no attribute backing.
- Numeric / list attribute values count as truthy when non-zero /
  non-empty. A ``capacity_l: 5`` grounds a ``"large capacity"`` claim
  if the alias points there *and* the scenario says 5 counts; the
  scenario, not the agent, owns the threshold.

## Output

``no_reply``. The offer-building skill (``catalog-serve`` or
``pricing-negotiate``) emits the single envelope with the filtered
``claims`` set.
