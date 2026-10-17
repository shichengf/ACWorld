---
name: listing-claim-manage
description: Ground, apply, correct, or retract public listing claims through Platform while keeping evidence references and catalog ownership intact.
when_to_use: Use for a claim-bearing owner or buyer message, and whenever platform.listing_claim_updated returns the authoritative result of a prior claim action.
group: trust
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.get_listing
  - world.get_evidence_record
  - world.get_listing_claim
  - world.list_listing_claims
context: main
---

# Listing Claim Management

## Native function guidance

Use authoritative listing, claim, and evidence reads before changing a public
claim. Apply, correct, or retract only a claim owned by this merchant and
supported for the exact listing. Preserve evidence references and do not
fabricate certification, transfer support across products, or expose private
catalog strategy. Use one supplied claim or listing action when the requested
change is grounded and authorized. Otherwise record the permitted decision or
give a public response without changing the claim.

## Detailed workflow

Manage public product claims without treating model text as catalog truth.
Every claim mutation is an actor intent sent to Platform.  Platform validates
the intent and World remains the only component that commits the claim or
listing change.

## Procedure

1. Read the referenced listing with `world.get_listing`.  When the request
   names evidence or an existing claim, read those exact records too.  Do not
   replace a missing record with a plausible value.
2. Select one operation that the inbound request actually authorizes:
   - apply or revise a grounded claim: emit `commerce.apply_listing_claim` to
     `platform:claims` with the claim, listing, evidence, and operation fields
     supplied by the request;
   - update public listing attributes: emit `commerce.update_listing` to
     `platform:catalog` with `op: "update"`, the owned `sku_id`, and only the
     public fields supported by the evidence;
   - answer a buyer: emit `commerce.respond_inquiry` to that buyer using only
     verified public facts;
   - finish an analysis-only request: emit `commerce.submit_decision_record`
     to `runtime:evidence` with the grounded conclusion.
3. On `platform.listing_claim_updated`, inspect the returned claim reference
   and status.  Continue the next requested claim operation only when its
   prerequisite is now authoritative.  Otherwise report the completed result
   or use `no_reply` when no continuation is required.

Never emit a `world.*` write.  Never publish private cost, floor, margin, or
supplier information as supporting evidence.  Do not mint record identifiers
that Platform or World has not returned.
