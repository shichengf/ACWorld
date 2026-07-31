---
name: market-governance
description: Handle sponsored placement disclosure, review integrity, governance cases, and remediation using only World-derived Platform notices.
when_to_use: Use for a marketplace governance instruction and on governance, case, remediation, audit, or evidence notices from Platform.
group: trust
user_invocable: false
disable_model_invocation: false
allowed_tools: []
context: main
---

# Marketplace Governance

## Native function guidance

Use public reputation and authoritative evidence when evaluating campaigns,
placement disclosure, review integrity, coordination cases, or remediation.
Disclose sponsorship truthfully and reject manipulation or prohibited
coordination. Accept or complete remediation only when the Platform case and
required evidence support it. Never fabricate a case, reputation signal, or
compliance record, and never expose private pricing strategy. Make one supplied
governance action for the current issue, or finish safely.

## Detailed workflow

Governance state belongs to World.  The merchant may publish a bounded intent
or respond to a Platform notice, but it cannot choose case outcomes, award its
own reputation, mark remediation complete without evidence, or create trusted
records.

## Campaign and review actions

- Publish a campaign with `commerce.publish_campaign` to `platform:ads`.
  After `platform.governance_updated` returns placements, disclose each exact
  `placement_id` with `commerce.disclose_placement`.  Activate the campaign
  only after every required disclosure is acknowledged.
- Submit an authenticated review only when acting in the permitted role.
  Reject review manipulation with `commerce.reject_review_manipulation` and
  reject prohibited coordination with `commerce.reject_coordination`.

## Cases and remediation

On `platform.governance_case_notice`, use the `current_case_reference` and
the listed `response_actions`.  Never derive a case identifier from prose.
Use `commerce.read_governance_history` to `platform:governance` when a safe
projection is needed.

On `platform.remediation_plan_notice`, accept only the returned draft plan
with `commerce.accept_remediation_plan`.  For an active plan, select the
pending step returned by Platform.  Wait for
`platform.evidence_record_persisted` from the independent audit path before
emitting `commerce.complete_remediation_step` to `platform:remediation` with
the exact `plan_id` and `step_id`.

`platform.governance_updated` and `platform.governance_snapshot` are
authoritative projections, not invitations to rewrite state.  A
`platform.remediation_audit_request` is addressed to the registered audit
service; a merchant must not impersonate that service or mint its evidence.
Never emit a `world.*` write.
