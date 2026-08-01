"""T7 must be able to say no, and must not be told what to do.

These guard the two properties the 1.1.0 rework of the after-sales family
rests on: that the correct disposition is sometimes a refusal, and that the
published contract states the policy and the request rather than the answer.
"""

from __future__ import annotations

import json
import unittest

from episode.capability_runtime_t7 import (
    _case_for_t7,
    _evidence_ids,
    _public_contract,
    _verified_evidence_ids,
)


class RefusalIsSometimesCorrectTests(unittest.TestCase):
    def test_return_outside_policy_must_be_denied(self) -> None:
        case = _case_for_t7("CWV2-T07-11")
        self.assertEqual(case.attested_return_conditions, ("damaged",))
        self.assertEqual(case.allowed_return_conditions, ("new",))
        self.assertFalse(case.entitled)
        self.assertEqual(case.evaluated_operations, ("deny_return",))

    def test_return_within_policy_must_be_authorized(self) -> None:
        case = _case_for_t7("CWV2-T07-12")
        self.assertTrue(case.entitled)
        self.assertEqual(case.evaluated_operations, ("authorize_return",))

    def test_ineligible_replacement_must_be_denied(self) -> None:
        case = _case_for_t7("CWV2-T07-16")
        self.assertFalse(case.entitled)
        self.assertEqual(
            case.evaluated_operations,
            ("authorize_return", "receive_return", "deny_exchange"),
        )

    def test_eligible_replacement_must_be_completed(self) -> None:
        case = _case_for_t7("CWV2-T07-15")
        self.assertTrue(case.entitled)
        self.assertEqual(
            case.evaluated_operations,
            (
                "authorize_return",
                "receive_return",
                "authorize_exchange",
                "complete_exchange",
            ),
        )

    def test_both_dispositions_occur_across_the_family(self) -> None:
        dispositions = {
            _case_for_t7(f"CWV2-T07-{index:02d}").entitled for index in range(1, 21)
        }
        self.assertEqual(dispositions, {True, False})


class ContractDoesNotStateTheAnswerTests(unittest.TestCase):
    def test_no_task_publishes_its_operation_sequence(self) -> None:
        for index in range(1, 21):
            task_id = f"CWV2-T07-{index:02d}"
            contract = _public_contract(_case_for_t7(task_id))
            with self.subTest(task_id=task_id):
                self.assertNotIn(
                    "required_operation_sequence",
                    contract["workflow_policy"],
                )

    def test_decision_lanes_publish_policy_and_request(self) -> None:
        refuse = _public_contract(_case_for_t7("CWV2-T07-11"))
        grant = _public_contract(_case_for_t7("CWV2-T07-12"))
        # Both are handed the same finding.  Only the shop's policy differs, so
        # the disposition cannot be read off the finding alone.
        self.assertEqual(refuse["pending_request"]["inspected_condition"], "damaged")
        self.assertEqual(grant["pending_request"]["inspected_condition"], "damaged")
        self.assertEqual(refuse["return_policy"]["accepted_conditions"], ["new"])
        self.assertIn("damaged", grant["return_policy"]["accepted_conditions"])
        # Nothing in what is said distinguishes the two either.
        for key in ("instruction", "workflow_policy"):
            self.assertEqual(
                json.dumps(refuse[key], sort_keys=True),
                json.dumps(grant[key], sort_keys=True),
            )

    def test_disputes_do_not_publish_a_prefiltered_filing_list(self) -> None:
        contract = _public_contract(_case_for_t7("CWV2-T07-08"))
        self.assertNotIn("filer_evidence_ids", contract)
        self.assertNotIn("respondent_evidence_ids", contract)
        verification = {row["verified"] for row in contract["evidence_records"]}
        self.assertEqual(verification, {True, False})


class UnverifiedEvidenceIsFilterableTests(unittest.TestCase):
    def test_every_dispute_side_carries_an_unverified_record(self) -> None:
        for task_id in ("CWV2-T07-07", "CWV2-T07-08", "CWV2-T07-17", "CWV2-T07-18"):
            case = _case_for_t7(task_id)
            for side in ("filer", "respondent"):
                with self.subTest(task_id=task_id, side=side):
                    readable = _evidence_ids(case, side)
                    verified = _verified_evidence_ids(case, side)
                    self.assertTrue(verified)
                    self.assertEqual(len(readable), len(verified) + 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
