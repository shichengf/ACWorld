from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.results import ResultStatus
from experiments.strong_model_report import (
    StrongModelReportError,
    build_paper_report,
)


class CompleteResultDirectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results: list[Path] = []
        for index in range(200):
            family, task = divmod(index, 20)
            task_id = f"CWV2-T{family + 1:02d}-{task + 1:02d}"
            directory = self.root / "runs" / task_id
            directory.mkdir(parents=True)
            credit = (0.0, 0.5)[index] if index < 2 else 1.0
            score = {
                "capability_score": credit,
                "strict_success": credit == 1.0,
                "capability_checks": [
                    {"name": "candidate_evidence_coverage", "weight": 1.0, "credit": credit}
                ],
            }
            (directory / "score.json").write_text(json.dumps(score), encoding="utf-8")
            result = {
                "status": ResultStatus.SUCCEEDED.value,
                "run": {
                    "model_id": "local/test-model",
                    "task_id": task_id,
                    "task_family": f"T{family + 1}",
                    "evaluated_role": "buyer" if index < 116 else "merchant",
                },
                "artifacts": {"task_score_v3": "score.json"},
                "metrics": {"capability_score": credit},
            }
            path = directory / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            self.results.append(path)

    def report(self):
        return build_paper_report(
            self.root, expected_model_id="local/test-model", bootstrap_replicates=20
        )

    def change_run(self, index: int, field: str, value: str) -> None:
        path = self.results[index]
        result = json.loads(path.read_text(encoding="utf-8"))
        result["run"][field] = value
        path.write_text(json.dumps(result), encoding="utf-8")

    def test_complete_directory_preserves_zero_and_partial_scores(self) -> None:
        report = self.report()
        self.assertEqual(report["runs"], {"successful": 200, "buyer": 116, "merchant": 84})
        self.assertEqual(report["full_partial_zero"], [198, 1, 1])
        self.assertEqual(report["mean_score_pct"]["overall"], 99.25)
        self.assertEqual(report["families"]["T1"]["full_partial_zero"], [18, 1, 1])

    def test_incomplete_directory_is_rejected(self) -> None:
        self.results[-1].unlink()
        with self.assertRaisesRegex(StrongModelReportError, "requires 200"):
            self.report()

    def test_duplicate_tasks_are_rejected(self) -> None:
        self.change_run(1, "task_id", "CWV2-T01-01")
        with self.assertRaisesRegex(StrongModelReportError, "duplicate task IDs"):
            self.report()

    def test_mixed_models_are_rejected(self) -> None:
        self.change_run(199, "model_id", "local/another-model")
        with self.assertRaisesRegex(StrongModelReportError, "expected one model"):
            self.report()
