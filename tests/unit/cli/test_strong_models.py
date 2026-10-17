from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from cli import strong_models
from experiments.benchmark_plan import PAPER_MODELS_V2
from experiments.strong_model_runner import safe_model_slug


class StrongModelCommandTests(unittest.TestCase):
    def test_run_all_keeps_the_paper_model_panel(self) -> None:
        args = strong_models.build_parser().parse_args(
            ["run", "--model", "all", "--api-key-file", "unused-key.txt", "--skip-local-smoke"]
        )
        with (
            patch.object(strong_models, "check_openrouter_catalog", return_value=[]),
            patch.object(strong_models, "read_api_key", return_value="test-only-placeholder"),
            patch.object(strong_models, "run_openrouter_model", return_value={}) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(strong_models.run_command(args), 0)
        self.assertEqual([call.args[0] for call in run.call_args_list], list(PAPER_MODELS_V2))

    def test_report_and_analysis_use_each_models_own_directory(self) -> None:
        for command, writer in (("paper-report", "write_paper_report"),
                                ("paper-analysis", "write_paper_analysis")):
            with self.subTest(command=command):
                args = strong_models.build_parser().parse_args(
                    [command, "--model", "all", "--output-root", "test-results"]
                )
                with (
                    patch.object(strong_models, writer, return_value=({}, "report")) as write,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(strong_models.run_command(args), 0)
                self.assertEqual(write.call_count, len(PAPER_MODELS_V2))
                for call, model_id in zip(write.call_args_list, PAPER_MODELS_V2):
                    root = Path("test-results") / safe_model_slug(model_id)
                    self.assertEqual(call.args, (root,))
                    self.assertEqual(call.kwargs, {"output_root": root, "expected_model_id": model_id})

    def test_unknown_reporting_options_fail_before_execution(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                strong_models.build_parser().parse_args(
                    ["paper-report", "--model", "all", "--override-results", "another-run"]
                )
        self.assertEqual(raised.exception.code, 2)
