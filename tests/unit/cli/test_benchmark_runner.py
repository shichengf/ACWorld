from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cli import benchmark_runner


class UnifiedBenchmarkRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _base_args(self, task_count: str) -> list[str]:
        return [
            "--tasks",
            task_count,
            "--model",
            "google/gemini-3.5-flash",
            "--api-key-file",
            str(self.root / "key.txt"),
            "--catalog-db",
            str(self.root / "catalog.sqlite"),
            "--large-task-file",
            str(self.root / "tasks.json"),
            "--large-output-root",
            str(self.root / "large-output"),
        ]

    def test_default_run_remains_the_original_200_tasks(self) -> None:
        calls: list[str] = []
        with (
            patch.object(
                benchmark_runner,
                "_run_core",
                side_effect=lambda _args: calls.append("core") or 0,
            ),
            patch.object(
                benchmark_runner,
                "_run_large",
                side_effect=lambda _args: calls.append("large"),
            ),
        ):
            status = benchmark_runner._run(
                [
                    "--model",
                    "google/gemini-3.5-flash",
                    "--api-key-file",
                    str(self.root / "key.txt"),
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(calls, ["core"])

    def test_large_mode_validates_then_runs_only_sixty_tasks(self) -> None:
        (self.root / "catalog.sqlite").touch()
        (self.root / "tasks.json").write_text("[]", encoding="utf-8")
        calls: list[str] = []
        with (
            patch.object(
                benchmark_runner,
                "large_catalog_main",
                side_effect=lambda argv: calls.append(str(argv[0])),
            ),
            patch.object(
                benchmark_runner,
                "_run_core",
                side_effect=lambda _args: calls.append("core") or 0,
            ),
        ):
            status = benchmark_runner._run(self._base_args("60"))

        self.assertEqual(status, 0)
        self.assertEqual(calls, ["validate", "run", "summary"])

    def test_combined_mode_preflights_large_before_core(self) -> None:
        (self.root / "catalog.sqlite").touch()
        (self.root / "tasks.json").write_text("[]", encoding="utf-8")
        calls: list[str] = []
        with (
            patch.object(
                benchmark_runner,
                "large_catalog_main",
                side_effect=lambda argv: calls.append(str(argv[0])),
            ),
            patch.object(
                benchmark_runner,
                "_run_core",
                side_effect=lambda _args: calls.append("core") or 0,
            ),
        ):
            status = benchmark_runner._run(self._base_args("260"))

        self.assertEqual(status, 0)
        self.assertEqual(calls, ["validate", "core", "run", "summary"])

    def test_large_mode_rejects_nonpaper_optional_model(self) -> None:
        args = self._base_args("60")
        args[args.index("google/gemini-3.5-flash")] = "anthropic/claude-opus-4.8"

        with self.assertRaisesRegex(ValueError, "ten paper models"):
            benchmark_runner._run(args)

    def test_fresh_large_mode_downloads_when_raw_data_is_absent(self) -> None:
        calls: list[str] = []
        with (
            patch.object(
                benchmark_runner,
                "large_catalog_main",
                side_effect=lambda argv: calls.append(str(argv[0])),
            ),
            patch.object(benchmark_runner, "_run_large"),
        ):
            status = benchmark_runner._run(self._base_args("60"))

        self.assertEqual(status, 0)
        self.assertEqual(calls, ["download", "validate"])


if __name__ == "__main__":
    unittest.main()
