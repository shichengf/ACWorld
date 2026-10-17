from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from cli import large_catalog


def test_task_file_is_rebuilt_from_current_code(tmp_path: Path) -> None:
    task_path = tmp_path / "tasks.json"
    task_path.write_text('{"suite": "stale", "tasks": []}\n', encoding="utf-8")
    expected = (object(),)

    with (
        patch.object(large_catalog, "build_tasks", return_value=expected) as build,
        patch.object(large_catalog, "write_tasks") as write,
    ):
        actual = large_catalog._load_or_build_tasks(
            tmp_path / "catalog.sqlite",
            task_path,
        )

    assert actual is expected
    build.assert_called_once_with(tmp_path / "catalog.sqlite")
    write.assert_called_once_with(expected, task_path)
