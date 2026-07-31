from __future__ import annotations

import json
import sqlite3
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from large_catalog import download


class CatalogDownloadTests(unittest.TestCase):
    def test_downloads_extracts_and_validates_prepared_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            package = temporary / "package"
            package.mkdir()
            database_path = package / "catalog.sqlite"
            connection = sqlite3.connect(database_path)
            connection.executescript(
                """
                CREATE TABLE source_rows (id INTEGER PRIMARY KEY);
                CREATE TABLE listings (id INTEGER PRIMARY KEY);
                INSERT INTO source_rows DEFAULT VALUES;
                INSERT INTO listings DEFAULT VALUES;
                """
            )
            connection.commit()
            connection.close()
            (package / "tasks.json").write_text(
                json.dumps(
                    {
                        "suite": "test",
                        "tasks": [{"task_id": "LC-TEST-01"}],
                    }
                ),
                encoding="utf-8",
            )
            archive_path = temporary / "catalog.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(database_path, arcname="catalog.sqlite")
                archive.add(package / "tasks.json", arcname="tasks.json")

            destination = temporary / "output"
            with (
                patch.object(download, "EXPECTED_SOURCE_ROWS", 1),
                patch.object(download, "EXPECTED_LISTINGS", 1),
                patch.object(download, "EXPECTED_TASKS", 1),
            ):
                report = download.download_prepared_catalog(
                    output_root=destination,
                    catalog_db=destination / "catalog.sqlite",
                    task_path=destination / "tasks.json",
                    asset_urls=(archive_path.as_uri(),),
                )

            self.assertEqual(report["source_rows"], 1)
            self.assertEqual(report["listings"], 1)
            self.assertEqual(report["tasks"], 1)
            self.assertTrue((destination / "catalog.sqlite").is_file())
            self.assertTrue((destination / "tasks.json").is_file())


if __name__ == "__main__":
    unittest.main()
