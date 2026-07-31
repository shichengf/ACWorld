"""Download and validate the prepared large-catalog data package."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


DATA_RELEASE_URLS = tuple(
    "https://github.com/shichengf/ACWorld/releases/download/"
    f"large-catalog-data-v1/acworld-large-catalog-v1.tar.gz.part-{index:03d}"
    for index in range(65)
)
EXPECTED_SOURCE_ROWS = 791_431
EXPECTED_LISTINGS = 785_022
EXPECTED_TASKS = 60
REQUIRED_MEMBERS = frozenset({"catalog.sqlite", "tasks.json"})
OPTIONAL_MEMBERS = frozenset(
    {"catalog-summary.json", "validation-report.json"}
)


class CatalogDataError(RuntimeError):
    """The downloadable catalog package is missing or invalid."""


def validate_prepared_catalog(
    catalog_db: str | Path,
    task_path: str | Path,
) -> dict[str, Any]:
    """Run concise content checks without a separate hash or contract."""

    database_path = Path(catalog_db)
    tasks_path = Path(task_path)
    if not database_path.is_file():
        raise CatalogDataError(f"catalog database is missing: {database_path}")
    if not tasks_path.is_file():
        raise CatalogDataError(f"large-catalog task file is missing: {tasks_path}")

    connection = sqlite3.connect(
        f"file:{database_path.resolve()}?mode=ro",
        uri=True,
    )
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check != ("ok",):
            raise CatalogDataError(f"SQLite quick_check failed: {quick_check!r}")
        source_rows = int(
            connection.execute("SELECT COUNT(*) FROM source_rows").fetchone()[0]
        )
        listings = int(
            connection.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        )
    except sqlite3.Error as exc:
        raise CatalogDataError(f"catalog database cannot be read: {exc}") from exc
    finally:
        connection.close()

    payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise CatalogDataError("large-catalog task file has an invalid structure")
    task_ids = [
        str(row.get("task_id", ""))
        for row in payload["tasks"]
        if isinstance(row, dict)
    ]
    if source_rows != EXPECTED_SOURCE_ROWS:
        raise CatalogDataError(
            f"catalog contains {source_rows} source rows, "
            f"expected {EXPECTED_SOURCE_ROWS}"
        )
    if listings != EXPECTED_LISTINGS:
        raise CatalogDataError(
            f"catalog contains {listings} listings, expected {EXPECTED_LISTINGS}"
        )
    if len(task_ids) != EXPECTED_TASKS or len(set(task_ids)) != EXPECTED_TASKS:
        raise CatalogDataError(
            f"task file contains {len(task_ids)} valid rows, expected {EXPECTED_TASKS}"
        )
    return {
        "source_rows": source_rows,
        "listings": listings,
        "tasks": len(task_ids),
        "sqlite_quick_check": "ok",
    }


def download_prepared_catalog(
    *,
    output_root: str | Path,
    catalog_db: str | Path,
    task_path: str | Path,
    asset_urls: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Download, safely extract, and atomically install the prepared catalog."""

    root = Path(output_root)
    database_path = Path(catalog_db)
    tasks_path = Path(task_path)
    root.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)

    if database_path.is_file() and tasks_path.is_file():
        return validate_prepared_catalog(database_path, tasks_path)

    with tempfile.TemporaryDirectory(
        prefix=".catalog-download-",
        dir=root,
    ) as temporary_name:
        temporary = Path(temporary_name)
        archive_path = temporary / "catalog.tar.gz"
        extracted = temporary / "extracted"
        extracted.mkdir()
        print(
            "Downloading the ACWorld large-catalog data under the "
            "non-redistribution terms in README.md",
            file=sys.stderr,
        )
        _download_parts(
            tuple(asset_urls) if asset_urls is not None else DATA_RELEASE_URLS,
            archive_path,
            temporary,
        )
        _extract_archive(archive_path, extracted)
        report = validate_prepared_catalog(
            extracted / "catalog.sqlite",
            extracted / "tasks.json",
        )

        os.replace(extracted / "catalog.sqlite", database_path)
        os.replace(extracted / "tasks.json", tasks_path)
        for name in OPTIONAL_MEMBERS:
            source = extracted / name
            if source.is_file():
                os.replace(source, root / name)
        oracle_source = extracted / "oracle-reports"
        if oracle_source.is_dir():
            oracle_target = root / "oracle-reports"
            if oracle_target.exists():
                shutil.rmtree(oracle_target)
            os.replace(oracle_source, oracle_target)
    return report


def _download_parts(
    urls: Sequence[str],
    destination: Path,
    temporary: Path,
) -> None:
    if not urls:
        raise CatalogDataError("catalog data download has no asset URLs")
    with destination.open("wb") as combined:
        for index, url in enumerate(urls):
            part_path = temporary / f"part-{index:02d}"
            print(
                f"Downloading catalog part {index + 1}/{len(urls)}",
                file=sys.stderr,
            )
            _download(url, part_path)
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, combined, length=8 * 1024 * 1024)
            part_path.unlink()
    if destination.stat().st_size == 0:
        raise CatalogDataError("catalog data download produced an empty archive")


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ACWorld-large-catalog-downloader"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else 0
        copied = 0
        next_report = 100 * 1024 * 1024
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                copied += len(chunk)
                if copied >= next_report:
                    suffix = f"/{total}" if total else ""
                    print(
                        f"Downloaded {copied}{suffix} bytes",
                        file=sys.stderr,
                    )
                    next_report += 100 * 1024 * 1024
    if not destination.is_file() or destination.stat().st_size == 0:
        raise CatalogDataError(f"catalog data part is empty: {url}")


def _extract_archive(archive_path: Path, destination: Path) -> None:
    print("Extracting prepared large catalog", file=sys.stderr)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        if not REQUIRED_MEMBERS.issubset(names):
            missing = sorted(REQUIRED_MEMBERS - names)
            raise CatalogDataError(f"catalog archive is missing files: {missing}")
        for member in members:
            path = PurePosixPath(member.name)
            allowed = (
                member.name in REQUIRED_MEMBERS
                or member.name in OPTIONAL_MEMBERS
                or (member.isdir() and member.name.rstrip("/") == "oracle-reports")
                or (
                    len(path.parts) == 2
                    and path.parts[0] == "oracle-reports"
                    and path.suffix == ".json"
                )
            )
            if (
                not allowed
                or path.is_absolute()
                or ".." in path.parts
                or member.issym()
                or member.islnk()
            ):
                raise CatalogDataError(
                    f"catalog archive contains an unexpected member: {member.name!r}"
                )
        try:
            archive.extractall(destination, members=members, filter="data")
        except TypeError:
            archive.extractall(destination, members=members)
