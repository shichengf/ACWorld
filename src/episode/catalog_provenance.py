"""Machine-readable provenance checks for the shipped real-CSV catalog set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.merchant_data_csv import catalog_source_statistics


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2] / "data" / "catalog_provenance.json"
)


class CatalogProvenanceMismatch(ValueError):
    """A raw source no longer matches its immutable provenance declaration."""


def load_catalog_provenance(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and minimally validate the provenance manifest shape."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != "1.0":
        raise CatalogProvenanceMismatch("unsupported catalog provenance manifest_version")
    if manifest.get("data_label") != "real_csv":
        raise CatalogProvenanceMismatch("catalog provenance data_label must be 'real_csv'")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise CatalogProvenanceMismatch("catalog provenance sources must be non-empty")
    return manifest


def verify_catalog_provenance(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute source hashes/counts and fail on any undeclared data drift."""

    manifest = load_catalog_provenance(manifest_path)
    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(manifest_path).resolve().parents[1]
    )
    expected_fields = (
        "merchant",
        "source_file",
        "sha256",
        "bytes",
        "column_count",
        "row_count",
        "sellable_row_count",
        "canonical_listing_count",
        "duplicate_source_sku_rows",
        "in_stock_canonical_listing_count",
        "product_group_count",
        "source_domains",
    )
    actual_sources: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for expected in manifest["sources"]:
        merchant = str(expected["merchant"])
        actual = catalog_source_statistics(
            merchant,
            data_root=root / "data" / "raw_data",
        )
        actual_sources.append(actual)
        for field in expected_fields:
            if actual.get(field) != expected.get(field):
                mismatches.append(
                    f"{merchant}.{field}: expected {expected.get(field)!r}, "
                    f"found {actual.get(field)!r}"
                )

    aggregates = {
        "source_count": len(actual_sources),
        "row_count": sum(int(item["row_count"]) for item in actual_sources),
        "sellable_row_count": sum(
            int(item["sellable_row_count"]) for item in actual_sources
        ),
        "canonical_listing_count": sum(
            int(item["canonical_listing_count"]) for item in actual_sources
        ),
        "duplicate_source_sku_rows": sum(
            int(item["duplicate_source_sku_rows"]) for item in actual_sources
        ),
        "in_stock_canonical_listing_count": sum(
            int(item["in_stock_canonical_listing_count"])
            for item in actual_sources
        ),
    }
    for field, actual in aggregates.items():
        expected = manifest.get("aggregate", {}).get(field)
        if actual != expected:
            mismatches.append(
                f"aggregate.{field}: expected {expected!r}, found {actual!r}"
            )

    full_target = manifest.get("scale_profiles", {}).get("full")
    if full_target != aggregates["canonical_listing_count"]:
        mismatches.append(
            "scale_profiles.full must equal aggregate.canonical_listing_count"
        )
    if mismatches:
        raise CatalogProvenanceMismatch("; ".join(mismatches))
    return {
        "dataset_id": manifest["dataset_id"],
        "data_label": manifest["data_label"],
        "verified": True,
        "aggregate": aggregates,
        "sources": actual_sources,
        "governance": dict(manifest.get("governance") or {}),
    }


__all__ = [
    "CatalogProvenanceMismatch",
    "DEFAULT_MANIFEST",
    "load_catalog_provenance",
    "verify_catalog_provenance",
]
