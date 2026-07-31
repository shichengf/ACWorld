"""Sanitized feasibility evidence for the shipped real-CSV catalog corpus.

This module deliberately reuses the authoritative chain
``catalog_provenance -> merchant_data_csv -> iter_public_listings``.  It does
not parse CSV independently and never serializes a source row or attribute
value.  The report contains source hashes, aggregate counts, public category
and attribute-key coverage, and a deterministic five-merchant selection for
later local-only environment studies.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents.merchant_data_csv import KNOWN_MERCHANTS, iter_public_listings
from episode.catalog_provenance import (
    DEFAULT_MANIFEST,
    load_catalog_provenance,
    verify_catalog_provenance,
)


CATALOG_FEASIBILITY_REPORT_SCHEMA = "cwe.catalog-feasibility-report.v1"
DEFAULT_CATALOG_SEED = 42
DEFAULT_CATALOG_PROFILE = "medium"
DEFAULT_CATALOG_TARGET = 250
DEFAULT_IN_STOCK_ONLY = True
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_PROFILES = ("smoke", "standard", "medium", "full")


class CatalogFeasibilityError(ValueError):
    """The real-catalog feasibility report is invalid or no longer reproducible."""


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CatalogFeasibilityError(f"{field} must be a lowercase SHA-256")
    return value


def _pairs_to_dict(rows: Sequence[Sequence[object]], *, field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        if len(row) != 2:
            raise CatalogFeasibilityError(f"{field} rows must be [name, count]")
        name, raw_count = row
        if not isinstance(name, str) or not name or name in result:
            raise CatalogFeasibilityError(f"{field} names must be unique strings")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
            raise CatalogFeasibilityError(f"{field} counts must be positive integers")
        result[name] = raw_count
    return result


@dataclass(frozen=True)
class CatalogSourceFeasibilityV1:
    merchant: str
    merchant_id: str
    source_file: str
    source_sha256: str
    source_bytes: int
    sellable_row_count: int
    canonical_listing_count: int
    in_stock_canonical_listing_count: int
    category_listing_counts: tuple[tuple[str, int], ...]
    public_attribute_key_counts: tuple[tuple[str, int], ...]
    source_domains: tuple[str, ...]
    meets_medium_individually: bool

    def __post_init__(self) -> None:
        if self.merchant not in KNOWN_MERCHANTS:
            raise CatalogFeasibilityError("unknown real-catalog merchant")
        if self.merchant_id != f"merchant:{self.merchant}":
            raise CatalogFeasibilityError("merchant_id is not namespaced from merchant")
        if self.source_file != f"data/raw_data/{self.merchant}.csv":
            raise CatalogFeasibilityError("source_file is outside the canonical data path")
        _require_sha256(self.source_sha256, field="source_sha256")
        for name in (
            "source_bytes",
            "sellable_row_count",
            "canonical_listing_count",
            "in_stock_canonical_listing_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CatalogFeasibilityError(f"{name} must be a non-negative integer")
        if not (
            self.in_stock_canonical_listing_count
            <= self.canonical_listing_count
            <= self.sellable_row_count
        ):
            raise CatalogFeasibilityError("source listing counts are inconsistent")
        if sum(count for _, count in self.category_listing_counts) != (
            self.in_stock_canonical_listing_count
        ):
            raise CatalogFeasibilityError("category coverage does not cover in-stock listings")
        if tuple(sorted(self.category_listing_counts)) != self.category_listing_counts:
            raise CatalogFeasibilityError("category coverage must be sorted")
        if tuple(sorted(self.public_attribute_key_counts)) != (
            self.public_attribute_key_counts
        ):
            raise CatalogFeasibilityError("attribute coverage must be sorted")
        if self.meets_medium_individually is not (
            self.in_stock_canonical_listing_count >= DEFAULT_CATALOG_TARGET
        ):
            raise CatalogFeasibilityError("meets_medium_individually is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "merchant": self.merchant,
            "merchant_id": self.merchant_id,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "sellable_row_count": self.sellable_row_count,
            "canonical_listing_count": self.canonical_listing_count,
            "in_stock_canonical_listing_count": self.in_stock_canonical_listing_count,
            "category_listing_counts": [list(row) for row in self.category_listing_counts],
            "public_attribute_key_counts": [
                list(row) for row in self.public_attribute_key_counts
            ],
            "source_domains": list(self.source_domains),
            "meets_medium_individually": self.meets_medium_individually,
            "raw_rows_embedded": False,
            "attribute_values_embedded": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CatalogSourceFeasibilityV1":
        expected = {
            "merchant",
            "merchant_id",
            "source_file",
            "source_sha256",
            "source_bytes",
            "sellable_row_count",
            "canonical_listing_count",
            "in_stock_canonical_listing_count",
            "category_listing_counts",
            "public_attribute_key_counts",
            "source_domains",
            "meets_medium_individually",
            "raw_rows_embedded",
            "attribute_values_embedded",
        }
        if set(value) != expected:
            raise CatalogFeasibilityError("catalog source feasibility keys differ")
        raw_categories = value["category_listing_counts"]
        raw_attributes = value["public_attribute_key_counts"]
        raw_domains = value["source_domains"]
        if not all(isinstance(item, list) for item in (raw_categories, raw_attributes, raw_domains)):
            raise CatalogFeasibilityError("catalog source coverage fields must be arrays")
        if value["raw_rows_embedded"] is not False:
            raise CatalogFeasibilityError("catalog feasibility cannot embed raw rows")
        if value["attribute_values_embedded"] is not False:
            raise CatalogFeasibilityError("catalog feasibility cannot embed attribute values")
        categories = _pairs_to_dict(raw_categories, field="category_listing_counts")
        attributes = _pairs_to_dict(
            raw_attributes, field="public_attribute_key_counts"
        )
        return cls(
            merchant=str(value["merchant"]),
            merchant_id=str(value["merchant_id"]),
            source_file=str(value["source_file"]),
            source_sha256=str(value["source_sha256"]),
            source_bytes=int(value["source_bytes"]),
            sellable_row_count=int(value["sellable_row_count"]),
            canonical_listing_count=int(value["canonical_listing_count"]),
            in_stock_canonical_listing_count=int(
                value["in_stock_canonical_listing_count"]
            ),
            category_listing_counts=tuple(sorted(categories.items())),
            public_attribute_key_counts=tuple(sorted(attributes.items())),
            source_domains=tuple(map(str, raw_domains)),
            meets_medium_individually=bool(value["meets_medium_individually"]),
        )


@dataclass(frozen=True)
class CatalogScaleFeasibilityV1:
    profile: str
    target_listing_count: int
    available_in_stock_listing_count: int
    feasible: bool

    def __post_init__(self) -> None:
        if self.profile not in _EXPECTED_PROFILES:
            raise CatalogFeasibilityError("unknown catalog profile")
        if self.target_listing_count <= 0 or self.available_in_stock_listing_count < 0:
            raise CatalogFeasibilityError("catalog scale counts are invalid")
        if self.feasible is not (
            self.available_in_stock_listing_count >= self.target_listing_count
        ):
            raise CatalogFeasibilityError("catalog scale feasibility is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "target_listing_count": self.target_listing_count,
            "available_in_stock_listing_count": self.available_in_stock_listing_count,
            "feasible": self.feasible,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CatalogScaleFeasibilityV1":
        if set(value) != {
            "profile",
            "target_listing_count",
            "available_in_stock_listing_count",
            "feasible",
        }:
            raise CatalogFeasibilityError("catalog scale feasibility keys differ")
        return cls(
            profile=str(value["profile"]),
            target_listing_count=int(value["target_listing_count"]),
            available_in_stock_listing_count=int(
                value["available_in_stock_listing_count"]
            ),
            feasible=bool(value["feasible"]),
        )


@dataclass(frozen=True)
class CatalogFeasibilityReportV1:
    dataset_id: str
    provenance_manifest_sha256: str
    seed: int
    profile: str
    in_stock_only: bool
    governance: tuple[tuple[str, str], ...]
    sources: tuple[CatalogSourceFeasibilityV1, ...]
    selected_merchant_ids: tuple[str, ...]
    scales: tuple[CatalogScaleFeasibilityV1, ...]
    real_csv_5x5_status: str
    real_csv_5x5_reason: str

    def __post_init__(self) -> None:
        if self.dataset_id != "commerceworld-real-catalogs-v1":
            raise CatalogFeasibilityError("unexpected real-catalog dataset_id")
        _require_sha256(
            self.provenance_manifest_sha256,
            field="provenance_manifest_sha256",
        )
        if self.seed != DEFAULT_CATALOG_SEED:
            raise CatalogFeasibilityError("feasibility v1 is frozen to seed 42")
        if self.profile != DEFAULT_CATALOG_PROFILE or self.in_stock_only is not True:
            raise CatalogFeasibilityError("feasibility v1 requires medium/in_stock_only")
        if tuple(row.merchant for row in self.sources) != KNOWN_MERCHANTS:
            raise CatalogFeasibilityError("source order must follow KNOWN_MERCHANTS")
        if len(self.selected_merchant_ids) != 5:
            raise CatalogFeasibilityError("real-catalog 5x5 requires five merchants")
        expected_selected = tuple(
            row.merchant_id
            for row in sorted(
                self.sources,
                key=lambda row: (
                    -int(row.meets_medium_individually),
                    -row.in_stock_canonical_listing_count,
                    row.merchant_id,
                ),
            )[:5]
        )
        if self.selected_merchant_ids != expected_selected:
            raise CatalogFeasibilityError("five-merchant selection rule is inconsistent")
        if tuple(row.profile for row in self.scales) != _EXPECTED_PROFILES:
            raise CatalogFeasibilityError("catalog scale matrix is incomplete")
        selected = {
            row.merchant_id: row for row in self.sources if row.merchant_id in expected_selected
        }
        executable = (
            len(selected) == 5
            and all(row.in_stock_canonical_listing_count > 0 for row in selected.values())
            and next(row for row in self.scales if row.profile == "medium").feasible
        )
        expected_status = "executable_local_only" if executable else "not_applicable"
        if self.real_csv_5x5_status != expected_status:
            raise CatalogFeasibilityError("real CSV 5x5 status is inconsistent")
        if not self.real_csv_5x5_reason:
            raise CatalogFeasibilityError("real CSV 5x5 status requires a reason")

    @property
    def report_id(self) -> str:
        return _canonical_sha256(self.to_dict(include_report_id=False))

    def to_dict(self, *, include_report_id: bool = True) -> dict[str, Any]:
        selected_available = sum(
            row.in_stock_canonical_listing_count
            for row in self.sources
            if row.merchant_id in self.selected_merchant_ids
        )
        payload: dict[str, Any] = {
            "schema_version": CATALOG_FEASIBILITY_REPORT_SCHEMA,
            "dataset_id": self.dataset_id,
            "provenance_manifest_sha256": self.provenance_manifest_sha256,
            "seed": self.seed,
            "profile": self.profile,
            "target_listing_count": DEFAULT_CATALOG_TARGET,
            "in_stock_only": self.in_stock_only,
            "governance": dict(self.governance),
            "sources": [row.to_dict() for row in self.sources],
            "source_count": len(self.sources),
            "selected_merchant_ids": list(self.selected_merchant_ids),
            "merchant_selection_rule": (
                "meets_medium_desc,in_stock_canonical_listing_count_desc,merchant_id_asc"
            ),
            "selected_in_stock_listing_count": selected_available,
            "scale_feasibility": [row.to_dict() for row in self.scales],
            "cross_merchant_product_linkage": False,
            "real_csv_5x5_status": self.real_csv_5x5_status,
            "real_csv_5x5_reason": self.real_csv_5x5_reason,
            "paper_result": False,
            "raw_rows_embedded": False,
            "attribute_values_embedded": False,
        }
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CatalogFeasibilityReportV1":
        raw_sources = value.get("sources")
        raw_scales = value.get("scale_feasibility")
        raw_governance = value.get("governance")
        raw_selected = value.get("selected_merchant_ids")
        if not isinstance(raw_sources, list) or not isinstance(raw_scales, list):
            raise CatalogFeasibilityError("catalog feasibility arrays are missing")
        if not isinstance(raw_governance, Mapping) or not isinstance(raw_selected, list):
            raise CatalogFeasibilityError("catalog feasibility metadata is malformed")
        report = cls(
            dataset_id=str(value.get("dataset_id")),
            provenance_manifest_sha256=str(value.get("provenance_manifest_sha256")),
            seed=int(value.get("seed", -1)),
            profile=str(value.get("profile")),
            in_stock_only=value.get("in_stock_only") is True,
            governance=tuple(sorted((str(key), str(item)) for key, item in raw_governance.items())),
            sources=tuple(
                CatalogSourceFeasibilityV1.from_dict(row)
                for row in raw_sources
                if isinstance(row, Mapping)
            ),
            selected_merchant_ids=tuple(map(str, raw_selected)),
            scales=tuple(
                CatalogScaleFeasibilityV1.from_dict(row)
                for row in raw_scales
                if isinstance(row, Mapping)
            ),
            real_csv_5x5_status=str(value.get("real_csv_5x5_status")),
            real_csv_5x5_reason=str(value.get("real_csv_5x5_reason")),
        )
        if dict(value) != report.to_dict():
            raise CatalogFeasibilityError("catalog feasibility derived fields differ")
        return report


def build_catalog_feasibility_report(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    repository_root: str | Path | None = None,
    seed: int = DEFAULT_CATALOG_SEED,
) -> CatalogFeasibilityReportV1:
    """Recompute the source-bound, medium/in-stock feasibility report."""

    path = Path(manifest_path).resolve()
    root = Path(repository_root).resolve() if repository_root is not None else path.parents[1]
    manifest = load_catalog_provenance(path)
    verified = verify_catalog_provenance(
        manifest_path=path,
        repository_root=root,
    )
    source_by_merchant = {
        str(row["merchant"]): row for row in verified["sources"]
    }
    source_rows: list[CatalogSourceFeasibilityV1] = []
    for merchant in KNOWN_MERCHANTS:
        categories: dict[str, int] = {}
        attributes: dict[str, int] = {}
        for listing in iter_public_listings(
            merchant,
            data_root=root / "data" / "raw_data",
            seed=seed,
        ):
            if int(listing["inventory"]) <= 0:
                continue
            category = str(listing["category"])
            categories[category] = categories.get(category, 0) + 1
            raw_attributes = listing["attributes"]
            if not isinstance(raw_attributes, Mapping):
                raise CatalogFeasibilityError("public listing attributes must be a mapping")
            for key in raw_attributes:
                name = str(key)
                attributes[name] = attributes.get(name, 0) + 1
        stats = source_by_merchant[merchant]
        in_stock = int(stats["in_stock_canonical_listing_count"])
        if sum(categories.values()) != in_stock:
            raise CatalogFeasibilityError(
                f"{merchant} public iterator in-stock count differs from provenance"
            )
        source_rows.append(
            CatalogSourceFeasibilityV1(
                merchant=merchant,
                merchant_id=f"merchant:{merchant}",
                source_file=str(stats["source_file"]),
                source_sha256=str(stats["sha256"]),
                source_bytes=int(stats["bytes"]),
                sellable_row_count=int(stats["sellable_row_count"]),
                canonical_listing_count=int(stats["canonical_listing_count"]),
                in_stock_canonical_listing_count=in_stock,
                category_listing_counts=tuple(sorted(categories.items())),
                public_attribute_key_counts=tuple(sorted(attributes.items())),
                source_domains=tuple(map(str, stats["source_domains"])),
                meets_medium_individually=in_stock >= DEFAULT_CATALOG_TARGET,
            )
        )

    selected = tuple(
        row.merchant_id
        for row in sorted(
            source_rows,
            key=lambda row: (
                -int(row.meets_medium_individually),
                -row.in_stock_canonical_listing_count,
                row.merchant_id,
            ),
        )[:5]
    )
    selected_available = sum(
        row.in_stock_canonical_listing_count
        for row in source_rows
        if row.merchant_id in selected
    )
    targets = manifest.get("scale_profiles")
    if not isinstance(targets, Mapping):
        raise CatalogFeasibilityError("provenance scale_profiles is missing")
    scales = tuple(
        CatalogScaleFeasibilityV1(
            profile=profile,
            target_listing_count=int(targets[profile]),
            available_in_stock_listing_count=selected_available,
            feasible=selected_available >= int(targets[profile]),
        )
        for profile in _EXPECTED_PROFILES
    )
    medium_feasible = next(row for row in scales if row.profile == "medium").feasible
    selected_sources = [row for row in source_rows if row.merchant_id in selected]
    executable = (
        len(selected_sources) == 5
        and all(row.in_stock_canonical_listing_count > 0 for row in selected_sources)
        and medium_feasible
    )
    if executable:
        status = "executable_local_only"
        reason = (
            "Five merchants have in-stock public listings and jointly satisfy the medium "
            "profile; license and publication permission remain unresolved."
        )
    else:
        status = "not_applicable"
        reason = (
            "Five source-bound merchants cannot jointly satisfy the in-stock medium profile."
        )
    governance = tuple(
        sorted(
            (str(key), str(value))
            for key, value in dict(verified.get("governance") or {}).items()
        )
    )
    return CatalogFeasibilityReportV1(
        dataset_id=str(manifest["dataset_id"]),
        provenance_manifest_sha256=_file_sha256(path),
        seed=seed,
        profile=DEFAULT_CATALOG_PROFILE,
        in_stock_only=DEFAULT_IN_STOCK_ONLY,
        governance=governance,
        sources=tuple(source_rows),
        selected_merchant_ids=selected,
        scales=scales,
        real_csv_5x5_status=status,
        real_csv_5x5_reason=reason,
    )


def verify_catalog_feasibility_report(
    report: CatalogFeasibilityReportV1,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    repository_root: str | Path | None = None,
) -> None:
    expected = build_catalog_feasibility_report(
        manifest_path=manifest_path,
        repository_root=repository_root,
        seed=report.seed,
    )
    if report != expected:
        raise CatalogFeasibilityError(
            "catalog feasibility report differs from current verified source bytes"
        )


__all__ = [
    "CATALOG_FEASIBILITY_REPORT_SCHEMA",
    "CatalogFeasibilityError",
    "CatalogFeasibilityReportV1",
    "CatalogScaleFeasibilityV1",
    "CatalogSourceFeasibilityV1",
    "build_catalog_feasibility_report",
    "verify_catalog_feasibility_report",
]
