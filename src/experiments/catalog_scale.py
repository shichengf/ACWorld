"""Deterministic, model-free real-catalog scale experiment axis.

This suite is deliberately separate from :mod:`experiments.plan`.  The main
model benchmark identifies runs by model/variant/seed/role/population, while a
catalog-scale cell additionally needs a catalog profile.  Reusing a main-plan
identity would silently alias four different datasets.  The independent schema
below instead materializes twelve catalog probes (four profiles x three seeds)
and lets the S2/S12/S18 engineering cells reference them.

The probes establish only catalog construction, exact structured search/read,
and in-memory/SQLite parity.  They do not execute an agent, score S2 or S12, or
claim the full S18 market-comparison lane.  The shipped provenance manifest
explicitly has no verified cross-merchant product crosswalk, so S18 is
fail-closed to ``catalog_size_search_only``.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from episode.benchmark import CatalogScale
from episode.catalog_provenance import (
    DEFAULT_MANIFEST,
    load_catalog_provenance,
    verify_catalog_provenance,
)
from episode.seed_catalog import seed_world_catalog
from world.state import World
from world.store import DatabaseWorld
from world.tools import WorldTools


CATALOG_SCALE_PLAN_SCHEMA = "cwe.catalog-scale-plan.v1"
CATALOG_SCALE_REPORT_SCHEMA = "cwe.catalog-scale-report.v1"
CATALOG_SCALE_VERIFICATION_SCHEMA = "cwe.catalog-scale-verification.v1"
CATALOG_SCALE_SUITE_ID = "commerceworld-real-catalog-scale-v1"

CATALOG_SCALE_VARIANTS: tuple[str, ...] = ("S2", "S12", "S18")
CATALOG_SCALE_SEEDS: tuple[int, ...] = (42, 1337, 2024)
CATALOG_SCALE_PROFILES: tuple[CatalogScale, ...] = (
    CatalogScale.SMOKE,
    CatalogScale.STANDARD,
    CatalogScale.MEDIUM,
    CatalogScale.FULL,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _REPOSITORY_ROOT / "data" / "raw_data"
_S18_NO_CROSSWALK_REASON = (
    "The real-CSV provenance declares no verified cross-merchant product "
    "crosswalk; this cell measures catalog-size/search execution only and "
    "must not invoke the S18 equivalent-product market oracle."
)


class CatalogScaleVerificationError(ValueError):
    """A plan/report or its recomputed evidence failed closed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_digest(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = _json_sha256(result)
    return result


def _verify_digest(payload: Mapping[str, Any], field: str) -> None:
    stored = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    actual = _json_sha256(unsigned)
    if stored != actual:
        raise CatalogScaleVerificationError(
            f"{field} mismatch: stored={stored!r}, computed={actual!r}"
        )


def _manifest_context(
    manifest_path: str | Path,
    repository_root: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    path = Path(manifest_path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else path.parents[1]
    )
    manifest = load_catalog_provenance(path)
    verified = verify_catalog_provenance(
        manifest_path=path,
        repository_root=root,
    )
    return manifest, verified, path, root / "data" / "raw_data"


def _profile_targets(manifest: Mapping[str, Any]) -> dict[str, int]:
    raw = manifest.get("scale_profiles")
    if not isinstance(raw, Mapping):
        raise CatalogScaleVerificationError("provenance scale_profiles is missing")
    targets: dict[str, int] = {}
    for profile in CATALOG_SCALE_PROFILES:
        value = raw.get(profile.value)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CatalogScaleVerificationError(
                f"invalid real-catalog target for {profile.value}: {value!r}"
            )
        targets[profile.value] = value
    expected = {
        CatalogScale.SMOKE.value: 5,
        CatalogScale.STANDARD.value: 50,
        CatalogScale.MEDIUM.value: 250,
        CatalogScale.FULL.value: int(
            manifest.get("aggregate", {}).get("canonical_listing_count", -1)
        ),
    }
    if targets != expected:
        raise CatalogScaleVerificationError(
            f"unexpected catalog scale profile targets: {targets!r}"
        )
    return targets


def _probe_id(*, seed: int, profile: str, dataset_id: str) -> str:
    identity = {
        "schema_version": "cwe.catalog-scale-probe-identity.v1",
        "suite_id": CATALOG_SCALE_SUITE_ID,
        "dataset_id": dataset_id,
        "seed": seed,
        "catalog_profile": profile,
    }
    return f"catalog-probe-{_json_sha256(identity)[:24]}"


def _cell_id(*, variant_id: str, seed: int, profile: str, probe_id: str) -> str:
    identity = {
        "schema_version": "cwe.catalog-scale-cell-identity.v1",
        "suite": "catalog_scale",
        "variant_id": variant_id,
        "seed": seed,
        "catalog_profile": profile,
        "probe_id": probe_id,
    }
    return f"catalog-scale-cell-{_json_sha256(identity)[:24]}"


def _lane_definition(variant_id: str) -> dict[str, Any]:
    if variant_id == "S2":
        return {
            "lane": "real_catalog_constraint_query_readiness",
            "claim_scope": "catalog construction and exact structured query/read only",
            "agent_task_score_applicable": False,
            "reason": (
                "This engineering axis does not transplant the inline S2 answer key "
                "onto unrelated real products."
            ),
        }
    if variant_id == "S12":
        return {
            "lane": "real_catalog_authoritative_read_readiness",
            "claim_scope": "authoritative structured listing-read execution only",
            "decision_bound_agent_score_applicable": False,
            "reason": (
                "A catalog probe has no agent decision trace, so it cannot satisfy "
                "the S12 decision-bound grounding score by itself."
            ),
        }
    if variant_id == "S18":
        return {
            "lane": "catalog_size_search_only",
            "claim_scope": "catalog-size/search engineering axis only",
            "cross_merchant_market_oracle_applicable": False,
            "reason": _S18_NO_CROSSWALK_REASON,
        }
    raise CatalogScaleVerificationError(f"unsupported catalog-scale variant: {variant_id}")


def build_catalog_scale_plan(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the exact 36-cell offline plan, bound to verified source bytes."""

    manifest, verified, path, _data_root = _manifest_context(
        manifest_path, repository_root
    )
    targets = _profile_targets(manifest)
    canonicalization = manifest.get("canonicalization", {})
    linkage = str(canonicalization.get("cross_merchant_product_linkage", ""))
    if not linkage.startswith("not_inferred"):
        raise CatalogScaleVerificationError(
            "catalog-scale v1 expects the real sources to have no verified "
            "cross-merchant product crosswalk"
        )

    probes: list[dict[str, Any]] = []
    by_axis: dict[tuple[int, str], str] = {}
    for profile in CATALOG_SCALE_PROFILES:
        for seed in CATALOG_SCALE_SEEDS:
            probe_id = _probe_id(
                seed=seed,
                profile=profile.value,
                dataset_id=str(manifest["dataset_id"]),
            )
            by_axis[(seed, profile.value)] = probe_id
            probes.append({
                "probe_id": probe_id,
                "seed": seed,
                "catalog_profile": profile.value,
                "expected_listing_count": targets[profile.value],
                "catalog_source": "real_csv",
                "model_inference": False,
            })

    cells: list[dict[str, Any]] = []
    for variant_id in CATALOG_SCALE_VARIANTS:
        for profile in CATALOG_SCALE_PROFILES:
            for seed in CATALOG_SCALE_SEEDS:
                probe_id = by_axis[(seed, profile.value)]
                cells.append({
                    "cell_id": _cell_id(
                        variant_id=variant_id,
                        seed=seed,
                        profile=profile.value,
                        probe_id=probe_id,
                    ),
                    "suite": "catalog_scale",
                    "variant_id": variant_id,
                    "seed": seed,
                    "catalog_profile": profile.value,
                    "expected_listing_count": targets[profile.value],
                    "probe_id": probe_id,
                    **_lane_definition(variant_id),
                })

    payload = {
        "schema_version": CATALOG_SCALE_PLAN_SCHEMA,
        "suite_id": CATALOG_SCALE_SUITE_ID,
        "experiment_class": "offline_deterministic_engineering_axis",
        "main_plan_relationship": {
            "modifies_main_model_plan": False,
            "formal_main_plan_run_count": 2160,
            "included_in_formal_main_episode_count": False,
            # Backward-compatible field retained for readers of the original
            # preregistration draft; 3,888 is no longer the executable plan.
            "included_in_3888_episode_count": False,
            "uses_model_inference": False,
            "uses_scenario_catalog_identity": False,
            "identity_note": (
                "catalog profile is part of this suite identity; cells do not "
                "reuse main RunSpec or ScenarioCatalog keys"
            ),
        },
        "source": {
            "dataset_id": manifest["dataset_id"],
            "data_label": "real_csv",
            "provenance_manifest_sha256": _file_sha256(path),
            "source_count": verified["aggregate"]["source_count"],
            "canonical_listing_count": verified["aggregate"][
                "canonical_listing_count"
            ],
            "selection_rule": manifest["scale_profiles"]["selection_rule"],
            "cross_merchant_product_linkage": linkage,
            "raw_rows_embedded_in_plan": False,
        },
        "publication_gate": {
            "paper_use_status": verified["governance"].get("paper_use_status"),
            "publication_permission": verified["governance"].get(
                "publication_permission"
            ),
            "engineering_probe_allowed": True,
            "paper_claim_ready": False,
        },
        "seeds": list(CATALOG_SCALE_SEEDS),
        "profiles": [
            {"name": profile.value, "expected_listing_count": targets[profile.value]}
            for profile in CATALOG_SCALE_PROFILES
        ],
        "dataset_probes": probes,
        "cells": cells,
        "counts": {
            "dataset_probes": len(probes),
            "cells": len(cells),
            "variants": len(CATALOG_SCALE_VARIANTS),
            "profiles": len(CATALOG_SCALE_PROFILES),
            "seeds": len(CATALOG_SCALE_SEEDS),
        },
        "evidence_boundary": {
            "validates": [
                "source provenance and profile cardinality",
                "catalog and inventory reproducibility hashes",
                "exact structured query and authoritative listing read",
                "in-memory and SQLite backend parity",
            ],
            "does_not_validate": [
                "agent quality or any model leaderboard score",
                "S12 decision-bound agent grounding",
                "S18 cross-merchant product equivalence or market-optimal choice",
                "dataset publication permission",
            ],
        },
    }
    return _with_digest(payload, "plan_sha256")


def validate_catalog_scale_plan(
    plan: Mapping[str, Any],
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless ``plan`` is the exact source-bound canonical matrix."""

    if plan.get("schema_version") != CATALOG_SCALE_PLAN_SCHEMA:
        raise CatalogScaleVerificationError(
            f"unsupported catalog-scale plan schema: {plan.get('schema_version')!r}"
        )
    _verify_digest(plan, "plan_sha256")
    expected = build_catalog_scale_plan(
        manifest_path=manifest_path,
        repository_root=repository_root,
    )
    if dict(plan) != expected:
        raise CatalogScaleVerificationError(
            "catalog-scale plan differs from the canonical source-bound matrix"
        )
    return {
        "schema_version": CATALOG_SCALE_VERIFICATION_SCHEMA,
        "kind": "plan",
        "verified": True,
        "plan_sha256": plan["plan_sha256"],
        "dataset_probes": len(plan["dataset_probes"]),
        "cells": len(plan["cells"]),
        "modifies_main_model_plan": False,
    }


def write_catalog_scale_plan(plan: Mapping[str, Any], path: str | Path) -> Path:
    """Write a plan once; never overwrite an existing experiment manifest."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(plan), indent=2, sort_keys=True) + "\n")
    return target


def load_catalog_scale_plan(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CatalogScaleVerificationError("catalog-scale plan must be a JSON object")
    return raw


def _listing_payload(listing: Any) -> dict[str, Any]:
    amount = Decimal(listing.list_price.amount).quantize(Decimal("0.01"))
    return {
        "sku_id": str(listing.sku_id),
        "merchant_id": str(listing.merchant_id),
        "product_id": listing.product_id,
        "category": listing.category,
        "name": listing.name,
        "attributes": listing.attributes,
        "list_price": {
            "amount": format(amount, ".2f"),
            "currency": listing.list_price.currency,
        },
    }


def _snapshot_payload(snapshot: Any) -> dict[str, Any]:
    catalog = sorted(
        (_listing_payload(listing) for listing in snapshot.catalog),
        key=lambda item: item["sku_id"],
    )
    inventory = sorted(
        (
            {
                "sku_id": str(row.sku_id),
                "merchant_id": str(row.merchant_id),
                "qty_available": int(row.qty_available),
                "qty_reserved": int(row.qty_reserved),
            }
            for row in snapshot.inventory.values()
        ),
        key=lambda item: item["sku_id"],
    )
    return {"catalog": catalog, "inventory": inventory}


def _query_evidence(world: Any, snapshot: Any) -> dict[str, Any]:
    if not snapshot.catalog:
        raise CatalogScaleVerificationError("catalog probe seeded no listings")
    anchor = min(snapshot.catalog, key=lambda listing: str(listing.sku_id))
    filters: dict[str, object] = {
        "sku_id": str(anchor.sku_id),
        "merchant_id": str(anchor.merchant_id),
        "category": anchor.category,
    }
    if anchor.product_id is not None:
        filters["product_id"] = anchor.product_id
    tools = WorldTools(world, caller_id="buyer:catalog-scale-probe")
    results = tools.search_catalog(str(anchor.sku_id), filters, limit=5)
    fetched = tools.get_listing(anchor.sku_id)
    if [str(item.sku_id) for item in results] != [str(anchor.sku_id)]:
        raise CatalogScaleVerificationError(
            "exact catalog query did not return only the authoritative anchor listing"
        )
    if fetched != anchor:
        raise CatalogScaleVerificationError(
            "authoritative listing read differs from the catalog query anchor"
        )
    return {
        "query": str(anchor.sku_id),
        "filters": {key: str(value) for key, value in sorted(filters.items())},
        "limit": 5,
        "result_sku_ids": [str(item.sku_id) for item in results],
        "authoritative_sku_id": str(fetched.sku_id),
        "authoritative_listing_sha256": _json_sha256(_listing_payload(fetched)),
        "verified": True,
    }


def _execute_dataset_probe(
    spec: Mapping[str, Any],
    *,
    data_root: Path,
) -> dict[str, Any]:
    profile = str(spec["catalog_profile"])
    seed = int(spec["seed"])
    expected_count = int(spec["expected_listing_count"])

    memory = World()
    durable = DatabaseWorld()
    try:
        memory_summary = seed_world_catalog(
            memory,
            seed=seed,
            catalog_scale=profile,
            data_root=str(data_root),
        )
        durable_summary = seed_world_catalog(
            durable,
            seed=seed,
            catalog_scale=profile,
            data_root=str(data_root),
        )
        memory_snapshot = memory.snapshot()
        durable_snapshot = durable.snapshot()
        memory_payload = _snapshot_payload(memory_snapshot)
        durable_payload = _snapshot_payload(durable_snapshot)
        if memory_summary["listings"] != expected_count:
            raise CatalogScaleVerificationError(
                f"{spec['probe_id']} seeded {memory_summary['listings']} listings; "
                f"expected {expected_count}"
            )
        if durable_summary["listings"] != expected_count:
            raise CatalogScaleVerificationError(
                f"SQLite {spec['probe_id']} seeded {durable_summary['listings']} "
                f"listings; expected {expected_count}"
            )
        if memory_payload != durable_payload:
            raise CatalogScaleVerificationError(
                f"in-memory/SQLite catalog mismatch for {spec['probe_id']}"
            )

        sku_ids = [item["sku_id"] for item in memory_payload["catalog"]]
        if len(sku_ids) != len(set(sku_ids)):
            raise CatalogScaleVerificationError(
                f"duplicate global listing key in {spec['probe_id']}"
            )
        if any(
            item["attributes"].get("source_label") != "real_csv"
            for item in memory_payload["catalog"]
        ):
            raise CatalogScaleVerificationError(
                f"non-real_csv listing entered {spec['probe_id']}"
            )

        memory_query = _query_evidence(memory, memory_snapshot)
        durable_query = _query_evidence(durable, durable_snapshot)
        if memory_query != durable_query:
            raise CatalogScaleVerificationError(
                f"in-memory/SQLite query evidence mismatch for {spec['probe_id']}"
            )

        catalog_hash = _json_sha256(memory_payload["catalog"])
        inventory_hash = _json_sha256(memory_payload["inventory"])
        merchants = sorted({item["merchant_id"] for item in memory_payload["catalog"]})
        product_owners: dict[str, set[str]] = {}
        for item in memory_payload["catalog"]:
            product_id = item["product_id"]
            if product_id is not None:
                product_owners.setdefault(str(product_id), set()).add(item["merchant_id"])
        shared_ids = sorted(
            product_id
            for product_id, owners in product_owners.items()
            if len(owners) > 1
        )
        if shared_ids:
            raise CatalogScaleVerificationError(
                "real catalog contains cross-merchant product ids without a declared "
                "verified crosswalk"
            )

        payload = {
            "probe_id": spec["probe_id"],
            "seed": seed,
            "catalog_profile": profile,
            "expected_listing_count": expected_count,
            "listing_count": len(memory_payload["catalog"]),
            "inventory_row_count": len(memory_payload["inventory"]),
            "merchant_count_in_profile": len(merchants),
            "catalog_sha256": catalog_hash,
            "inventory_sha256": inventory_hash,
            "seeded_world_sha256": _json_sha256(memory_payload),
            "query_evidence": memory_query,
            "backend_parity": {
                "in_memory": True,
                "sqlite": True,
                "state_equal": True,
                "query_equal": True,
            },
            "cross_merchant_product_crosswalk": {
                "declared_available": False,
                "shared_product_ids_used": 0,
                "s18_market_oracle_applicable": False,
                "reason": _S18_NO_CROSSWALK_REASON,
            },
            "raw_catalog_rows_embedded": False,
        }
        return _with_digest(payload, "probe_sha256")
    finally:
        durable.close()


def _cell_results(
    plan: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(item["probe_id"]): item for item in probes}
    results: list[dict[str, Any]] = []
    for cell in plan["cells"]:
        evidence = by_id.get(str(cell["probe_id"]))
        if evidence is None:
            continue
        result = {
            "cell_id": cell["cell_id"],
            "variant_id": cell["variant_id"],
            "seed": cell["seed"],
            "catalog_profile": cell["catalog_profile"],
            "lane": cell["lane"],
            "probe_id": cell["probe_id"],
            "status": "verified",
            "catalog_sha256": evidence["catalog_sha256"],
            "query_evidence_sha256": evidence["query_evidence"][
                "authoritative_listing_sha256"
            ],
            "claim_scope": cell["claim_scope"],
        }
        if cell["variant_id"] == "S18":
            result["cross_merchant_market_oracle_applicable"] = False
            result["reason"] = _S18_NO_CROSSWALK_REASON
        results.append(result)
    return results


def _build_report(
    plan: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    completed = len(probes)
    total = len(plan["dataset_probes"])
    cells = _cell_results(plan, probes)
    payload = {
        "schema_version": CATALOG_SCALE_REPORT_SCHEMA,
        "suite_id": CATALOG_SCALE_SUITE_ID,
        "plan": dict(plan),
        "probes": [dict(item) for item in probes],
        "cells": cells,
        "progress": {
            "completed_dataset_probes": completed,
            "total_dataset_probes": total,
            "verified_cells": len(cells),
            "total_cells": len(plan["cells"]),
            "complete": completed == total,
        },
        "evidence_boundary": dict(plan["evidence_boundary"]),
        "raw_catalog_rows_embedded": False,
    }
    return _with_digest(payload, "report_sha256")


def _write_checkpoint(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_report(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CatalogScaleVerificationError("catalog-scale report must be a JSON object")
    if raw.get("schema_version") != CATALOG_SCALE_REPORT_SCHEMA:
        raise CatalogScaleVerificationError(
            f"unsupported catalog-scale report schema: {raw.get('schema_version')!r}"
        )
    _verify_digest(raw, "report_sha256")
    return raw


def _validate_stored_probes(
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    data_root: Path,
) -> list[dict[str, Any]]:
    if report.get("plan") != dict(plan):
        raise CatalogScaleVerificationError(
            "checkpoint embeds a different catalog-scale plan"
        )
    expected = {
        str(item["probe_id"]): item for item in plan["dataset_probes"]
    }
    stored: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in report.get("probes", []):
        if not isinstance(raw, Mapping):
            raise CatalogScaleVerificationError("stored probe must be a JSON object")
        probe_id = str(raw.get("probe_id", ""))
        if probe_id in seen:
            raise CatalogScaleVerificationError(f"duplicate stored probe: {probe_id}")
        spec = expected.get(probe_id)
        if spec is None:
            raise CatalogScaleVerificationError(f"unknown stored probe: {probe_id}")
        _verify_digest(raw, "probe_sha256")
        recomputed = _execute_dataset_probe(spec, data_root=data_root)
        if dict(raw) != recomputed:
            raise CatalogScaleVerificationError(
                f"stored probe differs from recomputed evidence: {probe_id}"
            )
        stored.append(dict(raw))
        seen.add(probe_id)
    canonical_order = {
        str(spec["probe_id"]): index
        for index, spec in enumerate(plan["dataset_probes"])
    }
    stored.sort(key=lambda item: canonical_order[str(item["probe_id"])])
    expected_report = _build_report(plan, stored)
    if dict(report) != expected_report:
        raise CatalogScaleVerificationError(
            "checkpoint metadata/cells differ from its recomputed probes"
        )
    return stored


def run_catalog_scale_probe(
    plan: Mapping[str, Any],
    report_path: str | Path,
    *,
    resume: bool = False,
    max_probes: int | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run or resume the source-bound probes, checkpointing after each dataset.

    ``max_probes`` limits new work in this invocation and is useful for testing
    interruption/recovery.  It never changes the plan or marks a partial report
    complete.
    """

    validate_catalog_scale_plan(
        plan,
        manifest_path=manifest_path,
        repository_root=repository_root,
    )
    _manifest, _verified, _path, data_root = _manifest_context(
        manifest_path, repository_root
    )
    if max_probes is not None and max_probes < 0:
        raise ValueError("max_probes must be non-negative")
    target = Path(report_path)
    if target.exists() and not resume:
        raise FileExistsError(f"catalog-scale report already exists: {target}")

    completed: list[dict[str, Any]] = []
    if target.exists():
        completed = _validate_stored_probes(
            _load_report(target),
            plan,
            data_root=data_root,
        )
    elif resume:
        raise FileNotFoundError(f"catalog-scale resume report does not exist: {target}")

    completed_ids = {str(item["probe_id"]) for item in completed}
    remaining = [
        spec
        for spec in plan["dataset_probes"]
        if str(spec["probe_id"]) not in completed_ids
    ]
    allowance = len(remaining) if max_probes is None else max_probes
    if not target.exists():
        _write_checkpoint(_build_report(plan, completed), target)
    for spec in remaining[:allowance]:
        completed.append(_execute_dataset_probe(spec, data_root=data_root))
        _write_checkpoint(_build_report(plan, completed), target)
    return _build_report(plan, completed)


def verify_catalog_scale_report(
    report_path: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Independently recompute every probe; partial reports are not evidence."""

    report = _load_report(Path(report_path))
    plan = report.get("plan")
    if not isinstance(plan, Mapping):
        raise CatalogScaleVerificationError("catalog-scale report has no embedded plan")
    validate_catalog_scale_plan(
        plan,
        manifest_path=manifest_path,
        repository_root=repository_root,
    )
    _manifest, _verified, _path, data_root = _manifest_context(
        manifest_path, repository_root
    )
    probes = _validate_stored_probes(report, plan, data_root=data_root)
    if len(probes) != len(plan["dataset_probes"]):
        raise CatalogScaleVerificationError(
            "catalog-scale report is an incomplete checkpoint, not a verified suite"
        )
    if not report["progress"]["complete"]:
        raise CatalogScaleVerificationError(
            "catalog-scale report does not declare complete progress"
        )
    s18_cells = [cell for cell in report["cells"] if cell["variant_id"] == "S18"]
    if not s18_cells or any(
        cell.get("cross_merchant_market_oracle_applicable") is not False
        for cell in s18_cells
    ):
        raise CatalogScaleVerificationError(
            "S18 catalog-scale cells must remain fail-closed to search-only scope"
        )
    return {
        "schema_version": CATALOG_SCALE_VERIFICATION_SCHEMA,
        "kind": "report",
        "verified": True,
        "report_sha256": report["report_sha256"],
        "dataset_probes_recomputed": len(probes),
        "cells_verified": len(report["cells"]),
        "listing_targets": [5, 50, 250, 1082],
        "model_inference": False,
        "modifies_main_model_plan": False,
        "s18_scope": "catalog_size_search_only",
        "s18_cross_merchant_market_oracle_applicable": False,
        "paper_result": False,
    }


__all__ = [
    "CATALOG_SCALE_PLAN_SCHEMA",
    "CATALOG_SCALE_PROFILES",
    "CATALOG_SCALE_REPORT_SCHEMA",
    "CATALOG_SCALE_SEEDS",
    "CATALOG_SCALE_SUITE_ID",
    "CATALOG_SCALE_VARIANTS",
    "CATALOG_SCALE_VERIFICATION_SCHEMA",
    "CatalogScaleVerificationError",
    "build_catalog_scale_plan",
    "load_catalog_scale_plan",
    "run_catalog_scale_probe",
    "validate_catalog_scale_plan",
    "verify_catalog_scale_report",
    "write_catalog_scale_plan",
]
