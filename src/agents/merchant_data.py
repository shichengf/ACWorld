"""Legacy Excel loader for the merchant tools.

Reads a compatible dataset layout from any supplied directory into a
``MerchantData`` store that ``merchant_tools.py``
consumes. Excel parsing uses pandas through a lazy import. The project ships
zero runtime deps, so the pure tool logic in ``merchant_tools`` stays usable
without pandas. Only this loader needs it.

Expected layout. All files are optional and loading is best effort:

    <data>/excel/traffic-pools-summary.xlsx        product/category summary
    <data>/excel/pool-L2-<category>.xlsx           per-category keyword pools
    <data>/excel/master-keyword-analysis.xlsx      fallback keyword detail
    <data>/reference/核心竞品分析.xlsx             competitor + product detail
    <data>/listings.xlsx (or merchant_listings)    OPTIONAL transactional
                                                    catalog

The transactional catalog (price / floor / inventory / policy) does **not**
exist in the research layout. If no ``listings.xlsx`` is present, the Tier-1
tools have an empty catalog. That gap is by design, not a loader bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.merchant_tools import MerchantData


def _require_pandas():
    try:
        import pandas as pd  # noqa: PLC0415
        return pd
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "merchant_data.load_from_xlsx needs pandas + openpyxl: "
            "pip install pandas openpyxl. (The pure tools in "
            "agents.merchant_tools work without them. Build MerchantData "
            "via MerchantData.from_rows for dependency-free use.)"
        ) from e


def _category_from_pool_filename(name: str) -> str:
    """``pool-L2-自动猫砂盆.xlsx`` -> ``自动猫砂盆``."""
    stem = Path(name).stem  # pool-L2-自动猫砂盆
    parts = stem.split("-", 2)
    return parts[2] if len(parts) == 3 else stem


def load_from_xlsx(data_dir: str | Path) -> MerchantData:
    """Load a compatible Excel dataset into a ``MerchantData`` store."""
    pd = _require_pandas()
    root = Path(data_dir)
    excel = root / "excel"
    ref = root / "reference"

    def read(p: Path) -> list[dict[str, Any]]:
        if not p.is_file():
            return []
        df = pd.read_excel(p)
        df = df.where(df.notna(), None)
        return df.to_dict("records")

    # --- pool summary (product/category list + visibility) ---
    pool_summary: list[dict[str, Any]] = []
    for r in read(excel / "traffic-pools-summary.xlsx"):
        pool_summary.append({
            "category": r.get("category"),
            "keyword_count": r.get("keyword_count"),
            "comp_total_clicks": r.get("comp_total_clicks"),
            "mw_covered": r.get("mw_covered"),
            "mw_not_covered": r.get("mw_not_covered"),
            "high_opp_count": r.get("high_opp_count"),
            "coverage_rate": r.get("coverage_rate"),
        })

    # --- per-category keyword pools (filename encodes the category) ---
    keywords: list[dict[str, Any]] = []
    for f in sorted(excel.glob("pool-L2-*.xlsx")) if excel.is_dir() else []:
        cat = _category_from_pool_filename(f.name)
        for r in read(f):
            keywords.append({
                "product": cat,
                "keyword": r.get("keyword"),
                "volume": r.get("volume"),
                "kd": r.get("kd"),
                "intent": r.get("intent"),
                "user_scenario": r.get("user_scenario"),
                "content_hook": r.get("content_hook"),
                "channel": r.get("channel"),
                "opp_score": r.get("opp_score"),
                "mw_status": r.get("mw_status"),
            })

    # --- competitors + product detail (price benchmarks parsed downstream) ---
    competitors: list[dict[str, Any]] = []
    for fname in ("核心竞品分析.xlsx", "core-competitors.xlsx"):
        for r in read(ref / fname):
            competitors.append({
                "domain": r.get("domain"),
                "productCategory": r.get("productCategory"),
                "productDetail": r.get("productDetail"),
                "totalTraffic": r.get("totalTraffic"),
                "organicTraffic": r.get("organicTraffic"),
                "relevance": r.get("relevance"),
                "maxOverlapScore": r.get("maxOverlapScore"),
            })

    # Optional transactional catalog.
    listings: list[dict[str, Any]] = []
    policy: dict[str, Any] | None = None
    for fname in ("listings.xlsx", "merchant_listings.xlsx"):
        rows = read(root / fname)
        if rows:
            for r in rows:
                listings.append({
                    "product": r.get("product"),
                    "list_price": r.get("list_price"),
                    "floor_price": r.get("floor_price"),
                    "inventory": r.get("inventory"),
                    "attributes": r.get("attributes") or {},
                    "auto_accept_threshold": r.get("auto_accept_threshold"),
                    "target_band": r.get("target_band"),
                    "permitted_claims": r.get("permitted_claims") or (),
                    "must_not_claim": r.get("must_not_claim") or (),
                })
            break

    return MerchantData.from_rows(
        listings=listings,
        pool_summary=pool_summary,
        keywords=keywords,
        competitors=competitors,
        policy=policy,
    )
