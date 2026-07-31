"""Streaming SQLite backend for the large real-product catalog."""

from __future__ import annotations

import base64
import csv
import json
import re
import sqlite3
import time
import zlib
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

from large_catalog.models import CatalogListing, SearchPage, SearchRequest


EXPECTED_COLUMNS = frozenset(
    {
        "Product Name",
        "Product Type",
        "Vendor",
        "Sale Price (USD)",
        "SKU",
        "Availability",
        "Variant",
        "Tags",
        "Original Price (USD)",
        "Options",
        "Material",
        "Key Features",
        "Warranty",
        "Product URL",
    }
)
PAGE_SIZE_MAX = 100
INGEST_BATCH_SIZE = 5_000
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]{1,}")


class CatalogDataError(ValueError):
    """The raw catalog or a catalog query is malformed."""


@dataclass(frozen=True, slots=True)
class CatalogIngestReport:
    files: int
    source_rows: int
    positive_price_rows: int
    source_in_stock_rows: int
    transactable_in_stock_rows: int
    missing_sku_rows: int
    unique_product_urls: int
    merchants: int
    database_bytes: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _money_minor(value: object) -> int | None:
    raw = _text(value).replace(",", "").replace("$", "")
    if not raw:
        return None
    try:
        amount = Decimal(raw)
    except InvalidOperation:
        return None
    if not amount.is_finite():
        return None
    return int((amount * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _merchant_id(product_url: str, source_file: Path) -> str:
    host = urlsplit(product_url).netloc.lower().split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    raw = host or source_file.stem.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return normalized or "unknown"


def _availability(value: object) -> bool:
    return _text(value).casefold() == "in stock"


def _cursor_encode(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode("ascii")).decode("ascii").rstrip("=")


def _cursor_decode(value: str | None) -> int:
    if value is None:
        return 0
    if not isinstance(value, str) or not value or len(value) > 32:
        raise CatalogDataError("cursor must be a short non-empty string or null")
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.urlsafe_b64decode(value + padding).decode("ascii")
        offset = int(decoded)
    except (ValueError, UnicodeError) as exc:
        raise CatalogDataError("cursor is invalid") from exc
    if offset < 0:
        raise CatalogDataError("cursor offset cannot be negative")
    return offset


def _fts_query(value: str) -> str:
    tokens = [match.group(0).casefold() for match in _TOKEN_RE.finditer(value)]
    if not tokens:
        return ""
    # Quoted tokens avoid exposing FTS operators from model-authored text.
    return " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens[:12])


def _row_to_listing(row: sqlite3.Row) -> CatalogListing:
    return CatalogListing(
        listing_ref=str(row["listing_ref"]),
        merchant_id=str(row["merchant_id"]),
        source_sku=str(row["source_sku"] or ""),
        name=str(row["name"] or ""),
        variant=str(row["variant"] or ""),
        category=str(row["category"] or ""),
        price_minor=int(row["price_minor"]),
        currency=str(row["currency"]),
        in_stock=bool(row["in_stock"]),
        material=str(row["material"] or ""),
        key_features=str(row["key_features"] or ""),
        tags=str(row["tags"] or ""),
        warranty=str(row["warranty"] or ""),
        product_url=str(row["product_url"] or ""),
    )


class CatalogDatabase:
    """Read-only query facade over a prepared catalog database."""

    def __init__(self, path: str | Path, *, read_only: bool = True) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise CatalogDataError(f"catalog database does not exist: {self.path}")
        if read_only:
            uri = f"file:{self.path.resolve()}?mode=ro"
            self.connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "CatalogDatabase":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def listing(self, listing_ref: str) -> CatalogListing | None:
        row = self.connection.execute(
            "SELECT * FROM listings WHERE listing_ref = ?",
            (listing_ref,),
        ).fetchone()
        return None if row is None else _row_to_listing(row)

    def search(self, request: SearchRequest) -> SearchPage:
        if not isinstance(request.query, str):
            raise CatalogDataError("query must be text")
        if not 1 <= request.page_size <= PAGE_SIZE_MAX:
            raise CatalogDataError(f"page_size must be between 1 and {PAGE_SIZE_MAX}")
        offset = _cursor_decode(request.cursor)
        where_sql, parameters, applied = self._where(request.query, request.filters)
        order_sql = self._order(request.sort)
        count_sql = (
            "SELECT COUNT(*) FROM listings l "
            + (
                "JOIN listings_fts ON listings_fts.rowid = l.rowid "
                if _fts_query(request.query)
                else ""
            )
            + where_sql
        )
        total_hits = int(self.connection.execute(count_sql, parameters).fetchone()[0])
        query_sql = (
            "SELECT l.* FROM listings l "
            + (
                "JOIN listings_fts ON listings_fts.rowid = l.rowid "
                if _fts_query(request.query)
                else ""
            )
            + where_sql
            + f" ORDER BY {order_sql} LIMIT ? OFFSET ?"
        )
        rows = self.connection.execute(
            query_sql,
            (*parameters, request.page_size + 1, offset),
        ).fetchall()
        visible = rows[: request.page_size]
        has_more = len(rows) > request.page_size
        next_cursor = _cursor_encode(offset + request.page_size) if has_more else None
        unseen_price: int | None = None
        if has_more and request.sort == "price_asc":
            unseen_price = int(rows[request.page_size]["price_minor"])
        return SearchPage(
            items=tuple(_row_to_listing(row) for row in visible),
            total_hits=total_hits,
            next_cursor=next_cursor,
            applied_filters=applied,
            applied_sort=request.sort,
            unseen_price_lower_bound_minor=unseen_price,
        )

    def full_candidates(
        self,
        *,
        query: str,
        filters: Mapping[str, Any],
        sort: str = "price_asc",
    ) -> tuple[CatalogListing, ...]:
        """Return the complete set for oracle code, never for the model."""

        where_sql, parameters, _applied = self._where(query, filters)
        order_sql = self._order(sort)
        sql = (
            "SELECT l.* FROM listings l "
            + (
                "JOIN listings_fts ON listings_fts.rowid = l.rowid "
                if _fts_query(query)
                else ""
            )
            + where_sql
            + f" ORDER BY {order_sql}"
        )
        return tuple(_row_to_listing(row) for row in self.connection.execute(sql, parameters))

    def not_exists_better(
        self,
        *,
        query: str,
        filters: Mapping[str, Any],
        chosen_price_minor: int,
    ) -> bool:
        where_sql, parameters, _applied = self._where(query, filters)
        join = (
            "JOIN listings_fts ON listings_fts.rowid = l.rowid "
            if _fts_query(query)
            else ""
        )
        better_where = (
            where_sql + " AND l.price_minor < ?"
            if where_sql
            else " WHERE l.price_minor < ?"
        )
        sql = (
            "SELECT NOT EXISTS(SELECT 1 FROM listings l "
            + join
            + better_where
            + ")"
        )
        return bool(
            self.connection.execute(sql, (*parameters, chosen_price_minor)).fetchone()[0]
        )

    def summary(self) -> dict[str, int]:
        row = self.connection.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM source_rows) AS source_rows,
              (SELECT COUNT(*) FROM listings) AS positive_price_rows,
              (SELECT COUNT(*) FROM source_rows
                 WHERE lower(trim(availability)) = 'in stock') AS source_in_stock_rows,
              (SELECT COUNT(*) FROM listings
                 WHERE in_stock = 1) AS transactable_in_stock_rows,
              (SELECT COUNT(DISTINCT product_url) FROM source_rows
                 WHERE product_url <> '') AS unique_product_urls,
              (SELECT COUNT(DISTINCT merchant_id) FROM source_rows) AS merchants,
              (SELECT COUNT(*) FROM source_rows WHERE source_sku = '') AS missing_sku_rows
            """
        ).fetchone()
        return {key: int(row[key]) for key in row.keys()}

    def diverse_listings(
        self,
        *,
        limit: int,
        in_stock: bool = True,
        min_price_minor: int = 100,
        max_price_minor: int = 50_000,
    ) -> tuple[CatalogListing, ...]:
        """Return deterministic anchors spread across distinct Merchants."""

        rows = self.connection.execute(
            """
            WITH ranked AS (
              SELECT l.*,
                     ROW_NUMBER() OVER (
                       PARTITION BY merchant_id
                       ORDER BY lower(category), price_minor, listing_ref
                     ) AS local_rank
              FROM listings l
              WHERE in_stock = ?
                AND price_minor BETWEEN ? AND ?
                AND name <> ''
                AND category <> ''
            )
            SELECT * FROM ranked
            WHERE local_rank = 1
            ORDER BY merchant_id, lower(category), listing_ref
            LIMIT ?
            """,
            (int(in_stock), min_price_minor, max_price_minor, limit),
        ).fetchall()
        return tuple(_row_to_listing(row) for row in rows)

    def _where(
        self,
        query: str,
        filters: Mapping[str, Any],
    ) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
        if not isinstance(filters, Mapping):
            raise CatalogDataError("filters must be an object")
        allowed = {
            "category",
            "merchant",
            "in_stock",
            "min_price_minor",
            "max_price_minor",
            "required_features",
        }
        unknown = sorted(set(filters) - allowed)
        if unknown:
            raise CatalogDataError(f"unsupported filters: {unknown}")
        clauses: list[str] = []
        parameters: list[Any] = []
        normalized_query = _fts_query(query)
        if normalized_query:
            clauses.append("listings_fts MATCH ?")
            parameters.append(normalized_query)
        applied: dict[str, Any] = {}
        category = _text(filters.get("category"))
        if category:
            clauses.append("lower(l.category) = lower(?)")
            parameters.append(category)
            applied["category"] = category
        merchant = _text(filters.get("merchant"))
        if merchant:
            clauses.append("l.merchant_id = ?")
            parameters.append(merchant)
            applied["merchant"] = merchant
        if "in_stock" in filters:
            value = filters["in_stock"]
            if type(value) is not bool:
                raise CatalogDataError("in_stock must be boolean")
            clauses.append("l.in_stock = ?")
            parameters.append(int(value))
            applied["in_stock"] = value
        for key, operator in (
            ("min_price_minor", ">="),
            ("max_price_minor", "<="),
        ):
            if key in filters:
                value = filters[key]
                if type(value) is not int or value < 0:
                    raise CatalogDataError(f"{key} must be a non-negative integer")
                clauses.append(f"l.price_minor {operator} ?")
                parameters.append(value)
                applied[key] = value
        features = filters.get("required_features")
        if features is not None:
            if (
                not isinstance(features, Sequence)
                or isinstance(features, (str, bytes))
                or any(not isinstance(item, str) or not item.strip() for item in features)
            ):
                raise CatalogDataError("required_features must be a list of text")
            normalized_features = tuple(_text(item).casefold() for item in features)
            for feature in normalized_features:
                clauses.append(
                    "lower(l.name || ' ' || l.tags || ' ' || l.key_features || ' ' || "
                    "l.material) LIKE ? ESCAPE '\\'"
                )
                escaped = feature.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                parameters.append(f"%{escaped}%")
            applied["required_features"] = list(normalized_features)
        return (
            (" WHERE " + " AND ".join(clauses)) if clauses else "",
            tuple(parameters),
            applied,
        )

    @staticmethod
    def _order(value: str) -> str:
        orders = {
            "price_asc": "l.price_minor ASC, l.listing_ref ASC",
            "price_desc": "l.price_minor DESC, l.listing_ref ASC",
            "name_asc": "lower(l.name) ASC, l.listing_ref ASC",
            "merchant_asc": "l.merchant_id ASC, l.price_minor ASC, l.listing_ref ASC",
        }
        if value not in orders:
            raise CatalogDataError(f"unsupported sort: {value!r}")
        return orders[value]


def prepare_catalog(
    data_root: str | Path,
    database_path: str | Path,
    *,
    batch_size: int = INGEST_BATCH_SIZE,
) -> CatalogIngestReport:
    """Stream all CSV rows into a fresh SQLite database."""

    root = Path(data_root)
    if not root.is_dir():
        raise CatalogDataError(f"raw data directory does not exist: {root}")
    files = tuple(sorted(root.glob("*.csv")))
    if not files:
        raise CatalogDataError(f"raw data directory contains no CSV files: {root}")
    destination = Path(database_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    started = time.monotonic()
    connection = sqlite3.connect(destination)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=FILE;
            CREATE TABLE source_rows (
              source_id INTEGER PRIMARY KEY,
              source_file TEXT NOT NULL,
              source_row INTEGER NOT NULL,
              merchant_id TEXT NOT NULL,
              price_minor INTEGER,
              source_sku TEXT NOT NULL,
              availability TEXT NOT NULL,
              product_url TEXT NOT NULL,
              quality_flags TEXT NOT NULL,
              raw_row_zlib BLOB NOT NULL
            );
            CREATE TABLE listings (
              listing_ref TEXT PRIMARY KEY,
              source_id INTEGER NOT NULL REFERENCES source_rows(source_id),
              merchant_id TEXT NOT NULL,
              source_sku TEXT NOT NULL,
              name TEXT NOT NULL,
              variant TEXT NOT NULL,
              category TEXT NOT NULL,
              price_minor INTEGER NOT NULL CHECK(price_minor > 0),
              currency TEXT NOT NULL,
              in_stock INTEGER NOT NULL CHECK(in_stock IN (0,1)),
              material TEXT NOT NULL,
              key_features TEXT NOT NULL,
              tags TEXT NOT NULL,
              warranty TEXT NOT NULL,
              product_url TEXT NOT NULL,
              search_text TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE listings_fts USING fts5(
              search_text,
              content='listings',
              content_rowid='rowid',
              tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        source_rows: list[tuple[Any, ...]] = []
        listings: list[tuple[Any, ...]] = []
        source_id = 0
        positive = 0
        source_stocked = 0
        transactable_stocked = 0
        missing_sku = 0

        def flush() -> None:
            nonlocal source_rows, listings
            if source_rows:
                connection.executemany(
                    """
                    INSERT INTO source_rows VALUES
                    (?,?,?,?,?,?,?,?,?,?)
                    """,
                    source_rows,
                )
            if listings:
                connection.executemany(
                    """
                    INSERT INTO listings VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    listings,
                )
            source_rows = []
            listings = []

        for source_file in files:
            with source_file.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = frozenset(reader.fieldnames or ())
                if not EXPECTED_COLUMNS.issubset(fields) or len(fields) not in {25, 27}:
                    raise CatalogDataError(
                        f"{source_file.name} has unsupported {len(fields)}-column schema"
                    )
                for logical_row, raw in enumerate(reader, start=2):
                    source_id += 1
                    name = _text(raw.get("Product Name"))
                    category = _text(raw.get("Product Type"))
                    vendor = _text(raw.get("Vendor"))
                    price = _money_minor(raw.get("Sale Price (USD)"))
                    sku = _text(raw.get("SKU"))
                    availability = _text(raw.get("Availability"))
                    source_stocked += int(_availability(availability))
                    variant = _text(raw.get("Variant"))
                    tags = _text(raw.get("Tags"))
                    material = _text(raw.get("Material"))
                    features = _text(raw.get("Key Features"))
                    warranty = _text(raw.get("Warranty"))
                    product_url = _text(raw.get("Product URL"))
                    merchant = _merchant_id(product_url, source_file)
                    flags: list[str] = []
                    if not sku:
                        flags.append("missing_sku")
                        missing_sku += 1
                    if price is None:
                        flags.append("missing_or_invalid_price")
                    elif price <= 0:
                        flags.append("nonpositive_price")
                    if availability.casefold() not in {"in stock", "out of stock"}:
                        flags.append("unrecognized_availability")
                    compressed = zlib.compress(
                        json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8"),
                        level=3,
                    )
                    source_rows.append(
                        (
                            source_id,
                            source_file.name,
                            logical_row,
                            merchant,
                            price,
                            sku,
                            availability,
                            product_url,
                            json.dumps(flags, separators=(",", ":")),
                            compressed,
                        )
                    )
                    if price is not None and price > 0:
                        listing_ref = f"merchant:{merchant}:row:{logical_row}:{source_file.stem}"
                        stock = _availability(availability)
                        search_text = " ".join(
                            value
                            for value in (name, category, vendor, variant, material)
                            if value
                        )
                        listings.append(
                            (
                                listing_ref,
                                source_id,
                                merchant,
                                sku,
                                name,
                                variant,
                                category,
                                price,
                                "USD",
                                int(stock),
                                material,
                                features,
                                tags,
                                warranty,
                                product_url,
                                search_text,
                            )
                        )
                        positive += 1
                        transactable_stocked += int(stock)
                    if len(source_rows) >= batch_size:
                        flush()
                        connection.commit()
        flush()
        connection.executescript(
            """
            INSERT INTO listings_fts(rowid, search_text)
              SELECT rowid, search_text FROM listings;
            CREATE INDEX listings_merchant_idx ON listings(merchant_id);
            CREATE INDEX listings_sku_idx ON listings(source_sku);
            CREATE INDEX listings_url_idx ON listings(product_url);
            CREATE INDEX listings_stock_price_idx
              ON listings(in_stock, price_minor, listing_ref);
            CREATE INDEX listings_category_price_idx
              ON listings(category COLLATE NOCASE, price_minor, listing_ref);
            CREATE INDEX source_rows_url_idx ON source_rows(product_url);
            CREATE INDEX source_rows_merchant_idx ON source_rows(merchant_id);
            ANALYZE;
            PRAGMA wal_checkpoint(TRUNCATE);
            """
        )
        connection.commit()
        unique_urls = int(
            connection.execute(
                "SELECT COUNT(DISTINCT product_url) FROM source_rows WHERE product_url <> ''"
            ).fetchone()[0]
        )
        merchants = int(
            connection.execute("SELECT COUNT(DISTINCT merchant_id) FROM source_rows").fetchone()[0]
        )
    finally:
        connection.close()
    return CatalogIngestReport(
        files=len(files),
        source_rows=source_id,
        positive_price_rows=positive,
        source_in_stock_rows=source_stocked,
        transactable_in_stock_rows=transactable_stocked,
        missing_sku_rows=missing_sku,
        unique_product_urls=unique_urls,
        merchants=merchants,
        database_bytes=destination.stat().st_size,
        elapsed_seconds=time.monotonic() - started,
    )


def iter_raw_rows(data_root: str | Path) -> Iterator[tuple[Path, int, Mapping[str, str]]]:
    """Small public iterator used by data validation tests."""

    for source_file in sorted(Path(data_root).glob("*.csv")):
        with source_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for logical_row, row in enumerate(reader, start=2):
                yield source_file, logical_row, row
