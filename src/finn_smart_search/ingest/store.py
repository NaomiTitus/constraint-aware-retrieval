"""DuckDB store: bronze (immutable raw) -> silver (normalised) -> gold (facets).

Bronze is append-only and never deleted: every downstream table is derivable from
it, so the corpus can be fully reprocessed without re-crawling NAV.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

DEFAULT_DB = Path("data/ads.duckdb")

BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS feed_entries (      -- append-only listing rows
    uuid           VARCHAR,
    feed_id        VARCHAR,
    status         VARCHAR,
    title          VARCHAR,
    business_name  VARCHAR,
    municipal      VARCHAR,
    sist_endret    TIMESTAMPTZ,
    url            VARCHAR,
    seen_at        TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS ads_raw (           -- full ad payload, ACTIVE ads only
    uuid           VARCHAR PRIMARY KEY,
    status         VARCHAR,
    sist_endret    TIMESTAMPTZ,
    fetched_at     TIMESTAMPTZ,
    content_sha256 VARCHAR,
    ad_content     JSON
);
CREATE TABLE IF NOT EXISTS fetch_log (         -- every non-200 / stub outcome
    uuid           VARCHAR,
    http_status    INTEGER,
    outcome        VARCHAR,
    fetched_at     TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS crawl_state (       -- resume cursor
    key            VARCHAR PRIMARY KEY,
    value          JSON,
    updated_at     TIMESTAMPTZ
);
"""


def connect(db_path: Path | str = DEFAULT_DB) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(BRONZE_DDL)
    return con


def save_state(con, key: str, value: dict) -> None:
    con.execute(
        "INSERT OR REPLACE INTO crawl_state VALUES (?, ?, ?)",
        [key, json.dumps(value), datetime.now(timezone.utc)],
    )


def load_state(con, key: str) -> dict | None:
    row = con.execute("SELECT value FROM crawl_state WHERE key = ?", [key]).fetchone()
    return json.loads(row[0]) if row else None
