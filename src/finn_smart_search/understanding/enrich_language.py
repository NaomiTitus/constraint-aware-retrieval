"""Language enrichment: populate `ad_language` from cleaned ad HTML.

GOLD, not silver. `doc_lang` is deliberately NOT a column on `ads`, because
`ads` is a pure function of bronze — rebuildable at any time from `ads_raw`
with no external dependency. langid output depends on a lingua version, so a
column on `ads` would mean a library bump silently invalidates the silver
layer. Rows carry `langid_version` so a bump is detectable after the fact and
triggers reprocessing.

Resumable: the full corpus is ~12 minutes of detection, long enough that an
interruption should not cost the whole run.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from . import langid
from .html_clean import to_blocks

LANGID_VERSION = "langid-1"

DDL = """
CREATE TABLE IF NOT EXISTS ad_language (
    uuid VARCHAR PRIMARY KEY, doc_lang VARCHAR, lang_mix JSON,
    detected_other VARCHAR, is_bilingual BOOLEAN, n_scored INTEGER,
    confidence VARCHAR, langid_version VARCHAR, detected_at TIMESTAMPTZ);
"""


def _todo(con, force: bool) -> list[tuple[str, str]]:
    """Ads still needing a verdict. A row from a different langid_version counts
    as stale, so a library bump does not leave old verdicts behind."""
    if force:
        return con.execute("SELECT uuid, description_html FROM ads").fetchall()
    return con.execute("""
        SELECT a.uuid, a.description_html
        FROM ads a LEFT JOIN ad_language l USING (uuid)
        WHERE l.uuid IS NULL OR l.langid_version IS DISTINCT FROM ?
    """, [LANGID_VERSION]).fetchall()


def _flush(con, buf: list) -> int:
    """Single write path. Duplicating the INSERT let a mutation land on dead
    code: with batch=500 and small tests the in-loop copy never ran, so an
    `INSERT OR IGNORE` mutation on it survived while the real bug — stale rows
    never replaced on a version bump — went undetected."""
    if not buf:
        return 0
    con.executemany("INSERT OR REPLACE INTO ad_language VALUES (?,?,?,?,?,?,?,?,?)", buf)
    n = len(buf)
    buf.clear()
    return n


def run(con, *, force: bool = False, log=print, batch: int = 500) -> dict:
    con.execute(DDL)
    todo = _todo(con, force)
    log(f"language: {len(todo)} ads to process")

    dist, buf, done = Counter(), [], 0
    for uuid, html in todo:
        r = langid.detect(to_blocks(html))
        dist[r["doc_lang"]] += 1
        buf.append([uuid, r["doc_lang"], json.dumps(r["lang_mix"]),
                    r["detected_other"], r["is_bilingual"], r["n_scored"],
                    r["confidence"], LANGID_VERSION, datetime.now(timezone.utc)])
        if len(buf) >= batch:
            done += _flush(con, buf)
            log(f"  {done}/{len(todo)}")
    done += _flush(con, buf)
    return {"processed": done, "distribution": dict(dist)}
