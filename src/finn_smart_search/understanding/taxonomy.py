"""Derived taxonomy facets. No LLM call, no new data source, no network.

Everything here comes from data already on disk under the NAV licence. A review
of a proposed 20-field taxonomy found that most of what it wanted was already
present and merely unsurfaced — so these are derivations, not extractions.

  job_title_standardised  The feed's `jobtitle` is free text: 4,878 distinct
      values over 8,351 ads, 87.1% of them occurring exactly once, 700 containing
      digits. Not an identity. ESCO's preferred Norwegian label joins on 8,339
      ads (82.0%) with a controlled vocabulary AND a free English counterpart;
      STYRK-08's name covers 10,161 (100.0%). Coalescing reaches 99.95%.

  role_family  ISCO sub-major = the first TWO digits of the STYRK code. A flat
      12-value `job_family` enum was measured against all 323 STYRK codes and
      fitted 66.4% cleanly, 19.6% ambiguously, 13.9% not at all — because STYRK's
      FIRST digit is ISCO's skill level while `job_family` is a domain axis, and
      the two are orthogonal. ARCHITECTURE.md specifies this instead.

  nav_category  `occupationCategories.level1`, already in ads_raw.ad_content at
      ~100% as a closed 13-value vocabulary, never loaded. This is the
      `industry_sector` a proposal wanted as a free string — controlled and
      joinable rather than unjoinable text. Distribution is Norwegian, not
      Anglo-tech: Helse og sosial 39%, Salg og service 17%, Utdanning 14%, IT ~1%.

NOT derived here, deliberately: `categoryList.score` is 1.0 on 99.5-99.6% of all
rows, so weighting anything by it is decorative. Recorded rather than used.
"""
from __future__ import annotations

import json
import re

_STYRK4 = re.compile(r"^\d{4}$")

DDL = """
CREATE TABLE IF NOT EXISTS ad_taxonomy (
    uuid VARCHAR PRIMARY KEY,
    job_title_standardised VARCHAR,
    job_title_en VARCHAR,
    title_source VARCHAR,
    styrk_code VARCHAR,
    role_family VARCHAR,
    skill_level VARCHAR,
    nav_category VARCHAR,
    nav_subcategory VARCHAR,
    built_at TIMESTAMPTZ);
"""


def role_family(styrk_code: str | None) -> str | None:
    """ISCO sub-major: the first two digits. None if the code is unreadable —
    loud beats a plausible-looking two characters nobody can trace."""
    if not styrk_code or not _STYRK4.match(styrk_code.strip()):
        return None
    return styrk_code.strip()[:2]


def skill_level(styrk_code: str | None) -> str | None:
    """ISCO major group: the first digit. This is a SKILL LEVEL, not a domain —
    2223 Sykepleiere and 2521 Databasedesignere share it and are unrelated."""
    if not styrk_code or not _STYRK4.match(styrk_code.strip()):
        return None
    return styrk_code.strip()[0]


def _nav_categories(con) -> dict[str, tuple[str | None, str | None]]:
    """(level1, level2) per ad from the raw feed payload, first entry wins."""
    out: dict[str, tuple[str | None, str | None]] = {}
    for uuid, raw in con.execute("SELECT uuid, ad_content FROM ads_raw").fetchall():
        try:
            d = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(d, dict):
            continue
        content = d.get("ad_content") if isinstance(d.get("ad_content"), dict) else d
        cats = content.get("occupationCategories") or []
        if cats and isinstance(cats[0], dict):
            out[uuid] = (cats[0].get("level1"), cats[0].get("level2"))
    return out


def build(con, log=print) -> dict:
    """(Re)build `ad_taxonomy`. Idempotent; derived purely from bronze + ESCO."""
    con.execute(DDL)
    con.execute("DELETE FROM ad_taxonomy")

    nav = _nav_categories(con)
    con.execute("CREATE OR REPLACE TEMP TABLE _nav(uuid VARCHAR, l1 VARCHAR, l2 VARCHAR)")
    if nav:
        con.executemany("INSERT INTO _nav VALUES (?,?,?)",
                        [[u, a, b] for u, (a, b) in nav.items()])

    # One STYRK code per ad, chosen deterministically. `score` cannot break ties:
    # it is 1.0 on 99.6% of rows, so min(code) is the honest tie-break.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _styrk AS
        SELECT uuid, min(code) AS code, min(name) AS name
        FROM ad_categories WHERE category_type = 'STYRK08' GROUP BY uuid
    """)
    # ESCO occupation URIs only — 15.8% of ads carry just an /isco/ group URI,
    # which has no preferred occupation label to offer.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _esco AS
        SELECT c.uuid,
               min(CASE WHEN e.lang = 'no' THEN e.title END) AS title_no,
               min(CASE WHEN e.lang = 'en' THEN e.title END) AS title_en
        FROM ad_categories c
        JOIN esco_occupation e ON e.uri = c.code
        WHERE c.category_type = 'ESCO' AND c.code LIKE '%/occupation/%'
        GROUP BY c.uuid
    """)
    con.execute("""
        INSERT INTO ad_taxonomy
        SELECT a.uuid,
               coalesce(e.title_no, s.name)                        AS job_title_standardised,
               e.title_en                                          AS job_title_en,
               CASE WHEN e.title_no IS NOT NULL THEN 'esco'
                    WHEN s.name    IS NOT NULL THEN 'styrk' END    AS title_source,
               s.code, NULL, NULL, n.l1, n.l2, now()
        FROM ads a
        LEFT JOIN _styrk s USING (uuid)
        LEFT JOIN _esco  e USING (uuid)
        LEFT JOIN _nav   n USING (uuid)
    """)
    # role_family / skill_level in Python so the rule lives in ONE place and is
    # unit-testable without the corpus.
    rows = con.execute("SELECT uuid, styrk_code FROM ad_taxonomy").fetchall()
    con.executemany("UPDATE ad_taxonomy SET role_family=?, skill_level=? WHERE uuid=?",
                    [[role_family(c), skill_level(c), u] for u, c in rows])

    stats = con.execute("""SELECT count(*),
        sum(CASE WHEN job_title_standardised IS NOT NULL THEN 1 ELSE 0 END),
        sum(CASE WHEN title_source='esco' THEN 1 ELSE 0 END),
        sum(CASE WHEN nav_category IS NOT NULL THEN 1 ELSE 0 END),
        sum(CASE WHEN role_family IS NOT NULL THEN 1 ELSE 0 END)
        FROM ad_taxonomy""").fetchone()
    out = {"ads": stats[0], "titled": stats[1], "esco_tier": stats[2],
           "nav_category": stats[3], "role_family": stats[4]}
    log(f"taxonomy: {out}")
    return out
