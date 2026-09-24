"""Bronze -> silver: normalised, typed, deterministic. Pure function of ads_raw."""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")
from finn_smart_search.understanding.html_clean import clean

SILVER_DDL = """
DROP TABLE IF EXISTS ads;
DROP TABLE IF EXISTS ad_locations;
DROP TABLE IF EXISTS ad_categories;
CREATE TABLE ads (
    uuid VARCHAR PRIMARY KEY, title VARCHAR, jobtitle VARCHAR,
    description_html VARCHAR, description_text VARCHAR, n_blocks INTEGER, n_chars INTEGER,
    published TIMESTAMPTZ, expires TIMESTAMPTZ, updated TIMESTAMPTZ, application_due VARCHAR,
    application_url VARCHAR, link VARCHAR,
    employer_name VARCHAR, employer_orgnr VARCHAR, employer_homepage VARCHAR,
    engagementtype VARCHAR, extent VARCHAR, sector VARCHAR, positioncount VARCHAR,
    starttime VARCHAR, source VARCHAR, sourceurl VARCHAR, content_sha256 VARCHAR
);
CREATE TABLE ad_locations (
    uuid VARCHAR, country VARCHAR, address VARCHAR, city VARCHAR,
    postal_code VARCHAR, county VARCHAR, municipal VARCHAR
);
CREATE TABLE ad_categories (
    uuid VARCHAR, category_type VARCHAR, code VARCHAR, name VARCHAR, score DOUBLE
);
"""


def _ts(v):
    return v or None


def build(con, log=print) -> dict:
    con.execute(SILVER_DDL)
    rows = con.execute("SELECT uuid, ad_content, content_sha256 FROM ads_raw").fetchall()
    log(f"silver: transforming {len(rows)} ads")

    ads, locs, cats = [], [], []
    for uu, raw, sha in rows:
        a = json.loads(raw) if isinstance(raw, str) else raw
        blocks, text = clean(a.get("description"))
        emp = a.get("employer") or {}
        ads.append((
            uu, a.get("title"), a.get("jobtitle") or None,
            a.get("description"), text, len(blocks), len(text),
            _ts(a.get("published")), _ts(a.get("expires")), _ts(a.get("updated")),
            a.get("applicationDue") or None, a.get("applicationUrl") or None, a.get("link"),
            emp.get("name"), emp.get("orgnr") or None, emp.get("homepage") or None,
            a.get("engagementtype") or None, a.get("extent") or None, a.get("sector") or None,
            a.get("positioncount") or None, a.get("starttime") or None,
            a.get("source"), a.get("sourceurl") or None, sha,
        ))
        for L in (a.get("workLocations") or []):
            locs.append((uu, L.get("country"), L.get("address"), L.get("city"),
                         L.get("postalCode"), L.get("county"), L.get("municipal")))
        for c in (a.get("categoryList") or []):
            try:
                sc = float(c.get("score")) if c.get("score") not in (None, "") else None
            except (TypeError, ValueError):
                sc = None
            cats.append((uu, c.get("categoryType"), c.get("code"), c.get("name"), sc))

    con.executemany(f"INSERT INTO ads VALUES ({','.join('?'*24)})", ads)
    con.executemany("INSERT INTO ad_locations VALUES (?,?,?,?,?,?,?)", locs)
    con.executemany("INSERT INTO ad_categories VALUES (?,?,?,?,?)", cats)
    return {"ads": len(ads), "locations": len(locs), "categories": len(cats)}
