"""Fetch ESCO occupation->skill relations for the occupations our corpus actually uses.

Gotchas encoded here (both cost real debugging time):
  * Norwegian is language code 'no'. 'nb' is accepted but SILENTLY returns English.
  * 104/287 of our ISCO URIs arrive lowercase ('.../isco/c1219') and 404 until the
    leading 'C' is uppercased.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://ec.europa.eu/esco/api/resource/"
UA = {"User-Agent": "finnno-smart-search/0.1 (portfolio PoC)", "Accept": "application/json"}
LANGS = ("en", "no")

DDL = """
CREATE TABLE IF NOT EXISTS esco_occupation (
    uri VARCHAR, lang VARCHAR, title VARCHAR, PRIMARY KEY (uri, lang));
CREATE TABLE IF NOT EXISTS esco_skill (
    uri VARCHAR, lang VARCHAR, title VARCHAR, PRIMARY KEY (uri, lang));
CREATE TABLE IF NOT EXISTS esco_occ_skill (
    occupation_uri VARCHAR, skill_uri VARCHAR, relation VARCHAR,
    PRIMARY KEY (occupation_uri, skill_uri, relation));
CREATE TABLE IF NOT EXISTS esco_isco_narrower (
    isco_uri VARCHAR, occupation_uri VARCHAR, PRIMARY KEY (isco_uri, occupation_uri));
CREATE TABLE IF NOT EXISTS esco_fetch_log (uri VARCHAR, lang VARCHAR, outcome VARCHAR);
"""


def normalise(uri: str) -> str:
    """Uppercase the ISCO code letter; leave occupation UUIDs alone."""
    return re.sub(r"(/esco/isco/)c(\d)", r"\1C\2", uri)


def _get(endpoint: str, uri: str, lang: str, tries: int = 4):
    url = f"{API}{endpoint}?uri={urllib.parse.quote(uri, safe='')}&language={lang}"
    for a in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.load(r)
        except Exception as e:
            code = getattr(e, "code", None)
            if code in (400, 404):
                return None
            if a == tries - 1:
                return None
            time.sleep(2 ** a)
    return None


def fetch_occupation(uri: str) -> dict | None:
    out = {"uri": uri, "titles": {}, "skills": {}}
    for lang in LANGS:
        d = _get("occupation", uri, lang)
        if not d:
            return None
        out["titles"][lang] = d.get("title")
        for rel, key in (("essential", "hasEssentialSkill"), ("optional", "hasOptionalSkill")):
            for s in d.get("_links", {}).get(key, []):
                su = s.get("uri")
                if not su:
                    continue
                out["skills"].setdefault(su, {"relation": rel, "titles": {}})
                out["skills"][su]["titles"][lang] = s.get("title")
    return out


def fetch_isco(uri: str) -> dict | None:
    d = _get("concept", uri, "en")
    if not d:
        return None
    return {"uri": uri, "title": d.get("title"),
            "narrower": [n["uri"] for n in d.get("_links", {}).get("narrowerOccupation", [])
                         if n.get("uri")]}


def run(con, uris: list[str], workers: int = 4, log=print) -> dict:
    con.execute(DDL)
    uris = [normalise(u) for u in uris]
    occ = sorted({u for u in uris if "/occupation/" in u})
    isco = sorted({u for u in uris if "/isco/" in u})
    log(f"ESCO: {len(occ)} occupations, {len(isco)} ISCO groups")

    # 1. ISCO groups -> narrower occupations (adds occupations worth fetching)
    narrow_rows, extra = [], set()
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(fetch_isco, isco):
            if not r:
                continue
            for n in r["narrower"]:
                narrow_rows.append((r["uri"], n))
                extra.add(n)
    con.executemany("INSERT OR IGNORE INTO esco_isco_narrower VALUES (?,?)", narrow_rows)
    log(f"ESCO: {len(narrow_rows)} isco->occupation edges, {len(extra - set(occ))} new occupations")

    targets = sorted(set(occ) | extra)
    log(f"ESCO: fetching skills for {len(targets)} occupations x {len(LANGS)} languages")

    o_rows, s_rows, os_rows, done = [], [], [], 0
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(fetch_occupation, targets):
            done += 1
            if done % 200 == 0:
                log(f"  {done}/{len(targets)}")
            if not r:
                continue
            for lang, t in r["titles"].items():
                o_rows.append((r["uri"], lang, t))
            for su, s in r["skills"].items():
                os_rows.append((r["uri"], su, s["relation"]))
                for lang, t in s["titles"].items():
                    s_rows.append((su, lang, t))
    con.executemany("INSERT OR IGNORE INTO esco_occupation VALUES (?,?,?)", o_rows)
    con.executemany("INSERT OR IGNORE INTO esco_skill VALUES (?,?,?)", s_rows)
    con.executemany("INSERT OR IGNORE INTO esco_occ_skill VALUES (?,?,?)", os_rows)
    return {"occupations": len(set(r[0] for r in o_rows)),
            "skills": len(set(r[0] for r in s_rows)),
            "occ_skill_edges": len(set(os_rows)),
            "isco_edges": len(narrow_rows)}
