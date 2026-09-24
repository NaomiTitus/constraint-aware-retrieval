"""Census runner: one LLM call per near-duplicate cluster, fanned out to members.

Design notes, each earned the hard way:

  NO TRUNCATION. Truncating at 6,000 chars saves $0.15 on a ~$10 run and costs
  the head+tail logic, a `truncated` flag, span validation against truncated
  rather than original text, and a "don't say absence_of_requirement when
  truncated" rule. The longest corpus ad is 18,081 chars (~5,166 tokens).

  PREFIX CACHING IS A COST CONTROL. The 3,030-token system+few-shot prefix
  billed at 0.1x is the difference between $10.45 and $22.61 for the same run.

  MAP BY custom_id, NEVER BY POSITION. Batch results are unordered; position
  mapping would assign one ad's language verdict to another, silently.

  DEMOTION IS COUNTED, NEVER SILENT. Demoting a record whose evidence failed
  validation converts a precision error into a recall error invisibly. Above
  15% the run aborts: if one record in seven cannot support its own verdict,
  the prompt is broken and spending the rest is throwing money at it.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from . import census_validate as validate_mod
from . import dedup
from .census_prompt import (PROMPT_VERSION, TOOL, USER_TEMPLATE,
                            build_request as _build_params,
                            derive_english_accessible)
from .text_norm import normalise

EXTRACTOR_VERSION = "census-runner-1"
MODEL = "claude-haiku-4-5-20251001"

# Haiku 4.5 Batch API, USD per token.
PRICE_IN = 0.50 / 1_000_000
PRICE_OUT = 2.50 / 1_000_000

DEFAULT_MAX_DEMOTION_RATE = 0.15

DDL = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key VARCHAR PRIMARY KEY, model VARCHAR, prompt_version VARCHAR,
    response VARCHAR, input_tokens INTEGER, output_tokens INTEGER, created_at TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS ad_facets (
    uuid VARCHAR PRIMARY KEY, facets JSON, english_accessible BOOLEAN,
    demoted BOOLEAN, reasons JSON, prompt_version VARCHAR,
    extractor_version VARCHAR, extracted_at TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS census_failures (
    uuid VARCHAR, reason VARCHAR, detail VARCHAR, failed_at TIMESTAMPTZ);
"""


class MissingResultError(RuntimeError):
    """A submitted custom_id never came back. Never drop it silently."""


class DemotionRateError(RuntimeError):
    """Too many records could not support their own verdict."""


class SpendLimitError(RuntimeError):
    """Running spend exceeded the cap."""


# ── requests ─────────────────────────────────────────────────────────────────

def build_request(ad: Mapping[str, Any], **_) -> dict:
    """One batch request. custom_id is the uuid, so results map back by identity."""
    params = _build_params(
        title=ad.get("title") or "",
        body=ad.get("description_text") or "",
        doc_lang=ad.get("doc_lang") or "no",
    )
    params.pop("truncated", None)
    return {"custom_id": ad["uuid"], "params": params}


# ── cache ────────────────────────────────────────────────────────────────────

def cache_key(model: str, prompt_version: str, text: str | None) -> str:
    """Keyed on the NORMALISED text, so an HTML-cleaning tweak that only moves
    whitespace does not re-bill the entire census."""
    payload = f"{model}\x00{prompt_version}\x00{normalise(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_get(con, model: str, prompt_version: str, text: str | None):
    row = con.execute("SELECT response FROM llm_cache WHERE key = ?",
                      [cache_key(model, prompt_version, text)]).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0]) if isinstance(row[0], str) else row[0]
    except (json.JSONDecodeError, TypeError):
        return None            # corrupt row is a miss, not a crash


def cache_put(con, model: str, prompt_version: str, text: str | None,
              facets: Mapping[str, Any], usage: Mapping[str, int]) -> None:
    con.execute(
        "INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?,?,?,?)",
        [cache_key(model, prompt_version, text), model, prompt_version,
         json.dumps(facets, ensure_ascii=False),
         usage.get("input_tokens", 0), usage.get("output_tokens", 0),
         datetime.now(timezone.utc)])


# ── run ──────────────────────────────────────────────────────────────────────

def _collect(client, batch_id, expected: set[str], poll_seconds: float) -> dict:
    while client.poll(batch_id) not in ("ended", "completed"):
        time.sleep(poll_seconds)
    got = {}
    for res in client.results(batch_id):
        got[res["custom_id"]] = res
    missing = expected - set(got)
    if missing:
        raise MissingResultError(f"{len(missing)} custom_id(s) never returned: {sorted(missing)[:5]}")
    return got


def run(ads: Iterable[Mapping[str, Any]], client, con, *,
        limit: int | None = None,
        max_spend_usd: float | None = None,
        max_demotion_rate: float = DEFAULT_MAX_DEMOTION_RATE,
        model: str = MODEL,
        poll_seconds: float = 0.0) -> dict:
    con.execute(DDL)
    ads = list(ads)
    by_uuid = {a["uuid"]: a for a in ads}

    groups = dedup.cluster(ads)
    reps = dedup.representatives(groups)
    if limit is not None:
        reps = reps[:limit]
        keep = set(reps)
        groups = {s: u for s, u in groups.items() if min(u) in keep}

    # 1. cache first
    facets_by_rep, cache_hits = {}, 0
    todo = []
    for uuid in reps:
        hit = cache_get(con, model, PROMPT_VERSION, by_uuid[uuid].get("description_text"))
        if hit is not None:
            facets_by_rep[uuid] = hit
            cache_hits += 1
        else:
            todo.append(uuid)

    # 2. submit, retrying expired ONCE (expired requests are not billed)
    spend, failed, raw = 0.0, {}, {}
    pending = list(todo)
    for attempt in range(2):
        if not pending:
            break
        requests = [build_request(by_uuid[u]) for u in pending]
        got = _collect(client, client.submit_batch(requests), set(pending), poll_seconds)
        retry = []
        for uuid, res in got.items():
            kind = res.get("type")
            if kind == "succeeded":
                usage = res.get("usage") or {}
                spend += usage.get("input_tokens", 0) * PRICE_IN
                spend += usage.get("output_tokens", 0) * PRICE_OUT
                if max_spend_usd is not None and spend > max_spend_usd:
                    raise SpendLimitError(f"spend ${spend:.2f} exceeded cap ${max_spend_usd:.2f}")
                raw[uuid] = (res["facets"], usage)
            elif kind == "expired" and attempt == 0:
                retry.append(uuid)
            else:
                failed[uuid] = res.get("error") or kind
        pending = retry

    for uuid in pending:                       # still expired after the retry
        failed[uuid] = "expired"

    # 3. validate, then cache only what survived
    demoted = 0
    for uuid, (facets, usage) in raw.items():
        text = by_uuid[uuid].get("description_text") or ""
        checked = validate_mod.validate(dict(facets), text)
        if checked["demoted"]:
            demoted += 1
        facets_by_rep[uuid] = checked["facets"]
        facets_by_rep[uuid]["_reasons"] = checked["reasons"]
        cache_put(con, model, PROMPT_VERSION, text, checked["facets"], usage)

    n_new = len(raw)
    rate = demoted / n_new if n_new else 0.0
    if n_new and rate > max_demotion_rate:
        raise DemotionRateError(
            f"demotion rate {rate:.0%} exceeds {max_demotion_rate:.0%} over {n_new} records — "
            "the prompt is not producing defensible evidence")

    # 4. fan out to every cluster member, derive, persist
    live = {s: u for s, u in groups.items() if min(u) in facets_by_rep}
    spread = dedup.fan_out(live, facets_by_rep)
    now = datetime.now(timezone.utc)
    out = {}
    for uuid, facets in spread.items():
        facets = dict(facets)
        reasons = facets.pop("_reasons", [])
        facets["english_accessible"] = derive_english_accessible(
            facets, by_uuid[uuid].get("doc_lang") or "no")
        out[uuid] = facets
        con.execute("INSERT OR REPLACE INTO ad_facets VALUES (?,?,?,?,?,?,?,?)",
                    [uuid, json.dumps(facets, ensure_ascii=False),
                     facets["english_accessible"], bool(reasons),
                     json.dumps(reasons, ensure_ascii=False),
                     PROMPT_VERSION, EXTRACTOR_VERSION, now])

    for uuid, reason in failed.items():
        con.execute("INSERT INTO census_failures VALUES (?,?,?,?)",
                    [uuid, "batch", str(reason), now])

    return {"facets": out, "calls": n_new, "cache_hits": cache_hits,
            "demoted": demoted, "demotion_rate": rate, "spend_usd": spend,
            "failed": failed,
            "levels": Counter(f["norwegian_requirement_level"] for f in out.values())}
