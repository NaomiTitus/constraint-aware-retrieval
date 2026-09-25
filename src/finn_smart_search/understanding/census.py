"""Census runner: one LLM call per near-duplicate cluster, fanned out to members.

Design notes, each earned the hard way:

  NO TRUNCATION. Truncating at 6,000 chars saves $0.15 on a ~$10 run and costs
  the head+tail logic, a `truncated` flag, span validation against truncated
  rather than original text, and a "don't say absence_of_requirement when
  truncated" rule. The longest corpus ad is 18,081 chars (~5,166 tokens).

  PREFIX CACHING IS A COST CONTROL. MEASURED against the live API: 9,217
  cached prefix tokens per request, billed at 0.1x, with only 456-1,092
  uncached. Without caching the same run roughly triples.

  COST, MEASURED not estimated. My estimate was 3,413 prefix tokens and 200
  output; the real figures are 9,217 and ~350. The estimate counted
  characters/4 over the prompt text and ignored the tool schema, the twelve
  tool_use/tool_result envelopes and JSON structural overhead. Full census is
  ~$14.50, not the ~$10.45 first quoted.

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
# Cache READS bill at 0.1x input; cache WRITES at 1.25x. Both were counted as
# zero, which under-reported by 27.1% because the prompt's cached prefix is
# 9,217 tokens against 456-1,092 uncached input per call (measured over the
# recorded responses in tests/fixtures/batch_results_real.json). Over 9,823
# clusters that is $12.18 reported against $16.71 billed -- and max_spend_usd
# was computed the same way, so a $20 cap actually permitted ~$27.43.
PRICE_CACHE_READ = PRICE_IN * 0.1
PRICE_CACHE_WRITE = PRICE_IN * 1.25

DEFAULT_MAX_DEMOTION_RATE = 0.15

DDL = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key VARCHAR PRIMARY KEY, model VARCHAR, prompt_version VARCHAR,
    response VARCHAR, input_tokens INTEGER, output_tokens INTEGER, created_at TIMESTAMPTZ,
    response_raw VARCHAR);
CREATE TABLE IF NOT EXISTS ad_facets (
    uuid VARCHAR PRIMARY KEY, facets JSON, english_accessible BOOLEAN,
    demoted BOOLEAN, reasons JSON, prompt_version VARCHAR,
    extractor_version VARCHAR, extracted_at TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS census_failures (
    uuid VARCHAR, reason VARCHAR, detail VARCHAR, failed_at TIMESTAMPTZ);
"""


def ensure_schema(con) -> None:
    """Create the tables, and MIGRATE an existing llm_cache.

    `CREATE TABLE IF NOT EXISTS` does not add a column to a table that already
    exists, so the corpus database -- which holds a 7-column llm_cache written
    before response_raw existed -- would fail every INSERT without this. The
    migration is additive: rows already paid for keep their response and their
    token counts, and simply carry NULL raw.
    """
    con.execute(DDL)
    cols = {r[0] for r in con.execute("DESCRIBE llm_cache").fetchall()}
    if "response_raw" not in cols:
        con.execute("ALTER TABLE llm_cache ADD COLUMN response_raw VARCHAR")


class MissingResultError(RuntimeError):
    """A submitted custom_id never came back. Never drop it silently."""


class DemotionRateError(RuntimeError):
    """Too many records could not support their own verdict."""


class SpendLimitError(RuntimeError):
    """Running spend exceeded the cap."""


class BatchFailedError(RuntimeError):
    """The batch reached a terminal state that is not success."""


# The Batch API's `processing_status` vocabulary is exactly three values:
# `in_progress` | `canceling` | `ended`. A cancelled batch reports "canceling"
# and then ENDS, carrying per-request `result.type == "canceled"`.
#
# The previous tuples mixed the two vocabularies: canceled/errored/expired/
# failed are RESULT types (handled in parse_result), and "completed" is emitted
# by neither. The cost was real — fed the actual "canceling", run() treated it
# as unrecognised and polled to its 5,000 bound, which at the default
# poll_seconds=0.0 is a busy loop.
OK_STATES = ("ended",)
DEAD_STATES = ("canceling", "cancelling")
# The ONLY status that means "keep waiting". Anything outside these three sets is
# unknown and raises at once rather than polling to the bound: an unrecognised
# status used to spin 5,000 times, and with the default poll_seconds=0.0 that is
# a busy loop. Failing loud on an unknown value also means a future API status
# surfaces as an error instead of a hang.
IN_PROGRESS_STATES = ("in_progress",)


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
              facets: Mapping[str, Any], usage: Mapping[str, int],
              raw_facets: Mapping[str, Any] | None = None) -> None:
    """Store the VALIDATED record and, separately, the model's RAW answer.

    Only the validated record is ever served (see cache_get). The raw copy
    exists so that a change to the VALIDATOR can be re-evaluated for $0.

    The pilot is why. It demoted 55% of records on a validator bug, not a
    prompt bug — the model's answers were fine, only the judgement of them was
    wrong. But the cache held post-validation records keyed on the ad text, so
    a plain re-run served the demoted verdicts back and re-measuring meant
    paying the API again. At 44 ads that was $0.06; at 9,823 clusters it is
    $14.50, and a validator bug is the likeliest reason to need a re-run.

    Storing both also answers the objection that motivated the original
    design: caching the raw response ALONE would serve an unvalidated verdict
    and skip validation on every rerun. Nothing unvalidated is served here.
    """
    con.execute(
        "INSERT OR REPLACE INTO llm_cache (key, model, prompt_version, response, "
        "input_tokens, output_tokens, created_at, response_raw) VALUES (?,?,?,?,?,?,?,?)",
        [cache_key(model, prompt_version, text), model, prompt_version,
         json.dumps(facets, ensure_ascii=False),
         usage.get("input_tokens", 0), usage.get("output_tokens", 0),
         datetime.now(timezone.utc),
         None if raw_facets is None else json.dumps(raw_facets, ensure_ascii=False)])


def cache_get_raw(con, model: str, prompt_version: str, text: str | None):
    """The model's unvalidated answer, for re-validation only.

    Never feed this to the pipeline. It exists so revalidate_cache can re-judge
    paid-for answers under new rules.
    """
    row = con.execute("SELECT response_raw FROM llm_cache WHERE key = ?",
                      [cache_key(model, prompt_version, text)]).fetchone()
    if not row or row[0] is None:
        return None
    try:
        return json.loads(row[0]) if isinstance(row[0], str) else row[0]
    except (json.JSONDecodeError, TypeError):
        return None


def revalidate_cache(con, key_to_text: Mapping[str, str]) -> int:
    """Re-run the CURRENT validator over stored raw answers. Returns the number
    of rows rewritten. Makes no API call and cannot: no client is passed.

    Rows predating the raw column are skipped, not blanked — losing paid-for
    work is the failure this whole mechanism exists to prevent.
    """
    n = 0
    for key, text in key_to_text.items():
        row = con.execute("SELECT response_raw FROM llm_cache WHERE key = ?",
                          [key]).fetchone()
        if not row or row[0] is None:
            continue
        try:
            raw = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        except (json.JSONDecodeError, TypeError):
            continue
        checked = validate_mod.validate(dict(raw), text or "")
        # `_reasons` is the audit trail run() attaches, and ad_facets.demoted is
        # derived from it. Writing the bare facets reset `demoted` to False and
        # `reasons` to [] on the next cached run while the demotion was still in
        # force -- a demotion whose audit flag says it never happened.
        out = dict(checked["facets"])
        out["_reasons"] = checked["reasons"]
        con.execute("UPDATE llm_cache SET response = ? WHERE key = ?",
                    [json.dumps(out, ensure_ascii=False), key])
        n += 1
    return n


# ── run ──────────────────────────────────────────────────────────────────────

def _collect(client, batch_id, expected: set[str], poll_seconds: float,
             max_polls: int = 5000) -> dict:
    """Poll to a terminal state.

    A status that is terminal but NOT successful must raise, not loop. The
    first version looped while `status not in ("ended","completed")`, so
    "canceled" span forever — and with poll_seconds=0.0 that is a busy loop at
    100% CPU with no timeout and no error. max_polls bounds the remaining case
    of a batch that never terminates at all.
    """
    for _ in range(max_polls):
        status = client.poll(batch_id)
        if status in OK_STATES:
            break
        if status in DEAD_STATES:
            raise BatchFailedError(f"batch {batch_id} reached terminal state {status!r}")
        if status not in IN_PROGRESS_STATES:
            raise BatchFailedError(
                f"batch {batch_id} reported unknown processing_status {status!r}; "
                f"expected one of {OK_STATES + DEAD_STATES + IN_PROGRESS_STATES}")
        time.sleep(poll_seconds)
    else:
        raise BatchFailedError(f"batch {batch_id} did not terminate after {max_polls} polls")
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
    ensure_schema(con)
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
                spend += usage.get("cache_read_input_tokens", 0) * PRICE_CACHE_READ
                spend += usage.get("cache_creation_input_tokens", 0) * PRICE_CACHE_WRITE
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
        cache_put(con, model, PROMPT_VERSION, text, checked["facets"], usage,
                  raw_facets=facets)

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
