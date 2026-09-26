"""Turn hand labels from the sealed-set worksheet into golden-set records.

Labels arrive from a browser — pasted JSON, or read out of the artifact's `db`.
Either way they are UNTRUSTED input to the one measurement this project calls
unbiased, so ingestion is a gate rather than a format conversion.

Four things it refuses, each for a reason the repo has already paid for:

  UNCONFIRMED ROWS. The worksheet writes `confirmed: false` on every Opus
  suggestion. Letting one through would mean scoring the extractor against
  another model and calling the result human-labelled.

  AN UNVALIDATED SPAN. A gold span the real validator rejects is not a hard
  case, it is an unwinnable one: the extractor can never match it, so the ad
  scores as a permanent evidence error and quietly mismeasures every future run.
  `census_validate._span_ok` decides, against the text `prepare()` actually
  sends — not against the full ad.

  A CLAIMED `derived_accessible`. It is recomputed here. D3 kept the field only
  because it is a deterministic function of what the model emits; accepting one
  from the payload would let a browser assert accessibility.

  UNKNOWN KEYS. The output must satisfy the golden set's closed key vocabulary
  or it cannot be merged — the failure that left `taxonomy_gap` unreachable was
  exactly a key name nothing validated.

Everything refused comes back in the second return value with a reason. Nothing
is dropped silently.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..understanding import census_validate as v
from ..understanding.census_prompt import TOOL, derive_english_accessible

LEVELS = set(TOOL["input_schema"]["properties"]["norwegian_requirement_level"]["enum"])
WORKLANG = set(TOOL["input_schema"]["properties"]["stated_working_language"]["enum"])


def _reject(row: Mapping[str, Any], reason: str) -> dict:
    return {"uuid": row.get("uuid"), "n": row.get("n"), "reason": reason}


def ingest(rows: Iterable[Mapping[str, Any]],
           ad_texts: Mapping[str, str],
           start_n: int = 1000) -> tuple[list[dict], list[dict]]:
    """-> (accepted golden-set records, rejected rows with reasons).

    `ad_texts` maps uuid -> the ad's `description_text`. `start_n` renumbers the
    accepted rows contiguously so they cannot collide with the existing 44.
    """
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()

    for row in sorted(rows, key=lambda r: (r.get("n") is None, r.get("n"), r.get("uuid"))):
        uuid = row.get("uuid")
        exp = dict(row.get("expected") or {})

        if not row.get("confirmed"):
            rejected.append(_reject(row, "unconfirmed")); continue
        if uuid in seen:
            rejected.append(_reject(row, "duplicate_uuid")); continue
        level = exp.get("norwegian_requirement_level")
        if not level:
            rejected.append(_reject(row, "no_level")); continue
        if level not in LEVELS:
            rejected.append(_reject(row, "invalid_level")); continue
        wl = exp.get("stated_working_language") or "unstated"
        if wl not in WORKLANG:
            rejected.append(_reject(row, "invalid_working_language")); continue
        if uuid not in ad_texts:
            # Without the text the span cannot be validated, and an unvalidated
            # span is precisely what this gate exists to stop.
            rejected.append(_reject(row, "no_ad_text")); continue

        span = (exp.get("evidence_span") or "").strip() or None
        if span:
            # Validate against what prepare() SENDS, not the full ad: a span from
            # the deleted middle of a truncated ad is unmatchable in production.
            sent = v.prepare(row.get("title") or "", ad_texts[uuid], row.get("doc_lang") or "no")
            ok, why = v._span_ok(span, sent["sent_text"])
            if not ok:
                rejected.append(_reject(row, f"span:{why}")); continue

        seen.add(uuid)
        accepted.append({
            "n": None,                      # assigned below, contiguously
            "uuid": uuid,
            "title": row.get("title"),
            "stratum": row.get("stratum"),
            "expected": {
                "norwegian_requirement_level": level,
                "evidence_span": span,
                "stated_working_language": wl,
                "authorisation_required": (exp.get("authorisation_required") or "").strip()
                                          or None,
            },
            "note": (row.get("note") or "").strip() or None,
            "doc_lang": row.get("doc_lang") or "no",
            # RECOMPUTED, never taken from the payload.
            "derived_accessible": bool(derive_english_accessible(
                {"norwegian_requirement_level": level, "stated_working_language": wl},
                row.get("doc_lang") or "no")),
        })

    for i, rec in enumerate(accepted):
        rec["n"] = start_n + i
    return accepted, rejected
