"""LLM relevance judge. Scenarios approved 2026-09-27; see eval/JUDGE_SCENARIOS.md.

THE ONE PROPERTY THAT VOIDS EVERY HEADLINE NUMBER IF IT BREAKS. The judge must never
see the extractor's output. If it does, `CVR@10` measures the extractor agreeing with
itself — `JUDGING_PROTOCOL.md` calls that "the single way this project could produce
a confidently wrong result" and STANDARDS §4 makes the isolation test a
non-negotiable. So this module reads only the advertisement text and the persona's
own words, and `tests/unit/test_judge_isolation.py` asserts both that no extractor
field name reaches the rendered prompt and that no such name appears in this code.

The prompt is prose with no snake_case identifier anywhere, which is not stylistic:
two extractor field names are ordinary English words (`skills`, `seniority`), so the
isolation test allowlists those two and relies on the snake_case rule to catch them
the moment they appear as identifiers. Field names live in the tool schema instead.

WHY EVIDENCE NORMALISATION IS THE CAREFUL PART. The protocol requires a violation to
quote the advertisement sentence, and STANDARDS §3.1 records what a naive check
costs: a corpus test passed blocks WITH their leading bullet while the model quotes
them WITHOUT it, and the result reported "0 false rejects" while the bug was live.
Counted over all 10,166 ads (JUDGE_SCENARIOS G2): `•` leads 4,600 blocks, `-` 3,558,
`·` 1,944, `*` 503, `✅` 182, `●` 162, `«` 143, `📍` 135, and U+200B appears in 96
ads, U+FEFF in 14, U+2060 in 4. All of that is stripped from both sides before
comparison, so a truthful quotation is never reported as fabricated.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

JUDGE_TOOL_NAME = "record_relevance_judgment"

# STANDARDS §4: every gold row carries its versions. A judgment IS gold data, and
# `JUDGING_PROTOCOL.md` permits exactly ONE revision of this prompt before the human
# calibration set is labelled and zero after — a rule that is unenforceable unless
# each stored judgment says which prompt produced it. Bump this when the prompt
# changes, and the protocol's clock becomes auditable instead of remembered.
JUDGE_PROMPT_VERSION = "judge-v1"

# 10,000 chars covers p99 = 9,743 of the corpus (JUDGE_SCENARIOS G1: p50 2,500,
# p90 5,013, max 18,081). A tighter cap would cut requirement sentences out of a
# fifth of ads — 19.9% exceed 4,000 — and a requirement the judge never saw reads
# as a clean advertisement.
AD_TEXT_CAP = 10_000
TRUNCATION_NOTICE = "[advertisement truncated here]"

VIOLATED_FACETS = ("language", "authorisation", "licence", "other")

# Invisibles measured in the corpus, plus the two space variants that survive HTML
# cleaning. Removed from both sides of an evidence comparison.
_INVISIBLE = dict.fromkeys(map(ord, "\u200b⁠﻿­"), None)
_NBSP = {0x00A0: " "}

_AUTHORISATION = re.compile(r"autoris|authoris|authoriz", re.IGNORECASE)


class JudgeValidationError(ValueError):
    """A judgment that would corrupt the metric if it were stored."""


@dataclass(frozen=True)
class Judgment:
    pair_id: str
    grade: int
    violates: bool
    violated_facet: str | None = None
    constraint_evidence: str = ""
    notes: str = ""
    judge_prompt_version: str = JUDGE_PROMPT_VERSION


def judge_tool_schema() -> dict[str, Any]:
    """The tool the judge must call.

    Kept out of `render_prompt` because the prompt may contain no snake_case
    identifier while a schema necessarily does. The judge's OUTPUT field names are
    not the extractor's, so they are safe here and would be noise there.
    """
    return {
        "name": JUDGE_TOOL_NAME,
        "description": ("Record one relevance judgment for one advertisement "
                        "against one job seeker."),
        "input_schema": {
            "type": "object",
            "properties": {
                "grade": {"type": "integer", "minimum": 0, "maximum": 3},
                "violates": {"type": "boolean"},
                "violated_facet": {"type": ["string", "null"],
                                   "enum": [*VIOLATED_FACETS, None]},
                "constraint_evidence": {
                    "type": "string",
                    "description": ("The sentence from the advertisement, copied "
                                    "exactly. Empty when nothing is violated."),
                },
                "notes": {"type": "string"},
            },
            "required": ["grade", "violates"],
        },
    }


def render_prompt(persona_query: str, ad_title: str, ad_text: str) -> str:
    """The judge's user message: the seeker's own words and the raw advertisement.

    Every rule the protocol states is written out here, because a rule that is not
    in the prompt is not enforced — it is only documented.
    """
    body = ad_text
    if len(body) > AD_TEXT_CAP:
        body = body[:AD_TEXT_CAP] + "\n" + TRUNCATION_NOTICE
    return f"""You are grading one job advertisement for one job seeker.

THE JOB SEEKER SAID, in their own words:
{persona_query}

THE ADVERTISEMENT
Title: {ad_title}
---
{body}
---

TASK ONE — a relevance grade from 0 to 3, judged against what this seeker says
they can do and want, and IGNORING language entirely. Language is recorded
separately in task two, and mixing the two makes a poor match impossible to tell
apart from a job this person simply cannot take.

  3  Squarely the job this person is looking for. Right occupation, right level.
  2  Plausible. Right field, but the level, specialism or contract type is off.
  1  Adjacent. Shares some of what they can do or the sector, but this person
     would not apply.
  0  Irrelevant. A different occupation entirely.

Grade the advertisement AS WRITTEN, not the employer's likely flexibility. "They
would probably consider her" is grade 1, not grade 3.

If a grade is genuinely uncertain, record the LOWER one and say why in your notes.
Optimism inflates the result for every system equally and destroys the comparison
against the human baseline.

TASK TWO — does this advertisement require something the seeker explicitly said
they do not have? Judge this INDEPENDENTLY of the grade, and only on evidence in
the advertisement text. Quote that sentence exactly as it appears.

Five rules decide this, and they are not interchangeable:

  1. SILENCE IS NOT A VIOLATION. An advertisement that says nothing about a
     requirement does not impose it, however Norwegian it looks. Over a third of
     this corpus says nothing about language. Inferring a requirement from
     silence is the exact mistake this evaluation exists to detect, so a judge
     that makes it cannot detect anything.
  2. "norsk autorisasjon" IS NOT A LANGUAGE REQUIREMENT. Professional
     authorisation to practise is a licence. If the seeker lacks it, that is a
     violation of authorisation — never of language.
  3. "norsk eller engelsk" IS NOT A VIOLATION for an English speaker. It is a
     choice: either language will do.
  4. A clause about what language the PAPERWORK may be in is not a requirement on
     the applicant at all.
  5. A grade 3 advertisement CAN be a violation, and that combination is the most
     informative thing you can record: a perfect match this person cannot take.

Call the tool once."""


def normalise_for_evidence(text: str) -> str:
    """Reduce text to what survives being quoted by a model.

    Leading block glyphs go because the model quotes the sentence and not the
    bullet in front of it — G2 counted `•` leading 4,600 corpus blocks. Invisibles
    go because U+200B, U+FEFF and U+2060 are present in 114 ads between them and a
    judgment rejected over a zero-width space is a false fabrication report.
    """
    cleaned = unicodedata.normalize("NFKC", text).translate(_INVISIBLE).translate(_NBSP)
    out: list[str] = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        # Drop leading non-alphanumeric run: bullets, dashes, emoji, quotes.
        i = 0
        while i < len(stripped) and not (stripped[i].isalnum() or stripped[i] == "«"):
            i += 1
        if i and i < len(stripped):
            stripped = stripped[i:].lstrip()
        elif stripped[:1] == "«":
            stripped = stripped[1:].lstrip()
        if stripped:
            out.append(" ".join(stripped.split()))
    return " ".join(out).casefold()


def validate_judgment(judgment: Judgment, ad_text: str) -> None:
    """Raise `JudgeValidationError` unless this judgment could be true of this ad.

    The fabrication check is `census_validate`'s discipline turned onto the judge:
    without it a judge can invent the requirement it reports and `CVR@10` counts it.
    """
    if not isinstance(judgment.grade, int) or isinstance(judgment.grade, bool):
        raise JudgeValidationError(f"grade must be an integer, got {judgment.grade!r}")
    if not 0 <= judgment.grade <= 3:
        raise JudgeValidationError(
            f"grade {judgment.grade} is outside the 0-3 scale; nDCG gains are "
            f"2**grade - 1, so an out-of-range grade corrupts the metric silently")
    if judgment.violated_facet is not None and judgment.violated_facet not in VIOLATED_FACETS:
        raise JudgeValidationError(
            f"unknown violated facet {judgment.violated_facet!r}; "
            f"expected one of {VIOLATED_FACETS}")

    if not judgment.violates:
        return

    evidence = judgment.constraint_evidence.strip()
    if not evidence:
        raise JudgeValidationError(
            "a violation must quote the advertisement sentence; the protocol "
            "records `true` only on evidence in the text")

    if normalise_for_evidence(evidence) not in normalise_for_evidence(ad_text):
        raise JudgeValidationError(
            f"quoted evidence not found in the advertisement: {evidence!r}")

    if judgment.violated_facet == "language" and _AUTHORISATION.search(evidence):
        raise JudgeValidationError(
            "professional authorisation recorded as a language violation; it is a "
            "licensing requirement and belongs under authorisation")


def build_request(pair_id: str, persona_query: str, ad_title: str, ad_text: str,
                  model: str) -> dict[str, Any]:
    """One Batch API request envelope, in the shape `build_batch` expects."""
    return {
        "custom_id": pair_id,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "params": {
            "model": model,
            "max_tokens": 1024,
            "tools": [judge_tool_schema()],
            "tool_choice": {"type": "tool", "name": JUDGE_TOOL_NAME},
            "messages": [{"role": "user",
                          "content": render_prompt(persona_query, ad_title, ad_text)}],
        },
    }


def parse_judgment(raw: Mapping[str, Any],
                   prompt_version: str = JUDGE_PROMPT_VERSION) -> Judgment | dict[str, Any]:
    """Recorded-envelope row -> `Judgment`, or the client's error dict.

    Reuses `anthropic_client.parse_result`, which already handles the nested
    `tool_use` lookup and the standing ruling that a `max_tokens` stop is an error
    rather than something to salvage — a half-parsed judgment is indistinguishable
    from a real one downstream.
    """
    from finn_smart_search.ingest.anthropic_client import parse_result

    res: dict[str, Any] = parse_result(raw, JUDGE_TOOL_NAME)
    if res.get("type") != "succeeded":
        return res
    payload = res["facets"]
    return Judgment(
        pair_id=str(res["custom_id"]),
        grade=int(payload["grade"]),
        violates=bool(payload["violates"]),
        violated_facet=payload.get("violated_facet") or None,
        constraint_evidence=str(payload.get("constraint_evidence") or ""),
        notes=str(payload.get("notes") or ""),
        judge_prompt_version=prompt_version,
    )
