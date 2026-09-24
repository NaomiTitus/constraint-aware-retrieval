"""Preparation and post-validation for the LLM facet census.

Post-validation exists because a verbatim-substring check alone is not enough:
578 ads contain "norsk og engelsk", so a model can quote the fragment "engelsk"
and pass a substring test while supporting the opposite of what the sentence says.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from .census_prompt import HEAD_CHARS, TAIL_CHARS

from .text_norm import normalise as norm   # the single shared implementation


LANG_TOKEN = re.compile(
    r"norsk|norwegian|engelsk|english|skandinavisk|scandinavian|spr[åa]k|"
    r"munnleg|muntlig|skriftlig|skriftleg|norskpr[øo]ve|bergenstest|"
    r"niv[åa]\s*[ABC][12]|cefr|nynorsk|bokm[åa]l|samisk",
    re.I,
)
# Norwegian job ads are written as subjectless bullet fragments
# ("Behersker norsk eller engelsk"), so a finite-verb test is the wrong tool.
# The defence against fragment-quoting is a BOUNDARY rule: a span must begin
# where a sentence or bullet begins.
BOUNDARY = set(".!?:;\u2022\u2013\u2014-*\n")


def prepare(title: str, body: str, doc_lang: str) -> dict:
    """Head+tail truncation. Head-only preferentially cuts the closing
    'Kvalifikasjoner' block (21 ads measured, but the fix is free)."""
    truncated = len(body) > HEAD_CHARS + TAIL_CHARS
    sent = (body if not truncated
            else body[:HEAD_CHARS] + "\n[...]\n" + body[-TAIL_CHARS:])
    return {"title": title, "body": sent, "doc_lang": doc_lang,
            "truncated": truncated, "sent_text": sent}


def _span_ok(span: str, sent_text: str) -> tuple[bool, str]:
    n_span, n_text = norm(span), norm(sent_text)
    if n_span not in n_text:
        return False, "not_verbatim"
    # Norwegian ads bullet short complete statements ("Gode norskkunnskaper."),
    # so length is a weak signal. The BOUNDARY rules below do the real work.
    if len(n_span) < 15:
        return False, "fragment_too_short"
    if not LANG_TOKEN.search(n_span):
        return False, "no_language_token"
    # Must begin where a sentence or bullet begins. This is what defeats the
    # 578-ad trap: "engelsk" lifted out of "du må beherske norsk og engelsk"
    # is preceded by "...og ", not by a boundary.
    i = n_text.find(n_span)
    if i > 0:
        j = i - 1
        while j >= 0 and n_text[j].isspace():
            j -= 1
        if j >= 0 and n_text[j] not in BOUNDARY:
            return False, "starts_mid_sentence"
    # ...and must END at a boundary, so a prefix of a longer sentence is rejected.
    # The span usually carries its own terminator ("Gode norskkunnskaper."), so
    # check that first before looking at what follows.
    if n_span[-1] not in BOUNDARY:
        k = i + len(n_span)
        while k < len(n_text) and n_text[k].isspace():
            k += 1
        if k < len(n_text) and n_text[k] not in BOUNDARY:
            return False, "ends_mid_sentence"
    return True, ""


# Combinations the schema permits but which are semantically impossible.
def _coherence(f: dict) -> list[str]:
    errs, spans = [], f.get("evidence_spans") or []
    basis, strength = f.get("evidence_basis"), f.get("evidence_strength")
    lvl = f.get("norwegian_requirement_level")

    if basis == "explicit_statement" and not spans:
        errs.append("explicit_statement_without_span")
    if basis == "no_mention" and spans:
        errs.append("no_mention_with_span")
    if strength == "none" and spans:
        errs.append("strength_none_with_span")
    if strength in ("explicit_and_unambiguous", "explicit_but_hedged") and not spans:
        errs.append("explicit_strength_without_span")
    # absence is not negative evidence - the most frequent invalid record
    if lvl == "explicitly_not_required" and basis != "explicit_statement":
        errs.append("not_required_without_explicit_evidence")
    if lvl == "either_norwegian_or_english" and basis != "explicit_statement":
        errs.append("disjunction_without_explicit_evidence")
    if lvl != "unstated" and basis == "no_mention":
        errs.append("verdict_without_evidence")
    if f.get("seniority") == "senior" and (f.get("min_years_experience") or 0) == 0 \
            and f.get("min_years_experience") is not None:
        errs.append("senior_with_zero_years")
    return errs


def validate(facets: dict, sent_text: str) -> dict:
    """Returns {ok, facets, demoted, reasons}. Demotion converts a precision
    error into a recall error INVISIBLY, so the caller must track the rate."""
    reasons = list(_coherence(facets))
    kept = []
    for e in facets.get("evidence_spans") or []:
        ok, why = _span_ok(e.get("span", ""), sent_text)
        (kept if ok else reasons).append(e if ok else f"span:{why}")
    out = dict(facets, evidence_spans=kept)

    # A verdict resting only on rejected spans cannot stand.
    if not kept and out.get("norwegian_requirement_level") not in ("unstated", None):
        if out.get("evidence_basis") == "explicit_statement":
            out["norwegian_requirement_level"] = "unstated"
            out["evidence_basis"] = "no_mention"
            out["evidence_strength"] = "none"
            reasons.append("demoted:verdict_lost_its_evidence")
    if facets.get("truncated") and out.get("evidence_basis") == "no_mention":
        reasons.append("flag:no_mention_on_truncated_ad")
    return {"ok": not reasons, "facets": out,
            "demoted": any(r.startswith("demoted:") for r in reasons), "reasons": reasons}


# Near-duplicate clustering lives in dedup.py. An earlier cluster_key() here
# keyed on title + first 400 chars and performed WORSE than exact hashing,
# because chain stores vary the title per location while the body is identical.
