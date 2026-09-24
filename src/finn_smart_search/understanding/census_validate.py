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

# Span matching must preserve BLOCK boundaries. `description_text` joins blocks
# with "\n", but norm() collapses all whitespace — destroying the newline that
# separated them. The boundary check then walked back to the PREVIOUS block's
# last character, and Norwegian ads are bullet lists whose items rarely end in
# punctuation. The pilot measured the cost: 26 of 28 span rejections were
# `starts_mid_sentence` on legitimate block quotes, demoting 55% of records.
_H_SPACE = re.compile(r"[^\S\n]+")      # horizontal whitespace only


def norm_keep_blocks(s: str | None) -> str:
    """Same folding as norm(), but newlines survive as block boundaries."""
    if not s:
        return ""
    return _H_SPACE.sub(" ", norm_chars(s)).strip()


def norm_chars(s: str) -> str:
    import unicodedata
    from .text_norm import _TRANSLATE
    return unicodedata.normalize("NFKC", s).translate(_TRANSLATE)


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
    """Is this span defensible evidence?

    Four rules. The boundary rules are what defeat fragment-quoting: 578 corpus
    ads contain "norsk og engelsk", so "engelsk" lifted out of it is a real
    substring that supports the OPPOSITE of what the sentence says.

    Newlines are preserved on both sides, because `description_text` joins
    BLOCKS with "\n" and a block edge is a boundary. Collapsing them made the
    check walk back to the previous block's last character — and Norwegian ads
    are bullet lists whose items rarely end in punctuation. The pilot measured
    the cost: 26 of 28 rejections were `starts_mid_sentence` on legitimate
    block quotes, demoting 55% of records against a 15% threshold.
    """
    n_span = norm(span)
    n_text = norm_keep_blocks(sent_text)

    if n_span not in n_text:
        return False, "not_verbatim"
    if len(n_span) < 15:
        return False, "fragment_too_short"
    if not LANG_TOKEN.search(n_span):
        return False, "no_language_token"

    # A span is defensible if it faithfully quotes SOME sentence or block, so
    # every occurrence gets a chance. Checking only the first rejected spans on
    # the strength of an occurrence nobody quoted; `rfind` would be no better.
    # Measured at 6 of 12,572 language-bearing corpus blocks (0.05%, 5 ads).
    first_reason = ""
    i = n_text.find(n_span)
    while i != -1:
        reason = _boundaries_ok(n_span, n_text, i)
        if not reason:
            return True, ""
        first_reason = first_reason or reason
        i = n_text.find(n_span, i + 1)
    return False, first_reason


def _boundaries_ok(n_span: str, n_text: str, i: int) -> str:
    """"" if this occurrence begins and ends at a boundary, else the reason.

    Horizontal space is skipped on both sides but newlines never are — the
    newline IS the boundary being looked for. `_H_SPACE` has already folded
    tabs, so the "\t" in these classes is defensive only.
    """
    if i > 0:
        j = i - 1
        while j >= 0 and n_text[j] in " \t":
            j -= 1
        if j >= 0 and n_text[j] not in BOUNDARY:
            return "starts_mid_sentence"

    # ...and END at one. The span usually carries its own terminator
    # ("Gode norskkunnskaper."), but a bullet block often does not.
    if n_span[-1] not in BOUNDARY:
        k = i + len(n_span)
        while k < len(n_text) and n_text[k] in " \t":
            k += 1
        if k < len(n_text) and n_text[k] not in BOUNDARY:
            return "ends_mid_sentence"
    return ""


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
