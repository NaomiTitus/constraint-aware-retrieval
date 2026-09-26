"""Preparation and post-validation for the LLM facet census.

Post-validation exists because a verbatim-substring check alone is not enough:
578 ads contain "norsk og engelsk", so a model can quote the fragment "engelsk"
and pass a substring test while supporting the opposite of what the sentence says.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping

from .census_prompt import HEAD_CHARS, TAIL_CHARS, TOOL

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
# BOUNDARY served two roles and they are NOT the same set.
#
#   START — what may precede a span. A colon or semicolon legitimately ends a
#           lead-in ("Krav: Gode norskkunnskaper"), so they belong here.
#   END   — what may terminate a span. A colon, semicolon or hyphen does NOT
#           end a sentence, so they must NOT be here: with them present,
#           `n_span[-1] not in BOUNDARY` was false and the whole end check was
#           SKIPPED. Measured: 1,568 block-prefix spans in 978 ads (9.6%) were
#           accepted while their block continued, and 87 of those in 64 ads
#           dropped a qualifier or negation. Ad 8da75b8c accepted "…i Øst-" and
#           dropped "Finnmark er positivt, men ikke et krav."
#
# Bullets and dashes are in neither. They are non-alphanumeric, so the furniture
# walk already steps over them to the block edge — and as terminators they also
# created FALSE START boundaries for Norwegian suspended compounding
# ("norsk- eller engelskkunnskaper", 6,350 ads / 62.5%, 507 blocks with a
# language token, 60 of those with a negation).
START_BOUNDARY = set(".!?:;\n")
END_BOUNDARY = set(".!?\n")

# Kept for callers that ask "is this a sentence edge at all". Not used by the
# span rules, which need the two sets above kept apart.
BOUNDARY = START_BOUNDARY


def prepare(title: str, body: str, doc_lang: str) -> dict:
    """Head+tail truncation. Head-only preferentially cuts the closing
    'Kvalifikasjoner' block (21 ads measured, but the fix is free)."""
    truncated = len(body) > HEAD_CHARS + TAIL_CHARS
    sent = (body if not truncated
            else body[:HEAD_CHARS] + "\n[...]\n" + body[-TAIL_CHARS:])
    return {"title": title, "body": sent, "doc_lang": doc_lang,
            "truncated": truncated, "sent_text": sent}


# The marker prepare() inserts at a truncation cut. The cut lands MID-BLOCK in
# 519 of the 528 truncated ads, and the surrounding newlines make it look like a
# block edge — so a sentence cut in half by US passed every boundary rule.
# Measured: 28 truncated ads accepted such a fragment, one of them quoting a
# disjunction whose "å engelsk." half our own truncation had deleted.
TRUNCATION_MARKER = "[...]"


def _touches_truncation(n_span: str, n_text: str, i: int) -> bool:
    """True if this occurrence begins or ends flush against a truncation cut."""
    before = n_text[:i]
    after = n_text[i + len(n_span):]
    return (before.rstrip(" \t\n").endswith(TRUNCATION_MARKER)
            or after.lstrip(" \t\n").startswith(TRUNCATION_MARKER))


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
        if not reason and _touches_truncation(n_span, n_text, i):
            reason = "truncated_fragment"
        if not reason:
            return True, ""
        first_reason = first_reason or reason
        i = n_text.find(n_span, i + 1)
    return False, first_reason


def _boundaries_ok(n_span: str, n_text: str, i: int) -> str:
    """"" if this occurrence begins and ends at a boundary, else the reason.

    Three kinds of character matter when walking outward from a span:

      TERMINATOR  . ! ? : ; and the newline that separates blocks -> accept
      FURNITURE   anything that is not a letter or digit -> skip and keep going
      TEXT        a letter or digit -> the span starts mid-sentence, reject

    FURNITURE is why this is not a character list. The corpus leads 2,998 blocks
    with a glyph that a hand-written BOUNDARY set missed: · U+00B7 (1,930
    blocks), ● U+25CF (141), 📍 ✅ ✨ 👉 ⭐ 🤝 🔹, U+200B zero-width space, U+2060
    word joiner, and U+F0B7 -- the Wingdings bullet Word emits into pasted ads.
    Enumerating them is a losing game; `not ch.isalnum()` is not.

    That mattered: the model quotes a bullet's TEXT without its glyph, so the
    preceding character IS the glyph. Nine probe ads had correct evidence thrown
    away and their verdicts demoted to `unstated`, which read as the model
    failing to see the line.
    """
    if i > 0:
        j = i - 1
        while j >= 0:
            ch = n_text[j]
            if ch in START_BOUNDARY:
                break                       # block edge or sentence terminator
            if ch.isalnum():
                return "starts_mid_sentence"
            j -= 1                          # list furniture: keep walking back

    if n_span[-1] not in END_BOUNDARY:
        k = i + len(n_span)
        while k < len(n_text):
            ch = n_text[k]
            if ch in END_BOUNDARY:
                break
            if ch.isalnum():
                return "ends_mid_sentence"
            k += 1
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


# Derived from the tool schema, never hand-listed: a field added to TOOL is
# validated the moment it exists. A maintained list would omit the next one —
# the failure mode every hardcoded pattern in this module has already had.
ENUMS = {k: frozenset(p["enum"])
         for k, p in TOOL["input_schema"]["properties"].items() if "enum" in p}

# Every enum field is `required` and `{"type": "string"}` in TOOL, so a missing
# or null value is a schema violation, not an omission. Derived, not asserted.
REQUIRED_ENUMS = frozenset(ENUMS) & frozenset(TOOL["input_schema"].get("required") or ())

# NESTED enums. `ENUMS` above walks `properties` ONE level, and three enums in
# TOOL live below that: a span's `section_language`, a skill's `level`, and
# `implicit_evidence`, whose ITEMS carry the enum rather than the field. All
# three accepted junk silently.
#
# The guard test meant to prove the derivation complete asserted
# `len(ENUM_FIELDS) >= 8`, and there are exactly 8 top-level enum fields — so it
# certified the blind spot, because it re-derived the list with the SAME
# one-level accessor the code used. A test that shares the code's accessor is
# not independent of the code.
#
#   path -> (kind, key within the item, allowed values)
#   "items" = the field is a list whose ITEMS are the enum
#   "key"   = the field is a list of objects, one of whose KEYS is the enum
def _nested() -> dict:
    out = {}
    for field, spec in TOOL["input_schema"]["properties"].items():
        items = (spec or {}).get("items")
        if not isinstance(items, dict):
            continue
        if "enum" in items:
            out[field] = ("items", None, frozenset(items["enum"]))
        for key, sub in (items.get("properties") or {}).items():
            if "enum" in sub:
                out[f"{field}.{key}"] = ("key", key, frozenset(sub["enum"]))
    return out


NESTED_ENUMS = _nested()

# What an out-of-enum value repairs TO. Not every enum carries `unstated`: the
# first version wrote `"unstated" if "unstated" in enum else None`, which
# repaired `evidence_basis` and `evidence_strength` to None — and both are
# required, non-nullable strings, so the "repaired" record violated the very
# schema the repair exists to satisfy. It was also undetectable on a second
# pass, because None reads as absent.
NEUTRAL = {"evidence_basis": "no_mention", "evidence_strength": "none"}
for _f, _allowed in ENUMS.items():
    NEUTRAL.setdefault(_f, "unstated")
    assert NEUTRAL[_f] in _allowed, f"{_f} repairs to a value not in its own enum"


# Type and size constraints, DERIVED from TOOL. Declared there and enforced by
# nothing: not by tool-use (which does not validate server-side), not here, not
# by scoring. Measured over all 196 persisted records: zero violations — so
# these are forward guards. But two of them RAISE rather than report, and a
# raise inside census.run()'s loop loses a paid-for batch:
#
#   evidence_spans = ["bare string"]  -> AttributeError: 'str' has no 'get'
#   application_language = ["both"]   -> TypeError: unhashable type
#
# A violation rate of 0 over 196 has a 95% upper bound near 1.5%; over the
# census's 9,823 records that is up to ~150. "Latent" measures the sample, not
# the risk.
_PROPS = TOOL["input_schema"]["properties"]
# Norwegian working life does not contain a 400-year career, and a negative
# requirement is not a requirement. Bounds are this module's, not the schema's,
# which declares only `integer`; recorded here rather than buried in a branch.
_MIN_YEARS_MAX = 60


def _type_ok(value, declared) -> bool:
    """`declared` is TOOL's `type`, which may be a string or a list of them."""
    names = declared if isinstance(declared, list) else [declared]
    for name in names:
        if name == "null" and value is None:
            return True
        if name == "string" and isinstance(value, str):
            return True
        # bool is a subclass of int in Python; the schema means them separately
        if name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if name == "boolean" and isinstance(value, bool):
            return True
        if name == "array" and isinstance(value, list):
            return True
        if name == "object" and isinstance(value, Mapping):
            return True
    return False


def _shape_errors(facets: Mapping) -> list[str]:
    """Type, size and item-shape violations. Never raises, whatever it is given."""
    out = []
    for field, spec in _PROPS.items():
        if field not in facets:
            continue
        value = facets[field]
        declared = spec.get("type")
        if declared and not _type_ok(value, declared):
            out.append(f"shape:{field}:expected_{declared}_got_{type(value).__name__}")
            continue                       # nothing further is meaningful
        if value is None:
            continue
        if isinstance(value, list):
            cap = spec.get("maxItems")
            if cap is not None and len(value) > cap:
                out.append(f"shape:{field}:maxItems_{cap}_got_{len(value)}")
            items = spec.get("items") or {}
            if items.get("type") == "object":
                for i, item in enumerate(value):
                    if not isinstance(item, Mapping):
                        out.append(f"shape:{field}[{i}]:expected_object_got_"
                                   f"{type(item).__name__}")
                        continue
                    for key, sub in (items.get("properties") or {}).items():
                        sv = item.get(key)
                        if sv is None:
                            continue
                        if sub.get("type") and not _type_ok(sv, sub["type"]):
                            out.append(f"shape:{field}[{i}].{key}:expected_"
                                       f"{sub['type']}_got_{type(sv).__name__}")
                        cap = sub.get("maxLength")
                        if cap is not None and isinstance(sv, str) and len(sv) > cap:
                            out.append(f"shape:{field}[{i}].{key}:maxLength_{cap}"
                                       f"_got_{len(sv)}")
        if field == "min_years_experience" and isinstance(value, int) \
                and not isinstance(value, bool):
            if value < 0 or value > _MIN_YEARS_MAX:
                out.append(f"shape:{field}:out_of_range_{value}")
    return out


def _member(v, allowed: frozenset) -> bool:
    """`v in frozenset` raises TypeError on a list or dict.

    Measured: `application_language: ["both"]` aborted the whole census, because
    census.run() does not guard validate(). The shape matters — the neighbouring
    field `implicit_evidence` IS an array, so borrowing a neighbour's SHAPE is
    the same failure class as borrowing its vocabulary, which is the failure
    this function was written for. An unhashable value is never a member.
    """
    try:
        return v in allowed
    except TypeError:
        return False


def _enum_errors(facets: dict) -> list[str]:
    """Out-of-enum values, which tool-use does NOT reject server-side.

    Found by human review, not by a test: three of the four
    `application_language` values in the corpus were borrowed from a NEIGHBOURING
    field's enum — `scandinavian_accepted` and `either_norwegian_or_english`
    from the level, `both` from the working language. The model recognises a
    documentation-language clause, has nowhere clean to put it, and reaches for
    vocabulary it has seen elsewhere in the same schema.

    These are the hard ones to spot by eye: each value is legal SOMEWHERE, so it
    reads as plausible until you check which field you are looking at.
    """
    out = []
    for field, allowed in ENUMS.items():
        v = facets.get(field)
        if v is None:
            # required and non-nullable in TOOL: absent IS a violation, and the
            # first version skipped it. A missing `norwegian_requirement_level`
            # then reached derive_english_accessible and raised KeyError inside
            # census.run()'s loop, after the batch was already paid for.
            if field in REQUIRED_ENUMS:
                out.append(f"enum:{field}=None")
            continue
        if not _member(v, allowed):
            out.append(f"enum:{field}={v!r}")

    for path, (kind, key, allowed) in NESTED_ENUMS.items():
        seq = facets.get(path.split(".", 1)[0])
        if not isinstance(seq, list):
            continue
        for item in seq:
            val = item if kind == "items" else (
                item.get(key) if isinstance(item, Mapping) else None)
            if val is not None and not _member(val, allowed):
                out.append(f"enum:{path}={val!r}")
    return out


def validate(facets: dict, sent_text: str) -> dict:
    """Returns {ok, facets, demoted, reasons}. Demotion converts a precision
    error into a recall error INVISIBLY, so the caller must track the rate."""
    reasons = list(_coherence(facets))
    reasons += _shape_errors(facets)
    reasons += _enum_errors(facets)
    kept = []
    spans = facets.get("evidence_spans")
    for e in (spans if isinstance(spans, list) else []):
        # A bare string here used to raise AttributeError and take the whole
        # census down with it. Reported by _shape_errors above; skipped here.
        if not isinstance(e, Mapping):
            continue
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

    # An out-of-enum value is not a warning: the record carries a value no
    # consumer can interpret, and a downstream filter comparing against the enum
    # silently drops the ad. Demote so it is counted, visible in the demotion
    # rate, and never persisted as if it were a real verdict.
    # A malformed field is not a weaker answer, it is an uninterpretable one —
    # same argument as the enum case. Demote, and replace the field with a value
    # its own schema permits so the persisted record is well-formed.
    bad_shape = [r for r in reasons if r.startswith("shape:")]
    if bad_shape:
        for r in bad_shape:
            # `shape:evidence_spans[1].span:maxLength_300_got_306` — the index
            # matters. The first version dropped the whole array on any item
            # fault, and the 400-ad census stage caught what that costs: ad
            # 08f980a7 lost BOTH its spans because one was six characters over,
            # and kept a `certified` verdict resting on nothing.
            locator = r.split(":", 2)[1]
            field = locator.split("[", 1)[0]
            idx = None
            if "[" in locator:
                try:
                    idx = int(locator.split("[", 1)[1].split("]", 1)[0])
                except ValueError:
                    idx = None
            spec = _PROPS.get(field) or {}
            declared = spec.get("type")
            names = declared if isinstance(declared, list) else [declared]

            # maxItems overflow: KEEP the cap, do not empty the field. Ad
            # 09f87e6b had 9 skills against a cap of 8 and lost all nine.
            if ":maxItems_" in r and isinstance(out.get(field), list):
                cap = spec.get("maxItems")
                if isinstance(cap, int):
                    out[field] = out[field][:cap]
                    continue
            if idx is not None and isinstance(out.get(field), list):
                # one bad ITEM: drop that item, keep the rest
                seq = list(out[field])
                if 0 <= idx < len(seq):
                    seq.pop(idx)
                    out[field] = seq
                continue
            if field in NEUTRAL:
                out[field] = NEUTRAL[field]
            elif "null" in names:
                out[field] = None
            elif "array" in names:
                out[field] = []
            elif "boolean" in names:
                out[field] = False
            else:
                out[field] = None
        reasons.append("demoted:invalid_shape")

    # RE-CHECK AFTER THE REPAIRS. The span-loss guard above ran before them, so
    # a repair that empties the evidence left the verdict standing on nothing —
    # a repair after the last check is an unchecked repair.
    if (not (out.get("evidence_spans") or [])
            and out.get("norwegian_requirement_level") not in ("unstated", None)
            and out.get("evidence_basis") == "explicit_statement"):
        out["norwegian_requirement_level"] = "unstated"
        out["evidence_basis"] = "no_mention"
        out["evidence_strength"] = "none"
        if "demoted:verdict_lost_its_evidence" not in reasons:
            reasons.append("demoted:verdict_lost_its_evidence")

    bad_enum = [r for r in reasons if r.startswith("enum:")]
    if bad_enum:
        for r in bad_enum:
            field = r.split(":", 1)[1].split("=", 1)[0]
            if field in NEUTRAL:                 # top-level scalar: repairable
                out[field] = NEUTRAL[field]
            # A nested violation (`evidence_spans.section_language`) is reported
            # and demotes the record, but is NOT rewritten: the right repair
            # depends on the item, and guessing one is the move that produced a
            # None in a required string field the first time.
        reasons.append("demoted:invalid_enum_value")
    return {"ok": not reasons, "facets": out,
            "demoted": any(r.startswith("demoted:") for r in reasons), "reasons": reasons}


# Near-duplicate clustering lives in dedup.py. An earlier cluster_key() here
# keyed on title + first 400 chars and performed WORSE than exact hashing,
# because chain stores vary the title per location while the body is identical.
