"""Document language detection over cleaned ad blocks.

Feeds two things: the `document_language` attribute given to the LLM extractor,
and the silence-fallback in `derive_english_accessible`. A wrong verdict hides
jobs from people, so every constant here has a measurement behind it.

DETECT UNRESTRICTED. Restricting the detector to five Nordic/English languages
does not make a Polish ad fail — it force-projects it onto the nearest of the
five. Measured: 10 of 13 Polish ads got a confident "no", which under the
fallback means NOT accessible. The system would have hidden the ads most
obviously written for non-Norwegian speakers. Anything that is neither
Norwegian nor English becomes `other`, which is ACCESSIBLE: a Polish ad is not
being kept from you by a Norwegian language requirement.

NO sv/da VERDICTS. Zero ads in 1,500 are majority Swedish or Danish, and bokmål
descends from Danish — "Vi søker en dyktig medarbeider" is valid Danish
orthography. A class with no true instances has nothing to dilute error into,
so a single misread block would be 100% of its evidence. The Swedish-speaking
SEEKER is served in the matcher, where `scandinavian_accepted` meets a seeker
profile, not here on the ad side.
"""
from __future__ import annotations

import re
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

# Floor 10, not 40. Per-block accuracy (90% at 10-19 chars, 96% at 40-49) is the
# wrong level of analysis: char-weighting makes a 15-char block 15 of ~2,800
# characters. Measured at document level, floor 10 leaves ZERO ads decided on
# under half their text while floor 40 strands 19 of 1,500 and costs 6 unknowns
# — for no verdict change at all.
MIN_BLOCK_CHARS = 10

# Re-measured AT FLOOR 10, not inherited from the floor-40 run. The english-share
# distribution is strongly bimodal (1,136 ads below 0.1, 41 above 0.9, only 16
# anywhere between 0.15 and 0.85), so the band is insensitive: across 0.05-0.95
# to 0.40-0.60 just 4 ads of 1,500 change classification.
MIXED_LO, MIXED_HI = 0.20, 0.80

# An ad in a third language usually still carries a Norwegian contact or
# location line; that must not flip the verdict.
OTHER_DOMINANCE = 0.50

NORWEGIAN = {Language.BOKMAL, Language.NYNORSK}

# English legal/ATS furniture appended to otherwise Norwegian ads (104 corpus
# ads). Stripped BEFORE aggregating rather than absorbed by widening the band:
# a 450-char footer on a 1,200-char Norwegian ad gives share 0.27 -> "mixed",
# and the 0.20 bound is an artifact of this corpus's length distribution rather
# than a principle. Remove the cause, not the symptom.
_BOILERPLATE = re.compile(
    r"personal data|privacy polic|\bGDPR\b|equal opportunit|"
    r"regardless of (race|gender|age)|applicants? (of all|from all)|"
    r"powered by|cookie polic|terms of service",
    re.I,
)


@lru_cache(maxsize=1)
def _detector():
    """Full language set. The restriction that seemed prudent was the bug."""
    return LanguageDetectorBuilder.from_all_languages().build()


def is_boilerplate(text: str) -> bool:
    """Legal/ATS furniture, not content. Keyed on PHRASING, not on being
    English — keying on language would collapse every bilingual ad."""
    return bool(_BOILERPLATE.search(text))


def _classify(text: str) -> tuple[str, str | None]:
    """-> (class, detected_iso). class is 'no' | 'en' | 'other' | None."""
    lang = _detector().detect_language_of(text)
    if lang is None:
        return None, None
    if lang in NORWEGIAN:
        return "no", "no"
    if lang == Language.ENGLISH:
        return "en", "en"
    return "other", lang.iso_code_639_1.name.lower()


def detect(blocks: list[dict]) -> dict:
    """Char-weighted document verdict over already-cleaned blocks."""
    mass = {"no": 0, "en": 0, "other": 0}
    other_langs: dict[str, int] = {}
    n_scored = 0

    for b in blocks:
        text = b["text"]
        if b["n_chars"] < MIN_BLOCK_CHARS or is_boilerplate(text):
            continue
        cls, iso = _classify(text)
        if cls is None:
            continue
        mass[cls] += b["n_chars"]
        n_scored += 1
        if cls == "other":
            other_langs[iso] = other_langs.get(iso, 0) + b["n_chars"]

    total = sum(mass.values())
    if total == 0:
        return {"doc_lang": "unknown", "lang_mix": {}, "detected_other": None,
                "is_bilingual": False, "n_scored": 0, "confidence": "low"}

    mix = {k: v / total for k, v in mass.items()}
    detected_other = max(other_langs, key=other_langs.get) if other_langs else None

    if mix["other"] > OTHER_DOMINANCE:
        return {"doc_lang": "other", "lang_mix": mix, "detected_other": detected_other,
                "is_bilingual": False, "n_scored": n_scored, "confidence": "high"}

    # No `scand_en == 0` guard: if no+en were 0 then other would be the whole
    # mass, mix["other"] would be 1.0, and the dominance branch above would
    # already have returned. The branch was unreachable.
    share = mass["en"] / (mass["no"] + mass["en"])
    if share >= MIXED_HI:
        verdict = "en"
    elif share <= MIXED_LO:
        verdict = "no"
    else:
        verdict = "mixed"

    return {"doc_lang": verdict, "lang_mix": mix, "detected_other": detected_other,
            "is_bilingual": verdict == "mixed", "n_scored": n_scored,
            "confidence": "high" if n_scored >= 2 else "low"}
