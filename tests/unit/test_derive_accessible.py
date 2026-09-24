"""The english_accessible truth table.

This field is DERIVED, never asked of the model — it is a deterministic function
of `norwegian_requirement_level` + `stated_working_language` + the ad's own
language. It decides whether an ad reaches a non-Norwegian speaker at all, so it
is the single most consequential derivation in the project.

Written after the fact: a spec change (making `desirable` accessible) landed
with all 99 tests green because nothing pinned this table. That is exactly the
failure STANDARDS.md warns about.
"""
import pytest

from finn_smart_search.understanding.census_prompt import derive_english_accessible as derive

pytestmark = pytest.mark.unit


def f(level, working="unstated"):
    return {"norwegian_requirement_level": level, "stated_working_language": working}


# ── accessible regardless of the ad's own language ───────────────────────────

@pytest.mark.parametrize("doc_lang", ["no", "en", "mixed"])
def test_disjunction_is_accessible(doc_lang):
    """The 274-ad correction. "Behersker norsk ELLER engelsk" in a Norwegian-
    written ad must reach an English speaker; a naive reader ranks it DOWN."""
    assert derive(f("either_norwegian_or_english"), doc_lang) is True


@pytest.mark.parametrize("doc_lang", ["no", "en", "mixed"])
def test_explicitly_not_required_is_accessible(doc_lang):
    assert derive(f("explicitly_not_required"), doc_lang) is True


@pytest.mark.parametrize("doc_lang", ["no", "en", "mixed"])
def test_desirable_is_accessible(doc_lang):
    """Owner ruling. "norsk er en fordel" and "trenger ikke flytende norsk når
    du starter" both say Norwegian is NOT required — you can learn it on the
    job, so you can apply. Before this, `desirable` fell back to document
    language and was indistinguishable from `unstated`."""
    assert derive(f("desirable"), doc_lang) is True


# ── blocking regardless of the ad's own language ─────────────────────────────

@pytest.mark.parametrize("level", ["certified", "fluent", "professional", "conversational"])
@pytest.mark.parametrize("doc_lang", ["no", "en", "mixed"])
def test_requirement_levels_block(level, doc_lang):
    """An English-WRITTEN ad can still demand fluent Norwegian."""
    assert derive(f(level), doc_lang) is False


@pytest.mark.parametrize("doc_lang", ["no", "en", "mixed"])
def test_scandinavian_is_not_english(doc_lang):
    """Swedish and Danish qualify; English does not."""
    assert derive(f("scandinavian_accepted"), doc_lang) is False


# ── unstated falls back to the ad's own language ─────────────────────────────

@pytest.mark.parametrize("doc_lang", ["unknown", None, "", "NO", "sv"])
def test_unstated_is_inaccessible_unless_positively_english(doc_lang):
    """Language ID abstains on very short ads (13 in the corpus). `doc_lang !=
    "no"` is the refactor someone writes to "handle unknown", and it silently
    flips every one of them to accessible."""
    assert derive(f("unstated"), doc_lang) is False


def test_unstated_norwegian_ad_is_not_accessible():
    """Silence is not openness. Norwegian employers omit the requirement
    BECAUSE it is obvious — absence is negatively correlated with accessibility
    in exactly the care and retail roles that dominate the corpus."""
    assert derive(f("unstated"), "no") is False


@pytest.mark.parametrize("doc_lang", ["en", "mixed"])
def test_unstated_english_ad_is_accessible(doc_lang):
    assert derive(f("unstated"), doc_lang) is True


@pytest.mark.parametrize("working,expected", [
    ("english", True),        # explicit: English is the working language
    ("both", True),           # both -> an English speaker can work there
    ("norwegian", False),     # stated Norwegian beats an English-written ad
    ("scandinavian", False),  # Scandinavian is not English
    ("unstated", False),      # falls through to doc_lang
])
def test_working_language_axis_on_a_norwegian_ad(working, expected):
    """Only "english" was previously exercised, so accepting "scandinavian"
    too would have survived the whole suite."""
    assert derive(f("unstated", working=working), "no") is expected


def test_stated_norwegian_beats_an_english_written_ad():
    """The bug this suite missed: the doc_lang fallback never consulted a
    NON-English stated language, so an English-written ad declaring Norwegian
    as its working language came out accessible."""
    assert derive(f("unstated", working="norwegian"), "en") is False


def test_stated_working_language_does_not_override_a_requirement():
    """Rule 4: on contradiction the stronger Norwegian requirement wins."""
    assert derive(f("professional", working="english"), "en") is False


# ── the levels must not drift apart ──────────────────────────────────────────

EXPECTED_BY_LEVEL = {
    "certified": False, "fluent": False, "professional": False,
    "conversational": False, "scandinavian_accepted": False,
    "desirable": True, "either_norwegian_or_english": True,
    "explicitly_not_required": True,
}


def test_every_enum_level_has_a_pinned_verdict():
    """Drives the FUNCTION, not set arithmetic. The previous version asserted
    only that two constants partitioned the enum — it never called derive(), so
    it passed even if the function stopped reading those sets entirely."""
    from finn_smart_search.understanding.census_prompt import (ACCESSIBLE_LEVELS,
                                                               BLOCKING_LEVELS, TOOL)
    enum = set(TOOL["input_schema"]["properties"]["norwegian_requirement_level"]["enum"])
    assert enum == set(EXPECTED_BY_LEVEL) | {"unstated"}
    assert not (ACCESSIBLE_LEVELS & BLOCKING_LEVELS), "a level in both sets is ambiguous"
    for level, expected in EXPECTED_BY_LEVEL.items():
        for doc_lang in ("no", "en", "mixed", "unknown"):
            assert derive(f(level), doc_lang) is expected, (
                f"{level} must dominate doc_lang={doc_lang}")


def test_missing_level_fails_loud():
    """Fail-loud on the required facet. A future .get(..., "unstated") would
    convert crashes into a silent False for every malformed extraction."""
    with pytest.raises(KeyError):
        derive({"stated_working_language": "english"}, "en")
