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

def test_unstated_norwegian_ad_is_not_accessible():
    """Silence is not openness. Norwegian employers omit the requirement
    BECAUSE it is obvious — absence is negatively correlated with accessibility
    in exactly the care and retail roles that dominate the corpus."""
    assert derive(f("unstated"), "no") is False


@pytest.mark.parametrize("doc_lang", ["en", "mixed"])
def test_unstated_english_ad_is_accessible(doc_lang):
    assert derive(f("unstated"), doc_lang) is True


def test_stated_english_working_language_overrides_a_norwegian_ad():
    assert derive(f("unstated", working="english"), "no") is True


def test_stated_working_language_does_not_override_a_requirement():
    """Rule 4: on contradiction the stronger Norwegian requirement wins."""
    assert derive(f("professional", working="english"), "en") is False


# ── the levels must not drift apart ──────────────────────────────────────────

def test_every_level_is_classified():
    """A new level added to the enum without a rule here would silently fall
    through to the document-language default."""
    from finn_smart_search.understanding.census_prompt import (ACCESSIBLE_LEVELS,
                                                               BLOCKING_LEVELS, TOOL)
    enum = set(TOOL["input_schema"]["properties"]["norwegian_requirement_level"]["enum"])
    assert enum - ACCESSIBLE_LEVELS - BLOCKING_LEVELS == {"unstated"}, (
        "every level must be explicitly accessible, blocking, or the documented "
        "unstated fallback")
