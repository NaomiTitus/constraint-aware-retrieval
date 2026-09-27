"""Skills extraction. Scenarios 1-9 approved 2026-09-27.

The exclusions matter more than the inclusion here, and the measurement says why:
`Personlige egenskaper` heads 1,730 health ads and `Arbeidsoppgaver` 1,881, so the
two largest categories of bullet in this corpus are personal qualities and duties —
neither of which is a skill a seeker can claim. A prompt that admits them would
populate the field impressively and match nothing.

Every fixture is real corpus text or carries the count that justifies its shape,
per STANDARDS §3.0.
"""
from __future__ import annotations

import pytest

from finn_smart_search.understanding import skills_prompt as S
from finn_smart_search.understanding.skills_prompt import Skill, SkillsValidationError

# Verbatim from ad 4ccebbd0 and the Helse og sosial sample: block-joined with `\n`,
# bullet-led. G1 records p50 of 32 newlines per ad; G2 records `•` leading 4,600
# corpus blocks.
AD_TITLE = "Sykepleier søkes til medisinsk avdeling"
AD = ("Arbeidsoppgaver\n"
      "• Medikamentadministrasjon og deltakelse i legevisitt\n"
      "• Triagering og systematisk scoring (NEWS, ProAct)\n"
      "• Blodprøvetaking\n"
      "• Betjene kasse\n"
      "Kvalifikasjoner\n"
      "• Norsk autorisasjon som sykepleier\n"
      "• Gode norskkunnskaper, både muntlig og skriftlig\n"
      "Personlige egenskaper\n"
      "• Strukturert, selvstendig og engasjert\n"
      "• Gode samarbeidsevner og trives med teamarbeid")

NYNORSK = ("Sentrale arbeidsoppgåver\n"
           "• Sikkerheitskurs i høve til kvalifikasjonsføreskrifta\n"
           "• Krise- og passasjerhandtering")

ENGLISH = ("What you'll need to succeed\n"
           "• Experience with Cisco network administration\n"
           "• Comfortable working both independently and as part of a team")


def _s(**kw):
    base = dict(phrase="Blodprøvetaking", gloss_en="taking blood samples",
                level="required")
    base.update(kw)
    return Skill(**base)


# ── the prompt must carry the four-way distinction ────────────────────────────

def test_the_prompt_contains_the_advertisement_verbatim():
    p = S.render_prompt(AD_TITLE, AD)
    assert AD in p and AD_TITLE in p


def test_the_prompt_names_all_three_exclusions():
    """Scenarios 3-5. A rule that is not in the prompt is not enforced — it is only
    documented. These are the three categories that outnumber real skills in this
    corpus."""
    low = S.render_prompt(AD_TITLE, AD).lower()
    assert "arbeidsoppgaver" in low or "duties" in low or "tasks" in low
    assert "personlige egenskaper" in low or "personal qualit" in low
    assert "fagbrev" in low or "credential" in low or "licence" in low


def test_the_prompt_asks_for_a_verbatim_phrase_and_an_english_gloss():
    """The two-field design: evidence in the original language, match key in
    English, because ESCO's Norwegian labels miss 8 of 10 corpus phrasings."""
    low = S.render_prompt(AD_TITLE, AD).lower()
    assert "verbatim" in low or "exactly as" in low or "word for word" in low
    assert "english" in low


def test_the_prompt_tells_the_model_nynorsk_and_english_ads_both_count():
    """G4: nynorsk is common and whole ads are English. A prompt silent on this
    under-extracts on two of three languages — the same asymmetry that made BM25
    fail on English queries."""
    low = S.render_prompt(AD_TITLE, NYNORSK).lower()
    assert "nynorsk" in low
    assert "english" in low


def test_the_prompt_does_not_show_an_empty_skills_example():
    """THE MEASURED CAUSE OF THE 32.8%. Nine of the census's fifteen few-shots show
    `skills: []` and the model learned to return nothing. An example list that
    teaches emptiness is the one thing this prompt must not contain."""
    p = S.render_prompt(AD_TITLE, AD)
    assert "[]" not in p


# ── validation: the evidence must be real ─────────────────────────────────────

def test_a_phrase_present_verbatim_is_accepted():
    S.validate_skill(_s(), AD)


def test_a_fabricated_phrase_is_rejected():
    """Scenario 1. `census_validate`'s discipline: without it the extractor can
    invent the qualification it reports."""
    with pytest.raises(SkillsValidationError, match="not found"):
        S.validate_skill(_s(phrase="Sertifisert i robotkirurgi"), AD)


@pytest.mark.parametrize("glyph", ["•", "·", "-", "*", "●"])
def test_a_phrase_quoted_without_its_leading_glyph_is_accepted(glyph):
    """Scenario 2, and STANDARDS §3.1 row 3: the model quotes the block WITHOUT the
    bullet. Counted over the corpus — `•` leads 4,600 blocks, `-` 3,558, `·` 1,944."""
    ad = f"Kvalifikasjoner\n{glyph} Erfaring med sårbehandling"
    S.validate_skill(_s(phrase="Erfaring med sårbehandling",
                        gloss_en="wound care experience"), ad)


def test_a_missing_english_gloss_is_rejected():
    """Scenario 6. The gloss is the MATCH KEY — a skill without one cannot resolve
    against ESCO, where the English labels carry the coverage."""
    with pytest.raises(SkillsValidationError, match="gloss"):
        S.validate_skill(_s(gloss_en=""), AD)


@pytest.mark.parametrize("bad", ["mandatory", "nice-to-have", "", "REQUIRED"])
def test_a_level_outside_the_enum_is_rejected(bad):
    """Scenario 9. The level is what carries hard/soft into D7's weighted coverage,
    where a missing hard requirement disqualifies and a missing preference demotes."""
    with pytest.raises(SkillsValidationError, match="level"):
        S.validate_skill(_s(level=bad), AD)


@pytest.mark.parametrize("level", LEVELS := ("required", "preferred"))
def test_both_levels_are_accepted(level):
    S.validate_skill(_s(level=level), AD)


def test_a_nynorsk_phrase_validates_against_a_nynorsk_ad():
    """Scenario 7. The phrase stays in the language the employer wrote; only the
    gloss is English. ESCO has zero nynorsk labels, which is exactly why the gloss
    and not the phrase is the match key."""
    S.validate_skill(
        _s(phrase="Krise- og passasjerhandtering",
           gloss_en="crisis and passenger handling"), NYNORSK)


def test_an_english_ad_may_have_phrase_equal_to_gloss():
    """Scenario 8. For an English-language advertisement the evidence and the match
    key coincide, and forbidding that would reject 7% of the corpus."""
    S.validate_skill(
        _s(phrase="Experience with Cisco network administration",
           gloss_en="Experience with Cisco network administration"), ENGLISH)


def test_the_validator_does_not_itself_judge_whether_a_phrase_is_a_skill():
    """A DELIBERATE LIMIT, stated so nobody mistakes the validator for the gate.
    `Betjene kasse` is a DUTY and appears verbatim in the fixture, so it validates.
    Excluding duties is the PROMPT's job and the golden set's job to score; a
    regex-based classifier here would be the seventh hand-written pattern in this
    repo to look green and measure nothing (§3.1)."""
    S.validate_skill(_s(phrase="Betjene kasse", gloss_en="operating the till"), AD)
