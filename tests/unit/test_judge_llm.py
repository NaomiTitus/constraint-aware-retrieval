"""The judge prompt and judgment validation. Scenarios 1, 4–16, approved 2026-09-27.

The half of this file that matters most is `validate_judgment`, and specifically
the glyph normalisation. `JUDGING_PROTOCOL.md` requires a violation to quote the
advertisement sentence, and STANDARDS §3.1 row 3 records what happens when that
check is written against the wrong shape:

    the test's fixture   corpus check passing blocks WITH their glyph
    production reality   the model quotes the text WITHOUT it
    cost                 reported "0 false rejects" while the bug was live

So every leading glyph tested here is one that was COUNTED over the full corpus
(JUDGE_SCENARIOS G2), not one that seemed likely.
"""
from __future__ import annotations

import pytest

from finn_smart_search.eval import judge_llm as J
from finn_smart_search.eval.judge_llm import Judgment, JudgeValidationError
from tests.conftest import real_text

PERSONA = ("I am a nurse with hospital and nursing-home experience. I do not "
           "speak Norwegian. I am looking for a permanent position, preferably "
           "day shifts.")
TITLE = "Sykepleier søkes til sykehjem"

# Block-joined with `\n` and bullet-led: the production shape. G1 — p50 is 32
# newlines per ad; G2 — `•` leads 4,600 corpus blocks.
AD = ("Vi søker sykepleier til vårt sykehjem i Bergen.\n"
      "• Norsk autorisasjon som sykepleier\n"
      "• Gode norskkunnskaper, både muntlig og skriftlig\n"
      "• Erfaring fra sykehjem er en fordel, men ikke et krav")


# ── render_prompt ─────────────────────────────────────────────────────────────

def test_the_prompt_contains_the_advertisement_and_the_persona_verbatim():
    """Scenario 1. The judge grades what it is shown; if either side is reshaped,
    the grade is about something else."""
    p = J.render_prompt(PERSONA, TITLE, AD)
    assert AD in p
    assert PERSONA in p
    assert TITLE in p


def test_a_corpus_maximum_length_advertisement_is_truncated_and_says_so():
    """Scenario 4. G1: max is 18,081 chars. Silent truncation could cut the
    requirement sentence out and a violation the judge never saw reads as a clean
    ad — so the notice is part of the contract, not a nicety."""
    long_ad = "Vi søker sykepleier.\n" + ("fyllstoff " * 3000)
    assert len(long_ad) > J.AD_TEXT_CAP
    p = J.render_prompt(PERSONA, TITLE, long_ad)
    assert J.TRUNCATION_NOTICE in p
    assert len(p) < len(long_ad)


def test_a_median_length_advertisement_is_not_truncated():
    """Scenario 5. G1: p50 is 2,500 chars and p99 is 9,743, so the common case
    must pass through whole — a cap that fired at the median would corrupt most
    judgments."""
    mid = "Vi søker sykepleier.\n" + ("detaljer " * 250)
    assert len(mid) < J.AD_TEXT_CAP
    p = J.render_prompt(PERSONA, TITLE, mid)
    assert J.TRUNCATION_NOTICE not in p
    assert mid in p


@pytest.mark.parametrize("phrase", [
    "norsk autorisasjon",        # licensing, never a language violation
    "norsk eller engelsk",       # a disjunction: either suffices
])
def test_the_prompt_carries_the_violation_carve_outs_by_name(phrase):
    """Scenario 6. A protocol rule that is not in the prompt is not enforced."""
    assert phrase in J.render_prompt(PERSONA, TITLE, AD).lower()


def test_the_prompt_says_silence_is_not_a_violation():
    """Scenario 6, the carve-out that matters most: 36.5% of the corpus says
    nothing about language (LIMITATIONS §3), and inferring a requirement from
    silence is the exact error the extractor is built to avoid. A judge that does
    it cannot detect the extractor doing it."""
    low = J.render_prompt(PERSONA, TITLE, AD).lower()
    assert "silence" in low or "says nothing" in low


def test_the_prompt_says_a_top_grade_ad_can_still_be_a_violation():
    """Scenario 6. The protocol calls that combination "the most informative row in
    the whole dataset" — a perfect match the seeker cannot take."""
    low = J.render_prompt(PERSONA, TITLE, AD).lower()
    assert "grade 3" in low or "highest grade" in low


def test_the_prompt_instructs_the_lower_grade_when_uncertain():
    """Scenario 7. Systematic optimism inflates nDCG for every system equally but
    destroys the comparison against the manual FINN baseline."""
    low = J.render_prompt(PERSONA, TITLE, AD).lower()
    assert "lower" in low


def test_the_prompt_separates_the_grade_from_the_language_violation():
    """Scenario 8. Mixing them makes a bad match indistinguishable from an
    inaccessible one, which is the distinction the whole project is about."""
    low = J.render_prompt(PERSONA, TITLE, AD).lower()
    assert "ignor" in low            # the grade ignores language
    assert "independent" in low or "separately" in low


# ── validate_judgment: the scale and the evidence requirement ──────────────────

def _j(**kw):
    base = dict(pair_id="p1::abc", grade=3, violates=False,
                violated_facet=None, constraint_evidence="", notes="")
    base.update(kw)
    return Judgment(**base)


@pytest.mark.parametrize("grade", [-1, 4, 7])
def test_a_grade_outside_the_scale_is_rejected(grade):
    """Scenario 9. The protocol's scale is 0–3; anything else silently corrupts
    nDCG, whose gains are 2**grade - 1."""
    with pytest.raises(JudgeValidationError):
        J.validate_judgment(_j(grade=grade), AD)


@pytest.mark.parametrize("grade", [0, 1, 2, 3])
def test_every_grade_on_the_scale_is_accepted(grade):
    J.validate_judgment(_j(grade=grade), AD)


def test_a_violation_without_quoted_evidence_is_rejected():
    """Scenario 10. The protocol: record `true` only on evidence IN the ad text."""
    with pytest.raises(JudgeValidationError):
        J.validate_judgment(
            _j(violates=True, violated_facet="language",
               constraint_evidence=""), AD)


def test_no_violation_needs_no_evidence():
    """Scenario 15."""
    J.validate_judgment(_j(violates=False, constraint_evidence=""), AD)


# ── the census bug class: the model quotes without the glyph ───────────────────

@pytest.mark.parametrize("glyph", ["•", "·", "-", "*", "●", "✅", "«", "📍"])
def test_evidence_quoted_without_its_leading_block_glyph_is_accepted(glyph):
    """Scenarios 11–12, and the whole reason this function exists. Each glyph was
    COUNTED leading corpus blocks (G2): • 4,600, - 3,558, · 1,944, * 503, ✅ 182,
    ● 162, « 143, 📍 135. The model is shown `{glyph} Gode norskkunnskaper` and
    quotes `Gode norskkunnskaper`, which is correct behaviour and must not be
    rejected as fabricated."""
    ad = f"Vi søker sykepleier.\n{glyph} Gode norskkunnskaper er et krav"
    J.validate_judgment(
        _j(violates=True, violated_facet="language",
           constraint_evidence="Gode norskkunnskaper er et krav"), ad)


@pytest.mark.parametrize("invis", ["​", "﻿", "⁠"])
def test_evidence_differing_only_by_an_invisible_character_is_accepted(invis):
    """Scenario 13. G2: U+200B in 96 ads, U+FEFF in 14, U+2060 in 4. A judgment
    rejected over a zero-width space is a false fabrication report."""
    ad = f"Vi søker sykepleier.\n• Gode{invis} norskkunnskaper er et krav"
    J.validate_judgment(
        _j(violates=True, violated_facet="language",
           constraint_evidence="Gode norskkunnskaper er et krav"), ad)


def test_fabricated_evidence_is_rejected():
    """Scenario 14. The span must be real. This is `census_validate`'s discipline
    turned onto the judge — without it a judge can invent the requirement it is
    reporting, and CVR@10 would count it."""
    with pytest.raises(JudgeValidationError, match="not found"):
        J.validate_judgment(
            _j(violates=True, violated_facet="language",
               constraint_evidence="Applicants must hold a Norwegian passport"),
            AD)


def test_norsk_autorisasjon_cannot_be_recorded_as_a_language_violation():
    """Scenario 16. The protocol is explicit: professional authorisation to
    practise is a LICENSING requirement. This is the `norsk autorisasjon` trap the
    plan named on day one, and the judge is a new place to fall into it."""
    with pytest.raises(JudgeValidationError, match="authorisation"):
        J.validate_judgment(
            _j(violates=True, violated_facet="language",
               constraint_evidence="Norsk autorisasjon som sykepleier"), AD)


def test_norsk_autorisasjon_recorded_as_authorisation_is_accepted():
    """The same evidence under the right facet must pass, or the rule above is a
    blanket ban rather than a classification."""
    J.validate_judgment(
        _j(violates=True, violated_facet="authorisation",
           constraint_evidence="Norsk autorisasjon som sykepleier"), AD)


# ── the normaliser, against real corpus text ──────────────────────────────────

def test_the_normaliser_handles_real_corpus_blocks():
    """Grounded in the corpus rather than in this file's fixtures: every block of
    real ad text must survive normalisation and still be locatable in its own ad.
    If this fails, the validator would reject truthful judgments on real data —
    the failure mode §3.1 row 3 describes."""
    texts = real_text(20)   # plain helper, not a fixture — see tests/conftest.py
    checked = 0
    for t in texts:
        for block in (b for b in t.split("\n") if len(b.strip()) > 25):
            assert (J.normalise_for_evidence(block.strip())
                    in J.normalise_for_evidence(t))
            checked += 1
    assert checked > 20, f"only {checked} real blocks exercised"


# ── gold-data versioning (STANDARDS §4) ───────────────────────────────────────

def test_a_judgment_carries_the_prompt_version_that_produced_it():
    """STANDARDS §4: every gold row carries its versions. `JUDGING_PROTOCOL.md`
    permits ONE revision of this prompt before human calibration and zero after —
    a rule nothing can audit unless each judgment records which prompt it came
    from."""
    assert _j().judge_prompt_version == J.JUDGE_PROMPT_VERSION
    assert J.JUDGE_PROMPT_VERSION


def test_the_request_envelope_records_the_prompt_version_too():
    """So a submitted batch can be reconciled with the prompt it was built from,
    even before any result comes back."""
    req = J.build_request("p1_nurse-4821", PERSONA, TITLE, AD, model="claude-opus-5")
    assert req["judge_prompt_version"] == J.JUDGE_PROMPT_VERSION
    assert req["custom_id"] == "p1_nurse-4821"
    assert req["params"]["tool_choice"]["name"] == J.JUDGE_TOOL_NAME


# ── the Batch API's custom_id contract, learned from a live 400 ────────────────

def test_a_pair_id_with_a_colon_is_rejected_at_build_time():
    """GROUNDED IN AN ACTUAL API REJECTION, not the docs. Submitting 346 requests
    whose ids used `::` returned:

        400 requests.0.custom_id: String should match pattern '^[a-zA-Z0-9_-]{1,64}$'

    The API rejects the ENTIRE batch for one bad id and the 400 does not say which
    request was at fault, so this has to fail locally."""
    with pytest.raises(ValueError, match="Batch API pattern"):
        J.build_request("p1_nurse::4821", PERSONA, TITLE, AD, model="claude-opus-5")


def test_make_pair_id_produces_an_id_the_api_accepts():
    pid = J.make_pair_id("p1_sykepleier_no_norsk", 4821)
    assert pid == "p1_sykepleier_no_norsk-4821"
    assert J.PAIR_ID.match(pid)


def test_every_dev_persona_id_yields_a_legal_pair_id():
    """The real persona ids, not invented ones: the longest is 32 characters and a
    corpus index is up to 5 digits, so the 64-character ceiling has headroom — but
    it is asserted rather than assumed."""
    from tests.conftest import ROOT
    import yaml
    people = yaml.safe_load(
        (ROOT / "eval" / "personas.yaml").read_text(encoding="utf-8"))["personas"]
    for p in people:
        pid = J.make_pair_id(p["id"], 10165)
        assert J.PAIR_ID.match(pid), pid
        assert len(pid) <= 64


def test_a_pair_id_over_the_length_ceiling_is_rejected():
    with pytest.raises(ValueError, match="Batch API pattern"):
        J.make_pair_id("x" * 70, 1)
