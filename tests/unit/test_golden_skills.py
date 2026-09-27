"""The golden skills labels. Scenario 10, approved 2026-09-27.

WHAT THIS FILE IS AND IS NOT. `eval/golden_skills.json` was produced by Claude
Opus, not by a human, because the project owner could not validate 44 ads of
Norwegian in the time available and chose to prove the concept first
(LIMITATIONS §15). So it is a PROVISIONAL gate, and a number computed against it
measures AGREEMENT BETWEEN TWO MODELS, not accuracy.

That is worth being blunt about: STANDARDS §3.1's clearest row is `test_9_1`, a
test that set the very field it then asserted on and measured nothing for weeks.
An LLM gate over LLM extraction is the same shape. What makes this better than
nothing rather than worse than nothing is that every label carries a VERBATIM
phrase, so a human can audit any row against the advertisement without re-reading
the corpus — and these tests enforce that.
"""
from __future__ import annotations

import json

import pytest

from finn_smart_search.understanding import skills_prompt as S
from tests.conftest import ROOT, requires_corpus

GOLDEN_SKILLS = ROOT / "eval" / "golden_skills.json"


@pytest.fixture(scope="module")
def labels():
    return json.loads(GOLDEN_SKILLS.read_text(encoding="utf-8"))


def test_the_provenance_says_a_model_produced_this(labels):
    """The single most important field in the file. Without it a later reader would
    take these for human labels, and every number derived from them would be
    reported as accuracy rather than as model agreement."""
    assert labels["labelled_by"] == "claude-opus-5"
    assert labels["provisional"] is True
    assert "LIMITATIONS" in labels["note"]


def test_every_golden_ad_is_covered(labels):
    golden = {r["uuid"] for r in json.loads(
        (ROOT / "eval" / "golden_set.json").read_text(encoding="utf-8"))}
    got = {a["uuid"] for a in labels["ads"]}
    assert got == golden, f"missing {sorted(golden - got)}; extra {sorted(got - golden)}"


def test_every_level_is_in_the_enum(labels):
    for ad in labels["ads"]:
        for sk in ad["skills"]:
            assert sk["level"] in S.LEVELS, (ad["uuid"], sk)


def test_every_skill_has_an_english_gloss(labels):
    """The gloss is the match key — ESCO's Norwegian labels miss 8 of 10 corpus
    phrasings, so a label without one cannot resolve."""
    for ad in labels["ads"]:
        for sk in ad["skills"]:
            assert sk["gloss_en"].strip(), (ad["uuid"], sk["phrase"])


@requires_corpus
def test_every_phrase_is_verbatim_in_its_advertisement(labels, corpus_con):
    """THE AUDIT GUARANTEE. This is what lets a human spot-check any row without
    trusting the labeller: the phrase must be findable in the advertisement."""
    texts = dict(corpus_con.execute(
        "SELECT uuid, description_text FROM ads").fetchall())
    for ad in labels["ads"]:
        for sk in ad["skills"]:
            S.validate_skill(
                S.Skill(phrase=sk["phrase"], gloss_en=sk["gloss_en"],
                        level=sk["level"]),
                texts[ad["uuid"]])


@pytest.mark.parametrize("excluded", [
    "fagbrev", "autorisasjon", "førerkort", "politiattest",   # credentials
    "norskkunnskaper", "behersker norsk",                      # language
    "personlig egnethet", "samarbeidsevner",                   # personal qualities
])
def test_no_excluded_category_leaked_into_the_labels(labels, excluded):
    """The three exclusions, checked on the labels rather than only stated in the
    prompt. `Personlige egenskaper` heads 1,730 health ads, so this is the easiest
    mistake to make and the one that would populate the field impressively while
    matching nothing."""
    leaked = [(a["uuid"], s["phrase"]) for a in labels["ads"] for s in a["skills"]
              if excluded in s["phrase"].lower()]
    assert not leaked, f"{excluded!r} leaked: {leaked[:3]}"


def test_a_legitimate_zero_is_recorded_rather_than_padded(labels):
    """Two golden ads genuinely ask for no skill at all — one is a 3-line cleaning
    advert whose only requirement is a driving licence. Padding them would teach the
    prompt to invent, which is the failure the whole exercise exists to fix in the
    other direction."""
    zeros = [a["uuid"] for a in labels["ads"] if not a["skills"]]
    assert zeros, "no ad has zero skills — suspicious; the corpus contains some"


@requires_corpus
def test_the_labels_find_materially_more_than_the_census_did(labels, corpus_con):
    """THE LEVER, QUANTIFIED on the same 44 advertisements. The census populated
    `skills` on 32.8% of the corpus; if these labels do not find substantially more
    on the golden ads, the re-prompt is not worth its cost and that is the finding."""
    rows = dict(corpus_con.execute(
        "SELECT uuid, facets FROM ad_facets").fetchall())
    census = 0
    for ad in labels["ads"]:
        raw = rows[ad["uuid"]]
        f = json.loads(raw) if isinstance(raw, str) else raw
        census += len(f.get("skills") or [])
    mine = sum(len(a["skills"]) for a in labels["ads"])
    assert mine > 2 * census, (
        f"golden labels {mine} skills vs census {census} on the same 44 ads — "
        f"not a material lift")
