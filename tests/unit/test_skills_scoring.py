"""Scoring extracted skills. Scenarios 1-13, approved 2026-09-27.

THE VOCABULARY IS NOT MINE. `eval/scoring.py` already carries the answer, with the
measurement attached: over the 32 golden spans, 20 were strict-equal, 5 differed
ONLY by trailing punctuation, 7 were a containment, and 0 were genuinely different
sentences — so `exact` reads 62.5% while substantive agreement is 32 of 32, and
"five points of the gap are a full stop". Reporting an exact-match rate would put
that 62.5% in the README as extraction quality. This module reuses the same
four-way counter rather than inventing a second one.

AND THE LABELS ARE MODEL-PRODUCED (LIMITATIONS §15), which changes what a miss
means. An unpaired PREDICTION may be my own omission rather than the extractor's
error, so it is reported on its own line and never called a false positive. Calling
it one would let my omissions read as extractor precision failures — the same
self-agreement trap §15 records.
"""
from __future__ import annotations

import pytest

from finn_smart_search.eval import skills_scoring as S

# Real golden rows, verbatim from eval/golden_skills.json (ad 3dfb11f3, the GIS
# developer — the richest ad in the set at 7 skills).
G_GIS = [
    {"phrase": "ArcGIS Enterprise", "gloss_en": "ArcGIS Enterprise", "level": "required"},
    {"phrase": "Erfaring med Python-programmering og automatisering",
     "gloss_en": "Python programming and automation", "level": "required"},
    {"phrase": "Kunnskap om PostgreSQL og romlige databaser",
     "gloss_en": "PostgreSQL and spatial databases", "level": "required"},
    {"phrase": "Erfaring med smidig metodikk", "gloss_en": "agile methodology",
     "level": "preferred"},
]


def _p(phrase, gloss="x", level="required"):
    return {"phrase": phrase, "gloss_en": gloss, "level": level}


# ── classify: the four-way agreement counter ──────────────────────────────────

def test_identical_phrases_are_strict():
    assert S.classify("ArcGIS Enterprise", "ArcGIS Enterprise") == "strict"


def test_a_trailing_full_stop_is_its_own_category():
    """Scenario 2. Five of 32 golden spans differed by exactly this, and folding it
    into `disjoint` is what would report 62.5% as extraction quality."""
    assert S.classify("Erfaring med smidig metodikk",
                      "Erfaring med smidig metodikk.") == "trailing_punct_only"


def test_a_longer_quote_of_the_same_requirement_is_containment():
    """Scenario 3. Seven of 32. The model quoted a superset sentence — correct
    behaviour, not a miss."""
    assert S.classify(
        "Erfaring med Python-programmering og automatisering",
        "Erfaring med Python-programmering og automatisering av prosesser"
    ) == "containment"


def test_a_shorter_quote_is_also_containment():
    assert S.classify("Kunnskap om PostgreSQL og romlige databaser",
                      "PostgreSQL og romlige databaser") == "containment"


def test_an_unrelated_phrase_is_disjoint():
    assert S.classify("ArcGIS Enterprise", "Førerkort klasse B") == "disjoint"


@pytest.mark.parametrize("glyph", ["•", "·", "-", "*"])
def test_a_leading_block_glyph_does_not_break_the_match(glyph):
    """Scenario 5. `•` leads 4,600 corpus blocks and the model quotes without it."""
    assert S.classify("ArcGIS Enterprise", f"{glyph} ArcGIS Enterprise") in (
        "strict", "containment")


def test_case_and_whitespace_differences_do_not_break_the_match():
    assert S.classify("ArcGIS Enterprise", "arcgis   enterprise") == "strict"


# ── pairing ───────────────────────────────────────────────────────────────────

def test_every_golden_skill_pairs_when_the_prediction_is_perfect():
    pairs, ug, up = S.pair_skills(G_GIS, [dict(g) for g in G_GIS])
    assert len(pairs) == 4 and not ug and not up
    assert all(p.agreement == "strict" for p in pairs)


def test_pairing_is_one_to_one_even_when_two_predictions_could_match():
    """Scenario 11. Recall must never exceed 1. Measured on the labels: zero pairs
    of golden phrases on one ad contain each other, so ambiguity can only come from
    the PREDICTION side."""
    preds = [_p("Erfaring med smidig metodikk"),
             _p("Erfaring med smidig metodikk og DevOps")]
    pairs, ug, up = S.pair_skills([G_GIS[3]], preds)
    assert len(pairs) == 1
    assert len(up) == 1


def test_a_missed_golden_skill_is_unpaired_golden():
    pairs, ug, up = S.pair_skills(G_GIS, [dict(G_GIS[0])])
    assert len(pairs) == 1 and len(ug) == 3 and not up


def test_an_extra_prediction_is_unpaired_prediction_and_not_a_false_positive():
    """Scenario 10's sibling, and the §15 point. The label set is model-produced, so
    an extra prediction may be MY omission. It is reported on its own line and the
    report must not name it a false positive."""
    pairs, ug, up = S.pair_skills([G_GIS[0]], [dict(G_GIS[0]), _p("Azure")])
    assert len(pairs) == 1 and len(up) == 1


def test_a_level_disagreement_still_pairs(): 
    """Scenario 9. A required/preferred error is a DIFFERENT error from failing to
    find the skill, and the 121/23 split means level mistakes are asymmetric."""
    pairs, _, _ = S.pair_skills([G_GIS[3]],
                                [_p(G_GIS[3]["phrase"], level="required")])
    assert len(pairs) == 1
    assert pairs[0].level_match is False


def test_a_gloss_disagreement_still_pairs():
    """Scenario 10. Two models gloss differently by nature; that is not a pairing
    failure and must not enter F1."""
    pairs, _, _ = S.pair_skills([G_GIS[0]], [_p("ArcGIS Enterprise", gloss="GIS")])
    assert len(pairs) == 1 and pairs[0].gloss_match is False


# ── the aggregate report ──────────────────────────────────────────────────────

def test_a_correct_zero_is_scored_and_does_not_divide_by_zero():
    """Scenario 6. Two of the 44 golden ads genuinely have no skills — one is a
    three-line cleaning advert. A scorer that crashed or scored them 0 would punish
    the correct answer."""
    r = S.score({"a": []}, {"a": []})
    assert r["n_ads"] == 1
    assert r["correct_zeros"] == 1
    assert r["precision"] is None or r["precision"] == 1.0


def test_predicting_skills_where_golden_has_none_does_not_zero_recall():
    """Scenario 7. Recall is undefined with no golden skills, not zero."""
    r = S.score({"a": [_p("Azure")]}, {"a": []})
    assert r["recall"] is None or r["recall"] == 1.0
    assert r["unpaired_predictions"] == 1


def test_finding_nothing_where_golden_has_skills_scores_recall_zero():
    """Scenario 8. THE FAILURE BEING FIXED: this is what the census does on 26 of
    the 44 golden ads."""
    r = S.score({"a": []}, {"a": G_GIS})
    assert r["recall"] == 0.0


def test_an_ad_missing_from_golden_is_skipped_and_counted():
    """Scenario 12, following `scoring.py`, which tracks `unscored` rather than
    silently dropping."""
    r = S.score({"a": [], "zz": [_p("x")]}, {"a": []})
    assert r["unscored"] == ["zz"]


def test_the_report_carries_the_agreement_breakdown_not_just_a_rate():
    """The whole point of the house vocabulary: a single number would hide that five
    points of the gap are a full stop."""
    r = S.score({"a": [_p("ArcGIS Enterprise.")]},
                {"a": [G_GIS[0]]})
    assert set(r["agreement"]) == set(S.AGREEMENT)
    assert r["agreement"]["trailing_punct_only"] == 1


def test_the_report_includes_the_census_baseline_when_given():
    """Scenario 13. Without the baseline beside it the lift is unmeasurable, and the
    lift is the entire question the pilot exists to answer."""
    r = S.score({"a": [dict(g) for g in G_GIS]}, {"a": G_GIS},
                census_baseline={"a": [_p("Førerkort klasse B")]})
    assert r["census"]["n_skills"] == 1
    assert r["n_predicted"] == 4
    assert r["census"]["lift"] == pytest.approx(4.0)


def test_gloss_agreement_is_reported_separately_from_f1():
    """Two models gloss differently by nature. Folding gloss agreement into F1 would
    make the headline a translation-similarity score."""
    r = S.score({"a": [_p("ArcGIS Enterprise", gloss="mapping platform")]},
                {"a": [G_GIS[0]]})
    assert r["f1"] == 1.0
    assert r["gloss_agreement"] == 0.0
