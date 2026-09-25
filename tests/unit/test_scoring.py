"""Scoring the census against the frozen golden set.

Three design decisions, each with a reason:

  PER-LEVEL, NEVER POOLED. The golden set over-represents `desirable` by ~200x
  (4 of 44 ads for a level covering ~4 of 10,166). A single pooled accuracy
  number would be dominated by sampling design rather than by performance.

  THE HEADLINE METRIC IS THE ACCESSIBILITY FLIP, not level accuracy. Confusing
  `professional` with `fluent` changes nothing — both hide the ad. Confusing
  `either_norwegian_or_english` with `professional` flips an ad from SHOWN to
  HIDDEN. Level accuracy treats those as equally wrong; they are not.

  FLIPS ARE SPLIT BY DIRECTION. Wrongly hiding an ad is invisible to everyone;
  wrongly showing one costs a click. Reporting a single "flip rate" would
  average away the asymmetry the whole project turns on.
"""
import pytest

from finn_smart_search.eval import scoring

pytestmark = pytest.mark.unit


def gold(n, level, span=None, uuid=None, working="unstated", auth=None, doc_lang="no"):
    return {"n": n, "uuid": uuid or f"u{n}", "doc_lang": doc_lang,
            "expected": {"norwegian_requirement_level": level, "evidence_span": span,
                         "stated_working_language": working, "authorisation_required": auth}}


def pred(level, span=None, working="unstated", auth=None):
    spans = [{"span": span, "section_language": "no"}] if span else []
    return {"norwegian_requirement_level": level, "evidence_spans": spans,
            "stated_working_language": working, "authorisation_required": auth}


# ── 1 · level accuracy, per level ───────────────────────────────────────────

def test_1_1_per_level_breakdown_not_just_a_pooled_number():
    g = [gold(1, "unstated"), gold(2, "unstated"), gold(3, "desirable")]
    p = {"u1": pred("unstated"), "u2": pred("professional"), "u3": pred("desirable")}
    r = scoring.score(p, g)
    assert r["per_level"]["unstated"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert r["per_level"]["desirable"] == {"n": 1, "correct": 1, "accuracy": 1.0}


def test_1_2_pooled_accuracy_is_reported_but_flagged():
    """Present, because reviewers will ask — but carrying an explicit warning,
    because the golden set's level mix is a sampling artifact."""
    g = [gold(1, "unstated"), gold(2, "desirable")]
    p = {"u1": pred("unstated"), "u2": pred("unstated")}
    r = scoring.score(p, g)
    assert r["pooled_accuracy"] == 0.5
    assert "over-represent" in r["pooled_accuracy_caveat"]


def test_1_3_confusion_pairs_are_recorded():
    """Which level it confused for which is the actionable output; a scalar is
    not. `either_norwegian_or_english` -> `professional` is the dangerous pair."""
    g = [gold(1, "either_norwegian_or_english")]
    p = {"u1": pred("professional")}
    assert scoring.score(p, g)["confusion"][("either_norwegian_or_english", "professional")] == 1


# ── 2 · the headline metric: accessibility flips ────────────────────────────

def test_2_1_hidden_wrongly_is_counted_separately_from_shown_wrongly():
    g = [gold(1, "either_norwegian_or_english"),     # accessible
         gold(2, "professional")]                    # not accessible
    p = {"u1": pred("professional"),                 # -> wrongly HIDDEN
         "u2": pred("desirable")}                    # -> wrongly SHOWN
    f = scoring.score(p, g)["accessibility"]
    assert f["hidden_wrongly"] == 1 and f["shown_wrongly"] == 1


def test_2_2_a_level_error_that_does_NOT_flip_accessibility_is_not_a_flip():
    """professional -> fluent is wrong but harmless: both hide the ad. Counting
    it as a flip would drown the errors that actually change what a user sees."""
    g = [gold(1, "professional")]
    p = {"u1": pred("fluent")}
    r = scoring.score(p, g)
    assert r["per_level"]["professional"]["correct"] == 0
    assert r["accessibility"]["hidden_wrongly"] == 0
    assert r["accessibility"]["shown_wrongly"] == 0


def test_2_3_accessibility_uses_the_ads_real_doc_lang():
    """`unstated` resolves by doc_lang, so an English-written ad predicted
    `unstated` is accessible while a Norwegian one is not. Scoring with a
    hardcoded "no" would invent flips that did not happen."""
    g = [gold(1, "unstated", doc_lang="en")]
    p = {"u1": pred("unstated")}
    assert scoring.score(p, g)["accessibility"]["hidden_wrongly"] == 0


def test_2_4_flip_rates_are_reported_against_the_relevant_denominator():
    """hidden_wrongly is a share of ads that SHOULD be accessible, not of all
    ads — otherwise it looks tiny simply because most ads are Norwegian."""
    g = [gold(1, "either_norwegian_or_english"), gold(2, "professional"),
         gold(3, "professional"), gold(4, "professional")]
    p = {"u1": pred("professional"), "u2": pred("professional"),
         "u3": pred("professional"), "u4": pred("professional")}
    a = scoring.score(p, g)["accessibility"]
    assert a["hidden_wrongly"] == 1
    assert a["hidden_wrongly_rate"] == 1.0, "1 of 1 accessible ads was hidden"


# ── 3 · evidence spans ──────────────────────────────────────────────────────

def test_3_1_an_exactly_matching_span_scores():
    g = [gold(1, "professional", span="Gode norskkunnskaper.")]
    p = {"u1": pred("professional", span="Gode norskkunnskaper.")}
    s = scoring.score(p, g)["evidence"]
    assert s["exact"] == 1 and s["n_expected"] == 1


def test_3_2_a_different_but_present_span_is_scored_separately_from_a_miss():
    """The model may quote a different valid sentence. That is not the same
    failure as quoting nothing, and pooling them hides which is happening."""
    g = [gold(1, "professional", span="Gode norskkunnskaper.")]
    p = {"u1": pred("professional", span="Du må beherske norsk godt.")}
    s = scoring.score(p, g)["evidence"]
    assert s["exact"] == 0 and s["different"] == 1 and s["missing"] == 0


def test_3_3_a_missing_span_where_one_was_expected_is_a_miss():
    g = [gold(1, "professional", span="Gode norskkunnskaper.")]
    p = {"u1": pred("professional")}
    assert scoring.score(p, g)["evidence"]["missing"] == 1


def test_3_4_a_span_offered_where_NONE_was_expected_is_spurious():
    """The silent ads are 12 of 44. A model that invents evidence for them is
    failing in the way the verbatim validator exists to catch."""
    g = [gold(1, "unstated", span=None)]
    p = {"u1": pred("unstated", span="Vi søker en medarbeider.")}
    assert scoring.score(p, g)["evidence"]["spurious"] == 1


# ── 4 · the other model-produced fields ─────────────────────────────────────

def test_4_1_working_language_and_authorisation_are_scored():
    g = [gold(1, "unstated", auth="Norsk autorisasjon som sykepleier"),
         gold(2, "explicitly_not_required", working="english")]
    p = {"u1": pred("unstated", auth="Norsk autorisasjon som sykepleier"),
         "u2": pred("explicitly_not_required", working="norwegian")}
    r = scoring.score(p, g)
    assert r["authorisation"]["correct"] == 1
    assert r["working_language"]["correct"] == 1   # u1 matches "unstated"; u2 does not


def test_4_2_authorisation_is_scored_on_presence_not_exact_string():
    """Requiring an exact match would punish "norsk autorisasjon" against
    "Norsk autorisasjon som sykepleier" — the same finding, differently quoted."""
    g = [gold(1, "unstated", auth="Norsk autorisasjon som sykepleier")]
    p = {"u1": pred("unstated", auth="norsk autorisasjon")}
    assert scoring.score(p, g)["authorisation"]["correct"] == 1


# ── 5 · bookkeeping ─────────────────────────────────────────────────────────

def test_5_1_golden_ads_with_no_prediction_are_reported_not_ignored():
    """Silently dropping them inflates every accuracy figure."""
    g = [gold(1, "unstated"), gold(2, "professional")]
    r = scoring.score({"u1": pred("unstated")}, g)
    assert r["unscored"] == ["u2"]
    assert r["n_scored"] == 1 and r["n_golden"] == 2


def test_5_2_an_unscored_ad_does_not_count_as_correct_or_incorrect():
    g = [gold(1, "professional"), gold(2, "professional")]
    r = scoring.score({"u1": pred("professional")}, g)
    assert r["per_level"]["professional"] == {"n": 1, "correct": 1, "accuracy": 1.0}


def test_5_3_predictions_not_in_the_golden_set_are_ignored_with_a_count():
    g = [gold(1, "unstated")]
    r = scoring.score({"u1": pred("unstated"), "stranger": pred("fluent")}, g)
    assert r["n_scored"] == 1 and r["extra_predictions"] == 1


def test_5_4_an_empty_prediction_set_does_not_divide_by_zero():
    r = scoring.score({}, [gold(1, "unstated")])
    assert r["n_scored"] == 0 and r["pooled_accuracy"] is None
    assert r["accessibility"]["hidden_wrongly_rate"] is None


# ── 6 · the mapping itself, not just the harness ────────────────────────────

def test_6_1_the_accessibility_sets_are_frozen_by_value():
    """Every flip assertion runs both sides through the SAME mapping, so if
    score() imports the production function the suite cannot detect an error IN
    that mapping. Moving `conversational` into ACCESSIBLE_LEVELS would leave
    every other test green while the product wrongly shows every conversational
    ad. Pin the membership literally."""
    from finn_smart_search.understanding.census_prompt import (ACCESSIBLE_LEVELS,
                                                               BLOCKING_LEVELS)
    assert ACCESSIBLE_LEVELS == {"either_norwegian_or_english",
                                 "explicitly_not_required", "desirable"}
    assert BLOCKING_LEVELS == {"certified", "fluent", "professional",
                               "conversational", "scandinavian_accepted"}


@pytest.mark.parametrize("working,doc_lang,expected", [
    ("english", "no", True), ("both", "no", True),
    ("norwegian", "en", False),        # stated language beats the ad's own
    ("scandinavian", "en", False),
    ("unstated", "no", False), ("unstated", "en", True),
    ("unstated", "mixed", True), ("unstated", "other", True),
    ("unstated", "unknown", True),
])
def test_6_2_unstated_resolves_through_every_branch(working, doc_lang, expected):
    """12 of 44 goldens are `unstated`, and only one branch was covered."""
    from finn_smart_search.understanding.census_prompt import derive_english_accessible
    assert derive_english_accessible(
        {"norwegian_requirement_level": "unstated",
         "stated_working_language": working}, doc_lang) is expected


# ── 7 · trivial baselines must score badly ──────────────────────────────────

def test_7_1_an_always_blocking_predictor_is_not_flattered():
    """`shown_wrongly == 0` is trivially achievable by hiding everything — and
    hiding is the direction framed as costly. If the report does not make that
    obvious, the metric rewards the exact failure the project exists to fix."""
    g = [gold(1, "either_norwegian_or_english"), gold(2, "desirable"),
         gold(3, "professional")]
    r = scoring.score({f"u{i}": pred("professional") for i in (1, 2, 3)}, g)
    assert r["accessibility"]["shown_wrongly"] == 0
    assert r["accessibility"]["hidden_wrongly"] == 2, "both accessible ads hidden"


def test_7_2_an_always_unstated_predictor_is_not_flattered_on_working_language():
    """"unstated" is the overwhelmingly common value, so plain accuracy on this
    field rewards a constant predictor. Score it where it MATTERS: the ads whose
    level is `unstated`, where english/both is what flips an ad to shown."""
    g = [gold(1, "unstated", working="english"), gold(2, "unstated", working="both"),
         gold(3, "unstated", working="unstated")]
    r = scoring.score({f"u{i}": pred("unstated", working="unstated") for i in (1, 2, 3)}, g)
    assert r["working_language"]["non_default_recall"] == 0.0, (
        "a constant 'unstated' predictor must score ZERO on the values that matter")


# ── 8 · the thesis ──────────────────────────────────────────────────────────

def test_8_1_disjunction_recall_is_reported_on_its_own():
    """The 304-ad correction is what the project exists for. It must not be
    averaged into a pooled number."""
    g = [gold(1, "either_norwegian_or_english"), gold(2, "either_norwegian_or_english")]
    r = scoring.score({"u1": pred("either_norwegian_or_english"),
                       "u2": pred("professional")}, g)
    assert r["per_level"]["either_norwegian_or_english"]["accuracy"] == 0.5


# ── 9 · the taxonomy gap (#15) ──────────────────────────────────────────────

def test_9_1_derived_and_annotated_disagreement_is_reported_not_scored():
    """Golden #15 is `explicitly_not_required` yet a human judged it NOT
    accessible — the ad addresses Polish speakers and never mentions English.

    Scoring against the annotation makes that ad unwinnable: a CORRECT level is
    charged a permanent flip, the metric can never reach zero, and the cheapest
    way to green it is to mislabel the level. So score against the DERIVED
    value, and surface the disagreement as a first-class number: it is evidence
    the nine-level enum cannot express 'is English mentioned'."""
    g = [gold(1, "explicitly_not_required")]
    g[0]["annotated_accessible"] = False          # the human override
    r = scoring.score({"u1": pred("explicitly_not_required")}, g)
    assert r["accessibility"]["hidden_wrongly"] == 0, "a correct level is never a flip"
    assert r["taxonomy_gap"] == ["u1"]


# ── 10 · rates must carry their denominator and uncertainty ─────────────────

def test_10_1_rate_denominator_is_accessible_ads_not_all_ads():
    """With 1 accessible ad and 1 flip, 'expected-accessible' and 'all ads' both
    give 1.0. Two accessible ads and one flip separates them: 0.5 vs 0.25."""
    g = [gold(1, "either_norwegian_or_english"), gold(2, "desirable"),
         gold(3, "professional"), gold(4, "professional")]
    p = {"u1": pred("professional"), "u2": pred("desirable"),
         "u3": pred("professional"), "u4": pred("professional")}
    a = scoring.score(p, g)["accessibility"]
    assert a["hidden_wrongly"] == 1 and a["n_accessible"] == 2
    assert a["hidden_wrongly_rate"] == 0.5


def test_10_2_every_rate_ships_with_its_counts_and_an_interval():
    """At n=13 accessible ads, 1/13 has a 95% Wilson interval of roughly
    [0.01, 0.33]. A bare percentage invites false precision on a sample whose
    level mix is a 200x-distorted sampling artifact."""
    g = [gold(i, "either_norwegian_or_english") for i in range(1, 14)]
    p = {f"u{i}": pred("either_norwegian_or_english") for i in range(1, 14)}
    p["u1"] = pred("professional")
    a = scoring.score(p, g)["accessibility"]
    assert a["hidden_wrongly"] == 1 and a["n_accessible"] == 13
    lo, hi = a["hidden_wrongly_ci95"]
    assert lo < 0.077 < hi and hi > 0.25, "interval must convey how little n=13 pins down"


# ── 11 · fabricated spans ───────────────────────────────────────────────────

def test_11_1_a_span_not_present_in_the_ad_is_fabricated_not_merely_different():
    """`different` conflates three failures: another valid sentence, an
    irrelevant one, and an invention. Only the last is a falsification, and it
    is the one the verbatim validator exists to catch."""
    g = [gold(1, "professional", span="Gode norskkunnskaper.")]
    g[0]["source_text"] = "Vi søker en medarbeider. Gode norskkunnskaper. Oppstart snarest."
    p = {"u1": pred("professional", span="Our working language is English.")}
    e = scoring.score(p, g)["evidence"]
    assert e["fabricated"] == 1 and e["different"] == 0


def test_11_2_a_different_but_REAL_sentence_is_different_not_fabricated():
    g = [gold(1, "professional", span="Gode norskkunnskaper.")]
    g[0]["source_text"] = "Vi søker en medarbeider. Gode norskkunnskaper. Du må beherske norsk."
    p = {"u1": pred("professional", span="Du må beherske norsk.")}
    e = scoring.score(p, g)["evidence"]
    assert e["different"] == 1 and e["fabricated"] == 0


# ── 12 · integrity and robustness ───────────────────────────────────────────

def test_12_1_coverage_is_reported_beside_every_rate():
    """Unscored ads are EXCLUDED from denominators, so refusing the hard ads
    improves every rate. The coverage figure must travel with the numbers."""
    g = [gold(1, "professional"), gold(2, "either_norwegian_or_english")]
    r = scoring.score({"u1": pred("professional")}, g)
    assert r["coverage"] == 0.5


def test_12_2_an_unknown_predicted_level_is_an_error_not_silent_coercion(caplog):
    """Silently coercing an invalid level to blocking makes a broken extractor
    look merely conservative."""
    g = [gold(1, "professional")]
    r = scoring.score({"u1": pred("not_a_real_level")}, g)
    assert r["invalid_levels"] == 1
    assert r["per_level"]["professional"]["correct"] == 0


def test_12_3_authorisation_does_not_match_a_different_profession():
    """Substring matching must not make 'Norsk autorisasjon som lege' correct
    against '...som sykepleier'."""
    g = [gold(1, "unstated", auth="Norsk autorisasjon som sykepleier")]
    p = {"u1": pred("unstated", auth="Norsk autorisasjon som lege")}
    assert scoring.score(p, g)["authorisation"]["correct"] == 0


def test_12_4_spurious_authorisation_is_counted():
    g = [gold(1, "professional", auth=None)]
    p = {"u1": pred("professional", auth="Norsk autorisasjon")}
    assert scoring.score(p, g)["authorisation"]["spurious"] == 1


# ── 13 · gaps found by mutation analysis ────────────────────────────────────

@pytest.mark.parametrize("working,doc_lang,wrong_pred,expect_hidden,expect_shown", [
    ("unstated", "mixed", "fluent", 1, 0),       # accessible -> predicted blocking
    ("unstated", "other", "certified", 1, 0),
    ("norwegian", "en", "desirable", 0, 1),      # blocking -> predicted accessible
    ("english", "no", "professional", 1, 0),
])
def test_13_1_unstated_flips_are_detected_with_a_WRONG_prediction(
        working, doc_lang, wrong_pred, expect_hidden, expect_shown):
    """test 2.3 predicted `unstated` CORRECTLY, so both sides ran the same
    derivation and every mismatch cancelled — the rule was effectively
    untested. `unstated` is the largest golden bucket (12 of 44) and the one
    the extractor gets wrong most."""
    g = [gold(1, "unstated", working=working, doc_lang=doc_lang)]
    a = scoring.score({"u1": pred(wrong_pred)}, g)["accessibility"]
    assert (a["hidden_wrongly"], a["shown_wrongly"]) == (expect_hidden, expect_shown)


def test_13_2_the_confusion_matrix_includes_the_diagonal():
    """Dropping correct pairs makes per-level recall computed from the matrix
    garbage, and shows a model that appears to confuse everything."""
    g = [gold(1, "either_norwegian_or_english"), gold(2, "professional")]
    r = scoring.score({"u1": pred("either_norwegian_or_english"),
                       "u2": pred("fluent")}, g)
    c = r["confusion"]
    assert c[("either_norwegian_or_english", "either_norwegian_or_english")] == 1
    assert c[("professional", "fluent")] == 1
    assert sum(c.values()) == r["n_scored"], "cells must account for every scored ad"


def test_13_3_authorisation_reports_its_denominator_and_the_null_case():
    """With no `n`, `authorisation.correct` swings between ~4 and ~40 depending
    on whether None/None counts, and nobody can tell which scale they are
    reading."""
    g = [gold(1, "unstated", auth="Norsk autorisasjon som sykepleier"),
         gold(2, "professional", auth=None)]
    p = {"u1": pred("unstated", auth="Norsk autorisasjon som sykepleier"),
         "u2": pred("professional", auth=None)}
    a = scoring.score(p, g)["authorisation"]
    assert a["n_expected"] == 1, "the denominator is ads that SHOULD have one"
    assert a["correct"] == 1 and a["spurious"] == 0


@pytest.mark.parametrize("variant", [
    "Gode norskkunnskaper.",           # identical
    "gode norskkunnskaper.",           # case
    "Gode  norskkunnskaper.",          # collapsed inner whitespace
    "Gode norskkunnskaper.\n",         # trailing newline
    "Gode norskkunnskaper.",      # non-breaking space
])
def test_13_4_span_matching_is_normalised(variant):
    """`exact` is the number people will cite as extraction quality. A trailing
    newline must not halve it — the same normaliser the validator uses."""
    g = [gold(1, "professional", span="Gode norskkunnskaper.")]
    g[0]["source_text"] = "Vi søker. Gode norskkunnskaper. Oppstart snarest."
    assert scoring.score({"u1": pred("professional", span=variant)},
                         g)["evidence"]["exact"] == 1


def test_13_5_a_matching_span_anywhere_in_the_list_counts():
    """Real extractors return several spans; comparing only the first would
    misreport a correct extraction as `different`."""
    g = [gold(1, "professional", span="Gode norskkunnskaper.")]
    g[0]["source_text"] = "Du må beherske norsk. Gode norskkunnskaper."
    p = {"u1": {"norwegian_requirement_level": "professional",
                "evidence_spans": [{"span": "Du må beherske norsk.", "section_language": "no"},
                                   {"span": "Gode norskkunnskaper.", "section_language": "no"}],
                "stated_working_language": "unstated", "authorisation_required": None}}
    assert scoring.score(p, g)["evidence"]["exact"] == 1


def test_13_6_rates_do_not_dilute_when_hard_ads_go_unscored():
    """An extractor that refuses the hard ads must not have its harm rate
    diluted toward zero. Both denominators are GOLDEN, and coverage is
    reported so refusal is visible rather than rewarded."""
    g = [gold(1, "either_norwegian_or_english"), gold(2, "professional")]
    r = scoring.score({"u1": pred("professional")}, g)          # u2 unscored
    a = r["accessibility"]
    assert a["hidden_wrongly"] == 1 and a["n_accessible"] == 1
    assert a["hidden_wrongly_rate"] == 1.0
    assert r["coverage"] == 0.5
    assert r["pooled_accuracy"] == 0.0, "pooled is over SCORED ads, and coverage sits beside it"


def test_14_1_an_invalid_predicted_level_must_not_count_as_accessible():
    """test_12_2 checked that an invalid level is COUNTED, but not what it does
    to accessibility. Treating it as accessible would wrongly SHOW ads on the
    strength of a level that does not exist — a broken extractor reading as
    permissive rather than as broken."""
    g = [gold(1, "professional")]                      # expected: blocking
    r = scoring.score({"u1": pred("not_a_real_level")}, g)
    a = r["accessibility"]
    assert r["invalid_levels"] == 1
    assert a["shown_wrongly"] == 0, "an invalid level must never derive as accessible"
    assert a["hidden_wrongly"] == 0


def test_14_2_an_invalid_level_on_an_ACCESSIBLE_ad_is_a_hidden_flip():
    g = [gold(1, "either_norwegian_or_english")]        # expected: accessible
    a = scoring.score({"u1": pred("garbage")}, g)["accessibility"]
    assert a["hidden_wrongly"] == 1


def test_14_3_a_string_that_is_not_an_authorisation_does_not_match():
    """Presence-matching must key on the authorisation itself. Without that, any
    non-null string scores correct — including the model echoing a job title or
    a language requirement into the field."""
    g = [gold(1, "unstated", auth="Norsk autorisasjon som sykepleier")]
    for wrong in ("Gode norskkunnskaper", "sykepleier", "Bachelor i sykepleie", "ja"):
        r = scoring.score({"u1": pred("unstated", auth=wrong)}, g)
        assert r["authorisation"]["correct"] == 0, f"{wrong!r} is not an authorisation"


# ── 9b · the golden FILE, not a fixture the test builds itself ───────────────
#
# test_9_1 passes and proves nothing about the data. It constructs a golden ad
# and sets `annotated_accessible` on it, so the mechanism is exercised while the
# real eval/golden_set.json — which recorded the override as prose in
# `accessibility_note` — never reached the check at all. Measured: 0 of 44 ads
# carry `annotated_accessible`, so `taxonomy_gap` was unreachable in production.
#
# That is the same failure shape as the span tests that used single-line prose
# and the corpus check that passed blocks with their bullet glyph attached: the
# test built its own input, so it agreed with itself.
#
# These tests read the file. The closed-vocabulary one is the root-cause fix:
# the bug was a KEY NAME that nothing validated, and any future typo in a field
# name fails here rather than silently disabling a metric.

import json as _json
import pathlib as _pathlib

GOLDEN_PATH = _pathlib.Path(__file__).resolve().parents[2] / "eval" / "golden_set.json"

# Every key the harness or a human is allowed to use. Adding one is a
# deliberate act that shows up in this list and in review.
GOLDEN_KEYS = {
    "n", "uuid", "title", "stratum", "expected", "note", "doc_lang",
    "derived_accessible",        # documentation of what the derivation yields
    "annotated_accessible",      # BOOLEAN human override, read by scoring
    "accessibility_note",        # prose explaining an override
}
# Any facet a golden ad may pin. These are real fields of the tool schema, not
# free-form annotation: #3 pins `conflicting_statements`. The point of the closed
# set is that adding one is visible here, not that the list is short.
EXPECTED_KEYS = {"norwegian_requirement_level", "evidence_span",
                 "stated_working_language", "authorisation_required",
                 "conflicting_statements"}


def _golden_file():
    return _json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_9b_1_golden_set_uses_no_unknown_keys():
    """ROOT CAUSE. `accessibility_note` was written where scoring reads
    `annotated_accessible`; both are plausible names and nothing objected, so a
    documented metric quietly measured nothing for the whole project.

    A closed vocabulary makes the next such typo a failing test instead of a
    silent hole."""
    for g in _golden_file():
        unknown = set(g) - GOLDEN_KEYS
        assert not unknown, f"golden #{g['n']} has unknown key(s) {unknown}"
        unknown_exp = set(g["expected"]) - EXPECTED_KEYS
        assert not unknown_exp, f"golden #{g['n']} expected has {unknown_exp}"


def test_9b_2_a_prose_override_must_carry_the_boolean_scoring_reads():
    """The specific bug, as a permanent guard. Prose alone cannot be compared to
    a derived boolean, so an `accessibility_note` without `annotated_accessible`
    is an override that no metric can see."""
    for g in _golden_file():
        if "accessibility_note" in g:
            assert "annotated_accessible" in g, (
                f"golden #{g['n']} explains an accessibility override in prose but "
                "omits `annotated_accessible`, which is the field scoring reads — "
                "the override would be invisible to taxonomy_gap"
            )
            assert isinstance(g["annotated_accessible"], bool)


def test_9b_3_stored_derived_accessible_matches_the_derivation():
    """`derived_accessible` is present on all 44 ads and, before this test, was
    read by nothing. A documentation field no test checks is free to drift away
    from the code it documents — and it is the field a reader would trust when
    auditing the set by hand."""
    for g in _golden_file():
        e = g["expected"]
        want = scoring._accessible(e["norwegian_requirement_level"],
                                   e["stated_working_language"],
                                   g.get("doc_lang", "no"))
        assert g["derived_accessible"] == want, (
            f"golden #{g['n']}: stored derived_accessible={g['derived_accessible']} "
            f"but derive() gives {want} for level "
            f"{e['norwegian_requirement_level']!r}"
        )


def test_9b_4_taxonomy_gap_runs_against_the_real_file():
    """Exercises the check on the real data rather than a built fixture. It is
    allowed to be empty — after #15's level was revised to `certified` the
    derivation and the human reading AGREE, so the set legitimately contains no
    taxonomy-gap ad and this is a forward guard.

    What it pins is that the check RUNS and returns the documented type. The
    previous state was indistinguishable from that, which is why the closed
    vocabulary above is the real protection."""
    golden = _golden_file()
    preds = {g["uuid"]: pred(g["expected"]["norwegian_requirement_level"])
             for g in golden}
    r = scoring.score(preds, golden)
    assert isinstance(r["taxonomy_gap"], list)
    # every listed uuid must actually carry an override that disagrees
    for uuid in r["taxonomy_gap"]:
        g = next(x for x in golden if x["uuid"] == uuid)
        assert "annotated_accessible" in g
