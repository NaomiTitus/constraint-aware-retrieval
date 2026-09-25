"""Scoring against REAL predictions and REAL ad text (STANDARDS.md § 3.0).

WHY THIS FILE EXISTS. Every test in test_scoring.py builds its golden records
with `gold()` and its predictions with `pred()`. Those helpers supply 4 of the
16 keys a real prediction carries, and one of them — `test_9_1` — SET the very
key it then asserted scoring reads, which is how `taxonomy_gap` came to measure
nothing for the whole project while its test passed.

So `score()` had never once been run against a production prediction inside the
suite. These tests do that. All 44 golden uuids have a real record in
`ad_facets`, so the grounding is exact rather than a sample.

Numbers here are a REGRESSION BASELINE measured 2026-09-25 under census-v6.
They change when the extractor changes, which is the point — regenerate them
deliberately, never to make a red test green.
"""
import json

import pytest

from tests.conftest import golden_file, open_corpus, requires_corpus

pytestmark = [pytest.mark.unit, requires_corpus]

# Measured over all 44 golden ads with real predictions. Regenerate ONLY with a
# recorded reason.
#
#   2026-09-25  census-v6: pooled 0.909, hidden 0, exact 20/different 12/missing 0
#   2026-09-25  census-v8: pooled 0.864, hidden 1, exact 19/different 12/missing 1
#
# WHY THE v8 NUMBERS ARE ACCEPTED DESPITE BEING LOWER. The difference is TWO ADS
# (40/44 vs 38/44) on a set that has driven five prompt iterations, so it is noise
# on an over-fitted instrument, not a measurement. Specifically:
#
#   the single hidden-wrongly is #7, where the model quoted "arbeidsspåket" for
#   "arbeidsspråket" — one missing letter. Its LEVEL (desirable) is correct; the
#   validator rejected the quote, which is the designed trade ("a demoted record
#   beats a falsified one"). It made the identical slip under v7 and v8, so the
#   explicit character-for-character instruction did not fix it and no prompt text
#   will — it is a transcription limit, recorded in the backlog.
#
#   every other miss leaves ACCESSIBILITY UNCHANGED: #10/#33/#44 unstated ->
#   professional (both blocking), #20 explicitly_not_required -> desirable (both
#   accessible). The per-level dips are within-class, not flips.
#
# v8 also GAINED working_language recall 3/4 -> 4/4, and carries rules the golden
# set cannot measure at all: the documentation-language clause (135 corpus ads),
# English surface forms (103), three-way comma disjunctions (104).
#
# The unbiased comparison is the held-out probes, not this set.
BASELINE = {
    "pooled_accuracy": 0.864,
    "coverage": 1.0,
    "invalid_levels": 0,
    "hidden_wrongly": 1,
    "shown_wrongly": 1,
    "n_blocking": 32,
    "evidence": {"n_expected": 32, "exact": 19, "different": 12,
                 "missing": 1, "spurious": 3, "fabricated": 0},
    "authorisation": {"n_expected": 5, "correct": 5, "spurious": 1},
    "working_language": {"non_default_total": 4, "non_default_correct": 4},
}


@pytest.fixture(scope="module")
def real_inputs(request):
    """Real predictions and real ad text, assembled the way run_pilot.py does.

    run_pilot.py injects `source_text` from the corpus at runtime — it is NOT a
    field of golden_set.json, and must not become one: duplicating 44 ad bodies
    into the eval set would put derived data in a frozen artifact. Assembling it
    here the same way is what makes this test a check on the real call shape.
    """
    con = open_corpus()
    golden, preds = [], {}
    for g in golden_file():
        row = con.execute("""SELECT a.description_text, l.doc_lang, f.facets
                             FROM ads a
                             JOIN ad_language l USING (uuid)
                             JOIN ad_facets  f USING (uuid)
                             WHERE a.uuid = ?""", [g["uuid"]]).fetchone()
        if row is None:
            pytest.skip("corpus lacks facets for the golden set; run the pilot first")
        text, doc_lang, facets = row
        golden.append(dict(g, doc_lang=doc_lang, source_text=text))
        preds[g["uuid"]] = json.loads(facets) if isinstance(facets, str) else facets
    return preds, golden


def test_score_runs_against_real_16_key_predictions(real_inputs):
    """`pred()` supplies 4 of 16 keys. Nothing had ever handed `score()` a real
    facet dict, so any shape assumption between the two was unexercised."""
    preds, golden = real_inputs
    assert len(preds) == 44
    assert all(len(p) >= 15 for p in preds.values()), \
        "real predictions carry the full facet set; if not, the corpus is stale"
    r = scoring_score(preds, golden)
    assert r["coverage"] == BASELINE["coverage"]
    assert r["invalid_levels"] == BASELINE["invalid_levels"]
    assert round(r["pooled_accuracy"], 3) == BASELINE["pooled_accuracy"]


def test_accessibility_baseline_on_real_predictions(real_inputs):
    preds, golden = real_inputs
    a = scoring_score(preds, golden)["accessibility"]
    assert a["hidden_wrongly"] == BASELINE["hidden_wrongly"]
    assert a["shown_wrongly"] == BASELINE["shown_wrongly"]
    assert a["n_blocking"] == BASELINE["n_blocking"]


def test_fabrication_checking_actually_RAN(real_inputs):
    """THE point of this file's existence.

    `fabricated: 0` is ambiguous: it means both "checked every span against the
    ad, found none invented" and "no source text was supplied, so nothing was
    checked". Those are opposite facts reported by the same number — the exact
    ambiguity that let `taxonomy_gap` report [] forever.

    `score()` now reports `fabrication_checked`, and this asserts it is True on
    the real call shape. A caller that stops injecting source_text makes this
    red instead of silently reporting a clean bill of health."""
    preds, golden = real_inputs
    r = scoring_score(preds, golden)
    assert r["evidence"]["fabrication_checked"] is True, \
        "fabrication was NOT verified; `fabricated: 0` would be meaningless"
    assert r["evidence"]["fabricated"] == BASELINE["evidence"]["fabricated"]


def test_fabrication_checked_is_false_when_no_source_text_is_supplied(real_inputs):
    """The other half. Without source text the metric must say so rather than
    report 0 and look clean."""
    preds, golden = real_inputs
    stripped = [{k: v for k, v in g.items() if k != "source_text"} for g in golden]
    r = scoring_score(preds, stripped)
    assert r["evidence"]["fabrication_checked"] is False
    assert r["evidence"]["fabricated"] is None, \
        "an unchecked count must not be reported as 0"


def test_evidence_span_agreement_is_broken_down_not_just_exact(real_inputs):
    """`exact` alone is misleading and would have gone into the README as
    extraction quality.

    Measured over the 32 golden spans (census-v8): 19 strict-equal, 4 differ ONLY
    by trailing punctuation, 8 are a containment (the model quoted a superset or
    subset sentence), 1 missing, and ZERO are a genuinely different sentence. So
    `exact` reports 59% while substantive agreement is 31 of 32 — the gap is
    punctuation and sentence boundaries, not wrong quotes.

    `different` therefore means "boundary mismatch" here, not "quoted something
    else", and the breakdown has to travel with the number."""
    preds, golden = real_inputs
    b = scoring_score(preds, golden)["evidence"]["agreement"]
    # census-v8 measured: 19 strict, 4 trailing-punct-only, 8 containment, 0 disjoint
    assert b["strict"] == 19
    assert b["trailing_punct_only"] == 4
    assert b["containment"] == 8
    assert b["disjoint"] == 0, \
        "a genuinely different sentence appeared; `different` now means what its name says"
    assert b["strict"] + b["trailing_punct_only"] + b["containment"] + b["disjoint"] == 31


def test_derive_agrees_with_the_stored_english_accessible(real_inputs):
    """Production computes english_accessible at extraction time from the FULL
    facet dict; scoring rebuilds a 2-key dict and re-derives. Nothing pinned
    that the two agree. Measured: 169 of 169 rows agree."""
    from finn_smart_search.understanding.census_prompt import derive_english_accessible
    con = open_corpus()
    rows = con.execute("""SELECT f.facets, f.english_accessible, l.doc_lang
                          FROM ad_facets f JOIN ad_language l USING (uuid)""").fetchall()
    assert rows, "no facets in the corpus"
    bad = []
    for facets, stored, doc_lang in rows:
        d = json.loads(facets) if isinstance(facets, str) else facets
        if derive_english_accessible(d, doc_lang) != stored:
            bad.append(d.get("norwegian_requirement_level"))
    assert bad == [], f"derive() disagrees with stored english_accessible for {bad}"


def test_golden_doc_lang_matches_the_detector(real_inputs):
    """Golden `doc_lang` is hand-recorded; `ad_language.doc_lang` is produced by
    langid. A divergence means the golden set describes an ad differently from
    the pipeline, and doc_lang feeds the accessibility derivation.

    Measured 2026-09-25: #37 and #38 are recorded `en` while the detector says
    `mixed` — both are genuinely bilingual ads carrying parallel Norwegian and
    English requirement lines. Pinned as a KNOWN divergence so it is visible
    rather than discovered again."""
    con = open_corpus()
    KNOWN = {37: ("en", "mixed"), 38: ("en", "mixed")}
    found = {}
    for g in golden_file():
        det = con.execute("SELECT doc_lang FROM ad_language WHERE uuid=?",
                          [g["uuid"]]).fetchone()[0]
        if det != g["doc_lang"]:
            found[g["n"]] = (g["doc_lang"], det)
    assert found == KNOWN, (
        f"doc_lang divergence changed: {found} vs known {KNOWN}. "
        "Golden doc_lang is hand-recorded and langid is the pipeline's answer; "
        "reconcile deliberately."
    )


def scoring_score(preds, golden):
    from finn_smart_search.eval import scoring
    return scoring.score(preds, golden)
