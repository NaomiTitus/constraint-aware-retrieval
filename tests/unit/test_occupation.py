"""Occupation as a predicate (D18). Pure unit tests — the gazetteer is injected.

The two properties that matter most are non-effects, and both mirror
`constraints.py`: a stage that fires when it has nothing to say destroys recall
silently. So an unstated occupation must leave the ranking bit-identical, and an
occupation the gazetteer cannot RESOLVE must do the same rather than zeroing every
ad — those two states are different and conflating them would hide a gazetteer gap
as a ranking result.
"""
from __future__ import annotations

import pytest

from finn_smart_search.retrieval.occupation import (
    OccupationConstraint, OccupationGazetteer, PREFIX_WEIGHTS, apply,
    from_gold_parse, normalise, prefix_proximity)

# (styrk_code, norwegian label, english label) — one row per ad, as the corpus
# gives them, so label weights are ad counts. Shaped after the real data,
# including its noise: `sykepleier` genuinely maps to three codes in NAV's feed.
ROWS = (
    [("2223", "sykepleier", "nurse responsible for general care")] * 580
    + [("3412", "sykepleier", "nurse responsible for general care")] * 12
    + [("5311", "sykepleier", "nurse responsible for general care")] * 3
    + [("2221", "spesialsykepleier", "specialist nurse")] * 40
    + [("2341", "grunnskolelærer", "primary school teacher")] * 200
    + [("2342", "førskolelærer", "early years teacher")] * 365
    + [("7115", "tømrer", "carpenter")] * 94
    + [("2152", "programvareutvikler", "software developer")] * 120
)


@pytest.fixture(scope="module")
def gaz():
    return OccupationGazetteer.build_from_rows(ROWS)


# ── normalisation ─────────────────────────────────────────────────────────────

def test_normalisation_folds_case_and_punctuation():
    assert normalise("Lærer, spesialskole") == "lærer spesialskole"


def test_normalisation_keeps_norwegian_letters():
    assert normalise("Tømrer") == "tømrer"


# ── prefix proximity ──────────────────────────────────────────────────────────

def test_an_identical_code_is_full_proximity():
    assert prefix_proximity("2223", "2223") == 1.0


def test_a_shared_unit_group_scores_below_exact_but_well_above_zero():
    """2223 sykepleier vs 2221 nursing professionals — same unit group 222."""
    p = prefix_proximity("2223", "2221")
    assert 0.0 < p < 1.0
    assert p == PREFIX_WEIGHTS[3]


def test_related_teaching_codes_share_a_unit_group():
    """2341 grunnskolelærer vs 2342 førskolelærer — genuinely adjacent work."""
    assert prefix_proximity("2341", "2342") == PREFIX_WEIGHTS[3]


def test_an_unrelated_major_group_scores_exactly_zero():
    """7115 tømrer vs 2223 sykepleier. Exactly 0.0, not a small floor: a floor
    would let all 10,166 ads score something and make the predicate decorative."""
    assert prefix_proximity("7115", "2223") == 0.0


def test_codes_of_different_length_compare_over_the_shorter():
    assert prefix_proximity("222", "2223") == PREFIX_WEIGHTS[3]


def test_a_missing_code_scores_zero_rather_than_raising():
    assert prefix_proximity("", "2223") == 0.0
    assert prefix_proximity("2223", "") == 0.0


def test_proximity_is_monotone_in_shared_digits():
    base = "2223"
    seq = [prefix_proximity(base, c) for c in ("7115", "2115", "2215", "2221", "2223")]
    assert seq == sorted(seq), seq


# ── gazetteer resolution ──────────────────────────────────────────────────────

def test_a_norwegian_label_resolves_to_its_dominant_code(gaz):
    codes = gaz.resolve("sykepleier")
    assert max(codes, key=codes.get) == "2223"


def test_the_english_label_resolves_to_the_same_code(gaz):
    """THE PROPERTY THE WHOLE MODULE EXISTS FOR. `nurse` and `sykepleier` must
    reach the same STYRK code, because that is what fixes the cross-language
    failure D1 measured — without relying on a shared token appearing."""
    no = gaz.resolve("sykepleier")
    en = gaz.resolve("nurse responsible for general care")
    assert max(no, key=no.get) == max(en, key=en.get) == "2223"


def test_upstream_noise_codes_are_dropped_below_the_share_threshold(gaz):
    """NAV maps `sykepleier` to 5311 (teaching assistant) on 3 of 595 ads. Keeping
    it would make a nurse query score teaching-assistant work."""
    assert "5311" not in gaz.resolve("sykepleier")


def test_a_genuinely_ambiguous_code_is_kept(gaz):
    """3412 carries 12 of 595 — 2% — so it is dropped at the default, but a lower
    threshold must keep it. The knob exists because ambiguity and noise are not
    distinguishable from counts alone."""
    assert "3412" in gaz.resolve("sykepleier", min_share=0.0)


def test_a_partially_overlapping_phrase_resolves_by_token_overlap(gaz):
    """`backend developer` is not a corpus label; `software developer` is. Subset
    containment does NOT connect them — neither contains the other — so the
    fallback is Jaccard overlap: they share `developer` out of three tokens."""
    codes = gaz.resolve("backend developer")
    assert codes and max(codes, key=codes.get) == "2152"


def test_a_phrase_below_the_overlap_threshold_does_not_resolve(gaz):
    """The threshold is what stops one shared generic token from matching
    everything. A wrong code demotes every correct ad, so failing to resolve is
    the safer error."""
    assert gaz.resolve("backend developer", min_overlap=0.9) == {}


def test_token_set_matching_absorbs_word_order(gaz):
    """The Norwegian labels use a comma-qualifier shape (`lærer, spesialskole`),
    so a seeker writing the qualifier first must still resolve."""
    g = OccupationGazetteer.build_from_rows(
        [("2330", "lærer, spesialskole", "special education teacher")] * 10)
    assert g.resolve("spesialskole lærer") == g.resolve("lærer, spesialskole")


def test_an_unknown_phrase_resolves_to_nothing(gaz):
    """Empty, not a guess. A wrong code is worse than no code, because it would
    demote every correct ad."""
    assert gaz.resolve("kvantekryptograf") == {}


def test_resolution_weights_sum_to_at_most_one(gaz):
    assert sum(gaz.resolve("sykepleier").values()) <= 1.0 + 1e-9


# ── the constraint, and its two non-effects ───────────────────────────────────

def test_proximity_takes_the_best_code_not_the_average(gaz):
    """The codes are alternative readings of ONE phrase, so matching any of them
    well is a good match. Averaging would penalise a phrase NAV maps to several
    codes — punishing the seeker for upstream ambiguity."""
    c = OccupationConstraint("sykepleier", gaz.resolve("sykepleier", min_share=0.0))
    assert c.proximity("2223") > 0.9


def test_an_unrelated_ad_gets_zero_proximity(gaz):
    c = OccupationConstraint("sykepleier", gaz.resolve("sykepleier"))
    assert c.proximity("7115") == 0.0


def test_an_unstated_occupation_leaves_the_ranking_bit_identical():
    """The non-effect rule, same as `constraints.apply`. `None` means the seeker
    said nothing, and the stage must contribute exactly nothing."""
    scores = [0.9, 0.5, 0.31, 0.02]
    assert apply(scores, ["2223", "7115", "2341", "5223"], None) == scores


def test_an_unresolvable_occupation_also_leaves_the_ranking_unchanged(gaz):
    """DISTINCT FROM UNSTATED, and it must fail open. If the gazetteer cannot
    resolve the phrase, zeroing every ad would report a gazetteer gap as 'no
    relevant jobs exist' — the worst possible way to be wrong."""
    c = OccupationConstraint("kvantekryptograf", gaz.resolve("kvantekryptograf"))
    assert not c.resolved
    scores = [0.9, 0.5, 0.31]
    assert apply(scores, ["2223", "7115", "2341"], c) == scores


def test_lambda_zero_switches_the_stage_off(gaz):
    c = OccupationConstraint("sykepleier", gaz.resolve("sykepleier"))
    scores = [0.9, 0.5]
    assert apply(scores, ["2223", "7115"], c, lam=0.0) == scores


def test_applying_the_predicate_promotes_the_matching_occupation(gaz):
    """The whole point: a bricklayer ad that outscored a nurse ad lexically must
    end up below it once occupation is applied as a predicate."""
    c = OccupationConstraint("nurse", gaz.resolve("nurse responsible for general care"))
    scores = [10.0, 3.0]                      # bricklayer wins on text
    out = apply(scores, ["7115", "2223"], c)  # but is not a nurse
    assert out[1] > out[0]


def test_a_floor_keeps_non_matching_ads_alive(gaz):
    """Hard by default, softenable. Some seekers are open to adjacent work, and
    that is a product decision with a dial rather than a fixed rule."""
    c = OccupationConstraint("sykepleier", gaz.resolve("sykepleier"))
    hard = apply([1.0], ["7115"], c, floor=0.0)
    soft = apply([1.0], ["7115"], c, floor=0.25)
    assert hard[0] == 0.0
    assert soft[0] == pytest.approx(0.25)


def test_adjacent_occupations_survive_where_unrelated_ones_do_not(gaz):
    """Proximity, not a filter: `spesialsykepleier` (2221) must outrank a
    carpenter (7115) for a `sykepleier` query, while still sitting below an exact
    2223 match."""
    c = OccupationConstraint("sykepleier", gaz.resolve("sykepleier"))
    out = apply([1.0, 1.0, 1.0], ["2223", "2221", "7115"], c)
    assert out[0] > out[1] > out[2] == 0.0


# ── the bridge from a gold parse ──────────────────────────────────────────────

def test_from_gold_parse_returns_none_when_no_occupation_is_stated(gaz):
    class _P:
        constraints = ()
    assert from_gold_parse(_P(), gaz) is None


def test_from_gold_parse_resolves_a_stated_occupation(gaz):
    class _C:
        facet, value, priority = "occupation", "tømrer", "hard"
    class _P:
        constraints = (_C(),)
    c = from_gold_parse(_P(), gaz)
    assert c is not None and c.resolved
    assert max(c.codes, key=c.codes.get) == "7115"


# ════════════════════════════════════════════════════════════════════════════
# THE ESCO PATH — bilingual identity, compound affixes, candidate sets
# ════════════════════════════════════════════════════════════════════════════

from finn_smart_search.retrieval.occupation import (  # noqa: E402
    AdOccupation, EscoGazetteer, EscoOccupationConstraint, OCCUPATION_STOPWORDS,
    apply_esco, esco_from_gold_parse)

# Shaped after the real ESCO table: 1,242 occupations, every one with BOTH a
# Norwegian and an English label under one URI. The nurse family is included in
# full because it is the case that broke the first gazetteer.
ESCO = [
    ("u_nurse_general", "en", "nurse responsible for general care"),
    ("u_nurse_general", "no", "sykepleier"),
    ("u_nurse_assist", "en", "nurse assistant"),
    ("u_nurse_assist", "no", "pleiemedhjelper"),
    ("u_nurse_spec", "en", "specialist nurse"),
    ("u_nurse_spec", "no", "spesialsykepleier"),
    ("u_carpenter", "en", "carpenter"),
    ("u_carpenter", "no", "tømrer"),
    ("u_teacher_primary", "en", "primary school teacher"),
    ("u_teacher_primary", "no", "grunnskolelærer"),
    ("u_dev_software", "en", "software developer"),
    ("u_dev_software", "no", "programvareutvikler"),
    ("u_warehouse", "en", "warehouse operative"),
    ("u_warehouse", "no", "lagermedarbeider"),
    ("u_socialwork", "en", "social work assistant"),
    ("u_socialwork", "no", "sosialarbeiderassistent"),
]
# (esco_uri, styrk_code) per ad — the corpus co-occurrence that supplies hierarchy
AD_ROWS = (
    [("u_nurse_general", "2223")] * 660
    + [("u_nurse_assist", "5321")] * 554
    + [("u_nurse_spec", "2221")] * 40
    + [("u_carpenter", "7115")] * 93
    + [("u_teacher_primary", "2341")] * 119
    + [("u_dev_software", "2512")] * 23
    + [("u_warehouse", "4321")] * 200
    + [("u_socialwork", "5321")] * 82
)


@pytest.fixture(scope="module")
def esco():
    return EscoGazetteer.build(ESCO, AD_ROWS)


# ── bilingual identity: the property D18 was built for ────────────────────────

def test_the_two_languages_of_one_occupation_resolve_to_the_same_uri(esco):
    """THE POINT. `carpenter` and `tømrer` are two labels of one identity, so they
    must reach the same ads. The STYRK-label gazetteer never managed this."""
    assert esco.resolve("carpenter") == esco.resolve("tømrer")
    assert set(esco.resolve("carpenter")) == {"u_carpenter"}


def test_an_english_multiword_phrase_resolves_cross_language(esco):
    assert set(esco.resolve("primary school teacher")) == {"u_teacher_primary"}
    assert set(esco.resolve("grunnskolelærer")) == {"u_teacher_primary"}


# ── containment, not Jaccard ──────────────────────────────────────────────────

def test_a_specific_label_is_not_penalised_for_being_long(esco):
    """Symmetric Jaccard scored `nurse` against `nurse responsible for general care`
    at 0.25 and dropped it below threshold, which is how the first gazetteer
    resolved `nurse` to the assistant. Containment scores it 1.0."""
    assert "u_nurse_general" in esco.resolve("nurse")


# ── the Norwegian closed-compound rule ────────────────────────────────────────

def test_a_compound_head_reaches_its_compound(esco):
    """`utvikler` is not a TOKEN of `programvareutvikler`, it is a suffix of it.
    Whole-token comparison cannot connect them; this is why `backend-utvikler`
    failed to resolve at all before."""
    assert "u_dev_software" in esco.resolve("backend-utvikler")


def test_prefix_matching_is_deliberately_not_supported(esco):
    """`lager` is a PREFIX of `lagermedarbeider`, and prefix matching was removed
    anyway: on the real 2,470-label ESCO set it resolved `lager` to `anklager` and
    `kobber- og blikkenslager`, which merely end with those letters. The false
    matches outnumbered the true one, so an unresolved phrase is handed to D6's
    later stages instead."""
    assert esco.resolve("lager") == {}


def test_a_short_accidental_suffix_does_not_match(esco):
    """The 6-character floor. `lager` is a suffix of `anklager` (accuser) and that
    match is pure noise."""
    from finn_smart_search.retrieval.occupation import _token_match
    assert not _token_match("lager", "anklager")
    assert _token_match("utvikler", "programvareutvikler")
    assert _token_match("sykepleier", "spesialsykepleier")


def test_a_short_token_does_not_affix_match(esco):
    """The length floor is what stops `er` or `for` matching half the taxonomy."""
    from finn_smart_search.retrieval.occupation import _token_match
    assert not _token_match("er", "erfaring")
    assert _token_match("utvikler", "programvareutvikler")


# ── the occupational stoplist ─────────────────────────────────────────────────

def test_a_match_resting_only_on_generic_filler_does_not_resolve(esco):
    """`warehouse work` matched `social work assistant` on the word `work` and
    resolved to a social worker. A match carrying no occupational information is
    not a match."""
    codes = esco.resolve("warehouse work")
    assert "u_socialwork" not in codes
    assert "u_warehouse" in codes


def test_the_stoplist_covers_both_languages(esco):
    assert {"work", "job", "position"} <= OCCUPATION_STOPWORDS
    assert {"arbeid", "jobb", "stilling"} <= OCCUPATION_STOPWORDS


def test_a_phrase_of_only_stopwords_does_not_resolve(esco):
    assert esco.resolve("looking for a job position") == {}


# ── candidate sets, and no tiebreak ───────────────────────────────────────────

def test_a_genuinely_ambiguous_phrase_returns_EVERY_candidate(esco):
    """`nurse` is four ESCO occupations at different skill levels. The information
    that would choose between them is not in the word, so all of them come back."""
    codes = esco.resolve("nurse")
    assert {"u_nurse_general", "u_nurse_assist", "u_nurse_spec"} <= set(codes)


def test_candidate_weights_are_uniform_so_there_is_no_hidden_tiebreak(esco):
    """A weight difference here would be a tiebreak in disguise — and a tiebreak by
    ad count is exactly what resolved `nurse` to the 554-ad assistant over the
    660-ad nurse and got it wrong."""
    codes = esco.resolve("nurse")
    assert len(set(round(v, 12) for v in codes.values())) == 1


def test_ad_counts_are_available_for_reporting_but_do_not_affect_resolution(esco):
    assert esco.ads_per_uri["u_nurse_general"] == 660
    assert esco.ads_per_uri["u_nurse_assist"] == 554
    codes = esco.resolve("nurse")
    assert codes["u_nurse_assist"] == codes["u_nurse_general"]


def test_an_overly_ambiguous_phrase_fails_open_rather_than_guessing(esco):
    """Beyond `max_candidates` the phrase has not identified an occupation. Failing
    open leaves the lexical channel untouched; narrowing to an arbitrary subset
    would demote every ad outside a guess."""
    assert esco.resolve("nurse", max_candidates=1) == {}


def test_coverage_below_the_threshold_does_not_resolve(esco):
    assert esco.resolve("nurse", min_coverage=1.01) == {}


# ── proximity: identity first, hierarchy as fallback ──────────────────────────

def test_an_esco_identity_match_is_full_proximity(esco):
    c = EscoOccupationConstraint("tømrer", esco.resolve("tømrer"),
                                 esco.styrk_for(esco.resolve("tømrer")))
    assert c.proximity(AdOccupation("u_carpenter", "7115")) == 1.0


def test_the_compound_rule_makes_a_specialist_a_candidate_of_the_general_query(esco):
    """`sykepleier` is a suffix of `spesialsykepleier`, so a specialist nurse is a
    CANDIDATE for a nurse query — which is the right answer and not a bug. It is
    also why the next test must use a code outside the candidate set to exercise
    the STYRK fallback at all."""
    uris = esco.resolve("sykepleier")
    assert {"u_nurse_general", "u_nurse_spec"} <= set(uris)


def test_an_adjacent_styrk_code_still_scores_without_an_esco_match(esco):
    """ESCO URIs are flat, so identity alone cannot reach an ad whose URI the
    seeker's phrase never matched. 2224 shares the unit group 222 with the nurse
    codes but is not among them, so only the STYRK fallback can score it."""
    uris = esco.resolve("sykepleier")
    c = EscoOccupationConstraint("sykepleier", uris, esco.styrk_for(uris))
    p = c.proximity(AdOccupation("u_unknown_uri", "2224"))
    assert 0.0 < p < 1.0


def test_an_unrelated_ad_scores_zero_on_both_paths(esco):
    uris = esco.resolve("sykepleier")
    c = EscoOccupationConstraint("sykepleier", uris, esco.styrk_for(uris))
    assert c.proximity(AdOccupation("u_carpenter", "7115")) == 0.0


def test_an_ad_missing_its_esco_uri_still_scores_via_styrk(esco):
    """2.1% of the corpus has no ESCO tag but 100% has a STYRK code, so the
    fallback is load-bearing rather than theoretical."""
    uris = esco.resolve("sykepleier")
    c = EscoOccupationConstraint("sykepleier", uris, esco.styrk_for(uris))
    assert c.proximity(AdOccupation(None, "2223")) > 0.0


# ── the two non-effects, again, on this path ──────────────────────────────────

def test_an_unstated_occupation_leaves_the_esco_ranking_identical():
    scores = [0.9, 0.5, 0.31]
    ads = [AdOccupation("u_carpenter", "7115")] * 3
    assert apply_esco(scores, ads, None) == scores


def test_an_unresolved_occupation_leaves_the_esco_ranking_identical(esco):
    """D18's outcome measured why this matters: a predicate applied to one side of
    a paired comparison and not the other is a confound, so a gazetteer miss must
    change NOTHING rather than change a little."""
    c = EscoOccupationConstraint("kvantekryptograf", esco.resolve("kvantekryptograf"))
    assert not c.resolved
    scores = [0.9, 0.5]
    ads = [AdOccupation("u_carpenter", "7115"), AdOccupation("u_nurse_general", "2223")]
    assert apply_esco(scores, ads, c) == scores


def test_the_esco_predicate_promotes_the_right_occupation(esco):
    uris = esco.resolve("nurse")
    c = EscoOccupationConstraint("nurse", uris, esco.styrk_for(uris))
    out = apply_esco([10.0, 3.0],
                     [AdOccupation("u_carpenter", "7115"),
                      AdOccupation("u_nurse_general", "2223")], c)
    assert out[1] > out[0] == 0.0


def test_esco_from_gold_parse_returns_none_without_an_occupation(esco):
    class _P:
        constraints = ()
    assert esco_from_gold_parse(_P(), esco) is None


def test_esco_from_gold_parse_carries_both_uris_and_styrk(esco):
    class _C:
        facet, value, priority = "occupation", "carpenter", "hard"
    class _P:
        constraints = (_C(),)
    c = esco_from_gold_parse(_P(), esco)
    assert c is not None and c.resolved
    assert set(c.uris) == {"u_carpenter"}
    assert "7115" in c.styrk
