"""Location as a predicate. Grounded 2026-09-27 against ad_locations.

FOUR MEASUREMENTS SHAPED THIS, all over the real table:

  county is better covered than municipality — 99.8% vs 99.1% of 10,166 ads — and is
  a real hierarchy: 16 counties over 350 municipalities. So proximity is graded the
  way STYRK proximity is, not a boolean.

  7.1% of advertisements carry MORE THAN ONE location (12,003 rows over 10,165
  uuids), so the predicate must take the BEST of an ad's locations. A first-match
  implementation would silently reject 722 advertisements that do list the city the
  seeker asked for.

  `Oslo-området` — c4's own gold parse value — resolves to NOTHING: no municipality
  and no county carries that name. A suffix normaliser is required or a real dev
  persona fails to resolve.

  Oslo is BOTH a municipality and a county, with 1,630 ads under each name, so the
  resolver must handle a name that is both without double counting it.
"""
from __future__ import annotations

import pytest

from finn_smart_search.retrieval.location import (
    EXACT, SAME_COUNTY, AdLocation, LocationGazetteer, apply)

# (municipal, county) per ad row, shaped after the real table including its
# upper-case storage and the Oslo municipality/county coincidence.
ROWS = (
    [("OSLO", "OSLO")] * 1630
    + [("BERGEN", "VESTLAND")] * 669
    + [("VOSS", "VESTLAND")] * 40
    + [("TRONDHEIM", "TRØNDELAG")] * 389
    + [("NARVIK", "NORDLAND")] * 60
)


@pytest.fixture(scope="module")
def gaz():
    return LocationGazetteer.build_from_rows(ROWS)


def test_a_municipality_resolves(gaz):
    c = gaz.resolve("Bergen")
    assert c is not None and c.resolved
    assert "bergen" in c.municipals


def test_resolution_is_case_and_diacritic_tolerant(gaz):
    """The table stores upper case; seekers type mixed case."""
    assert gaz.resolve("TRONDHEIM").municipals == gaz.resolve("Trondheim").municipals


def test_a_county_name_resolves_as_a_county(gaz):
    c = gaz.resolve("Vestland")
    assert c is not None and "vestland" in c.counties


def test_oslo_resolves_as_both_without_double_counting(gaz):
    """Oslo is a municipality AND a county, 1,630 ads under each name."""
    c = gaz.resolve("Oslo")
    assert "oslo" in c.municipals and "oslo" in c.counties


def test_an_area_suffix_is_stripped(gaz):
    """`Oslo-området` is c4's own gold parse value and matches no row verbatim.
    Without this a real dev persona's hard location constraint never resolves."""
    c = gaz.resolve("Oslo-området")
    assert c is not None and "oslo" in c.municipals


@pytest.mark.parametrize("phrase", ["Oslo området", "oslo omegn", "Bergen og omegn"])
def test_other_area_phrasings_resolve(gaz, phrase):
    assert gaz.resolve(phrase) is not None


def test_an_unknown_place_does_not_resolve(gaz):
    """Empty, not a guess. A wrong municipality would demote every correct ad."""
    assert gaz.resolve("Atlantis") is None


# ── proximity ─────────────────────────────────────────────────────────────────

def test_the_exact_municipality_is_full_proximity(gaz):
    c = gaz.resolve("Bergen")
    assert c.proximity([AdLocation("BERGEN", "VESTLAND")]) == EXACT


def test_the_same_county_scores_partially(gaz):
    """Voss is in Vestland like Bergen. A commutable near-miss is not a match and is
    not nothing, which is why this is graded rather than boolean."""
    c = gaz.resolve("Bergen")
    p = c.proximity([AdLocation("VOSS", "VESTLAND")])
    assert 0.0 < p < EXACT and p == SAME_COUNTY


def test_a_different_county_scores_zero(gaz):
    c = gaz.resolve("Bergen")
    assert c.proximity([AdLocation("TRONDHEIM", "TRØNDELAG")]) == 0.0


def test_a_multi_location_ad_takes_its_BEST_location(gaz):
    """THE 7.1%. 722 advertisements list more than one location; taking the first
    would reject ads that do offer the city the seeker asked for."""
    c = gaz.resolve("Bergen")
    ads = [AdLocation("TRONDHEIM", "TRØNDELAG"), AdLocation("BERGEN", "VESTLAND")]
    assert c.proximity(ads) == EXACT


def test_an_ad_with_no_location_scores_zero_rather_than_raising(gaz):
    c = gaz.resolve("Bergen")
    assert c.proximity([AdLocation(None, None)]) == 0.0
    assert c.proximity([]) == 0.0


def test_a_county_query_matches_every_municipality_in_it(gaz):
    """Asking for Vestland should reach Bergen and Voss alike."""
    c = gaz.resolve("Vestland")
    assert c.proximity([AdLocation("BERGEN", "VESTLAND")]) == EXACT
    assert c.proximity([AdLocation("VOSS", "VESTLAND")]) == EXACT


# ── the two non-effects, as in constraints.py and occupation.py ────────────────

def test_an_unstated_location_leaves_the_ranking_bit_identical():
    scores = [0.9, 0.5, 0.31]
    ads = [[AdLocation("OSLO", "OSLO")]] * 3
    assert apply(scores, ads, None) == scores


def test_an_unresolvable_place_leaves_the_ranking_bit_identical(gaz):
    """Distinct from unstated, and it must fail OPEN. D18's outcome measured why: a
    predicate applied to one side of a comparison and not the other is a confound."""
    c = gaz.resolve("Atlantis")
    assert c is None
    scores = [0.9, 0.5]
    assert apply(scores, [[AdLocation("OSLO", "OSLO")]] * 2, c) == scores


def test_anywhere_in_norway_penalises_nothing(gaz):
    """p6 states `location.anywhere: true` — "I would like to teach in Norway". A
    seeker open to everywhere must not be penalised for it."""
    c = gaz.resolve("anywhere")
    scores = [0.9, 0.5]
    ads = [[AdLocation("OSLO", "OSLO")], [AdLocation("NARVIK", "NORDLAND")]]
    assert apply(scores, ads, c) == scores


def test_lambda_zero_switches_the_stage_off(gaz):
    c = gaz.resolve("Bergen")
    scores = [1.0]
    assert apply(scores, [[AdLocation("OSLO", "OSLO")]], c, lam=0.0) == scores


def test_the_predicate_promotes_the_right_city(gaz):
    c = gaz.resolve("Bergen")
    out = apply([10.0, 3.0],
                [[AdLocation("OSLO", "OSLO")], [AdLocation("BERGEN", "VESTLAND")]], c)
    assert out[1] > out[0]


def test_a_floor_keeps_out_of_area_ads_alive(gaz):
    """Hard by default, softenable — some seekers would relocate, and that is a
    product decision with a dial rather than a fixed rule."""
    c = gaz.resolve("Bergen")
    hard = apply([1.0], [[AdLocation("OSLO", "OSLO")]], c, floor=0.0)
    soft = apply([1.0], [[AdLocation("OSLO", "OSLO")]], c, floor=0.3)
    assert hard[0] == 0.0 and soft[0] == pytest.approx(0.3)
