"""The golden set itself must not drift.

Golden drift is the classic route to a flattering number: relabel the hard ads
and every metric improves. These assertions make that a visible, deliberate act
rather than a quiet edit.
"""
import json
from collections import Counter

import pytest

pytestmark = [pytest.mark.integration]

EXPECTED_MIX = {"unstated": 12, "scandinavian_accepted": 8,
                "either_norwegian_or_english": 5, "professional": 5,
                "certified": 4, "desirable": 4, "explicitly_not_required": 3,
                "conversational": 2, "fluent": 1}


@pytest.fixture(scope="module")
def golden():
    return json.load(open("eval/golden_set.json"))


def test_the_golden_set_is_44_ads(golden):
    assert len(golden) == 44
    assert len({g["uuid"] for g in golden}) == 44


def test_the_level_mix_has_not_drifted(golden):
    mix = Counter(g["expected"]["norwegian_requirement_level"] for g in golden)
    assert dict(mix) == EXPECTED_MIX


def test_thirteen_ads_derive_as_accessible(golden):
    assert sum(1 for g in golden if g["derived_accessible"]) == 13


def test_every_level_in_the_enum_is_represented(golden):
    from finn_smart_search.understanding.census_prompt import TOOL
    enum = set(TOOL["input_schema"]["properties"]["norwegian_requirement_level"]["enum"])
    assert set(EXPECTED_MIX) == enum, "a level with no golden example cannot be measured"


def test_expected_spans_are_verbatim_in_their_ads(golden):
    """A golden label quoting a sentence the ad does not contain would make that
    ad permanently unwinnable."""
    from finn_smart_search.ingest import store
    from finn_smart_search.understanding.text_norm import normalise
    con = store.connect("data/ads.duckdb")
    bad = []
    for g in golden:
        span = g["expected"]["evidence_span"]
        if not span:
            continue
        row = con.execute("SELECT description_text FROM ads WHERE uuid=?", [g["uuid"]]).fetchone()
        if row and normalise(span) not in normalise(row[0]):
            bad.append(g["n"])
    assert bad == [], f"golden spans not found verbatim in their ads: {bad}"
