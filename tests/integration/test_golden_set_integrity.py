"""The golden set itself must not drift.

Golden drift is the classic route to a flattering number: relabel the hard ads
and every metric improves. These assertions make that a visible, deliberate act
rather than a quiet edit.

WHY MOST OF THIS FILE IS NOW `unit`. It was entirely `integration`, and
`addopts` carries `-m "not integration"`, so none of it ran in the default
suite. The consequence was demonstrated on 2026-09-25: golden #15's level was
revised by owner ruling and COMMITTED with the drift guards red, because
`make test` never executed them. A guard that does not run is worse than no
guard, because it reads as coverage.

Every assertion here except the span check is a pure read of a JSON file with no
I/O, so there was never a reason for it to be integration-marked. Only
`test_expected_spans_are_verbatim_in_their_ads` needs data/ads.duckdb and stays
integration.
"""
import json
from collections import Counter

import pytest

from tests.conftest import requires_corpus

# Drift guards run BY DEFAULT. See the module docstring: these are JSON reads.
pytestmark = [pytest.mark.unit]

# Regenerate ONLY on a recorded owner ruling, never to make a red test green.
#   2026-09-25  #15 explicitly_not_required -> certified (D15, owner ruling):
#               certified 4->5, explicitly_not_required 3->2, accessible 13->12.
EXPECTED_MIX = {"unstated": 12, "scandinavian_accepted": 8,
                "either_norwegian_or_english": 5, "professional": 5,
                "certified": 5, "desirable": 4, "explicitly_not_required": 2,
                "conversational": 2, "fluent": 1}
N_ACCESSIBLE = 12


@pytest.fixture(scope="module")
def golden():
    return json.load(open("eval/golden_set.json"))


def test_the_golden_set_is_44_ads(golden):
    assert len(golden) == 44
    assert len({g["uuid"] for g in golden}) == 44


def test_the_level_mix_has_not_drifted(golden):
    mix = Counter(g["expected"]["norwegian_requirement_level"] for g in golden)
    assert dict(mix) == EXPECTED_MIX


def test_the_accessible_count_has_not_drifted(golden):
    """Renamed off the literal: `test_thirteen_...` had to be renamed as well as
    edited when the count changed, which is friction that encourages editing the
    number and leaving the stale name."""
    assert sum(1 for g in golden if g["derived_accessible"]) == N_ACCESSIBLE


def test_every_level_in_the_enum_is_represented(golden):
    from finn_smart_search.understanding.census_prompt import TOOL
    enum = set(TOOL["input_schema"]["properties"]["norwegian_requirement_level"]["enum"])
    assert set(EXPECTED_MIX) == enum, "a level with no golden example cannot be measured"


@pytest.mark.integration
@requires_corpus
def test_expected_spans_are_verbatim_in_their_ads(golden):
    """A golden label quoting a sentence the ad does not contain would make that
    ad permanently unwinnable."""
    from finn_smart_search.understanding.text_norm import normalise
    from tests.conftest import open_corpus
    # READ-ONLY: store.connect opens read-write, runs DDL against the artifact
    # under audit, and takes duckdb's exclusive lock.
    con = open_corpus()
    bad = []
    for g in golden:
        span = g["expected"]["evidence_span"]
        if not span:
            continue
        row = con.execute("SELECT description_text FROM ads WHERE uuid=?", [g["uuid"]]).fetchone()
        if row and normalise(span) not in normalise(row[0]):
            bad.append(g["n"])
    assert bad == [], f"golden spans not found verbatim in their ads: {bad}"
