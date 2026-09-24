"""Corpus-level pinning for language enrichment. Requires data/ads.duckdb, so
it is marked integration and excluded from CI."""
import re

import pytest

pytestmark = [pytest.mark.integration]

# Measured on a 1,500-ad sample. The corpus is frozen (DECISIONS.md D2), so
# these are pinned tight: if it is ever re-pulled this SHOULD break, because
# that is a conversation rather than a nuisance.
EXPECTED_SHARE = {"no": (0.93, 0.97), "en": (0.03, 0.05),
                  "mixed": (0.004, 0.015), "other": (0.001, 0.010)}


@pytest.fixture(scope="module")
def con():
    from finn_smart_search.ingest import store
    return store.connect("data/ads.duckdb")


def test_3_1_every_ad_has_exactly_one_language_row(con):
    n_ads = con.execute("SELECT count(*) FROM ads").fetchone()[0]
    n_rows, n_uuid = con.execute(
        "SELECT count(*), count(DISTINCT uuid) FROM ad_language").fetchone()
    assert n_rows == n_ads == n_uuid, "no ad lost, none duplicated"


def test_3_2_distribution_matches_the_measured_corpus(con):
    total = con.execute("SELECT count(*) FROM ad_language").fetchone()[0]
    dist = dict(con.execute(
        "SELECT doc_lang, count(*) FROM ad_language GROUP BY 1").fetchall())
    for lang, (lo, hi) in EXPECTED_SHARE.items():
        share = dist.get(lang, 0) / total
        assert lo <= share <= hi, f"{lang}: {share:.4f} outside [{lo}, {hi}]"


def test_3_3_the_polish_ads_are_not_hidden(con):
    """THE regression test for the bug this module exists to prevent. A
    restricted detector labelled 10 of 13 Polish ads a confident "no", which
    under the accessibility fallback means hidden from the people they target."""
    rows = con.execute("""
        SELECT a.uuid, a.description_text, l.doc_lang, l.detected_other
        FROM ads a JOIN ad_language l USING (uuid)""").fetchall()
    pol = re.compile(r"\b(praca|zatrudnimy|poszukujemy|wymagania|oferujemy|umowa)\b", re.I)
    polish = [(u, d, o) for u, t, d, o in rows if pol.search(t or "")]
    assert len(polish) >= 10, "the Polish ads must still be in the corpus"
    as_other = [u for u, d, o in polish if d == "other"]
    assert len(as_other) >= 10, (
        f"only {len(as_other)}/{len(polish)} Polish ads are `other` — "
        "a regression toward hiding them")
    assert any(o == "pl" for _, _, o in polish)


def test_3_4_no_null_verdicts(con):
    assert con.execute(
        "SELECT count(*) FROM ad_language WHERE doc_lang IS NULL").fetchone()[0] == 0
