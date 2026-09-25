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
    from tests.conftest import CORPUS
    if not CORPUS.exists():
        pytest.skip(f"needs {CORPUS.name} (gitignored; run `make ingest`)")
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


# Frozen ground truth, reviewed by hand against the real corpus. A pinned map
# rather than a re-derived count: the previous version found these ads by regex
# and asserted `len(as_other) >= 10` against `len(polish) >= 10`, so with 13 ads
# and 3 regressed it read 10 >= 10 and PASSED. The historical incident was
# exactly 10 of 13 hidden — a regression test that permits its own regression.
POLISH_ADS = {
    "1760997c": "other",   # Ventilasjonsmontører Haugesund
    "615dbddf": "other",   # Erfarne malere 2026
    "d0b49779": "other",   # Taktakkere til Trondheim
    "377095b8": "other",   # ELEKTRIKERE i Stavanger
    "6588ea35": "other",   # Elektrikere med DSB til Trondheim
    "1d3ca7b0": "other",   # VIL DU HA EN TØMRERJOBB
    "c0df04f0": "other",   # Build, Reinforce, and Shape the Future
    "cc048ca6": "other",   # VIL DU HA EN TØMRERJOBB (second posting)
    "1aff4e2b": "other",   # MURER – FAST JOBB | ALL10 AS
    "8642dd8d": "other",   # Murere 2026 til Trondheim
    "782181d3": "other",   # Flisleggere 2026 til Trondheim
    # KNOWN AND ACCEPTED, not overlooked. Both are genuinely Norwegian-dominant
    # by character mass, so "no" is the honest verdict rather than a failure:
    "e2bcda12": "no",      # trilingual: "Flisleggere søkes / Job for tilers /
                           #   Praca dla glazurników" — Norwegian mass wins
    "51eca9ff": "no",      # blocks detect as Bosnian, so Polish never clears
                           #   the 0.50 dominance gate
}


def test_3_3_the_polish_ads_are_not_hidden(con):
    """THE regression test for the bug this module exists to prevent: a
    restricted detector labelled 10 of these 13 a confident "no", which under
    the accessibility fallback means hidden from the very people they target.

    Zero tolerance in BOTH directions. An improvement fails this test too —
    which is correct, because it should be a deliberate decision with the map
    updated, not a silent drift."""
    rows = dict(con.execute("""
        SELECT substr(a.uuid, 1, 8), l.doc_lang
        FROM ads a JOIN ad_language l USING (uuid)
        WHERE substr(a.uuid, 1, 8) IN ({})
    """.format(",".join(f"'{u}'" for u in POLISH_ADS))).fetchall())

    assert set(rows) == set(POLISH_ADS), "the Polish ads must all still be in the corpus"
    assert rows == POLISH_ADS, (
        "language verdicts drifted for the Polish ads:\n" +
        "\n".join(f"  {u}: expected {POLISH_ADS[u]}, got {rows[u]}"
                  for u in POLISH_ADS if rows.get(u) != POLISH_ADS[u]))


def test_3_3b_the_polish_corpus_has_not_shrunk(con):
    """The denominator must be pinned separately. A regex re-derivation makes it
    a function of the corpus, which is what let the old assertion be vacuous."""
    pol = re.compile(r"\b(praca|zatrudnimy|poszukujemy|wymagania|oferujemy|umowa)\b", re.I)
    texts = con.execute("SELECT description_text FROM ads").fetchall()
    assert sum(1 for (t,) in texts if pol.search(t or "")) == len(POLISH_ADS) == 13


def test_3_4_no_null_verdicts(con):
    assert con.execute(
        "SELECT count(*) FROM ad_language WHERE doc_lang IS NULL").fetchone()[0] == 0
