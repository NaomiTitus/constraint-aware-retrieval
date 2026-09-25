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
    """READ-ONLY, via the shared helper.

    This used `store.connect("data/ads.duckdb")`, which opens read-WRITE, runs
    BRONZE_DDL (`CREATE TABLE IF NOT EXISTS ...`) against the artifact under
    audit, and takes duckdb's exclusive lock — so this test blocked every other
    reader and errored whenever an ingest or probe was running. It also used a
    RELATIVE path while the skip-guard checked the absolute one, so from any
    other working directory the guard passed and the connect created a fresh
    empty database that every assertion then ran against.
    """
    from tests.conftest import open_corpus
    return open_corpus()


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
# Multilingual ads: a Norwegian block list followed by a full parallel
# translation. Frozen uuid -> verdict, updated 2026-09-25 for the
# NORDIC_ADJACENT / OTHER_MINORITY change (see DECISIONS and langid.py).
#
# WHY THE NAME CHANGED: one of these is Lithuanian, so "MULTILINGUAL_ADS" was already
# wrong. The failure mode is language-agnostic — a parallel translation whose
# foreign mass sits below the dominance gate.
MULTILINGUAL_ADS = {
    "1760997c": "other",   # Ventilasjonsmontører Haugesund
    "615dbddf": "other",   # Erfarne malere 2026
    "d0b49779": "other",   # Taktakkere til Trondheim
    "377095b8": "other",   # ELEKTRIKERE i Stavanger
    "1d3ca7b0": "other",   # VIL DU HA EN TØMRERJOBB
    "c0df04f0": "other",   # Build, Reinforce, and Shape the Future
    "cc048ca6": "other",   # VIL DU HA EN TØMRERJOBB (second posting)
    "1aff4e2b": "other",   # MURER – FAST JOBB | ALL10 AS
    "782181d3": "other",   # Flisleggere 2026 til Trondheim
    # MOVED other -> mixed by OTHER_MINORITY. Both derive as accessible, so no
    # accessibility change; `mixed` is the more accurate description of an ad
    # that is majority Norwegian with a full Polish translation.
    "6588ea35": "mixed",   # Elektrikere med DSB til Trondheim
    "8642dd8d": "mixed",   # Murere 2026 til Trondheim
    # RECOVERED from hidden. Its previous comment read "blocks detect as Bosnian,
    # so Polish never clears the 0.50 dominance gate" — OTHER_MINORITY (0.20)
    # now catches it and the ad is accessible.
    "51eca9ff": "mixed",   # Vi søker etter hjelpearbeidere innen bygg
    # NEWLY FOUND by the diacritic denominator below; all three were `no`, i.e.
    # hidden from the seekers they target, and none used any of the six words the
    # old keyword denominator looked for.
    "4b35916f": "mixed",   # Murpussere til Trondheim (pl, other mass 0.376)
    "510aa5d9": "mixed",   # Erfarne steinleggere (pl, 0.331)
    "11fd761e": "mixed",   # tømrerarbeid / ieškome stalių (lt, 0.445)
    # KNOWN AND ACCEPTED, not overlooked. Genuinely Norwegian-dominant by
    # character mass, so "no" is the honest verdict rather than a failure:
    "e2bcda12": "no",      # trilingual: "Flisleggere søkes / Job for tilers /
                           #   Praca dla glazurników" — Norwegian mass wins
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
    """.format(",".join(f"'{u}'" for u in MULTILINGUAL_ADS))).fetchall())

    assert set(rows) == set(MULTILINGUAL_ADS), "the Polish ads must all still be in the corpus"
    assert rows == MULTILINGUAL_ADS, (
        "language verdicts drifted for the Polish ads:\n" +
        "\n".join(f"  {u}: expected {MULTILINGUAL_ADS[u]}, got {rows[u]}"
                  for u in MULTILINGUAL_ADS if rows.get(u) != MULTILINGUAL_ADS[u]))


def test_3_3b_the_multilingual_population_is_pinned_independently():
    """The denominator must NOT be the same instrument as the numerator.

    The old version counted ads matching six Polish words — the same regex family
    that defined the map — so an ad in a parallel translation that used none of
    those six words was outside BOTH the numerator and the denominator. Three
    were: 4b35916f, 510aa5d9 (Polish) and 11fd761e (Lithuanian), all sitting at
    `no` and hidden.

    My first replacement was worse: a "broader" Polish word list matched 2,439
    ads, because `kontakt` and `prace` collide with Norwegian. Recorded because
    it is the same hand-written-pattern failure this repo keeps finding, and I
    made it while fixing exactly that class of bug.

    What works is a signal Norwegian does not have: the Slavic and Baltic
    diacritics ł ż ś ć ę ą ź ń ų ė į š ž appear in no Norwegian word. Measured:
    15 ads carry 5 or more.
    """
    from tests.conftest import open_corpus
    con = open_corpus()
    SLAVIC_BALTIC = re.compile(r"[łżśćęąźńŁŻŚĆĘĄŹŃųėįšžĮŠŽ]")
    texts = con.execute("SELECT substr(uuid,1,8), description_text FROM ads").fetchall()
    pop = {u for u, t in texts if len(SLAVIC_BALTIC.findall(t or "")) >= 5}
    assert len(pop) == 15, (
        f"the multilingual population moved: {len(pop)} ads carry >=5 non-Norwegian "
        "diacritics, was 15. A new one appearing needs a verdict in the map."
    )
    # Every diacritic-bearing ad must be either in the frozen map or explicitly
    # accounted for here, so a newcomer cannot slip through unlabelled.
    ACCOUNTED = set(MULTILINGUAL_ADS) | {
        "3c17459a",   # Sami, misdetected as Swahili -> mixed. See the note below.
        "7602201b",   # Samiskspråklig barnehagelærer, same.
        "c7e0fd0e",   # Giellakonsulenta: Sami, misdetected Welsh, still `no`.
    }
    assert pop <= ACCOUNTED, f"unaccounted multilingual ads: {sorted(pop - ACCOUNTED)}"


def test_3_3c_the_sami_ads_are_a_known_limitation():
    """HONEST RECORD of a cost the OTHER_MINORITY change introduced.

    lingua has no Sami model and misdetects it as Swahili or Welsh. Two
    Sami/Norwegian ads therefore now score foreign-minority and land on `mixed`,
    which derives as English-ACCESSIBLE — and they are not: an English-only
    speaker can read neither half. Against that, the change recovered four
    Polish/Lithuanian ads from being hidden.

    Net 4 recovered against 2 wrongly exposed, and a wrongly-shown ad costs one
    click while a wrongly-hidden ad is invisible — so the trade is taken
    deliberately, and pinned here so it cannot drift further unnoticed."""
    from tests.conftest import open_corpus
    con = open_corpus()
    rows = dict(con.execute("""SELECT substr(uuid,1,8), doc_lang FROM ad_language
        WHERE substr(uuid,1,8) IN ('3c17459a','7602201b','c7e0fd0e')""").fetchall())
    assert rows == {"3c17459a": "mixed", "7602201b": "mixed", "c7e0fd0e": "no"}, rows

def test_3_4_no_null_verdicts(con):
    assert con.execute(
        "SELECT count(*) FROM ad_language WHERE doc_lang IS NULL").fetchone()[0] == 0
