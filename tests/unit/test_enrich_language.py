"""Wiring langid into the store: bronze -> silver -> GOLD.

`doc_lang` is deliberately NOT a column on `ads`. The architecture treats
`ads` as SILVER — a pure function of bronze, rebuildable at any time from
`ads_raw` with no external dependency. langid output depends on a lingua
version, so it is derived/gold. A column on `ads` would mean a library bump
silently invalidates the silver layer.

Resumability matters here: the full corpus is ~12 minutes of detection, long
enough that an interruption should not cost the whole run.
"""
import pytest

from finn_smart_search.understanding import enrich_language as el

pytestmark = pytest.mark.unit

NO = ("Vi søker en dyktig medarbeider til vårt team i Oslo. Du vil jobbe med "
      "daglig drift og oppfølging av kunder. Stillingen er fast.")
EN = ("We are looking for a skilled engineer to join our team in Oslo. You will "
      "work on daily operations and customer follow-up.")
PL = ("Poszukujemy wykwalifikowanych monterów wentylacji na projekt w "
      "miejscowości Haugesund i okolicach. Oferujemy umowę o pracę.")


@pytest.fixture
def con(tmp_path):
    import duckdb
    c = duckdb.connect(str(tmp_path / "t.duckdb"))
    c.execute("CREATE TABLE ads (uuid VARCHAR PRIMARY KEY, description_html VARCHAR, "
              "description_text VARCHAR, n_chars INTEGER)")
    c.execute(el.DDL)
    return c


def add(con, uuid, text):
    html = "".join(f"<p>{p}</p>" for p in text.split("\n")) if text else ""
    con.execute("INSERT INTO ads VALUES (?,?,?,?)", [uuid, html, text, len(text or "")])


def rows(con):
    return con.execute("SELECT uuid, doc_lang, detected_other, n_scored, confidence, "
                       "langid_version FROM ad_language ORDER BY uuid").fetchall()


# ── 1 · the transform ────────────────────────────────────────────────────────

def test_1_1_a_norwegian_ad_gets_a_row():
    pass  # covered by 1_2, kept for scenario-table traceability


def test_1_2_full_detect_output_is_persisted(con):
    add(con, "a", NO)
    el.run(con)
    r = con.execute("SELECT doc_lang, lang_mix, n_scored, confidence FROM ad_language").fetchone()
    assert r[0] == "no"
    assert r[1] is not None, "lang_mix must be stored, not just the verdict"
    assert r[2] >= 1 and r[3] in ("high", "low")


def test_1_3_rerun_is_idempotent(con):
    add(con, "a", NO)
    el.run(con)
    el.run(con, force=True)
    assert con.execute("SELECT count(*) FROM ad_language WHERE uuid='a'").fetchone()[0] == 1


def test_1_4_an_empty_ad_is_written_as_unknown_not_skipped(con):
    """Silently skipping would leave the ad with no doc_lang, and the census
    would then default it to "no" — hidden."""
    add(con, "a", "")
    el.run(con)
    assert rows(con) == [("a", "unknown", None, 0, "low", el.LANGID_VERSION)]


def test_1_5_rows_carry_the_langid_version(con):
    add(con, "a", NO)
    el.run(con)
    assert rows(con)[0][5] == el.LANGID_VERSION


def test_1_6_the_ads_table_is_not_touched(con):
    """Silver stays a pure function of bronze."""
    add(con, "a", NO)
    before = con.execute("SELECT * FROM ads").fetchall()
    cols_before = [c[0] for c in con.execute("DESCRIBE ads").fetchall()]
    el.run(con)
    assert con.execute("SELECT * FROM ads").fetchall() == before
    assert [c[0] for c in con.execute("DESCRIBE ads").fetchall()] == cols_before


def test_1_7_detected_other_is_recorded(con):
    add(con, "a", PL)
    el.run(con)
    r = rows(con)[0]
    assert r[1] == "other" and r[2] == "pl"


# ── 2 · resumability ─────────────────────────────────────────────────────────

def test_2_1_only_unprocessed_ads_are_reprocessed(con):
    for u, t in (("a", NO), ("b", EN), ("c", PL)):
        add(con, u, t)
    assert el.run(con)["processed"] == 3
    add(con, "d", NO)
    assert el.run(con)["processed"] == 1, "an interrupted run must resume, not restart"


def test_2_2_force_reprocesses_everything(con):
    add(con, "a", NO)
    el.run(con)
    assert el.run(con, force=True)["processed"] == 1


def test_2_3_a_version_change_triggers_reprocessing(con):
    """A lingua bump must not leave stale verdicts behind."""
    add(con, "a", NO)
    el.run(con)
    con.execute("UPDATE ad_language SET langid_version = 'langid-0'")
    assert el.run(con)["processed"] == 1
    assert rows(con)[0][5] == el.LANGID_VERSION


def test_2_4_summary_reports_the_distribution(con):
    for u, t in (("a", NO), ("b", EN), ("c", PL)):
        add(con, u, t)
    out = el.run(con)
    assert out["processed"] == 3
    assert out["distribution"]["no"] == 1
    assert out["distribution"]["other"] == 1
