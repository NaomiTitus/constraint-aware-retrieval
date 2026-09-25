"""Wiring langid into the store: bronze -> silver -> GOLD.

`doc_lang` is deliberately NOT a column on `ads`. The architecture treats
`ads` as SILVER — a pure function of bronze, rebuildable at any time from
`ads_raw` with no external dependency. langid output depends on a lingua
version, so it is derived/gold. A column on `ads` would mean a library bump
silently invalidates the silver layer.

Resumability matters here: the full corpus is ~12 minutes of detection, long
enough that an interruption should not cost the whole run.
"""
from datetime import datetime, timezone

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

def test_1_2_full_detect_output_is_persisted(con):
    """`lang_mix is not None` catches nothing: json.dumps({}) is the non-None
    string "{}". Parse it back. Without the evidence behind a verdict you
    cannot tell a 0.51/0.49 "no" from a 0.99 "no" without another 12-minute
    run — which is exactly what you need when the next regression appears."""
    import json
    add(con, "a", NO + "\n" + NO)        # two blocks -> confidence "high"
    add(con, "b", NO + "\n" + EN)        # one of each -> mixed, bilingual
    add(con, "c", NO)                    # ONE block -> confidence "low"
    el.run(con)
    r = con.execute("SELECT doc_lang, lang_mix, n_scored, confidence, is_bilingual, "
                    "detected_at FROM ad_language WHERE uuid='a'").fetchone()
    assert r[0] == "no"
    mix = json.loads(r[1])
    assert max(mix, key=mix.get) == "no" and mix["no"] > 0.5
    assert sum(mix.values()) == pytest.approx(1.0)
    assert r[2] == 2 and r[3] == "high"
    assert r[4] is False, "a monolingual ad is not bilingual"
    assert r[5].tzinfo is not None, "detected_at must be tz-aware"
    assert abs((datetime.now(timezone.utc) - r[5]).total_seconds()) < 120

    bi = con.execute("SELECT doc_lang, is_bilingual FROM ad_language WHERE uuid='b'").fetchone()
    assert bi == ("mixed", True), "is_bilingual was asserted nowhere; True pins the other pole"

    # A single scored block is LOW confidence by the n_scored >= 2 rule. Pinning
    # both poles is what kills a hardcoded "high".
    assert con.execute("SELECT n_scored, confidence FROM ad_language "
                       "WHERE uuid='c'").fetchone() == (1, "low")


def test_1_3_a_second_run_does_NO_WORK(con):
    """The old version called run() then run(force=True), which deliberately
    redoes the work — so the skip path was untested, and `count == 1` was
    guaranteed by the PRIMARY KEY rather than by anything in run().

    The timestamp check is the load-bearing part: without it, a mutant that
    rewrites every row on every run still passes, and on 10,166 ads that is 12
    wasted minutes — resumability silently gone."""
    add(con, "a", NO)
    el.run(con)
    ts = con.execute("SELECT detected_at FROM ad_language WHERE uuid='a'").fetchone()[0]

    out = el.run(con)
    assert out["processed"] == 0
    assert out["distribution"] == {}
    assert con.execute("SELECT count(*) FROM ad_language").fetchone()[0] == 1
    assert con.execute("SELECT detected_at FROM ad_language WHERE uuid='a'").fetchone()[0] == ts


@pytest.mark.parametrize("html", ["", None, "<p></p>", "   "])
def test_1_4_an_empty_ad_is_written_as_unknown_not_skipped(con, html):
    """Silently skipping leaves the ad with no row, and no row defaults to "no"
    downstream — hidden. NULL is the real-database shape and was untested; the
    add() helper only ever produced "" or "<p></p>"."""
    con.execute("INSERT INTO ads VALUES (?,?,?,?)", ["a", html, html, 0])
    el.run(con)
    assert rows(con) == [("a", "unknown", None, 0, "low", el.LANGID_VERSION)]


def test_1_4b_every_ad_gets_a_row_whatever_its_body(con):
    """The count-equality invariant, as a UNIT test. It previously existed only
    in the integration suite, which CI excludes — so a skip-empties mutation
    would ship green."""
    con.execute("INSERT INTO ads VALUES ('a', NULL, NULL, 0)")
    con.execute("INSERT INTO ads VALUES ('b', '', '', 0)")
    add(con, "c", NO)
    out = el.run(con)
    n_ads = con.execute("SELECT count(*) FROM ads").fetchone()[0]
    n_rows = con.execute("SELECT count(*) FROM ad_language").fetchone()[0]
    assert n_rows == n_ads == 3
    assert out["processed"] == n_rows, "`processed` is a loop counter; cross-check it"


def test_1_5_the_version_written_is_the_CONSTANT_not_a_literal(con, monkeypatch):
    """Comparing the stored value to the constant passes even if the literal
    "langid-1" was inlined — they are equal today. Patching the constant breaks
    the tie.

    Why it matters: the day someone bumps to langid-2, _todo() selects all
    10,166 ads, writes "langid-1" back, and EVERY subsequent run reprocesses
    the whole corpus forever. "A bump triggers reprocessing" inverts into "a
    bump never completes"."""
    monkeypatch.setattr(el, "LANGID_VERSION", "langid-test-9")
    add(con, "a", NO)
    el.run(con)
    assert rows(con)[0][5] == "langid-test-9"


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


def test_2_3b_a_NULL_version_also_triggers_reprocessing(con):
    """`IS DISTINCT FROM` -> `!=` is the standard "simplify the SQL" edit. Under
    `!=`, a NULL version yields NULL rather than TRUE and the row is never
    re-selected: permanently stale, permanently skipped."""
    add(con, "a", NO)
    el.run(con)
    con.execute("UPDATE ad_language SET langid_version = NULL")
    assert el.run(con)["processed"] == 1


def test_2_4_summary_reports_the_distribution(con):
    for u, t in (("a", NO), ("b", EN), ("c", PL)):
        add(con, u, t)
    out = el.run(con)
    assert out["processed"] == 3
    assert out["distribution"] == {"no": 1, "en": 1, "other": 1}


def test_2_5_a_crash_preserves_completed_work(con):
    """The actual interruption scenario: N ads flushed, process killed, restart
    must skip those N and finish the rest. test_2_1 only proves NEW ads get
    picked up, which is a different property.

    Note there is deliberately no transaction around the whole run — per-batch
    commits are what make resume possible. Wrapping it in one transaction later
    would silently destroy that, so this test locks the behaviour."""
    for u, txt in (("a", NO), ("b", EN), ("c", PL), ("d", NO), ("e", EN)):
        add(con, u, txt)
    el.run(con)
    before = dict(con.execute("SELECT uuid, detected_at FROM ad_language").fetchall())

    con.execute("DELETE FROM ad_language WHERE uuid IN ('d','e')")     # simulated crash
    assert el.run(con)["processed"] == 2, "must resume, not restart"

    after = dict(con.execute("SELECT uuid, detected_at FROM ad_language").fetchall())
    assert len(after) == 5
    for u in ("a", "b", "c"):
        assert after[u] == before[u], "completed work must not be redone"


@pytest.mark.parametrize("batch", [1, 2, 500])
def test_2_6_all_rows_are_written_across_batch_boundaries(con, batch):
    """batch=500 with 1-4 ads per test meant the in-loop flush NEVER ran. A
    broken flush — wrong column order, a missing buf.clear(), a miscounted
    `done` — would survive the entire unit suite and surface only in the
    12-minute production run."""
    for i, txt in enumerate((NO, EN, PL, NO, EN)):
        add(con, f"u{i}", txt)
    out = el.run(con, batch=batch)
    assert out["processed"] == 5
    assert con.execute("SELECT count(*) FROM ad_language").fetchone()[0] == 5
    assert sum(out["distribution"].values()) == out["processed"]


# ── every verdict field must be persisted ────────────────────────────────────
#
# WHY. `detected_nordic` was added to langid.detect() and to no table. It was
# computed on all 10,166 ads during an 885-second enrichment run and thrown
# away, and the only reason it surfaced was a query that happened to select it.
# That is the "field nothing writes" pattern in the other direction — and the
# third instance this session, after `annotated_accessible` (read, never
# written) and `_reasons` (written, then erased by revalidation).
#
# So the guard is on the CONTRACT, not on the field: any key detect() returns
# must have a column, forever.

def test_every_detect_field_has_a_column(tmp_con):
    """A verdict key with no column is silently discarded. Compares the real
    return shape against the real DDL rather than a hand-written list of either."""
    from finn_smart_search.understanding import enrich_language as el
    from finn_smart_search.understanding.langid import detect

    el.ensure_schema(tmp_con)
    cols = {r[0] for r in tmp_con.execute("DESCRIBE ad_language").fetchall()}
    verdict = detect([{"index": 0, "tag": "p", "n_chars": 41,
                       "text": "Vi søker en dyktig medarbeider til teamet"}])
    missing = set(verdict) - cols
    assert missing == set(), f"detect() returns {missing} with nowhere to store it"


def test_the_writer_round_trips_every_field(tmp_con):
    """Columns existing is not enough — the INSERT must name them. The previous
    INSERT used positional VALUES, so a new column was accepted by the schema and
    left NULL by the writer."""
    from finn_smart_search.understanding import enrich_language as el

    el.ensure_schema(tmp_con)
    tmp_con.execute("CREATE TABLE IF NOT EXISTS ads_raw (uuid VARCHAR, ad_content JSON)")
    tmp_con.execute("""CREATE TABLE IF NOT EXISTS ads (
        uuid VARCHAR PRIMARY KEY, description_html VARCHAR, n_chars INTEGER)""")
    # a Norwegian body with a Danish block: detected_nordic must survive the write
    html = ("<p>Vi søker en dyktig medarbeider til vårt team i Oslo nå</p>"
            "<p>Vi søger en dygtig medarbejder til vores afdeling i København</p>")
    tmp_con.execute("INSERT INTO ads VALUES (?,?,?)", ["a", html, len(html)])
    el.run(tmp_con, force=True, log=lambda *_: None)
    row = tmp_con.execute("SELECT doc_lang, detected_nordic FROM ad_language "
                          "WHERE uuid='a'").fetchone()
    assert row[0] == "no"
    assert row[1] == "da", "detected_nordic was computed and then dropped by the writer"


def test_ensure_schema_migrates_a_table_without_detected_nordic(tmp_con):
    """The corpus table predates the field. CREATE TABLE IF NOT EXISTS does not
    add a column, so without an ALTER every INSERT would fail — and rows already
    paid for (885 seconds of lingua) must survive the migration."""
    from finn_smart_search.understanding import enrich_language as el
    from datetime import datetime, timezone

    tmp_con.execute("DROP TABLE IF EXISTS ad_language")
    tmp_con.execute("""CREATE TABLE ad_language (
        uuid VARCHAR PRIMARY KEY, doc_lang VARCHAR, lang_mix JSON,
        detected_other VARCHAR, is_bilingual BOOLEAN, n_scored INTEGER,
        confidence VARCHAR, langid_version VARCHAR, detected_at TIMESTAMPTZ)""")
    tmp_con.execute("INSERT INTO ad_language VALUES (?,?,?,?,?,?,?,?,?)",
                    ["keep", "no", "{}", None, False, 3, "high", "v1",
                     datetime.now(timezone.utc)])
    el.ensure_schema(tmp_con)
    cols = {r[0] for r in tmp_con.execute("DESCRIBE ad_language").fetchall()}
    assert "detected_nordic" in cols
    kept = tmp_con.execute("SELECT doc_lang, n_scored, detected_nordic "
                           "FROM ad_language WHERE uuid='keep'").fetchone()
    assert kept == ("no", 3, None), "the existing verdict must survive the migration"
