"""Derived taxonomy facets — no LLM call, no new data source.

WHY THESE THREE. An external review of a proposed 20-field taxonomy found that
most of what it wanted already sits in licensed data we had not surfaced:

  job_title_standardised  The feed's `jobtitle` is free text: 4,878 distinct
      values on 8,351 ads, of which 87.1% occur exactly once, 700 contain
      digits. It is not an identity. But ESCO's preferred Norwegian label joins
      on 8,339 ads (82.0%) with 613 controlled values AND a free English
      counterpart, and STYRK-08's name covers 10,161 (100.0%). Coalescing the
      two reaches 99.95% with a controlled vocabulary at both tiers.

  role_family  A proposed flat 12-value `job_family` enum was measured against
      all 323 STYRK codes: 66.4% clean fit, 19.6% ambiguous, 13.9% NO fit —
      because STYRK's first digit is ISCO's SKILL LEVEL while `job_family` is a
      DOMAIN axis. They are orthogonal, which is why the enum has no Management
      value while 390 ads sit on domain-free management codes. ARCHITECTURE.md
      already specifies `role_family` as the ISCO sub-major instead: derivable,
      100% covered, grounded in a public vocabulary rather than an invented one.

  nav_category  `occupationCategories.level1` is ALREADY in ads_raw.ad_content
      at ~100% as a closed 13-value vocabulary, and was never loaded. It is the
      `industry_sector` the proposal wanted as a free string — controlled and
      joinable instead. Measured distribution is not an Anglo tech board's:
      Helse og sosial 39% · Salg og service 17% · Utdanning 14% · IT ~1%.

All three cost zero tokens. The census is ~$16; nothing here touches it.
"""
import pytest

from finn_smart_search.understanding import taxonomy
from tests.conftest import requires_corpus

pytestmark = [pytest.mark.unit]


# ── role_family: pure function, no corpus needed ─────────────────────────────

@pytest.mark.parametrize("code,expected", [
    ("2223", "22"),   # Sykepleiere
    ("5223", "52"),   # Butikkmedarbeidere
    ("5321", "53"),   # Helsefagarbeidere
    ("9112", "91"),   # Renholdere i bedrifter
    ("1219", "12"),   # Andre administrative ledere
])
def test_role_family_is_the_isco_sub_major(code, expected):
    assert taxonomy.role_family(code) == expected


@pytest.mark.parametrize("bad", ["223", "22", "22345", "2.23", " 2223x"])
def test_role_family_requires_exactly_four_digits(bad):
    """A 3-digit code would silently yield a plausible 2-character family that
    traces back to nothing. Every corpus code is length 4, so this guards the
    public function's contract rather than an observed input — and without it,
    widening the pattern to ^\\d{3,}$ survived every other test."""
    assert taxonomy.role_family(bad) is None, bad
    assert taxonomy.skill_level(bad) is None, bad


def test_role_family_rejects_a_code_it_cannot_read():
    """Loud rather than silent: a short or non-numeric code returning a plausible
    2 characters would create a family nobody can trace back."""
    for bad in (None, "", "2", "abcd", "2-23"):
        assert taxonomy.role_family(bad) is None, bad


def test_role_family_is_not_the_skill_level():
    """The distinction the 12-value enum got wrong. The FIRST digit is ISCO's
    skill level (2 = professionals); the first TWO are the sub-major, which is
    the domain axis. 2223 Sykepleiere and 2521 Databasedesignere share a skill
    level and must NOT share a family."""
    assert taxonomy.role_family("2223") != taxonomy.role_family("2521")
    assert taxonomy.skill_level("2223") == taxonomy.skill_level("2521") == "2"


# ── the derived view, against the real corpus ────────────────────────────────

# The corpus assertions read the BUILT table, exactly as the silver and langid
# corpus tests do. `build()` is a pipeline step (run it with `make taxonomy`);
# opening a second, WRITABLE session connection to the same duckdb file inside
# pytest is a hard process crash, not a catchable error — found the hard way.
@requires_corpus
def test_job_title_standardised_reaches_the_measured_coverage(corpus_con):
    """Two tiers: ESCO preferred label where an occupation URI joins, STYRK name
    otherwise. Measured 2026-09-25 — regenerate only with a recorded reason."""
    r = corpus_con.execute("""
        SELECT count(*) AS n,
               sum(CASE WHEN job_title_standardised IS NOT NULL THEN 1 ELSE 0 END) AS titled,
               sum(CASE WHEN title_source = 'esco' THEN 1 ELSE 0 END) AS esco,
               sum(CASE WHEN title_source = 'styrk' THEN 1 ELSE 0 END) AS styrk
        FROM ad_taxonomy""").fetchone()
    n, titled, esco, styrk = r
    assert n == 10166
    assert titled == 10161, f"coverage moved: {titled}/10166 (was 10161 = 99.95%)"
    assert esco == 8339, f"ESCO tier moved: {esco} (was 8339 = 82.0%)"
    assert styrk == titled - esco


@requires_corpus
def test_the_esco_tier_carries_an_english_label_for_free(corpus_con):
    """The whole multilingual argument: identity is the URI, so an English label
    comes without a translation step. 8,339 ads get one."""
    r = corpus_con.execute("""SELECT count(*) FROM ad_taxonomy
        WHERE title_source='esco' AND job_title_en IS NOT NULL""").fetchone()[0]
    assert r == 8339, f"English labels: {r}"
    row = corpus_con.execute("""SELECT job_title_standardised, job_title_en
        FROM ad_taxonomy WHERE title_source='esco' AND job_title_en IS NOT NULL
        LIMIT 1""").fetchone()
    assert row[0] and row[1] and row[0] != row[1]


@requires_corpus
def test_nav_category_is_a_closed_vocabulary_at_full_coverage(corpus_con):
    rows = corpus_con.execute("""SELECT nav_category, count(*) FROM ad_taxonomy
        WHERE nav_category IS NOT NULL GROUP BY 1 ORDER BY 2 DESC""").fetchall()
    cats = {c for c, _ in rows}
    assert 10 <= len(cats) <= 16, f"vocabulary size {len(cats)}: {sorted(cats)}"
    covered = sum(n for _, n in rows)
    assert covered >= 10_100, f"nav_category coverage {covered}/10166"
    assert rows[0][0] == "Helse og sosial", f"modal category is {rows[0][0]}"


@requires_corpus
def test_role_family_covers_every_ad_with_a_styrk_code(corpus_con):
    r = corpus_con.execute("""SELECT
          sum(CASE WHEN role_family IS NOT NULL THEN 1 ELSE 0 END),
          count(DISTINCT role_family) FROM ad_taxonomy""").fetchone()
    assert r[0] == 10161, f"role_family coverage {r[0]}"
    assert 30 <= r[1] <= 60, f"{r[1]} distinct sub-majors"


def test_the_styrk_tie_break_is_pinned_because_it_is_arbitrary(tmp_con):
    """1,486 corpus ads carry more than one STYRK code, and for 958 of them
    (9.4%) min() and max() give a DIFFERENT role_family. `score` cannot
    adjudicate (1.0 on 99.6% of rows), so the choice is arbitrary — which makes
    pinning it the only way it stays reproducible.

    Switching to max() survived every other test in this file."""
    tmp_con.execute("CREATE TABLE ads (uuid VARCHAR PRIMARY KEY, jobtitle VARCHAR)")
    tmp_con.execute("CREATE TABLE ads_raw (uuid VARCHAR, ad_content JSON)")
    tmp_con.execute("""CREATE TABLE ad_categories (uuid VARCHAR, category_type VARCHAR,
                       code VARCHAR, name VARCHAR, score DOUBLE)""")
    tmp_con.execute("CREATE TABLE esco_occupation (uri VARCHAR, lang VARCHAR, title VARCHAR)")
    tmp_con.execute("INSERT INTO ads VALUES ('m','Miljøterapeut / sykepleier')")
    tmp_con.execute("INSERT INTO ads_raw VALUES ('m','{}')")
    # two codes in DIFFERENT sub-majors, both score 1.0 — the real shape
    tmp_con.execute("""INSERT INTO ad_categories VALUES
        ('m','STYRK08','2223','Sykepleiere',1.0),
        ('m','STYRK08','3412','Miljøarbeidere',1.0)""")
    taxonomy.build(tmp_con, log=lambda *_: None)
    got = tmp_con.execute(
        "SELECT styrk_code, role_family FROM ad_taxonomy WHERE uuid='m'").fetchone()
    assert got == ("2223", "22"), (
        f"tie-break drifted to {got}; min(code) is the pinned choice and 958 ads "
        "change role_family if it moves"
    )


def test_build_is_idempotent_and_derives_from_bronze(tmp_con):
    """Built on a temp db from the real table SHAPES, so it exercises build()
    without a second connection to the corpus."""
    tmp_con.execute("CREATE TABLE ads (uuid VARCHAR PRIMARY KEY, jobtitle VARCHAR)")
    tmp_con.execute("CREATE TABLE ads_raw (uuid VARCHAR, ad_content JSON)")
    tmp_con.execute("""CREATE TABLE ad_categories (uuid VARCHAR, category_type VARCHAR,
                       code VARCHAR, name VARCHAR, score DOUBLE)""")
    tmp_con.execute("CREATE TABLE esco_occupation (uri VARCHAR, lang VARCHAR, title VARCHAR)")
    URI = "http://data.europa.eu/esco/occupation/8d3e8aaa"
    tmp_con.execute("INSERT INTO ads VALUES ('a','Sykepleier 100% vikar'),('b','Tømrar')")
    tmp_con.execute("""INSERT INTO ads_raw VALUES
        ('a', '{"ad_content":{"occupationCategories":[{"level1":"Helse og sosial",
                 "level2":"Sykepleiere"}]}}'),
        ('b', '{"ad_content":{"occupationCategories":[{"level1":"Håndverkere",
                 "level2":"Tømrere"}]}}')""")
    tmp_con.execute(f"""INSERT INTO ad_categories VALUES
        ('a','ESCO','{URI}','Sykepleier',1.0),
        ('a','STYRK08','2223','Sykepleiere',1.0),
        ('b','STYRK08','7115','Tømrere og snekkere',1.0)""")
    tmp_con.execute(f"""INSERT INTO esco_occupation VALUES
        ('{URI}','no','sykepleier'),('{URI}','en','nurse')""")

    taxonomy.build(tmp_con, log=lambda *_: None)
    got = dict((r[0], r[1:]) for r in tmp_con.execute(
        "SELECT uuid, job_title_standardised, job_title_en, title_source, role_family,"
        " skill_level, nav_category FROM ad_taxonomy ORDER BY uuid").fetchall())
    # ESCO tier wins where a URI joins, and brings English for free
    assert got["a"] == ("sykepleier", "nurse", "esco", "22", "2", "Helse og sosial")
    # STYRK fallback where it does not; no English label to offer
    assert got["b"] == ("Tømrere og snekkere", None, "styrk", "71", "7", "Håndverkere")

    taxonomy.build(tmp_con, log=lambda *_: None)
    assert tmp_con.execute("SELECT count(*) FROM ad_taxonomy").fetchone()[0] == 2
