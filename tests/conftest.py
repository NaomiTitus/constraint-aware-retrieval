import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = ROOT / "tests" / "fixtures"


import pytest


CORPUS = ROOT / "data" / "ads.duckdb"

# Integration tests run BY DEFAULT (see pyproject addopts). `data/` is
# gitignored, so a fresh clone has no corpus — those tests SKIP with a stated
# reason rather than erroring, which keeps `pytest` green on a clean checkout
# while still running every corpus assertion wherever the data exists.
requires_corpus = pytest.mark.skipif(
    not CORPUS.exists(),
    reason=f"needs {CORPUS.relative_to(ROOT)} (gitignored; run `make ingest`)",
)


def open_corpus():
    """Read-only corpus connection, or SKIP with a stated reason.

    Two environment conditions, neither a test failure:
      - the file is absent (data/ is gitignored, so a fresh clone has none)
      - another process holds duckdb's write lock (an ingest, a probe, a census)
    Erroring on either turns "you have work running" into a red suite, which
    trains people to ignore red. duckdb.connect() also CREATES a missing file,
    so the existence check must come first or an unguarded test silently
    asserts against an empty database."""
    import duckdb
    if not CORPUS.exists():
        pytest.skip(f"needs {CORPUS.relative_to(ROOT)} (gitignored; run `make ingest`)")
    try:
        return duckdb.connect(str(CORPUS), read_only=True)
    except duckdb.IOException as e:
        pytest.skip(f"corpus is locked by another process: {str(e).splitlines()[0][:90]}")


@pytest.fixture(scope="session")
def corpus_con():
    """Read-only connection to the real corpus. Session-scoped: duckdb takes a
    file lock, so per-test connections collide when tests run in one process."""
    return open_corpus()


# ── grounding sources (STANDARDS.md § 3.0) ───────────────────────────────────
#
# A test fixture is either loaded from a real artifact or carries the
# measurement that justifies its shape. These helpers make the grounded route
# the shortest route, because seven bugs in this repo were tests that built
# their own inputs and therefore agreed with themselves.
#
# Match the source to the assumption:
#   labels / semantics    -> golden_file()
#   text SHAPE            -> corpus_con (counted, not sampled) — the golden set
#                            is stratified for LABEL diversity and contains
#                            none of `·` `●` U+F0B7, the glyphs that caused the
#                            55% demotion
#   HTML / block structure -> real_blocks()
#   LLM response shape     -> recorded_batch_results()

GOLDEN = ROOT / "eval" / "golden_set.json"


def golden_file():
    """The golden set AS STORED. Never build a golden record by hand in a test:
    `test_9_1` did, setting the very key it then asserted on, and
    `taxonomy_gap` measured nothing for the whole project as a result."""
    import json
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def real_blocks(limit: int | None = None) -> list[str]:
    """Block texts as html_clean actually emits them, from the 71 committed raw
    ad descriptions. No database needed, so this is the default grounding for
    anything about block structure.

    Use `real_text()` when what matters is how blocks are JOINED."""
    import json

    from finn_smart_search.understanding.html_clean import clean

    ads = json.loads((FIXTURES / "ads_sample_71.json").read_text(encoding="utf-8"))
    out = []
    for a in ads[:limit]:
        blocks, _ = clean(a.get("description"))
        out += [b["text"] for b in blocks]
    return out


def real_text(limit: int | None = None) -> list[str]:
    """`description_text` as stored: blocks joined with "\n".

    THE shape that broke span validation. Every earlier span test used
    single-line prose with full stops, so the newline join — and the fact that
    Norwegian bullets carry no terminal punctuation — was never exercised."""
    import json

    from finn_smart_search.understanding.html_clean import clean

    ads = json.loads((FIXTURES / "ads_sample_71.json").read_text(encoding="utf-8"))
    return [clean(a.get("description"))[1] for a in ads[:limit]]


def recorded_batch_results():
    """Real Batch API responses, recorded live. Grounds anything about response
    shape, tool-use envelopes or usage accounting."""
    import json
    return json.loads((FIXTURES / "batch_results_real.json").read_text(encoding="utf-8"))


@pytest.fixture
def tmp_con(tmp_path):
    """An isolated in-memory-ish duckdb with the census schema applied."""
    import duckdb

    from finn_smart_search.understanding import census

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    con.execute(census.DDL)
    return con
