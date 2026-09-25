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


@pytest.fixture(scope="session")
def corpus_con():
    """Read-only connection to the real corpus. Session-scoped: duckdb takes a
    file lock, so per-test connections collide when tests run in one process."""
    if not CORPUS.exists():
        pytest.skip(f"needs {CORPUS.relative_to(ROOT)}")
    from finn_smart_search.ingest import store
    return store.connect(str(CORPUS))


@pytest.fixture
def tmp_con(tmp_path):
    """An isolated in-memory-ish duckdb with the census schema applied."""
    import duckdb

    from finn_smart_search.understanding import census

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    con.execute(census.DDL)
    return con
