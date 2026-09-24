import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = ROOT / "tests" / "fixtures"


import pytest


@pytest.fixture
def tmp_con(tmp_path):
    """An isolated in-memory-ish duckdb with the census schema applied."""
    import duckdb

    from finn_smart_search.understanding import census

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    con.execute(census.DDL)
    return con
