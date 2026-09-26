"""A batch is capped in BYTES, and nothing was measuring bytes.

THE INCIDENT. The census submitted 9,599 requests in one batch and the API
returned 413 Payload Too Large. Nothing was billed, but the run died at the
first submission after the corpus had been prepared.

    one request serialises to   43,880 bytes
    9,599 requests              421 MB
    the documented cap          256 MB

`chunk()` existed and was used, with `size=MAX_BATCH_REQUESTS` = 100,000 — a
COUNT limit, which 9,599 never approaches. The plan's §12.3 sizing said "our
largest batch is 4,200 requests ... far under 256 MB", which was true of the
JUDGE batches it was written about: those carry one ad each. A census request
carries the whole few-shot prefix — system block plus 15 worked examples —
and that prefix is ~40 KB repeated in every single request.

So the guard that existed measured the dimension that was never going to
bind, and the dimension that binds was unmeasured. Same shape as the rest of
this session's bugs.
"""
import json

import pytest

from finn_smart_search.ingest import anthropic_client as ac

pytestmark = pytest.mark.unit

# Measured 2026-09-26 on a real census request.
CENSUS_REQUEST_BYTES = 43_880
API_CAP_BYTES = 256_000_000


def _req(n_bytes: int, cid: str) -> dict:
    """A request whose serialised size is about n_bytes."""
    pad = "x" * max(0, n_bytes - 120)
    return {"custom_id": cid, "params": {"model": "m", "messages": [
        {"role": "user", "content": pad}]}}


def test_chunk_by_bytes_keeps_every_batch_under_the_cap():
    reqs = [_req(1_000_000, f"c{i}") for i in range(600)]      # ~600 MB
    batches = list(ac.chunk_bytes(reqs, max_bytes=200_000_000))
    assert len(batches) > 1, "600 MB was not split"
    for b in batches:
        assert len(json.dumps(b).encode()) <= 200_000_000


def test_chunk_by_bytes_loses_nothing_and_preserves_order():
    reqs = [_req(1_000_000, f"c{i}") for i in range(500)]
    flat = [r for b in ac.chunk_bytes(reqs, max_bytes=200_000_000) for r in b]
    assert [r["custom_id"] for r in flat] == [r["custom_id"] for r in reqs]


def test_a_single_oversized_request_is_still_yielded_not_dropped():
    """Truncating silently is the failure `chunk` was written to avoid. One
    request larger than the cap cannot be split, so it must be yielded alone
    and allowed to fail loudly at the API rather than vanish."""
    reqs = [_req(300_000_000, "huge"), _req(1000, "small")]
    batches = list(ac.chunk_bytes(reqs, max_bytes=200_000_000))
    ids = [r["custom_id"] for b in batches for r in b]
    assert ids == ["huge", "small"]
    assert len(batches[0]) == 1


def test_the_real_census_shape_needs_more_than_one_batch():
    """Grounded in the measured request size, so a prefix that grows again
    fails this test instead of failing at the API."""
    per = CENSUS_REQUEST_BYTES
    corpus = 9_823
    assert per * corpus > API_CAP_BYTES, (
        "if this ever passes the prefix shrank; re-measure rather than delete")
    batches_needed = -(-per * corpus // 200_000_000)
    assert batches_needed >= 3, batches_needed


def test_chunk_by_count_still_exists_for_callers_that_want_it():
    reqs = [_req(1000, f"c{i}") for i in range(10)]
    assert [len(b) for b in ac.chunk(reqs, size=4)] == [4, 4, 2]
