"""The real Batch API client — translation layer only.

The 40 census tests run against `FakeClient`, which pins a protocol I wrote
from reading the docs. Nothing has ever verified that the real API matches it.
That is the classic fake-versus-real divergence: the suite can be entirely
green while the fake and reality disagree about response shape, and you find
out when you spend $10.

So this file tests the four translations, and `tests/fixture/` holds a REAL
recorded response to settle the contract question for about a third of a cent.

Owner ruling: a truncated tool call (stop_reason == "max_tokens") is `errored`
and retried. Never salvaged — a half-parsed facet set is indistinguishable from
a real one downstream.
"""
import json

import pytest

from finn_smart_search.ingest import anthropic_client as ac

pytestmark = pytest.mark.unit

FACETS = {"norwegian_requirement_level": "professional", "evidence_spans": []}


def result(custom_id="a", *, content=None, stop_reason="tool_use",
           result_type="succeeded", usage=None):
    """Shape of one entry in the Batch API results stream."""
    if content is None:
        content = [{"type": "tool_use", "id": "t1", "name": "record_ad_facets",
                    "input": FACETS}]
    if result_type != "succeeded":
        return {"custom_id": custom_id, "result": {"type": result_type}}
    return {"custom_id": custom_id, "result": {"type": "succeeded", "message": {
        "content": content, "stop_reason": stop_reason,
        "usage": usage or {"input_tokens": 1000, "output_tokens": 200}}}}


# ── 1 · request translation ─────────────────────────────────────────────────

def test_1_1_custom_id_and_params_are_preserved():
    reqs = [{"custom_id": "abc-123", "params": {"model": "m", "messages": []}}]
    env = ac.build_batch(reqs)
    assert env[0]["custom_id"] == "abc-123"
    assert env[0]["params"]["model"] == "m"


def test_1_2_batches_are_chunked_not_truncated():
    """The API caps a batch at 100,000 requests. Silently dropping the tail
    would lose ads with no error."""
    reqs = [{"custom_id": str(i), "params": {}} for i in range(250_000)]
    chunks = list(ac.chunk(reqs))
    assert sum(len(c) for c in chunks) == 250_000
    assert all(len(c) <= ac.MAX_BATCH_REQUESTS for c in chunks)
    assert [r["custom_id"] for c in chunks for r in c] == [str(i) for i in range(250_000)]


# ── 2 · response translation ────────────────────────────────────────────────

def test_2_1_facets_come_from_the_tool_use_block():
    r = ac.parse_result(result())
    assert r == {"custom_id": "a", "type": "succeeded", "facets": FACETS,
                 "usage": {"input_tokens": 1000, "output_tokens": 200}}


def test_2_2_a_leading_text_block_does_not_break_extraction():
    """Positional indexing (content[0]) breaks the moment the model emits any
    preamble. Search by block type and tool name instead."""
    content = [{"type": "text", "text": "Let me analyse this advertisement."},
               {"type": "tool_use", "id": "t1", "name": "record_ad_facets", "input": FACETS}]
    assert ac.parse_result(result(content=content))["facets"] == FACETS


def test_2_3_a_wrongly_named_tool_is_an_error_not_a_silent_miss():
    content = [{"type": "tool_use", "id": "t1", "name": "some_other_tool", "input": {}}]
    r = ac.parse_result(result(content=content))
    assert r["type"] == "errored" and "record_ad_facets" in r["error"]


def test_2_4_no_tool_use_block_at_all_is_an_error():
    content = [{"type": "text", "text": "I cannot determine the language."}]
    r = ac.parse_result(result(content=content))
    assert r["type"] == "errored"
    assert "facets" not in r, "never return a partial record"


@pytest.mark.parametrize("result_type", ["errored", "expired", "canceled"])
def test_2_5_non_success_result_types_map_through(result_type):
    r = ac.parse_result(result(result_type=result_type))
    assert r["type"] == result_type
    assert "facets" not in r


# ── 2.6 · truncation: owner ruling is retry, never salvage ──────────────────

def test_2_6_truncated_tool_call_is_errored_not_salvaged():
    """stop_reason == "max_tokens" means the JSON is cut mid-structure. A
    half-parsed facet set is indistinguishable from a real one downstream, so
    it must NOT be returned. census.run() already retries errored requests."""
    partial = {"norwegian_requirement_level": "professional"}   # missing 14 fields
    content = [{"type": "tool_use", "id": "t1", "name": "record_ad_facets", "input": partial}]
    r = ac.parse_result(result(content=content, stop_reason="max_tokens"))
    assert r["type"] == "errored"
    assert "max_tokens" in r["error"]
    assert "facets" not in r, "a truncated record must never reach validation"


def test_2_7_usage_includes_cache_reads():
    """The 3,413-token cached prefix is what keeps the census at ~$10 instead of
    ~$23. Dropping cache fields would misreport spend by more than 2x."""
    u = {"input_tokens": 300, "output_tokens": 200,
         "cache_read_input_tokens": 3000, "cache_creation_input_tokens": 0}
    assert ac.parse_result(result(usage=u))["usage"] == u


# ── 3 · credentials ─────────────────────────────────────────────────────────

def test_3_1_keychain_is_preferred(monkeypatch):
    monkeypatch.setattr(ac, "_keychain_key", lambda: "sk-ant-from-keychain")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    assert ac.api_key() == "sk-ant-from-keychain"


def test_3_2_env_is_the_fallback(monkeypatch):
    monkeypatch.setattr(ac, "_keychain_key", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    assert ac.api_key() == "sk-ant-from-env"


def test_3_3_no_credentials_fails_before_any_network_call(monkeypatch):
    monkeypatch.setattr(ac, "_keychain_key", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ac.NoCredentialsError):
        ac.api_key()


def test_3_4_the_key_never_appears_in_repr_or_str(monkeypatch):
    """A key in a traceback ends up in logs and bug reports."""
    monkeypatch.setattr(ac, "_keychain_key", lambda: "sk-ant-secret-value")
    c = ac.AnthropicBatchClient()
    assert "sk-ant-secret-value" not in repr(c)
    assert "sk-ant-secret-value" not in str(c)


# ── 4 · the contract with FakeClient ────────────────────────────────────────

def test_4_1_the_real_client_satisfies_the_fake_protocol():
    """If the fake and the real client disagree, 40 green census tests are
    worthless. Pin the method surface the census actually calls."""
    for method in ("submit_batch", "poll", "results"):
        assert callable(getattr(ac.AnthropicBatchClient, method, None)), \
            f"census.run() calls client.{method}()"


def test_4_2_parse_result_output_matches_what_census_expects():
    """census.run() branches on `type`, reads `facets` and `usage` on success,
    and `error` otherwise. Anything else is a contract break."""
    ok = ac.parse_result(result())
    assert set(ok) == {"custom_id", "type", "facets", "usage"}
    bad = ac.parse_result(result(result_type="errored"))
    assert set(bad) == {"custom_id", "type", "error"}


# ── 5 · contract against a RECORDED REAL response ───────────────────────────

def _real():
    import json
    from tests.conftest import FIXTURES
    return json.loads((FIXTURES / "batch_results_real.json").read_text())


def test_5_1_the_parser_handles_a_real_recorded_response():
    """Recorded from an actual Batch API call. Everything above tests MY idea of
    the response shape; this tests the real one. If the fake and reality
    disagree, 40 green census tests are worthless — and settling it cost about
    a third of a cent."""
    from finn_smart_search.understanding.census_prompt import TOOL
    required = set(TOOL["input_schema"]["required"])
    for raw in _real():
        p = ac.parse_result(raw)
        assert p["type"] == "succeeded", p.get("error")
        assert set(p) == {"custom_id", "type", "facets", "usage"}
        missing = required - set(p["facets"])
        assert not missing, f"model omitted required fields: {sorted(missing)}"


def test_5_2_real_usage_carries_the_cache_read_field():
    """Prefix caching is a COST CONTROL, not a nicety: the measured prefix is
    9,217 tokens read at 0.1x. Losing this field would under-report spend and
    silently hide a caching regression that triples the bill."""
    for raw in _real():
        u = ac.parse_result(raw)["usage"]
        assert u["cache_read_input_tokens"] > 5000, "prefix caching is not engaging"
        assert u["output_tokens"] > 0


def test_5_3_real_responses_put_tool_use_first_but_we_do_not_rely_on_it():
    """The recorded responses happen to lead with the tool_use block. Relying on
    that is one preamble away from breaking, so parse_result searches by type
    and name — verified here by re-parsing with a text block prepended."""
    for raw in _real():
        doctored = json.loads(json.dumps(raw))
        doctored["result"]["message"]["content"].insert(
            0, {"type": "text", "text": "Let me analyse this."})
        assert ac.parse_result(doctored)["facets"] == ac.parse_result(raw)["facets"]
