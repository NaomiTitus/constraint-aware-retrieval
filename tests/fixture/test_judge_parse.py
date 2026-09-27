"""Judge response parsing, tier 2 — against the RECORDED envelope. Scenarios 17-19.

Grounded in `tests/fixtures/batch_results_real.json` (JUDGE_SCENARIOS G3), three
live Batch API rows. The envelope is the translation `anthropic_client` docstring
calls "the fiddly one": the tool input is nested in `result.message.content[]` as a
`tool_use` block, so positional indexing breaks the moment the model emits any
preamble. Reusing `parse_result` rather than re-deriving that is the point.
"""
from __future__ import annotations

import copy

import pytest

from finn_smart_search.eval import judge_llm as J
from tests.conftest import recorded_batch_results

pytestmark = pytest.mark.fixture

JUDGMENT = {"grade": 2, "violates": True, "violated_facet": "language",
            "constraint_evidence": "Gode norskkunnskaper er et krav",
            "notes": "right occupation, blocked by language"}


def _envelope(**over):
    """Take a REAL recorded row and swap the tool block for the judge's, so the
    envelope shape — message keys, stop_reason, usage — stays exactly as the API
    produced it rather than as I imagine it."""
    row = copy.deepcopy(recorded_batch_results()[0])
    row["custom_id"] = "p1_sykepleier_no_norsk::0f2c"
    msg = row["result"]["message"]
    msg["content"] = [{"type": "tool_use", "id": "tu_1",
                       "name": J.JUDGE_TOOL_NAME, "input": dict(JUDGMENT)}]
    for k, v in over.items():
        msg[k] = v
    return row


def test_a_recorded_envelope_parses_into_a_judgment():
    """Scenario 17."""
    got = J.parse_judgment(_envelope())
    assert isinstance(got, J.Judgment)
    assert got.grade == 2
    assert got.violates is True
    assert got.violated_facet == "language"
    assert got.pair_id == "p1_sykepleier_no_norsk::0f2c"


def test_a_truncated_response_errors_and_is_never_salvaged():
    """Scenario 18. The client's standing ruling: `stop_reason=max_tokens` means
    the JSON was cut mid-structure, and a half-parsed judgment is indistinguishable
    from a real one downstream."""
    got = J.parse_judgment(
        _envelope(stop_reason="max_tokens"))
    assert not isinstance(got, J.Judgment)
    assert got["type"] == "errored"


def test_the_wrong_tool_name_errors():
    """Scenario 19. The census tool and the judge tool have different contracts;
    accepting either would let a facet record be stored as a judgment."""
    row = _envelope()
    row["result"]["message"]["content"][0]["name"] = "record_ad_facets"
    got = J.parse_judgment(row)
    assert not isinstance(got, J.Judgment)
    assert got["type"] == "errored"


def test_a_failed_request_errors_rather_than_returning_a_judgment():
    row = _envelope()
    row["result"] = {"type": "errored", "error": "overloaded"}
    got = J.parse_judgment(row)
    assert not isinstance(got, J.Judgment)
    assert got["type"] == "errored"
