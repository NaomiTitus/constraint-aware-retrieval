"""B2a — census runner mechanics.

Scenario table agreed 2026-09-24. Everything here runs offline against a fake
client: no API calls, no key, so CI stays green without credentials.

Decisions encoded:
  * No truncation. Measured: truncating at 6,000 chars saves $0.15 on a ~$10
    run, and costs the head+tail logic, a `truncated` flag, span validation
    against truncated-rather-than-original text, and a "don't say
    absence_of_requirement when truncated" rule. The longest corpus ad is
    18,081 chars (~5,166 tokens) — trivial for the context window.
  * Demotion rate above 15% aborts the run. If more than one record in seven
    cannot support its own verdict, the prompt is broken and spending the rest
    is throwing money at it.
  * Expired batch requests retry once, then are recorded as failed. Expired
    requests are not billed, so the retry is free.
"""
import json
from datetime import datetime, timezone

import pytest

from finn_smart_search.understanding import census
from finn_smart_search.understanding.census_prompt import PROMPT_VERSION

pytestmark = pytest.mark.unit

MODEL = "claude-haiku-4-5-20251001"

SILENT = {
    "evidence_spans": [], "evidence_basis": "no_mention",
    "evidence_strength": "none", "norwegian_requirement_level": "unstated",
    "stated_working_language": "unstated", "application_language": "unstated",
    "conflicting_statements": False, "implicit_evidence": ["none"],
    "authorisation_required": None, "security_clearance_required": False,
    "visa_sponsorship": "unstated", "relocation_support": "unstated",
    "seniority": "unstated", "min_years_experience": None, "skills": [],
}

VALID = {
    "evidence_spans": [{"span": "Gode norskkunnskaper.", "section_language": "no"}],
    "evidence_basis": "explicit_statement",
    "evidence_strength": "explicit_and_unambiguous",
    "norwegian_requirement_level": "professional",
    "stated_working_language": "unstated", "application_language": "unstated",
    "conflicting_statements": False, "implicit_evidence": ["none"],
    "authorisation_required": None, "security_clearance_required": False,
    "visa_sponsorship": "unstated", "relocation_support": "unstated",
    "seniority": "unstated", "min_years_experience": None, "skills": [],
}
AD_TEXT = "Vi soker medarbeider. Gode norskkunnskaper. Oppstart etter avtale."

# Distinct bodies must differ in WORDS, not digits: canonical() masks digits, so
# "Ad number 0" and "Ad number 1" collapse into one cluster. This caught two of
# my own tests.
OTHER = {
    "a": "Vi soker sykepleier til nattevakt ved sykehjemmet.",
    "b": "Vi soker tomrer til nybygg i sentrum av byen.",
    "c": "Vi soker kokk til restauranten var ved havnen.",
    "d": "Vi soker laerer til barneskolen i kommunen.",
    "e": "Vi soker sjafor med fagbrev til distribusjon.",
    "f": "Vi soker regnskapsforer til okonomiavdelingen.",
    "g": "Vi soker renholder til kontorlokaler pa kveldstid.",
    "h": "Vi soker butikkmedarbeider til klesbutikken.",
    "i": "Vi soker frisor til salongen i gagaten.",
    "j": "Vi soker mekaniker til bilverkstedet vart.",
}



def ad(uuid, text=AD_TEXT, lang="no"):
    return {"uuid": uuid, "title": "Stilling", "description_text": text, "doc_lang": lang}


class FakeClient:
    """Records what it was asked to do and replays canned results."""

    def __init__(self, results=None, statuses=None, reverse=False, mangle_id=False):
        self.submitted = []          # list[list[request]]
        self._results = results or {}
        self._statuses = list(statuses or ["ended"])
        self.reverse = reverse       # batch APIs do not guarantee ordering
        self.mangle_id = mangle_id
        self.calls = 0
        self.polls = 0

    def submit_batch(self, requests):
        self.submitted.append(requests)
        self.calls += len(requests)
        return f"batch_{len(self.submitted)}"

    def poll(self, batch_id):
        self.polls += 1
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    def results(self, batch_id):
        idx = int(batch_id.split("_")[1]) - 1
        reqs = list(reversed(self.submitted[idx])) if self.reverse else self.submitted[idx]
        for req in reqs:
            cid = req["custom_id"]
            if self.mangle_id:
                yield {"custom_id": cid.upper(), "type": "succeeded",
                       "facets": dict(SILENT), "usage": {}}
                continue
            yield self._results.get(cid, {
                "custom_id": cid, "type": "succeeded",
                "facets": dict(SILENT), "usage": {"input_tokens": 1000, "output_tokens": 200},
            })


# ── 1 · request preparation ──────────────────────────────────────────────────

def test_1_1_ads_are_sent_whole_without_truncation():
    long_text = "Norsk. " * 3000          # ~21,000 chars, longer than any corpus ad
    req = census.build_request(ad("a", long_text))
    body = req["params"]["messages"][-1]["content"]
    assert long_text.strip() in body
    assert "[...]" not in body


def test_1_4_doc_lang_is_passed_not_inferred():
    assert 'document_language="en"' in census.build_request(ad("a", lang="en"))["params"]["messages"][-1]["content"]


def test_1_5_request_shape():
    """Derived from FEWSHOT, not hardcoded. A literal 31 breaks the moment an
    example is added — which is a maintenance cost, not a defect signal."""
    from finn_smart_search.understanding.census_prompt import FEWSHOT
    p = census.build_request(ad("a"))["params"]
    assert len(p["messages"]) == len(FEWSHOT) * 3 + 1
    assert p["messages"][-1]["role"] == "user", "the live ad comes last"
    assert p["temperature"] == 0
    assert p["tool_choice"] == {"type": "tool", "name": "record_ad_facets"}


def test_1_5b_fewshot_turns_alternate_correctly():
    """Each example is user(ad) -> assistant(tool_use) -> user(tool_result)."""
    from finn_smart_search.understanding.census_prompt import FEWSHOT
    m = census.build_request(ad("a"))["params"]["messages"]
    for i in range(len(FEWSHOT)):
        assert m[i * 3]["role"] == "user"
        assert m[i * 3 + 1]["content"][0]["type"] == "tool_use"
        assert m[i * 3 + 2]["content"][0]["type"] == "tool_result"


def test_1_6_cache_breakpoint_is_on_the_prefix_not_the_live_ad():
    """The 3,030-token prefix billed at 0.1x is what makes this affordable:
    without caching the same run costs $22.61 instead of $10.45."""
    p = census.build_request(ad("a"))["params"]
    assert "cache_control" in p["system"][0]
    assert "cache_control" not in str(p["messages"][-1])
    assert "cache_control" in str(p["messages"][-2])


def test_1_9_custom_id_is_the_uuid():
    assert census.build_request(ad("abc-123"))["custom_id"] == "abc-123"


# ── 2 · cache ────────────────────────────────────────────────────────────────

def test_2_1_same_ad_twice_is_a_cache_hit(tmp_con):
    census.cache_put(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT, VALID, {"input_tokens": 1, "output_tokens": 1})
    assert census.cache_get(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT) == VALID


def test_2_2_prompt_version_change_is_a_miss(tmp_con):
    census.cache_put(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT, VALID, {})
    assert census.cache_get(tmp_con, MODEL, "census-v99", AD_TEXT) is None


def test_2_3_model_change_is_a_miss(tmp_con):
    census.cache_put(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT, VALID, {})
    assert census.cache_get(tmp_con, "other-model", PROMPT_VERSION, AD_TEXT) is None


def test_2_4_text_change_is_a_miss(tmp_con):
    census.cache_put(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT, VALID, {})
    assert census.cache_get(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT + " Ekstra krav.") is None


def test_2_5_whitespace_only_change_is_a_hit(tmp_con):
    """Keyed on the normalised form, so re-running after an HTML cleaning tweak
    that only moves whitespace does not re-bill the whole census."""
    census.cache_put(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT, VALID, {})
    assert census.cache_get(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT.replace(" ", "  ")) == VALID


def test_2_7_corrupt_cache_row_is_a_miss_not_a_crash(tmp_con):
    tmp_con.execute(
        "INSERT INTO llm_cache (key, model, prompt_version, response, "
        "input_tokens, output_tokens, created_at) VALUES (?,?,?,?,?,?,?)",
        [census.cache_key(MODEL, PROMPT_VERSION, AD_TEXT), MODEL, PROMPT_VERSION,
         "{not json", 0, 0, None])   # truncated write; VARCHAR column permits it
    assert census.cache_get(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT) is None


# ── 3 · batch submission ─────────────────────────────────────────────────────

def test_3_2_results_are_mapped_by_custom_id_not_position(tmp_con):
    b_text = "Vi soker tomrer til nybygg. Helst bestatt norskprove B1. Oppstart snarest."
    out = census.run([ad("a"), ad("b", b_text)], FakeClient(results={
        "b": {"custom_id": "b", "type": "succeeded",
              "facets": dict(SILENT, norwegian_requirement_level="certified",
                             evidence_basis="explicit_statement",
                             evidence_strength="explicit_and_unambiguous",
                             evidence_spans=[{"span": "Helst bestatt norskprove B1.",
                                              "section_language": "no"}]),
              "usage": {"input_tokens": 1, "output_tokens": 1}}}), tmp_con)
    assert out["facets"]["a"]["norwegian_requirement_level"] == "unstated"
    assert out["facets"]["b"]["norwegian_requirement_level"] == "certified"


def test_3_3_missing_custom_id_raises(tmp_con):
    class Dropping(FakeClient):
        def results(self, batch_id):
            yield from list(super().results(batch_id))[:-1]
    with pytest.raises(census.MissingResultError):
        census.run([ad("a"), ad("b", OTHER["b"])], Dropping(), tmp_con)


def test_3_4_errored_result_is_recorded_and_the_run_continues(tmp_con):
    b_text = "Vi soker tomrer til nybygg. Helst bestatt norskprove B1. Oppstart snarest."
    out = census.run([ad("a"), ad("b", b_text)], FakeClient(results={
        "a": {"custom_id": "a", "type": "errored", "error": "overloaded"}}), tmp_con)
    assert "a" in out["failed"] and "b" in out["facets"]


def test_3_5_expired_results_are_retried_once_then_failed(tmp_con):
    expired = {"custom_id": "a", "type": "expired"}
    client = FakeClient(results={"a": expired})
    out = census.run([ad("a"), ad("b", OTHER["b"])], client, tmp_con)
    assert len(client.submitted) == 2, "expired must be retried in a second batch"
    assert "a" in out["failed"]


# ── 4 · validation wiring ────────────────────────────────────────────────────

def test_4_2_falsified_span_is_demoted_and_counted(tmp_con):
    bad = dict(VALID, evidence_spans=[{"span": "Our working language is English",
                                       "section_language": "en"}])
    out = census.run([ad("a")], FakeClient(results={
        "a": {"custom_id": "a", "type": "succeeded", "facets": bad,
              "usage": {"input_tokens": 1, "output_tokens": 1}}}), tmp_con,
        max_demotion_rate=1.0)        # this test measures counting, not the threshold
    assert out["facets"]["a"]["norwegian_requirement_level"] == "unstated"
    assert out["demoted"] == 1


def test_4_4_demotion_rate_above_threshold_aborts(tmp_con):
    """A high rate means the prompt is broken; spending the rest is throwing
    money at it."""
    bad = dict(VALID, evidence_spans=[{"span": "Invented sentence entirely", "section_language": "no"}])
    ads = [ad(str(i)) for i in range(20)]
    results = {str(i): {"custom_id": str(i), "type": "succeeded", "facets": bad,
                        "usage": {"input_tokens": 1, "output_tokens": 1}} for i in range(20)}
    with pytest.raises(census.DemotionRateError):
        census.run(ads, FakeClient(results=results), tmp_con, max_demotion_rate=0.15)


# ── 5 · fan-out and cost ─────────────────────────────────────────────────────

def test_5_1_one_call_per_cluster_fanned_to_every_member(tmp_con):
    """A 44-ad KIWI cluster costs one call, and all 44 rows are written."""
    ads = [ad(u) for u in ("a", "b", "c")] + [ad("d", OTHER["d"])]
    client = FakeClient()
    out = census.run(ads, client, tmp_con)
    assert client.calls == 2, "three identical bodies must cost one call"
    assert set(out["facets"]) == {"a", "b", "c", "d"}


def test_5_3_spend_is_tracked_from_real_usage(tmp_con):
    out = census.run([ad("a")], FakeClient(results={
        "a": {"custom_id": "a", "type": "succeeded", "facets": dict(VALID),
              "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}}}), tmp_con)
    assert out["spend_usd"] == pytest.approx(0.50 + 2.50, rel=1e-6)


def test_5_4_spend_over_the_cap_aborts(tmp_con):
    keys = list(OTHER)
    ads = [ad(k, OTHER[k]) for k in keys]
    results = {k: {"custom_id": k, "type": "succeeded", "facets": dict(SILENT),
                   "usage": {"input_tokens": 1_000_000, "output_tokens": 0}} for k in keys}
    with pytest.raises(census.SpendLimitError):
        census.run(ads, FakeClient(results=results), tmp_con, max_spend_usd=1.0)


def test_5_5_limit_caps_representatives(tmp_con):
    ads = [ad(k, OTHER[k]) for k in OTHER]
    client = FakeClient()
    census.run(ads, client, tmp_con, limit=3)
    assert client.calls == 3


# ── 6 · output ───────────────────────────────────────────────────────────────

def test_6_1_rows_carry_version_fields(tmp_con):
    census.run([ad("a")], FakeClient(), tmp_con)
    row = tmp_con.execute("SELECT prompt_version, extractor_version FROM ad_facets WHERE uuid='a'").fetchone()
    assert row[0] == PROMPT_VERSION and row[1]


def test_6_2_rerun_is_idempotent(tmp_con):
    census.run([ad("a")], FakeClient(), tmp_con)
    census.run([ad("a")], FakeClient(), tmp_con)
    assert tmp_con.execute("SELECT count(*) FROM ad_facets WHERE uuid='a'").fetchone()[0] == 1


def test_6_3_english_accessible_is_derived_not_read_from_the_model(tmp_con):
    """The model never emits this field. A disjunction in a Norwegian-written ad
    must derive True — the 274-ad correction."""
    disj = dict(SILENT, norwegian_requirement_level="either_norwegian_or_english",
                evidence_basis="explicit_statement",
                evidence_strength="explicit_and_unambiguous",
                evidence_spans=[{"span": "Behersker norsk eller engelsk.", "section_language": "no"}])
    out = census.run([ad("a", "Krav: Behersker norsk eller engelsk. Oppstart snarest.")],
                     FakeClient(results={"a": {"custom_id": "a", "type": "succeeded",
                                               "facets": disj,
                                               "usage": {"input_tokens": 1, "output_tokens": 1}}}), tmp_con)
    assert out["facets"]["a"]["english_accessible"] is True


def test_6_4_summary_reports_the_numbers_that_matter(tmp_con):
    out = census.run([ad("a"), ad("b", OTHER["b"])], FakeClient(), tmp_con)
    for k in ("calls", "cache_hits", "demoted", "demotion_rate", "spend_usd", "levels", "failed"):
        assert k in out, f"summary must report {k}"


# ── gaps found by external review + an honest mutation run ──────────────────

def test_results_are_mapped_by_id_when_returned_OUT_OF_ORDER(tmp_con):
    """The fake previously returned results in submission order, so position
    mapping and custom_id mapping were indistinguishable. Reversing makes the
    difference observable."""
    b_text = "Vi soker tomrer til nybygg. Helst bestatt norskprove B1. Oppstart snarest."
    certified = dict(SILENT, norwegian_requirement_level="certified",
                     evidence_basis="explicit_statement",
                     evidence_strength="explicit_and_unambiguous",
                     evidence_spans=[{"span": "Helst bestatt norskprove B1.",
                                      "section_language": "no"}])
    out = census.run([ad("a"), ad("b", b_text)],
                     FakeClient(reverse=True, results={
                         "b": {"custom_id": "b", "type": "succeeded", "facets": certified,
                               "usage": {"input_tokens": 1, "output_tokens": 1}}}), tmp_con)
    assert out["facets"]["a"]["norwegian_requirement_level"] == "unstated"
    assert out["facets"]["b"]["norwegian_requirement_level"] == "certified"


def test_unknown_custom_id_in_results_raises(tmp_con):
    with pytest.raises(census.MissingResultError):
        census.run([ad("a")], FakeClient(mangle_id=True), tmp_con)


def test_cache_hit_avoids_the_api_call_entirely(tmp_con):
    """The only end-to-end proof that caching works. Counting rows after two
    runs is a tautology — INSERT OR REPLACE gives one row whether run two was a
    hit or a full re-bill. Without this, a disabled cache costs $22.61."""
    ads = [ad("a"), ad("b", OTHER["b"])]
    c1 = FakeClient(); first = census.run(ads, c1, tmp_con)
    c2 = FakeClient(); second = census.run(ads, c2, tmp_con)
    assert second["calls"] == 0
    assert second["cache_hits"] == 2
    assert second["spend_usd"] == 0.0
    assert c2.submitted == [], "a fully cached run must not submit a batch at all"
    assert second["facets"] == first["facets"]


def test_cache_stores_the_validated_record_not_the_raw_model_output(tmp_con):
    """Caching before validation would store a falsified verdict forever, and
    every rerun would skip validation and fan the lie out to the whole cluster."""
    bad = dict(SILENT, norwegian_requirement_level="certified",
               evidence_basis="explicit_statement",
               evidence_strength="explicit_and_unambiguous",
               evidence_spans=[{"span": "Invented sentence entirely.", "section_language": "no"}])
    res = {"a": {"custom_id": "a", "type": "succeeded", "facets": bad,
                 "usage": {"input_tokens": 1, "output_tokens": 1}}}
    census.run([ad("a")], FakeClient(results=res), tmp_con, max_demotion_rate=1.0)
    second = census.run([ad("a")], FakeClient(results=res), tmp_con, max_demotion_rate=1.0)
    assert second["cache_hits"] == 1
    assert second["facets"]["a"]["norwegian_requirement_level"] == "unstated"


def test_spend_uses_asymmetric_prices(tmp_con):
    """1M in AND 1M out is symmetric: swapping PRICE_IN and PRICE_OUT gives the
    same total, so the original test could not detect it."""
    out = census.run([ad("a")], FakeClient(results={
        "a": {"custom_id": "a", "type": "succeeded", "facets": dict(SILENT),
              "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}}), tmp_con)
    assert out["spend_usd"] == pytest.approx(0.50, rel=1e-6)


def test_demotion_rate_is_over_new_records_not_all_representatives(tmp_con):
    """With a warm cache, dividing by len(reps) makes one demotion out of one
    new call read as a small fraction — and a broken prompt runs to completion."""
    ads = [ad(k, OTHER[k]) for k in list(OTHER)[:10]]
    census.run(ads[:9], FakeClient(), tmp_con)          # warm the cache
    bad = dict(SILENT, norwegian_requirement_level="certified",
               evidence_basis="explicit_statement",
               evidence_strength="explicit_and_unambiguous",
               evidence_spans=[{"span": "Invented sentence entirely.", "section_language": "no"}])
    last = ads[9]["uuid"]
    with pytest.raises(census.DemotionRateError):
        census.run(ads, FakeClient(results={
            last: {"custom_id": last, "type": "succeeded", "facets": bad,
                   "usage": {"input_tokens": 1, "output_tokens": 1}}}), tmp_con)


def test_cache_key_components_cannot_collide(tmp_con):
    """A ":" separator makes model="haiku:v1"/pv="x" collide with
    model="haiku"/pv="v1:x", serving one version's facets under another."""
    assert census.cache_key("haiku:v1", "x", AD_TEXT) != census.cache_key("haiku", "v1:x", AD_TEXT)


def test_poll_accepts_completed_as_well_as_ended(tmp_con):
    census.run([ad("a")], FakeClient(statuses=["completed"]), tmp_con)


def test_poll_waits_for_a_terminal_state(tmp_con):
    c = FakeClient(statuses=["in_progress", "in_progress", "ended"])
    census.run([ad("a")], c, tmp_con)
    assert c.polls == 3


def test_terminal_failure_state_raises_instead_of_hanging(tmp_con):
    """"canceled" is terminal but not successful. The first version looped while
    status not in ("ended","completed"), so it span forever — a busy loop at
    100% CPU with poll_seconds=0.0, no timeout, no error."""
    c = FakeClient(statuses=["canceled"])
    with pytest.raises(census.BatchFailedError):
        census.run([ad("a")], c, tmp_con)
    # Must fail FAST. Without the DEAD_STATES check it still raises, but only
    # after exhausting max_polls — 5,000 wasted polls, and in production a busy
    # loop. Asserting the exception alone passes for the wrong reason.
    assert c.polls == 1


def test_doc_lang_is_used_when_deriving_accessibility(tmp_con):
    """A hardcoded "no" would misclassify every English-written ad."""
    out = census.run([ad("a", "We are hiring a barista for our coffee bar.", lang="en")],
                     FakeClient(), tmp_con)
    assert out["facets"]["a"]["english_accessible"] is True
    out2 = census.run([ad("z", "Vi soker barista til kaffebaren var.", lang="no")],
                      FakeClient(), tmp_con)
    assert out2["facets"]["z"]["english_accessible"] is False


def test_every_ad_is_accounted_for(tmp_con):
    """Conservation. The `live` filter and the limit filter both key on min(u);
    if a representative were ever not the lexicographic minimum, whole clusters
    would vanish with no error."""
    ads = [ad("a"), ad("b"), ad("c", OTHER["c"]), ad("d", OTHER["d"])]
    out = census.run(ads, FakeClient(results={
        "c": {"custom_id": "c", "type": "errored", "error": "overloaded"}}), tmp_con)
    assert set(out["facets"]) | set(out["failed"]) == {a["uuid"] for a in ads}


def test_fan_out_spreads_a_non_default_verdict(tmp_con):
    """The original asserted only that members appear in the output, while every
    fake result was SILENT — so members and representative were trivially equal."""
    text = "Krav: Behersker norsk eller engelsk. Oppstart snarest."
    disj = dict(SILENT, norwegian_requirement_level="either_norwegian_or_english",
                evidence_basis="explicit_statement",
                evidence_strength="explicit_and_unambiguous",
                evidence_spans=[{"span": "Behersker norsk eller engelsk.",
                                 "section_language": "no"}])
    ads = [ad(u, text) for u in ("a", "b", "c")]
    out = census.run(ads, FakeClient(results={
        "a": {"custom_id": "a", "type": "succeeded", "facets": disj,
              "usage": {"input_tokens": 1, "output_tokens": 1}}}), tmp_con)["facets"]
    assert all(out[u]["norwegian_requirement_level"] == "either_norwegian_or_english"
               for u in ("a", "b", "c"))
    assert all(out[u]["english_accessible"] is True for u in ("a", "b", "c"))
    assert "_reasons" not in out["a"], "internal bookkeeping must not leak into facets"


def test_retry_resubmits_only_the_expired_request(tmp_con):
    """len(submitted)==2 passes even if the retry re-sends every request,
    re-billing the whole census."""
    client = FakeClient(results={"a": {"custom_id": "a", "type": "expired"}})
    census.run([ad("a"), ad("b", OTHER["b"])], client, tmp_con)
    assert [r["custom_id"] for r in client.submitted[1]] == ["a"]


def test_long_ad_is_sent_in_full(tmp_con):
    """"[...]" not in body catches one marker spelling; a silent body[:8000]
    would pass."""
    long_text = "Norsk. " * 3000
    body = census.build_request(ad("a", long_text))["params"]["messages"][-1]["content"]
    assert long_text.strip() in body and len(body) >= 21_000


# ── the cache must survive a VALIDATOR change, not only a prompt change ──────
#
# WHAT THIS COST. The 50-ad pilot demoted 55% of records. The cause was a bug in
# the span validator, not in the prompt. Fixing it should have been free to
# re-measure -- the model's answers were fine, only the judgement of them was
# wrong -- but cache_put stored `checked["facets"]`, the POST-validation record,
# and discarded the raw response. The cache key is the ad text, so a plain
# re-run served the demoted records straight back, and the only way to
# re-measure was to clear the cache and pay the API again.
#
# The original design was a deliberate answer to a real objection: caching the
# RAW response and returning it unvalidated would store a falsified verdict
# forever and skip validation on every rerun. Storing both satisfies that
# objection and this one -- nothing unvalidated is ever served, and a validator
# change can be re-evaluated for $0.
#
# At 44 ads this cost $0.06. At 9,823 clusters it costs $14.50, and validator
# bugs are the likeliest reason to need a re-run.

RAW_WITH_BAD_SPAN = dict(
    VALID,
    evidence_spans=[{"span": "norsk", "section_language": "no"}],   # too short
)


def test_the_raw_response_is_stored_alongside_the_validated_one(tmp_con):
    """The core of the fix. Both must be retrievable from one cache row."""
    census.cache_put(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT, VALID, {},
                     raw_facets=RAW_WITH_BAD_SPAN)
    assert census.cache_get(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT) == VALID
    assert census.cache_get_raw(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT) == RAW_WITH_BAD_SPAN


def test_cache_get_still_returns_the_validated_record(tmp_con):
    """Non-negotiable: an unvalidated record must never be served. This is the
    objection that motivated the original post-validation-only design and it
    still holds."""
    census.cache_put(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT, VALID, {},
                     raw_facets=RAW_WITH_BAD_SPAN)
    got = census.cache_get(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT)
    assert got != RAW_WITH_BAD_SPAN
    assert got == VALID


def test_revalidate_rewrites_verdicts_with_no_api_call(tmp_con):
    """The operation that was impossible. Store a record validated under an
    over-strict rule, then re-validate against the current one and see the
    verdict change -- without a client, so an accidental API call cannot pass."""
    strict = dict(VALID, evidence_spans=[])          # as if everything was stripped
    census.cache_put(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT, strict, {},
                     raw_facets=VALID)
    n = census.revalidate_cache(tmp_con, {census.cache_key(MODEL, PROMPT_VERSION, AD_TEXT): AD_TEXT})
    assert n == 1
    assert census.cache_get(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT)["evidence_spans"] \
        == VALID["evidence_spans"]


def test_revalidate_is_idempotent(tmp_con):
    census.cache_put(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT, VALID, {}, raw_facets=VALID)
    keys = {census.cache_key(MODEL, PROMPT_VERSION, AD_TEXT): AD_TEXT}
    census.revalidate_cache(tmp_con, keys)
    first = census.cache_get(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT)
    census.revalidate_cache(tmp_con, keys)
    assert census.cache_get(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT) == first


def test_a_row_with_no_raw_response_is_left_alone_by_revalidate(tmp_con):
    """Rows written before this change have no raw column. They must be skipped,
    not crash and not be silently blanked -- the whole point is not to lose
    paid-for work."""
    tmp_con.execute(
        "INSERT INTO llm_cache (key, model, prompt_version, response, "
        "input_tokens, output_tokens, created_at) VALUES (?,?,?,?,?,?,?)",
        [census.cache_key(MODEL, PROMPT_VERSION, AD_TEXT), MODEL, PROMPT_VERSION,
         json.dumps(VALID), 0, 0, datetime.now(timezone.utc)])
    n = census.revalidate_cache(
        tmp_con, {census.cache_key(MODEL, PROMPT_VERSION, AD_TEXT): AD_TEXT})
    assert n == 0
    assert census.cache_get(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT) == VALID


def test_cache_put_without_raw_still_works(tmp_con):
    """Backwards compatible: raw_facets is optional, so no caller breaks."""
    census.cache_put(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT, VALID, {})
    assert census.cache_get(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT) == VALID
    assert census.cache_get_raw(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT) is None


def test_an_existing_seven_column_cache_is_migrated_not_recreated(tmp_con):
    """CREATE TABLE IF NOT EXISTS does NOT add a column to a table that already
    exists, so the corpus database -- which already holds a 7-column llm_cache
    -- would keep failing every INSERT until it is migrated.

    The migration must be additive: rows already paid for must survive it.
    """
    tmp_con.execute("DROP TABLE IF EXISTS llm_cache")
    tmp_con.execute("""CREATE TABLE llm_cache (
        key VARCHAR PRIMARY KEY, model VARCHAR, prompt_version VARCHAR,
        response VARCHAR, input_tokens INTEGER, output_tokens INTEGER,
        created_at TIMESTAMPTZ)""")
    key = census.cache_key(MODEL, PROMPT_VERSION, AD_TEXT)
    tmp_con.execute(
        "INSERT INTO llm_cache (key, model, prompt_version, response, "
        "input_tokens, output_tokens, created_at) VALUES (?,?,?,?,?,?,?)",
        [key, MODEL, PROMPT_VERSION, json.dumps(VALID), 7, 3,
         datetime.now(timezone.utc)])

    census.ensure_schema(tmp_con)

    cols = [r[0] for r in tmp_con.execute("DESCRIBE llm_cache").fetchall()]
    assert "response_raw" in cols
    # the paid-for row survived, with its token counts
    row = tmp_con.execute("SELECT response, input_tokens, response_raw "
                          "FROM llm_cache WHERE key = ?", [key]).fetchone()
    assert json.loads(row[0]) == VALID and row[1] == 7 and row[2] is None
    # and the new path works against the migrated table
    census.cache_put(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT + " Nytt.", VALID, {},
                     raw_facets=VALID)
    assert census.cache_get_raw(tmp_con, MODEL, PROMPT_VERSION, AD_TEXT + " Nytt.") == VALID


def test_ensure_schema_is_idempotent(tmp_con):
    census.ensure_schema(tmp_con)
    census.ensure_schema(tmp_con)
    cols = [r[0] for r in tmp_con.execute("DESCRIBE llm_cache").fetchall()]
    assert cols.count("response_raw") == 1


def test_run_stores_the_raw_response_so_a_validator_change_is_free(tmp_con):
    """The mutation that survived everything above: deleting `raw_facets=facets`
    from run()'s cache_put left every other cache test green, so the mechanism
    could be dead in production while the unit tests all passed.

    This is the end-to-end proof. run() demotes a fabricated span, the demoted
    verdict is what gets served -- and the raw answer is still there to be
    re-judged if the VALIDATOR later turns out to have been wrong, which is
    exactly what happened in the 50-ad pilot.
    """
    bad = dict(SILENT, norwegian_requirement_level="certified",
               evidence_basis="explicit_statement",
               evidence_strength="explicit_and_unambiguous",
               evidence_spans=[{"span": "Invented sentence entirely.", "section_language": "no"}])
    res = {"a": {"custom_id": "a", "type": "succeeded", "facets": bad,
                 "usage": {"input_tokens": 1, "output_tokens": 1}}}
    census.run([ad("a")], FakeClient(results=res), tmp_con, max_demotion_rate=1.0)

    text = ad("a")["description_text"]
    served = census.cache_get(tmp_con, census.MODEL, PROMPT_VERSION, text)
    raw = census.cache_get_raw(tmp_con, census.MODEL, PROMPT_VERSION, text)

    assert served["norwegian_requirement_level"] == "unstated", "the demotion must be served"
    assert raw is not None, "run() must store the raw answer"
    assert raw["norwegian_requirement_level"] == "certified", \
        "the raw answer must be the model's own, before demotion"
    assert raw["evidence_spans"] == bad["evidence_spans"]


def test_revalidating_a_run_recovers_verdicts_without_a_client(tmp_con):
    """The whole point, exercised through run(): after a validator fix, the
    previously demoted record can be re-judged for $0. Simulated by demoting
    under a stub validator that rejects everything, then revalidating under the
    real one."""
    good_span = "Behersker norsk eller engelsk."
    facets = dict(SILENT, norwegian_requirement_level="either_norwegian_or_english",
                  evidence_basis="explicit_statement",
                  evidence_strength="explicit_and_unambiguous",
                  evidence_spans=[{"span": good_span, "section_language": "no"}])
    res = {"a": {"custom_id": "a", "type": "succeeded", "facets": facets,
                 "usage": {"input_tokens": 1, "output_tokens": 1}}}
    text = good_span + "\nVi tilbyr opplæring."
    a = dict(ad("a"), description_text=text)
    census.run([a], FakeClient(results=res), tmp_con, max_demotion_rate=1.0)

    # a validator bug demoted it; overwrite the served record to simulate that
    key = census.cache_key(census.MODEL, PROMPT_VERSION, text)
    tmp_con.execute("UPDATE llm_cache SET response = ? WHERE key = ?",
                    [json.dumps(dict(facets, norwegian_requirement_level="unstated",
                                     evidence_spans=[])), key])
    assert census.cache_get(tmp_con, census.MODEL, PROMPT_VERSION,
                            text)["norwegian_requirement_level"] == "unstated"

    assert census.revalidate_cache(tmp_con, {key: text}) == 1
    back = census.cache_get(tmp_con, census.MODEL, PROMPT_VERSION, text)
    assert back["norwegian_requirement_level"] == "either_norwegian_or_english"
    assert back["evidence_spans"][0]["span"] == good_span
