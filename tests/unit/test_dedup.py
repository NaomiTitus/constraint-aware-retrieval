"""Group B — near-duplicate clustering by normalised-text exact hash.

Scenario table agreed 2026-09-24. Supersedes the MinHash implementation (D6).

Why exact hashing rather than MinHash. Four strategies measured over the full
corpus: raw HTML 249 redundant (2.4%), cleaned text 320 (3.1%), normalised text
345 (3.4%), MinHash 423 (4.2%). MinHash's extra 78 merges were inspected and are
legitimate — but they sit in clusters of 2-6, where error amplification is
negligible. The large clusters that actually matter (44, 32, 21) are caught
identically by every strategy. Exact hashing is explainable in one sentence,
deterministic, and has zero false merges by construction.

The purpose is NOT cost. Deduping saves about $0.12 on a $3 census. It exists to
stop one template mistake becoming 44 corpus rows that look like consistent
signal and survive random-sample evaluation.
"""
import pytest

from tests.conftest import requires_corpus

from finn_smart_search.understanding import dedup

pytestmark = pytest.mark.unit

# Pinned goldens, measured after the Group A block-traversal change.
EXPECTED_CLUSTERS = 9823
EXPECTED_REDUNDANT = 343
EXPECTED_LARGEST = 44

BODY = "Vi soker en dyktig medarbeider til vart team i Oslo. Gode norskkunnskaper kreves."


def ad(uuid, body, title="Stilling"):
    return {"uuid": uuid, "title": title, "description_text": body}


# ── B1-B5 · what must collapse into one cluster ──────────────────────────────

def test_b1_identical_body_different_titles_cluster_together():
    """The KIWI case. Chain stores vary the title per location while the body is
    byte-identical; a key including the title found only 157 redundant ads,
    WORSE than plain exact hashing at 320."""
    g = dedup.cluster([ad("a", BODY, "KIWI Mørkvedvegen"), ad("b", BODY, "KIWI Frognerveien")])
    assert len(g) == 1 and sorted(next(iter(g.values()))) == ["a", "b"]


def test_b2_digits_are_masked():
    assert dedup.signature("Vi soker 5 stillinger") == dedup.signature("Vi soker 7 stillinger")
    assert dedup.signature("Vi soker 5 stillinger") == dedup.signature("Vi soker 15 stillinger")


def test_b3_case_is_folded():
    assert dedup.signature("GODE NORSKKUNNSKAPER") == dedup.signature("gode norskkunnskaper")


def test_b4_punctuation_is_collapsed():
    assert dedup.signature("norsk, engelsk") == dedup.signature("norsk - engelsk")
    assert dedup.signature("norsk engelsk!") == dedup.signature("norsk engelsk?")


def test_b5_unicode_variants_are_folded():
    """Escapes, not literals. Two visually identical strings assert nothing a
    reviewer can check, and silently degrade to `x == x` if one is ever
    re-typed. The invisible U+00AD in the previous version also defeated
    programmatic editing of this very file."""
    assert dedup.signature("Gode\u00a0norskkunnskaper") == dedup.signature("Gode norskkunnskaper")
    assert dedup.signature("norsk\u00adkunnskaper") == dedup.signature("norskkunnskaper")
    assert dedup.signature("O\ufb03siell") == dedup.signature("Offisiell")


def test_canonical_preserves_norwegian_letters():
    """Assert the canonical FORM, not merely that two strings differ. An
    ASCII-only \\w class turns "Ålesund" into " lesund" — the strings still
    differ, so an inequality assertion passes while every Norwegian word is
    quietly mangled."""
    assert dedup.canonical("Sykepleier søkes til Ålesund") == "sykepleier søkes til ålesund"
    assert dedup.canonical("Tømrer på Nøtterøy, 3 år") == "tømrer på nøtterøy # år"


# ── B6-B7 · what must stay apart ─────────────────────────────────────────────

def test_b6_near_duplicates_differing_by_a_sentence_stay_apart():
    """ACCEPTED LOSS, documented in DECISIONS.md. The Adecco case differed by
    ~20 characters and MinHash merged it; exact hashing cannot span that."""
    a = BODY
    b = BODY + " Vi tilbyr konkurransedyktige betingelser."
    assert dedup.signature(a) != dedup.signature(b)


def test_b7_unrelated_bodies_stay_apart():
    assert dedup.signature("Sykepleier til nattevakt") != dedup.signature("Tomrer soekes til nybygg")


def test_b10_differing_by_one_word_stays_apart():
    """Zero false merges by construction — the property MinHash could not offer."""
    assert dedup.signature("norsk kreves") != dedup.signature("norsk onskes")


# ── B8-B9 · robustness ───────────────────────────────────────────────────────

@pytest.mark.parametrize("empty", ["", "   ", None])
def test_b8_empty_body_does_not_raise(empty):
    assert isinstance(dedup.signature(empty), str)


def test_b8b_empty_bodies_cluster_together_and_do_not_absorb_real_ads():
    g = dedup.cluster([ad("a", ""), ad("b", "   "), ad("c", BODY)])
    # len(g) == 2 is also consistent with {"a","c"} + {"b"}. Pin membership.
    assert sorted(g.values()) == [["a", "b"], ["c"]]


def test_b9_signature_is_a_stable_golden_value():
    """A GOLDEN VALUE, not a shape check. `len(s) == 64` plus a hex-alphabet
    test passes for sha512[:64], latin-1 encoding, or a versioned prefix — none
    of which change the partition, so no other test notices either. All of them
    invalidate the entire LLM cache and re-bill the census silently.

    Regenerate this literal DELIBERATELY, never to make a red test green."""
    assert dedup.signature("Gode norskkunnskaper.") == "2b02f9402c14344b242b98b52e00b097f868c133def2033e5f93207eb2530319"
    assert dedup.signature(BODY) == dedup.signature(BODY)


# ── B11-B12 · representatives and fan-out ────────────────────────────────────

def test_cluster_returns_uuids_sorted_regardless_of_input_order():
    """Cluster CONTENTS must not depend on input order, or fan_out picking
    uuids[0] silently selects a different representative between runs and the
    LLM cache keys stop lining up."""
    fwd = dedup.cluster([ad("c", BODY), ad("a", BODY), ad("b", BODY)])
    rev = dedup.cluster([ad("b", BODY), ad("c", BODY), ad("a", BODY)])
    assert list(fwd.values()) == [["a", "b", "c"]]
    assert fwd == rev


def test_fan_out_representative_does_not_depend_on_list_order():
    """Guards fan_out against uuids[0] when a caller hands it unsorted groups."""
    groups = {"sig": ["c", "a", "b"]}
    out = dedup.fan_out(groups, {"a": "from-a"})
    assert out == {"a": "from-a", "b": "from-a", "c": "from-a"}


def test_b11_one_representative_per_cluster_chosen_stably():
    ads = [ad("b", BODY), ad("a", BODY), ad("c", "Noe helt annet her")]
    g = dedup.cluster(ads)
    reps = dedup.representatives(g)
    assert len(reps) == 2
    # deterministic choice, not dict insertion order
    assert dedup.representatives(dedup.cluster(list(reversed(ads)))) == reps


def test_b12_fan_out_gives_every_member_the_representative_result():
    ads = [ad("a", BODY), ad("b", BODY), ad("c", "Noe helt annet her")]
    g = dedup.cluster(ads)
    reps = dedup.representatives(g)
    results = {r: {"level": "professional", "from": r} for r in reps}
    out = dedup.fan_out(g, results)
    assert set(out) == {"a", "b", "c"}
    assert out["a"] == out["b"], "cluster members must share the representative's facets"
    assert out["c"] != out["a"]


def test_b12b_fan_out_raises_when_a_representative_is_missing():
    """Silently dropping ads whose representative failed extraction would shrink
    the corpus without anyone noticing."""
    g = dedup.cluster([ad("a", BODY), ad("b", BODY)])
    with pytest.raises(KeyError):
        dedup.fan_out(g, {})


# ── B13 · corpus characterisation ────────────────────────────────────────────

@pytest.mark.integration
@requires_corpus
def test_b13_corpus_clustering():
    """Pinned against the real corpus. Requires data/ads.duckdb, so it is marked
    integration and excluded from CI."""
    from tests.conftest import open_corpus

    # READ-ONLY, lock-tolerant: store.connect() takes an exclusive lock and runs
    # DDL against the corpus this test is auditing.
    con = open_corpus()
    rows = con.execute("SELECT uuid, description_text FROM ads WHERE n_chars > 0").fetchall()
    ads = [{"uuid": u, "description_text": t} for u, t in rows]
    g = dedup.cluster(ads)
    assert len(g) == EXPECTED_CLUSTERS
    assert max(len(v) for v in g.values()) == EXPECTED_LARGEST
    # Invariants, not just counts: nothing lost, nothing duplicated.
    members = [u for v in g.values() for u in v]
    assert len(members) == len(ads), "clustering must not lose ads"
    assert len(set(members)) == len(ads), "an ad must appear in exactly one cluster"
    # Cache stability at corpus scale, not just on three toy ads.
    assert dedup.representatives(g) == dedup.representatives(dedup.cluster(list(reversed(ads))))


# ── gaps found by external review + mutation testing ────────────────────────

def test_canonical_strips_trailing_punctuation_residue():
    """_NON_WORD.sub(" ", ...) re-introduces a trailing space AFTER normalise
    has stripped. Without the final .strip(), ads differing only by a closing
    "." or ":" split into separate clusters — and every signature in the corpus
    changes, invalidating the whole LLM cache."""
    assert dedup.canonical("Gode norskkunnskaper.") == "gode norskkunnskaper"
    assert dedup.canonical("Krav:") == "krav"
    assert dedup.signature("Gode norskkunnskaper.") == dedup.signature("Gode norskkunnskaper")


def test_cefr_levels_survive_digit_masking():
    """B1 and B2 are DIFFERENT language requirements. A blanket \\d+ mask merges
    them, handing two ads one verdict on the attribute this pipeline exists to
    read. 1,268 corpus ads carry a CEFR token."""
    assert dedup.signature("Krav: bestått norskprøve B1") != dedup.signature("Krav: bestått norskprøve B2")
    assert dedup.canonical("norskprøve B2, 3 år") == "norskprøve b2 # år"


def test_digit_masking_accepted_over_merge():
    """ACCEPTED TRADE-OFF, pinned so a future widening of the mask is visible.
    Percentages, salaries and dates ARE merged — two ads identical but for the
    stillingsprosent share their language requirements, which is what the
    census reads, so merging them is correct for this purpose."""
    assert dedup.signature("Stilling 20 % fast") == dedup.signature("Stilling 100 % fast")
    assert dedup.signature("kr 450 000") == dedup.signature("kr 650 000")


def test_long_shared_prefix_does_not_over_merge():
    """The only unit-level detector of over-eager clustering. Truncating the
    canonical form ("hash the first 200 chars, for speed") passes every other
    test — only the integration corpus test notices, and that is skipped in CI."""
    boiler = "Vi er en stor arbeidsgiver med lang historie i regionen og tilbyr gode " * 5
    assert len(boiler) > 300
    g = dedup.cluster([ad("a", boiler + " Vi soker sykepleier."),
                       ad("b", boiler + " Vi soker tomrer.")])
    assert len(g) == 2


def test_cluster_raises_on_missing_uuid():
    """A loud KeyError is the feature. `adv.get("uuid")` would make an ad with a
    renamed field a None-keyed member that silently vanishes from the corpus."""
    with pytest.raises(KeyError):
        dedup.cluster([{"description_text": BODY}])


def test_representatives_and_fan_out_agree_on_the_representative():
    """Round-trip. `representatives` using max while `fan_out` uses min raises
    KeyError for every multi-ad cluster — after the whole census is billed."""
    ads = [ad("c", BODY), ad("a", BODY), ad("b", "Noe helt annet her")]
    g = dedup.cluster(ads)
    dedup.fan_out(g, {r: "ok" for r in dedup.representatives(g)})  # must not raise
    assert dedup.representatives({"sig": ["c", "a", "b"]}) == ["a"]


def test_representatives_tolerates_an_empty_cluster():
    assert dedup.representatives({"sig": []}) == []


def test_fan_out_shares_one_object_across_a_cluster():
    """Explicit identity. Tests comparing with == pass whether members share the
    object or hold copies; the distinction decides whether an in-place mutation
    downstream hits all 44 members or one."""
    g = {"sig": ["a", "b"]}
    out = dedup.fan_out(g, {"a": {"level": "professional"}})
    assert out["a"] is out["b"]
