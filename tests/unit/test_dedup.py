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
    assert dedup.signature(BODY) == dedup.signature(BODY)
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
    assert dedup.signature("Gode norskkunnskaper") == dedup.signature("Gode norskkunnskaper")
    assert dedup.signature("norsk­kunnskaper") == dedup.signature("norskkunnskaper")


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
    assert len(g) == 2


def test_b9_signature_is_deterministic_and_a_plain_string():
    """No RNG, no seed, no permutations — unlike MinHash. A hex digest also
    means cluster keys are stable across processes and serialisable."""
    s = dedup.signature(BODY)
    assert isinstance(s, str) and len(s) == 64
    assert all(c in "0123456789abcdef" for c in s)
    assert s == dedup.signature(BODY)


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
def test_b13_corpus_clustering():
    """Pinned against the real corpus. Requires data/ads.duckdb, so it is marked
    integration and excluded from CI."""
    from finn_smart_search.ingest import store

    con = store.connect("data/ads.duckdb")
    rows = con.execute("SELECT uuid, description_text FROM ads WHERE n_chars > 0").fetchall()
    ads = [{"uuid": u, "description_text": t} for u, t in rows]
    g = dedup.cluster(ads)
    assert len(g) == EXPECTED_CLUSTERS
    assert len(ads) - len(g) == EXPECTED_REDUNDANT
    assert max(len(v) for v in g.values()) == EXPECTED_LARGEST
