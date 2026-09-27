"""D1's lexical channel. Pure unit tests — no corpus, no optional dependency.

Every test injects an explicit stemmer, so nothing here depends on whether
`PyStemmer` happens to be installed. STANDARDS.md requires CI to pass on a fresh
clone with no network, and `PyStemmer` is in the optional `search` extra; a test
whose expected values changed with the environment would be worse than no test.

The behaviours worth pinning are the ones that are easy to get quietly wrong:
IDF must never go negative, n-grams must not inflate document length or dominate
word matches, a repeated query term must not score twice, and the title boost must
actually move ranking.
"""
from __future__ import annotations

import numpy as np
import pytest

from finn_smart_search.retrieval.bm25 import (
    STOPWORDS, Bm25Index, IdentityStemmer, NGRAM_PREFIX, Tokenizer,
    default_stemmers)


class _SuffixStemmer:
    """A deterministic fake: strips a trailing 'er'/'e'. Enough to prove dual
    stemming merges variants, without importing Snowball."""

    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    def stemWord(self, word: str) -> str:
        return word[:-len(self.suffix)] if word.endswith(self.suffix) else word


PLAIN = Tokenizer(stemmers=(IdentityStemmer(),))


# ── tokenisation ──────────────────────────────────────────────────────────────

def test_norwegian_letters_survive_tokenisation():
    """A tokenizer that dropped æøå would silently mangle 95% of the corpus."""
    assert PLAIN.words("Tømrer søkes på Sørøya") == ["tømrer", "søkes", "på", "sørøya"]


def test_punctuation_and_underscores_are_not_terms():
    assert PLAIN.words("Python, PostgreSQL og_Kubernetes.") == [
        "python", "postgresql", "og", "kubernetes"]


def test_tokenisation_is_case_folded():
    assert PLAIN.words("NORSK Norsk norsk") == ["norsk"] * 3


def test_the_surface_form_is_always_the_first_term():
    """The surface form is the most precise term and keeps the highest IDF, so it
    must survive stemming rather than be replaced by it."""
    tok = Tokenizer(stemmers=(_SuffixStemmer("e"),))
    assert tok.stems("sykepleiere")[0] == "sykepleiere"


def test_stemming_yields_the_whole_chain_not_just_the_final_stem():
    """Norwegian Snowball strips ONE suffix per call, so `sykepleier` and
    `sykepleiere` land on different stems and would never meet. Indexing the chain
    is what makes them share a term."""
    tok = Tokenizer(stemmers=(_SuffixStemmer("e"),))
    assert tok.stems("sykepleiere") == ["sykepleiere", "sykepleier"]


def test_inflected_and_base_forms_share_at_least_one_chain_term():
    """The property the chain exists to guarantee, stated as a property rather
    than as a specific stem string."""
    tok = Tokenizer(stemmers=(_SuffixStemmer("e"),))
    assert set(tok.stems("sykepleiere")) & set(tok.stems("sykepleier"))


def test_a_token_that_stems_identically_contributes_one_term_not_two():
    """Two stemmers agreeing must not double a term's frequency — that would
    silently weight agreement between languages as repetition."""
    tok = Tokenizer(stemmers=(IdentityStemmer(), IdentityStemmer()))
    assert tok.stems("norsk") == ["norsk"]


def test_two_stemmers_that_disagree_both_contribute():
    tok = Tokenizer(stemmers=(_SuffixStemmer("er"), IdentityStemmer()))
    assert tok.stems("tømrer") == ["tømrer", "tømr"]


def test_a_chain_terminates_and_does_not_loop():
    """A stemmer that never reaches a fixed point must not hang the indexer."""
    class _Shrink:
        def stemWord(self, w):
            return w[:-1] if len(w) > 1 else w
    tok = Tokenizer(stemmers=(_Shrink(),), max_stem_steps=3)
    assert tok.stems("sykepleier") == ["sykepleier", "sykepleie", "sykeplei", "sykeple"]


def test_short_words_get_no_ngrams():
    """N-gramming every word would bloat the vocabulary for no recall."""
    assert PLAIN.ngrams("norsk") == []


def test_long_compounds_get_ngrams_that_include_the_head_word():
    """The reason this exists: the Norwegian Snowball stemmer leaves
    `sykepleierstilling` intact, so `sykepleier` can only be reached through the
    inside of the string."""
    grams = PLAIN.ngrams("sykepleierstilling")
    assert NGRAM_PREFIX + "syk" in grams
    assert NGRAM_PREFIX + "pleie" in grams
    assert all(g.startswith(NGRAM_PREFIX) for g in grams)


def test_ngram_terms_cannot_collide_with_real_words():
    """Without the prefix, the 3-gram `nor` and the word `nor` would share a
    term and pool their document frequencies."""
    words, grams = PLAIN.terms("barnehagelærer nor")
    assert "nor" in words
    assert NGRAM_PREFIX + "nor" not in words
    assert all(g.startswith(NGRAM_PREFIX) for g in grams)


# ── index construction ────────────────────────────────────────────────────────

def _idx(docs, **kw):
    kw.setdefault("tokenizer", PLAIN)
    return Bm25Index.build(docs, **kw)


def test_idf_is_never_negative_even_for_a_term_in_every_document():
    """The unsmoothed IDF goes negative for a term in more than half the corpus,
    which would let a common Norwegian function word SUBTRACT from a score."""
    idx = _idx(["norsk jobb", "norsk stilling", "norsk arbeid"])
    assert (idx.idf >= 0).all()
    assert idx.idf[idx.vocab["norsk"]] == pytest.approx(
        np.log(1.0 + 0.5 / 3.5), abs=1e-12)


def test_document_length_counts_words_not_ngrams():
    """N-grams must not inflate length normalisation: a document with one long
    compound would otherwise be treated as enormous and penalised for it."""
    idx = _idx(["barnehagelærer"])
    assert idx.doc_len[0] == 1.0
    assert idx.tf.shape[1] > 1     # but its n-grams ARE indexed


def test_rebuilding_the_same_corpus_is_deterministic():
    docs = ["tømrer i Bergen", "sykepleier i Oslo"]
    a, b = _idx(docs), _idx(docs)
    assert a.vocab == b.vocab
    assert np.allclose(a.score("tømrer"), b.score("tømrer"))


def test_mismatched_titles_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        _idx(["a", "b"], titles=["only one"])


# ── scoring ───────────────────────────────────────────────────────────────────

def test_an_unmatched_query_scores_exactly_zero():
    """Exactly 0.0, not merely small: RRF fusion and the constraint stage both
    key on whether a channel retrieved a document at all."""
    idx = _idx(["tømrer i Bergen"])
    assert idx.score("sykepleier").tolist() == [0.0]


def test_an_unknown_term_does_not_crash_or_contribute():
    idx = _idx(["tømrer i Bergen"])
    assert idx.score("tømrer kvantekryptografi")[0] > 0.0


def test_the_document_containing_the_rarer_term_scores_higher():
    idx = _idx(["norsk sykepleier", "norsk tømrer", "norsk kokk"])
    s = idx.score("sykepleier")
    assert s[0] > 0 and s[1] == 0.0 and s[2] == 0.0


def test_a_repeated_query_term_does_not_score_twice():
    """BM25 scores a term once per document. Rewarding a query that merely says
    the same word again would make query length a ranking signal."""
    idx = _idx(["tømrer i Bergen", "sykepleier i Oslo"])
    assert np.allclose(idx.score("tømrer"), idx.score("tømrer tømrer tømrer"))


def test_ngrams_are_down_weighted_relative_to_word_matches():
    """One 18-character compound yields ~45 n-grams against one word token. At
    equal weight a single compound would dominate every query it touched."""
    docs = ["sykepleierstilling ved sykehjem", "sykepleier ved sykehjem"]
    hi = Bm25Index.build(docs, tokenizer=PLAIN, ngram_weight=1.0)
    lo = Bm25Index.build(docs, tokenizer=PLAIN, ngram_weight=0.0)
    q = "sykepleierstilling"
    # with n-grams off, only the exact compound matches
    assert lo.score(q)[1] == 0.0
    # with them on, the compound reaches the plain word too
    assert hi.score(q)[1] > 0.0
    # and the weight is what controls how far
    mid = Bm25Index.build(docs, tokenizer=PLAIN, ngram_weight=0.3)
    assert 0.0 < mid.score(q)[1] < hi.score(q)[1]


def test_ngrams_let_a_head_word_reach_a_closed_compound():
    """The whole point of D1's n-grams: `sykepleier` must find
    `sykepleierstilling`, which stemming alone cannot do."""
    idx = _idx(["sykepleierstilling ved sykehjem", "tømrer på annet arbeid"])
    s = idx.score("sykepleier")
    assert s[0] > 0.0
    assert s[0] > 10 * s[1]


def test_ngram_false_positives_are_bounded_not_absent():
    """`sykepleier` and `byggeplass` share the 3-gram `epl`, so a query for one
    DOES score the other. Asserting zero would assert something untrue of
    character n-grams; what matters is that the down-weighting keeps it marginal."""
    idx = _idx(["sykepleier ved sykehjem", "tømrer på byggeplass"])
    s = idx.score("sykepleier")
    assert s[1] > 0.0, "expected a small spurious n-gram match"
    assert s[1] < 0.05 * s[0], f"spurious match too large: {s[1] / s[0]:.1%}"


def test_title_boost_changes_ranking():
    """Two documents with the same body; the term appears in one title. The
    titled one must win, or the boost is decorative."""
    docs = ["arbeid med mennesker", "arbeid med mennesker"]
    titles = ["Sykepleier søkes", "Vaktmester søkes"]
    idx = Bm25Index.build(docs, titles=titles, tokenizer=PLAIN, title_boost=3)
    s = idx.score("sykepleier")
    assert s[0] > s[1] == 0.0


def test_a_higher_title_boost_raises_the_titled_document_further():
    docs = ["arbeid", "sykepleier arbeid"]
    titles = ["Sykepleier", "Annet"]
    low = Bm25Index.build(docs, titles=titles, tokenizer=PLAIN, title_boost=1)
    high = Bm25Index.build(docs, titles=titles, tokenizer=PLAIN, title_boost=8)
    q = "sykepleier"
    assert (high.score(q)[0] / high.score(q).sum()
            > low.score(q)[0] / low.score(q).sum())


def test_length_normalisation_prefers_the_shorter_document():
    """b=0.75 means a term in a short document is stronger evidence. Without
    normalisation, long ads would win every query by accumulating terms."""
    idx = _idx(["sykepleier", "sykepleier " + " ".join(["fyll"] * 200)])
    s = idx.score("sykepleier")
    assert s[0] > s[1]


def test_b_zero_disables_length_normalisation():
    docs = ["sykepleier", "sykepleier " + " ".join(["fyll"] * 200)]
    idx = Bm25Index.build(docs, tokenizer=PLAIN, b=0.0)
    s = idx.score("sykepleier")
    assert s[0] == pytest.approx(s[1])


def test_top_k_returns_only_documents_that_matched():
    idx = _idx(["tømrer", "sykepleier", "kokk"])
    hits = idx.top_k("tømrer", k=10)
    assert [i for i, _ in hits] == [0]


def test_top_k_is_ordered_by_descending_score():
    idx = _idx(["sykepleier sykepleier", "sykepleier", "kokk"])
    hits = idx.top_k("sykepleier", k=3)
    assert [i for i, _ in hits] == [0, 1]
    assert hits[0][1] >= hits[1][1]


def test_top_k_larger_than_the_corpus_is_safe():
    idx = _idx(["tømrer"])
    assert idx.top_k("tømrer", k=99) == [(0, pytest.approx(idx.score("tømrer")[0]))]


# ── the optional dependency, when it is present ───────────────────────────────

@pytest.mark.skipif(len(default_stemmers()) < 2,
                    reason="PyStemmer not installed (optional `search` extra)")
@pytest.mark.parametrize("base,inflected", [
    ("sykepleier", "sykepleiere"), ("utvikler", "utviklere"),
    ("lærer", "lærere"), ("tømrer", "tømrere"), ("kokk", "kokker"),
])
def test_real_snowball_inflections_share_a_chain_term(base, inflected):
    """The defect this design exists to fix, pinned against the real stemmer.
    Single-pass stemming puts `sykepleier` on `sykeplei` and `sykepleiere` on
    `sykepleier`, which do not match; the chain makes them share a term."""
    tok = Tokenizer(stemmers=default_stemmers())
    shared = set(tok.stems(base)) & set(tok.stems(inflected))
    assert shared, f"{base} and {inflected} share no term: " \
                   f"{tok.stems(base)} vs {tok.stems(inflected)}"


@pytest.mark.skipif(len(default_stemmers()) < 2,
                    reason="PyStemmer not installed (optional `search` extra)")
def test_real_snowball_does_not_reduce_a_closed_compound_to_its_head():
    """Documents the limitation that motivates the n-grams: inflection is handled
    by stemming, closed Norwegian compounds are not."""
    tok = Tokenizer(stemmers=default_stemmers())
    assert "sykepleier" not in tok.stems("sykepleierstilling")


@pytest.mark.skipif(len(default_stemmers()) < 2,
                    reason="PyStemmer not installed (optional `search` extra)")
def test_the_over_stemming_collision_is_survivable_because_idf_damps_it():
    """`leder` and `ledig` both reach `led`, which is why fixed-point stemming was
    rejected. With chains the collision remains but the precise surface term
    carries a much higher IDF, so a `leder` query still prefers the leader ad."""
    tok = Tokenizer(stemmers=default_stemmers())
    docs = ["vi søker en leder til avdelingen"] + \
           ["ledig stilling i kommunen"] * 20
    idx = Bm25Index.build(docs, tokenizer=tok)
    s = idx.score("leder")
    assert s[0] > s[1], "the over-stemmed collision outranked the true match"


# ── stopwords: the fix for a ranking inversion IDF cannot reach ────────────────

def test_stopwords_are_dropped_before_stemming():
    """A stopword's stem chain is noise too. Filtering after stemming would leave
    `shift` behind from `shifts`."""
    tok = Tokenizer(stemmers=(IdentityStemmer(),))
    a = tok.analyse("nurse shifts permanent")
    assert a.words == ["nurse"]


def test_stopword_ngrams_do_not_leak_back_through_the_compound_channel():
    """`preferably` is 10 characters, so it would generate n-grams and smuggle the
    same boilerplate back in at 0.3 weight."""
    tok = Tokenizer(stemmers=(IdentityStemmer(),))
    assert tok.analyse("preferably").ngrams == []


def test_stopwords_do_not_count_toward_document_length():
    """Length normalisation must reflect content, or a wordy ad is penalised for
    its function words."""
    tok = Tokenizer(stemmers=(IdentityStemmer(),))
    assert tok.analyse("i am a nurse with experience").n_words == 1


def test_english_boilerplate_no_longer_outranks_the_occupation_term():
    """THE REGRESSION THIS EXISTS FOR, in miniature. In a Norwegian corpus the
    English words `shifts`/`permanent` look rare and get high IDF, so a chef ad
    sharing them beat every nurse ad on an English nurse query. With the corpus
    term `nurse` present in both, the nurse ad must win."""
    tok = Tokenizer(stemmers=(IdentityStemmer(),))
    docs = [
        "nurse at the hospital",                                     # the right one
        "chef permanent position day shifts evening weekend work",    # boilerplate
    ] + ["sykepleier ved sykehuset"] * 20                            # corpus bulk
    idx = Bm25Index.build(docs, tokenizer=tok)
    s = idx.score("I am a nurse looking for a permanent position, preferably day shifts")
    assert s[0] > s[1], (
        f"boilerplate ad still outranks the occupation match: "
        f"{s[0]:.2f} vs {s[1]:.2f}")


def test_both_languages_are_covered_by_the_stopword_list():
    """A list that covered only one language would reintroduce the asymmetry it
    exists to remove."""
    assert {"og", "jeg", "ikke", "med"} <= STOPWORDS
    assert {"and", "i", "not", "with"} <= STOPWORDS


def test_a_caller_can_disable_stopwords():
    """They are a default, not a law: the ablation (E6) has to be able to turn
    them off and measure what they were worth."""
    tok = Tokenizer(stemmers=(IdentityStemmer(),), stopwords=frozenset())
    assert tok.analyse("i am a nurse").n_words == 4
