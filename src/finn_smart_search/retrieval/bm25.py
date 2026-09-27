"""D1 — the lexical channel. BM25 with dual stemming and compound n-grams.

WHY THIS CHANNEL CARRIES MORE WEIGHT HERE THAN IT USUALLY WOULD. LIMITATIONS §14
measured the dense channel on this corpus and it was not merely weak, it was
inverted: a negated query ranked the accessible ads BELOW the blocking ones
(AUC 0.386–0.393), and indexing extracted spans instead of whole ads made the
absolute number worse, not better. So the lexical channel is not a baseline to be
beaten here — it is the channel most likely to do the actual work, and the
constraint stage (D5) is what handles what neither channel can express.

THREE DESIGN DECISIONS, each with the measurement behind it.

STEM CHAINS, NOT SINGLE-PASS STEMS, and this was found by a failing test rather
than assumed. NORWEGIAN SNOWBALL STRIPS ONLY ONE SUFFIX PER CALL:

    sykepleier   -> sykeplei          sykepleiere -> sykepleier
    utvikler     -> utvikl            utviklere   -> utvikler
    lærer        -> lær               lærere      -> lær

So `sykepleier` and `sykepleiere` land on DIFFERENT stems and do not match — the
single most common Norwegian occupational inflection pair, in the channel that has
to carry this corpus. Dual stemming does not rescue it either: the English stemmer
reduces `sykepleier` to `sykeplei` as well.

Iterating to a fixed point converges the inflections, and it was rejected: it
collapses `leder` into `ledig` (both -> `led`), and `ledig stilling` — "vacant
position" — appears across a large share of Norwegian ads, so every leadership
query would match nearly everything. 641 ads carry `seniority: manager`.

What is indexed instead is the whole CHAIN: the surface form plus every successive
stem, de-duplicated. `sykepleiere` yields {sykepleiere, sykepleier, sykeplei} and
`sykepleier` yields {sykepleier, sykeplei}, so they share two terms. `leder` and
`ledig` still share `led`, but that term is now common to both families, so its
document frequency is high and IDF drives its contribution down — while the precise
surface term keeps a high IDF and dominates. The precision is recovered by BM25's
own weighting rather than by a hand-tuned exception.

Document length counts ORIGINAL WORDS, not chain terms, or a chain would inflate
every document's length and distort the b normalisation.

DUAL STEMMING (no + en), per PLAN D1. The corpus is 95% Norwegian and the seekers
write in both languages — five of the personas are English, and the accessible ads
are frequently English too (§14: 81 of 93 `explicitly_not_required` ads carry an
English evidence span). Stemming with one language only would cost recall on
whichever half it was not. Each token contributes the chain from EVERY stemmer, so
`nursing` and `nurses` collide on `nurs` while the Norwegian forms collide through
their own chain.

A side effect worth naming rather than discovering: the English stemmer applied to
Norwegian acts as a crude extra suffix-stripper — `sykepleierstilling` →
`sykepleierstil`. That is not linguistically meaningful, but it is CONSISTENT, so a
query and a document reduce to the same garbage and still match. Harmless, and
mildly useful.

CHARACTER 3–5 GRAMS FOR COMPOUNDS, per PLAN D1, and the reason is measurable in one
line: the Norwegian Snowball stemmer leaves `sykepleierstilling` completely intact.
Norwegian compounds are written closed, so `sykepleier` never matches
`sykepleierstilling` by stemming at all — the head word is buried inside the
string. N-grams are the cheap route to a partial match.

  They are DOWN-WEIGHTED, and must be. One 18-character compound yields ~45
  n-grams against one word token, so at equal weight a single compound would
  dominate every query it touched. `ngram_weight` scales their contribution; a
  3-gram match is genuinely weaker evidence than a word match, and the weight says
  so instead of pretending otherwise.

  THEY ALSO FIRE FALSELY, and the cost is bounded rather than absent: `sykepleier`
  and `byggeplass` share the 3-gram `epl`, so a query for one scores the other.
  Measured on that pair the spurious score is ~1.3% of the true match, which is
  what the down-weighting is for. A test pins the RATIO, because asserting zero
  would be asserting something untrue of character n-grams.

TITLE BOOST by repetition. The title is the highest-signal field in a job ad —
`ad_taxonomy` derives the standardised occupation from it for 100% of the corpus —
and repeating it inflates its term frequencies through the ordinary BM25 path
rather than bolting on a second scoring rule.

A BILINGUAL STOPWORD LIST, WHICH THIS MODULE ORIGINALLY ARGUED AGAINST. The first
version said IDF would handle function words because they have near-maximal
document frequency. THAT IS FALSE IN A MIXED-LANGUAGE CORPUS, and the probe on the
real corpus proved it. IDF suppresses the function words of the MAJORITY language
only. Measured over 10,166 ads, 95% Norwegian:

    i           94.4% of ads   idf 0.06     correctly suppressed
    for         95.7% of ads   idf 0.04     correctly suppressed
    shifts       0.3% of ads   idf 5.84     ENGLISH BOILERPLATE, looks rare
    preferably   0.3% of ads   idf 5.75     ENGLISH BOILERPLATE, looks rare
    permanent    0.6% of ads   idf 5.03     ENGLISH BOILERPLATE, looks rare
    nurse       13.0% of ads   idf 2.04     THE ONLY CONTENT TERM, weakest of all

So an English query for a nurse ranked chefs and bricklayers first: `shifts`,
`permanent`, `day` and `position` each outweighed `nurse`, and a long English-written
ad in any occupation accumulated them. The stopword list is not tidying — it is the
fix for a real ranking inversion that IDF cannot reach.

Facet-bearing words are removed too (`permanent`, `shift`, `experience`) because
they are handled STRUCTURALLY and better: `contract.permanence` resolves against
`ads.engagementtype` at 99.9% coverage, `contract.extent` against `ads.extent` at
100%. Leaving them in the lexical channel means competing with the occupation term
for the same evidence the structured comparison already has.

`ikke` and `not` are stopwords, which follows from the thesis rather than
contradicting it: the operation is containment over typed attributes, negation in
the query text is not retrieval's job, and as a free-text term `ikke` is noise.

AND THE DEEPER POINT THE SAME PROBE MADE: raw query text is the wrong input. The
gold parse for that persona already isolates `occupation: nurse` as HARD and
`contract.shift: day` as SOFT. Querying the PARSE instead of the prose removes the
boilerplate by construction, and no stopword list can be as good as not having the
noise in the first place. The list is the floor for when a parse is unavailable.

THIS MODULE DEPENDS ONLY ON numpy AND scipy, both core dependencies, because
`PyStemmer` and `bm25s` sit in the OPTIONAL `search` extra and STANDARDS.md
requires CI to pass on a fresh clone with no network. So the stemmer is INJECTED:
`default_stemmers()` resolves Snowball when it is installed, tests pass an explicit
identity stemmer to stay pure and deterministic, and no test result depends on
whether an optional package happens to be present.

(`search` is also currently uninstallable on this machine as declared: it pulls
`sentence-transformers`, which needs the PyTorch wheel that does not exist for
Python 3.13 on Intel macOS — the same wall D2 hit. See LIMITATIONS §12.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

import numpy as np
import scipy.sparse as sp

# Unicode word characters minus underscore: keeps æøå and every accented letter
# without enumerating an alphabet that would be wrong for one of the languages.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

NGRAM_PREFIX = "#"      # keeps n-gram terms from colliding with real words

# Bilingual, because IDF only suppresses the majority language (see the module
# docstring for the measurement). Three groups, kept separate so a reviewer can
# disagree with one without discarding the others.
_STOP_NO = frozenset("""
og i jeg er på som en et til for med av det den de har kan vi du ikke å om fra
ved eller men også vil skal være blir her der hva hvor når man sin sitt våre
min mitt din ditt hans hennes dem oss meg deg seg at så noe noen alle andre
""".split())
_STOP_EN = frozenset("""
i am a an the with and for do not is are was were be been of to in on at my me
you we our your they them this that these those it its as by or but if then
have has had will would can could should may might must
""".split())
# Job-ad boilerplate. These carry meaning, but as FACETS — and every one of them
# is compared structurally instead: contract.permanence -> ads.engagementtype
# (99.9%), contract.extent -> ads.extent (100%), experience.years ->
# ad_facets.min_years_experience. In the lexical channel they only compete with
# the occupation term for evidence the structured comparison already holds.
_STOP_BOILERPLATE = frozenset("""
looking look seeking seek searching search position positions role roles job jobs
work working stilling stillinger jobb arbeid søker søkes permanent fast vikariat
experience experienced erfaring erfaren preferably prefer preferred gjerne ønskelig
day days evening evenings weekend weekends shift shifts dagvakt kveld helg skift
fulltime full part time deltid heltid speak speaks speaking snakker språk language
like want wish interested apply application søknad ansettelse tiltredelse snarest
""".split())

STOPWORDS = _STOP_NO | _STOP_EN | _STOP_BOILERPLATE


class Stemmer(Protocol):
    def stemWord(self, word: str) -> str: ...


class IdentityStemmer:
    """No stemming. The default for tests, so a unit test asserts tokenisation
    rather than a third-party library's behaviour."""

    def stemWord(self, word: str) -> str:
        return word


def default_stemmers() -> tuple[Stemmer, ...]:
    """Snowball for Norwegian and English when `PyStemmer` is installed.

    Returns a single identity stemmer when it is not, so importing this module
    never fails on a fresh clone — but callers that care are expected to check,
    because silently switching stemmers would make two runs incomparable.
    """
    try:
        import Stemmer as _PyStemmer
    except ImportError:
        return (IdentityStemmer(),)
    return (_PyStemmer.Stemmer("norwegian"), _PyStemmer.Stemmer("english"))


@dataclass(frozen=True)
class Analysis:
    n_words: int
    words: list[str]
    ngrams: list[str]


@dataclass(frozen=True)
class Tokenizer:
    """Text -> BM25 terms. Deterministic and order-preserving.

    `min_compound_len` is 9 because that is where Norwegian closed compounds
    start to dominate: `sykepleier` (10) is itself long, but the words this is
    aimed at — `sykepleierstilling`, `barnehagelærer`, `truckførerbevis` — are
    longer still, and n-gramming every short word would bloat the vocabulary for
    no recall.
    """
    stemmers: tuple[Stemmer, ...] = field(default_factory=lambda: (IdentityStemmer(),))
    min_compound_len: int = 9
    ngram_min: int = 3
    ngram_max: int = 5
    # Snowball converges in one or two steps; the cap only stops a pathological
    # stemmer from looping, it is not a tuning knob.
    max_stem_steps: int = 4
    stopwords: frozenset[str] = STOPWORDS

    def words(self, text: str) -> list[str]:
        return _WORD.findall(text.lower())

    def stems(self, word: str) -> list[str]:
        """The surface form plus the full stem CHAIN from every stemmer.

        Chained because Norwegian Snowball strips one suffix per call, so
        `sykepleier` and `sykepleiere` otherwise never meet. De-duplicated and
        order-stable: a token that stems identically in both languages contributes
        ONE term, not two, so its frequency is not silently doubled.
        """
        out: list[str] = [word]
        for s in self.stemmers:
            cur = word
            for _ in range(self.max_stem_steps):
                nxt = s.stemWord(cur)
                if not nxt or nxt == cur:
                    break
                cur = nxt
                if cur not in out:
                    out.append(cur)
        return out

    def ngrams(self, word: str) -> list[str]:
        if len(word) < self.min_compound_len:
            return []
        out: list[str] = []
        for n in range(self.ngram_min, self.ngram_max + 1):
            for i in range(len(word) - n + 1):
                out.append(NGRAM_PREFIX + word[i:i + n])
        return out

    def analyse(self, text: str) -> "Analysis":
        """Word terms and n-gram terms kept apart so the index can weight them
        differently, plus the ORIGINAL word count for length normalisation —
        chain terms and n-grams must not inflate document length.
        """
        w: list[str] = []
        g: list[str] = []
        n = 0
        for word in self.words(text):
            # Dropped BEFORE stemming: a stopword's chain and n-grams are noise
            # too, and `#shi` from `shifts` would leak the same boilerplate back
            # in through the compound channel.
            if word in self.stopwords:
                continue
            n += 1
            w.extend(self.stems(word))
            g.extend(self.ngrams(word))
        return Analysis(n_words=n, words=w, ngrams=g)

    def terms(self, text: str) -> tuple[list[str], list[str]]:
        """Back-compatible view of `analyse`."""
        a = self.analyse(text)
        return a.words, a.ngrams


@dataclass
class Bm25Index:
    """Okapi BM25 over a frozen corpus.

    k1=1.2 and b=0.75 are the standard defaults and are NOT tuned: tuning them
    needs relevance judgments, and `eval/JUDGING_PROTOCOL.md` is pre-registered
    with zero pairs judged so far. Any value chosen now would be chosen by taste
    and would then look like a measurement. They are parameters so the ablation
    (E6) can move them once there is something to move them against.
    """
    vocab: dict[str, int]
    tf: sp.csc_matrix          # docs x terms, raw term frequencies
    doc_len: np.ndarray        # effective length per doc (words only)
    idf: np.ndarray
    is_ngram: np.ndarray       # bool per term
    k1: float = 1.2
    b: float = 0.75
    ngram_weight: float = 0.3
    tokenizer: Tokenizer = field(default_factory=Tokenizer)

    @property
    def n_docs(self) -> int:
        return self.tf.shape[0]

    @classmethod
    def build(cls, docs: Sequence[str], *, titles: Sequence[str] | None = None,
              tokenizer: Tokenizer | None = None, title_boost: int = 3,
              k1: float = 1.2, b: float = 0.75,
              ngram_weight: float = 0.3) -> "Bm25Index":
        """Index `docs`, optionally repeating each title `title_boost` times.

        Repetition is how the title boost is applied — it raises the title's term
        frequencies through the normal BM25 path instead of adding a second
        scoring rule with its own parameters to justify.
        """
        tok = tokenizer or Tokenizer()
        if titles is not None and len(titles) != len(docs):
            raise ValueError("titles and docs must be the same length")

        vocab: dict[str, int] = {}
        is_ngram: list[bool] = []
        indptr = [0]
        indices: list[int] = []
        data: list[int] = []
        doc_len = np.zeros(len(docs), dtype=np.float64)

        for i, body in enumerate(docs):
            text = body
            if titles is not None:
                text = ((titles[i] + " ") * title_boost) + body
            a = tok.analyse(text)
            words, grams = a.words, a.ngrams
            # ORIGINAL word count: chain terms and n-grams must not inflate length
            doc_len[i] = float(a.n_words)
            counts: dict[int, int] = {}
            for term, ngram in ((w, False) for w in words):
                j = vocab.get(term)
                if j is None:
                    j = vocab[term] = len(vocab)
                    is_ngram.append(ngram)
                counts[j] = counts.get(j, 0) + 1
            for term in grams:
                j = vocab.get(term)
                if j is None:
                    j = vocab[term] = len(vocab)
                    is_ngram.append(True)
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                indices.append(j)
                data.append(c)
            indptr.append(len(indices))

        tf = sp.csr_matrix((data, indices, indptr),
                           shape=(len(docs), len(vocab)), dtype=np.float64).tocsc()
        df = np.diff(tf.indptr).astype(np.float64)
        n = float(len(docs))
        # Robertson/Sparck-Jones IDF in the +0.5 form, which stays positive for
        # every df including df == n. The unsmoothed variant goes negative for a
        # term in more than half the corpus, which would let a common Norwegian
        # function word SUBTRACT from a score.
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        return cls(vocab=vocab, tf=tf, doc_len=doc_len, idf=idf,
                   is_ngram=np.array(is_ngram, dtype=bool),
                   k1=k1, b=b, ngram_weight=ngram_weight, tokenizer=tok)

    def score(self, query: str) -> np.ndarray:
        """BM25 score per document. Unmatched documents score exactly 0.0."""
        words, grams = self.tokenizer.terms(query)
        return self.score_terms(words, grams)

    def score_terms(self, words: Iterable[str],
                    grams: Iterable[str] = ()) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float64)
        avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        if avgdl <= 0:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * (self.doc_len / avgdl))

        for terms, weight in ((words, 1.0), (grams, self.ngram_weight)):
            if weight == 0.0:
                continue
            # A repeated query term should not count twice: BM25 scores a term
            # once per document, and duplicating it would reward a query that
            # merely says the same word again.
            for term in dict.fromkeys(terms):
                j = self.vocab.get(term)
                if j is None:
                    continue
                lo, hi = self.tf.indptr[j], self.tf.indptr[j + 1]
                rows = self.tf.indices[lo:hi]
                freq = self.tf.data[lo:hi]
                out[rows] += weight * self.idf[j] * (
                    freq * (self.k1 + 1.0) / (freq + norm[rows]))
        return out

    def top_k(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        s = self.score(query)
        if k >= len(s):
            order = np.argsort(-s, kind="stable")
        else:
            part = np.argpartition(-s, k)[:k]
            order = part[np.argsort(-s[part], kind="stable")]
        return [(int(i), float(s[i])) for i in order if s[i] > 0.0]
