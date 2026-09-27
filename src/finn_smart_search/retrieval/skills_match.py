"""Resolve extracted skill text to ESCO skill concepts. Grounded 2026-09-27.

WHY A RESOLVER RATHER THAN STRING MATCHING. Occupation is compared as a taxonomy
code with graded proximity; skills are still compared as text. That asymmetry means
`PyTorch` does not imply `machine learning`, a seeker writing `ML` gets nothing from
an advertisement saying `maskinlæring` unless a gloss happens to spell it out, and
nothing can be normalised across the 66,392 extracted phrases.

ESCO has the vocabulary: 10,063 skills, and 100% of them carry BOTH a Norwegian and
an English label — the same bilingual identity that made the occupation path work.

THE HARD PART, MEASURED BEFORE ANY CODE. ESCO skill labels are VERB PHRASES —
`develop business case`, `implement sales strategies`, `comply with food safety and
hygiene` — while the census extractor produces NOUN PHRASES — `medication
administration`, `quality control of finished products`. So:

    exact label match, 5,000 sampled glosses     2.5%
    any shared content token, 2,000 sampled      99%

Exact is useless and any-token is uselessly loose. What is needed is a score between
them, and the head noun is what carries the meaning: `administration of medication`
and `administer medication` share `medication`, which is the content word, not the
grammatical frame.

SO THE SCORE IS CONTAINMENT-WEIGHTED AND IDF-WEIGHTED. Containment because a long
ESCO label should not be penalised for being specific — the mistake D18's outcome
measured. IDF because `management` appears in hundreds of ESCO labels and `dialysis`
in one, and a match on the second means far more.

DELIBERATELY NOT WIRED INTO RANKING YET. There is no way to measure a skill-matching
change: the judged ablation isolates occupation and constraints, not skills, and
today's session is a record of what happens when ranking is tuned by eye against a
single query. This module resolves and reports; whether resolution helps is a
question for a judged arm.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# Grammatical frame, not content. ESCO labels are verb phrases and the extractor
# produces noun phrases, so these words are exactly the ones that differ between two
# descriptions of the SAME skill and must not drive a match.
STOP = frozenset(["the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with", "from", "by", "as", "is", "are", "be", "being", "been", "og", "i", "på", "som", "en", "et", "til", "for", "med", "av", "fra", "ved", "er", "the", "use", "using", "used", "ensure", "ensuring", "apply", "applying", "perform", "performing", "carry", "out", "conduct", "manage", "managing", "work", "working", "utføre", "bruke", "bruk", "av", "og", "eller", "samt"])

MIN_SCORE = 0.45


# STEMMING WAS TRIED FOR THE VERB/NOUN GAP AND DOES NOT CLOSE IT. `forklift
# operation` against ESCO's `operate forklift` is a perfect semantic match scoring
# only 0.45, because `operation` and `operate` are different strings — and English
# Snowball reduces them to `oper` and `operat`, still different. The measured lift
# from 17% to 21% came from index-side IDF shifts, not from morphology being solved.
# Kept as a parameter because it changes results and might help elsewhere; not
# presented as the fix for the gap it was reached for.
def _stemmers() -> tuple[object, ...]:
    from .bm25 import default_stemmers
    return default_stemmers()


_STEM_CACHE: dict[str, str] = {}


def _stem(word: str) -> str:
    hit = _STEM_CACHE.get(word)
    if hit is not None:
        return hit
    cur = word
    for st in _stemmers():
        for _ in range(4):
            nxt = st.stemWord(cur)          # type: ignore[attr-defined]
            if not nxt or nxt == cur:
                break
            cur = nxt
    _STEM_CACHE[word] = cur
    return cur


def tokens(text: str, *, stem: bool = False) -> frozenset[str]:
    """Content tokens, optionally stemmed.

    STEMMING IS A PARAMETER RATHER THAN A DECISION, because measuring it produced a
    genuine trade and nothing here can settle it:

        unstemmed   17% of real glosses resolve; `medication administration` maps to
                    `assist in the administration of medication` (0.89) — correct
        stemmed     21% resolve; the same phrase maps to `manage medical supply
                    chains` (0.90) — WRONG, because stemming collapses
                    medication/medical and administration/administer

    Higher resolution rate is not higher quality. Picking between them by reading a
    handful of examples is what LIMITATIONS §16 records going wrong three times in
    one sitting, so the default is the conservative one (unstemmed, correct on the
    flagship case) and the choice is left to a judged arm.
    """
    raw = [w.lower() for w in _WORD.findall(text or "")
           if len(w) > 2 and w.lower() not in STOP]
    return frozenset(_stem(w) for w in raw) if stem else frozenset(raw)


@dataclass(frozen=True)
class SkillMatch:
    uri: str
    label_en: str
    label_no: str
    score: float


class EscoSkillIndex:
    """Bilingual ESCO skill labels, searchable by content-token overlap.

    Built from `esco_skill` directly. An inverted index over content tokens keeps
    resolution linear in the query's tokens rather than in the 10,063 labels.
    """

    def __init__(self, labels: Mapping[str, tuple[str, str]],
                 postings: Mapping[str, frozenset[str]], n_labels: int):
        self.labels = dict(labels)              # uri -> (en, no)
        self.postings = dict(postings)          # token -> uris
        self.n = max(n_labels, 1)

    @classmethod
    def build(cls, rows: Iterable[Sequence[object]]) -> EscoSkillIndex:
        """`rows` are (uri, lang, title) from `esco_skill`."""
        en: dict[str, str] = {}
        no: dict[str, str] = {}
        for uri, lang, title in rows:
            (en if lang == "en" else no)[str(uri)] = str(title)
        labels = {u: (en.get(u, ""), no.get(u, "")) for u in set(en) | set(no)}
        post: dict[str, set[str]] = {}
        for uri, (e, n) in labels.items():
            for t in (tokens(e) | tokens(n) | tokens(e, stem=True)
                      | tokens(n, stem=True)):
                post.setdefault(t, set()).add(uri)
        return cls(labels, {k: frozenset(v) for k, v in post.items()}, len(labels))

    def _idf(self, token: str) -> float:
        """A token in one label is worth far more than one in hundreds.

        `management` appears across the taxonomy; `dialysis` does not. Without this,
        every phrase containing a common verb resolves to the same crowd of labels.
        """
        df = len(self.postings.get(token, ()))
        return math.log(1.0 + self.n / (df + 1))

    def resolve(self, phrase: str, *, top_k: int = 3,
                min_score: float = MIN_SCORE,
                stem: bool = False) -> list[SkillMatch]:
        """Skill text -> ESCO concepts, best first. Empty when nothing scores.

        The score is the IDF-weighted share of the PHRASE's content tokens that the
        label covers, lightly penalised for label tokens the phrase does not have.
        Containment rather than symmetric overlap, so a specific ESCO label is not
        punished for carrying extra words — the failure D18's outcome measured.
        """
        q = tokens(phrase, stem=stem)
        if not q:
            return []
        weight = {t: self._idf(t) for t in q}
        total = sum(weight.values()) or 1.0

        cand: dict[str, float] = {}
        for t in q:
            for uri in self.postings.get(t, ()):
                cand[uri] = cand.get(uri, 0.0) + weight[t]

        out: list[SkillMatch] = []
        for uri, hit in cand.items():
            e, n = self.labels[uri]
            lt = tokens(e, stem=stem) | tokens(n, stem=stem)
            cov = hit / total
            extra = len(lt - q) / max(len(lt), 1)
            score = cov - 0.15 * extra
            if score >= min_score:
                out.append(SkillMatch(uri=uri, label_en=e, label_no=n,
                                      score=round(score, 4)))
        out.sort(key=lambda m: -m.score)
        return out[:top_k]
