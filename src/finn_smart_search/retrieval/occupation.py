"""Occupation as a predicate over metadata — D18, and D7's `occupation_proximity`.

WHY THIS EXISTS RATHER THAN A BETTER LEXICAL TRICK. D1's probe on the real corpus
measured the failure precisely: an English query for a nurse returned a
psychiatrist, a Norwegian teacher, a caretaker, a sales rep and a bricklayer. The
cause was not missing vocabulary — indexing `ad_taxonomy.job_title_en` made `nurse`
match 1,330 ads and changed almost nothing — it was that BM25 weights terms by
RARITY, so the one term identifying the occupation scored 2.04 while incidental
English boilerplate scored 5.0–5.8.

    A hard constraint weighted by rarity is the wrong ordering by construction.

So occupation stops being a scored term and becomes a predicate over metadata, and
the metadata already exists at 100% coverage: `ad_taxonomy.styrk_code`, 313 distinct
values, hierarchical.

WHY STYRK RATHER THAN LABELS. STYRK-08 is Norway's ISCO-08 variant and its digits
are a taxonomy, so proximity is a prefix comparison:

    2223  sykepleier                    exact
    2223  spesialsykepleier             exact — same code
    2221  nursing professionals         shares 222  (unit group)
    2341  grunnskolelærer      vs
    2342  førskolelærer                 shares 234  (unit group)
    7115  tømrer                        shares nothing with either

AND IT IS LANGUAGE-INDEPENDENT, which is the property the whole thing is for. The
ad's code is derived from the ad by A4's taxonomy build, not from the query, so a
seeker writing `nurse` and a seeker writing `sykepleier` resolve to the SAME code
and retrieve the same ads. That fixes the cross-language failure by construction
instead of hoping a shared token appears.

THE GAZETTEER IS MANY-TO-MANY AND NOISY, and pretending otherwise would break it.
NAV's own data maps one label to several codes:

    sykepleier          -> 2223, 3412, 5311
    førskolelærer       -> 2342, 2341, 1341
    programvareutvikler -> 2152, 2512

Some of that is genuine ambiguity (a `sykepleier` ad for an assistant role) and
some is upstream noise. So a label resolves to a WEIGHTED SET of codes, weighted by
how many ads carry each pairing, and proximity takes the best match across the set.
Collapsing to a single code would silently discard the ads under the other codes.

THE PREFIX WEIGHTS ARE STRUCTURAL, NOT TUNED. 1.0 / 0.7 / 0.4 / 0.15 follow the
four STYRK levels. They are NOT fitted, because fitting them needs relevance
judgments and `eval/JUDGING_PROTOCOL.md` is pre-registered with zero pairs judged —
a value chosen now by taste would later read as a measurement. They are parameters
so E6's ablation can move them once there is something to move them against.

This module takes its gazetteer as an argument and touches no database, so its
tests are pure and run on a fresh clone.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

# One weight per STYRK level. Index i = number of leading digits shared.
# 0 shared digits is 0.0: a different major group is a different kind of job, and a
# small non-zero floor there would let every ad in the corpus score something.
PREFIX_WEIGHTS: tuple[float, ...] = (0.0, 0.15, 0.40, 0.70, 1.00)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# Words that carry no occupational information but appear inside labels and in
# seeker phrasing. Stripped only for the FALLBACK token-overlap match, never from
# the exact-match path, so `lærer, spesialskole` still matches verbatim.
_LABEL_NOISE = frozenset("""
og i på som en et til for med av the a an and of in at
mv andre annet øvrige diverse generell generelle
""".split())


def normalise(label: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation. Applied to both sides so
    `Lagermedarbeidere og materia...` and a seeker's `lagermedarbeider` meet."""
    return " ".join(_WORD.findall(str(label).lower()))


def _tokens(label: str) -> frozenset[str]:
    return frozenset(w for w in normalise(label).split() if w not in _LABEL_NOISE)


@dataclass(frozen=True)
class OccupationGazetteer:
    """Normalised occupation label -> {styrk_code: ad count}.

    Built from BOTH `job_title_standardised` (Norwegian) and `job_title_en`, which
    is what makes the lookup bilingual. `build_from_rows` is the only constructor
    that touches corpus shape; the class itself is a plain mapping so tests can
    hand-build one.
    """
    labels: Mapping[str, Mapping[str, int]]
    _token_index: Mapping[frozenset[str], tuple[str, ...]] = field(
        default_factory=dict, repr=False)

    @classmethod
    def build_from_rows(cls, rows: Iterable[Sequence[object]]) -> "OccupationGazetteer":
        """`rows` are (styrk_code, job_title_standardised, job_title_en) triples,
        one per ad — so a label's weight is the number of ads that used it."""
        labels: dict[str, Counter] = {}
        for code, no_label, en_label in rows:
            if not code:
                continue
            for raw in (no_label, en_label):
                if not raw:
                    continue
                key = normalise(raw)
                if key:
                    labels.setdefault(key, Counter())[str(code)] += 1
        tok_index: dict[frozenset[str], list[str]] = {}
        for key in labels:
            tok_index.setdefault(_tokens(key), []).append(key)
        return cls(labels={k: dict(v) for k, v in labels.items()},
                   _token_index={k: tuple(v) for k, v in tok_index.items()})

    def resolve(self, phrase: str, *, min_share: float = 0.05,
                min_overlap: float = 0.30) -> dict[str, float]:
        """A seeker's occupation phrase -> {styrk_code: weight in 0..1}.

        Three passes, cheapest first, and the order matters: an exact label match
        is authoritative and must not be diluted by a fuzzy one.

          1. exact normalised label
          2. exact token SET, which absorbs word order and the comma-qualifier
             shape of the Norwegian labels — `lærer, spesialskole` vs
             `spesialskole lærer`
          3. best token OVERLAP by Jaccard, above `min_overlap`. This is what
             connects `backend developer` to `software developer`: they share
             `developer` out of three distinct tokens, 0.33. Subset containment was
             tried first and is too strict — neither phrase contains the other.

        Overlap deliberately does NOT reach across a missing shared token:
        `spesialsykepleier` is one closed compound, so it shares nothing with
        `sykepleier spesial` and will not resolve that way. Closed compounds are
        the lexical channel's problem, not this module's.

        `min_share` drops codes carrying under 5% of a label's ads: those are
        upstream noise (`sykepleier` -> 5311 on a handful of rows) and keeping them
        would make a nurse query score teaching-assistant ads.
        """
        key = normalise(phrase)
        if not key:
            return {}

        counts: Counter = Counter()
        if key in self.labels:
            counts.update(self.labels[key])
        if not counts:
            toks = _tokens(key)
            for cand in self._token_index.get(toks, ()):
                counts.update(self.labels[cand])
        if not counts:
            toks = _tokens(key)
            if toks:
                best = 0.0
                winners: list[str] = []
                for cand_toks, cands in self._token_index.items():
                    if not cand_toks:
                        continue
                    shared = toks & cand_toks
                    if not shared:
                        continue
                    j = len(shared) / len(toks | cand_toks)
                    if j > best + 1e-12:
                        best, winners = j, list(cands)
                    elif abs(j - best) <= 1e-12:
                        winners.extend(cands)
                if best >= min_overlap:
                    for cand in winners:
                        counts.update(self.labels[cand])
        if not counts:
            return {}
        total = sum(counts.values())
        return {c: n / total for c, n in counts.items()
                if n / total >= min_share}


def prefix_proximity(a: str, b: str,
                     weights: Sequence[float] = PREFIX_WEIGHTS) -> float:
    """How close two STYRK codes are, by shared leading digits.

    Codes of differing length compare over the shorter one, so a 3-digit group code
    and a 4-digit occupation code still meet.
    """
    if not a or not b:
        return 0.0
    shared = 0
    for x, y in zip(str(a), str(b)):
        if x != y:
            break
        shared += 1
    return float(weights[min(shared, len(weights) - 1)])


@dataclass(frozen=True)
class OccupationConstraint:
    """The seeker's occupation, resolved to weighted STYRK codes.

    `codes` empty means the gazetteer could not resolve the phrase. That is NOT the
    same as "no occupation stated", and callers must distinguish them: an
    unresolvable occupation should fall back to the lexical channel, never silently
    behave as though the seeker asked for nothing. `stated` records which it is.
    """
    phrase: str
    codes: Mapping[str, float]
    stated: bool = True

    @property
    def resolved(self) -> bool:
        return bool(self.codes)

    def proximity(self, ad_code: str,
                  weights: Sequence[float] = PREFIX_WEIGHTS) -> float:
        """Best weighted proximity between this constraint and one ad.

        The MAXIMUM over the seeker's codes, not a weighted mean: the codes are
        alternative readings of one phrase, so matching any of them well is a good
        match. Averaging would penalise a phrase that NAV maps to several codes.
        """
        if not ad_code or not self.codes:
            return 0.0
        return max(prefix_proximity(c, ad_code, weights) * w
                   for c, w in self.codes.items())


def from_gold_parse(parse, gaz: OccupationGazetteer) -> OccupationConstraint | None:
    """Build the constraint from a gold parse, or None when no occupation was
    stated — mirroring `constraints.SeekerProfile.language_constraint`, where
    absence must never be defaulted."""
    occ = [c for c in parse.constraints if c.facet == "occupation"]
    if not occ:
        return None
    phrase = str(occ[0].value)
    return OccupationConstraint(phrase=phrase, codes=gaz.resolve(phrase))


def apply(scores: Sequence[float], ad_codes: Sequence[str],
          constraint: OccupationConstraint | None,
          *, lam: float = 1.0, floor: float = 0.0) -> list[float]:
    """Multiply scores by occupation proximity.

    Returns the input UNCHANGED, element for element, when no occupation was
    stated or the gazetteer could not resolve it — the same non-effect rule
    `constraints.apply` follows, and for the same reason: a stage that fires when
    it has nothing to say silently destroys recall.

    `floor` keeps a non-matching ad at a fraction of its score rather than zero.
    At the default 0.0 this is a hard predicate; raise it to soften. `lam` scales
    how much of the proximity is applied.
    """
    if constraint is None or not constraint.resolved or lam == 0.0:
        return list(scores)
    out = []
    for s, code in zip(scores, ad_codes):
        p = constraint.proximity(code)
        out.append(s * (floor + (1.0 - floor) * (1.0 - lam + lam * p)))
    return out
