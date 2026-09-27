"""Score extracted skills against the golden labels. Scenarios approved 2026-09-27.

THE AGREEMENT VOCABULARY IS NOT NEW, and reusing it is the point. `eval/scoring.py`
already solved this for evidence spans, with the measurement attached: over the 32
golden spans, 20 were strict-equal, 5 differed ONLY by trailing punctuation, 7 were
a containment, and 0 were genuinely different sentences. So an exact-match rate
reads 62.5% while substantive agreement is 32 of 32, and — in that module's own
words — "five points of the gap are a full stop". A second, coarser matcher here
would put that 62.5% in the README as extraction quality.

WHAT `unpaired_predictions` MEANS, AND WHY IT IS NOT A FALSE POSITIVE. The golden
labels were produced by a model, not a person (LIMITATIONS §15). So a prediction
with no golden counterpart is ambiguous: it may be an extractor error, or it may be
a skill the labeller missed. This module reports the count and never names it a
false positive, because doing so would let the labeller's omissions read as
extractor precision failures — the same self-agreement trap §15 records. Precision
is computed and reported, but its denominator is stated to be provisional.

WHY LEVEL AND GLOSS DISAGREEMENTS DO NOT AFFECT F1. Failing to find a skill and
misjudging whether it was required are different errors with different costs: the
first loses the match entirely, the second only mis-weights it in D7's coverage.
And two models gloss the same Norwegian phrase differently by nature, so folding
gloss agreement into F1 would make the headline a translation-similarity score.
Both are reported on their own lines.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..understanding.text_norm import normalise

AGREEMENT = ("strict", "trailing_punct_only", "containment", "disjoint")

# Leading block glyphs and trailing sentence punctuation. Both measured: `•` leads
# 4,600 corpus blocks, `-` 3,558, `·` 1,944, and 5 of 32 golden spans differed from
# the model's quote by a trailing full stop alone.
_LEAD = "•·-*✅●«»📍→▪◦– \t"
_TRAIL = ".,;:!? "

# A containment match must share real content, not just a common word. 15 characters
# is `census_validate`'s own `fragment_too_short` floor, reused rather than re-chosen.
MIN_CONTAINMENT_CHARS = 15


@dataclass(frozen=True)
class Pair:
    golden_phrase: str
    predicted_phrase: str
    agreement: str
    level_match: bool
    gloss_match: bool


def _key(phrase: str) -> str:
    """Normalise for comparison via the single shared implementation, then strip the
    bullet the model does not quote and the full stop it sometimes adds."""
    return normalise(phrase).strip(_LEAD).strip(_TRAIL).casefold()


def classify(golden_phrase: str, predicted_phrase: str) -> str:
    """One of AGREEMENT, strongest first."""
    g_raw, p_raw = normalise(golden_phrase), normalise(predicted_phrase)
    g, p = _key(golden_phrase), _key(predicted_phrase)
    if not g or not p:
        return "disjoint"
    if g == p:
        # Distinguish "identical" from "identical once a full stop is removed",
        # because the second is the category worth five points of the span gap.
        if g_raw.casefold() != p_raw.casefold() and g_raw.strip(_TRAIL) != g_raw:
            return "trailing_punct_only"
        if g_raw.casefold() != p_raw.casefold() and p_raw.strip(_TRAIL) != p_raw:
            return "trailing_punct_only"
        return "strict"
    if (g in p or p in g) and min(len(g), len(p)) >= MIN_CONTAINMENT_CHARS:
        return "containment"
    return "disjoint"


def _rank(agreement: str) -> int:
    return AGREEMENT.index(agreement)


def pair_skills(golden: Sequence[Mapping[str, Any]],
                predicted: Sequence[Mapping[str, Any]]
                ) -> tuple[list[Pair], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Greedy one-to-one pairing, best agreement first.

    One-to-one because recall must never exceed 1. Ambiguity can only come from the
    prediction side: measured over the labels, ZERO pairs of golden phrases on the
    same advertisement contain one another, so no prediction can legitimately claim
    two golden rows.

    Ties break on the longer prediction, which prefers the fuller quotation — the
    model quoting a whole requirement sentence rather than a fragment of it.
    """
    candidates = []
    for gi, g in enumerate(golden):
        for pi, p in enumerate(predicted):
            a = classify(str(g.get("phrase", "")), str(p.get("phrase", "")))
            if a != "disjoint":
                candidates.append((_rank(a), -len(str(p.get("phrase", ""))), gi, pi, a))
    candidates.sort()

    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs: list[Pair] = []
    for _, _, gi, pi, a in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        g, p = golden[gi], predicted[pi]
        pairs.append(Pair(
            golden_phrase=str(g.get("phrase", "")),
            predicted_phrase=str(p.get("phrase", "")),
            agreement=a,
            level_match=g.get("level") == p.get("level"),
            gloss_match=_key(str(g.get("gloss_en", ""))) == _key(str(p.get("gloss_en", ""))),
        ))
    return (pairs,
            [g for i, g in enumerate(golden) if i not in used_g],
            [p for i, p in enumerate(predicted) if i not in used_p])


def score(predictions: Mapping[str, Sequence[Mapping[str, Any]]],
          golden: Mapping[str, Sequence[Mapping[str, Any]]],
          census_baseline: Mapping[str, Sequence[Mapping[str, Any]]] | None = None
          ) -> dict[str, Any]:
    """Aggregate over every advertisement present in both.

    `precision` and `recall` are `None` rather than 0.0 when their denominator is
    empty. Two of the 44 golden advertisements genuinely ask for no skill at all, and
    scoring a correct zero as 0.0 would punish the right answer — while a silent 0/0
    would crash the pilot on exactly those rows.
    """
    ads = [u for u in golden if u in predictions]
    unscored = sorted(u for u in predictions if u not in golden)

    agreement: Counter[str] = Counter({k: 0 for k in AGREEMENT})
    n_pairs = n_gold = n_pred = 0
    level_hits = gloss_hits = 0
    correct_zeros = 0
    unpaired_pred: list[dict[str, Any]] = []
    unpaired_gold: list[dict[str, Any]] = []
    per_ad: dict[str, Any] = {}

    for uuid in ads:
        g = list(golden[uuid])
        p = list(predictions[uuid])
        n_gold += len(g)
        n_pred += len(p)
        if not g and not p:
            correct_zeros += 1
        pairs, ug, up = pair_skills(g, p)
        n_pairs += len(pairs)
        for pr in pairs:
            agreement[pr.agreement] += 1
            level_hits += pr.level_match
            gloss_hits += pr.gloss_match
        unpaired_gold += [{"uuid": uuid, **dict(x)} for x in ug]
        unpaired_pred += [{"uuid": uuid, **dict(x)} for x in up]
        per_ad[uuid] = {"golden": len(g), "predicted": len(p), "paired": len(pairs)}

    precision = (n_pairs / n_pred) if n_pred else None
    recall = (n_pairs / n_gold) if n_gold else None
    f1 = ((2 * precision * recall / (precision + recall))
          if precision and recall else
          (0.0 if (precision == 0.0 or recall == 0.0) else None))

    out: dict[str, Any] = {
        "n_ads": len(ads),
        "unscored": unscored,
        "n_golden": n_gold,
        "n_predicted": n_pred,
        "n_paired": n_pairs,
        "correct_zeros": correct_zeros,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "agreement": dict(agreement),
        # Reported apart from F1 on purpose — see the module docstring.
        "level_agreement": (level_hits / n_pairs) if n_pairs else None,
        "gloss_agreement": (gloss_hits / n_pairs) if n_pairs else None,
        # NOT false positives. The label set is model-produced (§15), so an extra
        # prediction may be a skill the labeller missed. Sampled for human review.
        "unpaired_predictions": len(unpaired_pred),
        "unpaired_predictions_sample": unpaired_pred[:20],
        "unpaired_golden": len(unpaired_gold),
        "unpaired_golden_sample": unpaired_gold[:20],
        "per_ad": per_ad,
    }

    if census_baseline is not None:
        n_census = sum(len(census_baseline.get(u) or ()) for u in ads)
        zeros = sum(1 for u in ads if not (census_baseline.get(u) or ()))
        out["census"] = {
            "n_skills": n_census,
            "ads_with_zero": zeros,
            "lift": (n_pred / n_census) if n_census else None,
        }
    return out
