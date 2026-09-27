"""Retrieval metrics over the judged pool. `eval/JUDGING_PROTOCOL.md`, pre-registered.

CVR@10 IS THE HEADLINE and the protocol defines it exactly: mean over queries of
`#{top-10 with violates=true} / 10`. It is a CONSTRAINT VIOLATION rate, not a
relevance metric, and that separation is the project's whole argument — an
advertisement can be a perfect occupational match and still be one the seeker cannot
take. The judged data carries 8 such rows (grade 3 AND violates), which the protocol
calls "the most informative row in the whole dataset".

ΔCVR_paired = CVR(no-Norwegian variant) − CVR(Norwegian variant). A system that does
not react to the stated constraint sits near zero; the proposed system should be
strongly negative. This is the paired comparison the five persona pairs exist for.

REPORTED ALONGSIDE, NEVER INSTEAD, per the protocol: nDCG@10 with gains 2**grade − 1,
MRR@10, and the pooling caveat that Recall is an upper bound because a document no
system retrieved was never judged.

UNJUDGED DOCUMENTS ARE NOT SCORED AS IRRELEVANT. A pooled evaluation judges the union
of every arm's top-10, so a document another arm never surfaced has no grade. Treating
missing as grade 0 would punish an arm for retrieving something novel, which is
exactly backwards; these functions report coverage so the reader can see how much of a
ranking was judged at all.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def cvr_at_k(ranking: Sequence[str], violates: Mapping[str, bool], k: int = 10) -> float:
    """Share of the top-k that violate a stated constraint. Unjudged rows count as
    non-violating, which is CONSERVATIVE: it can only understate the problem."""
    top = list(ranking)[:k]
    if not top:
        return 0.0
    return sum(1 for p in top if violates.get(p)) / len(top)


def ndcg_at_k(ranking: Sequence[str], grades: Mapping[str, int], k: int = 10) -> float:
    """nDCG with gains 2**grade - 1, per the protocol. The ideal ranking is drawn
    from the JUDGED pool for this query, so a score of 1.0 means "as good as the
    judged set allows", not "perfect in absolute terms"."""
    top = list(ranking)[:k]
    dcg = sum((2 ** grades.get(p, 0) - 1) / math.log2(i + 2)
              for i, p in enumerate(top))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return (dcg / idcg) if idcg else 0.0


def mrr_at_k(ranking: Sequence[str], grades: Mapping[str, int], k: int = 10,
             relevant_at: int = 2) -> float:
    """Reciprocal rank of the first grade>=2 hit. Grade 2 is "plausible: right field,
    level or contract type off" — the lowest grade a seeker would actually apply to."""
    for i, p in enumerate(list(ranking)[:k]):
        if grades.get(p, 0) >= relevant_at:
            return 1.0 / (i + 1)
    return 0.0


def judged_share(ranking: Sequence[str], grades: Mapping[str, Any],
                 k: int = 10) -> float:
    """How much of this top-k carries a judgment at all. Below 1.0 the metrics above
    are optimistic, because unjudged rows contribute nothing to DCG and nothing to
    CVR. Reported so a thin pool cannot be mistaken for a good result."""
    top = list(ranking)[:k]
    return (sum(1 for p in top if p in grades) / len(top)) if top else 0.0


def evaluate_arm(rankings: Mapping[str, Sequence[str]],
                 grades: Mapping[str, int], violates: Mapping[str, bool],
                 k: int = 10) -> dict[str, Any]:
    """Aggregate one arm over every persona, macro-averaged.

    Macro, not micro: each persona counts once regardless of how many of its
    advertisements happened to be judged, so a persona with a large pool cannot
    dominate the headline.
    """
    per: dict[str, dict[str, float]] = {}
    for pid, ranking in rankings.items():
        g = {p: v for p, v in grades.items() if p.startswith(pid + "-")}
        per[pid] = {
            "cvr": cvr_at_k(ranking, violates, k),
            "ndcg": ndcg_at_k(ranking, g, k),
            "mrr": mrr_at_k(ranking, g, k),
            "judged": judged_share(ranking, g, k),
        }
    n = len(per) or 1
    return {
        "n_personas": len(per),
        "cvr_at_10": sum(v["cvr"] for v in per.values()) / n,
        "ndcg_at_10": sum(v["ndcg"] for v in per.values()) / n,
        "mrr_at_10": sum(v["mrr"] for v in per.values()) / n,
        "judged_share": sum(v["judged"] for v in per.values()) / n,
        "per_persona": per,
    }


def paired_delta_cvr(per_persona: Mapping[str, Mapping[str, float]],
                     personas: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """ΔCVR = CVR(no-Norwegian) − CVR(Norwegian), per pair.

    THE DIRECTION IS THE WHOLE POINT. Near zero means the system did not react to
    the seeker stating they cannot work in Norwegian. Strongly negative means it did.
    A POSITIVE value means stating the constraint made things WORSE, which is what
    the dense channel did on 5 of 5 personas (LIMITATIONS §12, cell 3).
    """
    pairs: dict[str, dict[str, float]] = {}
    for pid, row in per_persona.items():
        rec = personas.get(pid) or {}
        pair = rec.get("pair_id")
        if not pair:
            continue
        side = "no_norsk" if "no_norsk" in pid else "norsk"
        pairs.setdefault(pair, {})[side] = row["cvr"]
    deltas = {p: v["no_norsk"] - v["norsk"]
              for p, v in pairs.items() if len(v) == 2}
    return {
        "per_pair": {p: {**pairs[p], "delta": d} for p, d in deltas.items()},
        "mean_delta": (sum(deltas.values()) / len(deltas)) if deltas else None,
        "n_pairs": len(deltas),
    }
