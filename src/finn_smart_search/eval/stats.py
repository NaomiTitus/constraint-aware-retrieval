"""Bootstrap confidence intervals on adjacent-rung deltas. PLAN E7, pre-registered.

WHY THIS IS OWED RATHER THAN OPTIONAL. The headline — CVR@10 0.154 -> 0.077 — rests
on THIRTEEN personas. With an n that small a halving can be produced by two or three
advertisements moving, and §17 records that no interval has been computed and no
adjacent-rung delta should be quoted as significant. E7 pre-registers this; until it
runs the headline is a point estimate with no idea of its own precision.

THE RESAMPLING UNIT IS THE PERSONA, NOT THE ADVERTISEMENT, and that choice is the
whole design. Ten advertisements from one persona are not ten independent
observations: they share a query, a pool, and whatever that persona's occupation
happens to resolve to. Resampling advertisements would treat them as independent and
report an interval several times too narrow. Resampling personas asks the question
that matters — would this result hold on a different set of job seekers?

PAIRED BY CONSTRUCTION. Both arms are evaluated on the SAME resampled personas in
every replicate, so the interval is on the DIFFERENCE and the between-persona
variance that dominates raw CVR cancels out. An unpaired interval on thirteen
personas would be so wide as to say nothing.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any


def bootstrap_delta(a: Mapping[str, float], b: Mapping[str, float],
                    *, n: int = 10_000, seed: int = 0,
                    alpha: float = 0.05) -> dict[str, Any]:
    """Percentile bootstrap on mean(b) - mean(a), paired over shared keys.

    `n=10_000` and a fixed `seed` so the interval is reproducible: a CI that moves
    between runs is not a measurement. The percentile method is used rather than BCa
    because with thirteen units the bias correction is itself unstable, and an
    honest slightly-wide interval beats a clever one that cannot be checked.
    """
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return {"n_units": len(keys), "delta": None, "lo": None, "hi": None,
                "excludes_zero": None}
    diffs = [b[k] - a[k] for k in keys]
    point = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(s) / len(s))
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[min(int((1 - alpha / 2) * n), n - 1)]
    return {
        "n_units": len(keys),
        "delta": point,
        "lo": lo,
        "hi": hi,
        # The honest headline test: does the interval cross zero? If it does, the
        # rung's improvement is not distinguishable from noise at this sample size.
        "excludes_zero": (lo > 0) or (hi < 0),
    }


def ladder_intervals(arms: Sequence[str],
                     per_persona: Mapping[str, Mapping[str, Mapping[str, float]]],
                     metric: str, **kw: Any) -> list[dict[str, Any]]:
    """One interval per ADJACENT pair on the ladder, as E7 specifies.

    Adjacent rather than all-pairs because each rung is meant to justify itself
    against the one below it — "does adding this stage help?" — and testing every
    pair would multiply comparisons without a correction for it.
    """
    out = []
    for lo_arm, hi_arm in pairwise(arms):
        a = {p: v[metric] for p, v in per_persona[lo_arm].items()}
        b = {p: v[metric] for p, v in per_persona[hi_arm].items()}
        r = bootstrap_delta(a, b, **kw)
        out.append({"from": lo_arm, "to": hi_arm, "metric": metric, **r})
    return out
