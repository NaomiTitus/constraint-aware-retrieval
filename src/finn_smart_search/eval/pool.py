"""TREC depth-10 judgment pool. Scenarios 20-25, approved 2026-09-27.

`JUDGING_PROTOCOL.md`: "the union of the top-10 from every system on the ablation
ladder, deduplicated. Judged blind to which system contributed a result."

THE BLINDNESS IS STRUCTURAL, NOT PROCEDURAL. A pool is a tuple of advertisement
indices and nothing else — there is no field in which an arm identity could be
recorded, so neither a judge nor a later analysis can condition on which system
produced a result. Returning `{ad: [arms]}` would have been more informative and
would have made the protocol's rule a matter of discipline instead of a property.

Pooling bias is real and the protocol already puts it in the limitations: a
document no system retrieved is never judged, so Recall@50 is an upper bound.
Depth 10 rather than 20 was chosen there deliberately — it roughly halves the judge
cost and only mildly worsens that bias. Measured on the current ladder
(JUDGE_SCENARIOS G6): 5 arms x 13 dev personas x depth 10 = 650 slots collapsing to
346 unique pairs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DEPTH = 10


class SealedPersonaError(ValueError):
    """A sealed persona reached a pooling call.

    D16 and D17: sealed personas are opened ONCE, at the end. Pooling one puts its
    advertisements in front of a judge and its results in front of a developer,
    which is what breaking the seal means in practice. Refusing here costs nothing;
    noticing afterwards costs the held-out measurement.
    """


def build_pool(arm_rankings: Mapping[str, Sequence[tuple[int, float]]],
               depth: int = DEPTH) -> tuple[int, ...]:
    """Union of each arm's top-`depth` SCORED hits, deduplicated and ordered.

    Depth applies per system, as the protocol specifies, not as a cap on the pool.
    Only hits with a score above zero are taken: BM25 scores exactly 0.0 for no
    match, and padding a short arm up to depth would spend judgments on documents
    no system actually retrieved.

    The result is sorted, so it does not depend on the order the arms were passed
    in — otherwise the judged set would encode which system was listed first.
    """
    seen: set[int] = set()
    for hits in arm_rankings.values():
        for ad_index, score in list(hits)[:depth]:
            if score > 0.0:
                seen.add(int(ad_index))
    return tuple(sorted(seen))


def build_pools(personas: Mapping[str, Mapping[str, Any]],
                rankings: Mapping[str, Mapping[str, Sequence[tuple[int, float]]]],
                depth: int = DEPTH) -> dict[str, tuple[int, ...]]:
    """Pools for every persona in `rankings`.

    Raises on a sealed persona rather than filtering it out: silently dropping one
    would let a caller believe the sealed set had been pooled and judged when it had
    not. An unknown persona id raises `KeyError` for the same reason — a typo would
    otherwise produce a pool nothing can join back to a query.
    """
    out: dict[str, tuple[int, ...]] = {}
    for persona_id, arms in rankings.items():
        record = personas[persona_id]
        if record.get("split") == "sealed":
            raise SealedPersonaError(
                f"{persona_id} is sealed; it is opened once, at the end (D16/D17)")
        out[persona_id] = build_pool(arms, depth)
    return out
