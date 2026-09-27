"""TREC depth-10 pooling. Scenarios 20-25, approved 2026-09-27.

`JUDGING_PROTOCOL.md`: "the union of the top-10 from every system on the ablation
ladder, deduplicated. Judged blind to which system contributed a result."

Blindness is structural rather than procedural here: a pool carries ad indices and
nothing about which arm produced them, so there is no arm identity for a judge to
see or for a later analysis to condition on by accident.
"""
from __future__ import annotations

import pytest

from finn_smart_search.eval import pool as P
from finn_smart_search.eval.pool import SealedPersonaError

# Two arms with deliberate overlap. Shapes match what `Bm25Index.top_k` returns —
# (index, score) pairs, descending, zero-scoring hits already dropped.
ARM_A = [(5, 9.0), (3, 8.0), (1, 7.0)]
ARM_B = [(3, 6.0), (9, 5.0), (1, 4.0)]


def test_the_pool_is_the_deduplicated_union_of_every_arm():
    """Scenario 20."""
    got = P.build_pool({"a": ARM_A, "b": ARM_B}, depth=10)
    assert sorted(got) == [1, 3, 5, 9]


def test_arm_order_does_not_change_the_pool():
    """Scenario 21. If the pool depended on arm order, the judged set would encode
    which system was listed first — a contributor identity leaking through the
    back door."""
    one = P.build_pool({"a": ARM_A, "b": ARM_B}, depth=10)
    two = P.build_pool({"b": ARM_B, "a": ARM_A}, depth=10)
    assert one == two


def test_the_pool_carries_no_arm_identity():
    """Scenario 22. The returned rows are ad indices. There is nowhere to record
    which arm contributed, which is the point."""
    got = P.build_pool({"a": ARM_A, "b": ARM_B}, depth=10)
    assert all(isinstance(x, int) for x in got)


def test_depth_limits_each_arm_before_the_union():
    """Depth is per system, as the protocol specifies — not a cap on the pool."""
    got = P.build_pool({"a": ARM_A, "b": ARM_B}, depth=1)
    assert sorted(got) == [3, 5]


def test_an_arm_with_fewer_than_depth_hits_contributes_only_what_it_scored():
    """Scenario 23. BM25 scores exactly 0.0 for no match and `top_k` drops those,
    so a short arm means few matches — padding the pool would spend judgments on
    documents no system actually retrieved."""
    got = P.build_pool({"a": [(7, 1.0)], "b": ARM_B}, depth=10)
    assert sorted(got) == [1, 3, 7, 9]


def test_an_empty_arm_contributes_nothing_and_does_not_raise():
    got = P.build_pool({"a": [], "b": ARM_B}, depth=10)
    assert sorted(got) == [1, 3, 9]


def test_no_arms_at_all_gives_an_empty_pool():
    assert P.build_pool({}, depth=10) == ()


PERSONAS = {
    "p_dev": {"id": "p_dev", "split": "dev"},
    "p_sealed": {"id": "p_sealed", "split": "sealed"},
}


def test_pools_are_built_for_every_dev_persona():
    got = P.build_pools(PERSONAS, {"p_dev": {"a": ARM_A}}, depth=10)
    assert sorted(got["p_dev"]) == [1, 3, 5]


def test_a_sealed_persona_reaching_a_pooling_call_is_refused():
    """Scenario 25. D16 and D17: sealed personas are opened ONCE, at the end.
    Pooling one would put its ads in front of a judge and its results in front of
    a developer, which is what breaking the seal means in practice. Refusing here
    is cheaper than noticing later."""
    with pytest.raises(SealedPersonaError, match="p_sealed"):
        P.build_pools(PERSONAS, {"p_sealed": {"a": ARM_A}}, depth=10)


def test_an_unknown_persona_is_refused_rather_than_silently_pooled():
    """A typo'd persona id would otherwise produce a pool nothing can join back to
    a query."""
    with pytest.raises(KeyError):
        P.build_pools(PERSONAS, {"p_typo": {"a": ARM_A}}, depth=10)
