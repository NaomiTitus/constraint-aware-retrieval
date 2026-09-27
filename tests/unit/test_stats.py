"""Bootstrap intervals on adjacent-rung deltas. PLAN E7, pre-registered.

The property that matters is not the arithmetic — it is that the RESAMPLING UNIT is
the persona. Ten advertisements from one persona share a query, a pool and an
occupation resolution; treating them as ten independent observations would report an
interval several times too narrow and turn a noisy result into a significant one.
"""
from __future__ import annotations

from finn_smart_search.eval.stats import bootstrap_delta, ladder_intervals


def test_a_consistent_improvement_excludes_zero():
    a = {f"p{i}": 0.30 for i in range(13)}
    b = {f"p{i}": 0.10 for i in range(13)}
    r = bootstrap_delta(a, b, n=2000)
    assert r["delta"] < 0 and r["hi"] < 0 and r["excludes_zero"]


def test_noise_does_not_exclude_zero():
    """A difference that changes sign across units must not read as significant."""
    a = {f"p{i}": 0.2 for i in range(13)}
    b = {f"p{i}": 0.2 + (0.1 if i % 2 else -0.1) for i in range(13)}
    r = bootstrap_delta(a, b, n=2000)
    assert not r["excludes_zero"]


def test_it_is_paired_over_shared_units_only():
    """Both arms are evaluated on the SAME resampled personas, so the interval is on
    the difference and between-persona variance cancels."""
    a = {"p1": 0.4, "p2": 0.2, "gone": 0.9}
    b = {"p1": 0.1, "p2": 0.1}
    assert bootstrap_delta(a, b, n=500)["n_units"] == 2


def test_too_few_units_returns_no_interval_rather_than_a_fake_one():
    r = bootstrap_delta({"p1": 0.3}, {"p1": 0.1}, n=500)
    assert r["delta"] is None and r["excludes_zero"] is None


def test_the_interval_is_reproducible():
    """A confidence interval that moves between runs is not a measurement."""
    a = {f"p{i}": 0.3 + i * 0.01 for i in range(13)}
    b = {f"p{i}": 0.1 + i * 0.02 for i in range(13)}
    assert bootstrap_delta(a, b, n=1000) == bootstrap_delta(a, b, n=1000)


def test_the_point_estimate_sits_inside_its_own_interval():
    a = {f"p{i}": 0.3 + (i % 3) * 0.05 for i in range(13)}
    b = {f"p{i}": 0.1 + (i % 4) * 0.05 for i in range(13)}
    r = bootstrap_delta(a, b, n=2000)
    assert r["lo"] <= r["delta"] <= r["hi"]


def test_the_ladder_compares_adjacent_rungs_only():
    """Each rung justifies itself against the one below it; all-pairs would multiply
    comparisons without correcting for them."""
    pp = {a: {f"p{i}": {"cvr": 0.3 - j * 0.1} for i in range(13)}
          for j, a in enumerate(["x", "y", "z"])}
    out = ladder_intervals(["x", "y", "z"], pp, "cvr", n=500)
    assert [(r["from"], r["to"]) for r in out] == [("x", "y"), ("y", "z")]
