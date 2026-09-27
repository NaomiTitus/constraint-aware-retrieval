"""The dev/sealed persona split is pre-registered — these tests stop it drifting.

WHY A TEST AND NOT A COMMENT. A held-out set is worth exactly as much as the
discipline keeping it held out, and discipline is not a mechanism. `PLAN.md` E1
pre-registered "10 dev, 6 SEALED" and `STANDARDS.md` said sealed personas stay
sealed — and for the whole project up to 2026-09-27 no split existed in the repo
at all. Both statements read as commitments while enforcing nothing.

So the invariants live here, where breaking them fails CI:

  - every persona declares a split, so a new one cannot default into dev silently;
  - the SEALED COUNT stays at the pre-registered number — 6 in D16, amended to 7
    in D17 before any result existed, and frozen for good once the first CVR@10
    is computed;
  - pairs are never split across dev/sealed, because the paired variants differ
    only in the language sentence and the comparison IS the measurement;
  - the non-effect controls are represented on BOTH sides;
  - the sealed set can still run the paired negation stress-test E7 asks for.

  - the "Norwegian genuinely required, return few results" property is guarded in
    dev and verified held-out.

None of these assert that the split is a GOOD one — that is a judgment recorded in
D16 and D17. They assert it is the split that was pre-registered, unchanged.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

PERSONAS = Path(__file__).resolve().parents[2] / "eval" / "personas.yaml"

# The pre-registered quantities. Changing these numbers means changing a
# pre-registration, which is a DECISIONS.md entry and not a test edit.
N_SEALED = 7   # 6 at pre-registration, amended to 7 pre-results — D17
N_DEV = 13
VALID = {"dev", "sealed"}


@pytest.fixture(scope="module")
def personas() -> list[dict]:
    return yaml.safe_load(PERSONAS.read_text(encoding="utf-8"))["personas"]


def test_every_persona_declares_a_split(personas):
    """A persona added without a split would silently join the dev set."""
    missing = [p["id"] for p in personas if "split" not in p]
    assert not missing, f"personas with no `split`: {missing}"
    bad = [(p["id"], p["split"]) for p in personas if p["split"] not in VALID]
    assert not bad, f"split must be one of {VALID}: {bad}"


def test_the_split_sizes_are_the_pre_registered_ones(personas):
    """PLAN.md E1 and DECISIONS.md D16. The sealed count is the load-bearing one:
    raising it after seeing results would be a post-hoc change to a
    pre-registered quantity."""
    c = Counter(p["split"] for p in personas)
    assert c["sealed"] == N_SEALED, f"sealed is {c['sealed']}, pre-registered {N_SEALED}"
    assert c["dev"] == N_DEV, f"dev is {c['dev']}, pre-registered {N_DEV}"
    assert len(personas) == N_SEALED + N_DEV


def test_no_pair_is_split_across_dev_and_sealed(personas):
    """The paired personas differ ONLY in the language sentence. Splitting a pair
    does not hold half of it back — it destroys the controlled comparison that
    cells 1 and 2 are built on, and E7's paired negation stress-test with it."""
    by_pair: dict[str, set[str]] = {}
    for p in personas:
        if p.get("pair_id"):
            by_pair.setdefault(p["pair_id"], set()).add(p["split"])
    broken = {k: v for k, v in by_pair.items() if len(v) > 1}
    assert not broken, f"pairs straddling the split: {broken}"


def test_every_pair_is_intact_as_two_instances(personas):
    """A `pair_id` seen once is a pair that lost a variant somewhere."""
    c = Counter(p["pair_id"] for p in personas if p.get("pair_id"))
    odd = {k: n for k, n in c.items() if n != 2}
    assert not odd, f"pair_ids not appearing exactly twice: {odd}"


def test_sealed_can_run_the_paired_negation_stress_test(personas):
    """E7 is a PAIRED test. A sealed set with no intact pair cannot run it, and
    the headline stress-test would then only ever have been measured on data the
    system was tuned against."""
    sealed_pairs = {p["pair_id"] for p in personas
                    if p["split"] == "sealed" and p.get("pair_id")}
    assert sealed_pairs, "no pair is sealed — E7 cannot be run held-out"


def test_controls_are_represented_on_both_sides(personas):
    """The non-effect — a seeker who says nothing about language must get an
    unchanged ranking — needs guarding in dev, where the bug is cheap to fix, AND
    verifying in sealed, where it is not tuned against. D16."""
    controls = [p for p in personas if p.get("control")]
    assert len(controls) >= 4, f"only {len(controls)} control personas"
    splits = Counter(p["split"] for p in controls)
    assert splits["dev"] >= 1, "no control in dev: the non-effect bug would be found late"
    assert splits["sealed"] >= 1, "no control in sealed: the non-effect is never held out"


def test_sealed_verticals_are_disjoint_from_dev(personas):
    """Not forced, but deliberate and load-bearing on how the number is read: the
    sealed run measures generalisation to unseen verticals, so it is a LOWER
    bound on dev rather than a replicate. D16 says so; if this ever stops being
    true, that framing has to change with it."""
    dev = {p["vertical"] for p in personas if p["split"] == "dev"}
    sealed = {p["vertical"] for p in personas if p["split"] == "sealed"}
    assert not (dev & sealed), (
        f"verticals now in both: {sorted(dev & sealed)} — D16's 'lower bound' "
        f"reading assumed disjointness")


def test_persona_ids_are_unique(personas):
    dupes = [k for k, n in Counter(p["id"] for p in personas).items() if n > 1]
    assert not dupes, f"duplicate persona ids: {dupes}"


def test_the_originating_bug_persona_is_in_dev(personas):
    """p5 is the query that started the project. It is the reference failure and
    development has to iterate against it; sealing it would mean never being able
    to work on the thing the project exists to fix. D16."""
    p5 = [p for p in personas if p.get("pair_id") == "p5"]
    assert p5, "p5 is missing entirely"
    assert all(p["split"] == "dev" for p in p5), "p5 (originating bug) must stay in dev"


def test_the_return_few_results_property_is_held_out_too(personas):
    """`expect_few_results` marks a persona where Norwegian is GENUINELY required,
    so the correct behaviour is to return few results or none and say so. D16
    recorded that the sealed set lacked such a case; D17 added
    `p11_barnevernspedagog` to fix it before any results existed. A system that
    manufactures matches here is worse than one returning nothing, so the property
    needs guarding in dev AND verifying held-out."""
    marked = [p for p in personas if p.get("expect_few_results")]
    assert marked, "no persona marks `expect_few_results`"
    splits = {p["split"] for p in marked}
    assert "dev" in splits, "the return-few property is not developed against"
    assert "sealed" in splits, (
        "the return-few property is not held out — a system that manufactures "
        "matches for a genuinely-Norwegian role would only ever be measured on "
        "data it was tuned against")
