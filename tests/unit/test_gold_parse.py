"""The gold parses are the oracle — these tests are what stop them flattering it.

A hand-authored seeker profile is the easiest place in this project to produce a
confidently wrong number. Two failure modes, both silent:

  1. WRITING S IN THE CORPUS'S VOCABULARY. Reach for ESCO labels and census enum
     values and the oracle stops describing a seeker. It would delete the
     vocabulary mismatch LIMITATIONS §14 measured — the seeker negates Norwegian,
     the accessible ad affirms English — and retrieval would score well because
     the input was pre-translated into the ads' own words. The verbatim-evidence
     rule is the mechanism against it, and it only works if it is checked.

  2. DEFAULTING AN UNSTATED FACET. A parse that fills in `language.norwegian:
     fluent` for a persona who never mentioned language makes every seeker
     constrained, and the corpus shrinks to 9.4% for everyone. That is the bug the
     four control personas were added to catch; a bad parse reintroduces it
     upstream of them.

The negative tests matter as much as the positive ones: they prove the validator
actually rejects these, rather than having a rule written down that nothing runs.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from finn_smart_search.eval import gold_parse as gp

PERSONAS = Path(__file__).resolve().parents[2] / "eval" / "personas.yaml"
PARSES = Path(__file__).resolve().parents[2] / "eval" / "gold_parses.yaml"


@pytest.fixture(scope="module")
def personas():
    return gp.load_personas(PERSONAS)


@pytest.fixture(scope="module")
def parses(personas):
    return gp.load(PARSES, personas)


# ── the file as committed loads and is internally consistent ──────────────────

def test_the_committed_parses_load(parses):
    assert parses, "no gold parses loaded"
    for pid, parse in parses.items():
        assert parse.constraints, f"{pid} has no constraints"


def test_every_evidence_string_is_verbatim_in_the_persona_query(parses, personas):
    """RULE 1, on the real file. `load` enforces it, so this asserts the file
    committed actually satisfies it rather than trusting that it was run."""
    for pid, parse in parses.items():
        query = gp._norm(personas[pid]["query"])
        for c in parse.constraints:
            assert c.evidence in query, (
                f"{pid}: {c.evidence!r} is not a substring of the frozen query")


def test_controls_have_no_language_constraint(parses, personas):
    """RULE 2, cross-file. Raises rather than returning, because a control
    persona with a language constraint is not a control."""
    gp.check_against_personas(parses, personas)


def test_the_authorisation_trap_is_not_parsed_as_language(parses):
    """c2 contains the word `norsk` inside `norsk autorisasjon` — a LICENSING
    credential. This is the trap the plan named on day one, and a gold parse is a
    new place to fall into it."""
    c2 = parses["c2_sykepleier_authorisation_only"]
    assert not [c for c in c2.constraints if c.facet.startswith("language.")], (
        "c2's parse states a language constraint; `norsk autorisasjon` is a "
        "credential, not a language claim")
    auth = [c for c in c2.constraints if c.facet == "credential.authorisation"]
    assert auth, "c2's authorisation credential is missing from its parse"
    assert all(c.priority == "hard" for c in auth), (
        "an authorisation is a hard constraint — a job requiring one the seeker "
        "lacks is not merely a worse match")


def test_language_values_are_capability_levels_not_negations(parses):
    """A seeker states what they HAVE. `none` is a level, not a denial — that is
    the containment framing in the data rather than only in the README."""
    for pid, parse in parses.items():
        for c in parse.constraints:
            if c.facet.startswith("language.") and c.facet != "language.other":
                assert c.value in gp.LANGUAGE_LEVELS, (
                    f"{pid}: {c.facet} = {c.value!r} is not a capability level")


# ── the validator rejects what it claims to reject ────────────────────────────

def _write(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "gold_parses.yaml"
    p.write_text(yaml.safe_dump({"gold_parses": rows}, allow_unicode=True),
                 encoding="utf-8")
    return p


def test_invented_evidence_is_rejected(tmp_path, personas):
    """The load-bearing check. A constraint the seeker never stated must not
    load, however plausible it looks."""
    bad = [{"persona_id": "p1_sykepleier_no_norsk", "constraints": [
        {"facet": "skill", "value": "intensive care",
         "priority": "soft", "evidence": "I have intensive care experience",
         "ad_side": "skills"}]}]
    with pytest.raises(gp.GoldParseError, match="not found verbatim"):
        gp.load(_write(tmp_path, bad), personas)


def test_paraphrased_evidence_is_rejected(tmp_path, personas):
    """Close is not verbatim. `census_validate` checks ad spans byte-for-byte for
    the same reason: a paraphrase is where an assumption enters unnoticed."""
    bad = [{"persona_id": "p1_sykepleier_no_norsk", "constraints": [
        {"facet": "language.norwegian", "value": "none", "priority": "hard",
         "evidence": "I don't speak Norwegian",
         "ad_side": "norwegian_requirement_level"}]}]
    with pytest.raises(gp.GoldParseError, match="not found verbatim"):
        gp.load(_write(tmp_path, bad), personas)


def test_a_control_persona_given_a_language_constraint_is_rejected(tmp_path, personas):
    """RULE 2. The evidence here IS verbatim — `norsk autorisasjon` really is in
    c2's query — so rule 1 passes and only the cross-check catches it. That is
    precisely the trap."""
    rows = [{"persona_id": "c2_sykepleier_authorisation_only", "constraints": [
        {"facet": "language.norwegian", "value": "fluent", "priority": "hard",
         "evidence": "norsk autorisasjon",
         "ad_side": "norwegian_requirement_level"}]}]
    parses = gp.load(_write(tmp_path, rows), personas)
    with pytest.raises(gp.GoldParseError, match="not a control"):
        gp.check_against_personas(parses, personas)


def test_unknown_facet_is_rejected(tmp_path, personas):
    bad = [{"persona_id": "p1_sykepleier_no_norsk", "constraints": [
        {"facet": "langauge.norwegian", "value": "none", "priority": "hard",
         "evidence": "I do not speak Norwegian", "ad_side": None}]}]
    with pytest.raises(gp.GoldParseError, match="unknown facet"):
        gp.load(_write(tmp_path, bad), personas)


def test_a_rebound_ad_side_is_rejected(tmp_path, personas):
    """The facet-to-ad_side mapping is fixed by the schema. Letting a parse
    declare its own would let an unscoreable constraint claim coverage it does
    not have, inflating `coverage()`."""
    bad = [{"persona_id": "p1_sykepleier_no_norsk", "constraints": [
        {"facet": "contract.shift", "value": "day", "priority": "soft",
         "evidence": "preferably day shifts", "ad_side": "seniority"}]}]
    with pytest.raises(gp.GoldParseError, match="ad_side"):
        gp.load(_write(tmp_path, bad), personas)


def test_bad_priority_is_rejected(tmp_path, personas):
    bad = [{"persona_id": "p1_sykepleier_no_norsk", "constraints": [
        {"facet": "occupation", "value": "nurse", "priority": "nice_to_have",
         "evidence": "I am a nurse", "ad_side": "occupation"}]}]
    with pytest.raises(gp.GoldParseError, match="priority"):
        gp.load(_write(tmp_path, bad), personas)


def test_unknown_persona_is_rejected(tmp_path, personas):
    bad = [{"persona_id": "p99_does_not_exist", "constraints": []}]
    with pytest.raises(gp.GoldParseError, match="unknown persona_id"):
        gp.load(_write(tmp_path, bad), personas)


# ── the bridge into the constraint stage ──────────────────────────────────────

def test_a_stated_constraint_becomes_a_language_constraint(parses, personas):
    prof = gp.to_seeker_profile(parses["p1_sykepleier_no_norsk"], personas)
    assert prof.language_constraint is not None
    assert prof.language_constraint.norwegian == "none"


def test_an_unstated_facet_becomes_none_not_a_default(parses, personas):
    """The single most important line in this file. `None` is what makes
    `language_severity` return exactly 0.0 and `apply` return the ranking
    unchanged; any default here re-creates the bug that would cost 3,507 ads."""
    prof = gp.to_seeker_profile(parses["c2_sykepleier_authorisation_only"], personas)
    assert prof.language_constraint is None


def test_an_unstated_facet_leaves_the_ranking_bit_identical(parses, personas):
    """End-to-end on the property, not just the field: a gold parse with no
    language row must not perturb a single score."""
    from finn_smart_search.retrieval import constraints as K

    prof = gp.to_seeker_profile(parses["c2_sykepleier_authorisation_only"], personas)
    scores = [0.9, 0.5, 0.31, 0.02]
    ads = [({"norwegian_requirement_level": lvl}, "no")
           for lvl in ("professional", "certified", "unstated",
                       "explicitly_not_required")]
    assert K.apply(scores, ads, prof, lam=0.7) == scores


# ── the ceiling this schema is designed to expose ─────────────────────────────

def test_coverage_reports_the_unscoreable_constraints(parses):
    """`coverage()` is not decoration. Most seeker-side facets have no corpus
    counterpart, so a perfect parser and a perfect encoder still cannot honour
    them, and that fraction belongs beside any headline number."""
    cov = gp.coverage(parses.values())
    assert cov["n_constraints"] > 0
    assert 0.0 <= cov["scoreable_share"] <= 1.0
    assert cov["unscoreable_facets"], (
        "no unscoreable facets found — either the corpus got richer or a parse "
        "is claiming coverage it does not have")
    assert cov["n_scoreable"] <= cov["n_constraints"]


def test_hard_constraints_are_tracked_separately_in_coverage(parses):
    """An unscoreable HARD constraint is worse than an unscoreable soft one: it
    is a must-have the system cannot even see."""
    cov = gp.coverage(parses.values())
    assert cov["n_hard"] > 0
    assert cov["n_hard_scoreable"] <= cov["n_hard"]


def test_every_dev_persona_has_a_parse(parses, personas):
    """The dev oracle cannot be run on a persona with no S. A dev persona added
    later without a parse would silently shrink the ideal-case eval rather than
    failing it."""
    dev = {i for i, p in personas.items() if p["split"] == "dev"}
    missing = sorted(dev - set(parses))
    assert not missing, f"dev personas with no gold parse: {missing}"


def test_every_parse_names_a_persona_that_still_exists(parses, personas):
    """A parse for a renamed or removed persona would load and score nothing."""
    orphans = sorted(set(parses) - set(personas))
    assert not orphans, f"parses with no persona: {orphans}"


def test_both_variants_of_a_parsed_pair_are_parsed(parses, personas):
    """A pair is only a controlled comparison if BOTH sides have an S. Parsing one
    variant and not the other would make the paired test compare a parsed seeker
    against nothing."""
    by_pair: dict[str, list[str]] = {}
    for pid in parses:
        pair = personas[pid].get("pair_id")
        if pair:
            by_pair.setdefault(pair, []).append(pid)
    half = {k: v for k, v in by_pair.items() if len(v) != 2}
    assert not half, f"pairs with only one variant parsed: {half}"


def test_paired_variants_differ_only_in_the_language_constraint(parses, personas):
    """The personas differ ONLY in the language sentence, so their parses must
    differ only in `language.*`. If anything else diverges, the gold parse has
    introduced a confound the frozen queries do not have — and the paired
    comparison would no longer isolate language."""
    by_pair: dict[str, list[str]] = {}
    for pid in parses:
        pair = personas[pid].get("pair_id")
        if pair:
            by_pair.setdefault(pair, []).append(pid)
    for pair, ids in by_pair.items():
        if len(ids) != 2:
            continue
        shapes = []
        for pid in sorted(ids):
            shapes.append(sorted(
                (c.facet, c.priority) for c in parses[pid].constraints
                if not c.facet.startswith("language.")))
        assert shapes[0] == shapes[1], (
            f"pair {pair}: the two parses differ outside language.*\n"
            f"  {sorted(ids)[0]}: {shapes[0]}\n"
            f"  {sorted(ids)[1]}: {shapes[1]}")
