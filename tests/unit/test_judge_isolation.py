"""THE NON-NEGOTIABLE. STANDARDS.md §4, and the scenarios approved 2026-09-27.

> The judge sees the raw advertisement text and the persona's stated constraints.
> The judge NEVER sees the extractor's output.

If this fails, `CVR@10` measures the extractor agreeing with itself and every
headline number in the project is void. `JUDGING_PROTOCOL.md` calls it "the single
way this project could produce a confidently wrong result".

HOW THE RULE IS ENCODED, AND WHY IT IS NOT THE LITERAL ONE. §4 says the prompt must
contain no `ad_facets` field name. Grounding the closed key vocabulary against the
table (JUDGE_SCENARIOS G4) showed two of the sixteen keys are ordinary English
words — `skills` and `seniority` — and `JUDGING_PROTOCOL.md`'s own grade rubric
reads "judged against the persona's stated **skills** and preferences". A literal
test fails on the protocol's own required wording.

Ruled 2026-09-27, recorded as an erratum rather than an edit to the pre-registered
protocol: test the 14 DISTINCTIVE keys, every column name, the literal `ad_facets`,
and additionally that NO snake_case identifier appears anywhere in the prompt.
`skills` and `seniority` are allowlisted as English, and the snake_case rule still
catches them the moment they appear as identifiers. Strict where it matters, honest
about the two exceptions.
"""
from __future__ import annotations

import re

import pytest

from finn_smart_search.eval import judge_llm
from tests.conftest import requires_corpus

# JUDGE_SCENARIOS G4 — the closed vocabulary, read off `ad_facets` itself
# (16 JSON keys), not invented.
FACET_KEYS = frozenset({
    "application_language", "authorisation_required", "conflicting_statements",
    "english_accessible", "evidence_basis", "evidence_spans", "evidence_strength",
    "implicit_evidence", "min_years_experience", "norwegian_requirement_level",
    "relocation_support", "security_clearance_required", "seniority", "skills",
    "stated_working_language", "visa_sponsorship",
})
# G4 — the 8 columns of `ad_facets`.
FACET_COLUMNS = frozenset({
    "uuid", "facets", "english_accessible", "demoted", "reasons",
    "prompt_version", "extractor_version", "extracted_at",
})
# Ordinary English words that happen to be facet names. Allowlisted by ruling; the
# snake_case assertion below is what keeps them from leaking as identifiers.
ENGLISH_ALLOWLIST = frozenset({"skills", "seniority", "facets", "reasons", "uuid"})

DISTINCTIVE = (FACET_KEYS | FACET_COLUMNS) - ENGLISH_ALLOWLIST

_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

# A real persona query and a real ad shape. The persona is `p1_sykepleier_no_norsk`
# verbatim from eval/personas.yaml; the ad is block-joined with `\n` and led by
# `•`, which is the production shape — G1 records p50 of 32 newlines per ad and G2
# records `•` leading 4,600 blocks.
PERSONA = ("I am a nurse with hospital and nursing-home experience. I do not "
           "speak Norwegian. I am looking for a permanent position, preferably "
           "day shifts.")
AD_TITLE = "Sykepleier søkes til sykehjem"
AD_TEXT = ("Vi søker sykepleier til vårt sykehjem.\n"
           "• Norsk autorisasjon som sykepleier\n"
           "• Gode norskkunnskaper, både muntlig og skriftlig\n"
           "• Erfaring fra sykehjem er en fordel, men ikke et krav")


@pytest.fixture(scope="module")
def prompt() -> str:
    return judge_llm.render_prompt(PERSONA, AD_TITLE, AD_TEXT)


def test_no_distinctive_facet_field_name_appears_in_the_prompt(prompt):
    """Scenario 2. The 14 keys and 8 columns that are NOT ordinary English."""
    low = prompt.lower()
    leaked = sorted(n for n in DISTINCTIVE if n in low)
    assert not leaked, (
        f"extractor field name(s) leaked into the judge prompt: {leaked}. "
        f"CVR@10 would measure the extractor agreeing with itself.")


def test_the_literal_string_ad_facets_never_appears(prompt):
    assert "ad_facets" not in prompt.lower()


def test_no_snake_case_identifier_appears_anywhere_in_the_prompt(prompt):
    """Scenario 3. The catch-all: this is what still traps `skills` or `seniority`
    if they are ever written as identifiers rather than as English, and it traps
    any facet name added to the schema after this test was written."""
    found = sorted(set(_SNAKE.findall(prompt)))
    assert not found, (
        f"snake_case identifier(s) in the judge prompt: {found}. The prompt is "
        f"prose for a judge; field names belong in the tool schema.")


def test_the_allowlist_is_only_the_two_words_the_ruling_covers():
    """A guard on the guard. If someone widens the allowlist, the isolation test
    silently weakens, and that must be a deliberate visible change."""
    assert ENGLISH_ALLOWLIST & FACET_KEYS == {"skills", "seniority"}


@requires_corpus
def test_the_facet_vocabulary_matches_the_table_it_was_grounded_in(corpus_con):
    """G4 was measured once; the schema can change. If a key is added to
    `ad_facets` and not to FACET_KEYS here, the isolation test stops covering it —
    so the vocabulary is re-derived from the live table and compared."""
    import json
    keys: set[str] = set()
    for (raw,) in corpus_con.execute("SELECT facets FROM ad_facets LIMIT 500").fetchall():
        f = json.loads(raw) if isinstance(raw, str) else raw
        keys |= set(f)
    missing = sorted(keys - FACET_KEYS)
    assert not missing, (
        f"ad_facets gained key(s) {missing} that the isolation test does not "
        f"cover. Add them to FACET_KEYS.")


def test_the_module_does_not_reference_the_extractor_table_at_all():
    """Isolation "by construction", the first of the protocol's three mechanisms.
    A prompt can be clean while the module still loads facets and uses them to
    pick which ad to show."""
    import inspect
    src = inspect.getsource(judge_llm).lower()
    for forbidden in ("ad_facets", "norwegian_requirement_level", "english_accessible"):
        # The docstring is allowed to NAME what it must not import; only code is
        # checked, so strip string literals first.
        code = re.sub(r'""".*?"""', "", src, flags=re.S)
        code = re.sub(r"'''.*?'''", "", code, flags=re.S)
        assert forbidden not in code, f"{forbidden} referenced in judge_llm code"
