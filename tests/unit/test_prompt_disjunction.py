"""Regression guard for the Nordic-OR-English disjunction.

WHY THIS FILE EXISTS. The 50-ad pilot classified both of these as
`scandinavian_accepted`, hiding them from the English-only seeker this system
is built for:

    "Må beherske skandinavisk eller engelsk tale."          (ad #38)
    "kunne prate engelsk eller ett nordisk språk"           (ad #24)

In both the model quoted the correct span and then picked the wrong level. The
SYSTEM prompt already carried the governing rule -- precedence 2, SCOPE BEATS
BAR, "If English is among them, either_norwegian_or_english -- whatever the
bar". It was not enough, for two reasons this file pins:

  1. The LEVEL NAME is `either_norwegian_or_english`. Neither ad contains the
     word "norsk", so the name itself argues against the correct answer.
  2. The only Nordic few-shot (#6, "Beherske et nordisk språk flytende") maps a
     Nordic-language sentence to `scandinavian_accepted`. A sentence containing
     "nordisk"/"skandinavisk" therefore had exactly one demonstrated landing
     place, and the `eller engelsk` half had none.

Measured over the 10,166-ad corpus: 436 ads carry a Nordic-OR-English
disjunction and 364 of them (3.6% of the corpus) never also say "norsk eller
engelsk" -- so no other rule rescues them. That is larger than the original
"norsk eller engelsk" finding (304 ads) which motivated the whole level scheme.

These are STATIC tests over the prompt text. They cannot prove the model obeys
the rule -- only the pilot can, and it is the real verification. What they do
is stop the rule and its demonstration being edited away silently, which is the
failure mode that produced the bug: the rule was present in prose the entire
time and still lost to a competing few-shot.
"""
import re

import pytest

from finn_smart_search.understanding import census_prompt as cp

pytestmark = pytest.mark.unit

SYSTEM = cp.SYSTEM


def levels_demonstrated():
    return [f["norwegian_requirement_level"] for *_, f in cp.FEWSHOT]


def spans_of(level):
    out = []
    for _t, _b, _l, f in cp.FEWSHOT:
        if f["norwegian_requirement_level"] == level:
            out += [s["span"] for s in f.get("evidence_spans", [])]
    return out


# ── the rule must say the disjunction need not name Norwegian ────────────────

def test_rule_states_english_wins_without_the_word_norsk():
    """The governing sentence must survive a prompt edit. Asserting merely that
    "engelsk" appears somewhere would pass against any version of this file --
    including the one that shipped the bug."""
    low = SYSTEM.lower()
    assert "either_norwegian_or_english" in low
    # The rule must explicitly cover the case where Norwegian is NOT named.
    assert re.search(r"need not|does not need to|even when norwegian is not|without naming norsk"
                     r"|whether or not the sentence names", low), \
        "prompt must state the disjunction need not contain the word 'norsk'"


def test_rule_names_the_scandinavian_or_english_shape():
    """The exact surface forms the corpus uses. 436 ads carry one of these."""
    low = SYSTEM.lower()
    assert "skandinavisk eller engelsk" in low, "the commonest failing surface form"
    assert "nordisk" in low and "engelsk" in low


def test_scandinavian_accepted_is_scoped_to_no_english():
    """`scandinavian_accepted` must be defined by the ABSENCE of English, not
    merely by the presence of a Nordic language -- that is precisely the
    inference the model made.

    Order-sensitive by design: the level name must come FIRST and the
    English-absent condition must follow it. The old prompt stated the
    condition as a subordinate clause ahead of the name ("...but English is
    not, `scandinavian_accepted`"), where it reads as one branch of a rule
    rather than as the definition of the level. Verified to fail against
    HEAD~1 before this test was kept."""
    low = SYSTEM.lower()
    assert re.search(r"scandinavian_accepted[\s\S]{0,400}?(but english is not|no english|"
                     r"english is absent|without english|and english is not)", low, re.S), \
        "scandinavian_accepted must be conditioned on English being absent"


# ── the rule must be demonstrated, not merely stated ─────────────────────────

def test_a_fewshot_demonstrates_nordic_or_english():
    """THE test. The bug was a stated rule losing to a demonstrated counter-
    example; a rule with no demonstration of its own is how that happens.

    Requires a few-shot whose span contains a Nordic term, contains English,
    and lands on either_norwegian_or_english."""
    nordic = re.compile(r"skandinavisk|nordisk|svensk|dansk|scandinavian", re.I)
    english = re.compile(r"engelsk|english", re.I)
    ok = [s for s in spans_of("either_norwegian_or_english")
          if nordic.search(s) and english.search(s)]
    assert ok, (
        "no few-shot demonstrates a Nordic-OR-English disjunction landing on "
        "either_norwegian_or_english; the only Nordic demonstration maps to "
        "scandinavian_accepted, which is what the model copied"
    )


def test_the_nordic_scandinavian_accepted_fewshot_has_no_english():
    """Guards the contrast pair. If a future edit adds 'eller engelsk' to
    few-shot #6 it becomes a demonstration of the WRONG answer, and this whole
    file would otherwise still pass."""
    english = re.compile(r"engelsk|english", re.I)
    for s in spans_of("scandinavian_accepted"):
        assert not english.search(s), \
            f"a scandinavian_accepted example must not contain English: {s!r}"


def test_both_pilot_failures_are_representable():
    """The two real sentences the pilot got wrong, asserted against the rule
    text rather than paraphrases of it."""
    for sentence in ("Må beherske skandinavisk eller engelsk tale.",
                     "Kandidater bør ha førerkort for bil, og kunne prate engelsk "
                     "eller ett nordisk språk."):
        assert re.search(r"(skandinavisk|nordisk)", sentence, re.I)
        assert re.search(r"engelsk", sentence, re.I)


# ── the version must move when the prompt does ───────────────────────────────

def test_prompt_version_advanced_past_the_pilot():
    """census-v4 is the version that produced the 2 misses. A prompt edit that
    does not bump the version silently serves cached v4 answers -- the cache key
    includes PROMPT_VERSION, so this is the difference between re-measuring and
    re-reading the bug."""
    assert cp.PROMPT_VERSION != "census-v4", \
        "bump PROMPT_VERSION or the cache returns the pre-fix classifications"


# ── v6: the conjunction guard ────────────────────────────────────────────────
#
# v5 fixed the disjunction under-call and introduced a LARGER over-call. It told
# the model to "ask one question first: is English in the accepted set?" and to
# stop at the first matching rule -- so precedence 2 fired before precedence 4
# ("OR is not AND") could be read. Both v5 regressions were conjunctions:
#
#   "Du må kunne gjøre deg forstått på norsk OG engelsk"        (#18)
#   "Kommuniserer godt på et skandinavisk språk OG engelsk"     (#37)
#
# In a conjunction English is an ADDITIONAL requirement, not an alternative, so
# Norwegian/Scandinavian is still required and the ad is NOT English-accessible.
#
# Measured over the corpus: 781 ads carry the disjunction shape, 863 the
# conjunction shape. The v5 over-call was 1.1x the size of the under-call it
# fixed -- which is why the English-wins rule has to be gated on the connective
# rather than on English merely appearing.

def test_english_wins_is_gated_on_the_connective_not_mere_presence():
    """The v5 bug in one assertion. The rule must turn on eller/or vs og/and,
    not on whether English is mentioned."""
    low = SYSTEM.lower()
    i = low.find("either_norwegian_or_english")
    assert i != -1
    window = low[max(0, i - 1200):i + 1200]
    assert re.search(r"\beller\b|\bor\b", window), "the rule must name the disjunctive connective"
    assert re.search(r"\bog\b|\band\b", window), \
        "the rule must contrast the conjunctive connective in the same breath"


def test_conjunction_is_not_accessible():
    """`norsk og engelsk` must be stated as NOT accessible, adjacent enough to
    the English-wins rule that a model reading rule 2 cannot miss it."""
    low = SYSTEM.lower()
    assert re.search(r"(norsk|skandinavisk)\w*\s+og\s+engelsk", low), \
        "the prompt must show a conjunction surface form verbatim"
    assert re.search(r"og\s+engelsk[\s\S]{0,500}?(not|ikke|still required|begge|both)", low), \
        "the conjunction example must be labelled as still requiring Norwegian"


def test_a_fewshot_demonstrates_the_conjunction():
    """v5's lesson: a rule with no demonstration loses to a rule that has one.
    The conjunction needs its own worked example or the same failure recurs."""
    conj = re.compile(r"(norsk|skandinavisk|nordisk)\w*\s+og\s+(engelsk|english)", re.I)
    hits = []
    for _t, _b, _l, f in cp.FEWSHOT:
        for sp in f.get("evidence_spans", []):
            if conj.search(sp["span"]):
                hits.append((f["norwegian_requirement_level"], sp["span"]))
    assert hits, "no few-shot demonstrates an X-AND-English conjunction"
    for level, span in hits:
        assert level != "either_norwegian_or_english", (
            f"a conjunction must NOT be demonstrated as accessible: {span!r} -> {level}"
        )


def test_precedence_puts_the_connective_check_before_the_english_check():
    """Rule 2 says "stop at the first that matches". If the English test is
    reachable before the OR/AND test, v5's ordering bug is still live."""
    low = SYSTEM.lower()
    or_not_and = low.find("or is not and")
    scope = low.find("scope beats bar")
    assert or_not_and != -1 and scope != -1
    assert or_not_and < scope, (
        "the OR-is-not-AND test must precede SCOPE BEATS BAR, or the English "
        "rule fires first and conjunctions are misread as accessible"
    )


def test_prompt_version_advanced_past_v5():
    assert cp.PROMPT_VERSION not in ("census-v4", "census-v5"), \
        "bump PROMPT_VERSION or the cache returns the v5 over-calls"


# ── v7: the documentation-language clause, and the shape of the few-shots ─────
#
# FINDING. 135 corpus ads say "dokumentasjon ... må være på et skandinavisk
# språk eller engelsk" — a statement about what language your CV may be in, not
# a demand on you. That is 16.6% of the 813 ads carrying any X-or-English
# disjunction, and 51 of the 135 contain NO other applicant-directed language
# wording, so the documentation clause is the only thing a verdict can come
# from. 14 are in English ("documentation to be considered must be in a
# Scandinavian language or English").
#
# The prompt had no rule for it and no few-shot, while SIX few-shots demonstrate
# "skandinavisk eller engelsk" as accessible. One of the 169 real outputs
# already quoted the clause as language evidence (c514de15); the verdict
# happened to be carried by a different span, which is what made it invisible.
#
# This is the same failure the probes found twice in MY OWN regex, recorded in
# DECISIONS D11 — and it was never carried back into the prompt.

def test_the_documentation_language_clause_is_named_as_a_non_requirement():
    low = SYSTEM.lower()
    assert re.search(r"dokumentasjon|vitnem|attest|documentation|diploma|transcript", low), \
        "the prompt must name the documentation-language clause"
    assert re.search(r"(dokumentasjon|documentation)[\s\S]{0,400}?"
                     r"(not a language requirement|never a language requirement|"
                     r"ikke et språkkrav|unstated)", low), \
        "and must say it is not a language requirement"


def test_no_fewshot_quotes_a_documentation_clause_as_evidence():
    """A demonstration of the wrong answer beats a stated rule — that is how the
    census-v4 disjunction bug happened. So no example may quote one."""
    doc = re.compile(r"dokumentasjon|vitnem[åa]l|documentation|diploma|transcript", re.I)
    bad = [(f["norwegian_requirement_level"], s["span"])
           for *_x, f in cp.FEWSHOT for s in f.get("evidence_spans", [])
           if doc.search(s["span"])]
    assert bad == [], f"a documentation clause quoted as language evidence: {bad}"


def test_a_fewshot_demonstrates_the_documentation_clause_landing_on_unstated():
    """51 of the 135 have no other language wording, so the model must have seen
    one worked example that ends in silence."""
    doc = re.compile(r"dokumentasjon|vitnem[åa]l|documentation", re.I)
    hits = [f for _t, body, _l, f in cp.FEWSHOT if doc.search(body)]
    assert hits, "no few-shot body contains a documentation-language clause"
    assert any(f["norwegian_requirement_level"] == "unstated" and not f.get("evidence_spans")
               for f in hits), "one must land on `unstated` with no spans"


def test_the_rule_names_the_ENGLISH_surface_forms_too():
    """6.5% of the corpus carries English text and 103 ads word the requirement
    in English: "Norwegian or English" 31, "fluent in Norwegian" 20,
    "Scandinavian or English" 12. A Norwegian-only rule is half a rule — and
    writing the exclusion only in Norwegian is exactly the `dokument` /
    `documentation` blind spot from D11, at the prompt level."""
    low = SYSTEM.lower()
    assert "scandinavian or english" in low
    assert "norwegian or english" in low


def test_a_three_way_comma_disjunction_is_distinguished_from_a_comma_list():
    """104 ads write "norsk, engelsk eller polsk" — a comma LIST closed by a
    disjunction, which is accessible. 21 write a bare "norsk, engelsk" with no
    connective, which is a conjunction. The prompt stated only the second rule,
    which is actively wrong for the 104."""
    low = SYSTEM.lower()
    assert re.search(r"norsk, engelsk eller", low), \
        "the three-way form must appear verbatim; it is 5x commoner than the bare list"


def test_fewshot_bodies_have_the_production_block_shape():
    """census.build_request sends body = description_text = blocks joined with
    "\\n" (html_clean.blocks_to_text). Real ads: median 31 blocks, median 2,500
    chars. The few-shot bodies were single-line prose averaging 201 chars — so
    every demonstration the model saw was 1/12th the length of a real ad, in a
    shape no ad has. Same class as STANDARDS § 3.1 row 1.

    NOT held to production LENGTH, deliberately. Real ads are median 31 blocks;
    matching that across 13 examples would roughly triple a cached prefix that
    already measures 9,217 tokens, at 0.1x input on every one of 9,823 calls.
    The load-bearing property is the SHAPE — newline-joined blocks, bullets
    without terminal punctuation — not the block count. So: at least 3 blocks,
    and a stated trade-off rather than a silently weakened assertion."""
    flat = [t for t, body, _l, _f in cp.FEWSHOT if body.count("\n") < 2]
    assert flat == [], f"few-shot bodies still single-line prose: {flat}"
    assert sum(b.count("\n") + 1 for _t, b, _l, _f in cp.FEWSHOT) >= 60, \
        "aggregate block count collapsed; the examples are prose again"


def test_fewshot_spans_are_whole_blocks_of_their_own_body():
    """The model imitates span shape. A demonstration that appends a full stop to
    a bullet teaches it to quote a string that is not a substring of the ad —
    which is the 55%-demotion mechanism. Measured: 3 of 9 spans did not occur
    verbatim in the corpus, each existing only WITHOUT the added period."""
    bad = []
    for title, body, _l, f in cp.FEWSHOT:
        blocks = set(body.split("\n"))
        for s in f.get("evidence_spans", []):
            if s["span"] not in blocks:
                bad.append((title, s["span"]))
    assert bad == [], f"spans that are not a whole block of their body: {bad}"


def test_at_least_one_fewshot_span_is_an_unterminated_bullet():
    """52% of the 908 real blocks carrying a disjunction have no terminal full
    stop, and 116 of 169 real emitted spans carry none."""
    unterminated = [s for lvl in ("either_norwegian_or_english", "professional",
                                  "certified", "scandinavian_accepted")
                    for s in spans_of(lvl) if not s.rstrip().endswith((".", "!", "?"))]
    assert unterminated, "every demonstrated span ends in punctuation; real ones do not"
