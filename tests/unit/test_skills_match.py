"""ESCO skill resolution. Grounded against `esco_skill` and the extracted glosses.

WHAT THIS MODULE IS FOR AND WHAT IT IS NOT WIRED INTO. Occupation is compared as a
taxonomy code with graded proximity; skills are still compared as text, so `PyTorch`
does not imply `machine learning`. ESCO has 10,063 skills, all bilingual. This
resolves extracted skill text onto them — and is deliberately NOT in the ranking,
because the judged ablation isolates occupation and constraints, not skills, and
there is no way to measure whether resolution helps.
"""
from __future__ import annotations

import pytest

from finn_smart_search.retrieval.skills_match import EscoSkillIndex, tokens

# Shaped after the real table: ESCO labels are VERB phrases, bilingual, median three
# words. These are real labels from `esco_skill`.
ROWS = [
    ("u_wound", "en", "carry out wound care"), ("u_wound", "no", "utføre sårstell"),
    ("u_med", "en", "assist in the administration of medication"),
    ("u_med", "no", "bistå ved legemiddelhåndtering"),
    ("u_fork", "en", "operate forklift"), ("u_fork", "no", "kjøre truck"),
    ("u_python", "en", "Python (computer programming)"),
    ("u_python", "no", "Python (dataprogrammering)"),
    ("u_supply", "en", "manage medical supply chains"),
    ("u_supply", "no", "administrere medisinske forsyningskjeder"),
]


@pytest.fixture(scope="module")
def idx():
    return EscoSkillIndex.build(ROWS)


def test_the_index_is_bilingual(idx):
    """100% of ESCO skills carry both labels; a Norwegian phrase must reach the same
    concept an English one does — the identity that made the occupation path work."""
    assert idx.resolve("wound care")[0].uri == idx.resolve("sårstell")[0].uri


def test_a_noun_phrase_reaches_a_verb_phrase_label(idx):
    """THE CORE DIFFICULTY, measured before any code: ESCO labels are verb phrases
    (`carry out wound care`) and the extractor produces noun phrases (`wound care`),
    so exact matching resolves only 2.5% of 5,000 real glosses."""
    m = idx.resolve("wound care")
    assert m and m[0].label_en == "carry out wound care"


def test_a_specific_label_is_not_penalised_for_being_long(idx):
    """Containment, not symmetric overlap. `assist in the administration of
    medication` is six words against a two-word phrase; symmetric scoring would
    reject the correct answer for being specific — the failure D18's outcome
    measured on occupations."""
    m = idx.resolve("medication administration")
    assert m and "administration of medication" in m[0].label_en


def test_grammatical_frame_words_do_not_drive_a_match(idx):
    """`carry out`, `assist in`, `manage` are the words that DIFFER between two
    descriptions of one skill, so they are stopped."""
    assert "carry" not in tokens("carry out wound care")
    assert "manage" not in tokens("manage medical supply chains")


def test_an_unrelated_phrase_does_not_resolve(idx):
    assert not idx.resolve("underwater basket weaving")


def test_an_empty_phrase_is_safe(idx):
    assert idx.resolve("") == []
    assert idx.resolve("the and of") == []


def test_results_are_ordered_best_first(idx):
    m = idx.resolve("medication", top_k=3, min_score=0.0)
    assert m == sorted(m, key=lambda x: -x.score)


def test_stemming_is_a_parameter_not_a_decision(idx):
    """THE TRADE THIS MODULE REFUSES TO SETTLE BY EYE, asserted as a capability here
    and measured at corpus scale below. Stemming lifts resolution on real glosses
    from 17% to 21% but collapses medication/medical and administration/administer,
    so `medication administration` switches from the correct label to `manage
    medical supply chains`. Higher resolution rate is not higher quality, and
    choosing needs a judged arm — §16 records three such choices going wrong in one
    sitting."""
    # AND STEMMING DOES NOT BRIDGE THE VERB/NOUN GAP, which is the honest finding:
    # English Snowball gives `operation` -> `oper` and `operate` -> `operat`, two
    # different stems. The measured lift from 17% to 21% came from index-side IDF
    # shifts, not from solving morphology. Recorded so nobody reaches for stemming
    # expecting it to fix what it does not.
    assert tokens("operation", stem=True) != tokens("operate", stem=True)
    assert tokens("operation", stem=True) != tokens("operation", stem=False)


def test_the_default_is_the_conservative_path(idx):
    """Unstemmed by default: lower resolution, correct on the flagship case."""
    assert idx.resolve("medication administration")[0].uri == "u_med"


def test_idf_prefers_the_rare_token():
    """`management` spans the taxonomy, `dialysis` does not. Without IDF every phrase
    containing a common verb resolves to the same crowd. Asserted on a fixture built
    to HAVE that skew, since a five-label fixture has no natural rarity."""
    rows = [(f"u{i}", "en", f"manage thing {i}") for i in range(20)]
    rows += [("u_rare", "en", "perform dialysis")]
    ix = EscoSkillIndex.build(rows)
    assert ix._idf("dialysis") > ix._idf("thing")


@pytest.mark.parametrize("phrase,expect", [
    ("wound care", "u_wound"),
    ("sårstell", "u_wound"),
    ("operate forklift", "u_fork"),
    ("Python", "u_python"),
])
def test_known_phrases_reach_their_concept(idx, phrase, expect):
    m = idx.resolve(phrase, min_score=0.3)
    assert m and m[0].uri == expect
