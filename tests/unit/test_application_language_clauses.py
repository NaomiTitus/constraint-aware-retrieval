"""What `application_language` must produce, on clauses measured in the corpus.

WHY THIS FILE EXISTS. The field has three legal values and the golden set
scores none of them, so until now the only evidence about it was 4 rare values
a human read. A recall probe over 40 ads selected BY THE CLAUSE — not by the
model's output — put census-v10 at 36/40, and the 4 failures were not random:

  1. "Søknadstekst og CV må vere på norsk eller eit anna skandinavisk språk"
     -> the model answered `english_accepted`. No English is named anywhere.
        This is the failure direction that MATTERS: it tells a seeker who reads
        no Norwegian that they may apply in English, when they may not.
  2. "Attester må være oversatt til norsk"  (3 ads, one employer's boilerplate)
     -> `unstated`. A rule about the language of the attachments IS this field.

Case 1 was the argument for adding a fourth enum value. It is answered more
cheaply: LIMITATIONS §10 measured only 3 corpus ads needing `scandinavian_
accepted`, and adding it would reintroduce the exact vocabulary the model was
borrowing when this field held three values from other enums. What was missing
was not a value but a RULE — census-v11 says which of the three a
Scandinavian-only clause takes, and that it is not the English one.

GROUNDING (STANDARDS.md §3.0): every clause below is verbatim from the corpus,
with the uuid it came from. None is invented.
"""
import pytest

from finn_smart_search.understanding.census_prompt import SYSTEM, TOOL

pytestmark = pytest.mark.unit

LEGAL = set(TOOL["input_schema"]["properties"]["application_language"]["enum"])

# uuid, clause verbatim, the value the prompt's rules must yield
CLAUSES = [
    # English named -> english_accepted. 135 ads carry a variant of this.
    ("163ab8be", "Dokumentene må være på norsk/skandinavisk eller engelsk.",
     "english_accepted"),
    ("034c70a1", "Please note that all documentation must be in English or a "
                 "Scandinavian language.", "english_accepted"),
    ("08f980a7", "Your application and supporting documentation must be in English.",
     "english_accepted"),
    # A REAL TYPO in the ad: "ellerengelsk", no space. English IS named and the
    # model reads it correctly; the probe's own expectation regex used \bengelsk
    # and did not. Kept because a word-boundary assumption broke here once.
    ("1ef6e61b", "Dokumentene må være på norsk/skandinavisk ellerengelsk.",
     "english_accepted"),
    # Scandinavian named, English NOT -> norwegian_required, never english.
    ("1242f465", "Søknadstekst og CV må vere på norsk eller eit anna "
                 "skandinavisk språk.", "norwegian_required"),
    ("47331148", "Vedlagt dokumentasjon må være i pdf-format på skandinavisk.",
     "norwegian_required"),
    # Norwegian only.
    ("25b35173", "Søknaden må skrives på norsk.", "norwegian_required"),
    ("09f70cee", "Søknad skrives på norsk.", "norwegian_required"),
    # Translation of attachments IS the language of the application's documents.
    ("159eab4e", "Attester må være oversatt til norsk", "norwegian_required"),
]


@pytest.mark.parametrize("uuid,clause,expected", CLAUSES)
def test_the_expected_value_is_legal(uuid, clause, expected):
    """Guards the fixture itself: an expectation outside the enum would make
    every assertion below unsatisfiable, which is how this field got into
    trouble in the first place."""
    assert expected in LEGAL, f"{expected!r} is not a legal value: {sorted(LEGAL)}"


def test_the_prompt_states_which_value_a_scandinavian_only_clause_takes():
    """The measured failure: with no rule, the model chose `english_accepted`
    for a clause that names only Scandinavian — wrong in the direction that
    costs a seeker a real application."""
    assert "skandinavisk" in SYSTEM.lower()
    low = SYSTEM.lower()
    i = low.find("uten å nevne engelsk")
    if i < 0:
        i = low.find("without naming english")
    assert i >= 0, ("the prompt must say what a Scandinavian-only clause takes; "
                    "measured, the model guesses english_accepted")
    window = low[i:i + 400]
    assert "norwegian_required" in window, window[:200]


def test_the_prompt_says_a_translation_requirement_is_this_field():
    """`Attester må være oversatt til norsk` was recorded `unstated` on 3 ads."""
    assert "oversatt" in SYSTEM.lower(), (
        "the prompt must say a translation-of-attachments rule sets "
        "application_language; measured, it was recorded `unstated`")


def test_every_legal_value_is_named_in_the_prose():
    for v in LEGAL:
        assert f"`{v}`" in SYSTEM, f"{v!r} is legal but never named in the prompt"
