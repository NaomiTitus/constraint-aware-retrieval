"""Two facets had no definition anywhere, and behaved accordingly.

FOUND BY THE HUMAN REVIEW, on the full 46-row exceptions set.

`security_clearance_required` was declared as a bare `{"type": "boolean"}`
with no description, and the prompt never mentioned it. Measured on ads naming
a politiattest and no clearance, the extractor answered True on 4 and False on
23 — the same signature as the `application_language` bug: near-identical
inputs, unstable output. Nothing was wrong with the reviewer's verdicts; there
was no rule for them to be wrong against.

The reviewer set the rule: ANY security-type check counts. Measured
consequences, which is why the exclusions below are load-bearing:

    broad rule matches           3,643 ads  35.8%
    clearance only                 377 ads   3.7%
    `autorisasjon` as a LICENCE   1,625 ads, 726 with no security check at all
    MRSA / tuberculosis            261 ads

So the trap is `autorisasjon`, which is a professional licence far more often
than a security term — the same word that has been this project's precision
trap since the plan was written. And health screening is not security.

Against the reviewer's rule the extractor scored 13 correct True, 0 wrong True
and 28 MISSED — it under-fires, it does not over-fire.

`relocation_support` had the same hole. The reviewer separated three things the
extractor was collapsing: help MOVING there (the field), accommodation provided
BECAUSE OF the job (rotation, staff housing, an assignment), and travel while
working. 16 of 21 `offered` rows were rejected on that basis.

GROUNDING (STANDARDS.md §3.0): every clause below is verbatim from the corpus
with the uuid it came from.
"""
import pytest

from finn_smart_search.understanding.census_prompt import SYSTEM, TOOL

pytestmark = pytest.mark.unit

PROPS = TOOL["input_schema"]["properties"]

# --- security_clearance_required ------------------------------------------
SECURITY_TRUE = [
    ("001afe25", "Før oppstart må det legges fram gyldig politiattest."),
    ("002f1ad1", "Du må kunne legge fram politiattest uten merknader før du tiltrer i stillingen"),
    ("013fda3d", "DFD samarbeider med Semac for bakgrunnssjekk av aktuelle kandidater."),
    ("0301c23a", "Vi gjennomfører bakgrunnssjekk av aktuelle kandidater."),
    ("35", "Du må kunne sikkerhetsklareres til HEMMELIG og NATO SECRET før tiltredelse"),
    ("39", "God vandel, uttømmende politiattest vil bli etterspurt"),
]
SECURITY_FALSE = [
    # A PROFESSIONAL LICENCE. 726 corpus ads carry one and no security check.
    ("00b19841", "Norsk autorisasjon som sykepleier"),
    ("009955bb", "Norsk autorisasjon som helsefagarbeider"),
    ("0068a96d", "Bioingeniør med norsk autorisasjon"),
    # HEALTH screening, not security. 261 ads.
    ("009955bb", "Skjema for forhåndsundersøkelse av tuberkulose og MRSA må leveres før tiltredelse"),
]

# --- relocation_support ----------------------------------------------------
RELOCATION_OFFERED = [
    ("0729e888", "Hjelp til å finne bolig"),
    ("079b283e", "Vi er behjelpelig med bolig."),
    ("0865631f", "Vi kan være behjelpelig med å finne bolig."),
    ("010d4669", "Dekning av flytteutgifter i h.h.t reglement"),
    ("02a54ddc", "dekning av flytteutgifter etter vedtatte retningslinjer"),
    # the two the reviewer rejected and the evidence does not support rejecting
    ("row14", "Vi hjelper nye arbeidstakere i Norge med å finne bolig"),
    ("row18", "Hjelp å skaffe leilighet i Bergen"),
]
RELOCATION_NOT = [
    ("085b441c", "Betalt bolig under hele oppdragsperioden"),
    ("08cc1183", "Betalt bolig under oppdrag"),
    ("142cab29", "Kostnadsfri bolig når oppdraget krever at du bor borte"),
    ("row15", "Ansattbolig i Noresund."),
    ("row13", "dekning av reise og overnatting, betalt reisetid"),
]


def test_security_clearance_required_has_a_description_at_all():
    """It was `{"type": "boolean"}`. A field with no description is a field the
    model defines for itself, differently on different ads."""
    d = PROPS["security_clearance_required"].get("description") or ""
    assert len(d) > 40, f"still undefined: {d!r}"


def test_relocation_support_has_a_description_at_all():
    d = PROPS["relocation_support"].get("description") or ""
    assert len(d) > 40, f"still undefined: {d!r}"


def test_the_security_definition_names_what_counts():
    d = (PROPS["security_clearance_required"].get("description") or "").lower()
    for token in ("politiattest", "vandel", "bakgrunnssjekk", "sikkerhetsklarer"):
        assert token in d, f"{token!r} must be named as counting: {d}"


def test_the_security_definition_excludes_the_licence_trap():
    """`autorisasjon` is a professional licence in 726 ads with no security
    check. It is also a security term. The description must separate them."""
    d = (PROPS["security_clearance_required"].get("description") or "").lower()
    assert "autorisasjon" in d
    assert "sykepleier" in d or "lisens" in d or "licence" in d or "profesjon" in d, d


def test_the_security_definition_excludes_health_screening():
    d = (PROPS["security_clearance_required"].get("description") or "").lower()
    assert "mrsa" in d or "tuberkulose" in d, d


def test_the_relocation_definition_separates_moving_from_being_housed():
    d = (PROPS["relocation_support"].get("description") or "").lower()
    for token in ("flytte", "rotasjon", "oppdrag"):
        assert token in d, f"{token!r} must appear so the three cases are separated: {d}"


@pytest.mark.parametrize("uuid,clause", SECURITY_TRUE + SECURITY_FALSE
                         + RELOCATION_OFFERED + RELOCATION_NOT)
def test_the_fixture_clauses_are_real_text_not_paraphrase(uuid, clause):
    """Guards the fixture: these must be verbatim, because the prompt is being
    written against them. A paraphrase here would tune the prompt to language
    no advertisement uses."""
    assert clause.strip() == clause and len(clause) > 15


def test_the_prompt_prose_carries_both_rules():
    low = SYSTEM.lower()
    assert "politiattest" in low and "bakgrunnssjekk" in low
    assert "flytteutgifter" in low or "hjelp til å finne bolig" in low


# ===========================================================================
# census-v12 was measured on the 46 rows and was only partly right. These pin
# the three specific failures so a later prompt edit cannot undo them.
#
#   row 34  "Det er krav om at gyldig politiattest fremvises før tiltredelse"
#           flipped True -> False. The description led with ONE inclusion
#           sentence and then THREE "FALSE for..." sentences; the negatives
#           drowned the positive. A rule whose exceptions outweigh it reads as
#           an exception rule.
#   row 17  "Betalt bolig under hele oppdragsperioden"   stayed `offered`
#   row 32  "Gratis bolig i hele arbeidsperioden"        stayed `offered`
#           — both almost verbatim the examples already in the prompt, so
#           listing more examples was not the fix. v13 states the TEST instead:
#           does the help end when the job ends?
# ===========================================================================

def test_a_bare_politiattest_requirement_is_stated_to_be_enough():
    """Row 34's exact clause, and the failure it caused."""
    d = (PROPS["security_clearance_required"].get("description") or "").lower()
    assert "alone is enough" in d or "alene" in d, d
    assert "politiattest" in d
    # the inclusions must not be outnumbered by the exclusions
    assert d.count("false") <= 2, (
        f"{d.count('false')} negative clauses; v12 had three and the model "
        f"stopped saying True at all")


def test_the_relocation_rule_states_a_TEST_not_only_examples():
    """v12 listed 'betalt bolig under oppdraget' and rows 17 and 32 still came
    back `offered` on near-identical wording. More examples was not the fix."""
    d = (PROPS["relocation_support"].get("description") or "").lower()
    assert "does the help end when the job ends" in d or "so lenge" in d, d
    for token in ("gratis bolig", "fri bolig", "betalt bolig"):
        assert token in d, f"{token!r} is the commonest wording and must be named: {d}"


def test_the_prompt_prose_states_the_relocation_test_too():
    low = SYSTEM.lower()
    assert "does the help end when the job ends" in low, \
        "the discriminating test must be in the prose, not only the schema"
