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
