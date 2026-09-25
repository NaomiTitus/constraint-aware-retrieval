"""Contact PII must not ride out of the corpus inside a model-generated facet.

THE FINDING. `contactList` is excluded from every export (DECISIONS D1 era), but
the same details are in the ad BODY: measured over the corpus, the contact
person's full name also appears in `description_text` on 11.8% of ads, their
email on 10.1%, their phone on 9.6% (independently reproduced on a 379-ad sample
at 11.1 / 9.0 / 7.7%). So excluding the structured field is a PARTIAL control.

`skills[].phrase` is free-form model text quoting the body, which makes it an
uncontrolled egress path. It is not currently leaking — 0 of 169 real facet rows
match an email or phone pattern — so this is a latent channel being closed
before the 9,823-cluster census, not an incident.

GROUNDED IN REAL FORMATS, sampled from 2,810 real contact phone values and
2,575 emails:
    '+4797940668'   '48157761'   '950 27 028'   '+47 55 58 85 95'   '47632395'
    'renate.bjergene@eidsvoll.kommune.no'   'renhold@trondervask.no'

AND THE FALSE-POSITIVE CLASS THAT MATTERS: a naive 8-digit rule matches
NORWEGIAN DATES. In 2,000 bodies a first-draft pattern returned 639 hits of
which the len-8 bucket was dominated by application deadlines — '28.10.2026',
'30.01.2027', '16.10.2026'. Redacting those would corrupt the deadline text
every ad carries, so date exclusion is load-bearing, not polish.
"""
import pytest

from finn_smart_search import pii

pytestmark = pytest.mark.unit

# Verbatim from ads_raw.contactList — the shapes that actually occur.
REAL_PHONES = ["+4797940668", "48157761", "950 27 028", "+47 55 58 85 95",
               "47632395", "75 54 22 20", "+47 46912415", "78 97 76 00"]
REAL_EMAILS = ["paul@utelivpartner.no", "renate.bjergene@eidsvoll.kommune.no",
               "renhold@trondervask.no", "line.saglien@horten.kommune.no"]
# Norwegian application deadlines, measured as the dominant naive-match class.
REAL_DATES = ["28.10.2026", "30.01.2027", "16.10.2026", "01.12.2026"]


@pytest.mark.parametrize("phone", REAL_PHONES)
def test_real_contact_phones_are_detected(phone):
    assert pii.contains_pii(f"Ring oss på {phone} for mer informasjon"), phone


@pytest.mark.parametrize("date", REAL_DATES)
def test_norwegian_dates_are_NOT_treated_as_phone_numbers(date):
    """The load-bearing exclusion. Every ad carries a deadline; treating those as
    phone numbers would redact the one date the seeker needs."""
    text = f"Søknadsfrist {date} og vi behandler søknader fortløpende"
    assert not pii.contains_pii(text), f"{date} was read as a phone number"
    assert pii.scrub(text) == text, "a date must survive scrubbing untouched"


@pytest.mark.parametrize("email", REAL_EMAILS)
def test_real_contact_emails_are_detected(email):
    assert pii.contains_pii(f"Send søknad til {email} snarest")


def test_scrub_replaces_phone_and_email_but_keeps_the_rest():
    text = ("Gode norskkunnskaper er et krav. Kontakt Renate Bjergene på "
            "renate.bjergene@eidsvoll.kommune.no eller 950 27 028.")
    out = pii.scrub(text)
    assert "renate.bjergene@eidsvoll.kommune.no" not in out
    assert "950 27 028" not in out
    assert "Gode norskkunnskaper er et krav." in out, "non-PII text must survive"
    assert pii.EMAIL_REDACTION in out and pii.PHONE_REDACTION in out


def test_a_contact_NAME_is_only_removable_with_that_ads_contact_list():
    """Names cannot be detected by pattern — 'Renate Bjergene' is
    indistinguishable from any Norwegian noun phrase. So name redaction is
    AD-SCOPED: it uses that ad's own contactList values as the targets, which is
    both precise and available (94.2% of ads have a contactList).

    Without the list, a name must pass through rather than be guessed at."""
    text = "Kontakt Renate Bjergene for spørsmål om stillingen"
    assert "Renate Bjergene" in pii.scrub(text)
    out = pii.scrub(text, contacts=[{"name": "Renate Bjergene"}])
    assert "Renate Bjergene" not in out
    assert pii.NAME_REDACTION in out


def test_a_short_contact_name_is_not_used_as_a_redaction_target():
    """A one- or two-character name would redact fragments of ordinary words.
    Guard the target, not just the output."""
    text = "Vi søker en person som kan norsk og engelsk godt"
    assert pii.scrub(text, contacts=[{"name": "Ka"}]) == text


def test_scrub_is_idempotent():
    text = "Ring 48157761 eller mail paul@utelivpartner.no"
    once = pii.scrub(text)
    assert pii.scrub(once) == once


def test_scrub_handles_none_and_empty():
    assert pii.scrub(None) == ""
    assert pii.scrub("") == ""
    assert pii.contains_pii(None) is False


def test_facet_skills_are_scrubbed_by_the_guard():
    """The actual egress path. A model-generated skill phrase that quoted a
    contact line must not reach the index."""
    facets = {"skills": [{"phrase": "Kontakt oss på 950 27 028", "level": "required"},
                         {"phrase": "Førerkort klasse B", "level": "required"}]}
    out = pii.scrub_facets(facets, contacts=[{"name": "Renate Bjergene"}])
    assert pii.PHONE_REDACTION in out["skills"][0]["phrase"]
    assert out["skills"][1]["phrase"] == "Førerkort klasse B", "clean phrases untouched"


def test_scrub_facets_leaves_evidence_spans_ALONE():
    """DELIBERATE. `evidence_spans` must stay byte-identical to the ad or the
    verbatim check in census_validate rejects them, and that check is the defence
    against fabricated evidence. Scrubbing a span would trade a fabrication
    guarantee for a redaction — so spans are handled at the EXPORT boundary
    instead, and this test pins that the guard does not touch them."""
    facets = {"evidence_spans": [{"span": "Ring 48157761 for spørsmål",
                                  "section_language": "no"}],
              "skills": []}
    out = pii.scrub_facets(facets)
    assert out["evidence_spans"][0]["span"] == "Ring 48157761 for spørsmål"
