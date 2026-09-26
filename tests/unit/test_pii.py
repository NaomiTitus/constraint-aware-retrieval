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
# Numeric runs that must NOT be read as phone numbers.
# Dates: these do NOT match the final pattern anyway (its grouped alternatives
# need four 2-digit runs; a date has three), so they are a weak guard — kept, but
# not the reason the masking exists.
REAL_DATES = ["28.10.2026", "30.01.2027", "16.10.2026", "01.12.2026"]
# WORKING-HOURS RANGES are the real case: `08.00-16.00` IS four 2-digit groups
# with separators and matches exactly. Measured over 4,000 real bodies, masking
# suppresses 92 such matches and every one is a shift time. Redacting them would
# delete the hours from every shift ad — 14.8% of the corpus.
REAL_HOURS = ["08.00-16.00", "07.30-16.00", "06.00-16.00", "08.00-12.00"]


@pytest.mark.parametrize("phone", REAL_PHONES)
def test_real_contact_phones_are_detected(phone):
    assert pii.contains_pii(f"Ring oss på {phone} for mer informasjon"), phone


@pytest.mark.parametrize("hours", REAL_HOURS)
def test_working_hour_ranges_are_NOT_treated_as_phone_numbers(hours):
    """THE load-bearing exclusion, and it is not the one I first wrote a test for.

    `08.00-16.00` is four 2-digit groups with separators — the phone pattern
    matches it exactly. Removing the masking survived every other test in this
    file, so this assertion is what makes the guard real."""
    text = f"Arbeidstid er {hours} mandag til fredag"
    assert not pii.contains_pii(text), f"{hours} was read as a phone number"
    assert pii.scrub(text) == text, "working hours must survive scrubbing"


@pytest.mark.parametrize("date", REAL_DATES)
def test_norwegian_dates_are_NOT_treated_as_phone_numbers(date):
    """A weaker guard than it looks: verified that none of these match the final
    pattern even with masking removed, because its grouped alternatives need four
    2-digit runs and a date has three. Kept as a boundary record, not as the
    justification for the masking."""
    text = f"Søknadsfrist {date} og vi behandler søknader fortløpende"
    assert not pii.contains_pii(text), f"{date} was read as a phone number"
    assert pii.scrub(text) == text


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

    The fixture must contain the short name AS A SUBSTRING or the test passes
    whatever the floor is — my first version used "Ka" against text containing
    only lowercase "kan", so lowering MIN_NAME_CHARS to 1 survived it."""
    text = "Kan du norsk og engelsk, og har du fagbrev?"
    assert "Ka" in text, "fixture must actually contain the short name"
    assert pii.scrub(text, contacts=[{"name": "Ka"}]) == text


def test_a_full_name_is_redacted_before_any_part_of_it():
    """Targets are applied longest-first. Shortest-first leaves an orphan: with
    contacts [Renate, Renate Bjergene], redacting "Renate" first turns the text
    into "[navn fjernet] Bjergene" and the surname survives. Two entries are
    needed to detect it — one name cannot."""
    text = "Kontakt Renate Bjergene for spørsmål om stillingen"
    out = pii.scrub(text, contacts=[{"name": "Renate"},
                                    {"name": "Renate Bjergene"}])
    assert "Bjergene" not in out, f"surname survived: {out!r}"
    assert out.count(pii.NAME_REDACTION) == 1


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


# ---------------------------------------------------------------------------
# GROUPING BLINDNESS — found by measuring the corpus, not by reading the code.
#
# The pattern above ENUMERATES groupings (2-2-2-2, 3-2-3, bare-8). Ads do not
# restrict themselves to those. Measured over all 10,166 ads by matching each
# ad's own `contactList` phone digits against the scrubbed `description_text`:
# 18 contact phones survived in 15 ads, every one a grouping the enumeration
# does not list.
#
# The fixture that justified the old pattern was sampled from `contactList`
# VALUES. The leak is in how the BODY writes the number, which is a different
# artefact — STANDARDS.md §3.0 applied one layer up from where it was applied.
#
# Each pair below is verbatim: the contactList value, and the rendering of that
# same number in that ad's description_text.
LEAKED_IN_BODY = [
    ("+47 99518681",  "995 18681"),    # 3-5
    ("+4798244776",   "982 447 76"),   # 3-3-2
    ("45205365",      "4520 5365"),    # 4-4
    ("97086084",      "97 086 084"),   # 2-3-3
    ("905 15 152",    "90515 152"),    # 5-3
    ("+47 41529304",  "415 29304"),    # 3-5
    ("+4795927567",   "959 27567"),    # 3-5
    ("94194115",      "941 941 15"),   # 3-3-2
    ("+4790503566",   "905 035 66"),   # 3-3-2
    ("+ 47 776 26614", "+ 47 776 26614"),  # spaced country code, 3-5
]


@pytest.mark.parametrize("listed,in_body", LEAKED_IN_BODY)
def test_contact_phone_is_redacted_however_the_body_groups_it(listed, in_body):
    """A number known from contactList must not survive in the body text."""
    contacts = [{"name": "Test Kontakt", "phone": listed}]
    text = f"Spørsmål om stillingen rettes til Test Kontakt, {in_body}."
    out = pii.scrub(text, contacts)
    assert in_body not in out, f"{in_body!r} survived (contactList {listed!r})"
    assert pii.PHONE_REDACTION in out


# Numbers NOT in contactList are still phones and still contact details. These
# are verbatim from the corpus; each has a cue word or a +47 prefix.
CUED_PHONES = [
    "For eventuell veiledning, kontakt servicekontoret tlf: 78 97 76 00.",
    "ta kontakt på tlf. +47 413 52 617 for en uforpliktende prat",
    "kan du kontakte Adecco på telefon 23 29 00 00.",
    "eller på TLF: 915 60 156.",
    "tlf. 928 02 537",
    "Tlf.: 95361613",
]


@pytest.mark.parametrize("line", CUED_PHONES)
def test_cued_phone_numbers_are_redacted_in_any_grouping(line):
    out = pii.scrub(line, [])
    assert pii.PHONE_REDACTION in out, line
    assert not __import__("re").search(r"\d[\s.-]?\d[\s.-]?\d[\s.-]?\d[\s.-]?"
                                       r"\d[\s.-]?\d[\s.-]?\d[\s.-]?\d", out), out


# The precision side. These carry 8 digits with separators and are NOT phones.
# Verbatim from the corpus; a general 8-digit rule matches all of them, which is
# why the rule is anchored on a cue or a +47 prefix rather than left free.
NOT_PHONES_IN_CONTEXT = [
    "Kveldsvakt: Kl 1430-2000",
    "Workplace Ålesundsvegen 1136 6240 Ørskog Norway",
    "Dagskift: kl 07.30-16.00",
    "Perioden: 28.10.2026 - 30.01.2027",
    "Stillingen er ledig i perioden 2026-2028",
]


@pytest.mark.parametrize("line", NOT_PHONES_IN_CONTEXT)
def test_eight_digit_runs_without_a_phone_cue_are_left_alone(line):
    assert pii.PHONE_REDACTION not in pii.scrub(line, []), line
