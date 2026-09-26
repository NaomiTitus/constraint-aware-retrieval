"""Redact contact PII from text that leaves the corpus.

WHY THIS EXISTS. `contactList` is excluded from every export, but the same
details sit in the ad BODY. Measured over the corpus: the contact person's full
name also appears in `description_text` on 11.8% of ads, their email on 10.1%,
their phone on 9.6%. So the structured exclusion is a PARTIAL control, and any
facet carrying free-form body text is an egress path around it.

`skills[].phrase` is exactly that. It is not currently leaking — 0 of 169 real
facet rows match an email or phone pattern — so this closes a latent channel
before the 9,823-cluster census rather than cleaning up after one.

WHAT IS AND IS NOT PATTERN-DETECTABLE
  phone, email   pattern-detectable, and done here
  NAME           NOT detectable by pattern: "Renate Bjergene" is
                 indistinguishable from any Norwegian noun phrase. Name removal
                 is therefore AD-SCOPED — it uses that ad's own contactList
                 values as the redaction targets. 94.2% of ads carry a
                 contactList, so the targets are available. With no list, a name
                 passes through rather than being guessed at; a guessing name
                 redactor would eat ordinary words.

THE FALSE POSITIVE THAT MATTERS — and it is NOT what I first thought.

A naive 8-digit rule matches Norwegian dates, which is what a first draft did.
The FINAL pattern does not: 28.10.2026, 1.12.2026, 2026-10-28 and 28-10-2026 all
fail it, because its grouped alternatives need four 2-digit runs and a date has
three. Verified for six date forms.

What the masking actually earns its place against is WORKING-HOURS RANGES.
`08.00-16.00` is four 2-digit groups with separators and matches the phone
pattern exactly. Measured over 4,000 real bodies: masking suppresses 92 matches,
every one of them a shift time — 08.00-16.00, 07.30-16.00, 06.00-16.00,
08.00-12.00. Redacting those would delete the working hours from every shift
advertisement, which is 14.8% of the corpus.

So the guard is real and the reason is different from the one first recorded.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

EMAIL_REDACTION = "[e-post fjernet]"
PHONE_REDACTION = "[telefon fjernet]"
NAME_REDACTION = "[navn fjernet]"

# Shortest contact name used as a redaction target. A one- or two-character
# target would redact fragments of ordinary words.
MIN_NAME_CHARS = 4

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")

# Norwegian numbers are 8 digits, optionally +47, optionally grouped 2-2-2-2 or
# 3-2-3. Formats taken from 2,810 real contactList values:
#   +4797940668 · 48157761 · 950 27 028 · +47 55 58 85 95 · 78 97 76 00
# Dotted numeric runs that are NOT phone numbers: dates (28.10.2026) and, the
# case that actually bites, working-hours ranges (08.00-16.00). Parked behind a
# placeholder before the phone pattern runs. Named for what it matches, not for
# what I assumed it matched.
_NOT_A_PHONE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
# Boundaries are DIGIT-only, not \w or "." — a trailing "." is usually the end of
# the sentence ("...på 950 27 028."), and blocking on it silently skipped every
# phone that ends a sentence. Dates are masked before this runs, so "." needs no
# special treatment here.
PHONE = re.compile(
    r"(?<![\w])(?:\+\s?47[\s.-]?)?"             # optional country code
    r"(?:\d{3}[\s.-]\d{2}[\s.-]\d{3}"          # 950 27 028
    r"|\d{2}[\s.-]\d{2}[\s.-]\d{2}[\s.-]\d{2}"  # 55 58 85 95
    r"|\d{8})"                                  # 48157761
    r"(?!\d)"
)


def _mask_non_phones(text: str) -> tuple[str, list[str]]:
    """Park dates and working-hours ranges so the phone pattern cannot see them."""
    found: list[str] = []

    def take(m: re.Match) -> str:
        found.append(m.group(0))
        return f"{len(found) - 1}"

    return _NOT_A_PHONE.sub(take, text), found


def _restore(text: str, dates: list[str]) -> str:
    for i, d in enumerate(dates):
        text = text.replace(f"{i}", d)
    return text


# AD-SCOPED PHONE TARGETS — the general rule the enumeration above is not.
#
# The pattern above enumerates groupings. Real bodies use others: measured over
# all 10,166 ads, 18 contactList phones survived in 15 ads as 3-5, 3-3-2, 4-4,
# 2-3-3 and 5-3 groupings. Enumerating harder is the same mistake again, and a
# CONTEXT-FREE general 8-digit rule is not available: measured on the corpus it
# also matches 1,100 non-phones — shift times (`Kl 1430-2000`), postcodes
# (`1136 6240 Ørskog`) and year ranges.
#
# So the general rule is scoped to what this ad's contactList already declares.
# The digits are known to be a phone number, so ANY rendering of them is one and
# no context test is needed. 94.2% of ads carry a contactList; where one is
# absent the enumerated pattern is still the floor.
def _phone_targets(contacts: Iterable[Mapping[str, Any]] | None) -> list[re.Pattern]:
    out = []
    for c in contacts or []:
        digits = re.sub(r"\D", "", (c.get("phone") or ""))
        # A contactList value can hold several numbers ("+47 92099272 +47 78942585").
        for nat in {digits[i:i + 8] for i in range(0, max(len(digits) - 7, 1), 8)} | {digits[-8:]}:
            if len(nat) != 8:
                continue
            # the same 8 digits, however the body separates them, with an
            # optional country code in front
            body = r"[\s.\u00a0-]{0,2}".join(nat)
            out.append(re.compile(r"(?<![\d\w])(?:(?:\+|00)[\s.\u00a0-]{0,2}47[\s.\u00a0-]{0,2})?"
                                  + body + r"(?!\d)"))
    return out


def _name_targets(contacts: Iterable[Mapping[str, Any]] | None) -> list[str]:
    out = []
    for c in contacts or []:
        n = (c.get("name") or "").strip()
        if len(n) >= MIN_NAME_CHARS:
            out.append(n)
    # longest first, so a full name is removed before any part of it
    return sorted(out, key=len, reverse=True)


def contains_pii(text: str | None,
                 contacts: Iterable[Mapping[str, Any]] | None = None) -> bool:
    """True if text carries an email, a Norwegian phone number, or a name from
    this ad's contact list."""
    if not text:
        return False
    masked, _ = _mask_non_phones(text)
    if EMAIL.search(masked) or PHONE.search(masked):
        return True
    if any(rx.search(masked) for rx in _phone_targets(contacts)):
        return True
    return any(n in text for n in _name_targets(contacts))


def scrub(text: str | None,
          contacts: Iterable[Mapping[str, Any]] | None = None) -> str:
    """Return text with contact PII replaced by a visible marker.

    Idempotent: the markers contain no PII pattern, so re-scrubbing is a no-op.
    """
    if not text:
        return ""
    masked, dates = _mask_non_phones(text)
    masked = EMAIL.sub(EMAIL_REDACTION, masked)
    masked = PHONE.sub(PHONE_REDACTION, masked)
    for rx in _phone_targets(contacts):
        masked = rx.sub(PHONE_REDACTION, masked)
    out = _restore(masked, dates)
    for n in _name_targets(contacts):
        out = out.replace(n, NAME_REDACTION)
    return out


def scrub_facets(facets: Mapping[str, Any],
                 contacts: Iterable[Mapping[str, Any]] | None = None) -> dict:
    """Scrub the free-form fields of a facet record.

    `evidence_spans` are LEFT ALONE, deliberately. They must stay byte-identical
    to the ad or the verbatim check in census_validate rejects them — and that
    check is the defence against fabricated evidence. Trading a fabrication
    guarantee for a redaction would be the wrong exchange, so spans are handled
    at the export boundary instead.
    """
    out = dict(facets)
    skills = out.get("skills")
    if isinstance(skills, list):
        out["skills"] = [
            dict(s, phrase=scrub(s.get("phrase"), contacts))
            if isinstance(s, Mapping) else s
            for s in skills
        ]
    for field in ("authorisation_required",):
        if isinstance(out.get(field), str):
            out[field] = scrub(out[field], contacts)
    return out
