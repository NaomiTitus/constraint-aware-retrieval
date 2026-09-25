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

THE FALSE POSITIVE THAT MATTERS. A naive 8-digit rule matches NORWEGIAN DATES.
Over 2,000 bodies a first draft returned 639 hits whose len-8 bucket was
dominated by application deadlines: 28.10.2026, 30.01.2027, 16.10.2026. Every ad
carries a deadline, so excluding dd.mm.yyyy is load-bearing.
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
_DATE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
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


def _mask_dates(text: str) -> tuple[str, list[str]]:
    """Park dates behind a placeholder so the phone pattern cannot see them."""
    found: list[str] = []

    def take(m: re.Match) -> str:
        found.append(m.group(0))
        return f"{len(found) - 1}"

    return _DATE.sub(take, text), found


def _restore(text: str, dates: list[str]) -> str:
    for i, d in enumerate(dates):
        text = text.replace(f"{i}", d)
    return text


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
    masked, _ = _mask_dates(text)
    if EMAIL.search(masked) or PHONE.search(masked):
        return True
    return any(n in text for n in _name_targets(contacts))


def scrub(text: str | None,
          contacts: Iterable[Mapping[str, Any]] | None = None) -> str:
    """Return text with contact PII replaced by a visible marker.

    Idempotent: the markers contain no PII pattern, so re-scrubbing is a no-op.
    """
    if not text:
        return ""
    masked, dates = _mask_dates(text)
    masked = EMAIL.sub(EMAIL_REDACTION, masked)
    masked = PHONE.sub(PHONE_REDACTION, masked)
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
