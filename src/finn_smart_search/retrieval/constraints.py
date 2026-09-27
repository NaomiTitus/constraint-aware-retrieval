"""Constraints are predicates over metadata, not directions in embedding space.

THE OPERATION IS CONTAINMENT. A job requires a set R; the seeker has a set S; the
job is viable iff R ⊆ S. Containment is asymmetric — needing Norwegian you lack is
fatal, having Norwegian the job never asked for is free — and cosine similarity is
symmetric, so no encoder expresses it. That, not negation, is why this module
exists: the seeker's phrasing is irrelevant, and `NorwegianLevel` below types what
they HAVE rather than what they deny. See README and LIMITATIONS §14.

This module is the project's thesis in code, and its most important property
is a NON-effect: a constraint the seeker did not state must change nothing.

WHY THAT IS THE FIRST THING HERE, with the number that makes it urgent. The
census of all 10,166 ads:

    accessible to an English speaker      956   9.4%
    blocked by a STATED requirement     5,703  56.1%
    blocked by SILENCE alone            3,507  34.5%

`understanding.derive_english_accessible` calls an `unstated` level on a
Norwegian-written ad inaccessible. That is a reasonable default for the EVAL,
which needs a binary to compute hidden/shown-wrongly. Inherited here it would
be a catastrophe: a seeker who never mentioned language would be shown 9.4%
of the corpus, the product would be ruined for the ~90% who speak Norwegian,
and the language metric would look excellent the whole time — because until
the four control personas were added, every persona in the evaluation stated
a language constraint.

So this module keeps two things apart that the eval deliberately conflates:

    a BOOLEAN accessibility fact, for measuring the extractor
    a GRADED severity, applied only on request, for ranking

Severity is multiplied by lambda, which the plan exposes as a product knob and
the demo as a slider. lambda=0 switches the stage off; the `hard` mode drops
above a threshold instead of penalising. The recall-vs-CVR curve as lambda
sweeps 0 to 1 is the second chart in the report, and it is what makes this a
product decision with a dial rather than a binary fix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

# What the seeker can say about Norwegian. `None` — the attribute absent — is
# not one of these: it means they said nothing, which is the whole point.
NorwegianLevel = Literal["none", "basic", "conversational", "fluent", "native"]

# A level that cannot block anyone.
NO_BARRIER = frozenset({"explicitly_not_required", "either_norwegian_or_english"})
# A level that demands Norwegian outright.
BLOCKING = frozenset({"certified", "fluent", "professional", "conversational"})
# Nordic but not English — blocks a non-Scandinavian speaker.
NORDIC_ONLY = frozenset({"scandinavian_accepted"})

# Graded severities for a seeker with NO Norwegian. Taken from the plan's §6
# design, which chose a graded penalty over a filter because hard-filtering
# silence removes a third of the corpus.
SEVERITY = {
    "certified": 1.0,
    "professional": 1.0,
    "fluent": 1.0,
    "conversational": 0.8,
    "scandinavian_accepted": 0.9,   # Danish or Swedish would do; English will not
    "desirable": 0.4,               # asked for, not required
    "either_norwegian_or_english": 0.0,
    "explicitly_not_required": 0.0,
}
# Silence. NOT 1.0 — see the module docstring. The ad has not said it needs
# Norwegian; the language it is written in is weak evidence, nothing more.
SEVERITY_SILENT_NORWEGIAN_AD = 0.5
SEVERITY_SILENT_OTHER = 0.1

# How much a seeker's own Norwegian discounts the penalty.
SPEAKER_DISCOUNT = {"none": 1.0, "basic": 0.8, "conversational": 0.6,
                    "fluent": 0.0, "native": 0.0}


@dataclass(frozen=True)
class LanguageConstraint:
    """Present only when the seeker actually said something about language."""
    norwegian: NorwegianLevel


@dataclass(frozen=True)
class SeekerProfile:
    raw_query: str
    # None means NOT STATED, and is distinct from every stated value including
    # "fluent". A parser that defaults this to anything has broken the rule
    # this module exists to enforce.
    language_constraint: LanguageConstraint | None = None
    other: Mapping[str, object] = field(default_factory=dict)


def language_severity(facets: Mapping, doc_lang: str,
                      profile: SeekerProfile) -> float:
    """How badly this ad violates the seeker's STATED language constraint.

    Returns 0.0 when no constraint was stated — exactly zero, not merely
    small, because 3,507 ads hang on the difference.
    """
    c = profile.language_constraint
    if c is None:
        return 0.0                      # THE RULE. Nothing else may precede it.

    discount = SPEAKER_DISCOUNT.get(c.norwegian, 0.0)
    if discount == 0.0:
        return 0.0                      # they speak it; nothing can block them

    level = facets.get("norwegian_requirement_level")
    if level in SEVERITY:
        base = SEVERITY[level]
    else:
        # `unstated`, or a level this module has not been taught. Fall back to
        # the language the ad is WRITTEN in, weakly.
        working = facets.get("stated_working_language", "unstated")
        if working in ("english", "both"):
            base = 0.0
        elif working in ("norwegian", "scandinavian"):
            base = SEVERITY_SILENT_NORWEGIAN_AD
        else:
            base = (SEVERITY_SILENT_NORWEGIAN_AD if doc_lang == "no"
                    else SEVERITY_SILENT_OTHER)

    # An ad that states English as its working language cannot block, whatever
    # its level says — the two are different facts and the plan's fact E says
    # all four combinations occur.
    if facets.get("stated_working_language") == "english" and level not in BLOCKING:
        base = 0.0

    return base * discount


def apply(scores: Sequence[float], ads: Sequence[tuple[Mapping, str]],
          profile: SeekerProfile, lam: float = 0.7) -> list[float]:
    """Soft mode: score *= 1 - lambda * severity.

    With no stated constraint this returns the input unchanged, element for
    element — the ranking is bit-identical to running without the stage, which
    is what tests/unit/test_constraints_only_when_stated.py asserts.
    """
    if profile.language_constraint is None or lam == 0.0:
        return list(scores)
    return [s * (1.0 - lam * language_severity(f, dl, profile))
            for s, (f, dl) in zip(scores, ads)]
