"""Score the census against the frozen golden set.

Three design decisions, each with a reason, and each defended by a test:

  PER-LEVEL, NEVER POOLED ALONE. The golden set's level mix is a stratified
  SAMPLING ARTIFACT: `desirable` is 4 of 44 here (9%) for a level covering ~4
  of 10,166 in the corpus (0.04%) — over-represented ~200x. A single pooled
  number would be dominated by sampling design.

  THE HEADLINE IS THE ACCESSIBILITY FLIP, not level accuracy. Confusing
  `professional` with `fluent` changes nothing; both hide the ad. Confusing
  `either_norwegian_or_english` with `professional` flips an ad from SHOWN to
  HIDDEN. Level accuracy treats those as equally wrong. They are not.

  FLIPS ARE SPLIT BY DIRECTION, and rates carry their denominator, counts and
  a Wilson interval. Wrongly hiding is invisible to everyone; wrongly showing
  costs a click. And at n=13 accessible ads, 1/13 has a 95% interval of roughly
  [0.01, 0.33] — a bare percentage invites false precision.

Two traps this harness is built to avoid:

  REFUSAL MUST NOT BE REWARDED. Unscored ads are excluded from per-level
  counts, so an extractor that skips the hard ads would improve every rate.
  `coverage` therefore travels with the numbers, and flip denominators are
  GOLDEN counts, not scored counts.

  THE TAXONOMY GAP IS REPORTED, NOT SCORED. Golden #15 has a correct level
  (`explicitly_not_required`) that a human judged NOT accessible, because the
  ad addresses Polish speakers and never mentions English. Scoring against that
  annotation would make a correct answer unwinnable and make mislabelling the
  cheapest fix. It is surfaced separately as evidence the nine-level enum
  cannot express "is English mentioned".
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..understanding.census_prompt import (ACCESSIBLE_LEVELS, BLOCKING_LEVELS, TOOL,
                                           derive_english_accessible)
from ..understanding.text_norm import normalise

VALID_LEVELS = set(TOOL["input_schema"]["properties"]["norwegian_requirement_level"]["enum"])

POOLED_CAVEAT = (
    "Pooled accuracy is dominated by the golden set's level mix, which is a "
    "stratified sampling artifact: `desirable` is over-represented ~200x "
    "relative to the corpus. Read per_level instead."
)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval. At the sizes here (n=13 accessible ads) a bare
    point estimate conveys far more precision than the data supports."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _same_span(a: str | None, b: str | None) -> bool:
    """Same normaliser as the validator, plus case folding: `exact` is the
    number people cite as extraction quality, and a trailing newline or a
    capital must not halve it."""
    return normalise(a).lower() == normalise(b).lower()


def _accessible(level: str, working: str, doc_lang: str) -> bool:
    return derive_english_accessible(
        {"norwegian_requirement_level": level, "stated_working_language": working},
        doc_lang)


# Professions named in authorisation clauses. A bare subset rule would score
# "Norsk autorisasjon som lege" correct against "...som sykepleier": both
# contain "autorisasjon", and one is not a prefix of the other.
_PROFESSIONS = re.compile(
    r"sykepleier|vernepleier|lege|tannlege|helsefagarbeider|fysioterapeut|"
    r"ergoterapeut|jordmor|psykolog|farmas[øo]yt|radiograf|bioingeni[øo]r")


def _auth_matches(expected: str | None, got: str | None) -> bool:
    """Presence-based on the authorisation itself, but a NAMED profession must
    not contradict. "norsk autorisasjon" matches "...som sykepleier" (the same
    finding, differently quoted); "...som lege" does not."""
    if not expected:
        return not got
    if not got:
        return False
    e, g = normalise(expected).lower(), normalise(got).lower()
    if "autorisasjon" not in g:
        return False
    ep, gp = set(_PROFESSIONS.findall(e)), set(_PROFESSIONS.findall(g))
    return not (ep and gp and ep != gp)


def score(predictions: dict, golden: list) -> dict:
    by_uuid = {g["uuid"]: g for g in golden}
    scored = [g for g in golden if g["uuid"] in predictions]
    unscored = [g["uuid"] for g in golden if g["uuid"] not in predictions]

    per_level: dict[str, dict] = {}
    confusion: Counter = Counter()
    invalid_levels = 0
    hidden_wrongly = shown_wrongly = 0
    n_accessible = 0
    taxonomy_gap: list[str] = []

    ev = Counter()
    n_span_expected = 0
    auth_expected = auth_correct = auth_spurious = 0
    wl_non_default_total = wl_non_default_hit = 0

    # Denominators over the WHOLE golden set, so refusing hard ads cannot
    # dilute a harm rate.
    for g in golden:
        exp = g["expected"]
        if _accessible(exp["norwegian_requirement_level"],
                       exp["stated_working_language"], g.get("doc_lang", "no")):
            n_accessible += 1
        if "annotated_accessible" in g:
            derived = _accessible(exp["norwegian_requirement_level"],
                                  exp["stated_working_language"], g.get("doc_lang", "no"))
            if derived != g["annotated_accessible"]:
                taxonomy_gap.append(g["uuid"])

    for g in scored:
        uuid, exp = g["uuid"], g["expected"]
        p = predictions[uuid]
        want, got = exp["norwegian_requirement_level"], p.get("norwegian_requirement_level")
        doc_lang = g.get("doc_lang", "no")

        if got not in VALID_LEVELS:
            invalid_levels += 1

        slot = per_level.setdefault(want, {"n": 0, "correct": 0, "accuracy": 0.0})
        slot["n"] += 1
        slot["correct"] += int(got == want)
        confusion[(want, got)] += 1

        want_acc = _accessible(want, exp["stated_working_language"], doc_lang)
        got_acc = (_accessible(got, p.get("stated_working_language", "unstated"), doc_lang)
                   if got in VALID_LEVELS else False)
        if want_acc and not got_acc:
            hidden_wrongly += 1
        elif got_acc and not want_acc:
            shown_wrongly += 1

        # ── evidence ──
        want_span = exp["evidence_span"]
        got_spans = [s["span"] for s in p.get("evidence_spans") or []]
        source = normalise(g.get("source_text") or "")
        if want_span:
            n_span_expected += 1
            if any(_same_span(s, want_span) for s in got_spans):
                ev["exact"] += 1
            elif not got_spans:
                ev["missing"] += 1
            elif source and not any(normalise(s).lower() in source.lower() for s in got_spans):
                ev["fabricated"] += 1
            else:
                ev["different"] += 1
        elif got_spans:
            ev["spurious"] += 1

        # ── authorisation: denominator is ads that SHOULD have one ──
        if exp["authorisation_required"]:
            auth_expected += 1
            if _auth_matches(exp["authorisation_required"], p.get("authorisation_required")):
                auth_correct += 1
        elif p.get("authorisation_required"):
            auth_spurious += 1

        # ── working language: scored where it MATTERS, not as plain accuracy ──
        if exp["stated_working_language"] != "unstated":
            wl_non_default_total += 1
            wl_non_default_hit += int(
                p.get("stated_working_language") == exp["stated_working_language"])

    for slot in per_level.values():
        slot["accuracy"] = slot["correct"] / slot["n"] if slot["n"] else 0.0

    n_scored = len(scored)
    correct = sum(s["correct"] for s in per_level.values())
    return {
        "n_golden": len(golden), "n_scored": n_scored, "unscored": unscored,
        "extra_predictions": len(set(predictions) - set(by_uuid)),
        "coverage": n_scored / len(golden) if golden else None,
        "pooled_accuracy": (correct / n_scored) if n_scored else None,
        "pooled_accuracy_caveat": POOLED_CAVEAT,
        "per_level": per_level, "confusion": dict(confusion),
        "invalid_levels": invalid_levels,
        "taxonomy_gap": taxonomy_gap,
        "accessibility": {
            "hidden_wrongly": hidden_wrongly, "shown_wrongly": shown_wrongly,
            "n_accessible": n_accessible, "n_blocking": len(golden) - n_accessible,
            "hidden_wrongly_rate": (hidden_wrongly / n_accessible) if n_accessible else None,
            "shown_wrongly_rate": ((shown_wrongly / (len(golden) - n_accessible))
                                   if len(golden) - n_accessible else None),
            "hidden_wrongly_ci95": _wilson(hidden_wrongly, n_accessible),
            "shown_wrongly_ci95": _wilson(shown_wrongly, len(golden) - n_accessible),
        },
        "evidence": {"n_expected": n_span_expected, "exact": ev["exact"],
                     "different": ev["different"], "missing": ev["missing"],
                     "spurious": ev["spurious"], "fabricated": ev["fabricated"]},
        "working_language": {
            "non_default_total": wl_non_default_total,
            "non_default_correct": wl_non_default_hit,
            "non_default_recall": ((wl_non_default_hit / wl_non_default_total)
                                   if wl_non_default_total else None),
            "correct": wl_non_default_hit + sum(
                1 for g in scored
                if g["expected"]["stated_working_language"] == "unstated"
                and predictions[g["uuid"]].get("stated_working_language") == "unstated"),
        },
        "authorisation": {"n_expected": auth_expected, "correct": auth_correct,
                          "spurious": auth_spurious},
    }
