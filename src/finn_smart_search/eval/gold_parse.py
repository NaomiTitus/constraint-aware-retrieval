"""Load and validate the hand-authored gold parses — the seeker side, set S.

See `eval/GOLD_PARSE_SCHEMA.md` for the schema and the reasoning. This module is
the enforcement: the two rules that make these parses an oracle rather than a wish
list are mechanical checks here, not conventions in a document.

  1. EVERY CONSTRAINT CITES VERBATIM EVIDENCE from the persona's frozen query.
     Hand-authoring S invites writing the profile the CORPUS would like — ESCO
     labels, census enum values, the vocabulary the ads use — which would delete
     the vocabulary mismatch LIMITATIONS §14 measured and flatter retrieval by
     construction. If the seeker did not say it, it is not in S.

  2. ABSENCE MEANS NOT STATED, never a default. A facet with no entry is a
     distinct state from every stated value. This is
     `constraints.SeekerProfile.language_constraint = None`, which returns
     severity EXACTLY 0.0 — 3,507 ads hang on the difference.

`to_seeker_profile` is the bridge into `retrieval.constraints`, and it is
deliberately the only way a parse becomes a profile: it is where rule 2 is
converted from "no row in a file" into `language_constraint=None`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[3]
PERSONAS_PATH = ROOT / "eval" / "personas.yaml"
PARSES_PATH = ROOT / "eval" / "gold_parses.yaml"

PRIORITIES = frozenset({"hard", "soft"})

# Closed vocabulary. Extending it is an edit here plus a test update, on purpose:
# a typo'd facet name silently carried would be a constraint that never matches
# anything and never reports that it didn't.
FACETS: Mapping[str, str | None] = {
    "occupation": "occupation",
    "skill": "skills",
    "language.norwegian": "norwegian_requirement_level",
    "language.english": "stated_working_language",
    "language.other": None,
    "credential.authorisation": "authorisation_required",
    "credential.licence": None,
    "credential.trade_certificate": None,
    "location.place": None,
    "location.anywhere": None,
    "work.remote": None,
    "contract.permanence": None,
    "contract.extent": None,
    "contract.shift": None,
    "experience.years": "min_years_experience",
    "seniority": "seniority",
}

# The capability scale, identical to `constraints.NorwegianLevel`. A seeker states
# what they HAVE; there is no negation anywhere in the type.
LANGUAGE_LEVELS = frozenset({"none", "basic", "conversational", "fluent", "native"})


def _norm(s: str) -> str:
    """Whitespace-normalise for comparison. Persona queries are YAML folded
    scalars, so they carry folded newlines and a trailing one; comparing raw
    would fail on formatting rather than on content."""
    return " ".join(str(s).split())


@dataclass(frozen=True)
class Constraint:
    facet: str
    value: object
    priority: str
    evidence: str
    ad_side: str | None

    @property
    def scoreable(self) -> bool:
        """Does the corpus record anything this can be compared against? A
        `False` here is a constraint the seeker stated and the system cannot act
        on, whatever the encoder does."""
        return self.ad_side is not None


@dataclass(frozen=True)
class GoldParse:
    persona_id: str
    constraints: tuple[Constraint, ...]

    def by_facet(self, prefix: str) -> tuple[Constraint, ...]:
        return tuple(c for c in self.constraints
                     if c.facet == prefix or c.facet.startswith(prefix + "."))

    @property
    def hard(self) -> tuple[Constraint, ...]:
        return tuple(c for c in self.constraints if c.priority == "hard")


class GoldParseError(ValueError):
    """A parse that would silently weaken the evaluation if it loaded."""


def load_personas(path: Path | None = None) -> dict[str, dict]:
    p = path or PERSONAS_PATH
    ps = yaml.safe_load(p.read_text(encoding="utf-8"))["personas"]
    return {x["id"]: x for x in ps}


def load(path: Path | None = None,
         personas: Mapping[str, dict] | None = None) -> dict[str, GoldParse]:
    """Load every gold parse, validating as we go. Raises rather than warning:
    a parse that fails these checks produces a number that looks fine."""
    p = path or PARSES_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rows = raw.get("gold_parses") or []
    people = dict(personas) if personas is not None else load_personas()

    out: dict[str, GoldParse] = {}
    for row in rows:
        pid = row.get("persona_id")
        if pid not in people:
            raise GoldParseError(f"unknown persona_id {pid!r}")
        if pid in out:
            raise GoldParseError(f"duplicate parse for {pid!r}")
        query = _norm(people[pid]["query"])
        cons: list[Constraint] = []
        for i, c in enumerate(row.get("constraints") or []):
            where = f"{pid}[{i}]"
            facet = c.get("facet")
            if facet not in FACETS:
                raise GoldParseError(
                    f"{where}: unknown facet {facet!r}; "
                    f"extend FACETS deliberately, do not spell it differently")
            priority = c.get("priority")
            if priority not in PRIORITIES:
                raise GoldParseError(
                    f"{where}: priority must be one of {sorted(PRIORITIES)}, "
                    f"got {priority!r}")
            ev = c.get("evidence")
            if not ev or not str(ev).strip():
                raise GoldParseError(f"{where}: evidence is required")
            # RULE 1. The seeker has to have said it.
            if _norm(ev) not in query:
                raise GoldParseError(
                    f"{where}: evidence not found verbatim in the persona query.\n"
                    f"  evidence: {_norm(ev)!r}\n"
                    f"  query:    {query!r}")
            declared = c.get("ad_side", FACETS[facet])
            if declared != FACETS[facet]:
                raise GoldParseError(
                    f"{where}: ad_side for {facet!r} is {FACETS[facet]!r}, "
                    f"not {declared!r} — the mapping is fixed by the schema")
            if facet in ("language.norwegian", "language.english"):
                if c.get("value") not in LANGUAGE_LEVELS:
                    raise GoldParseError(
                        f"{where}: {facet} must be a capability level from "
                        f"{sorted(LANGUAGE_LEVELS)}, got {c.get('value')!r}")
            cons.append(Constraint(facet=facet, value=c.get("value"),
                                   priority=priority, evidence=_norm(ev),
                                   ad_side=FACETS[facet]))
        out[pid] = GoldParse(persona_id=pid, constraints=tuple(cons))
    return out


def check_against_personas(parses: Mapping[str, GoldParse],
                           personas: Mapping[str, dict]) -> None:
    """RULE 2, cross-file. The control personas state no language constraint, and
    a parse that invents one destroys the non-effect those four exist to protect
    — every seeker would be treated as constrained and the corpus would shrink to
    9.4% for everyone."""
    for pid, parse in parses.items():
        p = personas[pid]
        langs = parse.by_facet("language")
        stated = [c for c in langs if c.facet in ("language.norwegian",
                                                  "language.english")]
        if p.get("expect_language_constraint") is False and stated:
            raise GoldParseError(
                f"{pid} is marked `expect_language_constraint: false` but its "
                f"parse states {[c.facet for c in stated]}. A control persona "
                f"with a language constraint is not a control.")
        if p.get("control") and stated:
            raise GoldParseError(
                f"{pid} is a control persona but its parse states a language "
                f"constraint: {[c.facet for c in stated]}")


def to_seeker_profile(parse: GoldParse, personas: Mapping[str, dict]):
    """Bridge into `retrieval.constraints`.

    The ONLY place rule 2 becomes code: no `language.norwegian` row means
    `language_constraint=None`, which makes `language_severity` return exactly
    0.0 and `apply` return the ranking unchanged element for element.
    """
    from finn_smart_search.retrieval.constraints import (
        LanguageConstraint, SeekerProfile)

    no = [c for c in parse.constraints if c.facet == "language.norwegian"]
    constraint = LanguageConstraint(norwegian=no[0].value) if no else None
    other = {}
    for c in parse.constraints:
        if c.facet == "language.norwegian":
            continue
        other.setdefault(c.facet, []).append(c.value)
    return SeekerProfile(
        raw_query=_norm(personas[parse.persona_id]["query"]),
        language_constraint=constraint,
        other=other,
    )


def coverage(parses: Iterable[GoldParse]) -> dict:
    """The oracle's own ceiling, before any query runs.

    A `scoreable=False` constraint is one the seeker stated and the corpus never
    recorded — a day-shift preference the census does not extract cannot be
    honoured by a perfect parser and a perfect encoder. This fraction belongs
    beside any headline the ideal-case eval produces; see §4 on the fifteen
    extracted facets of which four are scored.
    """
    parses = list(parses)
    cons = [c for p in parses for c in p.constraints]
    scoreable = [c for c in cons if c.scoreable]
    hard = [c for c in cons if c.priority == "hard"]
    hard_scoreable = [c for c in hard if c.scoreable]
    per_facet: dict[str, int] = {}
    for c in cons:
        per_facet[c.facet] = per_facet.get(c.facet, 0) + 1
    return {
        "n_parses": len(parses),
        "n_constraints": len(cons),
        "n_scoreable": len(scoreable),
        "scoreable_share": (len(scoreable) / len(cons)) if cons else 0.0,
        "n_hard": len(hard),
        "n_hard_scoreable": len(hard_scoreable),
        "hard_scoreable_share": (len(hard_scoreable) / len(hard)) if hard else 0.0,
        "unscoreable_facets": sorted(
            {c.facet for c in cons if not c.scoreable}),
        "per_facet": dict(sorted(per_facet.items())),
    }
