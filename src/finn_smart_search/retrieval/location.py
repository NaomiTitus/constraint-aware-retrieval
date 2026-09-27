"""Location as a predicate over metadata. Grounded 2026-09-27.

WHY THIS EXISTS. The static demo made the gap visible: location was keyword-only, so
"nurse in Bergen" scored `bergen` as a search term and a Bergen advertisement in the
wrong occupation outranked a nurse advertisement elsewhere. A place is not a word in
a bag — it is a predicate over metadata, exactly as occupation is (D18), and the
metadata exists: `ad_locations` covers 99.1% of advertisements by municipality and
99.8% by county.

WHY PROXIMITY IS GRADED RATHER THAN BOOLEAN. 16 counties over 350 municipalities is
a real hierarchy, so a neighbouring municipality in the same county is a commutable
near-miss: not a match, and not nothing. `SAME_COUNTY = 0.55` sits between, and like
the STYRK weights it is STRUCTURAL and not fitted — fitting it needs relevance
judgments and `eval/JUDGING_PROTOCOL.md` is pre-registered with none applied yet.

THE 7.1% THAT FORCES ANY-MATCH. 722 advertisements carry more than one location
(12,003 rows over 10,165 uuids). So an advertisement's proximity is the BEST of its
locations. A first-match implementation — which the web export currently does — drops
advertisements that genuinely do list the city the seeker asked for.

THE SUFFIX NORMALISER IS NOT COSMETIC. `Oslo-området` is `c4_laerer_norsk_speaker`'s
own gold parse value, and no municipality or county carries that name, so without
stripping the area suffix a real dev persona's HARD location constraint never
resolves at all.

Both non-effects follow `constraints.py`: an unstated location and an unresolvable
one each leave the ranking bit-identical. D18's outcome measured why the second
matters — a predicate applied to one side of a paired comparison and not the other is
a confound, so a gazetteer miss must change nothing rather than change a little.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# Structural, not fitted. Exact municipality, same county, anywhere else.
EXACT, SAME_COUNTY, ELSEWHERE = 1.0, 0.55, 0.0

# "the Oslo area", "Bergen and surroundings". Stripped so a place phrase reaches the
# municipality it names — `Oslo-området` resolves to nothing otherwise, and it is a
# real gold parse value.
_AREA = re.compile(
    r"\s*(?:-|\s)?(?:omr[åa]det|omr[åa]de|omegn|og omegn|regionen|region|"
    r"kommune|sentrum|area)\s*$", re.IGNORECASE)
_ANYWHERE = re.compile(r"^(anywhere|hele landet|norge|norway|any)$", re.IGNORECASE)


def normalise_place(phrase: str) -> str:
    """Casefold, strip diacritic noise the table does not share, drop area suffixes.

    `ad_locations` stores upper case (`BERGEN`, `TRØNDELAG`) while seekers type mixed
    case, so both sides reduce through here.
    """
    s = unicodedata.normalize("NFKC", str(phrase)).strip().casefold()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    prev = None
    while prev != s:                      # "Bergen og omegn" needs two passes
        prev = s
        s = _AREA.sub("", s).strip(" -")
    return s


@dataclass(frozen=True)
class AdLocation:
    municipal: str | None
    county: str | None


@dataclass(frozen=True)
class LocationConstraint:
    """The seeker's stated place, resolved to municipality and county names.

    `anywhere` is its own state rather than an empty constraint: `p6` states
    "I would like to teach in Norway", and a seeker open to everywhere must not be
    penalised for saying so.
    """
    phrase: str
    municipals: frozenset[str] = frozenset()
    counties: frozenset[str] = frozenset()
    anywhere: bool = False

    @property
    def resolved(self) -> bool:
        return bool(self.municipals or self.counties or self.anywhere)

    def proximity(self, ads: Sequence[AdLocation]) -> float:
        """The BEST of an advertisement's locations — see the 7.1% note above."""
        if self.anywhere:
            return EXACT
        best = ELSEWHERE
        for ad in ads:
            m = normalise_place(ad.municipal or "")
            c = normalise_place(ad.county or "")
            if m and m in self.municipals:
                return EXACT
            if c and c in self.counties:
                # A county query matches every municipality inside it exactly; a
                # municipality query matching only the county is a near-miss.
                best = max(best, EXACT if c in self.counties and
                           not self.municipals else SAME_COUNTY)
            elif c and self.counties_of_municipals and c in self.counties_of_municipals:
                best = max(best, SAME_COUNTY)
        return best

    @property
    def counties_of_municipals(self) -> frozenset[str]:
        """Filled by the gazetteer: the counties the requested municipalities sit in,
        so a neighbouring municipality can score SAME_COUNTY."""
        return getattr(self, "_parent_counties", frozenset())


class LocationGazetteer:
    """Place name -> municipality and county names, built from the corpus itself.

    Built from `ad_locations` rather than an external list, so a name only resolves
    if some advertisement actually carries it.
    """

    def __init__(self, municipals: Mapping[str, str], counties: frozenset[str]):
        self.municipals = dict(municipals)      # municipality -> its county
        self.counties = frozenset(counties)

    @classmethod
    def build_from_rows(cls, rows: Iterable[Sequence[object]]) -> LocationGazetteer:
        """`rows` are (municipal, county) pairs, one per ad location row."""
        muni: dict[str, str] = {}
        counties: set[str] = set()
        for municipal, county in rows:
            m, c = normalise_place(municipal or ""), normalise_place(county or "")
            if c:
                counties.add(c)
            if m:
                muni.setdefault(m, c)
        return cls(muni, frozenset(counties))

    def resolve(self, phrase: str) -> LocationConstraint | None:
        """Seeker phrase -> constraint, or None when nothing carries that name.

        None rather than an empty constraint, because a wrong municipality would
        demote every correct advertisement — failing to resolve is the safer error.
        """
        key = normalise_place(phrase)
        if not key:
            return None
        if _ANYWHERE.match(key):
            return LocationConstraint(phrase=phrase, anywhere=True)

        municipals: set[str] = set()
        counties: set[str] = set()
        if key in self.municipals:
            municipals.add(key)
        if key in self.counties:
            counties.add(key)
        if not municipals and not counties:
            return None
        c = LocationConstraint(phrase=phrase, municipals=frozenset(municipals),
                               counties=frozenset(counties))
        # The counties the requested municipalities sit in, so a neighbouring
        # municipality scores SAME_COUNTY rather than zero.
        parents = frozenset(self.municipals[m] for m in municipals
                            if self.municipals.get(m))
        object.__setattr__(c, "_parent_counties", parents)
        return c


def from_gold_parse(parse, gaz: LocationGazetteer) -> LocationConstraint | None:
    """None when no location was stated — never a default.

    `location.anywhere` is honoured as such: `p6` states it, and treating it as an
    unstated location would be nearly right and quietly different.
    """
    anywhere = [c for c in parse.constraints if c.facet == "location.anywhere"]
    if anywhere:
        return LocationConstraint(phrase="anywhere", anywhere=True)
    place = [c for c in parse.constraints if c.facet == "location.place"]
    if not place:
        return None
    return gaz.resolve(str(place[0].value))


def apply(scores: Sequence[float],
          ad_locations: Sequence[Sequence[AdLocation]],
          constraint: LocationConstraint | None,
          *, lam: float = 1.0, floor: float = 0.0) -> list[float]:
    """Multiply scores by location proximity.

    Returns the input UNCHANGED when no location was stated, when the gazetteer could
    not resolve it, or when the seeker is open to anywhere — the same non-effect rule
    `constraints.apply` and `occupation.apply` follow.

    `floor` keeps an out-of-area advertisement at a fraction of its score rather than
    zero, because some seekers would relocate. That is a product decision with a dial.
    """
    if constraint is None or not constraint.resolved or constraint.anywhere \
            or lam == 0.0:
        return list(scores)
    out = []
    for s, ads in zip(scores, ad_locations):
        p = constraint.proximity(ads)
        out.append(s * (floor + (1.0 - floor) * (1.0 - lam + lam * p)))
    return out
