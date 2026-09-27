"""Skills extraction. Scenarios approved 2026-09-27.

WHY THIS EXISTS. The census populated `skills` on 32.8% of ads, and of the 6,828
with NOTHING extracted, 94.6% carry a requirements list 8+ lines long. So the
qualifications are in the text and were not taken. Two measured causes:

  - 9 of the 15 few-shot examples show `skills: []`, and all six non-empty ones
    contain ONLY certificates — `Truckførerbevis T4`, `Førerkort klasse D`, `YSK`,
    `fagbrev som tømrer`. The prompt taught "skills means named licences", which is
    why the modal non-zero count is exactly 1 and only 11 ads of 10,166 reach the
    schema cap of 8.
  - Nothing scored the field. The golden set covers five keys and `skills` is not
    among them, so this is LIMITATIONS §11's mechanism exactly: a gate that cannot
    see a field cannot certify it, and census-v10 scored byte-identically to v8
    while silently changing an unscored facet.

WHAT A SKILL IS HERE, because the corpus mixes four things in one bullet list and
only one of them is a skill:

    task/duty          `Betjene kasse`, `Blodprøvetaking`      what the JOB does
    credential         `Fagbrev som bilmekaniker`              -> credential.*
    personal quality   `Løsningsorientert`, `Gode samarbeidsevner`   not matchable
    SKILL              `Triagering og scoring`, `Cisco nettverk`     <- this one

A skill is a transferable competency, method, domain, or named tool/system a SEEKER
could claim to possess. `Personlige egenskaper` heads 1,730 health ads alone, so
the exclusions carry more weight than the inclusion.

THREE LANGUAGES, AND WHY THE ANSWER IS NOT TRANSLATION. ESCO carries 10,063 skills
with a Norwegian AND an English label for every one — but its Norwegian labels are
translations of ESCO concepts, not employer vocabulary. Measured: 8 of 10 corpus
phrasings get ZERO hits in ESCO's Norwegian labels while their English equivalents
are present (`medikamentadministrasjon` 0 vs `medication` 18; `vareeksponering` 0 vs
`merchandis` 15). Nynorsk gets zero hits on every form tested — ESCO Norwegian is
bokmål only.

So each skill carries BOTH:

    phrase     verbatim from the advertisement, any language. THE EVIDENCE, and
               byte-checkable, following §1's standing rule that the recorded span
               is the Norwegian original and never the gloss.
    gloss_en   a literal English canonical form. THE MATCH KEY, because the
               measurement says English is where ESCO's coverage lives. It also
               handles nynorsk for free: `sikkerheitskurs` and `sikkerhetskurs`
               gloss to one key, which no string matcher can do.

The cost is stated rather than hidden: matching now depends on the gloss being
right, which is §1's standing caveat. The verbatim phrase is what keeps it
auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .census_validate import norm_keep_blocks
from .text_norm import normalise

SKILLS_TOOL_NAME = "record_ad_skills"
SKILLS_PROMPT_VERSION = "skills-v1"
LEVELS = ("required", "preferred")
MAX_SKILLS = 12          # the census schema capped at 8 and 11 ads ever reached it


class SkillsValidationError(ValueError):
    """A skill record that would corrupt the facet if it were stored."""


@dataclass(frozen=True)
class Skill:
    phrase: str
    gloss_en: str
    level: str


def skills_tool_schema() -> dict[str, Any]:
    """`maxItems` is 12, not the census's 8.

    Only 11 ads of 10,166 ever reached 8, so the old cap was never the binding
    constraint — but a requirements list of 10 bullets is ordinary in this corpus
    and a cap the model can see shapes how much it looks for.
    """
    return {
        "name": SKILLS_TOOL_NAME,
        "description": ("Record the skills and competencies an advertisement asks "
                        "for. Not its duties, not its licences, not personality."),
        "input_schema": {
            "type": "object",
            "properties": {
                "skills": {
                    "type": "array",
                    "maxItems": MAX_SKILLS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "phrase": {
                                "type": "string",
                                "description": ("Copied from the advertisement word "
                                                "for word, in its own language."),
                            },
                            "gloss_en": {
                                "type": "string",
                                "description": ("The same skill as a short, literal "
                                                "English noun phrase."),
                            },
                            "level": {"type": "string", "enum": list(LEVELS)},
                        },
                        "required": ["phrase", "gloss_en", "level"],
                    },
                },
            },
            "required": ["skills"],
        },
    }


def render_prompt(ad_title: str, ad_text: str) -> str:
    """The extraction prompt.

    Deliberately contains NO example of an empty result. Nine of the census's
    fifteen few-shots showed an empty list and the model learned to return nothing,
    which is the measured cause of the 32.8% population rate.
    """
    return f"""Read this Norwegian job advertisement and list the SKILLS it asks for.

ADVERTISEMENT
Title: {ad_title}
---
{ad_text}
---

A SKILL is a transferable competency, method, subject domain, or named tool or
system that a job seeker could claim to possess.

THREE THINGS IN THIS LIST ARE NOT SKILLS, and in this corpus they outnumber the
real ones. Read the section headings: "Arbeidsoppgaver" and "Personlige
egenskaper" together head more bullets than every skill combined.

  1. DUTIES — what the job involves, not what the applicant brings. Under
     "Arbeidsoppgaver" or "Sentrale arbeidsoppgåver" you will find lines like
     "Betjene kasse", "Vaske og rydde bord", "Ha tilsyn med gutten når han sover".
     Those describe the work. Skip them, UNLESS the same line also names a
     competency the applicant must already have, such as "Triagering og
     systematisk scoring" or "Arbeid med Cisco nettverk".
  2. PERSONAL QUALITIES — under "Personlige egenskaper" or "Personlege
     eigenskapar": "Strukturert, selvstendig og engasjert", "Gode
     samarbeidsevner", "Løsningsorientert", "Personlig egnethet vektlegges".
     These are character, not capability. Skip all of them.
  3. CREDENTIALS — a licence, certificate, authorisation or degree is recorded
     elsewhere and must NOT appear here: "Fagbrev som bilmekaniker", "Norsk
     autorisasjon som sykepleier", "Førerkort klasse B", "Treårig utdanning som
     barnehagelærer", "Politiattest". Skip them. The SKILL a credential implies
     may still be listed if the advertisement names it separately.

Language requirements are recorded elsewhere too. Do not list "Gode
norskkunnskaper" or "Behersker norsk" as a skill.

FOR EACH SKILL, GIVE THREE THINGS.

  phrase     Copied from the advertisement WORD FOR WORD, in the language the
             employer wrote it. Do not translate it, do not tidy it, do not add
             the bullet character. It must appear in the text above exactly.
  gloss_en   The same skill written as a short, literal English noun phrase —
             "medication administration", "restocking shelves", "fault-finding
             and diagnostics". Literal, not fluent: this is how the skill is
             matched, and a loose translation matches the wrong thing.
  level      "required" if the advertisement demands it; "preferred" if it is
             framed as ønskelig, en fordel, gjerne, fortrinnsvis, a plus.

THREE LANGUAGES APPEAR IN THIS CORPUS AND ALL THREE COUNT. Most advertisements
are Norwegian bokmål. A substantial minority are NYNORSK — "arbeidsoppgåver",
"personlege eigenskapar", "sikkerheitskurs", "kunnskapar" — and those skills are
just as real; keep the nynorsk phrase and give the English gloss. Some
advertisements are written entirely in English, and there the phrase and the
gloss may be the same words.

WORKED EXAMPLES, all from real advertisements in this corpus.

  "Medikamentadministrasjon og deltakelse i legevisitt"
      -> phrase as written, gloss "medication administration", required
  "Triagering og systematisk scoring (NEWS, ProAct)"
      -> gloss "triage and early warning scoring", required
  "Vareeksponering og oppfølging av butikkens visuelle uttrykk"
      -> gloss "merchandising and visual presentation", required
  "Kjennskap til moderne forsyningskonsept med mellom anna just-in-time-leveransar"
      -> nynorsk; gloss "just-in-time supply chain knowledge", preferred
  "Sertifisering fra CISCO og erfaring fra nettverksdrift"
      -> gloss "Cisco network operations", preferred
  "Gode kunnskaper om mat og hygiene"
      -> gloss "food and hygiene knowledge", required

List every skill the advertisement asks for, up to {MAX_SKILLS}. Call the tool once."""


def validate_skill(skill: Skill, ad_text: str) -> None:
    """Raise `SkillsValidationError` unless this skill could be true of this ad.

    Normalisation goes through `text_norm.normalise`, the single shared
    implementation, so both sides reduce identically — two divergent normalisers
    silently reject correct evidence, which is why that module exists.

    NO BOUNDARY CHECK, unlike `census_validate._span_ok`, and the difference is
    deliberate. A language span must not be a fragment: 578 ads contain "norsk og
    engelsk", so "engelsk" lifted out of it supports the opposite of what the
    sentence says. A SKILL phrase is legitimately a fragment — "Erfaring med
    sårbehandling" taken from "Erfaring med sårbehandling er en fordel" is correct
    quoting — so importing the boundary rules would reject truthful skills.

    It also does NOT judge whether the phrase is really a skill rather than a duty
    or a personality trait. That is the prompt's job and the golden set's job to
    score; a regex classifier here would be the eighth hand-written pattern in this
    repository to look green and measure nothing.
    """
    if skill.level not in LEVELS:
        raise SkillsValidationError(
            f"level {skill.level!r} is not one of {LEVELS}; the level is what "
            f"carries hard/soft into the scorer's weighted coverage")
    if not skill.gloss_en.strip():
        raise SkillsValidationError(
            "gloss_en is empty; it is the match key, and ESCO's Norwegian labels "
            "miss 8 of 10 corpus phrasings so the English form is what resolves")
    if not skill.phrase.strip():
        raise SkillsValidationError("phrase is empty; it is the evidence")
    if normalise(skill.phrase) not in norm_keep_blocks(ad_text):
        raise SkillsValidationError(
            f"phrase not found verbatim in the advertisement: {skill.phrase!r}")
