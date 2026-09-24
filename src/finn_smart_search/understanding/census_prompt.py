"""Prompt + tool schema for the corpus-wide LLM facet census. Version 2.

v2 incorporates three independent reviews, each finding validated against the
corpus before acceptance. Counts below are measured over 10,166 real ads.

ACCEPTED (measured):
  * Disjunctions "norsk ELLER engelsk" - 304 ads, 274 of them NOT otherwise
    accessible. v1 would have scored these `required` and ranked them DOWN.
    This is the single largest correction: positive class 437 -> 711 (+63%).
  * Certification gates (norskprøve/Bergenstest/B2) - 654 ads. Categorically
    different from "required": a documentary gate you cannot negotiate. v1's
    "autorisasjon is not language" rule actively misfiled these.
  * Mangfoldserklæringen boilerplate - 493 ads of legally-templated diversity
    text with zero language content that reads as openness.
  * Nynorsk - 762 ads carry nynorsk markers; the bokmål-only lexicon missed 21%
    of explicit nynorsk language requirements.
  * Silence is the mode - 74.6% of ads say nothing. v1's few-shot had ZERO
    coverage of it while over-representing the 4% English case 11x.
  * Field order: evidence before verdict (tool JSON is autoregressive, so
    verdict-first invites confabulated support).
  * `english_accessible` removed - derivable, and the only field that let the
    model contradict itself.
  * `confidence` removed - mode-collapses onto demonstrated literals on a small
    model. Replaced by a discrete, verifiable `evidence_strength`.

REJECTED (measurement refuted the claim):
  * Truncation as a severe risk - only 21 ads have a requirement pattern falling
    entirely beyond 6000 chars, not the ~407 claimed. Head+tail applied anyway
    since it is free.
  * Visa-negative boilerplate - 0 occurrences.
  * "språkmodell for barna" - 5 ads, not the 727 implied by occupation counts.
"""

PROMPT_VERSION = "census-v2"

SYSTEM = """You extract language-requirement facts from Norwegian job advertisements for a \
search engine whose users include people who speak English but no Norwegian.

Decide what the advertisement STATES about language. Do not decide whether someone should \
apply. Absent evidence, say so — do not resolve silence in either direction.

## The decision that matters most

Norwegian is written in both bokmål and nynorsk. Treat them identically. Nynorsk forms you \
will encounter: norskkunnskapar, må meistre norsk, munnleg og skriftleg, framstillingsevne, \
ønskjeleg, søkjar, krav til stillinga.

### DISJUNCTIONS ARE ACCESSIBLE — the most commonly mishandled case
"Behersker norsk ELLER engelsk", "kommuniserer godt på norsk eller engelsk", \
"norsk eller engelsk" means an English speaker CAN apply. The sentence contains the word \
norsk, but it does not require Norwegian. Set \
`norwegian_requirement_level: "either_norwegian_or_english"`. This is common in production, \
industrial, warehouse and cleaning roles. Getting it wrong hides jobs from exactly the \
people this system serves.

Contrast: "norsk eller et annet skandinavisk språk" still EXCLUDES an English speaker.

### SILENCE IS SILENCE
Three of every four advertisements say nothing whatever about language. That is \
`"unstated"` — not `required`, not `not_required`. Do not infer a requirement from the fact \
that the advertisement is written in Norwegian. Do not infer openness from its absence.

### LANGUAGE CERTIFICATION IS A REQUIREMENT, NOT A LICENCE
"bestått norskprøve", "Bergenstesten", "norsk nivå B2", "CEFR B1", "Test i norsk – høyere \
nivå", "språkkrav" are DOCUMENTARY LANGUAGE GATES. Record them as \
`norwegian_requirement_level: "certified"`. Never as authorisation.

By contrast "norsk autorisasjon", "autorisasjon som sykepleier", "HPR-nummer" are \
professional licensing to practise a regulated profession. Put those in \
`authorisation_required` and leave the language level unstated unless language is \
separately mentioned. Both can appear in the same advertisement; keep them apart.

### BOILERPLATE THAT IS NOT ABOUT LANGUAGE
Norwegian public-sector advertisements carry a legally templated diversity clause: \
"Vi oppfordrer alle kvalifiserte til å søke uansett alder, kjønn, funksjonsevne, etnisitet, \
nasjonal opprinnelse eller hull i CV-en." It has NO language content. Never quote it as \
evidence and never let it suggest openness. The same applies to "offentlig søkerliste", \
"politiattest", "NOKUT-godkjent vitnemål" and MRSA/tuberculosis screening.

### CONTRADICTIONS
If one passage states English is the working language and another demands fluent Norwegian, \
record the STRONGER Norwegian requirement and set `conflicting_statements: true`.

## Evidence

Quote COMPLETE SENTENCES, copied character-for-character from the advertisement, in its own \
language. Never translate, tidy, shorten or paraphrase. A fragment such as "engelsk" lifted \
out of "du må beherske norsk og engelsk" is a FALSIFICATION — it is a real substring that \
supports the opposite of what the sentence says.

If nothing in the text addresses language, return an empty `evidence_spans` list. An empty \
list is correct and expected for most advertisements. An invented or fragmentary quote is \
far worse than none."""

TOOL = {
    "name": "record_ad_facets",
    "description": "Record language and job facets extracted from one advertisement.",
    "input_schema": {
        "type": "object",
        "properties": {
            # ---- evidence FIRST: the model must look before it judges --------
            "evidence_spans": {
                "type": "array",
                "maxItems": 4,
                "description": (
                    "Complete sentences about language, copied verbatim. Empty for the ~75% "
                    "of ads that say nothing. Never quote the diversity boilerplate."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "span": {"type": "string", "maxLength": 300},
                        "section_language": {"type": "string", "enum": ["no", "en", "other"]},
                    },
                    "required": ["span", "section_language"],
                },
            },
            "evidence_basis": {
                "type": "string",
                "enum": ["explicit_statement", "implicit_signal", "no_mention"],
            },
            "evidence_strength": {
                "type": "string",
                "enum": ["explicit_and_unambiguous", "explicit_but_hedged",
                         "inferred_from_context", "none"],
                "description": "Replaces a numeric confidence. Must be 'none' when evidence_spans is empty.",
            },
            # ---- then the graded verdict -------------------------------------
            "norwegian_requirement_level": {
                "type": "string",
                "enum": [
                    "certified",                  # norskprøve B2, Bergenstest, CEFR level
                    "fluent",                     # flytende norsk, beherske svært godt
                    "professional",               # gode norskkunnskaper muntlig og skriftlig
                    "conversational",             # kan gjøre seg forstått, kommunisere på norsk
                    "desirable",                  # ønskelig, en fordel, vektlegges
                    "scandinavian_accepted",      # skandinavisk språk - excludes English
                    "either_norwegian_or_english",# DISJUNCTION - accessible to English speakers
                    "explicitly_not_required",    # trenger ikke norsk, engelsk er tilstrekkelig
                    "unstated",                   # the modal case, ~75% of ads
                ],
            },
            "stated_working_language": {
                "type": "string",
                "enum": ["norwegian", "english", "both", "scandinavian", "unstated"],
                "description": "ONLY when the ad states it (arbeidsspråket/konsernspråket er ...). Do not infer from the language the ad is written in.",
            },
            "application_language": {
                "type": "string",
                "enum": ["norwegian_required", "english_accepted", "unstated"],
                "description": "Language the APPLICATION must be written in. A process gate, separate from the job requirement.",
            },
            "conflicting_statements": {"type": "boolean"},
            "implicit_evidence": {
                "type": "array",
                "description": "Duties implying written/spoken Norwegian though never named as a language requirement.",
                "items": {
                    "type": "string",
                    "enum": ["journalforing", "sprakmodell_barn", "brukerkontakt",
                             "kundebehandling", "telefonvakt", "undervisning", "none"],
                },
            },
            # ---- non-language facets last ------------------------------------
            "authorisation_required": {
                "type": ["string", "null"],
                "description": "Verbatim licensing phrase, e.g. 'norsk autorisasjon som sykepleier'. NOT language certification.",
            },
            "security_clearance_required": {"type": "boolean"},
            "visa_sponsorship": {"type": "string",
                                 "enum": ["offered", "explicitly_not_offered", "unstated"]},
            "relocation_support": {"type": "string",
                                   "enum": ["offered", "explicitly_not_offered", "unstated"]},
            "seniority": {
                "type": "string",
                "enum": ["intern", "junior", "mid", "senior", "lead", "manager",
                         "executive", "unstated"],
                "description": "Use 'unstated' unless the ad names a level or years of experience.",
            },
            "min_years_experience": {
                "type": ["integer", "null"],
                "description": "Lower bound of any stated range. null if experience is merely 'ønskelig'.",
            },
            "skills": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "phrase": {"type": "string", "description": "Surface form as written in the ad."},
                        "level": {"type": "string", "enum": ["required", "preferred"]},
                    },
                    "required": ["phrase", "level"],
                },
            },
        },
        "required": [
            "evidence_spans", "evidence_basis", "evidence_strength",
            "norwegian_requirement_level", "stated_working_language", "application_language",
            "conflicting_statements", "implicit_evidence", "authorisation_required",
            "security_clearance_required", "visa_sponsorship", "relocation_support",
            "seniority", "min_years_experience", "skills",
        ],
    },
}

# Few-shot in rough CORPUS PROPORTION. v1 had 3/4 explicit statements and 2/4
# accessible, against a corpus that is 74.6% silent and 7% accessible.
FEWSHOT_SPEC = [
    ("silent Norwegian retail ad",        "unstated",      "no_mention"),
    ("silent Norwegian kommune ad",       "unstated",      "no_mention"),
    ("silent Norwegian warehouse ad",     "unstated",      "no_mention"),
    ("gode norskkunnskaper",              "professional",  "explicit_statement"),
    ("DISJUNCTION: norsk eller engelsk",  "either_norwegian_or_english", "explicit_statement"),
    ("nurse: autorisasjon, no language",  "unstated",      "no_mention"),
    ("norskprøve B2 certification gate",  "certified",     "explicit_statement"),
    ("nynorsk: må meistre norsk",         "professional",  "explicit_statement"),
    ("English-written, silent",           "unstated",      "no_mention"),
    ("explicit: trenger ikke norsk",      "explicitly_not_required", "explicit_statement"),
]

# english_accessible is DERIVED, never asked of the model.
ACCESSIBLE_LEVELS = {"either_norwegian_or_english", "explicitly_not_required"}
BLOCKING_LEVELS = {"certified", "fluent", "professional", "conversational",
                   "scandinavian_accepted"}

def derive_english_accessible(facets: dict, doc_lang: str) -> bool:
    lvl = facets["norwegian_requirement_level"]
    if lvl in ACCESSIBLE_LEVELS:
        return True
    if lvl in BLOCKING_LEVELS:
        return False
    if facets.get("stated_working_language") == "english":
        return True
    if lvl == "desirable":
        return doc_lang in ("en", "mixed")
    return doc_lang in ("en", "mixed")        # 'unstated' falls back to document language

USER_TEMPLATE = (
    "<advertisement document_language=\"{doc_lang}\" truncated=\"{truncated}\">\n"
    "{title}\n\n{body}\n</advertisement>"
)
HEAD_CHARS, TAIL_CHARS = 3500, 2500   # head+tail beats head-only; affects 539 ads
