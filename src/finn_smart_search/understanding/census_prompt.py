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

PROMPT_VERSION = "census-v9"

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

THE DISJUNCTION NEED NOT NAME NORWEGIAN. The level is called \
`either_norwegian_or_english`, but it is decided by whether ENGLISH IS ACCEPTED, \
whether or not the sentence names norsk at all. All of these are \
`either_norwegian_or_english`:
  "Må beherske skandinavisk eller engelsk tale"
  "kunne prate engelsk eller ett nordisk språk"
  "Scandinavian or English"
  "Norwegian or English"
  "engelsk eller et skandinavisk språk"

6.5% of advertisements are written in English and 103 state the requirement in \
English wording — "Norwegian or English" (31), "fluent in Norwegian" (20), \
"Scandinavian or English" (12), "Norwegian language skills" (9). Every rule \
here applies to the English phrasing exactly as to the Norwegian.
An English speaker satisfies every one of them by speaking English. 436 \
advertisements use this shape and 364 of them never also say "norsk eller \
engelsk", so no other rule rescues them.

THE CONNECTIVE IS THE WHOLE TEST. Swap "eller" for "og" and the answer \
reverses, because a conjunction demands every language it lists:
  "skandinavisk eller engelsk"  → `either_norwegian_or_english`  (accessible)
  "skandinavisk og engelsk"     → `scandinavian_accepted`        (NOT accessible)
  "norsk eller engelsk"         → `either_norwegian_or_english`  (accessible)
  "norsk og engelsk"            → the Norwegian bar stated       (NOT accessible)
863 advertisements use the conjunction shape and 781 the disjunction — they are \
almost equally common, so reading the connective is not an edge case.

Contrast: "norsk eller et annet skandinavisk språk" still EXCLUDES an English speaker — \
there is no English in the set. That absence, not the presence of a Nordic language, \
is what makes it `scandinavian_accepted`.

### "EN FORDEL" MEANS NOT REQUIRED
"norsk er en fordel", "ønskelig med norsk", "du trenger ikke å snakke flytende \
norsk når du starter" all say Norwegian is NOT a requirement — you may apply and \
learn on the job. Use `desirable`.

Read the clause carefully: in "god formidlingsevne på norsk. Fordel med erfaring \
fra drift", the "fordel" attaches to EXPERIENCE, not to Norwegian, and Norwegian \
is required. The advantage must attach to the language itself.

### PRECEDENCE — when several of these rules fire at once
Most advertisements trigger more than one rule. Apply them in this order and \
stop at the first that matches:

1. **An explicit negation wins.** "Norsk er en fordel, men ikke et krav" is \
   `explicitly_not_required`, not `desirable` — "ikke et krav" overrides "en \
   fordel" in the same sentence.
2. **READ THE CONNECTIVE BEFORE ANYTHING ELSE: OR is not AND.** This decides \
   whether the languages named are ALTERNATIVES or a LIST OF REQUIREMENTS, and \
   nothing below can be applied until you know which.
   - **"eller" / "or" — alternatives.** Any ONE of them suffices. Go to rule 3.
   - **"og" / "and" — requirements.** ALL of them are demanded. English being \
     among them makes it an EXTRA requirement, not an escape route: \
     "gjøre deg forstått på norsk **og** engelsk" still requires Norwegian, so \
     the level is the Norwegian bar stated (`conversational` here) and the ad \
     is NOT English-accessible. "skandinavisk språk **og** engelsk" still \
     requires Scandinavian → `scandinavian_accepted`. Do NOT apply rule 3 to a \
     conjunction.
   A comma-separated list with NO connective at all ("norsk, engelsk", 21 \
   advertisements) is a conjunction — treat it as "og". But a comma list \
   CLOSED by a disjunction is a disjunction: "norsk, engelsk eller polsk" \
   (104 advertisements) offers three alternatives and is \
   `either_norwegian_or_english`. Read to the end of the list before deciding; \
   the closing connective governs the whole list, and the disjunctive form is \
   five times commoner than the bare one.
3. **SCOPE BEATS BAR — for alternatives only.** Having established a \
   DISJUNCTION in rule 2, the set of accepted languages decides the level and \
   the proficiency demanded does not. Ask: **is English one of the \
   alternatives?**
   - **Yes → `either_norwegian_or_english`.** Whatever the bar, whatever the \
     other languages offered, and whether or not norsk is named. "skandinavisk \
     eller engelsk" and "engelsk eller ett nordisk språk" are BOTH this level. \
     A Nordic word in the sentence does not change it — read the whole \
     disjunction before choosing, not the first language you recognise.
   - **No, but a Nordic/Scandinavian language is → `scandinavian_accepted`.** \
     This level means English is ABSENT from the alternatives. Use it only \
     then — even when the bar is only "gjøre seg forstått", and even when the \
     sentence also says "flytende".
4. **A REQUIREMENT outranks a parenthetical.** "Gode kommunikasjonsevner, \
   skandinavisk og engelsk (norsk er en fordel)" requires Scandinavian, so it \
   is `scandinavian_accepted`. The bracketed "fordel" does not soften a \
   requirement stated beside it. Contrast "gode kommunikasjonsevner i engelsk \
   (norsk er en fordel)", which requires only English and IS `desirable`.

### PROFICIENCY BARS, when scope does not decide
`certified` — any named test or level: norskprøve, Norskprøve 2, Bergenstest, \
CEFR A1-C2, "nivå 2", "språkkrav". A scale you can be examined on.
`fluent` — flytende, beherske svært godt, meistre.
`professional` — gode norskkunnskaper, god kunnskap i norsk, må beherske norsk.
`conversational` — gjøre seg forstått, kunne kommunisere, grunnleggende norsk.

### THE REQUIREMENT MUST APPLY TO THE APPLICANT
A sentence may mention Norwegian without demanding it of you. "Jeg har ikke \
verbalt språk" describes the employer. "Undervisning i norskopplæring" describes \
the work. "Fullført mastergrad ved norsk studieinstitusjon" describes an \
institution. None is a language requirement. Only a demand made of the applicant \
counts. A gate that binds only some applicants — "utenlandske søkere må ha \
dokumenterte norskkunnskaper på nivå B2" — still binds, and is `certified`.

### A CLAUSE ABOUT YOUR APPLICATION IS NOT A REQUIREMENT ON YOU
135 advertisements say some version of "dokumentasjon som skal vurderes må \
være på et skandinavisk språk eller engelsk", "vitnemål må vere på eit \
skandinavisk språk eller engelsk", or in English "documentation to be \
considered must be in a Scandinavian language or English". This states what \
language your PAPERWORK may be in. It says nothing about the language of the \
work, and it is NEVER a language requirement — do not quote it as evidence and \
do not let it set the level. If the advertisement contains no other language \
statement, the answer is `unstated`, even though the sentence contains both \
"skandinavisk" and "engelsk". 51 of those 135 have no other language wording \
at all, so this is the only sentence a verdict could come from — and the \
verdict must still be silence.

WHERE IT DOES GO. A statement about the language of the APPLICATION — \
"søknaden må skrives på norsk", "vi ber om at CV lastes opp på engelsk", \
"dokumentasjon må være på et skandinavisk språk eller engelsk" — is recorded \
in `application_language`, never in `norwegian_requirement_level`.

`application_language` HAS ONLY THREE VALUES, and they are its own:
  `norwegian_required`  the application or its documents must be in Norwegian
  `english_accepted`    English is accepted — including "skandinavisk ELLER \
                        engelsk", because that accepts English
  `unstated`            the ad says nothing about the application's language

Do NOT put a level value here. `scandinavian_accepted`, \
`either_norwegian_or_english`, `professional` and `both` belong to OTHER \
fields and are invalid in this one — a record carrying one is discarded. If a \
documents clause accepts English alongside a Scandinavian language, that is \
`english_accepted`.

### HOW TO QUOTE — a span that fails these rules is DISCARDED
Every quote is checked mechanically against the advertisement. A quote that \
fails is thrown away AND the verdict it supported is reset to `unstated`, so a \
bad quote costs you a correct answer. Four rules:

1. **Character for character.** Copy exactly as written — do not retype from \
   memory, do not correct spelling, do not fix the employer's typos. A single \
   wrong letter fails the check. "arbeidsspåket" for "arbeidsspråket" is a fail.
2. **A COMPLETE sentence or a complete bullet.** Start where the sentence or \
   bullet starts and end where it ends. Never quote a sub-clause from the middle: \
   in "Kandidater bør ha førerkort for bil, og kunne prate engelsk eller ett \
   nordisk språk", quote the WHOLE sentence — quoting only "kunne prate engelsk \
   eller ett nordisk språk" starts mid-sentence and fails.
3. **At least 15 characters, and it must contain the language word** that \
   justifies your verdict.
4. **No span beats an approximate span.** If you cannot reproduce the sentence \
   exactly, return an empty `evidence_spans` list. An empty list is a normal, \
   correct outcome — it is far better than a quote that is nearly right.

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

def _f(level, basis, strength, spans=(), **kw):
    """Build a complete facet record; every required field gets a default."""
    d = dict(
        evidence_spans=[{"span": s, "section_language": sl} for s, sl in spans],
        evidence_basis=basis, evidence_strength=strength,
        norwegian_requirement_level=level, stated_working_language="unstated",
        application_language="unstated", conflicting_statements=False,
        implicit_evidence=["none"], authorisation_required=None,
        security_clearance_required=False, visa_sponsorship="unstated",
        relocation_support="unstated", seniority="unstated",
        min_years_experience=None, skills=[],
    )
    d.update(kw)
    return d


# Ten examples, all excerpted from REAL ads in the corpus. Ordered so the modal
# case (silent) comes first and last - recency and primacy both favour it.
FEWSHOT = [
    # 1. THE MODAL CASE - silent, Norwegian-written. ~75% of the corpus.
    ("Ønsker du en nøkkelrolle i utviklingen av Samnanger?",
     "Samnanger har natur, plass, kraft og kort vei til byen.\n"
     "Samnanger Kommunale Utviklingsselskap er opprettet for å koble dette sammen og få mer til å skje.\n"
     "Målet er flere arbeidsplasser, flere boliger og en kommune der enda flere vil bo.",
     "no",
     _f("unstated", "no_mention", "none")),

    # 2. DISJUNCTION - 304 ads. The single most mishandled construction.
    ("Produksjonsmedarbeider – søm og tekniske tekstiler",
     "Kvalifikasjoner:\n"
     "Har gode praktiske ferdigheter.\n"
     "Er selvstendig og løsningsorientert.\n"
     "Trives med fysisk og variert arbeid.\n"
     "Behersker norsk eller engelsk\n"
     "Arbeidsoppgaver:\n"
     "Industrisømarbeider søkes til produksjon av tekniske tekstiler.",
     "no",
     _f("either_norwegian_or_english", "explicit_statement", "explicit_and_unambiguous",
        spans=[("Behersker norsk eller engelsk", "no")])),

    # 3. Silent again, different sector - reinforces the mode.
    ("Lagermedarbeider søkes",
     "Vi søker en lagermedarbeider til vårt team.\n"
     "Arbeidsoppgaver omfatter plukking, pakking og truckkjøring.\n"
     "Truckførerbevis T4 er en fordel.\n"
     "Oppstart etter avtale.",
     "no",
     _f("unstated", "no_mention", "none",
        skills=[{"phrase": "Truckførerbevis T4", "level": "preferred"}])),

    # 4. Standard explicit requirement - the commonest non-silent case.
    ("Søker helsesekretær/sykepleier til legekontoret",
     "Erfaring fra arbeid på legekontor.\n"
     "Erfaring med Infodoc journalsystem.\n"
     "Gode samarbeidsevner og et godt pasientfokus.\n"
     "Gode norskkunnskaper\n"
     "Arbeidsoppgaver:\n"
     "Pasientmottak og telefon, laboratoriearbeid og prøvetaking.",
     "no",
     _f("professional", "explicit_statement", "explicit_and_unambiguous",
        spans=[("Gode norskkunnskaper", "no")],
        implicit_evidence=["kundebehandling"],
        skills=[{"phrase": "Infodoc journalsystem", "level": "preferred"}])),

    # 5. CERTIFICATION GATE, and hedged ("Helst") - 654 ads carry a cert token.
    ("Er du en av våre nye bussjåfører i Hamar og Brumunddal?",
     "Kvalifikasjoner:\n"
     "Førerkort klasse D, YSK og kjøreseddel.\n"
     "God norsk ferdigheter.\n"
     "Helst bestått norskprøve B1\n"
     "Vy bruker bransjens egen Bussnorsktest.\n"
     "Erfaring som yrkessjåfør.",
     "no",
     _f("certified", "explicit_statement", "explicit_but_hedged",
        spans=[("Helst bestått norskprøve B1", "no")],
        skills=[{"phrase": "Førerkort klasse D", "level": "required"},
                {"phrase": "YSK", "level": "required"}])),

    # 5b. NORDIC *OR* ENGLISH - 436 ads, 364 with no other rescue. The pilot got
    # this wrong twice: it quoted the right span and then copied example 6 below,
    # the only other Nordic demonstration. The contrast is the point - keep them
    # adjacent so the discriminating word ("eller engelsk") is what differs.
    ("Vi trenger flere elektrikere i Bergen",
     "Vi søker erfarne elektrikere til oppdrag i Bergen.\n"
     "Krav:\n"
     "Fagbrev som elektriker.\n"
     "Minimum 1 års erfaring.\n"
     "Må beherske skandinavisk eller engelsk tale\n"
     "Vi tilbyr hjelp til å finne bolig.",
     "no",
     _f("either_norwegian_or_english", "explicit_statement", "explicit_and_unambiguous",
        spans=[("Må beherske skandinavisk eller engelsk tale", "no")],
        min_years_experience=1)),

    # 5c. DOCUMENTATION-LANGUAGE CLAUSE ONLY -> unstated. 135 ads carry one and
    # 51 have no other language wording, so this sentence is all a verdict could
    # rest on. It contains both "skandinavisk" and "engelsk" and is still not a
    # language requirement: it is about the paperwork. One of the first 169 real
    # outputs quoted it as evidence, which is what put this example here.
    ("Stipendiat innan marin biologi",
     "Om stillinga\n"
     "Vi har ledig ei treårig stipendiatstilling ved instituttet\n"
     "Kvalifikasjonar\n"
     "Mastergrad i biologi eller tilsvarande\n"
     "Erfaring med feltarbeid er ein fordel\n"
     "Søknad og vedlegg\n"
     "All dokumentasjon som skal vurderast må vere på eit skandinavisk språk eller engelsk\n"
     "Søknadsfrist 1. desember",
     "no",
     _f("unstated", "no_mention", "none")),

    # 6. AUTHORISATION + a SEPARATE Nordic-language line. Both present, kept apart.
    ("Intensivsykepleier til Sørlandet",
     "Kvalifikasjoner:\n"
     "Norsk autorisasjon.\n"
     "Respiratorkompetanse.\n"
     "2 års erfaring fra intensivavdeling.\n"
     "Oppdatert AHLR-kurs.\n"
     "Beherske et nordisk språk flytende, både muntlig og skriftlig",
     "no",
     _f("scandinavian_accepted", "explicit_statement", "explicit_and_unambiguous",
        spans=[("Beherske et nordisk språk flytende, både muntlig og skriftlig", "no")],
        authorisation_required="Norsk autorisasjon",
        implicit_evidence=["journalforing"], min_years_experience=2)),

    # 7. AUTHORISATION ALONE, no language line anywhere. Must NOT become a
    #    language verdict - otherwise the model learns the two travel together.
    ("Sykepleier til nattevakt, sykehjem",
     "Vi søker sykepleier til faste nattevakter.\n"
     "Krav:\n"
     "Norsk autorisasjon som sykepleier.\n"
     "Gyldig politiattest må leveres før oppstart.\n"
     "Turnus med arbeid hver tredje helg.\n"
     "Vi oppfordrer alle kvalifiserte til å søke uansett alder, kjønn, funksjonsevne, etnisitet, nasjonal opprinnelse eller hull i CV-en.",
     "no",
     _f("unstated", "no_mention", "none",
        authorisation_required="Norsk autorisasjon som sykepleier",
        implicit_evidence=["journalforing", "brukerkontakt"])),

    # 8. NYNORSK - 762 ads carry markers; a bokmål-only reader misses 21%.
    ("Fagansvarleg i Eining for Miljø- og velferdstenester",
     "Kvalifikasjonar: Høgskuleutdanning innan helse.\n"
     "Ønskjeleg med vidareutdanning innan pedagogikk eller rettleiing.\n"
     "God kunnskap i norsk, munnleg og skriftleg\n"
     "Førarkort kl.\n"
     "B.\n"
     "Gyldig politiattest må leverast før oppstart.",
     "no",
     _f("professional", "explicit_statement", "explicit_and_unambiguous",
        spans=[("God kunnskap i norsk, munnleg og skriftleg", "no")],
        implicit_evidence=["brukerkontakt"])),

    # 9. EXPLICITLY NOT REQUIRED - only ~5 such ads exist; the class is real but rare.
    ("Er du et nattmenneske? Vi søker tilkallingshjelp på natt",
     "Du er serviceinnstilt og liker å møte mennesker.\n"
     "Du trenger ikke å snakke norsk, men du må kunne kommunisere godt på engelsk.\n"
     "Andre språk er selvfølgelig en fordel.\n"
     "Send oss gjerne en kort søknad.",
     "no",
     _f("explicitly_not_required", "explicit_statement", "explicit_and_unambiguous",
        spans=[("Du trenger ikke å snakke norsk, men du må kunne kommunisere godt på "
                "engelsk.", "no")],
        stated_working_language="english")),

    # 10. ENGLISH-WRITTEN but SILENT. The verdict is still 'unstated' - accessibility
    #     is derived downstream from document_language, never asserted here.
    ("Barista – Oslo S",
     "We are looking for a friendly barista for our busy coffee bar.\n"
     "Experience with espresso machines is a plus.\n"
     "Shifts include weekends and early mornings.",
     "en",
     _f("unstated", "no_mention", "none",
        skills=[{"phrase": "espresso machines", "level": "preferred"}])),

    # 11. FLUENT — a hard proficiency bar, distinct from `professional`.
    ("Selger",
     "Er du klar for en spennende salgskarriere?\n"
     "Hos Verisure er vi stolte av å beskytte over 6,4 millioner hjem verden over.\n"
     "Kvalifikasjoner:\n"
     "Flytende norsk\n"
     "Førerkort klasse B.\n"
     "Resultatorientert og engasjert.",
     "no",
     _f("fluent", "explicit_statement", "explicit_and_unambiguous",
        spans=[("Flytende norsk", "no")],
        skills=[{"phrase": "Førerkort klasse B", "level": "required"}])),

    # 12. CONVERSATIONAL — a LOW bar, and note "norsk OG engelsk" is a
    #     conjunction: both are required, so it is NOT a disjunction.
    ("Burger King Sveberg søker skiftleder 80-100% stilling",
     "Om stillingen\n"
     "Vi søker skiftleder til Burger King Sveberg\n"
     "Stillingen krever ikke tidligere ledererfaring, men erfaring innen serviceyrket\n"
     "Grunnleggende norsk og engelsk kunnskaper er nødvendig for å kunne utføre rollen.",
     "no",
     _f("conversational", "explicit_statement", "explicit_and_unambiguous",
        spans=[("Grunnleggende norsk og engelsk kunnskaper er nødvendig for å "
                "kunne utføre rollen.", "no")])),
]


# english_accessible is DERIVED, never asked of the model.
# `desirable` is accessible: "norsk er en fordel" / "trenger ikke flytende norsk
# når du starter" both say Norwegian is NOT a requirement. Owner ruling: you can
# learn it on the job, so you can apply. Previously it fell back to document
# language, which made it behave identically to `unstated` and wasted a level.
ACCESSIBLE_LEVELS = {"either_norwegian_or_english", "explicitly_not_required",
                     "desirable"}
BLOCKING_LEVELS = {"certified", "fluent", "professional", "conversational",
                   "scandinavian_accepted"}

def derive_english_accessible(facets: dict, doc_lang: str) -> bool:
    """Can a fluent English speaker with NO Norwegian realistically apply?

    Derived, never asked of the model. Order matters: the requirement LEVEL
    dominates, because an English-written ad can still demand fluent Norwegian.
    Only when the level is `unstated` does anything else get consulted.
    """
    lvl = facets["norwegian_requirement_level"]        # fail loud if absent
    if lvl in ACCESSIBLE_LEVELS:
        return True
    if lvl in BLOCKING_LEVELS:
        return False

    # `unstated` only. A STATED working language beats the ad's own language in
    # both directions: an English-written ad that says the working language is
    # Norwegian is NOT accessible, which the doc_lang fallback alone got wrong.
    working = facets.get("stated_working_language", "unstated")
    if working in ("english", "both"):
        return True
    if working in ("norwegian", "scandinavian"):
        return False

    # Silence: fall back to the language the ad is WRITTEN in.
    #
    # "other" (Polish, Sámi, ...) and "unknown" are ACCESSIBLE. A Polish ad is
    # not being kept from you by a Norwegian language requirement, and a false
    # hide is invisible to everyone while a false show costs one click.
    # Measured: 13 Polish ads exist, and a restricted detector labelled 10 of
    # them a confident "no".
    #
    # Everything else — Norwegian, or a detection we could not place — stays
    # inaccessible: 95.3% of the corpus is Norwegian, and silence correlates
    # NEGATIVELY with accessibility in the care and retail roles that dominate.
    return doc_lang in ("en", "mixed", "other", "unknown")


USER_TEMPLATE = (
    "<advertisement document_language=\"{doc_lang}\" truncated=\"{truncated}\">\n"
    "{title}\n\n{body}\n</advertisement>"
)
HEAD_CHARS, TAIL_CHARS = 3500, 2500   # head+tail beats head-only; affects 539 ads


def build_request(title: str, body: str, doc_lang: str, truncated: bool = False) -> dict:
    """Assemble the exact Messages API payload for one advertisement.

    The system block and all ten few-shot turns are marked with cache_control,
    so the ~2.4k-token prefix is billed at 0.1x on every call after the first.
    """
    messages = []
    for ex_title, ex_body, ex_lang, ex_facets in FEWSHOT:
        messages.append({"role": "user", "content": USER_TEMPLATE.format(
            doc_lang=ex_lang, truncated="false", title=ex_title, body=ex_body)})
        messages.append({"role": "assistant", "content": [{
            "type": "tool_use", "id": f"fs_{len(messages)}",
            "name": TOOL["name"], "input": ex_facets}]})
        messages.append({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": f"fs_{len(messages)-1}",
            "content": "recorded"}]})
    # cache breakpoint at the end of the stable prefix
    messages[-1]["content"][0]["cache_control"] = {"type": "ephemeral"}
    messages.append({"role": "user", "content": USER_TEMPLATE.format(
        doc_lang=doc_lang, truncated=str(truncated).lower(), title=title, body=body)})
    return {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "temperature": 0,
        "system": [{"type": "text", "text": SYSTEM,
                    "cache_control": {"type": "ephemeral"}}],
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": TOOL["name"]},
        "messages": messages,
    }
