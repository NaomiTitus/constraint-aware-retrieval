"""Prompt + tool schema for the corpus-wide LLM facet census.

Design notes
------------
* The SCHEMA is the contract, not prose instructions. We use tool-use with a strict
  input_schema so the model cannot return malformed JSON.
* SYSTEM + few-shot are a STABLE PREFIX marked with cache_control. At ~1,400 tokens
  and 10,166 calls, cache reads (0.1x input) make the few-shot examples effectively
  free after the first call - so we can afford good ones.
* Every judgement must be defensible: either a verbatim span, or an explicit
  evidence_basis saying why no span exists. Post-validation rejects invented spans.
* Skills are extracted as SURFACE PHRASES here. Mapping to ESCO URIs happens
  afterwards, deterministically, via the Aho-Corasick gazetteer. Keeping the LLM
  out of taxonomy assignment stops it inventing URIs.
"""

PROMPT_VERSION = "census-v1"

SYSTEM = """You analyse Norwegian job advertisements for a job-search engine whose users \
include people who do not speak Norwegian.

Your single most important job is to decide, accurately and conservatively, whether a \
non-Norwegian speaker could realistically apply for this position.

CRITICAL DISTINCTIONS — these are the errors that matter most:

1. PROFESSIONAL AUTHORISATION IS NOT A LANGUAGE REQUIREMENT.
   "norsk autorisasjon", "autorisasjon som sykepleier i Norge", "HPR-nummer" describe
   licensing to practise a regulated profession. They say NOTHING about language.
   Record them under `authorisation_required`, never as a Norwegian language requirement.

2. SCANDINAVIAN IS NOT ENGLISH.
   "skandinavisk språk" admits Swedish and Danish but excludes an English-only speaker.
   That is `working_language: "scandinavian"`, and Norwegian IS effectively required.

3. WRITTEN LANGUAGE IS NOT A STATED REQUIREMENT.
   An ad WRITTEN in Norwegian has not thereby stated a requirement. An ad written in
   English may still demand fluent Norwegian. Judge these two things separately.

4. WHEN THE AD CONTRADICTS ITSELF, THE EXPLICIT REQUIREMENT WINS.
   "Our working language is English" plus "must speak fluent Norwegian" resolves to
   `norwegian_required: "required"`. Protect the applicant from a wasted application.

EVIDENCE RULES:
* If a sentence states the language situation, quote it EXACTLY as it appears —
  character for character, in the ad's own language. Do not translate, tidy or
  paraphrase it. Set `evidence_basis: "explicit_statement"`.
* If the ad never mentions language, DO NOT invent a quote. Set `evidence_span: null`
  and choose `evidence_basis`: "document_language" (inferring from the language the ad
  is written in) or "absence_of_requirement".
* An invented quote is worse than an abstention.

ABSTAIN when genuinely ambiguous. `"unknown"` is a valid, useful answer and is
preferred over a confident guess. `confidence` must reflect real uncertainty."""

TOOL = {
    "name": "record_ad_facets",
    "description": "Record the extracted facets for one job advertisement.",
    "input_schema": {
        "type": "object",
        "properties": {
            "norwegian_required": {
                "type": "string",
                "enum": ["required", "preferred", "not_required", "unknown"],
                "description": "Does the JOB require Norwegian? Not whether the ad is written in Norwegian.",
            },
            "working_language": {
                "type": "string",
                "enum": ["norwegian", "english", "both", "scandinavian", "unknown"],
                "description": "The language the team actually works in, if stated.",
            },
            "english_accessible": {
                "type": "boolean",
                "description": (
                    "True if a fluent English speaker with no Norwegian could realistically "
                    "apply. True when the ad is written in English and imposes no Norwegian "
                    "requirement, OR when English is stated as the working language."
                ),
            },
            "evidence_span": {
                "type": ["string", "null"],
                "description": "VERBATIM substring of the ad supporting the verdict, or null. Never paraphrase.",
            },
            "evidence_basis": {
                "type": "string",
                "enum": ["explicit_statement", "document_language", "absence_of_requirement"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "authorisation_required": {
                "type": ["string", "null"],
                "description": "Regulated-profession licence, e.g. 'norsk autorisasjon som sykepleier'. NOT language.",
            },
            "visa_sponsorship": {
                "type": "string",
                "enum": ["offered", "explicitly_not_offered", "unstated"],
            },
            "relocation_support": {
                "type": "string",
                "enum": ["offered", "explicitly_not_offered", "unstated"],
            },
            "skills": {
                "type": "array",
                "description": "Concrete skills/tools/qualifications as they appear in the ad.",
                "items": {
                    "type": "object",
                    "properties": {
                        "phrase": {"type": "string", "description": "Surface form, as written."},
                        "level": {"type": "string", "enum": ["required", "preferred"]},
                    },
                    "required": ["phrase", "level"],
                },
            },
            "min_years_experience": {"type": ["integer", "null"]},
            "seniority": {
                "type": "string",
                "enum": ["intern", "junior", "mid", "senior", "lead", "manager", "executive", "unknown"],
            },
        },
        "required": [
            "norwegian_required", "working_language", "english_accessible",
            "evidence_span", "evidence_basis", "confidence",
            "authorisation_required", "visa_sponsorship", "relocation_support",
            "skills", "min_years_experience", "seniority",
        ],
    },
}

# Few-shot examples chosen to cover the four failure modes named in SYSTEM.
FEWSHOT = [
    ("Sykepleier, nattevakt\n\nVi søker sykepleier til nattevakt. Krav: norsk autorisasjon "
     "som sykepleier og gode norskkunnskaper, muntlig og skriftlig.",
     {"norwegian_required": "required", "working_language": "norwegian",
      "english_accessible": False, "evidence_span": "gode norskkunnskaper, muntlig og skriftlig",
      "evidence_basis": "explicit_statement", "confidence": 0.97,
      "authorisation_required": "norsk autorisasjon som sykepleier",
      "visa_sponsorship": "unstated", "relocation_support": "unstated",
      "skills": [], "min_years_experience": None, "seniority": "unknown"}),

    ("Senior Backend Engineer\n\nWe are an international team building payment "
     "infrastructure. All internal communication is in English. You will work with Go, "
     "Kubernetes and PostgreSQL. 5+ years of backend experience required. "
     "We support relocation to Oslo.",
     {"norwegian_required": "not_required", "working_language": "english",
      "english_accessible": True, "evidence_span": "All internal communication is in English",
      "evidence_basis": "explicit_statement", "confidence": 0.95,
      "authorisation_required": None, "visa_sponsorship": "unstated",
      "relocation_support": "offered",
      "skills": [{"phrase": "Go", "level": "required"},
                 {"phrase": "Kubernetes", "level": "required"},
                 {"phrase": "PostgreSQL", "level": "required"}],
      "min_years_experience": 5, "seniority": "senior"}),

    ("Barista – Oslo S\n\nWe're looking for a friendly barista for our busy coffee bar. "
     "Experience with espresso machines is a plus. Shifts include weekends.",
     {"norwegian_required": "unknown", "working_language": "unknown",
      "english_accessible": True, "evidence_span": None,
      "evidence_basis": "document_language", "confidence": 0.6,
      "authorisation_required": None, "visa_sponsorship": "unstated",
      "relocation_support": "unstated",
      "skills": [{"phrase": "espresso machines", "level": "preferred"}],
      "min_years_experience": None, "seniority": "unknown"}),

    ("Kundebehandler\n\nDu må beherske skandinavisk språk. Vi tilbyr opplæring.",
     {"norwegian_required": "required", "working_language": "scandinavian",
      "english_accessible": False, "evidence_span": "Du må beherske skandinavisk språk",
      "evidence_basis": "explicit_statement", "confidence": 0.9,
      "authorisation_required": None, "visa_sponsorship": "unstated",
      "relocation_support": "unstated", "skills": [],
      "min_years_experience": None, "seniority": "unknown"}),
]

USER_TEMPLATE = "<advertisement>\n{title}\n\n{body}\n</advertisement>"
MAX_BODY_CHARS = 6000   # only 4% of ads exceed this
