# Gold parse schema — the seeker side, hand-authored

**Committed before any retrieval ran.** This is the oracle input for the ideal-case
evaluation: the structured attribute set **S** that a perfect query parser would
produce from each persona's frozen query text. It exists so retrieval can be
measured *without* the input layer, because a failure with a perfect parse and a
failure with a noisy parse are different failures and only one of them is the
retrieval stage's fault.

It has a second life. `PLAN.md` D6 is the query parser — gazetteer → distilled
classifier → LLM residue — and this file is the gold standard D6 gets scored
against. So the work is not throwaway: it is simultaneously retrieval's oracle and
the parser's test set.

---

## The operation these parses feed

From the README: a job states requirements **R**, the seeker has attributes **S**,
the job is viable iff **R ⊆ S**, and viable jobs rank by weighted coverage of what
the seeker asked for. These parses are **S**. The census's `ad_facets` are **R**.

Containment is asymmetric, which is why every constraint carries a `priority`: a
missing `hard` constraint disqualifies, a missing `soft` one only demotes. A flat
match count would rank a job missing `norsk autorisasjon` above one missing a
shift preference, and 462 ads carry an authorisation requirement.

## Shape

A flat list of constraints per persona. Flat, not nested, for three reasons: it
validates cleanly, coverage is a simple fold over it, and scoring D6 later becomes
a set comparison against the parser's output rather than a tree diff.

```yaml
gold_parses:
  - persona_id: p1_sykepleier_no_norsk
    constraints:
      - facet: occupation
        value: nurse
        priority: hard
        evidence: "I am a nurse"
        ad_side: occupation
      - facet: language.norwegian
        value: none
        priority: hard
        evidence: "I do not speak Norwegian"
        ad_side: norwegian_requirement_level
```

### Fields

| field | meaning |
|---|---|
| `facet` | From the closed vocabulary below. An unknown facet fails validation rather than being silently carried. |
| `value` | The seeker's stated value. For `language.*` these are capability levels (`none`…`native`), matching `constraints.NorwegianLevel`. |
| `priority` | `hard` — a job violating this is not viable. `soft` — a preference that ranks. |
| `evidence` | **A verbatim substring of the persona's `query`.** See below. |
| `ad_side` | The `ad_facets` field this compares against, or `null` when the corpus has no counterpart. |

## The two rules that make this an oracle and not a wish list

### 1. Every constraint cites verbatim evidence from the frozen query

`evidence` must appear **character-for-character** in that persona's `query` after
whitespace normalisation, and `tests/unit/test_gold_parse.py` asserts it.

This is the census's own discipline turned around. On the ad side,
`census_validate` checks every `evidence_span` byte-for-byte against the
advertisement, so the extractor cannot invent a requirement. The same risk exists
here and is easier to miss: hand-authoring **S** invites writing the profile the
*corpus* would like — ESCO labels, census enum values, the vocabulary the ads
actually use. That would quietly delete the vocabulary mismatch LIMITATIONS §14
measured, and the oracle would flatter retrieval by construction.

So the rule is mechanical: if the seeker did not say it, it is not in **S**. A
constraint the parser could never recover from the text has no business in the
gold standard it will be scored against.

### 2. Absence means NOT STATED, and is never defaulted

A facet with no entry means the seeker said nothing about it. That is a distinct
state from every stated value, including the ones that look like defaults.

This is `constraints.SeekerProfile.language_constraint = None`, which returns
severity **exactly** 0.0 rather than merely small, because 3,507 ads — 34.5% of
the corpus — hang on the difference. A gold parse that fills in
`language.norwegian: fluent` for a persona who never mentioned language would
break the non-effect the four control personas exist to protect. Enforced:
`tests/unit/test_gold_parse.py` cross-checks every persona marked
`expect_language_constraint: false` against its parse and fails if a
`language.*` constraint appears.

## Facet vocabulary

Closed. Extending it is an edit here plus a test update, deliberately.

| facet | example value | `ad_side` |
|---|---|---|
| `occupation` | `nurse` | `occupation` |
| `skill` | `Python` | `skills` |
| `language.norwegian` | `none` \| `basic` \| `conversational` \| `fluent` \| `native` | `norwegian_requirement_level` |
| `language.english` | same scale | `stated_working_language` |
| `language.other` | `Polish` | `null` |
| `credential.authorisation` | `norsk autorisasjon (sykepleier)` | `authorisation_required` |
| `credential.licence` | `førerkort klasse B` | `null` |
| `credential.trade_certificate` | `fagbrev tømrer` | `null` |
| `location.place` | `Oslo` | `null` |
| `location.anywhere` | `true` | `null` |
| `work.remote` | `required` \| `preferred` | `null` |
| `contract.permanence` | `permanent` | `null` |
| `contract.extent` | `full_time` | `null` |
| `contract.shift` | `day` \| `evening_weekend` \| `shift` | `null` |
| `experience.years` | `10` | `min_years_experience` |
| `seniority` | `senior` | `seniority` |

## The coverage ceiling this exposes, which is the point

Most rows above have `ad_side: null`. The census extracts fifteen facets and only
four are scored (`LIMITATIONS.md` §4); of the seeker-side facets the personas
actually state, a minority have any corpus counterpart at all.

That is not a defect in this schema — it is the measurement. `null` rows are
**stated constraints the system cannot act on**, and counting them gives the
oracle's own ceiling before a single query is run. A perfect parser and perfect
retrieval still cannot honour a day-shift preference the corpus never recorded.
`gold_parse.coverage()` reports that fraction, and it belongs beside any headline
number the ideal-case eval produces.

## Reading the result honestly

The ideal-case number this produces is a **ceiling**, not a product estimate. It
assumes a parser with zero error, which D6 will not be. The gap between this
number and the D6-parsed number is the price of the input layer, and reporting
both is the only way to attribute a failure to the right stage.

Sealed personas are parsed here too. Authoring a parse reads only the frozen query
text, which has been in the repo since before the split — it is not inspecting
results, and it does not touch the seal. Running the sealed parses is what waits
for the final run.
