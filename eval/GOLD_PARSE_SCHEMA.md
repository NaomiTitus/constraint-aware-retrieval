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
      - facet: language.norwegian
        value: none
        priority: hard
        evidence: "I do not speak Norwegian"
```

Four fields, and no `ad_side`: what a facet compares against is a property of the
facet, resolved from the table below.

### Fields

| field | meaning |
|---|---|
| `facet` | From the closed vocabulary below. An unknown facet fails validation rather than being silently carried. |
| `value` | The seeker's stated value. For `language.*` these are capability levels (`none`…`native`), matching `constraints.NorwegianLevel`. |
| `priority` | `hard` — a job violating this is not viable. `soft` — a preference that ranks. |
| `evidence` | **A verbatim substring of the persona's `query`.** See below. |

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

| facet | example value | compared against | populated |
|---|---|---|---:|
| `occupation` | `nurse` | `ad_taxonomy.job_title_standardised` | 100% |
| `skill` | `Python` | `ad_facets.skills` | 32.8% |
| `language.norwegian` | `none` \| `basic` \| `conversational` \| `fluent` \| `native` | `ad_facets.norwegian_requirement_level` | 63.5% |
| `language.english` | same scale | `ad_facets.stated_working_language` | 4.7% |
| `language.other` | `Polish` | — | — |
| `credential.authorisation` | `norsk autorisasjon` | `ad_facets.authorisation_required` | 16.7% |
| `credential.licence` | `førerkort klasse B` | — | — |
| `credential.trade_certificate` | `fagbrev` | — | — |
| `location.place` | `Oslo` | `ad_locations.municipal` | 99.1% |
| `location.anywhere` | `true` | `ad_locations.country` | 100% |
| `work.remote` | `required` \| `preferred` | — | — |
| `contract.permanence` | `permanent` | `ads.engagementtype` (`Fast`) | 99.9% |
| `contract.extent` | `full_time` | `ads.extent` | 100% |
| `contract.shift` | `day` \| `evening_weekend` \| `shift` | — | — |
| `experience.years` | `10` | `ad_facets.min_years_experience` | 6.1% |
| `seniority` | `senior` | `ad_facets.seniority` | 20.5% |

`ad_side` is **derived from the facet, never authored per constraint.** It is a
property of the facet, so hand-maintained copies could only drift — the original
need to validate 71 of them was the smell. A row may still state one, and then it
must agree.

Population figures are measured by `scripts/measure_oracle_ceiling.py`, which also
asserts that every mapping names a `table.column` that actually exists. That check
is not decoration: the first version of this mapping pointed `occupation` at
`ad_facets.occupation`, **a field that does not exist**, and counted 13 constraints
as scoreable against nothing — while marking `location.place` and
`contract.permanence` unscoreable although the corpus covers them at 99%+.

## Two ceilings, and conflating them is how the first version was wrong

**Schema coverage — does a field exist at all?** Structural and cheap; this is
what `gold_parse.coverage()` reports. On the 13 dev personas: **81.7%** of stated
constraints, **81.1%** of the hard ones.

**Population-weighted coverage — on a randomly drawn ad, does a VALUE exist?**
This is the operative ceiling. `ad_facets.skills` exists on every row and carries
something on 32.8%; `min_years_experience` on 6.1%. A facet at 6% is nominally
scoreable and practically not. On the dev personas: **53.5%** of stated
constraints, **66.6%** of the hard ones.

Where the dev ceiling is actually lost:

| facet | n | hard | populated | |
|---|---:|---:|---:|---|
| `skill` | 19 | 0 | 32.8% | sparse — the largest facet, and the weakest |
| `credential.licence` | 5 | **5** | — | no counterpart; `førerkort klasse B` gates real jobs |
| `contract.shift` | 4 | 0 | — | no counterpart |
| `credential.trade_certificate` | 2 | **2** | — | no counterpart; `fagbrev` gates trade work |
| `work.remote` | 2 | 0 | — | no counterpart |
| `credential.authorisation` | 2 | **2** | 16.7% | sparse, and it is a hard disqualifier |
| `experience.years` | 2 | 0 | 6.1% | sparse |

Nine of the 37 hard dev constraints have **no corpus counterpart at all**, and two
more sit on fields populated under 20%. These are must-haves, not preferences: a
seeker without `fagbrev` cannot hold the job, and the corpus does not record
whether the job wants it.

This is not a defect in the schema — it *is* the measurement, and it is §4's "ten
of fifteen facets scored by nothing" seen from the seeker's side. No encoder and no
parser raises it, because it is a property of what advertisers chose to write.

## Reading the result honestly

The ideal-case number this produces is a **ceiling**, not a product estimate. It
assumes a parser with zero error, which D6 will not be. The gap between this
number and the D6-parsed number is the price of the input layer, and reporting
both is the only way to attribute a failure to the right stage.

Sealed personas are parsed here too. Authoring a parse reads only the frozen query
text, which has been in the repo since before the split — it is not inspecting
results, and it does not touch the seal. Running the sealed parses is what waits
for the final run.
