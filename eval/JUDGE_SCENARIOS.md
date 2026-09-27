# LLM judge — grounding and scenario table

**STANDARDS.md §3 step 2. Awaiting approval; no implementation written.**

`eval/JUDGING_PROTOCOL.md` is pre-registered and names three enforcement
mechanisms for judge isolation. Two of them do not exist: `eval/judge_llm.py` and
`tests/unit/test_judge_isolation.py`. §4 lists the second as a **non-negotiable**.
This document grounds them before either is written.

---

## Step 1 — Grounding

Per §3.0: text shape is counted over the **full corpus**, not a sample; the
tool-use envelope comes from the **recorded fixture**; key vocabularies come from
**the file itself**.

### G1 · Advertisement text length — `SELECT description_text FROM ads`, n=10,166

| | chars |
|---|---:|
| min | 31 |
| p50 | 2,500 |
| p90 | 5,013 |
| p99 | 9,743 |
| max | 18,081 |
| over 4,000 | 2,024 ads — **19.9%** |
| over 8,000 | 166 ads — 1.6% |

Newlines per ad: p50 **32**, p90 52, max 138. So an ad is a *block list*, not
prose — the shape §3.1 row 1 records as having demoted 55% of the census pilot
when a test assumed single-line prose.

### G2 · Leading glyphs on blocks — full corpus, counted

The census bug class: the model quotes ad text **without** the bullet that leads
the block (§3.1 row 3, which reported "0 false rejects" while the bug was live).

| glyph | leads blocks |
|---|---:|
| `•` U+2022 | 4,600 |
| `-` U+002D | 3,558 |
| `·` U+00B7 | 1,944 |
| `*` U+002A | 503 |
| `✅` U+2705 | 182 |
| `●` U+25CF | 162 |
| `«` U+00AB | 143 |
| `📍` U+1F4CD | 135 |

Invisibles present: **U+200B** in 96 ads, **U+FEFF** in 14, **U+2060** in 4.

### G3 · Tool-use envelope — `tests/fixtures/batch_results_real.json`, 3 recorded rows

```
{custom_id, result{type:"succeeded", message{content:[{type:"tool_use", name:...}],
                                             stop_reason:"tool_use", usage{...}}}}
```

`parse_result` already handles this and now takes a `tool_name` argument, so the
judge reuses it rather than re-deriving the envelope.

### G4 · `ad_facets` key vocabulary — from the table itself, closed

16 JSON keys: `application_language`, `authorisation_required`,
`conflicting_statements`, `english_accessible`, `evidence_basis`, `evidence_spans`,
`evidence_strength`, `implicit_evidence`, `min_years_experience`,
`norwegian_requirement_level`, `relocation_support`, `security_clearance_required`,
`seniority`, `skills`, `stated_working_language`, `visa_sponsorship`.

8 columns: `uuid`, `facets`, `english_accessible`, `demoted`, `reasons`,
`prompt_version`, `extractor_version`, `extracted_at`.

### G5 · Persona query shape — `eval/personas.yaml`, n=20

99–167 chars, median 129. Two or three sentences. **14 of 20 written in English.**

### G6 · Pool size — measured, 5 arms × 13 dev personas, depth 10

650 slots → **346 unique (persona, ad) pairs**. Mean 26.6 per persona, min 19,
max 39. At a 10,000-char cap that is ~234k input tokens: under $5 on any current
model, so **cost does not constrain the design** and the judge can be a strong
model rather than the cheapest one.

---

## ⚠ One finding needs a ruling before code

**The non-negotiable as written is unsatisfiable.** §4 says the rendered prompt
"must contain no `ad_facets` field name". But G4 shows two of those field names are
ordinary English words — **`skills`** and **`seniority`** — and
`JUDGING_PROTOCOL.md`'s own grade rubric reads *"judged against the persona's
stated **skills** and preferences"*. A literal test fails on the protocol's own
required wording.

The intent is clear and worth preserving exactly: the judge must never see the
**extractor's output**. Three ways to encode that, and I recommend the third:

1. Test the literal 16 — fails on the protocol's own rubric wording. Rejected.
2. Test only `ad_facets` and facet *values* — too weak; `norwegian_requirement_level`
   could leak and pass.
3. **Test the 14 distinctive field names, plus every column name, plus the literal
   `ad_facets`, plus "no snake_case identifier appears anywhere in the prompt"** —
   with `skills` and `seniority` on a documented allowlist because they are
   ordinary English, and the snake_case rule catching them if they ever appear as
   identifiers.

Option 3 keeps the mechanism strict where it matters and is honest about the two
exceptions. **It does weaken a non-negotiable, so it needs your approval, not
mine** — and if approved it should be recorded as an erratum against the protocol
rather than an edit to it.

**Second, smaller erratum.** `JUDGING_PROTOCOL.md` justifies "silence is not a
violation" with *"74.6% of the corpus says nothing"*. That figure was superseded
by the census: LIMITATIONS §3 records the true `unstated` rate as **36.5%**, and
§3 already notes 74.6% was wrong by a factor of two. **The rule is unaffected** —
only its supporting number. Proposal: append a dated erratum, never rewrite the
pre-registered line.

---

## Step 2 — Scenario table

Tier 1 (`unit`) unless stated. Every fixture cites its grounding.

### `render_prompt(persona_query, ad_title, ad_text)`

| # | Input | Expected | Rationale · grounding |
|---|---|---|---|
| 1 | any persona + ad | prompt contains the ad text verbatim and the persona query verbatim | The judge grades what it is shown. G5, G1 |
| 2 | any | prompt contains **none** of the 14 distinctive `ad_facets` keys, no column name, not `ad_facets` | §4 non-negotiable. Closed vocabulary from G4 |
| 3 | any | prompt contains **no snake_case identifier at all** | Catches `skills`/`seniority` as identifiers while allowing them as English. Ruling above |
| 4 | ad of 18,081 chars (the corpus max, G1) | ad text truncated to 10,000 chars; prompt states it was truncated | 10,000 covers p99=9,743. Truncating silently could cut the requirement sentence and make `violates` wrong. G1 |
| 5 | ad of 2,500 chars (p50, G1) | not truncated; no truncation notice | The common case must be clean. G1 |
| 6 | any | prompt carries all five violation carve-outs: silence, `norsk autorisasjon`, document-language clause, `norsk eller engelsk`, and grade-3-can-violate | Protocol rules are only enforced if they are in the prompt |
| 7 | any | prompt instructs: uncertain → **lower** grade | Protocol "Ties and uncertainty" |
| 8 | any | prompt instructs the grade ignores language, and the violation flag is independent | Protocol; mixing them makes a bad match indistinguishable from an inaccessible one |

### `validate_judgment(judgment, ad_text)` — the census bug class

| # | Input | Expected | Rationale · grounding |
|---|---|---|---|
| 9 | `grade=4` or `grade=-1` | rejected | Scale is 0–3 |
| 10 | `violates=true`, `constraint_evidence=""` | rejected | Protocol: record `true` only on evidence in the ad text |
| 11 | evidence quoted **without** the leading `•` of its block | **accepted** | §3.1 row 3 exactly: the model quotes without the glyph. G2 — `•` leads 4,600 blocks |
| 12 | evidence quoted without a leading `·`, `-`, `*`, `●`, `✅`, `«`, `📍` | accepted | G2, all seven measured |
| 13 | evidence differing from the ad by U+200B / U+FEFF / U+2060 | accepted | G2 — present in 96 / 14 / 4 ads |
| 14 | evidence that appears **nowhere** in the ad, glyph-normalised | **rejected** as fabricated | The span must be real; this is `census_validate`'s discipline |
| 15 | `violates=false` with evidence `""` | accepted | Nothing to quote |
| 16 | `violated_facet="language"` while evidence quotes `norsk autorisasjon` | rejected | Protocol: authorisation is never a language violation |

### `parse_judgment(raw_row)` — tier 2 (`fixture`)

| # | Input | Expected | Rationale · grounding |
|---|---|---|---|
| 17 | recorded envelope with the judge tool | typed judgment | G3 |
| 18 | `stop_reason="max_tokens"` | `errored`, never salvaged | The client's existing owner ruling; a half-parsed judgment is indistinguishable from a real one |
| 19 | envelope whose `tool_use` name is `record_ad_facets` | `errored` | Wrong tool means wrong contract |

### `build_pool(arm_rankings, depth=10)`

| # | Input | Expected | Rationale · grounding |
|---|---|---|---|
| 20 | 5 arms, overlapping top-10s | union, de-duplicated | Protocol: TREC depth-10 pooling |
| 21 | same arms in a different order | identical pool | Blind to contributor; order must not leak |
| 22 | any | pool rows carry **no** arm identity | Protocol: "judged blind to which system contributed" |
| 23 | an arm returning fewer than 10 scored hits | pool takes only the scored ones | BM25 scores exactly 0.0 for no match; padding would judge noise |
| 24 | 13 dev personas, 5 arms | ~346 pairs | G6, measured |
| 25 | a **sealed** persona passed in | rejected | D16/D17; the seal is not broken by a pooling call |

---

## Not in scope here

The human calibration set (E5, 120 pairs, Cohen's κ) and the ablation runner
(E6). This step delivers the judge, its isolation test, and the pool. The
protocol's "one prompt revision, before human calibration, zero after" clock
starts when the first judgment is run, not when this is approved.
