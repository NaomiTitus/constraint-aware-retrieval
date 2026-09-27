# Decision log

Decisions that shape the system, with the reasoning and the evidence. Recorded so
a reviewer — or a future maintainer — can see what was chosen deliberately rather
than by default. Newest last.

---

### D1 · Corpus source: NAV licensed feed only
**2026-09-22**

Use `pam-stilling-feed.nav.no` under NAV's published terms. **No automated access
to finn.no of any kind.** finn.no/robots.txt requires written permission for
crawling under Norwegian copyright law and disallows `/job/` for GPTBot.

FINN-originated ads are excluded from the licensed feed — verified by requesting
`source:"FINN"` uuids from the feed and receiving HTTP 404 on all of them. Zero
FINN-sourced ads appear among the 10,166 fetched.

*Consequence:* the corpus is not FINN's index. The methods transfer; the
measurements are on the slice obtainable lawfully. Stated in the README above the
fold, not buried.

---

### D2 · Static frozen snapshot — no refresh
**2026-09-24**

The corpus is a single snapshot: 10,166 ads, 120-day window, fetched 2026-09-24.
No scheduled re-crawl, no incremental updates.

*Evidence:* the corpus decays fast — 37% of ads expire within 7 days, 59% within
14, 83% within 30. Median ad lifetime is 26 days; ~500 new ads publish per weekday.

*Reasoning:* the ablation compares eight ranking configurations over the same
documents. If the corpus shifts between runs the numbers stop being comparable and
the judged qrels pool shrinks. A frozen snapshot is the scientifically correct
choice, and it is what makes `reports/ablation.md` reproducible from the committed
cache.

*Consequences:*
- The feed cursor is for **crash resume within a run**, not weekly refresh.
  Scenario 2.6 is a within-session concern; no cursor-rotation check needed.
- No GitHub Actions refresh workflow.
- The index manifest needs no `supersedes` chain — versions are a set, not a line.
- The demo must **state the snapshot date prominently** and show expiry state per
  ad ("closed 14 Oct") rather than hiding it. Honest beats broken-looking.

---

### D3 · `english_accessible` is derived, never asked of the model
**2026-09-24**

Removed from the tool schema; computed in post from `norwegian_requirement_level`
+ `document_language`.

*Reasoning:* it is a deterministic function of fields the model already emits, it
has a ~7% base rate (rare positive classes get over-predicted), and it was the only
field where the model could contradict itself.

---

### D4 · Nine-level requirement scale, not four
**2026-09-24**

`certified · fluent · professional · conversational · desirable ·
scandinavian_accepted · either_norwegian_or_english · explicitly_not_required ·
unstated`

*Evidence:* the original four-value scale could express neither the disjunction
(*"behersker norsk eller engelsk"* — 304 ads, 274 not otherwise accessible) nor the
documentary certification gate (*norskprøve B2* — 654 ads). Both are categorically
distinct from "required", and the disjunction case would have been ranked **down**
rather than merely missed.

---

### D5 · LLM census over the whole corpus, not a sample
**2026-09-24**

Label all 10,166 (9,771 after clustering) with an LLM at ~$3. Human effort goes to
**auditing the labeller** (150 stratified ads), not producing labels.

*Reasoning:* at corpus-scale cost of $3, sampling to obtain labels is pointless.
The estimand changes from "model accuracy" to "labeller bias" — two-phase sampling
/ prediction-powered inference.

*Cut as a consequence:* Snorkel label model (labelling functions are correlated,
and the classes that matter have 0% LF coverage), active learning (incompatible
with an unbiased held-out set at this scale).

---

### D6 · Near-duplicate clustering on body only, via MinHash — SUPERSEDED by D8
**2026-09-24**

*Evidence:* a first attempt keyed on `title + first 400 chars` found **157**
redundant ads — worse than plain exact hashing at 320 — because chain stores vary
the title per location while the body is identical. MinHash over 5-word body
shingles finds 456 (4.5%), largest cluster 44.

*Purpose is error amplification, not cost.* Deduping saves $0.12. It prevents one
template mistake becoming 44 corpus rows that look like consistent signal and
survive random-sample evaluation.

*Not solved here:* whether 44 near-identical ads should appear on one results page.
That is a ranking concern (MMR / per-employer capping), deliberately deferred.

---

### D7 · No vector database by default — but benchmark one
**2026-09-24**

10,166 × 768 float32 = 30 MB; brute-force cosine is one numpy matmul, ~5 ms.
FAISS/Qdrant solve a problem this corpus does not have.

*But* ship a `DenseIndex` interface with numpy / FAISS / Qdrant behind one config
line, and measure the crossover: recall@10 vs exact, p50/p95, build time at
10k → 100k → 1M → 10M. A measured crossover curve is evidence; "a vector DB is
overkill" is an assertion.

Also measure filtered-search strategies (post / pre / in-engine) at the real ~7%
selectivity, where post-filtering should collapse.


---

### D8 · Exact hash of normalised text, superseding MinHash (D6)
**2026-09-24**

`signature(body) = sha256(normalise(body).lower() → digits masked → punctuation
collapsed)`. A 64-char hex digest, not a tuple of permutations.

*Evidence — four strategies over the full corpus:*

| Strategy | Redundant | Largest cluster |
|---|---:|---:|
| Raw HTML, exact | 249 (2.4%) | 28 |
| Cleaned text, exact | 320 (3.1%) | 44 |
| **Normalised text, exact** | **343 (3.4%)** | 44 |
| MinHash shingles | 423 (4.2%) | 44 |

MinHash's extra 78 merges were inspected and are legitimate (Kid Interiør across
three stores at 0.976 similarity; Adecco at 0.995). But they sit in clusters of
2–6, where error amplification is negligible, and the large clusters that
actually matter are caught identically by every strategy.

*Reasoning:* the purpose is error amplification, not cost — deduping saves $0.12
on a $3 census. Against a marginal 0.8% of the corpus, exact hashing buys:
explainable in one sentence, deterministic across processes with no seed to keep
in sync, serialisable cluster keys, and **zero false merges by construction**.

*Accepted loss:* ads differing by a sentence or so (the Adecco case, ~20 chars)
stay in separate clusters. Tested explicitly, so the tradeoff is visible rather
than forgotten.

*CEFR exception:* digits are masked so one template per location collapses, but
a digit directly preceded by A/B/C is left alone. `norskprøve B1` and `B2` are
DIFFERENT language requirements, and merging them would hand two ads one verdict
on the very attribute this pipeline exists to read. 1,268 corpus ads carry a
CEFR token; zero clusters currently differ by one, so the guard is latent — which
is exactly when it is cheapest to add.

*Kept from D6:* clustering is on the body only. Chain stores vary the title per
location, and a key including the title found only 157 redundant ads — worse
than plain exact hashing at 320.


---

### D9 · Batch API over REST, and the cost model corrected by measurement
**2026-09-24**

The client talks to `api.anthropic.com/v1/messages/batches` directly over httpx
rather than through the SDK: the installed SDK pulls an `httpx2` whose
decompressor signature mismatches this environment, and the Batch API is three
endpoints. Fewer dependencies, and the protocol is explicit in the source.

*Verified against the live API with a 3-request batch (~$0.003), recorded as
`tests/fixtures/batch_results_real.json`.* All three matched their golden
labels, including the disjunction case quoting "Behersker norsk eller engelsk"
— the 274-ad correction working end to end.

**The cost model was wrong and is now measured:**

| | Estimated | Measured |
|---|---:|---:|
| Cached prefix tokens | 3,413 | **9,217** |
| Output tokens | 200 | **~350** |
| Full census | ~$10.45 | **~$14.50** |

The estimate counted characters÷4 over the prompt text and ignored the tool
schema, the twelve tool_use/tool_result envelopes, and JSON structural
overhead. Still well inside the $50 ceiling, and still roughly a third of the
uncached cost — but measured rather than asserted.

*Truncation is not salvaged.* `stop_reason == "max_tokens"` maps to `errored`
and is retried by the existing path. A half-parsed facet set is
indistinguishable from a real one downstream.

---

## D10 — The connective decides accessibility, and it is read first

**Decision.** `norwegian_requirement_level` is chosen by reading the CONNECTIVE
in a language list before anything else. `eller`/`or` makes the languages
alternatives; `og`/`and` (and a bare comma list) makes them all requirements.
Only in a disjunction does English being present make an ad accessible.

**Why it needed deciding.** The pilot exposed the failure in both directions
across three prompt versions, on the same 44 ads:

| | v4 | v5 | v6 |
|---|---:|---:|---:|
| `either_norwegian_or_english` | 3/5 | 5/5 | 5/5 |
| `scandinavian_accepted` | 8/8 | 7/8 | 7/8 |
| `conversational` | 2/2 | 1/2 | 2/2 |
| hidden wrongly (of 13) | 3 | 2 | **1** |
| shown wrongly (of 31) | 0 | 2 | **1** |
| pooled | 84.1% | 81.8% | **88.6%** |

v4 under-read disjunctions: "Må beherske skandinavisk **eller** engelsk tale"
became `scandinavian_accepted`, hiding it from the English speaker who satisfies
it. The rule was already in the prompt and lost anyway, because the level is
NAMED `either_norwegian_or_english` and the ad never says "norsk", and because
the only Nordic few-shot mapped a Nordic sentence to `scandinavian_accepted` —
a stated rule losing to a demonstrated counter-example.

v5 fixed that and broke the opposite case, because "ask one question first: is
English in the accepted set?" sat above "OR is not AND" in a list headed *stop
at the first that matches*. "gjøre deg forstått på norsk **og** engelsk" became
accessible.

**Measured prevalence — the two classes are nearly the same size:**

| Shape | Ads | Correct reading |
|---|---:|---|
| `X eller engelsk` | 781 | accessible |
| `X og engelsk` | 863 | Norwegian/Scandinavian still required |
| Nordic-or-English with no "norsk eller engelsk" anywhere | 364 | accessible, and nothing else rescues them |

So v5 traded a 781-ad error for an 863-ad one. Neither direction is an edge
case, which is why the connective is read first rather than patched afterwards.

**Held-out validation (`probe_connective.py`).** The prompt had by then been
tuned against the same 44 ads four times, so those numbers stopped being an
unbiased estimate. The probe scores 40 ads the prompt has never seen, using the
connective itself as the oracle — no hand-labelling needed. Result after
auditing every disagreement: **disjunction 18/20, conjunction 20/20, 38/40.**

## D11 — Application-document language is a separate field, and the evidence is the probe's own false positives

**Decision.** `application_language` stays a distinct facet. A statement about
what language the APPLICATION may be written in is never evidence about the job.

**Why.** The probe's raw disjunction score was 11/20. Seven of the nine
disagreements were the ORACLE being wrong, and four of those seven were this:

> "Dokumentene må være på norsk/skandinavisk eller engelsk"
> "All dokumentasjon må foreligge på et skandinavisk språk eller engelsk"

Academic advertisements routinely accept applications in English while requiring
Norwegian for the work. In the UiA postdoc ad both sentences appear: the model
returned `scandinavian_accepted` with `working_language: norwegian`, quoting
"Arbeidsspråket ved Universitetet i Agder er norsk", and ignored the document
clause. The regex could not, and read the ad as English-accessible.

Three more were school SUBJECTS — "basisfagene (norsk, matematikk eller
engelsk)" — a list of things taught, not languages demanded.

**Why this matters beyond the probe.** The confusion is concentrated in exactly
the ads most likely to be English-friendly (universities, research institutes),
so a system that gets it wrong is wrong where it is most consequential. And it
is a *live risk for the planned BM25 channel*, which is a keyword surface and
will make precisely the mistake my regex made. Recorded here so the retrieval
work inherits the warning rather than rediscovering it.

One genuine model miss stands, unexplained away: `· Behersker norsk eller
engelsk` returned `unstated` with no spans — the canonical disjunction, and the
exact sentence few-shot 2 demonstrates.

## D12 — The LLM cache stores the raw answer as well as the validated one

**Decision.** `llm_cache` carries both `response` (validated, the only thing
ever served) and `response_raw` (the model's own answer). `revalidate_cache()`
re-judges stored raw answers under the current validator with no API call.

**Why.** The pilot demoted 55% of records on a **validator** bug — the span
checker destroyed block boundaries — not a prompt bug. The model's answers were
fine; only the judgement of them was wrong. Re-measuring should have been free,
but the cache held post-validation records keyed on ad text, so a plain re-run
served the demoted verdicts straight back and the fix cost a second API run.

The original post-validation-only design answered a real objection: caching the
raw answer ALONE would serve an unvalidated verdict and skip validation on every
rerun. Storing both satisfies both constraints — nothing unvalidated is served,
and a validator change costs nothing to re-evaluate.

At 44 ads the mistake cost $0.06. At 9,823 clusters it is $14.50, and a
validator bug is the likeliest reason to need a re-run. Migration is additive:
`ensure_schema()` ALTERs an existing table, and the 172 already-paid-for rows
were preserved.

## D13 — Span boundaries are decided by character CLASS, not by a list of glyphs

**Decision.** Walking outward from a candidate span, `_boundaries_ok` classifies
each character: a sentence terminator (`. ! ? : ;`) or a block edge (`\n`)
ACCEPTS; a letter or digit REJECTS; anything else is **list furniture** and is
skipped. No enumeration of bullet characters.

**How this was found — the measurement inverted twice.** A probe of ads stating
a disjunction as a bullet reported 12/22 "model accuracy" against 95% on the
general disjunction population, every miss returning `unstated` with no evidence
span. That reads unambiguously as the model failing to see the line.

It was not the model. **All 9 misses were DEMOTED**, 8 of them
`starts_mid_sentence`. The model had found and quoted every requirement
correctly; demotion reset the level to `unstated` and stripped the spans, which
is indistinguishable from a miss unless you look at `reasons`.

`BOUNDARY` enumerated `•` U+2022 but not `·` U+00B7 or `●` U+25CF. **The model
quotes a bullet's TEXT, not its glyph**, so the character immediately before the
span is the glyph — and it was not a boundary.

**Why the earlier corpus check missed it.** A prior measurement reported 0 false
rejects over 12,572 language-bearing blocks. It passed whole blocks *including*
the leading glyph, so the character before the span was always the newline. It
never reproduced what the model does. Re-measured with the glyph stripped — the
model's actual behaviour — the rate was the real one, and after the fix it is
**0 of 12,569**.

**Why a list was the wrong instrument.** Block-leading glyphs absent from the
old set, measured over the corpus:

| glyph | blocks | |
|---|---:|---|
| `·` U+00B7 middle dot | 1,930 | |
| `●` U+25CF black circle | 141 | |
| 📍 ✅ ✨ 👉 ⭐ 🤝 🔹 ✔ … | ~500 | emoji bullets |
| U+200B zero-width space | 106 | invisible |
| U+2060 word joiner | — | invisible, not whitespace, so unfolded |
| `` U+F0B7 | 29 | the Wingdings bullet Word emits into pasted ads |
| **total** | **2,998** | |

Nobody writes U+F0B7 into a hand-maintained list. `not ch.isalnum()` covers it
and everything else of its kind, and the guard against over-accepting is pinned
separately: a span preceded by a WORD is still `starts_mid_sentence`, which
matters because 863 ads say "norsk og engelsk" and "engelsk" lifted out of one
supports the opposite of its sentence.

**Result, and the first real use of D12.** Re-judging the affected ads cost
**$0.00 and made no API call** — `revalidate_cache` re-ran the fixed validator
over answers already paid for.

| | before | after |
|---|---:|---:|
| glyph subpopulation (all 22) | 12/22 · 55% | **21/22 · 95%** |
| general disjunction population | 19/20 · 95% | 19/20 · 95% |
| corpus false rejects / 12,569 blocks | 9+ | **0** |
| 44-ad pilot | 88.6% | 88.6% (no regression) |

The glyph subpopulation now matches the general population exactly, which is the
evidence that the subpopulation never had a problem of its own.

The single remaining miss is the validator being RIGHT: the model wrote
`passasjøres` where the ad says `passasjerers`, and `not_verbatim` rejected the
paraphrase. That is what the verbatim rule is for.

## D14 — Six hand-written patterns, zero model errors: why the LLM layer earns its place

Recorded because it is the strongest empirical argument in the project, and it
was not the argument originally planned.

Every "model error" chased in this module resolved into **my own pattern**:

| # | The pattern | What it did |
|---|---|---|
| 1 | `normalise()` collapsing `\n` in span matching | demoted 55% of the pilot; 26 of 28 rejections were legitimate block quotes |
| 2 | `BOUNDARY` enumerating `•` but not `·` `●` emoji U+F0B7 | demoted 9 of 22 glyph-bullet ads with correct evidence |
| 3 | corpus false-reject check passing blocks WITH their glyph | reported 0 false rejects while (2) was live |
| 4 | probe oracle matching document-language clauses | 4 of 9 "model errors" in probe run 1 |
| 5 | the same exclusion written `dokument`, missing English `documentation` | 1 of 2 "model errors" in probe run 2, in a ~7% English corpus |
| 6 | glyph filter selecting 20 of 289 affected ads | made a 289-ad population look like 26, nearly closing the investigation |

Against that, genuine model errors found across ~180 ads: the census-v4/v5
connective confusion (fixed, and it was a *prompt* defect, not a capability
one), one dropped disjunction, and one paraphrased quote — which the verbatim
check caught.

**Two consequences.**

*For the thesis.* The project argues that constraints are predicates over
metadata and cannot be expressed as similarity. The sharper version, evidenced
above, is that they cannot reliably be expressed as **surface patterns** either:
a language requirement in this corpus is stated in bokmål and nynorsk, in
English, as a bare bullet, behind a Wingdings glyph, inside a disjunction whose
other half may be Scandinavian or Polish, and next to a clause about what
language your CV may be in. Six careful attempts to pattern-match it failed.

*For the retrieval work.* The planned BM25 channel is a surface matcher and will
make mistakes 4 and 5 by construction — it cannot distinguish "the job needs
Norwegian" from "your diploma may be in English". That is an argument for the
constraint stage reading `ad_facets` rather than text, and it is now measured
rather than asserted.

## D15 — Golden #15 is `certified`; and the harness now reads the golden FILE

**Owner ruling, 2026-09-25.** #15 (*Jurist / saksbehandler / advokatfullmektig*)
requires Norwegian at B2 — "Svært god norsk muntlig og skriftlig (B2 eller
høyere)" — so Norwegian IS required. In this taxonomy a CEFR level is a
documentary gate, which is `certified` rather than a bare "required"; there is no
bare `required` level, because the scale splits by proficiency bar.

This supersedes the 2026-09-24 ruling of `explicitly_not_required`, which rested
on "Er du polsktallende jurist men norskkunnskapene er under utvikling? Søk!".
That line invites Polish speakers whose Norwegian is developing; it sits beside
the B2 requirement rather than cancelling it. The revised reading weights the
stated requirement over the invitation.

Accessibility now follows from the level — `certified` is blocking — so
`derived_accessible` is False and the hand annotation is gone.

**Effect on the pilot, stated plainly:** pooled 88.6% → 90.9% and hidden wrongly
1 → 0 of 12. **That gain is a relabelling, not a model improvement.** The census
output did not change; the same prediction (`certified`) is now scored correct.
Recorded here so no later reader mistakes 0/12 for extractor progress.

**The taxonomy gap now has zero instances.** #15 was the project's only example
of "a correct level that a human judged inaccessible". The underlying limitation
is unchanged — `explicitly_not_required` derives to accessible whether or not
English is mentioned, so an ad saying "no Norwegian needed, we work in Polish"
would still derive True — but no golden ad illustrates it any more, and
`taxonomy_gap` is a forward guard rather than a live measurement.

**The harness bug, which is the more general lesson.** `scoring.py` reads
`annotated_accessible`; the golden set recorded the override as prose in
`accessibility_note`. **0 of 44 ads carried the key scoring reads, so
`taxonomy_gap` was unreachable for the whole project** — a documented headline
number that measured nothing.

`test_9_1` passed throughout. It builds its own golden ad and sets
`annotated_accessible` on it, so it exercised the mechanism and agreed with
itself. Identical shape to the span tests that used single-line prose and the
corpus false-reject check that passed blocks with their bullet glyph attached:
**the test constructed the input, so it could not detect that real inputs never
reach the code.**

Four tests now read `eval/golden_set.json` itself:

| test | what it pins |
|---|---|
| closed key vocabulary | ROOT CAUSE — a typo'd field name now fails a test instead of silently disabling a metric |
| prose override ⇒ boolean | an `accessibility_note` without `annotated_accessible` is invisible to scoring |
| stored `derived_accessible` == `derive()` | 44 documentation fields that no test read, free to drift from the code they document |
| `taxonomy_gap` runs on the real file | the check executes against real data, and every uuid it lists genuinely carries a disagreeing override |

The closed vocabulary is the only one of the four that generalises: the bug was a
key name nothing validated, and that class of bug recurs wherever data files
carry fields by convention.

## D16 — The dev/sealed persona split, pre-registered before any retrieval ran

**Date: 2026-09-27. The git timestamp on this commit is the pre-registration.**
No retrieval ablation, no relevance judgment and no gold persona parse existed
when this was written, which is the only condition under which the split is worth
anything.

### Why it had to be decided first

`PLAN.md` E1 pre-registered "10 dev, 6 SEALED" and `STANDARDS.md` says sealed
personas stay sealed until the final run — but **no split existed anywhere in the
repo.** All 19 personas sat in one undifferentiated list. Left alone, the first
retrieval eval would have consumed every persona, leaving nothing held out and
voiding E1 silently.

The split is also the one artefact here that **decays with time**. A schema is as
good written tomorrow; a held-out set chosen after seeing which personas embarrass
the system is contaminated no matter how honest the chooser. So it goes first.

### E1's arithmetic was stale, and reconciling it is the decision

E1 says **16 personas across 10 verticals**. The file holds **19 across 11** — the
original 15 plus the four controls added in `4925087`, after E1 was written.

**The sealed COUNT is kept at 6, as pre-registered.** Raising it would be a
post-hoc change to a pre-registered quantity; lowering it weakens the held-out
measurement. Dev absorbs the growth and becomes 13. The sealed share therefore
falls from 6/16 = 37.5% to 6/19 = 31.6%, and that is a consequence of honouring
the pre-registered number rather than a new choice.

### Two structural constraints removed most of the freedom

**Pairs move as units of two.** `p1`–`p5` exist for the within-pair comparison —
the variants differ only in the language sentence, which is what makes the
negation stress-test controlled rather than two unrelated queries. Splitting a
pair across dev and sealed destroys the instrument rather than holding anything
back. So the sealed set is assembled from blocks of 2, 1 and 1, not from 19 free
choices.

**The controls' non-effect needs coverage on both sides.** `expect_language_constraint:
false` asserts that a seeker who says nothing about language gets a bit-identical
ranking. `tests/unit/test_constraints_only_when_stated.py` already guards that for
`constraints.py` in isolation, but end-to-end leakage through parser, fusion and
scorer is a different failure surface. So three controls sit in dev, where the bug
would be caught while it is cheap, and one sits in sealed, so the property is
verified on held-out data too.

### The split

**SEALED — 6 instances, 5 verticals.** Opened once, at the end.

| persona | vertical | why sealed |
|---|---|---|
| `p3_kokk_norsk` + `p3_kokk_no_norsk` | Hospitality | The paired negation stress-test on held-out data (E7). Chosen over `p4` — the easiest case, which cannot discriminate — and over `p5`, which is needed in dev as the reference failure. Hospitality is genuinely English-friendly, so a system that hides everything when Norwegian is absent fails here *visibly* rather than arguably. |
| `p8_regnskapsforer` | Finance | Authorisation-versus-language in a regulation-bound vertical. The distinction is developed against `c2` and `p1` in dev, so sealing this costs no iteration. |
| `p9_multilingual_support` | Customer service | The one persona where lacking Norwegian is an **asset**. A system that only ever penalises its absence gets this backwards — a discriminating property precisely because it must not be tuned on. |
| `p10_civil_engineer` | Engineering | The bilingual large-employer ad regime, covered nowhere in dev. If correct bilingual handling does not fall out of general correctness, that is exactly what a held-out set should catch. |
| `c3_utvikler_remote_seniority` | Technology | The control on held-out data, and the only persona exercising `remote` + `seniority` + `skills`. Technology is the thinnest vertical (240 ads), where a leaking language stage shows first. |

**DEV — 13 instances, 6 verticals.** `p1`, `p2`, `p4`, `p5` pairs; `p6`, `p7`;
`c1`, `c2`, `c4`.

Four kept in dev deliberately, each for a stated reason:

- **`p5` — the originating bug.** The query that started the project. It is the
  reference failure and development has to be able to iterate against it.
- **`p6` — the designed negative control.** Norwegian genuinely *is* required to
  teach in a Norwegian primary school, so a good system returns few results *and
  says so*. Calibrating "says so" is iterative work, not an end-of-project
  discovery.
- **`c1`, `c2`, `c4` — the non-effect guards.** `c1` is the majority case; `c2` is
  authorisation in isolation, the `norsk autorisasjon` trap named on day one;
  `c4` is a fluent speaker who never mentions it. These must work *during*
  development, not be verified after shipping.

### A consequence to state rather than discover

**Sealed verticals are disjoint from dev verticals.** That was not forced — it
fell out of stratifying on difficulty and facet coverage — and it is kept because
it makes the sealed run a test of generalisation to unseen verticals, which is
what a real product meets. The cost is that the sealed number is **not a
like-for-like replicate of dev**: it may be lower for reasons unrelated to the
architecture, and must be reported as a lower bound rather than as the headline.

### What this does not cover

The sealed set has no "Norwegian genuinely required, return few results" case —
`p6` holds that in dev, and `p8` covers it only partially through regulation. If
the final report wants that property measured held-out, it needs a new persona
written now, before any results exist, not a reassignment later.

## D17 — Amendment to D16: a seventh sealed persona, added before any results existed

**Date: 2026-09-27, hours after D16 and still with zero retrieval results, zero
relevance judgments and zero gold persona parses in the repo.** This is an
amendment to a quantity D16 pre-registered, and it is filed as one rather than
folded into D16, because the audit trail is the only thing that makes the original
pre-registration worth anything.

### What D16 left broken, in its own words

D16 closed with: *"The sealed set has no 'Norwegian genuinely required, return few
results' case — `p6` holds that in dev... If the final report wants that property
measured held-out, it needs a new persona written now, before any results exist,
not a reassignment later."*

That is the condition being met. The fix is a **new** persona, not a reassignment
of an existing one — reassignment would let knowledge of the dev set leak into the
choice of what to hold back, which is exactly the contamination D16 was written to
prevent.

### Why the property is worth a sealed slot

`p6_grunnskolelaerer` encodes a behaviour that is easy to get backwards: when
Norwegian genuinely IS required, the right answer is **few results or none, said
plainly**. A system that manufactures plausible matches there is *worse* than one
that returns nothing, because it spends the seeker's applications on jobs they
cannot lawfully hold. Measuring that only on a persona the system was tuned
against would tell us nothing about whether the behaviour generalises.

### `p11_barnevernspedagog` — Social services

Chosen against corpus evidence rather than intuition:

| | |
|---|---:|
| ads in the vertical | **308** |
| explicitly demanding Norwegian | 219 — 71% |
| census calls accessible | **0** |

308 ads is the point. A negative control in a vertical with four ads measures the
corpus, not the system — "returned nothing" would be trivially correct. At 308,
with zero accessible, "returns few and says so" is a real behaviour with a real
opportunity to fail. And the requirement is legitimate rather than incidental:
statutory child-welfare work is Norwegian-language client contact, case
documentation and court reporting.

**Vertical disjointness is preserved.** Social services appears in neither the dev
set (Data, Education, Healthcare, Logistics, Software, Trades) nor the existing
sealed set (Customer service, Engineering, Finance, Hospitality, Technology), so
D16's "sealed measures generalisation, read it as a lower bound" framing still
holds, and `test_sealed_verticals_are_disjoint_from_dev` still passes.

**Why sealed and not dev.** `p6` already carries this property in dev, which is
where the "and say so" wording gets calibrated. `p11` asks whether that
calibration generalises to a vertical never developed against — the one form of
the question no dev persona can answer.

### The counts, amended

| | pre-registered in D16 | amended |
|---|---:|---:|
| sealed | 6 | **7** |
| dev | 13 | 13 |
| total instances | 19 | **20** |
| sealed share | 31.6% | 35.0% |

Dev is untouched, so nothing the system will be tuned on changed. The sealed share
moves back toward E1's original 37.5%.

### The rule this establishes for any future amendment

Amending a pre-registration is legitimate **only** while no result exists that
could motivate the amendment, and **only** when recorded as an amendment with its
date and reason. Once the first CVR@10 is computed, the sealed set is frozen: after
that point adding, removing or reassigning a sealed persona invalidates the held-out
measurement whatever the justification, and the honest move is to report the
limitation instead. `expect_few_results` is now a machine-checked field
(`tests/unit/test_persona_split.py`) so this property cannot silently disappear
from either side of the split.

## D18 — Occupation is a predicate over metadata, not a BM25 term

**Date: 2026-09-27, decided from the D1 probe on the real corpus.** No relevance
judgments exist, so this rests on structural measurement only, and it is recorded
as a design decision rather than a quality result.

### What D1 does well, and where it inverts

The lexical channel works on Norwegian queries. `p1_sykepleier_norsk` returns
`sykepleier` for 10 of 10; `p2_tomrer_norsk` returns `tømrer` for 5 of 5;
`c4_laerer_norsk_speaker` returns `grunnskolelærer` for 5 of 5. Index builds in
~30s over 10,166 ads, queries in 6ms.

**English queries invert.** `p1_sykepleier_no_norsk` — the same seeker, same
occupation, language sentence swapped — returned a psychiatrist, a Norwegian
teacher, a caretaker, a sales rep and a bricklayer. No nurses.

### Why, measured

In a 95%-Norwegian corpus, ordinary English words are RARE and therefore score as
informative:

| term | df | idf |
|---|---:|---:|
| `i` | 94.4% | 0.06 |
| `for` | 95.7% | 0.04 |
| `shifts` | 0.3% | **5.84** |
| `preferably` | 0.3% | **5.75** |
| `permanent` | 0.6% | **5.03** |
| **`nurse`** | 13.0% | **2.04** |

IDF suppresses the *majority* language's function words and inflates the
minority's. The only content-bearing term in the query was the weakest in it, and
any long English-written ad accumulated the rest.

### Two fixes applied, and their measured worth

**A bilingual stopword list**, which `bm25.py` originally argued against on the
grounds that IDF would handle function words. That reasoning is false for a mixed
corpus and the docstring now carries the refutation. Boilerplate that maps to a
structured field is stopped too — `permanent`, `shift`, `experience` — because
`ads.engagementtype` (99.9%), `ads.extent` (100%) and
`ad_facets.min_years_experience` compare them properly instead of as bag-of-words
evidence competing with the occupation.

**Indexing `ad_taxonomy.job_title_en`**, the bilingual bridge A4's ESCO work
already built, populated for 100% of the corpus.

Mean paired overlap@10 across p1/p2/p4/p5, where the two variants differ only in
the language sentence:

| | mean overlap@10 |
|---|---:|
| raw query, title only, no stopwords | 0.8/10 |
| + bilingual stopwords | 2.0/10 |
| + `job_title_en` indexed | 1.8/10 |
| + querying the GOLD PARSE instead of prose | **2.5/10** |

Both fixes help and neither is sufficient. `p1` and `p2` remain at 0/10.

### The decision

**Occupation must be applied as a predicate against `ad_taxonomy`, not as an
IDF-weighted term.** The failure is not a tuning problem, and one more lexical
trick will not reach it: the taxonomy bridge *worked* — `nurse` matches 1,320 ads
— and it made things no better, because a term matching 1,320 ads necessarily has
low IDF while an incidental `hospital` matching 54 has high IDF. BM25 is being
asked to treat a **hard constraint** as graded similarity evidence, and it weights
it by rarity, which is the wrong ordering by construction.

This is the project's own thesis arriving a second time from a different
direction. Containment is not similarity: an occupation requirement is a predicate
over metadata, `ad_taxonomy.job_title_standardised` and `job_title_en` are exactly
that metadata at 100% coverage, and the seeker's `occupation` constraint is already
typed `hard` in every gold parse. Expressing it lexically is the same category
error as expressing negation in embedding space.

**So D7's `occupation_proximity` is promoted from a ranking feature to the
mechanism that carries occupation**, with BM25 scoring only the free-text residue —
skills and context — where graded lexical evidence is the right model. D3's graph
channel supplies proximity for near-miss occupations (`spesialsykepleier` for a
`sykepleier` query), which is what the ESCO `styrk_code` and `role_family` columns
are for.

### What this does not settle

Whether the resulting ranking is GOOD. `eval/JUDGING_PROTOCOL.md` is pre-registered
with zero pairs judged; everything above is structural — occupation labels read from
`ad_taxonomy`, which is derived independently of BM25, so the check is not circular,
but it is not a relevance measurement either.

### D18 — measured outcome, 2026-09-27

Built and measured the same day. **The predicate works; it does not fix what I
predicted it would fix, and the remaining error is the input layer.**

**What improved.** Share of the top-5 whose STYRK code is within one unit group of
the seeker's occupation, read off `ad_taxonomy` and therefore independent of both
BM25 and this module:

| | occupation precision@5 |
|---|---:|
| BM25 on raw query prose | 48% |
| BM25 on gold-parse lexical terms | 66% |
| + occupation predicate | **70%** |

`p6_grunnskolelaerer` went 1/5 → 5/5, `p4_backend_no_norsk` 2/5 → 5/5,
`p2_tomrer_no_norsk` 3/5 → 5/5. Most of the first jump comes from querying the
parse rather than the prose, which is the same finding D18 recorded from the other
side: the boilerplate was the problem, and removing it beats reweighting it.

**What did NOT improve, against the stated prediction.** Paired overlap@10 — the
number D18 predicted would rise, because both language variants resolve to the same
code:

| | mean paired overlap@10 |
|---|---:|
| raw query | 1.8/10 |
| gold parse | **2.5/10** |
| + occupation predicate | 2.2/10 |

**It went down.** The prediction failed and the reason is instructive rather than
fatal.

**Cause 1: asymmetric resolution makes a pair diverge MORE.** `backend developer`
resolves to 2512; `backend-utvikler` does not resolve at all. The fail-open rule
then leaves the Norwegian variant untouched while filtering the English one, so
`p4` fell from 8/10 to 4/10. Fail-open is still right — zeroing every ad on a
gazetteer miss would report a lookup gap as "no jobs exist" — but **a predicate
applied to one side of a controlled pair and not the other is a confound**, and any
future paired measurement has to report resolution status alongside it.

**Cause 2: the gazetteer misresolves the most important phrase.** `nurse` resolves
to **5321 `pleiemedhjelper`** (care assistant) at 65%, with the correct 2223
`sykepleier` at only 8%. ESCO's English label for `sykepleier` is `nurse
responsible for general care` — four content tokens — so symmetric Jaccard against
the one-token phrase `nurse` scores 0.25 and falls below threshold, while a short
label like `auxiliary nurse` scores 0.5 and wins. **Jaccard penalises the correct
label for being specific.** Containment-weighted scoring (`|shared| / |phrase|`)
is the obvious fix and is not attempted here.

**Cause 3: 3 of 13 dev occupations do not resolve at all** — `lager`,
`backend-utvikler`, `warehouse work`.

**Where this leaves it.** All three causes are seeker-side: mapping a person's words
onto a corpus code. That is `PLAN.md` D6, the query parser, and its gazetteer fast
path — the input layer. The ad side is fine (100% coverage, hierarchical, derived
independently); the predicate mechanism is fine (28 unit tests, exact/adjacent/
unrelated all behave); what is weak is the lookup from prose to code.

So the measurement redirects the work rather than endorsing more of it: **the next
gain is in the input layer, not in retrieval.** That is also what the ceiling said
from a different angle (§ gold parse coverage: 54% population-weighted), and what
D18's own premise said — extraction reliability is the binding constraint.

**Not settled.** Whether any of this ranking is GOOD. Zero relevance pairs judged.
`occupation precision@5` measures agreement with a taxonomy label, not relevance to
a person, and 5/10 of the top-10 for the no-Norwegian nurse still DEMAND Norwegian —
which occupation proximity should not fix, and D5 exists to.

## D19 — Free-text query is the entry point; CV upload is measurement, not product

**Date: 2026-09-27, before any input-layer code was written.**

### The decision

The product takes a **free-text query** — *"I am looking for nursing positions in
Oslo, I have six years of experience"*. CV upload is not a second entry point for
the product. It is retained for one purpose only: `LIMITATIONS.md` §1b commits to
running the manual baseline against FINN's live feature, and that feature is now
CV-upload matching. So a CV parser is owed to the **evaluation**, not to the search
surface, and it is scoped accordingly.

### Why free text, and the third reason is the one that decides it

**1. Everything already built assumes it.** All 20 personas are free-text queries,
all 13 dev gold parses derive from them, and `eval/JUDGING_PROTOCOL.md` — committed
before any ablation — grades relevance "against the persona's **stated** skills and
preferences". Switching the entry point orphans the personas, the parses and the
protocol at once.

**2. §1b already recorded that a CV is harder for this thesis, not easier.** A
query can say *"I do not speak Norwegian."* A CV cannot: it lists what you have,
never what you lack. The constraint becomes an ABSENCE — Norwegian missing from a
languages list — and §14 then applies with full force, because absence is expressed
by silence on both sides and there is no vector and no predicate for text that was
never written.

**3. A CV cannot carry the hard constraints, which are the entire subject.** The
containment model needs **R ⊆ S**. A CV populates the *soft* half of S well —
occupation, skills, years — and the *hard* half not at all. Measured on the 13 dev
gold parses, the hard constraints are:

| hard constraint | in a CV? |
|---|---|
| `occupation` (13) | yes |
| `language.norwegian` (10) | only as a positive list, never as a limit |
| `location.place` (5) | **no** — a CV states where you have worked, not where you will |
| `credential.licence` (5) | usually |
| `credential.trade_certificate` (2) | usually |
| `credential.authorisation` (2) | usually |

Location and language — 15 of the 37 hard constraints — are **preferences about the
future**, and a CV is a record of the past. No parser recovers them from it. So a
CV-only surface cannot express the constraint this project exists to respect, and
would need a query bolted on regardless.

**4. Cost.** PDF and DOCX parsing, layout recovery, PII handling, and extraction
from two to four pages of career history rather than one paragraph — before any of
it earns a single point of the metric.

### What this commits the input layer to

`PLAN.md` D6 stands as written — gazetteer fast path, then distilled classifier,
then LLM residue — with the gold parses as its test set and
`eval/GOLD_PARSE_SCHEMA.md` as its output contract. The parser's job is to produce
the same typed **S** the gold parses hold, from prose.

### What would reverse this

If the eval shows seekers cannot state their constraints in prose well enough for
the gazetteer to resolve — resolution is at 9 of 13 today — the answer is a better
parser or a guided form, **not** a CV. A CV would lose the constraints rather than
capture them better, which is the wrong direction on the axis that matters.
