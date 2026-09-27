# Smart Search — constraint-aware job search over Norwegian job ads

**Job search that treats what you *lack* as a fact about the world, not as words to
match.** Type a sentence in English or Norwegian. The constraints in it — the language a job
demands, where it is, the experience it expects — are extracted as typed metadata and
evaluated as **predicates**, so a job you cannot take ranks as one you cannot take. Every
result shows its own reasoning: what the parser understood, the occupation code it resolved
to, which skills matched, and why it was penalised.

**[▶ Live demo](https://naomititus.github.io/finnno_smart_search/)** · **[What these numbers do
not support](LIMITATIONS.md)** · [Architecture](ARCHITECTURE.md) · [Decision log](DECISIONS.md)

> **Scope.** One market: the NAV/arbeidsplassen licensed feed, 10,166 currently-active ads in
> a 120-day window (FINN ads are excluded from that feed by licence). **Four of the fifteen
> extracted facets reach the ranking** — language level, working language, evidence spans and
> authorisation — plus derived accessibility and three fields from outside the census
> (location, occupation, skills). **The other eleven are extracted, validated and stored but
> scored by nothing** (§4, §11). No CV upload, no personalisation, no
> learning-to-rank, no incremental re-crawl. **The shipped page's own ranking has not been
> judged** — every confidence interval below comes from the Python pipeline (§22).

---

## Headline

| | | |
|---|---:|---|
| **Language extraction** | **96.4%** (27/28) | sealed held-out set, never seen during prompt development · **0** accessible ads hidden |
| **Constraint violations @10** | **halved**, 0.154 → 0.077 | 378 relevance judgments; 95% CI excludes zero |
| **Query latency** | **166 ms** p50 · 198 ms p95 | full scan + rank of all 10,166 ads, no server · `scripts/bench_page.mjs` |
| **Time to interactive** | ~450 ms after download | 2.9 MB gzipped index · 100 ms parse · 343 ms index build |
| **Infrastructure at query time** | **none** | static GitHub Pages; no API call, no vector DB, no backend |
| **Cost** | **$0.0045**/ad → **~$54/month** | $46.15 ÷ 10,166 ads, at ~388 new ads/day. Plus $8.30 R&D once — split out below |
| **Extraction models** | `claude-haiku-4-5` | both censuses — 9,379 + 9,823 calls, 0 failures |
| **Judge + label model** | `claude-opus-5` | judging, and the golden skill labels haiku is scored against |
| **Throughput** | 9,823 ads in **10m13s** | Batch API, single submission |

**Why two models and not one.** The same skills prompt was piloted on both: Opus recall
**0.868**, Haiku recall **0.861** — inside noise — at **$130 vs $23** for the full corpus.
So extraction runs on Haiku and the money goes where a weak model cannot be repaired by
re-prompting: the **judge**, which decides whether any of this works, and the **golden
labels** Haiku is measured against. Scaling to a million ads is ~$4.5k of Haiku against ~$23k of Opus,
and the judging cost does not scale with the corpus at all.


### What it costs to run, separated properly

Three different kinds of money, and conflating them is how a pilot budget gets
mis-sold. Every figure below is **actual token usage pulled from the Batch API**, not an
estimate — my own estimates were out by 38% in one direction and 15% in the other.

**① Ongoing — the only cost that recurs.** Each new advertisement is tagged once:

| | total spend | per advertisement | per 1,000 ads |
|---|---:|---:|---:|
| facet census (`census-v15`, 15 typed facets) | $22.59 | $0.00222 | $2.22 |
| skills census (`skills-v1`, 6.53 skills/ad) | $23.56 | $0.00232 | $2.32 |
| **total taxonomy cost** | **$46.15** | **$0.00454** | **$4.54** |

All four figures are spend ÷ 10,166 advertisements. Per *API call* they are higher —
deduplication by body hash collapses the corpus to 9,379 and 9,823 calls — but per-ad is the
number that scales with a market.

Arrival rate measured from the corpus: **~388 new ads/day** over the two weeks before the
crawl, so **~11,800/month ≈ $54/month, $645/year** to keep the whole Norwegian market tagged.
Query serving adds nothing — the front end is static and calls no API.

**The obvious cost lever is not available yet, and saying so is the point.** The facet census
ran at a **97% prompt-cache hit rate** — *"the number that sets the bill"* — while the skills
census ran at **0%**. The static instruction block is ~62% of each skills call, so caching it
looks like an easy win. But the prompt renders the **advertisement first and the instructions
after it**: two rendered prompts share only ~98 characters, about 24 tokens, far below Haiku's
2,048-token minimum cacheable prefix. `cache_control` as the prompt stands today would cache
**nothing**. Realising it means reordering the prompt and re-validating the extraction — a
real change, not a flag flip. Deduplication already removes 3.4% of calls.

**② One-off R&D — $8.30, and it does not scale with the corpus.**

| | |
|---:|---|
| $2.42 | 18 prompt-development runs, 1,036 calls — fifteen census prompt versions |
| $1.33 | skills pilots on the 44 golden ads: Haiku $0.20, Opus $1.13 |
| $4.55 | **378 relevance judgments** across three rounds (`claude-opus-5`) |

This is the number that surprises people: **proving the thing works cost less than
$9.** It is fixed — judging 378 pairs costs the same whether the corpus is 10,000 ads
or 10 million. It excludes my own time, which dominated.

**③ One-off backfill — $46.15** to tag the existing 10,166 ads ($22.59 facets + $23.56
skills). This scales with corpus size, not with time, and it is the figure to quote for a
market you have not indexed yet: **~$4,500 per million advertisements.** Opus is 5× Haiku per
token, so the same backfill there is **~$230 per 10,166 ads ≈ $23,000 per million** — for
+0.007 recall. (The $130 figure in §18 is the *skills* half alone; the full backfill is both
censuses.)

---

## The problem: matching is containment, and similarity is symmetric

A job states the requirements it has — a set **R**. A seeker has attributes **S**. The job is
viable iff **R ⊆ S**, and among viable jobs the best match most of what the seeker asked for.
Containment is **asymmetric**, and that asymmetry is the whole difficulty:

| the job requires | seeker falls short | seeker exceeds it |
|---|---|---|
| Norwegian at professional level | fatal | free |
| a place — Bergen | another county is a *near-miss*, not a miss | free |
| 5 years' experience | disqualifying | free |
| a security clearance | fatal | free |

Cosine similarity is **symmetric**: it cannot tell "needs Norwegian, seeker has none" from
"seeker has Norwegian, job never asked". So no encoder, at any size, ranks on containment.
The fix is architectural — extract typed requirements from **both sides** and evaluate them
as **predicates over metadata**, in a separate stage with a dial on it.

**Language is the sharpest case,** and it was measured before it was asserted. 956 of 10,166
ads (9.4%) are accessible to an English speaker, so the constraint decides almost the entire
corpus. Worse, embeddings get it backwards: *"jeg snakker ikke norsk"* and *"jeg snakker
flytende norsk"* collapse to **cosine 0.914**, and stating the constraint moved the seeker
**closer** to the ads demanding Norwegian on **5 of 5** personas. Declaring what you lack
made results worse.

**Location has the same shape but a different grade.** It is metadata, not a word in a bag —
99.1% of ads carry a municipality — and proximity is not binary: a neighbouring municipality
in the same county is a commutable near-miss, so it scores 0.55 rather than 0. 7.1% of ads
list several places, so an ad is scored on its *best* one. Treating `bergen` as a search term
instead let a Bergen ad in the wrong occupation outrank a nurse ad elsewhere.

**And absence is expressed by silence, which has no vector.** An ad states what it requires
and never enumerates what it does not: 36.5% say nothing about language, 99.4% nothing about
visa sponsorship, and only **93 ads (0.9%)** carry the label `explicitly_not_required` — and
§14 retracts even that: 82 of the 93 contain no explicit negation, the census inferred it from
a positive English working language. The accessible
ads affirm *English* rather than negating Norwegian — which is exactly why a query containing
`norsk` drifts toward the ads it rules out. **This holds for every facet, not just language.**

*Negation was the frame this project started from and it turned out to be the wrong one: a
control showed a one-word **content** swap (nurse vs carpenter) is erased at the same rate as
a negation, so the real limit is that single-vector retrieval loses any single clause once
queries are long. Both disproved hypotheses are in
[LIMITATIONS.md](LIMITATIONS.md) §12–§14.*

---

## Architecture

**Budgets are asymmetric**, and everything follows from that: minutes and fractions of a cent
per ad at **index** time, against <300 ms and effectively free at **query** time. So every
expensive semantic operation is pushed to index time and the query path is arithmetic over
dense arrays — the LLM's best use here is not inference, it is **label generation**. Full
storage and latency argument in [ARCHITECTURE.md](ARCHITECTURE.md); numbered design
decisions in [DECISIONS.md](DECISIONS.md).

### Build time — how the index is made

There is no gradient descent here. What stands in for "training" is a **typed extraction
pass** over every advertisement, validated against human-authored scenario tables and a
sealed held-out set, then joined to two public taxonomies and compiled to a static index.

```mermaid
flowchart TB
  subgraph ETL["1 · ETL — bronze to gold"]
    A["NAV/arbeidsplassen<br/>licensed feed"] -->|"220,988 listing entries<br/>3m45s, resumable"| B["feed_entries"]
    B -->|"dedupe by uuid"| C["59,830 distinct ads"]
    C -->|"120-day active window"| D["ads_raw<br/>10,166 full bodies"]
    D -->|"html_clean · text_norm<br/>pii scrub"| E[("DuckDB<br/>ads · ad_locations<br/>ad_categories")]
  end

  subgraph TAX["2 · Taxonomies"]
    F["ESCO API"] -->|"bilingual no+en"| G[("1,242 occupations<br/>10,063 skills<br/>52,009 occ→skill edges")]
    H["STYRK-08<br/>hierarchical codes"] --> I[("ad_taxonomy<br/>100% coverage")]
  end

  subgraph EXT["3 · Extraction — the 'training' pass"]
    E -->|"dedup.signature<br/>exact body hash"| J["9,823 clusters<br/>from 10,166 ads"]
    J -->|"census-v15 · haiku-4-5<br/>97% prompt-cache hit<br/>$22.59 · 21 min"| K["15 typed facets<br/>language · contract<br/>location · authorisation"]
    J -->|"skills-v1 · haiku-4-5<br/>$23.56 · 10m13s"| L["66,392 skills<br/>phrase + gloss_en + level"]
    K -->|"census_validate<br/>2.3% demoted, 0 failures"| M[("ad_facets<br/>ad_language")]
    L -->|"verbatim validation<br/>6.3% REJECTED, not stored"| L2["reports/skills_census.json<br/>NOT merged into ad_facets:<br/>a mixed row would carry<br/>neither version honestly"]
  end

  subgraph VAL["4 · Validation"]
    M --> N["44 golden ads<br/>labelled by claude-opus-5"]
    N --> O["28 SEALED ads<br/>never seen in prompt dev"]
    O -->|"96.4% · 0 hidden wrongly"| P["gate passed"]
  end

  M --> Q["export_web.py<br/>DERIVED FIELDS ONLY"]
  L2 --> Q
  P -.->|"gates the export"| Q
  G --> Q
  I --> Q
  Q --> R["docs/data/index.json<br/>2.9 MB gzipped"]

  style EXT fill:#fff4e6
  style VAL fill:#e8f5e9
```

The **sealed set** is the honesty mechanism: 28 advertisements withheld from every round of
prompt development, scored once. Prompt iteration happened on the other 16 — the alternative
is tuning until the numbers look good and calling that a result.

### Query time — what happens when you type

Five stages. **Three of them fail open**: an unstated *or* unresolvable constraint leaves
the ranking bit-identical, so a gazetteer miss can never quietly change a little.

```mermaid
flowchart LR
  Q["'I am a nurse in Bergen.<br/>I do not speak norwegian'"] --> P["parse<br/>client-side"]

  P --> P1["terms<br/>bilingual stopwords"]
  P --> P2["bigrams<br/>adjacent pairs"]
  P --> P3["occupation<br/>→ ESCO label → STYRK"]
  P --> P4["place<br/>→ municipality/county"]
  P --> P5["norwegian level<br/>proximity negation, 42 ch"]

  P1 --> S1["① lexical base<br/>IDF over title/occ/skills<br/>field-weighted 1.0 / 0.55"]
  P2 --> S1
  P3 --> S2["② occupation predicate<br/>STYRK prefix proximity<br/>weighted OR bonus"]
  S1 --> S2
  S2 --> S3["③ location predicate<br/>1.0 / 0.55 / 0<br/>best of ad's locations"]
  P4 --> S3
  S3 --> S4["④ language constraint<br/>severity × discount × λ"]
  P5 --> S4
  S4 --> S5["⑤ accessibility boost<br/>×1.22 if English-accessible"]
  S5 --> R["ranked top 10<br/>+ badge + matched terms"]

  style S4 fill:#ffe0e0
  style S5 fill:#e0f0ff
```

**Where this architecture stops.** It works because the corpus fits in a browser: 10,166 ads
is 2.9 MB gzipped and a full scan is 166 ms. At ~100k ads the index needs sharding by region
or occupation and the linear scan becomes an inverted index; past ~1M it is a served
retrieval tier with the constraint stage as a filter pushed into the query planner. **The ETL
and the extraction economics are unchanged by that** — only the query path is
corpus-size-dependent, which is why the cost figures above scale and the latency figure does
not.

Search runs **entirely in the browser** over the precomputed index. Per
[DATA_LICENSE.md](DATA_LICENSE.md) the bundle carries **derived fields only** — title, ESCO
labels, STYRK code, municipality, census language level, skill glosses — and every result
links back to the original ad on arbeidsplassen.no. No ad body text, employer names or
contact details are redistributed. **No automated access to finn.no was performed at any
point.**

---

## Scoring

*This documents the **shipped page** (`docs/index.html`), which is the implementation a
reader can go and use. The judged Python pipeline shares the STYRK ladder, the location
values and the severity tables — verified identical — but not the ω weights or the label
bonus, which exist only in the page. That gap is limit 1 below.*

### ① Lexical base — rarity, weighted by field

Term overlap over title + occupation labels ($T$) and skills + glosses + category ($S$),
each term contributing its rarity, normalised by the rarity available so scores stay
comparable across query lengths:

$$\mathrm{idf}(t) = \ln\!\left(1 + \frac{N}{\mathrm{df}(t)+1}\right), \qquad N = 10{,}166$$

$$\text{base} = \frac{\displaystyle\sum_{t \in Q} \mathrm{idf}(t)\,w(t) \;+\; \sum_{p \in B} \mathrm{idf}(p)\,w'(p)}{\displaystyle\sum_{t \in Q}\mathrm{idf}(t) + \sum_{p \in B}\mathrm{idf}(p)}$$

$$w = \begin{cases}1.0 & \text{in } T\\ 0.55 & \text{in } S \text{ only}\end{cases} \qquad w' = \begin{cases}1.0 & \text{in } T\\ 0.80 & \text{in } S \text{ only}\end{cases}$$

A bigram found only in the skills field keeps **0.80** rather than 0.55: a matched *phrase*
is strong evidence wherever it sits, where a lone common word is not.

Field weighting is not cosmetic: a term in the **job title** is far stronger evidence than
the same term buried in a skill list. Matching is on **word boundaries** — `includes("nurse")`
matched `nursery` and returned a kindergarten manager first.

**Bigrams $B$ earn their place or are dropped.** A pair is admitted only if it is
genuinely rarer than either of its parts, which is what stops `low voltage` from scoring
high-voltage ads:

$$B = \{\, p = (a,b) \;:\; \mathrm{idf}(p) > \max(\mathrm{idf}(a), \mathrm{idf}(b)) + 0.5 \,\}$$

Measured across 12 contrastive pairs: **4 better, 8 flat, 0 worse** (§23). It generalises
narrowly — where both qualifiers are content words with their own IDF, the unigrams already
separated them.

### ② Occupation — a predicate over STYRK codes, trusted in proportion to how it resolved

STYRK-08 codes are hierarchical, so shared prefix length *is* taxonomic distance:

$$\mathrm{prox}(c_q, c_{ad}) = \big[\,0,\;0.15,\;0.40,\;0.70,\;1.00\,\big]_{\,\min(\ell,\,4)}, \qquad \ell = \text{common prefix length}$$

How much that is trusted scales with how specifically the phrase resolved — treating a
precise hit and a vague one alike was a real defect:

$$\text{base} \leftarrow \begin{cases} \omega\cdot\mathrm{prox} + (1-\omega)\cdot\text{base}, & \omega = 0.78 \text{ exact ESCO label} \\[2pt] \omega\cdot\mathrm{prox} + (1-\omega)\cdot\text{base}, & \omega = 0.62 \text{ resolved to} \le 2 \text{ codes} \\[2pt] \text{base} + 0.18\cdot\mathrm{prox}, & \text{bare token} \to \text{many codes (bonus, never a gate)} \end{cases}$$

`engineer` resolves to six broad codes spanning 145 ads. Gated at 0.78 it made those 145
unbeatable — an AI-engineer query could not rank a data-engineering ad above an HVAC one.
A direct label match adds a further $+0.02$, enough to break ties on evidence rather than
corpus order.

### ③ Location — graded, best-of, fails open

$$\mathrm{lprox} = \max_{\,l \in \mathrm{locs}(ad)} \begin{cases} 1.00 & \text{exact municipality} \\ 0.55 & \text{same county} \\ 0.00 & \text{elsewhere} \end{cases} \qquad \text{base} \leftarrow \text{base}\cdot\big(1 - \lambda + \lambda\cdot\mathrm{lprox}\big)$$

`max` because 722 ads (7.1%) list several locations; a first-match implementation drops ads
that genuinely do list the city asked for.

### ④ The constraint stage — the one component that measurably works

Severity is **graded, not binary**, and multiplied by how much Norwegian the seeker
actually has:

$$\sigma = \underbrace{\mathrm{sev}(\ell_{ad})}_{\text{ad's demand}} \times \underbrace{\delta(\ell_{seeker})}_{\text{speaker discount}}$$

| $\mathrm{sev}$ | ad's stated level | | $\delta$ | seeker states |
|---:|---|---|---:|---|
| 1.0 | certified · professional · fluent | | 1.0 | none |
| 0.9 | scandinavian_accepted | | 0.8 | basic |
| 0.8 | conversational | | 0.6 | conversational |
| 0.5 | **silent** on a Norwegian ad | | 0.0 | fluent · native |
| 0.4 | desirable | | | |
| 0.0 | either_no_or_en · explicitly_not_required | | | |

**Silence scores 0.5, not 1.0.** That single default decides more advertisements than every
stated requirement combined — 3,507 (34.5%) blocked by silence alone against 5,703 (56.1%)
by a stated requirement. It is a **design decision, not evidence** (§3b).

### ⑤ Final score

$$\text{final} = \text{base} \times \underbrace{\beta}_{\substack{1.22 \text{ if English-accessible} \\ \text{and seeker lacks Norwegian}}} \times \underbrace{(1 - \lambda\sigma)}_{\text{constraint penalty}}$$

$\lambda$ is the slider on the demo, from 0 (off) to 1 (full). **At $\lambda = 0$, or when the
seeker said nothing, the ranking is bit-identical to the unconstrained one** — that non-effect
is the property the whole design protects, because a stage applied to one side of a paired
comparison and not the other is a confound.

*That invariant was broken until an audit of this README caught it.* $\beta$ was a flat 1.22
sitting **outside** the $\lambda$ gate, so $\lambda = 0$ still reordered results — two boosted
ads in the top ten for "nurse in Bergen, I do not speak norwegian" — and the arm labelled
"constraints off" was never off. It is now $\beta = 1 + 0.22\lambda$. The fix cost one point on
the industry suite (83% → 82%) and did **not** resolve §23.1; it was made because the paired
comparison is meaningless without it.

**Almost every constant here is structural, not fitted** — 0.55, 0.80, 0.78, 0.62, 0.18,
1.22, the 0.5 silence default, the STYRK ladder. Each is chosen from the shape of the taxonomy
and left alone, because fitting them requires relevance judgments and
[JUDGING_PROTOCOL.md](eval/JUDGING_PROTOCOL.md) was pre-registered with none applied. **The
one exception is the 0.02 label bonus, which was selected against an outcome**: at 0.35 it
measured flat (82% either way), so it was cut to a tie-breaker and the suite read 83% (§20).
That is fitting on 23 self-authored queries, and it should be read as such. With
enough judgments this stage becomes a learned reranker; at n=13 personas, fitting seven
weights would be curve-fitting. That is a deliberate choice with a cost, not an omission.

$\beta$ pulls workable ads **up** rather than only pushing blocking ads down — 956 ads are
English-accessible, 551 of them `either_norwegian_or_english`. It was once stacked with a
second boost, double-counting derived evidence at 1.32×; that bug put a sales role titled
"GTM Engineer" above three genuine automation matches. **It is also the open defect**: see
§23.1 below.

---

## Measured performance

### Retrieval — 378 judgments, `claude-opus-5` judge, `judge-v1`, 0 validation failures

The judge never sees the extractor's output, grades 0–3 on relevance, and flags constraint
violations on an **independent** axis. The protocol was **committed before any ablation
ran**, with one prompt revision allowed before human calibration and zero after
([JUDGING_PROTOCOL.md](eval/JUDGING_PROTOCOL.md)).

**CVR@10** is the constraint-violation rate in the top ten — the share of returned ads that
demand something the seeker said they lack. **Lower is better.** nDCG@10 is standard graded
relevance; higher is better. So the row that matters below is negative on both: fewer
violations, at a real cost in relevance.

| rung | ΔCVR@10 ↓ | 95% CI | ΔnDCG@10 ↑ | 95% CI |
|---|---:|---|---:|---|
| raw → parsed query | +0.038 | [−0.008, +0.085] | **+0.124** | **[+0.017, +0.260]** |
| + occupation predicate | +0.000 | [−0.023, +0.023] | −0.014 | [−0.052, +0.015] |
| **+ constraint stage (λ=0.7)** | **−0.077** | **[−0.138, −0.023]** | **−0.049** | **[−0.103, −0.005]** |
| + ESCO skills (either resolver) | +0.000 | [−0.023, +0.023] | +0.024 | [−0.017, +0.071] |

Bootstrap: 10,000 replicates, percentile method, **resampling the persona (n=13) and not
the advertisement** — ten ads from one persona share a query and a pool, so treating them
as independent would report an interval several times too narrow and turn noise into
significance.

**Absolute, and the negative result the architecture rests on:**

| arm | CVR@10 ↓ | nDCG@10 ↑ | MRR@10 ↑ |
|---|---:|---:|---:|
| BM25 over the parsed query | 0.154 | **0.798** | 0.962 |
| + constraint stage (λ=0.7) | **0.077** | 0.735 | 0.949 |
| **dense retrieval** | **0.277** | **0.383** | 0.560 |

**The dense channel is the worst arm on every measure** — roughly half the nDCG of every
lexical arm at more than double the violation rate. That is the point, not a disappointment.

**One component clearly earns its place, and five measured as nothing.** The constraint stage
halves violations, and it is a genuine **trade**, because the nDCG cost is *also* significant.
The shipped page still runs seven scoring components, three of which are on that null list —
the occupation predicate, the label bonus and IDF weighting — because they were kept for
behaviour the judged metrics do not capture (tie-breaking, sense disambiguation) rather than
for a measured lift. That is a defensible call and it is not a measured one.
Quoting the first without the second would be dishonest. Five plausible components measured
as **no effect at all**: the occupation predicate, both skill resolvers, IDF weighting on
the query it was built for, and the label bonus. The one rung that clearly earns its place
on relevance is **parsing the query at all**.

### Extraction

| | measured | on |
|---|---|---|
| Language level, **sealed held-out set** | **27/28 = 96.4%** | 28 ads never seen during prompt development |
| — per level | 1.00 on all six stated levels; **0.67 on `unstated` (n=3)** | |
| — accessibility errors | **0 hidden wrongly** · 1 shown wrongly (4.5%) | CI [0, 0.39] — n too small to bound |
| Facet census, full corpus | 9,379 calls · $22.59 · 2.3% demoted · **0 failures** | 10,166 ads |
| Skills, coverage | **98.4%** (from 32.8%) · 6.53 skills/ad · 66,392 total | |
| Skills, recall vs golden | **0.861** (from 0.104) | Haiku, the model that ran the census; Opus scored 0.868 |
| — rejection rate | **6.3%** failed verbatim validation, **discarded not stored** | 4,433 of 70,825 returned |

Skills recall is agreement with labels **Opus produced**, not human labels — an upper bound
on agreement, not on correctness (§15).

### The shipped page — 23-query suite across ten industries

- **82% industry match in top-5** (94/115, down one from the λ-gate fix above) · 0 language parse failures · 0 place parse
  failures · 1 occupation unresolved
- **5/6 discrimination pairs**: data / electrical / software engineer and nurse all clean
  at 5/5 with no bleed; mechanical leaks 1

Run it yourself: `node scripts/check_demo_queries.mjs`.

### The three limits that matter most

1. **The page's ranking has never been judged.** Every interval above comes from the Python
   pipeline; `docs/index.html` is a *second implementation*. The 83% is my own standard, not
   the judge's. This is the largest outstanding item (§22).
2. **n = 13 personas.** The constraint result excludes zero. Nothing else does.
3. **The 1.22× accessibility boost can override topical fit** (§23.1). Ask for an upper-secondary
   teacher *and* say you do not speak Norwegian, and you get PhD fellowships — because the
   flat multiplier lifts the whole English pool over the Norwegian one, and this corpus's
   English ads are academic. **CVR@10 scores that as a success**, because it measures
   language accessibility. That is a measurement gap before it is a ranking gap.

---

## Everything used to make it work

| layer | what | why this one |
|---|---|---|
| **Extraction LLM** | `claude-haiku-4-5` | 0.861 recall vs Opus's 0.868 at a sixth of the cost; both censuses |
| **Judge + labeller** | `claude-opus-5` | a weak judge cannot be repaired by re-prompting; also authored the golden skill labels |
| **Batch API** | 50% discount, `custom_id` constrained | 9,823 calls in 10m13s; `^[a-zA-Z0-9_-]{1,64}$` rejected all 346 of my first attempt |
| **Prompt caching** | 97% hit on the facet census | *"the number that sets the bill"* — and 0% on the skills census, a 20% saving left on the table |
| **Structured output** | tool-use schema + verbatim validation | every skill must appear in the source text or be rejected |
| **Encoder** | `paraphrase-multilingual-mpnet-base-v2` via **ONNX Runtime** | bilingual, so a Norwegian ad and an English query share a space |
| **Lexical retrieval** | BM25 with **stem chains** + bilingual stopwords + char 3–5-grams | Norwegian Snowball strips one suffix per call; closed compounds need n-grams |
| **Occupation taxonomy** | **STYRK-08** hierarchical codes | prefix length is taxonomic distance, so proximity is free |
| **Skills/occupation graph** | **ESCO** — 1,242 occs · 10,063 skills · 52,009 edges, bilingual | `gloss_en` routes matching through English, where the noise floor is lowest |
| **Store** | **DuckDB** | single-file OLAP; 17 tables, 220,988 feed rows, embedded |
| **Evaluation** | TREC **depth-10 pooling** · CVR@10 · nDCG@10 (gains $2^g-1$) · MRR@10 · ΔCVR_paired | pooling bias is the failure mode, and it bit once (§17) |
| **Statistics** | percentile bootstrap, 10,000 replicates, **persona-level** resampling | ad-level resampling would fake significance |
| **Front end** | vanilla JS, zero dependencies, static Pages | no backend to run; the λ slider makes the effect visible rather than asserted |
| **Method** | TDD gate: ground → scenario table → **human approval** → failing tests → implement | **1,022 tests**, offline and keyless. [STANDARDS.md](STANDARDS.md) §3; §3.1 lists seven bugs that had green tests anyway |

**All of it, and what it does not support, is in [LIMITATIONS.md](LIMITATIONS.md)** — 23
sections including the two hypotheses the probes disproved, the ablation that could not
measure its own fix, and every component that measured as nothing.

---

## Corpus

| | |
|---|---|
| Corpus | **10,166** active ads, full body text, 120-day window |
| Feed walk | 220,988 listing entries → 59,830 distinct ads, 3m45s, resumable |
| DuckDB | 17 tables · `ad_locations` 12,003 rows (7.1% of ads list several) |
| ESCO graph | 1,242 occupations · 10,063 skills · 52,009 occ→skill edges, bilingual |
| Accessible to an English speaker | **956 ads (9.4%)** — the needle in the haystack |
| Blocked by a STATED requirement | 5,703 (56.1%) |
| Blocked by SILENCE alone | **3,507 (34.5%)** — a default, not a finding (§3b) |

Earlier drafts reported 4.33% accessible and 74.6% silent. Both predated the census and
were estimates; **the 74.6% figure was wrong by a factor of two.** It is corrected here
rather than quietly dropped.

Two further corpus facts worth stating because they shaped the design: **FINN ads are
excluded from the licensed feed** (`source:"FINN"` uuids return HTTP 404; zero appear among
10,166 — so this indexes the NAV/arbeidsplassen slice), and **the positive class has almost no
lexical signal** — a 22-pattern bilingual lexicon fires `not_required` on **5 ads in
10,166**. English-accessible ads do not announce themselves; they are simply written in
English. Detection is langid, not classification.

---

## What I would do next

In priority order, with the reason for the order:

1. **Judge the shipped page's ranking on the existing pool.** It is the largest gap: the page
   is a second implementation and none of the intervals above apply to it (§22). Cheap —
   the judge, protocol and pool already exist.
2. **Sweep the 0.5 silence default** at 0.3 and 0.7 and report the CVR/nDCG curve. It is the
   single most consequential hand-set constant in the system — it decides more ads than every
   stated requirement combined — and it has never been varied.
3. **Fix the β × topicality interaction** (§23.1). The 1.22 accessibility boost can override
   occupation fit, and CVR@10 is blind to it by construction. Needs nDCG on a topical pool.
4. **Raise n above 13 personas.** Five components measured as "no effect"; at this sample a
   true ΔnDCG of +0.03 is invisible, so some of those nulls are underpowered rather than flat.
5. **Turn on prompt caching for the skills census** for the measured 20% — the smallest and
   most certain win on this list.

## Layout

```
src/finn_smart_search/
  ingest/        nav_feed.py  silver.py  store.py  anthropic_client.py
  pii.py         scrub before anything is stored
  understanding/ census.py  census_prompt.py  census_validate.py  skills_prompt.py
                 enrich_language.py
                 dedup.py  html_clean.py  langid.py  taxonomy.py  text_norm.py
  esco/          fetch.py
  retrieval/     bm25.py  constraints.py  occupation.py  location.py  skills_match.py
  eval/          judge_llm.py  metrics.py  pool.py  stats.py  scoring.py
                 gold_parse.py  sealed_ingest.py  skills_scoring.py
eval/            JUDGING_PROTOCOL.md  GOLD_PARSE_SCHEMA.md  JUDGE_SCENARIOS.md
                 personas.yaml  gold_parses.yaml  golden_skills.json
                 judgments.json  demo_queries.json
scripts/         run_skills_census.py  run_judgments.py  export_web.py
                 check_demo_queries.mjs  (+ 11 probe_*.py)
docs/            index.html  data/index.json      <- the GitHub Pages demo
reports/         every probe, census and ablation log, kept including the wrong ones
```

## Running it

**The demo needs nothing installed.** It is live at
**[naomititus.github.io/finnno_smart_search](https://naomititus.github.io/finnno_smart_search/)**
— GitHub Pages serves a 2.9 MB gzipped index as a static file, and all search runs in the
visitor's browser. No server, no API key, no vector database, nothing to spin up.

To serve it locally you need a static file server, **not** a double-click: the page
`fetch()`es `data/index.json`, which browsers block over `file://`.

```bash
python3 -m http.server 8000 --directory docs   # then open localhost:8000
python3 -m pytest                              # 1,022 tests, 65s, offline, no API key
```

The index is committed, so both of those work on a fresh clone with no key and no spend.

## Rebuilding from source

These are for regenerating the index, not for running the demo. They need the licensed NAV
feed, an `ANTHROPIC_API_KEY`, and the gitignored `data/` directory.

```bash
pip install -e ".[dev]"
python run_ingest.py --days 120              # resumable; ~35 min, polite rate limits
python run_esco.py                           # ~20 min
python scripts/run_skills_census.py           # DRY RUN by default; --submit to spend
python scripts/export_web.py                  # rebuild docs/data/index.json
node scripts/check_demo_queries.mjs           # the 23-query suite
pytest                                        # the TDD gate
```

**Spend guards, stated precisely because "it's all safe" is the kind of claim that ages
badly.** `run_skills_census.py`, `run_judgments.py` and `run_skills_pilot.py` require
`--submit` and are dry runs by default. `run_census.py` **spends by default** — its
`--dry-run` is opt-in — and is bounded by `--max-spend` instead. `run_pilot.py` and
`run_sealed_eval.py` have no guard at all. All of them print an estimate before spending.
`data/` is gitignored — no ad text, employer names or contact details are redistributed.

**Licences.** Code is [MIT](LICENSE). The advertisement data is *not* redistributed and is
governed separately by [DATA_LICENSE.md](DATA_LICENSE.md) — **no automated access to finn.no
was performed at any point.**

**Not present, so as not to imply otherwise:** there is no CI workflow (the 1,022 tests run
locally only) and no container or deployment manifest. GitHub Pages deploys on push to
`main`.
