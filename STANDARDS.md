# Project standard

The reference this project is checked against at every step. If a change does not
fit this document, either the change is wrong or this document needs amending
first — in that order.

Status legend: `✓` exists · `~` partial · `✗` not yet built

---

## 1. Target layout

```
finnno_smart_search/
├── Makefile                      ✗  setup test ingest census index eval serve web all
├── pyproject.toml                ✓  extras: llm · search · api · dev
├── README.md                     ✓  thesis · headline result · honest data caveat
├── ARCHITECTURE.md               ✓  ETL, facet schema, latency argument
├── PLAN.md                       ✓  step-by-step status
├── STANDARDS.md                  ✓  this file
├── DATA_LICENSE.md               ✓  NAV terms · ESCO · "no finn.no crawling"
├── DECISIONS.md                  ✓  decision log with evidence
├── LICENSE                       ✗  MIT
├── .github/workflows/ci.yml      ✗  ruff + mypy + pytest (offline markers only)
├── configs/                      ~  ingest · extract · eval · ranking/NN_*.yaml
├── resources/                    ✓  lexicon_language.yaml · stopwords · esco/
├── src/finn_smart_search/
│   ├── cli.py                    ✗  typer; every operation reachable from here
│   ├── config.py                 ✗  pydantic-settings; no bare os.environ
│   ├── schemas.py                ✗  shared pydantic models
│   ├── api/                      ✗  FastAPI app · request/response schemas · DI
│   ├── ingest/                   ✓  nav_feed · silver · store  (arbeidsplassen_sampler ✗)
│   ├── understanding/            ~  html_clean ✓ census_prompt ✓ census_validate ✓
│   │                                dedup ✓ · langid ✗ · language_rules ✗ · classifier ✗
│   ├── esco/                     ✓  fetch  (graph ✗)
│   ├── query/                    ✗  parse_rules · parse_llm · personas
│   ├── retrieval/                ✗  bm25 · dense · graph_channel · fuse
│   │                                constraints · score · rerank · pipeline
│   ├── eval/                     ✗  pool · judge_llm · metrics · stats · run_ablation
│   └── export/                   ✗  quantize · build_web
├── tests/
│   ├── unit/                     ✗  tier 1 — pure functions, no I/O
│   ├── fixture/                  ✗  tier 2 — recorded I/O, offline
│   ├── golden/                   ✗  tier 3 — LLM evals, needs a key, skipped in CI
│   ├── integration/              ✗  tier 4 — end-to-end, marked slow
│   └── fixtures/                 ✓  recorded API responses
├── index/                        ✗  versioned build artifacts (gitignored)
├── data/                         ✓  gitignored — duckdb, exploration, logs
├── reports/                      ~  ablation.md ✗ · figures ✗ · occupation_distribution.json ✓
├── scripts/                      ✓  one-off generators
└── docs/                         ✓  GitHub Pages — demo + explainers
```

**Root scripts `run_ingest.py` / `run_esco.py` are temporary.** They fold into
`cli.py` at the first opportunity. Nothing new goes at the root.

---

## 2. Where things belong

| If it... | it goes in | and must |
|---|---|---|
| transforms data with no I/O | `src/.../<domain>/` | have a tier-1 unit test |
| talks to a network | `src/.../ingest/` or `esco/` | have a recorded fixture test |
| prompts an LLM | `src/.../understanding/` or `query/` | have a golden-set eval |
| serves a request | `src/.../api/` | have a contract test |
| is a one-off | `scripts/` | be excluded from coverage |
| is exploratory | a notebook | import from `src/`, define nothing |

Notebooks are **narrative only**. Logic that matters lives in the package and is
tested. This split is itself a hiring signal; do not blur it.

---

## 3. TDD workflow — the gate

Every step runs this loop. Steps 1 and 3 are hard gates; nothing proceeds
without them.

1. **GROUND THE INPUT.** Measure what the real input and output actually look
   like, and write the measurement down. Takes as many steps as it takes.
2. Produce a **scenario table** — input · expected output · rationale — whose
   every fixture **cites its grounding** from step 1.
3. **Human approves or amends it.** No code before this.
4. Write the tests. They fail.
5. Demonstrate they fail *for the right reason* — not an import error.
6. Implement until green.
7. Only then run against real data.

The scenario table is the artifact. Tests are its executable form.

### 3.0 Grounding — the step before the scenario table

**Rule: a test fixture is either loaded from a real artifact, or accompanied by
the measurement that justifies its shape. Never invented.**

A hand-written fixture is allowed — often it is clearer — but it must carry a
comment naming what it was derived from and what was measured, e.g.
*"the real shape: html_clean emits blocks, silver joins them with `\n`"*, or
*"· U+00B7 leads 1,930 corpus blocks"*. A fixture with no provenance is a
guess about production, and a test built on a guess agrees with the guess.

Being multi-step is expected and budgeted. Counting glyph frequencies across
10,166 ads before writing a boundary test is not overhead; it IS the test
design.

**Match the source to the kind of assumption.** This is the part that is easy to
get wrong — see the caveat below.

| Assumption under test | Ground it in |
|---|---|
| labels, semantics, the level taxonomy | `eval/golden_set.json` — 44 hand-labelled ads |
| **text shape**: block joins, whitespace, glyphs, invisibles | **the full corpus, counted** — `SELECT description_text FROM ads`, not a sample |
| LLM response shape, tool-use envelopes | `tests/fixtures/batch_results_real.json` — recorded live responses |
| HTML structure, markup variants | `tests/fixtures/ads_sample_71.json` — raw `description` values |
| HTTP/API behaviour, error bodies | recorded fixtures: `feedentry_active.json`, `feedentry_inactive_stub.json`, `search_index_429.json` |
| a data file's schema and key names | **the file itself, with a closed key vocabulary** |

**The golden set is not a general grounding source, and here is the proof.**
The 44-ad golden set is stratified for *label* diversity, not *surface*
diversity. Measured over its ad text, the glyphs that caused the 55%-demotion
bug are absent:

| glyph | corpus blocks | in golden set | in `ads_sample_71` |
|---|---:|---|---|
| `·` U+00B7 | 1,930 | **no** | no |
| `●` U+25CF | 141 | **no** | no |
| `` U+F0B7 (Wingdings, from Word) | 29 | **no** | no |

So grounding the span-boundary tests in the golden set would have produced
exactly the same bug. Only a **distributional** count over the whole corpus
surfaces them. Use the golden set for *what a correct answer is*; use the corpus
for *what the input looks like*.

### 3.1 The failure this rule exists to prevent

Seven bugs in the census module were a hand-written pattern of mine, not a model
error, and **every one had a green test**. The tests constructed their own
inputs, so they agreed with themselves:

| The test's fixture | Production reality | Cost |
|---|---|---|
| single-line prose, full stops between sentences | blocks joined with `\n` | 55% of the pilot demoted; 26 of 28 rejections legitimate |
| `BOUNDARY` listing `•` only | `·` `●` emoji U+200B U+2060 U+F0B7 lead 2,998 blocks | 9 of 22 ads demoted with correct evidence |
| corpus check passing blocks **with** their glyph | the model quotes the text **without** it | reported "0 false rejects" while the above was live |
| oracle regex spelled `dokument` | ~7% of the corpus is English — `documentation` | 1 of 2 "model errors" in a probe |
| oracle matching any `X eller engelsk` | document-language clauses say it too | 4 of 9 "model errors" in a probe |
| glyph filter requiring a typed bullet | `<li>` markup yields no glyph | made a 289-ad population look like 26 |
| `test_9_1` **setting** `annotated_accessible` itself | the file used `accessibility_note` | `taxonomy_gap` unreachable for the whole project |

The last one is the clearest statement of the rule: the test wrote the field it
then asserted on. It passed for weeks and measured nothing.

### 3.2 Helpers — make grounded the path of least resistance

`tests/conftest.py` exposes the canonical sources so using real data is shorter
than inventing it:

| Helper | Gives |
|---|---|
| `golden_file()` | `eval/golden_set.json` parsed — the file, not a built dict |
| `real_blocks()` | block texts from `ads_sample_71.json` via `html_clean` — the true production shape, no DB needed |
| `corpus_con` (fixture) | read-only session connection to `data/ads.duckdb` |
| `requires_corpus` | skip marker with a stated reason, for a fresh clone |

Corpus-dependent tests **run by default** and skip with a reason when `data/` is
absent. They were excluded once, and a golden-set relabelling was committed with
the drift guards red because `pytest` never executed them.

### Tiers

| Tier | Marker | What | Determinism | In CI |
|---|---|---|---|---|
| 1 | `unit` | pure functions | exact | yes |
| 2 | `fixture` | recorded I/O | exact | yes |
| 3 | `golden` | LLM behaviour | threshold on a frozen golden set | no |
| 4 | `integration` | end-to-end, latency | threshold | no |

Tier 3 asserts **accuracy ≥ an agreed threshold**, never exact match. The golden
set is frozen BEFORE prompt tuning, or you are fitting to your own test.

Tier 4 asserts **guardrails**, not optima: e.g. `p95 < 300ms`,
`CVR@10 <= 0.25`, `negation stress-test flips >= 5 of 10`.

### CI policy

CI runs `unit` and `fixture` only. It must pass on a fresh clone with **no
network and no API key**. A reviewer who cannot get green will not read further.

---

## 4. Non-negotiables

- **`tests/unit/test_judge_isolation.py`** — the rendered LLM-judge prompt must
  contain no `ad_facets` field name. If this fails, CVR@10 measures the extractor
  agreeing with itself and every headline number is void.
- **`eval/JUDGING_PROTOCOL.md` is committed BEFORE any ablation runs.** The git
  timestamp is the pre-registration.
- **Sealed personas stay sealed** until the final run.
- **No automated access to finn.no**, ever. See DATA_LICENSE.md.
- **Every gold row carries `extractor_version` + `prompt_version`.**
- **Demotion rate is reported**, never silent — it converts precision errors into
  recall errors invisibly.

---

## 5. Index manifest contract

Every build writes `index/v{N}/manifest.json`. The service loads a version and
reports it from `/readyz`. This is what makes "which model produced this ranking?"
answerable later.

```json
{
  "index_version": "v3",
  "built_at": "2026-09-24T18:00:00Z",
  "corpus": { "source": "nav-pam-stilling-feed", "window_days": 120,
              "fetched_at": "...", "n_ads": 10166, "n_clusters": 9771 },
  "versions": { "extractor": "...", "prompt": "census-v2",
                "embedding_model": "NbAiLab/nb-sbert-base", "esco": "v1.2" },
  "artifacts": { "embeddings": "emb.f32.npy", "bm25": "bm25.bin",
                 "graph": "graph.parquet", "facets": "facets.parquet" },
  "counts": { "occupations": 1242, "skills": 10063, "occ_skill_edges": 52009 },
  "quality": { "census_demotion_rate": null, "english_accessible": null }
}
```

Agreed in the Phase A retrofit, because census, retrieval build and the service
all read or write it.

---

## 6. Definition of done — per step

- [ ] Scenario table written and approved
- [ ] Tests exist at the right tier and failed first
- [ ] Implementation passes; no test weakened to make it pass
- [ ] `ruff` and `mypy --strict` clean on `src/`
- [ ] Public functions have docstrings saying *why*, not *what*
- [ ] Any measured claim in a docstring or README cites its number
- [ ] Reachable from `cli.py`; no new root scripts
- [ ] Version fields written where the step produces gold data
- [ ] README/PLAN status updated
- [ ] Committed with a message stating what was measured, not just what changed

---

## 7. The reviewer's path

What a hiring reviewer does, in order. Optimise for this sequence.

1. `git clone` → `make setup && make test` — **green, offline, no key**
2. README — thesis, headline number, data caveat
3. `src/.../retrieval/pipeline.py` — the architecture at a glance
4. Are there tests next to the claims?
5. `reports/ablation.md` — are the numbers reproducible?
6. `DATA_LICENSE.md` — is the data handling lawful?
7. The live demo

A README claiming a 12 ms query path with nothing in the suite measuring it is
worse than making no claim.
