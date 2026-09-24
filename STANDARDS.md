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

Every step runs this loop. Step 2 is a hard gate; nothing proceeds without it.

1. Produce a **scenario table** — input · expected output · rationale.
2. **Human approves or amends it.** No code before this.
3. Write the tests. They fail.
4. Demonstrate they fail *for the right reason* — not an import error.
5. Implement until green.
6. Only then run against real data.

The scenario table is the artifact. Tests are its executable form.

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
