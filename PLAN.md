# Project plan

Status as of 2026-09-24. Phase A complete. See README.md for measured findings.

## A. Ingest & graph — DONE
A1 NAV feed walk, 120 days, resumable cursor, RFC-1123 weekday guards ✓
A2 Detail fetch, ACTIVE-only (INACTIVE returns a 112-byte stub) ✓
A3 Bronze → silver: HTML→blocks→text, typed tables ✓
A4 ESCO graph: 1,242 occupations, 10,063 skills, 52,009 edges, bilingual ✓
A5 Bilingual lexicon + corpus measurement ✓
A6 Census prompt v2: 3 adversarial reviews, findings validated against corpus ✓

## A7 Corpus is FROZEN — see DECISIONS.md D2
Snapshot 2026-09-24, 10,166 ads. No refresh. 37% expire within 7 days, 83%
within 30 — so the demo states the snapshot date and shows per-ad expiry state.

## B. Facet census — NEXT
B1 ✓ dedup: exact hash of normalised text (D8). 9,823 clusters, 343 redundant,
   largest 44. signature/cluster/representatives/fan_out, 20 tests, 100% mutation score.
B2 Pilot 50 ads; verify schema adherence, span validity, demotion rate
B3 Batch census over ~9,500 unique clusters (Haiku 4.5, ~$3, <1h expected)
B4 Post-validate; track demotion rate as a first-class metric
B5 Fan out cluster results; derive english_accessible; build ad_facets (gold)
B6 Aho-Corasick: skill phrases → ESCO URIs via the 10,063-skill gazetteer

## C. Human audit — 150 ads, ~2h
C1 Stratify on (LLM label × signal agreement), cap 3 ads/employer
C2 Label 150 in a minimal CLI showing only the extracted span
C3 80 double-labelled → LLM noise floor, reported beside every number
C4 Hájek-weighted P/R/prevalence; report Kish ESS; bootstrap CIs by employer

## D. Retrieval
Framing (revised 2026-09-27): the operation is CONTAINMENT, not similarity. A job's
required set R against a seeker's attribute set S, viable iff R ⊆ S, ranked by weighted
coverage of what the seeker asked for. Each facet is THREE-valued — satisfied, violated,
or UNSTATED — because silence is the dominant way absence appears in this corpus
(LIMITATIONS §14) and treating it as a violation costs 3,507 ads. Negation is not a
separate problem: a seeker's "I do not speak X" is the attribute level `none`, which is
how `constraints.py` already types it.
D1 BM25, dual Snowball stemming (no + en), char 3-5grams for compounds. DONE.
   Stem CHAINS not single-pass stems (Norwegian Snowball strips one suffix, so
   sykepleier/sykepleiere never met). Bilingual STOPWORDS added against this
   module's original reasoning: IDF only suppresses the majority language, so
   English boilerplate scored idf 5.8 against `nurse` at 2.04 — DECISIONS D18.
D2 Dense: brute-force numpy (10k×768 = 30MB, ~5ms). DEVIATION: nb-sbert-base needs
   torch, and there is no PyTorch wheel for Python 3.13 on Intel macOS. Running
   paraphrase-multilingual-mpnet-base-v2 via ONNX Runtime instead — same family, same
   768 dims. No result may be attributed to nb-sbert. LIMITATIONS §12.
D3 Graph channel: k-hop from profile seeds, ≥5 shared skills threshold. Also
   supplies OCCUPATION PROXIMITY (styrk_code, role_family) so a `sykepleier`
   query reaches `spesialsykepleier` — promoted in importance by D18.
D4 RRF fusion (k=60)
D5 Constraint stage: graded severity from norwegian_requirement_level, λ dial
D6 Query parser: gazetteer fast path → distilled classifier → LLM residue.
   ENTRY POINT IS FREE TEXT (DECISIONS D19). Not CV upload: a CV carries the soft
   half of S (occupation, skills, years) and not the hard half — location and
   language are 15 of the 37 hard dev constraints and are preferences about the
   FUTURE, which a record of the past cannot hold. CV parsing is owed to the
   BASELINE only (LIMITATIONS §1b), not to the search surface.
   Output contract: the typed S of eval/GOLD_PARSE_SCHEMA.md. Test set: the 13 dev
   gold parses. Occupation resolves against ESCO's bilingual labels (1242/1242
   have both no+en) to a CANDIDATE SET, never a tiebroken winner — `nurse` is
   genuinely four ESCO occupations and picking one is guessing.
D7 Linear scorer: rrf + skill_coverage + occupation_proximity + recency + location.
   `occupation_proximity` is the MECHANISM that carries occupation, not a garnish:
   D18 measured that BM25 weights a hard occupation constraint by rarity, which is
   the wrong ordering by construction. BM25 scores the free-text residue only.
   Coverage is WEIGHTED and three-valued, not a match count: a missing hard requirement
   (`norsk autorisasjon`, 462 ads) disqualifies where a missing preference only demotes,
   and `unstated` must not score the same as `violated`.

## E. Evaluation
E1 20 persona instances across 12 verticals; 13 dev, 7 SEALED (opened once, at the
   end). Split pre-registered 2026-09-27 in personas.yaml `split:` — DECISIONS.md D16.
   Corrects this line's original "16 personas / 10 dev": four controls were added
   after E1 was written. The SEALED COUNT of 6 is preserved as pre-registered; dev
   absorbed the growth. Pairs are never split across dev/sealed — that would destroy
   the within-pair comparison. Sealed verticals are disjoint from dev, so the sealed
   run measures generalisation and reads as a LOWER bound, not a replicate. Sealed was
   amended 6 -> 7 the same day, pre-results, to add p11 (Social services) — the
   "Norwegian genuinely required, return few results" case D16 flagged as missing.
   DECISIONS.md D17.
E2 JUDGING_PROTOCOL.md committed BEFORE any ablation (git timestamp = pre-registration)
E3 Pool depth 10; coverage diagnostic; extend to 20 only if top-10s are unjudged
E4 LLM judge, isolated from extractor output (unit-tested). DONE: eval/judge_llm.py
   + eval/pool.py, 57 tests, built to the STANDARDS §3 gate (grounded, scenario
   table approved 2026-09-27, tests failed first on NotImplementedError). Isolation
   encoded per JUDGING_PROTOCOL erratum E2 — `skills`/`seniority` are also ordinary
   English, so the test uses 14 distinctive names plus a no-snake_case rule.
   Depth-10 pool over the current 5-arm ladder = 346 unique pairs (~$5 to judge).
   NOT YET RUN: no judgment exists, so every metric below is still unmeasured.
E5 120 human relevance pairs → Cohen's κ, quadratic-weighted
E6 Ablation: BM25 / dense / hybrid / +graph / +constraints soft / hard / +rerank
E7 Paired negation stress-test; CVR@10; bootstrap CIs on every adjacent-rung delta

## F. Vector-DB benchmark
F1 DenseIndex interface; NumpyFlat / FaissHNSW / Qdrant behind one config line
F2 Crossover curve: 10k → 100k → 1M → 10M, recall@10 vs exact, p50/p95, build time
F3 Filtered search: post-filter vs pre-filter vs in-engine at our 7% selectivity

## G. Deliverables
G1 Static demo: precomputed facets + int8 vectors, λ slider, negation toggle diff
G2 Eval dashboard: ablation table, CVR bars, negation dumbbell, λ tradeoff curve
G3 README with thesis, honest data caveat, headline result, limitations
G4 GitHub Pages deploy

## Cut after review
Snorkel label model (LFs correlated; 0% coverage on the classes that matter)
Active learning (incompatible with unbiased held-out eval; overhead without payoff)
4-class norwegian_required (unmeasurable at 0.05% lexicon coverage → 9-level scale)
Corpus-wide workLanguage fetch (rate-limited; sample-only stratifier)
Rebalanced training + prior correction (breaks label-shift; calibrate on weighted sample)
