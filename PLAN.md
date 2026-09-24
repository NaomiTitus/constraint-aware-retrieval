# Project plan

Status as of 2026-09-24. Phase A complete. See README.md for measured findings.

## A. Ingest & graph — DONE
A1 NAV feed walk, 120 days, resumable cursor, RFC-1123 weekday guards ✓
A2 Detail fetch, ACTIVE-only (INACTIVE returns a 112-byte stub) ✓
A3 Bronze → silver: HTML→blocks→text, typed tables ✓
A4 ESCO graph: 1,242 occupations, 10,063 skills, 52,009 edges, bilingual ✓
A5 Bilingual lexicon + corpus measurement ✓
A6 Census prompt v2: 3 adversarial reviews, findings validated against corpus ✓

## B. Facet census — NEXT
B1 Wire dedup.cluster() into the runner — MinHash over body shingles, 395 redundant
   ads (3.9%), largest cluster 44. Extract once per cluster, fan out results.
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
D1 BM25, dual Snowball stemming (no + en), char 3-5grams for compounds
D2 Dense: nb-sbert-base, brute-force numpy (10k×768 = 30MB, ~5ms)
D3 Graph channel: k-hop from profile seeds, ≥5 shared skills threshold
D4 RRF fusion (k=60)
D5 Constraint stage: graded severity from norwegian_requirement_level, λ dial
D6 Query parser: gazetteer fast path → distilled classifier → LLM residue
D7 Linear scorer: rrf + skill_coverage + occupation_proximity + recency + location

## E. Evaluation
E1 16 personas across 10 verticals; 10 dev, 6 SEALED (opened once, at the end)
E2 JUDGING_PROTOCOL.md committed BEFORE any ablation (git timestamp = pre-registration)
E3 Pool depth 10; coverage diagnostic; extend to 20 only if top-10s are unjudged
E4 LLM judge, isolated from extractor output (unit-tested)
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
