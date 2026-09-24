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
