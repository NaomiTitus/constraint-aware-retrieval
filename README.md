# Constraint-aware job search

Search over Norwegian job ads that actually respects *"I do not speak Norwegian."*

**The thesis.** A bi-encoder cannot represent negation. `"jeg snakker ikke norsk"` and
`"jeg snakker flytende norsk"` collapse to nearly the same vector under a multilingual
sentence encoder — **measured at cosine 0.914** across five paired personas. Constraints
are **predicates over metadata**; similarity is a **metric over content**. Set-difference is
not expressible in a metric space. The fix is architectural: extract typed constraints
*before* retrieval and apply them in a separate, tunable stage.

The measurement also corrected the claim. Near-identical query vectors do **not** produce
near-identical rankings — overlap@10 is 5.6/10 and rank correlation is ~0, so the result
list does reshuffle. What survives the reshuffle is the damage: **3.6 of every 10 top
results still demand Norwegian**, and stating the constraint moved the seeker *closer* to
those ads than saying nothing at all, on **5 of 5** personas. Encoder:
`paraphrase-multilingual-mpnet-base-v2` via ONNX Runtime — **not** nb-sbert-base, which
this project has never run. See [LIMITATIONS.md](LIMITATIONS.md) §12.

## Status

Phase A complete — ingest, storage, taxonomy graph. Facet census complete over all
10,166 ads. First retrieval component landed (constraint stage) and the premise behind
it is now measured, not assumed.

| | |
|---|---|
| Corpus | **10,166** active ads, full body text, 120-day window |
| Feed walk | 220,988 listing entries → 59,830 distinct ads, 3m45s |
| ESCO graph | 1,242 occupations · 10,063 skills · 52,009 occ→skill edges (bilingual) |
| Census | census-v15, 9,379 calls, **$22.59**, 2.3% demoted, 0 failures |
| Accessible to an English speaker | **956 ads (9.4%)** — the needle in the haystack |
| Blocked by a STATED requirement | 5,703 (56.1%) |
| Blocked by SILENCE alone | **3,507 (34.5%)** — a default, not a finding |

The last row is a design decision, not evidence, and it decides more ads than every
stated requirement combined — see [LIMITATIONS.md](LIMITATIONS.md) §3b. Earlier drafts of
this table reported 4.33% accessible and 74.6% silent; both predated the census and were
estimates. The 74.6% figure was wrong by a factor of two.

## Measured findings

- **Declaring a constraint makes dense retrieval worse, not better.** On 5 of 5 paired
  personas, adding *"jeg snakker ikke norsk"* to a query raised its mean similarity to
  Norwegian-demanding ads versus staying silent. The negation is not merely invisible to
  the encoder — the extra tokens about Norwegian pull the seeker *toward* the ads they rule
  out. This is the single strongest argument for a separate constraint stage.
- **What pooling erases is any single clause, not negation specifically.** Padding a query
  to realistic length drives cosine to 0.999, but a one-word *content* swap — nurse vs
  carpenter, Oslo vs Bergen — is erased at the same rate (margins of 0.004, noise). The
  control disconfirmed the tidier "mean pooling averages away the negation" story. The
  real claim is broader and stronger: single-vector retrieval loses single-clause
  distinctions once queries are long.
- **FINN ads are excluded from the licensed feed.** `source:"FINN"` uuids from the
  arbeidsplassen search index return HTTP 404 from the feed; zero appear among 10,166.
- **The positive class has almost no lexical signal.** A 22-pattern bilingual lexicon fires
  `not_required` on **5 ads in 10,166**. English-accessible ads do not announce themselves —
  they are simply written in English. Detection is langid, not classification.
- **The interesting signal is the severity gradient among the 95% Norwegian ads**, not the
  binary. That is what the ranking penalty operates on.
- **The skill layer is what makes the graph a retrieval channel.** Median ads reachable:
  92 via shared occupation, **1,213 via ≥5 shared skills**, 4,844 via ≥1 (hub-dominated,
  useless). The threshold is a tunable.
- **NAV's ESCO references are not case-normalised.** Joining on labels inflates the
  occupation graph ~2× (1,747 phantom nodes vs 817 real).

## Layout

```
src/finn_smart_search/
  ingest/       nav_feed.py  silver.py  store.py
  understanding/ html_clean.py  census_prompt.py
  esco/         fetch.py
  retrieval/    constraints.py        graded severity, off at lambda=0
resources/      lexicon_language.yaml
scripts/        probe_negation.py  probe_negation_dilution.py  (+ census probes)
reports/        negation_probe.json  negation_dilution.json  census_full.log
docs/           ann-hnsw-explainer.html  occupation-pruning.html
tests/fixtures/ recorded API responses
```

## Reproducing

```bash
pip install -e ".[dev]"
python run_ingest.py --days 120     # resumable; ~35 min, polite rate limits
python run_esco.py                  # ~20 min
```

`data/` is gitignored — no ad text, employer names or contact details are redistributed.
See [DATA_LICENSE.md](DATA_LICENSE.md). **No automated access to finn.no was performed.**
