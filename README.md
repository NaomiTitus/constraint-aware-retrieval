# Constraint-aware job search

Search over Norwegian job ads that actually respects *"I do not speak Norwegian."*

**The thesis.** A bi-encoder cannot represent negation. `"jeg snakker ikke norsk"` and
`"jeg snakker flytende norsk"` sit close together under any multilingual sentence encoder,
so a dense-retrieval search surface returns near-identical rankings for both. Constraints
are **predicates over metadata**; similarity is a **metric over content**. Set-difference is
not expressible in a metric space. The fix is architectural: extract typed constraints
*before* retrieval and apply them in a separate, tunable stage.

## Status

Phase A complete — ingest, storage, and the taxonomy graph.

| | |
|---|---|
| Corpus | **10,166** active ads, full body text, 120-day window |
| Feed walk | 220,988 listing entries → 59,830 distinct ads, 3m45s |
| ESCO graph | 1,242 occupations · 10,063 skills · 52,009 occ→skill edges (bilingual) |
| English-accessible | **438 ads (4.33%)** — the needle in the haystack |
| Norwegian ads explicitly requiring Norwegian | 2,563 (25.2%) |
| Norwegian ads silent on language | 7,583 (74.6%) |

## Measured findings

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
resources/      lexicon_language.yaml
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
