# Validation sets and human reviews

Every human review in this project is run through the same tool, recorded the
same way, and kept here. The record is deliberately auditable: a reviewer can
see what was selected, why, by what rule, with what seed, what the human
decided, and what changed as a result.

## Why this is a first-class artefact and not a footnote

Two claims in this project can only be supported by a human:

1. **that the extractor is accurate** — on ads chosen after the prompt was
   frozen and never tuned against, or the number is fitted rather than measured;
2. **that FINN's live product has the bug** — no code may touch finn.no
   (`DECISIONS.md` D1), so the baseline is recorded by hand in a browser.

Everything else in the repo is reproducible by running it. These are not, so
their provenance has to be written down or the results are unverifiable.

## The pattern every review follows

| step | artefact |
|---|---|
| selection rule + seed, stated before looking | `manifest.json` → `selection` |
| items rendered with the evidence needed to judge | the review tool (below) |
| a machine verdict shown as a SUGGESTION, never as an answer | `prefill` in the tool |
| the human decides; every value marked unconfirmed until touched | artifact db |
| an ingestion **gate** rejects anything unusable | `eval/sealed_ingest.py` |
| what changed, and what it cost | `manifest.json` → `outcome` |

### The review tool

`scripts/build_review_artifact.py` renders any review to a self-contained page:
per-item evidence, literal English glosses (the author does not read Norwegian —
see `LIMITATIONS.md` §1), validity computed by the **production validator**
rather than reimplemented, and labels synced to the artifact so a browser change
cannot lose an hour of work.

Three properties it is built to guarantee, each learned from a defect found by
review rather than by reasoning:

- **click-to-quote, never type.** A hand-typed span differing by one character
  fails `not_verbatim`, and a gold span the extractor can never match
  mismeasures every future run.
- **show only what the model is shown.** Blocks come from
  `census_validate.prepare()`'s output, not the full ad. An earlier version drew
  from the full text and offered 16 lines the model never sees — one ad's only
  language line was inside the truncated middle, making it unwinnable.
- **no rule reimplemented in the page.** Span verdicts are computed in Python by
  the same code that runs in production and embedded as data. The first version
  reimplemented the rule in JavaScript and drifted: it showed a green tick on 56
  of 98 spans the validator rejects.

## What is committed here, and what is not

Committed: the manifest, the selection rule, and the labels — including the
short `evidence_span` quotations, matching the existing `eval/golden_set.json`.

Not committed: ad bodies, employer contact details, and the rendered worksheets.
`data/` is gitignored. Excerpts are PII-scrubbed before they ever reach a page
(`src/finn_smart_search/pii.py`): the contact person's name appears in the ad
body on 11.8% of ads, their email on 10.1%, their phone on 9.6%, so excluding
the structured `contactList` alone is only a partial control.

## Pausing at a discovery is not finishing

`002` paused after 12 of 46 rows. The reviewer found a class bug — three of four
`application_language` values were borrowed from other fields' enums — and
stopped rather than judge 34 more rows against data about to change. Those 12
verdicts are discarded.

**Pausing is a step in a review, not the end of one.** A review ends when every
row has been judged against output that will ship and no finding is outstanding.
The temptation after a good discovery is to write it up and move on, which
converts a partial review into a finished-sounding artefact — and leaves the
other 34 rows, any one of which could hold a different class bug, unexamined.

What the pause bought, measured: two subagents audited the review page and the
fix while it was stopped, and found **ten further defects** between them — four
in the page, six in the validator. Among them, a cap that silently hid the
decisive sentence on two rows, and a `TypeError` that would have aborted the
entire census after the batch was paid for. Every one would have corrupted the
remaining verdicts or the run they were gating. The 34 unspent verdicts were
worth more after the audit than before it.

So the sequence is: **pause at the discovery → fix → audit the fix adversarially
→ re-extract → resume from row 1.** Not pause → write up → move on.

### Two faults were in the tool, not the reviewer

A verdict button conflated "does the ad support this fact" with "is this the
right value for this field" — on three rows the honest answers differed. And the
per-facet search terms hid the evidence on one row while the page said nothing
had matched. Both are recorded in the manifest, because a reviewer misled by the
instrument has not made an error; the instrument has.

### Process rules that came out of it

- Never `git commit -a` while a subagent holds the working tree. Commit `e31ef0e`
  swept up an agent's in-flight mutation and shipped a red suite for one commit.
- Run the suite in the same command as the commit, not before it.
- Never `git checkout --` to undo an experiment when the file also holds
  uncommitted work you want. Commit the good state first, then mutate.

## Reviews

| id | what | items | state |
|---|---|---:|---|
| `001-sealed-language` | held-out language labels — the unbiased accuracy set | 28 | labelled |
| `002-facet-exceptions` | the rare values of near-constant facets | 46 | **in progress** — paused at a class bug found in 4 rows; tool rebuilt, rows re-extracted under census-v10, resuming from row 1 |
| `003-finn-baseline` | the live FINN product, via CV upload | 15 | not started |

### A retired review

A query-paste baseline for FINN SmartSøk was built and then withdrawn: the
feature was replaced by CV-upload matching between 2026-09-22 and 2026-09-26.
Recorded in `LIMITATIONS.md` §1b rather than deleted silently, because "we
could not measure this" is itself a result, and a tool left in the repo would
imply evidence that cannot be produced.
