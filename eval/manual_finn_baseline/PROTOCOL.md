# Manual FINN SmartSøk baseline — protocol

**Written before any measurement is taken.** The git timestamp on this file is
the pre-registration: it fixes what will be recorded and how, so the result
cannot be shaped after seeing it.

This is the only evidence in the project that cannot be produced by code, and it
is what turns the central claim from an argument into a comparison.

---

## Why this exists at all

The thesis is that a bi-encoder cannot represent negation: *"jeg snakker ikke
norsk"* and *"jeg snakker flytende norsk"* sit at ~0.9 cosine under any
multilingual sentence encoder, so a dense-retrieval surface returns near-identical
rankings for both. The repo can demonstrate that on its own corpus. It cannot
demonstrate that **FINN's live product** has the problem — only a human running
the real queries can.

## The one methodological insight that makes this usable

**nDCG is not comparable across corpora. CVR@10 is.**

Constraint Violation Rate@10 is a property of the returned list *relative to the
constraint the user stated* — "of the ten ads this system showed me, how many
require a language I said I don't speak?" That question is answerable for any
system on any corpus, so a FINN row can sit legitimately beside the proposed
system's row **on CVR@10 only**, with nDCG left blank and a footnote saying why.

Without this, a hand-recorded CSV is an anecdote. With it, it is evidence.

## Hard rules

1. **No automation. None.** Ordinary browser, by hand. `finn.no/robots.txt`
   requires written permission for automated crawling under Norwegian copyright
   law and disallows `/job/` for GPTBot. See `DECISIONS.md` D1.
2. **Logged out, and in a private window**, so personalisation cannot influence
   the ranking. Note the browser and OS in `results.csv`.
3. **Paste the persona text verbatim** from `eval/personas.yaml`. Do not
   rephrase, do not shorten, do not "help" the search.
4. **Record what you see, not what you expected.** If the top-10 looks good,
   that is the finding. A baseline that loses on every query invites the question
   of whether the queries were chosen to make it lose.
5. **Judge each result yourself, before scoring anything.** Your own grade and
   violation flag, not the extractor's — see `eval/JUDGING_PROTOCOL.md`.
6. **One sitting per persona pair.** Run the "speaks Norwegian" and "does not
   speak Norwegian" variants back to back, so drift in the live index cannot be
   mistaken for the effect being measured.

## What to record, per query

Into `eval/manual_finn_baseline/results.csv`:

| column | meaning |
|---|---|
| `persona_id` | e.g. `p5_no_norsk` — must match `personas.yaml` |
| `pair_id` | links the Norwegian / no-Norwegian variants |
| `run_at` | ISO timestamp, local time, to the minute |
| `query_text_sha256` | first 12 chars of the sha256 of the pasted text — proves the text was not edited between runs |
| `rank` | 1–10 |
| `title`, `employer`, `url` | as displayed |
| `grade` | 0–3, per `JUDGING_PROTOCOL.md` |
| `violates` | `true`/`false` — does this ad require a language the persona said they do not speak? |
| `violated_constraint` | which one, in words |
| `constraint_evidence` | the sentence in the ad that shows it, quoted |
| `notes` | anything odd: a promoted listing, a deleted ad, an obvious duplicate |

And a screenshot per query into `screenshots/<persona_id>_<YYYYMMDD>.png`,
showing the query box and the top-10 together.

## Scope, and what it costs you

Ten personas, five of them paired → **15 query instances**. Ten results each.
About **one hour**, most of it judging rather than searching.

If time is short, the five PAIRED personas are the ones that carry the argument —
they are the only ones that measure whether stating the constraint changes
anything. Run those ten first.

## Two honest caveats to carry into the README

- **n = 15 queries.** Bootstrap intervals on a CVR difference will be wide. Report
  them wide rather than reporting a point estimate that implies precision.
- **A single point in time.** FINN's index and ranking change; this is one
  observation, dated. It is not a claim about the product's behaviour in general,
  and the README should say so in those words.

## Not to be done

Do not adjust the personas after seeing FINN's results. If a persona turns out to
be badly worded, record that in `notes`, finish the run, and change it afterwards
with the change recorded in git — never silently mid-measurement.
