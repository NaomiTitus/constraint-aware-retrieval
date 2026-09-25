# Judging protocol

**Committed before any ablation is run.** The git timestamp on this file is the
pre-registration. It costs twenty minutes to write and it is the difference
between a measurement and a number chosen after the fact.

Applies to both judges: the human grading the manual FINN baseline, and the LLM
judge grading the pooled results.

---

## The rule that matters more than the grades

> **The judge sees the raw advertisement text and the persona's stated
> constraints. The judge NEVER sees the extractor's output.**

If the judge is shown the extracted facets, `CVR@10` measures the extractor
agreeing with itself and the headline number is worthless. This is Risk 1 in the
plan and it is the single way this project could produce a confidently wrong
result.

Enforced three ways, because a rule this important should not rest on discipline:
- by construction — `eval/judge_llm.py` imports `ads.description_text` and
  `personas.yaml`, and never `ad_facets`;
- by test — `tests/unit/test_judge_isolation.py` asserts no facet field name
  appears in the rendered prompt;
- by this file, dated.

## Relevance grade, 0–3

Judged against the persona's stated skills and preferences, **ignoring language**
— language is scored separately as a violation, and mixing the two makes it
impossible to tell a bad match from an inaccessible one.

| grade | meaning |
|---|---|
| **3** | Squarely the job this person is looking for. Right occupation, right level. |
| **2** | Plausible. Right field, but the level, specialism or contract type is off. |
| **1** | Adjacent. Shares skills or sector, but this person would not apply. |
| **0** | Irrelevant. Different occupation entirely. |

Grade the **advertisement as written**, not the employer's likely flexibility.
"They would probably consider her" is grade 1, not 3.

## Violation flag — judged INDEPENDENTLY of the grade

> `violates = true` when the advertisement requires something the persona
> explicitly said they do not have.

Almost always language. Record it as `true` only on evidence **in the
advertisement text**, and quote the sentence in `constraint_evidence`. Then:

- A grade-3 ad can be a violation. That combination is the most informative row
  in the whole dataset: a perfect match the seeker cannot take.
- **Silence is not a violation.** An advertisement that says nothing about
  language is `false`, however Norwegian it looks. 74.6% of the corpus says
  nothing, and inferring a requirement from silence is exactly the error the
  extractor is built to avoid — a judge that does it cannot detect the extractor
  doing it.
- **`norsk autorisasjon` is NOT a language violation.** Professional
  authorisation to practise is a licensing requirement. If the persona lacks it,
  that is a violation of `authorisation`, recorded as such — never as language.
- A document-language clause ("dokumentasjon må være på et skandinavisk språk
  eller engelsk", 254 corpus ads) is **not** a requirement on the applicant. It
  states what language the paperwork may be in.
- `norsk eller engelsk` is **not** a violation for an English speaker. It is a
  disjunction: either language suffices.

## Headline metric

**CVR@10** — Constraint Violation Rate@10: mean over queries of
`#{top-10 with violates=true} / 10`.

**ΔCVR_paired** = CVR(no-Norwegian variant) − CVR(Norwegian variant). A
bi-encoder baseline should sit near zero: it did not react to the constraint. The
proposed system should be strongly negative.

Reported alongside, never instead: nDCG@10 (gains `2^grade − 1`), MRR@10,
Recall@50 with its pooling caveat, and Kendall τ_b between paired variants —
where **low τ is good**, because it means the system responded to the negation.

## Prompt revisions

**One revision of the LLM-judge prompt is permitted, BEFORE the human
calibration set is labelled. Zero after.**

If Cohen's κ against the human labels comes in low, that is the finding — report
it and downweight the LLM-judged numbers. Re-prompting until κ looks good is
p-hacking the judge, and it is undetectable in the output.

Targets, stated in advance so they cannot be moved: quadratic-weighted κ > 0.6 on
the 4-grade scale, κ > 0.75 on the binary violation flag.

## Pooling

TREC depth-10 pooling: the union of the top-10 from every system on the ablation
ladder, deduplicated. Judged blind to which system contributed a result.

Pooling bias is real and goes in the limitations: a document no system retrieved
is never judged, so Recall@50 is an upper bound. Depth 10 rather than 20 was
chosen deliberately — it roughly halves the judge cost and only mildly worsens
that bias.

## Ties and uncertainty

If a grade is genuinely uncertain, record the LOWER grade and say why in `notes`.
Systematic optimism inflates nDCG for every system equally but destroys the
comparison against the manual FINN baseline, where a human is grading a different
corpus.
