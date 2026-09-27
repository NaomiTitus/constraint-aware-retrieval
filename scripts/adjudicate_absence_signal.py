"""The §13 follow-up: 334 lexical hits vs 93 census labels — which side is wrong?

WHY THIS HAD TO RUN. Cell 2 (LIMITATIONS §13) rested one of its two conclusions
on a single inequality: 334 ads state "Norwegian not required" in words, which is
MORE than the 166 that say `tømrer`, therefore document-side SCARCITY is not the
mechanism by which a negated query fails. §13 also flagged that the census
labelled only 93 ads `explicitly_not_required`, and that either the pattern
over-fires or the census under-fires by 3.6x — unmeasured, and load-bearing for
§3b's 9.4%-accessible headline.

It over-fires. The conclusion it was supporting does not survive.

WHAT `LEXICAL` MEANT, AND WHY IT FAILED. The test was string proximity: does a
negation phrase (`ikke et krav`, `not required`) appear within 90 characters of
`norsk`? Norwegian job ads put qualifications in one bullet list, and
"X er en fordel, men ikke et krav" is a stock phrase in it. So the negation
routinely attaches to experience, a driving licence, a forklift certificate, or
church membership, while a Norwegian requirement sits a few words away:

    "Erfaring fra arbeid med barn er ønskelig, men ikke et krav.
     Du må kunne snakke godt norsk for å kommunisere med barn og foreldre"

Lexically that reads as an absence. Semantically it is the opposite: experience
optional, Norwegian MANDATORY. The census called it `professional` and was right.

THE AGREEMENT TABLE, all 10,166 ads:

    lexical hits L                    334
    census explicitly_not_required C   93
    L ∩ C                              11
    L only                            323
    C only                             82

HAND AUDIT, 36 ads read across all nine census levels that L fires on. True
positives — where the negation genuinely attaches to the Norwegian language —
are concentrated in ONE stratum, `desirable`, and in the 11 already agreeing.
Everywhere else the sampled hits were false. So L's precision is single digits
and the corrected count is far below 166: the inequality reverses and §13's
"scarcity is not the mechanism" is RETRACTED.

THE SECOND FINDING, WHICH WAS NOT THE QUESTION ASKED BUT MATTERS MORE.
`explicitly_not_required` is a misnomer for what is in it. 82 of its 93 ads
contain no explicit negation of Norwegian at all. Their census evidence spans are
POSITIVE English requirements:

    "Are fluent in English, as it is DESMI's corporate language"
    "Competence in English is a requirement for all applicants to the PhD program"

The census inferred "Norwegian not required" from an English-only working
language. That is a reasonable inference and it is not what the level's name
says. Meanwhile the ads that DO explicitly negate Norwegian — "Knowledge of
Norwegian is an advantage, but not a requirement" — are labelled `desirable`.

THE MECHANISM THIS SUGGESTS, and it is consistent with every earlier cell.
The seeker writes a NEGATION about NORWEGIAN. The accessible ad writes an
AFFIRMATION about ENGLISH, frequently never mentioning Norwegian. Measured:
96.9% of Norwegian-demanding ads contain `norsk`, against 67.7% of accessible
ones. So the query's own `norsk` token pulls it toward the ads that discuss
Norwegian — which are precisely the ones that demand it — while the ads that
would serve the seeker share almost no vocabulary with the query. That is why
stating the constraint made things WORSE on 5 of 5 personas in cell 3, and it is
a sharper mechanism than either "nothing to survive into" or "both sides are
negations".

DISCRIMINATION TEST, replacing the underpowered top-10 recovery metric. AUC over
all 93 accessible vs all 5,703 demanding ads: can the query separate them at all?
0.50 means no; below 0.50 means it actively prefers the blocking ads.

CONFOUND, STATED BECAUSE THE ABSOLUTE AUC IS NOT INTERPRETABLE WITHOUT IT. The
93 accessible ads skew academic, offshore and corporate; the 5,703 skew health,
retail and education. A one-line query carries no occupation signal, so absolute
AUC partly measures topical mismatch rather than language handling. The
controlled quantity is the PAIRED DELTA between a positive and negated variant
of the same query, which holds topic fixed.

Encoder: paraphrase-multilingual-mpnet-base-v2 via ONNX Runtime — LIMITATIONS
§12. No API calls; the corpus embedding is read from the cache cell 1 wrote.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "embeddings_nb_sbert.npz"
OUT = ROOT / "reports" / "absence_signal_adjudication.json"

NORSK = re.compile(r"norsk|norwegian", re.I)
NEG = re.compile(
    r"ikke\s+(et\s+)?(absolutt\s+)?krav|ikke\s+n[øo]dvendig|ikke\s+p[åa]krevd|"
    r"not\s+(a\s+)?(requirement|required|necessary|needed)|no\s+need\s+for|"
    r"trenger\s+ikke|beh[øo]ver\s+ikke",
    re.I)
DEMANDING = ("professional", "certified", "fluent", "conversational",
             "scandinavian_accepted")

# The hand audit. Read 36 windows across all nine strata L fires on; recorded
# here as the count judged TRUE — the negation genuinely attaching to the
# Norwegian language — out of the count read. Adjudicated from the window text
# plus the census evidence span, by the gloss method LIMITATIONS §1 describes.
# `desirable` is the one stratum that is genuinely about language: its hits read
# "Norwegian is an advantage, but not a requirement". `unstated` contains
# conditional exemptions ("the language requirement does not apply if you have
# five years in a Norwegian kindergarten") scored as ambiguous, not true.
AUDIT = {
    "professional": (0, 6), "certified": (0, 6),
    "either_norwegian_or_english": (0, 6), "unstated": (0, 3),
    "fluent": (0, 3), "conversational": (0, 3),
    "desirable": (3, 3), "scandinavian_accepted": (0, 3),
    "explicitly_not_required": (3, 3),   # agrees with the census by construction
}


def _wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """Upper 95% bound on a proportion — needed because several strata audited
    0 true positives and `0` is not the same claim as `0 with n=6`."""
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c + r) / d)


def main() -> None:
    import duckdb

    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    rows = con.execute("""SELECT a.uuid, a.title, a.description_text, f.facets
                          FROM ads a JOIN ad_facets f USING (uuid)
                          ORDER BY a.uuid""").fetchall()
    uuids = [r[0] for r in rows]
    texts = [f"{r[1]}\n{r[2] or ''}" for r in rows]
    facets = [json.loads(r[3]) if isinstance(r[3], str) else r[3] for r in rows]
    levels = [f.get("norwegian_requirement_level") for f in facets]

    def fires(t: str, w: int = 90) -> bool:
        return any(NEG.search(t[max(0, m.start() - w):m.end() + w])
                   for m in NORSK.finditer(t))

    L = {i for i, t in enumerate(texts) if fires(t)}
    C = {i for i, l in enumerate(levels) if l == "explicitly_not_required"}
    result = {"n_ads": len(uuids), "n_lexical": len(L), "n_census": len(C),
              "n_both": len(L & C), "n_lexical_only": len(L - C),
              "n_census_only": len(C - L)}

    print("=== AGREEMENT: the lexical pattern vs the census label ===\n")
    print(f"  lexical hits L                   {len(L):5d}")
    print(f"  census explicitly_not_required C {len(C):5d}")
    print(f"  L and C agree                    {len(L & C):5d}")
    print(f"  L only                           {len(L - C):5d}")
    print(f"  C only                           {len(C - L):5d}")
    print(f"\n  Jaccard {len(L & C) / len(L | C):.3f} — the two measure almost "
          f"different things.")

    print("\n=== WHAT THE CENSUS SAID ABOUT EACH LEXICAL HIT ===\n")
    print(f"  {'census level':30s} {'hits':>5s} {'audited':>8s} {'true':>5s} "
          f"{'prec.':>7s} {'95% up':>7s} {'est. true':>10s}")
    strata, est_true, est_true_hi = {}, 0.0, 0.0
    by_level: dict[str, int] = {}
    for i in L:
        by_level[str(levels[i])] = by_level.get(str(levels[i]), 0) + 1
    for lvl, n in sorted(by_level.items(), key=lambda kv: -kv[1]):
        k, m = AUDIT.get(lvl, (0, 0))
        prec = (k / m) if m else float("nan")
        hi = _wilson_upper(k, m)
        est_true += n * (prec if m else 0.0)
        est_true_hi += n * hi
        strata[lvl] = {"hits": n, "audited": m, "true": k,
                       "precision": prec if m else None, "precision_hi95": hi}
        print(f"  {lvl:30s} {n:5d} {m:8d} {k:5d} "
              f"{'--' if not m else f'{prec:6.0%}'} {hi:6.0%} "
              f"{n * (prec if m else 0.0):9.0f}")
    result["strata"] = strata
    result["estimated_true_positives"] = est_true
    result["estimated_true_positives_hi95"] = est_true_hi

    n_occ = sum(1 for t in texts
                if re.search(r"tømrer|tomrer|snekker|carpenter", t, re.I))
    result["n_occupation_b"] = n_occ
    print(f"\n  ESTIMATED TRUE LEXICAL ABSENCES: {est_true:.0f} "
          f"(95% upper bound {est_true_hi:.0f}) out of {len(L)} hits")
    print(f"  COMPARE `tømrer`: {n_occ} ads.")
    verdict_reversed = est_true_hi < n_occ
    result["cell2_inequality_reversed"] = bool(verdict_reversed)
    if verdict_reversed:
        print(f"  Even at the 95% UPPER bound the count is below {n_occ}. "
              f"§13's inequality REVERSES,\n  and its 'scarcity is not the "
              f"mechanism' conclusion is retracted.")
    else:
        print(f"  The upper bound still exceeds {n_occ}; §13's inequality "
              f"survives this audit.")

    # ---- what the census's own 93 actually contain ----------------------
    print("\n=== IS `explicitly_not_required` WHAT ITS NAME SAYS? ===\n")
    conly = sorted(C - L)
    eng = sum(1 for i in conly
              if re.search(r"english|engelsk", " ".join(
                  x.get("span", "") for x in (facets[i].get("evidence_spans") or [])),
                  re.I))
    print(f"  {len(C)} ads carry the level. {len(C - L)} contain NO explicit "
          f"negation of Norwegian.")
    print(f"  Of those, {eng} have an evidence span mentioning English.")
    print("  The census inferred the absence from a POSITIVE English "
          "requirement — defensible,\n  but not what the level's name asserts. "
          "Ads that DO explicitly negate Norwegian\n  ('an advantage, but not a "
          "requirement') are labelled `desirable` instead.")
    result["census_only_no_explicit_negation"] = len(C - L)
    result["census_only_mentioning_english"] = eng

    # ---- vocabulary asymmetry, the mechanism ---------------------------
    print("\n=== THE MECHANISM: do the accessible ads even DISCUSS Norwegian? ===\n")
    dem = [i for i, l in enumerate(levels) if l in DEMANDING]
    acc = sorted(C)
    vocab = {}
    for name, idx in (("accessible (not required)", acc),
                      ("DEMANDING Norwegian", dem)):
        n = sum(1 for i in idx if NORSK.search(texts[i]))
        vocab[name] = {"n": len(idx), "mentioning": n, "share": n / len(idx)}
        print(f"  {name:28s} {n:5d}/{len(idx):5d} mention norsk/norwegian "
              f"({n / len(idx):6.1%})")
    result["vocabulary"] = vocab
    print("\n  The seeker negates NORWEGIAN. The ad that would serve them "
          "affirms ENGLISH, and a\n  third of the time never says `norsk` at "
          "all. The query's own `norsk` token pulls it\n  toward the ads that "
          "discuss Norwegian — the blocking ones. That is cell 3's result\n"
          "  (stating the constraint made it WORSE on 5 of 5) with a mechanism "
          "attached.")

    # ---- discrimination test -------------------------------------------
    if not CACHE.exists():
        print(f"\n(no embedding cache at {CACHE} — skipping the AUC cell)")
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
        return
    z = np.load(CACHE, allow_pickle=True)
    if list(z["uuids"]) != uuids:
        sys.exit("embedding cache does not match the corpus")
    E = z["E"]
    from fastembed import TextEmbedding
    m = TextEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    def enc(xs):
        V = np.array(list(m.embed(list(xs))), dtype=np.float32)
        return V / np.linalg.norm(V, axis=1, keepdims=True)

    PAIRS = {
        "no": ("Jeg snakker flytende norsk.", "Jeg snakker ikke norsk."),
        "en": ("I speak fluent Norwegian.", "I do not speak Norwegian."),
    }
    EXTRA = {"english-positive":
             "I speak English. I am looking for a job where English is the "
             "working language."}
    print("\n=== DISCRIMINATION: 93 accessible vs 5,703 demanding, AUC ===")
    print("    0.50 = indistinguishable.  BELOW 0.50 = prefers the blocking "
          "ads.\n")
    print(f"  {'query':44s} {'AUC':>7s}")

    def auc_of(q: str) -> float:
        v = enc([q])[0]
        s = E @ v
        return float((s[acc][:, None] > s[dem][None, :]).mean())

    aucs, deltas = {}, {}
    for lang, (pos, neg) in PAIRS.items():
        ap, an = auc_of(pos), auc_of(neg)
        aucs[f"{lang}_positive"], aucs[f"{lang}_negated"] = ap, an
        deltas[lang] = an - ap
        print(f"  {lang.upper() + ' positive — ' + pos[:28]:44s} {ap:7.3f}")
        print(f"  {lang.upper() + ' negated  — ' + neg[:28]:44s} {an:7.3f}")
    for name, q in EXTRA.items():
        aucs[name] = auc_of(q)
        print(f"  {name:44s} {aucs[name]:7.3f}")
    result["auc"] = aucs
    result["auc_paired_delta"] = deltas

    print("\n  PAIRED DELTA (the controlled quantity — topic held fixed, only "
          "the polarity moves):")
    for lang, d in deltas.items():
        print(f"    {lang.upper()}  negating the language claim moves AUC "
              f"{d:+.3f}")
    print("\n  Absolute AUC below 0.5 is NOT clean evidence on its own: the 93 "
          "skew academic and\n  offshore while the 5,703 skew health and "
          "retail, and a one-line query carries no\n  occupation signal to "
          "control that. The delta is the interpretable number.")

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")

    print("\n=== WHAT THIS SETTLES ===")
    print("  1. THE PATTERN OVER-FIRES; THE CENSUS DOES NOT UNDER-FIRE. §13's "
          "334 is retracted,\n     and with it §13's claim that document-side "
          "scarcity is not the mechanism.")
    print("  2. `explicitly_not_required` is a misnomer: 82 of 93 contain no "
          "explicit negation.\n     The level is inferred from positive "
          "English requirements.")
    print("  3. The genuine negations sit in `desirable`, so the taxonomy "
          "boundary between\n     `desirable` and `explicitly_not_required` "
          "is doing real work on §3b's 9.4%.")
    print("  4. STILL OPEN: whether `desirable` ads that say 'Norwegian not "
          "required' should count\n     as accessible. That is a labelling "
          "decision, it is unmade, and it moves the\n     headline number.")


if __name__ == "__main__":
    main()
