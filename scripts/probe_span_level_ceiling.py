"""Does embedding the REQUIREMENT SENTENCE instead of the whole ad fix it?

THE OBJECTION THIS ANSWERS, and it is a fair one. Every probe so far embedded the
whole advertisement — title plus body. An ad states fifteen facets: occupation,
location, extent, licences, authorisation, clearance, seniority, language. Mean
pooling averages all of them into one vector, so "the encoder cannot find the
accessible ads" may be measuring DOCUMENT NOISE rather than anything about
negation. Language is one requirement among many and it is a small share of the
tokens.

The fix is to give the bi-encoder its best possible shot: retrieve against the
single sentence the census already extracted as the language evidence, instead of
against 1,200 characters of unrelated prose. If the encoder separates
`norsk er ikke et krav` from `du må beherske norsk` once the noise is gone, then
the earlier numbers were confounded and the thesis must be restated — what breaks
would be dilution in long documents, not negation as such.

This is deliberately a STEEL-MAN. Span-level indexing is a real alternative
architecture to the project's typed-constraint stage, and it deserves to be ruled
in or out on evidence rather than ignored.

THREE CELLS.

  A. THE SILENCE CENSUS, across every constraint facet, not just language. The
     generalised claim: an advertisement states the requirements it HAS and never
     enumerates the ones it LACKS, so absence is expressed by absent text.
     visa_sponsorship is 99.4% unstated; relocation_support has EIGHT ads that
     say it is not offered. If absence is silence, no encoder of any size can
     represent it, because there is nothing to encode.

  B. SPAN-LEVEL DISCRIMINATION. Re-run the 93-accessible vs 5,703-demanding AUC
     with each ad represented by its language evidence span alone. Compare
     against the whole-ad number (0.393 negated, 0.441 positive). The paired
     delta between polarities is the controlled quantity; absolute AUC is
     confounded by sector, as LIMITATIONS §13 records.

  C. THE CEILING THAT SURVIVES ANY ENCODER. Span-level retrieval needs a span.
     An ad blocked by SILENCE has no sentence about language to extract, so it
     cannot be reached by better embedding at all — only by a decision about how
     to treat silence, which is exactly what `constraints.py` makes tunable.
     §3b measured that population at 3,507 ads, 34.5% of the corpus.

WHAT WOULD FALSIFY THE PROJECT'S THESIS. If span-level AUC rises well above 0.5
and the polarity delta turns positive — negating the query HELPING, as it should
— then a bi-encoder over extracted spans handles this and the separate constraint
stage is unnecessary for the cases where a span exists. That would be a real
result against the architecture and it must be reported if it appears.

Encoder: paraphrase-multilingual-mpnet-base-v2 via ONNX Runtime — LIMITATIONS
§12. Span embeddings are cached; no API calls.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WHOLE_CACHE = ROOT / "data" / "embeddings_nb_sbert.npz"
SPAN_CACHE = ROOT / "data" / "embeddings_spans.npz"
OUT = ROOT / "reports" / "span_level_ceiling.json"

MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEMANDING = ("professional", "certified", "fluent", "conversational",
             "scandinavian_accepted")

# Facets whose "absent" state the census records explicitly, so silence can be
# counted against an explicit alternative rather than assumed.
SILENCE_FACETS = {
    "visa_sponsorship": ("unstated", "explicitly_not_offered"),
    "relocation_support": ("unstated", "explicitly_not_offered"),
    "stated_working_language": ("unstated", None),
    "application_language": ("unstated", None),
    "norwegian_requirement_level": ("unstated", "explicitly_not_required"),
}

PAIRS = {
    "no": ("Jeg snakker flytende norsk.", "Jeg snakker ikke norsk."),
    "en": ("I speak fluent Norwegian.", "I do not speak Norwegian."),
}


def main() -> None:
    import duckdb
    from fastembed import TextEmbedding

    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    rows = con.execute("""SELECT a.uuid, a.title, a.description_text, f.facets
                          FROM ads a JOIN ad_facets f USING (uuid)
                          ORDER BY a.uuid""").fetchall()
    uuids = [r[0] for r in rows]
    facets = [json.loads(r[3]) if isinstance(r[3], str) else r[3] for r in rows]
    levels = [f.get("norwegian_requirement_level") for f in facets]
    spans = [" ".join(" ".join(x.get("span", "").split())
                      for x in (f.get("evidence_spans") or [])).strip()
             for f in facets]

    result = {"model": MODEL, "n_ads": len(uuids)}

    # ---- CELL A: the silence census, across facets ----------------------
    print("=== CELL A: ACROSS EVERY FACET, 'NOT REQUIRED' IS SILENCE ===\n")
    print("  An ad states the requirements it HAS. It does not enumerate the "
          "ones it LACKS.\n")
    print(f"  {'facet':30s} {'silent':>8s} {'share':>7s} "
          f"{'explicitly absent':>18s} {'share':>7s}")
    sil = {}
    for key, (silent_val, explicit_val) in SILENCE_FACETS.items():
        c = collections.Counter(str(f.get(key)) for f in facets)
        n_sil = c.get(str(silent_val), 0)
        n_exp = c.get(str(explicit_val), 0) if explicit_val else None
        sil[key] = {"silent": n_sil, "silent_share": n_sil / len(facets),
                    "explicit_absent": n_exp}
        exp_s = "--" if n_exp is None else f"{n_exp:d}"
        exp_sh = "--" if n_exp is None else f"{n_exp / len(facets):6.2%}"
        print(f"  {key:30s} {n_sil:8d} {n_sil / len(facets):6.1%} "
              f"{exp_s:>18s} {exp_sh:>7s}")
    result["silence_census"] = sil
    print("\n  There is no vector for a sentence that was never written. This is "
          "not a limitation\n  of THIS encoder — it is a property of the data, "
          "and it holds for every facet.")

    # ---- CELL C first (free): the ceiling -------------------------------
    acc = [i for i, l in enumerate(levels) if l == "explicitly_not_required"]
    dem = [i for i, l in enumerate(levels) if l in DEMANDING]
    unst = [i for i, l in enumerate(levels) if l == "unstated"]
    n_acc_span = sum(1 for i in acc if spans[i])
    n_unst_span = sum(1 for i in unst if spans[i])
    print("\n=== CELL C: THE CEILING SPAN-LEVEL RETRIEVAL CANNOT PASS ===\n")
    print(f"  ads with a language evidence span at all: "
          f"{sum(1 for s in spans if s)} of {len(spans)} "
          f"({sum(1 for s in spans if s) / len(spans):.1%})")
    print(f"  of the {len(acc)} `explicitly_not_required`: {n_acc_span} have a "
          f"span ({n_acc_span / max(len(acc), 1):.0%})")
    print(f"  of the {len(unst)} `unstated`:              {n_unst_span} have a "
          f"span ({n_unst_span / max(len(unst), 1):.0%})")
    print("\n  An ad that never mentions language has no sentence to extract "
          "and no sentence to\n  embed. Span-level indexing cannot reach it by "
          "construction — only a DECISION about\n  how to treat silence can, "
          "which is what `retrieval/constraints.py` makes tunable.")
    result["ceiling"] = {"ads_with_span": sum(1 for s in spans if s),
                         "accessible_with_span": n_acc_span,
                         "unstated_total": len(unst),
                         "unstated_with_span": n_unst_span}

    # ---- CELL B: span-level discrimination ------------------------------
    need = sorted(set(acc) | set(dem))
    span_texts = [spans[i] if spans[i] else "" for i in need]
    have = [j for j, t in enumerate(span_texts) if t]
    print(f"\n=== CELL B: SPAN-LEVEL DISCRIMINATION ===\n")
    print(f"  {len(acc)} accessible + {len(dem)} demanding = {len(need)} ads; "
          f"{len(have)} have a span.")

    m = TextEmbedding(model_name=MODEL)

    def enc(xs):
        V = np.array(list(m.embed(list(xs))), dtype=np.float32)
        return V / np.linalg.norm(V, axis=1, keepdims=True)

    if SPAN_CACHE.exists():
        z = np.load(SPAN_CACHE, allow_pickle=True)
        if list(z["keys"]) == [uuids[i] for i in need]:
            S = z["S"]
            print(f"  span embeddings from cache: {S.shape}")
        else:
            S = None
    else:
        S = None
    if S is None:
        print(f"  embedding {len(have)} spans…", flush=True)
        S = np.zeros((len(need), 768), dtype=np.float32)
        V = enc([span_texts[j] for j in have])
        for j, v in zip(have, V):
            S[j] = v
        np.savez_compressed(SPAN_CACHE, S=S,
                            keys=np.array([uuids[i] for i in need]))
        print(f"  cached -> {SPAN_CACHE.name}")

    pos_local = [need.index(i) for i in acc if spans[i]]
    dem_local = [need.index(i) for i in dem if spans[i]]
    print(f"  usable: {len(pos_local)} accessible spans vs "
          f"{len(dem_local)} demanding spans\n")

    zw = np.load(WHOLE_CACHE, allow_pickle=True)
    assert list(zw["uuids"]) == uuids
    E = zw["E"]

    def auc(scores, a_idx, b_idx):
        a, b = scores[a_idx], scores[b_idx]
        return float((a[:, None] > b[None, :]).mean())

    print(f"  {'query':40s} {'whole-ad':>9s} {'span-level':>11s} {'change':>8s}")
    table, deltas = {}, {}
    for lang, (pos, neg) in PAIRS.items():
        for pol, q in (("positive", pos), ("negated", neg)):
            v = enc([q])[0]
            a_whole = auc(E @ v, [i for i in acc if spans[i]],
                          [i for i in dem if spans[i]])
            a_span = auc(S @ v, pos_local, dem_local)
            table[f"{lang}_{pol}"] = {"whole_ad": a_whole, "span": a_span}
            print(f"  {lang.upper() + ' ' + pol + ' — ' + q[:22]:40s} "
                  f"{a_whole:9.3f} {a_span:11.3f} {a_span - a_whole:+8.3f}")
        d_whole = (table[f"{lang}_negated"]["whole_ad"]
                   - table[f"{lang}_positive"]["whole_ad"])
        d_span = (table[f"{lang}_negated"]["span"]
                  - table[f"{lang}_positive"]["span"])
        deltas[lang] = {"whole_ad": d_whole, "span": d_span}
    result["auc"] = table
    result["polarity_delta"] = deltas

    print("\n  PAIRED POLARITY DELTA — negated minus positive, topic held fixed.")
    print("  A bi-encoder that handled negation would make this POSITIVE: "
          "saying 'not' should\n  move you TOWARD the ads that do not require "
          "it.\n")
    print(f"  {'':6s} {'whole-ad':>10s} {'span-level':>12s}")
    for lang, d in deltas.items():
        print(f"  {lang.upper():6s} {d['whole_ad']:+10.3f} {d['span']:+12.3f}")

    span_helps = all(d["span"] > 0 for d in deltas.values())
    span_better = all(d["span"] > d["whole_ad"] for d in deltas.values())
    # SIGN IS NOT A RESULT. A correctly-signed delta of +0.03 on a metric whose
    # absolute value is 0.14 describes an index that ranks the right ads in the
    # bottom fifth. Require the span index to be USABLE before crediting it.
    span_auc_usable = all(v["span"] > 0.5 for v in table.values())
    span_delta_material = all(abs(d["span"]) >= 0.05 for d in deltas.values())
    result["span_fixes_polarity"] = bool(span_helps)
    result["span_improves_polarity"] = bool(span_better)
    result["span_auc_usable"] = bool(span_auc_usable)
    result["span_delta_material"] = bool(span_delta_material)
    worst_span = min(v["span"] for v in table.values())
    best_span = max(v["span"] for v in table.values())

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")

    print("\n=== VERDICT ===")
    if span_helps and span_auc_usable:
        print("  THE OBJECTION IS SUSTAINED AND THE THESIS MUST BE RESTATED.")
        print("  On extracted spans the polarity delta is positive AND the "
              "index is usable, so a\n  span-level bi-encoder handles "
              "negation for ads that have a span.")
    elif span_helps:
        print("  PARTLY SUSTAINED, AND THE SIGN IS NOT THE STORY.")
        print(f"    The polarity delta DOES flip positive on spans "
              f"({deltas['no']['span']:+.3f} NO, "
              f"{deltas['en']['span']:+.3f} EN),")
        print("    so negation is WEAKLY representable once document noise is "
              "removed. Document noise\n    was a real confound and the "
              "objection was correct to raise it.")
        print(f"    BUT absolute AUC COLLAPSES from "
              f"{max(v['whole_ad'] for v in table.values()):.3f} whole-ad to "
              f"{best_span:.3f} at span level.")
        print(f"    An index like this ranks the accessible ads in the bottom "
              f"fifth ({worst_span:.2f}-{best_span:.2f}).")
        print("    Removing the job content removed the ONLY material the "
              "query and the ad shared:\n    the accessible ad's span talks "
              "about ENGLISH, the query talks about NORWEGIAN.\n    Span-level "
              "indexing sharpens the vocabulary mismatch it was meant to fix.")
        if not span_delta_material:
            print("    The delta is also below the 0.05 materiality floor, so "
                  "it is a direction, not\n    a usable signal.")
        print("\n  SO THE ARCHITECTURE CONCLUSION STANDS, FOR A CORRECTED "
              "REASON. Negation is not\n  literally unrepresentable — a small "
              "correctly-signed signal exists. It is far too\n  weak to "
              "overcome the vocabulary gap between a seeker negating Norwegian "
              "and an ad\n  affirming English. A typed predicate does not have "
              "to overcome anything.")
    else:
        print("  THE OBJECTION DOES NOT RESCUE THE BI-ENCODER. Cleaning the "
              "document side to a single\n  requirement sentence does not make "
              "the negation work — the polarity delta stays\n  negative.")
    print(f"\n  AND THE CEILING IS UNMOVED EITHER WAY: {len(unst)} ads "
          f"({len(unst) / len(uuids):.1%}) say nothing about\n  language at "
          f"all, and only {n_unst_span} of them have any span to embed. No "
          f"encoder reaches\n  that population at any granularity, because "
          f"there is no text. It needs a policy —\n  which is what "
          f"`constraints.py` makes tunable, and why severity for silence is "
          f"0.5\n  rather than 1.0.")


if __name__ == "__main__":
    main()
