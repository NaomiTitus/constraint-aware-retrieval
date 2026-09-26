"""Cell 1b: does the negation get DILUTED as the query grows?

WHY THIS EXISTS. Cell 1 reports one number — paired persona queries sit at
cosine 0.914 — and that number asserts the problem without showing its
mechanism. The mechanism is mean pooling: a transformer emits one vector per
token and a sentence encoder averages them, so every token's influence is
about 1/n. Measured on the real tokeniser:

    "Jeg snakker ikke norsk."     7 tokens, `▁ikke` is 1 of them   = 14%
    the full p1 persona query    30 tokens, `▁ikke` is 1 of them   = 3.3%

If that is the explanation, the prediction is exact and falsifiable: hold the
language sentence fixed, pad the query with more ordinary detail, and the
cosine between the "I speak Norwegian" and "I do not speak Norwegian"
variants must climb monotonically toward 1.0. If it does not climb, mean
pooling is NOT the mechanism and the explanation is wrong.

Showing the mechanism is worth more than asserting the number, because it
also says what cannot fix it: a larger encoder still averages.

RESULT, and it DISCONFIRMED the hypothesis as first stated.

    NO  at 12 clauses:  negation 0.999   occupation 0.998   place 0.995
    EN  at 12 clauses:  negation 0.999   occupation 0.994   place 0.995

The curve rises monotonically in both languages (Spearman 0.989 and 0.901,
non-decreasing on 12 of 12 steps) — but the CONTROL rises just as fast. A
one-word content difference, nurse vs carpenter or Oslo vs Bergen, is erased
at the same rate. The margins are 0.004, which is noise.

So "mean pooling averages away the NEGATION" is too specific. What pooling
averages away is ANY single-clause difference, once the query is long enough
to be realistic. That is a broader claim about single-vector retrieval and it
is the one the write-up must make.

THE REFINED HYPOTHESIS, which this probe cannot settle and cell 2 can. The
dilution is symmetric; its CONSEQUENCE is not. "Nurse vs carpenter" vanishes
from the query vector yet survives in retrieval, because the corpus is
saturated with occupation signal — thousands of ads say `sykepleier`. "I do
not speak Norwegian" has nothing to survive into: almost no advertisement
states the ABSENCE of a requirement, so once the clause is averaged out of
the query there is no document-side signal left to recover it. If that is
right, the language pairs show HIGH ranking overlap while occupation pairs
show low overlap, at the same query-side cosine.

GROUNDING (STANDARDS.md §3.0). The padding is not invented prose. It is
clauses taken from the project's own frozen, pre-registered personas, with
their language sentences removed — so every clause is text a real seeker
in this evaluation actually wrote. No API calls; CPU only; seconds to run.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OUT = ROOT / "reports" / "negation_dilution.json"

LANG_RE = re.compile(r"\b(snakker|speak|spr[åa]k|language|norsk|norwegian)\b", re.I)

# The only thing that differs between the two variants, in each language.
VARIANTS = {
    "no": ("Jeg snakker flytende norsk.", "Jeg snakker ikke norsk."),
    "en": ("I speak fluent Norwegian.", "I do not speak Norwegian."),
}

# THE CONTROL, without which the rising curve proves nothing.
#
# Padding BOTH variants with identical text pushes cosine up mechanically —
# the shared content comes to dominate both means whatever the difference is.
# So a rise on its own is not evidence about negation.
#
# The question is comparative: does a NEGATION dilute faster than an ordinary
# CONTENT difference of similar surface size? These pairs differ by one
# content word — an occupation, a place — and are padded identically. If they
# survive padding better than the negation does, the dilution is specific to
# negation and the mechanism claim holds. If they collapse at the same rate,
# the effect is generic and the negation framing is wrong.
CONTROLS = {
    "no": {
        "occupation": ("Jeg er sykepleier.", "Jeg er tømrer."),
        "place": ("Jeg ser etter jobb i Oslo.", "Jeg ser etter jobb i Bergen."),
    },
    "en": {
        "occupation": ("I am a nurse.", "I am a carpenter."),
        "place": ("I am looking for work in Oslo.",
                  "I am looking for work in Bergen."),
    },
}


def _padding_clauses() -> dict[str, list[str]]:
    """Clauses from the frozen personas, language sentences stripped."""
    ps = yaml.safe_load((ROOT / "eval" / "personas.yaml")
                        .read_text(encoding="utf-8"))["personas"]
    out: dict[str, list[str]] = {"no": [], "en": []}
    for p in ps:
        q = " ".join(p["query"].split())
        # crude but sufficient: a query with æøå or `Jeg ` is the Norwegian one
        lang = "no" if re.search(r"[æøå]|^Jeg ", q) else "en"
        for s in re.split(r"(?<=[.!?])\s+", q):
            s = s.strip()
            if len(s) > 15 and not LANG_RE.search(s) and s not in out[lang]:
                out[lang].append(s)
    return out


def main() -> None:
    from fastembed import TextEmbedding
    m = TextEmbedding(model_name=MODEL)
    tok = m.model.tokenizer

    def enc(xs):
        V = np.array(list(m.embed(list(xs))), dtype=np.float32)
        return V / np.linalg.norm(V, axis=1, keepdims=True)

    pads = _padding_clauses()
    result = {"model": MODEL, "series": {}}

    for lang, (pos, neg) in VARIANTS.items():
        clauses = pads[lang]
        print(f"\n=== {lang.upper()} — {len(clauses)} real persona clauses available "
              f"as padding ===")
        print(f"  positive: {pos}")
        print(f"  negative: {neg}")
        print(f"\n  {'clauses':>7s} {'tokens':>7s} {'ikke/not share':>15s} {'cosine':>8s}")
        rows = []
        for k in range(0, min(len(clauses), 12) + 1):
            pad = " ".join(clauses[:k])
            a = (pad + " " + pos).strip()
            b = (pad + " " + neg).strip()
            va, vb = enc([a, b])
            cos = float(va @ vb)
            n_tok = len(tok.encode(b).tokens)
            share = 1.0 / n_tok
            print(f"  {k:7d} {n_tok:7d} {share:14.1%} {cos:8.3f}")
            rows.append({"n_clauses": k, "n_tokens": n_tok,
                         "one_token_share": share, "cosine": cos})
        result["series"][lang] = rows

        cos = [r["cosine"] for r in rows]
        rises = sum(1 for i in range(1, len(cos)) if cos[i] >= cos[i - 1] - 0.005)
        print(f"\n  cosine {cos[0]:.3f} -> {cos[-1]:.3f} over "
              f"{rows[0]['n_tokens']} -> {rows[-1]['n_tokens']} tokens")
        print(f"  non-decreasing on {rises} of {len(cos)-1} steps "
              f"(tolerance 0.005)")
        from scipy.stats import spearmanr
        rho = spearmanr([r["n_tokens"] for r in rows], cos).statistic
        print(f"  Spearman(tokens, cosine) = {rho:.3f}")
        result["series"][lang + "_spearman"] = float(rho)

        # --- the control: a CONTENT difference, padded identically ---------
        print(f"\n  CONTROL — one content word instead of a negation, "
              f"same padding:")
        print(f"  {'clauses':>7s} {'negation':>9s}", end="")
        for name in CONTROLS[lang]:
            print(f" {name:>11s}", end="")
        print()
        ctrl_rows = []
        for k in (0, 1, 2, 3, 6, 12):
            if k > len(clauses):
                continue
            pad = " ".join(clauses[:k])
            row = {"n_clauses": k, "negation": rows[k]["cosine"]}
            print(f"  {k:7d} {rows[k]['cosine']:9.3f}", end="")
            for name, (c1, c2) in CONTROLS[lang].items():
                va, vb = enc([(pad + " " + c1).strip(), (pad + " " + c2).strip()])
                c = float(va @ vb)
                row[name] = c
                print(f" {c:11.3f}", end="")
            print()
            ctrl_rows.append(row)
        result["series"][lang + "_control"] = ctrl_rows
        last = ctrl_rows[-1]
        others = [v for k2, v in last.items()
                  if k2 not in ("n_clauses", "negation")]
        print(f"\n  at {last['n_clauses']} clauses: negation {last['negation']:.3f} "
              f"vs content differences {[round(x,3) for x in others]}")
        if last["negation"] > max(others) + 0.002:
            print("  -> the NEGATION is more diluted than a content difference. "
                  "The mechanism claim holds.")
        else:
            print("  -> the negation is NOT specially diluted. The effect is "
                  "generic and the framing needs revising.")

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")
    print("\nREAD IT LIKE THIS. A rising curve means the encoder DOES see the\n"
          "negation — it is simply averaged away as the query grows. That is a\n"
          "property of pooling a whole query into one vector, so a better or\n"
          "larger encoder does not fix it; extracting the clause as a typed\n"
          "predicate BEFORE pooling does.")


if __name__ == "__main__":
    main()
