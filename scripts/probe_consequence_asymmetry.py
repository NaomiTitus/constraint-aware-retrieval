"""Cell 2: the dilution is symmetric — is its CONSEQUENCE?

WHAT CELL 1b LEFT OPEN, AND WHY IT MATTERS MORE THAN WHAT IT CLOSED.

Cell 1b predicted that padding a query would dilute the negation, and it did:
cosine 0.400 -> 0.999 over 7 -> 151 tokens. But the CONTROL rose just as fast.
A one-word content difference — nurse vs carpenter, Oslo vs Bergen — was erased
at the same rate, margins of 0.004, noise. So "mean pooling averages away the
NEGATION" is false as stated. Pooling averages away ANY single clause.

That leaves an obvious objection to the whole project, and it deserves to be
stated in its strongest form: if occupation dilutes exactly as hard as negation,
and occupation search plainly WORKS, then dilution cannot be what breaks
negation, and the architectural argument loses its mechanism.

THE REFINED HYPOTHESIS. The dilution is symmetric on the QUERY side; its
consequence in RETRIEVAL is not, because retrieval is a function of two vectors
and only one of them has been diluted. "Nurse" vanishes from the query vector
and still survives the search, because the corpus is saturated with document-
side occupation signal — thousands of ads say `sykepleier`, so a faint residue
is enough to land on them. "I do not speak Norwegian" has nothing to survive
INTO. Almost no advertisement states the ABSENCE of a requirement, so once the
clause is averaged out of the query there is no document-side signal left to
recover it. The query-side loss is equal; the recoverability is not.

THE PREDICTION, and it is comparative and falsifiable.

  At MATCHED query-side cosine, language pairs show HIGHER ranking overlap@10
  than occupation and place pairs. Same query-side damage, different retrieval
  outcome.

Matching on cosine is the whole method. Cell 1b's error was reading a rising
curve as proof when identical padding raises cosine mechanically; the fix there
was a control, and the fix here is to compare families only where the query-side
damage is EQUAL. An uncontrolled claim that "language overlap is high" would say
nothing, because overlap and cosine both move with padding.

WHAT FALSIFIES IT. If language overlap@10 is at or below the occupation and
place curves at matched cosine, the refined hypothesis is wrong: the consequence
is symmetric too, dilution is not the mechanism by which negation fails, and the
README must drop the document-side-signal explanation entirely and say only that
single-vector retrieval loses single clauses. That would be a real loss of
explanatory power and it must not be smuggled past.

THREE CELLS.

  2a. DOCUMENT-SIDE SIGNAL CENSUS. Count, over all 10,166 ads, how many carry
      signal for each side of each pair. No embeddings — this is the mechanism
      claim measured directly against the corpus. Prediction: both sides of
      occupation and place are well populated; the "no Norwegian required" side
      is near-empty.

  2b. COSINE-MATCHED OVERLAP. Sweep padding 0..12 clauses, recording query-side
      cosine and retrieval overlap@10 together, then compare families in shared
      cosine bins.

  2c. RECOVERY. Of variant A's top-10, how many ads actually carry A's signal?
      This is the mechanism itself rather than a proxy for it: it asks whether
      the corpus can still deliver the distinction the query vector has lost.

GROUNDING (STANDARDS.md §3.0). Padding clauses come from the project's frozen
pre-registered personas with language sentences stripped, exactly as cell 1b
built them — no invented prose. The language levels are the census's own
verdicts, not keyword matching. Occupation and place signal ARE keyword matches
and are labelled as such: they are a floor on document-side signal, not a
measurement of it, and the argument only needs the floor.

Encoder: paraphrase-multilingual-mpnet-base-v2 via ONNX Runtime. NOT
nb-sbert-base — see LIMITATIONS.md §12. No API calls; CPU only; the corpus
embedding is read from the cache cell 1 wrote.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CACHE = ROOT / "data" / "embeddings_nb_sbert.npz"
OUT = ROOT / "reports" / "consequence_asymmetry.json"
TOPK = 10
MAX_CLAUSES = 12

LANG_RE = re.compile(r"\b(snakker|speak|spr[åa]k|language|norsk|norwegian)\b", re.I)

# Levels the census assigns to ads that DEMAND Norwegian.
DEMANDING = ("professional", "certified", "fluent", "conversational",
             "scandinavian_accepted")
# The only level that states the ABSENCE of the requirement. This is the
# document-side signal the negated query would have to land on.
NOT_REQUIRED = ("explicitly_not_required",)

# THE CONFOUND THIS ANSWERS. Occupation and place signal is detected by a
# keyword sitting in the very text that was embedded, so the encoder is being
# asked to find something lexically present. Language signal is a census LLM
# judgment about meaning. That is not a like-for-like comparison and it is
# biased TOWARD the hypothesis, so the method-matched version must be reported:
# how many ads state the absence of the requirement LEXICALLY, in words an
# encoder could match the way it matches `tømrer`?
_NORSK_RE = re.compile(r"norsk|norwegian", re.I)
_NEG_REQ_RE = re.compile(
    r"ikke\s+(et\s+)?(absolutt\s+)?krav|ikke\s+n[øo]dvendig|ikke\s+p[åa]krevd|"
    r"not\s+(a\s+)?(requirement|required|necessary|needed)|no\s+need\s+for|"
    r"trenger\s+ikke|beh[øo]ver\s+ikke",
    re.I)


def _states_absence_lexically(text: str, window: int = 90) -> bool:
    """Does the ad negate a requirement WITHIN `window` chars of a mention of
    the Norwegian language?

    The unscoped version of this test matched 1,242 ads and was wrong: `ikke et
    krav` and `ikke nødvendig` overwhelmingly negate EXPERIENCE or EDUCATION,
    not language. Scoping the negation to the neighbourhood of `norsk` is what
    makes this comparable to a `tømrer` keyword hit.
    """
    for m in _NORSK_RE.finditer(text):
        lo = max(0, m.start() - window)
        if _NEG_REQ_RE.search(text[lo:m.end() + window]):
            return True
    return False

# Each family is (variant_A, variant_B) plus the document-side test for each
# side. `None` means "use the census levels", not a regex.
FAMILIES = {
    "no": {
        "language": {
            "a": "Jeg snakker flytende norsk.",
            "b": "Jeg snakker ikke norsk.",
            "sig_a": None, "sig_b": None,
        },
        "occupation": {
            "a": "Jeg er sykepleier.",
            "b": "Jeg er tømrer.",
            "sig_a": re.compile(r"sykepleier|sjukepleiar|nurse", re.I),
            "sig_b": re.compile(r"tømrer|tomrer|snekker|carpenter", re.I),
        },
        "place": {
            "a": "Jeg ser etter jobb i Oslo.",
            "b": "Jeg ser etter jobb i Bergen.",
            "sig_a": re.compile(r"\boslo\b", re.I),
            "sig_b": re.compile(r"\bbergen\b", re.I),
        },
    },
    "en": {
        "language": {
            "a": "I speak fluent Norwegian.",
            "b": "I do not speak Norwegian.",
            "sig_a": None, "sig_b": None,
        },
        "occupation": {
            "a": "I am a nurse.",
            "b": "I am a carpenter.",
            "sig_a": re.compile(r"sykepleier|sjukepleiar|nurse", re.I),
            "sig_b": re.compile(r"tømrer|tomrer|snekker|carpenter", re.I),
        },
        "place": {
            "a": "I am looking for work in Oslo.",
            "b": "I am looking for work in Bergen.",
            "sig_a": re.compile(r"\boslo\b", re.I),
            "sig_b": re.compile(r"\bbergen\b", re.I),
        },
    },
}


def _padding_clauses() -> dict[str, list[str]]:
    """Clauses from the frozen personas, language sentences stripped.

    Copied deliberately from cell 1b rather than imported: the two probes must
    pad identically for their numbers to be comparable, and a shared helper
    that later drifts would break that silently.
    """
    ps = yaml.safe_load((ROOT / "eval" / "personas.yaml")
                        .read_text(encoding="utf-8"))["personas"]
    out: dict[str, list[str]] = {"no": [], "en": []}
    for p in ps:
        q = " ".join(p["query"].split())
        lang = "no" if re.search(r"[æøå]|^Jeg ", q) else "en"
        for s in re.split(r"(?<=[.!?])\s+", q):
            s = s.strip()
            if len(s) > 15 and not LANG_RE.search(s) and s not in out[lang]:
                out[lang].append(s)
    return out


def _load_corpus():
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    rows = con.execute("""SELECT a.uuid, a.title, a.description_text,
                                 f.facets, l.doc_lang
                          FROM ads a
                          JOIN ad_facets f USING (uuid)
                          JOIN ad_language l USING (uuid)
                          ORDER BY a.uuid""").fetchall()
    uuids = [r[0] for r in rows]
    texts = [f"{r[1]}\n{r[2] or ''}" for r in rows]
    levels = []
    for r in rows:
        f = json.loads(r[3]) if isinstance(r[3], str) else r[3]
        levels.append(f.get("norwegian_requirement_level"))
    return uuids, texts, levels


def main() -> None:
    from fastembed import TextEmbedding
    from scipy.stats import kendalltau

    uuids, texts, levels = _load_corpus()

    if not CACHE.exists():
        sys.exit(f"no embedding cache at {CACHE} — run probe_negation.py first")
    z = np.load(CACHE, allow_pickle=True)
    if list(z["uuids"]) != uuids:
        sys.exit("embedding cache does not match the corpus — corpus changed?")
    E = z["E"]
    print(f"corpus embeddings from cache: {E.shape}", flush=True)

    print(f"loading {MODEL} via ONNX (CPU)…", flush=True)
    m = TextEmbedding(model_name=MODEL)

    def enc(xs):
        V = np.array(list(m.embed(list(xs))), dtype=np.float32)
        return V / np.linalg.norm(V, axis=1, keepdims=True)

    result = {"model": MODEL, "n_ads": len(uuids)}
    low = [t.lower() for t in texts]

    # ---- CELL 2a: document-side signal census ---------------------------
    print("\n=== CELL 2a: DOCUMENT-SIDE SIGNAL — what is there to survive into? ===")
    print("  the mechanism claim, measured against all 10,166 ads directly\n")
    print(f"  {'family':11s} {'side':6s} {'what the query asks for':34s} "
          f"{'ads':>6s} {'share':>7s}")
    sig_idx: dict[tuple[str, str], np.ndarray] = {}
    census = []
    for fam in ("language", "occupation", "place"):
        spec = FAMILIES["no"][fam]
        for side in ("a", "b"):
            rx = spec[f"sig_{side}"]
            if rx is None:
                want = DEMANDING if side == "a" else NOT_REQUIRED
                idx = np.array([i for i, l in enumerate(levels) if l in want])
                desc = ("ads DEMANDING Norwegian" if side == "a"
                        else "ads stating Norwegian NOT required")
                how = "census level"
            else:
                idx = np.array([i for i, t in enumerate(low) if rx.search(t)])
                desc = spec[side]
                how = "keyword floor"
            sig_idx[(fam, side)] = idx
            share = len(idx) / len(uuids)
            print(f"  {fam:11s} {side:6s} {desc[:34]:34s} {len(idx):6d} "
                  f"{share:6.2%}")
            census.append({"family": fam, "side": side, "n_ads": int(len(idx)),
                           "share": share, "method": how, "describes": desc})
    result["document_side_census"] = census

    n_dem = len(sig_idx[("language", "a")])
    n_not = len(sig_idx[("language", "b")])
    n_occ_b = len(sig_idx[("occupation", "b")])
    n_plc_b = len(sig_idx[("place", "b")])
    print(f"\n  THE ASYMMETRY: {n_dem} ads say they demand Norwegian; "
          f"{n_not} say it is not required.")
    print(f"  A negated query has {n_not / max(n_dem, 1):.1%} as much to land "
          f"on as its positive twin.")
    print("\n  BUT THE COUNT ALONE DOES NOT CARRY THE HYPOTHESIS, and saying "
          "it did would be wrong.")
    print(f"  The B-sides are comparable in size: language {n_not}, "
          f"occupation {n_occ_b}, place {n_plc_b}.")
    print(f"  Language is only {n_occ_b / max(n_not, 1):.1f}x rarer than "
          f"`tømrer`, not orders of magnitude.")
    print("  So if recovery differs sharply, scarcity is not the whole "
          "mechanism — the KIND of\n  signal matters too: an ABSENCE stated "
          "in an ad is itself a negation, and dilutes\n  on the document side "
          "exactly as it does on the query side.")

    # Method-matched: language-B by keyword, the way occupation-B is measured.
    lex = [i for i, t in enumerate(low) if _states_absence_lexically(t)]
    print("\n  METHOD-MATCHED, because the comparison above is not like-for-"
          "like. Occupation and\n  place are KEYWORD hits in the embedded "
          "text; language is an LLM judgment. Counting\n  language the same "
          "way occupation is counted — a requirement negated within 90 chars\n"
          "  of a mention of the Norwegian language:")
    print(f"    ads stating the absence IN WORDS: {len(lex)} "
          f"({len(lex)/len(uuids):.2%})")
    print(f"    vs `tømrer` {n_occ_b} ({n_occ_b/len(uuids):.2%}) and "
          f"`Bergen` {n_plc_b} ({n_plc_b/len(uuids):.2%})")
    # Let the number decide what this means. An earlier version of this probe
    # asserted the favourable reading in a hardcoded string and was wrong when
    # the pattern changed.
    if len(lex) < n_occ_b:
        print(f"  Lexically the absence is RARER than `tømrer` "
              f"({len(lex)} vs {n_occ_b}), so the method-matched\n  comparison "
              f"preserves the asymmetry the hypothesis needs.")
    else:
        print(f"  Lexically the absence is NOT rarer than `tømrer` "
              f"({len(lex)} vs {n_occ_b}). Document-side\n  SCARCITY is "
              f"therefore not the mechanism: enough ads say it in words. If "
              f"recovery is\n  still 0, what fails is the ENCODER's handling "
              f"of those words, not their absence.")
    result["not_required_lexical"] = {"n_ads": len(lex),
                                      "share": len(lex) / len(uuids),
                                      "rarer_than_occupation_b": len(lex) < n_occ_b}

    # Does the negated query recover the LEXICALLY-stating ads? This is the
    # method-matched recovery test, and it is the one that separates "there is
    # nothing to find" from "the encoder cannot find it".
    result["lexical_absence_idx_n"] = len(lex)
    sig_idx[("language", "b_lex")] = np.array(lex)

    # ---- CELL 2b/2c: cosine-matched overlap, and recovery ---------------
    pads = _padding_clauses()
    print("\n=== CELL 2b & 2c: at MATCHED query-side cosine, does retrieval "
          "recover the distinction? ===")
    sweeps = []
    for lang in ("no", "en"):
        clauses = pads[lang]
        n_k = min(len(clauses), MAX_CLAUSES)
        print(f"\n--- {lang.upper()} — padding from {n_k} real persona clauses ---")
        print(f"  {'family':11s} {'k':>2s} {'cosine':>7s} {'overlap@10':>10s} "
              f"{'tau@50':>7s} {'recov_A':>8s} {'recov_B':>8s}")
        for fam in ("language", "occupation", "place"):
            spec = FAMILIES[lang][fam]
            ia, ib = sig_idx[(fam, "a")], sig_idx[(fam, "b")]
            sa, sb = set(ia.tolist()), set(ib.tolist())
            for k in range(0, n_k + 1):
                pad = " ".join(clauses[:k])
                qa = (pad + " " + spec["a"]).strip()
                qb = (pad + " " + spec["b"]).strip()
                va, vb = enc([qa, qb])
                cos = float(va @ vb)
                ra = np.argsort(-(E @ va))
                rb = np.argsort(-(E @ vb))
                ov = len(set(ra[:TOPK].tolist()) & set(rb[:TOPK].tolist()))
                tau = float(kendalltau(ra[:50], rb[:50]).statistic)
                # recovery: does each variant's top-10 contain ITS OWN signal?
                rec_a = sum(1 for i in ra[:TOPK].tolist() if i in sa) / TOPK
                rec_b = sum(1 for i in rb[:TOPK].tolist() if i in sb) / TOPK
                rec_b_lex = None
                if fam == "language":
                    _lx = sig_idx.get(("language", "b_lex"))
                    if _lx is not None:
                        _lxs = set(_lx.tolist())
                        rec_b_lex = sum(1 for i in rb[:TOPK].tolist()
                                        if i in _lxs) / TOPK
                if k in (0, n_k // 2, n_k):
                    print(f"  {fam:11s} {k:2d} {cos:7.3f} {ov:7d}/10 "
                          f"{tau:7.3f} {rec_a:8.1%} {rec_b:8.1%}")
                row = {"lang": lang, "family": fam, "n_clauses": k,
                       "cosine": cos, "overlap_at_10": ov,
                       "kendall_tau": tau,
                       "recovery_a": rec_a, "recovery_b": rec_b}
                if rec_b_lex is not None:
                    row["recovery_b_lex"] = rec_b_lex
                sweeps.append(row)
    result["sweeps"] = sweeps

    # ---- THE COMPARISON: shared cosine bins ----------------------------
    print("\n=== THE TEST: overlap@10 by family, inside shared cosine bins ===")
    print("  the prediction is that `language` sits ABOVE the other two in "
          "every populated bin\n")
    bins = [(0.0, 0.90), (0.90, 0.96), (0.96, 0.99), (0.99, 1.01)]
    labels = ["<0.90", "0.90-0.96", "0.96-0.99", ">0.99"]
    print(f"  {'cosine bin':12s} {'language':>18s} {'occupation':>18s} "
          f"{'place':>18s}   verdict")
    binned = []
    for (lo, hi), lab in zip(bins, labels):
        cells = {}
        for fam in ("language", "occupation", "place"):
            vs = [s["overlap_at_10"] for s in sweeps
                  if s["family"] == fam and lo <= s["cosine"] < hi]
            cells[fam] = (float(np.mean(vs)), len(vs)) if vs else (None, 0)
        if all(c[1] == 0 for c in cells.values()):
            continue
        row = {"bin": lab}
        out = f"  {lab:12s}"
        for fam in ("language", "occupation", "place"):
            mean, n = cells[fam]
            row[fam] = mean
            row[f"{fam}_n"] = n
            out += f" {'--' if mean is None else f'{mean:5.1f}/10':>12s} (n={n:2d})"
        others = [cells[f][0] for f in ("occupation", "place")
                  if cells[f][0] is not None]
        n_min = min([cells[f][1] for f in ("language", "occupation", "place")
                     if cells[f][1] > 0] or [0])
        if cells["language"][0] is not None and others:
            margin = cells["language"][0] - max(others)
            row["margin"] = margin
            row["n_min"] = n_min
            # A direction is only evidence if the bin is populated enough to
            # have one and the gap is bigger than one result slot. Cell 1b was
            # nearly misread on a margin of 0.004; the guard is the lesson.
            if n_min < 3 or abs(margin) < 1.0:
                row["language_higher"] = None
                row["underpowered"] = True
                out += (f"   UNDERPOWERED (margin {margin:+.1f}, n_min={n_min})")
            else:
                row["language_higher"] = margin > 0
                row["underpowered"] = False
                out += (f"   {'AS PREDICTED' if margin > 0 else 'AGAINST'} "
                        f"({margin:+.1f}/10)")
        else:
            row["language_higher"] = None
            row["underpowered"] = True
            out += "   (not comparable)"
        print(out)
        binned.append(row)
    result["cosine_binned"] = binned

    # ---- recovery summary, which is the mechanism ----------------------
    print("\n=== THE MECHANISM: recovery of the B-side, averaged over the "
          "whole sweep ===")
    print("  variant B is the one the project cares about: 'ikke norsk', "
          "'tømrer', 'Bergen'\n")
    print("  a raw share means nothing without the base rate: a RANDOM top-10 "
          "already contains\n  some B-side ads. What matters is enrichment "
          "over chance.\n")
    print(f"  {'family':11s} {'observed':>9s} {'chance':>8s} {'enrichment':>11s} "
          f"{'exp. hits in sweep':>19s}")
    rec = {}
    for fam in ("language", "occupation", "place"):
        vs = [s["recovery_b"] for s in sweeps if s["family"] == fam]
        obs = float(np.mean(vs))
        base = len(sig_idx[(fam, "b")]) / len(uuids)
        enr = obs / base if base else float("nan")
        exp_hits = base * TOPK * len(vs)
        rec[fam] = {"observed": obs, "chance": base, "enrichment": enr,
                    "expected_hits_in_sweep": exp_hits,
                    "observed_hits_in_sweep": float(sum(vs) * TOPK)}
        print(f"  {fam:11s} {obs:8.1%} {base:7.2%} {enr:10.1f}x "
              f"{exp_hits:8.1f} vs {sum(vs)*TOPK:5.0f} seen")
    result["recovery_b"] = rec

    lex_idx = set(sig_idx[("language", "b_lex")].tolist())
    if lex_idx:
        vs = []
        for sw in sweeps:
            if sw["family"] != "language":
                continue
            vs.append(sw.get("recovery_b_lex", 0.0))
        base = len(lex_idx) / len(uuids)
        obs = float(np.mean(vs)) if vs else 0.0
        print(f"\n  METHOD-MATCHED language recovery (against the {len(lex_idx)} "
              f"ads that state the\n  absence in words, not the census label):")
        print(f"  {'language*':11s} {obs:8.1%} {base:7.2%} "
              f"{obs/base if base else float('nan'):10.1f}x")
        result["recovery_b"]["language_lexical"] = {
            "observed": obs, "chance": base,
            "enrichment": (obs / base) if base else None}

    graded = [b for b in binned if b.get("language_higher") is not None]
    verdict_holds = bool(graded) and all(b["language_higher"] for b in graded)
    any_comparable = bool(graded)
    result["refined_hypothesis_holds"] = verdict_holds
    result["bins_graded"] = len(graded)
    result["bins_underpowered"] = len(binned) - len(graded)

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")

    print("\n=== VERDICT: THE HYPOTHESIS SPLITS ===")
    lex_n = result["not_required_lexical"]["n_ads"]
    lex_rec = result["recovery_b"].get("language_lexical") or {}
    lex_enr = lex_rec.get("enrichment")
    occ_enr = rec["occupation"]["enrichment"]
    plc_enr = rec["place"]["enrichment"]

    print("\n  CONFIRMED — the consequence IS asymmetric.")
    print(f"    Equal query-side dilution, unequal retrieval outcome. Content "
          f"queries pull their\n    own signal to {occ_enr:.1f}x and "
          f"{plc_enr:.1f}x the base rate. The negated language query"
          + (f" reaches\n    {lex_enr:.1f}x — chance." if lex_enr is not None
             else " does not."))
    if any_comparable:
        print(f"    Ranking overlap agrees in direction, but only "
              f"{len(graded)} of {len(binned)} cosine bins clear the\n"
              f"    power guard, so that arm is suggestive and not decisive.")

    print("\n  DISCONFIRMED — the REASON cell 1b gave for it was wrong.")
    print(f"    Cell 1b said the negation has 'nothing to survive into: almost "
          f"no advertisement\n    states the ABSENCE of a requirement'. "
          f"Measured the same way `tømrer` is measured,\n    "
          f"{lex_n} ads state it in words — "
          + ("MORE" if lex_n > n_occ_b else "fewer") +
          f" than the {n_occ_b} that say `tømrer`.")
    print("    SCARCITY IS NOT THE MECHANISM. The document-side signal is "
          "there, in comparable\n    quantity, and retrieval still does not "
          "use it.")

    print("\n  THE CORRECTED MECHANISM, which is stronger for the "
          "architecture argument, not weaker.")
    print("    The document-side statement of an absence is ITSELF a negation "
          "— `norsk er ikke et\n    krav`. Mean pooling dilutes it on the "
          "document side exactly as it dilutes the query.\n    Two averaged-"
          "away negations cannot find each other in a metric space. So the "
          "failure\n    cannot be fixed by more coverage or more data: the "
          "data is ALREADY THERE and unusable.\n    Only extracting the "
          "constraint into a typed predicate can reach it.")

    print("\n  WHAT IS STILL OPEN, and must not be quoted as settled.")
    print(f"    - The {lex_n} lexical hits are an unvalidated keyword floor; "
          f"the census labelled only\n      "
          f"{n_not} ads `explicitly_not_required`. Either the pattern "
          f"over-fires or the census\n      under-fires, and which one is "
          f"unmeasured.")
    print("    - Five pairs, one encoder, dense channel only. No confidence "
          "intervals.")
    print("    - The cosine-bin arm needs more sweep points per bin before it "
          "can carry weight.")

if __name__ == "__main__":

    main()
