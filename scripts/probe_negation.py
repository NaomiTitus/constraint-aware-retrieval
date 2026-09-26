"""Cell 1 and 2 of the negation probe: can a bi-encoder represent "not"?

THE PREMISE THE WHOLE PROJECT RESTS ON. If a multilingual sentence encoder
places "jeg snakker ikke norsk" and "jeg snakker flytende norsk" close
together, then a dense-retrieval surface returns near-identical rankings for
both, and no amount of better embedding fixes it — the problem is that
set-difference is not expressible in a metric space. Extracting typed
constraints BEFORE retrieval and applying them in a separate tunable stage is
then an architectural necessity, not a preference.

If the premise is FALSE — if the pairs sit far apart and the rankings differ —
this project's argument collapses and it is far better to learn that here,
from four numbers, than after building a retrieval stack on top of it.

WHAT IS MEASURED, on the five paired personas that differ ONLY in the
language sentence:

  1. cosine between each pair                      predicted 0.88-0.96
  2. Kendall tau and overlap@10 between the two
     dense-only rankings they produce              predicted tau > 0.85, 9/10
  3. THE KILLER CELL: score one `gode norskkunnskaper` ad against
       (a) "jeg snakker ikke norsk"
       (b) the same query with no language sentence at all
     If (a) > (b), stating the constraint made the result WORSE.

No API calls. CPU only. The corpus is embedded once and cached to disk, so a
re-run is free.

WHICH ENCODER, AND WHY NOT THE ONE THE PLAN NAMES. The plan specifies
`NbAiLab/nb-sbert-base` via sentence-transformers on torch. **There is no
PyTorch wheel for this machine** — Python 3.13 on Intel macOS (x86_64), and
PyTorch stopped building macOS x86_64 wheels after 2.2. So the probe runs
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` through ONNX
Runtime instead: the same family, the same 768 dimensions the plan's sizing
assumes, Norwegian in its training data, and no torch.

This matters for how the result may be quoted. The claim the probe tests is
about BI-ENCODERS AS A CLASS — that a metric space cannot express
set-difference — so a multilingual encoder is a fair instrument, and arguably
a fairer one here since the no-Norwegian queries are written in English. But
the README must name the encoder actually used and must NOT attribute the
number to nb-sbert, which has not been run.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CACHE = ROOT / "data" / "embeddings_nb_sbert.npz"
OUT = ROOT / "reports" / "negation_probe.json"
TOPK = 10


def _load_personas() -> dict:
    import yaml
    ps = yaml.safe_load((ROOT / "eval" / "personas.yaml")
                        .read_text(encoding="utf-8"))["personas"]
    return {p["id"]: p for p in ps}


def _strip_language_sentence(q: str) -> str:
    """Cell 3 needs the same query with the language claim removed and nothing
    else touched."""
    keep = [s for s in re.split(r"(?<=[.!?])\s+", q.strip())
            if not re.search(r"\b(snakker|speak|spr[åa]k|language|norsk|norwegian)\b",
                             s, re.I)]
    return " ".join(keep)


def main() -> None:
    from fastembed import TextEmbedding
    import duckdb

    print(f"loading {MODEL_NAME} via ONNX (CPU)…", flush=True)
    _m = TextEmbedding(model_name=MODEL_NAME)

    def _encode(items, **_kw):
        V = np.array(list(_m.embed(list(items))), dtype=np.float32)
        return V / np.linalg.norm(V, axis=1, keepdims=True)

    class _Shim:
        encode = staticmethod(_encode)
    model = _Shim()

    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    ads = con.execute("""SELECT a.uuid, a.title, a.description_text,
                                f.facets, l.doc_lang
                         FROM ads a
                         JOIN ad_facets f USING (uuid)
                         JOIN ad_language l USING (uuid)
                         ORDER BY a.uuid""").fetchall()
    uuids = [r[0] for r in ads]
    texts = [f"{r[1]}\n{(r[2] or '')[:1200]}" for r in ads]
    levels = [json.loads(r[3]).get("norwegian_requirement_level")
              if isinstance(r[3], str) else r[3].get("norwegian_requirement_level")
              for r in ads]

    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["uuids"]) == uuids:
            E = z["E"]
            print(f"embeddings from cache: {E.shape}", flush=True)
        else:
            E = None
    else:
        E = None
    if E is None:
        t0 = time.time()
        print(f"embedding {len(texts)} ads on CPU — this is the slow part…",
              flush=True)
        E = model.encode(texts)
        np.savez_compressed(CACHE, E=E, uuids=np.array(uuids))
        print(f"embedded in {time.time()-t0:.0f}s -> {CACHE.name}", flush=True)

    P = _load_personas()
    pairs = {}
    for p in P.values():
        if p.get("pair_id"):
            pairs.setdefault(p["pair_id"], []).append(p)
    pairs = {k: v for k, v in pairs.items() if len(v) == 2}

    def rank(q: str) -> np.ndarray:
        v = model.encode([q])[0]
        return np.argsort(-(E @ v))

    from scipy.stats import kendalltau
    result = {"model": MODEL_NAME, "n_ads": len(uuids), "pairs": []}
    print("\n=== CELL 1 & 2: paired personas, dense only ===")
    print(f"{'pair':6s} {'cosine':>7s} {'tau@50':>7s} {'overlap@10':>11s}  "
          f"{'blocking in top-10':>20s}")
    for pid, (a, b) in sorted(pairs.items()):
        qa, qb = a["query"].strip(), b["query"].strip()
        va, vb = model.encode([qa, qb])
        cos = float(va @ vb)
        ra, rb = rank(qa), rank(qb)
        tau = float(kendalltau(ra[:50], rb[:50]).statistic)
        ov = len(set(ra[:TOPK]) & set(rb[:TOPK]))
        # of the NO-NORWEGIAN variant's top 10, how many demand Norwegian?
        blocking = sum(1 for i in rb[:TOPK]
                       if levels[i] in ("certified", "fluent", "professional",
                                        "conversational", "scandinavian_accepted"))
        print(f"{pid:6s} {cos:7.3f} {tau:7.3f} {ov:8d}/10  {blocking:17d}/10")
        result["pairs"].append({"pair_id": pid, "cosine": cos, "kendall_tau": tau,
                                "overlap_at_10": ov, "blocking_in_top10": blocking})

    cs = [p["cosine"] for p in result["pairs"]]
    ovs = [p["overlap_at_10"] for p in result["pairs"]]
    bl = [p["blocking_in_top10"] for p in result["pairs"]]
    print(f"\nmean cosine {np.mean(cs):.3f}   mean overlap@10 {np.mean(ovs):.1f}/10"
          f"   mean blocking in top-10 {np.mean(bl):.1f}/10")

    print("\n=== CELL 3: does STATING the constraint make it worse? ===")
    killer = []
    for pid, (a, b) in sorted(pairs.items()):
        stated = b["query"].strip()                     # "I do not speak Norwegian"
        silent = _strip_language_sentence(stated)
        vs, vq = model.encode([stated, silent])
        # the most Norwegian-demanding ads, by the extractor's own verdict
        idx = [i for i, l in enumerate(levels) if l in ("professional", "certified")]
        sub = E[idx]
        mean_stated = float((sub @ vs).mean())
        mean_silent = float((sub @ vq).mean())
        worse = mean_stated > mean_silent
        killer.append({"pair_id": pid, "stated": mean_stated,
                       "silent": mean_silent, "stating_made_it_worse": worse})
        print(f"  {pid}: mean similarity to Norwegian-DEMANDING ads — "
              f"stating the constraint {mean_stated:.4f} vs saying nothing "
              f"{mean_silent:.4f}   {'WORSE' if worse else 'better'}")
    n_worse = sum(1 for k in killer if k["stating_made_it_worse"])
    print(f"\nstating 'I do not speak Norwegian' moved the seeker CLOSER to ads "
          f"that demand it on {n_worse} of {len(killer)} personas")
    result["killer_cell"] = killer

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")
    print("\nVERDICT: the premise HOLDS if cosine is high and overlap@10 is "
          "high — a bi-encoder cannot see the negation.\n"
          "         If cosine is low and overlap small, the thesis is wrong "
          "and the constraint stage is unnecessary.")


if __name__ == "__main__":
    main()
