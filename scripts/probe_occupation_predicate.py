"""Does occupation-as-predicate fix the cross-language failure D1 measured?

THE CLAIM UNDER TEST, from DECISIONS D18. BM25 weights a hard occupation
constraint by rarity, so `nurse` (idf 2.04, 1,330 ads) lost to English boilerplate
like `shifts` (idf 5.84, 0.3% of ads), and an English nurse query returned a
psychiatrist, a Norwegian teacher, a caretaker, a sales rep and a bricklayer. The
fix is to stop scoring occupation lexically and compare STYRK codes instead —
metadata at 100% coverage, hierarchical, and derived from the AD rather than the
query, so it is language-independent by construction.

THE PREDICTION, and it is falsifiable in one number. The paired personas differ
only in the language sentence, but one variant of each pair is written in Norwegian
and the other in English. Under BM25 alone their top-10s barely intersect
(mean 1.8/10). If the occupation predicate works, the two variants must CONVERGE,
because both resolve to the same STYRK code regardless of the language they were
typed in.

WHAT WOULD FALSIFY IT. If overlap does not rise, then either the gazetteer fails
to resolve the seeker phrases or STYRK proximity is too coarse, and the honest move
is to report that the 2-hour detour bought nothing rather than to keep adding
lexical tricks.

THREE ARMS, each adding one thing to the one before:

  1. BM25 on the raw query prose            — the D1 baseline
  2. BM25 on the gold-parse lexical terms   — boilerplate removed by construction
  3. arm 2 x occupation predicate           — D18

NOT AN EVALUATION. Zero relevance pairs are judged. Occupation labels are read from
`ad_taxonomy`, which is derived independently of both BM25 and this module, so the
check is not circular — but it is structural, and no number here is a quality
measurement.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "reports" / "occupation_predicate.json"
BODY_CHARS = 4000


def main() -> None:
    import duckdb
    from finn_smart_search.eval import gold_parse as gp
    from finn_smart_search.retrieval.bm25 import Bm25Index, Tokenizer, default_stemmers
    from finn_smart_search.retrieval import occupation as occ

    stemmers = default_stemmers()
    if len(stemmers) < 2:
        sys.exit("PyStemmer missing — install the search deps first")
    tok = Tokenizer(stemmers=stemmers)

    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    rows = con.execute("""SELECT a.uuid, a.title, a.description_text,
                                 t.job_title_standardised, t.job_title_en,
                                 t.styrk_code, f.facets
                          FROM ads a
                          JOIN ad_taxonomy t USING (uuid)
                          JOIN ad_facets f USING (uuid)
                          ORDER BY a.uuid""").fetchall()
    titles = [f"{r[1] or ''} {r[3] or ''} {r[4] or ''}" for r in rows]
    bodies = [(r[2] or "")[:BODY_CHARS] for r in rows]
    label = [r[3] or "" for r in rows]
    codes = [r[5] or "" for r in rows]
    levels = []
    for r in rows:
        f = json.loads(r[6]) if isinstance(r[6], str) else r[6]
        levels.append(f.get("norwegian_requirement_level"))
    print(f"corpus {len(rows)} ads")

    gaz = occ.OccupationGazetteer.build_from_rows(
        (r[5], r[3], r[4]) for r in rows)
    print(f"gazetteer: {len(gaz.labels):,} labels -> "
          f"{len({c for v in gaz.labels.values() for c in v})} styrk codes")

    t0 = time.time()
    idx = Bm25Index.build(bodies, titles=titles, tokenizer=tok, title_boost=3)
    print(f"bm25 built in {time.time()-t0:.0f}s\n")

    people = gp.load_personas()
    parses = gp.load(personas=people)
    dev = [i for i, p in sorted(people.items()) if p["split"] == "dev"]

    def parse_query(pid: str) -> str:
        parts: list[str] = []
        for c in parses[pid].constraints:
            if c.facet in ("occupation", "skill"):
                parts.extend([str(c.value)] * (3 if c.priority == "hard" else 1))
        return " ".join(parts)

    # ---- how well does the gazetteer resolve the seekers at all? --------
    print("=== GAZETTEER RESOLUTION on the 13 dev personas ===\n")
    print(f"  {'persona':32s} {'occupation phrase':26s} {'codes':>22s}")
    resolution = {}
    constraints: dict[str, occ.OccupationConstraint | None] = {}
    for pid in dev:
        c = occ.from_gold_parse(parses[pid], gaz)
        constraints[pid] = c
        if c is None:
            print(f"  {pid:32s} {'(none stated)':26s} {'—':>22s}")
            resolution[pid] = None
            continue
        top = sorted(c.codes.items(), key=lambda kv: -kv[1])[:3]
        shown = ", ".join(f"{k}:{v:.0%}" for k, v in top) or "UNRESOLVED"
        print(f"  {pid:32s} {c.phrase[:26]:26s} {shown:>22s}")
        resolution[pid] = {"phrase": c.phrase, "codes": dict(c.codes)}
    n_unres = sum(1 for pid in dev
                  if constraints[pid] is not None and not constraints[pid].resolved)
    if n_unres:
        print(f"\n  {n_unres} occupation(s) UNRESOLVED — those personas fall back to "
              f"the lexical\n  channel untouched, which is the fail-open rule.")

    # ---- the three arms -------------------------------------------------
    def rank(pid: str, arm: int, k: int = 10) -> list[int]:
        if arm == 1:
            s = idx.score(" ".join(people[pid]["query"].split()))
        else:
            s = idx.score(parse_query(pid))
        s = list(s)
        if arm == 3:
            s = occ.apply(s, codes, constraints[pid])
        order = sorted(range(len(s)), key=lambda i: -s[i])
        return [i for i in order[:k] if s[i] > 0.0]

    ARMS = {1: "raw query", 2: "gold parse", 3: "gold parse x occupation"}
    result = {"n_ads": len(rows), "resolution": resolution, "arms": {}}

    print("\n=== PAIRED OVERLAP@10 — do the NO and EN variants converge? ===\n")
    pairs: dict[str, list[str]] = {}
    for pid in dev:
        if people[pid].get("pair_id"):
            pairs.setdefault(people[pid]["pair_id"], []).append(pid)
    print(f"  {'pair':6s} " + " ".join(f"{ARMS[a]:>24s}" for a in (1, 2, 3)))
    means = {a: [] for a in (1, 2, 3)}
    for pair, ids in sorted(pairs.items()):
        if len(ids) != 2:
            continue
        a_id, b_id = sorted(ids)
        row = []
        for arm in (1, 2, 3):
            ov = len(set(rank(a_id, arm)) & set(rank(b_id, arm)))
            means[arm].append(ov)
            row.append(ov)
        print(f"  {pair:6s} " + " ".join(f"{v:21d}/10" for v in row))
    print(f"\n  {'mean':6s} " + " ".join(
        f"{sum(means[a])/len(means[a]):21.1f}/10" for a in (1, 2, 3)))
    result["pair_overlap"] = {ARMS[a]: sum(means[a]) / len(means[a])
                              for a in (1, 2, 3)}

    # ---- does the right occupation actually come back? ------------------
    print("\n=== IS THE TOP-5 THE OCCUPATION THE SEEKER ASKED FOR? ===")
    print("   proximity read off ad_taxonomy.styrk_code, derived independently\n")
    print(f"  {'persona':32s} " + " ".join(f"{ARMS[a]:>20s}" for a in (1, 2, 3)))
    hit_means = {a: [] for a in (1, 2, 3)}
    for pid in dev:
        c = constraints[pid]
        if c is None or not c.resolved:
            continue
        row = []
        for arm in (1, 2, 3):
            top = rank(pid, arm, k=5)
            good = sum(1 for i in top if c.proximity(codes[i]) >= 0.70)
            hit_means[arm].append(good / 5)
            row.append(good)
        print(f"  {pid:32s} " + " ".join(f"{v:17d}/5" for v in row))
    print(f"\n  {'mean share':32s} " + " ".join(
        f"{sum(hit_means[a])/len(hit_means[a]):19.0%}" for a in (1, 2, 3)))
    result["occupation_precision_at5"] = {
        ARMS[a]: sum(hit_means[a]) / len(hit_means[a]) for a in (1, 2, 3)}

    # ---- the case that started it ---------------------------------------
    print("\n=== p1_sykepleier_no_norsk — the English nurse query ===")
    for arm in (1, 2, 3):
        top = rank("p1_sykepleier_no_norsk", arm, k=5)
        print(f"  {ARMS[arm]:24s} {[label[i][:22] for i in top]}")
        result.setdefault("p1_top5", {})[ARMS[arm]] = [label[i] for i in top]

    DEM = ("professional", "certified", "fluent", "conversational",
           "scandinavian_accepted")
    print("\n=== and the constraint stage still has work left to do ===")
    top = rank("p1_sykepleier_no_norsk", 3, k=10)
    n_block = sum(1 for i in top if levels[i] in DEM)
    print(f"  arm 3 top-10 for the no-Norwegian nurse: {n_block}/10 still DEMAND "
          f"Norwegian.\n  Occupation proximity does not and should not fix that — "
          f"D5 does.")
    result["p1_blocking_in_top10_arm3"] = n_block

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")
    print("\nREMINDER: zero relevance pairs judged. Structural check only.")


if __name__ == "__main__":
    main()
