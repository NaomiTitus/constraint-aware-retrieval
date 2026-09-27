"""Which tag system should occupation resolve against — STYRK labels or ESCO URIs?

THE QUESTION, asked because "our tags versus ESCO" turned out to be a false
dichotomy: `ad_taxonomy` is already 82% ESCO-derived (`title_source='esco'`). The
real choice is which IDENTIFIER to match a seeker's words against, and the corpus
offers four with very different properties:

    ad_categories STYRK08      100%    hierarchical code       prefix proximity
    ad_categories ESCO URI    97.9%    flat identity, bilingual labels 1242/1242
    ad_taxonomy.styrk_code     100%    hierarchical (82% ESCO) prefix proximity
    ad_facets.skills          32.8%    LLM free text           lexical

`ad_facets.skills` is excluded: 32.8% coverage and the wrong grain for occupation.
It stays a soft skill signal.

FOUR ARMS, each adding one thing:

    1. BM25 on the raw query prose               the D1 baseline
    2. BM25 on the gold-parse lexical terms      boilerplate gone by construction
    3. arm 2 x STYRK-label predicate             the first gazetteer (D18)
    4. arm 2 x ESCO predicate                    bilingual identity + STYRK fallback

THE YARDSTICK IS DELIBERATELY NOT THE SYSTEM UNDER TEST. Occupation precision@5 is
measured with the STYRK-label constraint for EVERY arm, arm 4 included, so the ESCO
path is not graded by the identifier it uses. That still leaves it graded against a
taxonomy rather than against a person: zero relevance pairs are judged, and none of
this is a quality measurement.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "reports" / "esco_vs_styrk.json"
BODY_CHARS = 4000


def main() -> None:
    import duckdb
    from finn_smart_search.eval import gold_parse as gp
    from finn_smart_search.retrieval.bm25 import Bm25Index, Tokenizer, default_stemmers
    from finn_smart_search.retrieval import occupation as occ

    tok = Tokenizer(stemmers=default_stemmers())
    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)

    rows = con.execute("""SELECT a.uuid, a.title, a.description_text,
                                 t.job_title_standardised, t.job_title_en,
                                 t.styrk_code, f.facets
                          FROM ads a
                          JOIN ad_taxonomy t USING (uuid)
                          JOIN ad_facets f USING (uuid)
                          ORDER BY a.uuid""").fetchall()
    label = [r[3] or "" for r in rows]
    styrk = [r[5] or "" for r in rows]
    levels = []
    for r in rows:
        f = json.loads(r[6]) if isinstance(r[6], str) else r[6]
        levels.append(f.get("norwegian_requirement_level"))

    first_uri: dict[str, str] = {}
    for uuid, code in con.execute("""SELECT uuid, code FROM ad_categories
                                     WHERE category_type='ESCO'""").fetchall():
        first_uri.setdefault(uuid, code)
    ad_occs = [occ.AdOccupation(esco_uri=first_uri.get(r[0]), styrk_code=r[5] or None)
               for r in rows]
    n_uri = sum(1 for a in ad_occs if a.esco_uri)
    print(f"corpus {len(rows)} ads — {n_uri} carry an ESCO uri "
          f"({n_uri/len(rows):.1%}), {sum(1 for s in styrk if s)} a styrk code")

    styrk_gaz = occ.OccupationGazetteer.build_from_rows(
        (r[5], r[3], r[4]) for r in rows)
    esco_gaz = occ.EscoGazetteer.build(
        con.execute("SELECT uri,lang,title FROM esco_occupation").fetchall(),
        con.execute("""SELECT c.code, t.styrk_code FROM ad_categories c
                       JOIN ad_taxonomy t USING (uuid)
                       WHERE c.category_type='ESCO'""").fetchall())
    print(f"styrk-label gazetteer {len(styrk_gaz.labels):,} labels; "
          f"esco gazetteer {len(esco_gaz.label_uris):,} bilingual labels")

    t0 = time.time()
    idx = Bm25Index.build([(r[2] or "")[:BODY_CHARS] for r in rows],
                          titles=[f"{r[1] or ''} {r[3] or ''} {r[4] or ''}"
                                  for r in rows],
                          tokenizer=tok, title_boost=3)
    print(f"bm25 built in {time.time()-t0:.0f}s\n")

    people = gp.load_personas()
    parses = gp.load(personas=people)
    dev = sorted(i for i, p in people.items() if p["split"] == "dev")

    sc = {pid: occ.from_gold_parse(parses[pid], styrk_gaz) for pid in dev}
    ec = {pid: occ.esco_from_gold_parse(parses[pid], esco_gaz) for pid in dev}

    print("=== RESOLUTION: does the seeker's phrase reach an occupation? ===\n")
    print(f"  {'persona':32s} {'phrase':22s} {'styrk-label':>13s} {'ESCO':>13s}")
    res = {}
    for pid in dev:
        s_, e_ = sc[pid], ec[pid]
        if s_ is None and e_ is None:
            continue
        ph = (s_ or e_).phrase
        sv = f"{len(s_.codes)} codes" if s_ and s_.resolved else "—"
        ev = f"{len(e_.uris)} uris" if e_ and e_.resolved else "—"
        print(f"  {pid:32s} {ph[:22]:22s} {sv:>13s} {ev:>13s}")
        res[pid] = {"phrase": ph, "styrk": sv, "esco": ev}
    ns = sum(1 for p in dev if sc[p] and sc[p].resolved)
    ne = sum(1 for p in dev if ec[p] and ec[p].resolved)
    tot = sum(1 for p in dev if sc[p] is not None)
    print(f"\n  resolved: styrk-label {ns}/{tot}   ESCO {ne}/{tot}")

    def parse_query(pid: str) -> str:
        parts: list[str] = []
        for c in parses[pid].constraints:
            if c.facet in ("occupation", "skill"):
                parts.extend([str(c.value)] * (3 if c.priority == "hard" else 1))
        return " ".join(parts)

    def rank(pid: str, arm: int, k: int = 10) -> list[int]:
        s = list(idx.score(" ".join(people[pid]["query"].split()) if arm == 1
                           else parse_query(pid)))
        if arm == 3:
            s = occ.apply(s, styrk, sc[pid])
        elif arm == 4:
            s = occ.apply_esco(s, ad_occs, ec[pid])
        order = sorted(range(len(s)), key=lambda i: -s[i])
        return [i for i in order[:k] if s[i] > 0.0]

    ARMS = {1: "raw query", 2: "gold parse", 3: "x styrk-label", 4: "x ESCO"}
    ALL = (1, 2, 3, 4)
    result = {"resolution": res, "resolved": {"styrk": ns, "esco": ne, "of": tot}}

    print("\n=== PAIRED OVERLAP@10 — do the NO and EN variants converge? ===\n")
    pairs: dict[str, list[str]] = {}
    for pid in dev:
        if people[pid].get("pair_id"):
            pairs.setdefault(people[pid]["pair_id"], []).append(pid)
    print(f"  {'pair':6s}" + "".join(f"{ARMS[a]:>16s}" for a in ALL))
    means = {a: [] for a in ALL}
    for pair, ids in sorted(pairs.items()):
        if len(ids) != 2:
            continue
        a_id, b_id = sorted(ids)
        row = []
        for arm in ALL:
            ov = len(set(rank(a_id, arm)) & set(rank(b_id, arm)))
            means[arm].append(ov)
            row.append(ov)
        print(f"  {pair:6s}" + "".join(f"{v:13d}/10" for v in row))
    print(f"  {'mean':6s}" + "".join(
        f"{sum(means[a])/len(means[a]):13.1f}/10" for a in ALL))
    result["pair_overlap"] = {ARMS[a]: sum(means[a]) / len(means[a]) for a in ALL}

    print("\n=== OCCUPATION PRECISION@5 ===")
    print("   yardstick is the STYRK-label constraint for EVERY arm, so arm 4 is")
    print("   not graded by the identifier it uses\n")
    print(f"  {'persona':32s}" + "".join(f"{ARMS[a]:>15s}" for a in ALL))
    hit = {a: [] for a in ALL}
    for pid in dev:
        c = sc[pid]
        if c is None or not c.resolved:
            continue
        row = []
        for arm in ALL:
            good = sum(1 for i in rank(pid, arm, k=5) if c.proximity(styrk[i]) >= 0.70)
            hit[arm].append(good / 5)
            row.append(good)
        print(f"  {pid:32s}" + "".join(f"{v:13d}/5" for v in row))
    print(f"  {'mean share':32s}" + "".join(
        f"{sum(hit[a])/len(hit[a]):14.0%}" for a in ALL))
    result["precision_at5"] = {ARMS[a]: sum(hit[a]) / len(hit[a]) for a in ALL}

    print("\n=== p1_sykepleier_no_norsk — the English nurse query ===")
    for arm in ALL:
        tops = [label[i][:20] for i in rank("p1_sykepleier_no_norsk", arm, k=5)]
        print(f"  {ARMS[arm]:16s} {tops}")
        result.setdefault("p1_top5", {})[ARMS[arm]] = tops

    DEM = ("professional", "certified", "fluent", "conversational",
           "scandinavian_accepted")
    top = rank("p1_sykepleier_no_norsk", 4, k=10)
    nb = sum(1 for i in top if levels[i] in DEM)
    print(f"\n  arm 4 top-10 for the no-Norwegian nurse: {nb}/10 still DEMAND "
          f"Norwegian.\n  Occupation proximity should not fix that — D5 does.")
    result["p1_blocking_arm4"] = nb

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")
    print("\nREMINDER: zero relevance pairs judged. Structural check only.")


if __name__ == "__main__":
    main()
