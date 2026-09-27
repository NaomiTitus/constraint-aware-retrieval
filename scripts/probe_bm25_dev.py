"""D1 on the real corpus: does the lexical channel retrieve plausible ads?

NOT AN EVALUATION, AND THE DIFFERENCE MATTERS. `eval/JUDGING_PROTOCOL.md` is
pre-registered and zero pairs are judged, so nothing here is a relevance
measurement and no number from it may be quoted as one. This is a smoke test with
a specific job: find out whether D1 is worth judging before spending ~130 human
judgments on its output.

WHAT IT CAN CHECK WITHOUT JUDGMENTS, all structural:

  1. The index builds over 10,166 ads and queries in reasonable time.
  2. The top hits for an occupation query are that occupation, read off
     `ad_taxonomy.job_title_standardised` — which is derived independently of BM25,
     so it is a check and not a tautology.
  3. THE PAIRED VARIANTS DIVERGE. p1–p5 differ only in the language sentence, and
     one of each pair is written in Norwegian while the other is English. If the
     lexical channel returns similar ads for both, it is matching on occupation
     despite the language switch; if it collapses, cross-language recall is the
     problem D1 has to solve and BM25 alone cannot.
  4. Recall is not zero for the hard cases — `p6` and `p11` are supposed to find
     FEW results, but "few" must come from the constraint stage, not from the
     lexical channel failing to retrieve anything at all.

The queries are fed RAW here. The gold parses exist and D7 will consume them; this
probe deliberately uses the frozen query text so it measures the channel rather
than the oracle.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "reports" / "bm25_dev_probe.json"
BODY_CHARS = 4000     # ads have a long boilerplate tail; the requirements are early


def main() -> None:
    import duckdb
    from finn_smart_search.eval import gold_parse as gp
    from finn_smart_search.retrieval.bm25 import Bm25Index, Tokenizer, default_stemmers

    stemmers = default_stemmers()
    if len(stemmers) < 2:
        sys.exit("PyStemmer missing: install the search deps before trusting this")
    tok = Tokenizer(stemmers=stemmers)

    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    rows = con.execute("""SELECT a.uuid, a.title, a.description_text,
                                 t.job_title_standardised, t.nav_category,
                                 l.municipal, f.facets, t.job_title_en
                          FROM ads a
                          JOIN ad_taxonomy t USING (uuid)
                          LEFT JOIN ad_locations l USING (uuid)
                          JOIN ad_facets f USING (uuid)
                          ORDER BY a.uuid""").fetchall()
    # ad_locations has more rows than ads; keep the first per uuid
    seen: set[str] = set()
    ads = []
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        ads.append(r)
    titles = [r[1] or "" for r in ads]
    bodies = [(r[2] or "")[:BODY_CHARS] for r in ads]
    occ = [r[3] or "" for r in ads]
    levels = []
    for r in ads:
        f = json.loads(r[6]) if isinstance(r[6], str) else r[6]
        levels.append(f.get("norwegian_requirement_level"))
    print(f"corpus: {len(ads)} ads\n")

    # THE CONTROLLED COMPARISON. `ad_taxonomy` carries a standardised occupation
    # AND its English label for 100% of the corpus — `sykepleier` ->
    # "nurse responsible for general care". Indexing that label gives the LEXICAL
    # channel a bilingual bridge built from A4's ESCO work, with no encoder
    # involved. Whether it helps is measured here, not assumed: arm A is the raw
    # title alone, arm B adds the taxonomy labels, everything else identical.
    occ_en = [r[3] or "" for r in ads]
    occ_no = [r[3] or "" for r in ads]
    # job_title_standardised is the Norwegian label; job_title_en is the English one
    occ_no = [x or "" for x in (r[3] for r in ads)]
    occ_en = [x or "" for x in (r[7] for r in ads)]

    arms = {
        "A_title_only": titles,
        "B_title_plus_taxonomy": [f"{titles[i]} {occ_no[i]} {occ_en[i]}"
                                  for i in range(len(ads))],
    }
    built = {}
    for name, field_text in arms.items():
        t0 = time.time()
        built[name] = Bm25Index.build(bodies, titles=field_text, tokenizer=tok,
                                      title_boost=3)
        print(f"{name}: built in {time.time()-t0:.1f}s — "
              f"{len(built[name].vocab):,} terms "
              f"({built[name].tf.nnz:,} postings)")
    idx = built["B_title_plus_taxonomy"]
    build_s = 0.0

    people = gp.load_personas()
    dev = [(i, p) for i, p in sorted(people.items()) if p["split"] == "dev"]

    t0 = time.time()
    for _, p in dev:
        idx.score(" ".join(p["query"].split()))
    q_ms = (time.time() - t0) / len(dev) * 1000
    print(f"mean query time {q_ms:.0f}ms over {len(dev)} dev personas\n")

    DEM = ("professional", "certified", "fluent", "conversational",
           "scandinavian_accepted")
    result = {"n_ads": len(ads), "n_terms": len(idx.vocab),
              "build_seconds": build_s, "query_ms": q_ms, "personas": {}}

    for pid, p in dev:
        q = " ".join(p["query"].split())
        hits = idx.top_k(q, k=10)
        print(f"── {pid}  [{p['vertical']}]")
        for rank, (i, sc) in enumerate(hits[:5], 1):
            blocking = "BLOCKS" if levels[i] in DEM else "      "
            print(f"   {rank}. {sc:6.2f} {blocking}  {occ[i][:28]:30s} "
                  f"{titles[i][:46]}")
        n_block = sum(1 for i, _ in hits if levels[i] in DEM)
        result["personas"][pid] = {
            "n_hits": len(hits),
            "top_occupations": [occ[i] for i, _ in hits[:5]],
            "blocking_in_top10": n_block,
        }
        print(f"   {len(hits)} hits; {n_block}/10 of the top-10 demand Norwegian\n")

    # ---- the paired divergence check -----------------------------------
    print("=== DO THE PAIRED VARIANTS RETRIEVE THE SAME ADS? ===")
    print("   each pair differs ONLY in the language sentence — but one is written")
    print("   in Norwegian and the other in English, so this measures whether the")
    print("   lexical channel survives the language switch at all.\n")
    pairs: dict[str, list[str]] = {}
    for pid, p in dev:
        if p.get("pair_id"):
            pairs.setdefault(p["pair_id"], []).append(pid)
    all_ov = {}
    print(f"   {'pair':6s} {'A title only':>14s} {'B + taxonomy':>14s}")
    for pair, ids in sorted(pairs.items()):
        if len(ids) != 2:
            continue
        a, b = sorted(ids)
        row = {}
        for name, ix in built.items():
            ta = {i for i, _ in ix.top_k(" ".join(people[a]["query"].split()), k=10)}
            tb = {i for i, _ in ix.top_k(" ".join(people[b]["query"].split()), k=10)}
            row[name] = len(ta & tb)
        all_ov[pair] = row
        print(f"   {pair:6s} {row['A_title_only']:11d}/10 "
              f"{row['B_title_plus_taxonomy']:11d}/10")
    result["pair_overlap_at_10"] = all_ov
    if all_ov:
        ma = sum(r["A_title_only"] for r in all_ov.values()) / len(all_ov)
        mb = sum(r["B_title_plus_taxonomy"] for r in all_ov.values()) / len(all_ov)
        result["mean_overlap"] = {"A_title_only": ma, "B_title_plus_taxonomy": mb}
        print(f"\n   mean  A {ma:.1f}/10    B {mb:.1f}/10    delta {mb-ma:+.1f}")
        print("\n   Arm A is the cross-language failure: `nurse` never matches "
              "`sykepleier`, so the\n   English variant falls back to generic "
              "English vocabulary and retrieves chefs\n   and bricklayers. Arm B "
              "indexes ad_taxonomy.job_title_en, populated for 100% of\n   the "
              "corpus by A4's ESCO work, which bridges the two LEXICALLY — no "
              "encoder.")

    # ---- ARM C: query the GOLD PARSE instead of the prose ---------------
    #
    # The diagnosis said raw query text is the wrong input: English boilerplate
    # (`shifts`, `permanent`, `preferably`) looks rare in a Norwegian corpus and
    # outweighs `nurse`. A stopword list is the floor. The DESIGNED answer is to
    # query the parse, where the facets are already separated by priority — only
    # `occupation` and `skill` are lexical, and everything else is compared
    # structurally by D5/D7 rather than thrown into a bag of words.
    parses = gp.load(personas=people)

    def parse_query(pid: str) -> str:
        parts: list[str] = []
        for c in parses[pid].constraints:
            if c.facet in ("occupation", "skill"):
                parts.extend([str(c.value)] * (3 if c.priority == "hard" else 1))
        return " ".join(parts)

    print("\n=== THE CASE THAT EXPOSED IT: p1_sykepleier_no_norsk top-5 ===")
    pid = "p1_sykepleier_no_norsk"
    raw = " ".join(people[pid]["query"].split())
    trials = [("A_title_only  / raw query", built["A_title_only"], raw),
              ("B_+taxonomy   / raw query", idx, raw),
              ("C_+taxonomy   / GOLD PARSE", idx, parse_query(pid))]
    for name, ix, q in trials:
        tops = [occ_no[i] for i, _ in ix.top_k(q, k=5)]
        n_nurse = sum(1 for o in tops if "sykepleier" in o.lower())
        print(f"   {name:28s} {n_nurse}/5 nurses  {tops[:4]}")
        result.setdefault("p1_english_top5", {})[name] = tops

    print("\n=== PAIRED OVERLAP, raw query vs gold parse (arm B index) ===")
    print("   the parse strips the language sentence and the boilerplate, so the")
    print("   two variants of a pair should converge on the SAME ads.\n")
    pov = {}
    for pair, ids in sorted(pairs.items()):
        if len(ids) != 2:
            continue
        a, b = sorted(ids)
        ra = {i for i, _ in idx.top_k(" ".join(people[a]["query"].split()), k=10)}
        rb = {i for i, _ in idx.top_k(" ".join(people[b]["query"].split()), k=10)}
        pa = {i for i, _ in idx.top_k(parse_query(a), k=10)}
        pb = {i for i, _ in idx.top_k(parse_query(b), k=10)}
        pov[pair] = {"raw": len(ra & rb), "parse": len(pa & pb)}
        print(f"   {pair:6s} raw {len(ra & rb):2d}/10    gold parse "
              f"{len(pa & pb):2d}/10")
    result["pair_overlap_raw_vs_parse"] = pov
    if pov:
        mr = sum(v["raw"] for v in pov.values()) / len(pov)
        mp = sum(v["parse"] for v in pov.values()) / len(pov)
        print(f"\n   mean  raw {mr:.1f}/10   gold parse {mp:.1f}/10   "
              f"delta {mp-mr:+.1f}")
        result["mean_overlap_raw_vs_parse"] = {"raw": mr, "parse": mp}

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")
    print("\nREMINDER: no relevance judgment exists. Nothing above is a quality\n"
          "measurement — it says only whether D1 is worth judging.")


if __name__ == "__main__":
    main()
