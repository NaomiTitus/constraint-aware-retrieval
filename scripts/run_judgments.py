"""Run the depth-10 judgment pool through the LLM judge. One-off orchestration.

A `scripts/` one-off per STANDARDS §2, and deliberately thin: every decision that
could be wrong lives in the tested package — `eval/judge_llm.py` renders and
validates, `eval/pool.py` pools — and this file only wires them to the corpus and
the Batch API. Nothing here is unit-tested and nothing here makes a judgment call.

THIS SPENDS MONEY AND STARTS A CLOCK. `JUDGING_PROTOCOL.md` permits ONE revision of
the judge prompt before the human calibration set is labelled and zero after. The
first submitted batch starts that clock, which is why `--submit` is required and a
dry run is the default.

MODEL CHOICE, and a grounding figure that was wrong. G6 estimated ~234k input
tokens by measuring the advertisement text alone; the built pool is 446,471 tokens,
because the judge receives the FULL advertisement (cap 10,000, not the arms' 4,000)
plus a ~2,000-character instruction per request. Roughly double the estimate, and
still small: at batch pricing — 50% of standard — Opus 5 input is $2.50/MTok and
output $12.50/MTok, so this pool is a few dollars including thinking tokens.

Cost therefore does not constrain the choice, and the judge is `claude-opus-5`. The
protocol pre-registers quadratic-weighted kappa > 0.6 against human labels and
permits ONE prompt revision, so a weak judge cannot be repaired by re-prompting —
it can only be reported as weak and have its numbers downweighted. Recorded in the
manifest either way.

THE ARMS, which are the current ablation ladder and not the final one. E6 names
BM25 / dense / hybrid / +graph / +constraints soft / hard / +rerank; fusion and
rerank do not exist yet. Pooling what exists now means a later arm may retrieve
documents this pool never judged — that is exactly the pooling bias the protocol
already puts in the limitations, and Recall@50 is an upper bound because of it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_JUDGMENTS = ROOT / "eval" / "judgments.json"
OUT_MANIFEST = ROOT / "eval" / "judgments_manifest.json"
BODY_CHARS = 4000          # what the retrieval arms index; NOT the judge's cap
# claude-opus-5. Not a cost decision: the protocol pre-registers a target of
# quadratic-weighted kappa > 0.6 against human labels and permits ONE prompt
# revision, so a weak judge cannot be fixed by re-prompting later — it can only be
# reported as a weak judge and have its numbers downweighted. Batch pricing is 50%
# of standard, which puts this whole pool in single dollars either way.
DEFAULT_MODEL = "claude-opus-5"


def build_everything(model: str):
    import duckdb
    import numpy as np
    from finn_smart_search.eval import gold_parse as gp
    from finn_smart_search.eval import judge_llm as J
    from finn_smart_search.eval import pool as P
    from finn_smart_search.retrieval import occupation as occ
    from finn_smart_search.retrieval.bm25 import Bm25Index, Tokenizer, default_stemmers

    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    rows = con.execute("""SELECT a.uuid, a.title, a.description_text,
                                 t.job_title_standardised, t.job_title_en,
                                 t.styrk_code
                          FROM ads a JOIN ad_taxonomy t USING (uuid)
                          ORDER BY a.uuid""").fetchall()
    uuids = [r[0] for r in rows]
    titles = [r[1] or "" for r in rows]
    # The JUDGE gets the FULL advertisement text, capped only by its own
    # AD_TEXT_CAP. The 4,000-char slice below is what the retrieval arms index;
    # feeding the judge the same slice would hide requirements from it that the
    # ranking never had to see either, and a requirement the judge cannot read
    # scores as a clean advertisement.
    full_text = [r[2] or "" for r in rows]
    styrk = [r[5] or "" for r in rows]

    tok = Tokenizer(stemmers=default_stemmers())
    idx = Bm25Index.build([t[:BODY_CHARS] for t in full_text],
                          titles=[f"{titles[i]} {rows[i][3] or ''} {rows[i][4] or ''}"
                                  for i in range(len(rows))],
                          tokenizer=tok, title_boost=3)

    first_uri: dict[str, str] = {}
    for uuid, code in con.execute("""SELECT uuid, code FROM ad_categories
                                     WHERE category_type='ESCO'""").fetchall():
        first_uri.setdefault(uuid, code)
    ad_occs = [occ.AdOccupation(first_uri.get(u), s or None)
               for u, s in zip(uuids, styrk)]

    styrk_gaz = occ.OccupationGazetteer.build_from_rows(
        (r[5], r[3], r[4]) for r in rows)
    esco_gaz = occ.EscoGazetteer.build(
        con.execute("SELECT uri,lang,title FROM esco_occupation").fetchall(),
        con.execute("""SELECT c.code, t.styrk_code FROM ad_categories c
                       JOIN ad_taxonomy t USING (uuid)
                       WHERE c.category_type='ESCO'""").fetchall())

    z = np.load(ROOT / "data" / "embeddings_nb_sbert.npz", allow_pickle=True)
    assert list(z["uuids"]) == uuids, "embedding cache does not match the corpus"
    emb = z["E"]
    from fastembed import TextEmbedding
    m = TextEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    def encode(text: str):
        import numpy as _np
        v = _np.array(list(m.embed([text])), dtype=_np.float32)[0]
        return v / _np.linalg.norm(v)

    people = gp.load_personas()
    parses = gp.load(personas=people)
    dev = sorted(i for i, p in people.items() if p["split"] == "dev")

    def parse_query(pid: str) -> str:
        parts: list[str] = []
        for c in parses[pid].constraints:
            if c.facet in ("occupation", "skill"):
                parts.extend([str(c.value)] * (3 if c.priority == "hard" else 1))
        return " ".join(parts)

    def topk(scores, k=10):
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, float(scores[i])) for i in order if scores[i] > 0.0]

    import duckdb as _dd
    _fr = con.execute("""SELECT a.uuid, f.facets, l.doc_lang FROM ads a
                         JOIN ad_facets f USING (uuid) JOIN ad_language l USING (uuid)
                         ORDER BY a.uuid""").fetchall()
    ad_facets_seq = [(json.loads(r[1]) if isinstance(r[1], str) else r[1], r[2])
                     for r in _fr]
    from finn_smart_search.retrieval import constraints as K

    rankings: dict[str, dict[str, list]] = {}
    bm_parse_full: dict[str, list[float]] = {}
    for pid in dev:
        raw = " ".join(people[pid]["query"].split())
        bm_raw = list(idx.score(raw))
        bm_parse = list(idx.score(parse_query(pid)))
        bm_parse_full[pid] = bm_parse
        rankings[pid] = {
            "bm25_raw": topk(bm_raw),
            "bm25_parse": topk(bm_parse),
            "styrk": topk(occ.apply(list(bm_parse), styrk,
                                    occ.from_gold_parse(parses[pid], styrk_gaz))),
            "esco": topk(occ.apply_esco(list(bm_parse), ad_occs,
                                        occ.esco_from_gold_parse(parses[pid], esco_gaz))),
            "dense": topk(list(emb @ encode(raw))),
        }

    # ROUND 2 ADDS THE ARM THE FIRST POOL WAS MISSING (LIMITATIONS §17). The first
    # five arms were all UNCONSTRAINED, so the proposed system was never put in
    # front of the judge, and the constrained ranking promotes documents nothing
    # pooled — 17% of its top-10 had no grade. Pooling it now is the fix; the judge,
    # the prompt version and the protocol are unchanged, so nothing is re-registered.
    #
    # Soft (λ=0.7) and hard (λ=1.0) are both pooled because E6 names them as separate
    # rungs on the ladder, and they promote different documents.
    for pid in dev:
        prof = gp.to_seeker_profile(parses[pid], people)
        for lam, name in ((0.7, "constr_soft"), (1.0, "constr_hard")):
            adj = occ.apply_esco(list(bm_parse_full[pid]), ad_occs,
                                 occ.esco_from_gold_parse(parses[pid], esco_gaz))
            adj = K.apply(adj, ad_facets_seq, prof, lam=lam)
            rankings[pid][name] = topk(adj)

    # ROUND 3 ADDS THE +skills ARM (LIMITATIONS §18). Occupation is compared as a
    # taxonomy code; skills were still compared as text, so `PyTorch` did not imply
    # `machine learning`. This resolves both sides onto ESCO skill concepts — 10,063
    # skills, all bilingual — and scores their overlap. Whether that HELPS is exactly
    # what no existing arm could show, which is why it is pooled and judged rather
    # than tuned by eye.
    esco_sk = json.loads((ROOT / "reports" / "skills_esco_resolved.json")
                         .read_text(encoding="utf-8"))["ad_to_uris"]
    from finn_smart_search.retrieval.skills_match import EscoSkillIndex
    sk_index = EscoSkillIndex.build(
        con.execute("SELECT uri,lang,title FROM esco_skill").fetchall())
    ad_sk = [set(esco_sk.get(u, ())) for u in uuids]

    for pid in dev:
        want: set[str] = set()
        for c in parses[pid].constraints:
            if c.facet == "skill":
                want.update(m.uri for m in sk_index.resolve(str(c.value), top_k=2))
        base = list(bm_parse_full[pid])
        if want:
            # A BONUS, never a gate. Only 67% of advertisements carry any resolved
            # concept at all, so gating on it would silently drop a third of the
            # corpus for a signal that resolves 13% of glosses.
            base = [s * (1.0 + 0.45 * (len(want & ad_sk[i]) / len(want)))
                    for i, s in enumerate(base)]
        adj = occ.apply_esco(base, ad_occs,
                             occ.esco_from_gold_parse(parses[pid], esco_gaz))
        prof = gp.to_seeker_profile(parses[pid], people)
        rankings[pid]["skills_esco"] = topk(K.apply(adj, ad_facets_seq, prof, lam=0.7))

    pools = P.build_pools(people, rankings)          # raises on a sealed persona
    # Do not pay twice for a pair the first round already judged.
    already: set[str] = set()
    if OUT_JUDGMENTS.exists():
        prior = json.loads(OUT_JUDGMENTS.read_text(encoding="utf-8"))
        already = {j["pair_id"] for j in prior.get("judgments", [])}
        print(f"prior      {len(already)} pairs already judged — skipping those")

    requests = []
    index_of_pair: dict[str, tuple[str, int]] = {}
    for pid, ads in pools.items():
        for ad_i in ads:
            pair_id = J.make_pair_id(pid, ad_i)
            index_of_pair[pair_id] = (pid, ad_i)
            if pair_id in already:
                continue
            requests.append(J.build_request(
                pair_id=pair_id,
                persona_query=" ".join(people[pid]["query"].split()),
                ad_title=titles[ad_i],
                ad_text=full_text[ad_i],
                model=model))
    return dict(requests=requests, pools=pools, index_of_pair=index_of_pair,
                uuids=uuids, titles=titles, full_text=full_text, dev=dev,
                rankings=rankings, people=people)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true",
                    help="actually spend money and start the protocol's clock")
    ap.add_argument("--collect", metavar="BATCH_ID", nargs="?", const="manifest",
                    help="skip submission and collect an already-submitted batch. "
                         "Without this, a poll timeout would force a resubmission "
                         "and pay for the same pool twice.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--poll-seconds", type=int, default=20)
    ap.add_argument("--max-wait", type=int, default=3600)
    args = ap.parse_args()

    from finn_smart_search.eval import judge_llm as J

    built = build_everything(args.model)
    reqs = built["requests"]
    chars = sum(len(r["params"]["messages"][0]["content"]) for r in reqs)
    print(f"personas   {len(built['dev'])} dev")
    print(f"pool       {len(reqs)} unique (persona, advertisement) pairs")
    print(f"per persona min {min(len(v) for v in built['pools'].values())} "
          f"max {max(len(v) for v in built['pools'].values())}")
    tokens = chars // 4
    # Batch API is 50% of standard rates. Opus 5 standard is $5/$25 per MTok.
    in_usd = tokens / 1e6 * 2.50
    out_usd = len(reqs) * 800 / 1e6 * 12.50      # generous: thinking is on
    print(f"prompt     {chars:,} chars ~= {tokens:,} input tokens")
    print(f"est. cost  ~${in_usd:.2f} input + ~${out_usd:.2f} output = "
          f"~${in_usd + out_usd:.2f} at batch rates (50% of standard)")
    print(f"model      {args.model}")
    print(f"prompt ver {J.JUDGE_PROMPT_VERSION}")

    truncated = sum(1 for r in reqs
                    if J.TRUNCATION_NOTICE in r["params"]["messages"][0]["content"])
    print(f"truncated  {truncated} of {len(reqs)} advertisements hit the "
          f"{J.AD_TEXT_CAP:,}-char cap")

    if not args.submit and not args.collect:
        print("\nDRY RUN. Nothing submitted. Re-run with --submit to spend money "
              "and start\nthe protocol's one-revision clock.")
        return

    from finn_smart_search.ingest.anthropic_client import AnthropicBatchClient
    client = AnthropicBatchClient(model=args.model)

    if args.collect:
        batch_id = (json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))["batch_id"]
                    if args.collect == "manifest" else args.collect)
        print(f"\ncollecting existing batch {batch_id} — nothing resubmitted")
    else:
        batch_id = client.submit_batch(reqs)
        print(f"\nsubmitted batch {batch_id}")
    if not args.collect:
        OUT_MANIFEST.write_text(json.dumps({
            "batch_id": batch_id, "model": args.model,
            "judge_prompt_version": J.JUDGE_PROMPT_VERSION,
            "n_pairs": len(reqs),
            "arms": sorted(next(iter(built["rankings"].values()))),
            "personas": built["dev"], "depth": 10,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, indent=1), encoding="utf-8")
        print(f"manifest -> {OUT_MANIFEST.relative_to(ROOT)}")

    waited = 0
    while waited < args.max_wait:
        status = client.poll(batch_id)
        print(f"  [{waited:5d}s] {status}", flush=True)
        if status == "ended":
            break
        time.sleep(args.poll_seconds)
        waited += args.poll_seconds
    else:
        print(f"still running after {args.max_wait}s; batch {batch_id} is LIVE and "
              f"already paid for.\nRe-run with --collect (no argument) to pick it "
              f"up from the manifest — do NOT\nre-run with --submit, that would "
              f"pay for the same pool twice.")
        return

    ok, errored, rejected = [], [], []
    for raw in client.raw_results(batch_id):
        got = J.parse_judgment(raw)
        if not isinstance(got, J.Judgment):
            errored.append(got)
            continue
        pid, ad_i = built["index_of_pair"][got.pair_id]
        try:
            J.validate_judgment(got, built["full_text"][ad_i])
        except J.JudgeValidationError as e:
            rejected.append({"pair_id": got.pair_id, "reason": str(e),
                             "grade": got.grade, "violates": got.violates,
                             "evidence": got.constraint_evidence})
            continue
        ok.append({"pair_id": got.pair_id, "persona_id": pid,
                   "ad_uuid": built["uuids"][ad_i], "ad_index": ad_i,
                   "grade": got.grade, "violates": got.violates,
                   "violated_facet": got.violated_facet,
                   "constraint_evidence": got.constraint_evidence,
                   "notes": got.notes,
                   "judge_prompt_version": got.judge_prompt_version,
                   "model": args.model})

    total = len(ok) + len(errored) + len(rejected)
    print(f"\njudged     {len(ok)} of {total}")
    print(f"errored    {len(errored)}")
    # STANDARDS §4: a rejection rate is reported, never silent — it converts one
    # kind of error into another invisibly, exactly as the census's demotion rate
    # does.
    print(f"REJECTED   {len(rejected)} ({len(rejected)/max(total,1):.1%}) failed "
          f"validation")
    for r in rejected[:8]:
        print(f"   {r['pair_id']}: {r['reason'][:90]}")

    OUT_JUDGMENTS.write_text(json.dumps(
        {"judgments": ok, "rejected": rejected,
         "errored": [dict(e) for e in errored]}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nwritten {OUT_JUDGMENTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
