"""Re-extract `skills` over the whole corpus with the new prompt. Batch API.

WHY THIS IS WORTH RUNNING, measured rather than assumed. The original census
populated `skills` on 32.8% of advertisements, and of the 6,828 with nothing
extracted, 94.6% carry a requirements list 8+ lines long — the qualifications were
in the text and the prompt did not ask for them. Scored against the golden labels
(LIMITATIONS §15):

    census-v15    recall 0.104   F1 0.165    38 skills on 44 ads
    skills-v1     recall 0.861   F1 0.599   270 skills on 44 ads   (haiku-4.5)

MODEL: claude-haiku-4-5, the census's own model. The pilot measured Opus 5 at recall
0.868 and Haiku at 0.861 — within noise — for a sixth of the cost. Haiku paraphrases
more (5.6% of its skills failed verbatim validation against Opus's 0.8%), and every
one of those is REJECTED here rather than stored, so the cost of the cheaper model is
recall on the rejected rows, not fabrication in the data.

DEDUPLICATION, following D8 and the census's own path. `dedup.signature` is an exact
hash of normalised body text; the census used it to turn 10,166 advertisements into
~9,378 calls. Identical bodies get identical skills, so paying twice would be waste.

WHERE THE OUTPUT GOES, and why not into `ad_facets`. Overwriting
`ad_facets.facets.skills` would leave that row's `prompt_version` claiming
census-v15 for a value produced by skills-v1 — §4 requires every gold row to carry
its versions, and a mixed row carries neither honestly. So results land in
`reports/skills_census.json` with their own version stamp, and a separate load step
can put them in a new table when something downstream consumes them.

NOTHING READS `skills` YET. `constraints.py` reads only
`norwegian_requirement_level` and `stated_working_language`, and D7's
`skill_coverage` does not exist. This run makes the static demo substantively better
— it ships skill glosses at 32.8% coverage today — and prepares D7; it does not
change any current metric.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "reports" / "skills_census.json"
MANIFEST = ROOT / "reports" / "skills_census_manifest.json"
DEFAULT_MODEL = "claude-haiku-4-5"


def build(model: str):
    import duckdb
    from finn_smart_search.understanding import dedup
    from finn_smart_search.understanding import skills_prompt as S

    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    rows = con.execute("""SELECT uuid, title, description_text FROM ads
                          ORDER BY uuid""").fetchall()
    ads = [{"uuid": u, "title": t or "", "body": b or ""} for u, t, b in rows]

    # D8: exact hash of normalised text. One call per cluster, fanned out after.
    groups = dedup.cluster([{"uuid": a["uuid"], "description_text": a["body"]}
                            for a in ads])
    reps = set(dedup.representatives(groups))
    by_uuid = {a["uuid"]: a for a in ads}

    requests = []
    for u in sorted(reps):
        a = by_uuid[u]
        requests.append({
            "custom_id": u.replace("-", "")[:64],
            "params": {
                "model": model,
                "max_tokens": 8192,
                "tools": [S.skills_tool_schema()],
                "tool_choice": {"type": "tool", "name": S.SKILLS_TOOL_NAME},
                "messages": [{"role": "user",
                              "content": S.render_prompt(a["title"], a["body"])}],
            },
        })
    return ads, groups, requests, by_uuid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", nargs="?", const="manifest", metavar="BATCH_ID")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--max-wait", type=int, default=14400)
    args = ap.parse_args()

    from finn_smart_search.understanding import skills_prompt as S

    ads, groups, requests, by_uuid = build(args.model)
    chars = sum(len(r["params"]["messages"][0]["content"]) for r in requests)
    tok = chars // 4
    RATES = {"claude-haiku-4-5": (1.0, 5.0), "claude-opus-5": (5.0, 25.0)}
    p_in, p_out = RATES.get(args.model, (1.0, 5.0))
    # Batch is 50% of standard. Output estimated at the pilot's measured 365/ad.
    est = (tok / 1e6 * p_in + len(requests) * 365 / 1e6 * p_out) * 0.5
    print(f"corpus     {len(ads)} ads")
    print(f"clusters   {len(requests)} calls after dedup "
          f"({len(ads) - len(requests)} saved)")
    print(f"prompt     {tok:,} input tokens")
    print(f"model      {args.model}   prompt {S.SKILLS_PROMPT_VERSION}")
    print(f"est. cost  ~${est:.2f} at batch rates (50% of standard)")

    if not args.submit and not args.collect:
        print("\nDRY RUN. Re-run with --submit.")
        return

    from finn_smart_search.ingest.anthropic_client import (
        AnthropicBatchClient, chunk_bytes)
    client = AnthropicBatchClient(model=args.model)

    if args.collect:
        ids = (json.loads(MANIFEST.read_text(encoding="utf-8"))["batch_ids"]
               if args.collect == "manifest" else [args.collect])
        print(f"\ncollecting {len(ids)} existing batch(es) — nothing resubmitted")
    else:
        ids = []
        # A batch is capped in BYTES as well as in requests, and the census died on
        # a 413 because nothing counted bytes. chunk_bytes is the fix, reused.
        for chunk in chunk_bytes(requests):
            bid = client.submit_batch(chunk)
            ids.append(bid)
            print(f"  submitted {bid}  ({len(chunk)} requests)")
        MANIFEST.write_text(json.dumps({
            "batch_ids": ids, "model": args.model,
            "prompt_version": S.SKILLS_PROMPT_VERSION,
            "n_calls": len(requests), "n_ads": len(ads),
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, indent=1), encoding="utf-8")
        print(f"manifest -> {MANIFEST.relative_to(ROOT)}")

    waited = 0
    pending = set(ids)
    while pending and waited < args.max_wait:
        done = {b for b in pending if client.poll(b) == "ended"}
        pending -= done
        if pending:
            print(f"  [{waited:5d}s] {len(pending)} batch(es) in progress", flush=True)
            time.sleep(args.poll_seconds)
            waited += args.poll_seconds
    if pending:
        print(f"still running after {args.max_wait}s. Re-run with --collect.")
        return

    texts = {a["uuid"]: a["body"] for a in ads}
    short = {u.replace("-", "")[:64]: u for u in texts}
    per_rep: dict[str, list[dict]] = {}
    rejected: list[dict] = []
    errored: list[dict] = []
    for bid in ids:
        for raw in client.results(bid, S.SKILLS_TOOL_NAME):
            uuid = short.get(str(raw.get("custom_id")))
            if raw.get("type") != "succeeded" or uuid is None:
                errored.append({"custom_id": raw.get("custom_id"),
                                "type": raw.get("type"),
                                "error": str(raw.get("error"))[:160]})
                continue
            kept = []
            for rs in raw["facets"].get("skills") or []:
                sk = S.Skill(phrase=str(rs.get("phrase", "")),
                             gloss_en=str(rs.get("gloss_en", "")),
                             level=str(rs.get("level", "")))
                try:
                    S.validate_skill(sk, texts[uuid])
                except S.SkillsValidationError as e:
                    rejected.append({"uuid": uuid, "phrase": sk.phrase,
                                     "reason": str(e)[:120]})
                    continue
                kept.append({"phrase": sk.phrase, "gloss_en": sk.gloss_en,
                             "level": sk.level})
            per_rep[uuid] = kept

    # Fan the cluster representative's skills out to every member (D8).
    out: dict[str, list[dict]] = {}
    for _, members in groups.items():
        rep = next((m for m in members if m in per_rep), None)
        if rep is None:
            continue
        for m in members:
            out[m] = per_rep[rep]

    n_sk = sum(len(v) for v in out.values())
    n_zero = sum(1 for v in out.values() if not v)
    total_returned = n_sk + len(rejected)
    print(f"\nextracted  {len(out)} of {len(ads)} ads, {n_sk:,} skills")
    print(f"zero       {n_zero} ads ({n_zero / max(len(out), 1):.1%}) "
          f"— census had 6,828 (67.2%)")
    # STANDARDS §4: a rejection rate is reported, never silent.
    print(f"REJECTED   {len(rejected)} of {total_returned} returned "
          f"({len(rejected) / max(total_returned, 1):.1%}) failed verbatim validation")
    print(f"errored    {len(errored)}")

    OUT.write_text(json.dumps({
        "prompt_version": S.SKILLS_PROMPT_VERSION, "model": args.model,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_ads": len(out), "n_skills": n_sk, "n_zero": n_zero,
        "n_rejected": len(rejected), "n_errored": len(errored),
        "rejected_sample": rejected[:40], "errored_sample": errored[:20],
        "skills": out,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")
    print("NOT written into ad_facets: that row's prompt_version says census-v15 "
          "and\na mixed row would carry neither version honestly (§4).")


if __name__ == "__main__":
    main()
