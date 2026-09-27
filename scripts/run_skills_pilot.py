"""Pilot the skills prompt on the 44 golden advertisements. One-off orchestration.

WHY 44 AND WHY THESE 44. `PLAN.md` B2 is "Pilot 50 ads" and this repository has
thirteen pilot logs plus a commit titled "The 400-ad census stage found a bug this
session introduced", so pilot-before-scale is established practice here rather than
caution. The golden advertisements are the only ones with labels, so one run answers
both questions at once: does the prompt behave as intended, and does it beat the
census's measured recall of 0.104 against a gate.

WHAT THIS ANSWERS AND WHAT IT CANNOT. It measures agreement with labels a model
produced (LIMITATIONS §15), so a good number is an UPPER BOUND on quality and must
be reported as model agreement. What it can establish on its own terms:

  - the tool schema is accepted by the live endpoint at all, which the judge batch
    taught is not free — 346 requests were rejected wholesale on a `custom_id`
    pattern that looked fine locally;
  - whether the prompt returns credentials, duties or personal qualities, which the
    exclusions exist to prevent and which no local test can confirm;
  - the real cost per advertisement on a COLD prompt cache, since the census's
    $22.59 rested on a 97% hit rate that a new prompt does not inherit.

Synchronous Messages API rather than the Batch API: 44 requests, and the batch path
just spent 75 minutes on 346. Cost is the same per token; only the 50% batch
discount is forgone, which on this pilot is cents.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "reports" / "skills_pilot.json"
GOLDEN_SKILLS = ROOT / "eval" / "golden_skills.json"
DEFAULT_MODEL = "claude-opus-5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="spend money")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="first N ads only")
    args = ap.parse_args()

    import duckdb
    import httpx
    from finn_smart_search.eval import skills_scoring as SS
    from finn_smart_search.ingest.anthropic_client import api_key
    from finn_smart_search.understanding import skills_prompt as S

    labels = json.loads(GOLDEN_SKILLS.read_text(encoding="utf-8"))
    golden = {a["uuid"]: a["skills"] for a in labels["ads"]}
    uuids = sorted(golden)
    if args.limit:
        uuids = uuids[:args.limit]

    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    rows = dict(con.execute(
        "SELECT uuid, title FROM ads").fetchall())
    texts = dict(con.execute(
        "SELECT uuid, description_text FROM ads").fetchall())
    facets = dict(con.execute("SELECT uuid, facets FROM ad_facets").fetchall())

    census = {}
    for u in uuids:
        raw = facets[u]
        f = json.loads(raw) if isinstance(raw, str) else raw
        census[u] = [{"phrase": s.get("phrase", ""), "gloss_en": "",
                      "level": s.get("level", "required")}
                     for s in (f.get("skills") or [])]

    prompts = {u: S.render_prompt(rows[u] or "", texts[u] or "") for u in uuids}
    chars = sum(len(p) for p in prompts.values())
    print(f"ads        {len(uuids)}")
    print(f"prompt     {chars:,} chars ~= {chars // 4:,} input tokens")
    print(f"model      {args.model}   prompt {S.SKILLS_PROMPT_VERSION}")
    # Opus 5 standard rates, NOT batch: $5/MTok in, $25/MTok out.
    est_in = chars / 4 / 1e6 * 5.0
    est_out = len(uuids) * 1500 / 1e6 * 25.0
    print(f"est. cost  ~${est_in:.2f} in + ~${est_out:.2f} out = ~${est_in + est_out:.2f} "
          f"(standard rates; no batch discount)")
    print(f"golden     {sum(len(golden[u]) for u in uuids)} skills")
    print(f"census     {sum(len(census[u]) for u in uuids)} skills "
          f"({sum(1 for u in uuids if not census[u])} ads with zero)")

    if not args.submit:
        print("\nDRY RUN. Re-run with --submit to spend money.")
        return

    headers = {"x-api-key": api_key(), "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    tool = S.skills_tool_schema()
    predictions: dict[str, list[dict]] = {}
    rejected: list[dict] = []
    failures: list[dict] = []
    usage = {"in": 0, "out": 0}
    t0 = time.time()

    with httpx.Client(timeout=300) as client:
        for n, u in enumerate(uuids, 1):
            body = {
                "model": args.model,
                "max_tokens": 8192,
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": S.SKILLS_TOOL_NAME},
                "messages": [{"role": "user", "content": prompts[u]}],
            }
            try:
                r = client.post("https://api.anthropic.com/v1/messages",
                                headers=headers, json=body)
                if r.status_code != 200:
                    failures.append({"uuid": u, "status": r.status_code,
                                     "body": r.text[:300]})
                    print(f"  [{n:2d}/{len(uuids)}] {u[:8]} HTTP {r.status_code} "
                          f"{r.text[:120]}")
                    continue
                msg = r.json()
            except httpx.HTTPError as e:              # network, not a rejection
                failures.append({"uuid": u, "error": str(e)[:200]})
                continue

            usage["in"] += msg.get("usage", {}).get("input_tokens", 0)
            usage["out"] += msg.get("usage", {}).get("output_tokens", 0)

            # A truncated tool call is an ERROR, never salvaged — a half-parsed
            # skills array is indistinguishable from a complete one downstream.
            if msg.get("stop_reason") == "max_tokens":
                failures.append({"uuid": u, "error": "stop_reason=max_tokens"})
                continue
            block = next((b for b in msg.get("content") or []
                          if b.get("type") == "tool_use"
                          and b.get("name") == S.SKILLS_TOOL_NAME), None)
            if block is None:
                failures.append({"uuid": u, "error": "no tool_use block"})
                continue

            kept = []
            for raw_skill in block["input"].get("skills") or []:
                sk = S.Skill(phrase=str(raw_skill.get("phrase", "")),
                             gloss_en=str(raw_skill.get("gloss_en", "")),
                             level=str(raw_skill.get("level", "")))
                try:
                    S.validate_skill(sk, texts[u] or "")
                except S.SkillsValidationError as e:
                    rejected.append({"uuid": u, "phrase": sk.phrase,
                                     "reason": str(e)[:150]})
                    continue
                kept.append({"phrase": sk.phrase, "gloss_en": sk.gloss_en,
                             "level": sk.level})
            predictions[u] = kept
            print(f"  [{n:2d}/{len(uuids)}] {u[:8]} {len(kept):2d} skills "
                  f"(golden {len(golden[u])}, census {len(census[u])})", flush=True)

    elapsed = time.time() - t0
    cost = usage["in"] / 1e6 * 5.0 + usage["out"] / 1e6 * 25.0
    print(f"\n{len(predictions)} of {len(uuids)} ads extracted in {elapsed:.0f}s")
    print(f"tokens in={usage['in']:,} out={usage['out']:,}  ACTUAL COST ${cost:.2f} "
          f"= ${cost / max(len(predictions), 1):.4f}/ad")
    print(f"  -> full corpus at this rate: ${cost / max(len(predictions), 1) * 10166:.0f} "
          f"standard, ~${cost / max(len(predictions), 1) * 10166 / 2:.0f} batched")
    if failures:
        print(f"\nREQUEST FAILURES {len(failures)}")
        for f in failures[:6]:
            print(f"   {f}")

    # STANDARDS §4: a rejection rate is reported, never silent.
    total_returned = sum(len(v) for v in predictions.values()) + len(rejected)
    print(f"\nVALIDATION: {len(rejected)} of {total_returned} returned skills "
          f"rejected ({len(rejected) / max(total_returned, 1):.1%})")
    for r in rejected[:8]:
        print(f"   {r['uuid'][:8]} {r['phrase'][:48]!r}: {r['reason'][:70]}")

    report = SS.score(predictions, {u: golden[u] for u in predictions},
                      census_baseline=census)
    print("\n=== SCORED AGAINST THE GOLDEN LABELS ===")
    for k in ("n_ads", "n_golden", "n_predicted", "n_paired", "correct_zeros"):
        print(f"  {k:22s} {report[k]}")
    for k in ("precision", "recall", "f1", "level_agreement", "gloss_agreement"):
        v = report[k]
        print(f"  {k:22s} {'n/a' if v is None else round(v, 3)}")
    print(f"  agreement              {report['agreement']}")
    print(f"  unpaired golden        {report['unpaired_golden']} (missed)")
    print(f"  unpaired predicted     {report['unpaired_predictions']} "
          f"(NOT false positives — §15)")
    print(f"\n  CENSUS BASELINE        recall 0.104, F1 0.165, 38 skills")
    if report["recall"] is not None:
        print(f"  THIS RUN               recall {report['recall']:.3f}, "
              f"F1 {report['f1']:.3f}, {report['n_predicted']} skills")

    OUT.write_text(json.dumps({
        "model": args.model, "prompt_version": S.SKILLS_PROMPT_VERSION,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "usage": usage, "cost_usd": cost, "elapsed_s": elapsed,
        "report": report, "rejected": rejected, "failures": failures,
        "predictions": predictions,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
