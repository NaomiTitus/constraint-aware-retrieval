"""Run the facet census over the ACTIVE corpus.

STAGED BY DEFAULT, and staging is free: `llm_cache` is keyed on
model + PROMPT_VERSION + normalised text, so every ad done in a stage is a
cache hit in the next run and is never billed twice. `--limit N` does one
stage; re-running without it completes the corpus and pays only for what is
left.

WHAT A STAGE IS FOR, given ~215 ads have already been through census-v15
end to end (the 44-ad golden set with DB writes, the 46 exception rows, and
three probes). The prompt, schema, parsing, validation and persistence are
exercised. What is NOT exercised is scale:

  PROMPT-CACHE TTL over a long batch. Each read refreshes a 5-minute window.
  The 44-ad run held 98% over 470s; a 9,823-request batch runs far longer and
  if throughput ever stalls past the TTL the prefix expires and later
  requests re-write it at 1.25x. Measured, that is $22 against roughly $60.

  DEMOTION RATE. census.run aborts above --max-demotion-rate, AFTER the batch
  is paid for. A stage gives the rate a usable interval first.

  THE WRITE PATH at volume, and the fan-out to every cluster member.

So the number to watch in a stage is `prompt-cache hit rate`, then the
demotion rate, then cost per ad. A five-ad stage would show none of them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import duckdb                                                   # noqa: E402
from finn_smart_search.ingest import anthropic_client as ac     # noqa: E402
from finn_smart_search.understanding import census              # noqa: E402
from finn_smart_search.understanding.census_prompt import PROMPT_VERSION  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="stage size; omit to complete the corpus")
    ap.add_argument("--max-spend", type=float, default=60.0,
                    help="abort above this, in USD")
    ap.add_argument("--max-demotion-rate", type=float, default=0.15)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be sent and stop")
    a = ap.parse_args()

    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"))
    ads = [dict(uuid=u, title=t, description_text=b, doc_lang=dl)
           for u, t, b, dl in con.execute(
               """SELECT a.uuid, a.title, a.description_text, l.doc_lang
                  FROM ads a JOIN ad_language l USING (uuid)""").fetchall()]

    done = {r[0] for r in con.execute(
        "SELECT DISTINCT key FROM llm_cache WHERE prompt_version = ?",
        [PROMPT_VERSION]).fetchall()}
    remaining = [x for x in ads
                 if census.cache_key(census.MODEL, PROMPT_VERSION,
                                     x["description_text"]) not in done]
    print(f"{PROMPT_VERSION}  corpus {len(ads)}  already cached "
          f"{len(ads) - len(remaining)}  remaining {len(remaining)}", flush=True)
    if a.limit:
        print(f"STAGE: {a.limit} ads. Re-run without --limit to finish; the "
              f"stage is a cache hit then and is not billed again.", flush=True)
    if a.dry_run:
        return

    t0 = time.time()
    out = census.run(ads, ac.AnthropicBatchClient(), con,
                     limit=a.limit, max_spend_usd=a.max_spend,
                     max_demotion_rate=a.max_demotion_rate, poll_seconds=10)

    tok = out.get("tokens") or {}
    n = max(1, out["calls"])
    print(f"\ncensus: {time.time()-t0:.0f}s  calls={out['calls']}  "
          f"disk-cache hits={out['cache_hits']}  "
          f"demoted={out['demoted']} ({out['demotion_rate']:.1%})  "
          f"spend=${out['spend_usd']:.2f}")
    if tok:
        print(f"tokens: in={tok['in']:,} out={tok['out']:,} "
              f"cache_read={tok['cache_read']:,} cache_write={tok['cache_write']:,}")
        print(f"PROMPT-CACHE HIT RATE {out['prompt_cache_hit_rate']:.0%}"
              f"   <- the number that sets the bill")
        per = out["spend_usd"] / n
        print(f"${per:.5f}/ad   full corpus would be ${per * len(ads):.2f}")
    if out["failed"]:
        print(f"FAILED {len(out['failed'])}: "
              f"{list(out['failed'].items())[:3]}")
    print(f"levels: {dict(out['levels'])}")


if __name__ == "__main__":
    main()
