"""Does the prompt cache actually pay off inside ONE batch?

WHY THIS EXISTS. census_prompt.build_request marks the system block and the
few-shot prefix with cache_control and its docstring claims the prefix "is
billed at 0.1x on every call after the first". That is true of SERIAL calls. The
census submits a single batch of ~9,823 requests, and a batch's requests are
dispatched together — so an unknown fraction of them start before any of them
has written the cache, and each of those pays 1.25x instead of 0.1x.

The difference is not small. The prefix is ~10k tokens:
    every call reads   9,823 * 10,000 * 0.1  * $0.50/MTok  =  $4.9
    every call writes  9,823 * 10,000 * 1.25 * $0.50/MTok  = $61.4
    no cache_control   9,823 * 10,000 * 1.0  * $0.50/MTok  = $49.1
So the census bill is somewhere between $5 and $61 depending on a fact nobody
has measured, and the plan's figure was computed from the optimistic end.

This probe submits ONE batch of N requests with the identical prefix and reports
the read/write split. It costs about a cent and it decides a $45 question.

No DB write.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import duckdb                                                   # noqa: E402
from finn_smart_search.ingest import anthropic_client as ac     # noqa: E402
from finn_smart_search.understanding import census              # noqa: E402
from finn_smart_search.understanding.census_prompt import PROMPT_VERSION  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24


def main() -> None:
    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    ads = con.execute("""SELECT a.uuid, a.title, a.description_text, l.doc_lang
                         FROM ads a JOIN ad_language l USING (uuid)
                         WHERE length(a.description_text) BETWEEN 1500 AND 4000
                         ORDER BY a.uuid LIMIT ?""", [N]).fetchall()
    reqs = [census.build_request({"uuid": u, "title": t,
                                  "description_text": b, "doc_lang": dl})
            for u, t, b, dl in ads]
    print(f"{PROMPT_VERSION}: probing prompt-cache behaviour on {len(reqs)} "
          f"requests in ONE batch", flush=True)

    c = ac.AnthropicBatchClient()
    bid = c.submit_batch(reqs)
    print(f"batch {bid}", flush=True)
    for _ in range(400):
        st = c.poll(bid)
        if st == "ended":
            break
        if st != "in_progress":
            sys.exit(f"terminal state {st}")
        time.sleep(10)
    else:
        sys.exit("timed out")

    tin = tout = tread = twrite = 0
    per_call = []
    for p in c.results(bid):
        u = p.get("usage") or {}
        tin += u.get("input_tokens", 0)
        tout += u.get("output_tokens", 0)
        tread += u.get("cache_read_input_tokens", 0)
        twrite += u.get("cache_creation_input_tokens", 0)
        per_call.append((u.get("cache_read_input_tokens", 0),
                         u.get("cache_creation_input_tokens", 0)))

    n = len(per_call)
    readers = sum(1 for r, w in per_call if r > 0)
    writers = sum(1 for r, w in per_call if w > 0)
    spend = (tin * census.PRICE_IN + tout * census.PRICE_OUT
             + tread * census.PRICE_CACHE_READ + twrite * census.PRICE_CACHE_WRITE)

    print(f"\n  requests            {n}")
    print(f"  cache READERS       {readers} ({readers/n:.0%})")
    print(f"  cache WRITERS       {writers} ({writers/n:.0%})")
    print(f"  tokens  in={tin:,}  out={tout:,}  read={tread:,}  write={twrite:,}")
    print(f"  spend               ${spend:.4f}   (${spend/n:.5f}/ad)")
    print(f"\n  EXTRAPOLATED to 9,823 clusters: ${spend/n*9823:.2f}")
    print("  (the plan's figure was $16.71, computed assuming the prefix is "
          "read at 0.1x on every call after the first)")
    if writers > n * 0.5:
        print("\n  VERDICT: most requests in one batch WRITE the prefix rather "
              "than read it.\n  Prompt caching does not pay inside a single "
              "large batch — and at 1.25x it costs MORE\n  than not marking the "
              "prefix at all.")
    else:
        print("\n  VERDICT: the batch does share the prefix cache.")


if __name__ == "__main__":
    main()
