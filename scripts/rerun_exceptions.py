"""Re-extract the 46 exception ads under the CURRENT prompt, before review.

WHY. The exception rows were selected from a census run under census-v7/v8.
census-v9 changed what the model is told about where a documentation-language
clause goes, and the enum validator now demotes out-of-enum values. Reviewing
v8 output would be reviewing a version that will never ship: every verdict
would be spent on a value that has already been replaced.

So each row is re-run and carries BOTH values. Where they differ the page says
so, which also makes the review a direct test of the v9 fix.

No DB write: this is a review input, not a census. The full census runs after.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import duckdb                                                      # noqa: E402
from finn_smart_search.ingest import anthropic_client as ac        # noqa: E402
from finn_smart_search.understanding import census                 # noqa: E402
from finn_smart_search.understanding.census_prompt import PROMPT_VERSION  # noqa: E402

SRC = Path("/private/tmp/claude-501/-Users-naomi-Documents-projects/"
           "2d8ed1e8-68bf-4a23-bed2-3527063a5b59/scratchpad/exceptions.json")
OUT = ROOT / "data" / "worksheet" / "exceptions_current.json"


def main() -> None:
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)

    # one request per AD, not per row: 5 uuids carry two exception rows each
    uuids = sorted({r["uuid"] for r in rows})
    reqs = []
    for u in uuids:
        a = con.execute("""SELECT a.title, a.description_text, l.doc_lang
                           FROM ads a JOIN ad_language l USING (uuid)
                           WHERE a.uuid=?""", [u]).fetchone()
        reqs.append(census.build_request({"uuid": u, "title": a[0],
                                          "description_text": a[1], "doc_lang": a[2]}))

    print(f"{PROMPT_VERSION}: re-running {len(reqs)} ads for {len(rows)} rows",
          flush=True)
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

    got = {p["custom_id"]: p for p in c.results(bid)}
    missing = [u for u in uuids if u not in got]
    if missing:
        sys.exit(f"{len(missing)} ads returned nothing: {missing[:3]}")

    changed = 0
    out = []
    for r in rows:
        res = got[r["uuid"]]
        f = res.get("facets") or {}
        new_val = f.get(r["field"])
        rec = dict(r)
        rec["value_v8"] = r["value"]
        rec["value"] = new_val
        rec["changed"] = str(new_val) != str(r["value"])
        rec["level"] = f.get("norwegian_requirement_level")
        # evidence_spans are objects here and plain strings in the review's
        # item contract. Normalise at the boundary, not in the consumer.
        rec["spans"] = [e.get("span", "") if isinstance(e, dict) else str(e)
                        for e in (f.get("evidence_spans") or [])]
        rec["demoted"] = bool(res.get("demoted"))
        rec["reasons"] = res.get("_reasons") or []
        changed += rec["changed"]
        out.append(rec)

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{changed} of {len(out)} rows changed value under {PROMPT_VERSION}")
    for r in out:
        if r["changed"]:
            print(f"   {r['field']:28s} {r['value_v8']!r} -> {r['value']!r}")
    still_rare = sum(1 for r in out if str(r["value"]) == str(r["value_v8"]))
    print(f"\n{still_rare} rows still hold the rare value and remain exceptions")
    print(f"written {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
