"""Does the extractor FIND a security check, not just classify the ones it sees?

WHY. `security_clearance_required` was a bare boolean with no description
anywhere, and on ads naming a politiattest and no clearance the extractor said
True on 4 and False on 23. The reviewer set the rule — ANY security vetting
counts — and against it the persisted records scored 13 correct True, 0 wrong
True and 28 MISSED. It under-fires.

The exceptions review cannot see that, by construction: it selects rows where
the value is the RARE one, so it only ever holds rows the extractor fired on.
Every one of the 28 is invisible to it. This probe selects by the CLAUSE in the
ad text instead, so it measures recall.

THE TRAP, measured: 1,625 ads say `autorisasjon` and 726 of those have no
security check at all — it is 'norsk autorisasjon som sykepleier', a licence.
Those must stay False, so the probe scores them as a precision set alongside
the recall set. A prompt that reaches 100% recall by saying True to every
`autorisasjon` has not improved anything.

No DB write.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import duckdb                                                   # noqa: E402
from finn_smart_search.ingest import anthropic_client as ac     # noqa: E402
from finn_smart_search.understanding import census              # noqa: E402
from finn_smart_search.understanding.census_prompt import PROMPT_VERSION  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = ROOT / "reports" / "security_recall.json"

VET = re.compile(r"politiattest|vandelsattest|plettfri vandel|god vandel"
                 r"|bakgrunnssjekk|sikkerhetsklarer|sikkerhetsklarering"
                 r"|skjermingsverdig|security clearance", re.I)
# a professional licence, not vetting — must stay False
LIC = re.compile(r"autorisasjon som|norsk autorisasjon|HPR-nummer", re.I)
HEALTH = re.compile(r"\bMRSA\b|tuberkulose", re.I)


def main() -> None:
    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    rows = con.execute("""SELECT a.uuid, a.title, a.description_text, l.doc_lang
                          FROM ads a JOIN ad_language l USING (uuid)
                          ORDER BY a.uuid""").fetchall()

    recall, precision = [], []
    for u, t, b, dl in rows:
        ad = {"uuid": u, "title": t, "description_text": b, "doc_lang": dl}
        if VET.search(b or ""):
            if len(recall) < N:
                m = VET.search(b)
                ad["clause"] = b[max(0, m.start() - 40):m.end() + 60].replace("\n", " ")
                ad["expected"] = True
                recall.append(ad)
        elif (LIC.search(b or "") or HEALTH.search(b or "")):
            if len(precision) < N // 2:
                m = LIC.search(b) or HEALTH.search(b)
                ad["clause"] = b[max(0, m.start() - 40):m.end() + 60].replace("\n", " ")
                ad["expected"] = False
                precision.append(ad)
        if len(recall) >= N and len(precision) >= N // 2:
            break

    picked = recall + precision
    print(f"{PROMPT_VERSION}: {len(recall)} ads naming a security check "
          f"(expect True) + {len(precision)} naming only a licence or health "
          f"screening (expect False)", flush=True)

    c = ac.AnthropicBatchClient()
    bid = c.submit_batch([census.build_request(picked[0])])      # warm
    for _ in range(200):
        if c.poll(bid) == "ended":
            break
        time.sleep(10)
    got = {p["custom_id"]: p for p in c.results(bid)}
    bid = c.submit_batch([census.build_request(a) for a in picked[1:]])
    print(f"batch {bid}", flush=True)
    for _ in range(400):
        st = c.poll(bid)
        if st == "ended":
            break
        if st != "in_progress":
            sys.exit(f"terminal state {st}")
        time.sleep(10)
    got.update({p["custom_id"]: p for p in c.results(bid)})

    detail, tp, fn, tn, fp = [], 0, 0, 0, 0
    for a in picked:
        v = bool(((got.get(a["uuid"]) or {}).get("facets") or {})
                 .get("security_clearance_required"))
        if a["expected"] and v:
            tp += 1
        elif a["expected"]:
            fn += 1
        elif v:
            fp += 1
        else:
            tn += 1
        detail.append({"uuid": a["uuid"], "expected": a["expected"], "got": v,
                       "clause": a["clause"]})

    print(f"\n  RECALL    on ads naming a check : {tp}/{tp+fn}"
          f"  ({tp/max(1,tp+fn):.0%})   missed {fn}")
    print(f"  PRECISION on licence/health only: {tn}/{tn+fp}"
          f"  ({tn/max(1,tn+fp):.0%})   wrongly True {fp}")
    for d in detail:
        if d["expected"] != d["got"]:
            print(f"    {d['uuid'][:8]} want {d['expected']} got {d['got']}  "
                  f"{d['clause'][:90]}")
    OUT.write_text(json.dumps({"prompt_version": PROMPT_VERSION, "recall_tp": tp,
                               "recall_fn": fn, "prec_tn": tn, "prec_fp": fp,
                               "detail": detail}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
