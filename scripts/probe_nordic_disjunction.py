"""`norsk eller <another Nordic language>` — does English get asserted anyway?

WHY A PROBE AND NOT THE GOLDEN SET. The golden set holds 8 ads labelled
`scandinavian_accepted` and its score on them has bounced 8/8, 7/8, 7/8, 6/8
across four prompt versions, while three of its 44 ads had their decisive line
sitting in the prompt. Eight items cannot tell a real change from noise, and
four consecutive prompt versions have now been steered partly by it — which is
the over-fitting LIMITATIONS §2 already declares.

344 corpus ads (3.38%) are written `norsk eller <skandinavisk|nordisk|svensk|
dansk>` with no English anywhere. Answering `either_norwegian_or_english` on
them asserts English is accepted where it is not: a `shown wrongly` error, the
direction that costs a seeker an application, and the exact failure this
project exists to fix.

Selected by the CLAUSE, excluding every golden and sealed uuid. No DB write.

RESULT, census-v15, 40 held-out ads:
    asserted English (the error hunted)   0   (0%)
    scandinavian_accepted                24  (60%)
    other level                          16  (40%)

READ THE 40% BEFORE CALLING IT A FAILURE RATE — it is this probe's
mis-specification, not the extractor's error. The pattern
`norsk ... eller ... svensk|dansk|skandinavisk` matches two quite different
sentences, and only the first means "any Scandinavian language is acceptable":

    "Behersker norsk eller et annet skandinavisk språk"
        -> scandinavian_accepted.  The shape this probe was built for.

    "Søkere som har annet morsmål enn norsk, svensk eller dansk, og som ikke
     har fullført norskspråklig videregående skole, må dokumentere ..."
        -> `certified`, and correctly so. The Nordic languages are named to
           say WHO MUST DOCUMENT their Norwegian, not what is accepted.
           13 of the 16 are this boilerplate.

    "gratis norskkurs dersom du ikke behersker norsk, svensk eller dansk"
        -> `desirable`. They will teach you; Norwegian is not a requirement.

So the number to read is the FIRST line: the asserted-English error is absent
on held-out ads. The golden set shows 2 such errors in 8 items; this probe
shows 0 in 40. Eight items could not distinguish a live class from two hard
ads, which is why this probe exists.
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = ROOT / "reports" / "nordic_disjunction.json"

NORDIC_OR = re.compile(r"\bnorsk\w*\b[^.\n]{0,40}\beller\b[^.\n]{0,40}"
                       r"(skandinavisk|nordisk|svensk|dansk)\w*", re.I)
EN = re.compile(r"engelsk|english", re.I)
ACCEPT = {"scandinavian_accepted"}
# the error we are hunting: English asserted where the ad never mentions it
WRONG = {"either_norwegian_or_english"}


def main() -> None:
    golden = {g["uuid"] for g in json.loads((ROOT / "eval" / "golden_set.json")
                                            .read_text(encoding="utf-8"))}
    sealed = set()
    sp = ROOT / "reviews" / "001-sealed-language" / "labels.json"
    if sp.exists():
        sealed = {l["uuid"] for l in json.loads(sp.read_text(encoding="utf-8"))}

    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    rows = con.execute("""SELECT a.uuid, a.title, a.description_text, l.doc_lang
                          FROM ads a JOIN ad_language l USING (uuid)
                          ORDER BY a.uuid""").fetchall()
    picked = []
    for u, t, b, dl in rows:
        if u in golden or u in sealed:
            continue
        for s in re.split(r"(?<=[.!?])\s+|\n", b or ""):
            s = s.strip()
            if 15 < len(s) < 220 and NORDIC_OR.search(s) and not EN.search(s):
                picked.append({"uuid": u, "title": t, "description_text": b,
                               "doc_lang": dl, "clause": s})
                break
        if len(picked) >= N:
            break

    print(f"{PROMPT_VERSION}: {len(picked)} held-out ads with a Nordic-only "
          f"disjunction (no golden, no sealed)", flush=True)
    c = ac.AnthropicBatchClient()
    bid = c.submit_batch([census.build_request(picked[0])])
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

    ok = asserted_english = other = 0
    detail = []
    for a in picked:
        lvl = (((got.get(a["uuid"]) or {}).get("facets") or {})
               .get("norwegian_requirement_level"))
        if lvl in ACCEPT:
            ok += 1
            state = "ok"
        elif lvl in WRONG:
            asserted_english += 1
            state = "ASSERTED_ENGLISH"
        else:
            other += 1
            state = f"other:{lvl}"
        detail.append({"uuid": a["uuid"], "level": lvl, "state": state,
                       "clause": a["clause"]})

    n = len(picked)
    print(f"\n  scandinavian_accepted            {ok:3d}  ({ok/n:.0%})")
    print(f"  ASSERTED ENGLISH (shown wrongly) {asserted_english:3d}  "
          f"({asserted_english/n:.0%})   <- the error being hunted")
    print(f"  other level                      {other:3d}  ({other/n:.0%})")
    for d in detail:
        if d["state"] != "ok":
            print(f"    {d['uuid'][:8]} {str(d['level']):28s} {d['clause'][:78]}")
    OUT.write_text(json.dumps({"prompt_version": PROMPT_VERSION, "n": n,
                               "scandinavian_accepted": ok,
                               "asserted_english": asserted_english,
                               "other": other, "detail": detail},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
