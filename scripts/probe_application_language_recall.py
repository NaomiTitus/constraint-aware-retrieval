"""Does the extractor FIND a documentation-language clause, not just file it right?

WHY THIS EXISTS. Review 002 found 3 of 4 `application_language` values were
borrowed from a neighbouring enum, and census-v9/v10 fixed that. But every one
of those 4 was a FALSE POSITIVE — a value the model did set. The exceptions
review is selected on the RARE value, so by construction it can only ever see
false positives. The modal value is `unstated`, and a missed clause lands there
and is invisible to the review, to the golden set, and to every metric.

Measured on what is persisted today: of the records carrying a
documentation-language clause, **6 of 8 under census-v6 and 3 of 4 under
census-v8 recorded `unstated`** — a recall of about 25%. So the silent misses
are the larger error by a wide margin, and the v9/v10 work never touched them.

This probe selects ads by the CLAUSE, in the ad text, before the model sees
them — so it measures recall honestly rather than re-reading the model's own
output. No DB write.
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
OUT = ROOT / "reports" / "application_language_recall.json"

# A clause about the language of the PAPERWORK.
#
# THE FIRST VERSION OF THIS SELECTOR WAS WRONG, and its number must not be
# quoted. It required only a document word, a language word and a modal
# ANYWHERE in the same block, and reported 35% of ads "missed". Reading the
# misses showed most were not application-language clauses at all:
#
#   "norsk autorisasjon som helsefagarbeider (HPR-nummer oppgis i søknaden)"
#   "Må beherske norsk språk skriftlig og muntlig"
#   "For søknad og meir informasjon ta kontakt med: ... Tlf"
#
# The first is the `norsk autorisasjon` precision trap the project plan names
# explicitly; the second is a proficiency requirement; the third is noise. The
# model recording `unstated` for all three is CORRECT, and the probe was about
# to book them as extractor misses. Tenth instance of this project's signature
# failure — a hand-written pattern that looks right — and it nearly produced a
# headline number that was pure measurement error.
#
# The selector now requires the document noun and the language to be BOUND by
# a verb within one short clause, and excludes the licence traps by name.
# Hand-checked: 25 of 25 sampled hits are genuine, 188 ads corpus-wide.
DOCWORD = (r"(?:s[øo]knad\w*|application|dokument\w*|documentation|vedlegg"
           r"|vitnem[åa]l\w*|attest\w*|CV)")
LANGWORD = r"(?:norsk|engelsk|english|skandinavisk|scandinavian|norwegian|nordisk)\w*"
BIND = r"(?:m[åa]|skal|must|be|vere|være|skrives|skrivast|written|oversatt|omsett)"
CLAUSE = re.compile(
    rf"{DOCWORD}[^.\n]{{0,60}}\b{BIND}\b[^.\n]{{0,60}}?\bp[åa]\s+{LANGWORD}"
    rf"|{DOCWORD}[^.\n]{{0,60}}\b{BIND}\b[^.\n]{{0,40}}oversatt til\s+{LANGWORD}"
    rf"|{DOCWORD}[^.\n]{{0,60}}\bmust be\b[^.\n]{{0,60}}\bin (?:a )?{LANGWORD}", re.I)
# Licence and proficiency, NOT paperwork language. These are the trap.
TRAP = re.compile(r"autorisasjon|HPR|norskpr[øo]ve|Bergenstesten|godkjenning", re.I)
EN = re.compile(r"\b(engelsk\w*|english)\b", re.I)


def clause_of(text: str) -> str | None:
    for s in re.split(r"(?<=[.!?])\s+|\n", text or ""):
        s = s.strip()
        if 15 < len(s) < 300 and CLAUSE.search(s) and not TRAP.search(s):
            return s
    return None


def expected(clause: str) -> str:
    """What the prompt's own rule says this clause should produce."""
    return "english_accepted" if EN.search(clause) else "norwegian_required"


def main() -> None:
    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    rows = con.execute("""SELECT a.uuid, a.title, a.description_text, l.doc_lang
                          FROM ads a JOIN ad_language l USING (uuid)
                          ORDER BY a.uuid""").fetchall()

    picked = []
    for u, t, b, dl in rows:
        c = clause_of(b)
        if c:
            picked.append({"uuid": u, "title": t, "description_text": b,
                           "doc_lang": dl, "clause": c, "expected": expected(c)})
        if len(picked) >= N:
            break

    print(f"{PROMPT_VERSION}: recall probe on {len(picked)} ads selected by "
          f"CLAUSE, not by model output", flush=True)
    c = ac.AnthropicBatchClient()
    # warm the prefix first, same reason as census.run
    bid = c.submit_batch([census.build_request(picked[0])])
    for _ in range(200):
        if c.poll(bid) == "ended":
            break
        time.sleep(10)
    warm = {p["custom_id"]: p for p in c.results(bid)}

    bid = c.submit_batch([census.build_request(a) for a in picked[1:]])
    print(f"batch {bid}", flush=True)
    for _ in range(400):
        st = c.poll(bid)
        if st == "ended":
            break
        if st != "in_progress":
            sys.exit(f"terminal state {st}")
        time.sleep(10)
    got = dict(warm)
    got.update({p["custom_id"]: p for p in c.results(bid)})

    found = missed = wrong = 0
    detail = []
    for a in picked:
        f = (got.get(a["uuid"]) or {}).get("facets") or {}
        v = f.get("application_language")
        state = ("missed" if v == "unstated" else
                 "found" if v == a["expected"] else "wrong_value")
        found += state == "found"
        missed += state == "missed"
        wrong += state == "wrong_value"
        detail.append({"uuid": a["uuid"], "clause": a["clause"],
                       "expected": a["expected"], "got": v, "state": state})

    n = len(picked)
    print(f"\n  found        {found:3d}  ({found/n:.0%})")
    print(f"  MISSED       {missed:3d}  ({missed/n:.0%})   recorded `unstated`")
    print(f"  wrong value  {wrong:3d}  ({wrong/n:.0%})")
    print("\n--- misses ---")
    for d in detail:
        if d["state"] != "found":
            print(f"  {d['uuid'][:8]}  want {d['expected']:18s} got "
                  f"{str(d['got']):18s}  {d['clause'][:88]}")
    OUT.write_text(json.dumps({"prompt_version": PROMPT_VERSION, "n": n,
                               "found": found, "missed": missed,
                               "wrong_value": wrong, "detail": detail},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
