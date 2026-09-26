"""Literal English glosses for the worksheet's Norwegian blocks.

WHY THIS NEEDS CARE. The labeller does not read Norwegian, so these glosses are
what the gold labels will actually rest on. In this taxonomy the MODALITY *is*
the label:

    må beherske norsk        -> professional/fluent   (must master)
    bør kunne norsk          -> weaker demand         (should be able to)
    ønskelig med norsk       -> desirable             (desirable)
    norsk er en fordel       -> desirable             (is an advantage)
    ikke et krav             -> explicitly_not_required (not a requirement)
    eller                    -> disjunction, accessible
    og                       -> conjunction, NOT accessible

A fluent, natural translation is exactly the wrong thing: smoothing "bør" into
"must" or "eller" into "and" would silently change the correct label. So the
instruction is literal, modality-preserving, and leaves the Norwegian
connectives unambiguous.

The EVIDENCE SPAN always stays the Norwegian original — the gloss is an aid to
judgement, never the quoted text, because census_validate checks the span
byte-for-byte against the ad.

DISCLOSURE: labels produced with translation assistance are not the same as
labels produced by a Norwegian speaker. That belongs in the README's limitations
beside the kappa caveat, not hidden.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from finn_smart_search.ingest import anthropic_client as ac   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "worksheet"
MODEL = "claude-opus-5"

SYSTEM = """You produce LITERAL English glosses of Norwegian job-advertisement \
lines, for someone deciding what the advertisement demands about language.

Rules:
- Translate literally, clause by clause. Do NOT smooth or paraphrase.
- PRESERVE MODALITY EXACTLY. må = must · skal = shall · bør = should · \
kan = can · ønskelig = desirable · er en fordel = is an advantage · \
ikke et krav = not a requirement · krav = requirement · forutsetning = prerequisite.
- PRESERVE CONNECTIVES EXACTLY. eller = or · og = and. Never swap them; the \
difference decides whether an English speaker may apply.
- Keep language names as they are (norsk = Norwegian, engelsk = English, \
skandinavisk = Scandinavian, nordisk = Nordic, samisk = Sami).
- Keep test and certificate names verbatim (Bergenstesten, norskprøve, B2).
- If a line is already English, return it unchanged.
- Keep it terse. No commentary, no bracketed notes.

Return one gloss per input line, in the same order, via the tool."""

TOOL = {
    "name": ac.TOOL_NAME,
    "description": "Literal English glosses, one per input line, same order.",
    "input_schema": {
        "type": "object",
        "properties": {"glosses": {"type": "array", "items": {"type": "string"}}},
        "required": ["glosses"],
    },
}


def main() -> None:
    import re
    html = (OUT / "worksheet.html").read_text(encoding="utf-8")
    data = json.loads(re.search(r"const DATA = (\{.*?\});\nconst KEY", html, re.S).group(1))

    # Only gloss what is missing. Re-running after a single ad was swapped used
    # to re-submit all 28 and wait on a whole batch for one ad's worth of value.
    have = {}
    gp = OUT / "glosses.json"
    if gp.exists():
        have = {k: v for k, v in json.loads(gp.read_text(encoding="utf-8")).items()
                if v and any(v)}

    reqs = []
    for a in data["ads"]:
        lines = [b["text"] for b in a["blocks"]]
        if not lines:
            continue
        if len(have.get(a["uuid"]) or []) == len(lines):
            continue
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(lines))
        reqs.append({"custom_id": a["uuid"], "params": {
            "model": MODEL, "max_tokens": 2000, "system": SYSTEM,
            "tools": [TOOL], "tool_choice": {"type": "tool", "name": ac.TOOL_NAME},
            "messages": [{"role": "user", "content":
                          f"Gloss these {len(lines)} lines:\n\n{numbered}"}]}})

    if not reqs:
        print("nothing to gloss; all ads already have one per block")
        return
    print(f"translating {sum(len(a['blocks']) for a in data['ads'] if a['uuid'] not in have)} "
          f"blocks across {len(reqs)} ads with {MODEL}")
    c = ac.AnthropicBatchClient()
    bid = c.submit_batch(reqs)
    print(f"batch {bid}", flush=True)
    for _ in range(3000):
        st = c.poll(bid)
        if st == "ended":
            break
        if st != "in_progress":
            sys.exit(f"terminal state {st!r}")
        time.sleep(10)

    got = {}
    for p in c.results(bid):
        if p.get("type") == "succeeded":
            got[p["custom_id"]] = (p["facets"] or {}).get("glosses") or []

    filled = 0
    for a in data["ads"]:
        g = got.get(a["uuid"]) or have.get(a["uuid"]) or []
        for i, b in enumerate(a["blocks"]):
            b["en"] = g[i] if i < len(g) else None
            filled += bool(b.get("en"))
    merged = dict(have)
    for a in data["ads"]:
        g = [b.get("en") for b in a["blocks"]]
        if any(g):
            merged[a["uuid"]] = g
    (OUT / "glosses.json").write_text(
        json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"glossed {filled} blocks -> {OUT/'glosses.json'}")


if __name__ == "__main__":
    main()
