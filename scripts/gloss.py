"""Literal English glosses for any review's Norwegian lines.

Reusable across reviews (see reviews/README.md): give it a list of
{"id": str, "lines": [str]} and it returns {id: [gloss]} aligned by index.

WHY LITERAL AND NOT FLUENT. The reviewer does not read Norwegian, so these
glosses are what the judgement rests on, and in this domain the MODALITY IS THE
MEANING:

    må = must · skal = shall · bør = should · kan = can
    ønskelig / er en fordel = desirable, NOT required
    ikke et krav = not a requirement
    eller = OR   og = AND      <- swapping these reverses the answer

A fluent translation is the wrong tool: smoothing `bør` into "must" or `eller`
into "and" changes the fact being judged. Terms of art stay verbatim
(Bergenstesten, norskprøve, politiattest, sikkerhetsklarering, fagbrev), because
the whole point of several reviews is whether the ad says that exact thing.

Incremental: rows already present in the cache file are not re-sent.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from finn_smart_search.ingest import anthropic_client as ac   # noqa: E402

MODEL = "claude-opus-5"

SYSTEM = """You produce LITERAL English glosses of lines from Norwegian job \
advertisements, for a reader who does not know Norwegian and must judge exactly \
what the advertisement says.

Rules:
- Translate literally, clause by clause. Do NOT smooth, shorten or paraphrase.
- PRESERVE MODALITY EXACTLY: må = must · skal = shall · bør = should · \
kan = can · ønskelig = desirable · er en fordel = is an advantage · \
ikke et krav = not a requirement · krav = requirement · kreves = is required.
- PRESERVE CONNECTIVES EXACTLY: eller = or · og = and. Never swap them.
- Keep terms of art verbatim, untranslated: Bergenstesten, norskprøve, \
politiattest, sikkerhetsklarering, fagbrev, autorisasjon, HPR, \
lønnstrinn, vikariat, åremål, stillingsprosent.
- Keep language names as names (norsk = Norwegian, samisk = Sami, \
skandinavisk = Scandinavian, nordisk = Nordic).
- If a line is already English, return it unchanged.
- Terse. No commentary, no bracketed additions.

Return one gloss per input line, same order, via the tool."""

TOOL = {
    "name": ac.TOOL_NAME,
    "description": "Literal English glosses, one per input line, same order.",
    "input_schema": {"type": "object",
                     "properties": {"glosses": {"type": "array",
                                                "items": {"type": "string"}}},
                     "required": ["glosses"]},
}


def gloss(groups: list[dict], cache_path: Path, log=print) -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    if cache_path.exists():
        cache = {k: v for k, v in
                 json.loads(cache_path.read_text(encoding="utf-8")).items() if v}

    todo = [g for g in groups
            if g["lines"] and len(cache.get(g["id"]) or []) != len(g["lines"])]
    if not todo:
        log("nothing to gloss")
        return cache

    # SIZED TO THE INPUT. A flat 2,200 truncated 10 of 46 items the first time
    # the full ad was glossed (58-90 lines each): the response hit max_tokens and
    # was correctly discarded rather than salvaged, so those items came back with
    # no glosses at all. Budget per line plus headroom for the JSON envelope.
    def budget(n_lines: int) -> int:
        return min(16000, 1200 + 120 * n_lines)

    reqs = [{"custom_id": g["id"], "params": {
        "model": MODEL, "max_tokens": budget(len(g["lines"])), "system": SYSTEM,
        "tools": [TOOL], "tool_choice": {"type": "tool", "name": ac.TOOL_NAME},
        "messages": [{"role": "user", "content":
                      "Gloss these %d lines:\n\n%s" % (
                          len(g["lines"]),
                          "\n".join(f"{i+1}. {t}" for i, t in enumerate(g["lines"])))}]}}
            for g in todo]

    log(f"glossing {sum(len(g['lines']) for g in todo)} lines "
        f"across {len(reqs)} items with {MODEL}")
    c = ac.AnthropicBatchClient()
    bid = c.submit_batch(reqs)
    log(f"batch {bid}")
    for _ in range(3000):
        st = c.poll(bid)
        if st == "ended":
            break
        if st != "in_progress":
            raise SystemExit(f"terminal state {st!r}")
        time.sleep(10)

    for p in c.results(bid):
        if p.get("type") == "succeeded":
            cache[p["custom_id"]] = (p["facets"] or {}).get("glosses") or []
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False),
                          encoding="utf-8")
    log(f"glossed {sum(1 for v in cache.values() if v)} items -> {cache_path}")
    return cache


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    items = json.loads((ROOT / "data" / "worksheet" / "exceptions_items.json")
                       .read_text(encoding="utf-8"))
    # Gloss the FULL ad too: the toggle shows it, and a reviewer who cannot read
    # Norwegian gains nothing from untranslated text behind a button.
    # ORDER IS A CONTRACT with build_exceptions_artifact.py, which slices this
    # list back apart positionally. Both sides derive the offsets from the item's
    # own arrays and the artifact asserts the total length, so a change here
    # fails loudly instead of sliding every gloss onto the wrong line — which
    # would put confident English under the wrong Norwegian sentence, the worst
    # possible failure for a reviewer who cannot read the original.
    groups = [{"id": str(i["n"]),
               "lines": [i["title"]] + i["lines"] + i["spans"] + i["all_lines"]}
              for i in items]
    gloss(groups, ROOT / "data" / "worksheet" / "exceptions_glosses.json")
