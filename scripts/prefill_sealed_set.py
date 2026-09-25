"""Opus pre-fill for the SEALED golden set — a second opinion, never the gold.

WHAT THIS IS AND IS NOT.

  IS      a labour saver and a disagreement flag. Opus proposes; the human
          confirms or overrides; the human's label is the only one that counts.
  IS NOT  the gold label. Scoring Haiku against Opus measures MODEL AGREEMENT,
          not accuracy, and the two share failure modes — the census-v4
          disjunction bug ("skandinavisk eller engelsk" read as Scandinavian-only)
          is exactly the plausible-misreading kind a second LLM is most likely to
          reproduce. An agreement score would look clean while both were wrong.
          This project has already measured the shape of that risk: across the
          held-out probes, 7 of 9 apparent "model errors" were the ORACLE being
          wrong, and an LLM judge is another oracle — a less inspectable one than
          a regex, where at least the dokument/documentation bug could be read.

THE DESIGN CHOICE THAT MATTERS: Opus gets the LABEL SPACE, not the census prompt.

  It receives plain definitions of the nine levels and the span contract, and
  NOTHING ELSE — no precedence rules, no disjunction convention, no few-shots.
  Handing it census-v8 would make this "does Opus agree with the prompt", and the
  prompt's conventions are precisely what an independent opinion should be able to
  disagree with. A disagreement on a disjunction is then informative: it says the
  convention is not self-evident from the taxonomy alone.

SAFEGUARDS
  * Run BEFORE any Haiku output for these ads exists, and it never reads
    ad_facets. Sequencing, not politeness: a pre-fill that has seen the thing it
    is checking is not a second opinion.
  * Every value is written with `confirmed: false`. Ingestion REFUSES unconfirmed
    rows, so a field skimmed past cannot become gold by inattention.
  * Excerpts are PII-scrubbed and output stays under data/ (gitignored).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from finn_smart_search import pii                                 # noqa: E402
from finn_smart_search.ingest import anthropic_client as ac       # noqa: E402
from finn_smart_search.understanding.census_prompt import TOOL    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "worksheet"
# Opus 5. NOT hardcoded from memory a second time: the first attempt used
# "claude-opus-4-1-20250805", which the API rejected with
# not_found_error on all 28 requests. The plan says not to hardcode model IDs
# from memory and I did exactly that.
JUDGE_MODEL = "claude-opus-5"

LEVELS = TOOL["input_schema"]["properties"]["norwegian_requirement_level"]["enum"]
WORKLANG = TOOL["input_schema"]["properties"]["stated_working_language"]["enum"]

# Definitions only. No precedence rules, no examples — see the module docstring.
JUDGE_SYSTEM = f"""You are labelling Norwegian job advertisements for a research \
dataset. Decide what the advertisement STATES about language requirements.

Choose exactly one `norwegian_requirement_level` from:
  unstated                     the ad says nothing about language at all
  explicitly_not_required      it says Norwegian is not needed
  desirable                    Norwegian is an advantage, not a requirement
  either_norwegian_or_english  either language is accepted
  scandinavian_accepted        a Scandinavian language is accepted
  conversational               must make oneself understood
  professional                 good working Norwegian demanded
  fluent                       fluency demanded
  certified                    a named test or CEFR level is demanded

Also record `stated_working_language` (one of: {', '.join(WORKLANG)}) and
`authorisation_required` (the professional licence named, or null — this is a
licence to practise, not a language requirement).

For `evidence_span`, quote the sentence that justifies your level, copied
character for character from the advertisement, as a COMPLETE sentence or bullet.
If you cannot quote one exactly, return null. An honest null is better than an
approximate quote.

Answer only with the tool call."""

# Named for the shared parser, not for this task: anthropic_client.parse_result
# searches for TOOL_NAME, and reusing it keeps the tested max_tokens / preamble
# handling rather than writing a second, untested parser.
JUDGE_TOOL = {
    "name": ac.TOOL_NAME,
    "description": "Record the language-requirement label for one advertisement.",
    "input_schema": {
        "type": "object",
        "properties": {
            "norwegian_requirement_level": {"type": "string", "enum": LEVELS},
            "evidence_span": {"type": ["string", "null"]},
            "stated_working_language": {"type": "string", "enum": WORKLANG},
            "authorisation_required": {"type": ["string", "null"]},
            "reasoning": {"type": "string",
                          "description": "One sentence. What in the ad decided it."},
        },
        "required": ["norwegian_requirement_level", "evidence_span",
                     "stated_working_language", "authorisation_required", "reasoning"],
    },
}


def main() -> None:
    import duckdb

    skeleton = json.loads((OUT / "sealed_skeleton.json").read_text(encoding="utf-8"))
    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)

    requests = []
    for row in skeleton:
        t, body, raw = con.execute(
            "SELECT a.title, a.description_text, r.ad_content FROM ads a "
            "LEFT JOIN ads_raw r USING (uuid) WHERE a.uuid = ?", [row["uuid"]]).fetchone()
        contacts = []
        try:
            d = json.loads(raw) if isinstance(raw, str) else (raw or {})
            c = d.get("ad_content") if isinstance(d.get("ad_content"), dict) else d
            contacts = c.get("contactList") or []
        except (json.JSONDecodeError, TypeError):
            pass
        text = pii.scrub(body or "", contacts)[:9000]
        requests.append({
            "custom_id": row["uuid"],
            "params": {
                # NO `temperature`: Opus 5 rejects it outright with
                # invalid_request_error "`temperature` is deprecated for this
                # model". Haiku 4.5 (the census model) still accepts it, so this
                # is model-specific and will bite if the census model ever moves.
                "model": JUDGE_MODEL, "max_tokens": 700,
                "system": JUDGE_SYSTEM,
                "tools": [JUDGE_TOOL],
                "tool_choice": {"type": "tool", "name": ac.TOOL_NAME},
                "messages": [{"role": "user", "content":
                              f"Title: {pii.scrub(t or '', contacts)}\n\n{text}"}],
            },
        })

    print(f"pre-filling {len(requests)} sealed ads with {JUDGE_MODEL}")
    client = ac.AnthropicBatchClient()
    bid = client.submit_batch(requests)
    print(f"batch {bid}")
    for _ in range(3000):
        st = client.poll(bid)
        if st in ("ended",):
            break
        if st not in ("in_progress",):
            sys.exit(f"batch terminal state {st!r}")
        time.sleep(10)

    # client.results() ALREADY returns parsed dicts (it maps parse_result over
    # raw_results). Calling parse_result again re-parses a parsed record: the
    # second pass looks for a nested "result" key, finds none, and reports
    # error="None" — which is how the real cause (a rejected model id) was thrown
    # away and had to be recovered by querying the batch directly.
    got = {p["custom_id"]: p for p in client.results(bid)}

    filled = 0
    for row in skeleton:
        p = got.get(row["uuid"])
        if not p or p.get("type") != "succeeded":
            row["prefill"] = {"error": (p or {}).get("error", "missing")}
            continue
        f = p["facets"]
        row["prefill"] = {
            "norwegian_requirement_level": f.get("norwegian_requirement_level"),
            "evidence_span": f.get("evidence_span"),
            "stated_working_language": f.get("stated_working_language"),
            "authorisation_required": f.get("authorisation_required"),
            "reasoning": f.get("reasoning"),
            "model": JUDGE_MODEL,
            "confirmed": False,          # ingestion refuses rows with this false
        }
        filled += 1
    (OUT / "sealed_skeleton.json").write_text(
        json.dumps(skeleton, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"pre-filled {filled}/{len(skeleton)}; all marked confirmed=false")


if __name__ == "__main__":
    main()
