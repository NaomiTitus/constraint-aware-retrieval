"""Generate a hand-labelling worksheet for the SEALED golden set.

WHY A SECOND SET. The 44-ad golden set has driven five prompt iterations
(census-v4 -> v8), so it is a regression gate, not a measurement: tuning against
it is the overfitting this project has documented repeatedly. These ads are
selected to be HELD OUT and never tuned against, which is the only way the
project gets an unbiased accuracy number.

WHAT IS SELECTED, and why each stratum exists — every count measured over the
10,166-ad corpus, excluding the 44 already labelled:

  comma_disjunction   120 ads,   0 golden.  census-v7 rule, wholly untested.
  bullet_glyph        419 ads,   0 golden.  The glyph bug demoted 9 ads with
                                            correct evidence; no golden ad
                                            exercises a glyph-led language block.
  truncated           528 ads,   0 golden.  prepare() cuts mid-block on 519 of
                                            them; the truncated_fragment guard
                                            has no golden case.
  doc_clause          254 ads,   1 golden.  census-v7 rule, thin.
  english_worded       56 ads,   1 golden.  census-v7 rule, thin.
  it_vertical         240 ads,   0 golden.  Persona P5 — the originating bug.
  handverkere         787 ads,   1 golden.  Persona P2 Tømrer.
  utdanning         1,471 ads,   2 golden.  Persona P6, the negative control.
  working_lang_gap      -    ,   0 golden.  `both` and `scandinavian` are 2 of 5
                                            enum values never tested; the second
                                            is what the Nordic seeker matrix needs.
  quantified_years    311 ads,   1 golden.  min_years_experience is scored by
                                            nothing at all today.

DATA HANDLING, both non-negotiable:
  * Output goes to data/ (gitignored). Ad text is NEVER committed — DECISIONS D1
    and the NAV terms. The generator is committed so the SELECTION is auditable
    without the text.
  * Every excerpt passes through pii.scrub(). The contact person's name is in the
    body on 11.8% of ads, their email on 10.1%, their phone on 9.6%, and
    contactList exclusion does not cover the body.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from finn_smart_search import pii                                    # noqa: E402
from finn_smart_search.understanding import census_validate as v     # noqa: E402
from finn_smart_search.understanding.census_prompt import TOOL       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "worksheet"
SEED = 20260925
TARGET = {                      # stratum -> how many ads to label
    "comma_disjunction": 3, "bullet_glyph": 3, "truncated": 3,
    "doc_clause": 2, "english_worded": 2, "it_vertical": 3,
    "handverkere": 3, "utdanning": 2, "nordic_or_english": 2,
    "quantified_years": 2, "work_mode": 3,
}

LEVELS = TOOL["input_schema"]["properties"]["norwegian_requirement_level"]["enum"]
WORKLANG = TOOL["input_schema"]["properties"]["stated_working_language"]["enum"]

GLYPH = re.compile(r"^\s*[·●•*\-–⁠​\U0001F4CD✅✨👉⭐🤝🔹✔]")
PATTERNS = {
    "comma_disjunction": re.compile(
        r"\bnorsk\w*\s*,\s*\w+\s+eller\s+\w+|\w+\s*,\s*engelsk\s+eller\s+\w+", re.I),
    "doc_clause": re.compile(
        r"(dokumentasjon|vitnem[åa]l|attest)[^.\n]{0,60}(skandinavisk|engelsk)"
        r"|documentation[^.\n]{0,40}(Scandinavian|English)", re.I),
    "english_worded": re.compile(
        r"fluent in Norwegian|Norwegian or English|Scandinavian or English"
        r"|Norwegian language skills", re.I),
    "nordic_or_english": re.compile(
        r"(skandinavisk|nordisk)\w*[^.\n]{0,30}\b(eller|or)\b[^.\n]{0,30}(engelsk|english)", re.I),
    "quantified_years": re.compile(
        r"\d+\s*[åa]rs?\s+erfaring"
        r"|(en|to|tre|fire|fem|seks|sju|syv|åtte)\s+[åa]rs?\s+erfaring", re.I),
    "work_mode": re.compile(r"hjemmekontor|heimekontor|\bhybrid\w*|fjernarbeid", re.I),
}


def language_context(text: str, limit: int = 14) -> list[str]:
    """Blocks carrying a language token, plus the block before each for context —
    because whether a span starts at a boundary depends on its neighbour."""
    blocks = (text or "").split("\n")
    keep: list[int] = []
    for i, b in enumerate(blocks):
        if v.LANG_TOKEN.search(b):
            if i > 0:
                keep.append(i - 1)
            keep.append(i)
    seen, out = set(), []
    for i in keep:
        if i not in seen and blocks[i].strip():
            seen.add(i)
            out.append(blocks[i])
    return out[:limit]


def main() -> None:
    import duckdb

    db = ROOT / "data" / "ads.duckdb"
    if not db.exists():
        sys.exit("needs data/ads.duckdb — run the ingest first")
    con = duckdb.connect(str(db), read_only=True)

    already = {g["uuid"] for g in json.loads((ROOT / "eval" / "golden_set.json")
                                             .read_text(encoding="utf-8"))}
    rows = con.execute("""
        SELECT a.uuid, a.title, a.description_text, a.n_chars,
               t.nav_category, t.role_family, t.job_title_standardised,
               l.doc_lang, r.ad_content
        FROM ads a
        LEFT JOIN ad_taxonomy t USING (uuid)
        LEFT JOIN ad_language l USING (uuid)
        LEFT JOIN ads_raw     r USING (uuid)
        WHERE a.n_chars > 0
    """).fetchall()

    buckets: dict[str, list] = {k: [] for k in TARGET}
    for row in rows:
        uuid, title, text, n_chars, nav, fam, std, lang, raw = row
        if uuid in already:
            continue
        text = text or ""
        for name, pat in PATTERNS.items():
            if pat.search(text):
                buckets[name].append(row)
        if any(GLYPH.match(b) and v.LANG_TOKEN.search(b) for b in text.split("\n")):
            buckets["bullet_glyph"].append(row)
        if n_chars > 6000 and v.LANG_TOKEN.search(text):
            # The language line must SURVIVE prepare()'s head+tail cut, or the ad
            # is unwinnable by construction: whatever a human labels, the model is
            # never shown the sentence and can only emit `unstated`. One ad (8)
            # was selected this way before the check existed — its single language
            # line sat inside the deleted middle.
            surviving = v.prepare(title or "", text, "no")["sent_text"]
            if any(v.LANG_TOKEN.search(b) for b in surviving.split("\n")):
                buckets["truncated"].append(row)
        if nav == "IT":
            buckets["it_vertical"].append(row)
        if nav == "Håndverkere":
            buckets["handverkere"].append(row)
        if nav == "Utdanning":
            buckets["utdanning"].append(row)

    rng = random.Random(SEED)
    chosen, used = [], set()
    for name, want in TARGET.items():
        pool = [r for r in buckets[name] if r[0] not in used]
        # Ads carrying a language token first — an ad with none cannot exercise
        # the rule its stratum exists for. Then SHUFFLE IN PLACE and take N.
        #
        # The first version did `rng.shuffle(pool[:k])`, which shuffles a COPY and
        # leaves `pool` untouched, so selection was silently uuid-ascending while
        # claiming to be seeded-random. Fixed, and noted because a method section
        # that says "random sample" when it is not is a claim about the evidence.
        pool.sort(key=lambda r: (not v.LANG_TOKEN.search(r[2] or ""), r[0]))
        head = [r for r in pool if v.LANG_TOKEN.search(r[2] or "")]
        tail = [r for r in pool if not v.LANG_TOKEN.search(r[2] or "")]
        rng.shuffle(head)
        rng.shuffle(tail)
        for r in (head + tail)[:want]:
            used.add(r[0])
            chosen.append((name, r))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve work already paid for: a re-run must not silently discard the Opus
    # prefill or the glosses for ads that are still in the set.
    prior = {}
    sk = OUT_DIR / "sealed_skeleton.json"
    if sk.exists():
        for row in json.loads(sk.read_text(encoding="utf-8")):
            if row.get("prefill"):
                prior[row["uuid"]] = row["prefill"]
    md, skeleton = [], []
    md.append("# Sealed golden set — labelling worksheet\n")
    md.append("**These ads are HELD OUT. Do not tune the prompt against them.** "
              "The existing 44-ad set has driven five prompt iterations and is a "
              "regression gate; this set is the unbiased measurement.\n")
    md.append(f"Ads to label: **{len(chosen)}**. Excerpts are PII-scrubbed and this "
              "file lives under `data/` which is gitignored — ad text is never "
              "committed.\n")
    md.append("For each ad, fill the four fields. `evidence_span` must be "
              "**copied character-for-character** from the quoted blocks, and must "
              "be a complete sentence or bullet — the validator rejects anything "
              "else, which is what a wrong label would silently test.\n")
    md.append(f"\n**levels:** {' · '.join(LEVELS)}")
    md.append(f"\n**stated_working_language:** {' · '.join(WORKLANG)}\n")

    for i, (stratum, r) in enumerate(chosen, 1):
        uuid, title, text, n_chars, nav, fam, std, lang, raw = r
        contacts = []
        try:
            d = json.loads(raw) if isinstance(raw, str) else (raw or {})
            c = d.get("ad_content") if isinstance(d.get("ad_content"), dict) else d
            contacts = c.get("contactList") or []
        except (json.JSONDecodeError, TypeError):
            pass
        md.append(f"\n---\n\n## {i}. `{stratum}` — {pii.scrub(title, contacts)}\n")
        md.append(f"- uuid `{uuid}`")
        md.append(f"- occupation: **{nav or '?'}** · role_family `{fam or '?'}` "
                  f"· standardised title: *{std or '?'}*")
        md.append(f"- doc_lang `{lang}` · {n_chars} chars"
                  f"{' · **TRUNCATED at 3500+2500**' if n_chars > 6000 else ''}")
        md.append("\n**Language-bearing blocks** (preceding block shown for "
                  "boundary context):\n")
        for b in language_context(text):
            md.append(f"  - `{pii.scrub(b, contacts)}`")
        md.append("\n```")
        md.append("norwegian_requirement_level: ")
        md.append("evidence_span: ")
        md.append("stated_working_language: ")
        md.append("authorisation_required: ")
        md.append("note: ")
        md.append("```")
        skeleton.append({"n": i, "uuid": uuid, "stratum": stratum,
                         "title": pii.scrub(title, contacts),
                         "nav_category": nav, "doc_lang": lang,
                         "expected": {"norwegian_requirement_level": None,
                                      "evidence_span": None,
                                      "stated_working_language": None,
                                      "authorisation_required": None},
                         "note": None,
                         **({"prefill": prior[uuid]} if uuid in prior else {})})

    (OUT_DIR / "worksheet.md").write_text("\n".join(md), encoding="utf-8")
    (OUT_DIR / "sealed_skeleton.json").write_text(
        json.dumps(skeleton, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_DIR/'worksheet.md'}  ({len(chosen)} ads)")
    print(f"wrote {OUT_DIR/'sealed_skeleton.json'}")
    from collections import Counter
    for k, n in Counter(s for s, _ in chosen).most_common():
        print(f"   {k:20s} {n}")


if __name__ == "__main__":
    main()
