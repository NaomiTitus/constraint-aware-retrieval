"""Render the sealed-set worksheet as a self-contained labelling tool.

A working instrument, not a document: 28 ads to judge in about an hour, so the
craft goes into information design rather than decoration.

THE ONE DESIGN DECISION THAT MATTERS. `evidence_span` must be byte-identical to
the ad or census_validate rejects it — that is the single likeliest hand-labelling
error, and a wrong span silently tests nothing. So blocks are CLICKABLE and
clicking one sets the span verbatim. Typing is possible but never necessary.

Output goes to data/ (gitignored). Ad text is never committed.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from finn_smart_search import pii                                  # noqa: E402
from finn_smart_search.understanding import census_validate as v   # noqa: E402
from finn_smart_search.understanding.census_prompt import TOOL     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "worksheet"
LEVELS = TOOL["input_schema"]["properties"]["norwegian_requirement_level"]["enum"]
WORKLANG = TOOL["input_schema"]["properties"]["stated_working_language"]["enum"]

WHY = {
    "comma_disjunction": "104 corpus ads · 0 golden. The census-v7 rule's PRESENCE is tested; its BEHAVIOUR is not. Note the selecting pattern is looser than the prompt's, so some ads here match a mother-tongue clause (“annet morsmål enn norsk, svensk eller dansk”) rather than a list of accepted working languages — judge what the ad demands, not why it was selected.",
    "bullet_glyph": "419 corpus ads · 0 golden. The glyph bug demoted 9 ads that had correct evidence.",
    "truncated": "528 ads (5.2%) · 0 golden. prepare() cuts mid-block on 519 of them.",
    "doc_clause": "254 corpus ads · 1 golden. census-v7 rule, thin coverage.",
    "english_worded": "56 corpus ads · 1 golden. The requirement stated in English.",
    "it_vertical": "240 corpus ads · 0 golden. Persona P5 — the originating bug.",
    "handverkere": "787 corpus ads · 1 golden. Persona P2 Tømrer.",
    "utdanning": "1,471 corpus ads · 2 golden. Persona P6, the designed negative control.",
    "nordic_or_english": "251 corpus ads · 2 golden. English present without the word 'norsk'.",
    "quantified_years": "311 corpus ads · 1 golden. min_years_experience is scored by nothing today.",
    "work_mode": "362 corpus ads · 0 golden. Needed before the facet ships.",
}
LEVEL_HINT = {
    "unstated": "says nothing about language",
    "explicitly_not_required": "says Norwegian is NOT needed",
    "desirable": "Norwegian an advantage, not required",
    "either_norwegian_or_english": "either language accepted",
    "scandinavian_accepted": "a Scandinavian language accepted, English NOT",
    "conversational": "must make oneself understood",
    "professional": "good working Norwegian demanded",
    "fluent": "fluency demanded",
    "certified": "a named test or CEFR level demanded",
}


def main() -> None:
    import duckdb

    skeleton = json.loads((OUT / "sealed_skeleton.json").read_text(encoding="utf-8"))
    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    ads = []
    for row in skeleton:
        r = con.execute("""SELECT a.title, a.description_text, a.n_chars, r.ad_content,
                                  t.nav_category, t.role_family, t.job_title_standardised
                           FROM ads a LEFT JOIN ads_raw r USING (uuid)
                           LEFT JOIN ad_taxonomy t USING (uuid) WHERE a.uuid = ?""",
                        [row["uuid"]]).fetchone()
        title, body, n_chars, raw, nav, fam, std = r
        contacts = []
        try:
            d = json.loads(raw) if isinstance(raw, str) else (raw or {})
            c = d.get("ad_content") if isinstance(d.get("ad_content"), dict) else d
            contacts = c.get("contactList") or []
        except (json.JSONDecodeError, TypeError):
            pass
        blocks = (body or "").split("\n")
        keep, seen = [], set()
        for i, b in enumerate(blocks):
            if v.LANG_TOKEN.search(b):
                for j in (i - 1, i):
                    if j >= 0 and j not in seen and blocks[j].strip():
                        seen.add(j)
                        keep.append({"text": pii.scrub(blocks[j], contacts),
                                     "hit": j == i})
        ads.append({
            "n": row["n"], "stratum": row["stratum"], "uuid": row["uuid"],
            "title": pii.scrub(title or "", contacts),
            "nav": nav, "family": fam, "std": std,
            "doc_lang": row["doc_lang"], "n_chars": n_chars,
            "truncated": bool(n_chars and n_chars > 6000),
            "blocks": keep[:16],
            "prefill": row.get("prefill") or None,
        })

    data = {"ads": ads, "levels": LEVELS, "worklang": WORKLANG,
            "why": WHY, "hint": LEVEL_HINT}
    # English glosses, if scripts/translate_worksheet.py has run. Literal and
    # modality-preserving by construction: the labeller does not read Norwegian,
    # so må/bør/er en fordel and eller/og must survive translation intact — those
    # distinctions ARE the label. The span stays the Norwegian original.
    gl_path = OUT / "glosses.json"
    if gl_path.exists():
        gl = json.loads(gl_path.read_text(encoding="utf-8"))
        for a in ads:
            for i, b in enumerate(a["blocks"]):
                g = gl.get(a["uuid"]) or []
                b["en"] = g[i] if i < len(g) else None

    tpl = (ROOT / "scripts" / "worksheet_template.html").read_text(encoding="utf-8")
    out = tpl.replace("/*__DATA__*/null",
                      json.dumps(data, ensure_ascii=False))
    (OUT / "worksheet.html").write_text(out, encoding="utf-8")
    print(f"wrote {OUT/'worksheet.html'}  ({len(ads)} ads, "
          f"{sum(len(a['blocks']) for a in ads)} blocks)")


if __name__ == "__main__":
    main()
