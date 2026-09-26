"""Review 002 — the rare values of near-constant facets.

WHY THESE. Six facets take one value on 90-99% of records. A field that is
`False` 99% of the time is either capturing something genuinely rare or not
working, and the modal value cannot tell you which — only the exceptions can.
If the rare value is wrong, MOST OF THAT CLASS is wrong, and nothing downstream
would show it.

  conflicting_statements       True on   1 of 169
  visa_sponsorship             set  on   1
  application_language         set  on   4
  evidence_strength            hedged/inferred on 6
  security_clearance_required  True on  13
  relocation_support           offered on 21

The question per row is narrower than the language review's: not "what is the
right label" but "is this rare value justified by the ad". So the form is a
three-way verdict plus a note, and the evidence shown is the sentences that
could bear on that specific field — not the language lines.

Same guarantees as review 001 (see reviews/README.md): PII-scrubbed excerpts,
literal glosses, labels synced to the artifact, and nothing reimplemented that
production already decides.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from finn_smart_search import pii                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "worksheet"
SRC = Path("/private/tmp/claude-501/-Users-naomi-Documents-projects/"
           "2d8ed1e8-68bf-4a23-bed2-3527063a5b59/scratchpad/exceptions.json")

# What each field is ABOUT, and the wording that would justify its rare value.
FIELD = {
    "conflicting_statements": (
        "The ad contradicts itself about language — one passage says English is "
        "the working language, another demands fluent Norwegian.",
        r"engelsk|english|norsk|spr[åa]k"),
    "visa_sponsorship": (
        "The ad says something explicit about work permits or visa sponsorship.",
        r"visa|arbeidstillatelse|oppholdstillatelse|work permit|sponsor"
        r"|EU/E[ØO]S|tillatelse til [åa] jobbe|arbeidsl[øo]yve"),
    "application_language": (
        "The ad says which language the APPLICATION must be written in — not "
        "the working language, and not a requirement on the applicant.",
        r"s[øo]knad|application|skriv|dokumentasjon|vitnem[åa]l|CV|attest"
        r"|p[åa] (norsk|engelsk|skandinavisk)"),
    "security_clearance_required": (
        "The ad requires a security clearance — sikkerhetsklarering, "
        "autorisasjon for skjermingsverdig informasjon. NOT a police certificate "
        "(politiattest), which most care and education roles require.",
        r"sikkerhetsklarer|klarering|skjermingsverdig|politiattest|vandel"
        r"|bakgrunnssjekk|security clearance|autorisasjon"),
    "relocation_support": (
        "The ad offers help relocating — housing, moving costs, travel or "
        "accommodation cover. NOT merely mentioning that the job is in a "
        "particular place.",
        # WIDENED after the first build missed `boforhold` (housing conditions)
        # on two ads and `overnatting` on a third, and then told the reviewer
        # "no sentence matches this facet" — asserting the regex's blind spot as
        # evidence and arguing for the wrong verdict.
        r"boforhold|bolig|bosted|overnatting|innkvarter|husv[æe]r|flytte|reloc"
        r"|hjelp til [åa] (finne|skaffe)|accommodation|housing|zakwaterowanie"
        r"|pendl|reise til jobb|dekning av reise|dekker reise|per diem"
        r"|rotasjon|\d+ uker p[åa]"),
    "evidence_strength": (
        "How solid the language evidence is. `explicit_but_hedged` = stated but "
        "qualified; `inferred_from_context` = not stated at all, which should be "
        "rare and is worth challenging.",
        r"norsk|engelsk|spr[åa]k|flytende|beherske"),
}


def main() -> None:
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    raw = json.loads(SRC.read_text(encoding="utf-8"))

    items = []
    for i, r in enumerate(sorted(raw, key=lambda x: (x["field"], x["uuid"])), 1):
        about, pat = FIELD[r["field"]]
        cts = []
        row = con.execute("SELECT ad_content FROM ads_raw WHERE uuid=?", [r["uuid"]]).fetchone()
        try:
            d = json.loads(row[0]) if row and isinstance(row[0], str) else {}
            c = d.get("ad_content") if isinstance(d.get("ad_content"), dict) else d
            cts = c.get("contactList") or []
        except (json.JSONDecodeError, TypeError):
            pass
        rx = re.compile(pat, re.I)
        lines = [pii.scrub(b, cts) for b in (r["text"] or "").split("\n")
                 if rx.search(b) and len(b.strip()) > 8][:8]
        items.append({
            "n": i, "uuid": r["uuid"], "title": pii.scrub(r["title"] or "", cts),
            "field": r["field"], "value": str(r["value"]), "about": about,
            "level": r["level"],
            "spans": [pii.scrub(s, cts) for s in r["spans"]],
            "lines": lines,
            # The FULL ad, always available behind a toggle. A filter should
            # narrow attention, never bound the evidence: the first build showed
            # only matched lines and claimed their absence was "the finding",
            # which was wrong on 6 of 10 such rows.
            "all_lines": [pii.scrub(b, cts) for b in (r["text"] or "").split("\n")
                          if b.strip()],
        })
    (OUT / "exceptions_items.json").write_text(
        json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print(f"{len(items)} exception rows")
    for k, n in Counter(i["field"] for i in items).most_common():
        print(f"   {k:30s} {n}")
    noev = sum(1 for i in items if not i["lines"])
    print(f"\n   rows where no sentence matched the SEARCH TERMS: {noev}")
    print(f"   (was 10 before the vocabularies were widened; 6 of those 10 were the "
          f"pattern failing, not the model)")
    print(f"   full ad carried on every row: "
          f"{sum(1 for i in items if i.get('all_lines'))} of {len(items)}")


if __name__ == "__main__":
    main()
