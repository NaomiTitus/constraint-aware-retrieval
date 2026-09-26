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
from finn_smart_search.understanding.census_prompt import TOOL     # noqa: E402

# Derived, not transcribed: a hand-copied enum is the bug this review found.
ENUMS = {k: list(v["enum"])
         for k, v in TOOL["input_schema"]["properties"].items() if "enum" in v}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "worksheet"
# The rows re-extracted under the CURRENT prompt. Reviewing the older run would
# spend verdicts on values that will never ship. `scripts/rerun_exceptions.py`
# writes this and carries `value_v8` so the page can show what changed.
SRC = ROOT / "data" / "worksheet" / "exceptions_current.json"

# What each field is ABOUT, and the wording that would justify its rare value.
FIELD = {
    "conflicting_statements": (
        "The ad contradicts itself about language. NOT only the "
        "English-vs-Norwegian case: a conflict about the NORWEGIAN BAR counts "
        "too — one passage demanding good Norwegian, another asking only for "
        "basic. The schema declares a bare boolean with no definition, so the "
        "test is simply: do two passages about language disagree?",
        r"engelsk|english|norsk|spr[åa]k"),
    "visa_sponsorship": (
        "The recorded value is what to judge, not the topic. "
        "`explicitly_not_offered` means the ad REFUSES to sponsor — not merely "
        "that it requires the applicant to already hold the right to work. "
        "\"We only process applications from candidates with permission to work "
        "in the EU/EEA\" is a requirement ON THE APPLICANT, and is NOT a refusal "
        "to sponsor. The prompt records visa-negative boilerplate as a REJECTED "
        "claim: 0 occurrences measured.",
        r"visa|arbeidstillatelse|oppholdstillatelse|work permit|sponsor"
        r"|EU/E[ØO]S|tillatelse til [åa] jobbe|arbeidsl[øo]yve"),
    "application_language": (
        "The ad says which language the APPLICATION must be written in — not "
        "the working language, and not a requirement on the applicant. "
        "THIS FIELD HAS ONLY THREE LEGAL VALUES: `norwegian_required`, "
        "`english_accepted`, `unstated`. Anything else is borrowed from a "
        "neighbouring field and is discarded downstream, however well the ad "
        "supports the underlying fact.",
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
        # WIDENED AGAIN: row 18 says `Hjelp å skaffe leilighet i Bergen` — no
        # `til`, and `leilighet` was not in the vocabulary at all. The reviewer
        # saw one irrelevant line and would have marked a correct extraction
        # wrong. Same failure class as the `boforhold` miss above.
        r"|hjelp (til )?[åa] (finne|skaffe)|accommodation|housing|zakwaterowanie"
        r"|leilighet|apartment|hybel|bokollektiv|personalbolig|tjenestebolig"
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
        # NO CAPS. The first build took `[:8]` and dropped blocks of <=8 chars,
        # and said so nowhere. Measured consequence: on one row 12 of 20 matched
        # lines were dropped INCLUDING the only one naming the application
        # language, and on another the bare bullet `Bolig` vanished — in both
        # cases the reviewer was shown boilerplate and would have concluded the
        # ad says nothing. A filter may narrow attention; it must never bound
        # the evidence silently.
        lines = [pii.scrub(b, cts) for b in (r["text"] or "").split("\n")
                 if rx.search(b) and b.strip()]
        items.append({
            "n": i, "uuid": r["uuid"], "title": pii.scrub(r["title"] or "", cts),
            "field": r["field"], "value": str(r["value"]), "about": about,
            # Whether the recorded value is even a member of this field's enum.
            # Three of the four `application_language` values were not, and the
            # first build rendered them in the same style as legal ones — so the
            # page argued for "justified" on a value that is discarded anyway.
            "value_v8": (str(r["value_v8"]) if r.get("value_v8") is not None
                         else None),
            "changed": bool(r.get("changed")),
            "legal_values": ENUMS.get(r["field"]),
            "value_is_legal": (r["field"] not in ENUMS
                               or str(r["value"]) in ENUMS[r["field"]]),
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
