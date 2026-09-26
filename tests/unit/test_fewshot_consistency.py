"""The worked examples must demonstrate what the prose instructs.

WHY THIS FILE EXISTS. census-v9's prose gained a rule — a documentation-language
clause goes in `application_language`, and "skandinavisk eller engelsk" means
`english_accepted`. The FEWSHOT was not updated with it. So example 5c carried
EXACTLY that clause and labelled it `unstated`, and 5b's ad body offered housing
while its record said `relocation_support="unstated"`.

Prose says one thing; the only worked example demonstrates the opposite. On a
smaller model the demonstration usually wins, and neither contradiction was
visible to any test — the FEWSHOT was data nothing asserted about.

Found by an adversarial audit of the v9 fix, not by writing it.

GROUNDING (STANDARDS.md §3.0): the fixtures here are not constructed. They are
the FEWSHOT entries themselves, read from the module, and the cues are measured
corpus wording — the same wording the prompt's own prose quotes.
"""
import re

import pytest

from finn_smart_search.understanding.census_prompt import FEWSHOT, SYSTEM, TOOL

pytestmark = pytest.mark.unit

# A documentation/application-language clause that names English. The prompt's
# prose says these are `english_accepted`; 135 corpus ads carry one.
DOC_CLAUSE_EN = re.compile(
    r"(?=.*\b(dokumentasjon|documentation|vedlegg|vitnem[åa]l|s[øo]knad\w*|CV)\b)"
    r"(?=.*\b(engelsk|english)\b)", re.I)
# An explicit offer of help with housing or moving.
RELOC_OFFER = re.compile(
    r"(hjelp (til )?[åa] (finne|skaffe) (bolig|leilighet|husv[æe]re)"
    r"|vi (tilbyr|dekker) (bolig|flytte|overnatting)"
    r"|help (finding|to find) (housing|accommodation|an apartment))", re.I)


def _cases():
    return [(i, title, body, rec) for i, (title, body, _lang, rec) in enumerate(FEWSHOT)]


@pytest.mark.parametrize("i,title,body,rec", _cases())
def test_a_documentation_clause_naming_english_is_labelled_english_accepted(
        i, title, body, rec):
    """Example 5c is the documentation-clause DEMONSTRATION and said `unstated`."""
    for line in body.split("\n"):
        if DOC_CLAUSE_EN.search(line) and re.search(r"\bm[åa]\b|\bmust\b|\bskal\b|\bvere\b",
                                                    line, re.I):
            assert rec["application_language"] == "english_accepted", (
                f"FEWSHOT[{i}] {title!r} shows {line.strip()!r} "
                f"but records application_language="
                f"{rec['application_language']!r}. The prose says english_accepted.")


@pytest.mark.parametrize("i,title,body,rec", _cases())
def test_an_explicit_housing_offer_is_not_labelled_unstated(i, title, body, rec):
    """Example 5b's body says `Vi tilbyr hjelp til å finne bolig.` and its record
    said `unstated` — the few-shot taught the model to miss a housing offer, on
    the very facet whose exceptions were under review."""
    for line in body.split("\n"):
        if RELOC_OFFER.search(line):
            assert rec["relocation_support"] == "offered", (
                f"FEWSHOT[{i}] {title!r} shows {line.strip()!r} "
                f"but records relocation_support={rec['relocation_support']!r}.")


@pytest.mark.parametrize("i,title,body,rec", _cases())
def test_every_fewshot_record_is_schema_legal(i, title, body, rec):
    """A worked example carrying an illegal value would teach it. Derived from
    TOOL, never hand-listed."""
    from finn_smart_search.understanding import census_validate as v
    for field, allowed in v.ENUMS.items():
        assert field in rec, f"FEWSHOT[{i}] omits required field {field!r}"
        assert rec[field] in allowed, (
            f"FEWSHOT[{i}] {title!r}: {field}={rec[field]!r} is not in its enum")
    for path, (kind, key, allowed) in v.NESTED_ENUMS.items():
        seq = rec.get(path.split(".", 1)[0])
        if not isinstance(seq, list):
            continue
        for item in seq:
            val = item if kind == "items" else item.get(key)
            if val is not None:
                assert val in allowed, f"FEWSHOT[{i}]: {path}={val!r} is not in its enum"


def test_the_prose_rule_and_the_examples_name_the_same_legal_values():
    """The SYSTEM prose spells out `application_language`'s legal values inline.
    If the enum changes and the prose does not, the model is told the wrong set —
    which is how the field ended up holding three values from other enums."""
    declared = set(TOOL["input_schema"]["properties"]["application_language"]["enum"])
    for val in declared:
        assert f"`{val}`" in SYSTEM, (
            f"{val!r} is legal in the schema but never named in the prompt prose")


# ===========================================================================
# THE GENERAL FORM. Three times now the prose has been updated and the worked
# examples left behind, and each time the demonstration won:
#
#   census-v9  -> 5c showed a documentation clause and said `unstated`
#   census-v9  -> 5b showed a housing offer and said `unstated`
#   census-v12 -> examples 8 and 9 showed "Gyldig politiattest må leveres" and
#                 said security_clearance_required=False. They were the ONLY
#                 two politiattest ads in the few-shot, against prose saying
#                 "TRUE is the expected answer, 35.8% of ads".
#                 Measured: recall 60%, 12 of 30 politiattest ads missed.
#
# The one-off tests above pin the three. This pins the SHAPE: for any facet
# with an unambiguous trigger in the ad text, no worked example may contradict
# it. A new facet added to this table is checked against all 14 examples at
# once, which is the only version of this test that keeps working.
TRIGGERS = [
    ("security_clearance_required",
     r"politiattest|vandelsattest|plettfri vandel|bakgrunnssjekk|sikkerhetsklarer",
     True),
    ("relocation_support",
     r"hjelp til [åa] finne bolig|behjelpelig med (å skaffe )?bolig|flytteutgifter",
     "offered"),
    ("application_language",
     r"dokument\w*[^.\n]{0,60}m[åa][^.\n]{0,60}(engelsk|english)",
     "english_accepted"),
]


@pytest.mark.parametrize("field,trigger,expected", TRIGGERS)
def test_no_worked_example_contradicts_the_prose_for_its_own_trigger(
        field, trigger, expected):
    bad = []
    for i, (title, body, _lang, rec) in enumerate(FEWSHOT):
        if re.search(trigger, body, re.I) and rec.get(field) != expected:
            bad.append(f"FEWSHOT[{i}] {title!r}: ad triggers {field} but the "
                       f"record says {rec.get(field)!r}, prose says {expected!r}")
    assert not bad, "\n".join(bad)


def test_at_least_one_worked_example_demonstrates_each_non_default_value():
    """A facet whose every example carries the default has been demonstrated
    only in its silent form, whatever the prose claims. `security_clearance_
    required` was False on all 14."""
    for field, _trigger, expected in TRIGGERS:
        shown = {rec.get(field) for _t, _b, _l, rec in FEWSHOT}
        assert expected in shown, (
            f"no worked example ever shows {field}={expected!r}; "
            f"the few-shot only demonstrates {sorted(map(str, shown))}")


def test_a_nordic_only_disjunction_is_demonstrated_not_only_described():
    """THE PATTERN, stated once: prose without a demonstration does not hold.

    Three consecutive versions failed the same way — the rule was written into
    the prose and the worked examples were left showing the opposite, and each
    time the demonstration won:

        v9   documentation clause  prose english_accepted / example unstated
        v12  politiattest          prose "TRUE expected"   / 2 examples False
        v14  norsk eller skandinavisk   prose scandinavian_accepted, and NO
                                        example of the shape at all

    The third is the subtlest: nothing contradicted the rule, there was simply
    no instance of it. `eller` had only ever been demonstrated landing on
    English, so the model generalised from the examples it had. A rule with no
    instance is not taught.

    MEASURED: 344 ads (3.4%) use a Nordic-only disjunction, and v13/v14 answered
    `either_norwegian_or_english` on them — asserting English is accepted where
    it is not, the `shown wrongly` direction.
    """
    found = []
    for i, (title, body, _lang, rec) in enumerate(FEWSHOT):
        for line in body.split("\n"):
            if (re.search(r"\bnorsk\w*\b[^.\n]{0,40}\beller\b[^.\n]{0,40}"
                          r"(skandinavisk|nordisk|svensk|dansk)", line, re.I)
                    and not re.search(r"engelsk|english", line, re.I)):
                found.append((i, rec["norwegian_requirement_level"]))
    assert found, ("no worked example shows `norsk eller <Nordic>` with no "
                   "English; 344 corpus ads do, and prose alone did not hold")
    for i, level in found:
        assert level == "scandinavian_accepted", (
            f"FEWSHOT[{i}] shows a Nordic-only disjunction but is labelled "
            f"{level!r} — the demonstration would teach the error")


def test_no_worked_example_quotes_a_line_unique_to_an_evaluation_ad():
    """Leakage is showing the model a TEST ITEM, not a common phrase.

    Three few-shot examples shared their decisive language line with a
    golden-set ad of the same title. Measured, only one was leakage:

        "Behersker norsk eller engelsk"                    47 corpus ads
        "Beherske et nordisk språk flytende, både ..."     11 corpus ads
        "God kunnskap i norsk, munnleg og skriftleg"        1 corpus ad
                                                            — and it IS the
                                                              golden ad

    A phrase in 47 advertisements is boilerplate and demonstrating it is what
    a few-shot is for. A phrase in exactly one, which is an evaluation item,
    hands the model the answer to that item. The line is the frequency, not
    the coincidence of titles — which is why the first version of this test,
    matching on title, reported three leaks where there was one.
    """
    import json
    import re as _re
    from pathlib import Path
    import duckdb
    root = Path(__file__).resolve().parents[2]
    db, gp = root / "data" / "ads.duckdb", root / "eval" / "golden_set.json"
    if not (db.exists() and gp.exists()):
        pytest.skip("needs the corpus and the golden set")
    golden = {g["uuid"] for g in json.loads(gp.read_text(encoding="utf-8"))}
    con = duckdb.connect(str(db), read_only=True)
    ads = con.execute("SELECT uuid, description_text FROM ads").fetchall()
    con.close()

    def norm(x):
        return _re.sub(r"\s+", " ", x or "").strip().lower()

    bad = []
    for i, (_t, _b, _l, rec) in enumerate(FEWSHOT):
        for sp in rec.get("evidence_spans") or []:
            n = norm(sp["span"])
            if len(n) < 20:
                continue
            carriers = [u for u, t in ads if n in norm(t)]
            if len(carriers) == 1 and carriers[0] in golden:
                bad.append(f"FEWSHOT[{i}] quotes {sp['span']!r}, which occurs "
                           f"in exactly one ad and that ad is golden")
    assert not bad, "\n".join(bad)


def test_every_fewshot_span_is_verbatim_corpus_text():
    """The examples are excerpted from REAL advertisements. While closing a
    leak I shortened one to fit a source line — "Norsk er ikke et krav siden vi
    har ansatte fra flere nasjoner" against the ad's "Norsk er ikke et krav som
    språk siden vi har flere ansatte fra flere nasjoner" — and no test noticed.

    A paraphrased span tunes the prompt to language no advertisement uses, and
    it is indistinguishable from an invented one. STANDARDS.md §3.0 applies to
    the prompt's fixtures exactly as it applies to a test's.
    """
    import json
    import re as _re
    from pathlib import Path
    import duckdb
    root = Path(__file__).resolve().parents[2]
    db = root / "data" / "ads.duckdb"
    if not db.exists():
        pytest.skip("needs the corpus")
    con = duckdb.connect(str(db), read_only=True)
    ads = [t for (t,) in con.execute("SELECT description_text FROM ads").fetchall()]
    con.close()
    blob = "\n".join(_re.sub(r"\s+", " ", t or "") for t in ads).lower()
    missing = []
    for i, (_t, _b, _l, rec) in enumerate(FEWSHOT):
        for sp in rec.get("evidence_spans") or []:
            n = _re.sub(r"\s+", " ", sp["span"]).strip().lower()
            if len(n) >= 15 and n not in blob:
                missing.append(f"FEWSHOT[{i}] span is not verbatim corpus text: "
                               f"{sp['span']!r}")
    assert not missing, "\n".join(missing)
