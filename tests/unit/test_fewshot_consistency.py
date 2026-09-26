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
