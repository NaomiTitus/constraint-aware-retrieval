"""Every enum field must be validated, and the coverage is derived from the schema.

FOUND BY HUMAN REVIEW, not by a test. Of the four `application_language`
exceptions in the corpus, THREE carried a value borrowed from another field's
enum:

    application_language = "scandinavian_accepted"   legal in norwegian_requirement_level
    application_language = "either_norwegian_or_english"  same
    application_language = "both"                    legal in stated_working_language

`application_language` permits exactly {norwegian_required, english_accepted,
unstated}. The model recognises a documentation-language clause, has nowhere
clean to record it, and reaches for a neighbouring field's vocabulary.

WHY NOTHING CAUGHT IT. The tool schema declares the enums, but tool-use does not
enforce them server-side. `census_validate` checked no enums at all, and
`scoring` checked `norwegian_requirement_level` alone. So an invalid value flowed
into the data and out again unchallenged.

THE TESTS ARE DERIVED FROM `TOOL`, deliberately. A hand-written list of fields
would omit the next one added — the same failure as every hardcoded pattern in
this repo. Parametrising over the schema means a new enum field is covered the
moment it exists, or the coverage test fails.
"""
import pytest

from finn_smart_search.understanding import census_validate as v
from finn_smart_search.understanding.census_prompt import TOOL

pytestmark = pytest.mark.unit

PROPS = TOOL["input_schema"]["properties"]
ENUM_FIELDS = {k: set(p["enum"]) for k, p in PROPS.items() if "enum" in p}
AD = ("Krav til stillingen\n"
      "Gode norskkunnskaper er et krav for stillingen\n"
      "Oppstart snarest")
SPAN = "Gode norskkunnskaper er et krav for stillingen"


def base():
    """A record that passes validation, so any failure is the field under test."""
    return {
        "norwegian_requirement_level": "professional",
        "stated_working_language": "unstated",
        "application_language": "unstated",
        "evidence_basis": "explicit_statement",
        "evidence_strength": "explicit_and_unambiguous",
        "evidence_spans": [{"span": SPAN, "section_language": "no"}],
        "authorisation_required": None, "security_clearance_required": False,
        "visa_sponsorship": "unstated", "relocation_support": "unstated",
        "seniority": "unstated", "min_years_experience": None,
        "skills": [], "implicit_evidence": [], "conflicting_statements": False,
    }


def test_the_schema_has_the_enum_fields_this_file_expects():
    """Guards the derivation itself: if TOOL stops declaring enums, every
    parametrised test below would silently pass with an empty parameter list."""
    assert len(ENUM_FIELDS) >= 8, sorted(ENUM_FIELDS)
    assert "application_language" in ENUM_FIELDS


@pytest.mark.parametrize("field", sorted(ENUM_FIELDS))
def test_a_value_outside_the_enum_is_rejected(field):
    """Every enum field, derived from the schema — not a list someone maintains."""
    f = base()
    f[field] = "definitely_not_a_valid_value"
    out = v.validate(f, AD)
    assert out["demoted"], f"{field} accepted an out-of-enum value"
    assert any("enum" in r for r in out["reasons"]), out["reasons"]


@pytest.mark.parametrize("field,borrowed", [
    ("application_language", "scandinavian_accepted"),     # from the level enum
    ("application_language", "either_norwegian_or_english"),
    ("application_language", "both"),                      # from working_language
    ("stated_working_language", "certified"),              # from the level enum
    ("seniority", "professional"),                         # plausible, wrong field
])
def test_a_value_borrowed_from_ANOTHER_fields_enum_is_rejected(field, borrowed):
    """The real failure mode. These are not typos — each is a legal value
    somewhere else in the same schema, which is exactly why a human reviewing one
    field at a time can miss them and why the check must be per-field."""
    assert borrowed not in ENUM_FIELDS[field], "fixture must use an ILLEGAL value"
    assert any(borrowed in e for k, e in ENUM_FIELDS.items() if k != field), \
        "fixture must use a value that IS legal in another field"
    f = base()
    f[field] = borrowed
    out = v.validate(f, AD)
    assert out["demoted"], f"{field}={borrowed!r} accepted"


@pytest.mark.parametrize("field", sorted(ENUM_FIELDS))
def test_every_legal_value_is_accepted(field):
    """The other direction: validation must not reject the schema's own values.
    Parametrised over every declared value, so narrowing an enum in code without
    narrowing it in the schema fails here."""
    for value in sorted(ENUM_FIELDS[field]):
        f = base()
        f[field] = value
        out = v.validate(f, AD)
        bad = [r for r in out["reasons"] if "enum" in r]
        assert not bad, f"{field}={value!r} rejected as invalid: {bad}"


def test_a_missing_optional_enum_is_not_an_enum_error():
    """`None` means absent, not invalid — several fields are nullable and the
    census emits null rather than a sentinel."""
    f = base()
    f["authorisation_required"] = None
    out = v.validate(f, AD)
    assert not any("enum" in r for r in out["reasons"]), out["reasons"]


def test_the_corpus_currently_holds_three_known_violations():
    """Pins the finding so the fix can be shown to have worked, and so the
    number cannot drift unnoticed before the census re-runs."""
    import json
    import duckdb
    from tests.conftest import CORPUS
    if not CORPUS.exists():
        pytest.skip("needs the corpus")
    con = duckdb.connect(str(CORPUS), read_only=True)
    bad = []
    for (fac,) in con.execute("SELECT facets FROM ad_facets").fetchall():
        d = json.loads(fac) if isinstance(fac, str) else fac
        for k, allowed in ENUM_FIELDS.items():
            if d.get(k) is not None and d[k] not in allowed:
                bad.append((k, d[k]))
    assert len(bad) == 3, f"known violations changed: {bad}"
    assert all(k == "application_language" for k, _ in bad), bad
