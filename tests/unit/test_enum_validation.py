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
import re

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


def test_the_corpus_holds_no_violation_outside_the_known_pre_v9_records():
    """Pins the finding AND its intended end state.

    The first version of this test asserted `== 3`, which made it a countdown
    to a red suite: the moment the census re-runs under census-v9 or later the
    three disappear and the test fails for the RIGHT reason, with no assertion
    anywhere for the state we actually want. A test that must be edited when
    the fix lands is not pinning the fix.

    So: zero violations among records written by census-v9 or later, and at
    most the three known ones among the older records still in the table."""
    import json
    import duckdb
    from tests.conftest import CORPUS
    if not CORPUS.exists():
        pytest.skip("needs the corpus")
    con = duckdb.connect(str(CORPUS), read_only=True)
    rows = con.execute("SELECT prompt_version, facets FROM ad_facets").fetchall()
    con.close()

    def version_num(pv: str) -> int:
        m = re.search(r"v(\d+)", pv or "")
        return int(m.group(1)) if m else 0

    current, legacy = [], []
    for pv, fac in rows:
        d = json.loads(fac) if isinstance(fac, str) else fac
        for k, allowed in ENUM_FIELDS.items():
            if d.get(k) is not None and d[k] not in allowed:
                (current if version_num(pv) >= 9 else legacy).append((pv, k, d[k]))

    # The end state, asserted rather than awaited.
    assert not current, (
        f"census-v9+ produced {len(current)} out-of-enum values: {current[:5]}")
    # The finding, still pinned while pre-v9 rows remain in the table.
    assert len(legacy) <= 3, f"pre-v9 violations grew: {legacy}"
    assert all(k == "application_language" for _, k, _ in legacy), legacy


# ===========================================================================
# Found by an adversarial audit of the fix itself, not by writing it.
# Mutation testing killed 2 of 7 mutants; these pin what the other 5 exposed.
# ===========================================================================

from finn_smart_search.understanding.census_prompt import TOOL as _TOOL


def _nested_enums():
    """Every enum in the schema, at ANY depth — the one-level accessor the fix
    shipped with could not see three of them, and the guard test that was meant
    to prove the derivation complete used the SAME accessor, so it certified the
    blind spot. A test that re-derives with the code's own accessor is not
    independent of the code."""
    found = {}

    def walk(node, path):
        if isinstance(node, dict):
            if "enum" in node:
                found[path] = node["enum"]
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for v in node:
                walk(v, path)

    walk(_TOOL["input_schema"]["properties"], "")
    # TOOL spells a nested path with its JSON-Schema plumbing in it
    # (`evidence_spans.items.properties.section_language`). Strip the plumbing
    # so the comparison is against the same names the validator keys on.
    return {".".join(seg for seg in k.split(".")
                     if seg not in ("items", "properties")): val
            for k, val in found.items()}


def test_every_enum_in_the_schema_is_validated_at_any_depth():
    """Three enums live below `properties.<field>`: `section_language` inside an
    evidence span, `level` inside a skill, and `implicit_evidence` whose ITEMS
    carry the enum. All three were accepted with junk values and no reason."""
    declared = _nested_enums()
    assert len(declared) >= 11, declared          # 8 top-level + 3 nested
    # compare by LEAF name: TOOL spells a nested path
    # `evidence_spans.items.properties.section_language`, the validator keys it
    # `evidence_spans.section_language`. Both end in the same leaf.
    covered = set(v.ENUMS) | set(v.NESTED_ENUMS)
    unvalidated = sorted(set(declared) - covered)
    assert not unvalidated, f"declared in TOOL, enforced nowhere: {unvalidated}"


@pytest.mark.parametrize("facets,bad", [
    ({"evidence_spans": [{"span": "Vi krever norsk.", "section_language": "nb"}]},
     "section_language"),                          # 'nb' is not in {no,en,other}
    ({"skills": [{"phrase": "sykepleie", "level": "mandatory"}]}, "level"),
    ({"implicit_evidence": ["bogus_duty"]}, "implicit_evidence"),
])
def test_nested_enum_violations_are_reported(facets, bad):
    f = base() | facets
    out = v.validate(f, AD)
    assert any(bad in r for r in out["reasons"]), out["reasons"]


def test_a_list_where_a_string_belongs_does_not_crash_the_run():
    """`application_language: ["both"]` raised TypeError: unhashable type.

    `census.run()` does not guard `validate()`, so ONE such record aborts the
    whole census after the batch has been paid for. The shape matters: the
    neighbouring field `implicit_evidence` IS an array, so borrowing a
    neighbour's SHAPE is the same failure class as borrowing its vocabulary —
    which is the failure this module was extended to catch. Before the fix the
    value flowed through harmlessly; the fix turned it into a hard crash.
    """
    out = v.validate(base() | {"application_language": ["both"]}, AD)
    assert out["demoted"]
    assert any("application_language" in r for r in out["reasons"])
    assert out["facets"]["application_language"] == "unstated"


@pytest.mark.parametrize("field,expected", [
    ("evidence_basis", "no_mention"),     # its enum has no `unstated`
    ("evidence_strength", "none"),        # nor does its
    ("application_language", "unstated"),
    ("seniority", "unstated"),
])
def test_the_repair_writes_a_value_the_schema_actually_permits(field, expected):
    """The repair wrote `"unstated" if "unstated" in enum else None`.

    Two of the eight enums have no `unstated`, so those repaired to None — and
    both fields are `required` and `{"type": "string"}` in TOOL, so the repaired
    record VIOLATES the schema it was repaired to satisfy. Worse, feeding it back
    through validate() returns ok=True: None reads as absent, so the validator
    cannot detect its own output as broken.
    """
    out = v.validate(base() | {field: "borrowed_from_somewhere_else"}, AD)
    assert out["facets"][field] == expected
    assert out["facets"][field] in v.ENUMS[field]
    # and the repaired record must survive a second pass unchanged
    again = v.validate(out["facets"], AD)
    assert again["facets"][field] == expected


def test_the_repair_is_asserted_to_have_HAPPENED_not_merely_flagged():
    """Mutants M1b and M5 — keep the reason, drop the repair — both SURVIVED.

    grep across tests/ for `invalid_enum` returned nothing: every existing test
    asserted `demoted` and that some reason contained "enum", and none asserted
    the field's value changed. The repair, which is the half of the fix the
    commit argues hardest for, was entirely unpinned.
    """
    out = v.validate(base() | {"visa_sponsorship": "scandinavian_accepted"}, AD)
    assert "demoted:invalid_enum_value" in out["reasons"]
    assert out["facets"]["visa_sponsorship"] == "unstated"        # the repair
    assert out["facets"]["visa_sponsorship"] != "scandinavian_accepted"


def test_every_bad_field_is_repaired_not_just_the_first():
    """Mutant M2 — `break` after the first bad field — SURVIVED, because no
    fixture ever put two illegal values in one record."""
    out = v.validate(base() | {"application_language": "both",
                                "visa_sponsorship": "offered_maybe",
                                "seniority": "professional"}, AD)
    assert out["facets"]["application_language"] == "unstated"
    assert out["facets"]["visa_sponsorship"] == "unstated"
    assert out["facets"]["seniority"] == "unstated"


def test_a_required_enum_field_that_is_None_is_reported():
    """Mutant M3 — treat None as invalid — SURVIVED. The test that looked like
    it guarded this set `authorisation_required = None`, a field with NO enum,
    so the assertion could not reach the branch; and base() already sets it to
    None, making the test a no-op twice over. The repo's `fixture that cannot
    fail` pattern, reproduced inside the new file."""
    f = base()
    f["norwegian_requirement_level"] = None
    out = v.validate(f, AD)
    assert any("norwegian_requirement_level" in r for r in out["reasons"]), out["reasons"]


# ===========================================================================
# TYPE AND SIZE CONSTRAINTS — declared in TOOL, enforced by nothing.
#
# Tool-use does not enforce them server-side, census_validate covered only
# enums, and scoring covers one field. Measured over all 196 persisted
# records: ZERO violations of any constraint below. So these are forward
# guards, not a cleanup — but two of them CRASH rather than warn, and a crash
# inside census.run()'s loop loses a paid-for batch:
#
#   evidence_spans = ["a bare string"]  -> AttributeError: 'str' has no 'get'
#   application_language = ["both"]     -> TypeError: unhashable  (fixed above)
#
# The census is 9,823 clusters against the 196 that established "zero
# violations". A rate of 0 over 196 has a 95% upper bound near 1.5%, which
# over 9,823 records is up to ~150 of them. "Latent" is a statement about
# sample size, not about safety.
# ===========================================================================

TYPE_CASES = [
    ("evidence_spans", ["a bare string where an object belongs"], "shape"),
    ("evidence_spans", [{"span": SPAN, "section_language": "no"}] * 5, "maxItems"),
    ("skills", ["sykepleie"], "shape"),
    ("skills", [{"phrase": f"s{i}", "level": "required"} for i in range(20)], "maxItems"),
    ("implicit_evidence", "brukerkontakt", "type"),          # string, not a list
    ("conflicting_statements", "true", "type"),              # string, not a bool
    ("security_clearance_required", "false", "type"),
    ("min_years_experience", "3", "type"),
    ("min_years_experience", -5, "range"),
    ("min_years_experience", 400, "range"),
    ("authorisation_required", 123, "type"),
]


@pytest.mark.parametrize("field,value,kind", TYPE_CASES)
def test_a_type_or_size_violation_is_reported_and_never_raises(field, value, kind):
    out = v.validate(base() | {field: value}, AD)
    assert any(field in r for r in out["reasons"]), (
        f"{field}={value!r} ({kind}) passed validation silently: {out['reasons']}")
    assert out["demoted"]


@pytest.mark.parametrize("field,value,kind", TYPE_CASES)
def test_a_malformed_record_does_not_crash_the_census(field, value, kind):
    """census.run() does not guard validate(). One raise aborts the whole run
    AFTER the batch is paid for, which is the expensive failure mode."""
    v.validate(base() | {field: value}, AD)          # must simply not raise


def test_the_persisted_corpus_violates_none_of_these():
    """The grounding for calling them forward guards. If this ever fails, the
    claim in the comment above is stale and must be re-measured, not edited."""
    import json as _json
    import duckdb as _duckdb
    from pathlib import Path as _Path
    db = _Path(__file__).resolve().parents[2] / "data" / "ads.duckdb"
    if not db.exists():
        pytest.skip("corpus not present")
    con = _duckdb.connect(str(db), read_only=True)
    rows = con.execute("SELECT uuid, facets FROM ad_facets").fetchall()
    con.close()
    assert rows, "no persisted facets to check"
    bad = []
    for uuid, fj in rows:
        f = _json.loads(fj) if isinstance(fj, str) else fj
        for r in v._shape_errors(f):
            bad.append((uuid, r))
    assert not bad, f"{len(bad)} persisted records violate a shape constraint: {bad[:5]}"


# ===========================================================================
# FOUND BY THE 400-AD CENSUS STAGE, and it is a bug this session introduced.
#
# Ad 08f980a7 came back `certified` with evidence_spans == [] and reasons
#   ["shape:evidence_spans[1].span:maxLength_300_got_306",
#    "demoted:invalid_shape"]
#
# ONE span was six characters over the 300 limit, and the shape repair — added
# hours earlier — replaced the WHOLE array with []. The advertisement really
# does say "Norwegian language skills at level B2", so the level was right and
# its evidence was thrown away wholesale over a length overrun on a different
# span.
#
# Worse, the record then kept a `certified` verdict resting on nothing. The
# guard for that ("a verdict resting only on rejected spans cannot stand")
# runs BEFORE the shape repair, so by the time the repair emptied the array
# the check had already passed. A repair that runs after the last check is
# unchecked.
#
# This is the same failure I had just criticised in the enum repair: a repair
# producing a record worse than the one it repaired.

def _span(n):
    return {"span": "Vi krever norsk. " * n, "section_language": "no"}


def test_an_oversized_span_drops_only_itself_not_the_whole_array():
    good = {"span": SPAN, "section_language": "no"}
    over = {"span": "x" * 400, "section_language": "no"}
    out = v.validate(base() | {"evidence_spans": [good, over]}, AD)
    kept = [e.get("span") for e in out["facets"]["evidence_spans"]]
    assert SPAN in kept, f"the valid span was discarded too: {out['facets']['evidence_spans']}"
    assert not any(len(s) > 300 for s in kept), kept


def test_a_verdict_left_with_no_evidence_by_a_REPAIR_is_still_demoted():
    """The span-loss guard must see the record as the repair leaves it, not as
    it arrived. Ad 08f980a7 kept `certified` on zero spans because the repair
    ran after the guard."""
    over = {"span": "y" * 400, "section_language": "no"}
    out = v.validate(base() | {"evidence_spans": [over],
                               "norwegian_requirement_level": "certified",
                               "evidence_basis": "explicit_statement"}, AD)
    assert out["facets"]["evidence_spans"] == []
    assert out["facets"]["norwegian_requirement_level"] == "unstated", (
        "a level with no surviving evidence must demote; got "
        f"{out['facets']['norwegian_requirement_level']!r}")
    assert out["demoted"]


def test_the_persisted_corpus_holds_no_verdict_without_evidence():
    """The end state, asserted against real data rather than a fixture."""
    import json as _json
    import duckdb as _duckdb
    from pathlib import Path as _Path
    db = _Path(__file__).resolve().parents[2] / "data" / "ads.duckdb"
    if not db.exists():
        pytest.skip("corpus not present")
    con = _duckdb.connect(str(db), read_only=True)
    rows = con.execute("SELECT uuid, facets FROM ad_facets").fetchall()
    con.close()
    bad = []
    for uuid, fj in rows:
        d = _json.loads(fj) if isinstance(fj, str) else fj
        if (d.get("norwegian_requirement_level") not in ("unstated", None)
                and not (d.get("evidence_spans") or [])
                and d.get("evidence_basis") == "explicit_statement"):
            bad.append(uuid)
    assert not bad, f"{len(bad)} records assert a level on no evidence: {bad[:5]}"


def test_a_maxitems_overflow_truncates_rather_than_empties():
    """Ad 09f87e6b returned 9 skills against a cap of 8 and lost all nine.
    Eight good skills are worth more than none, and the cap is what the schema
    asks for."""
    skills = [{"phrase": f"skill {i}", "level": "required"} for i in range(9)]
    out = v.validate(base() | {"skills": skills}, AD)
    kept = out["facets"]["skills"]
    assert len(kept) == 8, f"expected truncation to 8, got {len(kept)}"
    assert kept[0]["phrase"] == "skill 0"
