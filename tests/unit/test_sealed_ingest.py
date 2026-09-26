"""Ingest hand labels from the sealed-set worksheet into golden-set records.

THE POINT OF THIS MODULE. Labels arrive from a browser — either pasted JSON or
read out of the artifact's db. Whatever the route, they are UNTRUSTED input to
the one measurement the project claims is unbiased, so ingestion is the gate:

  * a row that no human confirmed must NEVER become gold. The artifact writes
    `confirmed: false` on every Opus suggestion, and the previous failure in this
    repo was a field nothing validated -- `annotated_accessible` read by scoring
    and never written by the data.
  * `derived_accessible` is RECOMPUTED here, never trusted from the payload.
    D3 kept it only because it is a deterministic function of fields the model
    emits; a hand-supplied value would let the browser assert accessibility.
  * an `evidence_span` that the REAL validator rejects must not enter the gold
    set. A gold span the extractor can never match scores as a permanent
    evidence error and silently mismeasures every future run.
  * the output must satisfy the golden set's CLOSED KEY VOCABULARY
    (tests/unit/test_scoring.py GOLDEN_KEYS), or it cannot be merged at all.
"""
import json

import pytest

from finn_smart_search.eval import sealed_ingest as si

pytestmark = pytest.mark.unit

AD = ("Krav til stillingen\n"
      "Gode norskkunnskaper er et krav for stillingen\n"
      "Oppstart snarest")
SPAN = "Gode norskkunnskaper er et krav for stillingen"


def row(**kw):
    base = {
        "n": 1, "uuid": "u-1", "title": "Sykepleier", "stratum": "doc_clause",
        "doc_lang": "no", "confirmed": True,
        "expected": {"norwegian_requirement_level": "professional",
                     "evidence_span": SPAN,
                     "stated_working_language": "unstated",
                     "authorisation_required": None},
        "note": None,
    }
    base.update(kw)
    return base


TEXTS = {"u-1": AD}


# ── the gate ────────────────────────────────────────────────────────────────

def test_an_unconfirmed_row_is_refused():
    """The artifact marks every Opus suggestion `confirmed: false`. If an
    unconfirmed row could become gold, the 'unbiased' measurement would be
    scoring the extractor against another model."""
    out, rej = si.ingest([row(confirmed=False)], TEXTS)
    assert out == []
    assert rej[0]["reason"] == "unconfirmed"


def test_a_row_with_no_level_is_refused():
    out, rej = si.ingest([row(expected={**row()["expected"],
                                        "norwegian_requirement_level": None})], TEXTS)
    assert out == [] and rej[0]["reason"] == "no_level"


def test_an_invalid_level_is_refused():
    out, rej = si.ingest([row(expected={**row()["expected"],
                                        "norwegian_requirement_level": "very_norwegian"})], TEXTS)
    assert out == [] and rej[0]["reason"] == "invalid_level"


def test_a_span_the_validator_rejects_is_refused():
    """The expensive failure. A gold span the extractor can never match is not a
    hard case — it is an unwinnable one, and it mismeasures every future run."""
    bad = "norskkunnskaper er et krav"          # starts mid-sentence
    out, rej = si.ingest([row(expected={**row()["expected"], "evidence_span": bad})], TEXTS)
    assert out == []
    assert rej[0]["reason"].startswith("span:"), rej[0]


def test_an_empty_span_is_allowed():
    """`unstated` with no evidence is a correct and common outcome — 74.6% of the
    corpus says nothing. Requiring a span would make silence unlabellable."""
    out, rej = si.ingest([row(expected={**row()["expected"],
                                        "norwegian_requirement_level": "unstated",
                                        "evidence_span": None})], TEXTS)
    assert rej == [] and len(out) == 1
    assert out[0]["expected"]["evidence_span"] is None


# ── the output must be golden-set shaped ────────────────────────────────────

def test_output_uses_only_the_closed_key_vocabulary():
    """The exact check test_9b_1 applies to eval/golden_set.json. An extra key
    here fails that test on merge, which is how `accessibility_note` slipped past
    `annotated_accessible` in the first place."""
    from tests.unit.test_scoring import EXPECTED_KEYS, GOLDEN_KEYS
    out, _ = si.ingest([row()], TEXTS)
    assert set(out[0]) <= GOLDEN_KEYS, set(out[0]) - GOLDEN_KEYS
    assert set(out[0]["expected"]) <= EXPECTED_KEYS


def test_derived_accessible_is_recomputed_not_trusted():
    """A payload claiming accessibility must not be believed. `professional` is
    blocking, so the answer is False whatever the browser sent."""
    out, _ = si.ingest([row(derived_accessible=True)], TEXTS)
    assert out[0]["derived_accessible"] is False


def test_derived_accessible_uses_doc_lang_for_unstated():
    """The one case where doc_lang decides: silence in an English-written ad
    derives accessible."""
    r = row(doc_lang="en", expected={**row()["expected"],
                                     "norwegian_requirement_level": "unstated",
                                     "evidence_span": None})
    out, _ = si.ingest([r], TEXTS)
    assert out[0]["derived_accessible"] is True


def test_every_output_row_carries_the_required_keys():
    from tests.integration.test_golden_set_integrity import golden as _  # noqa: F401
    out, _ = si.ingest([row()], TEXTS)
    for k in ("n", "uuid", "title", "stratum", "expected", "note", "doc_lang",
              "derived_accessible"):
        assert k in out[0], k


# ── shape and provenance ────────────────────────────────────────────────────

def test_rows_are_ordered_and_renumbered_contiguously():
    """Merging into the 44-ad set needs stable, non-colliding `n`."""
    rows = [row(n=9, uuid="u-1"), row(n=2, uuid="u-2")]
    out, _ = si.ingest(rows, {"u-1": AD, "u-2": AD}, start_n=100)
    assert [o["n"] for o in out] == [100, 101]
    assert [o["uuid"] for o in out] == ["u-2", "u-1"]


def test_a_duplicate_uuid_is_refused():
    out, rej = si.ingest([row(uuid="dup"), row(uuid="dup")], {"dup": AD})
    assert len(out) == 1 and rej[0]["reason"] == "duplicate_uuid"


def test_ingest_is_idempotent():
    a, _ = si.ingest([row()], TEXTS)
    b, _ = si.ingest([row()], TEXTS)
    assert a == b


def test_a_uuid_with_no_ad_text_is_refused():
    """Without the ad text the span cannot be validated, and an unvalidated span
    is exactly what this gate exists to stop."""
    out, rej = si.ingest([row(uuid="ghost")], {})
    assert out == [] and rej[0]["reason"] == "no_ad_text"


def test_a_span_from_the_DELETED_MIDDLE_of_a_truncated_ad_is_refused():
    """THE ad-8 failure mode, as a gate.

    prepare() sends body[:3500] + "[...]" + body[-2500:]. A span from the deleted
    middle is verbatim in the ad and unmatchable in production, so validating
    against the FULL text instead of `sent_text` would let it into the gold set
    and make that ad unwinnable — the extractor is never shown the sentence and
    can only emit `unstated`.

    That is not hypothetical: one selected ad had its ONLY language line there,
    and it had to be replaced. Without this test, 'validate against the full ad'
    survived every other assertion in this file.
    """
    from finn_smart_search.understanding.census_prompt import HEAD_CHARS, TAIL_CHARS

    buried = "Gode norskkunnskaper er et absolutt krav for denne stillingen"
    body = ("Om stillingen\n"
            + "Vi er en stor arbeidsgiver i regionen. " * 100      # head
            + "\n" + buried + "\n"                                  # the middle
            + "Vi tilbyr gode betingelser og et godt miljø. " * 100  # tail
            + "\nSøknadsfrist snarest")
    assert len(body) > HEAD_CHARS + TAIL_CHARS, len(body)
    sent = si.v.prepare("t", body, "no")["sent_text"]
    assert buried in body and buried not in sent, "fixture must bury the span"

    out, rej = si.ingest(
        [row(uuid="trunc", expected={**row()["expected"], "evidence_span": buried})],
        {"trunc": body})
    assert out == [], "a span the model is never shown must not become gold"
    assert rej[0]["reason"].startswith("span:"), rej[0]


# ── authorisation_required must be grounded in the ad, not paraphrased ───────
#
# WHY. The field is SCORED: scoring._auth_matches compares the extractor's output
# against this string with a profession regex. The extractor emits Norwegian as
# the ad writes it, so an English gold value fails a correct answer.
#
# This is not hypothetical — the first label produced through the worksheet
# stored Opus's rendering, "Norwegian psychologist authorisation (norsk
# autorisasjon som psykolog)", for an ad whose own text says
# "Psykolog med norsk autorisasjon."
#
# The rule is checked against the existing gold before being imposed: all five
# golden ads carrying this field have a value that IS verbatim in their ad
# ("Norsk autorisasjon", "Gyldig norsk autorisasjon som helsefagarbeider", …),
# so this rejects nothing that is already accepted.

AUTH_AD = ("Kvalifikasjoner\n"
           "Psykolog med norsk autorisasjon\n"
           "Gode norskkunnskaper er et krav for stillingen\n"
           "Oppstart snarest")


def test_an_authorisation_value_not_in_the_ad_is_refused():
    r = row(uuid="auth", expected={**row()["expected"],
            "authorisation_required": "Norwegian psychologist authorisation "
                                      "(norsk autorisasjon som psykolog)"})
    out, rej = si.ingest([r], {"auth": AUTH_AD})
    assert out == [], "an English paraphrase would fail _auth_matches against a correct answer"
    assert rej[0]["reason"] == "authorisation_not_in_ad"


def test_the_ads_own_authorisation_wording_is_accepted():
    r = row(uuid="auth", expected={**row()["expected"],
            "authorisation_required": "Psykolog med norsk autorisasjon"})
    out, rej = si.ingest([r], {"auth": AUTH_AD})
    assert rej == [] and out[0]["expected"]["authorisation_required"] == \
        "Psykolog med norsk autorisasjon"


def test_a_shorter_grounded_form_is_accepted():
    """The existing gold uses both full and short forms — "Norsk autorisasjon"
    appears on its own in one ad. Substring grounding, not equality."""
    r = row(uuid="auth", expected={**row()["expected"],
            "authorisation_required": "norsk autorisasjon"})
    out, rej = si.ingest([r], {"auth": AUTH_AD})
    assert rej == [] and out


def test_no_authorisation_is_fine():
    out, rej = si.ingest([row()], TEXTS)
    assert rej == [] and out[0]["expected"]["authorisation_required"] is None


def test_the_rule_accepts_every_value_already_in_the_golden_set():
    """Grounded in the artefact it governs: a rule the current gold fails would
    be the wrong rule, so this asserts it against all five real values."""
    import json, pathlib
    import duckdb
    from tests.conftest import CORPUS
    if not CORPUS.exists():
        pytest.skip("needs the corpus")
    con = duckdb.connect(str(CORPUS), read_only=True)
    gold = json.loads((pathlib.Path("eval/golden_set.json")).read_text(encoding="utf-8"))
    checked = 0
    for g in gold:
        v = g["expected"].get("authorisation_required")
        if not v:
            continue
        text = con.execute("SELECT description_text FROM ads WHERE uuid=?",
                           [g["uuid"]]).fetchone()[0]
        assert si._auth_grounded(v, text), f"golden #{g['n']}: {v!r}"
        checked += 1
    assert checked == 5, f"expected 5 golden ads with an authorisation, saw {checked}"
