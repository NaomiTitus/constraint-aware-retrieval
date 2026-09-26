"""A constraint the seeker did not state must change nothing at all.

WHY THIS IS THE FIRST TEST IN THE RETRIEVAL LAYER, and why it is written
before the layer exists.

The census measured what happens if this rule is broken. Of 10,166 ads:

    accessible to an English speaker      956   9.4%
    blocked by a STATED requirement     5,703  56.1%
    blocked by SILENCE alone            3,507  34.5%   <- a DEFAULT

`derive_english_accessible` calls an `unstated` level on a Norwegian-written
ad inaccessible. That is a defensible default for the EVAL, which needs a
binary. Inherited by RETRIEVAL it is a catastrophe: a seeker who never
mentioned language would see 9.4% of the corpus, the product would be
destroyed for the ~90% who speak Norwegian, and the language metric would
look excellent throughout — because every persona in eval/personas.yaml
stated a language constraint until the four controls were added.

So the invariant is not "the penalty should be small". It is EXACTLY ZERO,
and the ranking must be bit-identical to the same pipeline with the
constraint stage switched off. An approximate version of this test would
pass while the product silently lost 90% of its inventory.

GROUNDING (STANDARDS.md §3.0): the facet records below are loaded from the
census, not constructed. A hand-built facet would let the rule be written
against a shape no advertisement has.
"""
import json

import pytest

from finn_smart_search.retrieval import constraints as C

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def real_facets():
    """A spread of REAL census records, one per requirement level."""
    import duckdb
    from pathlib import Path
    db = Path(__file__).resolve().parents[2] / "data" / "ads.duckdb"
    if not db.exists():
        pytest.skip("needs the corpus")
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute("""
        SELECT f.facets, l.doc_lang,
               json_extract_string(f.facets,'$.norwegian_requirement_level') lvl
        FROM ad_facets f JOIN ad_language l USING (uuid)
        QUALIFY row_number() OVER (PARTITION BY lvl ORDER BY f.uuid) <= 3
    """).fetchall()
    con.close()
    out = [(json.loads(f) if isinstance(f, str) else f, dl) for f, dl, _ in rows]
    assert len(out) >= 20, f"expected a spread of real records, got {len(out)}"
    return out


SILENT = C.SeekerProfile(raw_query="Jeg søker jobb på lager i Bergen.")
SPEAKS_NONE = C.SeekerProfile(
    raw_query="I do not speak Norwegian.",
    language_constraint=C.LanguageConstraint(norwegian="none"))
SPEAKS_FLUENT = C.SeekerProfile(
    raw_query="Jeg snakker flytende norsk.",
    language_constraint=C.LanguageConstraint(norwegian="fluent"))


def test_no_stated_language_constraint_means_severity_is_exactly_zero(real_facets):
    """Not small. Zero. 3,507 ads hang on this."""
    for facets, doc_lang in real_facets:
        s = C.language_severity(facets, doc_lang, SILENT)
        assert s == 0.0, (
            f"an unstated constraint penalised an ad by {s} "
            f"(level={facets.get('norwegian_requirement_level')}, "
            f"doc_lang={doc_lang})")


def test_a_silent_profile_leaves_the_ranking_bit_identical(real_facets):
    """The invariant as the product experiences it."""
    scores = [1.0 / (i + 1) for i in range(len(real_facets))]
    with_stage = C.apply(scores, real_facets, SILENT, lam=0.7)
    assert with_stage == scores, "the constraint stage moved a silent query"


def test_a_stated_constraint_does_change_the_ranking(real_facets):
    """The other half: if stating a constraint changed nothing, the whole
    project would be pointless. This is the guard against 'fix' the first
    test by making apply() a no-op."""
    scores = [1.0] * len(real_facets)
    out = C.apply(scores, real_facets, SPEAKS_NONE, lam=0.7)
    assert out != scores
    assert any(x < 1.0 for x in out)


def test_a_fluent_speaker_is_penalised_for_nothing(real_facets):
    """Stating that you DO speak Norwegian is still a stated constraint, and
    it must never hide an ad — it can only stop ads being hidden."""
    for facets, doc_lang in real_facets:
        assert C.language_severity(facets, doc_lang, SPEAKS_FLUENT) == 0.0


def test_silence_is_penalised_GRADUALLY_not_absolutely(real_facets):
    """The eval's boolean says an `unstated` level on a Norwegian ad is
    inaccessible. Retrieval must NOT inherit that: hard-filtering silence
    removes a third of the corpus. It is a partial penalty, and lambda can
    turn it off entirely."""
    silent_no = [(f, dl) for f, dl in real_facets
                 if f.get("norwegian_requirement_level") == "unstated" and dl == "no"]
    assert silent_no, "no silent Norwegian-written ad in the sample"
    for facets, doc_lang in silent_no:
        s = C.language_severity(facets, doc_lang, SPEAKS_NONE)
        assert 0.0 < s < 1.0, f"silence must be graded, got {s}"


def test_a_stated_requirement_outweighs_mere_silence(real_facets):
    """Ordering, not magnitudes: an ad that DEMANDS Norwegian must be
    penalised more than one that simply says nothing."""
    def sev(level, dl="no"):
        return C.language_severity({"norwegian_requirement_level": level,
                                    "stated_working_language": "unstated"},
                                   dl, SPEAKS_NONE)
    assert sev("professional") > sev("unstated") > sev("explicitly_not_required")
    assert sev("explicitly_not_required") == 0.0


def test_lambda_zero_disables_the_stage_entirely(real_facets):
    """The product knob. lambda is a dial, not a switch bolted shut."""
    scores = [0.5] * len(real_facets)
    assert C.apply(scores, real_facets, SPEAKS_NONE, lam=0.0) == scores


def test_every_control_persona_is_marked_as_stating_no_constraint():
    """The personas and this rule have to agree, or the eval measures nothing."""
    import yaml
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "eval" / "personas.yaml"
    ps = yaml.safe_load(p.read_text(encoding="utf-8"))["personas"]
    controls = [x for x in ps if x.get("control")]
    assert len(controls) >= 4, f"only {len(controls)} control personas"
    for c in controls:
        assert c.get("expect_language_constraint") is False, c["id"]


def test_two_controls_say_norsk_without_stating_a_language_constraint():
    """c2 says `norsk autorisasjon` (a LICENCE) and c4 says `norsk og
    matematikk` (a SUBJECT TAUGHT). A parser that greps for `norsk` applies a
    language penalty to both. They are in the set precisely to catch that."""
    import yaml
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "eval" / "personas.yaml"
    ps = {x["id"]: x for x in yaml.safe_load(p.read_text(encoding="utf-8"))["personas"]}
    assert "norsk autorisasjon" in ps["c2_sykepleier_authorisation_only"]["query"]
    assert "norsk og matematikk" in ps["c4_laerer_norsk_speaker"]["query"]
    for cid in ("c2_sykepleier_authorisation_only", "c4_laerer_norsk_speaker"):
        assert ps[cid]["expect_language_constraint"] is False
