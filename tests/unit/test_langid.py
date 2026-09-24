"""langid — document language detection over cleaned ad blocks.

Four decisions here reversed an earlier design, each because a MEASUREMENT
contradicted the reasoning rather than because an argument was better:

  DETECT UNRESTRICTED. Restricting the detector to five Nordic/English
  languages does not make a Polish ad fail — it force-projects it onto the
  nearest of the five. Measured: 10 of 13 Polish ads got a confident "no",
  which under the accessibility fallback means NOT accessible. The system would
  have hidden the ads most obviously written for non-Norwegian speakers.

  NO sv/da VERDICTS. Zero ads in 1,500 are majority Swedish or Danish, and
  bokmål descends from Danish — "Vi søker en dyktig medarbeider" is valid
  Danish orthography. A class with no true instances has nothing to dilute
  error into, so one misread block would be 100% of its evidence. The
  Swedish-speaking SEEKER is served in the matcher, not here.

  FLOOR = 10, not 40. Per-block accuracy is the wrong level of analysis: we
  char-weight, so a 15-char block is 15 of ~2,800 characters. Measured, floor
  10 leaves ZERO ads decided on under half their text; floor 40 strands 19.

  STRIP BOILERPLATE, don't tune the threshold. A 450-char English GDPR/EEO
  footer on a 1,200-char Norwegian ad gives an English share of 0.27 -> "mixed"
  under a pure ratio. The 0.20 band is an artifact of this corpus's length
  distribution, not a principle.
"""
import pytest

from finn_smart_search.understanding import langid

pytestmark = pytest.mark.unit

NO = ("Vi søker en dyktig medarbeider til vårt team i Oslo. Du vil jobbe med "
      "daglig drift og oppfølging av kunder. Stillingen er fast og starter etter avtale.")
NN = ("Vi søkjer ein dyktig medarbeidar til eininga vår. Du skal arbeide med "
      "dagleg drift og oppfølging av brukarane våre. Stillinga er fast.")
EN = ("We are looking for a skilled engineer to join our team in Oslo. You will "
      "work on daily operations and customer follow-up. The position is permanent.")
PL = ("Poszukujemy wykwalifikowanych monterów wentylacji na projekt w "
      "miejscowości Haugesund i okolicach. Oferujemy umowę o pracę oraz "
      "zakwaterowanie. Wymagania: doświadczenie w montażu.")
GDPR_EN = ("We process your personal data in accordance with GDPR. We are an equal "
           "opportunity employer and welcome applicants regardless of race, gender, "
           "age or disability. See our privacy policy for details.")


def blocks(*texts):
    return [{"index": i, "tag": "p", "text": t, "n_chars": len(t)}
            for i, t in enumerate(texts)]


# ── 1 · document verdicts ────────────────────────────────────────────────────

def test_1_1_bokmal_is_no():
    assert langid.detect(blocks(NO))["doc_lang"] == "no"


def test_1_2_nynorsk_is_no():
    """Two written standards of one language, needing identical competence.
    762 corpus ads carry nynorsk markers."""
    assert langid.detect(blocks(NN))["doc_lang"] == "no"


def test_1_3_english_is_en():
    assert langid.detect(blocks(EN))["doc_lang"] == "en"


def test_1_4_polish_is_other_and_records_what_it_was():
    """THE bug this module exists to prevent. 13 such ads in the corpus; the
    restricted detector labelled 10 of 13 a confident "no"."""
    r = langid.detect(blocks(PL))
    assert r["doc_lang"] == "other"
    assert r["detected_other"] is not None
    assert r["detected_other"] != "no"


def test_1_5_bilingual_is_mixed():
    r = langid.detect(blocks(NO, EN))
    assert r["doc_lang"] == "mixed"
    assert r["is_bilingual"] is True


def test_1_6_empty_is_unknown():
    r = langid.detect([])
    assert r["doc_lang"] == "unknown"
    assert r["n_scored"] == 0
    assert r["confidence"] == "low"


def test_1_7_blocks_below_the_floor_are_not_scored():
    assert langid.detect(blocks("Oslo", "Kontakt", "Frist"))["doc_lang"] == "unknown"


# ── 2 · boilerplate stripping ────────────────────────────────────────────────

def test_2_1_english_boilerplate_is_stripped_from_a_long_ad():
    r = langid.detect(blocks(NO, NO, NO, GDPR_EN))
    assert r["doc_lang"] == "no"


def test_2_2_english_boilerplate_is_stripped_from_a_SHORT_ad():
    """The case a pure ratio gets wrong. One Norwegian block (~150 chars) plus a
    ~200-char English GDPR footer is an English share near 0.57 -> "mixed"
    without stripping."""
    r = langid.detect(blocks(NO, GDPR_EN))
    assert r["doc_lang"] == "no", "boilerplate must not turn a Norwegian ad bilingual"


def test_2_3_a_genuinely_english_section_is_NOT_stripped():
    """Stripping must key on boilerplate phrasing, not merely on being English —
    otherwise every bilingual ad collapses to Norwegian."""
    assert langid.detect(blocks(NO, EN, EN))["doc_lang"] in ("mixed", "en")


def test_2_4_an_all_boilerplate_input_is_unknown_not_english():
    assert langid.detect(blocks(GDPR_EN))["doc_lang"] == "unknown"


# ── 3 · aggregation ──────────────────────────────────────────────────────────

def test_3_1_char_weighted_not_block_counted():
    """One long Norwegian block must outweigh three short English ones, or a
    bullet list swamps the body."""
    short_en = "Apply now today please"
    r = langid.detect(blocks(NO * 3, short_en, short_en, short_en))
    assert r["doc_lang"] == "no"


def test_3_2_lang_mix_sums_to_one():
    mix = langid.detect(blocks(NO, EN))["lang_mix"]
    assert abs(sum(mix.values()) - 1.0) < 1e-9


def test_3_3_deterministic():
    b = blocks(NO, EN)
    assert langid.detect(b) == langid.detect(b)


def test_3_4_other_dominance_beats_a_stray_norwegian_block():
    """A Polish ad almost always contains a Norwegian contact or location line.
    That must not flip the verdict."""
    r = langid.detect(blocks(PL, PL, NO))
    assert r["doc_lang"] == "other"


# ── 4 · pinned constants, each with a measurement behind it ──────────────────

def test_4_1_floor_is_ten():
    """Floor 10 leaves ZERO ads decided on under half their text; floor 40
    strands 19 of 1,500 and costs 6 unknowns, for no verdict change."""
    assert langid.MIN_BLOCK_CHARS == 10


def test_4_2_mixed_band():
    """Re-measured AT FLOOR 10, not inherited from the floor-40 run: across
    0.05-0.95 to 0.40-0.60 only 4 ads of 1,500 change classification."""
    assert (langid.MIXED_LO, langid.MIXED_HI) == (0.20, 0.80)


# ── 5 · the accessibility contract ───────────────────────────────────────────

@pytest.mark.parametrize("doc_lang", ["other", "unknown"])
def test_5_1_other_and_unknown_are_accessible(doc_lang):
    """A false hide is invisible to everyone; a false show costs one click. An
    ad written in Polish is not being kept from you by a Norwegian requirement."""
    from finn_smart_search.understanding.census_prompt import derive_english_accessible
    assert derive_english_accessible(
        {"norwegian_requirement_level": "unstated",
         "stated_working_language": "unstated"}, doc_lang) is True


def test_5_2_a_requirement_still_blocks_an_other_language_ad():
    """`other` relaxes the SILENCE fallback, never an explicit requirement."""
    from finn_smart_search.understanding.census_prompt import derive_english_accessible
    assert derive_english_accessible(
        {"norwegian_requirement_level": "professional",
         "stated_working_language": "unstated"}, "other") is False
