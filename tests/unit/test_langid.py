"""langid — document language detection over cleaned ad blocks.

Test design note. `detect()` weights by `b["n_chars"]` but detects on
`b["text"]`, so weights are FORGEABLE. Policy tests (bands, weighting,
dominance) therefore use exact forged arithmetic and never depend on lingua
classifying a short fragment — its least reliable regime. Only the oracle tests
in section 0 exercise lingua itself, so a version bump fails there, loudly and
alone, instead of surfacing as an undiagnosable band regression.

Four decisions reversed an earlier design, each because a MEASUREMENT
contradicted the reasoning:

  DETECT UNRESTRICTED. Restricting to five Nordic/English languages does not
  make a Polish ad fail — it force-projects it onto the nearest of the five.
  Measured: 10 of 13 Polish ads got a confident "no", which under the
  accessibility fallback means NOT accessible.

  NO sv/da VERDICTS. Zero ads are majority Swedish or Danish, and bokmål
  descends from Danish. A class with no true instances has nothing to dilute
  error into.

  FLOOR = 10, not 40. Char-weighting makes a 15-char block 15 of ~2,800
  characters. Floor 10 leaves ZERO ads decided on under half their text; floor
  40 strands 19 of 1,500.

  STRIP BOILERPLATE, don't tune the threshold. The 0.20 band is an artifact of
  this corpus's length distribution, not a principle.
"""
import pytest

from finn_smart_search.understanding import langid
from finn_smart_search.understanding.langid import detect, is_boilerplate

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


def blk(text, n_chars=None):
    return {"index": 0, "tag": "p", "text": text,
            "n_chars": len(text) if n_chars is None else n_chars}


def blocks(*texts):
    return [dict(blk(t), index=i) for i, t in enumerate(texts)]


# ── 0 · ORACLE — the only tests that exercise lingua itself ─────────────────

@pytest.mark.parametrize("text,expected_class", [
    (NO, "no"), (NN, "no"), (EN, "en"), (PL, "other")])
def test_0_classifier_oracle(text, expected_class):
    """Fails loudly and ALONE on a lingua version bump, instead of surfacing as
    an undiagnosable band or weighting regression."""
    assert langid._classify(text)[0] == expected_class


def test_0b_polish_is_identified_as_polish():
    """`!= "no"` was nearly unfalsifiable. Polish -> Slovak drift is exactly the
    silent regression worth surfacing."""
    assert langid._classify(PL)[1] == "pl"


# ── 1 · document verdicts ────────────────────────────────────────────────────

def test_1_1_bokmal_is_no():
    assert detect(blocks(NO))["doc_lang"] == "no"


def test_1_2_nynorsk_is_no():
    """Two written standards of one language. 762 corpus ads carry nynorsk."""
    assert detect(blocks(NN))["doc_lang"] == "no"


def test_1_3_english_is_en():
    assert detect(blocks(EN))["doc_lang"] == "en"


def test_1_4_polish_is_other_and_names_the_language():
    """THE bug this module prevents: a restricted detector labelled 10 of 13
    Polish ads a confident "no"."""
    r = detect(blocks(PL))
    assert r["doc_lang"] == "other"
    assert r["detected_other"] == "pl"


def test_1_5_bilingual_is_mixed():
    r = detect([blk(NO, 500), blk(EN, 500)])
    assert r["doc_lang"] == "mixed"
    assert r["is_bilingual"] is True


def test_1_6_empty_is_unknown():
    r = detect([])
    assert r == {"doc_lang": "unknown", "lang_mix": {}, "detected_other": None,
                 "is_bilingual": False, "n_scored": 0, "confidence": "low"}


# ── 2 · thresholds, with forged weights so the arithmetic is exact ──────────

@pytest.mark.parametrize("n_chars,scored", [(9, 0), (10, 1), (11, 1)])
def test_2_1_floor_boundary_is_exactly_ten(n_chars, scored):
    """Pins the floor at exactly 10 — the old fixture (4-7 char strings) could
    only distinguish <=7 from >=8. Also pins that n_chars, not len(text), is
    the weight: a desync with upstream cleaning would be silent."""
    assert detect([blk(NO, n_chars)])["n_scored"] == scored


@pytest.mark.parametrize("en_chars,expected", [
    (199, "no"), (200, "no"), (201, "mixed"),      # MIXED_LO = 0.20, inclusive
    (799, "mixed"), (800, "en"), (801, "en"),      # MIXED_HI = 0.80, inclusive
])
def test_2_2_band_edges(en_chars, expected):
    """Exercises the COMPARISONS, not the constants. Asserting
    MIXED_LO == 0.20 catches a literal edit and nothing else — >= becoming >
    survives it entirely."""
    assert detect([blk(EN, en_chars), blk(NO, 1000 - en_chars)])["doc_lang"] == expected


@pytest.mark.parametrize("other_chars,expected", [(500, "no"), (501, "other")])
def test_2_3_other_dominance_boundary(other_chars, expected):
    """OTHER_DOMINANCE = 0.50, strictly greater. At 0.0 a single foreign block
    would make a whole Norwegian ad `other`; at 0.99 the Polish ads come back
    as Norwegian."""
    assert detect([blk(PL, other_chars),
                   blk(NO, 1000 - other_chars)])["doc_lang"] == expected


def test_2_4_char_weighted_not_block_counted():
    """Forged weights: under block-counting both cases tie at 1-1."""
    assert detect([blk(EN, 1000), blk(NO, 10)])["doc_lang"] == "en"
    assert detect([blk(EN, 10), blk(NO, 1000)])["doc_lang"] == "no"


def test_2_5_share_denominator_excludes_other():
    """An ad that is 40% Polish, 40% Norwegian, 20% English is Norwegian-
    dominant among the languages the band judges. Dividing by the TOTAL would
    make the English share 0.20 and flip the verdict."""
    r = detect([blk(PL, 400), blk(NO, 400), blk(EN, 200)])
    assert r["doc_lang"] == "mixed"          # 200/(400+200) = 0.33
    assert r["lang_mix"]["other"] == pytest.approx(0.4)


# ── 3 · boilerplate ─────────────────────────────────────────────────────────

def test_3_1_boilerplate_stripped_from_a_long_ad():
    assert detect(blocks(NO, NO, NO, GDPR_EN))["doc_lang"] == "no"


def test_3_2_boilerplate_stripped_from_a_SHORT_ad():
    """The case a pure ratio gets wrong: without stripping this is ~0.57 English."""
    assert detect(blocks(NO, GDPR_EN))["doc_lang"] == "no"


def test_3_3_a_genuinely_english_section_is_NOT_stripped():
    """Exact, not a disjunction. `in ("mixed","en")` would let MIXED_HI drop to
    0.6, or the Norwegian block be dropped entirely, pass silently."""
    assert detect([blk(NO, 300), blk(EN, 700)])["doc_lang"] == "mixed"


def test_3_4b_boilerplate_does_not_downgrade_a_GENUINELY_bilingual_ad():
    """Stripping must act on the BLOCK, not on the verdict. Applying it to the
    verdict ("any boilerplate present -> downgrade mixed to no") passes every
    other boilerplate test, and hides a real bilingual ad from the exact seeker
    it targets."""
    r = detect([blk(NO, 400), blk(EN, 400), blk(GDPR_EN, 200)])
    assert r["doc_lang"] == "mixed"


def test_3_4_all_boilerplate_is_unknown_not_english():
    assert detect(blocks(GDPR_EN))["doc_lang"] == "unknown"


@pytest.mark.parametrize("phrase", [
    "we process your personal data", "see our privacy policy", "compliant with GDPR",
    "an equal opportunity employer", "regardless of race or gender",
    "powered by Webcruiter", "our cookie policy", "the terms of service"])
def test_3_5_every_boilerplate_alternative_matches(phrase):
    """Dropping any single alternative from the regex would otherwise survive."""
    assert is_boilerplate(phrase) is True


def test_3_6_ordinary_text_is_not_boilerplate():
    assert is_boilerplate(NO) is False and is_boilerplate(EN) is False


# ── 4 · reporting fields ────────────────────────────────────────────────────

def test_4_1_confidence_needs_two_scored_blocks():
    assert detect([blk(NO, 500)])["confidence"] == "low"
    assert detect([blk(NO, 500), blk(NO, 500)])["confidence"] == "high"


def test_4_2_is_bilingual_only_when_mixed():
    """`mix["en"] > 0` would mark a 99%-Norwegian ad bilingual."""
    assert detect([blk(NO, 990), blk(EN, 10)])["is_bilingual"] is False
    assert detect([blk(NO, 500), blk(EN, 500)])["is_bilingual"] is True


def test_4_3_detected_other_is_the_heaviest_not_the_first():
    de = "Wir suchen einen erfahrenen Mitarbeiter für unser Team in Oslo zum nächstmöglichen Zeitpunkt."
    r = detect([blk(de, 100), blk(PL, 900)])
    assert r["detected_other"] == "pl", "must be argmax by char mass, not first seen"


def test_4_4_lang_mix_sums_to_one_even_when_blocks_are_filtered():
    """The denominator must be SCORED CHARACTERS, not len(blocks) or n_scored.
    Dividing by a block count deflates mix["other"], letting a Polish ad with
    filtered short blocks slide past the dominance branch into the en/no share
    — which reinstates the original bug."""
    b = [blk(NO, 600), blk(EN, 400), blk(GDPR_EN, 300), blk("Oslo", 4)]
    mix = detect(b)["lang_mix"]
    assert sum(mix.values()) == pytest.approx(1.0)
    assert mix["no"] == pytest.approx(0.6), "weights are scored chars, not block counts"


def test_4_5_order_independent():
    """Real determinism. `detect(b) == detect(b)` on the same object is true of
    any function without a side-effecting counter."""
    b = [blk(NO, 600), blk(EN, 400)]
    assert detect(b) == detect(list(reversed(b)))


def test_4_6_unclassifiable_blocks_are_not_counted_as_norwegian(monkeypatch):
    """If lingua abstains, adding those chars to the Norwegian mass would push
    ads toward `no` — which means HIDDEN. Same failure class as the restricted
    detector, by a different route."""
    monkeypatch.setattr(langid, "_classify", lambda t: (None, None))
    r = detect([blk(NO, 500), blk(EN, 500)])
    assert r["doc_lang"] == "unknown" and r["n_scored"] == 0


def test_4_7_danish_is_other_not_norwegian():
    """Bokmål DESCENDS from Danish, so folding Danish into NORWEGIAN is a
    tempting "they're the same anyway" edit. No Danish fixture existed, so it
    passed every test while silently hiding real Danish ads."""
    da = ("Vi søger en dygtig medarbejder til vores team i København. Du kommer "
          "til at arbejde med daglig drift og opfølgning på kunder.")
    assert langid._classify(da)[0] == "other"


# ── 5 · the accessibility contract ──────────────────────────────────────────

@pytest.mark.parametrize("doc_lang,expected", [
    ("other", True),     # a Polish ad is not kept from you by a NORWEGIAN requirement
    ("unknown", True),   # a false hide is invisible; a false show costs one click
    ("en", True), ("mixed", True),
    ("no", False),       # 95% of the corpus; silence correlates negatively
])
def test_5_1_unstated_accessibility_by_doc_lang(doc_lang, expected):
    from finn_smart_search.understanding.census_prompt import derive_english_accessible
    assert derive_english_accessible(
        {"norwegian_requirement_level": "unstated",
         "stated_working_language": "unstated"}, doc_lang) is expected


def test_5_2_a_requirement_blocks_regardless_of_doc_lang():
    """`other` relaxes the SILENCE fallback, never an explicit requirement."""
    from finn_smart_search.understanding.census_prompt import derive_english_accessible
    for dl in ("no", "en", "mixed", "other", "unknown"):
        assert derive_english_accessible(
            {"norwegian_requirement_level": "professional",
             "stated_working_language": "unstated"}, dl) is False
