"""Span validation against the REAL stored text shape.

The pilot found this: 26 of 28 span rejections were `starts_mid_sentence` on
perfectly legitimate quotes, demoting 55% of records against a 15% threshold.

Cause: `description_text` joins blocks with "\\n", but the normaliser used for
matching collapses ALL whitespace, destroying the block boundary. The check
then walks back to the previous BLOCK's last character — and Norwegian ads are
bullet lists whose items rarely end in punctuation.

Every earlier test used single-line text with full stops between sentences, so
the production shape was never exercised.
"""
import pytest

from finn_smart_search.understanding import census_validate as v

pytestmark = pytest.mark.unit

# The real shape: html_clean emits blocks, silver joins them with newlines.
# Every span here carries a language token: a span with none is rejected by a
# different rule (no_language_token), which is correct — evidence must be ABOUT
# language — and would mask what these tests are for.
BULLETS = ("Du må beherske norsk godt\n"
           "Har erfaring med søm, enten fra arbeid eller utdanning\n"
           "Gode norskkunnskaper\n"
           "Engelsk er også nyttig i denne stillingen")


def test_a_whole_block_quote_is_valid_evidence():
    """A bullet with no terminal punctuation, preceded by another such bullet.
    This is the commonest evidence shape in the corpus."""
    ok, why = v._span_ok("Gode norskkunnskaper", BULLETS)
    assert ok, f"rejected a legitimate block quote: {why}"


def test_the_first_block_is_valid_evidence():
    ok, why = v._span_ok("Du må beherske norsk godt", BULLETS)
    assert ok, why


def test_the_last_block_is_valid_evidence():
    ok, why = v._span_ok("Engelsk er også nyttig i denne stillingen", BULLETS)
    assert ok, why


def test_a_fragment_of_a_block_is_still_rejected():
    """The 578-ad trap must still be caught: a block boundary is a boundary,
    but mid-block is not. The fragment carries a language token, so it is the
    BOUNDARY rule that must reject it."""
    ok, why = v._span_ok("beherske norsk godt", BULLETS)
    assert not ok and why == "starts_mid_sentence"


def test_a_span_spanning_two_blocks_is_rejected():
    ok, why = v._span_ok("Gode norskkunnskaper Engelsk er også nyttig", BULLETS)
    assert not ok and why == "not_verbatim", why


def test_sentence_boundaries_still_work_within_a_block():
    text = "Vi søker medarbeider. Gode norskkunnskaper. Oppstart snarest."
    assert v._span_ok("Gode norskkunnskaper.", text)[0]
    assert not v._span_ok("norskkunnskaper. Oppstart", text)[0]


def test_mixed_shape_blocks_and_sentences():
    """Real ads have both: prose blocks containing several sentences, and
    bullet blocks containing none."""
    text = ("Om stillingen\n"
            "Vi søker en medarbeider. Norsk er arbeidsspråket. Oppstart etter avtale.\n"
            "Krav\n"
            "Gode norskkunnskaper")
    assert v._span_ok("Norsk er arbeidsspråket.", text)[0], "sentence inside a prose block"
    assert v._span_ok("Gode norskkunnskaper", text)[0], "whole bullet block"
