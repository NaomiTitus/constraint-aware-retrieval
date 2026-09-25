"""The grounding helpers must actually reflect production (STANDARDS.md § 3.0).

A helper whose job is to supply real shapes is worthless if nothing checks that
it does — that would be the same self-agreement it exists to prevent. These
tests assert the helpers carry the specific properties that the seven documented
bugs turned on.
"""
import pytest

from tests.conftest import (golden_file, real_blocks, real_text,
                            recorded_batch_results, requires_corpus)

pytestmark = pytest.mark.unit


def test_real_text_is_newline_joined_blocks():
    """THE property. Every early span test used single-line prose, so the
    production join was never under test and the validator destroyed it."""
    texts = real_text()
    assert texts, "fixture yielded nothing"
    multi = [t for t in texts if "\n" in t]
    assert len(multi) > 40, f"only {len(multi)} of {len(texts)} texts have blocks"


def test_real_blocks_include_bullets_without_terminal_punctuation():
    """The second half of the same bug: Norwegian bullets rarely end in a full
    stop, which is why walking back to the previous block's last character
    rejected legitimate quotes."""
    blocks = real_blocks()
    unterminated = [b for b in blocks if b and b[-1] not in ".!?:;"]
    assert len(unterminated) > 500, f"only {len(unterminated)} unterminated of {len(blocks)}"


def test_golden_file_is_read_from_disk_not_constructed():
    g = golden_file()
    assert len(g) == 44
    assert all("expected" in x and "uuid" in x for x in g)
    # the shape a hand-built fixture reliably gets wrong: optional keys
    assert any("annotated_accessible" in x or "accessibility_note" in x for x in g) is False, \
        "no override currently recorded — if one is added, scoring must read it"


def test_recorded_batch_results_are_real_api_shape():
    r = recorded_batch_results()
    assert r, "no recorded results"
    body = str(r)
    for token in ("custom_id", "content", "usage"):
        assert token in body, f"recorded batch results lack {token!r}"


def test_the_golden_set_does_NOT_ground_surface_shape():
    """Pins the caveat in STANDARDS § 3.0, so nobody later 'simplifies' the
    standard to "just use the golden set".

    The 44 ads are stratified for LABEL diversity. The three glyphs that caused
    the 55%-demotion bug appear nowhere in them, so grounding span tests here
    would have reproduced the bug exactly. Surface shape must be grounded
    distributionally, over the corpus."""
    import json

    from finn_smart_search.understanding.html_clean import clean

    # use the committed HTML fixture as the stand-in available without the DB
    ads = json.loads((__import__("pathlib").Path("tests/fixtures/ads_sample_71.json")
                      ).read_text(encoding="utf-8"))
    leading = set()
    for a in ads:
        blocks, _ = clean(a.get("description"))
        for b in blocks:
            t = b["text"].strip()
            if t and not t[0].isalnum():
                leading.add(t[0])
    for glyph, n_corpus in [("·", 1930), ("●", 141), ("", 29)]:
        assert glyph not in leading, (
            f"{glyph!r} now appears in the committed sample; it leads {n_corpus} "
            "corpus blocks. Update STANDARDS § 3.0 — the caveat's evidence changed."
        )
