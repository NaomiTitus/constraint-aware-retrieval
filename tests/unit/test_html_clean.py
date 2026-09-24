"""Group A — html_clean: leaf-only traversal, no block dedup.

Scenario table agreed 2026-09-24. Two behaviour changes under test:

  1. Emit only blocks with no block DESCENDANT. Norwegian ads are marked up as
     <li><p>text</p></li>, so the previous div-only skip emitted every bullet
     twice and a 160-char prefix dedup silently cleaned up after it.

  2. Mixed content (option b): when a node has BOTH its own text and block
     children, emit its own text as a separate block first. Measured: 272 ads
     (2.68%), 418 nodes, 98,300 characters — including
     "God norsk ferdigheter. Helst bestått norskprøve B1.", a language
     requirement that option (a) would have discarded.

  3. The prefix dedup is DELETED. It existed only to mask (1), and it destroys
     genuinely repeated bullets in 1.4% of ads.
"""
import json

import pytest

from finn_smart_search.understanding.html_clean import blocks_to_text, clean, to_blocks

from tests.conftest import FIXTURES

pytestmark = pytest.mark.unit

# Pinned goldens for the 71-ad fixture. Regenerate DELIBERATELY when behaviour
# changes: a range wide enough to feel safe is a range too wide to catch the
# bugs this module exists to fix.
EXPECTED_TOTAL_BLOCKS = 2317
EXPECTED_TOTAL_CHARS = 189_023


def texts(html):
    return [b["text"] for b in to_blocks(html)]


def tags(html):
    return [b["tag"] for b in to_blocks(html)]


# ── A1–A4 · leaf-only traversal ──────────────────────────────────────────────

def test_a1_li_wrapping_p_emits_one_block():
    """<li><p>A</p></li> is ONE block from the <p>, not two."""
    assert texts("<li><p>Alpha</p></li>") == ["Alpha"]
    assert tags("<li><p>Alpha</p></li>") == ["p"]


def test_a2_td_wrapping_p_emits_one_block():
    assert texts("<table><tr><td><p>Alpha</p></td></tr></table>") == ["Alpha"]


def test_a3_arbitrary_nesting_depth():
    assert texts("<div><div><p>Alpha</p></div></div>") == ["Alpha"]


def test_a4_nested_lists():
    assert texts("<ul><li><ul><li>Alpha</li></ul></li></ul>") == ["Alpha"]


# ── A5–A6 · ordering and repetition ──────────────────────────────────────────

def test_a5_order_preserved():
    assert texts("<p>Alpha</p><p>Beta</p>") == ["Alpha", "Beta"]


def test_a6_genuine_repetition_is_preserved():
    """THE behaviour change. Two separate bullets with identical text stay two
    blocks. 1.4% of ads repeat a bullet, including language requirements."""
    assert texts("<ul><li>Gode norskkunnskaper</li><li>Gode norskkunnskaper</li></ul>") == [
        "Gode norskkunnskaper",
        "Gode norskkunnskaper",
    ]


# ── A13 · mixed content (option b) ───────────────────────────────────────────

def test_a13a_mixed_content_emits_own_text_first():
    """A node with its own text AND block children emits both, in order.
    Text must clear MIN_BLOCK_CHARS (2) or it is not a block at all."""
    b = to_blocks("<li>Krav: <p>Norsk</p></li>")
    assert [x["text"] for x in b] == ["Krav:", "Norsk"]
    assert [x["tag"] for x in b] == ["li", "p"]


def test_a13b_no_own_text_emits_only_the_child():
    assert texts("<li><p>Norsk</p></li>") == ["Norsk"]


def test_a13c_whitespace_only_own_text_is_not_a_block():
    assert texts("<li>   <p>Alpha</p></li>") == ["Alpha"]


def test_a13d_own_text_below_min_own_chars_is_dropped():
    """MIN_OWN_CHARS is 3. Own text of 2 chars clears MIN_BLOCK_CHARS (2) and so
    reaches the own-text threshold — the only length that discriminates. The
    earlier version used ":" (1 char), which emit() rejected first, leaving
    MIN_OWN_CHARS untested at any value from 0 to 2."""
    assert texts("<li>NB <p>Norsk</p></li>") == ["Norsk"]


def test_a13d2_own_text_at_min_own_chars_is_kept():
    assert texts("<li>Krav <p>Norsk</p></li>") == ["Krav", "Norsk"]


def test_a13e_real_case_language_requirement_survives():
    """From the Vy bus-driver ad — few-shot example 5. Option (a) would have
    discarded this sentence and turned the ad from `certified` to `unstated`."""
    html = "<ul><li>God norsk ferdigheter. Helst bestått norskprøve B1.<p>Vy bruker Bussnorsktest</p></li></ul>"
    assert texts(html) == [
        "God norsk ferdigheter. Helst bestått norskprøve B1.",
        "Vy bruker Bussnorsktest",
    ]


# ── A7–A12 · hygiene ─────────────────────────────────────────────────────────

def test_a7_script_style_noscript_removed():
    """Script text must not leak into an ancestor block's OWN text. The naive
    version passes with decompose() deleted, because script/style are not in
    BLOCK_TAGS and emit nothing on their own."""
    assert texts("<div><p>Alpha</p><script>var x=1</script></div>") == ["Alpha"]
    assert texts("<div><p>Alpha</p><style>p{color:red}</style></div>") == ["Alpha"]
    # noscript must not lead: lexbor hoists a leading <noscript> into <head> and
    # empties it, spilling its content into <body> before decompose() can run.
    # No corpus ad contains <noscript>; this is defensive.
    assert texts("<p>Alpha</p><noscript><p>Aktiver JavaScript</p></noscript>") == ["Alpha"]


def test_a8_plain_text_without_markup():
    assert texts("Line one\nLine two") == ["Line one", "Line two"]


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_a9_empty_input_returns_empty_list(bad):
    assert to_blocks(bad) == []


def test_a10_nfkc_and_punctuation_normalised():
    """Equality, not absence. The previous version asserted NBSP was absent from
    input that never contained one, so it passed against an identity normaliser."""
    src = "<p>Gode\u00a0norsk\u00adkunnskaper \u201cja\u201d \u2013 B1</p>"
    assert texts(src) == ['Gode norskkunnskaper "ja" - B1']


def test_a10b_nfkc_folds_ligatures_and_fullwidth():
    """Dropping NFKC while keeping the explicit character maps would let these
    through; the LLM retypes them as ASCII and the span validator then rejects
    a correct quote."""
    assert texts("<p>O\ufb03siell \uff12 norskprøve</p>") == ["Offisiell 2 norskprøve"]


def test_a11_block_index_is_sequential():
    b = to_blocks("<p>Alpha</p><p>Beta</p><p>Gamma</p>")
    assert [x["index"] for x in b] == [0, 1, 2]


def test_a12_blocks_to_text_joins_with_newline():
    """The old second assertion was a tautology: normalise() collapses all
    whitespace, so no block text can contain a newline and "\\n\\n\\n" was
    unreachable. It also passed if blocks_to_text returned "". """
    assert blocks_to_text(to_blocks("<p>Alpha</p><p>Beta</p>")) == "Alpha\nBeta"
    assert blocks_to_text(to_blocks("<p>Alpha</p><p></p><p>Beta</p>")) == "Alpha\nBeta"


def test_clean_returns_blocks_and_text():
    b, t = clean("<p>Alpha</p><p>Beta</p>")
    assert len(b) == 2 and t == "Alpha\nBeta"


# ── new: gaps found by mutation testing ─────────────────────────────────────

LONG = ("Gode norskkunnskaper muntlig og skriftlig kreves for stillingen fordi du skal "
        "kommunisere med kunder og kolleger i det daglige arbeidet hos oss i Norge.")


def test_a6b_identical_long_bullets_both_survive():
    """a6 alone used a 20-char string, so any prefix-keyed dedup with a length
    floor would survive it. LONG exceeds the old 160-char dedup key."""
    assert texts(f"<ul><li>{LONG}</li><li>{LONG}</li></ul>") == [LONG, LONG]


def test_a6c_distinct_bullets_sharing_a_long_prefix_both_survive():
    out = texts(f"<ul><li>{LONG} A1-nivå.</li><li>{LONG} B2-nivå.</li></ul>")
    assert len(out) == 2 and out[0] != out[1]


def test_a13f_own_text_overlapping_child_text_is_intact():
    """String subtraction removed the FIRST occurrence — often the parent's own
    words. Position-based collection fixes it. Zero corpus ads were affected,
    but the failure mode is exactly the one that matters: mangling a language
    requirement."""
    assert texts("<li>Norsk kreves: <p>Norsk</p></li>") == ["Norsk kreves:", "Norsk"]
    assert texts("<div>Norsk<p>Norsk</p></div>") == ["Norsk", "Norsk"]


def test_inline_markup_does_not_glue_words():
    """79% of ads carry inline markup. separator="" would produce
    "Godenorskferdigheter" and destroy every sentence the LLM must quote."""
    assert texts("<p>Gode <strong>norsk</strong> ferdigheter</p>") == ["Gode norsk ferdigheter"]


def test_min_block_chars_boundary():
    """Two-char blocks are real: B1, B2, C1 are CEFR levels and carry the
    language requirement. `>=` becoming `>` would silently delete them."""
    assert texts("<ul><li>A</li><li>B1</li><li>Norsk</li></ul>") == ["B1", "Norsk"]


def test_td_is_a_block_tag():
    """a2 passes with `td` removed, because the inner <p> is selected anyway.
    A table cell with no inner block would vanish entirely."""
    html = "<table><tr><td>Norsk</td><td>Engelsk</td></tr></table>"
    assert texts(html) == ["Norsk", "Engelsk"]
    # Text alone is insufficient: with `td` removed the raw-text fallback still
    # recovers both cells, but tags collapse to "p" and section structure is lost.
    assert tags(html) == ["td", "td"]


def test_headings_keep_their_tag():
    """Heading text is often recoverable as a parent div's own text, so counts
    alone do not catch h1-h6 being dropped — but the tag drives section splits."""
    b = to_blocks("<h2>Krav</h2><p>Norsk</p>")
    assert [x["text"] for x in b] == ["Krav", "Norsk"]
    assert [x["tag"] for x in b] == ["h2", "p"]


def test_index_is_contiguous_when_blocks_are_filtered():
    """Indexing off the css loop instead of len(blocks) yields gaps whenever a
    node is filtered, silently mis-addressing downstream block lookups."""
    assert [x["index"] for x in to_blocks("<p>Alpha</p><p></p><p>Beta</p>")] == [0, 1]


# ── A14 · corpus characterisation ────────────────────────────────────────────

@pytest.mark.unit
def test_a14_corpus_characterisation():
    """Locks behaviour against the 71-ad fixture. Values recorded AFTER the
    leaf-only change; this is a characterisation test, not a specification."""
    ads = json.loads((FIXTURES / "ads_sample_71.json").read_text())
    res = [clean(a.get("description")) for a in ads]
    n_blocks = [len(b) for b, _ in res]
    n_chars = [len(t) for _, t in res]
    assert len(ads) == 71
    assert sum(1 for n in n_blocks if n == 0) == 0, "no ad may yield zero blocks"

    # Pinned goldens, not ranges. The old 500-char window absorbed the entire
    # 98,300-char mixed-content change, so it could not have caught either bug.
    assert sum(n_blocks) == EXPECTED_TOTAL_BLOCKS
    assert sum(n_chars) == EXPECTED_TOTAL_CHARS

    # The fixture must actually exercise the bugs this module fixes.
    html = " ".join(a.get("description") or "" for a in ads)
    assert "<li><p>" in html.replace(" ", ""), "fixture must contain an li>p ad"
    assert any(any(x["tag"] == "li" for x in b) for b, _ in res), "fixture must yield li blocks"
