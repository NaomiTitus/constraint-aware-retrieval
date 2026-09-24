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


def test_a13d_own_text_under_three_chars_is_dropped():
    """Child text must be >= 2 chars, else the existing min-length rule drops it
    and the case no longer isolates A13d."""
    assert texts("<li>: <p>ABC</p></li>") == ["ABC"]


def test_a13e_real_case_language_requirement_survives():
    """From the Vy bus-driver ad — few-shot example 5. Option (a) would have
    discarded this sentence and turned the ad from `certified` to `unstated`."""
    html = "<ul><li>God norsk ferdigheter. Helst bestått norskprøve B1.<p>Vy bruker Bussnorsktest</p></li></ul>"
    assert "God norsk ferdigheter. Helst bestått norskprøve B1." in texts(html)


# ── A7–A12 · hygiene ─────────────────────────────────────────────────────────

def test_a7_script_and_style_removed():
    assert texts("<script>x=1</script><style>p{}</style><p>Alpha</p>") == ["Alpha"]


def test_a8_plain_text_without_markup():
    assert texts("Line one\nLine two") == ["Line one", "Line two"]


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_a9_empty_input_returns_empty_list(bad):
    assert to_blocks(bad) == []


def test_a10_nfkc_and_punctuation_normalised():
    t = texts("<p>Gode norsk­kunnskaper “ja”</p>")[0]
    assert " " not in t and "­" not in t
    assert '"ja"' in t


def test_a11_block_index_is_sequential():
    b = to_blocks("<p>Alpha</p><p>Beta</p><p>Gamma</p>")
    assert [x["index"] for x in b] == [0, 1, 2]


def test_a12_blocks_to_text_joins_with_newline():
    assert blocks_to_text(to_blocks("<p>Alpha</p><p>Beta</p>")) == "Alpha\nBeta"
    assert "\n\n\n" not in blocks_to_text(to_blocks("<p>Alpha</p><p></p><p>Beta</p>"))


def test_clean_returns_blocks_and_text():
    b, t = clean("<p>Alpha</p><p>Beta</p>")
    assert len(b) == 2 and t == "Alpha\nBeta"


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
    assert 30 <= sum(n_blocks) / len(n_blocks) <= 40
    assert 2500 <= sum(n_chars) / len(n_chars) <= 3000
