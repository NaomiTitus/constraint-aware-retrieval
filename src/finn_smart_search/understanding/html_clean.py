"""HTML -> ordered blocks -> text.

Blocks rather than a flattened string because ads are frequently bilingual: a
Norwegian "Om stillingen" section followed by an English "About the role".
Per-block language ID needs that structure, and evidence spans need to be
locatable.

Two rules, both established by measurement:

  LEAF-ONLY. Emit a node only if it has no block DESCENDANT. Norwegian ads are
  marked up as <li><p>text</p></li>, so a div-only skip emitted every bullet
  twice across 27% of ads, masked by a prefix dedup that also destroyed
  genuinely repeated bullets in 1.4% of ads. There is no block dedup here now:
  removing the cause beats tuning the workaround.

  MIXED CONTENT. When a node has both its own text and block children, emit its
  own text as a separate block first. Measured: 272 ads (2.68%), 418 nodes,
  98,300 characters — including "God norsk ferdigheter. Helst bestått
  norskprøve B1.", a language requirement that discarding would have turned
  from `certified` into `unstated`.
"""
from __future__ import annotations

from selectolax.lexbor import LexborHTMLParser

from .text_norm import normalise

BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "div"}
_SELECTOR = ", ".join(BLOCK_TAGS)

MIN_BLOCK_CHARS = 2      # a block must carry at least this much text
MIN_OWN_CHARS = 3        # mixed-content own text must carry at least this much

__all__ = ["BLOCK_TAGS", "normalise", "to_blocks", "blocks_to_text", "clean"]


def _has_block_descendant(node) -> bool:
    """True if any descendant is a block node.

    Walks children explicitly: selectolax's `node.css()` includes the node
    itself and returns a fresh wrapper object, so an identity check against the
    result never matches and every node looks like it contains a block.
    """
    child = node.child
    while child is not None:
        if child.tag in BLOCK_TAGS:
            return True
        if child.tag != "-text" and _has_block_descendant(child):
            return True
        child = child.next
    return False


def _own_text(node) -> str:
    """Text belonging directly to a node, excluding its block descendants'.

    Collected by POSITION, not by string subtraction. Subtracting a child's text
    with `full.replace(child, " ", 1)` removes the FIRST occurrence, which is
    often the parent's own words: "<li>Norsk kreves: <p>Norsk</p></li>" yielded
    "kreves: Norsk" instead of "Norsk kreves:". Verified identical to the old
    behaviour across all 10,166 corpus ads, so this is a safe swap.
    """
    parts: list[str] = []

    def walk(n):
        while n is not None:
            if n.tag in BLOCK_TAGS:
                pass                       # skip the entire block subtree
            elif n.tag == "-text":
                parts.append(n.text() or "")
            else:
                walk(n.child)              # descend through inline markup
            n = n.next

    walk(node.child)
    return normalise(" ".join(parts))


def to_blocks(html: str | None) -> list[dict]:
    """Ordered content blocks: [{index, tag, text, n_chars}]."""
    if not html or not html.strip():
        return []

    tree = LexborHTMLParser(html)
    for bad in tree.css("script, style, noscript"):
        bad.decompose()

    blocks: list[dict] = []

    def emit(tag: str, text: str) -> None:
        if len(text) >= MIN_BLOCK_CHARS:
            blocks.append({"index": len(blocks), "tag": tag, "text": text,
                           "n_chars": len(text)})

    for node in tree.css(_SELECTOR):
        if _has_block_descendant(node):
            own = _own_text(node)
            if len(own) >= MIN_OWN_CHARS:
                emit(node.tag, own)
            continue
        emit(node.tag, normalise(node.text(separator=" ", strip=True)))

    if not blocks:  # plain-text ad with no usable markup
        # Split on newlines BEFORE normalising: normalise() collapses all
        # whitespace, so normalising first would merge every line into one.
        raw = tree.text(separator="\n", strip=True) or ""
        for line in raw.split("\n"):
            emit("p", normalise(line))
    return blocks


def blocks_to_text(blocks: list[dict]) -> str:
    return "\n".join(b["text"] for b in blocks).strip()


def clean(html: str | None) -> tuple[list[dict], str]:
    b = to_blocks(html)
    return b, blocks_to_text(b)
