"""HTML -> ordered blocks -> text.

Blocks (not a flattened string) because ads are frequently bilingual: a Norwegian
"Om stillingen" section followed by an English "About the role". Per-block language
ID needs that structure, and evidence spans need to be locatable.
"""
from __future__ import annotations

import re
import unicodedata

from selectolax.lexbor import LexborHTMLParser

BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "div"}
_WS = re.compile(r"[ \t   ]+")
_NL = re.compile(r"\n{3,}")


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace(" ", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    return _WS.sub(" ", s).strip()


def to_blocks(html: str | None) -> list[dict]:
    """Ordered content blocks. Nested containers are skipped when a child already emits."""
    if not html:
        return []
    tree = LexborHTMLParser(html)
    for bad in tree.css("script, style, noscript"):
        bad.decompose()

    blocks: list[dict] = []
    seen: set[str] = set()
    for node in tree.css(", ".join(BLOCK_TAGS)):
        # skip containers whose text is already covered by a descendant block
        if node.tag == "div" and node.css_first(", ".join(BLOCK_TAGS - {"div"})):
            continue
        txt = normalise(node.text(separator=" ", strip=True) or "")
        if len(txt) < 2:
            continue
        key = txt[:160]
        if key in seen:
            continue
        seen.add(key)
        blocks.append({"index": len(blocks), "tag": node.tag, "text": txt, "n_chars": len(txt)})

    if not blocks:  # plain-text ad with no markup
        txt = normalise(tree.text(separator="\n", strip=True) or "")
        if txt:
            blocks = [{"index": i, "tag": "p", "text": t, "n_chars": len(t)}
                      for i, t in enumerate(x for x in txt.split("\n") if x.strip())]
    return blocks


def blocks_to_text(blocks: list[dict]) -> str:
    return _NL.sub("\n\n", "\n".join(b["text"] for b in blocks)).strip()


def clean(html: str | None) -> tuple[list[dict], str]:
    b = to_blocks(html)
    return b, blocks_to_text(b)
