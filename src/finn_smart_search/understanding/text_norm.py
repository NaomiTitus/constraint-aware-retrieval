"""The single text-normalisation implementation.

Both sides of evidence-span validation must normalise identically: the stored
`description_text` is produced by html_clean, and spans returned by the LLM are
checked against it. Two divergent implementations silently reject correct spans,
so there is exactly one here and everything imports it.
"""
from __future__ import annotations

import re
import unicodedata

# Characters that survive HTML extraction and would otherwise break matching.
_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'",          # curly single quotes
    "“": '"', "”": '"',          # curly double quotes
    " ": " ",                          # non-breaking space
    " ": " ", " ": " ",          # thin / narrow no-break space
    "­": "",                           # soft hyphen
    "–": "-", "—": "-",          # en / em dash
})

_WS = re.compile(r"\s+")


def normalise(s: str | None) -> str:
    """NFKC + punctuation/whitespace unification. Idempotent."""
    if not s:
        return ""
    return _WS.sub(" ", unicodedata.normalize("NFKC", s).translate(_TRANSLATE)).strip()
