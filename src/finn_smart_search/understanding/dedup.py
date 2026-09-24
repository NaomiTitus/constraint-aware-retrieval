"""Near-duplicate detection over ad bodies.

Three distinct dedup problems, which need different answers:

  1. REVISION dedup - the same ad appearing many times in the feed as it is
     edited. Solved at ingest by keeping max(sist_endret) per uuid.
     220,988 listing entries -> 59,830 distinct ads.

  2. CONTENT dedup - different uuids carrying substantially the same text,
     because chains and agencies post one template per location. Solved here.
     Purpose: extract once per cluster (saves spend) and stop a single template
     error becoming 44 rows that look like consistent signal.

  3. RESULT DIVERSITY - whether 44 near-identical ads should all appear in one
     result page. That is a RANKING concern (MMR / per-employer capping), not
     an ETL one, and is deliberately not solved here.

A first attempt keyed on `title + first 400 chars` performed WORSE than plain
exact hashing (157 vs 320 redundant ads), because chain stores vary the title
per location - "KIWI Mørkvedvegen...", "KIWI Frognerveien..." - while the body
is identical. Including the title split exactly the clusters we wanted merged.
MinHash over body shingles fixes it: 456 redundant (4.5%), largest cluster 44.
"""
from __future__ import annotations

import random
import re
import unicodedata
from collections import defaultdict

SHINGLE_WORDS = 5
N_PERMS = 32

_rng = random.Random(20260924)
_PERMS = [(_rng.getrandbits(32) | 1, _rng.getrandbits(32)) for _ in range(N_PERMS)]
_MASK = 0xFFFFFFFF


def _normalise(text: str) -> str:
    """Strip the variation that distinguishes sibling ads: digits, case, spacing."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = re.sub(r"\d+", "#", t)
    return re.sub(r"[^\w#æøå ]+", " ", t)


def signature(body: str) -> tuple[int, ...]:
    """MinHash signature over word shingles of the BODY ONLY (never the title)."""
    words = _normalise(body).split()
    if len(words) < SHINGLE_WORDS:
        return (hash(" ".join(words)) & _MASK,) * N_PERMS
    shingles = {
        hash(" ".join(words[i:i + SHINGLE_WORDS])) & _MASK
        for i in range(len(words) - SHINGLE_WORDS + 1)
    }
    return tuple(min(((s * a + b) & _MASK) for s in shingles) for a, b in _PERMS)


def cluster(ads: list[dict]) -> dict[tuple, list[str]]:
    """Group ads by MinHash signature. Returns {signature: [uuid, ...]}."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for ad in ads:
        groups[signature(ad.get("description_text", ""))].append(ad["uuid"])
    return dict(groups)


def representatives(groups: dict[tuple, list[str]]) -> list[str]:
    """One uuid per cluster - the ads actually sent to the LLM."""
    return [uuids[0] for uuids in groups.values()]
