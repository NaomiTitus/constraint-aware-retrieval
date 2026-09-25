"""Near-duplicate detection over ad bodies, by exact hash of normalised text.

Three distinct dedup problems, which need different answers:

  1. REVISION dedup — the same ad appearing many times in the feed as it is
     edited. Solved at ingest by keeping max(sist_endret) per uuid:
     220,988 listing entries -> 59,830 distinct ads.

  2. CONTENT dedup — different uuids carrying substantially the same text,
     because chains and agencies post one template per location. Solved here.

  3. RESULT DIVERSITY — whether 44 near-identical ads should all appear on one
     result page. That is a RANKING concern (MMR / per-employer capping), not
     an ETL one, and is deliberately not solved here.

Why exact hashing rather than MinHash (supersedes DECISIONS.md D6). Measured
over the full corpus: raw HTML 249 redundant (2.4%), cleaned text 320 (3.1%),
normalised text 345 (3.4%), MinHash 423 (4.2%). MinHash's extra 78 merges are
legitimate but sit in clusters of 2-6, where error amplification is negligible;
the large clusters that matter (44, 32, 21) are caught identically by every
strategy. Exact hashing is explainable in one sentence, deterministic across
processes, and has zero false merges by construction.

The purpose is NOT cost — deduping saves about $0.12 on a $3 census. It exists
to stop one template mistake becoming 44 corpus rows that look like consistent
signal and survive random-sample evaluation.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .text_norm import normalise

# Strip the variation that distinguishes sibling ads, and nothing else.
# \w under re.UNICODE keeps æøå; stripping them would collapse Norwegian ads
# toward a common key and make clustering catastrophically over-eager.
# Digits are masked so one template per location collapses ("5 stillinger" /
# "7 stillinger"). CEFR levels are the exception: B1 and B2 are DIFFERENT
# language requirements, and merging them would hand two ads one verdict on the
# very attribute this pipeline exists to read. 1,268 corpus ads carry a CEFR
# token. A digit directly preceded by A/B/C is therefore left alone.
# Digits are masked so that two ads differing only by stillingsprosent, salary
# or a date share a signature -- they share their LANGUAGE requirement, which is
# what the census reads. Two exceptions, both measured:
#
#   CEFR letter+digit  B1 vs B2 are different requirements (1,268 ads carry one)
#   BARE DIGIT after a level word  "norskprøve 3", "nivå 2", "trinn 4" -- 68 ads
#       state the level this way, and masking merged two different requirements
#       into one LLM verdict. Confirmed as load-bearing by a real emitted span:
#       "Norsk muntlig og skriftlige ferdigheter tilsvarende nivå 2".
#
# The bare-digit case needs preceding CONTEXT, which a fixed-width lookbehind
# cannot express, so the substitution is a function.
_LEVEL_WORD = re.compile(
    r"(?:niv[åa]|norskpr[øo]ve|trinn|spr[åa]kniv[åa]|level|grade)\W{0,3}$", re.I)
_DIGITS = re.compile(r"(?<![ABCabc])\d+")


def _mask_digit(m: "re.Match") -> str:
    """Keep a digit that states a language level; mask every other."""
    return m.group(0) if _LEVEL_WORD.search(m.string[:m.start()]) else "#"
_NON_WORD = re.compile(r"[^\w#]+", re.UNICODE)

__all__ = ["canonical", "signature", "cluster", "representatives", "fan_out"]


def canonical(body: str | None) -> str:
    """The comparison form: shared normaliser, case-folded, digits masked."""
    text = normalise(body).lower()
    text = _DIGITS.sub(_mask_digit, text)
    return _NON_WORD.sub(" ", text).strip()


def signature(body: str | None) -> str:
    """Stable 64-char hex digest of the canonical form.

    A hex string rather than MinHash's tuple of permutations: serialisable,
    comparable across processes, and with no seed to keep in sync.
    """
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def cluster(ads: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Group ads by signature. Returns {signature: [uuid, ...]} with uuids
    sorted, so cluster contents do not depend on input order."""
    groups: dict[str, list[str]] = defaultdict(list)
    for adv in ads:
        groups[signature(adv.get("description_text"))].append(adv["uuid"])
    return {sig: sorted(uuids) for sig, uuids in groups.items()}


def representatives(groups: Mapping[str, list[str]]) -> list[str]:
    """One uuid per cluster — the ads actually sent to the LLM.

    Deterministic: the lowest uuid in each cluster, returned in sorted order, so
    two runs over the same corpus send exactly the same requests and the cache
    keys line up.
    """
    return sorted(min(uuids) for uuids in groups.values() if uuids)


def fan_out(groups: Mapping[str, list[str]],
            results: Mapping[str, Any]) -> dict[str, Any]:
    """Spread each representative's result to every member of its cluster.

    Raises KeyError if a representative is missing from `results`. Silently
    dropping ads whose representative failed extraction would shrink the corpus
    without anyone noticing.
    """
    out: dict[str, Any] = {}
    for uuids in groups.values():
        if not uuids:
            continue
        rep = min(uuids)
        if rep not in results:
            raise KeyError(f"no result for cluster representative {rep!r}")
        for uuid in uuids:
            out[uuid] = results[rep]
    return out
