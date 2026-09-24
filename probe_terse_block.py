"""Does the model miss a language requirement stated as a TERSE STANDALONE BLOCK?

The glyph-bullet probe measured 12/22 (55%) against a 95% general-population
rate, every miss returning `unstated` with no evidence span. But that probe's
filter required a bullet character in the BLOCK TEXT, and html_clean takes block
text from <li> elements — so a properly marked-up list yields identical terse
blocks with NO glyph. The glyph is present only when the advertiser typed it.

Population, measured: a disjunction stated as a standalone block under 70 chars
occurs in 289 ads. Only 20 of them carry a typed glyph. The 22-ad probe
therefore measured 7% of the affected population and the glyph was very likely
incidental — the variable that matters is TERSENESS and list context, not the
character.

If the no-glyph 269 fail at the same ~45% rate, roughly 130 ads are wrongly
hidden from English speakers — the same order as the 364-ad Nordic-disjunction
finding that justified census-v6.

Sampled, not exhaustive: 30 ads, ~$0.045, enough to separate 45% from 5%.
"""
import json
import random
import re
import sys
import time

sys.path.insert(0, "src")
from finn_smart_search.ingest import anthropic_client as ac, store
from finn_smart_search.understanding import census
from finn_smart_search.understanding.census_prompt import ACCESSIBLE_LEVELS, PROMPT_VERSION

N = 30
SEED = 20260926

L = r"(norsk\w*|skandinavisk\w*|nordisk\w*)"
E = r"(engelsk\w*|english)"
DISJ = re.compile(rf"\b{L}\b[^.\n]{{0,30}}\b(eller|or)\b[^.\n]{{0,30}}\b{E}\b"
                  rf"|\b{E}\b[^.\n]{{0,30}}\b(eller|or)\b[^.\n]{{0,30}}\b{L}\b", re.I)
CONJ = re.compile(rf"\b{L}\b[^.\n]{{0,30}}\b(og|and)\b[^.\n]{{0,30}}\b{E}\b"
                  rf"|\b{E}\b[^.\n]{{0,30}}\b(og|and)\b[^.\n]{{0,30}}\b{L}\b", re.I)
GLYPH = re.compile(r"^\s*[·●•*\-–⁠\s]*[·●•*\-–]")
DOC = re.compile(r"dokument|document|vedlegg|s[øo]knad|vitnem[åa]l|attest|submit", re.I)
SUBJ = re.compile(r"\b(matematikk|naturfag|basisfag|fagene|undervise|undervisning)\b", re.I)

con = store.connect("data/ads.duckdb")
pool = []
for uuid, title, text, lang in con.execute(
        """SELECT a.uuid, a.title, a.description_text, l.doc_lang
           FROM ads a JOIN ad_language l USING (uuid) WHERE a.n_chars > 0""").fetchall():
    blocks = (text or "").split("\n")
    hit = [b for b in blocks if DISJ.search(b) and not DOC.search(b) and not SUBJ.search(b)]
    if not hit or CONJ.search(text or ""):
        continue
    terse = [b for b in hit if len(b.strip()) < 70 and not GLYPH.match(b)]
    if terse:
        pool.append({"uuid": uuid, "title": title, "description_text": text,
                     "doc_lang": lang, "_line": terse[0].strip()[:80]})

rng = random.Random(SEED); rng.shuffle(pool)
ads = pool[:N]
print(f"prompt {PROMPT_VERSION}  |  TERSE no-glyph disjunction blocks")
print(f"  population {len(pool)}   sampled {len(ads)}\n")

client = ac.AnthropicBatchClient()
t0 = time.time()
out = census.run([{k: v for k, v in a.items() if k != "_line"} for a in ads],
                 client, con, poll_seconds=10, max_demotion_rate=1.0)
print(f"census: {time.time()-t0:.0f}s  calls={out['calls']}  "
      f"cache_hits={out['cache_hits']}  spend=${out['spend_usd']:.3f}\n")

ACC = "either_norwegian_or_english"
ok = 0; no_span = 0; misses = []
for a in ads:
    f = out["facets"].get(a["uuid"], {})
    lvl = f.get("norwegian_requirement_level")
    if lvl == ACC:
        ok += 1
    else:
        if not f.get("evidence_spans"):
            no_span += 1
        misses.append((a["uuid"], lvl, lvl in ACCESSIBLE_LEVELS, a["_line"]))

n = len(ads)
print("RESULT")
print(f"  correct                        {ok}/{n}  {ok/n:.0%}")
print(f"  glyph subpopulation (all 22)   12/22  55%")
print(f"  general disjunction population 19/20  95%")
print(f"  of {len(misses)} misses, {no_span} returned NO evidence span")
hidden = sum(1 for *_x, acc, _l in [(m[0], m[1], m[2], m[3]) for m in misses] if not acc)
print(f"  wrongly NOT accessible         {hidden}/{n}")
print(f"\n  -> population {len(pool)} ads; at this rate "
      f"~{round(len(pool)*(n-ok)/n)} ads wrongly hidden")
if misses:
    print("\nMISSES")
    for uuid, lvl, acc, ln in misses:
        print(f"  {uuid[:8]}  {str(lvl):28s} accessible={acc}\n            {ln!r}")
json.dump({"prompt_version": PROMPT_VERSION, "population": len(pool), "n": n,
           "correct": ok, "no_span": no_span, "misses": misses},
          open("reports/probe_terse_block.json", "w"), indent=1)
