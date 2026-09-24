"""Targeted probe: is a disjunction inside a glyph bullet list missed more often?

HYPOTHESIS, from two held-out misses that shared a shape:

    · Behersker norsk eller engelsk      (df60fb97, probe run 1)
    ● Snakker norsk eller engelsk        (367d5967, probe run 2)

Both returned `unstated` with NO evidence spans at all — not a wrong level, no
answer. Both sit as one item among 5-7 personal-quality bullets in a short ad
("Hva vi ser etter"), glyph-prefixed, with no terminal punctuation.

Few-shot 2 demonstrates that exact sentence — "Behersker norsk eller engelsk" —
but renders it as FLOWING PROSE inside a paragraph of full stops. So the
demonstrated shape and the corpus shape differ, which is the same class of gap
that made every earlier span test pass while production failed.

WHY MEASURE BEFORE FIXING. This shape is only 26 of the 688 ads carrying a
disjunction (4%), i.e. 0.26% of the corpus. Two misses both landing in a 4%
subpopulation is suggestive but n=2. Rewriting the few-shot on that evidence is
the "extractor rabbit hole" the plan warns about (Risk 3: measure rather than
polish). So: run the whole 26-ad subpopulation, ~$0.04, and compare its miss
rate against the 5% measured on the general disjunction population.

  If the rate is materially higher, reformatting few-shot 2 is justified.
  If it is ~5%, the hypothesis is dead and gets recorded as dead.
"""
import json
import re
import sys
import time

sys.path.insert(0, "src")
from finn_smart_search.ingest import anthropic_client as ac, store
from finn_smart_search.understanding import census
from finn_smart_search.understanding.census_prompt import ACCESSIBLE_LEVELS, PROMPT_VERSION

L = r"(norsk\w*|skandinavisk\w*|nordisk\w*)"
E = r"(engelsk\w*|english)"
DISJ = re.compile(rf"\b{L}\b[^.\n]{{0,30}}\b(eller|or)\b[^.\n]{{0,30}}\b{E}\b"
                  rf"|\b{E}\b[^.\n]{{0,30}}\b(eller|or)\b[^.\n]{{0,30}}\b{L}\b", re.I)
CONJ = re.compile(rf"\b{L}\b[^.\n]{{0,30}}\b(og|and)\b[^.\n]{{0,30}}\b{E}\b"
                  rf"|\b{E}\b[^.\n]{{0,30}}\b(og|and)\b[^.\n]{{0,30}}\b{L}\b", re.I)
GLYPH = re.compile(r"^\s*[·●•*\-–]\s*")
DOC = re.compile(r"dokument|document|vedlegg|s[øo]knad|vitnem[åa]l|attest|submit", re.I)
SUBJ = re.compile(r"\b(matematikk|naturfag|basisfag|fagene|undervise|undervisning)\b", re.I)

con = store.connect("data/ads.duckdb")
rows = con.execute("""SELECT a.uuid, a.title, a.description_text, l.doc_lang
                      FROM ads a JOIN ad_language l USING (uuid)
                      WHERE a.n_chars > 0""").fetchall()

ads = []
for uuid, title, text, lang in rows:
    blocks = (text or "").split("\n")
    hit = [b for b in blocks
           if DISJ.search(b) and not DOC.search(b) and not SUBJ.search(b)]
    if not hit or CONJ.search(text or ""):
        continue
    if any(GLYPH.match(b) for b in hit) and sum(1 for b in blocks if GLYPH.match(b)) >= 4:
        ads.append({"uuid": uuid, "title": title, "description_text": text,
                    "doc_lang": lang, "_line": hit[0][:90]})

print(f"prompt {PROMPT_VERSION}  |  glyph-bullet disjunction subpopulation")
print(f"  {len(ads)} ads — the whole subpopulation, not a sample\n")

client = ac.AnthropicBatchClient()
t0 = time.time()
out = census.run([{k: v for k, v in a.items() if k != "_line"} for a in ads],
                 client, con, poll_seconds=10, max_demotion_rate=1.0)
print(f"census: {time.time()-t0:.0f}s  calls={out['calls']}  "
      f"cache_hits={out['cache_hits']}  spend=${out['spend_usd']:.3f}\n")

ACC = "either_norwegian_or_english"
line = {a["uuid"]: a["_line"] for a in ads}
ok = miss = no_span = 0
rows_out = []
for a in ads:
    f = out["facets"].get(a["uuid"], {})
    lvl = f.get("norwegian_requirement_level")
    good = lvl == ACC
    ok += good
    if not good:
        miss += 1
        if not f.get("evidence_spans"):
            no_span += 1
        rows_out.append((a["uuid"], lvl, lvl in ACCESSIBLE_LEVELS, line[a["uuid"]]))

n = len(ads)
print("RESULT")
print(f"  correct level ({ACC})   {ok}/{n}  {ok/max(n,1):.0%}")
print(f"  general disjunction population, measured  19/20  95%")
print(f"  of the {miss} misses, {no_span} returned NO evidence span at all")
if rows_out:
    print("\nMISSES")
    for uuid, lvl, acc, ln in rows_out:
        print(f"  {uuid[:8]}  {lvl:28s} accessible={acc}")
        print(f"            {ln!r}")
json.dump({"prompt_version": PROMPT_VERSION, "n": n, "correct": ok,
           "misses": rows_out}, open("reports/probe_bullet_shape.json", "w"), indent=1)
