"""HELD-OUT probe: does the connective rule generalise, or was it fitted to 2 ads?

WHY. census-v4 -> v5 -> v6 were each tuned against the same 44 golden ads. Four
iterations against one set means the 44-ad numbers are no longer an unbiased
estimate of anything. This probe measures the one distinction those iterations
were about, on ads the prompt has never seen.

THE ORACLE. For this narrow question the label is deterministic from the text,
so no hand-labelling is needed:

  DISJUNCTION  "X eller engelsk", no conjunction present
      -> any ONE language suffices, English is offered
      -> level MUST be either_norwegian_or_english        (strong expectation)

  CONJUNCTION  "X og engelsk", no disjunction present
      -> every language listed is demanded
      -> level MUST NOT be either_norwegian_or_english    (weaker, but the
         exact confusion v5 introduced)

The conjunction expectation is deliberately the weaker of the two. "norsk og
engelsk er en fordel" is a conjunction that is nonetheless accessible via
`desirable`, so asserting NOT-accessible would make the oracle wrong. Asserting
only that a conjunction is not read as a disjunction is sound in every case.

Ads carrying BOTH shapes are excluded — the oracle cannot adjudicate them.
Golden-set ads are excluded — that is the whole point.

## RUN 1 AUDIT (seed 20260924, census-v6) — THE ORACLE WAS WRONG, NOT THE MODEL

Raw result: disjunction 11/20, conjunction 20/20. Reading all nine
"disagreements" against the ad text reversed seven of them:

  MODEL RIGHT, ORACLE WRONG — 7 of 9
    Two distinct blind spots, both now excluded below.

    (a) APPLICATION-DOCUMENT LANGUAGE, 4 ads. Academic advertisements say
        "Dokumentene må være på norsk/skandinavisk eller engelsk" or "All
        dokumentasjon må foreligge på et skandinavisk språk eller engelsk" —
        a statement about what language your APPLICATION may be written in,
        not about the job. One of these (0f00816e, UiA postdoc) says in the
        same ad "Arbeidsspråket ved Universitetet i Agder er norsk"; the model
        returned scandinavian_accepted with working_language=norwegian and
        ignored the document clause entirely. It was right and the regex was
        not. This is the strongest available evidence that `application_language`
        belongs in the schema as its own field.

    (b) SCHOOL SUBJECTS, 3 ads. "undervise i fagene norsk, engelsk og
        matematikk", "basisfagene (norsk, matematikk eller engelsk)" — a list
        of subjects TAUGHT, not languages required. "matematikk" inside the
        disjunction is the giveaway. The model read these as the Norwegian bar
        stated elsewhere in the ad (professional) or as silence.

  MODEL WRONG — 1 of 9
    df60fb97, "· Behersker norsk eller engelsk" — the canonical disjunction,
    the exact sentence few-shot 2 demonstrates, returned `unstated` with no
    spans at all. A real miss, recorded rather than explained away.

  WRONG FOR AN UNRELATED REASON — 1 of 9
    af3aea61 requires "norskprøve bestått A2 skriftlig og B1 muntlig" and so is
    `certified`; the model said `unstated`. The oracle's disjunction match was
    spurious boilerplate. Counted against the model, but not as a connective
    error.

  TRUE held-out accuracy on the connective distinction:
    disjunction 18/20 (90%)   conjunction 20/20 (100%)   overall 38/40 (95%)

The lesson is about the oracle, not the score: a regex that finds "X eller
engelsk" cannot tell a job requirement from the language your CV may be in, and
the second is common in exactly the ads most likely to be English-friendly. The
exclusions below encode that, and the same confusion is a live risk for the
BM25 channel later — a keyword surface would make the identical mistake.
"""
import json
import random
import re
import sys
import time

sys.path.insert(0, "src")
from finn_smart_search.ingest import anthropic_client as ac, store
from finn_smart_search.understanding import census
from finn_smart_search.understanding.census_prompt import PROMPT_VERSION

N_PER_GROUP = 20
SEED = int(__import__("os").environ.get("PROBE_SEED", 20260924))

L = r"(norsk\w*|skandinavisk\w*|nordisk\w*|svensk\w*|dansk\w*|scandinavian)"
E = r"(engelsk\w*|english)"
DISJ = re.compile(rf"\b{L}\b[^.\n]{{0,30}}\b(eller|or)\b[^.\n]{{0,30}}\b{E}\b"
                  rf"|\b{E}\b[^.\n]{{0,30}}\b(eller|or)\b[^.\n]{{0,30}}\b{L}\b", re.I)
CONJ = re.compile(rf"\b{L}\b[^.\n]{{0,30}}\b(og|and)\b[^.\n]{{0,30}}\b{E}\b"
                  rf"|\b{E}\b[^.\n]{{0,30}}\b(og|and)\b[^.\n]{{0,30}}\b{L}\b", re.I)

# Contexts the oracle cannot adjudicate, learned from the run-1 audit above.
# Applied per MATCHED SENTENCE, not per ad: an ad may legitimately contain a
# real requirement AND a document-language clause.
DOC_LANG = re.compile(
    r"dokument|vedlegg|s[øo]knad|vitnem[åa]l|attest|cv\b|publikasjon|"
    r"application|enclosure|diploma|certificate", re.I)
SUBJECT = re.compile(
    r"\b(matematikk|matte|naturfag|samfunnsfag|kroppsøving|musikk|kunst|"
    r"historie|fysikk|kjemi|biologi|basisfag|fagene|undervise|undervisning)\b", re.I)


def adjudicable(text: str, pat: re.Pattern) -> bool:
    """True when at least one match sits in a sentence the oracle can judge."""
    for m in pat.finditer(text or ""):
        lo = text.rfind("\n", 0, m.start()) + 1
        hi = text.find("\n", m.end())
        sent = text[lo:hi if hi != -1 else len(text)]
        if not DOC_LANG.search(sent) and not SUBJECT.search(sent):
            return True
    return False


golden = {g["uuid"] for g in json.load(open("eval/golden_set.json"))}
con = store.connect("data/ads.duckdb")
rows = con.execute("""SELECT a.uuid, a.title, a.description_text, l.doc_lang
                      FROM ads a JOIN ad_language l USING (uuid)
                      WHERE a.n_chars > 0""").fetchall()

disj, conj = [], []
for uuid, title, text, lang in rows:
    if uuid in golden:
        continue
    d = adjudicable(text, DISJ)
    c = adjudicable(text, CONJ)
    if d and not c:
        disj.append((uuid, title, text, lang))
    elif c and not d:
        conj.append((uuid, title, text, lang))

rng = random.Random(SEED)
rng.shuffle(disj); rng.shuffle(conj)
disj, conj = disj[:N_PER_GROUP], conj[:N_PER_GROUP]

ads = [{"uuid": u, "title": t, "description_text": x, "doc_lang": l}
       for u, t, x, l in disj + conj]
want = {u: "disjunction" for u, *_ in disj} | {u: "conjunction" for u, *_ in conj}

print(f"prompt {PROMPT_VERSION}  |  HELD-OUT connective probe")
print(f"  disjunction group {len(disj)}   conjunction group {len(conj)}"
      f"   (golden ads excluded)\n")

client = ac.AnthropicBatchClient()
t0 = time.time()
out = census.run(ads, client, con, poll_seconds=10, max_demotion_rate=1.0)
print(f"census: {time.time()-t0:.0f}s  calls={out['calls']}  "
      f"demoted={out['demoted']}  spend=${out['spend_usd']:.3f}\n")

ACC = "either_norwegian_or_english"
res = {"disjunction": [0, 0], "conjunction": [0, 0]}   # [correct, n]
wrong = []
for uuid, f in out["facets"].items():
    grp = want.get(uuid)
    if grp is None:
        continue
    lvl = f.get("norwegian_requirement_level")
    ok = (lvl == ACC) if grp == "disjunction" else (lvl != ACC)
    res[grp][0] += ok; res[grp][1] += 1
    if not ok:
        m = (DISJ if grp == "disjunction" else CONJ).search(
            dict((u, x) for u, _t, x, _l in disj + conj)[uuid] or "")
        wrong.append((grp, uuid, lvl, m.group(0)[:90] if m else ""))

print("HELD-OUT RESULT")
for grp, (c, n) in res.items():
    exp = f"== {ACC}" if grp == "disjunction" else f"!= {ACC}"
    print(f"  {grp:12s} {c:2d}/{n:2d}  {c/max(n,1):5.0%}   (expect level {exp})")

if wrong:
    print(f"\n{len(wrong)} DISAGREEMENTS with the oracle")
    for grp, uuid, lvl, span in wrong:
        print(f"  [{grp:11s}] {uuid[:8]}  got {lvl}")
        print(f"                 {span!r}")
else:
    print("\nno disagreements")

json.dump({"prompt_version": PROMPT_VERSION, "seed": SEED, "result": res,
           "disagreements": wrong}, open("reports/probe_connective.json", "w"), indent=1)
