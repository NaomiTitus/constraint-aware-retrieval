"""Pilot the census on the 44 golden ads — the most informative 44 available,
since they are the only ones with hand-labelled ground truth.

Measures both things a pilot is for: that the pipeline works end to end, and
whether census-v4 produces defensible evidence.
"""
import json
import sys
import time

sys.path.insert(0, "src")
from finn_smart_search.eval import scoring
from finn_smart_search.ingest import anthropic_client as ac, store
from finn_smart_search.understanding import census
from finn_smart_search.understanding.census_prompt import PROMPT_VERSION

golden = json.load(open("eval/golden_set.json"))
con = store.connect("data/ads.duckdb")

ads, gold = [], []
for g in golden:
    row = con.execute("""SELECT a.uuid, a.title, a.description_text, l.doc_lang
                         FROM ads a JOIN ad_language l USING (uuid)
                         WHERE a.uuid = ?""", [g["uuid"]]).fetchone()
    if not row:
        print(f"  !! golden #{g['n']} not found in corpus"); continue
    ads.append({"uuid": row[0], "title": row[1], "description_text": row[2], "doc_lang": row[3]})
    gold.append(dict(g, doc_lang=row[3], source_text=row[2]))

print(f"prompt {PROMPT_VERSION}  |  {len(ads)} golden ads\n")
client = ac.AnthropicBatchClient()
t0 = time.time()
out = census.run(ads, client, con, poll_seconds=10, max_demotion_rate=1.0)
print(f"\ncensus: {time.time()-t0:.0f}s  calls={out['calls']}  cache_hits={out['cache_hits']}  "
      f"demoted={out['demoted']} ({out['demotion_rate']:.0%})  spend=${out['spend_usd']:.3f}")
if out["failed"]:
    print(f"  failed: {out['failed']}")

r = scoring.score(out["facets"], gold)
json.dump({k: v for k, v in r.items() if k != "confusion"} |
          {"confusion": {f"{a}->{b}": c for (a, b), c in r["confusion"].items()}},
          open("reports/pilot_scores.json", "w"), indent=1, default=str)

print(f"\ncoverage {r['coverage']:.0%}   pooled {r['pooled_accuracy']:.1%} "
      f"(caveat applies)   invalid_levels={r['invalid_levels']}")
print("\nPER LEVEL")
for lvl, s in sorted(r["per_level"].items(), key=lambda x: -x[1]["n"]):
    bar = "#" * int(s["accuracy"] * 20)
    print(f"  {lvl:28s} {s['correct']:2d}/{s['n']:2d}  {s['accuracy']:5.0%} {bar}")

a = r["accessibility"]
print(f"\nACCESSIBILITY  (the metric that matters)")
print(f"  hidden wrongly  {a['hidden_wrongly']:2d} of {a['n_accessible']:2d} accessible"
      f"   rate {a['hidden_wrongly_rate']:.0%}  ci95 "
      f"[{a['hidden_wrongly_ci95'][0]:.2f}, {a['hidden_wrongly_ci95'][1]:.2f}]")
print(f"  shown wrongly   {a['shown_wrongly']:2d} of {a['n_blocking']:2d} blocking"
      f"   rate {a['shown_wrongly_rate']:.0%}")

e = r["evidence"]
print(f"\nEVIDENCE SPANS (of {e['n_expected']} expected)")
print(f"  exact {e['exact']}  different {e['different']}  missing {e['missing']}  "
      f"fabricated {e['fabricated']}  spurious {e['spurious']}")
w, au = r["working_language"], r["authorisation"]
print(f"\nworking_language non-default recall "
      f"{w['non_default_correct']}/{w['non_default_total']}")
print(f"authorisation {au['correct']}/{au['n_expected']}  spurious {au['spurious']}")
if r["taxonomy_gap"]:
    print(f"taxonomy gap (derived != annotated): {len(r['taxonomy_gap'])}")

print("\nMISCLASSIFICATIONS")
for (want, got), n in sorted(r["confusion"].items(), key=lambda x: -x[1]):
    if want != got:
        print(f"  {want:28s} -> {got:28s} x{n}")
