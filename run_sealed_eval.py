"""Score census-v8 against the 28 HELD-OUT ads — the first unbiased number.

The 44-ad golden set has driven eight prompt versions and is a regression gate.
These 28 were selected after the prompt was frozen, labelled by a human
reviewing each ad, and NEVER tuned against. Whatever this prints is the number
the README can defend.

Run once. Tuning the prompt against this result would destroy the only clean
measurement the project has, and there would be no way to tell from the outside.
"""
import json
import sys
import time

sys.path.insert(0, "src")
from finn_smart_search.eval import scoring
from finn_smart_search.ingest import anthropic_client as ac, store
from finn_smart_search.understanding import census
from finn_smart_search.understanding.census_prompt import PROMPT_VERSION

gold = json.load(open("data/worksheet/sealed_labels.json"))
con = store.connect("data/ads.duckdb")

ads, golden = [], []
for g in gold:
    row = con.execute("""SELECT a.uuid, a.title, a.description_text, l.doc_lang
                         FROM ads a JOIN ad_language l USING (uuid)
                         WHERE a.uuid = ?""", [g["uuid"]]).fetchone()
    ads.append({"uuid": row[0], "title": row[1],
                "description_text": row[2], "doc_lang": row[3]})
    golden.append(dict(g, doc_lang=row[3], source_text=row[2]))

print(f"prompt {PROMPT_VERSION}  |  {len(ads)} SEALED ads (never tuned against)\n")
t0 = time.time()
out = census.run(ads, ac.AnthropicBatchClient(), con,
                 poll_seconds=10, max_demotion_rate=1.0)
print(f"census: {time.time()-t0:.0f}s  calls={out['calls']}  "
      f"cache_hits={out['cache_hits']}  demoted={out['demoted']} "
      f"({out['demotion_rate']:.0%})  spend=${out['spend_usd']:.3f}\n")

r = scoring.score(out["facets"], golden)
json.dump({k: v for k, v in r.items() if k != "confusion"} |
          {"confusion": {f"{a}->{b}": c for (a, b), c in r["confusion"].items()}},
          open("reports/sealed_scores.json", "w"), indent=1, default=str)

print(f"coverage {r['coverage']:.0%}   pooled {r['pooled_accuracy']:.1%}   "
      f"invalid_levels={r['invalid_levels']}")
print("\nPER LEVEL")
for lvl, s in sorted(r["per_level"].items(), key=lambda x: -x[1]["n"]):
    print(f"  {lvl:28s} {s['correct']:2d}/{s['n']:2d}  {s['accuracy']:5.0%} "
          + "#" * int(s["accuracy"] * 20))
a = r["accessibility"]
print(f"\nACCESSIBILITY  (the metric that matters)")
print(f"  hidden wrongly  {a['hidden_wrongly']:2d} of {a['n_accessible']:2d} accessible"
      f"   ci95 [{a['hidden_wrongly_ci95'][0]:.2f}, {a['hidden_wrongly_ci95'][1]:.2f}]")
print(f"  shown wrongly   {a['shown_wrongly']:2d} of {a['n_blocking']:2d} blocking")
e = r["evidence"]
print(f"\nEVIDENCE SPANS (of {e['n_expected']})")
print(f"  exact {e['exact']}  different {e['different']}  missing {e['missing']}  "
      f"fabricated {e['fabricated']}  spurious {e['spurious']}")
print(f"  agreement {e.get('agreement')}")
print(f"\nauthorisation {r['authorisation']['correct']}/{r['authorisation']['n_expected']}"
      f"  spurious {r['authorisation']['spurious']}")
print("\nMISCLASSIFICATIONS")
for (want, got), n in sorted(r["confusion"].items(), key=lambda x: -x[1]):
    if want != got:
        print(f"  {want:28s} -> {got:28s} x{n}")
