"""What fraction of what the seekers actually say can this corpus act on?

THE NUMBER EVERY LATER NUMBER GETS COMPARED TO, so it is measured rather than
asserted. The ideal-case evaluation feeds hand-authored gold parses — set S —
straight into retrieval, bypassing the query parser, to isolate retrieval failures
from extraction failures. Its ceiling is set by the corpus, not by the encoder: a
constraint the ads never recorded cannot be honoured by a perfect parser and a
perfect index.

TWO DIFFERENT CEILINGS, AND CONFLATING THEM IS HOW THE FIRST VERSION OF THIS WAS
WRONG.

  SCHEMA COVERAGE — does a field exist anywhere in the corpus for this facet?
  Cheap, structural, and the one `gold_parse.coverage()` reports. It answers "could
  this ever be compared", not "can it be compared for this ad".

  POPULATION-WEIGHTED COVERAGE — on a randomly drawn advertisement, what share of
  the seeker's constraints actually have a value on the other side? This is the
  operative ceiling. `ad_facets.skills` exists on every row and is POPULATED on
  32.8%; `min_years_experience` on 6.1%. A facet at 6% is nominally scoreable and
  practically not, and reporting only schema coverage hides that completely.

HOW THE FIRST MAPPING WAS WRONG IN BOTH DIRECTIONS, recorded because it is the
reason this script exists. It named bare `ad_facets` keys:

  - `occupation` -> `ad_facets.occupation`, WHICH DOES NOT EXIST. Thirteen
    constraints counted as scoreable against nothing. Occupation actually lives in
    `ad_taxonomy.job_title_standardised`, on 100% of the corpus.
  - `location.place` and `contract.permanence` -> None, i.e. declared
    unscoreable — although `ad_locations` covers 100% of ads and
    `ads.engagementtype` 99.9%, where `Fast` means permanent.

So the ceiling was understated for some facets and overstated for others, and the
net figure was meaningless. `table.column` is now used throughout precisely so
that a mapping naming nothing real fails HERE, loudly, against the live schema.

Writes `reports/oracle_ceiling.json`. Needs the corpus; no API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "reports" / "oracle_ceiling.json"
EMPTY = (None, "", "unstated", [], {}, "null", "Ikke oppgitt")


def _population(con, spec: str, n_corpus: int) -> float:
    """Share of the corpus where `table.column` carries a usable value.

    `ad_facets` columns are JSON fields, so they are extracted rather than
    selected. `ad_locations` has more rows than ads, so distinct uuids are counted
    — otherwise a multi-location ad would inflate the rate above 1.0.
    """
    table, column = spec.split(".", 1)
    if table == "ad_facets":
        rows = con.execute("SELECT facets FROM ad_facets").fetchall()
        hit = 0
        for (raw,) in rows:
            f = json.loads(raw) if isinstance(raw, str) else raw
            v = f.get(column)
            if v not in EMPTY and v is not False:
                hit += 1
        return hit / n_corpus
    q = (f'SELECT count(DISTINCT uuid) FROM {table} '
         f'WHERE "{column}" IS NOT NULL '
         f"AND CAST(\"{column}\" AS VARCHAR) NOT IN ('', 'unstated', 'Ikke oppgitt')")
    return con.execute(q).fetchone()[0] / n_corpus


def main() -> None:
    import duckdb
    from finn_smart_search.eval import gold_parse as gp

    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    n_corpus = con.execute("SELECT count(*) FROM ads").fetchone()[0]

    people = gp.load_personas()
    parses = gp.load(personas=people)
    gp.check_against_personas(parses, people)

    # Every mapping must name something that exists. This is the check the first
    # version of the schema would have failed on `ad_facets.occupation`.
    schema = {t: {r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()}
              for t in ("ads", "ad_facets", "ad_taxonomy", "ad_locations")}
    facet_keys = set()
    for (raw,) in con.execute("SELECT facets FROM ad_facets LIMIT 400").fetchall():
        f = json.loads(raw) if isinstance(raw, str) else raw
        facet_keys |= set(f)

    print("=== MAPPING CHECK: does every ad_side name something real? ===\n")
    bad = []
    pop: dict[str, float] = {}
    for facet, spec in sorted(gp.FACETS.items()):
        if spec is None:
            print(f"  {facet:30s} —  no corpus counterpart")
            continue
        table, column = spec.split(".", 1)
        exists = (column in facet_keys) if table == "ad_facets" \
            else (column in schema.get(table, ()))
        if not exists:
            bad.append((facet, spec))
            print(f"  {facet:30s} {spec:42s} *** DOES NOT EXIST ***")
            continue
        pop[spec] = _population(con, spec, n_corpus)
        print(f"  {facet:30s} {spec:42s} populated {pop[spec]:6.1%}")
    if bad:
        sys.exit(f"\nFATAL: {len(bad)} ad_side mapping(s) name nothing real: {bad}")
    print("\n  every mapping resolves against the live schema.")

    # ---- the two ceilings, per split -----------------------------------
    result = {"n_corpus": n_corpus, "population": pop, "splits": {}}
    for split in ("dev", "sealed", "all"):
        ids = [i for i, p in people.items()
               if (split == "all" or p["split"] == split) and i in parses]
        cons = [c for i in ids for c in parses[i].constraints]
        if not cons:
            continue
        hard = [c for c in cons if c.priority == "hard"]

        def _schema(cs):
            return sum(1 for c in cs if c.scoreable) / len(cs) if cs else 0.0

        def _weighted(cs):
            return (sum(pop.get(c.ad_side, 0.0) if c.scoreable else 0.0
                        for c in cs) / len(cs)) if cs else 0.0

        row = {
            "n_personas": len(ids), "n_constraints": len(cons),
            "schema_coverage": _schema(cons),
            "population_weighted_coverage": _weighted(cons),
            "n_hard": len(hard),
            "schema_coverage_hard": _schema(hard),
            "population_weighted_coverage_hard": _weighted(hard),
        }
        result["splits"][split] = row
        print(f"\n=== {split.upper()}: {len(ids)} personas, {len(cons)} "
              f"stated constraints ===")
        print(f"  schema coverage        (a field exists)      "
              f"{row['schema_coverage']:6.1%}   hard {row['schema_coverage_hard']:6.1%}")
        print(f"  population-weighted   (a VALUE exists)      "
              f"{row['population_weighted_coverage']:6.1%}   hard "
              f"{row['population_weighted_coverage_hard']:6.1%}")

    # ---- where the loss is ---------------------------------------------
    print("\n=== WHERE THE CEILING IS LOST, dev constraints ===\n")
    dev = [c for i, p in people.items() if p["split"] == "dev" and i in parses
           for c in parses[i].constraints]
    per: dict[str, dict] = {}
    for c in dev:
        d = per.setdefault(c.facet, {"n": 0, "hard": 0, "pop": None})
        d["n"] += 1
        d["hard"] += c.priority == "hard"
        d["pop"] = pop.get(c.ad_side) if c.scoreable else None
    print(f"  {'facet':30s} {'n':>3s} {'hard':>5s} {'populated':>10s}  verdict")
    for facet, d in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
        if d["pop"] is None:
            verdict, ps = "NO COUNTERPART", "  --"
        elif d["pop"] < 0.5:
            verdict, ps = "sparse", f"{d['pop']:6.1%}"
        else:
            verdict, ps = "usable", f"{d['pop']:6.1%}"
        print(f"  {facet:30s} {d['n']:3d} {d['hard']:5d} {ps:>10s}  {verdict}")
    result["dev_per_facet"] = {k: {kk: vv for kk, vv in v.items()}
                               for k, v in per.items()}

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.relative_to(ROOT)}")

    d = result["splits"]["dev"]
    print("\n=== HOW TO QUOTE THIS ===")
    print(f"  A field exists for {d['schema_coverage']:.0%} of what the dev "
          f"seekers state. But on a randomly\n  drawn ad only "
          f"{d['population_weighted_coverage']:.0%} of their constraints have a "
          f"value to compare against, and for\n  HARD constraints — must-haves "
          f"rather than preferences — it is "
          f"{d['population_weighted_coverage_hard']:.0%}.")
    print("  The second number is the ceiling. Report it beside any ideal-case "
          "result, because\n  no encoder and no parser can raise it: it is a "
          "property of what advertisers wrote.")


if __name__ == "__main__":
    main()
