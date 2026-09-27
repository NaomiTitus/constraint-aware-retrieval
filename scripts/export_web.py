"""Export a static search index for the GitHub Pages demo. PLAN G1.

WHAT MAY BE SHIPPED, AND WHY THIS IS NOT A STYLE CHOICE. `DATA_LICENSE.md` states:
"No ad text, employer names, or contact details are committed. Contact details are
stored locally but excluded from every export." A GitHub Pages bundle is the most
public export there is, so it carries DERIVED fields only, plus a link back to the
advertisement on arbeidsplassen.no where the real text lives.

  shipped        title, ESCO occupation label (no + en), STYRK code, municipality,
                 census language level, English skill glosses, expiry date, the
                 canonical arbeidsplassen URL
  NOT shipped    description_text, employer name, contactList

The title is necessary — a result list without one is unusable — and is the one
advertiser-written string included. Everything else is either a taxonomy code or a
model-derived gloss.

WHY THE SEARCH RUNS IN THE BROWSER. Pages is static hosting: there is no server to
run BM25 or the predicates on. So the index is precomputed here and the ranking is
computed client-side, which is what PLAN G1 specifies ("precomputed facets, lambda
slider, negation toggle diff").

A HONEST LIMITATION SHIPPED WITH THE DATA. `skills` comes from the ORIGINAL census
column, populated on 32.8% of advertisements. The re-prompt measured at recall 0.861
(LIMITATIONS §15) has NOT been run over the corpus, so skill matching in the demo is
weaker than the pilot shows is achievable. The index records `skills_source` so the
page can say so rather than implying otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "docs" / "data"
DEMOTED_LEVELS = ("professional", "certified", "fluent", "conversational",
                  "scandinavian_accepted")


SKILLS_CENSUS = ROOT / "reports" / "skills_census.json"


def main() -> None:
    import duckdb
    from finn_smart_search.retrieval import occupation as occ

    # The skills-v1 re-run when it exists, the census column otherwise. Kept in its
    # own file rather than merged into ad_facets: that row's prompt_version says
    # census-v15 and a mixed row would carry neither version honestly (§4).
    new_skills: dict[str, list[dict]] = {}
    skills_source = "census-v15 (32.8% of ads)"
    if SKILLS_CENSUS.exists():
        blob = json.loads(SKILLS_CENSUS.read_text(encoding="utf-8"))
        new_skills = blob["skills"]
        skills_source = (f"{blob['prompt_version']} / {blob['model']} — "
                         f"{blob['n_skills']:,} skills, "
                         f"{blob['n_zero'] / blob['n_ads']:.1%} of ads with none")

    con = duckdb.connect(str(ROOT / "data" / "ads.duckdb"), read_only=True)
    rows = con.execute("""SELECT a.uuid, a.title, a.expires, a.extent,
                                 a.engagementtype,
                                 t.job_title_standardised, t.job_title_en,
                                 t.styrk_code, t.nav_category,
                                 f.facets
                          FROM ads a
                          JOIN ad_taxonomy t USING (uuid)
                          JOIN ad_facets f USING (uuid)
                          ORDER BY a.uuid""").fetchall()
    # ALL of an advertisement's locations, not the first. 7.1% of ads carry more
    # than one (12,003 rows over 10,165 uuids), and `setdefault` kept only the
    # first — silently dropping 722 ads that DO list the city a seeker asked for.
    # County travels too: it covers 99.8% of ads against municipality's 99.1% and
    # is what gives `retrieval/location.py` its same-county tier.
    loc: dict[str, list[tuple[str, str]]] = {}
    for uuid, muni, county in con.execute(
            "SELECT uuid, municipal, county FROM ad_locations").fetchall():
        pair = ((muni or "").title(), (county or "").title())
        if pair != ("", ""):
            loc.setdefault(uuid, [])
            if pair not in loc[uuid]:
                loc[uuid].append(pair)

    ads = []
    for (uuid, title, expires, extent, engagement, occ_no, occ_en, styrk,
         cat, raw_facets) in rows:
        f = json.loads(raw_facets) if isinstance(raw_facets, str) else raw_facets
        lvl = f.get("norwegian_requirement_level") or "unstated"
        # BOTH the verbatim phrase AND its English gloss. The gloss is the MATCH
        # KEY — that is the entire reason skills-v1 produces one — and shipping only
        # the Norwegian phrase would leave an English query term matching Norwegian
        # skill text, which is the cross-language failure D18 measured.
        src = new_skills.get(uuid) or (f.get("skills") or [])
        skills, glosses = [], []
        for sk in src[:12]:
            ph, gl = sk.get("phrase", ""), sk.get("gloss_en", "")
            if ph:
                skills.append(ph)
            if gl and gl.lower() != ph.lower():
                glosses.append(gl)
        ads.append({
            "u": uuid[:8],                        # short id; the URL carries the full
            "t": title or "",
            "o": occ_no or "",
            "e": occ_en or "",
            "s": styrk or "",
            "c": cat or "",
            "m": (loc.get(uuid) or [("", "")])[0][0],   # primary, for display
            "L": loc.get(uuid, []),                       # every (municipal, county)
            "l": lvl,
            "w": f.get("stated_working_language") or "unstated",
            "a": bool(f.get("english_accessible")),
            "k": skills,
            "g": glosses,
            "x": expires.strftime("%Y-%m-%d") if expires else "",
            "ext": extent or "",
            "eng": engagement or "",
            "url": f"https://arbeidsplassen.nav.no/stillinger/stilling/{uuid}",
        })

    # The occupation gazetteer, so the BROWSER can resolve a typed phrase to STYRK
    # codes. This is D18's bilingual identity path, shipped as data.
    esco_rows = con.execute("SELECT uri, lang, title FROM esco_occupation").fetchall()
    ad_rows = con.execute("""SELECT c.code, t.styrk_code FROM ad_categories c
                             JOIN ad_taxonomy t USING (uuid)
                             WHERE c.category_type='ESCO'""").fetchall()
    gaz = occ.EscoGazetteer.build(esco_rows, ad_rows)
    # label -> the styrk codes its URIs appear under, deduplicated
    label_styrk: dict[str, list[str]] = {}
    for label, uris in gaz.label_uris.items():
        codes = sorted(gaz.styrk_for(uris))
        if codes:
            label_styrk[label] = codes[:6]

    # A TOKEN index, because exact label matching fails on the most important case.
    # ESCO's English label for `sykepleier` is "nurse responsible for general care",
    # so a seeker typing `nurse` matches no label at all — the same failure D18's
    # outcome measured, where symmetric matching penalised the correct label for
    # being specific. Mapping each content token to the STYRK codes of every label
    # containing it gives the CANDIDATE SET behaviour instead: `nurse` reaches the
    # whole nursing family, and picking one would be guessing.
    NOISE = set("og i på som en et til for med av the a an and of in at or mv andre "
                "annet øvrige diverse generell generelle".split())
    token_styrk: dict[str, dict[str, int]] = {}
    for label, codes in label_styrk.items():
        for tok in label.split():
            if len(tok) < 3 or tok in NOISE:
                continue
            bucket = token_styrk.setdefault(tok, {})
            for c in codes:
                bucket[c] = bucket.get(c, 0) + 1
    # Keep the codes a token points at most often; a token spanning half the
    # taxonomy identifies nothing and is dropped.
    token_index = {}
    for tok, codes in token_styrk.items():
        if len(codes) > 40:
            continue
        top = sorted(codes.items(), key=lambda kv: -kv[1])[:6]
        token_index[tok] = [c for c, _ in top]

    # A THIRD RESOLUTION TIER, BUILT FROM THE EMPLOYERS' OWN WORDS.
    #
    # ESCO has the AI-engineer concept — `ICT intelligent systems designer`,
    # `utvikler av IKT-intelligenssystemer` — and calls it something no job seeker
    # would ever type. Measured: ZERO ESCO labels contain `ai`, `ml` or `artificial`
    # as a word, so `ai engineer` shares no token at all with its own ESCO label and
    # neither exact nor token matching can reach it.
    #
    # The corpus solves this itself. `Software AI Engineer` IS an advertisement
    # title, and `ad_taxonomy` already assigned it STYRK 2511. So indexing the
    # employers' own title vocabulary against the codes those advertisements carry
    # bridges seeker language to taxonomy without inventing a synonym list: the
    # corpus supplies the synonyms.
    #
    # Titles are noisy ("Lyst til å jobbe med AI i et av Nordens største mediehus"),
    # so a token must appear in at least two advertisements and keeps only the codes
    # it points at most often.
    TITLE_NOISE = NOISE | set(
        "vi er du en et jobb stilling søker søkes til med som har vil ledig "
        "nye vår våre deg din ditt oss the a an and of in at or for to we you "
        "your our new job jobs position role work".split())
    title_tok: dict[str, dict[str, int]] = {}
    for _uuid, _t, _x, _e, _g, occ_no_, occ_en_, styrk_, _c, _f in rows:
        if not styrk_:
            continue
        for word in re.findall(r"[^\W_]+", (_t or "").lower(), re.UNICODE):
            if len(word) < 2 or word in TITLE_NOISE:
                continue
            b = title_tok.setdefault(word, {})
            b[str(styrk_)] = b.get(str(styrk_), 0) + 1
    title_index = {}
    for word, codes in title_tok.items():
        total = sum(codes.values())
        if total < 2 or len(codes) > 25:
            continue
        top = sorted(codes.items(), key=lambda kv: -kv[1])[:5]
        # only codes carrying at least a tenth of the token's advertisements
        title_index[word] = [c for c, n in top if n / total >= 0.10]

    index = {
        "built_at": __import__("time").strftime("%Y-%m-%d"),
        "n_ads": len(ads),
        "snapshot": "2026-09-24",
        "skills_source": skills_source,
        "levels_demoted": list(DEMOTED_LEVELS),
        "ads": ads,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    (OUT_DIR / "occupations.json").write_text(
        json.dumps({"labels": label_styrk, "tokens": token_index,
                    "titles": title_index},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")

    sz = (OUT_DIR / "index.json").stat().st_size
    gz = (OUT_DIR / "occupations.json").stat().st_size
    print(f"docs/data/index.json        {sz/1e6:.2f} MB  ({len(ads)} ads)")
    print(f"docs/data/occupations.json  {gz/1e6:.2f} MB  "
          f"({len(label_styrk)} bilingual labels, {len(token_index)} tokens)")
    print(f"    + {len(title_index)} tokens from employers' own advertisement titles")
    for probe in ("nurse", "tømrer", "ai", "ml", "vision", "scientist"):
        via = ("esco-label" if probe in label_styrk else
               "esco-token" if probe in token_index else
               "ad-title" if probe in title_index else "UNRESOLVED")
        codes = label_styrk.get(probe) or token_index.get(probe) or title_index.get(probe)
        print(f"    {probe:10s} {via:12s} {codes or ''}")
    multi = sum(1 for a in ads if len(a["L"]) > 1)
    with_county = sum(1 for a in ads if any(c for _, c in a["L"]))
    print(f"\nads with >1 location:  {multi} ({multi/len(ads):.1%}) — all shipped, "
          f"not just the first")
    print(f"ads with a county:     {with_county} ({with_county/len(ads):.1%})")
    n_sk = sum(1 for a in ads if a["k"])
    tot = sum(len(a["k"]) for a in ads)
    print(f"ads with >=1 skill:    {n_sk} ({n_sk/len(ads):.1%})")
    print(f"skills shipped:        {tot:,} ({tot/len(ads):.2f}/ad) + "
          f"{sum(len(a['g']) for a in ads):,} English glosses")
    print(f"skills source:         {skills_source}")
    print("NOT exported, per DATA_LICENSE.md: description_text, employer name, "
          "contact details")


if __name__ == "__main__":
    main()
