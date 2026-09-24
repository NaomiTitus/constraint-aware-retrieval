# Architecture — ETL, storage, and the latency argument

## 0. The principle everything follows from

**Budgets are asymmetric.**

| | Ad side (index time) | Query side (request time) |
|---|---|---|
| Volume | ~10k docs, once | 1 query, every keystroke |
| Latency budget | minutes per doc | **< 300 ms p95** |
| Cost budget | fractions of a cent per doc, amortised over every future query | must be ~free |
| LLM affordable? | **Yes** | **No** |

So: **push every expensive semantic operation to index time, and make the query path pure
arithmetic over dense arrays.** That single decision is what makes the ETL "a large part of the
problem" — correctly, because the ETL is where the latency is bought.

The corollary, which is the highest-leverage idea here: **the LLM's best use is not inference, it is
label generation.** Use it offline to label the corpus, distil those labels into a small local model,
and the online path becomes free. You pay once.

---

## 1. Storage — bronze → silver → gold

One DuckDB file (`data/ads.duckdb`): single file, SQL, columnar, no server, reads/writes Parquet.
Right for 10k rows on a laptop. At FINN scale this layer becomes a warehouse + a search engine; the
*shape* below is unchanged, which is the point.

### Bronze — immutable raw (built)
```
feed_entries(uuid, feed_id, status, title, business_name, municipal, sist_endret, url, seen_at)
ads_raw(uuid PK, status, sist_endret, fetched_at, content_sha256, ad_content JSON)
fetch_log(uuid, http_status, outcome, fetched_at)
crawl_state(key PK, value JSON, updated_at)
```
Append-only, never deleted, never edited. Every downstream table is derivable from it, so the whole
corpus can be reprocessed without re-crawling NAV. `feed_entries` keeps *all* revisions of an ad
(mean ~3.7 per ad), which is free change-history: republish cadence, how ads get edited, and a
natural source of near-duplicate pairs.

`content_sha256` over `title + description` is the **re-extraction key**: if an ad is republished but
the text is unchanged, nothing downstream re-runs. This is what keeps incremental ETL cheap.

### Silver — normalised, deterministic
```
ads(uuid PK, title, jobtitle, description_html, description_text, published, expires, updated,
    application_due, application_url, link, employer_name, employer_orgnr, employer_homepage,
    engagementtype, extent, sector, positioncount, source, content_sha256)
ad_locations(uuid, country, address, city, postal_code, county, municipal)
ad_categories(uuid, category_type, code, name, score)   -- ESCO | STYRK08 | JANZZ
ad_contacts(uuid, name, email, phone, role, title)      -- PII: excluded from all exports
```
Pure function of bronze. HTML → blocks → text happens here. Re-runnable at any time.

### Gold — the serving layer
```
ad_facets(uuid PK, <~40 typed facet columns>, extractor_version, prompt_version, extracted_at)
ad_skills(uuid, phrase, esco_uri, skill_group, requirement_level, confidence, char_span)
ad_embeddings         -> data/emb_{model}_{version}.npy   (float32, quantised int8 for the web)
bm25_index            -> data/bm25_{version}.bin
esco_graph            -> data/graph.parquet + scipy CSR
```

**Critical:** at query time the facet table is **not** queried as SQL rows. It is loaded once into a
dense `(n_ads × n_facets)` array — uint8 categorical codes plus float32 calibrated probabilities.
Constraint scoring is then a single vectorised numpy expression over the whole corpus. DuckDB is the
*ETL and analytics* layer; the *serving* layer is arrays in RAM. This is the design that makes
sub-10 ms achievable, and it is why the facet schema is worth getting right.

Every gold row carries `extractor_version` + `prompt_version`, so two extractor generations are
diffable and only genuinely-changed rows recompute.

---

## 2. Facet schema — organised by extraction cost

The central ETL insight: **do not spend an LLM on what is already structured.** Three tiers, cheapest
first; each tier only handles what the tier above could not.

### Tier 0 — free, already structured in the feed (zero extraction cost)
`extent` (Heltid/Deltid) · `engagementtype` (Fast/Vikariat/Sesong/Engasjement/Prosjekt/Trainee/
Lærling/Åremål) · `sector` (Privat/Offentlig) · `positioncount` · `published` / `expires` /
`application_due` · employer name + **orgnr** · full location hierarchy (city/postal/municipal/county)
· `occupationCategories` level1/level2 · **ESCO / STYRK08 / JANZZ codes with relevance scores** ·
`jobtitle`.

Roughly a third of the useful facets cost nothing. Free ESCO codes are the single most valuable item
here — they are the multilingual backbone of §3.

### Tier 1 — deterministic rules + gazetteers (~ms/ad, no API calls)

| Facet | Values | Notes |
|---|---|---|
| `body_lang`, `lang_mix`, `is_bilingual` | no / en / mixed / unknown | per-block langid, char-weighted |
| **`norwegian_required`** | required / preferred / not_required / unknown | + verbatim `evidence_span` |
| **`working_language`** | norwegian / english / both / scandinavian | distinct from the above |
| `other_languages` | list | Swedish, German, Spanish… |
| `min_years_experience` | int | `minst 3 års erfaring`, `3+ years` |
| `education_level` | none/vgs/fagbrev/fagskole/bachelor/master/phd | **ordinal**, so `>=` works |
| `education_field` | text → ISCED/ESCO | |
| **`authorisation_required`** | bool + profession | `norsk autorisasjon` — **never** conflated with language |
| `security_clearance` | bool | `sikkerhetsklarering`, `politiattest` |
| `drivers_license` | bool + class (B/C/CE/…) | partly structured already |
| `remote_mode` | onsite / hybrid / remote | + days-per-week when stated |
| `shift_work` | bool + type | `turnus`, `skift`, `helgearbeid` — large in health/industry |
| `travel_required` | bool + % | |
| `seniority` | intern…executive | title patterns + years |
| `position_pct` | int | `100% stilling`, `80% stilling` |
| `salary_stated` | bool + range | rare in NO, but present |
| `union_agreement` | bool | `tariffavtale` |
| `is_agency` | bool | staffing agencies — dedupe + quality signal |

### Tier 2 — genuinely needs semantics (classifier, LLM only on the residue)

- **`skills` with `requirement_level ∈ {required, preferred, implied}`.** The hard one, and the one
  that matters most. Needs requirements-section detection, phrase extraction, *and* a
  required-vs-nice-to-have judgement that regex cannot make reliably.
- **`relocation_support` / `visa_sponsorship`** ∈ {offered, not_offered, unstated}. Phrased a dozen
  ways, usually absent, and **directly decisive for a non-Norwegian applicant** — exactly the facet no
  incumbent exposes. High value, genuinely needs an LLM.
- **`role_family`** — normalised job function above ESCO's granularity.

### On skill categorisation — use ESCO's hierarchy, don't invent one

Rather than hand-rolling technical/management/other, use ESCO's own skill groups: `S1` communication
& collaboration · `S2` assisting & caring · `S3` **management** · `S4` **working with computers** ·
`S5` handling & moving · `S6` constructing · `S7` working with machinery · `S8` processing
information — crossed with ESCO's transversal-vs-occupation-specific axis.

Three reasons this beats bespoke categories: it is **multilingual for free** (every skill has a
Norwegian *and* English label, so an English query matches a Norwegian ad by URI identity, no
translation step); it **generalises to all 3,039 occupations**, which is what "must work for any job,
not just data science" actually requires; and it is a **stable public vocabulary**, so results are
auditable rather than dependent on a prompt.

Supplement it with one small hand-maintained **technology gazetteer** (languages, frameworks, clouds,
certifications), because ESCO lags fast-moving tech — that is the one place a bespoke list earns its
keep.

---

## 3. The query side — and the better method

### Why not text-to-SQL
Tempting and wrong: it puts an LLM in the hot path (500-1500 ms), generates invalid SQL, has no
graceful degradation, and does no semantic matching. Rejected.

### Why not hard `WHERE` filtering either
`unknown` will be a large share of every extracted facet. A hard filter on `norwegian_required` drops
every ad the extractor was unsure about — destroying recall to buy precision the user did not ask for.

**Store calibrated probabilities, not classes**, and turn constraints into *scores*:
```
penalty = λ · Σ_c w_c · severity_c(facets, profile)
```
with a hard filter reserved for high-confidence binary facts the user explicitly demanded. `λ` is a
product dial, not a constant — and being able to show its recall/violation tradeoff curve is worth
more than any single operating point.

### Three-tier query parsing — latency by construction

| Tier | Handles | Latency |
|---|---|---|
| **1. Gazetteer + regex fast path** | ~80% of real queries: language, location, remote, years, education, licence, extent | **< 1 ms** |
| **2. Distilled local classifier / token-tagger** | the residue, trained on LLM-generated labels | ~10 ms |
| **3. LLM, strict JSON schema** | genuinely novel phrasing only; cached by normalised query | 500-1500 ms |

Two multipliers on top:
- **Parse once per session, not per keystroke.** The profile is stable; only the *ranking* changes as
  the user adjusts filters. Moving the λ slider re-ranks in ~1 ms with no re-parse.
- **Cache by normalised query.** Job-search queries are heavily repeated across users.

### Measured latency budget (everything resident in RAM)

| Stage | 10k ads |
|---|---|
| Query parse (fast path) | < 1 ms |
| BM25 (`bm25s`, numpy-native) | ~3 ms |
| Dense (10k × 768 matmul) | ~5 ms |
| Graph k-hop expansion | ~1 ms |
| RRF + vectorised facet scoring | ~2 ms |
| **Total, no LLM** | **~12 ms** |

10k × 768 float32 = 30 MB. No FAISS, no vector DB, no service mesh. Knowing when *not* to reach for
infrastructure is part of the argument.

### Where the LLM actually belongs, ranked by leverage

1. **Label generation for distillation** — label the corpus offline, train a small local model, ship
   the small model. Highest leverage by a wide margin, and it is the honest answer to "how would this
   run in production at p95 < 300 ms".
2. **Index-time extraction of Tier-2 facets** (sponsorship, required-vs-nice skills) — amortised over
   every future query.
3. **Offline evaluation judging** — no latency constraint at all.
4. **Query parsing, residue only** — cached, once per session.
5. **Explanation generation** for the top-k — and *stream it after results render*, so it never blocks
   the ranking.

**Never:** inside the retrieval loop, synchronously reranking every query, or per keystroke.
