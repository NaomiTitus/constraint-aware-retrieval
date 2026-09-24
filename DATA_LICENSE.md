# Data provenance and terms

## Job advertisements — NAV (Arbeids- og velferdsetaten)
Source: `https://pam-stilling-feed.nav.no` — the public, documented job-ad feed of
Arbeidsplassen.no, operated by NAV. Free to use under the terms at
<https://arbeidsplassen.nav.no/vilkar-api>.

Corpus: 10,166 ads with `status = ACTIVE`, enumerated from a 120-day feed walk
(220,988 listing entries, 59,830 distinct ads) on 2026-09-24.

**INACTIVE ads expose no `ad_content`** — a 112-byte stub. Expired ads therefore cannot
contribute body text, and the corpus ceiling is the currently-active set.

## Occupation & skill taxonomy — ESCO (European Commission)
Source: `https://ec.europa.eu/esco/api`. ESCO v1.x, used under the European Commission's
terms. 1,242 occupations and 10,063 skills, each with English and Norwegian labels.

## finn.no — NOT used
`finn.no/robots.txt` states that automated crawling requires written permission under
Norwegian copyright law, and disallows `/job/` for GPTBot. **No automated access to
finn.no was performed at any point.** Any FINN comparison data in this repository was
recorded manually, by a human, in a browser, and is marked as such.

FINN-originated ads are excluded from the NAV licensed feed: `source:"FINN"` uuids taken
from the arbeidsplassen search index return HTTP 404 from the feed. Verified on 4 uuids;
zero FINN-sourced ads appear among the 10,166 fetched.

## What this repository redistributes
`data/` is gitignored. No ad text, employer names, or contact details are committed.
Contact details (`contactList`) are stored locally but excluded from every export.
The corpus is reproducible with `python run_ingest.py`, which is resumable and
deterministic given a feed date range.
