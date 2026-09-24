"""Phase A driver: crawl the NAV licensed feed into bronze."""
import asyncio, sys, argparse
from datetime import datetime, timedelta, timezone
sys.path.insert(0, "src")
import httpx
from finn_smart_search.ingest import nav_feed as nf, store


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--stage", choices=["listings", "details", "all"], default="all")
    ap.add_argument("--db", default="data/ads.duckdb")
    a = ap.parse_args()

    con = store.connect(a.db)
    limiter = nf.RateLimiter(nf.RATE_LIMIT_RPS)
    async with httpx.AsyncClient(http2=True, follow_redirects=True) as client:
        token = await nf.get_token(client)
        newest = await nf.liveness_check(client, token, limiter)
        print(f"feed is live; newest entry {newest.isoformat()}", flush=True)

        if a.stage in ("listings", "all"):
            since = datetime.now(timezone.utc) - timedelta(days=a.days)
            t0 = datetime.now()
            n = await nf.walk_listings(con, client, token, limiter, since,
                                       log=lambda m: print(m, flush=True))
            print(f"LISTINGS DONE: {n} entries in {datetime.now()-t0}", flush=True)

        if a.stage in ("details", "all"):
            rows = con.execute("""
                WITH latest AS (
                  SELECT uuid, status,
                         ROW_NUMBER() OVER (PARTITION BY uuid ORDER BY sist_endret DESC) rn
                  FROM feed_entries)
                SELECT uuid FROM latest
                WHERE rn = 1 AND status = 'ACTIVE'
                  AND uuid NOT IN (SELECT uuid FROM ads_raw)
            """).fetchall()
            uuids = [r[0] for r in rows]
            print(f"ACTIVE uuids still to fetch: {len(uuids)}", flush=True)
            t0 = datetime.now()
            stats = await nf.fetch_details(con, client, token, limiter, uuids,
                                           log=lambda m: print(m, flush=True))
            print(f"DETAILS DONE: {stats} in {datetime.now()-t0}", flush=True)

    print(con.execute("SELECT count(*) entries, count(DISTINCT uuid) uniq FROM feed_entries").df())
    print(con.execute("SELECT count(*) ads FROM ads_raw").df())

asyncio.run(main())
