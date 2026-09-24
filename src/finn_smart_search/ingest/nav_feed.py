"""Crawl the NAV licensed job-ad feed (https://pam-stilling-feed.nav.no).

Two stages:
  1. walk_listings  - follow the JSON-Feed cursor, recording (uuid, status) rows.
                      Sequential by construction: each page's next_url depends on
                      the previous one.
  2. fetch_details  - fetch ad_content for uuids whose LATEST status is ACTIVE.
                      INACTIVE entries return a 112-byte stub with no ad_content,
                      so expired ads can never contribute body text.

Terms of use: https://arbeidsplassen.nav.no/vilkar-api
"""
from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

BASE = "https://pam-stilling-feed.nav.no"
TOKEN_URL = f"{BASE}/api/publicToken"          # NOTE: no /v1 on this path
FEED_URL = f"{BASE}/api/v1/feed"
ENTRY_URL = f"{BASE}/api/v1/feedentry"
UA = "finnno-smart-search/0.1 (portfolio PoC; contact via github)"

RATE_LIMIT_RPS = 5.0


class FeedSeekError(RuntimeError):
    """The feed returned a window far from the one requested."""


def rfc1123(dt: datetime) -> str:
    """RFC-1123 with a CORRECT weekday.

    The feed silently returns 2019 data when the weekday name does not match the
    date, so `strftime('%a, %d %b %Y %H:%M:%S GMT')` (locale-dependent) is unsafe.
    `email.utils.format_datetime` derives the weekday from the timestamp itself.
    """
    return format_datetime(dt.astimezone(timezone.utc), usegmt=True)


def assert_weekday_correct(header: str, dt: datetime) -> None:
    expected = calendar.day_abbr[dt.astimezone(timezone.utc).weekday()]
    if not header.startswith(expected):
        raise ValueError(f"RFC-1123 weekday mismatch: {header!r} vs expected {expected}")


async def get_token(client: httpx.AsyncClient) -> str:
    r = await client.get(TOKEN_URL, timeout=30)
    r.raise_for_status()
    for line in reversed(r.text.strip().splitlines()):
        line = line.strip()
        if re.match(r"^ey[A-Za-z0-9_-]+\.", line):
            return line
    raise RuntimeError(f"no JWT found in token response: {r.text[:200]!r}")


class RateLimiter:
    """Simple token bucket so we stay polite regardless of concurrency."""

    def __init__(self, rps: float) -> None:
        self._interval = 1.0 / rps
        self._next = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if wait:
            await asyncio.sleep(wait)


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    wait=wait_exponential_jitter(initial=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _get(client: httpx.AsyncClient, url: str, token: str, limiter: RateLimiter,
               headers: dict | None = None) -> httpx.Response:
    await limiter.acquire()
    h = {"Authorization": f"Bearer {token}", "User-Agent": UA}
    if headers:
        h.update(headers)
    r = await client.get(url, headers=h, timeout=60)
    if r.status_code in (429, 500, 502, 503, 504):
        r.raise_for_status()
    return r


async def liveness_check(client: httpx.AsyncClient, token: str, limiter: RateLimiter) -> datetime:
    r = await _get(client, f"{FEED_URL}?last=true", token, limiter)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        raise RuntimeError("feed?last=true returned no items")
    newest = datetime.fromisoformat(items[0]["date_modified"])
    age = datetime.now(timezone.utc) - newest
    if age > timedelta(hours=24):
        raise RuntimeError(f"feed looks stale: newest entry is {age} old")
    return newest


def content_sha(ad_content: dict) -> str:
    text = (ad_content.get("description") or "") + (ad_content.get("title") or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def walk_listings(con, client, token, limiter, since: datetime,
                        max_pages: int = 2000, log=print) -> int:
    """Follow the feed cursor from `since`, appending listing rows to bronze.

    Resumable: the cursor is persisted after every page, so an interruption
    costs at most one page.
    """
    from . import store

    state = store.load_state(con, "listing_cursor")
    if state and state.get("next_url"):
        url, page = BASE + state["next_url"], state.get("pages_done", 0)
        headers = None
        log(f"resuming from page {page}: {state['next_url']}")
    else:
        url, page, headers = FEED_URL, 0, {"If-Modified-Since": rfc1123(since)}
        assert_weekday_correct(headers["If-Modified-Since"], since)
        log(f"seeking to {headers['If-Modified-Since']}")

    total = 0
    while url and page < max_pages:
        r = await _get(client, url, token, limiter, headers=headers)
        headers = None
        if r.status_code == 304:
            break
        r.raise_for_status()
        d = r.json()
        items = d.get("items", [])
        if not items:
            break

        if page == 0 and state is None:
            first = datetime.fromisoformat(items[0]["date_modified"])
            if abs(first - since) > timedelta(days=7):
                raise FeedSeekError(
                    f"feed seek landed at {first.isoformat()} but requested {since.isoformat()} "
                    "- check the RFC-1123 weekday"
                )

        rows = [
            (
                it["_feed_entry"]["uuid"], it["id"], it["_feed_entry"]["status"],
                it["_feed_entry"].get("title"), it["_feed_entry"].get("businessName"),
                it["_feed_entry"].get("municipal"),
                datetime.fromisoformat(it["_feed_entry"]["sistEndret"]),
                it.get("url"), datetime.now(timezone.utc),
            )
            for it in items
        ]
        con.executemany("INSERT INTO feed_entries VALUES (?,?,?,?,?,?,?,?,?)", rows)
        total += len(rows)
        page += 1

        nxt = d.get("next_url")
        store.save_state(con, "listing_cursor",
                         {"next_url": nxt, "pages_done": page,
                          "last_seen": items[-1]["date_modified"]})
        if page % 10 == 0 or not nxt:
            log(f"  page {page:4d}  entries={total:7d}  at {items[-1]['date_modified'][:19]}")
        if not nxt:
            break
        url = BASE + nxt
    return total


async def fetch_details(con, client, token, limiter, uuids, concurrency: int = 5,
                        log=print) -> dict:
    """Fetch ad_content for the given uuids. INACTIVE -> stub, recorded, no content."""
    sem = asyncio.Semaphore(concurrency)
    stats = {"active": 0, "inactive_stub": 0, "error": 0}
    buf, logbuf = [], []

    async def one(uu: str):
        async with sem:
            try:
                r = await _get(client, f"{ENTRY_URL}/{uu}", token, limiter)
            except Exception:
                stats["error"] += 1
                logbuf.append((uu, 0, "error", datetime.now(timezone.utc)))
                return
            if r.status_code != 200:
                stats["error"] += 1
                logbuf.append((uu, r.status_code, "http_error", datetime.now(timezone.utc)))
                return
            d = r.json()
            ad = d.get("ad_content")
            if not ad:
                stats["inactive_stub"] += 1
                logbuf.append((uu, 200, "inactive_stub", datetime.now(timezone.utc)))
                return
            stats["active"] += 1
            buf.append((uu, d.get("status"), datetime.fromisoformat(d["sistEndret"]),
                        datetime.now(timezone.utc), content_sha(ad),
                        json.dumps(ad, ensure_ascii=False)))

    for i in range(0, len(uuids), 500):
        chunk = uuids[i:i + 500]
        await asyncio.gather(*(one(u) for u in chunk))
        if buf:
            con.executemany("INSERT OR REPLACE INTO ads_raw VALUES (?,?,?,?,?,?)", buf)
            buf.clear()
        if logbuf:
            con.executemany("INSERT INTO fetch_log VALUES (?,?,?,?)", logbuf)
            logbuf.clear()
        log(f"  details {min(i+500, len(uuids)):6d}/{len(uuids)}  "
            f"active={stats['active']} stub={stats['inactive_stub']} err={stats['error']}")
    return stats
