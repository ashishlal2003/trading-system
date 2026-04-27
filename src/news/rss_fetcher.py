"""
RSS news fetcher for Indian financial markets.

Fetches from multiple RSS feeds concurrently using asyncio + run_in_executor
(feedparser is a synchronous library). Results are filtered by recency,
symbol-tagged, de-duplicated by URL, and capped at 50 items.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import asyncio
import time

import feedparser

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feed registry
# ---------------------------------------------------------------------------

RSS_FEEDS: dict[str, str] = {
    "economic_times": "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    "moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
    "livemint": "https://www.livemint.com/rss/markets",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
    "nse_news": "https://www.nseindia.com/api/rss-feed?type=news",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class NewsItem:
    source: str
    title: str
    summary: str
    published: datetime
    url: str
    symbols_mentioned: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


class RSSFetcher:
    """
    Fetches and parses RSS feeds for Indian financial news.

    Parameters
    ----------
    watchlist:
        List of NSE/BSE ticker symbols to tag inside news text.
        Symbols are stored in uppercase for case-insensitive matching.
    """

    def __init__(self, watchlist: list[str]) -> None:
        self.watchlist: list[str] = [s.upper() for s in watchlist]

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def fetch_all(self, hours_back: int = 12) -> list[NewsItem]:
        """
        Fetch all configured RSS feeds in parallel.

        Each feed is parsed in a thread-pool executor because feedparser is
        a blocking library. Results are merged, filtered by *cutoff*, tagged
        with watchlist symbols, de-duplicated by URL, sorted newest-first,
        and capped at 50 items.

        Parameters
        ----------
        hours_back:
            Only return items published within the last *hours_back* hours.

        Returns
        -------
        list[NewsItem]
            Up to 50 items, sorted by published date descending.
        """
        cutoff: datetime = datetime.now(tz=timezone.utc) - timedelta(hours=hours_back)
        loop = asyncio.get_running_loop()

        tasks = [
            loop.run_in_executor(None, self._parse_feed, source, url, cutoff)
            for source, url in RSS_FEEDS.items()
        ]

        results: list[list[NewsItem]] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        all_items: list[NewsItem] = []
        seen_urls: set[str] = set()

        for source, result in zip(RSS_FEEDS.keys(), results):
            if isinstance(result, Exception):
                logger.warning(
                    "Feed '%s' raised an unexpected exception: %s", source, result
                )
                continue
            for item in result:
                if item.url not in seen_urls:
                    seen_urls.add(item.url)
                    all_items.append(item)

        all_items.sort(key=lambda n: n.published, reverse=True)
        logger.info(
            "Fetched %d unique news items across %d feeds (last %dh).",
            len(all_items),
            len(RSS_FEEDS),
            hours_back,
        )
        return all_items[:50]

    # ------------------------------------------------------------------
    # Private helpers (synchronous — called from executor)
    # ------------------------------------------------------------------

    def _parse_feed(
        self, source: str, url: str, cutoff: datetime
    ) -> list[NewsItem]:
        """
        Synchronous: download and parse a single RSS feed.

        Errors are caught and logged so one bad feed never blocks the rest.

        Parameters
        ----------
        source:
            Human-readable feed name (key from RSS_FEEDS).
        url:
            RSS/Atom feed URL.
        cutoff:
            Items published before this datetime are discarded.

        Returns
        -------
        list[NewsItem]
            Parsed and filtered items for this feed.
        """
        try:
            logger.debug("Parsing feed '%s' from %s", source, url)
            parsed = feedparser.parse(url)

            # feedparser does not raise on HTTP errors; inspect the status.
            status: int = getattr(parsed, "status", 200)
            if status not in (200, 301, 302):
                logger.warning(
                    "Feed '%s' returned HTTP %d — skipping.", source, status
                )
                return []

            if parsed.bozo and parsed.entries == []:
                logger.warning(
                    "Feed '%s' is malformed and has no entries: %s",
                    source,
                    parsed.bozo_exception,
                )
                return []

        except Exception as exc:  # network errors, SSL issues, etc.
            logger.error("Failed to fetch feed '%s' (%s): %s", source, url, exc)
            return []

        items: list[NewsItem] = []

        for entry in parsed.entries:
            published = self._parse_date(getattr(entry, "published_parsed", None))
            if published is None:
                # Fall back to updated_parsed if available
                published = self._parse_date(
                    getattr(entry, "updated_parsed", None)
                )
            if published is None:
                # Cannot determine age — skip to avoid stale data
                continue

            # Ensure timezone-aware for comparison with cutoff
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)

            if published < cutoff:
                continue

            title: str = entry.get("title", "").strip()
            summary: str = entry.get("summary", entry.get("description", "")).strip()
            url_entry: str = entry.get("link", "").strip()

            if not title or not url_entry:
                continue

            symbols = self._extract_symbols(f"{title} {summary}")

            items.append(
                NewsItem(
                    source=source,
                    title=title,
                    summary=summary,
                    published=published,
                    url=url_entry,
                    symbols_mentioned=symbols,
                )
            )

        logger.debug(
            "Feed '%s': %d items after cutoff filter.", source, len(items)
        )
        return items

    def _extract_symbols(self, text: str) -> list[str]:
        """
        Find which watchlist symbols are mentioned in *text*.

        Uses simple word-boundary matching: a symbol is considered mentioned
        only when it appears as a standalone token (surrounded by spaces /
        punctuation), avoiding false positives like "ICICIGI" matching "ICICI".

        Parameters
        ----------
        text:
            Combined title + summary string to search.

        Returns
        -------
        list[str]
            De-duplicated list of matched symbols, in watchlist order.
        """
        if not text or not self.watchlist:
            return []

        upper_text = text.upper()
        found: list[str] = []

        for symbol in self.watchlist:
            # Check for the symbol as a whole word.  We avoid importing `re`
            # at module level for every call; a simple containment check with
            # boundary characters is fast enough for typical symbol lengths.
            idx = upper_text.find(symbol)
            while idx != -1:
                before_ok = idx == 0 or not upper_text[idx - 1].isalpha()
                after_idx = idx + len(symbol)
                after_ok = after_idx >= len(upper_text) or not upper_text[after_idx].isalpha()
                if before_ok and after_ok:
                    found.append(symbol)
                    break  # no need to find multiple occurrences
                idx = upper_text.find(symbol, idx + 1)

        return found

    def _parse_date(self, time_struct) -> datetime | None:
        """
        Convert a feedparser ``time.struct_time`` to a timezone-aware
        :class:`datetime` (UTC).

        feedparser always parses dates into UTC-normalised ``time.struct_time``
        objects (the ``*_parsed`` attributes).  We use :func:`time.mktime` via
        calendar to convert safely.

        Parameters
        ----------
        time_struct:
            A ``time.struct_time`` as returned by feedparser, or ``None``.

        Returns
        -------
        datetime | None
            UTC-aware datetime, or ``None`` if the input is ``None`` or
            invalid.
        """
        if time_struct is None:
            return None
        try:
            # time.mktime interprets the struct as *local* time, but feedparser
            # already normalises to UTC, so we use calendar.timegm instead.
            import calendar

            timestamp: float = calendar.timegm(time_struct)
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (TypeError, OverflowError, OSError) as exc:
            logger.debug("Could not parse time struct %s: %s", time_struct, exc)
            return None
