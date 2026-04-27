"""
NSE corporate announcements fetcher.

NSE's public API is unauthenticated but requires:
  - Browser-like HTTP headers (User-Agent, Accept, Referer)
  - A valid session cookie obtained by first visiting nseindia.com

Fetching is done sequentially across symbols (not in parallel) to avoid
triggering NSE's rate-limiting / Cloudflare protection.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class NSEAnnouncement:
    symbol: str
    subject: str
    description: str
    announcement_date: datetime
    category: str


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

_NSE_BASE = "https://www.nseindia.com"
_ANNOUNCEMENTS_URL = (
    f"{_NSE_BASE}/api/corporate-announcements?index=equities&symbol={{symbol}}"
)

# Headers that mimic a real browser session — required by NSE's CDN.
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": _NSE_BASE,
    "Connection": "keep-alive",
    "DNT": "1",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

# Seconds to sleep between consecutive symbol requests.
_INTER_REQUEST_SLEEP: float = 2.0

# Timeout for the initial cookie-grab GET (seconds).
_COOKIE_TIMEOUT: float = 5.0

# Timeout for data API calls (seconds).
_API_TIMEOUT: float = 10.0


class NSEAnnouncementFetcher:
    """
    Fetches corporate announcements from NSE's public (unauthenticated) API.

    Usage
    -----
    ::

        fetcher = NSEAnnouncementFetcher()
        results = await fetcher.fetch_multiple(["RELIANCE", "TCS"], days_back=3)

    Notes
    -----
    * The client must first visit the NSE homepage to obtain a session cookie;
      subsequent API calls include that cookie automatically.
    * Requests are sent sequentially to respect NSE rate limits.
    * HTTP 403 / 429 responses are logged as warnings and return an empty list
      for the affected symbol (the caller continues with remaining symbols).
    """

    def __init__(self) -> None:
        # The httpx client is created lazily (on first use) so the class can
        # be instantiated outside an async context.
        self._client: httpx.AsyncClient | None = None
        self._cookie_fetched: bool = False

    # ------------------------------------------------------------------
    # Internal client management
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """Return (and lazily create) the shared async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=_BROWSER_HEADERS,
                follow_redirects=True,
                timeout=_API_TIMEOUT,
            )
            self._cookie_fetched = False
        return self._client

    async def _ensure_cookies(self) -> None:
        """
        Visit the NSE homepage to acquire the session cookie.

        NSE's CDN (Cloudflare) sets cookies on the first request to the root
        URL.  Subsequent API calls must carry those cookies or they will be
        rejected with HTTP 403.

        This method is idempotent — it only issues the GET once per client
        lifetime.
        """
        if self._cookie_fetched:
            return

        client = await self._get_client()
        try:
            logger.debug("Fetching NSE homepage to acquire session cookie …")
            response = await client.get(
                _NSE_BASE, timeout=_COOKIE_TIMEOUT
            )
            response.raise_for_status()
            self._cookie_fetched = True
            logger.debug(
                "Cookie acquired (status=%d, cookies=%s).",
                response.status_code,
                dict(client.cookies),
            )
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "NSE homepage returned HTTP %d — cookie may be missing.",
                exc.response.status_code,
            )
            # Continue anyway; the API call will reveal the real problem.
            self._cookie_fetched = True
        except Exception as exc:
            logger.error("Failed to fetch NSE homepage for cookie: %s", exc)
            self._cookie_fetched = True

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def fetch(
        self, symbol: str, days_back: int = 3
    ) -> list[NSEAnnouncement]:
        """
        Fetch corporate announcements for a single *symbol*.

        Workflow
        --------
        1. Ensure the NSE session cookie is present (homepage GET).
        2. GET the corporate announcements API endpoint.
        3. Parse and filter by *days_back*.

        Parameters
        ----------
        symbol:
            NSE ticker symbol (e.g. ``"RELIANCE"``).  Case-insensitive;
            internally uppercased.
        days_back:
            Return announcements published within the last *days_back* days.

        Returns
        -------
        list[NSEAnnouncement]
            Announcements sorted newest-first.  Empty list on any error.
        """
        symbol = symbol.upper()
        cutoff: datetime = datetime.now(tz=timezone.utc) - timedelta(days=days_back)

        await self._ensure_cookies()
        client = await self._get_client()

        url = _ANNOUNCEMENTS_URL.format(symbol=symbol)
        logger.debug("Fetching NSE announcements for %s from %s", symbol, url)

        try:
            response = await client.get(url)
        except httpx.TimeoutException:
            logger.warning(
                "Timeout fetching NSE announcements for %s — returning empty.", symbol
            )
            return []
        except httpx.RequestError as exc:
            logger.error(
                "Network error fetching NSE announcements for %s: %s", symbol, exc
            )
            return []

        # Rate-limit / auth guard
        if response.status_code in (403, 429):
            logger.warning(
                "NSE API returned HTTP %d for symbol %s — skipping (rate limit / block).",
                response.status_code,
                symbol,
            )
            return []

        if response.status_code != 200:
            logger.warning(
                "NSE API returned unexpected HTTP %d for symbol %s.",
                response.status_code,
                symbol,
            )
            return []

        try:
            payload: Any = response.json()
        except Exception as exc:
            logger.error(
                "Could not decode JSON from NSE for symbol %s: %s", symbol, exc
            )
            return []

        return self._parse_announcements(symbol, payload, cutoff)

    async def fetch_multiple(
        self, symbols: list[str], days_back: int = 3
    ) -> dict[str, list[NSEAnnouncement]]:
        """
        Fetch announcements for *symbols* one at a time (sequential).

        Sequential execution is intentional: NSE aggressively rate-limits
        concurrent requests from the same IP.  A :data:`_INTER_REQUEST_SLEEP`
        second pause is inserted between each request.

        Parameters
        ----------
        symbols:
            List of NSE ticker symbols.
        days_back:
            Passed through to :meth:`fetch`.

        Returns
        -------
        dict[str, list[NSEAnnouncement]]
            ``{symbol: [announcements, …]}`` for every requested symbol,
            including symbols that returned an empty list due to errors.
        """
        results: dict[str, list[NSEAnnouncement]] = {}

        for i, symbol in enumerate(symbols):
            announcements = await self.fetch(symbol, days_back=days_back)
            results[symbol.upper()] = announcements
            logger.info(
                "NSE announcements for %s: %d item(s).",
                symbol.upper(),
                len(announcements),
            )
            # Sleep between requests (but not after the very last one).
            if i < len(symbols) - 1:
                await asyncio.sleep(_INTER_REQUEST_SLEEP)

        return results

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_announcements(
        self,
        symbol: str,
        payload: Any,
        cutoff: datetime,
    ) -> list[NSEAnnouncement]:
        """
        Parse the raw NSE API JSON response into :class:`NSEAnnouncement` objects.

        The NSE API returns a dict with a ``"data"`` key containing a list of
        announcement dicts.  Each dict typically has these keys:

        * ``"bflag"`` — board meeting flag (ignored here)
        * ``"smIndustry"`` — industry (ignored)
        * ``"sort_date"`` — ISO-style timestamp string e.g. "20240424120000"
        * ``"an_dt"``     — date string "DD-Mon-YYYY HH:MM:SS"
        * ``"subject"``   — short subject
        * ``"desc"``      — longer description
        * ``"attchmntFile"`` — filename (ignored)
        * ``"attchmntText"`` — attachment text (ignored)
        * ``"exchdisstime"`` — exchange dissemination time
        * ``"seq_no"``    — sequence number
        * ``"smIndustry"`` — industry
        * ``"symbol"``    — symbol (same as queried)

        Parameters
        ----------
        symbol:
            The queried symbol (used as fallback when absent in the record).
        payload:
            Decoded JSON from the API.
        cutoff:
            Announcements older than this are excluded.

        Returns
        -------
        list[NSEAnnouncement]
            Sorted newest-first.
        """
        if not isinstance(payload, dict):
            logger.warning(
                "Unexpected NSE payload type %s for %s.", type(payload).__name__, symbol
            )
            return []

        raw_items: list[Any] = payload.get("data", [])
        if not isinstance(raw_items, list):
            logger.warning(
                "NSE 'data' field is not a list for %s (got %s).",
                symbol,
                type(raw_items).__name__,
            )
            return []

        announcements: list[NSEAnnouncement] = []

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            ann_date = self._parse_nse_date(
                item.get("an_dt") or item.get("exchdisstime") or item.get("sort_date")
            )
            if ann_date is None:
                continue

            if ann_date.tzinfo is None:
                ann_date = ann_date.replace(tzinfo=timezone.utc)

            if ann_date < cutoff:
                continue

            subject: str = (item.get("subject") or "").strip()
            description: str = (item.get("desc") or "").strip()
            record_symbol: str = (item.get("symbol") or symbol).strip().upper()
            category: str = (item.get("smIndustry") or "General").strip()

            announcements.append(
                NSEAnnouncement(
                    symbol=record_symbol,
                    subject=subject,
                    description=description,
                    announcement_date=ann_date,
                    category=category,
                )
            )

        announcements.sort(key=lambda a: a.announcement_date, reverse=True)
        return announcements

    # ------------------------------------------------------------------
    # Date parsing helpers
    # ------------------------------------------------------------------

    def _parse_nse_date(self, raw: str | None) -> datetime | None:
        """
        Parse an NSE date string into a :class:`datetime`.

        NSE uses several inconsistent date formats across its APIs.  We try
        each known format in order and return the first successful parse.

        Known formats
        -------------
        * ``"DD-Mon-YYYY HH:MM:SS"``  e.g. ``"24-Apr-2024 12:00:00"``
        * ``"YYYYMMDDHHMMSS"``        e.g. ``"20240424120000"`` (sort_date)
        * ISO 8601                    e.g. ``"2024-04-24T12:00:00"``

        Returns
        -------
        datetime | None
        """
        if not raw:
            return None

        raw = raw.strip()

        _FORMATS = [
            "%d-%b-%Y %H:%M:%S",   # "24-Apr-2024 12:00:00"
            "%d-%b-%Y",             # "24-Apr-2024"
            "%Y%m%d%H%M%S",        # "20240424120000"
            "%Y-%m-%dT%H:%M:%S",   # ISO 8601 no TZ
            "%Y-%m-%d %H:%M:%S",   # ISO-like with space
            "%Y-%m-%d",            # date-only ISO
        ]

        for fmt in _FORMATS:
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue

        logger.debug("Could not parse NSE date string: %r", raw)
        return None

    # ------------------------------------------------------------------
    # Context manager support (optional — for explicit client cleanup)
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying HTTP client and release connections."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("NSEAnnouncementFetcher HTTP client closed.")

    async def __aenter__(self) -> "NSEAnnouncementFetcher":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
