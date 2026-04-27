"""
News summarizer — combines RSS items and NSE announcements into compact,
LLM-ready strings.

The hard character cap (MAX_CHARS = 3 500) keeps the summary within a
GPT-4o context budget while still surfacing the most actionable information.

Priority order for a per-symbol summary
----------------------------------------
1. NSE corporate announcements (highest alpha, exchange-verified)
2. RSS items that mention the symbol directly
3. General market/macro headlines (no specific symbol)
"""

from src.news.nse_announcements import NSEAnnouncement
from src.news.rss_fetcher import NewsItem
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NewsSummarizer:
    """
    Combines RSS + NSE announcements into compact strings for an LLM.

    Hard cap at 3 500 characters to stay within GPT-4o context budget.

    All methods are pure (no I/O) and synchronous — they can be called
    directly from async code without ``await``.
    """

    MAX_CHARS: int = 3_500

    # ------------------------------------------------------------------
    # Per-symbol summary
    # ------------------------------------------------------------------

    def summarize_for_symbol(
        self,
        symbol: str,
        news_items: list[NewsItem],
        announcements: list[NSEAnnouncement],
    ) -> str:
        """
        Build a prioritised news digest for *symbol*.

        Structure
        ---------
        1. Up to 3 NSE announcements (highest relevance, exchange-verified).
        2. Up to 5 RSS items that mention *symbol* in their text.
        3. Up to 3 general market headlines (items with no symbol tag).

        Each line is prefixed with a source tag in square brackets so the LLM
        can easily distinguish announcement type from source.

        Parameters
        ----------
        symbol:
            The NSE ticker being analysed (e.g. ``"RELIANCE"``).
        news_items:
            All RSS news items returned by :class:`~src.news.rss_fetcher.RSSFetcher`.
        announcements:
            NSE corporate announcements for *symbol* from
            :class:`~src.news.nse_announcements.NSEAnnouncementFetcher`.

        Returns
        -------
        str
            Summary string, at most :attr:`MAX_CHARS` characters, or the
            sentinel ``"No significant news found."`` when nothing is available.
        """
        symbol = symbol.upper()
        lines: list[str] = []

        # --- 1. NSE announcements (highest priority) -------------------
        for ann in announcements[:3]:
            # Truncate description to keep individual lines manageable.
            desc_snippet = ann.description[:250].replace("\n", " ").strip()
            subject = ann.subject.strip()
            date_str = ann.announcement_date.strftime("%d-%b-%Y")
            line = f"[NSE {date_str}] {subject}: {desc_snippet}"
            lines.append(line)

        # --- 2. Symbol-specific RSS items ------------------------------
        relevant: list[NewsItem] = [
            n for n in news_items if symbol in n.symbols_mentioned
        ]
        for item in relevant[:5]:
            summary_snippet = item.summary[:200].replace("\n", " ").strip()
            source_tag = item.source.upper().replace("_", " ")
            date_str = item.published.strftime("%d-%b %H:%M")
            line = f"[{source_tag} {date_str}] {item.title}: {summary_snippet}"
            lines.append(line)

        # --- 3. General market headlines (no symbol tag) ---------------
        general: list[NewsItem] = [n for n in news_items if not n.symbols_mentioned]
        for item in general[:3]:
            source_tag = item.source.upper().replace("_", " ")
            line = f"[MARKET/{source_tag}] {item.title}"
            lines.append(line)

        if not lines:
            logger.debug(
                "No news or announcements found for symbol %s.", symbol
            )
            return "No significant news found."

        result = "\n".join(lines)

        if len(result) > self.MAX_CHARS:
            logger.debug(
                "Summary for %s truncated from %d to %d chars.",
                symbol,
                len(result),
                self.MAX_CHARS,
            )
            result = result[: self.MAX_CHARS]

        return result

    # ------------------------------------------------------------------
    # Market overview summary
    # ------------------------------------------------------------------

    def summarize_market_overview(self, news_items: list[NewsItem]) -> str:
        """
        Build a pre-market context string from the top 5 general headlines.

        Intended for use at the start of a trading session to give the LLM
        a quick macro picture before analysing individual symbols.

        Parameters
        ----------
        news_items:
            All RSS news items (sorted newest-first) as returned by
            :class:`~src.news.rss_fetcher.RSSFetcher`.

        Returns
        -------
        str
            Newline-separated headlines (up to 5), or
            ``"No market news available."`` when the list is empty.
        """
        if not news_items:
            return "No market news available."

        lines: list[str] = []
        for item in news_items[:5]:
            source_tag = item.source.upper().replace("_", " ")
            date_str = item.published.strftime("%d-%b %H:%M")
            lines.append(f"[{source_tag} {date_str}] {item.title}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Batch helper (convenience)
    # ------------------------------------------------------------------

    def summarize_watchlist(
        self,
        symbols: list[str],
        news_items: list[NewsItem],
        announcements_by_symbol: dict[str, list[NSEAnnouncement]],
    ) -> dict[str, str]:
        """
        Summarise news for every symbol in *symbols* in one call.

        Parameters
        ----------
        symbols:
            List of NSE ticker symbols.
        news_items:
            All RSS items (shared across symbols).
        announcements_by_symbol:
            Mapping returned by
            :meth:`~src.news.nse_announcements.NSEAnnouncementFetcher.fetch_multiple`.

        Returns
        -------
        dict[str, str]
            ``{symbol: summary_string}`` for every symbol in *symbols*.
        """
        return {
            symbol.upper(): self.summarize_for_symbol(
                symbol=symbol,
                news_items=news_items,
                announcements=announcements_by_symbol.get(symbol.upper(), []),
            )
            for symbol in symbols
        }
