from datetime import datetime, timezone
import pytz
from src.utils.logger import get_logger

logger = get_logger(__name__)
IST = pytz.timezone("Asia/Kolkata")

_PRICE_CACHE_TTL_SECONDS = 30


class ContextBuilder:
    """
    Builds a tight, token-efficient context snapshot for the chat LLM.

    Live watchlist prices are fetched from Groww and cached for
    _PRICE_CACHE_TTL_SECONDS seconds so rapid back-and-forth messages
    don't hammer the API on every reply.
    """

    def __init__(self, trade_store, news_cache: list = None,
                 market_data=None, watchlist_symbols: list = None):
        self.store = trade_store
        self.news_cache = news_cache or []
        self._market_data = market_data
        self._watchlist_symbols = watchlist_symbols or []

        # In-memory price cache
        self._price_cache: dict[str, float] = {}
        self._price_cache_at: datetime | None = None

    async def _get_live_prices(self) -> dict[str, float]:
        """Return watchlist prices, fetching from Groww only if cache is stale."""
        if not self._market_data or not self._watchlist_symbols:
            return {}

        now = datetime.now(timezone.utc)
        cache_age = (
            (now - self._price_cache_at).total_seconds()
            if self._price_cache_at else float("inf")
        )

        if cache_age < _PRICE_CACHE_TTL_SECONDS and self._price_cache:
            logger.debug("context_builder.price_cache_hit", age_s=round(cache_age, 1))
            return self._price_cache

        try:
            quotes = await self._market_data.get_multiple_quotes(
                self._watchlist_symbols, exchange="NSE"
            )
            self._price_cache = {
                sym: float(q["ltp"]) for sym, q in quotes.items()
                if q.get("ltp") and float(q["ltp"]) > 0
            }
            self._price_cache_at = now
            logger.debug("context_builder.price_cache_refreshed",
                         count=len(self._price_cache))
        except Exception as e:
            logger.warning("context_builder.price_fetch_failed", error=str(e))

        return self._price_cache

    async def build(self) -> str:
        now = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
        sections = [f"CURRENT TIME: {now}", "PAPER TRADE MODE: Yes (no real money at risk)"]

        # Live watchlist prices
        prices = await self._get_live_prices()
        if prices:
            lines = ["WATCHLIST PRICES (live):"]
            for sym, ltp in sorted(prices.items()):
                lines.append(f"  {sym}: ₹{ltp:,.2f}")
            sections.append("\n".join(lines))

        # Open positions — include unrealised P&L if we have live price
        try:
            positions = await self.store.get_open_positions()
            if positions:
                lines = ["OPEN POSITIONS:"]
                for p in positions[:5]:
                    sym = p["symbol"]
                    entry = float(p["entry_price"])
                    qty = int(p["quantity"])
                    ltp = prices.get(sym)
                    if ltp:
                        direction = p["direction"].upper()
                        unrealised = (ltp - entry) * qty if direction == "BUY" else (entry - ltp) * qty
                        pnl_str = f" | unrealised ₹{unrealised:+.0f}"
                    else:
                        pnl_str = ""
                    lines.append(
                        f"  {sym} {p['direction']} {qty}qty "
                        f"entry=₹{entry} sl=₹{p['stop_loss']} "
                        f"target=₹{p['target_1']} type={p['trade_type']}{pnl_str}"
                    )
                sections.append("\n".join(lines))
            else:
                sections.append("OPEN POSITIONS: None")
        except Exception as e:
            logger.warning("context_positions_failed", error=str(e))
            sections.append("OPEN POSITIONS: unavailable")

        # Today's P&L
        try:
            daily_pnl = await self.store.get_daily_pnl()
            closed = await self.store.get_today_closed_trades()
            wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
            sections.append(
                f"TODAY P&L: ₹{daily_pnl:+.0f} | "
                f"Trades closed: {len(closed)} | Wins: {wins}"
            )
        except Exception as e:
            logger.warning("context_pnl_failed", error=str(e))
            sections.append("TODAY P&L: unavailable")

        # Recent signals (last 5, compressed)
        try:
            signals = await self.store.get_recent_signals(limit=5)
            if signals:
                lines = ["RECENT SIGNALS (last 5):"]
                for s in signals:
                    lines.append(
                        f"  {s['symbol']} → {s['action']} "
                        f"conf={s.get('confidence', 0):.0%} "
                        f"decision={s.get('user_decision', 'pending')}"
                    )
                sections.append("\n".join(lines))
        except Exception as e:
            logger.warning("context_signals_failed", error=str(e))

        # Top news headlines (title only, max 5)
        if self.news_cache:
            headlines = [n.title for n in self.news_cache[:5]]
            sections.append("MARKET NEWS (this morning):\n" + "\n".join(f"  • {h}" for h in headlines))

        return "\n\n".join(sections)
