from datetime import datetime, timezone
import asyncio
import pytz
import yfinance as yf
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

    @staticmethod
    def _yfinance_price(symbol: str) -> float | None:
        """Fetch last close from yfinance as offline fallback. Runs in thread pool."""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(period="2d", interval="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return None

    async def _get_live_prices(self) -> dict[str, float]:
        """
        Return prices for watchlist symbols.

        Primary: Groww live LTP (works during market hours).
        Fallback: yfinance last close (when Groww returns zeros outside hours).
        Results are cached for _PRICE_CACHE_TTL_SECONDS.
        """
        if not self._watchlist_symbols:
            return {}

        now = datetime.now(timezone.utc)
        cache_age = (
            (now - self._price_cache_at).total_seconds()
            if self._price_cache_at else float("inf")
        )

        if cache_age < _PRICE_CACHE_TTL_SECONDS and self._price_cache:
            logger.debug("context_builder.price_cache_hit", age_s=round(cache_age, 1))
            return self._price_cache

        live: dict[str, float] = {}

        # --- Primary: Groww ---
        if self._market_data:
            try:
                quotes = await self._market_data.get_multiple_quotes(
                    self._watchlist_symbols, exchange="NSE"
                )
                live = {
                    sym: float(q["ltp"]) for sym, q in quotes.items()
                    if q.get("ltp") and float(q["ltp"]) > 0
                }
                logger.debug("context_builder.groww_prices", count=len(live))
            except Exception as e:
                logger.warning("context_builder.groww_price_fetch_failed", error=str(e))

        # --- Fallback: yfinance for any symbol that Groww didn't fill ---
        missing = [s for s in self._watchlist_symbols if s not in live]
        if missing:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(None, self._yfinance_price, sym)
                for sym in missing
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, price in zip(missing, results):
                if isinstance(price, float) and price > 0:
                    live[sym] = price
            logger.debug("context_builder.yfinance_fallback",
                         requested=len(missing),
                         filled=sum(1 for s in missing if s in live))

        self._price_cache = live
        self._price_cache_at = now
        return self._price_cache

    async def build(self) -> str:
        now = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
        sections = [f"CURRENT TIME: {now}", "PAPER TRADE MODE: Yes (no real money at risk)"]

        # Prices — live during market hours, last close otherwise
        prices = await self._get_live_prices()
        ist_now = datetime.now(IST)
        market_open = ist_now.hour == 9 and ist_now.minute >= 15 or ist_now.hour > 9
        market_open = market_open and (ist_now.hour < 15 or (ist_now.hour == 15 and ist_now.minute <= 30))
        market_open = market_open and ist_now.weekday() < 5
        price_label = "live" if market_open else "last close"
        if prices:
            lines = [f"WATCHLIST PRICES ({price_label}):"]
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
                        tag = "live" if market_open else "est"
                        pnl_str = f" | unrealised ₹{unrealised:+.0f} ({tag})"
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
