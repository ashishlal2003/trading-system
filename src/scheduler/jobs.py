"""
APScheduler-based trading scheduler with IST timezone.

Job schedule
------------
- pre_market_scan      09:00 IST, Mon–Fri  (swing check + news cache)
- intraday_scan        every 5 min, 09:15–15:10 IST, Mon–Fri
- pre_close_square_off 15:10 IST, Mon–Fri  (INTRADAY positions squared off)
- eod_summary          15:35 IST, Mon–Fri  (end-of-day P&L report)
- evening_backtest     16:30 IST, Mon–Fri  (backtest active strategy, send report)
- sl_monitor           every 30 seconds    (stop-loss enforcement)
"""

import asyncio
from datetime import datetime

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.utils.logger import get_logger

logger = get_logger(__name__)
IST = pytz.timezone("Asia/Kolkata")


def _calc_charges(turnover: float, side: str = "both") -> float:
    """
    NSE intraday charges (Groww flat-fee structure, 2025-26 rates).

    turnover : total value traded (entry_price × qty)
    side     : "buy", "sell", or "both" (round trip)

    Breakdown per side:
      Groww brokerage : ₹20 flat per order (intraday)
      STT             : 0.025% on sell-side turnover only
      NSE txn charge  : 0.00297% each side
      SEBI charge     : 0.0001% each side
      GST             : 18% on (brokerage + txn + SEBI)
      Stamp duty      : 0.003% on buy-side turnover only
    """
    sides = 2 if side == "both" else 1

    brokerage   = 20.0 * sides
    stt         = turnover * 0.00025                          # sell side only
    nse_txn     = turnover * 0.0000297 * sides
    sebi        = turnover * 0.000001  * sides
    gst         = (brokerage + nse_txn + sebi) * 0.18
    stamp       = turnover * 0.00003                          # buy side only

    return round(brokerage + stt + nse_txn + sebi + gst + stamp, 2)

# ---------------------------------------------------------------------------
# NSE trading holiday list — 2026
# Extend annually; format: "YYYY-MM-DD"
# ---------------------------------------------------------------------------

NSE_HOLIDAYS_2026: set[str] = {
    # Fixed-date national holidays
    "2026-01-26",  # Republic Day
    "2026-05-01",  # Maharashtra Day
    "2026-08-15",  # Independence Day
    "2026-10-02",  # Gandhi Jayanti
    "2026-12-25",  # Christmas

    # Hindu festivals (lunar calendar — verify against NSE circular each year)
    "2026-02-19",  # Mahashivratri
    "2026-03-20",  # Holi (Dhuleti) — verify
    "2026-04-06",  # Ram Navami — verify
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-08-14",  # Janmashtami — verify (may coincide with Independence Day)
    "2026-10-21",  # Dussehra (Vijayadashami) — verify
    "2026-11-08",  # Diwali Laxmi Pujan — verify
    "2026-11-09",  # Diwali Balipratipada — verify
    "2026-11-05",  # Guru Nanak Jayanti — verify

    # Islamic holidays (moon-sighting dependent — approximate)
    "2026-06-27",  # Eid ul-Adha (Bakri Id) — approximate
    "2026-07-17",  # Muharram — approximate

    # Christian
    "2026-04-03",  # Good Friday (Easter = Apr 5, 2026)
}


class TradingScheduler:
    """
    Wraps APScheduler and registers all trading jobs.

    Parameters
    ----------
    data_pipeline:
        ``DataPipeline`` — scans symbols and computes indicators.
    rule_engine:
        ``RuleEngine`` — generates trading signals from the active strategy.
    risk_manager:
        ``RiskManager`` — gate-checks before allowing a new trade.
    telegram_bot:
        ``TelegramBot`` — delivers messages and signal cards.
    order_manager:
        ``OrderManager`` — places and manages broker orders.
    swing_tracker:
        ``SwingTracker`` — runs the morning swing-position review.
    news_fetcher:
        ``RSSFetcher`` — fetches RSS news feeds.
    news_summarizer:
        ``NewsSummarizer`` — converts raw news into LLM-ready strings.
    nse_announcements:
        ``NSEAnnouncementFetcher`` — fetches NSE corporate announcements.
    trade_store:
        ``TradeStore`` — reads/writes signals and position records.
    sl_enforcer:
        ``StopLossEnforcer`` — monitors and enforces stop-loss levels.
    watchlist:
        Dict loaded from ``config/watchlist.yaml``; keys ``"intraday"`` and
        ``"swing"``, values are lists of ticker strings.
    settings:
        ``Settings`` singleton — provides capital, paper-trade flag, etc.
    """

    def __init__(
        self,
        data_pipeline,
        rule_engine,
        risk_manager,
        telegram_bot,
        order_manager,
        swing_tracker,
        news_fetcher,
        news_summarizer,
        nse_announcements,
        trade_store,
        sl_enforcer,
        watchlist: dict,
        settings,
        context_builder=None,
        groww_client=None,
        backtest_engine=None,
        backtest_reporter=None,
    ) -> None:
        self.data_pipeline = data_pipeline
        self.rule_engine = rule_engine
        self.risk_manager = risk_manager
        self.telegram_bot = telegram_bot
        self.order_manager = order_manager
        self.swing_tracker = swing_tracker
        self.news_fetcher = news_fetcher
        self.news_summarizer = news_summarizer
        self.nse_announcements = nse_announcements
        self.trade_store = trade_store
        self.sl_enforcer = sl_enforcer
        self.groww_client = groww_client
        self.watchlist = watchlist
        self.settings = settings
        self.context_builder = context_builder
        self.backtest_engine = backtest_engine
        self.backtest_reporter = backtest_reporter

        # In-memory caches populated by pre_market_scan and reused by
        # intraday_scan for cheaper per-interval runs.
        self._cached_news: list = []
        self._cached_announcements: dict = {}

        # Regime filter: True = market is trending, ORB allowed.
        # Set every morning by pre_market_scan. Defaults to True so the
        # bot trades normally if the regime check fails to fetch data.
        self._market_regime_ok: bool = True

        self.scheduler = AsyncIOScheduler(timezone=IST)
        self._register_jobs()

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def _register_jobs(self) -> None:
        """Register all trading jobs on the APScheduler instance."""

        # 1. Pre-market scan — 09:00 IST, Mon–Fri
        self.scheduler.add_job(
            self.pre_market_scan,
            CronTrigger(
                hour=9,
                minute=0,
                day_of_week="mon-fri",
                timezone=IST,
            ),
            id="pre_market_scan",
            name="Pre-market scan (swing check + news)",
            replace_existing=True,
            misfire_grace_time=120,
        )

        # 2. Intraday scan — every 5 minutes between 09:15 and 15:10, Mon–Fri
        #    CronTrigger fires at minute=*/5 for hours 9 through 15.
        #    The job itself guards against firing outside 09:15–15:10 via
        #    _is_market_hours().
        self.scheduler.add_job(
            self.intraday_scan,
            CronTrigger(
                hour="9-15",
                minute="*/5",
                day_of_week="mon-fri",
                timezone=IST,
            ),
            id="intraday_scan",
            name="Intraday watchlist scan (every 5 min)",
            replace_existing=True,
            misfire_grace_time=60,
        )

        # 3. Pre-close square-off — 15:10 IST, Mon–Fri
        self.scheduler.add_job(
            self.pre_close_square_off,
            CronTrigger(
                hour=15,
                minute=10,
                day_of_week="mon-fri",
                timezone=IST,
            ),
            id="pre_close_square_off",
            name="Pre-close intraday square-off",
            replace_existing=True,
            misfire_grace_time=120,
        )

        # 4. EOD summary — 15:35 IST, Mon–Fri
        self.scheduler.add_job(
            self.eod_summary,
            CronTrigger(
                hour=15,
                minute=35,
                day_of_week="mon-fri",
                timezone=IST,
            ),
            id="eod_summary",
            name="End-of-day P&L summary",
            replace_existing=True,
            misfire_grace_time=300,
        )

        # 5. Groww auth health check — 08:00 and 09:00 IST, Mon–Fri
        #    Mirrors check_groww_apis.py: refreshes the TOTP token first,
        #    then confirms a live quote works. Self-contained — no separate
        #    token-refresh job needed.
        for hour, check_id in ((8, "groww_auth_check_0800"), (9, "groww_auth_check_0900")):
            self.scheduler.add_job(
                self.check_groww_auth,
                CronTrigger(hour=hour, minute=0, day_of_week="mon-fri", timezone=IST),
                id=check_id,
                name=f"Groww auth check ({hour:02d}:00 IST)",
                replace_existing=True,
                misfire_grace_time=120,
            )

        # 6. Evening backtest — 16:30 IST, Mon–Fri
        self.scheduler.add_job(
            self.evening_backtest,
            CronTrigger(
                hour=16,
                minute=30,
                day_of_week="mon-fri",
                timezone=IST,
            ),
            id="evening_backtest",
            name="Evening backtest report (16:30 IST)",
            replace_existing=True,
            misfire_grace_time=600,
        )

        # 7. Stop-loss monitor — every 30 seconds (all week, all day)
        #    The job itself bails out outside market hours via _is_market_hours().
        self.scheduler.add_job(
            self.sl_monitor,
            IntervalTrigger(seconds=30, timezone=IST),
            id="sl_monitor",
            name="Stop-loss enforcement (30 s)",
            replace_existing=True,
            misfire_grace_time=10,
        )

        logger.info(
            "scheduler.jobs_registered",
            jobs=[j.id for j in self.scheduler.get_jobs()],
        )

    # ------------------------------------------------------------------
    # Market-state helpers
    # ------------------------------------------------------------------

    def _is_market_holiday(self) -> bool:
        """
        Return ``True`` when today is in the NSE holiday list.

        The check is made against the current IST wall-clock date.
        """
        today_str: str = datetime.now(IST).strftime("%Y-%m-%d")
        return today_str in NSE_HOLIDAYS_2026

    def _is_market_hours(self) -> bool:
        """
        Return ``True`` when the current IST time falls within regular NSE
        trading hours (09:15 – 15:30 inclusive).
        """
        now = datetime.now(IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    async def check_groww_auth(self) -> None:
        """
        Groww auth health check — runs at 08:00 and 09:00 IST.

        Mirrors check_groww_apis.py:
          1. Refresh the TOTP access token
          2. Confirm a live quote works (SBIN as a canary symbol)
        Sends ✅ or 🚨 to Telegram so you know before market opens.
        """
        ist_time = datetime.now(IST).strftime("%I:%M %p")
        logger.info("scheduler.check_groww_auth.start", time=ist_time)
        try:
            # Step 1 — refresh token (same as check_groww_apis.py test_token)
            await self.groww_client.refresh_access_token()
            logger.info("scheduler.check_groww_auth.token_ok", time=ist_time)

            # Step 2 — confirm a live quote comes back
            quote = await self.data_pipeline._market_data.get_live_quote("SBIN", exchange="NSE")
            ltp = quote.get("last_price") or (quote.get("ohlc") or {}).get("close") or 0.0
            logger.info("scheduler.check_groww_auth.quote_ok", ltp=ltp, time=ist_time)

            await self.telegram_bot.send_message(
                f"✅ *Groww auth OK* at {ist_time} IST — token live, SBIN LTP ₹{float(ltp):,.2f}"
            )
        except Exception as exc:
            logger.error("scheduler.check_groww_auth.failed", error=str(exc), exc_info=True)
            await self.telegram_bot.send_message(
                f"🚨 *Groww auth FAILED* at {ist_time} IST\n\n"
                f"`{type(exc).__name__}: {str(exc)[:200]}`\n\n"
                f"Paste a fresh API key from groww.in/trade-api into "
                f"`GROWW_API_KEY` in `.env` and restart before 9:15 IST."
            )

    async def pre_market_scan(self) -> None:
        """
        Pre-market job — runs at 09:00 IST.

        1. Skip on market holidays.
        2. Run swing_tracker morning check.
        3. Fetch all RSS news and cache for the day.
        4. Send a pre-market summary to Telegram.
        """
        if self._is_market_holiday():
            logger.info("scheduler.pre_market_scan.holiday_skip")
            return

        logger.info("scheduler.pre_market_scan.start")

        # --- NIFTY regime info (notification only — does NOT block trading) ---
        # Per-symbol ADX in the rule engine selects the right strategy each scan.
        # A bearish NIFTY day is fine — ORB SHORTs and VWAP reversion still trade.
        try:
            import yfinance as yf
            import pandas as pd
            nifty = yf.download("^NSEI", period="70d", interval="1d", progress=False, auto_adjust=True)
            if not nifty.empty and len(nifty) >= 50:
                close_col = nifty["Close"] if "Close" in nifty.columns else nifty.iloc[:, 3]
                ema50 = close_col.ewm(span=50, adjust=False).mean()
                last_close = float(close_col.iloc[-1])
                last_ema50 = float(ema50.iloc[-1])
                above = last_close > last_ema50
                regime_label = "TRENDING UP" if above else "BEARISH/CHOPPY"
                logger.info(
                    "scheduler.regime_filter",
                    nifty_close=round(last_close, 2),
                    ema50=round(last_ema50, 2),
                    regime=regime_label,
                )
                await self.telegram_bot.send_message(
                    f"*Market Regime: {regime_label}*\n"
                    f"NIFTY: ₹{last_close:,.0f} | EMA50: ₹{last_ema50:,.0f}\n"
                    f"{'📈 Bias: LONG (ORB breakouts)' if above else '📉 Bias: SHORT (ORB breakdowns + VWAP reversion)'}"
                )
        except Exception as exc:
            logger.warning("scheduler.regime_filter.failed", error=str(exc))

        # --- Swing position morning check ---
        try:
            await self.swing_tracker.run_morning_check()
        except Exception as exc:
            logger.error(
                "scheduler.pre_market_scan.swing_check_error",
                error=str(exc),
                exc_info=True,
            )

        # --- Fetch RSS news and cache ---
        try:
            self._cached_news = await self.news_fetcher.fetch_all(hours_back=12)
            logger.info(
                "scheduler.pre_market_scan.news_cached",
                item_count=len(self._cached_news),
            )
        except Exception as exc:
            logger.error(
                "scheduler.pre_market_scan.news_fetch_error",
                error=str(exc),
                exc_info=True,
            )
            self._cached_news = []

        # Keep context_builder's news_cache in sync so the chat LLM sees fresh headlines
        if self.context_builder:
            self.context_builder.news_cache = self._cached_news

        # --- Fetch NSE corporate announcements and cache ---
        try:
            intraday_symbols = self.watchlist.get("intraday", [])
            if intraday_symbols:
                self._cached_announcements = await self.nse_announcements.fetch_multiple(
                    intraday_symbols, days_back=2
                )
                logger.info(
                    "scheduler.pre_market_scan.announcements_cached",
                    symbol_count=len(self._cached_announcements),
                )
        except Exception as exc:
            logger.error(
                "scheduler.pre_market_scan.announcements_fetch_error",
                error=str(exc),
                exc_info=True,
            )
            self._cached_announcements = {}

        # --- Pre-market Telegram summary ---
        try:
            overview = self.news_summarizer.summarize_market_overview(
                self._cached_news
            )
            mode_tag = "PAPER" if self.settings.PAPER_TRADE else "LIVE"
            ist_time = datetime.now(IST).strftime("%d %b %Y %I:%M %p")
            msg = (
                f"*Pre-Market Summary* | {ist_time} IST\n"
                f"Mode: `{mode_tag}` | Capital: ₹{self.settings.TOTAL_CAPITAL:,.2f}\n\n"
                f"{overview}"
            )
            await self.telegram_bot.send_message(msg)
        except Exception as exc:
            logger.error(
                "scheduler.pre_market_scan.telegram_summary_error",
                error=str(exc),
                exc_info=True,
            )

        logger.info("scheduler.pre_market_scan.complete")

    async def intraday_scan(self) -> None:
        """
        Intraday watchlist scan — runs every 5 minutes during market hours.

        1. Skip on holidays or outside market hours (09:15–15:30).
        2. Check risk manager gate.
        3. Scan the intraday watchlist via the data pipeline.
        4. For each scan result with valid indicators, ask the LLM for a signal.
        5. If the signal is actionable, deliver it via Telegram for approval.
        """
        if self._is_market_holiday() or not self._is_market_hours():
            return

        logger.info("scheduler.intraday_scan.start")

        # --- Risk gate ---
        try:
            can_trade, reason = await self.risk_manager.can_trade()
        except Exception as exc:
            logger.error(
                "scheduler.intraday_scan.risk_check_error",
                error=str(exc),
                exc_info=True,
            )
            return

        if not can_trade:
            logger.warning(
                "scheduler.intraday_scan.risk_gate_blocked",
                reason=reason,
            )
            return

        # --- Scan intraday watchlist ---
        intraday_symbols: list[str] = self.watchlist.get("intraday", [])
        if not intraday_symbols:
            logger.warning("scheduler.intraday_scan.empty_watchlist")
            return

        try:
            scan_results = await self.data_pipeline.scan_watchlist(
                symbols=intraday_symbols,
                exchange="NSE",
                interval="5m",
                mode="intraday",
            )
        except Exception as exc:
            logger.error(
                "scheduler.intraday_scan.pipeline_error",
                error=str(exc),
                exc_info=True,
            )
            try:
                await self.telegram_bot.send_message(
                    f"⚠️ *Intraday scan failed* — pipeline error\n`{type(exc).__name__}: {exc}`"
                )
            except Exception:
                pass
            return

        # Fetch live Groww prices for all successfully scanned symbols so the
        # LLM uses the real current price, not a potentially stale yfinance close.
        live_prices: dict[str, float] = {}
        live_symbols = [r.symbol for r in scan_results if r.indicators is not None]
        if live_symbols:
            try:
                quotes = await self.data_pipeline._market_data.get_multiple_quotes(
                    live_symbols, exchange="NSE"
                )
                live_prices = {
                    sym: q["ltp"] for sym, q in quotes.items()
                    if q.get("ltp") and float(q["ltp"]) > 0
                }
                logger.info(
                    "scheduler.intraday_scan.live_prices_fetched",
                    count=len(live_prices),
                )
            except Exception as exc:
                logger.warning(
                    "scheduler.intraday_scan.live_prices_failed",
                    error=str(exc),
                )

        # Alert if ALL symbols failed (likely expired Groww token)
        failed_results = [r for r in scan_results if r.error]
        if failed_results and len(failed_results) == len(scan_results):
            sample_error = failed_results[0].error or "unknown"
            try:
                await self.telegram_bot.send_message(
                    f"⚠️ *All symbols failed to scan* — Groww API error?\n"
                    f"Sample: `{sample_error[:200]}`\n"
                    f"Check if your Groww API key has expired and needs renewal."
                )
            except Exception:
                pass
            return

        # --- Generate signals via rule engine and forward actionable ones ---
        for result in scan_results:
            if result.candles_df.empty:
                if result.error:
                    logger.warning(
                        "scheduler.intraday_scan.symbol_error",
                        symbol=result.symbol,
                        error=result.error,
                    )
                continue

            try:
                signal = self.rule_engine.generate_signal(
                    df=result.candles_df,
                    symbol=result.symbol,
                    exchange=result.exchange,
                    live_price=live_prices.get(result.symbol),
                )

                if not signal.is_actionable:
                    logger.debug(
                        "scheduler.intraday_scan.signal_not_actionable",
                        symbol=result.symbol,
                        action=signal.action,
                        reasoning=signal.reasoning[:60],
                    )
                    continue

                # Pre-trade risk gate
                approved, reason = await self.risk_manager.pre_trade_check(
                    signal=signal.to_dict(),
                    current_price=live_prices.get(result.symbol, signal.entry_price),
                )
                if not approved:
                    logger.info(
                        "scheduler.intraday_scan.risk_rejected",
                        symbol=result.symbol,
                        reason=reason,
                    )
                    continue

                # Compute position size
                pos_size = self.risk_manager.compute_position_size(
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    trade_type=signal.trade_type,
                )

                await self.telegram_bot.send_signal(signal, pos_size.quantity)
                logger.info(
                    "scheduler.intraday_scan.signal_sent",
                    symbol=signal.symbol,
                    action=signal.action,
                    strategy=self.rule_engine.strategy.name,
                    quantity=pos_size.quantity,
                )

            except Exception as exc:
                logger.error(
                    "scheduler.intraday_scan.signal_error",
                    symbol=result.symbol,
                    error=str(exc),
                    exc_info=True,
                )

        logger.info(
            "scheduler.intraday_scan.complete",
            symbols_scanned=len(scan_results),
        )

    async def pre_close_square_off(self) -> None:
        """
        Pre-close square-off — runs at 15:10 IST.

        In paper-trade mode: reads open INTRADAY positions from the local DB,
        fetches live prices via yfinance, closes them in the DB, and notifies
        Telegram per position.

        In live mode: delegates to order_manager.square_off_all_intraday().
        """
        if self._is_market_holiday():
            logger.info("scheduler.pre_close_square_off.holiday_skip")
            return

        logger.info("scheduler.pre_close_square_off.start")

        if self.settings.PAPER_TRADE:
            await self._paper_square_off()
        else:
            await self._live_square_off()

    async def _paper_square_off(self) -> None:
        """Close all open INTRADAY positions from the local DB at last traded price."""
        try:
            positions = await self.trade_store.get_open_positions()
        except Exception as exc:
            logger.error("scheduler.pre_close_square_off.get_positions_error", error=str(exc))
            return

        intraday = [p for p in positions if p.get("trade_type", "INTRADAY").upper() == "INTRADAY"]

        if not intraday:
            logger.info("scheduler.pre_close_square_off.no_intraday_positions")
            return

        logger.info("scheduler.pre_close_square_off.squaring_off", count=len(intraday))

        async def _get_exit_price(symbol: str) -> float:
            """Groww live LTP at squareoff time, yfinance as fallback."""
            try:
                quote = await self.data_pipeline._market_data.get_live_quote(
                    symbol, exchange="NSE"
                )
                raw = quote.get("last_price") or (quote.get("ohlc") or {}).get("close") or 0.0
                ltp = float(raw)
                if ltp > 0:
                    return ltp
            except Exception as exc:
                logger.warning(
                    "scheduler.pre_close_square_off.groww_price_failed",
                    symbol=symbol,
                    error=str(exc),
                )
            # Fallback: yfinance last 1-min close
            try:
                import yfinance as _yf
                from functools import partial as _partial
                loop = asyncio.get_event_loop()

                def _yf_ltp(sym: str) -> float:
                    hist = _yf.Ticker(f"{sym}.NS").history(period="1d", interval="1m")
                    return float(hist["Close"].iloc[-1]) if not hist.empty else 0.0

                price = await loop.run_in_executor(None, _partial(_yf_ltp, symbol))
                if price > 0:
                    logger.info(
                        "scheduler.pre_close_square_off.yfinance_fallback_used",
                        symbol=symbol,
                        price=price,
                    )
                    return price
            except Exception as exc:
                logger.warning(
                    "scheduler.pre_close_square_off.yfinance_fallback_failed",
                    symbol=symbol,
                    error=str(exc),
                )
            return 0.0

        for p in intraday:
            symbol = p["symbol"]
            pos_id = p["id"]
            try:
                exit_price = await _get_exit_price(symbol)
                if exit_price <= 0:
                    logger.error(
                        "scheduler.pre_close_square_off.no_price",
                        symbol=symbol,
                    )
                    continue
                await self.trade_store.close_position(pos_id, exit_price, "EOD_SQUAREOFF")

                entry  = float(p["entry_price"])
                qty    = int(p["quantity"])
                direction = p["direction"].upper()
                pnl = (exit_price - entry) * qty if direction == "BUY" else (entry - exit_price) * qty
                emoji = "🟢" if pnl >= 0 else "🔴"
                sign  = "+" if pnl >= 0 else ""

                try:
                    await self.telegram_bot.send_message(
                        f"{emoji} SQUARED OFF — {symbol}\n"
                        f"Direction : {direction}\n"
                        f"Qty       : {qty}\n"
                        f"Entry     : Rs {entry:.2f}\n"
                        f"Exit      : Rs {exit_price:.2f}\n"
                        f"Gross P&L : {sign}Rs {pnl:.2f}\n"
                        f"Reason    : EOD_SQUAREOFF"
                    )
                except Exception:
                    pass

                logger.info(
                    "scheduler.pre_close_square_off.closed",
                    symbol=symbol, exit_price=exit_price, pnl=round(pnl, 2),
                )
            except Exception as exc:
                logger.error(
                    "scheduler.pre_close_square_off.close_error",
                    symbol=symbol, error=str(exc),
                )

    async def _live_square_off(self) -> None:
        """Delegate to broker for live mode square-off."""
        try:
            positions = await self.order_manager.get_open_positions()
        except Exception as exc:
            logger.error("scheduler.pre_close_square_off.get_positions_error", error=str(exc))
            return

        intraday_positions = [
            p for p in positions
            if p.get("product_type", p.get("productType", "")).upper() in ("INTRADAY", "MIS")
        ]

        if not intraday_positions:
            logger.info("scheduler.pre_close_square_off.no_intraday_positions")
            return

        n = len(intraday_positions)
        try:
            await self.telegram_bot.send_message(
                f"*Pre-Close Square-Off*\n\nSquaring off {n} intraday position(s) at 15:10 IST."
            )
        except Exception:
            pass

        try:
            results = await self.order_manager.square_off_all_intraday(intraday_positions)
            logger.info("scheduler.pre_close_square_off.complete", orders_placed=len(results))
        except Exception as exc:
            logger.error("scheduler.pre_close_square_off.square_off_error", error=str(exc))

    async def eod_summary(self) -> None:
        """
        End-of-day summary — runs at 15:35 IST.

        Reads today's closed trades from the trade store, computes total P&L,
        and sends a formatted summary to Telegram.
        """
        if self._is_market_holiday():
            logger.info("scheduler.eod_summary.holiday_skip")
            return

        logger.info("scheduler.eod_summary.start")

        try:
            closed_trades = await self.trade_store.get_today_closed_trades()
        except Exception as exc:
            logger.error(
                "scheduler.eod_summary.get_trades_error",
                error=str(exc),
                exc_info=True,
            )
            return

        total_gross: float = sum(
            float(t.get("pnl", 0.0) or 0.0) for t in closed_trades
        )
        n_trades = len(closed_trades)

        # Compute real brokerage + taxes per trade
        trade_charges = []
        for t in closed_trades:
            entry   = float(t.get("entry_price", 0.0) or 0.0)
            qty     = int(t.get("quantity", 0) or 0)
            turnover = entry * qty
            charges = _calc_charges(turnover, side="both")
            trade_charges.append(charges)

        total_charges = sum(trade_charges)
        total_net     = total_gross - total_charges

        winners = [t for t in closed_trades if float(t.get("pnl", 0.0) or 0.0) > 0]
        losers  = [t for t in closed_trades if float(t.get("pnl", 0.0) or 0.0) < 0]

        def _sign(v: float) -> str: return "+" if v >= 0 else ""

        ist_date = datetime.now(IST).strftime("%d %b %Y")

        lines = [
            f"*EOD Summary — {ist_date}*\n",
            f"Total trades : {n_trades}  |  Winners: {len(winners)}  Losers: {len(losers)}",
            f"Gross P&L    : `{_sign(total_gross)}₹{total_gross:,.2f}`",
            f"Charges      : `-₹{total_charges:,.2f}` _(brokerage + STT + GST + NSE)_",
            f"*Net P&L     : `{_sign(total_net)}₹{total_net:,.2f}`*",
        ]

        if closed_trades:
            lines.append("\n*Trade Breakdown:*")
            for trade, charges in zip(closed_trades, trade_charges):
                sym       = trade.get("symbol", "?")
                gross     = float(trade.get("pnl", 0.0) or 0.0)
                net       = gross - charges
                direction = trade.get("direction", "?")
                reason    = trade.get("exit_reason", "?")
                lines.append(
                    f"  {sym} ({direction}) — Gross `{_sign(gross)}₹{gross:,.2f}` "
                    f"→ Net `{_sign(net)}₹{net:,.2f}` after ₹{charges:.0f} charges  [{reason}]"
                )

        message = "\n".join(lines)

        try:
            await self.telegram_bot.send_message(message)
        except Exception as exc:
            logger.error(
                "scheduler.eod_summary.telegram_error",
                error=str(exc),
                exc_info=True,
            )

        logger.info(
            "scheduler.eod_summary.complete",
            trades=n_trades,
            gross_pnl=round(total_gross, 2),
            charges=round(total_charges, 2),
            net_pnl=round(total_net, 2),
        )

    async def evening_backtest(self) -> None:
        """
        Evening backtest job — runs at 16:30 IST after market close.

        1. Fetches last 6 months of 5-min candles for each intraday symbol.
        2. Runs BacktestEngine with the active strategy.
        3. Computes metrics and generates a report.
        4. Sends the report summary to Telegram.
        """
        if self._is_market_holiday():
            logger.info("scheduler.evening_backtest.holiday_skip")
            return

        if self.backtest_engine is None or self.backtest_reporter is None:
            logger.warning("scheduler.evening_backtest.not_configured")
            return

        logger.info("scheduler.evening_backtest.start")

        try:
            await self.telegram_bot.send_message(
                f"*Evening Backtest* | {self.rule_engine.strategy.name}\n"
                "Running strategy validation on today's data... ⏳"
            )
        except Exception:
            pass

        symbols = self.watchlist.get("intraday", [])[:3]  # limit to 3 to keep it fast
        for symbol in symbols:
            try:
                scan_results = await self.data_pipeline.scan_watchlist(
                    symbols=[symbol],
                    exchange="NSE",
                    interval="5m",
                    mode="intraday",
                )
                if not scan_results or scan_results[0].candles_df.empty:
                    continue

                df = scan_results[0].candles_df
                bt_result = self.backtest_engine.run(
                    df=df,
                    strategy=self.rule_engine.strategy,
                    initial_capital=float(self.settings.TOTAL_CAPITAL),
                    symbol=symbol,
                )
                report_msg = self.backtest_reporter.telegram_report(bt_result)
                await self.telegram_bot.send_message(report_msg)

                logger.info(
                    "scheduler.evening_backtest.symbol_done",
                    symbol=symbol,
                    trades=bt_result.trade_count,
                    return_pct=round(bt_result.total_return_pct, 2),
                )
            except Exception as exc:
                logger.error(
                    "scheduler.evening_backtest.symbol_error",
                    symbol=symbol,
                    error=str(exc),
                    exc_info=True,
                )

        logger.info("scheduler.evening_backtest.complete")

    async def sl_monitor(self) -> None:
        """
        Stop-loss monitor — runs every 30 seconds.

        1. Skip outside market hours.
        2. Fetch the set of open position symbols from the trade store.
        3. Get live quotes for each symbol from the market data service.
        4. Build a ``{symbol: price}`` dict and hand it to the enforcer.
        """
        if not self._is_market_hours():
            return

        try:
            symbols: list[str] = await self.trade_store.get_open_position_symbols()
        except Exception as exc:
            logger.error(
                "scheduler.sl_monitor.get_symbols_error",
                error=str(exc),
                exc_info=True,
            )
            return

        if not symbols:
            return

        # Fetch live quotes — one call per symbol (sequential to avoid rate limits)
        prices: dict[str, float] = {}
        for sym in symbols:
            try:
                quote = await self.data_pipeline._market_data.get_live_quote(
                    sym, exchange="NSE"
                )
                raw = quote.get("last_price") or (quote.get("ohlc") or {}).get("close") or 0.0
                ltp = float(raw)
                if ltp > 0:
                    prices[sym] = ltp
                else:
                    logger.warning(
                        "scheduler.sl_monitor.zero_ltp",
                        symbol=sym,
                    )
            except Exception as exc:
                logger.warning(
                    "scheduler.sl_monitor.quote_error",
                    symbol=sym,
                    error=str(exc),
                )

        if not prices:
            return

        try:
            await self.sl_enforcer.check_and_enforce(prices)
        except Exception as exc:
            logger.error(
                "scheduler.sl_monitor.enforcer_error",
                error=str(exc),
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the APScheduler background scheduler."""
        self.scheduler.start()
        logger.info("scheduler.started")

    def stop(self) -> None:
        """Stop the APScheduler background scheduler (waits for running jobs)."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("scheduler.stopped")
