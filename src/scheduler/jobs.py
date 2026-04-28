"""
APScheduler-based trading scheduler with IST timezone.

Job schedule
------------
- pre_market_scan      09:00 IST, Mon–Fri  (swing check + news cache)
- intraday_scan        every 5 min, 09:15–15:10 IST, Mon–Fri
- pre_close_square_off 15:10 IST, Mon–Fri  (INTRADAY positions squared off)
- eod_summary          15:35 IST, Mon–Fri  (end-of-day P&L report)
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

# ---------------------------------------------------------------------------
# NSE trading holiday list — 2026
# Extend annually; format: "YYYY-MM-DD"
# ---------------------------------------------------------------------------

NSE_HOLIDAYS_2026: set[str] = {
    "2026-01-26",  # Republic Day
    "2026-03-25",  # Holi
    "2026-04-02",  # Ram Navami
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-04-17",  # Good Friday
    "2026-05-01",  # Maharashtra Day
    "2026-08-15",  # Independence Day
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-24",  # Diwali Laxmi Pujan
    "2026-11-04",  # Diwali Balipratipada
    "2026-11-25",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
}


class TradingScheduler:
    """
    Wraps APScheduler and registers all trading jobs.

    Parameters
    ----------
    data_pipeline:
        ``DataPipeline`` — scans symbols and computes indicators.
    llm_engine:
        ``LLMSignalEngine`` — generates trading signals from indicator data.
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
        llm_engine,
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
    ) -> None:
        self.data_pipeline = data_pipeline
        self.llm_engine = llm_engine
        self.risk_manager = risk_manager
        self.telegram_bot = telegram_bot
        self.order_manager = order_manager
        self.swing_tracker = swing_tracker
        self.news_fetcher = news_fetcher
        self.news_summarizer = news_summarizer
        self.nse_announcements = nse_announcements
        self.trade_store = trade_store
        self.sl_enforcer = sl_enforcer
        self.groww_client = groww_client  # needed for TOTP token refresh
        self.watchlist = watchlist
        self.settings = settings
        self.context_builder = context_builder

        # In-memory caches populated by pre_market_scan and reused by
        # intraday_scan for cheaper per-interval runs.
        self._cached_news: list = []
        self._cached_announcements: dict = {}

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

        # 5. Daily TOTP token refresh — 06:05 IST every day
        #    Groww access tokens expire at 06:00 IST. Runs 5 minutes after to
        #    ensure Groww has cycled, then fetches a fresh token automatically.
        #    Only registered when TOTP is configured.
        if self.groww_client and getattr(self.settings, "GROWW_TOTP_SECRET", ""):
            self.scheduler.add_job(
                self.refresh_groww_token,
                CronTrigger(hour=6, minute=5, timezone=IST),
                id="groww_token_refresh",
                name="Groww TOTP access token refresh (06:05 IST)",
                replace_existing=True,
                misfire_grace_time=300,
            )

        # 6. Stop-loss monitor — every 30 seconds (all week, all day)
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

    async def refresh_groww_token(self) -> None:
        """
        Refresh the Groww access token using TOTP at 06:05 IST daily.

        Groww tokens expire at 06:00 IST. This job runs 5 minutes after to get
        a fresh token so the pre-market scan at 09:00 and intraday scans start
        with a valid credential — no manual intervention needed.
        """
        logger.info("scheduler.refresh_groww_token.start")
        try:
            await self.groww_client.refresh_access_token()
            logger.info("scheduler.refresh_groww_token.success")
            try:
                await self.telegram_bot.send_message(
                    "🔑 Groww access token refreshed automatically (06:05 IST). Ready for today's session."
                )
            except Exception:
                pass
        except Exception as exc:
            logger.error("scheduler.refresh_groww_token.failed", error=str(exc), exc_info=True)
            try:
                await self.telegram_bot.send_message(
                    f"🚨 *Groww token refresh FAILED at 06:05 IST!*\n"
                    f"`{type(exc).__name__}: {str(exc)[:200]}`\n\n"
                    f"No signals will be generated today. Check your TOTP secret in .env."
                )
            except Exception:
                pass

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
        if hasattr(self, 'context_builder') and self.context_builder:
            self.context_builder.news_cache = self._cached_news

        # --- Verify Groww API is reachable ---
        try:
            await self.data_pipeline._market_data.get_portfolio()
            logger.info("scheduler.pre_market_scan.groww_api_ok")
        except Exception as exc:
            logger.error("scheduler.pre_market_scan.groww_api_error", error=str(exc))
            try:
                await self.telegram_bot.send_message(
                    f"🚨 *Groww API unreachable at pre-market check!*\n"
                    f"`{type(exc).__name__}: {str(exc)[:200]}`\n\n"
                    f"Your API key may have expired. Renew it on the EC2 before 9:15 IST or no signals will be generated today."
                )
            except Exception:
                pass

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

        # --- Generate signals and forward actionable ones ---
        for result in scan_results:
            if result.indicators is None:
                if result.error:
                    logger.warning(
                        "scheduler.intraday_scan.symbol_error",
                        symbol=result.symbol,
                        error=result.error,
                    )
                else:
                    logger.debug(
                        "scheduler.intraday_scan.no_indicators",
                        symbol=result.symbol,
                    )
                continue

            try:
                # Build per-symbol news summary from cached data
                announcements = self._cached_announcements.get(result.symbol.upper(), [])
                news_summary = self.news_summarizer.summarize_for_symbol(
                    symbol=result.symbol,
                    news_items=self._cached_news,
                    announcements=announcements,
                )

                signal = await self.llm_engine.generate_signal(
                    symbol=result.symbol,
                    exchange=result.exchange,
                    trade_type="INTRADAY",
                    indicators=result.indicators,
                    patterns=result.patterns,
                    news_summary=news_summary,
                )

                if not signal.is_actionable:
                    logger.debug(
                        "scheduler.intraday_scan.signal_not_actionable",
                        symbol=result.symbol,
                        action=signal.action,
                        confidence=signal.confidence,
                    )
                    continue

                # Compute position size
                pos_size = self.risk_manager.compute_position_size(
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                )

                await self.telegram_bot.send_signal(signal, pos_size.quantity)
                logger.info(
                    "scheduler.intraday_scan.signal_sent",
                    symbol=signal.symbol,
                    action=signal.action,
                    confidence=signal.confidence,
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

        Fetches all open broker positions, filters for INTRADAY product type,
        and market-exits them via ``order_manager.square_off_all_intraday()``.
        """
        if self._is_market_holiday():
            logger.info("scheduler.pre_close_square_off.holiday_skip")
            return

        logger.info("scheduler.pre_close_square_off.start")

        try:
            positions = await self.order_manager.get_open_positions()
        except Exception as exc:
            logger.error(
                "scheduler.pre_close_square_off.get_positions_error",
                error=str(exc),
                exc_info=True,
            )
            return

        intraday_positions = [
            p
            for p in positions
            if p.get("product_type", p.get("productType", "")).upper()
            in ("INTRADAY", "MIS")
        ]

        if not intraday_positions:
            logger.info("scheduler.pre_close_square_off.no_intraday_positions")
            return

        n = len(intraday_positions)
        logger.info("scheduler.pre_close_square_off.squaring_off", count=n)

        try:
            await self.telegram_bot.send_message(
                f"*Pre-Close Square-Off*\n\nSquaring off {n} intraday position(s) at 15:10 IST."
            )
        except Exception as exc:
            logger.warning(
                "scheduler.pre_close_square_off.telegram_notify_error",
                error=str(exc),
            )

        try:
            results = await self.order_manager.square_off_all_intraday(intraday_positions)
            logger.info(
                "scheduler.pre_close_square_off.complete",
                orders_placed=len(results),
            )
        except Exception as exc:
            logger.error(
                "scheduler.pre_close_square_off.square_off_error",
                error=str(exc),
                exc_info=True,
            )

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

        total_pnl: float = sum(
            float(t.get("pnl", 0.0) or 0.0) for t in closed_trades
        )
        n_trades = len(closed_trades)
        winners = [t for t in closed_trades if float(t.get("pnl", 0.0) or 0.0) > 0]
        losers = [t for t in closed_trades if float(t.get("pnl", 0.0) or 0.0) < 0]

        pnl_sign = "+" if total_pnl >= 0 else ""
        ist_date = datetime.now(IST).strftime("%d %b %Y")

        lines = [
            f"*EOD Summary — {ist_date}*\n",
            f"Total trades: {n_trades}",
            f"Winners: {len(winners)}  |  Losers: {len(losers)}",
            f"Net P&L: `{pnl_sign}₹{total_pnl:,.2f}`",
        ]

        if closed_trades:
            lines.append("\n*Trade Breakdown:*")
            for trade in closed_trades:
                sym = trade.get("symbol", "?")
                pnl = float(trade.get("pnl", 0.0) or 0.0)
                direction = trade.get("direction", "?")
                reason = trade.get("exit_reason", "?")
                sign = "+" if pnl >= 0 else ""
                lines.append(f"  {sym} ({direction}) — `{sign}₹{pnl:,.2f}` [{reason}]")

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
            total_pnl=round(total_pnl, 2),
        )

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
                ltp = float(quote.get("ltp", quote.get("lastPrice", 0.0)))
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
