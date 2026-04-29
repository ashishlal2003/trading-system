#!/usr/bin/env python3
"""
Indian Stock Market Auto-Trader
NSE/BSE Intraday + Swing | Groww API | GPT-4o | Telegram Approval

Entry point — wires all components and starts the async trading system.

Usage
-----
    python main.py

The process runs until interrupted with SIGINT (Ctrl-C) or SIGTERM.
"""

import asyncio
import signal
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Make the project root importable regardless of where the script is invoked
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from src.utils.logger import configure_logging, get_logger

# ---------------------------------------------------------------------------
# Broker layer
# ---------------------------------------------------------------------------
from src.broker.groww_client import GrowwClient
from src.broker.market_data import MarketDataService
from src.broker.order_manager import OrderManager, OrderRequest, OrderType, ProductType

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
from src.data.store import CandleStore, TradeStore
from src.data.indicators import IndicatorEngine
from src.data.patterns import PatternDetector
from src.data.pipeline import DataPipeline

# ---------------------------------------------------------------------------
# News layer
# ---------------------------------------------------------------------------
from src.news.rss_fetcher import RSSFetcher
from src.news.nse_announcements import NSEAnnouncementFetcher
from src.news.summarizer import NewsSummarizer

# ---------------------------------------------------------------------------
# Signal & risk layer
# ---------------------------------------------------------------------------
from src.signals.llm_engine import LLMSignalEngine
from src.risk.manager import RiskManager
from src.risk.stop_loss import StopLossEnforcer

# ---------------------------------------------------------------------------
# Tracker & scheduler
# ---------------------------------------------------------------------------
from src.tracker.swing_tracker import SwingTracker
from src.telegram.bot import TelegramBot
from src.scheduler.jobs import TradingScheduler
from src.utils.context_builder import ContextBuilder
from src.signals.chat_engine import ChatEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_watchlist(path: str = "config/watchlist.yaml") -> dict:
    """
    Load the watchlist YAML from *path* (relative to the project root).

    Returns an empty dict with ``"intraday"`` and ``"swing"`` keys on error
    so the rest of the system can continue gracefully.
    """
    abs_path = Path(__file__).parent / path
    try:
        with abs_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return {
            "intraday": data.get("intraday", []),
            "swing":    data.get("swing", []),
        }
    except Exception as exc:
        # Logger may not be configured yet; use print as fallback.
        print(f"[main] WARNING: Could not load watchlist from {abs_path}: {exc}", file=sys.stderr)
        return {"intraday": [], "swing": []}


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

async def shutdown(
    scheduler: TradingScheduler,
    bot: TelegramBot,
    logger,
) -> None:
    """
    Gracefully stop the scheduler and Telegram bot.

    Called by the SIGINT / SIGTERM handler.
    """
    logger.info("main.shutdown_initiated")
    try:
        scheduler.stop()
    except Exception as exc:
        logger.error("main.scheduler_stop_error", error=str(exc))
    try:
        await bot.stop()
    except Exception as exc:
        logger.error("main.bot_stop_error", error=str(exc))
    logger.info("main.shutdown_complete")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Configure logging
    # -----------------------------------------------------------------------
    configure_logging(settings.LOG_LEVEL)
    logger = get_logger(__name__)

    mode_label = "PAPER" if settings.PAPER_TRADE else "LIVE"
    logger.info(
        "main.startup",
        mode=mode_label,
        capital=settings.TOTAL_CAPITAL,
        model=settings.OPENAI_MODEL,
        log_level=settings.LOG_LEVEL,
    )
    logger.info(
        "=== Indian Stock Market Auto-Trader ===",
        mode=mode_label,
        capital=f"₹{settings.TOTAL_CAPITAL:,.2f}",
    )

    # -----------------------------------------------------------------------
    # 2. Initialise persistent stores
    # -----------------------------------------------------------------------
    candle_store = CandleStore(settings.DB_PATH)
    trade_store = TradeStore(settings.DB_PATH)

    await candle_store.init_db()
    await trade_store.init_db()
    logger.info("main.stores_initialised", db_path=settings.DB_PATH)

    # -----------------------------------------------------------------------
    # 3. Load watchlist
    # -----------------------------------------------------------------------
    watchlist = _load_watchlist("config/watchlist.yaml")
    all_symbols: list[str] = list(
        set(watchlist.get("intraday", []) + watchlist.get("swing", []))
    )
    logger.info(
        "main.watchlist_loaded",
        intraday=watchlist.get("intraday", []),
        swing=watchlist.get("swing", []),
    )

    # -----------------------------------------------------------------------
    # 4. Wire all components inside the Groww async HTTP client context
    # -----------------------------------------------------------------------
    async with GrowwClient(
        api_key=settings.GROWW_API_KEY,
        api_secret=settings.GROWW_API_SECRET,
        base_url=settings.GROWW_BASE_URL,
        totp_secret=settings.GROWW_TOTP_SECRET,
    ) as groww:

        await groww.refresh_access_token()
        logger.info("main.groww_token_refreshed_at_startup")

        # --- Broker layer ---------------------------------------------------
        market_data = MarketDataService(client=groww)
        order_manager = OrderManager(
            client=groww,
            paper_trade=settings.PAPER_TRADE,
        )
        logger.info("main.broker_layer_ready", paper_trade=settings.PAPER_TRADE)

        # --- Data layer -----------------------------------------------------
        indicator_engine = IndicatorEngine()
        pattern_detector = PatternDetector()
        data_pipeline = DataPipeline(
            market_data=market_data,
            indicator_engine=indicator_engine,
            pattern_detector=pattern_detector,
            candle_store=candle_store,
        )
        logger.info("main.data_layer_ready")

        # --- News layer -----------------------------------------------------
        news_fetcher = RSSFetcher(watchlist=all_symbols)
        nse_announcements = NSEAnnouncementFetcher()
        news_summarizer = NewsSummarizer()
        logger.info("main.news_layer_ready")

        # --- Signal & risk layer --------------------------------------------
        llm_engine = LLMSignalEngine(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            max_tokens=settings.OPENAI_MAX_TOKENS,
        )
        risk_manager = RiskManager(
            total_capital=settings.TOTAL_CAPITAL,
            max_risk_per_trade_pct=settings.MAX_RISK_PER_TRADE_PCT,
            max_daily_loss_pct=settings.MAX_DAILY_LOSS_PCT,
            max_open_positions=settings.MAX_OPEN_POSITIONS,
            trade_store=trade_store,
            intraday_leverage=settings.INTRADAY_LEVERAGE,
        )
        sl_enforcer = StopLossEnforcer(
            order_manager=order_manager,
            trade_store=trade_store,
        )
        logger.info("main.risk_layer_ready")

        # --- Swing tracker --------------------------------------------------
        swing_tracker = SwingTracker(
            market_data=market_data,
            indicator_engine=indicator_engine,
            order_manager=order_manager,
            trade_store=trade_store,
            telegram_bot=None,  # patched below after TelegramBot is constructed
            max_hold_days=10,
        )

        # -----------------------------------------------------------------------
        # 5. Define approval / rejection callbacks
        #    These closures capture all the wired components.
        # -----------------------------------------------------------------------

        async def on_approve(signal, quantity: int) -> None:
            """
            Called by TelegramBot when the operator approves a signal.

            Steps
            -----
            1. Run the full pre-trade risk check.
            2. Compute final position size (may differ from quantity shown in card
               if risk limits tightened since card was sent).
            3. Build the entry OrderRequest (LIMIT for INTRADAY, DELIVERY for SWING).
            4. Place a bracket order (entry + GTT OCO stop/target).
            5. Save the position to the trade store.
            6. Send a confirmation message via Telegram.
            """
            symbol = signal.symbol
            logger.info(
                "main.on_approve",
                symbol=symbol,
                action=signal.action,
                quantity=quantity,
            )

            # -- 1. Pre-trade check ------------------------------------------
            # Fetch live Groww price at approval time as the actual fill price.
            # signal.entry_price (from yfinance close) is only used as fallback.
            entry_price = signal.entry_price
            try:
                quote = await market_data.get_live_quote(symbol, exchange="NSE")
                raw = quote.get("last_price") or (quote.get("ohlc") or {}).get("close") or 0.0
                ltp = float(raw)
                if ltp > 0:
                    entry_price = ltp
                    logger.info("main.on_approve.live_fill_price", symbol=symbol, ltp=ltp)
            except Exception as exc:
                logger.warning(
                    "main.on_approve.live_price_failed",
                    symbol=symbol,
                    error=str(exc),
                )

            approved, reason = await risk_manager.pre_trade_check(
                signal.to_dict(),
                entry_price,
            )
            if not approved:
                logger.warning(
                    "main.on_approve.pre_trade_blocked",
                    symbol=symbol,
                    reason=reason,
                )
                try:
                    await telegram_bot.send_message(
                        f"*Order Blocked — {symbol}*\n\nReason: {reason}"
                    )
                except Exception:
                    pass
                return

            # -- 2. Position size --------------------------------------------
            stop_loss = signal.stop_loss
            try:
                pos_size = risk_manager.compute_position_size(
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    trade_type=signal.trade_type,
                )
                final_qty = pos_size.quantity
            except ValueError:
                # entry == stop_loss edge case
                final_qty = max(1, quantity)

            # -- 3. Build OrderRequest ---------------------------------------
            trade_type = signal.trade_type.upper()
            action = signal.action.upper()
            target_1 = signal.target_1
            exchange = "NSE"

            product_type = (
                ProductType.DELIVERY if trade_type == "SWING" else ProductType.INTRADAY
            )

            entry_req = OrderRequest(
                symbol=symbol,
                exchange=exchange,
                transaction_type=action,
                quantity=final_qty,
                order_type=OrderType.LIMIT,
                product_type=product_type,
                price=entry_price,
                trigger_price=0.0,
                tag=f"algo-{trade_type.lower()}",
            )

            # -- 4. Place bracket order (entry + GTT OCO) --------------------
            try:
                order_result = await order_manager.place_bracket_order(
                    entry=entry_req,
                    stop_loss_price=stop_loss,
                    target_price=target_1,
                )
            except Exception as exc:
                logger.error(
                    "main.on_approve.bracket_order_failed",
                    symbol=symbol,
                    error=str(exc),
                    exc_info=True,
                )
                try:
                    await telegram_bot.send_message(
                        f"*Order Failed — {symbol}*\n\nError: {exc}"
                    )
                except Exception:
                    pass
                return

            entry_order_id = order_result.get("entry_order_id", "")
            gtt_id = order_result.get("gtt_id", "")

            # -- 5. Save position to trade store -----------------------------
            try:
                signal_id = await trade_store.save_signal(signal.to_dict())
                position_data = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "direction": action,
                    "trade_type": trade_type,
                    "quantity": final_qty,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "target_1": target_1,
                    "target_2": signal.target_2,
                    "entry_order_id": entry_order_id,
                    "gtt_id": gtt_id,
                    "status": "OPEN",
                }
                await trade_store.save_position(signal_id, position_data)
                logger.info(
                    "main.on_approve.position_saved",
                    symbol=symbol,
                    signal_id=signal_id,
                    entry_order_id=entry_order_id,
                )
            except Exception as exc:
                logger.error(
                    "main.on_approve.save_position_failed",
                    symbol=symbol,
                    error=str(exc),
                    exc_info=True,
                )

            # -- 6. Confirmation message -------------------------------------
            mode_tag = "PAPER" if settings.PAPER_TRADE else "LIVE"
            try:
                await telegram_bot.send_message(
                    f"*Order Placed — {symbol}* [{mode_tag}]\n\n"
                    f"Action: `{action}` | Type: `{trade_type}`\n"
                    f"Entry: ₹{entry_price:,.2f}  Qty: {final_qty}\n"
                    f"Stop-loss: ₹{stop_loss:,.2f}\n"
                    f"Target: ₹{target_1:,.2f}\n"
                    f"Order ID: `{entry_order_id}`"
                )
            except Exception as exc:
                logger.warning(
                    "main.on_approve.confirm_message_failed",
                    symbol=symbol,
                    error=str(exc),
                )

        async def on_reject(signal, quantity: int = 0) -> None:
            """
            Called by TelegramBot when the operator rejects a signal.

            Logs the rejection and records the decision in the trade store.
            """
            symbol = signal.symbol
            logger.info(
                "main.on_reject",
                symbol=symbol,
                action=signal.action,
            )

            try:
                signal_id = await trade_store.save_signal(signal.to_dict())
                await trade_store.save_signal_decision(signal_id, "REJECTED")
                logger.info(
                    "main.on_reject.decision_saved",
                    symbol=symbol,
                    signal_id=signal_id,
                )
            except Exception as exc:
                logger.error(
                    "main.on_reject.save_failed",
                    symbol=symbol,
                    error=str(exc),
                    exc_info=True,
                )

        # -----------------------------------------------------------------------
        # 6. Construct chat engine and context builder, then TelegramBot
        # -----------------------------------------------------------------------
        chat_engine = ChatEngine(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL)
        context_builder = ContextBuilder(trade_store=trade_store, news_cache=[])

        telegram_bot = TelegramBot(
            token=settings.TELEGRAM_BOT_TOKEN,
            chat_id=settings.TELEGRAM_CHAT_ID,
            on_approve=on_approve,
            on_reject=on_reject,
            capital=settings.TOTAL_CAPITAL,
            paper_trade=settings.PAPER_TRADE,
            chat_engine=chat_engine,
            context_builder=context_builder,
        )

        # Patch components that need telegram_bot (constructed after them)
        swing_tracker.telegram_bot = telegram_bot
        sl_enforcer.telegram_bot = telegram_bot

        # -----------------------------------------------------------------------
        # 7. Construct the scheduler
        # -----------------------------------------------------------------------
        scheduler = TradingScheduler(
            data_pipeline=data_pipeline,
            llm_engine=llm_engine,
            risk_manager=risk_manager,
            telegram_bot=telegram_bot,
            order_manager=order_manager,
            swing_tracker=swing_tracker,
            news_fetcher=news_fetcher,
            news_summarizer=news_summarizer,
            nse_announcements=nse_announcements,
            trade_store=trade_store,
            sl_enforcer=sl_enforcer,
            watchlist=watchlist,
            settings=settings,
            context_builder=context_builder,
            groww_client=groww,
        )

        # -----------------------------------------------------------------------
        # 8. Start Telegram bot and scheduler
        # -----------------------------------------------------------------------
        await telegram_bot.start()
        scheduler.start()
        logger.info("main.all_components_started")

        # -----------------------------------------------------------------------
        # 9. Send startup notification to Telegram
        # -----------------------------------------------------------------------
        startup_msg = (
            f"🤖 Trading bot started | "
            f"Mode: {mode_label} | "
            f"Capital: ₹{settings.TOTAL_CAPITAL:,.2f}"
        )
        try:
            await telegram_bot.send_message(startup_msg)
        except Exception as exc:
            logger.warning("main.startup_telegram_failed", error=str(exc))

        # -----------------------------------------------------------------------
        # 10. Signal handlers for graceful shutdown
        # -----------------------------------------------------------------------
        stop_event = asyncio.Event()

        def _signal_handler(sig_num: int, *_) -> None:
            logger.info("main.signal_received", signal=sig_num)
            # Schedule the shutdown coroutine on the running loop
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(stop_event.set)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_event_loop().add_signal_handler(
                    sig,
                    lambda s=sig: _signal_handler(s),
                )
            except NotImplementedError:
                # Windows does not support add_signal_handler on the event loop
                import signal as _signal_module
                _signal_module.signal(sig, _signal_handler)

        # -----------------------------------------------------------------------
        # 11. Run forever — wait for shutdown signal
        # -----------------------------------------------------------------------
        logger.info("main.running", hint="Send SIGINT or SIGTERM to stop")
        await stop_event.wait()

        # -----------------------------------------------------------------------
        # 12. Graceful teardown
        # -----------------------------------------------------------------------
        await shutdown(scheduler, telegram_bot, logger)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # asyncio.run() already propagates KeyboardInterrupt after cleanup.
        pass
