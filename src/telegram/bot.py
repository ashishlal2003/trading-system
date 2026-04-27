"""
Async Telegram bot for the algorithmic trading system.

Responsibilities
----------------
- Deliver formatted signal cards to the configured chat with APPROVE / REJECT
  inline buttons.
- Route operator decisions back to the trading engine via ``on_approve`` and
  ``on_reject`` callbacks.
- Provide lightweight operator commands (/start, /status, /positions, /help).

Dependencies
------------
- python-telegram-bot >= 21 (async API)

Usage
-----
    bot = TelegramBot(
        token=config.telegram_token,
        chat_id=config.telegram_chat_id,
        on_approve=order_manager.place_order,
        on_reject=lambda sig: None,
        capital=100_000.0,
    )
    await bot.start()   # begins polling in the background
    ...
    await bot.stop()    # graceful shutdown
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Callable, Awaitable, Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.constants import ParseMode

from src.utils.logger import get_logger
from src.telegram.formatters import format_signal_card, format_system_message

logger = get_logger(__name__)

# Indian Standard Time offset
_IST = timezone(timedelta(hours=5, minutes=30))

_BOT_COMMANDS = [
    BotCommand("start",     "Show welcome message"),
    BotCommand("status",    "Show system status"),
    BotCommand("positions", "Show open swing positions"),
    BotCommand("help",      "List available commands"),
]


def _ist_now_str() -> str:
    return datetime.now(_IST).strftime("%d %b %Y  %I:%M:%S %p IST")


class TelegramBot:
    """
    Async Telegram bot that bridges the trading engine and the operator.

    Parameters
    ----------
    token:
        Telegram Bot API token obtained from @BotFather.
    chat_id:
        Telegram chat / channel ID that receives all messages.
    on_approve:
        Async callable invoked when the operator taps APPROVE.
        Signature: ``async (signal: TradeSignal, quantity: int) -> None``.
    on_reject:
        Async callable invoked when the operator taps REJECT.
        Signature: ``async (signal: TradeSignal) -> None``.
    capital:
        Total available capital; forwarded to the signal formatter.
    paper_trade:
        When True, status messages advertise paper-trade mode.
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        on_approve: Callable[..., Awaitable[None]],
        on_reject:  Callable[..., Awaitable[None]],
        capital: float = 100_000.0,
        paper_trade: bool = True,
        chat_engine=None,
        context_builder=None,
    ) -> None:
        self.chat_id    = chat_id
        self.on_approve = on_approve
        self.on_reject  = on_reject
        self.capital    = capital
        self.paper_trade = paper_trade
        self.chat_engine = chat_engine
        self.context_builder = context_builder

        # Pending signals awaiting operator decision.
        # key -> {"signal": TradeSignal, "quantity": int}
        self._pending: dict[str, dict] = {}

        self.app = Application.builder().token(token).build()
        self._register_handlers()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        """Register all command and callback query handlers."""
        self.app.add_handler(CallbackQueryHandler(self._handle_callback))
        self.app.add_handler(CommandHandler("start",     self._handle_start))
        self.app.add_handler(CommandHandler("status",    self._handle_status))
        self.app.add_handler(CommandHandler("positions", self._handle_positions))
        self.app.add_handler(CommandHandler("help",      self._handle_help))
        # Free-text chat handler — must be registered AFTER command handlers
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_chat))
        logger.info("telegram_handlers_registered")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_signal(self, signal, quantity: int) -> None:
        """
        Send a formatted signal card to the operator with APPROVE / REJECT buttons.

        The signal is stored in ``_pending`` until the operator responds or it
        expires (stale keys are ignored at callback time).

        Parameters
        ----------
        signal:
            ``TradeSignal`` produced by the LLM engine.
        quantity:
            Number of shares / lots calculated by the risk manager.
        """
        # Unique key: symbol + millisecond timestamp
        def _get(attr: str, default="UNK"):
            if isinstance(signal, dict):
                return signal.get(attr, default)
            return getattr(signal, attr, default)

        symbol: str = str(_get("symbol", "UNK")).upper()
        ts_ms: int  = int(time.time() * 1000)
        key: str    = f"{symbol}_{ts_ms}"

        self._pending[key] = {"signal": signal, "quantity": quantity}

        text = format_signal_card(signal, capital=self.capital, quantity=quantity)

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ APPROVE", callback_data=f"APPROVE:{key}"),
                InlineKeyboardButton("❌ REJECT",  callback_data=f"REJECT:{key}"),
            ]
        ])

        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard,
            )
            logger.info("signal_sent", key=key, symbol=symbol, quantity=quantity)
        except Exception:
            logger.exception("send_signal_failed", key=key, symbol=symbol)
            # Remove from pending so it doesn't linger on error
            self._pending.pop(key, None)
            raise

    async def send_message(
        self,
        text: str,
        parse_mode: str = ParseMode.MARKDOWN,
    ) -> None:
        """
        Send a plain text message to the configured chat.

        Parameters
        ----------
        text:
            Message body (Markdown or plain text).
        parse_mode:
            Telegram parse mode; defaults to ``ParseMode.MARKDOWN``.
        """
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
            )
        except Exception:
            logger.exception("send_message_failed", length=len(text))
            raise

    # ------------------------------------------------------------------
    # Callback query handler
    # ------------------------------------------------------------------

    async def _handle_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """
        Handle APPROVE / REJECT inline button presses.

        Parses the callback_data (``"APPROVE:{key}"`` or ``"REJECT:{key}"``),
        looks up the pending signal, updates the message text to reflect the
        decision, and fires the appropriate callback.
        """
        query = update.callback_query
        await query.answer()  # acknowledge immediately to stop the loading spinner

        data: str = query.data or ""
        if ":" not in data:
            logger.warning("callback_malformed_data", data=data)
            return

        action_str, key = data.split(":", 1)
        action_str = action_str.upper()

        pending = self._pending.pop(key, None)
        if pending is None:
            # Signal already handled or expired
            logger.warning("callback_key_not_found", key=key, action=action_str)
            try:
                await query.edit_message_text(
                    text=(
                        query.message.text
                        + "\n\n_⚠️ This signal has already been handled or has expired._"
                    ),
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=None,
                )
            except Exception:
                logger.exception("callback_edit_message_failed", key=key)
            return

        signal   = pending["signal"]
        quantity = pending["quantity"]

        # Determine who pressed what
        user = update.effective_user
        operator = user.username or user.full_name if user else "operator"

        if action_str == "APPROVE":
            status_suffix = f"\n\n✅ *APPROVED* by @{operator}"
            logger.info("signal_approved", key=key, operator=operator, quantity=quantity)
            try:
                await self.on_approve(signal, quantity)
            except Exception:
                logger.exception("on_approve_callback_failed", key=key)
                status_suffix += "\n⚠️ _Order placement encountered an error — check logs._"

        elif action_str == "REJECT":
            status_suffix = f"\n\n❌ *REJECTED* by @{operator}"
            logger.info("signal_rejected", key=key, operator=operator)
            try:
                await self.on_reject(signal)
            except Exception:
                logger.exception("on_reject_callback_failed", key=key)

        else:
            logger.warning("callback_unknown_action", action=action_str, key=key)
            return

        # Edit the original signal card to remove buttons and add decision stamp
        try:
            await query.edit_message_text(
                text=query.message.text + status_suffix,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=None,   # remove inline keyboard
            )
        except Exception:
            logger.exception("callback_edit_message_failed", key=key)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _handle_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Send a welcome message listing available commands."""
        user = update.effective_user
        name = user.first_name if user else "Trader"
        text = (
            f"👋 *Welcome, {name}!*\n\n"
            "I am your algorithmic trading assistant for NSE/BSE markets.\n\n"
            "I will send you trade signals and wait for your approval before "
            "placing any orders.\n\n"
            "*Available commands:*\n"
            "  /status    — System health & mode\n"
            "  /positions — Open swing positions\n"
            "  /help      — This help text\n"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    async def _handle_status(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Send a brief system status line."""
        mode = "Paper" if self.paper_trade else "Live"
        pending_count = len(self._pending)
        ist_time = _ist_now_str()
        text = (
            f"✅ *System Running*\n\n"
            f"📋 *Mode:*            {mode} Trade\n"
            f"⏳ *Pending signals:* {pending_count}\n"
            f"💰 *Capital:*         ₹{self.capital:,.2f}\n"
            f"🕐 *Time (IST):*      {ist_time}"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    async def _handle_positions(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Placeholder for open positions (to be wired to the position tracker)."""
        text = (
            "📂 *Open Positions*\n\n"
            "_Use /status for now. Detailed position tracking is coming soon._"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    async def _handle_help(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """List all available bot commands."""
        text = (
            "🤖 *Trading Bot — Command Reference*\n\n"
            "/start      — Welcome message\n"
            "/status     — System health, mode, and IST time\n"
            "/positions  — Open swing positions summary\n"
            "/help       — Show this help message\n\n"
            "_Signals are sent automatically during market hours. "
            "Tap ✅ APPROVE or ❌ REJECT on each card._"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    async def _handle_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle free-text messages by forwarding them to the ChatEngine."""
        if not self.chat_engine or not self.context_builder:
            await update.message.reply_text("Chat mode not configured.")
            return

        # Only respond to messages from our chat_id (security)
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        user_msg = update.message.text
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        try:
            ctx = await self.context_builder.build()
            reply = await self.chat_engine.reply(user_msg, ctx)
            await update.message.reply_text(reply)
        except Exception as e:
            logger.error("chat_handler_failed", error=str(e))
            await update.message.reply_text("Something went wrong. Try again.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Initialise the Telegram application, register bot commands in the
        menu, and start long-polling.

        Call this once from the main async entry point.  Polling runs in a
        background task managed by ``python-telegram-bot``; this coroutine
        returns after the polling loop has been started.
        """
        logger.info("telegram_bot_starting")

        await self.app.initialize()

        # Register the slash-command menu visible in Telegram clients.
        try:
            await self.app.bot.set_my_commands(_BOT_COMMANDS)
            logger.info("bot_commands_registered", count=len(_BOT_COMMANDS))
        except Exception:
            logger.exception("set_bot_commands_failed")

        await self.app.updater.start_polling(drop_pending_updates=True)
        await self.app.start()

        # Notify the operator channel that the bot is online.
        try:
            startup_msg = format_system_message("Trading bot started successfully.", level="INFO")
            await self.send_message(startup_msg)
        except Exception:
            logger.warning("startup_notification_failed")

        logger.info("telegram_bot_started")

    async def stop(self) -> None:
        """
        Gracefully shut down the Telegram application.

        Stops the updater first (stops receiving new updates), then stops the
        application, then shuts down the underlying HTTP client.
        """
        logger.info("telegram_bot_stopping")

        # Notify before going offline (best-effort).
        try:
            shutdown_msg = format_system_message("Trading bot shutting down.", level="WARNING")
            await self.send_message(shutdown_msg)
        except Exception:
            logger.warning("shutdown_notification_failed")

        try:
            await self.app.updater.stop()
        except Exception:
            logger.exception("updater_stop_failed")

        try:
            await self.app.stop()
        except Exception:
            logger.exception("app_stop_failed")

        try:
            await self.app.shutdown()
        except Exception:
            logger.exception("app_shutdown_failed")

        logger.info("telegram_bot_stopped")
