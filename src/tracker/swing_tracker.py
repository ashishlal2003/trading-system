"""
SwingTracker — monitors overnight swing positions and runs morning checks.

Called once per trading day at 09:00 IST by the scheduler.  For each open
SWING position the tracker:
  1. Fetches the latest daily candles and computes indicators.
  2. Evaluates exit conditions in priority order.
  3. Exits the position (DELIVERY market order) or sends a morning update.
"""

from datetime import datetime, date
from typing import Optional

import pytz

from src.utils.logger import get_logger

logger = get_logger(__name__)
IST = pytz.timezone("Asia/Kolkata")


class SwingTracker:
    """
    Monitors overnight swing positions and executes or reports on them each
    morning at 09:00 IST.

    Parameters
    ----------
    market_data:
        ``MarketDataService`` — used to fetch the latest daily candles.
    indicator_engine:
        ``IndicatorEngine`` — computes technical indicators on candle data.
    order_manager:
        ``OrderManager`` — places DELIVERY market exit orders.
    trade_store:
        ``TradeStore`` — reads/writes position records.
    telegram_bot:
        ``TelegramBot`` — delivers morning updates and exit notifications.
    max_hold_days:
        Maximum number of calendar days to hold a swing position before
        forcing an exit.  Defaults to 10.
    """

    def __init__(
        self,
        market_data,
        indicator_engine,
        order_manager,
        trade_store,
        telegram_bot,
        max_hold_days: int = 10,
    ) -> None:
        self.market_data = market_data
        self.indicator_engine = indicator_engine
        self.order_manager = order_manager
        self.trade_store = trade_store
        self.telegram_bot = telegram_bot
        self.max_hold_days = max_hold_days

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_morning_check(self) -> None:
        """
        Main entry point called at 09:00 IST each trading day.

        Steps
        -----
        1. Load all open swing positions from the trade store.
        2. For each position: fetch the last 5 daily candles, compute
           indicators, and evaluate exit conditions.
        3. Exit if a condition is met; otherwise send a morning update.

        Each position is processed inside its own try/except so that a
        failure on one position never prevents the others from being checked.
        """
        logger.info("swing_tracker.morning_check_start")

        try:
            positions = await self.trade_store.get_open_swing_positions()
        except Exception as exc:
            logger.error(
                "swing_tracker.failed_to_load_positions",
                error=str(exc),
                exc_info=True,
            )
            return

        if not positions:
            logger.info("swing_tracker.no_open_swing_positions")
            return

        logger.info(
            "swing_tracker.checking_positions",
            count=len(positions),
        )

        for pos in positions:
            symbol = pos.get("symbol", "UNKNOWN")
            try:
                await self._check_position(pos)
            except Exception as exc:
                logger.error(
                    "swing_tracker.position_check_error",
                    symbol=symbol,
                    position_id=pos.get("id"),
                    error=str(exc),
                    exc_info=True,
                )

        logger.info("swing_tracker.morning_check_complete")

    # ------------------------------------------------------------------
    # Internal: per-position logic
    # ------------------------------------------------------------------

    async def _check_position(self, pos: dict) -> None:
        """
        Evaluate a single swing position, exit or update as needed.

        Exit conditions are checked in priority order:
          1. MAX_HOLD_DAYS — force exit after max_hold_days calendar days.
          2. TARGET_HIT    — price has reached target_1.
          3. SL_BREACH     — price has breached the stop-loss.
        """
        symbol: str = pos.get("symbol", "UNKNOWN")
        exchange: str = pos.get("exchange", "NSE")
        direction: str = (pos.get("direction") or "BUY").upper()
        entry_price: float = float(pos.get("entry_price", 0.0))
        stop_loss: float = float(pos.get("stop_loss", 0.0))
        target_1: float = float(pos.get("target_1", 0.0))

        logger.info(
            "swing_tracker.checking_position",
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
        )

        # --- Fetch recent daily candles (last 5 days) ---
        from datetime import timedelta

        to_date = datetime.utcnow()
        from_date = to_date - timedelta(days=7)  # extra buffer for weekends

        df = await self.market_data.get_candles(
            symbol=symbol,
            exchange=exchange,
            interval="1d",
            from_date=from_date,
            to_date=to_date,
        )

        # Trim to last 5 rows and compute indicators
        if not df.empty and len(df) > 5:
            df = df.tail(5).reset_index(drop=True)

        indicators = self.indicator_engine.compute(df=df, symbol=symbol) if not df.empty else None

        # Determine current price: prefer indicators.close, fall back to live
        current_price: Optional[float] = None

        if indicators is not None:
            current_price = indicators.close

        if current_price is None or current_price == 0.0:
            try:
                quote = await self.market_data.get_live_quote(symbol, exchange)
                current_price = float(
                    quote.get("ltp", quote.get("lastPrice", 0.0))
                )
            except Exception as exc:
                logger.warning(
                    "swing_tracker.live_quote_failed",
                    symbol=symbol,
                    error=str(exc),
                )
                current_price = entry_price  # safe fallback — no exit triggered

        hold_days: int = self._hold_days(pos)

        # ----------------------------------------------------------------
        # Exit condition checks — priority order
        # ----------------------------------------------------------------

        # 1. MAX_HOLD_DAYS
        if hold_days >= self.max_hold_days:
            logger.info(
                "swing_tracker.exit_max_hold_days",
                symbol=symbol,
                hold_days=hold_days,
                max_hold_days=self.max_hold_days,
            )
            await self._exit_swing(pos, current_price, "MAX_HOLD_DAYS")
            return

        # 2. TARGET_HIT
        target_hit = (
            (direction == "BUY" and current_price >= target_1)
            or (direction == "SELL" and current_price <= target_1)
        )
        if target_hit:
            logger.info(
                "swing_tracker.exit_target_hit",
                symbol=symbol,
                direction=direction,
                current_price=current_price,
                target_1=target_1,
            )
            await self._exit_swing(pos, current_price, "TARGET_HIT")
            return

        # 3. SL_BREACH
        sl_breached = (
            (direction == "BUY" and current_price <= stop_loss)
            or (direction == "SELL" and current_price >= stop_loss)
        )
        if sl_breached:
            logger.info(
                "swing_tracker.exit_sl_breach",
                symbol=symbol,
                direction=direction,
                current_price=current_price,
                stop_loss=stop_loss,
            )
            await self._exit_swing(pos, current_price, "SL_BREACH")
            return

        # ----------------------------------------------------------------
        # No exit — send morning update
        # ----------------------------------------------------------------
        await self._send_morning_update(pos, current_price, hold_days)

    # ------------------------------------------------------------------
    # Exit helper
    # ------------------------------------------------------------------

    async def _exit_swing(self, pos: dict, price: float, reason: str) -> None:
        """
        Place a DELIVERY market exit order, close the position in the trade
        store, and send a Telegram exit notification.

        Parameters
        ----------
        pos:
            Position dict from ``TradeStore.get_open_swing_positions()``.
        price:
            Current market price used for P&L estimation and exit order.
        reason:
            Exit reason tag (``"MAX_HOLD_DAYS"``, ``"TARGET_HIT"``, or
            ``"SL_BREACH"``).
        """
        from src.broker.order_manager import OrderRequest, OrderType, ProductType

        symbol: str = pos.get("symbol", "UNKNOWN")
        direction: str = (pos.get("direction") or "BUY").upper()
        quantity: int = int(pos.get("quantity", 1))
        position_id: int = pos.get("id", 0)

        exit_direction = "SELL" if direction == "BUY" else "BUY"
        pnl: float = self._compute_pnl(pos, price)

        req = OrderRequest(
            symbol=symbol,
            exchange=pos.get("exchange", "NSE"),
            transaction_type=exit_direction,
            quantity=quantity,
            order_type=OrderType.MARKET,
            product_type=ProductType.DELIVERY,
            price=0.0,
            trigger_price=0.0,
            tag=f"swing-exit-{reason.lower()}",
        )

        # Place exit order (errors are logged but do not prevent store update)
        order_result = {}
        try:
            order_result = await self.order_manager.place_order(req)
            logger.info(
                "swing_tracker.exit_order_placed",
                symbol=symbol,
                position_id=position_id,
                reason=reason,
                exit_direction=exit_direction,
                price=price,
                pnl=round(pnl, 2),
                order_id=order_result.get("order_id"),
            )
        except Exception as exc:
            logger.error(
                "swing_tracker.exit_order_failed",
                symbol=symbol,
                position_id=position_id,
                reason=reason,
                error=str(exc),
                exc_info=True,
            )

        # Close position in trade store
        try:
            await self.trade_store.close_position(position_id, price, reason)
        except Exception as exc:
            logger.error(
                "swing_tracker.close_position_failed",
                symbol=symbol,
                position_id=position_id,
                error=str(exc),
                exc_info=True,
            )

        # Telegram exit notification
        hold_days: int = self._hold_days(pos)
        entry_price: float = float(pos.get("entry_price", 0.0))
        pnl_sign = "+" if pnl >= 0 else ""

        message = (
            f"*Swing Exit — {symbol}*\n\n"
            f"Reason: `{reason}`\n"
            f"Direction: `{direction}`\n"
            f"Entry: ₹{entry_price:,.2f}  →  Exit: ₹{price:,.2f}\n"
            f"Qty: {quantity}\n"
            f"P&L: `{pnl_sign}₹{pnl:,.2f}`\n"
            f"Hold days: {hold_days}\n"
            f"Order ID: `{order_result.get('order_id', 'N/A')}`"
        )

        try:
            await self.telegram_bot.send_message(message)
        except Exception as exc:
            logger.warning(
                "swing_tracker.exit_notification_failed",
                symbol=symbol,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Morning update (no exit)
    # ------------------------------------------------------------------

    async def _send_morning_update(
        self, pos: dict, current_price: float, hold_days: int
    ) -> None:
        """
        Send a Telegram morning status update for a position that has not
        triggered an exit condition.
        """
        symbol: str = pos.get("symbol", "UNKNOWN")
        direction: str = (pos.get("direction") or "BUY").upper()
        entry_price: float = float(pos.get("entry_price", 0.0))
        stop_loss: float = float(pos.get("stop_loss", 0.0))
        target_1: float = float(pos.get("target_1", 0.0))
        quantity: int = int(pos.get("quantity", 1))

        pnl: float = self._compute_pnl(pos, current_price)
        pnl_pct: float = (pnl / (entry_price * quantity) * 100) if entry_price and quantity else 0.0
        pnl_sign = "+" if pnl >= 0 else ""

        message = (
            f"*Morning Update — {symbol}* ({direction})\n\n"
            f"Current price: ₹{current_price:,.2f}\n"
            f"Entry: ₹{entry_price:,.2f}\n"
            f"Stop-loss: ₹{stop_loss:,.2f}\n"
            f"Target: ₹{target_1:,.2f}\n"
            f"Qty: {quantity}\n"
            f"P&L: `{pnl_sign}₹{pnl:,.2f}` ({pnl_sign}{pnl_pct:.2f}%)\n"
            f"Hold days: {hold_days} / {self.max_hold_days}"
        )

        try:
            await self.telegram_bot.send_message(message)
        except Exception as exc:
            logger.warning(
                "swing_tracker.morning_update_failed",
                symbol=symbol,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Pure helpers
    # ------------------------------------------------------------------

    def _compute_pnl(self, pos: dict, current_price: float) -> float:
        """
        Direction-aware unrealised P&L calculation.

        Returns
        -------
        float
            (current_price - entry_price) * qty  for BUY
            (entry_price - current_price) * qty  for SELL
        """
        direction: str = (pos.get("direction") or "BUY").upper()
        entry_price: float = float(pos.get("entry_price", 0.0))
        quantity: int = int(pos.get("quantity", 1))

        if direction == "BUY":
            return (current_price - entry_price) * quantity
        else:  # SELL / SHORT
            return (entry_price - current_price) * quantity

    def _hold_days(self, pos: dict) -> int:
        """
        Return the number of calendar days elapsed since the position's
        ``entry_date``.

        Handles both ISO-string and datetime ``entry_date`` values.

        Returns
        -------
        int
            Days since entry.  Returns 0 when entry_date cannot be parsed.
        """
        entry_date_raw = pos.get("entry_date")
        if entry_date_raw is None:
            return 0

        try:
            if isinstance(entry_date_raw, datetime):
                entry_dt = entry_date_raw.date()
            elif isinstance(entry_date_raw, date):
                entry_dt = entry_date_raw
            else:
                # Parse ISO-8601 string (with or without time component)
                entry_dt = datetime.fromisoformat(str(entry_date_raw)).date()

            return (date.today() - entry_dt).days
        except Exception as exc:
            logger.warning(
                "swing_tracker.hold_days_parse_error",
                entry_date_raw=entry_date_raw,
                error=str(exc),
            )
            return 0
