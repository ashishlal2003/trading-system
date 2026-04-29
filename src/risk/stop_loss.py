from src.utils.logger import get_logger

logger = get_logger(__name__)


class StopLossEnforcer:
    """
    Monitors open positions and triggers an exit if the stop-loss level is
    breached.

    Also implements a trailing stop-loss rule: once a position has moved 1R
    in favour of the trade, the stop-loss is moved to the breakeven
    (entry price) so that no capital is ever lost on that trade.

    Intended to be called every 30 seconds by the scheduler.
    """

    def __init__(self, order_manager, trade_store, telegram_bot=None):
        self.order_manager = order_manager
        self.store = trade_store
        self.telegram_bot = telegram_bot

    # ------------------------------------------------------------------
    # Public entry-point (called by scheduler)
    # ------------------------------------------------------------------

    async def check_and_enforce(self, live_prices: dict[str, float]) -> None:
        """
        Iterate over every open position and apply stop-loss / trailing-stop
        logic based on the current market price.

        Args:
            live_prices: Mapping of symbol → latest market price.  Positions
                         whose symbol is absent from the dict are skipped
                         (the scheduler should log a separate warning in that
                         case).
        """
        open_positions = await self.store.get_open_positions()

        if not open_positions:
            logger.debug("stop_loss_enforcer_no_open_positions")
            return

        logger.debug("stop_loss_enforcer_checking", position_count=len(open_positions))

        for pos in open_positions:
            symbol = pos["symbol"]

            # Skip if we have no live price for this symbol.
            current_price = live_prices.get(symbol)
            if current_price is None:
                logger.warning(
                    "stop_loss_enforcer_price_missing",
                    symbol=symbol,
                    position_id=pos["id"],
                )
                continue

            direction = pos["direction"].upper()  # "BUY" or "SELL"
            stop_loss = float(pos["stop_loss"])

            # --- Stop-loss breach detection ---
            sl_breached = (
                (direction == "BUY" and current_price <= stop_loss)
                or (direction == "SELL" and current_price >= stop_loss)
            )

            if sl_breached:
                logger.warning(
                    "stop_loss_breached",
                    symbol=symbol,
                    position_id=pos["id"],
                    direction=direction,
                    stop_loss=stop_loss,
                    current_price=current_price,
                )
                await self._exit_position(pos, current_price, "SL_BREACH")
                continue

            # --- Target hit detection ---
            target_1 = pos.get("target_1")
            if target_1 is not None:
                target_1 = float(target_1)
                target_hit = (
                    (direction == "BUY" and current_price >= target_1)
                    or (direction == "SELL" and current_price <= target_1)
                )
                if target_hit:
                    logger.info(
                        "target_hit",
                        symbol=symbol,
                        position_id=pos["id"],
                        target=target_1,
                        current_price=current_price,
                    )
                    await self._exit_position(pos, current_price, "TARGET_HIT")
                    continue

            # --- Trailing stop-loss: move to breakeven at 1R gain ---
            sl_moved = bool(pos.get("sl_moved_to_breakeven", 0))
            if not sl_moved:
                r_multiple = self._compute_r_multiple(pos, current_price)

                if r_multiple >= 1.0:
                    entry_price = float(pos["entry_price"])
                    logger.info(
                        "trailing_sl_to_breakeven",
                        symbol=symbol,
                        position_id=pos["id"],
                        direction=direction,
                        r_multiple=round(r_multiple, 3),
                        new_sl=entry_price,
                    )
                    await self.store.update_stop_loss(pos["id"], entry_price)
                else:
                    logger.debug(
                        "stop_loss_check_ok",
                        symbol=symbol,
                        position_id=pos["id"],
                        current_price=current_price,
                        stop_loss=stop_loss,
                        r_multiple=round(r_multiple, 3),
                    )
            else:
                logger.debug(
                    "stop_loss_check_ok_breakeven_already_set",
                    symbol=symbol,
                    position_id=pos["id"],
                    current_price=current_price,
                    stop_loss=stop_loss,
                )

    # ------------------------------------------------------------------
    # R-multiple calculation
    # ------------------------------------------------------------------

    def _compute_r_multiple(self, pos: dict, current_price: float) -> float:
        """
        Calculate how many R-units (initial risk amounts) the position has
        moved in the trader's favour.

        Formula:
            risk   = |entry_price - stop_loss|
            gain   = (current_price - entry_price)  for BUY
                   = (entry_price - current_price)  for SELL
            result = gain / risk   (0.0 if risk == 0)

        Returns:
            float: R-multiple.  Positive means in profit, negative means
                   the trade is currently in the red (but hasn't yet hit SL).
        """
        entry_price = float(pos["entry_price"])
        stop_loss = float(pos["stop_loss"])
        direction = pos["direction"].upper()

        risk = abs(entry_price - stop_loss)
        if risk == 0:
            logger.warning(
                "compute_r_multiple_zero_risk",
                position_id=pos.get("id"),
                symbol=pos.get("symbol"),
            )
            return 0.0

        gain = (
            (current_price - entry_price)
            if direction == "BUY"
            else (entry_price - current_price)
        )

        return gain / risk

    # ------------------------------------------------------------------
    # Exit execution
    # ------------------------------------------------------------------

    async def _exit_position(self, pos: dict, price: float, reason: str) -> None:
        """
        Place a market exit order for the given position and close the
        internal record in the trade store.

        The exit order is placed in the direction opposite to the original
        entry.  ProductType is INTRADAY when the position was opened as an
        intraday trade, and DELIVERY for swing / delivery trades.

        Args:
            pos:    Position dict as returned by TradeStore.get_open_positions().
            price:  The current market price used to estimate exit P&L.
            reason: Textual tag recorded on the trade (e.g. "SL_BREACH").
        """
        # Import here to avoid a circular import at module load time.
        from src.broker.order_manager import OrderRequest, OrderType, ProductType

        symbol = pos["symbol"]
        direction = pos["direction"].upper()  # original entry direction
        quantity = int(pos["quantity"])
        entry_price = float(pos["entry_price"])
        trade_type = pos.get("trade_type", "INTRADAY").upper()

        # Exit is always opposite to entry direction.
        exit_direction = "SELL" if direction == "BUY" else "BUY"

        # ProductType mirrors the original order's product type.
        product_type = (
            ProductType.INTRADAY if trade_type == "INTRADAY" else ProductType.DELIVERY
        )

        # Estimate P&L for logging (actual P&L is stored by close_position).
        if direction == "BUY":
            estimated_pnl = (price - entry_price) * quantity
        else:
            estimated_pnl = (entry_price - price) * quantity

        req = OrderRequest(
            symbol=symbol,
            exchange=pos.get("exchange", "NSE"),
            transaction_type=exit_direction,
            quantity=quantity,
            order_type=OrderType.MARKET,
            product_type=product_type,
            price=0.0,  # MARKET order; price is ignored by the broker
            trigger_price=0.0,
            tag=f"risk-{reason.lower()}",
        )

        try:
            order_result = await self.order_manager.place_order(req)
            logger.info(
                "exit_order_placed",
                symbol=symbol,
                position_id=pos["id"],
                reason=reason,
                exit_direction=exit_direction,
                quantity=quantity,
                price=price,
                estimated_pnl=round(estimated_pnl, 2),
                order_id=order_result.get("order_id"),
                product_type=product_type.value,
            )
        except Exception as exc:
            # Log the error but still attempt to close the position in the store
            # to keep the internal state consistent.
            logger.error(
                "exit_order_failed",
                symbol=symbol,
                position_id=pos["id"],
                reason=reason,
                error=str(exc),
            )

        try:
            await self.store.close_position(pos["id"], price, reason)
            logger.info(
                "position_closed_in_store",
                symbol=symbol,
                position_id=pos["id"],
                exit_price=price,
                reason=reason,
                estimated_pnl=round(estimated_pnl, 2),
            )
        except Exception as exc:
            logger.error(
                "close_position_in_store_failed",
                symbol=symbol,
                position_id=pos["id"],
                reason=reason,
                error=str(exc),
            )

        if self.telegram_bot:
            try:
                mode_tag = "PAPER" if self.order_manager.paper_trade else "LIVE"
                pnl_str = f"+₹{estimated_pnl:.2f}" if estimated_pnl >= 0 else f"-₹{abs(estimated_pnl):.2f}"
                if reason == "TARGET_HIT":
                    emoji = "🎯"
                    label = "Target Hit"
                elif reason == "SL_BREACH":
                    emoji = "🛑"
                    label = "Stop Loss Hit"
                else:
                    emoji = "📤"
                    label = reason
                await self.telegram_bot.send_message(
                    f"{emoji} *{label} — {symbol}* [{mode_tag}]\n\n"
                    f"Direction: `{direction}` | Qty: {quantity}\n"
                    f"Entry: ₹{entry_price:.2f} → Exit: ₹{price:.2f}\n"
                    f"P&L: `{pnl_str}`"
                )
            except Exception as exc:
                logger.warning("exit_telegram_notify_failed", symbol=symbol, error=str(exc))
