import math
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionSize:
    quantity: int
    capital_at_risk: float
    risk_pct: float
    entry_price: float
    stop_loss: float
    max_loss: float  # quantity * |entry - stop_loss|


class RiskManager:
    """
    Enforces all pre-trade and post-trade risk rules.
    Must be called before every order is approved.
    """

    def __init__(
        self,
        total_capital: float,
        max_risk_per_trade_pct: float,
        max_daily_loss_pct: float,
        max_open_positions: int,
        trade_store,  # TradeStore instance
        intraday_leverage: float = 5.0,
    ):
        self.total_capital = total_capital
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_positions = max_open_positions
        self.trade_store = trade_store
        self.intraday_leverage = intraday_leverage

        # Absolute capital amount at which daily trading halts.
        self.max_daily_loss_limit = -(total_capital * (max_daily_loss_pct / 100))

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_position_size(
        self, entry_price: float, stop_loss: float, trade_type: str = "INTRADAY"
    ) -> PositionSize:
        """
        Calculate the number of shares to trade bounded by two constraints:

        1. Risk constraint  — if stopped out, loss <= max_risk_per_trade_pct% of capital
           qty_risk = floor(capital * risk_pct / risk_per_share)

        2. Capital constraint — can't deploy more than an equal share of buying power
           buying_power     = capital * leverage   (intraday) | capital (swing)
           max_per_trade    = buying_power / max_open_positions
           qty_capital      = floor(max_per_trade / entry_price)

        Final quantity = min(qty_risk, qty_capital)

        Raises:
            ValueError: if entry_price == stop_loss (zero risk per share).
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            raise ValueError(
                f"entry_price and stop_loss are equal ({entry_price}); "
                "cannot compute position size with zero risk per share."
            )

        # --- Constraint 1: risk-based quantity ---
        risk_amount = self.total_capital * (self.max_risk_per_trade_pct / 100)
        qty_risk = max(1, math.floor(risk_amount / risk_per_share))

        # --- Constraint 2: capital-based quantity ---
        leverage = self.intraday_leverage if trade_type.upper() == "INTRADAY" else 1.0
        max_capital_per_trade = (self.total_capital * leverage) / self.max_open_positions
        qty_capital = max(1, math.floor(max_capital_per_trade / entry_price))

        quantity = min(qty_risk, qty_capital)

        capital_at_risk = quantity * risk_per_share
        risk_pct = (capital_at_risk / self.total_capital) * 100
        capital_deployed = quantity * entry_price

        logger.info(
            "position_sizing",
            trade_type=trade_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_per_share=risk_per_share,
            risk_amount=risk_amount,
            qty_risk=qty_risk,
            qty_capital=qty_capital,
            quantity=quantity,
            capital_deployed=round(capital_deployed, 2),
            capital_at_risk=round(capital_at_risk, 2),
            risk_pct=round(risk_pct, 4),
            leverage=leverage,
        )

        return PositionSize(
            quantity=quantity,
            capital_at_risk=capital_at_risk,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_loss=stop_loss,
            max_loss=capital_at_risk,
        )

    # ------------------------------------------------------------------
    # Gate checks
    # ------------------------------------------------------------------

    async def can_trade(self) -> tuple[bool, str]:
        """
        Returns (True, "") when trading is permitted, or
        (False, <human-readable reason>) when a circuit-breaker is active.

        Check 1 – Daily loss limit
            If cumulative closed-trade P&L today has reached or exceeded the
            configured max_daily_loss_pct, halt all new orders.

        Check 2 – Max concurrent open positions
            If the number of currently open positions equals or exceeds
            max_open_positions, reject new entries.
        """
        # --- Check 1: daily loss ---
        daily_pnl = await self.trade_store.get_daily_pnl()
        if daily_pnl <= self.max_daily_loss_limit:
            reason = (
                f"Daily loss limit reached: pnl={daily_pnl:.2f}, "
                f"limit={self.max_daily_loss_limit:.2f} "
                f"({self.max_daily_loss_pct}% of capital)"
            )
            logger.warning("can_trade_blocked_daily_loss", daily_pnl=daily_pnl, limit=self.max_daily_loss_limit)
            return False, reason

        # --- Check 2: open position cap ---
        open_count = await self.trade_store.count_open_positions()
        if open_count >= self.max_open_positions:
            reason = (
                f"Max open positions reached: open={open_count}, "
                f"limit={self.max_open_positions}"
            )
            logger.warning("can_trade_blocked_max_positions", open_count=open_count, limit=self.max_open_positions)
            return False, reason

        logger.debug("can_trade_ok", daily_pnl=daily_pnl, open_count=open_count)
        return True, ""

    # ------------------------------------------------------------------
    # Signal-level validation helpers
    # ------------------------------------------------------------------

    def validate_signal_rr(
        self,
        entry: float,
        stop_loss: float,
        target: float,
        min_rr: float = 1.5,
    ) -> bool:
        """
        Return True when the signal's reward-to-risk ratio meets the minimum
        threshold.

        Args:
            entry:     Proposed entry price.
            stop_loss: Stop-loss level.
            target:    Profit target level.
            min_rr:    Minimum acceptable R:R ratio (default 1.5).

        Returns:
            True if (|target - entry| / |entry - stop_loss|) >= min_rr,
            False if risk == 0 or ratio is below threshold.
        """
        reward = abs(target - entry)
        risk = abs(entry - stop_loss)

        if risk <= 0:
            logger.warning("validate_rr_zero_risk", entry=entry, stop_loss=stop_loss)
            return False

        rr = reward / risk
        passed = rr >= min_rr

        logger.debug(
            "validate_signal_rr",
            entry=entry,
            stop_loss=stop_loss,
            target=target,
            reward=round(reward, 4),
            risk=round(risk, 4),
            rr=round(rr, 4),
            min_rr=min_rr,
            passed=passed,
        )
        return passed

    def validate_entry_proximity(
        self,
        entry_price: float,
        current_price: float,
        max_pct: float = 0.3,
    ) -> bool:
        """
        Return True when the current market price is within *max_pct* percent
        of the signal's entry price.  This prevents chasing moves that have
        already happened.

        Args:
            entry_price:   Signal's recommended entry price.
            current_price: Live market price of the instrument.
            max_pct:       Maximum allowable deviation in percent (default 0.3).

        Returns:
            True if abs(entry_price - current_price) / current_price * 100 <= max_pct.
        """
        if current_price == 0:
            logger.warning("validate_entry_proximity_zero_price")
            return False

        deviation_pct = abs(entry_price - current_price) / current_price * 100
        passed = deviation_pct <= max_pct

        logger.debug(
            "validate_entry_proximity",
            entry_price=entry_price,
            current_price=current_price,
            deviation_pct=round(deviation_pct, 4),
            max_pct=max_pct,
            passed=passed,
        )
        return passed

    # ------------------------------------------------------------------
    # Composite pre-trade gate
    # ------------------------------------------------------------------

    async def pre_trade_check(
        self, signal, current_price: float
    ) -> tuple[bool, str]:
        """
        Full pre-trade validation pipeline.  All checks must pass before an
        order is sent to the broker.

        Steps:
            1. can_trade()                  – daily loss + position cap
            2. validate_signal_rr()         – reward:risk ratio
            3. validate_entry_proximity()   – price hasn't moved away (INTRADAY only)

        Args:
            signal:        Signal dict with keys: entry_price, stop_loss,
                           target_1, trade_type (and optionally direction).
            current_price: Live market price at time of signal evaluation.

        Returns:
            (True, "")                on full approval.
            (False, <reason string>)  on first failure.
        """
        # 1. System-level gate
        tradeable, reason = await self.can_trade()
        if not tradeable:
            logger.info("pre_trade_check_failed", step="can_trade", reason=reason)
            return False, reason

        entry_price = signal["entry_price"]
        stop_loss = signal["stop_loss"]
        target = signal["target_1"]
        trade_type = signal.get("trade_type", "INTRADAY")

        # 2. Reward-to-risk check
        rr_ok = self.validate_signal_rr(entry_price, stop_loss, target)
        if not rr_ok:
            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            rr = (reward / risk) if risk > 0 else 0.0
            reason = (
                f"R:R ratio too low: {rr:.2f} < 1.5 "
                f"(entry={entry_price}, sl={stop_loss}, target={target})"
            )
            logger.info("pre_trade_check_failed", step="validate_signal_rr", reason=reason)
            return False, reason

        # 3. Entry proximity check – only meaningful for INTRADAY signals
        if trade_type == "INTRADAY":
            proximity_ok = self.validate_entry_proximity(entry_price, current_price)
            if not proximity_ok:
                deviation_pct = abs(entry_price - current_price) / current_price * 100
                reason = (
                    f"Entry price too far from current market: "
                    f"entry={entry_price}, current={current_price}, "
                    f"deviation={deviation_pct:.4f}% > 0.3%"
                )
                logger.info(
                    "pre_trade_check_failed",
                    step="validate_entry_proximity",
                    reason=reason,
                )
                return False, reason

        logger.info(
            "pre_trade_check_passed",
            symbol=signal.get("symbol", "unknown"),
            trade_type=trade_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            current_price=current_price,
        )
        return True, ""
