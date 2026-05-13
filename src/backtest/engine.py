"""
Vectorized Backtesting Engine
==============================
Replays historical OHLCV data bar-by-bar through a strategy and simulates
trade fills with realistic costs. No ML, no prediction — pure rule replay.

Key design decisions
--------------------
- NO lookahead bias: strategy.evaluate(df, i) may only read df.iloc[:i+1].
- Fill price for entry = bar i close (signal on bar close, fill at close).
- Stop and target are checked on subsequent bars (bar i+1 onward) using the
  bar's high and low to determine which level was hit first.
- Transaction costs and slippage are applied symmetrically on entry and exit.
- One open position at a time per BacktestEngine.run() call (single-symbol).

Usage
-----
    engine = BacktestEngine()
    result = engine.run(df, strategy=ORBStrategy(), initial_capital=100_000)
    print(result.trades)
    print(result.equity_curve)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.strategy.base import BaseStrategy, SignalResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Trade:
    """Record of a single completed trade."""
    symbol: str
    entry_bar: int
    exit_bar: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    action: str          # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    quantity: int
    gross_pnl: float     # before costs
    net_pnl: float       # after all costs
    exit_reason: str     # "TARGET" | "STOP" | "EOD" | "END_OF_DATA"
    strategy_reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0


@dataclass
class BacktestResult:
    """Full output of a single backtest run."""
    symbol: str
    strategy_name: str
    initial_capital: float
    final_capital: float
    trades: list[Trade]
    equity_curve: pd.Series    # index = bar number, value = portfolio value
    params: dict[str, Any]

    @property
    def total_return_pct(self) -> float:
        return (self.final_capital - self.initial_capital) / self.initial_capital * 100

    @property
    def trade_count(self) -> int:
        return len(self.trades)


class BacktestEngine:
    """
    Single-symbol, bar-by-bar backtest engine.

    Parameters
    ----------
    transaction_cost_pct : Round-trip transaction cost as a fraction of trade
                           value. 0.0006 = 0.06% (brokerage + STT + NSE fees
                           realistic for NSE intraday).
    slippage_pct         : One-way slippage as fraction. 0.0002 = 0.02%.
                           Applied symmetrically on entry and exit.
    quantity             : Fixed number of shares per trade. If None, uses
                           all available capital / entry_price (whole shares).
    eod_exit_hour        : Force-close any open position at this hour IST.
    eod_exit_minute      : ... and this minute.
    """

    def __init__(
        self,
        transaction_cost_pct: float = 0.0006,
        slippage_pct: float = 0.0002,
        quantity: int | None = None,
        eod_exit_hour: int = 15,
        eod_exit_minute: int = 15,
        leverage: float = 1.0,
    ) -> None:
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.quantity = quantity
        self.eod_exit_hour = eod_exit_hour
        self.eod_exit_minute = eod_exit_minute
        self.leverage = leverage

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float,
        symbol: str = "UNKNOWN",
    ) -> BacktestResult:
        """
        Run the backtest.

        df must have columns: timestamp, open, high, low, close, volume.
        Rows must be sorted ascending by timestamp.

        Returns BacktestResult with all trades and equity curve.
        """
        df = df.reset_index(drop=True)
        capital = initial_capital
        equity_curve: list[float] = []
        trades: list[Trade] = []

        # Active position state
        in_position = False
        position_action: str = ""
        entry_bar: int = 0
        entry_price: float = 0.0
        stop_loss: float = 0.0
        target: float = 0.0
        qty: int = 0
        signal_reasoning: str = ""
        signal_meta: dict = {}

        for i in range(len(df)):
            row = df.iloc[i]
            ts = pd.Timestamp(row["timestamp"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])

            # ------------------------------------------------------------------
            # Check stop/target FIRST — takes priority over EOD on the same bar
            # ------------------------------------------------------------------
            if in_position:
                hit_stop = False
                hit_target = False

                if position_action == "BUY":
                    # Conservative: assume stop hits before target when both
                    # levels are touched in the same bar.
                    if bar_low <= stop_loss:
                        hit_stop = True
                    elif bar_high >= target:
                        hit_target = True
                else:  # SELL (short)
                    if bar_high >= stop_loss:
                        hit_stop = True
                    elif bar_low <= target:
                        hit_target = True

                if hit_stop or hit_target:
                    exit_price = stop_loss if hit_stop else target
                    reason = "STOP" if hit_stop else "TARGET"
                    trade = self._close_position(
                        symbol=symbol,
                        action=position_action,
                        entry_bar=entry_bar,
                        exit_bar=i,
                        entry_time=pd.Timestamp(df.iloc[entry_bar]["timestamp"]),
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        stop_loss=stop_loss,
                        target=target,
                        qty=qty,
                        exit_reason=reason,
                        reasoning=signal_reasoning,
                        metadata=signal_meta,
                    )
                    capital += trade.net_pnl
                    trades.append(trade)
                    in_position = False

            # ------------------------------------------------------------------
            # EOD force-exit if stop/target was not hit on this bar
            # ------------------------------------------------------------------
            if in_position:
                is_eod = (
                    ts.hour > self.eod_exit_hour
                    or (ts.hour == self.eod_exit_hour and ts.minute >= self.eod_exit_minute)
                )
                if is_eod:
                    trade = self._close_position(
                        symbol=symbol,
                        action=position_action,
                        entry_bar=entry_bar,
                        exit_bar=i,
                        entry_time=pd.Timestamp(df.iloc[entry_bar]["timestamp"]),
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=bar_close,
                        stop_loss=stop_loss,
                        target=target,
                        qty=qty,
                        exit_reason="EOD",
                        reasoning=signal_reasoning,
                        metadata=signal_meta,
                    )
                    capital += trade.net_pnl
                    trades.append(trade)
                    in_position = False

            # ------------------------------------------------------------------
            # Evaluate strategy signal for this bar (no position open)
            # ------------------------------------------------------------------
            if not in_position:
                signal: SignalResult = strategy.evaluate(df, i)

                if signal.is_actionable:
                    qty = self._compute_qty(capital, signal.entry_price)
                    if qty > 0:
                        # Apply entry slippage
                        if signal.action == "BUY":
                            fill_price = signal.entry_price * (1 + self.slippage_pct)
                        else:
                            fill_price = signal.entry_price * (1 - self.slippage_pct)

                        entry_cost = fill_price * qty * (self.transaction_cost_pct / 2)
                        capital -= entry_cost  # deduct entry half of round-trip cost

                        in_position = True
                        position_action = signal.action
                        entry_bar = i
                        entry_price = fill_price
                        stop_loss = signal.stop_loss
                        target = signal.target
                        signal_reasoning = signal.reasoning
                        signal_meta = signal.metadata

                        logger.debug(
                            "backtest.entry",
                            bar=i,
                            action=signal.action,
                            fill_price=round(fill_price, 2),
                            qty=qty,
                            stop=stop_loss,
                            target=target,
                        )

            equity_curve.append(capital)

        # ------------------------------------------------------------------
        # Force-close any position still open at end of data
        # ------------------------------------------------------------------
        if in_position:
            last_idx = len(df) - 1
            last_row = df.iloc[last_idx]
            trade = self._close_position(
                symbol=symbol,
                action=position_action,
                entry_bar=entry_bar,
                exit_bar=last_idx,
                entry_time=pd.Timestamp(df.iloc[entry_bar]["timestamp"]),
                exit_time=pd.Timestamp(last_row["timestamp"]),
                entry_price=entry_price,
                exit_price=float(last_row["close"]),
                stop_loss=stop_loss,
                target=target,
                qty=qty,
                exit_reason="END_OF_DATA",
                reasoning=signal_reasoning,
                metadata=signal_meta,
            )
            capital += trade.net_pnl
            trades.append(trade)
            equity_curve[-1] = capital

        logger.info(
            "backtest.complete",
            symbol=symbol,
            strategy=strategy.name,
            trades=len(trades),
            initial_capital=initial_capital,
            final_capital=round(capital, 2),
            return_pct=round((capital - initial_capital) / initial_capital * 100, 2),
        )

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy.name,
            initial_capital=initial_capital,
            final_capital=capital,
            trades=trades,
            equity_curve=pd.Series(equity_curve, name="equity"),
            params=strategy.get_params(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_qty(self, capital: float, entry_price: float) -> int:
        if self.quantity is not None:
            return self.quantity
        if entry_price <= 0:
            return 0
        buying_power = capital * self.leverage
        return max(1, int(buying_power // entry_price))

    def _close_position(
        self,
        symbol: str,
        action: str,
        entry_bar: int,
        exit_bar: int,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        target: float,
        qty: int,
        exit_reason: str,
        reasoning: str,
        metadata: dict,
    ) -> Trade:
        # Apply exit slippage (adverse to us)
        if action == "BUY":
            fill_exit = exit_price * (1 - self.slippage_pct)
            gross_pnl = (fill_exit - entry_price) * qty
        else:
            fill_exit = exit_price * (1 + self.slippage_pct)
            gross_pnl = (entry_price - fill_exit) * qty

        exit_cost = fill_exit * qty * (self.transaction_cost_pct / 2)
        net_pnl = gross_pnl - exit_cost

        logger.debug(
            "backtest.exit",
            bar=exit_bar,
            reason=exit_reason,
            exit_price=round(fill_exit, 2),
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
        )

        return Trade(
            symbol=symbol,
            entry_bar=entry_bar,
            exit_bar=exit_bar,
            entry_time=entry_time,
            exit_time=exit_time,
            action=action,
            entry_price=entry_price,
            exit_price=fill_exit,
            stop_loss=stop_loss,
            target=target,
            quantity=qty,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            strategy_reasoning=reasoning,
            metadata=metadata,
        )
