"""
Performance Metrics Engine
===========================
Computes all standard quant performance metrics from a BacktestResult.

Metrics
-------
- Sharpe Ratio        : Risk-adjusted return (annualised). Target ≥ 1.2.
- Max Drawdown (%)    : Worst peak-to-trough loss. Target < 20%.
- Profit Factor       : Gross profit / gross loss. Target ≥ 1.5.
- Win Rate (%)        : % of trades that are profitable. Target 50–65%.
- Expectancy (₹)      : Average net P&L per trade. Must be > 0.
- Total Return (%)    : Final capital vs initial capital.
- Calmar Ratio        : Annualised return / max drawdown.
- Consecutive Losses  : Longest losing streak (psychological health check).
- Average Win / Loss  : Average ₹ per winning / losing trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, Trade


@dataclass
class MetricsResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    expectancy: float        # average net P&L per trade in ₹
    avg_win: float
    avg_loss: float
    max_consecutive_losses: int
    calmar_ratio: float
    # Viability verdict
    is_viable: bool
    verdict: str             # human-readable summary

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class MetricsEngine:
    """
    Computes MetricsResult from a BacktestResult.

    Parameters
    ----------
    bars_per_year : Number of 5-minute bars in a trading year.
                    NSE: ~6 h/day × 12 bars/h × 250 days ≈ 18 000.
                    Used to annualise Sharpe and Calmar.
    risk_free_rate: Annual risk-free rate (decimal). Default 0.065 = 6.5% (India 10Y).
    """

    def __init__(
        self,
        bars_per_year: int = 18_000,
        risk_free_rate: float = 0.065,
    ) -> None:
        self.bars_per_year = bars_per_year
        self.risk_free_rate = risk_free_rate

    def compute(self, result: BacktestResult) -> MetricsResult:
        trades = result.trades
        equity = result.equity_curve

        if not trades:
            return self._zero_metrics(result)

        # ------------------------------------------------------------------
        # Win / loss split
        # ------------------------------------------------------------------
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]

        total = len(trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total * 100 if total > 0 else 0.0

        # ------------------------------------------------------------------
        # P&L aggregates
        # ------------------------------------------------------------------
        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = gross_profit / win_count if win_count > 0 else 0.0
        avg_loss = gross_loss / loss_count if loss_count > 0 else 0.0
        expectancy = sum(t.net_pnl for t in trades) / total

        # ------------------------------------------------------------------
        # Sharpe ratio (annualised, based on per-bar equity returns)
        # ------------------------------------------------------------------
        returns = equity.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            rf_per_bar = (1 + self.risk_free_rate) ** (1 / self.bars_per_year) - 1
            excess = returns - rf_per_bar
            sharpe = float(excess.mean() / excess.std() * math.sqrt(self.bars_per_year))
        else:
            sharpe = 0.0

        # ------------------------------------------------------------------
        # Max drawdown
        # ------------------------------------------------------------------
        max_dd_pct = self._max_drawdown(equity)

        # ------------------------------------------------------------------
        # Calmar ratio
        # ------------------------------------------------------------------
        years = len(equity) / self.bars_per_year
        try:
            annualised_return = (
                (result.final_capital / result.initial_capital) ** (1 / years) - 1
                if years > 0 else 0.0
            )
        except (OverflowError, ZeroDivisionError):
            annualised_return = 0.0
        calmar = annualised_return / (max_dd_pct / 100) if max_dd_pct > 0 else float("inf")

        # ------------------------------------------------------------------
        # Max consecutive losses
        # ------------------------------------------------------------------
        max_consec = self._max_consecutive_losses(trades)

        # ------------------------------------------------------------------
        # Viability verdict
        # ------------------------------------------------------------------
        is_viable, verdict = self._verdict(
            sharpe=sharpe,
            max_dd_pct=max_dd_pct,
            profit_factor=profit_factor,
            win_rate=win_rate,
            total_trades=total,
        )

        return MetricsResult(
            total_trades=total,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate_pct=round(win_rate, 2),
            total_return_pct=round(result.total_return_pct, 2),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown_pct=round(max_dd_pct, 2),
            profit_factor=round(profit_factor, 3),
            expectancy=round(expectancy, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            max_consecutive_losses=max_consec,
            calmar_ratio=round(calmar, 3),
            is_viable=is_viable,
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        """Max peak-to-trough drawdown as a positive percentage."""
        if equity.empty:
            return 0.0
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        return float(abs(drawdown.min()) * 100)

    @staticmethod
    def _max_consecutive_losses(trades: list[Trade]) -> int:
        max_streak = 0
        streak = 0
        for t in trades:
            if t.net_pnl <= 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    @staticmethod
    def _verdict(
        sharpe: float,
        max_dd_pct: float,
        profit_factor: float,
        win_rate: float,
        total_trades: int,
    ) -> tuple[bool, str]:
        issues = []
        flags = []

        # "Too few trades" is always the first check — insufficient data
        # cannot be overridden by an overfitting flag.
        if total_trades < 20:
            return False, f"NEEDS WORK: too few trades ({total_trades}) for statistical significance"

        if sharpe < 1.2:
            issues.append(f"Sharpe {sharpe:.2f} below 1.2 threshold")
        if max_dd_pct > 20:
            issues.append(f"max drawdown {max_dd_pct:.1f}% exceeds 20% limit")
        if profit_factor < 1.5:
            issues.append(f"profit factor {profit_factor:.2f} below 1.5 threshold")

        # Overfitting red flags (too-good-to-be-true)
        if sharpe > 3.0:
            flags.append(f"Sharpe {sharpe:.2f} > 3.0 — likely overfitted")
        if profit_factor > 4.0:
            flags.append(f"profit factor {profit_factor:.2f} > 4.0 — likely overfitted")
        if win_rate > 80:
            flags.append(f"win rate {win_rate:.1f}% > 80% — suspicious")

        if flags:
            return False, "OVERFITTED: " + "; ".join(flags)
        if issues:
            return False, "NEEDS WORK: " + "; ".join(issues)
        return True, (
            f"VIABLE — Sharpe {sharpe:.2f}, MaxDD {max_dd_pct:.1f}%, "
            f"PF {profit_factor:.2f}, WR {win_rate:.1f}%"
        )

    def _zero_metrics(self, result: BacktestResult) -> MetricsResult:
        return MetricsResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            total_return_pct=round(result.total_return_pct, 2),
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_consecutive_losses=0,
            calmar_ratio=0.0,
            is_viable=False,
            verdict="NEEDS WORK: no trades generated",
        )
