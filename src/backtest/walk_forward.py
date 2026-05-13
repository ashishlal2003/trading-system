"""
Walk-Forward Validator
=======================
The gold standard for avoiding overfitting in algorithmic trading.

How it works
------------
1. Divide historical data into overlapping windows.
2. For each window:
   a. IN-SAMPLE  (e.g. 6 months): optimize strategy parameters.
   b. OUT-OF-SAMPLE (e.g. 1 month): test with those params on UNSEEN data.
3. Aggregate metrics across ALL out-of-sample windows.
4. If performance is consistent → strategy is robust, not curve-fitted.
   If performance collapses out-of-sample → overfitted, do not trade.

Analogy
-------
Imagine studying for 6 months (in-sample), then taking an exam on new questions
(out-of-sample). If you do well on the exam, you actually learned. If you only
memorised past answers (overfitting), you fail the exam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.metrics import MetricsEngine, MetricsResult
from src.strategy.base import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

# How many 5-min bars fit in one month (approximate)
# 6 h/day × 12 bars/h × 21 trading days/month ≈ 1512
_BARS_PER_MONTH = 1512


@dataclass
class WalkForwardFold:
    """Results for one in-sample/out-of-sample window pair."""
    fold_number: int
    in_sample_start: pd.Timestamp
    in_sample_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    in_sample_metrics: MetricsResult
    oos_metrics: MetricsResult
    optimised_params: dict[str, Any]


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward folds."""
    strategy_name: str
    symbol: str
    folds: list[WalkForwardFold]
    avg_oos_sharpe: float
    avg_oos_profit_factor: float
    avg_oos_win_rate: float
    avg_oos_max_drawdown: float
    avg_oos_return_pct: float
    total_oos_trades: int
    is_robust: bool
    verdict: str


class WalkForwardValidator:
    """
    Walk-forward validation for any BaseStrategy.

    Parameters
    ----------
    in_sample_months  : Months of data to optimise on (default 6).
    oos_months        : Months of data to test on (default 1).
    step_months       : How many months to advance the window each fold (default 1).
    initial_capital   : Capital to simulate per fold (default 100_000).
    param_grid        : Dict of parameter_name → list of values to try.
                        If empty, no optimisation — just run with current params.
    backtest_engine   : Pre-configured BacktestEngine (uses defaults if None).
    metrics_engine    : Pre-configured MetricsEngine (uses defaults if None).
    """

    def __init__(
        self,
        in_sample_months: int = 6,
        oos_months: int = 1,
        step_months: int = 1,
        initial_capital: float = 100_000.0,
        param_grid: dict[str, list] | None = None,
        backtest_engine: BacktestEngine | None = None,
        metrics_engine: MetricsEngine | None = None,
    ) -> None:
        self.in_sample_months = in_sample_months
        self.oos_months = oos_months
        self.step_months = step_months
        self.initial_capital = initial_capital
        self.param_grid = param_grid or {}
        self.bt_engine = backtest_engine or BacktestEngine()
        self.metrics_engine = metrics_engine or MetricsEngine()

    def validate(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str = "UNKNOWN",
    ) -> WalkForwardResult:
        """
        Run walk-forward validation on df.

        df must have: timestamp, open, high, low, close, volume — sorted ascending.
        Returns WalkForwardResult with per-fold and aggregate metrics.
        """
        df = df.reset_index(drop=True)
        in_sample_bars = self.in_sample_months * _BARS_PER_MONTH
        oos_bars = self.oos_months * _BARS_PER_MONTH
        step_bars = self.step_months * _BARS_PER_MONTH
        window_bars = in_sample_bars + oos_bars

        if len(df) < window_bars:
            logger.warning(
                "walk_forward.insufficient_data",
                rows=len(df),
                required=window_bars,
            )
            return self._empty_result(strategy.name, symbol)

        folds: list[WalkForwardFold] = []
        fold_num = 1
        start = 0

        while start + window_bars <= len(df):
            is_slice = df.iloc[start: start + in_sample_bars]
            oos_slice = df.iloc[start + in_sample_bars: start + window_bars]

            if oos_slice.empty:
                break

            # --- Optimise on in-sample ---
            best_params = self._optimise(is_slice, strategy, symbol)
            strategy.set_params(best_params)
            strategy.reset()

            # --- In-sample metrics (informational only) ---
            is_result = self.bt_engine.run(is_slice, strategy, self.initial_capital, symbol)
            is_metrics = self.metrics_engine.compute(is_result)

            # Reset strategy state for a fresh OOS run with same params
            strategy.set_params(best_params)
            strategy.reset()

            # --- Out-of-sample metrics (what matters) ---
            oos_result = self.bt_engine.run(oos_slice, strategy, self.initial_capital, symbol)
            oos_metrics = self.metrics_engine.compute(oos_result)

            fold = WalkForwardFold(
                fold_number=fold_num,
                in_sample_start=pd.Timestamp(is_slice.iloc[0]["timestamp"]),
                in_sample_end=pd.Timestamp(is_slice.iloc[-1]["timestamp"]),
                oos_start=pd.Timestamp(oos_slice.iloc[0]["timestamp"]),
                oos_end=pd.Timestamp(oos_slice.iloc[-1]["timestamp"]),
                in_sample_metrics=is_metrics,
                oos_metrics=oos_metrics,
                optimised_params=best_params,
            )
            folds.append(fold)

            logger.info(
                "walk_forward.fold_complete",
                fold=fold_num,
                oos_sharpe=oos_metrics.sharpe_ratio,
                oos_pf=oos_metrics.profit_factor,
                oos_trades=oos_metrics.total_trades,
                params=best_params,
            )

            fold_num += 1
            start += step_bars

        if not folds:
            return self._empty_result(strategy.name, symbol)

        return self._aggregate(folds, strategy.name, symbol)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _optimise(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str,
    ) -> dict[str, Any]:
        """
        Grid-search over param_grid on df. Returns best params by Sharpe.
        If param_grid is empty, returns current strategy params unchanged.
        """
        if not self.param_grid:
            return strategy.get_params()

        best_sharpe = float("-inf")
        best_params = strategy.get_params()

        for combo in self._param_combinations(self.param_grid):
            strategy.set_params(combo)
            strategy.reset()
            result = self.bt_engine.run(df, strategy, self.initial_capital, symbol)
            metrics = self.metrics_engine.compute(result)
            if metrics.sharpe_ratio > best_sharpe:
                best_sharpe = metrics.sharpe_ratio
                best_params = dict(combo)

        return best_params

    @staticmethod
    def _param_combinations(grid: dict[str, list]) -> list[dict[str, Any]]:
        """Cartesian product of all param values."""
        import itertools
        keys = list(grid.keys())
        values = list(grid.values())
        combos = []
        for vals in itertools.product(*values):
            combos.append(dict(zip(keys, vals)))
        return combos

    def _aggregate(
        self, folds: list[WalkForwardFold], strategy_name: str, symbol: str
    ) -> WalkForwardResult:
        oos_metrics_list = [f.oos_metrics for f in folds]

        avg_sharpe = sum(m.sharpe_ratio for m in oos_metrics_list) / len(oos_metrics_list)
        avg_pf = sum(m.profit_factor for m in oos_metrics_list) / len(oos_metrics_list)
        avg_wr = sum(m.win_rate_pct for m in oos_metrics_list) / len(oos_metrics_list)
        avg_dd = sum(m.max_drawdown_pct for m in oos_metrics_list) / len(oos_metrics_list)
        avg_ret = sum(m.total_return_pct for m in oos_metrics_list) / len(oos_metrics_list)
        total_oos_trades = sum(m.total_trades for m in oos_metrics_list)

        is_robust = (
            avg_sharpe >= 1.0
            and avg_pf >= 1.3
            and avg_dd <= 25.0
            and total_oos_trades >= 10
        )

        if is_robust:
            verdict = (
                f"ROBUST across {len(folds)} folds — "
                f"avg OOS Sharpe {avg_sharpe:.2f}, PF {avg_pf:.2f}, "
                f"WR {avg_wr:.1f}%, MaxDD {avg_dd:.1f}%"
            )
        else:
            verdict = (
                f"NOT ROBUST across {len(folds)} folds — "
                f"avg OOS Sharpe {avg_sharpe:.2f}, PF {avg_pf:.2f}, "
                f"WR {avg_wr:.1f}%, MaxDD {avg_dd:.1f}%, "
                f"OOS trades {total_oos_trades}"
            )

        return WalkForwardResult(
            strategy_name=strategy_name,
            symbol=symbol,
            folds=folds,
            avg_oos_sharpe=round(avg_sharpe, 3),
            avg_oos_profit_factor=round(avg_pf, 3),
            avg_oos_win_rate=round(avg_wr, 2),
            avg_oos_max_drawdown=round(avg_dd, 2),
            avg_oos_return_pct=round(avg_ret, 2),
            total_oos_trades=total_oos_trades,
            is_robust=is_robust,
            verdict=verdict,
        )

    @staticmethod
    def _empty_result(strategy_name: str, symbol: str) -> WalkForwardResult:
        return WalkForwardResult(
            strategy_name=strategy_name,
            symbol=symbol,
            folds=[],
            avg_oos_sharpe=0.0,
            avg_oos_profit_factor=0.0,
            avg_oos_win_rate=0.0,
            avg_oos_max_drawdown=0.0,
            avg_oos_return_pct=0.0,
            total_oos_trades=0,
            is_robust=False,
            verdict="NOT ROBUST: insufficient data for walk-forward validation",
        )
