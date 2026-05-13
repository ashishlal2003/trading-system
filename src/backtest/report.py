"""
Backtest Report Generator
==========================
Formats BacktestResult + WalkForwardResult into human-readable reports
for both console output and Telegram messages.
"""

from __future__ import annotations

from src.backtest.engine import BacktestResult
from src.backtest.metrics import MetricsEngine, MetricsResult
from src.backtest.walk_forward import WalkForwardResult


class BacktestReporter:
    """
    Generates formatted reports from backtest and walk-forward results.

    Parameters
    ----------
    metrics_engine : MetricsEngine instance (uses defaults if None).
    """

    def __init__(self, metrics_engine: MetricsEngine | None = None) -> None:
        self._metrics = metrics_engine or MetricsEngine()

    def console_report(
        self,
        result: BacktestResult,
        wf_result: WalkForwardResult | None = None,
    ) -> str:
        """Return a multi-line plain-text report for terminal output."""
        m = self._metrics.compute(result)
        lines = [
            "=" * 60,
            f"  BACKTEST REPORT: {result.strategy_name} on {result.symbol}",
            "=" * 60,
            f"  Period       : {self._period(result)}",
            f"  Capital      : ₹{result.initial_capital:,.0f} → ₹{result.final_capital:,.0f}",
            f"  Total Return : {m.total_return_pct:+.2f}%",
            "-" * 60,
            f"  Sharpe Ratio : {m.sharpe_ratio:.3f}  {'✓' if m.sharpe_ratio >= 1.2 else '✗'}",
            f"  Max Drawdown : {m.max_drawdown_pct:.2f}%  {'✓' if m.max_drawdown_pct < 20 else '✗'}",
            f"  Profit Factor: {m.profit_factor:.3f}  {'✓' if m.profit_factor >= 1.5 else '✗'}",
            f"  Win Rate     : {m.win_rate_pct:.1f}%  ({m.winning_trades}W / {m.losing_trades}L / {m.total_trades} total)",
            f"  Expectancy   : ₹{m.expectancy:+.2f} per trade",
            f"  Avg Win/Loss : ₹{m.avg_win:.2f} / ₹{m.avg_loss:.2f}",
            f"  Max Consec L : {m.max_consecutive_losses}",
            f"  Calmar Ratio : {m.calmar_ratio:.3f}",
            "-" * 60,
            f"  VERDICT      : {m.verdict}",
        ]

        if wf_result:
            lines += [
                "",
                "  WALK-FORWARD VALIDATION",
                f"  Folds        : {len(wf_result.folds)}",
                f"  OOS Sharpe   : {wf_result.avg_oos_sharpe:.3f}",
                f"  OOS PF       : {wf_result.avg_oos_profit_factor:.3f}",
                f"  OOS Win Rate : {wf_result.avg_oos_win_rate:.1f}%",
                f"  OOS Max DD   : {wf_result.avg_oos_max_drawdown:.2f}%",
                f"  OOS Trades   : {wf_result.total_oos_trades}",
                f"  WF VERDICT   : {wf_result.verdict}",
            ]
            for fold in wf_result.folds:
                lines.append(
                    f"    Fold {fold.fold_number}: OOS {fold.oos_start.date()} – {fold.oos_end.date()} "
                    f"| Sharpe {fold.oos_metrics.sharpe_ratio:.2f} "
                    f"| PF {fold.oos_metrics.profit_factor:.2f} "
                    f"| WR {fold.oos_metrics.win_rate_pct:.0f}% "
                    f"| Trades {fold.oos_metrics.total_trades}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def telegram_report(
        self,
        result: BacktestResult,
        wf_result: WalkForwardResult | None = None,
    ) -> str:
        """Return a Telegram-formatted message (Markdown compatible)."""
        m = self._metrics.compute(result)

        def tick(cond: bool) -> str:
            return "✅" if cond else "❌"

        lines = [
            f"*Backtest: {result.strategy_name} — {result.symbol}*",
            f"Period: {self._period(result)}",
            f"Capital: ₹{result.initial_capital:,.0f} → ₹{result.final_capital:,.0f}  ({m.total_return_pct:+.1f}%)",
            "",
            f"{tick(m.sharpe_ratio >= 1.2)} Sharpe: `{m.sharpe_ratio:.2f}`",
            f"{tick(m.max_drawdown_pct < 20)} Max Drawdown: `{m.max_drawdown_pct:.1f}%`",
            f"{tick(m.profit_factor >= 1.5)} Profit Factor: `{m.profit_factor:.2f}`",
            f"{tick(50 <= m.win_rate_pct <= 80)} Win Rate: `{m.win_rate_pct:.1f}%` ({m.winning_trades}W/{m.losing_trades}L/{m.total_trades} trades)",
            f"Expectancy: `₹{m.expectancy:+.0f}` per trade",
            f"Max consecutive losses: `{m.max_consecutive_losses}`",
            "",
            f"*Verdict:* {m.verdict}",
        ]

        if wf_result:
            lines += [
                "",
                f"*Walk-Forward ({len(wf_result.folds)} folds):*",
                f"Avg OOS Sharpe: `{wf_result.avg_oos_sharpe:.2f}` | PF: `{wf_result.avg_oos_profit_factor:.2f}` | WR: `{wf_result.avg_oos_win_rate:.0f}%`",
                f"Avg OOS Max DD: `{wf_result.avg_oos_max_drawdown:.1f}%` | OOS Trades: `{wf_result.total_oos_trades}`",
                f"{'✅' if wf_result.is_robust else '❌'} *{wf_result.verdict}*",
            ]

        return "\n".join(lines)

    @staticmethod
    def _period(result: BacktestResult) -> str:
        equity = result.equity_curve
        if equity.empty:
            return "N/A"
        # equity index = bar numbers; use trades for date range if available
        if result.trades:
            start = result.trades[0].entry_time.date()
            end = result.trades[-1].exit_time.date()
            return f"{start} → {end}"
        return f"{len(equity)} bars"
