"""
Rule-Based Signal Engine
=========================
Runs per-symbol ADX to pick the right strategy each scan:
  - ADX >= adx_trend_threshold (default 25) → trending → ORB
  - ADX <  adx_trend_threshold              → choppy   → VWAP Reversion

Both strategies share the same TradeSignal output shape used everywhere
(Telegram, RiskManager, OrderManager).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.signals.llm_engine import TradeSignal
from src.strategy.base import BaseStrategy, SignalResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RuleEngine:
    """
    Selects between ORB and VWAP Reversion based on per-symbol ADX,
    then generates a TradeSignal.

    Parameters
    ----------
    strategy        : Primary strategy (ORB). Also used as fallback if
                      vwap_strategy is None.
    vwap_strategy   : Secondary strategy for choppy/low-ADX days.
    trade_type      : "INTRADAY" or "SWING".
    adx_trend_threshold : ADX above this → trending → use ORB.
                          ADX below this → choppy  → use VWAP Reversion.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        vwap_strategy: BaseStrategy | None = None,
        trade_type: str = "INTRADAY",
        adx_trend_threshold: float = 25.0,
    ) -> None:
        self.strategy = strategy          # ORB template (backwards-compat attr)
        self.vwap_strategy = vwap_strategy
        self.trade_type = trade_type
        self.adx_trend_threshold = adx_trend_threshold
        self._last_strategy: BaseStrategy = strategy
        # Per-symbol strategy instances — each symbol must have its own state
        # (ORB tracks _or_high/_or_low/_traded_today per instance).
        self._orb_instances:  dict[str, BaseStrategy] = {}
        self._vwap_instances: dict[str, BaseStrategy] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str = "NSE",
        live_price: float | None = None,
    ) -> TradeSignal:
        try:
            if df is None or df.empty:
                return self._no_trade(symbol, "Empty DataFrame — no data available")

            adx = self._compute_adx(df)
            selected = self._select_strategy(adx, symbol)
            self._last_strategy = selected

            # Use last bar; if it has zero volume it's an incomplete in-progress
            # candle from yfinance — step back to the last complete bar.
            bar_idx = len(df) - 1
            if bar_idx > 0 and float(df.iloc[bar_idx]["volume"]) == 0:
                bar_idx -= 1

            result: SignalResult = selected.evaluate(df, bar_idx)

            # Use live price as entry when available; recalculate target so
            # stop stays at its absolute level and R:R multiple is preserved.
            entry_price = result.entry_price
            target_1 = result.target
            rr = result.risk_reward

            if live_price and result.is_actionable:
                entry_price = live_price
                actual_risk = abs(live_price - result.stop_loss)
                if actual_risk > 0:
                    if result.action == "BUY":
                        target_1 = live_price + actual_risk * rr
                    else:
                        target_1 = live_price - actual_risk * rr

            signal = TradeSignal(
                symbol=symbol,
                action=result.action,
                trade_type=self.trade_type,
                entry_price=entry_price,
                stop_loss=result.stop_loss,
                target_1=target_1,
                target_2=None,
                confidence=self._compute_confidence(result),
                risk_reward_ratio=round(rr, 3),
                reasoning=result.reasoning,
                key_risks=self._key_risks(result),
                invalidation_condition=self._invalidation(result),
                generated_at=datetime.now(),
            )

            logger.info(
                "rule_engine.signal",
                symbol=symbol,
                strategy=selected.name,
                adx=round(adx, 1),
                action=signal.action,
                entry=signal.entry_price,
                sl=signal.stop_loss,
                target=signal.target_1,
                rr=signal.risk_reward_ratio,
                is_actionable=signal.is_actionable,
            )

            return signal

        except Exception as exc:
            logger.error(
                "rule_engine.error",
                symbol=symbol,
                error=str(exc),
                exc_info=True,
            )
            return self._no_trade(symbol, f"Rule engine error: {type(exc).__name__}")

    def batch_generate(
        self,
        scan_results: list[dict[str, Any]],
    ) -> list[TradeSignal]:
        actionable: list[TradeSignal] = []

        for item in scan_results:
            symbol = item.get("symbol", "UNKNOWN")
            exchange = item.get("exchange", "NSE")
            df = item.get("df")
            live_price = item.get("live_price")

            signal = self.generate_signal(df, symbol, exchange, live_price)

            if signal.is_actionable:
                actionable.append(signal)
                logger.info(
                    "rule_engine.batch_actionable",
                    symbol=symbol,
                    action=signal.action,
                    rr=signal.risk_reward_ratio,
                )
            else:
                logger.debug(
                    "rule_engine.batch_skipped",
                    symbol=symbol,
                    reasoning=signal.reasoning[:80],
                )

        logger.info(
            "rule_engine.batch_complete",
            total=len(scan_results),
            actionable=len(actionable),
        )
        return actionable

    # ------------------------------------------------------------------
    # Strategy selection — per-symbol instances to avoid shared state
    # ------------------------------------------------------------------

    def _select_strategy(self, adx: float, symbol: str) -> BaseStrategy:
        if self.vwap_strategy is not None and adx < self.adx_trend_threshold:
            if symbol not in self._vwap_instances:
                params = self.vwap_strategy.get_params()
                self._vwap_instances[symbol] = self.vwap_strategy.__class__(**params)
            return self._vwap_instances[symbol]
        if symbol not in self._orb_instances:
            params = self.strategy.get_params()
            self._orb_instances[symbol] = self.strategy.__class__(**params)
        return self._orb_instances[symbol]

    # ------------------------------------------------------------------
    # ADX computation (Wilder's smoothing, no lookahead)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> float:
        """
        Returns ADX value on the last bar. Returns 0.0 if insufficient data.
        High ADX (>25) = trending. Low ADX (<25) = choppy/ranging.
        """
        if len(df) < period * 2:
            return 0.0

        high  = df["high"].astype(float).reset_index(drop=True)
        low   = df["low"].astype(float).reset_index(drop=True)
        close = df["close"].astype(float).reset_index(drop=True)

        prev_close = close.shift(1)
        prev_high  = high.shift(1)
        prev_low   = low.shift(1)

        tr = pd.concat(
            [high - low,
             (high - prev_close).abs(),
             (low  - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        plus_dm  = (high - prev_high).clip(lower=0)
        minus_dm = (prev_low - low).clip(lower=0)
        # Only keep the larger of the two; zero out ties
        plus_dm  = plus_dm.where(plus_dm  > minus_dm, 0.0)
        minus_dm = minus_dm.where(minus_dm > plus_dm,  0.0)

        # Wilder's smoothing uses alpha = 1/period (not span=period)
        _alpha = 1.0 / period
        atr14      = tr.ewm(alpha=_alpha, adjust=False).mean()
        plus_di14  = 100 * plus_dm.ewm(alpha=_alpha, adjust=False).mean() / atr14
        minus_di14 = 100 * minus_dm.ewm(alpha=_alpha, adjust=False).mean() / atr14

        di_sum  = (plus_di14 + minus_di14).replace(0, float("nan"))
        dx      = 100 * (plus_di14 - minus_di14).abs() / di_sum
        adx_ser = dx.ewm(alpha=_alpha, adjust=False).mean()

        val = adx_ser.iloc[-1]
        return float(val) if not pd.isna(val) else 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _no_trade(symbol: str, reason: str) -> TradeSignal:
        return TradeSignal(
            symbol=symbol,
            action="NO_TRADE",
            trade_type="INTRADAY",
            entry_price=0.0,
            stop_loss=0.0,
            target_1=0.0,
            target_2=None,
            confidence=0.0,
            risk_reward_ratio=0.0,
            reasoning=reason,
            key_risks=["No signal"],
            invalidation_condition="N/A",
            generated_at=datetime.now(),
        )

    @staticmethod
    def _compute_confidence(result: SignalResult) -> float:
        if not result.is_actionable:
            return 0.0
        rr = result.risk_reward
        rvol = result.metadata.get("rvol", 1.0) if result.metadata else 1.0
        if rr >= 3.0:
            base = 1.0
        elif rr >= 2.5:
            base = 0.85
        elif rr >= 2.0:
            base = 0.75
        elif rr >= 1.5:
            base = 0.65
        else:
            base = 0.50
        vol_bonus = min(0.15, (rvol - 1.0) * 0.05) if rvol > 1.0 else 0.0
        return min(1.0, round(base + vol_bonus, 2))

    @staticmethod
    def _key_risks(result: SignalResult) -> list[str]:
        risks = ["Stop loss breach", "Adverse news or gap"]
        if result.action == "BUY":
            risks.append("Trend reversal below stop")
        elif result.action == "SELL":
            risks.append("Short squeeze above stop")
        return risks

    @staticmethod
    def _invalidation(result: SignalResult) -> str:
        if result.action == "BUY":
            return f"Close below stop loss {result.stop_loss:.2f}"
        if result.action == "SELL":
            return f"Close above stop loss {result.stop_loss:.2f}"
        return "N/A"
