"""
VWAP Mean Reversion Strategy
==============================
Complements ORB — works when the market is trending and price pulls back to
VWAP before resuming. Aims for a bounce off VWAP with a tight stop.

Rules
-----
Trend filter (LONG side):
  EMA9 > EMA21 > EMA50  →  confirmed UPTREND
  Price must have been ABOVE VWAP in the most recent N bars before the touch.

Entry (LONG):
  Close is within touch_band_pct % of VWAP (touching it)
  AND close crosses back above VWAP (rebound candle)
  AND RSI < rsi_long_threshold (default 45) — oversold within uptrend
  AND relative_volume > 1.0 — at least average volume
  AND trend == UPTREND

Stop loss:
  VWAP − (atr_14 × atr_sl_mult)  — 1 ATR below VWAP

Target:
  entry_price + risk × target_r   (risk = entry − stop_loss)

SHORT side:
  Mirror — DOWNTREND, RSI > rsi_short_threshold (default 55),
  price bounces down off VWAP, stop above VWAP.

Hard exits:
  No new entries after cutoff time (default 15:10 IST).
  One trade per day.
"""

from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy, SignalResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MIN = 15


class VWAPReversionStrategy(BaseStrategy):
    """
    VWAP Mean Reversion — buy the dip to VWAP in an uptrend.

    Parameters
    ----------
    touch_band_pct      : Price is considered "touching" VWAP if within this
                          percentage of VWAP (default 0.15%).
    rsi_long_threshold  : RSI must be below this for a long entry (default 45).
    rsi_short_threshold : RSI must be above this for a short entry (default 55).
    atr_sl_mult         : ATR multiplier for stop-loss distance (default 1.0).
    target_r            : Target as a multiple of risk (default 1.5).
    min_rvol            : Minimum relative volume (default 1.0).
    warmup_bars         : Minimum bars needed before evaluating (default 50).
    cutoff_hour         : No new entries at or after this hour (default 15).
    cutoff_minute       : No new entries at or after this minute (default 10).
    """

    def __init__(
        self,
        touch_band_pct: float = 0.15,
        rsi_long_threshold: float = 45.0,
        rsi_short_threshold: float = 55.0,
        atr_sl_mult: float = 1.0,
        target_r: float = 1.5,
        min_rvol: float = 0.3,
        warmup_bars: int = 50,
        cutoff_hour: int = 15,
        cutoff_minute: int = 10,
    ) -> None:
        self.touch_band_pct = touch_band_pct
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_short_threshold = rsi_short_threshold
        self.atr_sl_mult = atr_sl_mult
        self.target_r = target_r
        self.min_rvol = min_rvol
        self.warmup_bars = warmup_bars
        self.cutoff_hour = cutoff_hour
        self.cutoff_minute = cutoff_minute

        self._traded_date: object | None = None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "VWAP-Reversion"

    def get_params(self) -> dict:
        return {
            "touch_band_pct": self.touch_band_pct,
            "rsi_long_threshold": self.rsi_long_threshold,
            "rsi_short_threshold": self.rsi_short_threshold,
            "atr_sl_mult": self.atr_sl_mult,
            "target_r": self.target_r,
            "min_rvol": self.min_rvol,
            "warmup_bars": self.warmup_bars,
            "cutoff_hour": self.cutoff_hour,
            "cutoff_minute": self.cutoff_minute,
        }

    def set_params(self, params: dict) -> None:
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def reset(self) -> None:
        """Reset per-day state — called before each backtest fold."""
        self._traded_date = None

    def evaluate(self, df: pd.DataFrame, bar_idx: int) -> SignalResult:
        no_trade = SignalResult(
            action="NO_TRADE",
            entry_price=0.0,
            stop_loss=0.0,
            target=0.0,
            reasoning="",
        )

        if bar_idx < self.warmup_bars:
            no_trade.reasoning = "Warming up — insufficient history"
            return no_trade

        row = df.iloc[bar_idx]
        ts = pd.Timestamp(row["timestamp"])
        bar_date = ts.date()
        bar_total = ts.hour * 60 + ts.minute

        # Market hours gate: NSE opens 9:15 AM, no pre-market data
        if bar_total < 9 * 60 + 15:
            no_trade.reasoning = "Before market open (9:15 AM IST)"
            return no_trade

        # Cutoff check
        if bar_total >= self.cutoff_hour * 60 + self.cutoff_minute:
            no_trade.reasoning = "Past entry cutoff time"
            return no_trade

        # One trade per day
        if self._traded_date == bar_date:
            no_trade.reasoning = "Already traded today"
            return no_trade

        # ------------------------------------------------------------------
        # Compute indicators from history (no lookahead)
        # ------------------------------------------------------------------
        window = df.iloc[: bar_idx + 1]
        close_series = window["close"].astype(float)

        ema9 = self._ema(close_series, 9).iloc[-1]
        ema21 = self._ema(close_series, 21).iloc[-1]
        ema50 = self._ema(close_series, 50).iloc[-1]
        rsi = self._rsi(close_series, 14).iloc[-1]
        atr = self._atr(window, 14).iloc[-1]
        vwap = self._session_vwap(window, bar_date)
        rvol = self._rvol(window, 20)

        close = float(row["close"])
        prev_close = float(df.iloc[bar_idx - 1]["close"]) if bar_idx > 0 else close

        if vwap is None or vwap <= 0 or pd.isna(atr) or atr <= 0:
            no_trade.reasoning = "VWAP or ATR unavailable"
            return no_trade

        # ------------------------------------------------------------------
        # Trend classification — EMA9 vs EMA21 is sufficient on 5-min bars;
        # EMA50 spans 4+ hours of intraday data and rarely aligns strictly.
        # ------------------------------------------------------------------
        if ema9 > ema21:
            trend = "UPTREND"
        elif ema9 < ema21:
            trend = "DOWNTREND"
        else:
            no_trade.reasoning = f"Sideways — EMA9={ema9:.2f} == EMA21={ema21:.2f}"
            return no_trade

        # ------------------------------------------------------------------
        # VWAP touch: within touch_band_pct of VWAP
        # ------------------------------------------------------------------
        band = vwap * (self.touch_band_pct / 100.0)
        touching_vwap = abs(close - vwap) <= band or abs(float(row["low"]) - vwap) <= band

        if not touching_vwap:
            no_trade.reasoning = (
                f"Price {close:.2f} not touching VWAP {vwap:.2f} (band ±{band:.2f})"
            )
            return no_trade

        # ------------------------------------------------------------------
        # Volume confirmation
        # ------------------------------------------------------------------
        if rvol < self.min_rvol:
            no_trade.reasoning = f"Low volume: RVOL {rvol:.2f} < {self.min_rvol}"
            return no_trade

        # ------------------------------------------------------------------
        # LONG setup: uptrend + price rebounds off VWAP
        # ------------------------------------------------------------------
        if trend == "UPTREND" and rsi < self.rsi_long_threshold:
            # Rebound confirmation: previous close was at/below VWAP, current close above
            prev_below = prev_close <= vwap * (1 + self.touch_band_pct / 100)
            curr_above = close > vwap
            if prev_below and curr_above:
                stop = vwap - atr * self.atr_sl_mult
                risk = close - stop
                if risk <= 0:
                    no_trade.reasoning = "Invalid stop: risk ≤ 0"
                    return no_trade
                target = close + risk * self.target_r
                self._traded_date = bar_date
                logger.info(
                    "vwap_reversion.long_signal",
                    close=close,
                    vwap=round(vwap, 2),
                    rsi=round(rsi, 2),
                    rvol=round(rvol, 2),
                    stop=round(stop, 2),
                    target=round(target, 2),
                )
                return SignalResult(
                    action="BUY",
                    entry_price=close,
                    stop_loss=stop,
                    target=target,
                    reasoning=(
                        f"VWAP Reversion LONG: price {close:.2f} rebounded off VWAP {vwap:.2f}, "
                        f"RSI {rsi:.1f} < {self.rsi_long_threshold}, RVOL {rvol:.2f}x, UPTREND"
                    ),
                    metadata={"vwap": vwap, "rsi": rsi, "rvol": rvol, "atr": atr, "trend": trend},
                )

        # ------------------------------------------------------------------
        # SHORT setup: downtrend + price bounces down off VWAP
        # ------------------------------------------------------------------
        if trend == "DOWNTREND" and rsi > self.rsi_short_threshold:
            prev_above = prev_close >= vwap * (1 - self.touch_band_pct / 100)
            curr_below = close < vwap
            if prev_above and curr_below:
                stop = vwap + atr * self.atr_sl_mult
                risk = stop - close
                if risk <= 0:
                    no_trade.reasoning = "Invalid stop: risk ≤ 0"
                    return no_trade
                target = close - risk * self.target_r
                self._traded_date = bar_date
                logger.info(
                    "vwap_reversion.short_signal",
                    close=close,
                    vwap=round(vwap, 2),
                    rsi=round(rsi, 2),
                    stop=round(stop, 2),
                    target=round(target, 2),
                )
                return SignalResult(
                    action="SELL",
                    entry_price=close,
                    stop_loss=stop,
                    target=target,
                    reasoning=(
                        f"VWAP Reversion SHORT: price {close:.2f} rejected off VWAP {vwap:.2f}, "
                        f"RSI {rsi:.1f} > {self.rsi_short_threshold}, RVOL {rvol:.2f}x, DOWNTREND"
                    ),
                    metadata={"vwap": vwap, "rsi": rsi, "rvol": rvol, "atr": atr, "trend": trend},
                )

        no_trade.reasoning = (
            f"Touching VWAP but conditions not met: trend={trend}, RSI={rsi:.1f}"
        )
        return no_trade

    # ------------------------------------------------------------------
    # Indicator helpers (pure pandas, no lookahead — operate on slices)
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        # Wilder's smoothing: alpha = 1/period
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        # Wilder's smoothing: alpha = 1/period
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def _session_vwap(df: pd.DataFrame, bar_date: object) -> float | None:
        mask = pd.to_datetime(df["timestamp"]).dt.date == bar_date
        day = df[mask]
        if day.empty:
            return None
        tp = (day["high"].astype(float) + day["low"].astype(float) + day["close"].astype(float)) / 3.0
        vol = day["volume"].astype(float)
        total_vol = vol.sum()
        return float((tp * vol).sum() / total_vol) if total_vol > 0 else None

    @staticmethod
    def _rvol(df: pd.DataFrame, window: int = 20) -> float:
        # Use intraday-only bars so early morning volume isn't diluted by
        # yesterday's higher mid-session bars.
        if df.empty:
            return 1.0
        bar_date = pd.Timestamp(df.iloc[-1]["timestamp"]).date()
        today = df[pd.to_datetime(df["timestamp"]).dt.date == bar_date]["volume"].astype(float)
        if len(today) < 2:
            return 1.0
        current = float(today.iloc[-1])
        avg = float(today.iloc[:-1].mean())
        return current / avg if avg > 0 else 1.0
