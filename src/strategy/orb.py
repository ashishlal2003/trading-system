"""
Opening Range Breakout (ORB) Strategy
======================================
One of the most studied and mechanical intraday strategies. No curve-fitting,
no subjectivity — if the price and volume conditions are met, the signal fires.

Rules
-----
Setup (computed once per day at 9:30 AM IST):
  - Opening Range = first 15-minute candle (9:15–9:30 AM)
  - ORH = that candle's high
  - ORL = that candle's low

Entry (evaluated on every 5-min candle after 9:30 AM):
  LONG  → candle closes ABOVE ORH
           AND relative_volume > rvol_threshold (default 1.5)
           AND close > VWAP
           AND no prior trade today on this symbol

  SHORT → candle closes BELOW ORL
           AND relative_volume > rvol_threshold
           AND close < VWAP
           AND no prior trade today on this symbol

Stop Loss:
  LONG  → ORL  (the other side of the opening range)
  SHORT → ORH

Target:
  entry_price ± (range_width × target_r)  where target_r = 2.0 by default

Hard exits:
  No new entries after cutoff_hour:cutoff_minute IST (default 15:10).
  Max one trade per symbol per day.
"""

from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy, SignalResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# IST market open
_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MIN = 15

# Opening range ends at 9:30 — that's 1 completed 15-min candle after open
_OR_END_HOUR = 9
_OR_END_MIN = 30


class ORBStrategy(BaseStrategy):
    """
    Opening Range Breakout strategy for 5-minute NSE intraday data.

    Parameters
    ----------
    or_minutes      : Width of the opening range in minutes (default 15).
    rvol_threshold  : Minimum relative volume for entry (default 1.5).
    target_r        : Target as a multiple of the range width (default 2.0).
    cutoff_hour     : No new entries at or after this hour IST (default 15).
    cutoff_minute   : No new entries at or after this minute of cutoff_hour (default 10).
    """

    def __init__(
        self,
        or_minutes: int = 15,
        rvol_threshold: float = 1.5,
        target_r: float = 2.0,
        cutoff_hour: int = 15,
        cutoff_minute: int = 10,
    ) -> None:
        self.or_minutes = or_minutes
        self.rvol_threshold = rvol_threshold
        self.target_r = target_r
        self.cutoff_hour = cutoff_hour
        self.cutoff_minute = cutoff_minute

        # Per-day state — reset when the calendar date changes
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._or_date: object | None = None   # datetime.date
        self._traded_today: bool = False

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"ORB-{self.or_minutes}m"

    def get_params(self) -> dict:
        return {
            "or_minutes": self.or_minutes,
            "rvol_threshold": self.rvol_threshold,
            "target_r": self.target_r,
            "cutoff_hour": self.cutoff_hour,
            "cutoff_minute": self.cutoff_minute,
        }

    def set_params(self, params: dict) -> None:
        self.or_minutes = params.get("or_minutes", self.or_minutes)
        self.rvol_threshold = params.get("rvol_threshold", self.rvol_threshold)
        self.target_r = params.get("target_r", self.target_r)
        self.cutoff_hour = params.get("cutoff_hour", self.cutoff_hour)
        self.cutoff_minute = params.get("cutoff_minute", self.cutoff_minute)

    def evaluate(self, df: pd.DataFrame, bar_idx: int) -> SignalResult:
        """
        Evaluate the ORB signal at bar bar_idx.

        Only reads df.iloc[:bar_idx+1] — no lookahead.
        """
        no_trade = SignalResult(
            action="NO_TRADE",
            entry_price=0.0,
            stop_loss=0.0,
            target=0.0,
            reasoning="",
        )

        row = df.iloc[bar_idx]
        ts = pd.Timestamp(row["timestamp"])
        bar_date = ts.date()
        bar_hour = ts.hour
        bar_minute = ts.minute

        # Pre-market guard: ignore any bars before 9:15 AM IST
        if bar_hour * 60 + bar_minute < _MARKET_OPEN_HOUR * 60 + _MARKET_OPEN_MIN:
            no_trade.reasoning = "Before market open (9:15 AM IST)"
            return no_trade

        # ------------------------------------------------------------------
        # 1. Reset daily state when a new trading day starts
        # ------------------------------------------------------------------
        if bar_date != self._or_date:
            self._or_high = None
            self._or_low = None
            self._or_date = bar_date
            self._traded_today = False

        # ------------------------------------------------------------------
        # 2. Build the opening range from all bars in the first or_minutes
        # ------------------------------------------------------------------
        if self._or_high is None:
            self._or_high, self._or_low = self._compute_opening_range(
                df, bar_idx, bar_date
            )

        if self._or_high is None:
            # Not enough bars yet to establish the opening range
            no_trade.reasoning = "Opening range not yet established"
            return no_trade

        # ------------------------------------------------------------------
        # 3. Skip bars inside the opening range window and after cutoff
        # ------------------------------------------------------------------
        bar_minutes_since_open = (bar_hour - _MARKET_OPEN_HOUR) * 60 + (bar_minute - _MARKET_OPEN_MIN)
        if bar_minutes_since_open < self.or_minutes:
            no_trade.reasoning = "Still inside opening range window"
            return no_trade

        cutoff_total = self.cutoff_hour * 60 + self.cutoff_minute
        bar_total = bar_hour * 60 + bar_minute
        if bar_total >= cutoff_total:
            no_trade.reasoning = "Past entry cutoff time"
            return no_trade

        # ------------------------------------------------------------------
        # 4. One trade per symbol per day
        # ------------------------------------------------------------------
        if self._traded_today:
            no_trade.reasoning = "Already traded today"
            return no_trade

        # ------------------------------------------------------------------
        # 5. Extract bar values
        # ------------------------------------------------------------------
        close = float(row["close"])
        volume = float(row["volume"])

        # Relative volume: average all completed bars before this one today so
        # the current bar's own volume doesn't dilute the comparison.
        prev_vols = [
            float(df.iloc[i]["volume"])
            for i in range(bar_idx)
            if pd.Timestamp(df.iloc[i]["timestamp"]).date() == bar_date
        ]
        vol_avg = sum(prev_vols) / len(prev_vols) if prev_vols else volume
        rvol = volume / vol_avg if vol_avg > 0 else 1.0

        # VWAP from the current day's bars up to and including this bar
        vwap = self._compute_session_vwap(df, bar_idx, bar_date)

        or_high = self._or_high
        or_low = self._or_low
        range_width = or_high - or_low

        if range_width <= 0:
            no_trade.reasoning = "Zero-width opening range"
            return no_trade

        # ------------------------------------------------------------------
        # 6. Check LONG entry conditions
        # ------------------------------------------------------------------
        if (
            close > or_high
            and rvol >= self.rvol_threshold
            and vwap is not None
            and close > vwap
        ):
            stop = or_low
            risk = close - stop  # actual risk from live entry to stop
            target = close + risk * self.target_r
            self._traded_today = True
            logger.info(
                "orb.long_signal",
                close=close,
                or_high=or_high,
                or_low=or_low,
                rvol=round(rvol, 2),
                vwap=round(vwap, 2) if vwap else None,
                stop=stop,
                target=target,
            )
            return SignalResult(
                action="BUY",
                entry_price=close,
                stop_loss=stop,
                target=target,
                reasoning=(
                    f"ORB LONG: close {close:.2f} > ORH {or_high:.2f}, "
                    f"RVOL {rvol:.2f}x, price above VWAP {vwap:.2f}"
                ),
                metadata={"or_high": or_high, "or_low": or_low, "rvol": rvol, "vwap": vwap},
            )

        # ------------------------------------------------------------------
        # 7. Check SHORT entry conditions
        # ------------------------------------------------------------------
        if (
            close < or_low
            and rvol >= self.rvol_threshold
            and vwap is not None
            and close < vwap
        ):
            stop = or_high
            risk = stop - close  # actual risk from live entry to stop
            target = close - risk * self.target_r
            self._traded_today = True
            logger.info(
                "orb.short_signal",
                close=close,
                or_high=or_high,
                or_low=or_low,
                rvol=round(rvol, 2),
                vwap=round(vwap, 2) if vwap else None,
                stop=stop,
                target=target,
            )
            return SignalResult(
                action="SELL",
                entry_price=close,
                stop_loss=stop,
                target=target,
                reasoning=(
                    f"ORB SHORT: close {close:.2f} < ORL {or_low:.2f}, "
                    f"RVOL {rvol:.2f}x, price below VWAP {vwap:.2f}"
                ),
                metadata={"or_high": or_high, "or_low": or_low, "rvol": rvol, "vwap": vwap},
            )

        no_trade.reasoning = (
            f"No breakout: close {close:.2f} in range [{or_low:.2f}, {or_high:.2f}], "
            f"RVOL {rvol:.2f}x"
        )
        return no_trade

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset per-day state — called before each backtest fold."""
        self._or_high = None
        self._or_low = None
        self._or_date = None
        self._traded_today = False

    def _compute_opening_range(
        self, df: pd.DataFrame, bar_idx: int, bar_date: object
    ) -> tuple[float | None, float | None]:
        """
        Return (high, low) of all bars that fall within the opening range
        window on bar_date, using only bars up to bar_idx (no lookahead).

        Returns (None, None) if no qualifying bars exist yet.
        """
        # Collect bars from market open up to or_minutes elapsed
        highs, lows = [], []
        for i in range(bar_idx + 1):
            r = df.iloc[i]
            ts = pd.Timestamp(r["timestamp"])
            if ts.date() != bar_date:
                continue
            minutes_elapsed = (ts.hour - _MARKET_OPEN_HOUR) * 60 + (ts.minute - _MARKET_OPEN_MIN)
            if 0 <= minutes_elapsed < self.or_minutes:
                highs.append(float(r["high"]))
                lows.append(float(r["low"]))

        if not highs:
            return None, None
        return max(highs), min(lows)

    def _compute_session_vwap(
        self, df: pd.DataFrame, bar_idx: int, bar_date: object
    ) -> float | None:
        """
        Compute cumulative VWAP from market open up to and including bar_idx,
        only for bars on bar_date.
        """
        tp_vol_sum = 0.0
        vol_sum = 0.0
        for i in range(bar_idx + 1):
            r = df.iloc[i]
            if pd.Timestamp(r["timestamp"]).date() != bar_date:
                continue
            tp = (float(r["high"]) + float(r["low"]) + float(r["close"])) / 3.0
            v = float(r["volume"])
            tp_vol_sum += tp * v
            vol_sum += v
        return tp_vol_sum / vol_sum if vol_sum > 0 else None
