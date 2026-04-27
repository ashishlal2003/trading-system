"""
Technical indicator computation engine for the algorithmic trading system.

All indicators are implemented manually using pure pandas/numpy — no pandas-ta
or TA-Lib dependency.  This avoids install-time issues and gives full control
over edge-case behaviour.

Indicators computed
-------------------
- RSI(14)         – exponential-weighted gains/losses ratio
- MACD            – EMA(12) − EMA(26), signal = EMA(9) of MACD, histogram
- EMA(9/21/50/200)
- Bollinger Bands – SMA(20) ± 2σ, plus %B
- ATR(14)         – Wilder's EMA of true range
- Volume SMA(20)
- Relative Volume – current volume vs. its 20-period SMA
- VWAP            – session VWAP (cumulative typical-price × volume)
- Trend           – UPTREND / DOWNTREND / SIDEWAYS from EMA alignment
- Price vs VWAP   – ABOVE_VWAP / BELOW_VWAP
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Minimum candle count required before any indicator is computed
# ---------------------------------------------------------------------------
_MIN_ROWS: int = 30


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class IndicatorResult:
    symbol: str
    timestamp: pd.Timestamp
    close: float
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    ema_9: float
    ema_21: float
    ema_50: float
    ema_200: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    bb_pct_b: float
    atr_14: float
    volume_sma_20: float
    relative_volume: float
    vwap: float
    trend: str          # "UPTREND" | "DOWNTREND" | "SIDEWAYS"
    price_vs_vwap: str  # "ABOVE_VWAP" | "BELOW_VWAP"

    def to_dict(self) -> dict:
        return {
            k: (str(v) if isinstance(v, pd.Timestamp) else v)
            for k, v in self.__dict__.items()
        }


# ---------------------------------------------------------------------------
# Helper functions (pure numpy/pandas)
# ---------------------------------------------------------------------------


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average — pandas ewm with adjust=False (Wilder-compatible)."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using exponential-weighted smoothing of gains and losses.

    Steps
    -----
    1. Compute price deltas.
    2. Separate gains (delta > 0) and losses (delta < 0, made positive).
    3. Smooth both with ewm(span=period, adjust=False).
    4. RSI = 100 − 100 / (1 + avg_gain / avg_loss).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    # Guard against zero division
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD line, signal line, and histogram.

    Returns
    -------
    (macd_line, macd_signal, macd_hist) — all as pd.Series aligned to *close*.
    """
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = _ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def _bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Returns
    -------
    (upper, mid, lower, pct_b)
    - pct_b = (close − lower) / (upper − lower)
    """
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)  # population std, common for BB
    upper = mid + num_std * std
    lower = mid - num_std * std
    band_width = upper - lower
    # Avoid zero division when band collapses
    pct_b = (close - lower) / band_width.replace(0, np.nan)
    return upper, mid, lower, pct_b


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range using EMA(14) of the true range.

    True Range = max(high − low, |high − prev_close|, |low − prev_close|)
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Session VWAP — cumulative sum of (typical_price × volume) / cumulative volume.

    typical_price = (high + low + close) / 3
    """
    typical_price = (high + low + close) / 3.0
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _trend(ema_9: float, ema_21: float, ema_50: float) -> str:
    """
    Classify trend from EMA alignment.

    - UPTREND   : ema_9 > ema_21 > ema_50
    - DOWNTREND : ema_9 < ema_21 < ema_50
    - SIDEWAYS  : anything else
    """
    if ema_9 > ema_21 > ema_50:
        return "UPTREND"
    if ema_9 < ema_21 < ema_50:
        return "DOWNTREND"
    return "SIDEWAYS"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class IndicatorEngine:
    """
    Computes a full suite of technical indicators for a single OHLCV DataFrame.

    Usage
    -----
    ::

        engine = IndicatorEngine()
        result: IndicatorResult | None = engine.compute(df, symbol="RELIANCE")

    The DataFrame *df* must have columns:
        timestamp, open, high, low, close, volume
    Rows must be sorted in ascending chronological order.

    Returns ``None`` when:
    - The DataFrame has fewer than ``_MIN_ROWS`` rows.
    - Any unexpected exception occurs (logged at ERROR level).
    """

    def compute(self, df: pd.DataFrame, symbol: str) -> IndicatorResult | None:
        """
        Compute all indicators and return an IndicatorResult for the last row.

        Parameters
        ----------
        df:
            OHLCV DataFrame sorted ascending by timestamp.
            Required columns: timestamp, open, high, low, close, volume.
        symbol:
            Instrument ticker string used for logging and result labelling.

        Returns
        -------
        IndicatorResult | None
        """
        try:
            if df is None or len(df) < _MIN_ROWS:
                logger.warning(
                    "insufficient_candles",
                    symbol=symbol,
                    rows=len(df) if df is not None else 0,
                    required=_MIN_ROWS,
                )
                return None

            # ------------------------------------------------------------------
            # Validate required columns
            # ------------------------------------------------------------------
            required = {"timestamp", "open", "high", "low", "close", "volume"}
            missing = required - set(df.columns)
            if missing:
                logger.error(
                    "missing_columns",
                    symbol=symbol,
                    missing=sorted(missing),
                )
                return None

            # Work on a copy to avoid mutating caller's data
            df = df.copy().reset_index(drop=True)
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            volume = df["volume"].astype(float)

            # ------------------------------------------------------------------
            # RSI(14)
            # ------------------------------------------------------------------
            rsi_series = _rsi(close, period=14)

            # ------------------------------------------------------------------
            # MACD
            # ------------------------------------------------------------------
            macd_line_s, macd_signal_s, macd_hist_s = _macd(close)

            # ------------------------------------------------------------------
            # EMAs
            # ------------------------------------------------------------------
            ema9_s = _ema(close, 9)
            ema21_s = _ema(close, 21)
            ema50_s = _ema(close, 50)
            ema200_s = _ema(close, 200)

            # ------------------------------------------------------------------
            # Bollinger Bands
            # ------------------------------------------------------------------
            bb_upper_s, bb_mid_s, bb_lower_s, bb_pct_b_s = _bollinger_bands(close)

            # ------------------------------------------------------------------
            # ATR(14)
            # ------------------------------------------------------------------
            atr_s = _atr(high, low, close, period=14)

            # ------------------------------------------------------------------
            # Volume SMA(20) & Relative Volume
            # ------------------------------------------------------------------
            vol_sma20_s = volume.rolling(window=20).mean()
            rel_vol_s = volume / vol_sma20_s.replace(0, np.nan)

            # ------------------------------------------------------------------
            # VWAP
            # ------------------------------------------------------------------
            vwap_s = _vwap(high, low, close, volume)

            # ------------------------------------------------------------------
            # Extract last-row scalar values
            # ------------------------------------------------------------------
            idx = len(df) - 1

            def _scalar(series: pd.Series) -> float:
                val = series.iloc[idx]
                return float(val) if not pd.isna(val) else float("nan")

            last_close = _scalar(close)
            last_vwap = _scalar(vwap_s)
            last_ema9 = _scalar(ema9_s)
            last_ema21 = _scalar(ema21_s)
            last_ema50 = _scalar(ema50_s)

            result = IndicatorResult(
                symbol=symbol,
                timestamp=df["timestamp"].iloc[idx],
                close=last_close,
                rsi_14=_scalar(rsi_series),
                macd_line=_scalar(macd_line_s),
                macd_signal=_scalar(macd_signal_s),
                macd_hist=_scalar(macd_hist_s),
                ema_9=last_ema9,
                ema_21=last_ema21,
                ema_50=last_ema50,
                ema_200=_scalar(ema200_s),
                bb_upper=_scalar(bb_upper_s),
                bb_mid=_scalar(bb_mid_s),
                bb_lower=_scalar(bb_lower_s),
                bb_pct_b=_scalar(bb_pct_b_s),
                atr_14=_scalar(atr_s),
                volume_sma_20=_scalar(vol_sma20_s),
                relative_volume=_scalar(rel_vol_s),
                vwap=last_vwap,
                trend=_trend(last_ema9, last_ema21, last_ema50),
                price_vs_vwap=(
                    "ABOVE_VWAP" if last_close >= last_vwap else "BELOW_VWAP"
                ),
            )

            logger.debug(
                "indicators_computed",
                symbol=symbol,
                close=last_close,
                rsi=result.rsi_14,
                trend=result.trend,
            )
            return result

        except Exception:
            logger.error(
                "indicator_compute_error",
                symbol=symbol,
                exc_info=True,
            )
            return None
