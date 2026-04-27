"""
Candlestick pattern detection for the algorithmic trading system.

All patterns are implemented manually using pure pandas — no TA-Lib dependency.

Patterns detected
-----------------
Bullish
  HAMMER              – small body in upper third of range, long lower shadow
  BULLISH_ENGULFING   – current bullish candle fully engulfs prior bearish candle
  MORNING_DOJI_STAR   – 3-bar: bearish → doji → bullish
  PIERCING_LINE       – bearish → bullish opening below prior low, closing above prior midpoint

Bearish
  SHOOTING_STAR       – small body in lower third, long upper shadow
  BEARISH_ENGULFING   – current bearish candle fully engulfs prior bullish candle
  EVENING_DOJI_STAR   – 3-bar: bullish → doji → bearish
  DARK_CLOUD_COVER    – bullish → bearish opening above prior high, closing below prior midpoint

Neutral
  DOJI                – open ≈ close (body ≤ 10 % of high-low range)

Bias is determined by counting bullish vs. bearish pattern detections:
  more bullish → "BULLISH", more bearish → "BEARISH", tie → "NEUTRAL"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Minimum rows required for 3-candle patterns
# ---------------------------------------------------------------------------
_MIN_ROWS_3: int = 3
_MIN_ROWS_2: int = 2

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PatternResult:
    symbol: str
    detected: list[str] = field(default_factory=list)
    bias: str = "NEUTRAL"   # "BULLISH" | "BEARISH" | "NEUTRAL"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _body(o: float, c: float) -> float:
    """Absolute body size."""
    return abs(c - o)


def _range(h: float, l: float) -> float:
    """Total candle range.  Returns a small positive floor to prevent ZeroDivisionError."""
    return max(h - l, 1e-8)


def _is_bullish(o: float, c: float) -> bool:
    return c > o


def _is_bearish(o: float, c: float) -> bool:
    return c < o


def _is_doji(o: float, c: float, h: float, l: float, threshold: float = 0.10) -> bool:
    """Body ≤ threshold × total range."""
    return _body(o, c) <= threshold * _range(h, l)


# ---------------------------------------------------------------------------
# Single-candle patterns (operate on the *last* candle of df)
# ---------------------------------------------------------------------------


def _detect_hammer(o: float, c: float, h: float, l: float) -> bool:
    """
    Hammer (bullish reversal at bottom of downtrend):
    - Body occupies the upper third of the total range.
    - Lower shadow ≥ 2× body size.
    - Upper shadow ≤ 10 % of total range.
    """
    rng = _range(h, l)
    body = _body(o, c)
    body_top = max(o, c)
    body_bot = min(o, c)
    lower_shadow = body_bot - l
    upper_shadow = h - body_top

    body_in_upper_third = body_bot >= (l + 2 * rng / 3)
    long_lower = lower_shadow >= 2 * body if body > 0 else lower_shadow >= 0.6 * rng
    small_upper = upper_shadow <= 0.10 * rng

    return body_in_upper_third and long_lower and small_upper


def _detect_shooting_star(o: float, c: float, h: float, l: float) -> bool:
    """
    Shooting Star (bearish reversal at top of uptrend):
    - Body occupies the lower third of the total range.
    - Upper shadow ≥ 2× body size.
    - Lower shadow ≤ 10 % of total range.
    """
    rng = _range(h, l)
    body = _body(o, c)
    body_top = max(o, c)
    body_bot = min(o, c)
    upper_shadow = h - body_top
    lower_shadow = body_bot - l

    body_in_lower_third = body_top <= (l + rng / 3)
    long_upper = upper_shadow >= 2 * body if body > 0 else upper_shadow >= 0.6 * rng
    small_lower = lower_shadow <= 0.10 * rng

    return body_in_lower_third and long_upper and small_lower


# ---------------------------------------------------------------------------
# Two-candle patterns (operate on the last two rows of df)
# ---------------------------------------------------------------------------


def _detect_bullish_engulfing(
    o1: float, c1: float,   # prior candle
    o2: float, c2: float,   # current candle
) -> bool:
    """
    Bullish Engulfing:
    - Prior candle is bearish (c1 < o1).
    - Current candle is bullish (c2 > o2).
    - Current body fully engulfs prior body: o2 ≤ c1 AND c2 ≥ o1.
    """
    return (
        _is_bearish(o1, c1)
        and _is_bullish(o2, c2)
        and o2 <= c1
        and c2 >= o1
    )


def _detect_bearish_engulfing(
    o1: float, c1: float,
    o2: float, c2: float,
) -> bool:
    """
    Bearish Engulfing:
    - Prior candle is bullish (c1 > o1).
    - Current candle is bearish (c2 < o2).
    - Current body fully engulfs prior body: o2 ≥ c1 AND c2 ≤ o1.
    """
    return (
        _is_bullish(o1, c1)
        and _is_bearish(o2, c2)
        and o2 >= c1
        and c2 <= o1
    )


def _detect_piercing_line(
    o1: float, c1: float, h1: float, l1: float,  # prior bearish candle
    o2: float, c2: float,                          # current candle
) -> bool:
    """
    Piercing Line (bullish reversal):
    - Prior candle is bearish.
    - Current candle is bullish.
    - Current open < prior low (opens in bearish territory below prior low).
    - Current close > midpoint of prior body.
    """
    prior_mid = (o1 + c1) / 2.0
    return (
        _is_bearish(o1, c1)
        and _is_bullish(o2, c2)
        and o2 < l1
        and c2 > prior_mid
    )


def _detect_dark_cloud_cover(
    o1: float, c1: float, h1: float, l1: float,  # prior bullish candle
    o2: float, c2: float,                          # current candle
) -> bool:
    """
    Dark Cloud Cover (bearish reversal):
    - Prior candle is bullish.
    - Current candle is bearish.
    - Current open > prior high.
    - Current close < midpoint of prior body.
    """
    prior_mid = (o1 + c1) / 2.0
    return (
        _is_bullish(o1, c1)
        and _is_bearish(o2, c2)
        and o2 > h1
        and c2 < prior_mid
    )


# ---------------------------------------------------------------------------
# Three-candle patterns
# ---------------------------------------------------------------------------


def _detect_morning_doji_star(
    o1: float, c1: float, h1: float, l1: float,  # candle -2 (bearish)
    o2: float, c2: float, h2: float, l2: float,  # candle -1 (doji)
    o3: float, c3: float,                          # candle  0 (bullish)
) -> bool:
    """
    Morning Doji Star (bullish reversal):
    1. First candle is bearish.
    2. Second candle is a doji.
    3. Third candle is bullish.

    Gap is not required (Indian market adaptation).
    """
    return (
        _is_bearish(o1, c1)
        and _is_doji(o2, c2, h2, l2)
        and _is_bullish(o3, c3)
    )


def _detect_evening_doji_star(
    o1: float, c1: float, h1: float, l1: float,  # candle -2 (bullish)
    o2: float, c2: float, h2: float, l2: float,  # candle -1 (doji)
    o3: float, c3: float,                          # candle  0 (bearish)
) -> bool:
    """
    Evening Doji Star (bearish reversal):
    1. First candle is bullish.
    2. Second candle is a doji.
    3. Third candle is bearish.
    """
    return (
        _is_bullish(o1, c1)
        and _is_doji(o2, c2, h2, l2)
        and _is_bearish(o3, c3)
    )


# ---------------------------------------------------------------------------
# PatternDetector
# ---------------------------------------------------------------------------


class PatternDetector:
    """
    Detect candlestick patterns from an OHLCV DataFrame.

    Usage
    -----
    ::

        detector = PatternDetector()
        result: PatternResult = detector.detect(df, symbol="RELIANCE")

    The DataFrame *df* must have columns: open, high, low, close (and optionally
    timestamp / volume).  Rows must be sorted ascending by time.
    Returns a ``PatternResult`` with ``detected`` patterns and overall ``bias``.
    """

    # Pattern tags grouped by bias for bias calculation
    _BULLISH_TAGS = frozenset(
        {"HAMMER", "BULLISH_ENGULFING", "MORNING_DOJI_STAR", "PIERCING_LINE"}
    )
    _BEARISH_TAGS = frozenset(
        {"SHOOTING_STAR", "BEARISH_ENGULFING", "EVENING_DOJI_STAR", "DARK_CLOUD_COVER"}
    )

    def detect(self, df: pd.DataFrame, symbol: str) -> PatternResult:
        """
        Run all candlestick pattern checks against the last 1-3 rows of *df*.

        Parameters
        ----------
        df:
            OHLCV DataFrame sorted ascending by timestamp.
            Required columns: open, high, low, close.
        symbol:
            Instrument ticker (used for logging and the result label).

        Returns
        -------
        PatternResult
            Always returns a result; ``detected`` is empty when no pattern fires
            or when *df* is too short.
        """
        result = PatternResult(symbol=symbol)

        try:
            if df is None or len(df) < 1:
                logger.warning("pattern_detect_empty_df", symbol=symbol)
                return result

            required = {"open", "high", "low", "close"}
            missing = required - set(df.columns)
            if missing:
                logger.error(
                    "pattern_detect_missing_columns",
                    symbol=symbol,
                    missing=sorted(missing),
                )
                return result

            df = df.copy().reset_index(drop=True)
            n = len(df)

            # ----------------------------------------------------------------
            # Extract last 3 rows (as floats)
            # ----------------------------------------------------------------
            def _row(i: int) -> tuple[float, float, float, float]:
                """Return (open, high, low, close) for row index i from end."""
                r = df.iloc[-(i + 1)]
                return (
                    float(r["open"]),
                    float(r["high"]),
                    float(r["low"]),
                    float(r["close"]),
                )

            o0, h0, l0, c0 = _row(0)   # most recent (current)

            detected: list[str] = []

            # ----------------------------------------------------------------
            # 1-candle patterns
            # ----------------------------------------------------------------

            # DOJI
            if _is_doji(o0, c0, h0, l0):
                detected.append("DOJI")

            # HAMMER
            if _detect_hammer(o0, c0, h0, l0):
                detected.append("HAMMER")

            # SHOOTING_STAR
            if _detect_shooting_star(o0, c0, h0, l0):
                detected.append("SHOOTING_STAR")

            # ----------------------------------------------------------------
            # 2-candle patterns
            # ----------------------------------------------------------------
            if n >= _MIN_ROWS_2:
                o1, h1, l1, c1 = _row(1)   # candle before current

                # BULLISH_ENGULFING
                if _detect_bullish_engulfing(o1, c1, o0, c0):
                    detected.append("BULLISH_ENGULFING")

                # BEARISH_ENGULFING
                if _detect_bearish_engulfing(o1, c1, o0, c0):
                    detected.append("BEARISH_ENGULFING")

                # PIERCING_LINE
                if _detect_piercing_line(o1, c1, h1, l1, o0, c0):
                    detected.append("PIERCING_LINE")

                # DARK_CLOUD_COVER
                if _detect_dark_cloud_cover(o1, c1, h1, l1, o0, c0):
                    detected.append("DARK_CLOUD_COVER")

            # ----------------------------------------------------------------
            # 3-candle patterns
            # ----------------------------------------------------------------
            if n >= _MIN_ROWS_3:
                o2, h2, l2, c2 = _row(2)   # two candles before current

                # MORNING_DOJI_STAR  (bearish, doji, bullish)
                if _detect_morning_doji_star(o2, c2, h2, l2, o1, c1, h1, l1, o0, c0):
                    detected.append("MORNING_DOJI_STAR")

                # EVENING_DOJI_STAR  (bullish, doji, bearish)
                if _detect_evening_doji_star(o2, c2, h2, l2, o1, c1, h1, l1, o0, c0):
                    detected.append("EVENING_DOJI_STAR")

            # ----------------------------------------------------------------
            # Compute bias
            # ----------------------------------------------------------------
            bullish_count = sum(1 for p in detected if p in self._BULLISH_TAGS)
            bearish_count = sum(1 for p in detected if p in self._BEARISH_TAGS)

            if bullish_count > bearish_count:
                bias = "BULLISH"
            elif bearish_count > bullish_count:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"

            result.detected = detected
            result.bias = bias

            logger.debug(
                "patterns_detected",
                symbol=symbol,
                patterns=detected,
                bias=bias,
            )

        except Exception:
            logger.error(
                "pattern_detect_error",
                symbol=symbol,
                exc_info=True,
            )

        return result
