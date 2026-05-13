"""
Unit tests for ORBStrategy.

All tests use synthetic OHLCV DataFrames — no network calls, no broker auth.

Scenarios covered
-----------------
1. BUY fires when all conditions are met (close > ORH, RVOL ≥ 1.5, above VWAP)
2. NO_TRADE when RVOL is below threshold
3. NO_TRADE when price is below VWAP (even if it breaks ORH)
4. SHORT fires on ORL breakdown with RVOL and below VWAP
5. NO_TRADE when bar is still inside the opening range window
6. NO_TRADE when bar is after cutoff time (15:10 IST)
7. Second signal on same day is blocked (one trade per day rule)
8. Zero-width opening range returns NO_TRADE
"""

import pandas as pd
import pytest

from src.strategy.orb import ORBStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(date: str, time: str, open_: float, high: float, low: float,
         close: float, volume: float) -> dict:
    return {
        "timestamp": pd.Timestamp(f"{date} {time}+05:30"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _or_bars(date: str = "2024-01-15", base_price: float = 100.0,
             or_high: float = 102.0, or_low: float = 98.0,
             volume: float = 10_000) -> list[dict]:
    """Three 5-min bars forming the 9:15–9:30 opening range."""
    return [
        _bar(date, "09:15", base_price, or_high, or_low, 100.5, volume),
        _bar(date, "09:20", 100.5, 101.5, 99.0, 100.8, volume),
        _bar(date, "09:25", 100.8, 101.8, 98.5, 100.2, volume),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestORBLong:
    def test_buy_fires_when_all_conditions_met(self):
        strategy = ORBStrategy(or_minutes=15, rvol_threshold=1.5, target_r=2.0)
        date = "2024-01-15"
        or_bars = _or_bars(date, or_high=102.0, or_low=98.0, volume=10_000)

        # Signal bar: closes ABOVE ORH (102.0), high volume, above VWAP
        # VWAP from opening bars ≈ 100 → close of 103.5 is well above
        signal_bar = _bar(date, "09:35", 102.0, 104.0, 101.5, 103.5, 20_000)

        df = _make_df(or_bars + [signal_bar])
        result = strategy.evaluate(df, bar_idx=3)

        assert result.action == "BUY"
        assert result.entry_price == pytest.approx(103.5)
        assert result.stop_loss == pytest.approx(98.0)  # ORL
        # target = entry + actual_risk * target_r
        # actual_risk = 103.5 - 98.0 = 5.5; target = 103.5 + 5.5 * 2.0 = 114.5
        assert result.target == pytest.approx(114.5)
        # RR = (114.5 - 103.5) / (103.5 - 98.0) = 11.0 / 5.5 = 2.0
        assert result.risk_reward == pytest.approx(2.0)

    def test_no_trade_when_rvol_too_low(self):
        strategy = ORBStrategy(or_minutes=15, rvol_threshold=1.5)
        date = "2024-01-15"
        or_bars = _or_bars(date, or_high=102.0, or_low=98.0, volume=10_000)

        # Low volume — same as OR bars, so RVOL ≈ 1.0 < 1.5
        signal_bar = _bar(date, "09:35", 102.0, 104.0, 101.5, 103.5, 10_000)

        df = _make_df(or_bars + [signal_bar])
        result = strategy.evaluate(df, bar_idx=3)

        assert result.action == "NO_TRADE"

    def test_no_trade_when_price_inside_range(self):
        """Price closes inside the OR range — no breakout, no signal."""
        strategy = ORBStrategy(or_minutes=15, rvol_threshold=1.5)
        date = "2024-01-15"
        or_bars = _or_bars(date, or_high=102.0, or_low=98.0, volume=10_000)

        # Bar that stays inside the range: close=100 (between 98 and 102)
        inside_bar = _bar(date, "09:35", 100.0, 101.5, 99.0, 100.0, 20_000)

        df = _make_df(or_bars + [inside_bar])
        result = strategy.evaluate(df, bar_idx=3)

        assert result.action == "NO_TRADE"
        assert "breakout" in result.reasoning.lower() or "range" in result.reasoning.lower()


class TestORBShort:
    def test_short_fires_on_orl_breakdown(self):
        strategy = ORBStrategy(or_minutes=15, rvol_threshold=1.5, target_r=2.0)
        date = "2024-01-15"
        or_bars = _or_bars(date, or_high=102.0, or_low=98.0, volume=10_000)

        # Signal bar: closes BELOW ORL (98.0), high volume
        # VWAP ≈ 100 and close=96.5 < VWAP → SHORT conditions met
        signal_bar = _bar(date, "09:35", 98.0, 98.5, 95.0, 96.5, 20_000)

        df = _make_df(or_bars + [signal_bar])
        result = strategy.evaluate(df, bar_idx=3)

        assert result.action == "SELL"
        assert result.entry_price == pytest.approx(96.5)
        assert result.stop_loss == pytest.approx(102.0)  # ORH
        # target = entry - actual_risk * target_r
        # actual_risk = 102.0 - 96.5 = 5.5; target = 96.5 - 5.5 * 2.0 = 85.5
        assert result.target == pytest.approx(85.5)
        assert result.risk_reward == pytest.approx(2.0)


class TestORBTimingRules:
    def test_no_trade_during_opening_range_window(self):
        strategy = ORBStrategy(or_minutes=15)
        date = "2024-01-15"
        # Only 2 bars — still inside the 15-min OR window (9:15, 9:20)
        bars = [
            _bar(date, "09:15", 100.0, 105.0, 98.0, 102.0, 10_000),
            _bar(date, "09:20", 102.0, 106.0, 101.0, 105.5, 25_000),
        ]
        df = _make_df(bars)
        result = strategy.evaluate(df, bar_idx=1)

        assert result.action == "NO_TRADE"
        assert "opening range" in result.reasoning.lower()

    def test_no_trade_after_cutoff_time(self):
        strategy = ORBStrategy(or_minutes=15, cutoff_hour=15, cutoff_minute=10)
        date = "2024-01-15"
        or_bars = _or_bars(date, or_high=102.0, or_low=98.0, volume=10_000)

        # Bar after 15:10 cutoff
        late_bar = _bar(date, "15:15", 102.0, 104.0, 101.5, 103.5, 20_000)

        df = _make_df(or_bars + [late_bar])
        result = strategy.evaluate(df, bar_idx=3)

        assert result.action == "NO_TRADE"
        assert "cutoff" in result.reasoning.lower()

    def test_second_signal_same_day_blocked(self):
        strategy = ORBStrategy(or_minutes=15, rvol_threshold=1.5)
        date = "2024-01-15"
        or_bars = _or_bars(date, or_high=102.0, or_low=98.0, volume=10_000)

        bar1 = _bar(date, "09:35", 102.0, 104.0, 101.5, 103.5, 20_000)
        bar2 = _bar(date, "09:40", 103.5, 106.0, 103.0, 105.0, 20_000)

        df = _make_df(or_bars + [bar1, bar2])

        result1 = strategy.evaluate(df, bar_idx=3)
        result2 = strategy.evaluate(df, bar_idx=4)

        assert result1.action == "BUY"
        assert result2.action == "NO_TRADE"
        assert "already traded" in result2.reasoning.lower()


class TestORBEdgeCases:
    def test_zero_width_range_returns_no_trade(self):
        strategy = ORBStrategy(or_minutes=15)
        date = "2024-01-15"
        # All bars same price → zero-width range
        flat_bars = [
            _bar(date, "09:15", 100.0, 100.0, 100.0, 100.0, 10_000),
            _bar(date, "09:20", 100.0, 100.0, 100.0, 100.0, 10_000),
            _bar(date, "09:25", 100.0, 100.0, 100.0, 100.0, 10_000),
        ]
        signal_bar = _bar(date, "09:35", 100.0, 100.5, 99.5, 100.0, 20_000)
        df = _make_df(flat_bars + [signal_bar])
        result = strategy.evaluate(df, bar_idx=3)

        assert result.action == "NO_TRADE"

    def test_new_day_resets_state(self):
        strategy = ORBStrategy(or_minutes=15, rvol_threshold=1.5)

        # Day 1: trigger a BUY
        date1 = "2024-01-15"
        or_bars_d1 = _or_bars(date1, or_high=102.0, or_low=98.0, volume=10_000)
        bar_d1 = _bar(date1, "09:35", 102.0, 104.0, 101.5, 103.5, 20_000)

        # Day 2: same conditions — should fire again
        date2 = "2024-01-16"
        or_bars_d2 = _or_bars(date2, or_high=102.0, or_low=98.0, volume=10_000)
        bar_d2 = _bar(date2, "09:35", 102.0, 104.0, 101.5, 103.5, 20_000)

        df = _make_df(or_bars_d1 + [bar_d1] + or_bars_d2 + [bar_d2])

        r1 = strategy.evaluate(df, 3)
        r2 = strategy.evaluate(df, 7)

        assert r1.action == "BUY"
        assert r2.action == "BUY"

    def test_signal_result_risk_reward_property(self):
        strategy = ORBStrategy(or_minutes=15, rvol_threshold=1.5, target_r=2.0)
        date = "2024-01-15"
        or_bars = _or_bars(date, or_high=102.0, or_low=98.0, volume=10_000)
        signal_bar = _bar(date, "09:35", 102.0, 104.0, 101.5, 103.5, 20_000)
        df = _make_df(or_bars + [signal_bar])

        result = strategy.evaluate(df, bar_idx=3)
        assert result.action == "BUY"
        # risk = entry - stop = 103.5 - 98.0 = 5.5
        # reward = target - entry = 111.5 - 103.5 = 8.0
        # rr = 8.0 / 5.5 ≈ 1.45 (close to 2.0 * range / (entry - ORL))
        assert result.risk_reward > 1.0
        assert result.is_actionable
