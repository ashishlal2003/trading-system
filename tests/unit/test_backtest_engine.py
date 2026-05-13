"""
Unit tests for BacktestEngine.

All tests use synthetic OHLCV data and a stub strategy — no network calls.

Scenarios covered
-----------------
1. Zero trades → equity curve is flat at initial capital
2. Single winning trade → equity increases by correct net amount
3. Single losing trade (stop hit) → equity decreases correctly
4. Transaction costs applied on both entry and exit
5. Target hit on the bar after entry fires
6. EOD force-exit closes any position remaining at cutoff
7. End-of-data force-exit closes remaining open position
"""

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.strategy.base import BaseStrategy, SignalResult


# ---------------------------------------------------------------------------
# Stub strategy — fires on demand
# ---------------------------------------------------------------------------

class SingleTradeStrategy(BaseStrategy):
    """Fires one signal at fire_bar_idx, then NO_TRADE forever."""

    def __init__(self, fire_bar_idx: int, signal: SignalResult):
        self._fire_bar = fire_bar_idx
        self._signal = signal
        self._fired = False

    @property
    def name(self) -> str:
        return "SingleTrade"

    def get_params(self) -> dict:
        return {"fire_bar": self._fire_bar}

    def set_params(self, params: dict) -> None:
        self._fire_bar = params.get("fire_bar", self._fire_bar)

    def reset(self) -> None:
        self._fired = False

    def evaluate(self, df: pd.DataFrame, bar_idx: int) -> SignalResult:
        if bar_idx == self._fire_bar and not self._fired:
            self._fired = True
            return self._signal
        return SignalResult(action="NO_TRADE", entry_price=0, stop_loss=0, target=0, reasoning="")


class NeverTradeStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "NeverTrade"

    def get_params(self) -> dict:
        return {}

    def set_params(self, params: dict) -> None:
        pass

    def reset(self) -> None:
        pass

    def evaluate(self, df: pd.DataFrame, bar_idx: int) -> SignalResult:
        return SignalResult(action="NO_TRADE", entry_price=0, stop_loss=0, target=0, reasoning="no trade")


# ---------------------------------------------------------------------------
# Helper: build synthetic OHLCV DataFrame
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _bar(ts: str, open_: float, high: float, low: float, close: float, volume: float = 10_000) -> dict:
    return {"timestamp": pd.Timestamp(ts), "open": open_, "high": high, "low": low, "close": close, "volume": volume}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoTrades:
    def test_zero_trades_flat_equity(self):
        df = _make_df([
            _bar("2024-01-15 09:15", 100, 101, 99, 100),
            _bar("2024-01-15 09:20", 100, 101, 99, 100),
            _bar("2024-01-15 09:25", 100, 101, 99, 100),
        ])
        engine = BacktestEngine(transaction_cost_pct=0.0006, slippage_pct=0.0002)
        result = engine.run(df, NeverTradeStrategy(), initial_capital=100_000, symbol="TEST")

        assert len(result.trades) == 0
        # Without any trades, no costs are incurred
        assert result.final_capital == pytest.approx(100_000.0, rel=1e-6)


class TestWinningTrade:
    def test_long_trade_hits_target(self):
        # Entry at bar 0 close=100. Target=110. Stop=95.
        # Bar 1: high=111 → target hit at 110.
        signal = SignalResult(action="BUY", entry_price=100.0, stop_loss=95.0, target=110.0, reasoning="test")
        df = _make_df([
            _bar("2024-01-15 09:15", 100, 100, 99, 100),  # entry bar
            _bar("2024-01-15 09:20", 100, 111, 99, 108),  # target hit (high=111 > 110)
        ])
        strategy = SingleTradeStrategy(fire_bar_idx=0, signal=signal)
        engine = BacktestEngine(
            transaction_cost_pct=0.0,  # disable costs for this test
            slippage_pct=0.0,
            quantity=100,
            eod_exit_hour=23,  # no EOD force-exit in test
        )
        result = engine.run(df, strategy, initial_capital=100_000, symbol="TEST")

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "TARGET"
        assert trade.exit_price == pytest.approx(110.0)
        # gross P&L = (110 - 100) * 100 = 1000
        assert trade.gross_pnl == pytest.approx(1_000.0)
        assert result.final_capital == pytest.approx(101_000.0)


class TestLosingTrade:
    def test_long_trade_hits_stop(self):
        # Entry at bar 0 close=100. Stop=95. Target=110.
        # Bar 1: low=94 → stop hit at 95.
        signal = SignalResult(action="BUY", entry_price=100.0, stop_loss=95.0, target=110.0, reasoning="test")
        df = _make_df([
            _bar("2024-01-15 09:15", 100, 101, 99, 100),
            _bar("2024-01-15 09:20", 99, 99, 94, 96),   # low=94 < stop=95
        ])
        strategy = SingleTradeStrategy(fire_bar_idx=0, signal=signal)
        engine = BacktestEngine(
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
            quantity=100,
            eod_exit_hour=23,
        )
        result = engine.run(df, strategy, initial_capital=100_000, symbol="TEST")

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "STOP"
        assert trade.exit_price == pytest.approx(95.0)
        # gross P&L = (95 - 100) * 100 = -500
        assert trade.gross_pnl == pytest.approx(-500.0)
        assert result.final_capital == pytest.approx(99_500.0)


class TestTransactionCosts:
    def test_costs_reduce_net_pnl(self):
        # 100 shares, entry=100, exit=110.
        # transaction_cost_pct = 0.006 (0.6% round trip)
        # Engine design: entry half-cost deducted from capital on entry;
        #                exit half-cost deducted inside _close_position → net_pnl.
        # entry half-cost = 100 * 100 * 0.003 = 30  (deducted from capital)
        # exit  half-cost = 110 * 100 * 0.003 = 33  (inside net_pnl)
        # gross = (110 - 100) * 100 = 1000
        # trade.net_pnl = 1000 - 33 = 967
        # final_capital = initial - 30 + 967 = 100_937
        signal = SignalResult(action="BUY", entry_price=100.0, stop_loss=95.0, target=110.0, reasoning="test")
        df = _make_df([
            _bar("2024-01-15 09:15", 100, 100, 99, 100),
            _bar("2024-01-15 09:20", 100, 111, 99, 108),
        ])
        strategy = SingleTradeStrategy(fire_bar_idx=0, signal=signal)
        engine = BacktestEngine(
            transaction_cost_pct=0.006,
            slippage_pct=0.0,
            quantity=100,
            eod_exit_hour=23,
        )
        result = engine.run(df, strategy, initial_capital=100_000, symbol="TEST")

        trade = result.trades[0]
        # net_pnl on trade = gross - exit_half_cost = 1000 - 33 = 967
        assert trade.net_pnl == pytest.approx(967.0, rel=1e-4)
        # final capital accounts for both halves: 100_000 - 30 + 967 = 100_937
        assert result.final_capital == pytest.approx(100_937.0, rel=1e-4)

    def test_costs_with_slippage(self):
        # slippage_pct=0.01 (1% for easy math), no transaction cost
        # entry fill = 100 * 1.01 = 101.0
        # exit fill  = 110 * 0.99 = 108.9
        # gross P&L = (108.9 - 101.0) * 100 = 790
        signal = SignalResult(action="BUY", entry_price=100.0, stop_loss=95.0, target=110.0, reasoning="test")
        df = _make_df([
            _bar("2024-01-15 09:15", 100, 100, 99, 100),
            _bar("2024-01-15 09:20", 100, 111, 99, 108),
        ])
        strategy = SingleTradeStrategy(fire_bar_idx=0, signal=signal)
        engine = BacktestEngine(
            transaction_cost_pct=0.0,
            slippage_pct=0.01,
            quantity=100,
            eod_exit_hour=23,
        )
        result = engine.run(df, strategy, initial_capital=100_000, symbol="TEST")

        trade = result.trades[0]
        assert trade.entry_price == pytest.approx(101.0)
        assert trade.exit_price == pytest.approx(108.9)
        assert trade.gross_pnl == pytest.approx(790.0, rel=1e-4)


class TestEODExit:
    def test_eod_exits_open_position(self):
        # Entry at 09:15. No stop/target hit. EOD at 09:25 bar.
        signal = SignalResult(action="BUY", entry_price=100.0, stop_loss=50.0, target=200.0, reasoning="test")
        df = _make_df([
            _bar("2024-01-15 09:15", 100, 101, 99, 100),   # entry
            _bar("2024-01-15 09:20", 100, 101, 99, 101),   # hold
            _bar("2024-01-15 09:25", 101, 102, 100, 102),  # EOD exit bar
        ])
        strategy = SingleTradeStrategy(fire_bar_idx=0, signal=signal)
        engine = BacktestEngine(
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
            quantity=100,
            eod_exit_hour=9,
            eod_exit_minute=25,   # EOD at 09:25
        )
        result = engine.run(df, strategy, initial_capital=100_000, symbol="TEST")

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "EOD"
        assert trade.exit_price == pytest.approx(102.0)  # close of EOD bar


class TestEndOfDataExit:
    def test_end_of_data_closes_position(self):
        signal = SignalResult(action="BUY", entry_price=100.0, stop_loss=50.0, target=200.0, reasoning="test")
        df = _make_df([
            _bar("2024-01-15 09:15", 100, 101, 99, 100),
            _bar("2024-01-15 09:20", 100, 101, 99, 105),  # last bar, no EOD hour hit
        ])
        strategy = SingleTradeStrategy(fire_bar_idx=0, signal=signal)
        engine = BacktestEngine(
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
            quantity=100,
            eod_exit_hour=23,  # no EOD trigger
        )
        result = engine.run(df, strategy, initial_capital=100_000, symbol="TEST")

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "END_OF_DATA"


class TestShortTrade:
    def test_short_trade_hits_target(self):
        # SELL entry at 100, stop=105, target=90.
        # Bar 1: low=89 → target hit at 90.
        signal = SignalResult(action="SELL", entry_price=100.0, stop_loss=105.0, target=90.0, reasoning="test")
        df = _make_df([
            _bar("2024-01-15 09:15", 100, 101, 99, 100),
            _bar("2024-01-15 09:20", 99, 100, 89, 91),  # low=89 < target=90
        ])
        strategy = SingleTradeStrategy(fire_bar_idx=0, signal=signal)
        engine = BacktestEngine(
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
            quantity=100,
            eod_exit_hour=23,
        )
        result = engine.run(df, strategy, initial_capital=100_000, symbol="TEST")

        trade = result.trades[0]
        assert trade.exit_reason == "TARGET"
        assert trade.exit_price == pytest.approx(90.0)
        # Short P&L = (entry - exit) * qty = (100 - 90) * 100 = 1000
        assert trade.gross_pnl == pytest.approx(1_000.0)
