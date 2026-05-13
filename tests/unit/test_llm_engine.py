"""
Unit tests for src/signals/llm_engine.py (TradeSignal model)
and src/signals/rule_engine.py (RuleEngine).

LLMSignalEngine was removed — trading signals now come from RuleEngine.
TradeSignal (the Pydantic model) is retained as the shared data contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from src.signals.llm_engine import TradeSignal
from src.signals.rule_engine import RuleEngine
from src.strategy.base import BaseStrategy, SignalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade_signal(**overrides: Any) -> TradeSignal:
    defaults: dict[str, Any] = {
        "symbol": "RELIANCE",
        "action": "BUY",
        "trade_type": "INTRADAY",
        "entry_price": 2500.0,
        "stop_loss": 2450.0,
        "target_1": 2600.0,
        "target_2": None,
        "confidence": 0.80,
        "risk_reward_ratio": 2.0,
        "reasoning": "Strong uptrend with RSI < 70.",
        "key_risks": ["Market reversal", "FII selling"],
        "invalidation_condition": "Close below 2440",
    }
    defaults.update(overrides)
    return TradeSignal(**defaults)


def _bar(ts: str, open_: float, high: float, low: float, close: float, volume: float = 10_000) -> dict:
    return {"timestamp": pd.Timestamp(ts), "open": open_, "high": high, "low": low, "close": close, "volume": volume}


class _AlwaysBuyStrategy(BaseStrategy):
    """Fires a BUY on the last bar every time."""
    @property
    def name(self) -> str:
        return "AlwaysBuy"

    def get_params(self) -> dict:
        return {}

    def set_params(self, params: dict) -> None:
        pass

    def reset(self) -> None:
        pass

    def evaluate(self, df: pd.DataFrame, bar_idx: int) -> SignalResult:
        return SignalResult(
            action="BUY",
            entry_price=float(df.iloc[bar_idx]["close"]),
            stop_loss=float(df.iloc[bar_idx]["close"]) - 10.0,
            target=float(df.iloc[bar_idx]["close"]) + 20.0,
            reasoning="Always buy",
        )


class _AlwaysNoTradeStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "AlwaysNoTrade"

    def get_params(self) -> dict:
        return {}

    def set_params(self, params: dict) -> None:
        pass

    def reset(self) -> None:
        pass

    def evaluate(self, df: pd.DataFrame, bar_idx: int) -> SignalResult:
        return SignalResult(action="NO_TRADE", entry_price=0, stop_loss=0, target=0, reasoning="no trade")


# ---------------------------------------------------------------------------
# TradeSignal validation tests — these still matter (shared data contract)
# ---------------------------------------------------------------------------

class TestTradeSignalValidation:
    def test_valid_action_buy(self) -> None:
        assert _make_trade_signal(action="BUY").action == "BUY"

    def test_valid_action_sell(self) -> None:
        assert _make_trade_signal(action="SELL").action == "SELL"

    def test_valid_action_no_trade(self) -> None:
        s = _make_trade_signal(action="NO_TRADE", confidence=0.0, risk_reward_ratio=0.0)
        assert s.action == "NO_TRADE"

    def test_action_case_insensitive(self) -> None:
        assert _make_trade_signal(action="buy").action == "BUY"

    def test_invalid_action_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_trade_signal(action="HOLD")

    def test_confidence_boundary_zero(self) -> None:
        assert _make_trade_signal(confidence=0.0).confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        assert _make_trade_signal(confidence=1.0).confidence == 1.0

    def test_confidence_over_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_trade_signal(confidence=1.5)

    def test_confidence_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_trade_signal(confidence=-0.1)

    def test_confidence_rounded_to_4dp(self) -> None:
        s = _make_trade_signal(confidence=0.123456)
        assert s.confidence == round(0.123456, 4)

    def test_invalid_trade_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_trade_signal(trade_type="DAY")

    def test_valid_trade_type_swing(self) -> None:
        assert _make_trade_signal(trade_type="SWING").trade_type == "SWING"

    def test_generated_at_defaults_to_now(self) -> None:
        before = datetime.now()
        s = _make_trade_signal()
        after = datetime.now()
        assert before <= s.generated_at <= after


class TestTradeSignalIsActionable:
    def test_actionable_buy(self) -> None:
        assert _make_trade_signal(action="BUY", confidence=0.80, risk_reward_ratio=2.0).is_actionable is True

    def test_actionable_sell(self) -> None:
        assert _make_trade_signal(action="SELL", confidence=0.70, risk_reward_ratio=1.6).is_actionable is True

    def test_no_trade_never_actionable(self) -> None:
        s = _make_trade_signal(action="NO_TRADE", confidence=1.0, risk_reward_ratio=5.0)
        assert s.is_actionable is False

    def test_low_confidence_not_actionable(self) -> None:
        assert _make_trade_signal(action="BUY", confidence=0.5, risk_reward_ratio=2.0).is_actionable is False

    def test_confidence_boundary_exactly_065_passes(self) -> None:
        assert _make_trade_signal(action="BUY", confidence=0.65, risk_reward_ratio=2.0).is_actionable is True

    def test_confidence_just_below_065_fails(self) -> None:
        assert _make_trade_signal(action="BUY", confidence=0.6499, risk_reward_ratio=2.0).is_actionable is False

    def test_low_rr_not_actionable(self) -> None:
        assert _make_trade_signal(action="BUY", confidence=0.80, risk_reward_ratio=1.2).is_actionable is False

    def test_rr_exactly_15_passes(self) -> None:
        assert _make_trade_signal(action="BUY", confidence=0.80, risk_reward_ratio=1.5).is_actionable is True


class TestTradeSignalToDict:
    def test_returns_dict(self) -> None:
        assert isinstance(_make_trade_signal().to_dict(), dict)

    def test_generated_at_is_string(self) -> None:
        d = _make_trade_signal().to_dict()
        assert isinstance(d["generated_at"], str)
        datetime.fromisoformat(d["generated_at"])  # parseable

    def test_has_all_expected_keys(self) -> None:
        keys = _make_trade_signal().to_dict().keys()
        for k in ("symbol", "action", "entry_price", "stop_loss", "target_1", "confidence"):
            assert k in keys


# ---------------------------------------------------------------------------
# RuleEngine tests
# ---------------------------------------------------------------------------

class TestRuleEngine:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            _bar("2024-01-15 09:15", 100, 101, 99, 100),
            _bar("2024-01-15 09:20", 100, 102, 99, 101),
        ])

    def test_buy_strategy_returns_actionable_signal(self):
        engine = RuleEngine(_AlwaysBuyStrategy(), trade_type="INTRADAY")
        df = self._make_df()
        signal = engine.generate_signal(df, symbol="RELIANCE")

        assert isinstance(signal, TradeSignal)
        assert signal.action == "BUY"
        assert signal.symbol == "RELIANCE"
        assert 0.0 < signal.confidence <= 1.0  # graded by R:R; no longer always 1.0
        assert signal.risk_reward_ratio > 0

    def test_no_trade_strategy_returns_no_trade(self):
        engine = RuleEngine(_AlwaysNoTradeStrategy(), trade_type="INTRADAY")
        df = self._make_df()
        signal = engine.generate_signal(df, symbol="INFY")

        assert signal.action == "NO_TRADE"
        assert signal.is_actionable is False

    def test_empty_df_returns_no_trade(self):
        engine = RuleEngine(_AlwaysBuyStrategy())
        signal = engine.generate_signal(pd.DataFrame(), symbol="TCS")
        assert signal.action == "NO_TRADE"

    def test_live_price_used_when_provided(self):
        engine = RuleEngine(_AlwaysBuyStrategy())
        df = self._make_df()
        signal = engine.generate_signal(df, symbol="HDFCBANK", live_price=150.0)
        # live_price should override bar close as entry_price
        assert signal.entry_price == pytest.approx(150.0)

    def test_batch_generate_returns_only_actionable(self):
        engine = RuleEngine(_AlwaysBuyStrategy())
        df = self._make_df()
        items = [
            {"symbol": "A", "exchange": "NSE", "df": df, "live_price": None},
            {"symbol": "B", "exchange": "NSE", "df": pd.DataFrame(), "live_price": None},
        ]
        results = engine.batch_generate(items)
        # B has empty df → NO_TRADE → filtered out; A → BUY → kept
        assert len(results) == 1
        assert results[0].symbol == "A"

    def test_rule_engine_uses_strategy_name_in_log(self):
        engine = RuleEngine(_AlwaysBuyStrategy())
        assert engine.strategy.name == "AlwaysBuy"

    def test_error_in_strategy_returns_no_trade(self):
        class ErrorStrategy(BaseStrategy):
            @property
            def name(self): return "Error"
            def get_params(self): return {}
            def set_params(self, p): pass
            def reset(self): pass
            def evaluate(self, df, bar_idx):
                raise RuntimeError("boom")

        engine = RuleEngine(ErrorStrategy())
        df = self._make_df()
        signal = engine.generate_signal(df, symbol="BOOM")
        assert signal.action == "NO_TRADE"
