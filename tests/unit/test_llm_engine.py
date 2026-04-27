"""
Unit tests for src/signals/llm_engine.py — TradeSignal and LLMSignalEngine.

OpenAI calls are fully mocked; no real API key or network access is required.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.signals.llm_engine import LLMSignalEngine, TradeSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade_signal(**overrides: Any) -> TradeSignal:
    """Return a valid TradeSignal, optionally overriding any fields."""
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


def _make_llm_engine(api_key: str = "test-key") -> LLMSignalEngine:
    """Instantiate an LLMSignalEngine with a dummy API key."""
    return LLMSignalEngine(api_key=api_key)


def _make_mock_indicators() -> MagicMock:
    """Return a MagicMock that satisfies the IndicatorResult interface."""
    ind = MagicMock()
    ind.close = 2500.0
    ind.rsi_14 = 58.0
    ind.macd_line = 12.5
    ind.macd_signal = 10.0
    ind.macd_hist = 2.5
    ind.ema_9 = 2490.0
    ind.ema_21 = 2480.0
    ind.ema_50 = 2460.0
    ind.ema_200 = 2400.0
    ind.bb_upper = 2560.0
    ind.bb_mid = 2500.0
    ind.bb_lower = 2440.0
    ind.bb_pct_b = 0.5
    ind.atr_14 = 30.0
    ind.vwap = 2495.0
    ind.relative_volume = 1.3
    ind.trend = "UPTREND"
    ind.price_vs_vwap = "ABOVE_VWAP"
    return ind


def _make_openai_response(payload: dict[str, Any]) -> MagicMock:
    """Wrap a dict in a mock that looks like an OpenAI chat completion response."""
    message = MagicMock()
    message.content = json.dumps(payload)

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# TradeSignal validation tests
# ---------------------------------------------------------------------------


class TestTradeSignalValidation:
    """Tests for the Pydantic model validators on TradeSignal."""

    def test_trade_signal_valid_action_buy(self) -> None:
        """action='BUY' must be accepted and normalised to upper-case."""
        signal = _make_trade_signal(action="BUY")
        assert signal.action == "BUY"

    def test_trade_signal_valid_action_sell(self) -> None:
        """action='SELL' must be accepted."""
        signal = _make_trade_signal(action="SELL")
        assert signal.action == "SELL"

    def test_trade_signal_valid_action_no_trade(self) -> None:
        """action='NO_TRADE' must be accepted."""
        signal = _make_trade_signal(action="NO_TRADE", confidence=0.0, risk_reward_ratio=0.0)
        assert signal.action == "NO_TRADE"

    def test_trade_signal_action_case_insensitive(self) -> None:
        """Lower-case 'buy' must be normalised to 'BUY'."""
        signal = _make_trade_signal(action="buy")
        assert signal.action == "BUY"

    def test_trade_signal_invalid_action_raises(self) -> None:
        """action='HOLD' is not a valid action and must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _make_trade_signal(action="HOLD")
        assert "action" in str(exc_info.value).lower() or "HOLD" in str(exc_info.value)

    def test_trade_signal_invalid_action_wait_raises(self) -> None:
        """action='WAIT' must also raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_trade_signal(action="WAIT")

    def test_trade_signal_confidence_valid_boundary_zero(self) -> None:
        """confidence=0.0 is the lower boundary and must be accepted."""
        signal = _make_trade_signal(confidence=0.0)
        assert signal.confidence == 0.0

    def test_trade_signal_confidence_valid_boundary_one(self) -> None:
        """confidence=1.0 is the upper boundary and must be accepted."""
        signal = _make_trade_signal(confidence=1.0)
        assert signal.confidence == 1.0

    def test_trade_signal_confidence_range_over_one_raises(self) -> None:
        """confidence=1.5 is out of range and must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _make_trade_signal(confidence=1.5)
        assert "confidence" in str(exc_info.value).lower() or "1.5" in str(exc_info.value)

    def test_trade_signal_confidence_range_negative_raises(self) -> None:
        """confidence=-0.1 must raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_trade_signal(confidence=-0.1)

    def test_trade_signal_confidence_rounded_to_4dp(self) -> None:
        """confidence=0.123456 must be rounded to 4 decimal places."""
        signal = _make_trade_signal(confidence=0.123456)
        assert signal.confidence == round(0.123456, 4)

    def test_trade_signal_invalid_trade_type_raises(self) -> None:
        """trade_type='DAY' is not valid and must raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_trade_signal(trade_type="DAY")

    def test_trade_signal_valid_trade_type_swing(self) -> None:
        """trade_type='SWING' must be accepted."""
        signal = _make_trade_signal(trade_type="SWING")
        assert signal.trade_type == "SWING"

    def test_trade_signal_generated_at_defaults_to_now(self) -> None:
        """generated_at must default to approximately now when not supplied."""
        before = datetime.now()
        signal = _make_trade_signal()
        after = datetime.now()
        assert before <= signal.generated_at <= after


# ---------------------------------------------------------------------------
# TradeSignal.is_actionable
# ---------------------------------------------------------------------------


class TestTradeSignalIsActionable:
    """Tests for the is_actionable computed property."""

    def test_trade_signal_is_actionable_true(self) -> None:
        """
        action=BUY, confidence=0.80, rr=2.0 → all three conditions met → True.
        """
        signal = _make_trade_signal(action="BUY", confidence=0.80, risk_reward_ratio=2.0)
        assert signal.is_actionable is True

    def test_trade_signal_is_actionable_sell_true(self) -> None:
        """action=SELL with sufficient confidence and rr must also be actionable."""
        signal = _make_trade_signal(action="SELL", confidence=0.70, risk_reward_ratio=1.6)
        assert signal.is_actionable is True

    def test_trade_signal_is_actionable_false_no_trade(self) -> None:
        """action=NO_TRADE is never actionable, regardless of confidence or rr."""
        signal = _make_trade_signal(
            action="NO_TRADE", confidence=1.0, risk_reward_ratio=5.0
        )
        assert signal.is_actionable is False

    def test_trade_signal_is_actionable_false_low_confidence(self) -> None:
        """confidence=0.5 (below the 0.65 threshold) → is_actionable False."""
        signal = _make_trade_signal(action="BUY", confidence=0.5, risk_reward_ratio=2.0)
        assert signal.is_actionable is False

    def test_trade_signal_is_actionable_false_at_confidence_boundary(self) -> None:
        """
        confidence exactly at 0.65 must pass (>= check).
        confidence exactly at 0.6499 must fail.
        """
        signal_pass = _make_trade_signal(action="BUY", confidence=0.65, risk_reward_ratio=2.0)
        assert signal_pass.is_actionable is True

        signal_fail = _make_trade_signal(action="BUY", confidence=0.6499, risk_reward_ratio=2.0)
        assert signal_fail.is_actionable is False

    def test_trade_signal_is_actionable_false_low_rr(self) -> None:
        """risk_reward_ratio=1.2 (below 1.5) → is_actionable False."""
        signal = _make_trade_signal(action="BUY", confidence=0.80, risk_reward_ratio=1.2)
        assert signal.is_actionable is False

    def test_trade_signal_is_actionable_false_rr_exactly_at_boundary(self) -> None:
        """risk_reward_ratio=1.5 must pass; 1.499 must fail."""
        signal_pass = _make_trade_signal(action="BUY", confidence=0.80, risk_reward_ratio=1.5)
        assert signal_pass.is_actionable is True

        signal_fail = _make_trade_signal(action="BUY", confidence=0.80, risk_reward_ratio=1.499)
        assert signal_fail.is_actionable is False


# ---------------------------------------------------------------------------
# TradeSignal.to_dict
# ---------------------------------------------------------------------------


class TestTradeSignalToDict:
    """Tests for the to_dict() serialisation helper."""

    def test_trade_signal_to_dict_returns_dict(self) -> None:
        """to_dict() must return a plain Python dict."""
        signal = _make_trade_signal()
        d = signal.to_dict()
        assert isinstance(d, dict)

    def test_trade_signal_to_dict_generated_at_is_string(self) -> None:
        """generated_at must be an ISO-8601 string, not a datetime object."""
        signal = _make_trade_signal()
        d = signal.to_dict()
        assert isinstance(d["generated_at"], str), (
            f"Expected str, got {type(d['generated_at'])}"
        )
        # Must be parseable as ISO-8601
        datetime.fromisoformat(d["generated_at"])

    def test_trade_signal_to_dict_has_all_expected_keys(self) -> None:
        """All expected top-level keys must be present in the output dict."""
        signal = _make_trade_signal()
        d = signal.to_dict()

        expected_keys = {
            "symbol", "action", "trade_type",
            "entry_price", "stop_loss", "target_1", "target_2",
            "confidence", "risk_reward_ratio",
            "reasoning", "key_risks", "invalidation_condition",
            "generated_at",
        }
        assert expected_keys.issubset(d.keys()), (
            f"Missing keys: {expected_keys - d.keys()}"
        )

    def test_trade_signal_to_dict_values_match_model(self) -> None:
        """Values in to_dict() must match the model fields (spot-check)."""
        signal = _make_trade_signal(action="SELL", confidence=0.75, risk_reward_ratio=1.8)
        d = signal.to_dict()

        assert d["action"] == "SELL"
        assert d["confidence"] == signal.confidence
        assert d["risk_reward_ratio"] == 1.8
        assert d["symbol"] == "RELIANCE"


# ---------------------------------------------------------------------------
# LLMSignalEngine.generate_signal — indicators=None fast-path
# ---------------------------------------------------------------------------


class TestLLMSignalEngineNoIndicators:
    """Tests verifying that generate_signal returns NO_TRADE without calling the API."""

    @pytest.mark.asyncio
    async def test_generate_signal_returns_no_trade_on_none_indicators(self) -> None:
        """
        When indicators=None, generate_signal must:
        - Return a TradeSignal with action == "NO_TRADE".
        - Never instantiate or call the OpenAI client.
        """
        engine = _make_llm_engine()

        # Patch the underlying AsyncOpenAI client that the engine created
        with patch.object(engine, "_client") as mock_client:
            signal = await engine.generate_signal(
                symbol="INFY",
                exchange="NSE",
                trade_type="INTRADAY",
                indicators=None,
                patterns=None,
                news_summary="No news.",
            )

        assert isinstance(signal, TradeSignal)
        assert signal.action == "NO_TRADE"
        assert signal.symbol == "INFY"
        # The OpenAI client must not have been touched
        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_signal_no_trade_is_not_actionable(self) -> None:
        """The NO_TRADE signal returned for None indicators must not be actionable."""
        engine = _make_llm_engine()
        with patch.object(engine, "_client"):
            signal = await engine.generate_signal(
                symbol="WIPRO",
                exchange="NSE",
                trade_type="INTRADAY",
                indicators=None,
                patterns=None,
                news_summary="",
            )
        assert signal.is_actionable is False

    @pytest.mark.asyncio
    async def test_generate_signal_with_valid_indicators_calls_api(self) -> None:
        """
        When indicators are provided, the engine must call the OpenAI API and
        return a TradeSignal built from the response.
        """
        engine = _make_llm_engine()
        indicators = _make_mock_indicators()

        api_payload: dict[str, Any] = {
            "action": "BUY",
            "trade_type": "INTRADAY",
            "entry_price": 2500.0,
            "stop_loss": 2450.0,
            "target_1": 2600.0,
            "target_2": None,
            "confidence": 0.75,
            "risk_reward_ratio": 2.0,
            "reasoning": "Momentum confirmed.",
            "key_risks": ["Broad market weakness"],
            "invalidation_condition": "Close below 2440",
        }

        mock_response = _make_openai_response(api_payload)

        with patch.object(
            engine._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            signal = await engine.generate_signal(
                symbol="RELIANCE",
                exchange="NSE",
                trade_type="INTRADAY",
                indicators=indicators,
                patterns=None,
                news_summary="No material news.",
            )

        assert isinstance(signal, TradeSignal)
        assert signal.action == "BUY"
        assert signal.symbol == "RELIANCE"
        assert signal.confidence == 0.75

    @pytest.mark.asyncio
    async def test_generate_signal_falls_back_on_api_error(self) -> None:
        """
        If the OpenAI API raises an exception, generate_signal must return a
        NO_TRADE signal rather than propagating the exception.
        """
        engine = _make_llm_engine()
        indicators = _make_mock_indicators()

        with patch.object(
            engine._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=RuntimeError("connection timeout")),
        ):
            signal = await engine.generate_signal(
                symbol="HDFC",
                exchange="NSE",
                trade_type="INTRADAY",
                indicators=indicators,
                patterns=None,
                news_summary="",
            )

        assert isinstance(signal, TradeSignal)
        assert signal.action == "NO_TRADE"
        assert signal.is_actionable is False

    @pytest.mark.asyncio
    async def test_generate_signal_falls_back_on_invalid_json(self) -> None:
        """
        Malformed JSON from the API must cause a graceful NO_TRADE fallback.
        """
        engine = _make_llm_engine()
        indicators = _make_mock_indicators()

        # Return a response whose content is not valid JSON
        bad_message = MagicMock()
        bad_message.content = "this is not json"
        bad_choice = MagicMock()
        bad_choice.message = bad_message
        bad_response = MagicMock()
        bad_response.choices = [bad_choice]

        with patch.object(
            engine._client.chat.completions,
            "create",
            new=AsyncMock(return_value=bad_response),
        ):
            signal = await engine.generate_signal(
                symbol="SBIN",
                exchange="NSE",
                trade_type="INTRADAY",
                indicators=indicators,
                patterns=None,
                news_summary="",
            )

        assert signal.action == "NO_TRADE"


# ---------------------------------------------------------------------------
# LLMSignalEngine._build_no_trade_signal
# ---------------------------------------------------------------------------


class TestBuildNoTradeSignal:
    """Tests for the internal _build_no_trade_signal helper."""

    def test_build_no_trade_signal_has_correct_action(self) -> None:
        """_build_no_trade_signal must always produce action == 'NO_TRADE'."""
        engine = _make_llm_engine()
        signal = engine._build_no_trade_signal("TCS")
        assert signal.action == "NO_TRADE"

    def test_build_no_trade_signal_propagates_symbol(self) -> None:
        """The symbol argument must appear in the returned signal."""
        engine = _make_llm_engine()
        signal = engine._build_no_trade_signal("INFOSYS")
        assert signal.symbol == "INFOSYS"

    def test_build_no_trade_signal_custom_reason(self) -> None:
        """When a reason is supplied, it must appear in the reasoning field."""
        engine = _make_llm_engine()
        reason = "Market closed for holiday."
        signal = engine._build_no_trade_signal("WIPRO", reason=reason)
        assert reason in signal.reasoning
