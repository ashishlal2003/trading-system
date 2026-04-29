"""
LLM Signal Engine — async GPT-4o powered trading signal generator.

Architecture
------------
- TradeSignal        : Pydantic model representing a single trading decision.
- LLMSignalEngine    : Async engine that calls OpenAI, parses the response,
                       validates it, and returns a TradeSignal.

The engine is intentionally sequential when processing batches (batch_generate)
to keep OpenAI API costs predictable.  Do NOT switch to asyncio.gather() here
without adding a rate-limiter and a cost-cap guard.

Typical usage
-------------
    engine = LLMSignalEngine(api_key=os.environ["OPENAI_API_KEY"])
    signal = await engine.generate_signal(
        symbol="RELIANCE",
        exchange="NSE",
        trade_type="INTRADAY",
        indicators=indicator_result,
        patterns=pattern_result,
        news_summary="No material news today.",
    )
    if signal.is_actionable:
        await place_order(signal)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

from src.data.indicators import IndicatorResult
from src.signals.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Optional import — PatternResult may not exist yet; fall back gracefully.
# ---------------------------------------------------------------------------
try:
    from src.data.patterns import PatternResult  # type: ignore[import]
except ModuleNotFoundError:
    PatternResult = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_ACTIONS: frozenset[str] = frozenset({"BUY", "SELL", "NO_TRADE"})
_VALID_TRADE_TYPES: frozenset[str] = frozenset({"INTRADAY", "SWING"})

_NO_TRADE_DEFAULTS: dict[str, Any] = {
    "action": "NO_TRADE",
    "trade_type": "INTRADAY",
    "entry_price": 0.0,
    "stop_loss": 0.0,
    "target_1": 0.0,
    "target_2": None,
    "confidence": 0.0,
    "risk_reward_ratio": 0.0,
    "reasoning": "Signal generation skipped — insufficient data or an error occurred.",
    "key_risks": ["No data available"],
    "invalidation_condition": "N/A",
}


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class TradeSignal(BaseModel):
    """Validated, serialisable representation of a single trading signal."""

    symbol: str
    action: str
    trade_type: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float | None = None
    confidence: float
    risk_reward_ratio: float
    reasoning: str
    key_risks: list[str]
    invalidation_condition: str
    generated_at: datetime = Field(default_factory=datetime.now)

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        normalised = v.upper().strip()
        if normalised not in _VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(_VALID_ACTIONS)}, got '{v}'"
            )
        return normalised

    @field_validator("trade_type")
    @classmethod
    def validate_trade_type(cls, v: str) -> str:
        normalised = v.upper().strip()
        if normalised not in _VALID_TRADE_TYPES:
            raise ValueError(
                f"trade_type must be one of {sorted(_VALID_TRADE_TYPES)}, got '{v}'"
            )
        return normalised

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return round(v, 4)

    # ------------------------------------------------------------------
    # Computed property
    # ------------------------------------------------------------------

    @property
    def is_actionable(self) -> bool:
        """
        Returns True only when all three conditions are met:
        - action is BUY or SELL (not NO_TRADE)
        - confidence is at least 0.65
        - risk/reward ratio is at least 1.5
        """
        return (
            self.action != "NO_TRADE"
            and self.confidence >= 0.65
            and self.risk_reward_ratio >= 1.5
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return a plain dict suitable for JSON serialisation or database
        persistence.  datetime fields are converted to ISO-8601 strings.
        """
        raw = self.model_dump()
        raw["generated_at"] = self.generated_at.isoformat()
        return raw


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LLMSignalEngine:
    """
    Async engine that calls OpenAI GPT-4o and converts the JSON response into
    a validated TradeSignal.

    Parameters
    ----------
    api_key   : OpenAI API key.
    model     : Model ID to use (default: "gpt-4o").
    max_tokens: Upper bound on response tokens (default: 800).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 800,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_no_trade_signal(
        self, symbol: str, reason: str | None = None
    ) -> TradeSignal:
        """Return a safe NO_TRADE signal, optionally overriding the reasoning."""
        payload = dict(_NO_TRADE_DEFAULTS)
        payload["symbol"] = symbol
        if reason:
            payload["reasoning"] = reason
        return TradeSignal(**payload)

    def _format_float(self, value: float | None, decimals: int = 2) -> str:
        """Safely format a float for prompt insertion."""
        if value is None:
            return "N/A"
        return f"{value:.{decimals}f}"

    def _build_user_prompt(
        self,
        symbol: str,
        exchange: str,
        trade_type: str,
        indicators: IndicatorResult,
        patterns: Any | None,
        news_summary: str,
        live_price: float | None = None,
    ) -> str:
        """Populate USER_PROMPT_TEMPLATE with live data."""
        # ------ Pattern fields ------
        if patterns is not None:
            patterns_detected = getattr(patterns, "detected", "None")
            patterns_bias = getattr(patterns, "bias", "NEUTRAL")
            # Support list or string for detected patterns
            if isinstance(patterns_detected, list):
                patterns_detected = ", ".join(patterns_detected) if patterns_detected else "None"
        else:
            patterns_detected = "None"
            patterns_bias = "NEUTRAL"

        # Use Groww live price if available; fall back to last yfinance candle close
        current_price = live_price if live_price is not None else indicators.close

        return USER_PROMPT_TEMPLATE.format(
            symbol=symbol,
            exchange=exchange,
            trade_type=trade_type,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            current_price=self._format_float(current_price),
            rsi_14=self._format_float(indicators.rsi_14),
            macd_line=self._format_float(indicators.macd_line, 4),
            macd_signal=self._format_float(indicators.macd_signal, 4),
            macd_hist=self._format_float(indicators.macd_hist, 4),
            ema_9=self._format_float(indicators.ema_9),
            ema_21=self._format_float(indicators.ema_21),
            ema_50=self._format_float(indicators.ema_50),
            ema_200=self._format_float(indicators.ema_200),
            bb_upper=self._format_float(indicators.bb_upper),
            bb_mid=self._format_float(indicators.bb_mid),
            bb_lower=self._format_float(indicators.bb_lower),
            bb_pct_b=self._format_float(indicators.bb_pct_b, 4),
            atr_14=self._format_float(indicators.atr_14),
            vwap=self._format_float(indicators.vwap),
            relative_volume=self._format_float(indicators.relative_volume, 2),
            trend=indicators.trend,
            price_vs_vwap=indicators.price_vs_vwap,
            patterns_detected=patterns_detected,
            patterns_bias=patterns_bias,
            news_summary=news_summary.strip() if news_summary else "No recent news available.",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_signal(
        self,
        symbol: str,
        exchange: str,
        trade_type: str,
        indicators: IndicatorResult | None,
        patterns: Any | None,
        news_summary: str,
        live_price: float | None = None,
    ) -> TradeSignal:
        """
        Generate a trading signal for a single symbol.

        Steps
        -----
        1. Short-circuit to NO_TRADE if indicators are missing.
        2. Build the user prompt from live data.
        3. Call OpenAI with json_object response format.
        4. Parse, enrich with metadata, and validate via TradeSignal.
        5. Log the outcome.
        6. Return TradeSignal (always — never raises to the caller).

        Parameters
        ----------
        symbol       : Ticker symbol, e.g. "RELIANCE".
        exchange     : Exchange name, e.g. "NSE" or "BSE".
        trade_type   : "INTRADAY" or "SWING".
        indicators   : Computed IndicatorResult, or None if unavailable.
        patterns     : PatternResult (or compatible object), or None.
        news_summary : Free-text news context fed verbatim into the prompt.

        Returns
        -------
        TradeSignal  : Always returns a valid TradeSignal; falls back to
                       NO_TRADE on any error.
        """
        # Step 1 — guard: no indicators means we cannot form a valid prompt
        if indicators is None:
            logger.warning(
                "llm_engine.no_indicators",
                symbol=symbol,
                reason="IndicatorResult is None; returning NO_TRADE without API call",
            )
            return self._build_no_trade_signal(
                symbol,
                reason="Indicators unavailable — cannot generate a signal without technical data.",
            )

        try:
            # Step 2 — build prompt
            user_prompt = self._build_user_prompt(
                symbol=symbol,
                exchange=exchange,
                trade_type=trade_type,
                indicators=indicators,
                patterns=patterns,
                news_summary=news_summary,
                live_price=live_price,
            )

            # Step 3 — call OpenAI
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_content: str = response.choices[0].message.content or "{}"

            # Step 4 — parse JSON response
            parsed: dict[str, Any] = json.loads(raw_content)

            # Step 5 — inject metadata that the model does not produce
            parsed["symbol"] = symbol
            parsed["generated_at"] = datetime.now()

            # Step 6 — validate through Pydantic
            signal = TradeSignal(**parsed)

            # Step 7 — log outcome
            logger.info(
                "llm_engine.signal_generated",
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence,
                risk_reward_ratio=signal.risk_reward_ratio,
                is_actionable=signal.is_actionable,
            )

            # Step 8 — return
            return signal

        except Exception as exc:  # noqa: BLE001
            # Step 9 — safe fallback on any failure
            logger.error(
                "llm_engine.generate_signal_error",
                symbol=symbol,
                error=str(exc),
                exc_info=True,
            )
            return self._build_no_trade_signal(
                symbol,
                reason=f"Signal generation failed due to an internal error: {type(exc).__name__}.",
            )

    async def batch_generate(
        self,
        scan_results: list[dict[str, Any]],
        trade_type: str,
        news_cache: dict[str, str],
    ) -> list[TradeSignal]:
        """
        Generate signals for a list of scan results sequentially.

        Sequential (not concurrent) execution is intentional — parallel calls
        would spike OpenAI costs and hit rate limits on large scan lists.

        Parameters
        ----------
        scan_results : List of dicts, each containing at minimum:
                       - "symbol"     (str)
                       - "exchange"   (str)
                       - "indicators" (IndicatorResult | None)
                       - "patterns"   (PatternResult-compatible | None)
        trade_type   : "INTRADAY" or "SWING", applied to every result.
        news_cache   : Mapping of symbol -> news summary string.
                       If a symbol is absent, an empty summary is used.

        Returns
        -------
        list[TradeSignal]
            Only actionable signals (action != NO_TRADE, confidence >= 0.65,
            risk_reward_ratio >= 1.5) are returned.  Non-actionable signals
            are logged at debug level and discarded.
        """
        actionable: list[TradeSignal] = []

        for result in scan_results:
            symbol: str = result.get("symbol", "UNKNOWN")
            exchange: str = result.get("exchange", "NSE")
            indicators: IndicatorResult | None = result.get("indicators")
            patterns: Any | None = result.get("patterns")
            news_summary: str = news_cache.get(symbol, "No recent news available.")

            signal = await self.generate_signal(
                symbol=symbol,
                exchange=exchange,
                trade_type=trade_type,
                indicators=indicators,
                patterns=patterns,
                news_summary=news_summary,
            )

            if signal.is_actionable:
                actionable.append(signal)
                logger.info(
                    "llm_engine.batch_actionable_signal",
                    symbol=signal.symbol,
                    action=signal.action,
                    confidence=signal.confidence,
                    risk_reward_ratio=signal.risk_reward_ratio,
                )
            else:
                logger.debug(
                    "llm_engine.batch_signal_skipped",
                    symbol=symbol,
                    action=signal.action,
                    confidence=signal.confidence,
                    risk_reward_ratio=signal.risk_reward_ratio,
                )

        logger.info(
            "llm_engine.batch_complete",
            total_scanned=len(scan_results),
            actionable_count=len(actionable),
        )

        return actionable
