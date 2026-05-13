"""
TradeSignal model — shared data contract for the trading system.

Signal generation is now done by RuleEngine (src/signals/rule_engine.py),
not the LLM. This file retains only the TradeSignal Pydantic model so that
all downstream consumers (RiskManager, OrderManager, TelegramBot, TradeStore)
continue to work without changes.

The LLM is used exclusively for Telegram QnA via ChatEngine (chat_engine.py).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

_VALID_ACTIONS: frozenset[str] = frozenset({"BUY", "SELL", "NO_TRADE"})
_VALID_TRADE_TYPES: frozenset[str] = frozenset({"INTRADAY", "SWING"})


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

    @property
    def is_actionable(self) -> bool:
        """
        True when:
          - action is BUY or SELL
          - confidence >= 0.65  (rule engine always sets 1.0 when firing)
          - risk_reward_ratio >= 1.5
        """
        return (
            self.action != "NO_TRADE"
            and self.confidence >= 0.65
            and self.risk_reward_ratio >= 1.5
        )

    def to_dict(self) -> dict[str, Any]:
        raw = self.model_dump()
        raw["generated_at"] = self.generated_at.isoformat()
        return raw
