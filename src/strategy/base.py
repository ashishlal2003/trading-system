from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class SignalResult:
    """
    Output of a strategy's evaluate() call for one bar.

    action        : "BUY", "SELL", or "NO_TRADE"
    entry_price   : Suggested fill price (typically the bar close).
    stop_loss     : Hard stop-loss level.
    target        : Primary profit target.
    reasoning     : Human-readable explanation of why the signal fired.
    """
    action: str           # "BUY" | "SELL" | "NO_TRADE"
    entry_price: float
    stop_loss: float
    target: float
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.action in ("BUY", "SELL")

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target - self.entry_price)
        return reward / risk if risk > 0 else 0.0


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Every strategy must implement:
      - evaluate(df, current_bar_index) → SignalResult
      - get_params() → dict
      - set_params(params)

    The evaluate() contract:
      - df       : Full OHLCV DataFrame sorted ascending by timestamp.
                   Columns required: timestamp, open, high, low, close, volume.
      - bar_idx  : Index of the *current* bar being evaluated.
                   The strategy MUST NOT read df.iloc[bar_idx + 1] or beyond.
      - Returns  : SignalResult for this bar.
    """

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, bar_idx: int) -> SignalResult:
        """Evaluate one bar. Must not read ahead of bar_idx."""
        ...

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return current strategy parameters as a plain dict."""
        ...

    @abstractmethod
    def set_params(self, params: dict[str, Any]) -> None:
        """Apply a parameter dict (used by walk-forward optimizer)."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all intraday state (called before each backtest fold)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable strategy name, e.g. 'ORB-15m'."""
        ...
