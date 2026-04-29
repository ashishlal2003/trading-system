"""
Central configuration for the trading system.

All values are read from environment variables or a .env file at the project
root.  Import the ready-made singleton:

    from config.settings import settings
"""

from __future__ import annotations

import sys
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # Groww broker
    # ------------------------------------------------------------------ #
    GROWW_API_KEY: str = Field(..., description="Groww API key (required)")
    GROWW_API_SECRET: str = Field(..., description="Groww API secret (required)")
    GROWW_BASE_URL: str = Field(
        default="https://api.groww.in/v1",
        description="Groww REST API base URL",
    )
    GROWW_TOTP_SECRET: str = Field(
        default="",
        description="TOTP secret from Groww API dashboard (32-char string). Required to exchange the API key JWT for a live access token.",
    )

    # ------------------------------------------------------------------ #
    # OpenAI / LLM
    # ------------------------------------------------------------------ #
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key (required)")
    OPENAI_MODEL: str = Field(
        default="gpt-4o",
        description="OpenAI model to use for signal analysis",
    )
    OPENAI_MAX_TOKENS: int = Field(
        default=800,
        description="Maximum tokens for each LLM completion",
    )

    # ------------------------------------------------------------------ #
    # Telegram
    # ------------------------------------------------------------------ #
    TELEGRAM_BOT_TOKEN: str = Field(
        ..., description="Telegram bot token from @BotFather (required)"
    )
    TELEGRAM_CHAT_ID: str = Field(
        ..., description="Telegram chat/channel ID to send alerts (required)"
    )

    # ------------------------------------------------------------------ #
    # Capital & risk management
    # ------------------------------------------------------------------ #
    TOTAL_CAPITAL: float = Field(
        default=100_000.0,
        description="Total deployable capital in INR",
    )
    MAX_RISK_PER_TRADE_PCT: float = Field(
        default=1.0,
        description="Maximum capital at risk per trade as a percentage of TOTAL_CAPITAL",
    )
    MAX_DAILY_LOSS_PCT: float = Field(
        default=3.0,
        description="Kill-switch threshold: stop trading if daily P&L < -MAX_DAILY_LOSS_PCT %",
    )
    MAX_OPEN_POSITIONS: int = Field(
        default=5,
        description="Maximum number of concurrent open positions",
    )
    INTRADAY_LEVERAGE: float = Field(
        default=5.0,
        description="Intraday MIS leverage multiplier (5x on Groww for most stocks)",
    )
    INTRADAY_SL_PCT: float = Field(
        default=0.5,
        description="Default intraday stop-loss as a percentage of entry price",
    )
    SWING_SL_PCT: float = Field(
        default=2.0,
        description="Default swing trade stop-loss as a percentage of entry price",
    )

    # ------------------------------------------------------------------ #
    # Market timing (IST, 24-hour HH:MM strings)
    # ------------------------------------------------------------------ #
    PRE_MARKET_SCAN_TIME: str = Field(
        default="09:00",
        description="Time to run pre-market watchlist scan (IST)",
    )
    MARKET_OPEN: str = Field(
        default="09:15",
        description="NSE/BSE market open time (IST)",
    )
    MARKET_CLOSE: str = Field(
        default="15:30",
        description="NSE/BSE market close time (IST)",
    )
    INTRADAY_SCAN_INTERVAL_SECONDS: int = Field(
        default=300,
        description="How often (seconds) to re-scan intraday watchlist during market hours",
    )
    SQUARE_OFF_TIME: str = Field(
        default="15:10",
        description="Time to auto-square-off all intraday positions (IST)",
    )

    # ------------------------------------------------------------------ #
    # Data & indicators
    # ------------------------------------------------------------------ #
    CANDLE_HISTORY_DAYS: int = Field(
        default=60,
        description="Number of calendar days of OHLCV history to fetch",
    )
    INDICATOR_CANDLES: int = Field(
        default=200,
        description="Number of candles required to compute indicators reliably",
    )

    # ------------------------------------------------------------------ #
    # Operational
    # ------------------------------------------------------------------ #
    PAPER_TRADE: bool = Field(
        default=True,
        description="When True, log orders without sending them to the broker",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Python logging level: DEBUG | INFO | WARNING | ERROR",
    )
    DB_PATH: str = Field(
        default="db/trading.db",
        description="Path to the SQLite database file (relative to project root)",
    )

    # ------------------------------------------------------------------ #
    # Pydantic-settings configuration
    # ------------------------------------------------------------------ #
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    @property
    def max_risk_per_trade_inr(self) -> float:
        """Absolute INR amount at risk per trade."""
        return self.TOTAL_CAPITAL * self.MAX_RISK_PER_TRADE_PCT / 100.0

    @property
    def max_daily_loss_inr(self) -> float:
        """Absolute INR daily-loss kill-switch level."""
        return self.TOTAL_CAPITAL * self.MAX_DAILY_LOSS_PCT / 100.0


# --------------------------------------------------------------------------- #
# Module-level singleton
# Wrapped in try/except so that importing this module in CI or during testing
# without a real .env file does not raise a hard ValidationError at import time.
# --------------------------------------------------------------------------- #
try:
    settings: Settings = Settings()  # type: ignore[call-arg]
except Exception as exc:  # pragma: no cover
    print(
        f"\n[config/settings.py] WARNING: Could not load Settings — {exc}\n"
        "Make sure you have a valid .env file (copy .env.example and fill in "
        "the required values) before running the trading system.\n",
        file=sys.stderr,
    )
    settings = None  # type: ignore[assignment]
