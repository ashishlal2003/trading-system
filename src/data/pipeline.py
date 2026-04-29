"""
Async data pipeline for the algorithmic trading system.

The pipeline orchestrates:
1. Fetching OHLCV candles from the broker's MarketDataService.
2. Persisting / updating candles in the CandleStore (SQLite).
3. Running the IndicatorEngine over the candle data.
4. Running the PatternDetector over the candle data.

Two public entry points are provided:

- ``scan_symbol``    – fetch + analyse a single symbol; returns a ScanResult.
- ``scan_watchlist`` – concurrently scan many symbols; returns list[ScanResult].
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytz

import pandas as pd

from src.broker.market_data import MarketDataService
from src.data.indicators import IndicatorEngine, IndicatorResult
from src.data.patterns import PatternDetector, PatternResult
from src.data.store import CandleStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ScanResult
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    """
    The aggregated output for one symbol after a full pipeline run.

    Attributes
    ----------
    symbol:
        Instrument ticker (e.g. ``"RELIANCE"``).
    exchange:
        Exchange code (e.g. ``"NSE"``).
    interval:
        Candle interval string (e.g. ``"5m"``).
    indicators:
        IndicatorResult for the most recent candle, or ``None`` if computation
        failed (too few rows or internal error).
    patterns:
        PatternResult for the most recent candle, or ``None`` if detection
        failed.
    candles_df:
        The raw OHLCV DataFrame returned by the broker / stored in cache.
        Empty DataFrame when the fetch failed.
    scanned_at:
        UTC timestamp when this scan was executed.
    error:
        Human-readable error message when the pipeline raised an exception;
        ``None`` on success.
    """

    symbol: str
    exchange: str
    interval: str
    indicators: IndicatorResult | None
    patterns: PatternResult | None
    candles_df: pd.DataFrame
    scanned_at: datetime
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the scan completed without a pipeline-level error."""
        return self.error is None


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------

# How far back to fetch candles for each scan mode
_DAYS_BACK: dict[str, int] = {
    "intraday": 5,
    "swing": 60,
}
_DEFAULT_DAYS_BACK: int = 5
_IST = pytz.timezone("Asia/Kolkata")


class DataPipeline:
    """
    Orchestrates candle fetching, storage, and analysis.

    Parameters
    ----------
    market_data:
        An initialised ``MarketDataService`` instance used for fetching OHLCV
        data from the broker.
    indicator_engine:
        An ``IndicatorEngine`` instance for computing technical indicators.
    pattern_detector:
        A ``PatternDetector`` instance for identifying candlestick patterns.
    candle_store:
        A ``CandleStore`` instance for persisting and retrieving candles.

    Example
    -------
    ::

        pipeline = DataPipeline(
            market_data=market_svc,
            indicator_engine=IndicatorEngine(),
            pattern_detector=PatternDetector(),
            candle_store=candle_store,
        )
        result = await pipeline.scan_symbol("RELIANCE", exchange="NSE", interval="5m")
    """

    def __init__(
        self,
        market_data: MarketDataService,
        indicator_engine: IndicatorEngine,
        pattern_detector: PatternDetector,
        candle_store: CandleStore,
    ) -> None:
        self._market_data = market_data
        self._indicator_engine = indicator_engine
        self._pattern_detector = pattern_detector
        self._candle_store = candle_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scan_symbol(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "5m",
        mode: str = "intraday",
    ) -> ScanResult:
        """
        Fetch candles for *symbol*, persist them, compute indicators and
        detect candlestick patterns.

        Parameters
        ----------
        symbol:
            Instrument ticker string.
        exchange:
            Exchange code; defaults to ``"NSE"``.
        interval:
            Candle interval key understood by ``MarketDataService``
            (e.g. ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"1d"``).
        mode:
            ``"intraday"`` fetches the last 5 days of candles;
            ``"swing"`` fetches the last 60 days.

        Returns
        -------
        ScanResult
            Always returns a ScanResult.  Check ``result.error`` for failures.
        """
        scanned_at = datetime.now(_IST)
        empty_df = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        try:
            # ------------------------------------------------------------------
            # 1. Determine look-back window (IST dates so yfinance aligns with NSE)
            # ------------------------------------------------------------------
            days_back = _DAYS_BACK.get(mode, _DEFAULT_DAYS_BACK)
            to_date = datetime.now(_IST)
            from_date = to_date - timedelta(days=days_back)

            logger.info(
                "scan_symbol_start",
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                mode=mode,
                days_back=days_back,
            )

            # ------------------------------------------------------------------
            # 2. Fetch candles from broker
            # ------------------------------------------------------------------
            df = await self._market_data.get_candles(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                from_date=from_date,
                to_date=to_date,
            )

            if df.empty:
                logger.warning(
                    "scan_symbol_no_candles",
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                )
                return ScanResult(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    indicators=None,
                    patterns=None,
                    candles_df=empty_df,
                    scanned_at=scanned_at,
                    error="No candles returned from broker",
                )

            # ------------------------------------------------------------------
            # 3. Persist / upsert to CandleStore
            # ------------------------------------------------------------------
            try:
                await self._candle_store.upsert(symbol=symbol, interval=interval, df=df)
                logger.debug(
                    "candles_upserted",
                    symbol=symbol,
                    interval=interval,
                    rows=len(df),
                )
            except Exception as store_err:
                # Non-fatal: log but continue so analysis still runs
                logger.warning(
                    "candle_store_upsert_failed",
                    symbol=symbol,
                    interval=interval,
                    error=str(store_err),
                )

            # ------------------------------------------------------------------
            # 4. Compute indicators
            # ------------------------------------------------------------------
            indicators: IndicatorResult | None = self._indicator_engine.compute(
                df=df, symbol=symbol
            )

            # ------------------------------------------------------------------
            # 5. Detect candlestick patterns
            # ------------------------------------------------------------------
            patterns: PatternResult | None = self._pattern_detector.detect(
                df=df, symbol=symbol
            )

            logger.info(
                "scan_symbol_complete",
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                candle_rows=len(df),
                has_indicators=indicators is not None,
                pattern_bias=patterns.bias if patterns else None,
                patterns_found=patterns.detected if patterns else [],
            )

            return ScanResult(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                indicators=indicators,
                patterns=patterns,
                candles_df=df,
                scanned_at=scanned_at,
            )

        except Exception as exc:
            logger.error(
                "scan_symbol_error",
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                error=str(exc),
                exc_info=True,
            )
            return ScanResult(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                indicators=None,
                patterns=None,
                candles_df=empty_df,
                scanned_at=scanned_at,
                error=str(exc),
            )

    async def scan_watchlist(
        self,
        symbols: list[str],
        exchange: str = "NSE",
        interval: str = "5m",
        mode: str = "intraday",
    ) -> list[ScanResult]:
        """
        Concurrently scan all symbols in *symbols* and return results in the
        same order as the input list.

        All symbols are scanned concurrently via ``asyncio.gather``.
        Individual failures are captured inside each ``ScanResult.error``
        field — a failure for one symbol does not abort the others.

        Parameters
        ----------
        symbols:
            List of ticker strings to scan.
        exchange:
            Exchange code applied to every symbol.
        interval:
            Candle interval applied to every symbol.
        mode:
            ``"intraday"`` or ``"swing"`` — controls look-back window.

        Returns
        -------
        list[ScanResult]
            One ScanResult per symbol, in the same order as *symbols*.
        """
        if not symbols:
            logger.warning("scan_watchlist_empty_symbols")
            return []

        logger.info(
            "scan_watchlist_start",
            symbol_count=len(symbols),
            exchange=exchange,
            interval=interval,
            mode=mode,
        )

        # Build one coroutine per symbol; gather preserves order
        tasks = [
            self.scan_symbol(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                mode=mode,
            )
            for symbol in symbols
        ]

        # return_exceptions=False is intentional: scan_symbol already catches
        # all exceptions internally and wraps them in ScanResult.error, so
        # gather will never see a raised exception from our coroutines.
        results: list[ScanResult] = await asyncio.gather(*tasks)

        # ------------------------------------------------------------------
        # Summary log
        # ------------------------------------------------------------------
        succeeded = sum(1 for r in results if r.ok)
        errored = len(results) - succeeded

        logger.info(
            "scan_watchlist_complete",
            total=len(results),
            succeeded=succeeded,
            errored=errored,
            exchange=exchange,
            interval=interval,
        )

        if errored:
            failed_symbols = [r.symbol for r in results if not r.ok]
            logger.warning(
                "scan_watchlist_errors",
                failed_symbols=failed_symbols,
                error_details={r.symbol: r.error for r in results if not r.ok},
            )

        return list(results)
