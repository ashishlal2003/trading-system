#!/usr/bin/env python3
"""
Backfill historical OHLCV candles for all watchlist symbols.
Run once before starting the trading bot.
Usage: python scripts/backfill_candles.py
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import yaml

from config.settings import settings
from src.broker.groww_client import GrowwClient
from src.broker.market_data import MarketDataService
from src.data.store import CandleStore
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

async def backfill(symbols: list[str], interval: str, days_back: int):
    store = CandleStore(settings.DB_PATH)
    await store.init_db()

    async with GrowwClient(
        settings.GROWW_API_KEY,
        settings.GROWW_API_SECRET,
        settings.GROWW_BASE_URL,
        settings.GROWW_ALGO_ID,
    ) as client:
        market_data = MarketDataService(client)
        from_date = datetime.now() - timedelta(days=days_back)
        to_date = datetime.now()

        for symbol in symbols:
            try:
                logger.info("backfilling", symbol=symbol, interval=interval)
                df = await market_data.get_candles(symbol, "NSE", interval, from_date, to_date)
                await store.upsert(symbol, interval, df)
                logger.info("backfill_done", symbol=symbol, rows=len(df))
            except Exception as e:
                logger.error("backfill_failed", symbol=symbol, error=str(e))

async def main():
    configure_logging()
    with open("config/watchlist.yaml") as f:
        watchlist = yaml.safe_load(f)

    all_symbols = watchlist["intraday"] + watchlist["swing"]
    # Remove duplicates while preserving order
    seen = set()
    unique_symbols = [s for s in all_symbols if not (s in seen or seen.add(s))]

    logger.info("starting_backfill", symbols=unique_symbols)

    # Backfill 5-minute candles (60 days) and daily candles (365 days)
    await backfill(unique_symbols, "5m", 60)
    await backfill(unique_symbols, "1d", 365)

    logger.info("backfill_complete")

if __name__ == "__main__":
    asyncio.run(main())
