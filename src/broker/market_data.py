import asyncio
from datetime import datetime, timedelta
from functools import partial

import pandas as pd
import yfinance as yf

from src.broker.groww_client import GrowwClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

# yfinance uses the same short codes we do ("1m", "5m", "1d", etc.)
_YF_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
}


def _yf_symbol(symbol: str, exchange: str) -> str:
    return f"{symbol}.NS" if exchange.upper() in ("NSE", "") else f"{symbol}.BO"


def _fetch_candles_sync(yf_sym: str, interval: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(yf_sym, start=start, end=end, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    # Flatten MultiIndex columns present in newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    df.index.name = "timestamp"
    df = df.reset_index()[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df["volume"] = df["volume"].astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


class MarketDataService:
    """
    Fetches OHLCV candles and live quotes for the trading system.

    Historical candles come from Yahoo Finance (yfinance) because the Groww
    API's basic-tier token does not permit the /historical/candles endpoint.

    Live quotes and LTP come from the Groww REST API (correct parameter names
    discovered via live testing):
      Live quote  : GET /live-data/quote  — params: trading_symbol, exchange, segment
      Multiple LTP: GET /live-data/ltp   — params: exchange_symbols (NSE_SYM format), segment
      Holdings    : GET /holdings/user
      Positions   : GET /positions/user
    """

    def __init__(self, client: GrowwClient):
        self.client = client

    async def get_candles(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        """Returns DataFrame: [timestamp, open, high, low, close, volume] via yfinance."""
        yf_sym = _yf_symbol(symbol, exchange)
        yf_interval = _YF_INTERVAL_MAP.get(interval, interval)
        start = from_date.strftime("%Y-%m-%d")
        # yfinance end is exclusive (midnight UTC), so add 1 day to include today's candles
        end = (to_date + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info("fetch_candles_yf", symbol=symbol, yf_sym=yf_sym, interval=yf_interval)
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, partial(_fetch_candles_sync, yf_sym, yf_interval, start, end))
        if df.empty:
            logger.warning("no_candles_returned", symbol=symbol)
        return df

    async def get_live_quote(self, symbol: str, exchange: str = "NSE") -> dict:
        """Returns dict with ltp and other quote fields from Groww live-data/quote."""
        data = await self.client.get("/live-data/quote", params={
            "trading_symbol": symbol,
            "exchange": exchange,
            "segment": "CASH",
        })
        return data.get("payload", data)

    async def get_multiple_quotes(self, symbols: list[str], exchange: str = "NSE") -> dict[str, dict]:
        """
        Returns {symbol: {"ltp": price, "symbol": symbol}} for all symbols.

        Groww LTP endpoint uses NSE_SYMBOL underscore format and requires segment=CASH.
        """
        exchange_symbols = ",".join(f"{exchange}_{s}" for s in symbols)
        data = await self.client.get("/live-data/ltp", params={
            "exchange_symbols": exchange_symbols,
            "segment": "CASH",
        })
        payload = data.get("payload", data)
        # payload is {"NSE_RELIANCE": 1421.6, "NSE_TCS": 2482.0}
        result: dict[str, dict] = {}
        for key, price in payload.items():
            sym = key.split("_", 1)[1] if "_" in key else key
            result[sym] = {"ltp": price, "symbol": sym}
        return result

    async def get_portfolio(self) -> list[dict]:
        """Returns list of holdings: symbol, qty, avg_price, current_price."""
        data = await self.client.get("/holdings/user")
        return data.get("holdings", data.get("data", []))

    async def get_positions(self) -> list[dict]:
        """Returns list of current day positions."""
        data = await self.client.get("/positions/user")
        return data.get("positions", data.get("data", []))
