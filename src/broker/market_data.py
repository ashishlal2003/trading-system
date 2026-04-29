import pandas as pd
from datetime import datetime
from src.broker.groww_client import GrowwClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

INTERVAL_MAP = {
    "1m": "1minute",
    "5m": "5minute",
    "15m": "15minute",
    "30m": "30minute",
    "1h": "1hour",
    "1d": "1day",
}


class MarketDataService:
    """
    Fetches OHLCV candles and live quotes from Groww REST API.

    Endpoint reference (base: https://api.groww.in/v1):
      Historical candles : GET /historical/candles
      Live quote         : GET /live-data/quote
      Multiple LTPs      : GET /live-data/ltp
      Holdings           : GET /holdings/user
      Positions          : GET /positions/user
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
        """Returns DataFrame: [timestamp, open, high, low, close, volume]"""
        params = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": INTERVAL_MAP.get(interval, interval),
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
        }
        logger.info("fetch_candles", symbol=symbol, interval=interval)
        data = await self.client.get("/historical/candles", params=params)

        candles = data.get("candles", data.get("data", []))
        if not candles:
            logger.warning("no_candles_returned", symbol=symbol)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
        df["volume"] = df["volume"].astype(int)
        return df.sort_values("timestamp").reset_index(drop=True)

    async def get_live_quote(self, symbol: str, exchange: str = "NSE") -> dict:
        """Returns dict with ltp and other quote fields."""
        data = await self.client.get("/live-data/quote", params={"symbol": symbol, "exchange": exchange})
        return data.get("data", data)

    async def get_multiple_quotes(self, symbols: list[str], exchange: str = "NSE") -> dict[str, dict]:
        """Returns {symbol: ltp_dict} for all symbols via the LTP endpoint."""
        params = {"symbols": ",".join(symbols), "exchange": exchange}
        data = await self.client.get("/live-data/ltp", params=params)
        quotes = data.get("data", data)
        if isinstance(quotes, list):
            return {q["symbol"]: q for q in quotes}
        return quotes

    async def get_portfolio(self) -> list[dict]:
        """Returns list of holdings: symbol, qty, avg_price, current_price."""
        data = await self.client.get("/holdings/user")
        return data.get("holdings", data.get("data", []))

    async def get_positions(self) -> list[dict]:
        """Returns list of current day positions."""
        data = await self.client.get("/positions/user")
        return data.get("positions", data.get("data", []))
