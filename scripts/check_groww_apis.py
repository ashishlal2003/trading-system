#!/usr/bin/env python3
"""
Groww API health check — tests every endpoint the trading system uses.

Run this before market hours to confirm everything is working:
    python scripts/check_groww_apis.py

Tests (in order):
  1. TOTP token exchange
  2. Single live quote  (/live-data/quote)
  3. Batch LTP          (/live-data/ltp)
  4. Holdings           (/holdings/user)
  5. Positions          (/positions/user)
  6. yfinance candle fetch (comparison sanity check)
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.broker.groww_client import GrowwClient, GrowwAPIError
from src.broker.market_data import MarketDataService

# Symbols to test with — taken from the intraday watchlist
TEST_SYMBOLS = ["SBIN", "WIPRO", "NHPC"]
EXCHANGE = "NSE"

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def _header(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def _result(label: str, ok: bool, detail: str = "") -> None:
    icon = PASS if ok else FAIL
    line = f"{icon}  {label}"
    if detail:
        line += f"\n     {detail}"
    print(line)


async def test_token(client: GrowwClient) -> bool:
    _header("1. TOTP token exchange")
    try:
        token = await client.refresh_access_token()
        _result("Token refreshed", True, f"prefix: {token[:20]}...")
        return True
    except GrowwAPIError as e:
        _result("Token exchange", False, f"HTTP {e.status_code}: {e.message}")
        print("\n  Check GROWW_API_KEY (fresh from groww.in/trade-api) and GROWW_TOTP_SECRET in .env")
        return False
    except Exception as e:
        _result("Token exchange", False, str(e))
        return False


async def test_live_quote(svc: MarketDataService) -> bool:
    _header("2. Single live quote  (/live-data/quote)")
    all_ok = True
    for sym in TEST_SYMBOLS:
        try:
            data = await svc.get_live_quote(sym, exchange=EXCHANGE)
            ltp = data.get("last_price") or (data.get("ohlc") or {}).get("close")
            if ltp:
                _result(f"{sym}", True, f"last_price = ₹{float(ltp):,.2f}")
            else:
                _result(f"{sym}", False, f"no last_price in response: {str(data)[:120]}")
                all_ok = False
        except GrowwAPIError as e:
            _result(f"{sym}", False, f"HTTP {e.status_code}: {e.message}")
            all_ok = False
        except Exception as e:
            _result(f"{sym}", False, str(e))
            all_ok = False
    return all_ok


async def test_batch_ltp(svc: MarketDataService) -> bool:
    _header("3. Batch LTP  (/live-data/ltp)")
    try:
        quotes = await svc.get_multiple_quotes(TEST_SYMBOLS, exchange=EXCHANGE)
        if not quotes:
            _result("Batch LTP", False, "empty response")
            return False
        for sym, q in quotes.items():
            ltp = q.get("ltp")
            if ltp:
                _result(f"{sym}", True, f"ltp = ₹{float(ltp):,.2f}")
            else:
                _result(f"{sym}", False, f"no ltp in: {q}")
        missing = [s for s in TEST_SYMBOLS if s not in quotes]
        if missing:
            print(f"{WARN} Missing symbols in response: {missing}")
        return True
    except GrowwAPIError as e:
        _result("Batch LTP", False, f"HTTP {e.status_code}: {e.message}")
        return False
    except Exception as e:
        _result("Batch LTP", False, str(e))
        return False


async def test_holdings(svc: MarketDataService) -> bool:
    _header("4. Holdings  (/holdings/user)")
    try:
        holdings = await svc.get_portfolio()
        _result("Holdings fetch", True, f"{len(holdings)} holding(s) returned")
        for h in holdings[:3]:
            sym = h.get("trading_symbol") or h.get("symbol") or "?"
            qty = h.get("quantity") or h.get("qty") or "?"
            avg = h.get("average_price") or h.get("avg_price") or "?"
            print(f"     {sym}  qty={qty}  avg=₹{avg}")
        return True
    except GrowwAPIError as e:
        _result("Holdings fetch", False, f"HTTP {e.status_code}: {e.message}")
        return False
    except Exception as e:
        _result("Holdings fetch", False, str(e))
        return False


async def test_positions(svc: MarketDataService) -> bool:
    _header("5. Positions  (/positions/user)")
    try:
        positions = await svc.get_positions()
        _result("Positions fetch", True, f"{len(positions)} position(s) returned")
        for p in positions[:3]:
            sym = p.get("trading_symbol") or p.get("symbol") or "?"
            qty = p.get("quantity") or p.get("qty") or "?"
            print(f"     {sym}  qty={qty}")
        return True
    except GrowwAPIError as e:
        _result("Positions fetch", False, f"HTTP {e.status_code}: {e.message}")
        return False
    except Exception as e:
        _result("Positions fetch", False, str(e))
        return False


def test_yfinance_candles() -> bool:
    _header("6. yfinance candle fetch (indicator data source)")
    try:
        import yfinance as yf
        now = datetime.utcnow()
        start = (now - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        all_ok = True
        for sym in TEST_SYMBOLS:
            yf_sym = f"{sym}.NS"
            df = yf.download(yf_sym, start=start, end=end, interval="5m",
                             progress=False, auto_adjust=True)
            if df.empty:
                _result(f"{sym}", False, "no candles returned")
                all_ok = False
                continue
            last_ts = df.index[-1]
            last_close = float(df["Close"].iloc[-1]) if "Close" in df.columns else float(df.iloc[-1, 3])
            age_minutes = (datetime.utcnow() - last_ts.to_pydatetime().replace(tzinfo=None)).total_seconds() / 60
            stale = age_minutes > 30
            icon = WARN if stale else PASS
            status = f"STALE ({age_minutes:.0f} min old)" if stale else f"{age_minutes:.0f} min old"
            print(f"{icon}  {sym}  last_close=₹{last_close:,.2f}  last_candle={status}  rows={len(df)}")
        return all_ok
    except Exception as e:
        _result("yfinance", False, str(e))
        return False


async def main() -> None:
    print(f"\n{'═' * 55}")
    print("  Groww API Health Check")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Base URL: {settings.GROWW_BASE_URL}")
    print(f"{'═' * 55}")

    results: dict[str, bool] = {}

    async with GrowwClient(
        api_key=settings.GROWW_API_KEY,
        api_secret=settings.GROWW_API_SECRET,
        base_url=settings.GROWW_BASE_URL,
        totp_secret=settings.GROWW_TOTP_SECRET,
    ) as client:

        token_ok = await test_token(client)
        results["TOTP token"] = token_ok

        if not token_ok:
            print(f"\n{FAIL}  Token exchange failed — skipping all API tests.\n")
            sys.exit(1)

        svc = MarketDataService(client=client)

        results["Live quote"]  = await test_live_quote(svc)
        results["Batch LTP"]   = await test_batch_ltp(svc)
        results["Holdings"]    = await test_holdings(svc)
        results["Positions"]   = await test_positions(svc)

    results["yfinance candles"] = test_yfinance_candles()

    # Summary
    _header("Summary")
    all_passed = True
    for name, ok in results.items():
        _result(name, ok)
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print(f"{PASS}  All checks passed — system ready.\n")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"{FAIL}  {len(failed)} check(s) failed: {', '.join(failed)}\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
