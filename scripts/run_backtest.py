#!/usr/bin/env python3
"""
Backtest runner — pull historical NSE data via yfinance and test a strategy.

Usage
-----
    # ORB strategy on RELIANCE, last 6 months, 5-min candles
    python scripts/run_backtest.py

    # Custom symbol, period, strategy
    python scripts/run_backtest.py --symbol INFY --period 3mo --strategy orb
    python scripts/run_backtest.py --symbol TCS  --period 6mo --strategy vwap
    python scripts/run_backtest.py --symbol HDFCBANK --period 12mo --walk-forward
"""

import argparse
import sys
from pathlib import Path

# Make sure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd

from src.strategy.orb import ORBStrategy
from src.strategy.vwap_reversion import VWAPReversionStrategy
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import MetricsEngine
from src.backtest.walk_forward import WalkForwardValidator
from src.backtest.report import BacktestReporter


def fetch_data(symbol: str, period: str, interval: str = "5m") -> pd.DataFrame:
    """Download OHLCV from yfinance and normalise column names.

    yfinance hard limits:
      5m  → last 60 days only
      1h  → last 730 days (2 years)
      1d  → unlimited (years of daily data)
    """
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    print(f"  Downloading {ticker} ({period} of {interval} data)...")
    raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

    if raw.empty:
        print(f"  ERROR: No data returned for {ticker}. Check symbol or internet connection.")
        sys.exit(1)

    # yfinance returns MultiIndex columns when auto_adjust=True; flatten them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw = raw.reset_index()
    raw = raw.rename(columns={"datetime": "timestamp", "date": "timestamp"})

    # Ensure timestamp column exists
    if "timestamp" not in raw.columns:
        raw = raw.rename(columns={raw.columns[0]: "timestamp"})

    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.dropna(subset=["close", "volume"])
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    print(f"  {len(raw)} bars downloaded ({raw['timestamp'].iloc[0].date()} → {raw['timestamp'].iloc[-1].date()})")
    return raw


def main():
    parser = argparse.ArgumentParser(description="Run a strategy backtest on NSE data")
    parser.add_argument("--symbol",   default="RELIANCE",  help="NSE symbol without .NS suffix (default: RELIANCE)")
    parser.add_argument("--period",   default="60d",       help="yfinance period for 5-min data: max 60d (yfinance hard limit). Use --interval 1d for longer history (default: 60d)")
    parser.add_argument("--strategy", default="orb",       choices=["orb", "vwap"], help="Strategy to test (default: orb)")
    parser.add_argument("--capital",  default=10_000, type=float, help="Starting capital in ₹ (default: 10000)")
    parser.add_argument("--walk-forward", action="store_true", help="Also run walk-forward validation (needs 90d+ of data — use daily interval)")
    parser.add_argument("--quantity", default=None, type=int, help="Fixed quantity per trade (default: auto-size)")
    parser.add_argument("--leverage", default=5.0, type=float, help="Intraday leverage multiplier (default: 5.0 — matches Groww MIS margin)")
    parser.add_argument("--interval", default="5m", choices=["5m", "1h", "1d"], help="Candle interval: 5m (max 60d), 1h (max 2y), 1d (unlimited) (default: 5m)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  NSE BACKTEST")
    print(f"  Symbol  : {args.symbol}")
    print(f"  Strategy: {args.strategy.upper()}")
    print(f"  Period  : {args.period}  (interval: {args.interval})")
    print(f"  Capital : ₹{args.capital:,.0f}  (buying power: ₹{args.capital * args.leverage:,.0f} at {args.leverage}x leverage)")
    print(f"{'='*60}\n")

    # --- Data ---
    df = fetch_data(args.symbol, args.period, args.interval)

    # --- Strategy ---
    if args.strategy == "orb":
        strategy = ORBStrategy(or_minutes=15, rvol_threshold=1.5, target_r=2.0)
    else:
        strategy = VWAPReversionStrategy()

    # --- Backtest ---
    print(f"\nRunning backtest...")
    engine = BacktestEngine(
        transaction_cost_pct=0.0006,  # 0.06% round-trip (realistic NSE cost)
        slippage_pct=0.0002,          # 0.02% slippage
        quantity=args.quantity,
        leverage=args.leverage,
    )
    result = engine.run(df, strategy, initial_capital=args.capital, symbol=args.symbol)

    # --- Report ---
    reporter = BacktestReporter()
    wf_result = None

    if args.walk_forward:
        print(f"\nRunning walk-forward validation (6-month in-sample / 1-month out-of-sample)...")
        validator = WalkForwardValidator(
            in_sample_months=6,
            oos_months=1,
            initial_capital=args.capital,
        )
        wf_result = validator.validate(df, strategy, symbol=args.symbol)

    print()
    print(reporter.console_report(result, wf_result))

    # Quick equity curve summary
    metrics = MetricsEngine().compute(result)
    if result.trades:
        print(f"\n  Trade log ({len(result.trades)} trades):")
        for t in result.trades:
            sign = "+" if t.net_pnl >= 0 else ""
            print(
                f"    {t.entry_time.strftime('%m-%d %H:%M')} → {t.exit_time.strftime('%H:%M')}"
                f"  {t.action:<4}  {sign}₹{t.net_pnl:,.0f}  [{t.exit_reason}]"
            )


if __name__ == "__main__":
    main()
