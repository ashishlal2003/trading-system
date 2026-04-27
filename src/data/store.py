"""
Async SQLite database layer for the algorithmic trading system.

Provides two stores:
  - CandleStore  : OHLCV candle cache (read/write via aiosqlite)
  - TradeStore   : Signals, positions, and daily P&L tracking
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import aiosqlite
import pandas as pd

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Project root is three levels above this file:
#   src/data/store.py  ->  src/data  ->  src  ->  project-root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SCHEMA_PATH = _PROJECT_ROOT / "db" / "schema.sql"


def _resolve_db_path(db_path: str) -> Path:
    """Return an absolute Path for a db_path string relative to project root."""
    return _PROJECT_ROOT / db_path


# ---------------------------------------------------------------------------
# CandleStore
# ---------------------------------------------------------------------------


class CandleStore:
    """Persistent cache for OHLCV candles backed by SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path: Path = _resolve_db_path(db_path)

    async def init_db(self) -> None:
        """Read schema.sql and execute it; create the db directory if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        async with aiosqlite.connect(self._db_path) as conn:
            # Enable WAL mode for better concurrent read performance
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.executescript(schema_sql)
            await conn.commit()

    async def upsert(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """Bulk-upsert OHLCV rows from a DataFrame.

        The DataFrame must contain columns: timestamp, open, high, low, close, volume.
        Existing rows with the same (symbol, interval, timestamp) are replaced.
        """
        if df.empty:
            return

        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        rows = [
            (
                symbol,
                interval,
                row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
                float(row["open"])   if row["open"]   is not None else None,
                float(row["high"])   if row["high"]   is not None else None,
                float(row["low"])    if row["low"]    is not None else None,
                float(row["close"])  if row["close"]  is not None else None,
                int(row["volume"])   if row["volume"] is not None else None,
            )
            for _, row in df.iterrows()
        ]

        sql = """
            INSERT OR REPLACE INTO candles
                (symbol, interval, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.executemany(sql, rows)
            await conn.commit()

    async def get_candles(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Return the latest *limit* candles for (symbol, interval) sorted ascending.

        Returns an empty DataFrame when no rows are found.
        """
        sql = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND interval = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(sql, (symbol, interval, limit)) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            [dict(r) for r in rows],
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Return in ascending chronological order (most-recent last)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# TradeStore
# ---------------------------------------------------------------------------


class TradeStore:
    """Persistence layer for signals, positions, and daily P&L."""

    def __init__(self, db_path: str) -> None:
        self._db_path: Path = _resolve_db_path(db_path)

    async def init_db(self) -> None:
        """Read schema.sql and execute it; create the db directory if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.executescript(schema_sql)
            await conn.commit()

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    async def save_signal(self, signal_data: dict) -> int:
        """Insert a signal row and return its new primary-key id.

        List/dict fields (key_risks) are JSON-serialised automatically.
        """
        key_risks = signal_data.get("key_risks")
        if isinstance(key_risks, (list, dict)):
            key_risks = json.dumps(key_risks)

        sql = """
            INSERT INTO signals
                (symbol, action, trade_type, entry_price, stop_loss,
                 target_1, target_2, confidence, risk_reward_ratio,
                 reasoning, key_risks, invalidation_condition, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            signal_data.get("symbol"),
            signal_data.get("action"),
            signal_data.get("trade_type"),
            signal_data.get("entry_price"),
            signal_data.get("stop_loss"),
            signal_data.get("target_1"),
            signal_data.get("target_2"),
            signal_data.get("confidence"),
            signal_data.get("risk_reward_ratio"),
            signal_data.get("reasoning"),
            key_risks,
            signal_data.get("invalidation_condition"),
            signal_data.get("generated_at", datetime.utcnow().isoformat()),
        )
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute("PRAGMA foreign_keys=ON")
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return cursor.lastrowid

    async def save_signal_decision(self, signal_id: int, decision: str) -> None:
        """Record the user's BUY / SKIP / etc. decision against a signal."""
        sql = """
            UPDATE signals
            SET user_decision = ?,
                decided_at    = ?
            WHERE id = ?
        """
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute(
                sql, (decision, datetime.utcnow().isoformat(), signal_id)
            )
            await conn.commit()

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def save_position(self, signal_id: int, position_data: dict) -> int:
        """Insert a new position linked to *signal_id* and return its id."""
        sql = """
            INSERT INTO positions
                (signal_id, symbol, exchange, direction, trade_type,
                 quantity, entry_price, stop_loss, target_1, target_2,
                 entry_order_id, gtt_id, status, entry_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            signal_id,
            position_data.get("symbol"),
            position_data.get("exchange", "NSE"),
            position_data.get("direction"),
            position_data.get("trade_type"),
            position_data.get("quantity"),
            position_data.get("entry_price"),
            position_data.get("stop_loss"),
            position_data.get("target_1"),
            position_data.get("target_2"),
            position_data.get("entry_order_id"),
            position_data.get("gtt_id"),
            position_data.get("status", "OPEN"),
            position_data.get("entry_date", datetime.utcnow().isoformat()),
        )
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute("PRAGMA foreign_keys=ON")
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return cursor.lastrowid

    async def get_open_positions(self) -> list[dict]:
        """Return all positions with status='OPEN'."""
        sql = "SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_date"
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(sql) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_open_swing_positions(self) -> list[dict]:
        """Return all OPEN positions where trade_type = 'SWING'."""
        sql = """
            SELECT * FROM positions
            WHERE status = 'OPEN' AND trade_type = 'SWING'
            ORDER BY entry_date
        """
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(sql) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def count_open_positions(self) -> int:
        """Return the count of OPEN positions."""
        sql = "SELECT COUNT(*) FROM positions WHERE status = 'OPEN'"
        async with aiosqlite.connect(self._db_path) as conn:
            async with conn.execute(sql) as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_open_position_symbols(self) -> list[str]:
        """Return a deduplicated list of symbols with at least one OPEN position."""
        sql = "SELECT DISTINCT symbol FROM positions WHERE status = 'OPEN'"
        async with aiosqlite.connect(self._db_path) as conn:
            async with conn.execute(sql) as cursor:
                rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def update_stop_loss(self, position_id: int, new_sl: float) -> None:
        """Overwrite the stop_loss for a position; also sets sl_moved_to_breakeven=1."""
        sql = """
            UPDATE positions
            SET stop_loss            = ?,
                sl_moved_to_breakeven = 1
            WHERE id = ?
        """
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute(sql, (new_sl, position_id))
            await conn.commit()

    async def close_position(
        self, position_id: int, exit_price: float, exit_reason: str
    ) -> None:
        """Mark a position CLOSED, recording exit details and computing P&L.

        P&L is calculated as:
          (exit_price - entry_price) * quantity  for LONG direction
          (entry_price - exit_price) * quantity  for SHORT direction
        """
        # Fetch the position first so we can compute P&L server-side in Python
        fetch_sql = """
            SELECT direction, entry_price, quantity
            FROM positions
            WHERE id = ?
        """
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute("PRAGMA foreign_keys=ON")
            async with conn.execute(fetch_sql, (position_id,)) as cursor:
                row = await cursor.fetchone()

            if row is None:
                raise ValueError(f"Position {position_id} not found.")

            direction, entry_price, quantity = row[0], row[1], row[2]

            if direction.upper() == "SHORT":
                pnl = (entry_price - exit_price) * quantity
            else:
                pnl = (exit_price - entry_price) * quantity

            update_sql = """
                UPDATE positions
                SET status      = 'CLOSED',
                    exit_price  = ?,
                    exit_reason = ?,
                    pnl         = ?,
                    exit_date   = ?
                WHERE id = ?
            """
            await conn.execute(
                update_sql,
                (exit_price, exit_reason, pnl, datetime.utcnow().isoformat(), position_id),
            )
            await conn.commit()

    # ------------------------------------------------------------------
    # Daily P&L helpers
    # ------------------------------------------------------------------

    async def get_daily_pnl(self) -> float:
        """Return the sum of pnl for all positions closed today (UTC date)."""
        today = date.today().isoformat()
        sql = """
            SELECT COALESCE(SUM(pnl), 0.0)
            FROM positions
            WHERE status = 'CLOSED'
              AND DATE(exit_date) = ?
        """
        async with aiosqlite.connect(self._db_path) as conn:
            async with conn.execute(sql, (today,)) as cursor:
                row = await cursor.fetchone()
        return float(row[0]) if row else 0.0

    async def get_today_closed_trades(self) -> list[dict]:
        """Return all positions closed today (UTC) ordered by exit_date.

        Intended for the end-of-day summary report.
        """
        today = date.today().isoformat()
        sql = """
            SELECT p.*, s.action, s.confidence, s.reasoning
            FROM positions p
            LEFT JOIN signals s ON s.id = p.signal_id
            WHERE p.status = 'CLOSED'
              AND DATE(p.exit_date) = ?
            ORDER BY p.exit_date
        """
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(sql, (today,)) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_recent_signals(self, limit: int = 5) -> list[dict]:
        """Return the most recent *limit* signals ordered newest-first."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT symbol, action, confidence, user_decision, generated_at
                   FROM signals ORDER BY generated_at DESC LIMIT ?""",
                (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
