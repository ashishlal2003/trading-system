-- OHLCV candle cache
CREATE TABLE IF NOT EXISTS candles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    interval    TEXT NOT NULL,
    timestamp   DATETIME NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      INTEGER,
    UNIQUE(symbol, interval, timestamp)
);

-- All generated signals (including NO_TRADE)
CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT NOT NULL,
    action              TEXT NOT NULL,
    trade_type          TEXT NOT NULL,
    entry_price         REAL,
    stop_loss           REAL,
    target_1            REAL,
    target_2            REAL,
    confidence          REAL,
    risk_reward_ratio   REAL,
    reasoning           TEXT,
    key_risks           TEXT,
    invalidation_condition TEXT,
    generated_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_decision       TEXT,
    decided_at          DATETIME
);

-- Open and closed trade positions
CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       INTEGER REFERENCES signals(id),
    symbol          TEXT NOT NULL,
    exchange        TEXT DEFAULT 'NSE',
    direction       TEXT NOT NULL,
    trade_type      TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    entry_price     REAL NOT NULL,
    stop_loss       REAL NOT NULL,
    target_1        REAL NOT NULL,
    target_2        REAL,
    entry_order_id  TEXT,
    gtt_id          TEXT,
    status          TEXT DEFAULT 'OPEN',
    exit_price      REAL,
    exit_reason     TEXT,
    pnl             REAL,
    entry_date      DATETIME DEFAULT CURRENT_TIMESTAMP,
    exit_date       DATETIME,
    sl_moved_to_breakeven INTEGER DEFAULT 0
);

-- Daily P&L summary
CREATE TABLE IF NOT EXISTS daily_summary (
    date            TEXT PRIMARY KEY,
    total_trades    INTEGER DEFAULT 0,
    winning_trades  INTEGER DEFAULT 0,
    total_pnl       REAL DEFAULT 0.0,
    max_drawdown    REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval ON candles(symbol, interval);
CREATE INDEX IF NOT EXISTS idx_signals_generated_at ON signals(generated_at);
