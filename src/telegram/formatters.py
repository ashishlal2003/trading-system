"""
Telegram message formatters for the algorithmic trading system.

All public functions return plain Markdown (v1) strings safe for use with
``ParseMode.MARKDOWN`` in python-telegram-bot.  MarkdownV2 is deliberately
avoided because it requires escaping almost every punctuation character,
which becomes fragile when values come from live market data.

Usage
-----
    from src.telegram.formatters import format_signal_card, format_eod_summary

    text = format_signal_card(signal, capital=100_000, quantity=25)
    await bot.send_message(text)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Indian Standard Time offset (UTC+5:30)
_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    """Return the current time in IST."""
    return datetime.now(_IST)


def _fmt_ist(dt: datetime) -> str:
    """Format a datetime as a human-readable IST string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ist_dt = dt.astimezone(_IST)
    return ist_dt.strftime("%d %b %Y  %I:%M:%S %p IST")


def _confidence_bar(confidence: float, width: int = 10) -> str:
    """
    Build a visual confidence bar using block characters.

    Parameters
    ----------
    confidence:
        Float in [0.0, 1.0].
    width:
        Total number of blocks in the bar.

    Returns
    -------
    str
        E.g. ``"████████░░  80%"``
    """
    confidence = max(0.0, min(1.0, confidence))
    filled = round(confidence * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    pct = int(confidence * 100)
    return f"{bar}  {pct}%"


def _action_emoji(action: str) -> str:
    mapping = {"BUY": "📈", "SELL": "📉", "NO_TRADE": "⏸"}
    return mapping.get(action.upper(), "❓")


def _pnl_emoji(pnl: float) -> str:
    return "✅" if pnl >= 0 else "❌"


def _sign(value: float) -> str:
    """Prefix positive numbers with '+', negatives already carry '-'."""
    return f"+{value:.2f}" if value >= 0 else f"{value:.2f}"


# ---------------------------------------------------------------------------
# Public formatters
# ---------------------------------------------------------------------------


def format_signal_card(signal, capital: float, quantity: int) -> str:
    """
    Format a TradeSignal into a Telegram plain-Markdown signal card.

    Parameters
    ----------
    signal:
        A ``TradeSignal`` (or dict-like) produced by the LLM engine.
        Expected attributes / keys:
            symbol, action, trade_type, entry_price, stop_loss,
            target_1, target_2, confidence, risk_reward_ratio,
            reasoning, key_risks, invalidation_condition,
            generated_at (optional datetime).
    capital:
        Total available capital (used for context in the card header).
    quantity:
        Number of shares / lots for this trade.

    Returns
    -------
    str
        Telegram-ready Markdown v1 string.
    """
    # Support both attribute access (dataclass / namedtuple) and dict access.
    def _get(attr: str, default=None):
        if isinstance(signal, dict):
            return signal.get(attr, default)
        return getattr(signal, attr, default)

    symbol: str = str(_get("symbol", "UNKNOWN")).upper()
    action: str = str(_get("action", "NO_TRADE")).upper()
    trade_type: str = str(_get("trade_type", "INTRADAY")).upper()
    entry: float = float(_get("entry_price") or 0.0)
    stop_loss: float = float(_get("stop_loss") or 0.0)
    target_1: Optional[float] = _get("target_1")
    target_2: Optional[float] = _get("target_2")
    confidence: float = float(_get("confidence") or 0.0)
    rr_ratio: float = float(_get("risk_reward_ratio") or 0.0)
    reasoning: str = str(_get("reasoning") or "No reasoning provided.")
    key_risks: list = _get("key_risks") or []
    invalidation: str = str(_get("invalidation_condition") or "Not specified.")
    generated_at: Optional[datetime] = _get("generated_at")

    emoji = _action_emoji(action)
    capital_at_risk = quantity * abs(entry - stop_loss)

    # Timestamp
    if generated_at is not None:
        ts_str = _fmt_ist(generated_at) if isinstance(generated_at, datetime) else str(generated_at)
    else:
        ts_str = _fmt_ist(_now_ist())

    # Build key risks bullet list
    if key_risks:
        risks_block = "\n".join(f"  • {r}" for r in key_risks)
    else:
        risks_block = "  • None identified"

    # Target lines (only show target_2 if present)
    target_1_line = f"🎯 *Target 1:*    ₹{target_1:.2f}" if target_1 is not None else ""
    target_2_line = f"🎯 *Target 2:*    ₹{target_2:.2f}" if target_2 is not None else ""
    targets_block = "\n".join(filter(None, [target_1_line, target_2_line]))

    lines = [
        f"━━━━━━━━━━━━━━━━━━━━━━",
        f"📊 *SIGNAL: {symbol}* — _{trade_type}_",
        f"━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"{emoji} *Action:*       *{action}*",
        f"",
        f"💰 *Entry:*        ₹{entry:.2f}",
        f"🛑 *Stop Loss:*    ₹{stop_loss:.2f}",
    ]

    if targets_block:
        lines.append(targets_block)

    lines += [
        f"",
        f"⚖️ *Risk-Reward:*  {rr_ratio:.2f}x",
        f"📦 *Quantity:*     {quantity} shares",
        f"💸 *Capital Risk:* ₹{capital_at_risk:,.2f}",
        f"🏦 *Capital:*      ₹{capital:,.2f}",
        f"",
        f"🔢 *Confidence:*",
        f"  `{_confidence_bar(confidence)}`",
        f"",
        f"📝 *Reasoning:*",
        f"_{reasoning}_",
        f"",
        f"⚠️ *Key Risks:*",
        risks_block,
        f"",
        f"🚫 *Invalidation:*",
        f"_{invalidation}_",
        f"",
        f"🕐 _{ts_str}_",
        f"━━━━━━━━━━━━━━━━━━━━━━",
    ]

    card = "\n".join(lines)
    logger.debug("signal_card_formatted", symbol=symbol, action=action, confidence=confidence)
    return card


def format_eod_summary(trades: list[dict], daily_pnl: float) -> str:
    """
    Format an end-of-day trade summary.

    Parameters
    ----------
    trades:
        List of dicts, each with keys:
            ``symbol`` (str), ``action`` (str), ``pnl`` (float).
    daily_pnl:
        Aggregate P&L for the day.

    Returns
    -------
    str
        Telegram-ready Markdown v1 string.
    """
    now_str = _fmt_ist(_now_ist())
    total_trades = len(trades)
    wins = [t for t in trades if float(t.get("pnl", 0)) >= 0]
    win_count = len(wins)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "📋 *END-OF-DAY SUMMARY*",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
    ]

    if not trades:
        lines.append("_No trades executed today._")
    else:
        lines.append("*Trades:*")
        for t in trades:
            symbol = str(t.get("symbol", "?")).upper()
            action = str(t.get("action", "?")).upper()
            pnl = float(t.get("pnl", 0))
            label = "WIN" if pnl >= 0 else "LOSS"
            emoji = _pnl_emoji(pnl)
            pnl_str = _sign(pnl)
            lines.append(f"  {emoji} *{symbol}* [{action}]  ₹{pnl_str}  _{label}_")

    lines += [
        "",
        "─────────────────────",
        f"📊 *Total Trades:*  {total_trades}",
        f"✅ *Wins:*          {win_count}",
        f"📈 *Win Rate:*      {win_rate:.1f}%",
        f"",
        f"💰 *Day P&L:*       ₹{_sign(daily_pnl)}",
        "",
        f"🕐 _{now_str}_",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    summary = "\n".join(lines)
    logger.debug("eod_summary_formatted", total_trades=total_trades, daily_pnl=daily_pnl)
    return summary


def format_position_update(pos: dict, current_price: float) -> str:
    """
    Format a morning update for a swing position.

    Parameters
    ----------
    pos:
        Dict with keys:
            ``symbol`` (str), ``direction`` (str, "LONG"/"SHORT"),
            ``entry_price`` (float), ``stop_loss`` (float),
            ``target`` (float), ``hold_days`` (int),
            ``quantity`` (int, optional).
    current_price:
        Latest market price for the symbol.

    Returns
    -------
    str
        Telegram-ready Markdown v1 string.
    """
    symbol = str(pos.get("symbol", "UNKNOWN")).upper()
    direction = str(pos.get("direction", "LONG")).upper()
    entry = float(pos.get("entry_price", 0))
    stop_loss = float(pos.get("stop_loss", 0))
    target = float(pos.get("target", 0))
    hold_days = int(pos.get("hold_days", 0))
    quantity = int(pos.get("quantity", 0))

    # P&L calculation — sign depends on direction
    if direction == "SHORT":
        raw_pnl_per_share = entry - current_price
    else:
        raw_pnl_per_share = current_price - entry

    pnl_pct = (raw_pnl_per_share / entry * 100) if entry else 0.0
    pnl_amount = raw_pnl_per_share * quantity if quantity else raw_pnl_per_share

    dir_emoji = "📈" if direction == "LONG" else "📉"
    pnl_emoji = _pnl_emoji(raw_pnl_per_share)
    now_str = _fmt_ist(_now_ist())

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"🌅 *SWING UPDATE: {symbol}*",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"{dir_emoji} *Direction:*   {direction}",
        f"📅 *Hold Days:*   {hold_days}",
        "",
        f"🏷️ *Entry:*       ₹{entry:.2f}",
        f"📍 *Current:*     ₹{current_price:.2f}",
        "",
        f"{pnl_emoji} *P&L:*         ₹{_sign(pnl_amount)}  ({_sign(pnl_pct)}%)",
        "",
        f"🛑 *Stop Loss:*   ₹{stop_loss:.2f}",
        f"🎯 *Target:*      ₹{target:.2f}",
        "",
        f"🕐 _{now_str}_",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    text = "\n".join(lines)
    logger.debug("position_update_formatted", symbol=symbol, current_price=current_price)
    return text


def format_system_message(msg: str, level: str = "INFO") -> str:
    """
    Format a system status / alert message.

    Parameters
    ----------
    msg:
        The message body.
    level:
        One of ``"INFO"``, ``"WARNING"``, ``"ERROR"``.  Defaults to ``"INFO"``.

    Returns
    -------
    str
        Telegram-ready Markdown v1 string.
    """
    level = level.upper()
    level_map = {
        "INFO":    ("ℹ️", "INFO"),
        "WARNING": ("⚠️", "WARNING"),
        "ERROR":   ("🚨", "ERROR"),
    }
    emoji, label = level_map.get(level, ("ℹ️", level))
    now_str = _fmt_ist(_now_ist())

    lines = [
        f"{emoji} *[{label}]*  {msg}",
        f"🕐 _{now_str}_",
    ]

    text = "\n".join(lines)
    logger.debug("system_message_formatted", level=level, msg=msg)
    return text
