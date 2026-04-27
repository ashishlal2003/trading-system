from datetime import datetime
import pytz
from src.utils.logger import get_logger

logger = get_logger(__name__)
IST = pytz.timezone("Asia/Kolkata")

class ContextBuilder:
    """
    Builds a tight, token-efficient context snapshot for the chat LLM.
    Pulls from TradeStore only — no candle data, no full news articles.
    """
    def __init__(self, trade_store, news_cache: list = None):
        self.store = trade_store
        self.news_cache = news_cache or []

    async def build(self) -> str:
        now = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
        sections = [f"CURRENT TIME: {now}", "PAPER TRADE MODE: Yes (no real money at risk)"]

        # Open positions (max 5, 1 line each)
        try:
            positions = await self.store.get_open_positions()
            if positions:
                lines = ["OPEN POSITIONS:"]
                for p in positions[:5]:
                    pnl_str = ""
                    # rough unrealised pnl if we have current price somehow — skip if not
                    lines.append(
                        f"  {p['symbol']} {p['direction']} {p['quantity']}qty "
                        f"entry=₹{p['entry_price']} sl=₹{p['stop_loss']} "
                        f"target=₹{p['target_1']} type={p['trade_type']}"
                    )
                sections.append("\n".join(lines))
            else:
                sections.append("OPEN POSITIONS: None")
        except Exception as e:
            logger.warning("context_positions_failed", error=str(e))
            sections.append("OPEN POSITIONS: unavailable")

        # Today's P&L
        try:
            daily_pnl = await self.store.get_daily_pnl()
            closed = await self.store.get_today_closed_trades()
            wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
            sections.append(
                f"TODAY P&L: ₹{daily_pnl:+.0f} | "
                f"Trades closed: {len(closed)} | Wins: {wins}"
            )
        except Exception as e:
            logger.warning("context_pnl_failed", error=str(e))
            sections.append("TODAY P&L: unavailable")

        # Recent signals (last 5, compressed)
        try:
            # fetch last 5 signals from store
            signals = await self.store.get_recent_signals(limit=5)
            if signals:
                lines = ["RECENT SIGNALS (last 5):"]
                for s in signals:
                    lines.append(
                        f"  {s['symbol']} → {s['action']} "
                        f"conf={s.get('confidence', 0):.0%} "
                        f"decision={s.get('user_decision', 'pending')}"
                    )
                sections.append("\n".join(lines))
        except Exception as e:
            logger.warning("context_signals_failed", error=str(e))

        # Top news headlines (title only, max 5)
        if self.news_cache:
            headlines = [n.title for n in self.news_cache[:5]]
            sections.append("MARKET NEWS (this morning):\n" + "\n".join(f"  • {h}" for h in headlines))

        return "\n\n".join(sections)
