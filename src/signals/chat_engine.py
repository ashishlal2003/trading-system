from collections import deque
from openai import AsyncOpenAI
from src.utils.logger import get_logger

logger = get_logger(__name__)

CHAT_SYSTEM_PROMPT = """You are a friendly trading assistant for an Indian retail trader using NSE/BSE markets.
You help them understand their positions, P&L, signals, and market news in plain simple English.

RULES:
- Answer in 2-4 short sentences max. Be direct and conversational.
- Use ₹ for amounts. Use simple words — no jargon unless they ask.
- If you don't know something from the context provided, say so honestly.
- Never give financial advice or tell them to buy/sell. Just report facts from their data.
- If they ask about a specific stock not in their watchlist, say you don't have data on it.
- Keep it friendly. A little humour is fine.
- You have access to the last few messages of conversation — use that for follow-up questions.
"""

HISTORY_LIMIT = 10  # sliding window: 5 user + 5 assistant turns


class ChatEngine:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        # Sliding window: stores {"role": "user"/"assistant", "content": "..."}
        # Max HISTORY_LIMIT messages (each turn = 2 messages)
        self._history: deque = deque(maxlen=HISTORY_LIMIT)

    def clear_history(self):
        self._history.clear()

    async def reply(self, user_message: str, context: str) -> str:
        """
        Takes user's free-text message + live context snapshot.
        Maintains a 10-message sliding window of conversation history.
        Returns a short conversational reply (max 300 tokens).
        """
        # Build messages: system → history → fresh context + current question
        system_msg = {
            "role": "system",
            "content": CHAT_SYSTEM_PROMPT,
        }

        # Inject fresh context only in the latest user turn (not history)
        # so old turns don't re-inject stale context
        user_msg_with_context = {
            "role": "user",
            "content": f"TRADER CONTEXT (live):\n{context}\n\nQUESTION: {user_message}",
        }

        messages = [system_msg] + list(self._history) + [user_msg_with_context]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=300,
                temperature=0.7,
                messages=messages,
            )
            assistant_reply = response.choices[0].message.content.strip()

            # Slide the window: add user turn (without context blob to save tokens)
            self._history.append({"role": "user", "content": user_message})
            self._history.append({"role": "assistant", "content": assistant_reply})

            logger.info(
                "chat_reply",
                history_len=len(self._history),
                reply_len=len(assistant_reply),
            )
            return assistant_reply

        except Exception as e:
            logger.error("chat_engine_failed", error=str(e))
            return "Sorry, I couldn't process that right now. Try again in a moment."
