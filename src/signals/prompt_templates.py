"""
Prompt templates for the LLM signal engine.

SYSTEM_PROMPT defines the persona, constraints, and output rules for the
language model acting as a quantitative trading analyst.

USER_PROMPT_TEMPLATE is a format-string that is populated at runtime with
live indicator data, detected candlestick patterns, and a news summary before
being sent to the model.
"""

SYSTEM_PROMPT = """
You are a quantitative trading analyst specializing in Indian equity markets (NSE/BSE).
Your task is to analyze technical indicators, candlestick patterns, and recent news for a stock
and return a precise trading signal in JSON format.

RULES:
- Only recommend a trade if confidence >= 0.65
- For INTRADAY: entry must be within 0.3% of current price. Target/SL must be achievable same day.
- For SWING: hold period 2-10 days. Use daily candles perspective.
- If news contains material negative event (earnings miss, regulatory action, fraud), return NO_TRADE.
- Risk-reward ratio must be >= 1.5 before recommending a trade.
- Output ONLY valid JSON. No prose before or after.
- If uncertain, return NO_TRADE with low confidence.
"""

USER_PROMPT_TEMPLATE = """
SYMBOL: {symbol}
EXCHANGE: {exchange}
TRADE_TYPE: {trade_type}
CURRENT_TIME: {current_time}
CURRENT_PRICE: ₹{current_price}

--- TECHNICAL INDICATORS ---
RSI(14): {rsi_14}
MACD Line: {macd_line} | Signal: {macd_signal} | Histogram: {macd_hist}
EMA 9: {ema_9} | EMA 21: {ema_21} | EMA 50: {ema_50} | EMA 200: {ema_200}
Bollinger Bands: Upper={bb_upper} Mid={bb_mid} Lower={bb_lower} | %B={bb_pct_b}
ATR(14): {atr_14}
VWAP: {vwap}
Relative Volume: {relative_volume}x average
Trend: {trend}
Price vs VWAP: {price_vs_vwap}

--- CANDLESTICK PATTERNS (recent candles) ---
Detected: {patterns_detected}
Bias: {patterns_bias}

--- RECENT NEWS & ANNOUNCEMENTS ---
{news_summary}

--- REQUIRED JSON OUTPUT ---
Return exactly this structure:
{{
  "action": "BUY" | "SELL" | "NO_TRADE",
  "trade_type": "INTRADAY" | "SWING",
  "entry_price": <float>,
  "stop_loss": <float>,
  "target_1": <float>,
  "target_2": <float or null>,
  "confidence": <float 0.0-1.0>,
  "risk_reward_ratio": <float>,
  "reasoning": "<2-3 sentence explanation>",
  "key_risks": ["<risk1>", "<risk2>"],
  "invalidation_condition": "<what would invalidate this trade>"
}}
"""
