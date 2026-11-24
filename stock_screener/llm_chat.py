"""LLM-powered "ask about your stocks" chatbot, using the Gemini API.

Design: the LLM never does the maths. We compute every number ourselves
with pandas (via the tested functions in `indicators.py`), hand Gemini a
compact table of those facts, and ask it only to *read* the table and
answer the user's question in plain English. This is "grounding": it
stops the model inventing prices or RSI values, and keeps the numeric
logic unit-testable without any API calls.

The Gemini client is passed in rather than created here (dependency
injection), so tests can substitute a fake client and never hit the
network. If no API key is configured, or the API call fails, the app
falls back to the rule-based parser in `query_parser.py`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import pandas as pd

from stock_screener.indicators import (
    relative_strength_index,
    simple_moving_average,
    volume_ratio,
)

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

SYSTEM_PROMPT = """You are a helpful assistant inside a technical stock screener app.
You will be given a JSON list of stocks with pre-computed indicators, then a user question.

Rules:
- Be friendly and conversational. Reply naturally to greetings and small talk (e.g. "hey"),
  then briefly mention what you can help with (e.g. which stocks rose, are oversold, had volume spikes).
- You can explain general concepts (what RSI, moving averages or volume spikes mean) from your own knowledge.
- For anything about these specific stocks, use ONLY the data provided. Never invent prices,
  percentages or indicator values.
- If the data can't answer a stock question (e.g. news, fundamentals, predictions), say so briefly.
- RSI below 30 is conventionally "oversold", above 70 "overbought".
- A volume_ratio of 1.5 or more means volume is a spike vs its 20-day average.
- sma_20 above sma_50 is a bullish trend signal; below is bearish.
- Keep answers short (a few sentences or a short list) and name the tickers.
- Do not give personalised financial advice or tell the user to buy or sell."""


def _round(value: Any, ndigits: int = 2) -> Optional[float]:
    """Round to a plain float, turning NaN into None so it serialises as JSON null."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), ndigits)


def build_snapshot(price_data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Summarise each ticker's latest indicator values as plain dicts.

    This is the only data the LLM sees. Tickers with fewer than 2 rows are
    skipped, since we can't compute even a 1-day change for them.
    """
    snapshot = []
    for ticker, df in price_data.items():
        if df is None or df.empty or len(df) < 2:
            continue
        close = df["Close"]
        snapshot.append({
            "ticker": ticker,
            "date": str(df.index[-1])[:10],
            "close": _round(close.iloc[-1]),
            "change_1d_pct": _round((close.iloc[-1] / close.iloc[-2] - 1) * 100),
            "change_5d_pct": _round((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(df) >= 6 else None,
            "rsi_14": _round(relative_strength_index(close).iloc[-1], 1),
            "sma_20": _round(simple_moving_average(close, 20).iloc[-1]),
            "sma_50": _round(simple_moving_average(close, 50).iloc[-1]),
            "volume_ratio": _round(volume_ratio(df["Volume"]).iloc[-1]),
        })
    return snapshot


def build_prompt(question: str, snapshot: list[dict[str, Any]]) -> str:
    """Combine the computed data and the user's question into one prompt."""
    return (
        "Stock data (computed by the app, null = not enough history):\n"
        f"{json.dumps(snapshot, indent=2)}\n\n"
        f"User question: {question}"
    )


def make_client(api_key: Optional[str] = None):
    """Create a Gemini client, or return None if no API key is available."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    from google import genai  # imported lazily so the app still runs without the package

    return genai.Client(api_key=api_key)


def ask_llm(
    question: str,
    price_data: dict[str, pd.DataFrame],
    client,
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask Gemini `question`, grounded in the computed snapshot of `price_data`.

    Raises whatever the Gemini SDK raises on failure (network, bad key,
    rate limit); the caller decides how to fall back.
    """
    from google.genai import types

    snapshot = build_snapshot(price_data)
    if not snapshot:
        return "I don't have any price data to answer from yet."

    response = client.models.generate_content(
        model=model,
        contents=build_prompt(question, snapshot),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,  # low = more factual, less creative
        ),
    )
    return (response.text or "").strip() or "Sorry, I couldn't generate an answer."
