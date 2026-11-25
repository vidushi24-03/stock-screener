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
# turns the data into text for the prompt
import os
# reads environment variables 
from typing import Any, Optional
# type hints - labels in your code saying what typw of value a variable or return value should be: any means "any type at all"

import pandas as pd

from stock_screener.indicators import (
    relative_strength_index,
    simple_moving_average,
    volume_ratio,
)

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


# instructions for gemini to follow on every request:
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
    # NaN stands for "not a number", its a special value pandas uses for a missing or undefined number like an empty cell in Excel.
    # this matters because NaN isn't valid JSON, while None becomes null, which the prompt tells Gemini to read as "not enough history"
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
        if df is None or df.empty or len(df) < 2: # it skips any that are empty or have fewer than 2 days of data, because it can't work out a change without them.
            continue
        close = df["Close"]
        snapshot.append({
            "ticker": ticker,
            "date": str(df.index[-1])[:10], # date is the last trading day.
            "close": _round(close.iloc[-1]),# close is the latest price 
            "change_1d_pct": _round((close.iloc[-1] / close.iloc[-2] - 1) * 100),
            # works out how much a stocks price has gone up or down as a percentage over the last day
            "change_5d_pct": _round((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(df) >= 6 else None,
            # works out how much a stocks price has gone up or down as a percentage over the last 5 days
            "rsi_14": _round(relative_strength_index(close).iloc[-1], 1),
            "sma_20": _round(simple_moving_average(close, 20).iloc[-1]),
            "sma_50": _round(simple_moving_average(close, 50).iloc[-1]),
            "volume_ratio": _round(volume_ratio(df["Volume"]).iloc[-1]),
        })
    return snapshot # returns the list, this list is the only data that gemini sees.


def build_prompt(question: str, snapshot: list[dict[str, Any]]) -> str:
    """Combine the computed data and the user's question into one prompt."""
    return (
        "Stock data (computed by the app, null = not enough history):\n"
        f"{json.dumps(snapshot, indent=2)}\n\n"
        f"User question: {question}"
    )
# this basically sends the heading, the json data as text, and the users question to gemini

def make_client(api_key: Optional[str] = None):
    """Create a Gemini client, or return None if no API key is available."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None # with no key, it returns none, and the app uses the fallback 
    from google import genai  # imported lazily so the app still runs without the package

    return genai.Client(api_key=api_key) # creates the app's connection with gemini


def ask_llm(
    question: str,
    price_data: dict[str, pd.DataFrame],
    client, # client talks to google's gemini servers, it packages the prompt into a web request, sends it over to google with API key attached,
    # waits for geminis reply, and hands the reply back to your code as response.txt
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask Gemini `question`, grounded in the computed snapshot of `price_data`.

    Raises whatever the Gemini SDK raises on failure (network, bad key,
    rate limit); the caller decides how to fall back.
    """
    from google.genai import types

    snapshot = build_snapshot(price_data) # the price data
    if not snapshot: # if there is no data, it returns a message without calling the API which saves a wasted request
        return "I don't have any price data to answer from yet."

    response = client.models.generate_content( #calls gemini
        model=model,
        contents=build_prompt(question, snapshot),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,  # low = keeps answers more factual and consistent, less creative
            # temperature runs from 0(predictable) to 2(creative) 
        ),
    )
    return (response.text or "").strip() or "Sorry, I couldn't generate an answer."
# returns the text or a fallback messsage if Gemini returned nothing.
