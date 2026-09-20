"""Rule-based parser for the "ask in plain English" box.

Each handler is a (predicate, answer) pair. `answer_query` finds the
first handler whose predicate matches the question, then applies its
answer function to every ticker's price data. There's no ML here -- it's
pattern matching over keywords -- but it's structured so a smarter
matcher (e.g. embedding similarity against example phrasings) could be
dropped in later without changing the public `answer_query` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from stock_screener.indicators import relative_strength_index, volume_ratio


def _pct_change_1day(df: pd.DataFrame) -> float:
    return (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100


def _matches_increase(q: str) -> bool:
    return any(w in q for w in ("increas", "up", "gain", "ris")) and any(
        w in q for w in ("yesterday", "today", "day")
    )


def _answer_increase(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 2:
        return None
    pct = _pct_change_1day(df)
    return f"+{pct:.2f}%" if pct > 0 else None


def _matches_decrease(q: str) -> bool:
    return any(w in q for w in ("decreas", "down", "fell", "drop")) and any(
        w in q for w in ("yesterday", "today", "day")
    )


def _answer_decrease(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 2:
        return None
    pct = _pct_change_1day(df)
    return f"{pct:.2f}%" if pct < 0 else None


def _matches_oversold(q: str) -> bool:
    return "oversold" in q or ("rsi" in q and ("below" in q or "low" in q))


def _answer_oversold(df: pd.DataFrame) -> Optional[str]:
    rsi = relative_strength_index(df["Close"]).iloc[-1]
    if pd.isna(rsi) or rsi >= 30:
        return None
    return f"RSI {rsi:.1f}"


def _matches_overbought(q: str) -> bool:
    return "overbought" in q or ("rsi" in q and ("above" in q or "high" in q))


def _answer_overbought(df: pd.DataFrame) -> Optional[str]:
    rsi = relative_strength_index(df["Close"]).iloc[-1]
    if pd.isna(rsi) or rsi <= 70:
        return None
    return f"RSI {rsi:.1f}"


def _matches_volume_spike(q: str) -> bool:
    return "volume" in q and any(w in q for w in ("spike", "high", "surge"))


def _answer_volume_spike(df: pd.DataFrame) -> Optional[str]:
    ratio = volume_ratio(df["Volume"]).iloc[-1]
    if pd.isna(ratio) or ratio <= 1.5:
        return None
    return f"{ratio:.2f}x avg volume"


@dataclass
class QueryHandler:
    name: str
    matches: Callable[[str], bool]
    answer: Callable[[pd.DataFrame], Optional[str]]


HANDLERS: list[QueryHandler] = [
    QueryHandler("price_increase", _matches_increase, _answer_increase),
    QueryHandler("price_decrease", _matches_decrease, _answer_decrease),
    QueryHandler("oversold", _matches_oversold, _answer_oversold),
    QueryHandler("overbought", _matches_overbought, _answer_overbought),
    QueryHandler("volume_spike", _matches_volume_spike, _answer_volume_spike),
]


def answer_query(query: str, price_data: dict[str, pd.DataFrame]) -> Optional[list[str]]:
    """Match `query` against known handlers and list the tickers that satisfy it.

    Returns:
        None if no handler recognised the phrasing at all.
        [] if a handler matched but no ticker satisfied its condition.
        Otherwise a list of "TICKER (detail)" strings.
    """
    q = query.lower()
    handler = next((h for h in HANDLERS if h.matches(q)), None)
    if handler is None:
        return None

    matches = []
    for ticker, df in price_data.items():
        if df.empty or len(df) < 2:
            continue
        detail = handler.answer(df)
        if detail is not None:
            matches.append(f"{ticker} ({detail})")
    return matches