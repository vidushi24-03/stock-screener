"""Filter logic: turn indicator values into pass/fail decisions.

Each filter is a small class with an `evaluate(df)` method, following the
Strategy pattern -- `run_filters` doesn't need to know anything about MA,
RSI, or volume specifically, only that each filter can evaluate itself
against a price DataFrame. Adding a new criterion (MACD, Bollinger Bands,
...) means adding one new class here, not touching the screening loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol

import pandas as pd

from stock_screener.indicators import (
    relative_strength_index,
    simple_moving_average,
    volume_ratio,
)


class Filter(Protocol):
    """Any screening criterion: given price history, pass/fail + a reason.

    `evaluate` returns (None, reason) when there isn't enough price
    history to compute the indicator yet, rather than treating that as
    an automatic failure.
    """

    def evaluate(self, df: pd.DataFrame) -> tuple[Optional[bool], str]: ...


@dataclass
class MovingAverageFilter:
    short_window: int
    long_window: int
    condition: Literal["bullish", "bearish"] = "bullish"

    def evaluate(self, df: pd.DataFrame) -> tuple[Optional[bool], str]:
        short = simple_moving_average(df["Close"], self.short_window).iloc[-1]
        long = simple_moving_average(df["Close"], self.long_window).iloc[-1]
        if pd.isna(short) or pd.isna(long):
            return None, "MA: not enough data"

        passed = short > long if self.condition == "bullish" else short < long
        return bool(passed), f"MA: {'✅' if passed else '❌'}"


@dataclass
class RSIFilter:
    window: int = 14
    low: float = 0
    high: float = 30

    def evaluate(self, df: pd.DataFrame) -> tuple[Optional[bool], str]:
        rsi = relative_strength_index(df["Close"], self.window).iloc[-1]
        if pd.isna(rsi):
            return None, "RSI: not enough data"

        passed = self.low <= rsi <= self.high
        return bool(passed), f"RSI: {'✅' if passed else '❌'} ({rsi:.1f})"


@dataclass
class VolumeSpikeFilter:
    window: int = 20
    multiplier: float = 1.5

    def evaluate(self, df: pd.DataFrame) -> tuple[Optional[bool], str]:
        ratio = volume_ratio(df["Volume"], self.window).iloc[-1]
        if pd.isna(ratio):
            return None, "Vol: not enough data"

        passed = ratio >= self.multiplier
        return bool(passed), f"Vol: {'✅' if passed else '❌'} ({ratio:.2f}x avg)"


@dataclass
class ScreenResult:
    ticker: str
    price: Optional[float]
    passed: bool
    details: str


def run_filters(ticker: str, df: pd.DataFrame, filters: list[Filter]) -> ScreenResult:
    """Apply every filter in `filters` to one stock's price history.

    A stock only fails if a filter that *could* be evaluated returned
    False; filters that returned None (not enough data) are recorded in
    the details but don't count against the stock.
    """
    if df.empty or len(df) < 5:
        return ScreenResult(ticker=ticker, price=None, passed=False, details="no data")

    passed = True
    reasons: list[str] = []
    for f in filters:
        result, reason = f.evaluate(df)
        reasons.append(reason)
        if result is False:
            passed = False

    price = round(float(df["Close"].iloc[-1]), 2)
    return ScreenResult(ticker=ticker, price=price, passed=passed, details=" | ".join(reasons))