"""Data fetching utilities for the stock screener.

Wraps yfinance so the rest of the app doesn't need to know about Yahoo's
multi-index columns or handle missing-data edge cases itself. Deliberately
has no dependency on Streamlit, so it can be unit tested and reused (e.g.
in a CLI or notebook) without spinning up a UI.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Download OHLCV price history for a single ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        period: Lookback window accepted by yfinance, e.g. "3mo", "6mo", "1y".

    Returns:
        A DataFrame indexed by date with columns Open, High, Low, Close,
        Volume. Returns an empty DataFrame (not an exception) if no data
        was returned, e.g. for an invalid ticker.
    """
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df