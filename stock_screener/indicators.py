"""Technical indicator calculations.

Every function here is pure: it takes a pandas Series and returns a new
one, with no I/O and no framework dependency. That's what makes them easy
to unit test in isolation, rather than only being verifiable by eyeballing
a chart.
"""
# pure here means the same input will always give the same output


from __future__ import annotations

import pandas as pd


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Rolling simple moving average over `window` periods.

    The first `window - 1` entries are NaN, since there isn't enough
    history yet to compute a full window.
    """
    return series.rolling(window=window).mean()
# the average closing price over the last n days recalculated every day. 


def relative_strength_index(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI using a simple rolling average of gains and losses.

    This is the "cutler" variant of RSI (simple moving average of gains
    and losses) rather than Wilder's original smoothed version -- simpler
    to reason about and test, and close enough for screening purposes.

    Returns values in [0, 100]. NaN for the first `window` entries.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs)) # the standard RSI formula
# a score from 0 to 100 measuring how strongly a stock has been rising vs falling recently.
# below 30: oversold - fallen a lot, might bounce back
# abovw 70: overbought - risen a lot, might pull back


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Ratio of each day's volume to its trailing rolling average.

    A value of 2.0 means that day's volume was double the recent average
    -- useful as a simple "volume spike" signal.
    """
    avg = volume.rolling(window=window).mean()
    return volume / avg

# today's trading volume compared with the usual (20-day average) volume
# big volume often means something is happening, like news, earnings or big investors buying or selling. A price move on high volume is usually seen as more meaningful than one on low volume. The screener flags ratios of 1.5 or more by default.
