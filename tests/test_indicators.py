import pandas as pd
import pytest

from stock_screener.indicators import (
    relative_strength_index,
    simple_moving_average,
    volume_ratio,
)


def test_simple_moving_average_basic():
    series = pd.Series([1, 2, 3, 4, 5])
    sma = simple_moving_average(series, window=2)
    assert sma.iloc[-1] == pytest.approx(4.5)
    assert pd.isna(sma.iloc[0])


def test_rsi_all_gains_approaches_100():
    series = pd.Series(range(1, 20))
    rsi = relative_strength_index(series, window=14)
    assert rsi.iloc[-1] == pytest.approx(100)


def test_rsi_all_losses_approaches_0():
    series = pd.Series(list(range(20, 1, -1)))
    rsi = relative_strength_index(series, window=14)
    assert rsi.iloc[-1] == pytest.approx(0)


def test_rsi_has_nan_before_window_fills():
    series = pd.Series(range(1, 10))
    rsi = relative_strength_index(series, window=14)
    assert rsi.isna().all()


def test_volume_ratio_detects_spike():
    volume = pd.Series([100] * 20 + [500])
    ratio = volume_ratio(volume, window=20)
    # Window includes the current (spike) row: avg = (19*100 + 500)/20 = 120
    assert ratio.iloc[-1] == pytest.approx(500 / 120)


def test_volume_ratio_flat_volume_is_one():
    volume = pd.Series([100] * 25)
    ratio = volume_ratio(volume, window=20)
    assert ratio.iloc[-1] == pytest.approx(1.0)
