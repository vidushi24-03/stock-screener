import pandas as pd

from stock_screener.screener import (
    MovingAverageFilter,
    RSIFilter,
    VolumeSpikeFilter,
    run_filters,
)


def _make_df(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    volumes = volumes or [1000] * n
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": volumes}
    )


def test_ma_filter_passes_when_bullish():
    closes = list(range(1, 61))
    df = _make_df(closes)
    f = MovingAverageFilter(short_window=5, long_window=20, condition="bullish")
    passed, reason = f.evaluate(df)
    assert passed is True
    assert "✅" in reason


def test_ma_filter_fails_when_condition_not_met():
    closes = list(range(1, 61))
    df = _make_df(closes)
    f = MovingAverageFilter(short_window=5, long_window=20, condition="bearish")
    passed, _ = f.evaluate(df)
    assert passed is False


def test_ma_filter_returns_none_with_insufficient_history():
    df = _make_df([100, 101, 102])
    f = MovingAverageFilter(short_window=5, long_window=20)
    passed, reason = f.evaluate(df)
    assert passed is None
    assert "not enough data" in reason


def test_rsi_filter_flags_oversold_stock():
    closes = list(range(60, 1, -1))
    df = _make_df(closes)
    f = RSIFilter(window=14, low=0, high=30)
    passed, _ = f.evaluate(df)
    assert passed is True


def test_volume_filter_flags_spike():
    closes = [100] * 25
    volumes = [1000] * 24 + [5000]
    df = _make_df(closes, volumes)
    f = VolumeSpikeFilter(window=20, multiplier=1.5)
    passed, reason = f.evaluate(df)
    assert passed is True


def test_run_filters_reports_no_data_for_short_history():
    df = _make_df([100, 101])
    result = run_filters("TEST", df, filters=[RSIFilter()])
    assert result.passed is False
    assert result.details == "no data"


def test_run_filters_only_fails_on_a_false_not_a_none():
    closes = list(range(1, 26))
    df = _make_df(closes)
    filters = [MovingAverageFilter(5, 20, "bullish"), RSIFilter(window=14, low=0, high=100)]
    result = run_filters("TEST", df, filters)
    assert result.passed is True


def test_run_filters_combines_multiple_filters():
    closes = list(range(1, 61))
    volumes = [1000] * 59 + [5000]
    df = _make_df(closes, volumes)
    filters = [
        MovingAverageFilter(short_window=5, long_window=20, condition="bullish"),
        VolumeSpikeFilter(window=20, multiplier=1.5),
    ]
    result = run_filters("TEST", df, filters)
    assert result.passed is True
    assert "MA" in result.details and "Vol" in result.details
