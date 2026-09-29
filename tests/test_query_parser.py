import pandas as pd

from stock_screener.query_parser import answer_query


def _make_df(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": [1000] * 
len(closes)}
    )


def test_unrecognised_query_returns_none():
    data = {"AAPL": _make_df([100, 101])}
    assert answer_query("what's the weather like", data) is None


def test_increase_query_matches_only_rising_stock():
    data = {"UP": _make_df([100, 110]), "DOWN": _make_df([100, 90])}
    result = answer_query("which stocks increased yesterday", data)
    assert result == ["UP (+10.00%)"]


def test_decrease_query_matches_only_falling_stock():
    data = {"UP": _make_df([100, 110]), "DOWN": _make_df([100, 90])}
    result = answer_query("which stocks decreased today", data)
    assert result == ["DOWN (-10.00%)"]


def test_oversold_query_matches_falling_stock():
    falling = list(range(60, 1, -1))
    data = {"OVERSOLD": _make_df(falling)}
    result = answer_query("which stocks are oversold", data)
    assert len(result) == 1
    assert "OVERSOLD" in result[0]


def test_no_match_returns_empty_list_not_none():
    data = {"FLAT": _make_df([100, 100])}
    result = answer_query("which stocks increased today", data)
    assert result == []
