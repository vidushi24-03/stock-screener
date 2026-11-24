"""Streamlit UI for the technical stock screener.

All the actual logic -- fetching, indicators, filtering, query parsing --
lives in the `stock_screener` package so it can be unit tested without
Streamlit. This file only wires that logic to widgets; it should stay
"dumb" on purpose.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_screener.data import fetch_price_history
from stock_screener.indicators import (
    relative_strength_index,
    simple_moving_average,
)
from stock_screener.llm_chat import ask_llm, make_client
from stock_screener.query_parser import answer_query
from stock_screener.screener import (
    MovingAverageFilter,
    RSIFilter,
    VolumeSpikeFilter,
    run_filters,
)

st.set_page_config(page_title="Technical Stock Screener", layout="wide")
st.title("📊 Technical Stock Screener")
st.caption("Filter a custom list of tickers on moving averages, RSI, and volume.")


@st.cache_data(ttl=3600)
def cached_fetch(ticker: str, period: str) -> pd.DataFrame:
    """Streamlit-cached wrapper around the framework-agnostic fetch function."""
    return fetch_price_history(ticker, period)


#sidebar
st.sidebar.header("Stock Universe")
tickers_input = st.sidebar.text_area(
    "Enter tickers (comma-separated)", value="AAPL, MSFT, GOOGL, AMZN, TSLA", height=80
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
period = st.sidebar.selectbox("Data period", ["3mo", "6mo", "1y", "2y"], index=1)

st.sidebar.header("Filters")
filters = []

use_ma = st.sidebar.checkbox("Moving Average filter", value=True)
if use_ma:
    ma_short = st.sidebar.number_input("Short MA period", 5, 100, 20)
    ma_long = st.sidebar.number_input("Long MA period", 10, 300, 50)
    ma_condition = st.sidebar.radio("Condition", ["bullish", "bearish"])
    filters.append(MovingAverageFilter(ma_short, ma_long, ma_condition))

use_rsi = st.sidebar.checkbox("RSI filter", value=True)
if use_rsi:
    rsi_period = st.sidebar.number_input("RSI period", 5, 50, 14)
    rsi_min, rsi_max = st.sidebar.slider("RSI range", 0, 100, (0, 30))
    filters.append(RSIFilter(rsi_period, rsi_min, rsi_max))

use_vol = st.sidebar.checkbox("Volume filter", value=True)
if use_vol:
    vol_multiplier = st.sidebar.number_input(
        "Volume spike multiplier (x average)", 0.5, 10.0, 1.5, step=0.1
    )
    vol_avg_period = st.sidebar.number_input("Volume average lookback (days)", 5, 100, 20)
    filters.append(VolumeSpikeFilter(vol_avg_period, vol_multiplier))

run_button = st.sidebar.button("Run Screener", type="primary")

#main panel
data_store: dict[str, pd.DataFrame] = {}

if run_button:
    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        rows = []
        progress = st.progress(0)
        for i, ticker in enumerate(tickers):
            try:
                df = cached_fetch(ticker, period)
                result = run_filters(ticker, df, filters)
                rows.append(vars(result))
                if not df.empty:
                    data_store[ticker] = df
            except Exception as e:  # network errors, bad tickers, Yahoo hiccups
                rows.append({"ticker": ticker, "price": None, "passed": False, "details": str(e)})
            progress.progress((i + 1) / len(tickers))

        results_df = pd.DataFrame(rows).rename(
            columns={"ticker": "Ticker", "price": "Price", "passed": "Pass", "details": "Details"}
        )
        results_df["Pass"] = results_df["Pass"].map({True: "✅", False: "❌"})

        st.subheader("Screener Results")
        st.dataframe(results_df, use_container_width=True)

        passed = results_df.loc[results_df["Pass"] == "✅", "Ticker"].tolist()
        if passed:
            st.success(f"{len(passed)} stock(s) passed all filters: {', '.join(passed)}")
        else:
            st.info("No stocks passed all active filters.")

        st.subheader("Chart Inspector")
        chartable = [t for t in tickers if t in data_store]
        if chartable:
            chosen = st.selectbox("Select a ticker to view", chartable)
            df = data_store[chosen]

            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name="Price",
                )
            )
            if use_ma:
                fig.add_trace(go.Scatter(
                    x=df.index, y=simple_moving_average(df["Close"], ma_short), name=f"SMA {ma_short}"
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=simple_moving_average(df["Close"], ma_long), name=f"SMA {ma_long}"
                ))
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

            if use_rsi:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=df.index, y=relative_strength_index(df["Close"], rsi_period), name="RSI"
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(height=250, yaxis_range=[0, 100], margin=dict(t=10))
                st.plotly_chart(fig_rsi, use_container_width=True)

            if use_vol:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
                fig_vol.update_layout(height=250, margin=dict(t=10))
                st.plotly_chart(fig_vol, use_container_width=True)
else:
    st.info("Enter your tickers on the left, set your filters, and click **Run Screener**.")

#chatbot
def get_gemini_key() -> str | None:
    """Read the key from Streamlit secrets (deployment) or an env var (local)."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


@st.cache_resource
def get_llm_client(api_key: str | None):
    return make_client(api_key)


def rule_based_answer(query: str, price_data: dict[str, pd.DataFrame]) -> None:
    """Fallback: the original keyword parser, used when Gemini isn't available."""
    result = answer_query(query, price_data)
    if result is None:
        st.warning(
            "I didn't recognise that phrasing. Try: 'which stocks increased "
            "yesterday', 'which are oversold', 'which had a volume spike', "
            "'which are overbought'."
        )
    elif not result:
        st.info("No stocks matched that question.")
    else:
        st.success(", ".join(result))


st.divider()
st.subheader("💬 Ask about your stocks")
llm_client = get_llm_client(get_gemini_key())
st.caption(
    "Powered by Gemini — answers are grounded in the indicators computed above."
    if llm_client
    else "Gemini API key not set — using the basic keyword matcher."
)
query = st.text_input("Ask a question", placeholder="e.g. Which stocks look oversold and had a volume spike?")

if query:
    if not tickers:
        st.warning("Enter some tickers in the sidebar first.")
    else:
        price_data = {t: cached_fetch(t, period) for t in tickers}
        if llm_client is None:
            rule_based_answer(query, price_data)
        else:
            try:
                with st.spinner("Thinking..."):
                    st.markdown(ask_llm(query, price_data, llm_client))
            except Exception as e:  # bad key, rate limit, network error
                st.warning(f"Gemini request failed ({type(e).__name__}); falling back to keyword matcher.")
                rule_based_answer(query, price_data)
