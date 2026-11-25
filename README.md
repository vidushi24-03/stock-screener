# Technical Stock Screener

A Streamlit web app that screens a custom list of stocks on technical indicators (**moving-average crossovers, RSI and volume spikes**), with interactive charts and a **Gemini-powered chatbot** for asking questions about your stocks in plain English.

## Tech stack

- **Python**
- **Streamlit** for the web app UI
- **pandas** for data handling and indicator calculations
- **Plotly** for interactive charts
- **yfinance** for Yahoo Finance market data
- **Google Gemini API** (`google-genai`) for the chatbot
- **pytest** for unit testing

---

## Features

- **Custom stock universe**: enter any comma-separated list of tickers (e.g. `AAPL, MSFT, TSLA`) and pick a lookback period (3 months to 2 years).
- **Configurable filters**
  - **Moving Average**: bullish (short MA above long MA) or bearish crossover, with adjustable periods.
  - **RSI**: keep stocks whose Relative Strength Index falls within a chosen range (default 0–30, i.e. oversold).
  - **Volume spike**: flag stocks trading at a multiple of their average volume (default 1.5×).
- **Results table**: pass/fail for each stock with the reason for every filter.
- **Chart inspector**: candlestick chart with moving-average overlays, plus RSI and volume charts.
- **AI chatbot**: ask things like *"Which stocks look oversold?"*, *"What went up the most today?"* or *"What does RSI mean?"*. Answers are grounded in indicators the app computes itself.
- **Graceful fallback**: with no API key (or if the API call fails), the chatbot falls back to a rule-based keyword matcher, so the app always works.

---

## How the chatbot works

The key design choice: **the LLM never does the maths.**

```
User question
      │
      ▼
Fetch price data (Yahoo Finance, cached for 1 hour)
      │
      ▼
pandas computes indicators per ticker          ← exact, unit-tested
(1-day & 5-day % change, RSI, SMA 20/50, volume ratio)
      │
      ▼
Snapshot sent to Gemini as JSON + the question
      │
      ▼
Gemini explains the data in plain English      ← told never to invent numbers
      │
      ▼
Answer shown in the app
```

LLMs are unreliable at arithmetic and can produce plausible-looking but wrong numbers. By computing every value with pandas and instructing Gemini to answer stock-specific questions **only** from that data (*grounding*), the chatbot stays accurate while still understanding free-form questions. It's also told not to give buy/sell advice.

---

## Project structure

```
stock-screener/
├── app.py                      # Streamlit UI: wires widgets to the logic below
├── stock_screener/
│   ├── data.py                 # Fetches price history from Yahoo Finance
│   ├── indicators.py           # Pure functions: SMA, RSI, volume ratio
│   ├── screener.py             # Filter classes + run_filters (Strategy pattern)
│   ├── llm_chat.py             # Gemini chatbot: snapshot, prompt, API call
│   └── query_parser.py         # Rule-based keyword chatbot (fallback)
├── tests/                      # pytest unit tests (no network calls)
├── requirements.txt
└── requirements-dev.txt
```

All logic lives in the `stock_screener` package with **no dependency on Streamlit**, so it can be unit tested and reused elsewhere. `app.py` only handles the UI.

---

## Getting started

### 1. Clone and set up a virtual environment

```bash
git clone https://github.com/vidushi24-03/stock-screener.git
cd stock-screener
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your Gemini API key (optional but recommended)

Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey), then create `.streamlit/secrets.toml` in the project root:

```toml
GEMINI_API_KEY = "your-key-here"
```

This file is in `.gitignore`, so your key is never committed. Alternatively, set it as an environment variable:

```bash
export GEMINI_API_KEY="your-key-here"
```

Without a key, the app still runs and the chatbot uses the keyword matcher.

To use a different Gemini model, set `GEMINI_MODEL` (default: `gemini-3.5-flash`).

### 3. Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

---

## Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests use small synthetic datasets and a **fake Gemini client**, so they run in about a second with no internet connection, API key or cost.

---

## Design decisions

| Decision | Why |
|---|---|
| **Pure indicator functions** | No I/O or framework code, so they can be tested in isolation with known inputs. |
| **Strategy pattern for filters** | Each filter is a class with an `evaluate()` method. Adding a new one (e.g. MACD) means one new class; the screening loop doesn't change. |
| **"Not enough data" ≠ failure** | Filters return `None` when there isn't enough history (e.g. a 50-day MA on 1 month of data); only a real `False` fails a stock. |
| **Grounded LLM answers** | pandas does the maths; Gemini only explains it, which prevents hallucinated numbers. |
| **Dependency injection for the Gemini client** | `ask_llm` takes the client as a parameter, so tests can pass a fake one. |
| **Caching** | `st.cache_data` caches price data for an hour; `st.cache_resource` reuses one Gemini client across reruns. |
| **Fallback chatbot** | If the API key is missing or the call fails, the rule-based parser answers instead. |

---

## Limitations & future work

- The chatbot answers each question independently, with no conversation memory yet (could use `st.chat_message` and `st.session_state`).
- Selecting a different ticker in the chart inspector reruns the app and clears results; storing results in `st.session_state` would fix this.
- Large ticker lists would make the prompt big, so it would need filtering or retrieval before sending to the LLM.
- Possible additions: Gemini function calling so the chatbot can run screener filters itself, adding more indicators

---

## Disclaimer

This project is for educational purposes only and is **not financial advice**. Market data comes from Yahoo Finance via `yfinance` and may be delayed or inaccurate.
