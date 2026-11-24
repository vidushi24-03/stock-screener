import pandas as pd

from stock_screener.llm_chat import ask_llm, build_prompt, build_snapshot, make_client


def _make_df(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes,
         "Volume": volumes or [1000] * n},
        index=pd.date_range("2026-01-01", periods=n),
    )


class FakeModels:
    """Stands in for client.models; records the call instead of hitting the API."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return type("Response", (), {"text": self.reply})()


class FakeClient:
    def __init__(self, reply: str = "AAPL rose 10%."):
        self.models = FakeModels(reply)


def test_snapshot_computes_one_day_change():
    snap = build_snapshot({"UP": _make_df([100, 110])})
    assert snap[0]["ticker"] == "UP"
    assert snap[0]["change_1d_pct"] == 10.0


def test_snapshot_uses_none_not_nan_when_history_is_short():
    snap = build_snapshot({"NEW": _make_df([100, 101, 102])})
    assert snap[0]["rsi_14"] is None
    assert snap[0]["sma_50"] is None
    assert snap[0]["change_5d_pct"] is None


def test_snapshot_skips_tickers_without_enough_data():
    snap = build_snapshot({"EMPTY": pd.DataFrame(), "ONE": _make_df([100])})
    assert snap == []


def test_prompt_contains_question_and_data():
    prompt = build_prompt("which went up?", build_snapshot({"UP": _make_df([100, 110])}))
    assert "which went up?" in prompt
    assert '"ticker": "UP"' in prompt


def test_ask_llm_sends_grounded_prompt_and_returns_text():
    client = FakeClient(reply="  UP rose 10%.  ")
    answer = ask_llm("which went up?", {"UP": _make_df([100, 110])}, client, model="test-model")
    assert answer == "UP rose 10%."
    call = client.models.calls[0]
    assert call["model"] == "test-model"
    assert '"change_1d_pct": 10.0' in call["contents"]


def test_ask_llm_does_not_call_api_without_data():
    client = FakeClient()
    answer = ask_llm("anything?", {"EMPTY": pd.DataFrame()}, client)
    assert "don't have any price data" in answer
    assert client.models.calls == []


def test_make_client_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert make_client() is None
