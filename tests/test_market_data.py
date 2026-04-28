from __future__ import annotations

from portfolio_opt import market_data


def test_load_close_history_uses_yfinance_cache_loader(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_yfinance_fetch(symbols, **kwargs):
        captured["symbols"] = symbols
        captured["use_cache"] = kwargs["use_cache"]
        return {
            "SPY": [1.0, 2.0, 3.0, 4.0],
            "QQQ": [2.0, 3.0, 4.0, 5.0],
            "^HSI": [3.0, 4.0, 5.0, 6.0],
        }

    monkeypatch.setattr(market_data, "yf_fetch_closes", fake_yfinance_fetch)

    result = market_data.load_close_history(
        symbols=["SPY", "QQQ"],
        total_days=3,
        data_source="yfinance",
        benchmark_symbols=["^HSI"],
        use_cache=True,
    )

    assert captured == {"symbols": ["SPY", "QQQ", "^HSI"], "use_cache": True}
    assert result.closes_by_symbol == {
        "SPY": [2.0, 3.0, 4.0],
        "QQQ": [3.0, 4.0, 5.0],
    }
    assert result.benchmark_closes_by_symbol["^HSI"] == [4.0, 5.0, 6.0]
    assert result.benchmark_symbols_universe == ["SPY", "QQQ", "^HSI"]


def test_load_close_history_uses_alpaca_close_cache(monkeypatch) -> None:
    class FakeAlpaca:
        def get_daily_closes_for_period(self, symbols, total_days, **kwargs):
            assert symbols == ["SPY"]
            assert total_days == 3
            assert kwargs["offline"] is True
            return {"SPY": [1.0, 2.0, 3.0]}

    monkeypatch.setattr(
        market_data,
        "AlpacaClient",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("provided Alpaca client should be reused")
        ),
    )

    result = market_data.load_close_history(
        symbols=["SPY"],
        total_days=3,
        data_source="alpaca",
        alpaca=FakeAlpaca(),
        offline=True,
    )

    assert result.closes_by_symbol == {"SPY": [1.0, 2.0, 3.0]}
