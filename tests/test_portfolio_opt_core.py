from __future__ import annotations

from argparse import Namespace
from datetime import UTC, datetime
import json

import numpy as np

from portfolio_opt.alpaca_interface import AlpacaClient
from portfolio_opt import cli
from portfolio_opt.backtest import BacktestResult, compute_dual_momentum_weights, run_backtest
from portfolio_opt.config import AlpacaConfig, OptimizationConfig
from portfolio_opt.model import ModelInputs
from portfolio_opt.optimizer import optimize_weights
from portfolio_opt.rebalance import build_order_plan, build_trailing_stop_plan
from portfolio_opt.types import AccountSnapshot, Position, TrailingStopPlan


def test_optimize_weights_aligns_asset_class_max_constraints() -> None:
    config = OptimizationConfig(
        risk_aversion=0.01,
        min_weight=0.0,
        max_weight=1.0,
        force_full_investment=True,
        class_min_weights={"equity": 0.2},
        class_max_weights={"bond": 0.6},
    )
    asset_class_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    weights = optimize_weights(
        expected_returns=np.array([0.10, 0.30], dtype=float),
        covariance=np.eye(2, dtype=float) * 1e-4,
        config=config,
        asset_class_matrix=asset_class_matrix,
    )

    assert round(float(weights[0]), 6) == 0.4
    assert round(float(weights[1]), 6) == 0.6


def test_run_backtest_dispatches_black_litterman(monkeypatch) -> None:
    called = {"black_litterman": False, "optimize": False}

    def fake_bl(*, symbols, closes_by_symbol, momentum_window, mean_shrinkage):
        called["black_litterman"] = True
        assert momentum_window == 2
        return Namespace(
            expected_returns=np.array([0.1, 0.2], dtype=float),
            covariance=np.eye(2, dtype=float),
            observations=2,
        )

    def fake_optimize(*, expected_returns, covariance, config, current_weights, asset_class_matrix):
        called["optimize"] = True
        return np.array([0.25, 0.75], dtype=float)

    monkeypatch.setattr("portfolio_opt.backtest.estimate_inputs_from_black_litterman", fake_bl)
    monkeypatch.setattr("portfolio_opt.backtest.optimize_weights", fake_optimize)

    result = run_backtest(
        symbols=["SPY", "IEF"],
        closes_by_symbol={
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
            "IEF": [100.0, 100.5, 101.0, 101.5, 102.0],
        },
        lookback_days=2,
        rebalance_every=1,
        return_model="black-litterman",
        mean_shrinkage=0.5,
        momentum_window=2,
        opt_config=OptimizationConfig(),
        asset_class_matrix=None,
    )

    assert called == {"black_litterman": True, "optimize": True}
    assert result.latest_weights.tolist() == [0.25, 0.75]


def test_run_backtest_dispatches_risk_parity_projection(monkeypatch) -> None:
    called = {"risk_parity": False, "project": False}

    def fake_risk_parity(*, symbols, closes_by_symbol, lookback_days):
        called["risk_parity"] = True
        assert lookback_days == 2
        return Namespace(
            weights=np.array([0.6, 0.4], dtype=float),
            covariance=np.eye(2, dtype=float),
            observations=2,
        )

    def fake_project(*, target_weights, config, current_weights, asset_class_matrix):
        called["project"] = True
        assert np.allclose(target_weights, np.array([0.6, 0.4], dtype=float))
        return target_weights

    def fail_optimize(**_kwargs):
        raise AssertionError("risk-parity backtest should not call optimize_weights")

    monkeypatch.setattr("portfolio_opt.backtest.estimate_inputs_risk_parity", fake_risk_parity)
    monkeypatch.setattr("portfolio_opt.backtest.project_weights", fake_project)
    monkeypatch.setattr("portfolio_opt.backtest.optimize_weights", fail_optimize)

    result = run_backtest(
        symbols=["SPY", "IEF"],
        closes_by_symbol={
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
            "IEF": [100.0, 100.5, 101.0, 101.5, 102.0],
        },
        lookback_days=2,
        rebalance_every=1,
        return_model="risk-parity",
        mean_shrinkage=0.5,
        momentum_window=2,
        opt_config=OptimizationConfig(),
        asset_class_matrix=None,
    )

    assert called == {"risk_parity": True, "project": True}
    assert result.latest_weights.tolist() == [0.6, 0.4]


def test_compute_dual_momentum_weights_uses_full_lookback_window() -> None:
    weights = compute_dual_momentum_weights(
        symbols=["SPY", "QQQ", "SGOV"],
        closes_by_symbol={
            "SPY": [100.0, 130.0, 120.0],
            "QQQ": [100.0, 101.0, 102.0],
            "SGOV": [100.0, 100.0, 100.0],
        },
        asset_classes={
            "SPY": "equity_us_large",
            "QQQ": "equity_us_growth",
            "SGOV": "cash_like",
        },
        lookback_days=2,
        top_k=1,
        absolute_threshold=0.0,
    )

    assert weights == {"SPY": 1.0, "QQQ": 0.0, "SGOV": 0.0}


def test_order_plan_ignores_protective_trailing_stop_orders() -> None:
    plan = build_order_plan(
        symbols=["SPY"],
        target_weights=[1.0],
        account=AccountSnapshot(equity=1000.0),
        positions=[Position(symbol="SPY", qty=10.0, market_value=1000.0)],
        latest_prices={"SPY": 100.0},
        config=OptimizationConfig(rebalance_threshold=0.02),
        open_orders=[
            {
                "symbol": "SPY",
                "qty": 10.0,
                "side": "sell",
                "type": "trailing_stop",
            }
        ],
    )

    assert plan == []


def test_build_trailing_stop_plan_skips_existing_protection() -> None:
    plan = build_trailing_stop_plan(
        symbols=["SPY", "QQQ"],
        target_weights=[0.5, 0.5],
        positions=[
            Position(symbol="SPY", qty=10.0, market_value=1000.0),
            Position(symbol="QQQ", qty=2.5, market_value=500.0),
        ],
        open_orders=[
            {
                "symbol": "SPY",
                "qty": 10.0,
                "side": "sell",
                "type": "trailing_stop",
            }
        ],
        trailing_stop=0.15,
        rebalance_threshold=0.02,
    )

    assert plan == [
        TrailingStopPlan(
            symbol="QQQ",
            qty=2.5,
            side="sell",
            trail_percent=15.0,
            time_in_force="gtc",
        )
    ]


def test_cli_yfinance_backtest_does_not_require_alpaca(monkeypatch, capsys) -> None:
    args = Namespace(
        model="dummy.json",
        dynamic_universe=False,
        filter_before=None,
        ticker_basket=[],
        risk_aversion=4.0,
        min_weight=0.0,
        max_weight=0.35,
        rebalance_threshold=0.02,
        turnover_penalty=0.02,
        allow_cash=False,
        min_cash_weight=0.0,
        max_turnover=None,
        min_invested_weight=0.0,
        estimate_from_history=False,
        lookback_days=2,
        mean_shrinkage=0.75,
        return_model="sample-mean",
        strategy="mean-variance",
        momentum_window=2,
        top_k=1,
        dual_momentum_weighting="equal",
        softmax_temperature=0.05,
        absolute_momentum_threshold=0.0,
        target_vol=None,
        vol_window=63,
        max_single_weight=None,
        trailing_stop=None,
        basket_opt=None,
        basket_risk_aversion=1.0,
        data_source="yfinance",
        backtest_days=3,
        rebalance_every=1,
        rolling_window_days=0,
        rolling_step_days=1,
        sweep=False,
        top_n=5,
        submit=False,
        use_cache=False,
        refresh_cache=False,
        offline=False,
        dry_run=True,
    )

    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        cli,
        "load_model_inputs",
        lambda _path: ModelInputs(
            symbols=["SPY", "IEF"],
            expected_returns=np.array([0.1, 0.2], dtype=float),
            covariance=np.eye(2, dtype=float),
            asset_classes={"SPY": "equity", "IEF": "bond"},
            class_min_weights={},
            class_max_weights={},
        ),
    )
    monkeypatch.setattr(
        cli,
        "yf_fetch_closes",
        lambda symbols, period="max", **_kwargs: {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "IEF": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_backtest",
        lambda **_kwargs: BacktestResult(
            final_value=1.1,
            total_return=0.1,
            annualized_return=0.12,
            annualized_volatility=0.08,
            max_drawdown=0.03,
            rebalance_count=2,
            average_turnover=0.1,
            latest_weights=np.array([0.4, 0.6], dtype=float),
            daily_values=(1.0, 1.05, 1.1),
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_fixed_weight_benchmark",
        lambda **_kwargs: {
            "final_value": 1.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "max_drawdown": 0.0,
        },
    )
    monkeypatch.setattr(
        cli,
        "AlpacaClient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Alpaca should not be constructed for yfinance backtests")
        ),
    )

    cli.main()
    out = capsys.readouterr().out

    assert '"final_value": 1.1' in out


def test_cli_dual_momentum_backtest_passes_vol_window(monkeypatch, capsys) -> None:
    args = Namespace(
        model="dummy.json",
        dynamic_universe=False,
        filter_before=None,
        ticker_basket=[],
        risk_aversion=4.0,
        min_weight=0.0,
        max_weight=0.35,
        rebalance_threshold=0.02,
        turnover_penalty=0.02,
        allow_cash=False,
        min_cash_weight=0.0,
        max_turnover=None,
        min_invested_weight=0.0,
        estimate_from_history=True,
        lookback_days=2,
        mean_shrinkage=0.75,
        return_model="sample-mean",
        strategy="dual-momentum",
        momentum_window=2,
        top_k=1,
        dual_momentum_weighting="equal",
        softmax_temperature=0.05,
        absolute_momentum_threshold=0.0,
        target_vol=0.35,
        vol_window=21,
        max_single_weight=None,
        trailing_stop=None,
        basket_opt=None,
        basket_risk_aversion=1.0,
        data_source="yfinance",
        backtest_days=3,
        rebalance_every=1,
        rolling_window_days=0,
        rolling_step_days=1,
        sweep=False,
        top_n=5,
        submit=False,
        use_cache=False,
        refresh_cache=False,
        offline=False,
        dry_run=True,
    )
    captured: dict[str, int | None] = {}

    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        cli,
        "load_model_inputs",
        lambda _path: ModelInputs(
            symbols=["SPY", "IEF"],
            expected_returns=None,
            covariance=None,
            asset_classes={"SPY": "equity", "IEF": "bond_intermediate"},
            class_min_weights={},
            class_max_weights={},
        ),
    )
    monkeypatch.setattr(
        cli,
        "yf_fetch_closes",
        lambda symbols, period="max", **_kwargs: {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "IEF": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5],
        },
    )

    def fake_dual_momentum_backtest(**kwargs):
        captured["vol_window"] = kwargs["vol_window"]
        return BacktestResult(
            final_value=1.1,
            total_return=0.1,
            annualized_return=0.12,
            annualized_volatility=0.08,
            max_drawdown=0.03,
            rebalance_count=2,
            average_turnover=0.1,
            latest_weights=np.array([1.0, 0.0], dtype=float),
            daily_values=(1.0, 1.05, 1.1),
        )

    monkeypatch.setattr(cli, "run_dual_momentum_backtest", fake_dual_momentum_backtest)
    monkeypatch.setattr(
        cli,
        "run_fixed_weight_benchmark",
        lambda **_kwargs: {
            "final_value": 1.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "max_drawdown": 0.0,
        },
    )
    monkeypatch.setattr(
        cli,
        "AlpacaClient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Alpaca should not be constructed for yfinance backtests")
        ),
    )

    cli.main()

    assert captured["vol_window"] == 21
    assert '"vol_window": 21' in capsys.readouterr().out


def test_cli_rejects_backtest_when_common_history_is_too_short(monkeypatch) -> None:
    args = Namespace(
        model="dummy.json",
        dynamic_universe=False,
        filter_before=None,
        ticker_basket=[],
        risk_aversion=4.0,
        min_weight=0.0,
        max_weight=0.35,
        rebalance_threshold=0.02,
        turnover_penalty=0.02,
        allow_cash=False,
        min_cash_weight=0.0,
        max_turnover=None,
        min_invested_weight=0.0,
        estimate_from_history=True,
        lookback_days=2,
        mean_shrinkage=0.75,
        return_model="sample-mean",
        strategy="mean-variance",
        momentum_window=2,
        top_k=1,
        dual_momentum_weighting="equal",
        softmax_temperature=0.05,
        absolute_momentum_threshold=0.0,
        target_vol=None,
        vol_window=63,
        max_single_weight=None,
        trailing_stop=None,
        basket_opt=None,
        basket_risk_aversion=1.0,
        data_source="yfinance",
        backtest_days=10,
        rebalance_every=1,
        rolling_window_days=0,
        rolling_step_days=1,
        sweep=False,
        top_n=5,
        submit=False,
        use_cache=False,
        refresh_cache=False,
        offline=False,
        dry_run=True,
    )

    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        cli,
        "load_model_inputs",
        lambda _path: ModelInputs(
            symbols=["SPY", "IEF"],
            expected_returns=None,
            covariance=None,
            asset_classes={"SPY": "equity", "IEF": "bond"},
            class_min_weights={},
            class_max_weights={},
        ),
    )
    monkeypatch.setattr(
        cli,
        "yf_fetch_closes",
        lambda symbols, period="max", **_kwargs: {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
            "IEF": [100.0, 100.5, 101.0, 101.5, 102.0],
        },
    )

    try:
        cli.main()
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("cli.main should fail when common history is too short")

    assert "Not enough common history for the requested backtest" in message
    assert "supports at most 2 backtest days" in message


def test_offline_close_fallback_reuses_cached_superset_in_requested_order(
    monkeypatch,
) -> None:
    client = AlpacaClient(AlpacaConfig(api_key="key", api_secret="secret"))
    payload = {
        "SPY": [100.0, 101.0, 102.0, 103.0],
        "QQQ": [200.0, 201.0, 202.0, 203.0],
        "TLT": [300.0, 301.0, 302.0, 303.0],
    }

    monkeypatch.setattr(
        "portfolio_opt.alpaca_interface.Path.glob",
        lambda self, pattern: ["dummy.json"],
    )
    monkeypatch.setattr(
        "portfolio_opt.alpaca_interface.read_cache",
        lambda _path: payload,
    )

    closes = client._find_offline_closes_fallback(["TLT", "SPY"], lookback_days=3)

    assert closes == {
        "TLT": [301.0, 302.0, 303.0],
        "SPY": [101.0, 102.0, 103.0],
    }


def test_latest_prices_payload_accepts_trade_objects_and_dicts() -> None:
    class FakeTrade:
        def __init__(self, price: float) -> None:
            self.price = price

    class FakeDataClient:
        def get_stock_latest_trade(self, _request):
            return Namespace(
                data={
                    "APP": FakeTrade(510.25),
                    "SNDK": [FakeTrade(72.5)],
                    "GLD": {"p": 310.0},
                    "XLU": {"price": 85.75},
                }
            )

    client = object.__new__(AlpacaClient)
    client._data = FakeDataClient()

    assert client._latest_prices_payload(["APP", "SNDK", "GLD", "XLU"]) == {
        "APP": {"p": 510.25},
        "SNDK": {"p": 72.5},
        "GLD": {"p": 310.0},
        "XLU": {"p": 85.75},
    }


def test_submit_trailing_stop_plan_uses_qty_and_percent() -> None:
    class FakeTrading:
        def __init__(self) -> None:
            self.submitted = None

        def submit_order(self, order_data):
            self.submitted = order_data
            return Namespace(id="order-1")

    fake_trading = FakeTrading()
    client = object.__new__(AlpacaClient)
    client._trading = fake_trading

    submitted = client.submit_trailing_stop_plan(
        [
            TrailingStopPlan(
                symbol="SPY",
                qty=12.345678,
                side="sell",
                trail_percent=15.0,
                time_in_force="gtc",
            )
        ]
    )

    assert submitted == [
        {
            "id": "order-1",
            "symbol": "SPY",
            "side": "sell",
            "trail_percent": 15.0,
        }
    ]
    assert fake_trading.submitted.symbol == "SPY"
    assert fake_trading.submitted.qty == 12.345678
    assert fake_trading.submitted.notional is None
    assert fake_trading.submitted.trail_percent == 15.0
    assert fake_trading.submitted.type.value == "trailing_stop"
    assert fake_trading.submitted.time_in_force.value == "gtc"


def test_cancel_open_trailing_stops_only_cancels_matching_symbols() -> None:
    class FakeTrading:
        def __init__(self) -> None:
            self.canceled: list[str] = []

        def cancel_order_by_id(self, order_id: str) -> None:
            self.canceled.append(order_id)

    fake_trading = FakeTrading()
    client = object.__new__(AlpacaClient)
    client._trading = fake_trading

    canceled = client.cancel_open_trailing_stops(
        ["SPY"],
        open_orders=[
            {
                "id": "stop-1",
                "symbol": "SPY",
                "qty": 10,
                "side": "sell",
                "type": "trailing_stop",
            },
            {
                "id": "limit-1",
                "symbol": "SPY",
                "qty": 1,
                "side": "sell",
                "type": "limit",
            },
            {
                "id": "stop-2",
                "symbol": "QQQ",
                "qty": 2,
                "side": "sell",
                "type": "trailing_stop",
            },
        ],
    )

    assert canceled == [{"id": "stop-1", "symbol": "SPY"}]
    assert fake_trading.canceled == ["stop-1"]


def test_daily_closes_refresh_appends_only_missing_bars(monkeypatch, tmp_path) -> None:
    client = object.__new__(AlpacaClient)
    cache_file = tmp_path / "spy.json"
    cache_file.write_text(
        json.dumps(
            {
                "symbol": "SPY",
                "source": "alpaca",
                "feed": "iex",
                "adjustment": "all",
                "closes": {
                    "2026-01-01": 101.0,
                    "2026-01-02": 102.0,
                },
            }
        )
    )
    calls: list[tuple[list[str], datetime, int]] = []

    def fake_daily_bar_rows_payload(symbols, *, start, end, limit):
        calls.append((symbols, start, limit))
        return {"SPY": [{"timestamp": "2026-01-05", "close": 103.0}]}

    monkeypatch.setattr(client, "_daily_closes_v2_cache_path", lambda _symbol: cache_file)
    monkeypatch.setattr(client, "_daily_bar_rows_payload", fake_daily_bar_rows_payload)
    monkeypatch.setattr(
        "portfolio_opt.alpaca_interface.datetime",
        type(
            "FakeDateTime",
            (),
            {
                "now": staticmethod(lambda _tz=None: datetime(2026, 1, 5, tzinfo=UTC)),
                "strptime": staticmethod(datetime.strptime),
            },
        ),
    )

    closes = client.get_daily_closes(
        ["SPY"],
        lookback_days=2,
        use_cache=True,
        refresh_cache=True,
    )

    assert closes == {"SPY": [102.0, 103.0]}
    assert calls == [(["SPY"], datetime(2026, 1, 3, tzinfo=UTC), 7)]
    assert json.loads(cache_file.read_text()) == {
        "symbol": "SPY",
        "source": "alpaca",
        "feed": "iex",
        "adjustment": "all",
        "closes": {
            "2026-01-02": 102.0,
            "2026-01-05": 103.0,
        },
    }
