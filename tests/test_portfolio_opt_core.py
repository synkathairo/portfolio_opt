from __future__ import annotations

from argparse import Namespace
from datetime import UTC, datetime, timedelta
import json
from uuid import UUID

import numpy as np

from portfolio_opt.alpaca_interface import AlpacaClient
from portfolio_opt import cli
from portfolio_opt.cache import read_cache, write_cache
from portfolio_opt.backtest import (
    BacktestResult,
    compute_dual_momentum_weights,
    run_backtest,
    summarize_return_series,
)
from portfolio_opt.config import AlpacaConfig, OptimizationConfig
from portfolio_opt.execution import submit_rebalance_sell_first
from portfolio_opt.model import ModelInputs, load_model_inputs
from portfolio_opt.optimizer import _finalize_solution, optimize_weights, project_weights
from portfolio_opt.risk_parity import risk_parity_weights
from portfolio_opt.rebalance import build_order_plan, build_trailing_stop_plan
from portfolio_opt.types import (
    AccountSnapshot,
    OrderPlan,
    Position,
    TrailingStopPlan,
    UnprotectedTrailingStopQty,
)


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


def test_optimize_weights_preserves_constraints_after_cleanup() -> None:
    config = OptimizationConfig(
        risk_aversion=0.01,
        min_weight=0.0,
        max_weight=0.5,
        force_full_investment=True,
        max_turnover=0.2,
        class_min_weights={"equity": 0.6},
        class_max_weights={"bond": 0.4},
    )
    current_weights = np.array([0.3, 0.3, 0.4], dtype=float)
    asset_class_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    weights = optimize_weights(
        expected_returns=np.array([0.5, 0.1, 0.0], dtype=float),
        covariance=np.eye(3, dtype=float) * 1e-4,
        config=config,
        current_weights=current_weights,
        asset_class_matrix=asset_class_matrix,
    )

    class_exposures = asset_class_matrix @ weights
    assert abs(float(weights.sum()) - 1.0) <= 1e-6
    assert float(weights.max()) <= 0.5 + 1e-6
    assert float(np.abs(weights - current_weights).sum()) <= 0.2 + 1e-6
    assert class_exposures[0] >= 0.6 - 1e-6
    assert class_exposures[1] <= 0.4 + 1e-6


def test_project_weights_preserves_max_weight_after_cleanup() -> None:
    weights = project_weights(
        target_weights=np.array([1.0, 0.0, 0.0], dtype=float),
        config=OptimizationConfig(max_weight=0.4, force_full_investment=True),
    )

    assert abs(float(weights.sum()) - 1.0) <= 1e-6
    assert float(weights.max()) <= 0.4 + 1e-6


def test_finalize_solution_accepts_small_solver_feasibility_residual() -> None:
    weights = _finalize_solution(
        np.array([0.4, 0.599998991866], dtype=float),
        OptimizationConfig(max_weight=1.0, force_full_investment=True),
        context="Optimization",
    )

    assert abs(float(weights.sum()) - 1.0) <= 1e-5


def test_finalize_solution_rejects_material_full_investment_residual() -> None:
    try:
        _finalize_solution(
            np.array([0.4, 0.59998], dtype=float),
            OptimizationConfig(max_weight=1.0, force_full_investment=True),
            context="Optimization",
        )
    except RuntimeError as exc:
        assert "full_investment_sum" in str(exc)
    else:
        raise AssertionError("Material full-investment residual should fail.")


def _risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    marginal = covariance @ weights
    return weights * marginal / float(weights @ marginal)


def test_risk_parity_equal_vol_assets_are_equal_weight() -> None:
    weights = risk_parity_weights(np.eye(3, dtype=float) * 0.04)

    assert np.allclose(weights, np.full(3, 1.0 / 3.0), atol=1e-5)
    assert np.allclose(
        _risk_contributions(weights, np.eye(3, dtype=float) * 0.04),
        np.full(3, 1.0 / 3.0),
        atol=1e-5,
    )


def test_risk_parity_diagonal_covariance_inverse_vol_weights() -> None:
    covariance = np.diag([0.01, 0.04, 0.16])
    weights = risk_parity_weights(covariance)

    inverse_vol = np.array([10.0, 5.0, 2.5], dtype=float)
    expected = inverse_vol / inverse_vol.sum()
    assert np.allclose(weights, expected, atol=1e-5)
    assert np.allclose(
        _risk_contributions(weights, covariance),
        np.full(3, 1.0 / 3.0),
        atol=1e-5,
    )


def test_cache_write_is_atomic_and_round_trips(tmp_path) -> None:
    path = tmp_path / "nested" / "cache.json"
    write_cache(path, {"symbols": ["SPY"], "values": [1.0]})

    assert read_cache(path) == {"symbols": ["SPY"], "values": [1.0]}
    assert not path.with_name("cache.json.tmp").exists()


def test_summarize_return_series_uses_configurable_annualization() -> None:
    returns = np.array([0.01, -0.02, 0.03, -0.01], dtype=float)

    summary_252 = summarize_return_series(returns, trading_days_per_year=252)
    summary_365 = summarize_return_series(returns, trading_days_per_year=365)

    assert summary_252.annualized_return != summary_365.annualized_return
    assert summary_252.annualized_volatility != summary_365.annualized_volatility
    assert summary_252.sortino_ratio != summary_365.sortino_ratio
    assert round(summary_252.max_drawdown, 6) == 0.02


def test_load_model_inputs_rejects_malformed_static_inputs(tmp_path) -> None:
    bad_models = [
        {"symbols": ["SPY", 123]},
        {
            "symbols": ["SPY"],
            "expected_returns": {"SPY": float("nan")},
            "covariance": [[1.0]],
        },
        {
            "symbols": ["SPY", "IEF"],
            "expected_returns": {"SPY": 0.1, "IEF": 0.2},
            "covariance": [[1.0, 0.2], [0.1, 1.0]],
        },
        {
            "symbols": ["SPY"],
            "asset_classes": {"SPY": "equity"},
            "class_min_weights": {"equity": 1.2},
        },
    ]

    for index, payload in enumerate(bad_models):
        path = tmp_path / f"bad-{index}.json"
        path.write_text(json.dumps(payload))
        try:
            load_model_inputs(path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Malformed model should fail: {payload}")


def test_dynamic_universe_cache_writes_model_and_sidecar(tmp_path) -> None:
    ticker_basket = ["nasdaq100", "sp500"]
    fetched = {
        "symbols": ["AAPL", "MSFT"],
        "asset_classes": {
            "AAPL": "Apple Inc. Common Stock",
            "MSFT": "Microsoft Corporation",
        },
    }

    cli._write_dynamic_universe_cache(
        fetched,
        ticker_basket=ticker_basket,
        cache_dir=tmp_path,
    )
    model_path, meta_path = cli._dynamic_universe_cache_paths(ticker_basket, tmp_path)

    assert json.loads(model_path.read_text()) == fetched
    meta = json.loads(meta_path.read_text())
    assert meta["kind"] == "dynamic_universe_cache"
    assert meta["ticker_basket"] == ticker_basket
    assert meta["symbol_count"] == 2
    assert (
        cli._read_dynamic_universe_cache(
            ticker_basket=ticker_basket,
            cache_dir=tmp_path,
            max_age_days=1,
        )
        == fetched
    )


def test_dynamic_universe_cache_rejects_too_stale_sidecar(tmp_path) -> None:
    ticker_basket = ["nasdaq100"]
    fetched = {
        "symbols": ["AAPL"],
        "asset_classes": {"AAPL": "Apple Inc. Common Stock"},
    }

    cli._write_dynamic_universe_cache(
        fetched,
        ticker_basket=ticker_basket,
        cache_dir=tmp_path,
    )
    _model_path, meta_path = cli._dynamic_universe_cache_paths(
        ticker_basket,
        tmp_path,
    )
    meta = json.loads(meta_path.read_text())
    meta["fetched_at"] = (datetime.now(UTC) - timedelta(days=3)).isoformat()
    meta_path.write_text(json.dumps(meta))

    try:
        cli._read_dynamic_universe_cache(
            ticker_basket=ticker_basket,
            cache_dir=tmp_path,
            max_age_days=1,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("stale dynamic universe cache should fail")

    assert "exceeding" in message


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

    def fake_optimize(
        *, expected_returns, covariance, config, current_weights, asset_class_matrix
    ):
        called["optimize"] = True
        return np.array([0.25, 0.75], dtype=float)

    monkeypatch.setattr(
        "portfolio_opt.backtest.estimate_inputs_from_black_litterman", fake_bl
    )
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

    monkeypatch.setattr(
        "portfolio_opt.backtest.estimate_inputs_risk_parity", fake_risk_parity
    )
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


def test_order_plan_skips_buys_below_threshold_when_buying_power_is_low() -> None:
    plan = build_order_plan(
        symbols=["APP", "SNDK"],
        target_weights=[0.5, 0.5],
        account=AccountSnapshot(equity=1178.12, buying_power=5.70),
        positions=[Position(symbol="SNDK", qty=0.633232, market_value=583.2)],
        latest_prices={"APP": 500.0},
        config=OptimizationConfig(rebalance_threshold=0.02),
    )

    assert plan == []


def test_order_plan_scales_buys_to_buying_power() -> None:
    plan = build_order_plan(
        symbols=["APP", "SNDK"],
        target_weights=[0.5, 0.5],
        account=AccountSnapshot(equity=1000.0, buying_power=300.0),
        positions=[],
        latest_prices={"APP": 500.0, "SNDK": 50.0},
        config=OptimizationConfig(rebalance_threshold=0.02),
    )

    assert plan == [
        OrderPlan("APP", 0.0, 0.15, 0.15, "buy", 150.0),
        OrderPlan("SNDK", 0.0, 0.15, 0.15, "buy", 150.0),
    ]


def test_build_trailing_stop_plan_skips_existing_protection() -> None:
    result = build_trailing_stop_plan(
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

    assert result.orders == [
        TrailingStopPlan(
            symbol="QQQ",
            qty=2.0,
            side="sell",
            trail_percent=15.0,
            time_in_force="gtc",
        )
    ]
    assert result.unprotected_qty == [
        UnprotectedTrailingStopQty(
            symbol="QQQ",
            position_qty=2.5,
            unprotected_qty=0.5,
        )
    ]


def test_build_trailing_stop_plan_floors_fractional_qty() -> None:
    result = build_trailing_stop_plan(
        symbols=["SPY"],
        target_weights=[1.0],
        positions=[Position(symbol="SPY", qty=12.345678, market_value=1234.57)],
        open_orders=[],
        trailing_stop=0.15,
        rebalance_threshold=0.02,
    )

    assert result.orders == [
        TrailingStopPlan(
            symbol="SPY",
            qty=12.0,
            side="sell",
            trail_percent=15.0,
            time_in_force="gtc",
        )
    ]
    assert result.unprotected_qty == [
        UnprotectedTrailingStopQty(
            symbol="SPY",
            position_qty=12.345678,
            unprotected_qty=0.345678,
        )
    ]


def test_build_trailing_stop_plan_reports_subshare_qty_without_order() -> None:
    result = build_trailing_stop_plan(
        symbols=["SPY"],
        target_weights=[1.0],
        positions=[Position(symbol="SPY", qty=0.75, market_value=75.0)],
        open_orders=[],
        trailing_stop=0.15,
        rebalance_threshold=0.02,
    )

    assert result.orders == []
    assert result.unprotected_qty == [
        UnprotectedTrailingStopQty(
            symbol="SPY",
            position_qty=0.75,
            unprotected_qty=0.75,
        )
    ]


def test_cli_dry_run_reports_unprotected_fractional_trailing_stop_qty(
    monkeypatch,
    capsys,
) -> None:
    args = Namespace(
        model="dummy.json",
        dynamic_universe=False,
        filter_before=None,
        ticker_basket=[],
        risk_aversion=4.0,
        min_weight=0.0,
        max_weight=1.0,
        rebalance_threshold=0.02,
        turnover_penalty=0.02,
        allow_cash=False,
        min_cash_weight=0.0,
        max_turnover=None,
        min_invested_weight=0.0,
        estimate_from_history=False,
        lookback_days=60,
        mean_shrinkage=0.75,
        return_model="sample-mean",
        strategy="mean-variance",
        momentum_window=63,
        top_k=3,
        factor_top_k=1,
        dual_momentum_weighting="equal",
        softmax_temperature=0.05,
        absolute_momentum_threshold=0.0,
        target_vol=None,
        vol_window=63,
        max_single_weight=None,
        trailing_stop=0.15,
        basket_opt=None,
        basket_risk_aversion=1.0,
        data_source="alpaca",
        backtest_days=0,
        rebalance_every=21,
        rolling_window_days=0,
        rolling_step_days=21,
        sweep=False,
        top_n=5,
        submit=False,
        use_cache=False,
        refresh_cache=False,
        offline=False,
        dry_run=True,
    )

    class FakeAlpaca:
        def __init__(self, _config) -> None:
            pass

        def get_account(self, **_kwargs):
            return AccountSnapshot(equity=1234.5678)

        def get_positions(self, **_kwargs):
            return [Position(symbol="SPY", qty=12.345678, market_value=1234.5678)]

        def get_open_orders(self):
            return []

        def get_latest_prices(self, *_args, **_kwargs):
            return {}

    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        cli,
        "load_model_inputs",
        lambda _path: ModelInputs(
            symbols=["SPY"],
            expected_returns=np.array([0.1], dtype=float),
            covariance=np.eye(1, dtype=float),
            asset_classes={"SPY": "equity"},
            class_min_weights={},
            class_max_weights={},
        ),
    )
    monkeypatch.setattr(cli, "AlpacaClient", FakeAlpaca)
    monkeypatch.setattr(cli, "optimize_weights", lambda **_kwargs: np.array([1.0]))

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["trailing_stop_orders"] == [
        {
            "symbol": "SPY",
            "qty": 12.0,
            "side": "sell",
            "trail_percent": 15.0,
            "time_in_force": "gtc",
        }
    ]
    assert payload["unprotected_fractional_trailing_stop_qty"] == [
        {
            "symbol": "SPY",
            "position_qty": 12.345678,
            "unprotected_qty": 0.345678,
        }
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


def test_cli_yfinance_backtest_supports_external_benchmark(
    monkeypatch,
    capsys,
) -> None:
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
        target_vol=None,
        vol_window=21,
        max_single_weight=None,
        trailing_stop=None,
        basket_opt=None,
        basket_risk_aversion=1.0,
        data_source="yfinance",
        yfinance_max_workers=10,
        yfinance_retry_delay=1.0,
        benchmark=["^HSI"],
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
    fetched_symbols: list[list[str]] = []
    benchmark_calls: list[dict[str, float]] = []

    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        cli,
        "load_model_inputs",
        lambda _path: ModelInputs(
            symbols=["0005.HK", "0700.HK"],
            expected_returns=None,
            covariance=None,
            asset_classes={"0005.HK": "equity", "0700.HK": "equity"},
            class_min_weights={},
            class_max_weights={},
        ),
    )

    def fake_fetch_closes(symbols, period="max", **_kwargs):
        fetched_symbols.append(list(symbols))
        return {
            "0005.HK": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "0700.HK": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5],
            "^HSI": [200.0, 201.0, 202.0, 203.0, 204.0, 205.0],
        }

    def fake_dual_momentum_backtest(**_kwargs):
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

    def fake_benchmark(**kwargs):
        if kwargs["weights_by_symbol"] == {"^HSI": 1.0}:
            benchmark_calls.append(kwargs["weights_by_symbol"])
        return {
            "final_value": 1.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "max_drawdown": 0.0,
        }

    monkeypatch.setattr(cli, "yf_fetch_closes", fake_fetch_closes)
    monkeypatch.setattr(cli, "run_dual_momentum_backtest", fake_dual_momentum_backtest)
    monkeypatch.setattr(cli, "run_fixed_weight_benchmark", fake_benchmark)
    monkeypatch.setattr(
        cli,
        "AlpacaClient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Alpaca should not be constructed for yfinance backtests")
        ),
    )

    cli.main()

    assert fetched_symbols == [["0005.HK", "0700.HK", "^HSI"]]
    assert benchmark_calls == [{"^HSI": 1.0}]
    assert '"benchmark_hsi"' in capsys.readouterr().out


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


def test_alpaca_client_honors_configured_live_endpoint(monkeypatch) -> None:
    trading_kwargs = {}
    data_kwargs = {}

    class FakeTradingClient:
        def __init__(self, **kwargs) -> None:
            trading_kwargs.update(kwargs)

    class FakeDataClient:
        def __init__(self, **kwargs) -> None:
            data_kwargs.update(kwargs)

    monkeypatch.setattr(
        "portfolio_opt.alpaca_interface.TradingClient",
        FakeTradingClient,
    )
    monkeypatch.setattr(
        "portfolio_opt.alpaca_interface.StockHistoricalDataClient",
        FakeDataClient,
    )

    AlpacaClient(
        AlpacaConfig(
            api_key="key",
            api_secret="secret",
            base_url="https://api.alpaca.markets",
            data_url="https://data.alpaca.markets",
        )
    )

    assert trading_kwargs == {
        "api_key": "key",
        "secret_key": "secret",
        "paper": False,
        "url_override": "https://api.alpaca.markets",
    }
    assert data_kwargs == {
        "api_key": "key",
        "secret_key": "secret",
        "url_override": "https://data.alpaca.markets",
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
                qty=12.0,
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
    assert fake_trading.submitted.qty == 12.0
    assert fake_trading.submitted.notional is None
    assert fake_trading.submitted.trail_percent == 15.0
    assert fake_trading.submitted.type.value == "trailing_stop"
    assert fake_trading.submitted.time_in_force.value == "gtc"


def test_submitted_order_results_are_json_serializable_with_uuid_ids() -> None:
    order_id = UUID("12345678-1234-5678-1234-567812345678")

    class FakeOrder:
        def __init__(self, status: str = "new") -> None:
            self.id = order_id
            self.status = status
            self.filled_qty = "1"

    class FakeTrading:
        def __init__(self) -> None:
            self.submitted = None

        def submit_order(self, order_data):
            self.submitted = order_data
            return FakeOrder()

        def get_order_by_id(self, order_id_arg: str):
            assert order_id_arg == str(order_id)
            return FakeOrder(status="filled")

    fake_trading = FakeTrading()
    client = object.__new__(AlpacaClient)
    client._trading = fake_trading

    submitted = client.submit_order_plan(
        [OrderPlan("SPY", 0.0, 1.0, 1.0, "buy", 123.45)]
    )
    statuses = client.wait_for_submitted_orders(
        submitted, timeout_seconds=1.0, poll_seconds=0.0
    )
    trailing_stops = client.submit_trailing_stop_plan(
        [
            TrailingStopPlan(
                symbol="SPY",
                qty=1.0,
                side="sell",
                trail_percent=15.0,
                time_in_force="gtc",
            )
        ]
    )

    assert submitted[0]["id"] == str(order_id)
    assert statuses[0]["id"] == str(order_id)
    assert statuses[0]["status"] == "filled"
    assert trailing_stops[0]["id"] == str(order_id)
    json.dumps(
        {
            "submitted_orders": submitted,
            "order_fill_statuses": statuses,
            "submitted_trailing_stops": trailing_stops,
        }
    )


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


def test_submit_rebalance_sell_first_waits_before_buying() -> None:
    class FakeBroker:
        def __init__(self) -> None:
            self.submitted_symbols: list[list[str]] = []

        def submit_order_plan(self, plans: list[OrderPlan]):
            self.submitted_symbols.append([plan.symbol for plan in plans])
            return [
                {"id": f"order-{plan.symbol}", "symbol": plan.symbol, "side": plan.side}
                for plan in plans
            ]

        def wait_for_submitted_orders(self, submitted_orders, **_kwargs):
            return [dict(order, status="filled") for order in submitted_orders]

        def get_account(self, **_kwargs):
            return AccountSnapshot(equity=1175.92)

        def get_positions(self, **_kwargs):
            return []

        def get_open_orders(self):
            return []

        def get_latest_prices(self, symbols, **_kwargs):
            assert symbols == ["APP"]
            return {"APP": 500.0}

    broker = FakeBroker()

    result = submit_rebalance_sell_first(
        broker=broker,
        plan=[
            OrderPlan("APP", 0.0, 0.5, 0.5, "buy", 587.96),
            OrderPlan("PLTR", 0.504958, 0.0, -0.504958, "sell", 593.79),
        ],
        symbols=["APP", "PLTR"],
        target_weights=[0.5, 0.0],
        config=OptimizationConfig(rebalance_threshold=0.02),
    )

    assert broker.submitted_symbols == [["PLTR"], ["APP"]]
    assert result.sell_fill_statuses == [
        {"id": "order-PLTR", "symbol": "PLTR", "side": "sell", "status": "filled"}
    ]
    assert result.buy_plan == [OrderPlan("APP", 0.0, 0.5, 0.5, "buy", 587.96)]
    assert result.skipped_buys_reason is None


def test_submit_rebalance_sell_first_skips_buys_when_sell_rejected() -> None:
    class FakeBroker:
        def __init__(self) -> None:
            self.submitted_symbols: list[list[str]] = []

        def submit_order_plan(self, plans: list[OrderPlan]):
            self.submitted_symbols.append([plan.symbol for plan in plans])
            return [
                {"id": f"order-{plan.symbol}", "symbol": plan.symbol, "side": plan.side}
                for plan in plans
            ]

        def wait_for_submitted_orders(self, submitted_orders, **_kwargs):
            return [dict(order, status="rejected") for order in submitted_orders]

        def get_account(self, **_kwargs):
            raise AssertionError("buy leg should not refresh account")

        def get_positions(self, **_kwargs):
            raise AssertionError("buy leg should not refresh positions")

        def get_open_orders(self):
            raise AssertionError("buy leg should not refresh open orders")

        def get_latest_prices(self, symbols, **_kwargs):
            raise AssertionError("buy leg should not fetch prices")

    broker = FakeBroker()

    result = submit_rebalance_sell_first(
        broker=broker,
        plan=[
            OrderPlan("APP", 0.0, 0.5, 0.5, "buy", 587.96),
            OrderPlan("PLTR", 0.504958, 0.0, -0.504958, "sell", 593.79),
        ],
        symbols=["APP", "PLTR"],
        target_weights=[0.5, 0.0],
        config=OptimizationConfig(rebalance_threshold=0.02),
    )

    assert broker.submitted_symbols == [["PLTR"]]
    assert result.buy_plan == []
    assert result.skipped_buys_reason == "one or more sell orders did not fill"


def test_submit_rebalance_sell_first_skips_buys_when_sell_not_accepted() -> None:
    class FakeBroker:
        def __init__(self) -> None:
            self.submitted_symbols: list[list[str]] = []

        def submit_order_plan(self, plans: list[OrderPlan]):
            self.submitted_symbols.append([plan.symbol for plan in plans])
            return []

        def wait_for_submitted_orders(self, submitted_orders, **_kwargs):
            raise AssertionError("unaccepted sell leg should not be polled")

        def get_account(self, **_kwargs):
            raise AssertionError("buy leg should not refresh account")

        def get_positions(self, **_kwargs):
            raise AssertionError("buy leg should not refresh positions")

        def get_open_orders(self):
            raise AssertionError("buy leg should not refresh open orders")

        def get_latest_prices(self, symbols, **_kwargs):
            raise AssertionError("buy leg should not fetch prices")

    broker = FakeBroker()

    result = submit_rebalance_sell_first(
        broker=broker,
        plan=[
            OrderPlan("APP", 0.0, 0.5, 0.5, "buy", 587.96),
            OrderPlan("PLTR", 0.504958, 0.0, -0.504958, "sell", 593.79),
        ],
        symbols=["APP", "PLTR"],
        target_weights=[0.5, 0.0],
        config=OptimizationConfig(rebalance_threshold=0.02),
    )

    assert broker.submitted_symbols == [["PLTR"]]
    assert result.buy_plan == []
    assert result.skipped_buys_reason == "one or more sell orders were not accepted"


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

    monkeypatch.setattr(
        client, "_daily_closes_v2_cache_path", lambda _symbol: cache_file
    )
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
