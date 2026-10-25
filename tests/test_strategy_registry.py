from __future__ import annotations

import numpy as np
import pytest

from portfolio_opt.config import OptimizationConfig
from portfolio_opt.instruments import Leg, SpreadSpec
from portfolio_opt.model import ModelInputs
from portfolio_opt.pair_layer import PairLayerConfig
from portfolio_opt.strategy_data import build_strategy_data
from portfolio_opt.strategy_registry import run_strategy
from portfolio_opt.universe_builder import build_instrument_universe


def _data():
    model = ModelInputs(
        symbols=["AAA"],
        expected_returns=None,
        covariance=None,
        asset_classes={"AAA": "sector_test"},
        class_min_weights={},
        class_max_weights={},
        pairs=(
            SpreadSpec(
                id="AAA/BBB",
                legs=(Leg("AAA", 1.0), Leg("BBB", -1.0)),
                category="test_pair",
                asset_class="sector_test",
            ),
        ),
    )
    prices = {
        "AAA": [100.0 + index for index in range(30)],
        "BBB": [100.0 + 0.3 * index for index in range(30)],
    }
    universe = build_instrument_universe(
        model=model,
        closes_by_symbol=prices,
        pair_config=PairLayerConfig(
            formation_window=5,
            formation_every=5,
            borrow_rate=0.0,
        ),
    )
    return build_strategy_data(
        universe=universe,
        cash_nav=np.ones(universe.period_count),
    )


def test_run_strategy_dual_momentum_returns_instrument_weights() -> None:
    data = _data()

    result = run_strategy(
        "dual-momentum",
        data,
        lookback_days=3,
        rebalance_every=1,
        top_k=1,
        linear_trade_cost=0.0,
    )

    assert result.instrument_ids == data.ids
    assert set(result.latest_weights) == set(data.ids)
    assert result.latest_cash_weight >= 0.0


def test_run_strategy_mean_variance_runs_on_mixed_universe() -> None:
    data = _data()

    result = run_strategy(
        "mean-variance",
        data,
        lookback_days=3,
        rebalance_every=1,
        return_model="sample-mean",
        mean_shrinkage=0.0,
        momentum_window=2,
        opt_config=OptimizationConfig(
            force_full_investment=True,
            max_weight=1.0,
        ),
        asset_class_matrix=None,
    )

    assert result.result.final_value > 0.0
    assert set(result.latest_weights) == set(data.ids)


def test_run_strategy_factor_momentum_runs_on_mixed_universe() -> None:
    data = _data()

    result = run_strategy(
        "factor-momentum",
        data,
        lookback_days=3,
        rebalance_every=1,
        top_k=1,
        factor_top_k=1,
        absolute_threshold=0.0,
        linear_trade_cost=0.0,
    )

    assert result.result.final_value > 0.0


def test_run_strategy_protective_momentum_runs_on_mixed_universe() -> None:
    data = _data()

    result = run_strategy(
        "protective-momentum",
        data,
        lookback_days=3,
        rebalance_every=1,
        top_k=1,
        absolute_threshold=0.0,
        linear_trade_cost=0.0,
    )

    assert result.result.final_value > 0.0


def test_run_strategy_regime_adaptive_runs_on_mixed_universe() -> None:
    data = _data()

    result = run_strategy(
        "regime-adaptive",
        data,
        lookback_days=3,
        rebalance_every=1,
        top_k=1,
        absolute_threshold=0.0,
        selection_window=2,
        mean_reversion_window=2,
        linear_trade_cost=0.0,
    )

    assert result.result.final_value > 0.0


def test_run_strategy_fixed_allocation_runs_on_mixed_universe() -> None:
    data = _data()

    result = run_strategy(
        "fixed-allocation",
        data,
        weights_by_symbol={"AAA": 0.5, "AAA/BBB": 0.5},
        start_day=3,
        rebalance_every=1,
    )

    assert result.result.final_value > 0.0


def test_run_strategy_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_strategy("not-a-strategy", _data())
