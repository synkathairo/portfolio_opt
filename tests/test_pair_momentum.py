from __future__ import annotations

import numpy as np
import pytest

from portfolio_opt.pair_momentum import (
    PairBacktestConfig,
    build_pair_navs,
    estimate_pair_beta,
    latest_formation,
    pair_spread_returns,
    run_pair_momentum_backtest,
    select_pairs,
)


def test_pair_spread_returns_is_beta_neutral_and_charges_borrow() -> None:
    long_returns = np.array([0.10, -0.05, 0.02])
    short_returns = np.array([0.02, 0.01, -0.01])

    result = pair_spread_returns(
        long_returns,
        short_returns,
        2.0,
        borrow_rate=0.03,
        trading_days_per_year=252,
    )
    expected = (long_returns - 2.0 * short_returns) / 3.0 - 0.03 / 252 * (2.0 / 3.0)

    assert result == pytest.approx(expected)


def test_pair_spread_returns_rejects_invalid_beta() -> None:
    with pytest.raises(ValueError, match="beta"):
        pair_spread_returns(np.array([0.0]), np.array([0.0]), 0.0)


def test_select_pairs_prefers_correlated_pairs() -> None:
    returns = np.zeros((4, 80), dtype=float)
    returns[0] = np.linspace(-0.02, 0.02, 80)
    returns[1] = returns[0]
    returns[2] = np.sin(np.linspace(0.0, 20.0, 80)) * 0.02
    returns[3] = np.cos(np.linspace(0.0, 20.0, 80)) * 0.02
    config = PairBacktestConfig(
        formation_window=60,
        formation_every=20,
        momentum_min=5,
        pairs_per_sector=1,
        min_corr=0.5,
    )

    selected = select_pairs(
        returns,
        {"sector_test": [0, 1, 2, 3]},
        formation_index=60,
        config=config,
    )

    assert len(selected) == 1
    assert selected[0].long_index == 0
    assert selected[0].short_index == 1
    assert selected[0].beta == pytest.approx(1.0)


def test_build_pair_navs_is_flat_before_first_valid_beta() -> None:
    returns = np.zeros((2, 20), dtype=float)
    returns[0, 5:] = 0.01 + 0.001 * np.sin(np.arange(15))
    returns[1, 5:] = 0.005 + 0.001 * np.sin(np.arange(15))
    config = PairBacktestConfig(
        formation_window=5,
        formation_every=5,
        momentum_min=2,
        pair_cost=0.0,
        borrow_rate=0.0,
    )

    navs = build_pair_navs(returns, [(0, 1)], config, net=False)

    nav = navs[(0, 1)]
    assert len(nav) == 21
    assert nav[0] == pytest.approx(1.0)
    assert nav[5] == pytest.approx(1.0)
    assert nav[10] == pytest.approx(1.0)
    assert nav[15] > 1.0


def test_latest_formation_waits_for_momentum_window() -> None:
    config = PairBacktestConfig(
        formation_window=10,
        formation_every=5,
        momentum_min=3,
    )

    assert latest_formation(12, config) is None
    assert latest_formation(13, config) == 10
    assert latest_formation(17, config) == 10
    assert latest_formation(18, config) == 15


def test_estimate_pair_beta_uses_formation_window() -> None:
    returns = np.zeros((2, 20), dtype=float)
    returns[0, 5:15] = np.linspace(-0.01, 0.01, 10)
    returns[1, 5:15] = returns[0, 5:15] / 2.0
    config = PairBacktestConfig(formation_window=10, formation_every=5, momentum_min=2)

    beta = estimate_pair_beta(returns, 0, 1, 15, config)

    assert beta == pytest.approx(2.0)


def test_pair_backtest_produces_finite_result_and_nonnegative_weights() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 0.01, size=(80,))
    returns = np.vstack(
        [
            base + rng.normal(0.0, 0.001, size=80),
            base + rng.normal(0.0, 0.001, size=80),
            base + rng.normal(0.0, 0.001, size=80),
            base + rng.normal(0.0, 0.001, size=80),
        ]
    )
    config = PairBacktestConfig(
        formation_window=20,
        formation_every=10,
        momentum_min=5,
        rebalance_every=5,
        pairs_per_sector=1,
        top_k=1,
        pair_cost=0.0,
        borrow_rate=0.0,
    )

    result = run_pair_momentum_backtest(
        symbols=["AAA", "BBB", "CCC", "DDD"],
        returns=returns,
        sector_indices={"sector_a": [0, 1], "sector_b": [2, 3]},
        config=config,
        mode="pair-rank",
        signal="momentum",
    )

    assert np.isfinite(result.final_value)
    assert result.daily_values
    assert result.rebalance_count > 0
    assert sum(result.latest_pair_weights.values()) <= 1.0 + 1e-12


def test_fixed_pair_indices_limit_the_pair_universe() -> None:
    rng = np.random.default_rng(13)
    base = rng.normal(0.0, 0.01, size=80)
    returns = np.vstack(
        [
            base + rng.normal(0.0, 0.001, size=80),
            base + rng.normal(0.0, 0.001, size=80),
            base + rng.normal(0.0, 0.001, size=80),
            base + rng.normal(0.0, 0.001, size=80),
        ]
    )
    config = PairBacktestConfig(
        formation_window=20,
        formation_every=10,
        momentum_min=5,
        rebalance_every=5,
        top_k=1,
        pair_cost=0.0,
        borrow_rate=0.0,
    )

    result = run_pair_momentum_backtest(
        symbols=["AAA", "BBB", "CCC", "DDD"],
        returns=returns,
        sector_indices={"sector_a": [0, 1], "sector_b": [2, 3]},
        config=config,
        mode="pair-rank",
        fixed_pair_indices=[(0, 1)],
    )

    assert set(result.latest_pair_weights) <= {"AAA/BBB"}


def test_pair_backtest_accepts_category_mode() -> None:
    rng = np.random.default_rng(11)
    base = rng.normal(0.0, 0.01, size=(80,))
    returns = np.vstack([base + rng.normal(0.0, 0.001, size=80) for _ in range(4)])
    config = PairBacktestConfig(
        formation_window=20,
        formation_every=10,
        momentum_min=5,
        rebalance_every=5,
        pairs_per_sector=1,
        top_k=1,
        category_top_k=1,
        pairs_per_category=1,
        pair_cost=0.0,
        borrow_rate=0.0,
    )

    result = run_pair_momentum_backtest(
        symbols=["AAA", "BBB", "CCC", "DDD"],
        returns=returns,
        sector_indices={"sector_a": [0, 1], "sector_b": [2, 3]},
        config=config,
        mode="category-index",
        signal="momentum",
    )

    assert np.isfinite(result.final_value)
    assert all(
        isinstance(pair, str) and "/" in pair for pair in result.latest_pair_weights
    )
