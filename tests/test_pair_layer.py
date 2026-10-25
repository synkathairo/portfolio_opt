from __future__ import annotations

import numpy as np
import pytest

from portfolio_opt.instruments import Leg, SpreadSpec
from portfolio_opt.pair_layer import (
    PairLayerConfig,
    build_spread_nav,
    estimate_pair_beta,
    normalized_leg_weights,
    spread_period_returns,
)


def test_normalized_leg_weights_use_gross_exposure() -> None:
    weights = normalized_leg_weights(
        (Leg("AAA", 1.0), Leg("BBB", -2.0), Leg("CCC", -1.0))
    )

    assert weights == pytest.approx(np.array([0.25, -0.5, -0.25]))


def test_spread_period_returns_charge_borrow_on_short_leg() -> None:
    leg_returns = np.array([[0.10, 0.02], [-0.04, 0.00]])
    weights = np.array([1.0, -1.0])

    result = spread_period_returns(
        leg_returns,
        weights,
        borrow_rate=0.03,
        trading_days_per_year=252,
    )

    assert result == pytest.approx(np.array([0.07, 0.01]) - 0.03 / 252 * 0.5)


def test_estimate_pair_beta_recovers_ols_slope() -> None:
    short_returns = np.array([0.01, -0.02, 0.03, 0.00, 0.01])
    long_returns = 2.0 * short_returns

    beta = estimate_pair_beta(long_returns, short_returns)

    assert beta == pytest.approx(2.0)


def test_build_spread_nav_is_flat_until_first_formation() -> None:
    returns = {
        "AAA": np.array([0.01 + 0.001 * np.sin(i) for i in range(20)]),
        "BBB": np.array([0.005 + 0.001 * np.sin(i) for i in range(20)]),
    }
    spec = SpreadSpec(
        id="AAA/BBB",
        legs=(Leg("AAA", 1.0), Leg("BBB", -1.0)),
        category="test",
        asset_class="sector_test",
    )
    config = PairLayerConfig(
        formation_window=5,
        formation_every=5,
        borrow_rate=0.0,
    )

    nav = build_spread_nav(spec, returns, config)

    assert len(nav) == 21
    assert nav[0] == pytest.approx(1.0)
    assert nav[5] == pytest.approx(1.0)
    assert nav[10] > 1.0
    assert nav[15] > nav[10]


def test_build_spread_nav_uses_fixed_multi_leg_weights() -> None:
    returns = {
        "AAA": np.full(20, 0.01),
        "BBB": np.full(20, 0.005),
        "CCC": np.full(20, 0.002),
    }
    spec = SpreadSpec(
        id="AAA_vs_peers",
        legs=(Leg("AAA", 1.0), Leg("BBB", -0.5), Leg("CCC", -0.5)),
        category="test",
        asset_class="sector_test",
        hedge="fixed",
    )
    config = PairLayerConfig(
        formation_window=5,
        formation_every=5,
        borrow_rate=0.0,
    )

    nav = build_spread_nav(spec, returns, config)

    assert len(nav) == 21
    assert nav[5] == pytest.approx(1.0)
    assert nav[-1] > 1.0
