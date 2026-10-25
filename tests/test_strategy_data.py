from __future__ import annotations

import numpy as np
import pytest

from portfolio_opt.instruments import Leg, SpreadSpec
from portfolio_opt.model import ModelInputs
from portfolio_opt.pair_layer import PairLayerConfig
from portfolio_opt.strategy_data import build_strategy_data
from portfolio_opt.universe_builder import build_instrument_universe


def _universe():
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
        "AAA": [100.0 + index for index in range(20)],
        "BBB": [100.0 + 0.3 * index for index in range(20)],
    }
    return build_instrument_universe(
        model=model,
        closes_by_symbol=prices,
        pair_config=PairLayerConfig(
            formation_window=5,
            formation_every=5,
            borrow_rate=0.0,
        ),
    )


def test_build_strategy_data_adds_cash_and_metadata() -> None:
    universe = _universe()
    cash = np.linspace(1.0, 1.01, universe.period_count)

    data = build_strategy_data(universe=universe, cash_nav=cash)

    assert data.ids == ("AAA", "AAA/BBB")
    assert set(data.closes) == {"AAA", "AAA/BBB", "CASH"}
    assert data.asset_classes["CASH"] == "cash_like"
    assert data.kinds["CASH"] == "cash"
    assert data.returns.shape == (3, universe.period_count - 1)


def test_build_strategy_data_rejects_bad_cash_length() -> None:
    universe = _universe()

    with pytest.raises(ValueError, match="cash_nav"):
        build_strategy_data(
            universe=universe,
            cash_nav=np.ones(universe.period_count - 1),
        )


def test_build_strategy_data_rejects_cash_id_collision() -> None:
    universe = _universe()

    with pytest.raises(ValueError, match="collides"):
        build_strategy_data(
            universe=universe,
            cash_nav=np.ones(universe.period_count),
            cash_id="AAA",
        )
