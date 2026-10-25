from __future__ import annotations

import numpy as np
import pytest

from portfolio_opt.instruments import Leg, SpreadSpec
from portfolio_opt.model import ModelInputs
from portfolio_opt.pair_layer import PairLayerConfig
from portfolio_opt.universe_builder import build_instrument_universe


def _prices() -> dict[str, list[float]]:
    return {
        "AAA": [100.0 + index for index in range(30)],
        "BBB": [100.0 + 0.5 * index for index in range(30)],
        "CCC": [100.0 + 0.2 * index for index in range(30)],
    }


def _model(symbols: list[str], pairs=()) -> ModelInputs:
    return ModelInputs(
        symbols=symbols,
        expected_returns=None,
        covariance=None,
        asset_classes={symbol: "sector_test" for symbol in symbols},
        class_min_weights={},
        class_max_weights={},
        pairs=pairs,
    )


def _pair() -> SpreadSpec:
    return SpreadSpec(
        id="AAA/BBB",
        legs=(Leg("AAA", 1.0), Leg("BBB", -1.0)),
        category="test_pair",
        asset_class="sector_test",
    )


def test_universe_builder_includes_naked_and_spread_instruments() -> None:
    universe = build_instrument_universe(
        model=_model(["AAA", "BBB"], (_pair(),)),
        closes_by_symbol=_prices(),
        pair_config=PairLayerConfig(
            formation_window=5,
            formation_every=5,
            borrow_rate=0.0,
        ),
    )

    assert set(universe.instruments) == {"AAA", "BBB", "AAA/BBB"}
    assert universe.instruments["AAA"].kind == "stock"
    assert universe.instruments["AAA/BBB"].kind == "spread"
    assert universe.instruments["AAA/BBB"].legs is not None
    assert universe.period_count == 30


def test_universe_builder_allows_pair_leg_not_in_naked_symbols() -> None:
    universe = build_instrument_universe(
        model=_model(["AAA"], (_pair(),)),
        closes_by_symbol=_prices(),
        pair_config=PairLayerConfig(
            formation_window=5,
            formation_every=5,
            borrow_rate=0.0,
        ),
    )

    assert set(universe.instruments) == {"AAA", "AAA/BBB"}


def test_universe_builder_allows_overlap_between_naked_and_pair_leg() -> None:
    universe = build_instrument_universe(
        model=_model(["AAA", "BBB"], (_pair(),)),
        closes_by_symbol=_prices(),
        pair_config=PairLayerConfig(
            formation_window=5,
            formation_every=5,
            borrow_rate=0.0,
        ),
    )

    assert universe.instruments["AAA"].kind == "stock"
    assert universe.instruments["AAA/BBB"].kind == "spread"


def test_universe_builder_rejects_missing_pair_leg_prices() -> None:
    with pytest.raises(ValueError, match="Missing price history"):
        build_instrument_universe(
            model=_model(["AAA"], (_pair(),)),
            closes_by_symbol={"AAA": _prices()["AAA"]},
            pair_config=PairLayerConfig(
                formation_window=5,
                formation_every=5,
                borrow_rate=0.0,
            ),
        )


def test_universe_builder_outputs_aligned_nav_matrix() -> None:
    universe = build_instrument_universe(
        model=_model(["AAA"], (_pair(),)),
        closes_by_symbol=_prices(),
        pair_config=PairLayerConfig(
            formation_window=5,
            formation_every=5,
            borrow_rate=0.0,
        ),
    )

    ids, matrix = universe.nav_matrix()

    assert ids == ["AAA", "AAA/BBB"]
    assert matrix.shape == (2, 30)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix > 0.0)
