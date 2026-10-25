from __future__ import annotations

import pytest

from portfolio_opt.instruments import Instrument, InstrumentUniverse, Leg, SpreadSpec


def _spread() -> SpreadSpec:
    return SpreadSpec(
        id="AAA/BBB",
        legs=(Leg("AAA", 1.0), Leg("BBB", -1.0)),
        category="test",
        asset_class="sector_test",
    )


def test_spread_spec_accepts_signed_legs() -> None:
    spread = _spread()

    assert spread.legs[0].weight == 1.0
    assert spread.legs[1].weight == -1.0
    assert spread.hedge == "beta"


def test_spread_spec_rejects_duplicate_legs() -> None:
    with pytest.raises(ValueError, match="unique"):
        SpreadSpec(
            id="AAA/AAA",
            legs=(Leg("AAA", 1.0), Leg("AAA", -1.0)),
            category="test",
            asset_class="sector_test",
        )


def test_spread_spec_requires_both_signs() -> None:
    with pytest.raises(ValueError, match="negative"):
        SpreadSpec(
            id="AAA/BBB",
            legs=(Leg("AAA", 1.0), Leg("BBB", 1.0)),
            category="test",
            asset_class="sector_test",
        )


def test_instrument_universe_builds_nav_matrix() -> None:
    universe = InstrumentUniverse(
        instruments={
            "AAA": Instrument(
                id="AAA",
                kind="stock",
                nav=(1.0, 1.1),
                asset_class="sector_test",
                category="sector_test",
                available=(True, True),
            ),
            "AAA/BBB": Instrument(
                id="AAA/BBB",
                kind="spread",
                nav=(1.0, 1.05),
                asset_class="sector_test",
                category="test",
                available=(True, True),
                legs=_spread().legs,
                shortable=True,
            ),
        }
    )

    ids, matrix = universe.nav_matrix()

    assert ids == ["AAA", "AAA/BBB"]
    assert matrix.shape == (2, 2)
    assert universe.asset_classes() == {
        "AAA": "sector_test",
        "AAA/BBB": "sector_test",
    }
