from __future__ import annotations

from datetime import date, timedelta

import pytest

from portfolio_opt.residual_reversion import run_residual_reversion_backtest


def test_residual_reversion_uses_only_prices_available_at_rebalance() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(8)]
    result = run_residual_reversion_backtest(
        symbols=["AAA", "BBB", "CCC"],
        closes_by_symbol={
            "AAA": [100, 100, 100, 100, 90, 90, 100, 100],
            "BBB": [100, 102, 101, 104, 104, 104, 104, 104],
            "CCC": [100, 98, 100, 99, 99, 99, 99, 99],
        },
        trading_dates=dates,
        start_day=4,
        rebalance_every=99,
        lookback_days=3,
        volatility_window=3,
        top_k=1,
        max_single_weight=1.0,
    )

    # AAA is selected after its observed selloff and benefits from its rebound.
    assert result.final_value == pytest.approx(100 / 90)


def test_residual_reversion_rejects_infeasible_weight_cap() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(5)]
    with pytest.raises(ValueError, match="max_single_weight"):
        run_residual_reversion_backtest(
            symbols=["AAA", "BBB"],
            closes_by_symbol={"AAA": [100] * 5, "BBB": [100] * 5},
            trading_dates=dates,
            start_day=3,
            lookback_days=2,
            top_k=2,
            max_single_weight=0.4,
        )


def test_residual_reversion_caps_rebalance_turnover() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(8)]
    result = run_residual_reversion_backtest(
        symbols=["AAA", "BBB"],
        closes_by_symbol={
            "AAA": [100, 100, 100, 90, 90, 100, 100, 100],
            "BBB": [100] * 8,
        },
        trading_dates=dates,
        start_day=4,
        rebalance_every=1,
        lookback_days=3,
        volatility_window=3,
        top_k=1,
        max_single_weight=1.0,
        max_turnover=0.10,
    )

    assert result.average_turnover <= 0.10 + 1e-12
