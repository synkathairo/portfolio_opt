from __future__ import annotations

import pytest

from portfolio_opt.fixed_allocation import (
    parse_fixed_weights,
    run_fixed_allocation_backtest,
)


def test_parse_fixed_weights_accepts_complete_allocation() -> None:
    assert parse_fixed_weights(
        ["SPY=0.6", "QQQ=0.4"], allowed_symbols=["SPY", "QQQ", "BIL"]
    ) == {"SPY": 0.6, "QQQ": 0.4}


def test_parse_fixed_weights_rejects_incomplete_allocation() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        parse_fixed_weights(["SPY=0.6"], allowed_symbols=["SPY"])


def test_fixed_allocation_drifts_and_rebalances_with_costs() -> None:
    result = run_fixed_allocation_backtest(
        symbols=["SPY", "QQQ"],
        closes_by_symbol={
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
            "QQQ": [100.0, 102.0, 104.0, 106.0, 108.0],
        },
        weights_by_symbol={"SPY": 0.6, "QQQ": 0.4},
        start_day=0,
        rebalance_every=2,
        linear_trade_cost=0.001,
    )

    assert result.rebalance_count == 2
    assert result.average_turnover > 0.5
    assert result.final_value > 1.0
    assert round(float(result.latest_weights.sum()), 8) == 1.0
