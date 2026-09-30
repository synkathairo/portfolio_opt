from __future__ import annotations

from portfolio_opt.regime_adaptive import (
    compute_trend_filtered_mean_reversion_weights,
    run_regime_adaptive_backtest,
)


def test_mean_reversion_selects_short_term_laggard_above_trend() -> None:
    weights = compute_trend_filtered_mean_reversion_weights(
        symbols=["A", "B", "SGOV"],
        closes_by_symbol={
            "A": [100.0, 110.0, 120.0, 118.0],
            "B": [100.0, 105.0, 110.0, 108.0],
            "SGOV": [100.0, 100.1, 100.2, 100.3],
        },
        asset_classes={"A": "equity", "B": "equity", "SGOV": "cash_like"},
        trend_window=3,
        mean_reversion_window=1,
        top_k=1,
    )

    assert weights == {"A": 0.0, "B": 1.0, "SGOV": 0.0}


def test_mean_reversion_holds_cash_without_defensive_asset() -> None:
    weights = compute_trend_filtered_mean_reversion_weights(
        symbols=["A", "B"],
        closes_by_symbol={"A": [100.0, 95.0, 90.0], "B": [100.0, 98.0, 96.0]},
        asset_classes={"A": "equity", "B": "equity"},
        trend_window=2,
        mean_reversion_window=1,
        top_k=1,
    )

    assert weights == {"A": 0.0, "B": 0.0}


def test_regime_adaptive_backtest_applies_costs_and_returns_valid_weights() -> None:
    result = run_regime_adaptive_backtest(
        symbols=["A", "B", "SGOV"],
        closes_by_symbol={
            "A": [100.0, 104.0, 108.0, 106.0, 110.0, 114.0, 112.0, 116.0],
            "B": [100.0, 101.0, 102.0, 105.0, 108.0, 106.0, 110.0, 113.0],
            "SGOV": [100.0, 100.01, 100.02, 100.03, 100.04, 100.05, 100.06, 100.07],
        },
        asset_classes={"A": "equity", "B": "equity", "SGOV": "cash_like"},
        lookback_days=2,
        rebalance_every=1,
        top_k=1,
        absolute_threshold=0.0,
        selection_window=2,
        mean_reversion_window=1,
        linear_trade_cost=0.001,
    )

    assert result.rebalance_count > 0
    assert result.average_turnover >= 0.0
    assert round(float(result.latest_weights.sum()), 8) == 1.0
