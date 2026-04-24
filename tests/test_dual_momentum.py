from __future__ import annotations

from portfolio_opt.backtest import (
    compute_dual_momentum_weights,
    rolling_window_comparison,
    run_dual_momentum_backtest,
    run_factor_momentum_backtest,
)
from portfolio_opt.config import OptimizationConfig


def test_dual_momentum_prefers_top_risky_asset_when_it_beats_cash() -> None:
    closes_by_symbol = {
        "SPY": [100.0, 101.0, 103.0, 106.0, 109.0],
        "QQQ": [100.0, 100.5, 101.0, 101.5, 102.0],
        "SGOV": [100.0, 100.1, 100.2, 100.3, 100.4],
        "IEF": [100.0, 100.0, 99.9, 99.8, 99.7],
    }
    asset_classes = {
        "SPY": "equity_us_large",
        "QQQ": "equity_us_growth",
        "SGOV": "cash_like",
        "IEF": "bond_intermediate",
    }

    result = run_dual_momentum_backtest(
        symbols=["SPY", "QQQ", "SGOV", "IEF"],
        closes_by_symbol=closes_by_symbol,
        asset_classes=asset_classes,
        lookback_days=2,
        rebalance_every=1,
        top_k=1,
        absolute_threshold=0.0,
    )

    assert result.latest_weights.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_dual_momentum_falls_back_to_defensive_assets_when_risk_assets_fail_filter() -> (
    None
):
    closes_by_symbol = {
        "SPY": [100.0, 99.0, 98.0, 97.0, 96.0],
        "QQQ": [100.0, 99.5, 99.0, 98.5, 98.0],
        "SGOV": [100.0, 100.1, 100.2, 100.3, 100.4],
        "IEF": [100.0, 100.2, 100.4, 100.5, 100.7],
    }
    asset_classes = {
        "SPY": "equity_us_large",
        "QQQ": "equity_us_growth",
        "SGOV": "cash_like",
        "IEF": "bond_intermediate",
    }

    result = run_dual_momentum_backtest(
        symbols=["SPY", "QQQ", "SGOV", "IEF"],
        closes_by_symbol=closes_by_symbol,
        asset_classes=asset_classes,
        lookback_days=2,
        rebalance_every=1,
        top_k=1,
        absolute_threshold=0.0,
    )

    assert result.latest_weights.tolist() == [0.0, 0.0, 0.5, 0.5]


def test_rolling_window_comparison_counts_windows_against_spy() -> None:
    closes_by_symbol = {
        "SPY": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
        "QQQ": [100.0, 104.0, 108.0, 112.0, 116.0, 120.0, 124.0],
        "SGOV": [100.0, 100.05, 100.1, 100.15, 100.2, 100.25, 100.3],
    }
    asset_classes = {
        "SPY": "equity_us_large",
        "QQQ": "equity_us_growth",
        "SGOV": "cash_like",
    }

    comparison = rolling_window_comparison(
        strategy="dual-momentum",
        symbols=["SPY", "QQQ", "SGOV"],
        closes_by_symbol=closes_by_symbol,
        asset_classes=asset_classes,
        lookback_days=2,
        window_days=3,
        step_days=1,
        rebalance_every=1,
        return_model="momentum",
        mean_shrinkage=0.75,
        momentum_window=2,
        opt_config=OptimizationConfig(),
        asset_class_matrix=None,
        top_k=1,
        absolute_threshold=0.0,
        weighting="equal",
        softmax_temperature=0.05,
    )

    assert comparison["windows"] == 2
    assert comparison["beat_spy_return_windows"] == 2
    assert comparison["beat_spy_sharpe_windows"] == 2


def test_rolling_window_comparison_rejects_missing_spy() -> None:
    try:
        rolling_window_comparison(
            strategy="dual-momentum",
            symbols=["QQQ", "SGOV"],
            closes_by_symbol={
                "QQQ": [100.0, 101.0, 102.0, 103.0, 104.0],
                "SGOV": [100.0, 100.1, 100.2, 100.3, 100.4],
            },
            asset_classes={"QQQ": "equity", "SGOV": "cash_like"},
            lookback_days=1,
            window_days=2,
            step_days=1,
            rebalance_every=1,
            return_model="momentum",
            mean_shrinkage=0.75,
            momentum_window=1,
            opt_config=OptimizationConfig(),
            asset_class_matrix=None,
            top_k=1,
            absolute_threshold=0.0,
            weighting="equal",
            softmax_temperature=0.05,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("rolling comparison should fail without SPY")

    assert "requires SPY" in message


def test_dual_momentum_score_weighting_tilts_toward_stronger_asset() -> None:
    closes_by_symbol = {
        "SPY": [100.0, 102.0, 106.0, 112.0, 120.0],
        "QQQ": [100.0, 101.0, 103.0, 106.0, 110.0],
        "SGOV": [100.0, 100.1, 100.2, 100.3, 100.4],
    }
    asset_classes = {
        "SPY": "equity_us_large",
        "QQQ": "equity_us_growth",
        "SGOV": "cash_like",
    }

    result = run_dual_momentum_backtest(
        symbols=["SPY", "QQQ", "SGOV"],
        closes_by_symbol=closes_by_symbol,
        asset_classes=asset_classes,
        lookback_days=2,
        rebalance_every=1,
        top_k=2,
        absolute_threshold=0.0,
        weighting="score",
    )

    assert result.latest_weights[0] > result.latest_weights[1]
    assert result.latest_weights[2] == 0.0
    assert round(float(result.latest_weights.sum()), 6) == 1.0


def test_dual_momentum_inverse_vol_weighting_tilts_toward_lower_vol_asset() -> None:
    closes_by_symbol = {
        "SPY": [100.0, 110.0, 90.0, 120.0, 95.0],
        "QQQ": [100.0, 102.0, 104.0, 106.0, 108.0],
        "SGOV": [100.0, 100.1, 100.2, 100.3, 100.4],
    }
    asset_classes = {
        "SPY": "equity_us_large",
        "QQQ": "equity_us_growth",
        "SGOV": "cash_like",
    }

    result = run_dual_momentum_backtest(
        symbols=["SPY", "QQQ", "SGOV"],
        closes_by_symbol=closes_by_symbol,
        asset_classes=asset_classes,
        lookback_days=2,
        rebalance_every=1,
        top_k=2,
        absolute_threshold=0.0,
        weighting="inverse-vol",
    )

    assert result.latest_weights[1] > result.latest_weights[0]
    assert result.latest_weights[2] == 0.0
    assert round(float(result.latest_weights.sum()), 6) == 1.0


def test_dual_momentum_max_single_weight_redistributes_to_available_capacity() -> None:
    weights = compute_dual_momentum_weights(
        symbols=["A", "B", "C", "D", "SGOV"],
        closes_by_symbol={
            "A": [100.0, 140.0],
            "B": [100.0, 129.0],
            "C": [100.0, 120.0],
            "D": [100.0, 101.0],
            "SGOV": [100.0, 100.0],
        },
        asset_classes={
            "A": "equity",
            "B": "equity",
            "C": "equity",
            "D": "equity",
            "SGOV": "cash_like",
        },
        lookback_days=1,
        top_k=4,
        weighting="score",
        max_single_weight=0.3,
    )

    assert max(weights.values()) <= 0.3
    assert round(sum(weights.values()), 6) == 1.0


def test_dual_momentum_trailing_stop_peak_starts_at_entry() -> None:
    result = run_dual_momentum_backtest(
        symbols=["A", "B", "SGOV"],
        closes_by_symbol={
            "A": [100.0, 120.0, 140.0, 130.0, 125.0, 125.0],
            "B": [100.0, 100.0, 25.0, 50.0, 75.0, 100.0],
            "SGOV": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
        asset_classes={
            "A": "equity_a",
            "B": "equity_b",
            "SGOV": "cash_like",
        },
        lookback_days=2,
        rebalance_every=2,
        top_k=1,
        absolute_threshold=0.0,
        trailing_stop=0.1,
    )

    assert result.latest_weights.tolist() == [0.0, 1.0, 0.0]


def test_factor_momentum_selects_names_inside_top_factor_sleeve() -> None:
    closes_by_symbol = {
        "A_FAST": [100.0, 105.0, 111.0, 118.0, 126.0],
        "A_SLOW": [100.0, 101.0, 102.0, 103.0, 104.0],
        "B_FAST": [100.0, 110.0, 120.0, 130.0, 140.0],
        "B_BAD": [100.0, 80.0, 70.0, 65.0, 60.0],
        "SGOV": [100.0, 100.1, 100.2, 100.3, 100.4],
    }
    asset_classes = {
        "A_FAST": "Alpha Corp (factor_a)",
        "A_SLOW": "Alpha Slow (factor_a)",
        "B_FAST": "Beta Fast (factor_b)",
        "B_BAD": "Beta Bad (factor_b)",
        "SGOV": "cash_like",
    }

    result = run_factor_momentum_backtest(
        symbols=["A_FAST", "A_SLOW", "B_FAST", "B_BAD", "SGOV"],
        closes_by_symbol=closes_by_symbol,
        asset_classes=asset_classes,
        lookback_days=2,
        rebalance_every=1,
        top_k=1,
        factor_top_k=1,
        absolute_threshold=0.0,
    )

    assert result.latest_weights.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
