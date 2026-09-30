from __future__ import annotations

from datetime import date, timedelta

import pytest

from portfolio_opt.quality_backtest import run_quality_backtest


def _facts(net_income: int, filed: str) -> dict:
    tags = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": 1000,
        "NetIncomeLoss": net_income,
        "Assets": 1000,
        "StockholdersEquity": 400,
        "Liabilities": 600,
        "NetCashProvidedByUsedInOperatingActivities": net_income + 20,
    }
    return {
        "facts": {
            "us-gaap": {
                tag: {
                    "units": {
                        "USD": [
                            {
                                "val": value,
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": filed,
                                "form": "10-K",
                                "fy": 2022,
                                "fp": "FY",
                            }
                        ]
                    }
                }
                for tag, value in tags.items()
            }
        }
    }


def test_quality_backtest_does_not_use_future_filing() -> None:
    dates = [date(2023, 1, 1) + timedelta(days=index) for index in range(4)]
    result = run_quality_backtest(
        symbols=["AAA", "BBB"],
        closes_by_symbol={"AAA": [100, 110, 120, 130], "BBB": [100, 100, 100, 100]},
        trading_dates=dates,
        companyfacts_by_symbol={
            "AAA": _facts(200, "2023-01-03"),
            "BBB": _facts(50, "2023-01-03"),
        },
        start_day=0,
        rebalance_every=10,
        top_k=1,
        linear_trade_cost=0.001,
    )

    # No position exists before the filing date; the position starts next day.
    assert result.rebalance_count == 1
    assert result.final_value == pytest.approx(1.0)


def test_quality_backtest_applies_selected_position_after_filing() -> None:
    dates = [date(2023, 1, 1) + timedelta(days=index) for index in range(5)]
    result = run_quality_backtest(
        symbols=["AAA", "BBB"],
        closes_by_symbol={"AAA": [100, 100, 100, 110, 121], "BBB": [100] * 5},
        trading_dates=dates,
        companyfacts_by_symbol={
            "AAA": _facts(200, "2023-01-01"),
            "BBB": _facts(50, "2023-01-01"),
        },
        start_day=0,
        rebalance_every=10,
        top_k=1,
    )

    assert result.final_value == pytest.approx(1.21)


def test_quality_backtest_weights_drift_between_rebalances() -> None:
    dates = [date(2023, 1, 1) + timedelta(days=index) for index in range(3)]
    result = run_quality_backtest(
        symbols=["AAA", "BBB"],
        closes_by_symbol={"AAA": [100, 200, 400], "BBB": [100, 100, 100]},
        trading_dates=dates,
        companyfacts_by_symbol={
            "AAA": _facts(200, "2023-01-01"),
            "BBB": _facts(50, "2023-01-01"),
        },
        rebalance_every=10,
        top_k=2,
    )

    assert result.final_value == pytest.approx(2.5)
    assert result.latest_weights.tolist() == pytest.approx([0.8, 0.2])
