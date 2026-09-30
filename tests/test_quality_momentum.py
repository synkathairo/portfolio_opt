from __future__ import annotations

from datetime import date, timedelta

import pytest

from portfolio_opt.quality_momentum import run_quality_momentum_backtest


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


def test_quality_momentum_uses_point_in_time_quality_and_prices() -> None:
    dates = [date(2023, 1, 1) + timedelta(days=index) for index in range(8)]
    result = run_quality_momentum_backtest(
        symbols=["AAA", "BBB"],
        closes_by_symbol={
            "AAA": [100, 100, 100, 100, 110, 121, 133, 146],
            "BBB": [100] * 8,
        },
        trading_dates=dates,
        companyfacts_by_symbol={
            "AAA": _facts(200, "2023-01-01"),
            "BBB": _facts(50, "2023-01-01"),
        },
        lookback_days=3,
        rebalance_every=10,
        top_k=1,
        quality_weight=0.5,
        linear_trade_cost=0.001,
    )

    assert result.rebalance_count == 1
    expected = (1.1 - 0.001) * (121 / 110) * (133 / 121) * (146 / 133)
    assert result.final_value == pytest.approx(expected)
