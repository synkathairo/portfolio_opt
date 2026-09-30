from __future__ import annotations

from datetime import date, timedelta

import pytest

from portfolio_opt.value_quality_trend import run_value_quality_trend_backtest


def _facts(net_income: int, shares: int) -> dict:
    tags = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": 1000,
        "NetIncomeLoss": net_income,
        "Assets": 1000,
        "StockholdersEquity": 400,
        "Liabilities": 600,
        "NetCashProvidedByUsedInOperatingActivities": net_income + 20,
    }
    annual = {
        tag: {
            "units": {
                "USD": [
                    {
                        "val": value,
                        "start": "2022-01-01",
                        "end": "2022-12-31",
                        "filed": "2023-01-01",
                        "form": "10-K",
                        "fy": 2022,
                        "fp": "FY",
                    }
                ]
            }
        }
        for tag, value in tags.items()
    }
    return {
        "facts": {
            "us-gaap": annual,
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {
                                "val": shares,
                                "end": "2022-12-31",
                                "filed": "2023-01-01",
                                "form": "10-K",
                                "fy": 2022,
                                "fp": "FY",
                            }
                        ]
                    }
                }
            },
        }
    }


def test_value_quality_trend_uses_filed_shares_and_fundamentals() -> None:
    dates = [date(2023, 1, 1) + timedelta(days=index) for index in range(8)]
    result = run_value_quality_trend_backtest(
        symbols=["AAA", "BBB"],
        closes_by_symbol={
            "AAA": [100, 100, 100, 100, 110, 121, 133, 146],
            "BBB": [100] * 8,
        },
        market_closes_by_symbol={
            "AAA": [100, 100, 100, 100, 110, 121, 133, 146],
            "BBB": [100] * 8,
        },
        trading_dates=dates,
        companyfacts_by_symbol={"AAA": _facts(200, 100), "BBB": _facts(50, 100)},
        groups={"AAA": "technology", "BBB": "technology"},
        trend_window=3,
        rebalance_every=10,
        top_k=1,
        max_per_group=1,
        linear_trade_cost=0.001,
    )

    expected = (1.1 - 0.001) * (121 / 110) * (133 / 121) * (146 / 133)
    assert result.rebalance_count == 1
    assert result.final_value == pytest.approx(expected)


def test_value_ranking_uses_unadjusted_market_price() -> None:
    dates = [date(2023, 1, 1) + timedelta(days=index) for index in range(3)]
    result = run_value_quality_trend_backtest(
        symbols=["AAA", "BBB"],
        closes_by_symbol={"AAA": [100, 100, 100], "BBB": [100, 100, 110]},
        market_closes_by_symbol={"AAA": [200, 200, 200], "BBB": [100, 100, 110]},
        trading_dates=dates,
        companyfacts_by_symbol={"AAA": _facts(100, 100), "BBB": _facts(100, 100)},
        groups={"AAA": "technology", "BBB": "technology"},
        trend_window=1,
        rebalance_every=10,
        top_k=1,
        quality_weight=0.0,
        value_weight=1.0,
        trend_weight=0.0,
    )

    assert result.final_value == pytest.approx(1.1)
