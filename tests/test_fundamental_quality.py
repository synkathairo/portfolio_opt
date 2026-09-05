from __future__ import annotations

from datetime import date

from portfolio_opt.fundamental_quality import build_snapshot, rank_quality_scores


def _payload(net_income: int, revenue: int, filed: str = "2023-02-01") -> dict:
    tags = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": revenue,
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


def test_build_snapshot_uses_only_facts_filed_by_as_of() -> None:
    payload = _payload(120, 1000, filed="2023-02-01")

    before = build_snapshot("AAA", payload, as_of=date(2023, 1, 31))
    after = build_snapshot("AAA", payload, as_of=date(2023, 2, 2))

    assert before is None
    assert after is not None
    assert after.metrics()["return_on_assets"] == 0.12
    assert after.metrics()["liability_ratio"] == 0.6


def test_quality_ranking_rewards_profitability_and_lower_leverage() -> None:
    first = build_snapshot("AAA", _payload(200, 1000), as_of=date(2023, 3, 1))
    second = build_snapshot("BBB", _payload(50, 1000), as_of=date(2023, 3, 1))
    assert first is not None and second is not None

    scores = rank_quality_scores([first, second], groups={"AAA": "tech", "BBB": "tech"})

    assert scores["AAA"] > scores["BBB"]
    assert 0.0 <= scores["AAA"] <= 1.0
