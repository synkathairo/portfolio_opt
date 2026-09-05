from __future__ import annotations

from datetime import date

import pytest

from portfolio_opt.sec_edgar import (
    SecEdgarClient,
    iter_facts,
    latest_facts_as_of,
    normalize_cik,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _Session:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return _Response(self.responses.pop(0))


def _facts_payload() -> dict:
    return {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "val": 100,
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-02-01",
                                "form": "10-K",
                                "fy": 2022,
                                "fp": "FY",
                                "accn": "old",
                            },
                            {
                                "val": 110,
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "filed": "2023-04-01",
                                "form": "10-K/A",
                                "fy": 2022,
                                "fp": "FY",
                            },
                            {
                                "val": 120,
                                "start": "2023-01-01",
                                "end": "2023-03-31",
                                "filed": "2023-05-01",
                                "form": "10-Q",
                                "fy": 2023,
                                "fp": "Q1",
                                "accn": "new",
                            },
                        ]
                    }
                }
            }
        }
    }


def test_normalize_cik() -> None:
    assert normalize_cik(320193) == "0000320193"
    assert normalize_cik("CIK0000320193") == "0000320193"
    with pytest.raises(ValueError, match="Invalid CIK"):
        normalize_cik("AAPL")


def test_client_caches_ticker_and_companyfacts_responses(tmp_path) -> None:
    session = _Session(
        [
            {"0": {"ticker": "AAPL", "cik_str": 320193}},
            _facts_payload(),
        ]
    )
    client = SecEdgarClient(
        user_agent="portfolio-opt tests test@example.com",
        cache_dir=tmp_path,
        session=session,
    )

    first = client.company_facts_for_symbol("aapl")
    second = client.company_facts_for_symbol("AAPL", offline=True)

    assert first == second
    assert len(session.calls) == 2
    assert session.calls[0]["headers"] == {
        "User-Agent": "portfolio-opt tests test@example.com",
        "Accept-Encoding": "gzip, deflate",
    }
    assert session.calls[1]["url"].endswith("CIK0000320193.json")


def test_client_requires_compliant_minimum_interval() -> None:
    with pytest.raises(ValueError, match="at least 0.1"):
        SecEdgarClient(
            user_agent="portfolio-opt tests test@example.com",
            min_request_interval_seconds=0.05,
        )


def test_iter_facts_keeps_only_supported_forms_and_valid_dated_values() -> None:
    observations = iter_facts(_facts_payload(), tag="NetIncomeLoss")

    assert [(item.value, item.form, item.filed) for item in observations] == [
        (100.0, "10-K", date(2023, 2, 1)),
        (120.0, "10-Q", date(2023, 5, 1)),
    ]


def test_latest_facts_as_of_never_uses_future_filing() -> None:
    observations = iter_facts(_facts_payload(), tag="NetIncomeLoss")

    before_q1 = latest_facts_as_of(observations, date(2023, 4, 15))
    after_q1 = latest_facts_as_of(observations, date(2023, 5, 2))

    assert before_q1[date(2022, 12, 31)].value == 100.0
    assert date(2023, 3, 31) not in before_q1
    assert after_q1[date(2023, 3, 31)].value == 120.0
