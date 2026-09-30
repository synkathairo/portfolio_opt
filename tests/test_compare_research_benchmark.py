from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from runpy import run_path

import pytest

benchmark = run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/compare_research_benchmark.py")
)
STRATEGIES = benchmark["STRATEGIES"]
compare_report = benchmark["compare_report"]


def test_comparison_rebases_every_strategy_at_same_warmup_date() -> None:
    dates = [date(2022, 1, 1) + timedelta(days=index) for index in range(66)]
    report = {
        "dates": [day.isoformat() for day in dates],
        "symbols": ["AAA", "BBB"],
        "kind": "test",
        "survivorship_warning": "test universe",
        "linear_trade_cost": 0.001,
        "max_turnover": 0.5,
        "dual_momentum_config": {"lookback_days": 2},
        "regime_adaptive_config": {"lookback_days": 2},
        "quality_momentum_config": {"lookback_days": 2},
        "value_quality_trend_config": {"trend_window": 2},
    }
    for key in STRATEGIES:
        base = 10.0 if key == "strategy" else 1.0
        report[f"{key}_values"] = [base] * 64 + [base * 1.1, base * 1.21]

    comparison = compare_report(report)
    full = comparison["windows"]["full"]

    assert full["start"] == dates[63].isoformat()
    assert full["trading_days"] == 2
    for key in STRATEGIES:
        assert full["strategies"][key]["total_return"] == pytest.approx(0.21)
