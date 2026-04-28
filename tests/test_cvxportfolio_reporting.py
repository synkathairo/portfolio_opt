from argparse import Namespace
import json

import pandas as pd

from cvxportfolio_impl import cli as cvx_cli
from cvxportfolio_impl import backtest as cvx_backtest
from cvxportfolio_impl.backtest import clamp_for_display, clean_constraint_mapping
from portfolio_opt.market_data import CloseHistory
from portfolio_opt.model import ModelInputs


def test_clamp_for_display_snaps_near_upper_bound() -> None:
    assert clamp_for_display(0.600828, upper_bound=0.6) == 0.6


def test_clamp_for_display_snaps_near_lower_bound() -> None:
    assert clamp_for_display(-0.000001, lower_bound=0.0) == 0.0


def test_clean_constraint_mapping_respects_declared_bounds() -> None:
    cleaned = clean_constraint_mapping(
        {"equity": 0.600828, "commodity": -0.0, "bond": 0.298828},
        lower_bounds={"commodity": 0.0},
        upper_bounds={"equity": 0.6, "bond": 0.4},
    )
    assert cleaned == {"equity": 0.6, "commodity": 0.0, "bond": 0.298828}


def test_cvxportfolio_cli_remains_compatibility_entrypoint(
    monkeypatch,
    capsys,
) -> None:
    args = Namespace(model="dummy.json")

    monkeypatch.setattr(cvx_cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        cvx_cli,
        "run_from_args",
        lambda received_args: {
            "symbols": [received_args.model],
            "cvxportfolio_backtest": {"annualized_return": 0.1},
        },
    )

    cvx_cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["symbols"] == ["dummy.json"]
    assert payload["cvxportfolio_backtest"]["annualized_return"] == 0.1


def test_cvxportfolio_context_uses_shared_close_loader(monkeypatch) -> None:
    captured: dict[str, object] = {}
    returns_frame = pd.DataFrame({"SPY": [0.01, 0.02], "USDOLLAR": [0.0, 0.0]})
    prices_frame = pd.DataFrame({"SPY": [100.0, 101.0]})

    monkeypatch.setattr(
        cvx_backtest,
        "load_model_inputs",
        lambda _path: ModelInputs(
            symbols=["SPY"],
            expected_returns=None,
            covariance=None,
            asset_classes={"SPY": "equity"},
            class_min_weights={},
            class_max_weights={},
        ),
    )

    def fake_load_close_history(**kwargs):
        captured["data_source"] = kwargs["data_source"]
        captured["total_days"] = kwargs["total_days"]
        return CloseHistory(
            closes_by_symbol={"SPY": [100.0, 101.0, 103.0]},
            benchmark_closes_by_symbol={},
            benchmark_symbols_universe=["SPY"],
        )

    def fake_closes_to_market_data(closes_by_symbol):
        captured["closes"] = closes_by_symbol
        return returns_frame, prices_frame

    monkeypatch.setattr(cvx_backtest, "load_close_history", fake_load_close_history)
    monkeypatch.setattr(
        cvx_backtest, "closes_to_market_data", fake_closes_to_market_data
    )

    model, closes, returns, prices, warmup_days = (
        cvx_backtest.prepare_cvxportfolio_context(
            model_path="dummy.json",
            lookback_days=2,
            backtest_days=3,
            data_source="yfinance",
            offline=True,
        )
    )

    assert model.symbols == ["SPY"]
    assert closes == {"SPY": [100.0, 101.0, 103.0]}
    assert returns is returns_frame
    assert prices is prices_frame
    assert warmup_days == 252
    assert captured == {
        "data_source": "yfinance",
        "total_days": 256,
        "closes": {"SPY": [100.0, 101.0, 103.0]},
    }
