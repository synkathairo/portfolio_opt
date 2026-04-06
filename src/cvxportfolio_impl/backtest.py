from __future__ import annotations

import json

from portfolio_opt.alpaca import AlpacaClient
from portfolio_opt.backtest import run_fixed_weight_benchmark, summarize_return_series
from portfolio_opt.config import AlpacaConfig, OptimizationConfig
from portfolio_opt.model import load_model_inputs

from .data import closes_to_market_data, momentum_forecast
from .policy import build_policy


def run_cvxportfolio_backtest(
    model_path: str,
    lookback_days: int,
    backtest_days: int,
    risk_aversion: float,
    min_cash_weight: float,
    min_invested_weight: float,
    max_weight: float,
    mean_shrinkage: float,
    momentum_window: int,
) -> dict:
    try:
        import cvxportfolio as cvx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "cvxportfolio is required for this path. Run `uv sync` after adding the dependency."
        ) from exc

    model = load_model_inputs(model_path)
    alpaca = AlpacaClient(AlpacaConfig.from_env())
    # cvxportfolio's built-in covariance forecaster needs its own history
    # window. Use an explicit warmup so the realized backtest horizon matches
    # the requested backtest_days instead of silently consuming test periods.
    warmup_days = max(lookback_days, 252)
    total_days = warmup_days + backtest_days + 1
    closes_by_symbol = alpaca.get_daily_closes_for_period(model.symbols, total_days)
    returns_frame, prices_frame = closes_to_market_data(closes_by_symbol)
    forecasts = momentum_forecast(
        returns_frame=returns_frame,
        momentum_window=momentum_window,
        mean_shrinkage=mean_shrinkage,
    )

    policy = build_policy(
        cvx=cvx,
        symbols=model.symbols,
        forecasts=forecasts,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
        min_cash_weight=min_cash_weight,
        min_invested_weight=min_invested_weight,
        class_min_weights=model.class_min_weights,
        class_max_weights=model.class_max_weights,
        asset_classes=model.asset_classes,
    )
    simulator = cvx.MarketSimulator(returns=returns_frame, prices=prices_frame, cash_key="USDOLLAR")
    result = simulator.backtest(policy, start_time=returns_frame.index[warmup_days])
    latest_weights = result.w.iloc[-1].drop(labels=["USDOLLAR"], errors="ignore")
    latest_cash_weight = float(result.w.iloc[-1].get("USDOLLAR", 0.0))
    initial_value = float(result.v.iloc[0])
    final_value = float(result.v.iloc[-1])
    normalized_final_value = final_value / initial_value if initial_value else final_value
    realized_returns = result.v.pct_change().dropna().to_numpy()
    realized_final_value, realized_total_return, geometric_annualized_return, realized_annualized_volatility, realized_max_drawdown = (
        summarize_return_series(realized_returns)
    )
    geometric_sharpe = (
        geometric_annualized_return / realized_annualized_volatility
        if realized_annualized_volatility > 0
        else 0.0
    )
    first_timestamp = str(result.v.index[0])
    last_timestamp = str(result.v.index[-1])
    realized_periods = int(len(realized_returns))
    latest_class_exposures = {
        class_name: round(
            float(
                sum(
                    latest_weights.get(symbol, 0.0)
                    for symbol in model.symbols
                    if model.asset_classes.get(symbol) == class_name
                )
            ),
            6,
        )
        for class_name in sorted(set(model.asset_classes.values()))
    }
    benchmark_results = {
        "spy": run_fixed_weight_benchmark(
            symbols=model.symbols,
            closes_by_symbol=closes_by_symbol,
            weights_by_symbol={"SPY": 1.0},
            start_day=lookback_days,
        ),
        "sixty_forty_spy_tlt": run_fixed_weight_benchmark(
            symbols=model.symbols,
            closes_by_symbol=closes_by_symbol,
            weights_by_symbol={"SPY": 0.6, "TLT": 0.4},
            start_day=lookback_days,
        ),
        "equal_weight": run_fixed_weight_benchmark(
            symbols=model.symbols,
            closes_by_symbol=closes_by_symbol,
            weights_by_symbol={symbol: 1.0 / len(model.symbols) for symbol in model.symbols},
            start_day=lookback_days,
        ),
        "half_spy_half_cash": run_fixed_weight_benchmark(
            symbols=model.symbols,
            closes_by_symbol=closes_by_symbol,
            weights_by_symbol={"SPY": 0.5},
            start_day=lookback_days,
        ),
    }

    return {
        "symbols": model.symbols,
        "cvxportfolio_backtest": {
            "days": backtest_days,
            "warmup_days": warmup_days,
            "risk_aversion": risk_aversion,
            "mean_shrinkage": mean_shrinkage,
            "momentum_window": momentum_window,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "realized_periods": realized_periods,
            "initial_value": round(initial_value, 6),
            "final_value": round(final_value, 6),
            "normalized_final_value": round(normalized_final_value, 6),
            "total_return": round(realized_total_return, 6),
            "value_ratio_total_return": round(normalized_final_value - 1.0, 6),
            "realized_return_series_final_value": round(realized_final_value, 6),
            "annualized_return": round(float(geometric_annualized_return), 6),
            "annualized_volatility": round(float(realized_annualized_volatility), 6),
            "max_drawdown": round(float(realized_max_drawdown), 6),
            "average_turnover": round(float(result.turnover.mean()), 6),
            "sharpe_ratio": round(float(geometric_sharpe), 6),
            "cvxportfolio_annualized_average_return": round(float(result.annualized_average_return), 6),
        },
        "latest_target_weights": {
            symbol: round(max(0.0, float(latest_weights.get(symbol, 0.0))), 6)
            for symbol in model.symbols
        },
        "latest_cash_weight": round(latest_cash_weight, 6),
        "latest_asset_class_exposures": latest_class_exposures,
        "benchmarks": benchmark_results,
    }


def format_backtest(result: dict) -> str:
    return json.dumps(result, indent=2)
