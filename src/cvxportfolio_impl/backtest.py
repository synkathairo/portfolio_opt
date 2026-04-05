from __future__ import annotations

import itertools
import json

from portfolio_opt.alpaca import AlpacaClient
from portfolio_opt.backtest import run_fixed_weight_benchmark, summarize_return_series
from portfolio_opt.config import AlpacaConfig
from portfolio_opt.model import load_model_inputs

from .data import closes_to_market_data, momentum_forecast
from .policy import build_policy


def clean_value(value: float, tolerance: float = 1e-5) -> float:
    return 0.0 if abs(value) < tolerance else value


def clamp_for_display(
    value: float,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    tolerance: float = 1e-3,
) -> float:
    cleaned = clean_value(float(value))
    if lower_bound is not None and abs(cleaned - lower_bound) < tolerance:
        cleaned = lower_bound
    if upper_bound is not None and abs(cleaned - upper_bound) < tolerance:
        cleaned = upper_bound
    return cleaned


def clean_mapping(values: dict[str, float], tolerance: float = 1e-5) -> dict[str, float]:
    return {key: round(clean_value(float(value), tolerance), 6) for key, value in values.items()}


def clean_constraint_mapping(
    values: dict[str, float],
    *,
    lower_bounds: dict[str, float] | None = None,
    upper_bounds: dict[str, float] | None = None,
    tolerance: float = 1e-3,
) -> dict[str, float]:
    lower_bounds = lower_bounds or {}
    upper_bounds = upper_bounds or {}
    return {
        key: round(
            clamp_for_display(
                float(value),
                lower_bound=lower_bounds.get(key),
                upper_bound=upper_bounds.get(key),
                tolerance=tolerance,
            ),
            6,
        )
        for key, value in values.items()
    }


def prepare_cvxportfolio_context(
    model_path: str,
    lookback_days: int,
    backtest_days: int,
    use_cache: bool = False,
    refresh_cache: bool = False,
    offline: bool = False,
) -> tuple:
    model = load_model_inputs(model_path)
    alpaca = AlpacaClient(AlpacaConfig.from_env())
    warmup_days = max(lookback_days, 252)
    total_days = warmup_days + backtest_days + 1
    closes_by_symbol = alpaca.get_daily_closes_for_period(
        model.symbols,
        total_days,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        offline=offline,
    )
    returns_frame, prices_frame = closes_to_market_data(closes_by_symbol)
    return model, closes_by_symbol, returns_frame, prices_frame, warmup_days


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
    linear_trade_cost: float = 0.0,
    planning_horizon: int = 1,
    use_cache: bool = False,
    refresh_cache: bool = False,
    offline: bool = False,
) -> dict:
    try:
        import cvxportfolio as cvx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "cvxportfolio is required for this path. Run `uv sync` after adding the dependency."
        ) from exc

    model, closes_by_symbol, returns_frame, prices_frame, warmup_days = prepare_cvxportfolio_context(
        model_path=model_path,
        lookback_days=lookback_days,
        backtest_days=backtest_days,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        offline=offline,
    )
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
        planning_horizon=planning_horizon,
    )
    simulator = cvx.MarketSimulator(returns=returns_frame, prices=prices_frame, cash_key="USDOLLAR")
    result = simulator.backtest(policy, start_time=returns_frame.index[warmup_days])
    latest_weights = result.w.iloc[-1].drop(labels=["USDOLLAR"], errors="ignore")
    latest_cash_weight = float(result.w.iloc[-1].get("USDOLLAR", 0.0))
    initial_value = float(result.v.iloc[0])
    final_value = float(result.v.iloc[-1])
    normalized_final_value = final_value / initial_value if initial_value else final_value
    realized_returns = result.v.pct_change().dropna().to_numpy()
    turnover_series = result.turnover.reindex(result.v.index).fillna(0.0).to_numpy()
    # Apply a simple proportional transaction cost ex-post using reported turnover.
    net_realized_returns = realized_returns - linear_trade_cost * turnover_series[1:]
    realized_final_value, realized_total_return, geometric_annualized_return, realized_annualized_volatility, realized_max_drawdown = (
        summarize_return_series(net_realized_returns)
    )
    geometric_sharpe = (
        geometric_annualized_return / realized_annualized_volatility
        if realized_annualized_volatility > 0
        else 0.0
    )
    first_timestamp = str(result.v.index[0])
    last_timestamp = str(result.v.index[-1])
    realized_periods = int(len(realized_returns))
    latest_class_exposures = clean_constraint_mapping({
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
    }, lower_bounds=model.class_min_weights, upper_bounds=model.class_max_weights)
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
            "linear_trade_cost": linear_trade_cost,
            "planning_horizon": planning_horizon,
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
            symbol: round(
                clamp_for_display(
                    max(0.0, float(latest_weights.get(symbol, 0.0))),
                    lower_bound=0.0,
                    upper_bound=max_weight,
                ),
                6,
            )
            for symbol in model.symbols
        },
        "latest_cash_weight": round(clamp_for_display(latest_cash_weight, lower_bound=min_cash_weight, upper_bound=1.0), 6),
        "latest_asset_class_exposures": latest_class_exposures,
        "benchmarks": benchmark_results,
    }


def run_cvxportfolio_sweep(
    model_path: str,
    lookback_days: int,
    backtest_days: int,
    top_n: int,
    linear_trade_cost: float = 0.0,
    planning_horizon: int = 1,
    use_cache: bool = False,
    refresh_cache: bool = False,
    offline: bool = False,
) -> dict:
    try:
        import cvxportfolio as cvx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "cvxportfolio is required for this path. Run `uv sync` after adding the dependency."
        ) from exc

    model, _closes_by_symbol, returns_frame, prices_frame, warmup_days = prepare_cvxportfolio_context(
        model_path=model_path,
        lookback_days=lookback_days,
        backtest_days=backtest_days,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        offline=offline,
    )
    closes_by_symbol = _closes_by_symbol
    risk_grid = [0.5, 1.0, 2.0]
    shrinkage_grid = [0.5, 0.75, 0.9]
    cash_grid = [0.05, 0.10, 0.20]
    invested_grid = [0.20, 0.30, 0.40]
    momentum_grid = [42, 63, 84]
    results: list[dict[str, float | int]] = []
    skipped: list[dict[str, float | int | str]] = []

    for risk_aversion, mean_shrinkage, min_cash_weight, min_invested_weight, momentum_window in itertools.product(
        risk_grid,
        shrinkage_grid,
        cash_grid,
        invested_grid,
        momentum_grid,
    ):
        forecasts = momentum_forecast(
            returns_frame=returns_frame,
            momentum_window=momentum_window,
            mean_shrinkage=mean_shrinkage,
        )
        try:
            policy = build_policy(
                cvx=cvx,
                symbols=model.symbols,
                forecasts=forecasts,
                risk_aversion=risk_aversion,
                max_weight=0.35,
                min_cash_weight=min_cash_weight,
                min_invested_weight=min_invested_weight,
                class_min_weights=model.class_min_weights,
                class_max_weights=model.class_max_weights,
                asset_classes=model.asset_classes,
                planning_horizon=planning_horizon,
            )
            simulator = cvx.MarketSimulator(returns=returns_frame, prices=prices_frame, cash_key="USDOLLAR")
            result = simulator.backtest(policy, start_time=returns_frame.index[warmup_days])
            realized_returns = result.v.pct_change().dropna().to_numpy()
            turnover_series = result.turnover.reindex(result.v.index).fillna(0.0).to_numpy()
            net_realized_returns = realized_returns - linear_trade_cost * turnover_series[1:]
            _, _, annualized_return, annualized_volatility, max_drawdown = summarize_return_series(net_realized_returns)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0
            results.append(
                {
                    "risk_aversion": risk_aversion,
                    "mean_shrinkage": mean_shrinkage,
                    "min_cash_weight": min_cash_weight,
                    "min_invested_weight": min_invested_weight,
                    "momentum_window": momentum_window,
                    "annualized_return": round(float(annualized_return), 6),
                    "annualized_volatility": round(float(annualized_volatility), 6),
                    "max_drawdown": round(float(max_drawdown), 6),
                    "average_turnover": round(float(result.turnover.mean()), 6),
                    "sharpe_ratio": round(float(sharpe_ratio), 6),
                }
            )
        except Exception as exc:  # pragma: no cover
            skipped.append(
                {
                    "risk_aversion": risk_aversion,
                    "mean_shrinkage": mean_shrinkage,
                    "min_cash_weight": min_cash_weight,
                    "min_invested_weight": min_invested_weight,
                    "momentum_window": momentum_window,
                    "reason": str(exc),
                }
            )

    results.sort(
        key=lambda item: (
            float(item["sharpe_ratio"]),
            float(item["annualized_return"]),
        ),
        reverse=True,
    )
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
        "cvxportfolio_sweep": {
            "days": backtest_days,
            "warmup_days": warmup_days,
            "linear_trade_cost": linear_trade_cost,
            "planning_horizon": planning_horizon,
            "top_n": top_n,
            "tested": len(results),
            "skipped": len(skipped),
            "results": results[:top_n],
            "skipped_examples": skipped[: min(5, len(skipped))],
        },
        "benchmarks": benchmark_results,
    }


def format_backtest(result: dict) -> str:
    return json.dumps(result, indent=2)
