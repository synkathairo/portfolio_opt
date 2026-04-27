from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from portfolio_opt.alpaca_interface import AlpacaClient
from portfolio_opt.backtest import (
    TRADING_DAYS_PER_YEAR,
    run_fixed_weight_benchmark,
    summarize_return_series,
)
from portfolio_opt.backtest import run_backtest as run_custom_backtest
from portfolio_opt.config import AlpacaConfig, OptimizationConfig
from portfolio_opt.model import load_model_inputs

from .data import bars_to_market_data, momentum_forecast
from .policy import build_policy


def _run_single_sweep(params: tuple) -> dict[str, float | int]:
    """Run one grid point of the parameter sweep. Module-level for pickling."""
    (
        risk_aversion,
        mean_shrinkage,
        min_cash_weight,
        min_invested_weight,
        momentum_window,
        symbols,
        returns_array,
        prices_array,
        warmup_days,
        class_min_weights,
        class_max_weights,
        asset_classes,
        core_symbol,
        core_weight,
        target_volatility,
        max_leverage,
        benchmark_df,
        linear_trade_cost,
        planning_horizon,
        trading_days_per_year,
    ) = params

    try:
        import cvxportfolio as cvx
    except ImportError as exc:
        raise RuntimeError("cvxportfolio is not installed") from exc

    forecasts = momentum_forecast(
        returns_frame=returns_array,
        momentum_window=momentum_window,
        mean_shrinkage=mean_shrinkage,
    )
    policy = build_policy(
        cvx=cvx,
        symbols=symbols,
        forecasts=forecasts,
        risk_aversion=risk_aversion,
        max_weight=0.35,
        min_cash_weight=min_cash_weight,
        min_invested_weight=min_invested_weight,
        class_min_weights=class_min_weights,
        class_max_weights=class_max_weights,
        asset_classes=asset_classes,
        core_symbol=core_symbol,
        core_weight=core_weight,
        target_volatility=target_volatility,
        max_leverage=max_leverage,
        benchmark=benchmark_df,
        planning_horizon=planning_horizon,
    )
    simulator = cvx.MarketSimulator(
        returns=returns_array, prices=prices_array, cash_key="USDOLLAR"
    )
    result = simulator.backtest(policy, start_time=returns_array.index[warmup_days])
    realized_returns = result.v.pct_change().dropna().to_numpy()
    turnover_series = result.turnover.reindex(result.v.index).fillna(0.0).to_numpy()
    net_realized_returns = realized_returns - linear_trade_cost * turnover_series[1:]
    summary = summarize_return_series(
        net_realized_returns,
        trading_days_per_year=trading_days_per_year,
    )
    sharpe_ratio = (
        summary.annualized_return / summary.annualized_volatility
        if summary.annualized_volatility > 0
        else 0.0
    )
    return {
        "risk_aversion": risk_aversion,
        "mean_shrinkage": mean_shrinkage,
        "min_cash_weight": min_cash_weight,
        "min_invested_weight": min_invested_weight,
        "momentum_window": momentum_window,
        "annualized_return": round(float(summary.annualized_return), 6),
        "annualized_volatility": round(float(summary.annualized_volatility), 6),
        "max_drawdown": round(float(summary.max_drawdown), 6),
        "average_turnover": round(float(result.turnover.mean()), 6),
        "sharpe_ratio": round(float(sharpe_ratio), 6),
        "sortino_ratio": round(float(summary.sortino_ratio), 6),
    }


def clean_value(value: float, tolerance: float = 1e-5) -> float:
    return 0.0 if abs(value) < tolerance else value


def clamp_for_display(
    value: float,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    tolerance: float = 2e-3,
) -> float:
    cleaned = clean_value(float(value))
    if lower_bound is not None and abs(cleaned - lower_bound) < tolerance:
        cleaned = lower_bound
    if upper_bound is not None and abs(cleaned - upper_bound) < tolerance:
        cleaned = upper_bound
    return cleaned


def clean_mapping(
    values: dict[str, float], tolerance: float = 1e-5
) -> dict[str, float]:
    return {
        key: round(clean_value(float(value), tolerance), 6)
        for key, value in values.items()
    }


def clean_constraint_mapping(
    values: dict[str, float],
    *,
    lower_bounds: dict[str, float] | None = None,
    upper_bounds: dict[str, float] | None = None,
    tolerance: float = 2e-3,
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


def build_asset_class_matrix(
    symbols: list[str],
    asset_classes: dict[str, str],
    class_names: list[str],
) -> list[list[float]]:
    matrix: list[list[float]] = [[0.0 for _ in symbols] for _ in class_names]
    class_index = {class_name: idx for idx, class_name in enumerate(class_names)}
    for symbol_index, symbol in enumerate(symbols):
        class_name = asset_classes.get(symbol)
        if class_name is None or class_name not in class_index:
            continue
        matrix[class_index[class_name]][symbol_index] = 1.0
    return matrix


def build_benchmark_weights(
    index,
    symbols: list[str],
    benchmark_symbol: str | None,
    benchmark_weight: float,
):
    if benchmark_symbol is None:
        return None
    if benchmark_symbol not in symbols:
        raise ValueError(
            f"Benchmark symbol {benchmark_symbol} is not in the model universe."
        )
    data = {symbol: np.zeros(len(index), dtype=float) for symbol in symbols}
    data["USDOLLAR"] = np.full(
        len(index), max(0.0, 1.0 - benchmark_weight), dtype=float
    )
    data[benchmark_symbol] = np.full(len(index), benchmark_weight, dtype=float)
    return pd.DataFrame(data, index=index)


def rolling_window_comparison(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    *,
    window_days: int,
    step_days: int,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float | int]:
    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError("Strategy and benchmark returns must have the same length.")
    if window_days <= 0 or step_days <= 0:
        raise ValueError("window_days and step_days must be positive.")
    if len(strategy_returns) < window_days:
        raise ValueError(
            "Not enough realized periods for the requested rolling window."
        )

    window_results: list[dict[str, float | bool]] = []
    for start in range(0, len(strategy_returns) - window_days + 1, step_days):
        end = start + window_days
        strategy_slice = strategy_returns[start:end]
        benchmark_slice = benchmark_returns[start:end]
        strategy_summary = summarize_return_series(
            strategy_slice,
            trading_days_per_year=trading_days_per_year,
        )
        benchmark_summary = summarize_return_series(
            benchmark_slice,
            trading_days_per_year=trading_days_per_year,
        )
        strategy_sharpe = (
            strategy_summary.annualized_return / strategy_summary.annualized_volatility
            if strategy_summary.annualized_volatility > 0
            else 0.0
        )
        benchmark_sharpe = (
            benchmark_summary.annualized_return / benchmark_summary.annualized_volatility
            if benchmark_summary.annualized_volatility > 0
            else 0.0
        )
        window_results.append(
            {
                "beat_return": strategy_summary.total_return
                > benchmark_summary.total_return,
                "beat_sharpe": strategy_sharpe > benchmark_sharpe,
                "lower_drawdown": strategy_summary.max_drawdown
                < benchmark_summary.max_drawdown,
                "excess_total_return": strategy_summary.total_return
                - benchmark_summary.total_return,
            }
        )

    total_windows = len(window_results)
    beat_return_count = sum(
        1 for result in window_results if bool(result["beat_return"])
    )
    beat_sharpe_count = sum(
        1 for result in window_results if bool(result["beat_sharpe"])
    )
    lower_drawdown_count = sum(
        1 for result in window_results if bool(result["lower_drawdown"])
    )
    average_excess_return = float(
        np.mean([float(result["excess_total_return"]) for result in window_results])
    )
    return {
        "window_days": window_days,
        "step_days": step_days,
        "windows": total_windows,
        "beat_spy_return_windows": beat_return_count,
        "beat_spy_return_rate": round(beat_return_count / total_windows, 6),
        "beat_spy_sharpe_windows": beat_sharpe_count,
        "beat_spy_sharpe_rate": round(beat_sharpe_count / total_windows, 6),
        "lower_drawdown_than_spy_windows": lower_drawdown_count,
        "lower_drawdown_than_spy_rate": round(lower_drawdown_count / total_windows, 6),
        "average_excess_total_return_vs_spy": round(average_excess_return, 6),
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
    bars_by_symbol = alpaca.get_daily_bars_for_period(
        model.symbols,
        total_days,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        offline=offline,
    )
    returns_frame, prices_frame, closes_by_symbol = bars_to_market_data(bars_by_symbol)
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
    core_symbol: str | None = None,
    core_weight: float = 0.0,
    target_volatility: float | None = None,
    max_leverage: float | None = None,
    benchmark_symbol: str | None = None,
    benchmark_weight: float = 1.0,
    linear_trade_cost: float = 0.0,
    planning_horizon: int = 1,
    rolling_window_days: int = 0,
    rolling_step_days: int = 21,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
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

    model, closes_by_symbol, returns_frame, prices_frame, warmup_days = (
        prepare_cvxportfolio_context(
            model_path=model_path,
            lookback_days=lookback_days,
            backtest_days=backtest_days,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )
    )
    forecasts = momentum_forecast(
        returns_frame=returns_frame,
        momentum_window=momentum_window,
        mean_shrinkage=mean_shrinkage,
    )
    benchmark = build_benchmark_weights(
        returns_frame.index,
        model.symbols,
        benchmark_symbol=benchmark_symbol,
        benchmark_weight=benchmark_weight,
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
        core_symbol=core_symbol,
        core_weight=core_weight,
        target_volatility=target_volatility,
        max_leverage=max_leverage,
        benchmark=benchmark,
        planning_horizon=planning_horizon,
    )
    simulator = cvx.MarketSimulator(
        returns=returns_frame, prices=prices_frame, cash_key="USDOLLAR"
    )
    result = simulator.backtest(policy, start_time=returns_frame.index[warmup_days])
    latest_weights = result.w.iloc[-1].drop(labels=["USDOLLAR"], errors="ignore")
    latest_cash_weight = float(result.w.iloc[-1].get("USDOLLAR", 0.0))
    initial_value = float(result.v.iloc[0])
    final_value = float(result.v.iloc[-1])
    normalized_final_value = (
        final_value / initial_value if initial_value else final_value
    )
    realized_returns = result.v.pct_change().dropna().to_numpy()
    turnover_series = result.turnover.reindex(result.v.index).fillna(0.0).to_numpy()
    # Apply a simple proportional transaction cost ex-post using reported turnover.
    net_realized_returns = realized_returns - linear_trade_cost * turnover_series[1:]
    realized_summary = summarize_return_series(
        net_realized_returns,
        trading_days_per_year=trading_days_per_year,
    )
    geometric_sharpe = (
        realized_summary.annualized_return / realized_summary.annualized_volatility
        if realized_summary.annualized_volatility > 0
        else 0.0
    )
    first_timestamp = str(result.v.index[0])
    last_timestamp = str(result.v.index[-1])
    realized_periods = int(len(realized_returns))
    latest_class_exposures = clean_constraint_mapping(
        {
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
        },
        lower_bounds=model.class_min_weights,
        upper_bounds=model.class_max_weights,
    )
    benchmark_results = {
        "spy": run_fixed_weight_benchmark(
            symbols=model.symbols,
            closes_by_symbol=closes_by_symbol,
            weights_by_symbol={"SPY": 1.0},
            start_day=warmup_days,
            trading_days_per_year=trading_days_per_year,
        ),
        "sixty_forty_spy_tlt": run_fixed_weight_benchmark(
            symbols=model.symbols,
            closes_by_symbol=closes_by_symbol,
            weights_by_symbol={"SPY": 0.6, "TLT": 0.4},
            start_day=warmup_days,
            trading_days_per_year=trading_days_per_year,
        ),
        "equal_weight": run_fixed_weight_benchmark(
            symbols=model.symbols,
            closes_by_symbol=closes_by_symbol,
            weights_by_symbol={
                symbol: 1.0 / len(model.symbols) for symbol in model.symbols
            },
            start_day=warmup_days,
            trading_days_per_year=trading_days_per_year,
        ),
        "half_spy_half_cash": run_fixed_weight_benchmark(
            symbols=model.symbols,
            closes_by_symbol=closes_by_symbol,
            weights_by_symbol={"SPY": 0.5},
            start_day=warmup_days,
            trading_days_per_year=trading_days_per_year,
        ),
    }
    rolling_comparison = None
    if rolling_window_days > 0:
        spy_prices = np.array(closes_by_symbol["SPY"], dtype=float)
        spy_returns = spy_prices[1:] / spy_prices[:-1] - 1.0
        benchmark_aligned_returns = spy_returns[
            warmup_days : warmup_days + realized_periods
        ]
        rolling_comparison = rolling_window_comparison(
            net_realized_returns,
            benchmark_aligned_returns,
            window_days=rolling_window_days,
            step_days=rolling_step_days,
            trading_days_per_year=trading_days_per_year,
        )

    result_payload = {
        "symbols": model.symbols,
        "cvxportfolio_backtest": {
            "days": backtest_days,
            "warmup_days": warmup_days,
            "risk_aversion": risk_aversion,
            "mean_shrinkage": mean_shrinkage,
            "momentum_window": momentum_window,
            "core_symbol": core_symbol,
            "core_weight": core_weight,
            "target_volatility": target_volatility,
            "max_leverage": max_leverage,
            "benchmark_symbol": benchmark_symbol,
            "benchmark_weight": benchmark_weight,
            "linear_trade_cost": linear_trade_cost,
            "planning_horizon": planning_horizon,
            "trading_days_per_year": trading_days_per_year,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "realized_periods": realized_periods,
            "initial_value": round(initial_value, 6),
            "final_value": round(final_value, 6),
            "normalized_final_value": round(normalized_final_value, 6),
            "total_return": round(realized_summary.total_return, 6),
            "value_ratio_total_return": round(normalized_final_value - 1.0, 6),
            "realized_return_series_final_value": round(realized_summary.final_value, 6),
            "annualized_return": round(float(realized_summary.annualized_return), 6),
            "annualized_volatility": round(
                float(realized_summary.annualized_volatility), 6
            ),
            "max_drawdown": round(float(realized_summary.max_drawdown), 6),
            "average_turnover": round(float(result.turnover.mean()), 6),
            "sharpe_ratio": round(float(geometric_sharpe), 6),
            "sortino_ratio": round(float(realized_summary.sortino_ratio), 6),
            "cvxportfolio_annualized_average_return": round(
                float(result.annualized_average_return), 6
            ),
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
        "latest_cash_weight": round(
            clamp_for_display(
                latest_cash_weight, lower_bound=min_cash_weight, upper_bound=1.0
            ),
            6,
        ),
        "latest_asset_class_exposures": latest_class_exposures,
        "benchmarks": benchmark_results,
    }
    if benchmark_symbol is not None:
        result_payload["cvxportfolio_backtest"]["annualized_active_return"] = round(
            float(result.annualized_average_active_return), 6
        )
        result_payload["cvxportfolio_backtest"]["annualized_active_volatility"] = round(
            float(result.annualized_active_volatility), 6
        )
    if rolling_comparison is not None:
        result_payload["rolling_vs_spy"] = rolling_comparison
    return result_payload


def run_framework_comparison(
    model_path: str,
    lookback_days: int,
    backtest_days: int,
    cvxportfolio_config: dict[str, float | int],
    custom_config: dict[str, float | int],
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    use_cache: bool = False,
    refresh_cache: bool = False,
    offline: bool = False,
) -> dict:
    model, closes_by_symbol, _returns_frame, _prices_frame, _warmup_days = (
        prepare_cvxportfolio_context(
            model_path=model_path,
            lookback_days=lookback_days,
            backtest_days=backtest_days,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )
    )
    cvxportfolio_result = run_cvxportfolio_backtest(
        model_path=model_path,
        lookback_days=lookback_days,
        backtest_days=backtest_days,
        risk_aversion=float(cvxportfolio_config["risk_aversion"]),
        min_cash_weight=float(cvxportfolio_config["min_cash_weight"]),
        min_invested_weight=float(cvxportfolio_config["min_invested_weight"]),
        max_weight=float(cvxportfolio_config["max_weight"]),
        core_symbol=(
            str(cvxportfolio_config["core_symbol"])
            if cvxportfolio_config.get("core_symbol")
            else None
        ),
        core_weight=float(cvxportfolio_config.get("core_weight", 0.0)),
        target_volatility=(
            float(cvxportfolio_config["target_volatility"])
            if cvxportfolio_config.get("target_volatility") is not None
            else None
        ),
        max_leverage=(
            float(cvxportfolio_config["max_leverage"])
            if cvxportfolio_config.get("max_leverage") is not None
            else None
        ),
        benchmark_symbol=(
            str(cvxportfolio_config["benchmark_symbol"])
            if cvxportfolio_config.get("benchmark_symbol")
            else None
        ),
        benchmark_weight=float(cvxportfolio_config.get("benchmark_weight", 1.0)),
        mean_shrinkage=float(cvxportfolio_config["mean_shrinkage"]),
        momentum_window=int(cvxportfolio_config["momentum_window"]),
        linear_trade_cost=float(cvxportfolio_config["linear_trade_cost"]),
        planning_horizon=int(cvxportfolio_config["planning_horizon"]),
        trading_days_per_year=trading_days_per_year,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        offline=offline,
    )

    constrained_class_names = list(model.class_min_weights) + [
        name for name in model.class_max_weights if name not in model.class_min_weights
    ]
    asset_class_matrix = build_asset_class_matrix(
        model.symbols, model.asset_classes, constrained_class_names
    )
    custom_opt_config = OptimizationConfig(
        risk_aversion=float(custom_config["risk_aversion"]),
        min_weight=0.0,
        max_weight=float(custom_config["max_weight"]),
        rebalance_threshold=0.02,
        turnover_penalty=float(custom_config["turnover_penalty"]),
        force_full_investment=not bool(custom_config["allow_cash"]),
        min_cash_weight=float(custom_config["min_cash_weight"]),
        max_turnover=float(custom_config["max_turnover"]),
        min_invested_weight=float(custom_config["min_invested_weight"]),
        class_min_weights=model.class_min_weights,
        class_max_weights=model.class_max_weights,
    )
    custom_closes_by_symbol = {
        symbol: closes[-(lookback_days + backtest_days + 1) :]
        for symbol, closes in closes_by_symbol.items()
    }
    custom_backtest = run_custom_backtest(
        symbols=model.symbols,
        closes_by_symbol=custom_closes_by_symbol,
        lookback_days=lookback_days,
        rebalance_every=int(custom_config["rebalance_every"]),
        return_model=str(custom_config["return_model"]),
        mean_shrinkage=float(custom_config["mean_shrinkage"]),
        momentum_window=int(custom_config["momentum_window"]),
        opt_config=custom_opt_config,
        asset_class_matrix=(
            np.array(asset_class_matrix, dtype=float)
            if constrained_class_names
            else None
        ),
        trading_days_per_year=trading_days_per_year,
    )
    custom_latest_weights = clean_constraint_mapping(
        {
            symbol: float(weight)
            for symbol, weight in zip(
                model.symbols, custom_backtest.latest_weights, strict=True
            )
        },
        upper_bounds={
            symbol: float(custom_config["max_weight"]) for symbol in model.symbols
        },
    )
    custom_cash_weight = round(
        clamp_for_display(
            max(0.0, 1.0 - float(sum(custom_backtest.latest_weights))),
            lower_bound=float(custom_config["min_cash_weight"]),
            upper_bound=1.0,
        ),
        6,
    )
    benchmark_results = cvxportfolio_result["benchmarks"]
    return {
        "symbols": model.symbols,
        "comparison": {
            "lookback_days": lookback_days,
            "backtest_days": backtest_days,
        },
        "custom_baseline": {
            "config": custom_config,
            "metrics": {
                "annualized_return": round(float(custom_backtest.annualized_return), 6),
                "annualized_volatility": round(
                    float(custom_backtest.annualized_volatility), 6
                ),
                "max_drawdown": round(float(custom_backtest.max_drawdown), 6),
                "sortino_ratio": round(float(custom_backtest.sortino_ratio), 6),
                "average_turnover": round(float(custom_backtest.average_turnover), 6),
                "rebalance_count": int(custom_backtest.rebalance_count),
                "total_return": round(float(custom_backtest.total_return), 6),
            },
            "latest_target_weights": custom_latest_weights,
            "latest_cash_weight": custom_cash_weight,
        },
        "cvxportfolio": {
            "config": cvxportfolio_config,
            "metrics": cvxportfolio_result["cvxportfolio_backtest"],
            "latest_target_weights": cvxportfolio_result["latest_target_weights"],
            "latest_cash_weight": cvxportfolio_result["latest_cash_weight"],
            "latest_asset_class_exposures": cvxportfolio_result[
                "latest_asset_class_exposures"
            ],
        },
        "benchmarks": benchmark_results,
    }


def run_cvxportfolio_sweep(
    model_path: str,
    lookback_days: int,
    backtest_days: int,
    top_n: int,
    linear_trade_cost: float = 0.0,
    planning_horizon: int = 1,
    core_symbol: str | None = None,
    core_weight: float = 0.0,
    target_volatility: float | None = None,
    max_leverage: float | None = None,
    benchmark_symbol: str | None = None,
    benchmark_weight: float = 1.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
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

    model, _closes_by_symbol, returns_frame, prices_frame, warmup_days = (
        prepare_cvxportfolio_context(
            model_path=model_path,
            lookback_days=lookback_days,
            backtest_days=backtest_days,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )
    )
    closes_by_symbol = _closes_by_symbol
    risk_grid = [0.5, 1.0, 2.0]
    shrinkage_grid = [0.5, 0.75, 0.9]
    cash_grid = [0.05, 0.10, 0.20]
    invested_grid = [0.20, 0.30, 0.40]
    momentum_grid = [42, 63, 84]

    benchmark_df = build_benchmark_weights(
        returns_frame.index,
        model.symbols,
        benchmark_symbol=benchmark_symbol,
        benchmark_weight=benchmark_weight,
    )

    grid_params = list(
        itertools.product(
            risk_grid,
            shrinkage_grid,
            cash_grid,
            invested_grid,
            momentum_grid,
        )
    )
    worker_args = [
        (
            risk_aversion,
            mean_shrinkage,
            min_cash_weight,
            min_invested_weight,
            momentum_window,
            model.symbols,
            returns_frame,
            prices_frame,
            warmup_days,
            model.class_min_weights,
            model.class_max_weights,
            model.asset_classes,
            core_symbol,
            core_weight,
            target_volatility,
            max_leverage,
            benchmark_df,
            linear_trade_cost,
            planning_horizon,
            trading_days_per_year,
        )
        for (
            risk_aversion,
            mean_shrinkage,
            min_cash_weight,
            min_invested_weight,
            momentum_window,
        ) in grid_params
    ]

    results: list[dict[str, float | int]] = []
    skipped: list[dict[str, float | int | str]] = []

    with ProcessPoolExecutor() as executor:
        for combo_params, outcome in zip(
            grid_params, executor.map(_run_single_sweep, worker_args)
        ):
            results.append(outcome)

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
            start_day=warmup_days,
            trading_days_per_year=trading_days_per_year,
        ),
        "sixty_forty_spy_tlt": run_fixed_weight_benchmark(
            symbols=model.symbols,
            closes_by_symbol=closes_by_symbol,
            weights_by_symbol={"SPY": 0.6, "TLT": 0.4},
            start_day=warmup_days,
            trading_days_per_year=trading_days_per_year,
        ),
        "equal_weight": run_fixed_weight_benchmark(
            symbols=model.symbols,
            closes_by_symbol=closes_by_symbol,
            weights_by_symbol={
                symbol: 1.0 / len(model.symbols) for symbol in model.symbols
            },
            start_day=warmup_days,
            trading_days_per_year=trading_days_per_year,
        ),
        "half_spy_half_cash": run_fixed_weight_benchmark(
            symbols=model.symbols,
            closes_by_symbol=closes_by_symbol,
            weights_by_symbol={"SPY": 0.5},
            start_day=warmup_days,
            trading_days_per_year=trading_days_per_year,
        ),
    }
    return {
        "symbols": model.symbols,
        "cvxportfolio_sweep": {
            "days": backtest_days,
            "warmup_days": warmup_days,
            "linear_trade_cost": linear_trade_cost,
            "planning_horizon": planning_horizon,
            "trading_days_per_year": trading_days_per_year,
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
