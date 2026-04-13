from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .black_litterman import estimate_inputs_from_black_litterman
from .config import OptimizationConfig
from .estimation import estimate_inputs_from_momentum, estimate_inputs_from_prices
from .optimizer import optimize_basket_weights, optimize_weights, project_weights
from .risk_parity import estimate_inputs_risk_parity

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestResult:
    final_value: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    rebalance_count: int
    average_turnover: float
    latest_weights: np.ndarray
    daily_values: tuple[float, ...]


def _find_symbol_indices(
    symbols: list[str],
    subset: list[str],
) -> list[int]:
    symbol_index = {symbol: idx for idx, symbol in enumerate(symbols)}
    result = [symbol_index[s] for s in subset if s in symbol_index]
    return result


def compute_dual_momentum_weights(
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    asset_classes: dict[str, str],
    lookback_days: int,
    top_k: int,
    weighting: str = "equal",
    softmax_temperature: float = 0.05,
    absolute_threshold: float = 0.0,
    basket_opt: str | None = None,
    basket_risk_aversion: float = 1.0,
    target_vol: float | None = None,
    max_single_weight: float | None = None,
    vol_window: int = 63,
    trailing_stop: float | None = None,
) -> dict[str, float]:
    """Compute dual-momentum target weights for a single point in time.

    Used by the live rebalancing path to produce target weights from current
    Alpaca data, identical to the backtest logic at a single rebalance step.
    """
    # The live CLI handles trailing stops with broker-side protective orders.
    # This single-period weight calculation cannot simulate an entry peak.
    aligned_closes = align_close_history(symbols, closes_by_symbol)
    price_matrix = np.array([aligned_closes[symbol] for symbol in symbols], dtype=float)
    if price_matrix.ndim != 2 or price_matrix.shape[1] < lookback_days + 1:
        raise ValueError("Not enough price history to compute dual momentum weights.")

    returns = price_matrix[:, 1:] / price_matrix[:, :-1] - 1.0
    trailing_returns = price_matrix[:, -1] / price_matrix[:, -(lookback_days + 1)] - 1.0
    trailing_volatility = returns.std(axis=1, ddof=0)

    risky_symbols = [
        s
        for s in symbols
        if not asset_classes.get(s, "").startswith("bond")
        and asset_classes.get(s) != "cash_like"
    ]
    defensive_symbols = [
        s
        for s in symbols
        if asset_classes.get(s, "").startswith("bond")
        or asset_classes.get(s) == "cash_like"
    ]
    risky_indices = _find_symbol_indices(symbols, risky_symbols)
    defensive_indices = _find_symbol_indices(symbols, defensive_symbols)
    if not risky_indices:
        raise ValueError("Dual momentum requires at least one risky symbol.")

    cash_like_index = next(
        (i for i, s in enumerate(symbols) if asset_classes.get(s) == "cash_like"),
        None,
    )
    defensive_floor = (
        float(trailing_returns[cash_like_index])
        if cash_like_index is not None
        else absolute_threshold
    )
    risky_ranked = sorted(
        (
            (idx, float(trailing_returns[idx]))
            for idx in risky_indices
            if float(trailing_returns[idx]) > max(absolute_threshold, defensive_floor)
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    target_weights = np.zeros(len(symbols), dtype=float)
    if risky_ranked:
        selected = risky_ranked[:top_k]
        selected_indices = [idx for idx, _ in selected]

        if basket_opt == "mean-variance":
            basket_returns = trailing_returns[selected_indices]
            basket_returns_history = returns[
                :, max(0, returns.shape[1] - lookback_days) :
            ][selected_indices]
            if len(selected_indices) == 1:
                basket_cov = np.array([[np.var(basket_returns_history[0])]])
            else:
                basket_cov = np.cov(basket_returns_history)
            basket_cov += 1e-8 * np.eye(len(selected_indices))
            basket_weights = optimize_basket_weights(
                expected_returns=basket_returns,
                covariance=basket_cov,
                min_weight=0.0,
                max_weight=1.0,
                risk_aversion=basket_risk_aversion,
                force_full_investment=True,
            )
            for i, idx in enumerate(selected_indices):
                target_weights[idx] = basket_weights[i]
        else:
            for index, weight in _dual_momentum_selected_weights(
                selected=selected,
                trailing_returns=trailing_returns,
                trailing_volatility=trailing_volatility,
                weighting=weighting,
                softmax_temperature=softmax_temperature,
            ).items():
                target_weights[index] = weight

        if max_single_weight is not None:
            capped = target_weights.copy()
            excess = 0.0
            for i in range(len(symbols)):
                if capped[i] > max_single_weight:
                    excess += capped[i] - max_single_weight
                    capped[i] = max_single_weight
                elif capped[i] < 0:
                    excess += abs(capped[i])
                    capped[i] = 0.0
            remaining = capped[capped > 0].sum()
            if remaining > 0:
                for i in range(len(symbols)):
                    if 0 < capped[i] < max_single_weight:
                        capped[i] += excess * (capped[i] / remaining)
            target_weights = capped

        if target_vol is not None:
            recent_returns_window = returns[:, max(0, returns.shape[1] - vol_window) :]
            risky_indices_active = [
                i for i in range(len(symbols)) if target_weights[i] > 0
            ]
            if len(risky_indices_active) > 0:
                w = target_weights[risky_indices_active]
                recent_risky_returns = recent_returns_window[risky_indices_active]
                portfolio_recent_returns = np.dot(w, recent_risky_returns)
                portfolio_vol = float(
                    np.std(portfolio_recent_returns, ddof=0)
                ) * np.sqrt(TRADING_DAYS_PER_YEAR)
            else:
                portfolio_vol = 0.0

            if portfolio_vol > 0 and portfolio_vol > target_vol:
                scale = target_vol / portfolio_vol
                target_weights = target_weights * scale
    elif defensive_indices:
        weight = 1.0 / len(defensive_indices)
        for index in defensive_indices:
            target_weights[index] = weight

    return {s: float(target_weights[i]) for i, s in enumerate(symbols)}


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0.0:
        return np.full(len(values), 1.0 / len(values), dtype=float)
    return values / total


def _dual_momentum_selected_weights(
    *,
    selected: list[tuple[int, float]],
    trailing_returns: np.ndarray,
    trailing_volatility: np.ndarray,
    weighting: str,
    softmax_temperature: float,
) -> dict[int, float]:
    selected_indices = [index for index, _ in selected]
    if not selected_indices:
        return {}

    if weighting == "equal":
        weight_vector = np.full(
            len(selected_indices), 1.0 / len(selected_indices), dtype=float
        )
    elif weighting == "score":
        scores = np.array([max(score, 0.0) for _index, score in selected], dtype=float)
        weight_vector = _normalize_positive(scores)
    elif weighting == "inverse-vol":
        vol_values = np.array(
            [
                max(float(trailing_volatility[index]), 1e-8)
                for index in selected_indices
            ],
            dtype=float,
        )
        weight_vector = _normalize_positive(1.0 / vol_values)
    elif weighting == "softmax":
        scaled_scores = np.array(
            [
                float(trailing_returns[index]) / max(softmax_temperature, 1e-6)
                for index in selected_indices
            ],
            dtype=float,
        )
        shifted = scaled_scores - float(np.max(scaled_scores))
        weight_vector = _normalize_positive(np.exp(shifted))
    else:
        raise ValueError(f"Unknown dual momentum weighting mode: {weighting}")

    return {
        index: float(weight)
        for index, weight in zip(selected_indices, weight_vector, strict=True)
    }


def summarize_return_series(
    returns: np.ndarray,
) -> tuple[float, float, float, float, float]:
    portfolio_value = 1.0
    peak_value = portfolio_value
    max_drawdown = 0.0
    for period_return in returns:
        portfolio_value *= 1.0 + float(period_return)
        peak_value = max(peak_value, portfolio_value)
        max_drawdown = max(max_drawdown, 1.0 - portfolio_value / peak_value)

    periods = max(len(returns), 1)
    annualized_return = portfolio_value ** (TRADING_DAYS_PER_YEAR / periods) - 1.0
    annualized_volatility = returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return (
        portfolio_value,
        portfolio_value - 1.0,
        annualized_return,
        annualized_volatility,
        max_drawdown,
    )


def align_close_history(
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
) -> dict[str, list[float]]:
    lengths = [len(closes_by_symbol[symbol]) for symbol in symbols]
    common_length = min(lengths, default=0)
    if common_length < 2:
        raise ValueError("Not enough aligned price history to run the backtest.")
    if len(set(lengths)) == 1:
        return closes_by_symbol
    return {symbol: closes_by_symbol[symbol][-common_length:] for symbol in symbols}


def run_backtest(
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    lookback_days: int,
    rebalance_every: int,
    return_model: str,
    mean_shrinkage: float,
    momentum_window: int,
    opt_config: OptimizationConfig,
    asset_class_matrix: np.ndarray | None,
) -> BacktestResult:
    aligned_closes = align_close_history(symbols, closes_by_symbol)
    price_matrix = np.array([aligned_closes[symbol] for symbol in symbols], dtype=float)
    if price_matrix.ndim != 2 or price_matrix.shape[1] < lookback_days + 1:
        raise ValueError("Not enough price history to run the backtest.")

    returns = price_matrix[:, 1:] / price_matrix[:, :-1] - 1.0
    portfolio_value = 1.0
    weights = np.zeros(len(symbols), dtype=float)
    portfolio_returns: list[float] = []
    daily_values: list[float] = [1.0]
    turnovers: list[float] = []
    rebalance_count = 0
    peak_value = portfolio_value
    max_drawdown = 0.0

    # Re-estimate inputs at each rebalance date using only prior history, then
    # hold those weights until the next rebalance.
    for step in range(lookback_days, returns.shape[1]):
        if (step - lookback_days) % rebalance_every == 0:
            window_closes = {
                symbol: price_matrix[index, step - lookback_days : step + 1].tolist()
                for index, symbol in enumerate(symbols)
            }
            if return_model == "momentum":
                estimated = estimate_inputs_from_momentum(
                    symbols=symbols,
                    closes_by_symbol=window_closes,
                    mean_shrinkage=mean_shrinkage,
                    momentum_window=momentum_window,
                )
                target_weights = optimize_weights(
                    expected_returns=estimated.expected_returns,
                    covariance=estimated.covariance,
                    config=opt_config,
                    current_weights=weights,
                    asset_class_matrix=asset_class_matrix,
                )
            elif return_model == "black-litterman":
                estimated = estimate_inputs_from_black_litterman(
                    symbols=symbols,
                    closes_by_symbol=window_closes,
                    momentum_window=momentum_window,
                    mean_shrinkage=mean_shrinkage,
                )
                target_weights = optimize_weights(
                    expected_returns=estimated.expected_returns,
                    covariance=estimated.covariance,
                    config=opt_config,
                    current_weights=weights,
                    asset_class_matrix=asset_class_matrix,
                )
            elif return_model == "risk-parity":
                estimated = estimate_inputs_risk_parity(
                    symbols=symbols,
                    closes_by_symbol=window_closes,
                    lookback_days=lookback_days,
                )
                target_weights = project_weights(
                    target_weights=estimated.weights,
                    config=opt_config,
                    current_weights=weights,
                    asset_class_matrix=asset_class_matrix,
                )
            else:
                estimated = estimate_inputs_from_prices(
                    symbols=symbols,
                    closes_by_symbol=window_closes,
                    mean_shrinkage=mean_shrinkage,
                )
                target_weights = optimize_weights(
                    expected_returns=estimated.expected_returns,
                    covariance=estimated.covariance,
                    config=opt_config,
                    current_weights=weights,
                    asset_class_matrix=asset_class_matrix,
                )
            turnovers.append(float(np.abs(target_weights - weights).sum()))
            weights = target_weights
            rebalance_count += 1

        cash_weight = max(0.0, 1.0 - float(weights.sum()))
        period_return = float(np.dot(weights, returns[:, step]) + cash_weight * 0.0)
        portfolio_returns.append(period_return)
        portfolio_value *= 1.0 + period_return
        daily_values.append(portfolio_value)
        peak_value = max(peak_value, portfolio_value)
        max_drawdown = max(max_drawdown, 1.0 - portfolio_value / peak_value)

    returns_array = np.array(portfolio_returns, dtype=float)
    _, _, annualized_return, annualized_volatility, _ = summarize_return_series(
        returns_array
    )
    average_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    return BacktestResult(
        final_value=portfolio_value,
        total_return=portfolio_value - 1.0,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        max_drawdown=max_drawdown,
        rebalance_count=rebalance_count,
        average_turnover=average_turnover,
        latest_weights=weights,
        daily_values=tuple(daily_values),
    )


def run_fixed_weight_benchmark(
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    weights_by_symbol: dict[str, float],
    start_day: int,
) -> dict[str, float]:
    aligned_closes = align_close_history(symbols, closes_by_symbol)
    price_matrix = np.array([aligned_closes[symbol] for symbol in symbols], dtype=float)
    returns = price_matrix[:, 1:] / price_matrix[:, :-1] - 1.0
    benchmark_weights = np.array(
        [weights_by_symbol.get(symbol, 0.0) for symbol in symbols], dtype=float
    )
    benchmark_returns = np.array(
        [
            float(np.dot(benchmark_weights, returns[:, step]))
            for step in range(start_day, returns.shape[1])
        ],
        dtype=float,
    )
    (
        final_value,
        total_return,
        annualized_return,
        annualized_volatility,
        max_drawdown,
    ) = summarize_return_series(benchmark_returns)
    return {
        "final_value": round(float(final_value), 6),
        "total_return": round(float(total_return), 6),
        "annualized_return": round(float(annualized_return), 6),
        "annualized_volatility": round(float(annualized_volatility), 6),
        "max_drawdown": round(float(max_drawdown), 6),
    }


def run_dual_momentum_backtest(
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    asset_classes: dict[str, str],
    lookback_days: int,
    rebalance_every: int,
    top_k: int,
    absolute_threshold: float,
    weighting: str = "equal",
    softmax_temperature: float = 0.05,
    target_vol: float | None = None,
    max_single_weight: float | None = None,
    vol_window: int = 63,
    basket_opt: str | None = None,
    basket_risk_aversion: float = 1.0,
    trailing_stop: float | None = None,
) -> BacktestResult:
    aligned_closes = align_close_history(symbols, closes_by_symbol)
    price_matrix = np.array([aligned_closes[symbol] for symbol in symbols], dtype=float)
    if price_matrix.ndim != 2 or price_matrix.shape[1] < lookback_days + 1:
        raise ValueError("Not enough price history to run the backtest.")

    returns = price_matrix[:, 1:] / price_matrix[:, :-1] - 1.0
    portfolio_value = 1.0
    weights = np.zeros(len(symbols), dtype=float)
    portfolio_returns: list[float] = []
    daily_values: list[float] = [1.0]
    turnovers: list[float] = []

    # Compute SPY benchmark curve if available
    spy_values: list[float] = []
    if "SPY" in symbols:
        spy_idx = symbols.index("SPY")
        # SPY returns start from day 1 relative to lookback start
        spy_prices = price_matrix[spy_idx, 1:]
        start_price = price_matrix[spy_idx, 0]
        spy_values = [(p / start_price) for p in spy_prices]

    rebalance_count = 0
    peak_value = portfolio_value
    max_drawdown = 0.0

    # Per-asset peak prices since entry, for trailing stop-loss
    asset_peak_price: np.ndarray | None = None

    risky_symbols = [
        symbol
        for symbol in symbols
        if not asset_classes.get(symbol, "").startswith("bond")
        and asset_classes.get(symbol) != "cash_like"
    ]
    defensive_symbols = [
        symbol
        for symbol in symbols
        if asset_classes.get(symbol, "").startswith("bond")
        or asset_classes.get(symbol) == "cash_like"
    ]
    risky_indices = _find_symbol_indices(symbols, risky_symbols)
    defensive_indices = _find_symbol_indices(symbols, defensive_symbols)
    if not risky_indices:
        raise ValueError("Dual momentum requires at least one risky symbol.")

    cash_like_index = next(
        (
            index
            for index, symbol in enumerate(symbols)
            if asset_classes.get(symbol) == "cash_like"
        ),
        None,
    )

    for step in range(lookback_days, returns.shape[1]):
        if (step - lookback_days) % rebalance_every == 0:
            trailing_returns = (
                price_matrix[:, step] / price_matrix[:, step - lookback_days] - 1.0
            )
            trailing_volatility = returns[:, step - lookback_days : step].std(
                axis=1, ddof=0
            )
            defensive_floor = (
                float(trailing_returns[cash_like_index])
                if cash_like_index is not None
                else absolute_threshold
            )
            risky_ranked = sorted(
                (
                    (index, float(trailing_returns[index]))
                    for index in risky_indices
                    if float(trailing_returns[index])
                    > max(absolute_threshold, defensive_floor)
                ),
                key=lambda item: item[1],
                reverse=True,
            )

            target_weights = np.zeros(len(symbols), dtype=float)
            if risky_ranked:
                selected = risky_ranked[:top_k]
                selected_indices = [idx for idx, _ in selected]

                # Momentum pre-selects the basket, then size via one of:
                #  - weighting: equal / score / inverse-vol / softmax
                #  - basket_opt: cvxpy mean-variance on the small basket
                if basket_opt == "mean-variance":
                    basket_returns = trailing_returns[selected_indices]
                    basket_returns_history = returns[
                        :, max(0, step - lookback_days) : step
                    ][selected_indices]
                    if len(selected_indices) == 1:
                        basket_cov = np.array([[np.var(basket_returns_history[0])]])
                    else:
                        basket_cov = np.cov(basket_returns_history)
                    # Add diagonal ridge for stability
                    basket_cov += 1e-8 * np.eye(len(selected_indices))
                    basket_weights = optimize_basket_weights(
                        expected_returns=basket_returns,
                        covariance=basket_cov,
                        min_weight=0.0,
                        max_weight=1.0,
                        risk_aversion=basket_risk_aversion,
                        force_full_investment=True,
                    )
                    for i, idx in enumerate(selected_indices):
                        target_weights[idx] = basket_weights[i]
                else:
                    for index, weight in _dual_momentum_selected_weights(
                        selected=selected,
                        trailing_returns=trailing_returns,
                        trailing_volatility=trailing_volatility,
                        weighting=weighting,
                        softmax_temperature=softmax_temperature,
                    ).items():
                        target_weights[index] = weight

                # Apply single-asset cap
                if max_single_weight is not None:
                    capped = target_weights.copy()
                    excess = 0.0
                    for i in range(len(symbols)):
                        if capped[i] > max_single_weight:
                            excess += capped[i] - max_single_weight
                            capped[i] = max_single_weight
                        elif capped[i] < 0:
                            excess += abs(capped[i])
                            capped[i] = 0.0
                    remaining = capped[capped > 0].sum()
                    if remaining > 0:
                        for i in range(len(symbols)):
                            if 0 < capped[i] < max_single_weight:
                                capped[i] += excess * (capped[i] / remaining)
                    target_weights = capped

                # Scale risky basket to hit target portfolio volatility
                if target_vol is not None:
                    recent_returns_window = returns[:, max(0, step - vol_window) : step]
                    risky_indices_active = [
                        i for i in range(len(symbols)) if target_weights[i] > 0
                    ]
                    if len(risky_indices_active) > 0:
                        w = target_weights[risky_indices_active]
                        recent_risky_returns = recent_returns_window[
                            risky_indices_active
                        ]
                        # Portfolio returns over the lookback window
                        portfolio_recent_returns = np.dot(w, recent_risky_returns)
                        portfolio_vol = float(
                            np.std(portfolio_recent_returns, ddof=0)
                        ) * np.sqrt(TRADING_DAYS_PER_YEAR)
                    else:
                        portfolio_vol = 0.0

                    if portfolio_vol > 0 and portfolio_vol > target_vol:
                        scale = target_vol / portfolio_vol
                        target_weights = target_weights * scale
            elif defensive_indices:
                weight = 1.0 / len(defensive_indices)
                for index in defensive_indices:
                    target_weights[index] = weight

            turnovers.append(float(np.abs(target_weights - weights).sum()))
            weights = target_weights
            rebalance_count += 1

        # Trailing stop-loss: track peak prices since entry per asset
        if trailing_stop is not None:
            if asset_peak_price is None:
                asset_peak_price = price_matrix[:, step - 1].copy()
            # Update peaks for held positions
            for i in range(len(symbols)):
                if weights[i] > 0:
                    asset_peak_price[i] = max(
                        asset_peak_price[i], price_matrix[i, step]
                    )
            # Check for stop breaches and flatten
            for i in range(len(symbols)):
                if weights[i] > 0 and asset_peak_price[i] > 0:
                    drawdown_from_peak = (
                        asset_peak_price[i] - price_matrix[i, step]
                    ) / asset_peak_price[i]
                    if drawdown_from_peak > trailing_stop:
                        weights[i] = 0.0
                        asset_peak_price[i] = 0.0  # reset until re-entered

        period_return = float(np.dot(weights, returns[:, step]))
        portfolio_returns.append(period_return)
        portfolio_value *= 1.0 + period_return
        daily_values.append(portfolio_value)
        peak_value = max(peak_value, portfolio_value)
        max_drawdown = max(max_drawdown, 1.0 - portfolio_value / peak_value)

    returns_array = np.array(portfolio_returns, dtype=float)
    _, _, annualized_return, annualized_volatility, _ = summarize_return_series(
        returns_array
    )
    average_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    return BacktestResult(
        final_value=portfolio_value,
        total_return=portfolio_value - 1.0,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        max_drawdown=max_drawdown,
        rebalance_count=rebalance_count,
        average_turnover=average_turnover,
        latest_weights=weights,
        daily_values=tuple(daily_values),
    )


def _run_single_window(
    *,
    strategy: str,
    symbols: list[str],
    window_closes: dict[str, list[float]],
    asset_classes: dict[str, str],
    lookback_days: int,
    rebalance_every: int,
    return_model: str,
    mean_shrinkage: float,
    momentum_window: int,
    opt_config: OptimizationConfig,
    asset_class_matrix: np.ndarray | None,
    top_k: int,
    absolute_threshold: float,
    weighting: str,
    softmax_temperature: float,
) -> dict[str, float | bool]:
    """Run a single rolling window backtest and compare against SPY."""
    if strategy == "dual-momentum":
        backtest = run_dual_momentum_backtest(
            symbols=symbols,
            closes_by_symbol=window_closes,
            asset_classes=asset_classes,
            lookback_days=lookback_days,
            rebalance_every=rebalance_every,
            top_k=top_k,
            absolute_threshold=absolute_threshold,
            weighting=weighting,
            softmax_temperature=softmax_temperature,
        )
    else:
        backtest = run_backtest(
            symbols=symbols,
            closes_by_symbol=window_closes,
            lookback_days=lookback_days,
            rebalance_every=rebalance_every,
            return_model=return_model,
            mean_shrinkage=mean_shrinkage,
            momentum_window=momentum_window,
            opt_config=opt_config,
            asset_class_matrix=asset_class_matrix,
        )

    spy_benchmark = run_fixed_weight_benchmark(
        symbols=symbols,
        closes_by_symbol=window_closes,
        weights_by_symbol={"SPY": 1.0},
        start_day=lookback_days,
    )
    strategy_sharpe = (
        backtest.annualized_return / backtest.annualized_volatility
        if backtest.annualized_volatility > 0
        else 0.0
    )
    spy_sharpe = (
        float(spy_benchmark["annualized_return"])
        / float(spy_benchmark["annualized_volatility"])
        if float(spy_benchmark["annualized_volatility"]) > 0
        else 0.0
    )
    return {
        "beat_return": backtest.total_return > float(spy_benchmark["total_return"]),
        "beat_sharpe": strategy_sharpe > spy_sharpe,
        "lower_drawdown": backtest.max_drawdown < float(spy_benchmark["max_drawdown"]),
        "excess_total_return": backtest.total_return
        - float(spy_benchmark["total_return"]),
    }


def rolling_window_comparison(
    *,
    strategy: str,
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    asset_classes: dict[str, str],
    lookback_days: int,
    window_days: int,
    step_days: int,
    rebalance_every: int,
    return_model: str,
    mean_shrinkage: float,
    momentum_window: int,
    opt_config: OptimizationConfig,
    asset_class_matrix: np.ndarray | None,
    top_k: int,
    absolute_threshold: float,
    weighting: str,
    softmax_temperature: float,
) -> dict[str, float | int]:
    if window_days <= 0 or step_days <= 0:
        raise ValueError("window_days and step_days must be positive.")

    aligned_closes = align_close_history(symbols, closes_by_symbol)
    total_periods = len(aligned_closes[symbols[0]]) - 1
    if total_periods < lookback_days + window_days:
        raise ValueError(
            "Not enough price history to run the requested rolling-window comparison."
        )

    window_results: list[dict[str, float | bool]] = []
    last_start = total_periods - (lookback_days + window_days)

    # Build window data for parallel execution
    window_args = []
    for start in range(0, last_start + 1, step_days):
        end = start + lookback_days + window_days + 1
        window_closes = {
            symbol: closes[start:end] for symbol, closes in aligned_closes.items()
        }
        window_args.append(window_closes)

    try:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    _run_single_window,
                    strategy=strategy,
                    symbols=symbols,
                    window_closes=wc,
                    asset_classes=asset_classes,
                    lookback_days=lookback_days,
                    rebalance_every=rebalance_every,
                    return_model=return_model,
                    mean_shrinkage=mean_shrinkage,
                    momentum_window=momentum_window,
                    opt_config=opt_config,
                    asset_class_matrix=asset_class_matrix,
                    top_k=top_k,
                    absolute_threshold=absolute_threshold,
                    weighting=weighting,
                    softmax_temperature=softmax_temperature,
                )
                for wc in window_args
            ]
            for future in futures:
                window_results.append(future.result())
    except (NotImplementedError, PermissionError, OSError):
        for wc in window_args:
            window_results.append(
                _run_single_window(
                    strategy=strategy,
                    symbols=symbols,
                    window_closes=wc,
                    asset_classes=asset_classes,
                    lookback_days=lookback_days,
                    rebalance_every=rebalance_every,
                    return_model=return_model,
                    mean_shrinkage=mean_shrinkage,
                    momentum_window=momentum_window,
                    opt_config=opt_config,
                    asset_class_matrix=asset_class_matrix,
                    top_k=top_k,
                    absolute_threshold=absolute_threshold,
                    weighting=weighting,
                    softmax_temperature=softmax_temperature,
                )
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
