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


def _apply_max_single_weight(
    weights: np.ndarray,
    max_single_weight: float | None,
) -> np.ndarray:
    if max_single_weight is None:
        return weights
    if max_single_weight <= 0.0:
        raise ValueError("max_single_weight must be positive.")

    capped = np.maximum(np.array(weights, dtype=float), 0.0)
    total = float(capped.sum())
    if total <= 0.0:
        return capped

    target_total = min(total, max_single_weight * np.count_nonzero(capped > 0.0))
    remaining_total = target_total
    result = np.zeros_like(capped)
    active = capped > 0.0

    while np.any(active):
        active_weights = capped[active]
        active_sum = float(active_weights.sum())
        if active_sum <= 0.0:
            break
        proposed = active_weights / active_sum * remaining_total
        capped_active = proposed >= max_single_weight
        active_indices = np.flatnonzero(active)

        if not np.any(capped_active):
            result[active_indices] = proposed
            break

        capped_indices = active_indices[capped_active]
        result[capped_indices] = max_single_weight
        remaining_total -= max_single_weight * len(capped_indices)
        active[capped_indices] = False
        if remaining_total <= 0.0:
            break

    return result


def _momentum_asset_indices(
    symbols: list[str],
    asset_classes: dict[str, str],
) -> tuple[list[int], list[int], int | None]:
    risky_indices = [
        index
        for index, symbol in enumerate(symbols)
        if not asset_classes.get(symbol, "").startswith("bond")
        and asset_classes.get(symbol) != "cash_like"
    ]
    defensive_indices = [
        index
        for index, symbol in enumerate(symbols)
        if asset_classes.get(symbol, "").startswith("bond")
        or asset_classes.get(symbol) == "cash_like"
    ]
    cash_like_index = next(
        (
            index
            for index, symbol in enumerate(symbols)
            if asset_classes.get(symbol) == "cash_like"
        ),
        None,
    )
    return risky_indices, defensive_indices, cash_like_index


def _momentum_target_weights(
    *,
    symbols: list[str],
    asset_classes: dict[str, str],
    returns: np.ndarray,
    trailing_returns: np.ndarray,
    trailing_volatility: np.ndarray,
    lookback_days: int,
    top_k: int,
    weighting: str,
    softmax_temperature: float,
    absolute_threshold: float,
    basket_opt: str | None,
    basket_risk_aversion: float,
    target_vol: float | None,
    max_single_weight: float | None,
    vol_window: int,
    factor_top_k: int | None,
) -> np.ndarray:
    risky_indices, defensive_indices, cash_like_index = _momentum_asset_indices(
        symbols,
        asset_classes,
    )
    if not risky_indices:
        raise ValueError("Dual momentum requires at least one risky symbol.")

    defensive_floor = (
        float(trailing_returns[cash_like_index])
        if cash_like_index is not None
        else absolute_threshold
    )
    threshold = max(absolute_threshold, defensive_floor)
    candidate_risky_indices = _factor_momentum_candidate_indices(
        risky_indices=risky_indices,
        asset_classes=asset_classes,
        symbols=symbols,
        trailing_returns=trailing_returns,
        factor_top_k=factor_top_k,
        threshold=threshold,
    )
    risky_ranked = sorted(
        (
            (idx, float(trailing_returns[idx]))
            for idx in candidate_risky_indices
            if float(trailing_returns[idx]) > threshold
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

        target_weights = _apply_max_single_weight(target_weights, max_single_weight)

        if target_vol is not None:
            recent_returns_window = returns[:, max(0, returns.shape[1] - vol_window) :]
            risky_indices_active = [
                i for i in range(len(symbols)) if target_weights[i] > 0
            ]
            if risky_indices_active:
                w = target_weights[risky_indices_active]
                recent_risky_returns = recent_returns_window[risky_indices_active]
                portfolio_recent_returns = np.dot(w, recent_risky_returns)
                portfolio_vol = float(
                    np.std(portfolio_recent_returns, ddof=0)
                ) * np.sqrt(TRADING_DAYS_PER_YEAR)
            else:
                portfolio_vol = 0.0

            if portfolio_vol > 0 and portfolio_vol > target_vol:
                target_weights = target_weights * (target_vol / portfolio_vol)
    elif defensive_indices:
        weight = 1.0 / len(defensive_indices)
        for index in defensive_indices:
            target_weights[index] = weight

    return target_weights


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
    factor_top_k: int | None = None,
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

    target_weights = _momentum_target_weights(
        symbols=symbols,
        asset_classes=asset_classes,
        returns=returns,
        trailing_returns=trailing_returns,
        trailing_volatility=trailing_volatility,
        lookback_days=lookback_days,
        top_k=top_k,
        weighting=weighting,
        softmax_temperature=softmax_temperature,
        absolute_threshold=absolute_threshold,
        basket_opt=basket_opt,
        basket_risk_aversion=basket_risk_aversion,
        target_vol=target_vol,
        max_single_weight=max_single_weight,
        vol_window=vol_window,
        factor_top_k=factor_top_k,
    )

    return {s: float(target_weights[i]) for i, s in enumerate(symbols)}


def compute_factor_momentum_weights(
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    asset_classes: dict[str, str],
    lookback_days: int,
    top_k: int,
    factor_top_k: int = 1,
    weighting: str = "equal",
    softmax_temperature: float = 0.05,
    absolute_threshold: float = 0.0,
    basket_opt: str | None = None,
    basket_risk_aversion: float = 1.0,
    target_vol: float | None = None,
    max_single_weight: float | None = None,
    vol_window: int = 63,
) -> dict[str, float]:
    return compute_dual_momentum_weights(
        symbols=symbols,
        closes_by_symbol=closes_by_symbol,
        asset_classes=asset_classes,
        lookback_days=lookback_days,
        top_k=top_k,
        weighting=weighting,
        softmax_temperature=softmax_temperature,
        absolute_threshold=absolute_threshold,
        basket_opt=basket_opt,
        basket_risk_aversion=basket_risk_aversion,
        target_vol=target_vol,
        max_single_weight=max_single_weight,
        vol_window=vol_window,
        factor_top_k=factor_top_k,
    )


def _factor_label(asset_class: str) -> str:
    text = asset_class.strip()
    if text.endswith(")") and "(" in text:
        return text.rsplit("(", 1)[1][:-1].strip()
    return text


def _factor_momentum_candidate_indices(
    *,
    risky_indices: list[int],
    asset_classes: dict[str, str],
    symbols: list[str],
    trailing_returns: np.ndarray,
    factor_top_k: int | None,
    threshold: float,
) -> list[int]:
    if factor_top_k is None:
        return risky_indices
    if factor_top_k <= 0:
        raise ValueError("factor_top_k must be positive.")

    groups: dict[str, list[int]] = {}
    for index in risky_indices:
        factor = _factor_label(asset_classes.get(symbols[index], ""))
        groups.setdefault(factor, []).append(index)

    ranked = sorted(
        (
            (factor, float(np.mean(trailing_returns[indices])))
            for factor, indices in groups.items()
            if indices and float(np.mean(trailing_returns[indices])) > threshold
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    selected_factors = {factor for factor, _score in ranked[:factor_top_k]}
    return [
        index
        for index in risky_indices
        if _factor_label(asset_classes.get(symbols[index], "")) in selected_factors
    ]


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
    factor_top_k: int | None = None,
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

    # Per-asset peak prices since entry, for trailing stop-loss
    asset_peak_price: np.ndarray | None = None

    for step in range(lookback_days, returns.shape[1]):
        if (step - lookback_days) % rebalance_every == 0:
            trailing_returns = (
                price_matrix[:, step] / price_matrix[:, step - lookback_days] - 1.0
            )
            trailing_volatility = returns[:, step - lookback_days : step].std(
                axis=1, ddof=0
            )
            target_weights = _momentum_target_weights(
                symbols=symbols,
                asset_classes=asset_classes,
                returns=returns[:, :step],
                trailing_returns=trailing_returns,
                trailing_volatility=trailing_volatility,
                lookback_days=lookback_days,
                top_k=top_k,
                weighting=weighting,
                softmax_temperature=softmax_temperature,
                absolute_threshold=absolute_threshold,
                basket_opt=basket_opt,
                basket_risk_aversion=basket_risk_aversion,
                target_vol=target_vol,
                max_single_weight=max_single_weight,
                vol_window=vol_window,
                factor_top_k=factor_top_k,
            )

            previous_weights = weights
            turnovers.append(float(np.abs(target_weights - previous_weights).sum()))
            weights = target_weights
            rebalance_count += 1

        # Trailing stop-loss: track peak prices since entry per asset
        if trailing_stop is not None:
            if asset_peak_price is None:
                asset_peak_price = np.zeros(len(symbols), dtype=float)
            for i in range(len(symbols)):
                if weights[i] > 0 and asset_peak_price[i] <= 0.0:
                    asset_peak_price[i] = price_matrix[i, step]
                elif weights[i] <= 0.0:
                    asset_peak_price[i] = 0.0
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


def run_factor_momentum_backtest(
    *,
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    asset_classes: dict[str, str],
    lookback_days: int,
    rebalance_every: int,
    top_k: int,
    factor_top_k: int,
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
    return run_dual_momentum_backtest(
        symbols=symbols,
        closes_by_symbol=closes_by_symbol,
        asset_classes=asset_classes,
        lookback_days=lookback_days,
        rebalance_every=rebalance_every,
        top_k=top_k,
        absolute_threshold=absolute_threshold,
        weighting=weighting,
        softmax_temperature=softmax_temperature,
        target_vol=target_vol,
        max_single_weight=max_single_weight,
        vol_window=vol_window,
        basket_opt=basket_opt,
        basket_risk_aversion=basket_risk_aversion,
        trailing_stop=trailing_stop,
        factor_top_k=factor_top_k,
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
    factor_top_k: int,
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
    elif strategy == "factor-momentum":
        backtest = run_factor_momentum_backtest(
            symbols=symbols,
            closes_by_symbol=window_closes,
            asset_classes=asset_classes,
            lookback_days=lookback_days,
            rebalance_every=rebalance_every,
            top_k=top_k,
            factor_top_k=factor_top_k,
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
    factor_top_k: int = 1,
) -> dict[str, float | int]:
    if window_days <= 0 or step_days <= 0:
        raise ValueError("window_days and step_days must be positive.")
    if "SPY" not in symbols:
        raise ValueError("Rolling-window comparison requires SPY in the universe.")

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
                    factor_top_k=factor_top_k,
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
                    factor_top_k=factor_top_k,
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
