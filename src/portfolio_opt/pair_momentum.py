"""Research-only pair-as-asset momentum backtest.

The pair is the synthetic asset. Its two legs are implementation details inside
the pair return; top-level portfolio weights remain nonnegative capital
allocations to pair strategies. This module does not place orders and is not
connected to the live CLI.

Pair formation and pair momentum use disjoint windows: pairs are formed using
data through ``f`` and a signal at ``t`` uses pair returns only from ``f`` to
``t`` with ``t - f >= momentum_min``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .backtest import TRADING_DAYS_PER_YEAR, summarize_return_series

PairMode = Literal["equal-all", "pair-rank", "category-index", "category-pairs"]
PairSignal = Literal["momentum", "reversal"]


@dataclass(frozen=True)
class PairBacktestConfig:
    formation_window: int = 252
    formation_every: int = 126
    momentum_min: int = 63
    rebalance_every: int = 21
    pairs_per_sector: int = 8
    max_pairs_per_leg: int = 2
    min_corr: float = 0.2
    min_beta: float = 0.1
    max_beta: float = 5.0
    pair_cost: float = 0.001
    borrow_rate: float = 0.03
    top_k: int = 10
    category_top_k: int = 3
    pairs_per_category: int = 3
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR

    def __post_init__(self) -> None:
        if self.formation_window < 2:
            raise ValueError("formation_window must be at least 2.")
        if self.formation_every < 1 or self.momentum_min < 1:
            raise ValueError("formation_every and momentum_min must be positive.")
        if self.rebalance_every < 1:
            raise ValueError("rebalance_every must be positive.")
        if self.pairs_per_sector < 1 or self.max_pairs_per_leg < 1:
            raise ValueError("Pair-count limits must be positive.")
        if not 0.0 <= self.min_corr <= 1.0:
            raise ValueError("min_corr must be between 0 and 1.")
        if self.min_beta <= 0.0 or self.max_beta < self.min_beta:
            raise ValueError("Beta bounds must be positive and ordered.")
        if self.pair_cost < 0.0 or self.borrow_rate < 0.0:
            raise ValueError("pair_cost and borrow_rate cannot be negative.")
        if self.top_k < 1 or self.category_top_k < 1 or self.pairs_per_category < 1:
            raise ValueError("Ranking limits must be positive.")
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be positive.")


@dataclass(frozen=True)
class PairSpec:
    long_index: int
    short_index: int
    beta: float


@dataclass(frozen=True)
class PairBacktestResult:
    final_value: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    rebalance_count: int
    average_turnover: float
    daily_values: tuple[float, ...]
    sortino_ratio: float
    latest_pair_weights: dict[str, float]


def pair_spread_returns(
    long_returns: np.ndarray,
    short_returns: np.ndarray,
    beta: float,
    *,
    borrow_rate: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Return a beta-neutral pair return series with gross exposure 1.0."""
    long_values = np.asarray(long_returns, dtype=float)
    short_values = np.asarray(short_returns, dtype=float)
    if long_values.shape != short_values.shape:
        raise ValueError("Pair leg returns must have the same shape.")
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("Pair beta must be positive and finite.")
    if borrow_rate < 0.0 or trading_days_per_year <= 0:
        raise ValueError("Borrow parameters must be nonnegative and positive.")
    short_weight = beta / (1.0 + beta)
    result = (long_values - beta * short_values) / (1.0 + beta)
    if borrow_rate:
        result = result - borrow_rate / trading_days_per_year * short_weight
    return result


def select_pairs(
    returns: np.ndarray,
    sector_indices: dict[str, list[int]],
    formation_index: int,
    config: PairBacktestConfig,
) -> list[PairSpec]:
    """Select low-residual-volatility pairs inside each sector through ``f``."""
    start = max(0, formation_index - config.formation_window)
    if formation_index - start < 2:
        return []
    selected: list[PairSpec] = []
    for indices in sector_indices.values():
        if len(indices) < 2:
            continue
        window = returns[np.ix_(indices, range(start, formation_index))]
        if window.shape[1] < 2:
            continue
        variances = window.var(axis=1, ddof=0)
        valid = variances > 1e-18
        if not np.any(valid):
            continue
        covariance = np.cov(window, ddof=0)
        std = np.sqrt(np.maximum(variances, 1e-18))
        correlations = covariance / np.outer(std, std)
        betas = covariance / variances[None, :]
        residual_scale = np.sqrt(
            np.maximum(variances[:, None] - (covariance**2) / variances[None, :], 0.0)
        )

        candidates: list[tuple[float, int, int, float]] = []
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                if not valid[left] or not valid[right]:
                    continue
                correlation = float(correlations[left, right])
                beta = float(betas[left, right])
                scale = float(residual_scale[left, right])
                if not np.isfinite(correlation) or correlation < config.min_corr:
                    continue
                if not np.isfinite(beta) or beta <= config.min_beta:
                    continue
                beta = float(np.clip(beta, config.min_beta, config.max_beta))
                candidates.append((scale, indices[left], indices[right], beta))

        candidates.sort(key=lambda item: item[0])
        used: dict[int, int] = {}
        count = 0
        for _scale, left, right, beta in candidates:
            if (
                used.get(left, 0) >= config.max_pairs_per_leg
                or used.get(right, 0) >= config.max_pairs_per_leg
            ):
                continue
            selected.append(PairSpec(left, right, beta))
            used[left] = used.get(left, 0) + 1
            used[right] = used.get(right, 0) + 1
            count += 1
            if count >= config.pairs_per_sector:
                break
    return selected


def estimate_pair_beta(
    returns: np.ndarray,
    long_index: int,
    short_index: int,
    formation_index: int,
    config: PairBacktestConfig,
) -> float | None:
    """Estimate a point-in-time OLS beta for a fixed pair."""
    start = max(0, formation_index - config.formation_window)
    if formation_index - start < 2:
        return None
    long_values = np.asarray(returns[long_index, start:formation_index], dtype=float)
    short_values = np.asarray(returns[short_index, start:formation_index], dtype=float)
    variance = float(np.var(short_values, ddof=0))
    if variance <= 1e-18:
        return None
    beta = float(np.cov(long_values, short_values, ddof=0)[0, 1] / variance)
    if not np.isfinite(beta) or beta <= config.min_beta:
        return None
    return float(np.clip(beta, config.min_beta, config.max_beta))


def build_pair_navs(
    returns: np.ndarray,
    pairs: list[tuple[int, int]],
    config: PairBacktestConfig,
    *,
    net: bool = True,
) -> dict[tuple[int, int], np.ndarray]:
    """Build point-in-time pair NAV series for a fixed pair list.

    Betas are re-estimated at each formation date using only data through that
    date. Pair NAVs are flat before the first formation and compound net pair
    returns afterward.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("Returns must be a two-dimensional array.")
    navs: dict[tuple[int, int], np.ndarray] = {}
    for long_index, short_index in pairs:
        if (
            long_index < 0
            or long_index >= values.shape[0]
            or short_index < 0
            or short_index >= values.shape[0]
        ):
            raise ValueError("Pair indices are out of range.")
        pair_values = np.zeros(values.shape[1], dtype=float)
        for formation in range(
            config.formation_window,
            values.shape[1],
            config.formation_every,
        ):
            beta = estimate_pair_beta(
                values, long_index, short_index, formation, config
            )
            if beta is None:
                continue
            end = min(formation + config.formation_every, values.shape[1])
            pair_values[formation:end] = pair_spread_returns(
                values[long_index, formation:end],
                values[short_index, formation:end],
                beta,
                borrow_rate=config.borrow_rate if net else 0.0,
                trading_days_per_year=config.trading_days_per_year,
            )
        navs[(long_index, short_index)] = np.concatenate(
            ([1.0], np.cumprod(1.0 + pair_values))
        )
    return navs


def latest_formation(index: int, config: PairBacktestConfig) -> int | None:
    first = config.formation_window
    eligible = index - config.momentum_min
    if eligible < first:
        return None
    step = (eligible - first) // config.formation_every
    return first + step * config.formation_every


def pair_label(pair: PairSpec, symbols: list[str]) -> str:
    return f"{symbols[pair.long_index]}/{symbols[pair.short_index]}"


def run_pair_momentum_backtest(
    *,
    symbols: list[str],
    returns: np.ndarray,
    sector_indices: dict[str, list[int]],
    config: PairBacktestConfig,
    mode: PairMode = "pair-rank",
    signal: PairSignal = "momentum",
    net: bool = True,
    fixed_pair_indices: list[tuple[int, int]] | None = None,
) -> PairBacktestResult:
    """Backtest pair-as-asset allocation with optional category momentum.

    ``mode="pair-rank"`` ranks all eligible pairs directly. Category modes first
    choose sectors and then rank pairs inside them. ``signal="reversal"`` flips
    the pair score sign.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("Returns must be a two-dimensional array.")
    if len(symbols) != values.shape[0]:
        raise ValueError("Symbols and returns must have the same asset count.")
    if signal not in {"momentum", "reversal"}:
        raise ValueError(f"Unknown pair signal: {signal}")
    if mode not in {"equal-all", "pair-rank", "category-index", "category-pairs"}:
        raise ValueError(f"Unknown pair mode: {mode}")
    for indices in sector_indices.values():
        if any(index < 0 or index >= values.shape[0] for index in indices):
            raise ValueError("Sector indices are out of range.")

    start_index = config.formation_window + config.momentum_min
    if start_index >= values.shape[1]:
        raise ValueError("Not enough return history for the requested windows.")

    formations = range(
        config.formation_window,
        values.shape[1],
        config.formation_every,
    )
    if fixed_pair_indices is None:
        pairs_by_formation = {
            formation: select_pairs(values, sector_indices, formation, config)
            for formation in formations
        }
    else:
        for long_index, short_index in fixed_pair_indices:
            if (
                long_index < 0
                or long_index >= values.shape[0]
                or short_index < 0
                or short_index >= values.shape[0]
            ):
                raise ValueError("Fixed pair indices are out of range.")
        pairs_by_formation = {}
        for formation in formations:
            selected: list[PairSpec] = []
            for long_index, short_index in fixed_pair_indices:
                beta = estimate_pair_beta(
                    values, long_index, short_index, formation, config
                )
                if beta is not None:
                    selected.append(PairSpec(long_index, short_index, beta))
            pairs_by_formation[formation] = selected
    category_nav = {
        sector: np.cumprod(1.0 + values[indices].mean(axis=0))
        for sector, indices in sector_indices.items()
        if indices
    }

    pair_cache: dict[tuple[int, int, float, bool], np.ndarray] = {}

    def get_pair(pair: PairSpec) -> np.ndarray:
        key = (pair.long_index, pair.short_index, round(pair.beta, 6), net)
        if key not in pair_cache:
            pair_cache[key] = pair_spread_returns(
                values[pair.long_index],
                values[pair.short_index],
                pair.beta,
                borrow_rate=config.borrow_rate if net else 0.0,
                trading_days_per_year=config.trading_days_per_year,
            )
        return pair_cache[key]

    weights: dict[PairSpec, float] = {}
    period_returns: list[float] = []
    turnovers: list[float] = []
    active_formation: int | None = None
    active_pairs: list[PairSpec] = []

    for index in range(start_index, values.shape[1]):
        if (index - start_index) % config.rebalance_every != 0:
            period_returns.append(
                sum(weight * get_pair(pair)[index] for pair, weight in weights.items())
            )
            continue

        formation = latest_formation(index, config)
        if formation is not None and formation != active_formation:
            active_formation = formation
            active_pairs = pairs_by_formation[formation]

        candidates: list[tuple[float, PairSpec]] = []
        for pair in active_pairs:
            pair_values = get_pair(pair)
            score = float(np.prod(1.0 + pair_values[formation:index]) - 1.0)
            if signal == "reversal":
                score = -score
            candidates.append((score, pair))

        if not candidates:
            target: dict[PairSpec, float] = {}
        elif mode == "equal-all":
            target = {pair: 1.0 / len(candidates) for _score, pair in candidates}
        elif mode == "pair-rank":
            candidates.sort(key=lambda item: item[0], reverse=True)
            chosen = candidates[: config.top_k]
            target = {pair: 1.0 / len(chosen) for _score, pair in chosen}
        else:
            sector_pairs: dict[str, list[tuple[float, PairSpec]]] = {}
            for score, pair in candidates:
                sector = next(
                    name
                    for name, indices in sector_indices.items()
                    if pair.long_index in indices
                )
                sector_pairs.setdefault(sector, []).append((score, pair))
            sector_scores: dict[str, float] = {}
            for sector, pairs in sector_pairs.items():
                if mode == "category-index":
                    sector_scores[sector] = float(
                        category_nav[sector][index]
                        / category_nav[sector][index - config.momentum_min]
                        - 1.0
                    )
                else:
                    pair_values = [
                        get_pair(pair)[formation:index] for _score, pair in pairs
                    ]
                    sector_scores[sector] = float(
                        np.prod(1.0 + np.mean(pair_values, axis=0)) - 1.0
                    )
            chosen_sectors = sorted(
                sector_scores, key=lambda sector: sector_scores[sector], reverse=True
            )[: config.category_top_k]
            target = {}
            for sector in chosen_sectors:
                chosen_pairs = sorted(
                    sector_pairs[sector], key=lambda item: item[0], reverse=True
                )[: config.pairs_per_category]
                for _score, pair in chosen_pairs:
                    target[pair] = target.get(pair, 0.0) + (
                        1.0 / len(chosen_sectors)
                    ) / len(chosen_pairs)

        turnover = sum(
            abs(target.get(pair, 0.0) - weights.get(pair, 0.0))
            for pair in set(target) | set(weights)
        )
        cost = config.pair_cost * turnover if net else 0.0
        weights = target
        turnovers.append(turnover)
        period_returns.append(
            sum(weight * get_pair(pair)[index] for pair, weight in weights.items())
            - cost
        )

    summary = summarize_return_series(
        np.array(period_returns, dtype=float),
        trading_days_per_year=config.trading_days_per_year,
    )
    daily_values = [1.0]
    for period_return in period_returns:
        daily_values.append(daily_values[-1] * (1.0 + period_return))
    return PairBacktestResult(
        final_value=summary.final_value,
        total_return=summary.total_return,
        annualized_return=summary.annualized_return,
        annualized_volatility=summary.annualized_volatility,
        max_drawdown=summary.max_drawdown,
        rebalance_count=len(turnovers),
        average_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        daily_values=tuple(daily_values),
        sortino_ratio=summary.sortino_ratio,
        latest_pair_weights={
            pair_label(pair, symbols): round(weight, 6)
            for pair, weight in weights.items()
        },
    )
