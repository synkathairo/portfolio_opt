from __future__ import annotations

import argparse
import bisect
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import exchange_calendars as xcals
import numpy as np

from .alpaca_interface import AlpacaClient, format_order_plans
from .backtest import (
    TRADING_DAYS_PER_YEAR,
    compute_dual_momentum_weights,
    compute_factor_momentum_weights,
    compute_protective_momentum_weights,
    rolling_window_comparison,
    run_backtest,
    run_dual_momentum_backtest,
    run_factor_momentum_backtest,
    run_fixed_weight_benchmark,
    run_protective_momentum_backtest,
)
from .config import AlpacaConfig, OptimizationConfig
from .estimation import estimate_inputs_from_momentum, estimate_inputs_from_prices
from .execution import submit_rebalance_sell_first
from .black_litterman import estimate_inputs_from_black_litterman
from .market_data import load_close_history
from .risk_parity import estimate_inputs_risk_parity
from .model import ModelInputs, load_model_inputs
from .optimizer import effective_turnover_penalty, optimize_weights, project_weights
from .rebalance import build_order_plan, build_trailing_stop_plan, current_weights
from .runtime import configure_local_cache_dirs
from utils.fetch_tickers import (
    DEFAULT_TICKER_BASKET,
    fetch_ticker_dict,
    filter_tickers_before,
)
from cvxportfolio_impl.backtest import format_backtest as format_cvxportfolio_backtest
from cvxportfolio_impl.backtest import run_cvxportfolio_current_target
from cvxportfolio_impl.cli import run_from_args as run_cvxportfolio_from_args

configure_local_cache_dirs()

_DYNAMIC_UNIVERSE_CACHE_FORMAT = 1


def _dynamic_universe_cache_paths(
    ticker_basket: list[str],
    cache_dir: str | Path,
) -> tuple[Path, Path]:
    key_payload = {
        "format": _DYNAMIC_UNIVERSE_CACHE_FORMAT,
        "ticker_basket": ticker_basket,
    }
    digest = sha256(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    model_path = Path(cache_dir) / f"dynamic_universe-{digest}.json"
    meta_path = model_path.with_name(f"{model_path.stem}.meta.json")
    return model_path, meta_path


def _dynamic_universe_payload(fetched: dict[str, Any]) -> dict[str, Any]:
    symbols = fetched.get("symbols")
    asset_classes = fetched.get("asset_classes", {})
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("Dynamic universe fetch returned no symbols.")
    if any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        raise ValueError("Dynamic universe fetch returned invalid symbols.")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Dynamic universe fetch returned duplicate symbols.")
    if not isinstance(asset_classes, dict):
        raise ValueError("Dynamic universe fetch returned invalid asset classes.")
    unknown_asset_class_symbols = sorted(set(asset_classes) - set(symbols))
    if unknown_asset_class_symbols:
        raise ValueError(
            "Dynamic universe asset classes reference unknown symbols: "
            f"{unknown_asset_class_symbols}"
        )
    return {
        "symbols": list(symbols),
        "asset_classes": dict(asset_classes),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(path)


def _write_dynamic_universe_cache(
    fetched: dict[str, Any],
    *,
    ticker_basket: list[str],
    cache_dir: str | Path,
) -> None:
    model_payload = _dynamic_universe_payload(fetched)
    model_path, meta_path = _dynamic_universe_cache_paths(ticker_basket, cache_dir)
    fetched_at = datetime.now(UTC).isoformat()
    meta_payload: dict[str, Any] = {
        "kind": "dynamic_universe_cache",
        "format": _DYNAMIC_UNIVERSE_CACHE_FORMAT,
        "fetched_at": fetched_at,
        "ticker_basket": ticker_basket,
        "symbol_count": len(model_payload["symbols"]),
    }
    _write_json_atomic(model_path, model_payload)
    _write_json_atomic(meta_path, meta_payload)


def _read_dynamic_universe_cache(
    *,
    ticker_basket: list[str],
    cache_dir: str | Path,
    max_age_days: float,
) -> dict[str, Any]:
    model_path, meta_path = _dynamic_universe_cache_paths(ticker_basket, cache_dir)
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No cached dynamic universe exists at {model_path}")

    meta = json.loads(meta_path.read_text())
    fetched_at_raw = meta.get("fetched_at") if isinstance(meta, dict) else None
    if not isinstance(fetched_at_raw, str):
        raise ValueError(
            f"Cached dynamic universe metadata is missing fetched_at: {meta_path}"
        )
    fetched_at = datetime.fromisoformat(fetched_at_raw)
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=UTC)
    age = datetime.now(UTC) - fetched_at.astimezone(UTC)
    if age > timedelta(days=max_age_days):
        raise ValueError(
            f"Cached dynamic universe is {age.days} days old, exceeding "
            f"--max-stale-dynamic-universe-days={max_age_days}."
        )

    cached_model = load_model_inputs(model_path)
    return {
        "symbols": cached_model.symbols,
        "asset_classes": cached_model.asset_classes,
    }


def _calculate_trading_date_offset(trading_days: int) -> datetime:
    """Return the date exactly `trading_days` trading sessions ago."""
    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions

    # Convert index to list of datetime objects for easy bisect
    session_dates: list[datetime] = sessions.tolist()
    # exchange_calendars returns naive timestamps (no timezone)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Find the index of the last trading day on or before today
    idx = bisect.bisect_right(session_dates, today)
    if idx == len(session_dates) or session_dates[idx] > today:
        idx -= 1

    cutoff_idx = idx - trading_days
    if cutoff_idx < 0:
        raise ValueError(
            f"Cannot look back {trading_days} trading days; not enough history in calendar."
        )
    return cast(datetime, session_dates[cutoff_idx])


def _resolve_model_inputs(
    *,
    args: argparse.Namespace,
    alpaca: AlpacaClient | None,
    dynamic_universe_cache_dir: str,
    allow_stale_dynamic_universe: bool,
    max_stale_dynamic_universe_days: float,
) -> ModelInputs:
    if not args.dynamic_universe:
        return load_model_inputs(args.model)

    ticker_basket = (
        list(args.ticker_basket) if args.ticker_basket else list(DEFAULT_TICKER_BASKET)
    )
    try:
        fetched = fetch_ticker_dict(ticker_basket=ticker_basket)
        _write_dynamic_universe_cache(
            fetched,
            ticker_basket=ticker_basket,
            cache_dir=dynamic_universe_cache_dir,
        )
    except Exception as exc:
        if not allow_stale_dynamic_universe:
            raise
        try:
            fetched = _read_dynamic_universe_cache(
                ticker_basket=ticker_basket,
                cache_dir=dynamic_universe_cache_dir,
                max_age_days=max_stale_dynamic_universe_days,
            )
        except Exception as cache_exc:
            raise RuntimeError(
                "Dynamic universe fetch failed and no usable stale cache was found."
            ) from cache_exc
        print(
            "WARNING: using stale dynamic universe cache after fresh fetch failed: "
            f"{exc}",
            file=sys.stderr,
        )

    dynamic_symbols = set(fetched["symbols"])
    asset_classes = dict(fetched["asset_classes"])

    client = alpaca if alpaca is not None else AlpacaClient(AlpacaConfig.from_env())
    positions = client.get_positions(
        use_cache=args.use_cache,
        refresh_cache=args.refresh_cache,
        offline=args.offline,
    )
    held_symbols = {position.symbol for position in positions}
    all_symbols = sorted(dynamic_symbols | held_symbols)

    for symbol in held_symbols:
        if symbol not in asset_classes:
            asset_classes[symbol] = f"{symbol} (Unknown)"

    cutoff = None
    if args.filter_before:
        cutoff = datetime.strptime(args.filter_before, "%Y-%m-%d")
    else:
        try:
            cutoff = _calculate_trading_date_offset(args.lookback_days)
        except ValueError:
            pass

    if cutoff:
        all_symbols = filter_tickers_before(all_symbols, cutoff)

    return ModelInputs(
        symbols=all_symbols,
        expected_returns=None,
        covariance=None,
        asset_classes=asset_classes,
        class_min_weights={},
        class_max_weights={},
    )


def _emit_rebalance_result(
    *,
    args: argparse.Namespace,
    alpaca: AlpacaClient,
    model: ModelInputs,
    target_weights: np.ndarray,
    estimation_metadata: dict[str, Any],
    opt_config: OptimizationConfig,
    account,
    positions,
    existing_weights: np.ndarray,
    expected_returns: np.ndarray | None = None,
) -> None:
    target_weights = clean_weights(target_weights)
    target_cash_weight = max(0.0, 1.0 - float(target_weights.sum()))
    current_cash_weight = max(0.0, 1.0 - float(existing_weights.sum()))
    realized_turnover = float(np.abs(target_weights - existing_weights).sum())
    current_weights_clean = clean_weights(existing_weights)
    scaled_turnover_penalty = effective_turnover_penalty(opt_config, existing_weights)

    open_orders = alpaca.get_open_orders()
    symbols_with_open_orders = {
        str(order.get("symbol"))
        for order in open_orders
        if order.get("symbol") in model.symbols
    }
    symbols_needing_prices = [
        symbol
        for symbol, target_weight, current_weight in zip(
            model.symbols, target_weights, existing_weights, strict=True
        )
        if abs(float(target_weight - current_weight)) >= args.rebalance_threshold
        or symbol in symbols_with_open_orders
    ]
    latest_prices = (
        alpaca.get_latest_prices(
            symbols_needing_prices,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )
        if symbols_needing_prices
        else {}
    )

    plan = build_order_plan(
        symbols=model.symbols,
        target_weights=target_weights.tolist(),
        account=account,
        positions=positions,
        latest_prices=latest_prices,
        config=opt_config,
        open_orders=open_orders,
    )
    changed_symbols = [item.symbol for item in plan]
    trailing_stop_cancellations = (
        [
            {
                "id": order.get("id"),
                "symbol": order.get("symbol"),
                "qty": order.get("qty"),
                "trail_percent": order.get("trail_percent"),
            }
            for order in open_orders
            if order.get("symbol") in changed_symbols
            and order.get("type") == "trailing_stop"
            and order.get("side") == "sell"
        ]
        if args.trailing_stop is not None
        else []
    )
    trailing_stop_plan_result = (
        build_trailing_stop_plan(
            symbols=model.symbols,
            target_weights=target_weights.tolist(),
            positions=positions,
            open_orders=open_orders,
            trailing_stop=args.trailing_stop,
            rebalance_threshold=args.rebalance_threshold,
        )
        if args.trailing_stop is not None
        else None
    )
    trailing_stop_plan = (
        trailing_stop_plan_result.orders
        if trailing_stop_plan_result is not None
        else []
    )
    unprotected_trailing_stop_qty = (
        trailing_stop_plan_result.unprotected_qty
        if trailing_stop_plan_result is not None
        else []
    )

    result = {
        "symbols": model.symbols,
        "estimation": estimation_metadata,
        "optimization": {
            "risk_aversion": args.risk_aversion,
            "turnover_penalty": args.turnover_penalty,
            "effective_turnover_penalty": round(float(scaled_turnover_penalty), 6),
            "max_turnover": args.max_turnover,
            "min_weight": args.min_weight,
            "max_weight": args.max_weight,
            "rebalance_threshold": args.rebalance_threshold,
            "allow_cash": args.allow_cash,
            "min_cash_weight": args.min_cash_weight,
            "min_invested_weight": args.min_invested_weight,
            "class_min_weights": model.class_min_weights,
            "class_max_weights": model.class_max_weights,
        },
        "target_weights": {
            symbol: round(float(weight), 6)
            for symbol, weight in zip(model.symbols, target_weights, strict=True)
        },
        "current_weights": {
            symbol: round(float(weight), 6)
            for symbol, weight in zip(model.symbols, current_weights_clean, strict=True)
        },
        "expected_returns": (
            {
                symbol: round(float(value), 6)
                for symbol, value in zip(model.symbols, expected_returns, strict=True)
            }
            if expected_returns is not None
            else None
        ),
        "cash": {
            "current_weight": round(current_cash_weight, 6),
            "target_weight": round(target_cash_weight, 6),
            "buying_power": (
                round(account.buying_power, 2)
                if account.buying_power is not None
                else None
            ),
        },
        "asset_class_exposures": {
            "current": asset_class_exposures(
                symbols=model.symbols,
                weights=current_weights_clean,
                asset_classes=model.asset_classes,
            ),
            "target": asset_class_exposures(
                symbols=model.symbols,
                weights=target_weights,
                asset_classes=model.asset_classes,
            ),
        },
        "turnover": {
            "proposed": round(realized_turnover, 6),
            "max_allowed": args.max_turnover,
        },
        "orders": [asdict(item) for item in plan],
        "trailing_stop_cancellations": trailing_stop_cancellations,
        "trailing_stop_orders": [asdict(item) for item in trailing_stop_plan],
        "unprotected_fractional_trailing_stop_qty": [
            asdict(item) for item in unprotected_trailing_stop_qty
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.submit:
        canceled_trailing_stops = (
            alpaca.cancel_open_trailing_stops(changed_symbols, open_orders=open_orders)
            if args.trailing_stop is not None
            else []
        )
        execution_result = submit_rebalance_sell_first(
            broker=alpaca,
            plan=plan,
            symbols=model.symbols,
            target_weights=target_weights.tolist(),
            config=opt_config,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )
        submitted_orders = execution_result.submitted_orders
        fill_statuses: list[dict] = execution_result.sell_fill_statuses
        submitted_trailing_stops: list[dict] = []
        if args.trailing_stop is not None:
            fill_statuses = alpaca.wait_for_submitted_orders(submitted_orders)
            refreshed_positions = alpaca.get_positions()
            refreshed_open_orders = alpaca.get_open_orders()
            trailing_stop_plan_result = build_trailing_stop_plan(
                symbols=model.symbols,
                target_weights=target_weights.tolist(),
                positions=refreshed_positions,
                open_orders=refreshed_open_orders,
                trailing_stop=args.trailing_stop,
                rebalance_threshold=args.rebalance_threshold,
            )
            trailing_stop_plan = trailing_stop_plan_result.orders
            unprotected_trailing_stop_qty = trailing_stop_plan_result.unprotected_qty
            submitted_trailing_stops = alpaca.submit_trailing_stop_plan(
                trailing_stop_plan
            )
        print(format_order_plans(plan))
        if args.trailing_stop is not None:
            print(
                json.dumps(
                    {
                        "order_fill_statuses": fill_statuses,
                        "canceled_trailing_stops": canceled_trailing_stops,
                        "trailing_stop_orders": [
                            asdict(item) for item in trailing_stop_plan
                        ],
                        "unprotected_fractional_trailing_stop_qty": [
                            asdict(item) for item in unprotected_trailing_stop_qty
                        ],
                        "submitted_trailing_stops": submitted_trailing_stops,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )


def _run_sweep_point(
    *,
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    lookback_days: int,
    rebalance_every: int,
    return_model: str,
    mean_shrinkage: float,
    min_weight: float,
    max_weight: float,
    rebalance_threshold: float,
    max_turnover: float | None,
    class_min_weights: dict[str, float],
    class_max_weights: dict[str, float],
    risk_aversion: float,
    min_cash_weight: float,
    min_invested_weight: float,
    turnover_penalty: float,
    momentum_window: int,
    asset_class_matrix: np.ndarray | None,
    trading_days_per_year: int,
) -> dict[str, float | int | dict[str, float]] | tuple[str, str]:
    """Run a single sweep parameter combination. Returns result or error tuple."""
    sweep_config = OptimizationConfig(
        risk_aversion=risk_aversion,
        min_weight=min_weight,
        max_weight=max_weight,
        rebalance_threshold=rebalance_threshold,
        turnover_penalty=turnover_penalty,
        force_full_investment=False,
        min_cash_weight=min_cash_weight,
        max_turnover=max_turnover,
        min_invested_weight=min_invested_weight,
        class_min_weights=class_min_weights,
        class_max_weights=class_max_weights,
    )
    try:
        sweep_backtest = run_backtest(
            symbols=symbols,
            closes_by_symbol=closes_by_symbol,
            lookback_days=lookback_days,
            rebalance_every=rebalance_every,
            return_model=return_model,
            mean_shrinkage=mean_shrinkage,
            momentum_window=momentum_window,
            opt_config=sweep_config,
            asset_class_matrix=asset_class_matrix,
            trading_days_per_year=trading_days_per_year,
        )
    except RuntimeError as exc:
        return (
            "error",
            json.dumps(
                {
                    "risk_aversion": risk_aversion,
                    "min_cash_weight": min_cash_weight,
                    "min_invested_weight": min_invested_weight,
                    "turnover_penalty": turnover_penalty,
                    "momentum_window": momentum_window,
                    "reason": str(exc),
                }
            ),
        )
    return {
        "risk_aversion": risk_aversion,
        "min_cash_weight": min_cash_weight,
        "min_invested_weight": min_invested_weight,
        "turnover_penalty": turnover_penalty,
        "momentum_window": momentum_window,
        "annualized_return": round(float(sweep_backtest.annualized_return), 6),
        "annualized_volatility": round(float(sweep_backtest.annualized_volatility), 6),
        "max_drawdown": round(float(sweep_backtest.max_drawdown), 6),
        "sortino_ratio": round(float(sweep_backtest.sortino_ratio), 6),
        "average_turnover": round(float(sweep_backtest.average_turnover), 6),
    }


def build_asset_class_matrix(
    symbols: list[str],
    asset_classes: dict[str, str],
    class_names: list[str],
) -> np.ndarray:
    if not class_names:
        return np.zeros((0, len(symbols)), dtype=float)
    matrix = np.zeros((len(class_names), len(symbols)), dtype=float)
    class_index = {class_name: index for index, class_name in enumerate(class_names)}
    for symbol_index, symbol in enumerate(symbols):
        class_name = asset_classes.get(symbol)
        if class_name is None:
            continue
        if class_name in class_index:
            matrix[class_index[class_name], symbol_index] = 1.0
    return matrix


def clean_weights(weights: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    cleaned = np.array(weights, dtype=float)
    cleaned[np.abs(cleaned) < tolerance] = 0.0
    return cleaned


def asset_class_exposures(
    symbols: list[str],
    weights: np.ndarray,
    asset_classes: dict[str, str],
) -> dict[str, float]:
    return {
        class_name: round(
            float(
                sum(
                    weights[index]
                    for index, symbol in enumerate(symbols)
                    if asset_classes.get(symbol) == class_name
                )
            ),
            6,
        )
        for class_name in sorted(set(asset_classes.values()))
    }


def _validate_backtest_history(
    closes_by_symbol: dict[str, list[float]],
    *,
    lookback_days: int,
    backtest_days: int,
) -> None:
    if not closes_by_symbol:
        raise ValueError("No price history was loaded for the requested backtest.")

    lengths = {symbol: len(closes) for symbol, closes in closes_by_symbol.items()}
    common_history_days = min(lengths.values())
    required_history_days = lookback_days + backtest_days + 1
    if common_history_days >= required_history_days:
        return

    available_backtest_days = max(0, common_history_days - lookback_days - 1)
    limiting_symbols = sorted(lengths.items(), key=lambda item: item[1])[:5]
    limiting_summary = ", ".join(
        f"{symbol}={length}" for symbol, length in limiting_symbols
    )
    raise ValueError(
        "Not enough common history for the requested backtest. "
        f"Requested {backtest_days} backtest days with lookback {lookback_days}, "
        f"which needs {required_history_days} prices, but only {common_history_days} "
        f"common prices are available across the universe. That supports at most "
        f"{available_backtest_days} backtest days. Shortest histories: {limiting_summary}"
    )


def _json_key(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _value_differs(value: Any, default: Any) -> bool:
    if isinstance(default, float):
        return abs(float(value) - default) > 1e-12
    return value != default


def _validate_cvxportfolio_engine_args(args: argparse.Namespace) -> None:
    backtest_days = getattr(args, "backtest_days", 0)
    submit = bool(getattr(args, "submit", False))
    dry_run = bool(getattr(args, "dry_run", False))
    execution_mode = submit or dry_run
    if backtest_days <= 0 and not execution_mode:
        raise SystemExit(
            "--backtest-engine cvxportfolio requires --backtest-days > 0, "
            "--dry-run, or --submit"
        )
    if backtest_days > 0 and execution_mode:
        raise SystemExit(
            "--backtest-engine cvxportfolio cannot combine --backtest-days with "
            "--dry-run or --submit"
        )
    if backtest_days > 0 and getattr(args, "dynamic_universe", False):
        raise SystemExit("--dynamic-universe cannot be used with --backtest-days")

    unsupported_defaults = [
        ("min_weight", 0.0, "--min-weight"),
        ("allow_cash", False, "--allow-cash"),
        ("estimate_from_history", False, "--estimate-from-history"),
        ("return_model", "sample-mean", "--return-model"),
        ("strategy", "mean-variance", "--strategy"),
        ("top_k", 3, "--top-k"),
        ("factor_top_k", 1, "--factor-top-k"),
        ("dual_momentum_weighting", "equal", "--dual-momentum-weighting"),
        ("softmax_temperature", 0.05, "--softmax-temperature"),
        ("absolute_momentum_threshold", 0.0, "--absolute-momentum-threshold"),
        ("target_vol", None, "--target-vol"),
        ("vol_window", 63, "--vol-window"),
        ("max_single_weight", None, "--max-single-weight"),
        ("basket_opt", None, "--basket-opt"),
        ("basket_risk_aversion", 1.0, "--basket-risk-aversion"),
        ("breadth_min_risky", 0.0, "--breadth-min-risky"),
        ("breadth_max_risky", 1.0, "--breadth-max-risky"),
        ("defensive_weighting", "equal", "--defensive-weighting"),
        ("benchmark", [], "--benchmark"),
        ("rebalance_every", 21, "--rebalance-every"),
    ]
    unsupported = [
        flag
        for name, default, flag in unsupported_defaults
        if _value_differs(getattr(args, name, default), default)
    ]
    if unsupported:
        flags = ", ".join(unsupported)
        raise SystemExit(
            "--backtest-engine cvxportfolio does not support these native-only "
            f"options yet: {flags}"
        )


def _validate_native_engine_args(args: argparse.Namespace) -> None:
    cvxportfolio_defaults = [
        ("core_symbol", None, "--core-symbol"),
        ("core_weight", 0.0, "--core-weight"),
        ("target_volatility", None, "--target-volatility"),
        ("max_leverage", None, "--max-leverage"),
        ("benchmark_symbol", None, "--benchmark-symbol"),
        ("benchmark_weight", 1.0, "--benchmark-weight"),
        ("linear_trade_cost", 0.0, "--linear-trade-cost"),
        ("planning_horizon", 1, "--planning-horizon"),
        ("compare_custom", False, "--compare-custom"),
    ]
    unsupported = [
        flag
        for name, default, flag in cvxportfolio_defaults
        if _value_differs(getattr(args, name, default), default)
    ]
    if unsupported:
        flags = ", ".join(unsupported)
        raise SystemExit(
            "--backtest-engine native does not support these cvxportfolio-only "
            f"options: {flags}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native portfolio rebalances/backtests or the cvxportfolio comparison engine."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Path to model input JSON file.")
    group.add_argument(
        "--dynamic-universe",
        action="store_true",
        help="[native] Fetch current index constituents dynamically instead of using a model file.",
    )
    core_options = parser.add_argument_group("core options")
    native_options = parser.add_argument_group("native engine options")
    data_source_options = parser.add_argument_group("historical data-source options")
    cvxportfolio_options = parser.add_argument_group("cvxportfolio engine options")
    execution_options = parser.add_argument_group("execution and cache options")

    native_options.add_argument(
        "--filter-before",
        type=str,
        default=None,
        help="[native] Only include tickers that started trading before this ISO date (e.g. 2020-01-01).",
    )
    native_options.add_argument(
        "--ticker-basket",
        nargs="*",
        default=[],
        help="[native] Universe components for --dynamic-universe (uses fetch_ticker_dict defaults if empty).",
    )
    native_options.add_argument(
        "--dynamic-universe-cache-dir",
        default=".cache/models",
        help="[native] Directory for latest-known-good generated dynamic universe model caches.",
    )
    native_options.add_argument(
        "--allow-stale-dynamic-universe",
        action="store_true",
        help="[native] Use the previous cached dynamic universe if a fresh fetch fails.",
    )
    native_options.add_argument(
        "--max-stale-dynamic-universe-days",
        type=float,
        default=14.0,
        help="[native] Maximum age in days for --allow-stale-dynamic-universe fallback.",
    )
    core_options.add_argument("--risk-aversion", type=float, default=4.0)
    native_options.add_argument("--min-weight", type=float, default=0.0)
    core_options.add_argument("--max-weight", type=float, default=0.35)
    native_options.add_argument("--rebalance-threshold", type=float, default=0.02)
    native_options.add_argument("--turnover-penalty", type=float, default=0.02)
    native_options.add_argument(
        "--allow-cash",
        action="store_true",
        help="[native] Allow the optimizer to leave part of the portfolio in cash.",
    )
    core_options.add_argument(
        "--min-cash-weight",
        type=float,
        default=0.0,
        help="Minimum cash weight. Native uses this when --allow-cash is enabled; cvxportfolio uses it as an invested exposure cap.",
    )
    native_options.add_argument(
        "--max-turnover",
        type=float,
        default=None,
        help="[native] Hard cap on one-step turnover, measured as sum(abs(target-current)).",
    )
    core_options.add_argument(
        "--min-invested-weight",
        type=float,
        default=0.0,
        help="Minimum total risky-asset weight.",
    )
    native_options.add_argument(
        "--estimate-from-history",
        action="store_true",
        help="[native] Estimate expected returns and covariance from historical closes.",
    )
    core_options.add_argument("--lookback-days", type=int, default=60)
    core_options.add_argument(
        "--mean-shrinkage",
        type=float,
        default=0.75,
        help="Shrink sample mean returns toward zero to reduce estimation noise.",
    )
    native_options.add_argument(
        "--return-model",
        choices=("sample-mean", "momentum", "black-litterman", "risk-parity"),
        default="sample-mean",
        help="[native] How to estimate expected returns when using --estimate-from-history.",
    )
    native_options.add_argument(
        "--strategy",
        choices=(
            "mean-variance",
            "dual-momentum",
            "factor-momentum",
            "protective-momentum",
        ),
        default="mean-variance",
        help="[native] Strategy for live or native backtest rebalancing. Momentum strategies use live prices when --estimate-from-history is set.",
    )
    core_options.add_argument(
        "--momentum-window",
        type=int,
        default=63,
        help="Trailing trading-day window used by the momentum return model.",
    )
    native_options.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="[native] Number of assets to hold in dual momentum mode.",
    )
    native_options.add_argument(
        "--factor-top-k",
        type=int,
        default=1,
        help="[native] Number of top factor/sleeve groups to search in factor momentum mode.",
    )
    native_options.add_argument(
        "--dual-momentum-weighting",
        choices=("equal", "score", "inverse-vol", "softmax"),
        default="equal",
        help="[native] How to weight the selected basket in dual momentum mode.",
    )
    native_options.add_argument(
        "--softmax-temperature",
        type=float,
        default=0.05,
        help="[native] Temperature for softmax weighting in dual momentum mode.",
    )
    native_options.add_argument(
        "--absolute-momentum-threshold",
        type=float,
        default=0.0,
        help="[native] Minimum trailing return required for dual momentum if no cash proxy is present.",
    )
    native_options.add_argument(
        "--target-vol",
        type=float,
        default=None,
        help="[native] Target annualized portfolio volatility for the risky basket (vol targeting).",
    )
    native_options.add_argument(
        "--vol-window",
        type=int,
        default=63,
        help="[native] Trailing trading-day window used to estimate volatility for --target-vol.",
    )
    native_options.add_argument(
        "--max-single-weight",
        type=float,
        default=None,
        help="[native] Maximum weight for any single asset in the dual momentum basket.",
    )
    native_options.add_argument(
        "--trailing-stop",
        type=float,
        default=None,
        help="[native] Trailing stop-loss threshold per asset (e.g. 0.08 to exit an 8%% drawdown from peak).",
    )
    native_options.add_argument(
        "--basket-opt",
        choices=("mean-variance",),
        default=None,
        help="[native] How to size the momentum-selected basket (overrides --dual-momentum-weighting).",
    )
    native_options.add_argument(
        "--basket-risk-aversion",
        type=float,
        default=1.0,
        help="[native] Risk aversion for basket mean-variance optimization.",
    )
    native_options.add_argument(
        "--breadth-min-risky",
        type=float,
        default=0.0,
        help="[native] Minimum total risky exposure for protective momentum.",
    )
    native_options.add_argument(
        "--breadth-max-risky",
        type=float,
        default=1.0,
        help="[native] Maximum total risky exposure for protective momentum.",
    )
    native_options.add_argument(
        "--defensive-weighting",
        choices=("equal",),
        default="equal",
        help="[native] How protective momentum allocates capital not assigned to risky assets.",
    )
    data_source_options.add_argument(
        "--data-source",
        choices=("alpaca", "yfinance", "csv", "csv+yfinance", "stockanalysis"),
        default="alpaca",
        help="Source for historical close data in backtest mode.",
    )
    data_source_options.add_argument(
        "--csv-dir",
        default=".cache/csv",
        help=(
            "Directory of local OHLCV CSV files when --data-source csv is used. "
            "Rows must be symbol,date,open,high,low,close,volume."
        ),
    )
    data_source_options.add_argument(
        "--csv-write-json-cache",
        action="store_true",
        help="Write provider-neutral JSON close caches from --csv-dir before running.",
    )
    data_source_options.add_argument(
        "--stockanalysis-start",
        default="1980-01-01",
        help="Start date for --data-source stockanalysis chart JSON.",
    )
    data_source_options.add_argument(
        "--stockanalysis-end",
        default=None,
        help="End date for --data-source stockanalysis chart JSON. Defaults to today.",
    )
    data_source_options.add_argument(
        "--yfinance-max-workers",
        type=int,
        default=10,
        help="Maximum concurrent yfinance symbol downloads when --data-source yfinance is used.",
    )
    data_source_options.add_argument(
        "--yfinance-retry-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between yfinance retry attempts.",
    )
    data_source_options.add_argument(
        "--yfinance-symbol-delay",
        type=float,
        default=0.02,
        help=(
            "Seconds to wait between yfinance symbol downloads when "
            "--yfinance-max-workers is 1."
        ),
    )
    data_source_options.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help=(
            "[native] Additional benchmark ticker to compare in backtest mode. "
            "Can be repeated, e.g. --benchmark ^HSI."
        ),
    )
    core_options.add_argument(
        "--backtest-days",
        type=int,
        default=0,
        help="Run a simple offline backtest over this many trading days instead of a live rebalance.",
    )
    core_options.add_argument(
        "--backtest-engine",
        choices=("native", "cvxportfolio"),
        default="native",
        help=(
            "Backtest engine to use. 'native' runs the built-in optimizer and "
            "momentum strategies; 'cvxportfolio' runs the experimental "
            "framework-based comparison path."
        ),
    )
    native_options.add_argument(
        "--rebalance-every",
        type=int,
        default=21,
        help="[native] Trading-day interval between rebalances in native backtest mode.",
    )
    core_options.add_argument(
        "--trading-days-per-year",
        type=int,
        default=TRADING_DAYS_PER_YEAR,
        help="Trading sessions per year used for annualized metrics.",
    )
    core_options.add_argument(
        "--rolling-window-days",
        type=int,
        default=0,
        help="If set, compare the strategy to SPY over rolling windows of this many trading days.",
    )
    core_options.add_argument(
        "--rolling-step-days",
        type=int,
        default=21,
        help="Trading-day step between rolling comparison windows.",
    )
    core_options.add_argument(
        "--sweep",
        action="store_true",
        help="Run a simple parameter sweep in backtest mode.",
    )
    core_options.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top parameter combinations to show in sweep mode.",
    )
    cvxportfolio_options.add_argument(
        "--core-symbol",
        default=None,
        help="[cvxportfolio] Required minimum core holding symbol.",
    )
    cvxportfolio_options.add_argument(
        "--core-weight",
        type=float,
        default=0.0,
        help="[cvxportfolio] Required minimum weight for --core-symbol.",
    )
    cvxportfolio_options.add_argument(
        "--target-volatility",
        type=float,
        default=None,
        help="[cvxportfolio] Annualized volatility constraint.",
    )
    cvxportfolio_options.add_argument(
        "--max-leverage",
        type=float,
        default=None,
        help="[cvxportfolio] Maximum leverage constraint.",
    )
    cvxportfolio_options.add_argument(
        "--benchmark-symbol",
        default=None,
        help="[cvxportfolio] Benchmark symbol for benchmark-relative policy metrics.",
    )
    cvxportfolio_options.add_argument(
        "--benchmark-weight",
        type=float,
        default=1.0,
        help="[cvxportfolio] Weight assigned to --benchmark-symbol.",
    )
    cvxportfolio_options.add_argument(
        "--linear-trade-cost",
        type=float,
        default=0.0,
        help="[cvxportfolio] Simple proportional transaction-cost adjustment.",
    )
    cvxportfolio_options.add_argument(
        "--planning-horizon",
        type=int,
        default=1,
        help="[cvxportfolio] Planning horizon; values above 1 use multi-period optimization.",
    )
    cvxportfolio_options.add_argument(
        "--compare-custom",
        action="store_true",
        help=(
            "[cvxportfolio] Compare the cvxportfolio configuration against "
            "the repo's custom baseline preset."
        ),
    )
    execution_options.add_argument(
        "--submit",
        action="store_true",
        help="[native] Submit market orders to Alpaca. Default behavior is dry-run output only.",
    )
    execution_options.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached Alpaca data when available.",
    )
    execution_options.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refresh cached Alpaca data from the API.",
    )
    execution_options.add_argument(
        "--offline",
        action="store_true",
        help="Use cached data only and never call Alpaca.",
    )
    execution_options.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit dry-run mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trading_days_per_year = int(
        getattr(args, "trading_days_per_year", TRADING_DAYS_PER_YEAR)
    )
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive.")
    alpaca: AlpacaClient | None = None
    yfinance_max_workers = getattr(args, "yfinance_max_workers", 10)
    yfinance_retry_delay = getattr(args, "yfinance_retry_delay", 1.0)
    yfinance_symbol_delay = getattr(args, "yfinance_symbol_delay", 0.02)
    csv_dir = getattr(args, "csv_dir", ".cache/csv")
    stockanalysis_start = getattr(args, "stockanalysis_start", "1980-01-01")
    stockanalysis_end = getattr(args, "stockanalysis_end", None)
    factor_top_k = getattr(args, "factor_top_k", 1)
    breadth_min_risky = float(getattr(args, "breadth_min_risky", 0.0))
    breadth_max_risky = float(getattr(args, "breadth_max_risky", 1.0))
    defensive_weighting = str(getattr(args, "defensive_weighting", "equal"))
    dynamic_universe_cache_dir = getattr(
        args,
        "dynamic_universe_cache_dir",
        ".cache/models",
    )
    allow_stale_dynamic_universe = getattr(
        args,
        "allow_stale_dynamic_universe",
        False,
    )
    max_stale_dynamic_universe_days = getattr(
        args,
        "max_stale_dynamic_universe_days",
        14.0,
    )
    if args.trailing_stop is not None and args.trailing_stop <= 0:
        raise ValueError("--trailing-stop must be greater than 0.")
    if factor_top_k <= 0:
        raise ValueError("--factor-top-k must be greater than 0.")
    if not 0.0 <= breadth_min_risky <= 1.0:
        raise ValueError("--breadth-min-risky must be between 0 and 1.")
    if not 0.0 <= breadth_max_risky <= 1.0:
        raise ValueError("--breadth-max-risky must be between 0 and 1.")
    if breadth_min_risky > breadth_max_risky:
        raise ValueError("--breadth-min-risky cannot exceed --breadth-max-risky.")
    if defensive_weighting != "equal":
        raise ValueError("--defensive-weighting currently supports only 'equal'.")
    if yfinance_max_workers < 1:
        raise ValueError("--yfinance-max-workers must be at least 1.")
    if yfinance_retry_delay < 0:
        raise ValueError("--yfinance-retry-delay cannot be negative.")
    if yfinance_symbol_delay < 0:
        raise ValueError("--yfinance-symbol-delay cannot be negative.")
    if max_stale_dynamic_universe_days < 0:
        raise ValueError("--max-stale-dynamic-universe-days cannot be negative.")

    backtest_engine = getattr(args, "backtest_engine", "native")
    if backtest_engine == "cvxportfolio":
        _validate_cvxportfolio_engine_args(args)
        if args.backtest_days > 0:
            print(format_cvxportfolio_backtest(run_cvxportfolio_from_args(args)))
            return
    else:
        _validate_native_engine_args(args)

    # Dynamic universe is only useful for live/dry-run trading, not backtesting
    if args.dynamic_universe and args.backtest_days > 0:
        raise SystemExit("--dynamic-universe cannot be used with --backtest-days")
    if args.dynamic_universe and args.rolling_window_days > 0:
        raise SystemExit("--dynamic-universe cannot be used with --rolling-window-days")

    if args.dynamic_universe and alpaca is None:
        alpaca = AlpacaClient(AlpacaConfig.from_env())
    model = _resolve_model_inputs(
        args=args,
        alpaca=alpaca,
        dynamic_universe_cache_dir=dynamic_universe_cache_dir,
        allow_stale_dynamic_universe=allow_stale_dynamic_universe,
        max_stale_dynamic_universe_days=max_stale_dynamic_universe_days,
    )

    opt_config = OptimizationConfig(
        risk_aversion=args.risk_aversion,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        rebalance_threshold=args.rebalance_threshold,
        turnover_penalty=args.turnover_penalty,
        force_full_investment=not args.allow_cash,
        min_cash_weight=args.min_cash_weight,
        max_turnover=args.max_turnover,
        min_invested_weight=args.min_invested_weight,
        class_min_weights=model.class_min_weights,
        class_max_weights=model.class_max_weights,
    )

    constrained_class_names = list(model.class_min_weights) + [
        name for name in model.class_max_weights if name not in model.class_min_weights
    ]
    asset_class_matrix = build_asset_class_matrix(
        symbols=model.symbols,
        asset_classes=model.asset_classes,
        class_names=constrained_class_names,
    )

    if args.backtest_days > 0:
        total_days = args.lookback_days + args.backtest_days + 1
        benchmark_symbols = [
            symbol for symbol in getattr(args, "benchmark", []) if symbol
        ]
        close_history = load_close_history(
            symbols=model.symbols,
            total_days=total_days,
            data_source=args.data_source,
            benchmark_symbols=benchmark_symbols,
            alpaca=alpaca,
            csv_dir=csv_dir,
            csv_write_json_cache=getattr(args, "csv_write_json_cache", False),
            stockanalysis_start=stockanalysis_start,
            stockanalysis_end=stockanalysis_end,
            yfinance_max_workers=yfinance_max_workers,
            yfinance_retry_delay=yfinance_retry_delay,
            yfinance_symbol_delay=yfinance_symbol_delay,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )
        closes_by_symbol = close_history.closes_by_symbol
        closes_for_benchmarks = close_history.benchmark_closes_by_symbol
        symbols_for_benchmarks = close_history.benchmark_symbols_universe
        _validate_backtest_history(
            closes_by_symbol,
            lookback_days=args.lookback_days,
            backtest_days=args.backtest_days,
        )
        if args.sweep:
            if args.strategy in {
                "dual-momentum",
                "factor-momentum",
                "protective-momentum",
            }:
                raise ValueError(
                    "Sweep mode is only implemented for the mean-variance path."
                )
            risk_grid = [1.0, 2.0, 4.0]
            cash_grid = [0.05, 0.10, 0.20]
            invested_grid = [0.20, 0.30, 0.40]
            turnover_grid = [0.02, 0.05]
            momentum_grid = [42, 63, 84]

            grid_params = list(
                itertools.product(
                    risk_grid,
                    cash_grid,
                    invested_grid,
                    turnover_grid,
                    momentum_grid,
                )
            )

            try:
                with ProcessPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            _run_sweep_point,
                            symbols=model.symbols,
                            closes_by_symbol=closes_by_symbol,
                            lookback_days=args.lookback_days,
                            rebalance_every=args.rebalance_every,
                            return_model=args.return_model,
                            mean_shrinkage=args.mean_shrinkage,
                            min_weight=args.min_weight,
                            max_weight=args.max_weight,
                            rebalance_threshold=args.rebalance_threshold,
                            max_turnover=args.max_turnover,
                            class_min_weights=model.class_min_weights,
                            class_max_weights=model.class_max_weights,
                            risk_aversion=risk_aversion,
                            min_cash_weight=min_cash_weight,
                            min_invested_weight=min_invested_weight,
                            turnover_penalty=turnover_penalty,
                            momentum_window=momentum_window,
                            asset_class_matrix=(
                                asset_class_matrix if constrained_class_names else None
                            ),
                            trading_days_per_year=trading_days_per_year,
                        )
                        for risk_aversion, min_cash_weight, min_invested_weight, turnover_penalty, momentum_window in grid_params
                    ]
                    outcomes = [future.result() for future in futures]
            except (NotImplementedError, PermissionError, OSError):
                outcomes = [
                    _run_sweep_point(
                        symbols=model.symbols,
                        closes_by_symbol=closes_by_symbol,
                        lookback_days=args.lookback_days,
                        rebalance_every=args.rebalance_every,
                        return_model=args.return_model,
                        mean_shrinkage=args.mean_shrinkage,
                        min_weight=args.min_weight,
                        max_weight=args.max_weight,
                        rebalance_threshold=args.rebalance_threshold,
                        max_turnover=args.max_turnover,
                        class_min_weights=model.class_min_weights,
                        class_max_weights=model.class_max_weights,
                        risk_aversion=risk_aversion,
                        min_cash_weight=min_cash_weight,
                        min_invested_weight=min_invested_weight,
                        turnover_penalty=turnover_penalty,
                        momentum_window=momentum_window,
                        asset_class_matrix=(
                            asset_class_matrix if constrained_class_names else None
                        ),
                        trading_days_per_year=trading_days_per_year,
                    )
                    for risk_aversion, min_cash_weight, min_invested_weight, turnover_penalty, momentum_window in grid_params
                ]

            results: list[dict[str, float | int | dict[str, float]]] = []
            skipped: list[dict[str, float | int | str]] = []

            for outcome in outcomes:
                if isinstance(outcome, tuple):
                    skipped.append(json.loads(outcome[1]))
                else:
                    results.append(outcome)

            results.sort(
                key=lambda item: (
                    float(item["annualized_return"]),
                    -float(item["max_drawdown"]),
                ),
                reverse=True,
            )
            print(
                json.dumps(
                    {
                        "symbols": model.symbols,
                        "sweep": {
                            "days": args.backtest_days,
                            "rebalance_every": args.rebalance_every,
                            "top_n": args.top_n,
                            "tested": len(results),
                            "skipped": len(skipped),
                            "results": results[: args.top_n],
                            "skipped_examples": skipped[: min(5, len(skipped))],
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return
        if args.strategy == "dual-momentum":
            backtest = run_dual_momentum_backtest(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                asset_classes=model.asset_classes,
                lookback_days=args.lookback_days,
                rebalance_every=args.rebalance_every,
                top_k=args.top_k,
                absolute_threshold=args.absolute_momentum_threshold,
                weighting=args.dual_momentum_weighting,
                softmax_temperature=args.softmax_temperature,
                target_vol=args.target_vol,
                max_single_weight=args.max_single_weight,
                vol_window=args.vol_window,
                trailing_stop=args.trailing_stop,
                basket_opt=args.basket_opt,
                basket_risk_aversion=args.basket_risk_aversion,
                trading_days_per_year=trading_days_per_year,
            )
        elif args.strategy == "factor-momentum":
            backtest = run_factor_momentum_backtest(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                asset_classes=model.asset_classes,
                lookback_days=args.lookback_days,
                rebalance_every=args.rebalance_every,
                top_k=args.top_k,
                factor_top_k=factor_top_k,
                absolute_threshold=args.absolute_momentum_threshold,
                weighting=args.dual_momentum_weighting,
                softmax_temperature=args.softmax_temperature,
                target_vol=args.target_vol,
                max_single_weight=args.max_single_weight,
                vol_window=args.vol_window,
                trailing_stop=args.trailing_stop,
                basket_opt=args.basket_opt,
                basket_risk_aversion=args.basket_risk_aversion,
                trading_days_per_year=trading_days_per_year,
            )
        elif args.strategy == "protective-momentum":
            backtest = run_protective_momentum_backtest(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                asset_classes=model.asset_classes,
                lookback_days=args.lookback_days,
                rebalance_every=args.rebalance_every,
                top_k=args.top_k,
                absolute_threshold=args.absolute_momentum_threshold,
                weighting=args.dual_momentum_weighting,
                softmax_temperature=args.softmax_temperature,
                target_vol=args.target_vol,
                max_single_weight=args.max_single_weight,
                vol_window=args.vol_window,
                basket_opt=args.basket_opt,
                basket_risk_aversion=args.basket_risk_aversion,
                breadth_min_risky=breadth_min_risky,
                breadth_max_risky=breadth_max_risky,
                defensive_weighting=defensive_weighting,
                trading_days_per_year=trading_days_per_year,
            )
        else:
            backtest = run_backtest(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                lookback_days=args.lookback_days,
                rebalance_every=args.rebalance_every,
                return_model=args.return_model,
                mean_shrinkage=args.mean_shrinkage,
                momentum_window=args.momentum_window,
                opt_config=opt_config,
                asset_class_matrix=(
                    asset_class_matrix if constrained_class_names else None
                ),
                trading_days_per_year=trading_days_per_year,
            )
        latest_weights = clean_weights(backtest.latest_weights)
        rolling_comparison = None
        if args.rolling_window_days > 0:
            rolling_comparison = rolling_window_comparison(
                strategy=args.strategy,
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                asset_classes=model.asset_classes,
                lookback_days=args.lookback_days,
                window_days=args.rolling_window_days,
                step_days=args.rolling_step_days,
                rebalance_every=args.rebalance_every,
                return_model=args.return_model,
                mean_shrinkage=args.mean_shrinkage,
                momentum_window=args.momentum_window,
                opt_config=opt_config,
                asset_class_matrix=(
                    asset_class_matrix if constrained_class_names else None
                ),
                top_k=args.top_k,
                factor_top_k=factor_top_k,
                absolute_threshold=args.absolute_momentum_threshold,
                weighting=args.dual_momentum_weighting,
                softmax_temperature=args.softmax_temperature,
                breadth_min_risky=breadth_min_risky,
                breadth_max_risky=breadth_max_risky,
                defensive_weighting=defensive_weighting,
                trading_days_per_year=trading_days_per_year,
            )
        benchmark_results = {
            "spy": run_fixed_weight_benchmark(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                weights_by_symbol={"SPY": 1.0},
                start_day=args.lookback_days,
                trading_days_per_year=trading_days_per_year,
            ),
            "qqq": run_fixed_weight_benchmark(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                weights_by_symbol={"QQQ": 1.0},
                start_day=args.lookback_days,
                trading_days_per_year=trading_days_per_year,
            ),
            "sixty_forty_spy_tlt": run_fixed_weight_benchmark(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                weights_by_symbol={"SPY": 0.6, "TLT": 0.4},
                start_day=args.lookback_days,
                trading_days_per_year=trading_days_per_year,
            ),
            "equal_weight": run_fixed_weight_benchmark(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                weights_by_symbol={
                    symbol: 1.0 / len(model.symbols) for symbol in model.symbols
                },
                start_day=args.lookback_days,
                trading_days_per_year=trading_days_per_year,
            ),
            "half_spy_half_cash": run_fixed_weight_benchmark(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                weights_by_symbol={"SPY": 0.5},
                start_day=args.lookback_days,
                trading_days_per_year=trading_days_per_year,
            ),
        }
        for symbol in benchmark_symbols:
            key = f"benchmark_{_json_key(symbol)}"
            benchmark_results[key] = run_fixed_weight_benchmark(
                symbols=symbols_for_benchmarks,
                closes_by_symbol=closes_for_benchmarks,
                weights_by_symbol={symbol: 1.0},
                start_day=args.lookback_days,
                trading_days_per_year=trading_days_per_year,
            )
        result = {
            "symbols": model.symbols,
            "backtest": {
                "strategy": args.strategy,
                "dual_momentum_weighting": (
                    args.dual_momentum_weighting
                    if args.strategy
                    in {"dual-momentum", "factor-momentum", "protective-momentum"}
                    else None
                ),
                "factor_top_k": (
                    factor_top_k if args.strategy == "factor-momentum" else None
                ),
                "breadth_min_risky": (
                    breadth_min_risky
                    if args.strategy == "protective-momentum"
                    else None
                ),
                "breadth_max_risky": (
                    breadth_max_risky
                    if args.strategy == "protective-momentum"
                    else None
                ),
                "defensive_weighting": (
                    defensive_weighting
                    if args.strategy == "protective-momentum"
                    else None
                ),
                "target_vol": args.target_vol,
                "vol_window": (
                    args.vol_window
                    if args.strategy
                    in {"dual-momentum", "factor-momentum", "protective-momentum"}
                    else None
                ),
                "max_single_weight": args.max_single_weight,
                "basket_opt": args.basket_opt,
                "basket_risk_aversion": (
                    args.basket_risk_aversion
                    if args.strategy
                    in {"dual-momentum", "factor-momentum", "protective-momentum"}
                    and args.basket_opt
                    else None
                ),
                "days": args.backtest_days,
                "rebalance_every": args.rebalance_every,
                "trading_days_per_year": trading_days_per_year,
                "final_value": round(float(backtest.final_value), 6),
                "total_return": round(float(backtest.total_return), 6),
                "annualized_return": round(float(backtest.annualized_return), 6),
                "annualized_volatility": round(
                    float(backtest.annualized_volatility), 6
                ),
                "max_drawdown": round(float(backtest.max_drawdown), 6),
                "sortino_ratio": round(float(backtest.sortino_ratio), 6),
                "rebalance_count": backtest.rebalance_count,
                "average_turnover": round(float(backtest.average_turnover), 6),
                "daily_values": [round(v, 6) for v in backtest.daily_values],
            },
            "latest_target_weights": {
                symbol: round(float(weight), 6)
                for symbol, weight in zip(model.symbols, latest_weights, strict=True)
            },
            "latest_cash_weight": round(max(0.0, 1.0 - float(latest_weights.sum())), 6),
            "latest_asset_class_exposures": asset_class_exposures(
                symbols=model.symbols,
                weights=latest_weights,
                asset_classes=model.asset_classes,
            ),
            "benchmarks": benchmark_results,
        }
        if rolling_comparison is not None:
            result["rolling_vs_spy"] = rolling_comparison
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if alpaca is None:
        alpaca = AlpacaClient(AlpacaConfig.from_env())

    # Even in dry-run mode we fetch live account state for the rebalance path,
    # because the order plan depends on current equity, holdings, and prices.
    account = alpaca.get_account(
        use_cache=args.use_cache, refresh_cache=args.refresh_cache, offline=args.offline
    )
    positions = alpaca.get_positions(
        use_cache=args.use_cache, refresh_cache=args.refresh_cache, offline=args.offline
    )
    existing_weights_map = current_weights(model.symbols, account, positions)
    existing_weights = np.array(
        [existing_weights_map[symbol] for symbol in model.symbols], dtype=float
    )
    target_weights: np.ndarray | None = None
    expected_returns: np.ndarray | None = None
    covariance: np.ndarray | None = None

    if backtest_engine == "cvxportfolio":
        target_weights = run_cvxportfolio_current_target(
            model=model,
            account=account,
            positions=positions,
            lookback_days=args.lookback_days,
            risk_aversion=args.risk_aversion,
            min_cash_weight=args.min_cash_weight,
            min_invested_weight=args.min_invested_weight,
            max_weight=args.max_weight,
            mean_shrinkage=args.mean_shrinkage,
            momentum_window=args.momentum_window,
            core_symbol=args.core_symbol,
            core_weight=args.core_weight,
            target_volatility=args.target_volatility,
            max_leverage=args.max_leverage,
            benchmark_symbol=args.benchmark_symbol,
            benchmark_weight=args.benchmark_weight,
            planning_horizon=args.planning_horizon,
            data_source=args.data_source,
            csv_dir=csv_dir,
            csv_write_json_cache=getattr(args, "csv_write_json_cache", False),
            stockanalysis_start=stockanalysis_start,
            stockanalysis_end=stockanalysis_end,
            yfinance_max_workers=yfinance_max_workers,
            yfinance_retry_delay=yfinance_retry_delay,
            yfinance_symbol_delay=yfinance_symbol_delay,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )
        _emit_rebalance_result(
            args=args,
            alpaca=alpaca,
            model=model,
            target_weights=target_weights,
            estimation_metadata={
                "method": "cvxportfolio_current_target",
                "risk_aversion": args.risk_aversion,
                "mean_shrinkage": args.mean_shrinkage,
                "momentum_window": args.momentum_window,
                "max_weight": args.max_weight,
                "min_cash_weight": args.min_cash_weight,
                "min_invested_weight": args.min_invested_weight,
                "core_symbol": args.core_symbol,
                "core_weight": args.core_weight,
                "target_volatility": args.target_volatility,
                "max_leverage": args.max_leverage,
                "benchmark_symbol": args.benchmark_symbol,
                "benchmark_weight": args.benchmark_weight,
                "linear_trade_cost": args.linear_trade_cost,
                "planning_horizon": args.planning_horizon,
            },
            opt_config=opt_config,
            account=account,
            positions=positions,
            existing_weights=existing_weights,
        )
        return

    if args.estimate_from_history:
        history_days = args.lookback_days
        if args.strategy in {
            "dual-momentum",
            "factor-momentum",
            "protective-momentum",
        }:
            history_days = max(history_days, args.lookback_days + 1)
            if args.target_vol is not None:
                history_days = max(history_days, args.vol_window + 1)
        closes_by_symbol = alpaca.get_daily_closes(
            model.symbols,
            history_days,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )

        # Filter out symbols with insufficient history.
        # This handles cases where Alpaca (IEX data) or Yahoo has gaps,
        # or stocks are new IPOs/SPACs with limited history.
        original_count = len(closes_by_symbol)
        closes_by_symbol = {
            s: bars for s, bars in closes_by_symbol.items() if len(bars) >= history_days
        }
        dropped_count = original_count - len(closes_by_symbol)
        if dropped_count > 0:
            print(
                f"Warning: Dropped {dropped_count} symbols with insufficient history "
                f"(< {history_days} prices).",
                file=sys.stderr,
            )
            # Update model to match available data
            model = ModelInputs(
                symbols=list(closes_by_symbol.keys()),
                expected_returns=model.expected_returns,
                covariance=model.covariance,
                asset_classes=model.asset_classes,
                class_min_weights=model.class_min_weights,
                class_max_weights=model.class_max_weights,
            )
            # Recompute existing weights for the filtered symbol list
            existing_weights_map = current_weights(model.symbols, account, positions)
            existing_weights = np.array(
                [existing_weights_map[symbol] for symbol in model.symbols], dtype=float
            )
            asset_class_matrix = build_asset_class_matrix(
                symbols=model.symbols,
                asset_classes=model.asset_classes,
                class_names=constrained_class_names,
            )

        if args.strategy == "dual-momentum":
            dm_weights = compute_dual_momentum_weights(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                asset_classes=model.asset_classes,
                lookback_days=args.lookback_days,
                top_k=args.top_k,
                weighting=args.dual_momentum_weighting,
                softmax_temperature=args.softmax_temperature,
                absolute_threshold=args.absolute_momentum_threshold,
                basket_opt=args.basket_opt,
                basket_risk_aversion=args.basket_risk_aversion,
                target_vol=args.target_vol,
                max_single_weight=args.max_single_weight,
                vol_window=args.vol_window,
                trailing_stop=None,
            )
            target_weights = np.array(
                [dm_weights[s] for s in model.symbols], dtype=float
            )
            estimation_metadata = {
                "method": "dual_momentum",
                "lookback_days": args.lookback_days,
                "top_k": args.top_k,
                "weighting": args.dual_momentum_weighting,
                "vol_window": args.vol_window,
            }
        elif args.strategy == "factor-momentum":
            fm_weights = compute_factor_momentum_weights(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                asset_classes=model.asset_classes,
                lookback_days=args.lookback_days,
                top_k=args.top_k,
                factor_top_k=factor_top_k,
                weighting=args.dual_momentum_weighting,
                softmax_temperature=args.softmax_temperature,
                absolute_threshold=args.absolute_momentum_threshold,
                basket_opt=args.basket_opt,
                basket_risk_aversion=args.basket_risk_aversion,
                target_vol=args.target_vol,
                max_single_weight=args.max_single_weight,
                vol_window=args.vol_window,
            )
            target_weights = np.array(
                [fm_weights[s] for s in model.symbols], dtype=float
            )
            estimation_metadata = {
                "method": "factor_momentum",
                "lookback_days": args.lookback_days,
                "top_k": args.top_k,
                "factor_top_k": factor_top_k,
                "weighting": args.dual_momentum_weighting,
                "vol_window": args.vol_window,
            }
        elif args.strategy == "protective-momentum":
            pm_weights = compute_protective_momentum_weights(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                asset_classes=model.asset_classes,
                lookback_days=args.lookback_days,
                top_k=args.top_k,
                weighting=args.dual_momentum_weighting,
                softmax_temperature=args.softmax_temperature,
                absolute_threshold=args.absolute_momentum_threshold,
                basket_opt=args.basket_opt,
                basket_risk_aversion=args.basket_risk_aversion,
                target_vol=args.target_vol,
                max_single_weight=args.max_single_weight,
                vol_window=args.vol_window,
                breadth_min_risky=breadth_min_risky,
                breadth_max_risky=breadth_max_risky,
                defensive_weighting=defensive_weighting,
            )
            target_weights = np.array(
                [pm_weights[s] for s in model.symbols], dtype=float
            )
            estimation_metadata = {
                "method": "protective_momentum",
                "lookback_days": args.lookback_days,
                "top_k": args.top_k,
                "weighting": args.dual_momentum_weighting,
                "breadth_min_risky": breadth_min_risky,
                "breadth_max_risky": breadth_max_risky,
                "defensive_weighting": defensive_weighting,
                "vol_window": args.vol_window,
            }
        elif args.return_model == "momentum":
            estimated = estimate_inputs_from_momentum(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                mean_shrinkage=args.mean_shrinkage,
                momentum_window=args.momentum_window,
            )
            expected_returns = estimated.expected_returns
            covariance = estimated.covariance
            estimation_metadata = {
                "method": "alpaca_daily_bars",
                "return_model": args.return_model,
                "lookback_days": args.lookback_days,
                "mean_shrinkage": args.mean_shrinkage,
                "observations": estimated.observations,
                "momentum_window": min(args.momentum_window, args.lookback_days),
            }
        elif args.return_model == "black-litterman":
            estimated = estimate_inputs_from_black_litterman(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                momentum_window=args.momentum_window,
                mean_shrinkage=args.mean_shrinkage,
            )
            expected_returns = estimated.expected_returns
            covariance = estimated.covariance
            estimation_metadata = {
                "method": "black_litterman",
                "lookback_days": args.lookback_days,
                "mean_shrinkage": args.mean_shrinkage,
                "observations": estimated.observations,
                "momentum_window": min(args.momentum_window, args.lookback_days),
            }
        elif args.return_model == "risk-parity":
            estimated = estimate_inputs_risk_parity(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                lookback_days=args.lookback_days,
            )
            target_weights = project_weights(
                target_weights=estimated.weights,
                config=opt_config,
                current_weights=existing_weights,
                asset_class_matrix=(
                    asset_class_matrix if constrained_class_names else None
                ),
            )
            estimation_metadata = {
                "method": "risk_parity",
                "lookback_days": args.lookback_days,
                "observations": estimated.observations,
            }
        else:
            estimated = estimate_inputs_from_prices(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                mean_shrinkage=args.mean_shrinkage,
            )
            expected_returns = estimated.expected_returns
            covariance = estimated.covariance
            estimation_metadata = {
                "method": "alpaca_daily_bars",
                "return_model": args.return_model,
                "lookback_days": args.lookback_days,
                "mean_shrinkage": args.mean_shrinkage,
                "observations": estimated.observations,
            }
    else:
        if model.expected_returns is None or model.covariance is None:
            raise ValueError(
                "Model file must include expected_returns and covariance unless "
                "--estimate-from-history is used."
            )
        expected_returns = model.expected_returns
        covariance = model.covariance
        estimation_metadata = {"method": "model_file"}

    if (
        args.strategy not in {"dual-momentum", "factor-momentum", "protective-momentum"}
        and target_weights is None
    ):
        if expected_returns is None or covariance is None:
            raise RuntimeError("Expected returns and covariance were not estimated.")
        target_weights = optimize_weights(
            expected_returns=expected_returns,
            covariance=covariance,
            config=opt_config,
            current_weights=existing_weights,
            asset_class_matrix=asset_class_matrix if constrained_class_names else None,
        )
    if target_weights is None:
        raise RuntimeError(
            "Failed to produce target weights for the requested strategy."
        )
    _emit_rebalance_result(
        args=args,
        alpaca=alpaca,
        model=model,
        target_weights=target_weights,
        estimation_metadata=estimation_metadata,
        opt_config=opt_config,
        account=account,
        positions=positions,
        existing_weights=existing_weights,
        expected_returns=(
            expected_returns
            if args.strategy
            not in {"dual-momentum", "factor-momentum", "protective-momentum"}
            else None
        ),
    )


if __name__ == "__main__":
    main()
