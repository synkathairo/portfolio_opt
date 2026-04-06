from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from .alpaca import AlpacaClient, format_order_plans
from .config import AlpacaConfig, OptimizationConfig
from .estimation import estimate_inputs_from_momentum, estimate_inputs_from_prices
from .model import load_model_inputs
from .optimizer import effective_turnover_penalty, optimize_weights
from .rebalance import build_order_plan, current_weights


def build_asset_class_matrix(
    symbols: list[str],
    asset_classes: dict[str, str],
    class_names: list[str],
) -> np.ndarray:
    if not class_names:
        return np.zeros((0, len(symbols)), dtype=float)
    matrix = np.zeros((len(class_names), len(symbols)), dtype=float)
    for symbol_index, symbol in enumerate(symbols):
        class_name = asset_classes.get(symbol)
        if class_name is None:
            continue
        matrix[class_names.index(class_name), symbol_index] = 1.0
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mean-variance rebalance against Alpaca.")
    parser.add_argument("--model", required=True, help="Path to model input JSON file.")
    parser.add_argument("--risk-aversion", type=float, default=4.0)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--rebalance-threshold", type=float, default=0.02)
    parser.add_argument("--turnover-penalty", type=float, default=0.02)
    parser.add_argument(
        "--allow-cash",
        action="store_true",
        help="Allow the optimizer to leave part of the portfolio in cash.",
    )
    parser.add_argument(
        "--min-cash-weight",
        type=float,
        default=0.0,
        help="Minimum cash weight to hold when --allow-cash is enabled.",
    )
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=None,
        help="Hard cap on one-step turnover, measured as sum(abs(target-current)).",
    )
    parser.add_argument(
        "--min-invested-weight",
        type=float,
        default=0.0,
        help="Minimum total risky-asset weight when cash is allowed.",
    )
    parser.add_argument(
        "--estimate-from-history",
        action="store_true",
        help="Estimate expected returns and covariance from Alpaca daily bars.",
    )
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument(
        "--mean-shrinkage",
        type=float,
        default=0.75,
        help="Shrink sample mean returns toward zero to reduce estimation noise.",
    )
    parser.add_argument(
        "--return-model",
        choices=("sample-mean", "momentum"),
        default="sample-mean",
        help="How to estimate expected returns when using --estimate-from-history.",
    )
    parser.add_argument(
        "--momentum-window",
        type=int,
        default=63,
        help="Trailing trading-day window used by the momentum return model.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit market orders to Alpaca. Default behavior is dry-run output only.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Explicit dry-run mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model_inputs(args.model)
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

    # The optimizer is intentionally decoupled from data acquisition so the
    # expected return and covariance model can be replaced later.
    alpaca = AlpacaClient(AlpacaConfig.from_env())
    # Even in dry-run mode we fetch live account state, because the rebalance
    # plan depends on current equity, holdings, and market prices.
    account = alpaca.get_account()
    positions = alpaca.get_positions()
    existing_weights_map = current_weights(model.symbols, account, positions)
    existing_weights = np.array([existing_weights_map[symbol] for symbol in model.symbols], dtype=float)
    scaled_turnover_penalty = effective_turnover_penalty(opt_config, existing_weights)
    constrained_class_names = list(model.class_min_weights) + [
        name for name in model.class_max_weights if name not in model.class_min_weights
    ]
    asset_class_matrix = build_asset_class_matrix(
        symbols=model.symbols,
        asset_classes=model.asset_classes,
        class_names=constrained_class_names,
    )

    if args.estimate_from_history:
        closes_by_symbol = alpaca.get_daily_closes(model.symbols, args.lookback_days)
        if args.return_model == "momentum":
            estimated = estimate_inputs_from_momentum(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                mean_shrinkage=args.mean_shrinkage,
                momentum_window=args.momentum_window,
            )
        else:
            estimated = estimate_inputs_from_prices(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                mean_shrinkage=args.mean_shrinkage,
            )
        # Both return models reuse the same covariance estimate from recent
        # daily returns; only the expected return signal changes.
        expected_returns = estimated.expected_returns
        covariance = estimated.covariance
        estimation_metadata = {
            "method": "alpaca_daily_bars",
            "return_model": args.return_model,
            "lookback_days": args.lookback_days,
            "mean_shrinkage": args.mean_shrinkage,
            "observations": estimated.observations,
        }
        if args.return_model == "momentum":
            estimation_metadata["momentum_window"] = min(args.momentum_window, args.lookback_days - 1)
    else:
        if model.expected_returns is None or model.covariance is None:
            raise ValueError(
                "Model file must include expected_returns and covariance unless "
                "--estimate-from-history is used."
            )
        expected_returns = model.expected_returns
        covariance = model.covariance
        estimation_metadata = {"method": "model_file"}

    target_weights = optimize_weights(
        expected_returns=expected_returns,
        covariance=covariance,
        config=opt_config,
        current_weights=existing_weights,
        asset_class_matrix=asset_class_matrix if constrained_class_names else None,
    )
    target_cash_weight = max(0.0, 1.0 - float(target_weights.sum()))
    current_cash_weight = max(0.0, 1.0 - float(existing_weights.sum()))
    realized_turnover = float(np.abs(target_weights - existing_weights).sum())
    class_exposures = {
        class_name: round(
            float(sum(target_weights[index] for index, symbol in enumerate(model.symbols) if model.asset_classes.get(symbol) == class_name)),
            6,
        )
        for class_name in sorted(set(model.asset_classes.values()))
    }

    # Convert target weights into dollar notional orders using the latest
    # available prices and a minimum rebalance threshold.
    latest_prices = alpaca.get_latest_prices(model.symbols)
    plan = build_order_plan(
        symbols=model.symbols,
        target_weights=target_weights.tolist(),
        account=account,
        positions=positions,
        latest_prices=latest_prices,
        config=opt_config,
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
            for symbol, weight in zip(model.symbols, existing_weights, strict=True)
        },
        "expected_returns": {
            symbol: round(float(value), 6)
            for symbol, value in zip(model.symbols, expected_returns, strict=True)
        },
        "cash": {
            "current_weight": round(current_cash_weight, 6),
            "target_weight": round(target_cash_weight, 6),
        },
        "asset_class_exposures": class_exposures,
        "turnover": {
            "proposed": round(realized_turnover, 6),
            "max_allowed": args.max_turnover,
        },
        "orders": [asdict(item) for item in plan],
    }
    print(json.dumps(result, indent=2))

    if args.submit:
        alpaca.submit_order_plan(plan)
        print(format_order_plans(plan))


if __name__ == "__main__":
    main()
