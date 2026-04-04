from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .alpaca import AlpacaClient, format_order_plans
from .config import AlpacaConfig, OptimizationConfig
from .model import load_model_inputs
from .optimizer import optimize_weights
from .rebalance import build_order_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mean-variance rebalance against Alpaca.")
    parser.add_argument("--model", required=True, help="Path to model input JSON file.")
    parser.add_argument("--risk-aversion", type=float, default=4.0)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--rebalance-threshold", type=float, default=0.02)
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
    )

    target_weights = optimize_weights(
        expected_returns=model.expected_returns,
        covariance=model.covariance,
        config=opt_config,
    )

    alpaca = AlpacaClient(AlpacaConfig.from_env())
    account = alpaca.get_account()
    positions = alpaca.get_positions()
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
        "target_weights": {
            symbol: round(float(weight), 6)
            for symbol, weight in zip(model.symbols, target_weights, strict=True)
        },
        "orders": [asdict(item) for item in plan],
    }
    print(json.dumps(result, indent=2))

    if args.submit:
        alpaca.submit_order_plan(plan)
        print(format_order_plans(plan))


if __name__ == "__main__":
    main()
