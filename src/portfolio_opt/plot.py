"""Plot backtest equity curves against a benchmark (default SPY).

Usage:
    # Pipe backtest JSON directly into the plotter:
    uv run portfolio-opt ... --offline | uv run python -m portfolio_opt.plot

    # Or pass a saved JSON file:
    uv run python -m portfolio_opt.plot backtest_result.json

    # Or fetch live portfolio history from Alpaca:
    uv run python -m portfolio_opt.plot --alpaca-history
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from .alpaca import AlpacaClient
from .config import AlpacaConfig

matplotlib.use("Agg")


def _plot_from_result(result: dict, save_path: str, benchmark: str = "SPY") -> None:
    daily_values = result.get("backtest", {}).get("daily_values", [])
    if not daily_values:
        print("No daily_values found in result.", file=sys.stderr)
        return

    n = len(daily_values)
    # Generate a business day range ending today so the x-axis shows dates
    end_date = pd.Timestamp(datetime.now()).normalize()
    dates = pd.bdate_range(end=end_date, periods=n)

    # Fetch benchmark for the same date range
    ticker = yf.Ticker(benchmark)
    hist = ticker.history(start=dates[0] - timedelta(days=5), end=end_date + timedelta(days=1))
    if hist.empty or "Close" not in hist.columns:
        # Fallback: just fetch the last 'n' days if the exact date range fails
        hist = ticker.history(period=f"{n + 10}d")

    if hist.empty:
        print(f"Could not fetch {benchmark} data from yfinance.", file=sys.stderr)
        return

    # Strip timezone from SPY data so it aligns with our naive dates
    hist = hist.copy()
    hist.index = hist.index.tz_localize(None)

    # Align and fill missing data
    bench_series = hist["Close"].reindex(dates).ffill().bfill()
    bench_values = bench_series.values
    bench_values = bench_values / bench_values[0]

    # Ensure lengths match
    common_len = min(len(daily_values), len(bench_values), len(dates))
    dates = dates[:common_len]
    daily_values = daily_values[:common_len]
    bench_values = bench_values[:common_len]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, daily_values, label="Strategy", linewidth=2)
    ax.plot(dates, bench_values, label=benchmark, linestyle="--", alpha=0.7)

    ax.set_title("Strategy vs Benchmark")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Growth of $1")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved chart to {save_path}")


def _plot_from_alpaca_history(
    save_path: str,
    benchmark: str = "SPY",
    period: str = "1M",
    timeframe: str = "1D",
) -> None:
    """Fetch portfolio history from Alpaca and plot against benchmark."""
    alpaca = AlpacaClient(AlpacaConfig.from_env())
    query = urlencode({"period": period, "timeframe": timeframe})
    payload = alpaca._request_json("GET", f"/v2/account/portfolio/history?{query}")

    timestamps = payload.get("timestamp", [])
    equity_curve = payload.get("equity", [])
    if not timestamps or not equity_curve:
        print("No portfolio history found. Is your account funded?", file=sys.stderr)
        return

    # Convert millisecond timestamps to datetime
    dates = pd.to_datetime([t / 1000 for t in timestamps], unit="s")
    # Normalize to start at 1.0
    equity_values = [e / equity_curve[0] if equity_curve[0] > 0 else 1.0 for e in equity_curve]

    # Fetch benchmark for the same date range
    ticker = yf.Ticker(benchmark)
    start = dates.min() - timedelta(days=5)
    end = dates.max() + timedelta(days=1)
    hist = ticker.history(start=start, end=end)
    if hist.empty:
        print(f"Could not fetch {benchmark} data from yfinance.", file=sys.stderr)
        return

    hist = hist.copy()
    hist.index = hist.index.tz_localize(None)
    bench_series = hist["Close"].reindex(dates.normalize()).ffill().bfill()
    bench_values = bench_series.values / bench_series.values[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, equity_values, label="Portfolio", linewidth=2)
    ax.plot(dates, bench_values, label=benchmark, linestyle="--", alpha=0.7)
    ax.set_title("Live Portfolio vs Benchmark")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Growth of $1")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved chart to {save_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot portfolio performance")
    parser.add_argument("input", nargs="?", help="JSON result file from backtest")
    parser.add_argument("--output", default="performance.png", help="Output image path")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker")
    parser.add_argument(
        "--alpaca-history",
        action="store_true",
        help="Fetch live portfolio history from Alpaca API and plot it.",
    )
    parser.add_argument("--period", default="1M", help="Alpaca history period (e.g. 1M, 3M, 1Y).")
    args = parser.parse_args()

    if args.alpaca_history:
        _plot_from_alpaca_history(args.output, args.benchmark, args.period)
        return

    if args.input:
        result = json.loads(Path(args.input).read_text())
    else:
        result = json.loads(sys.stdin.read())

    _plot_from_result(result, args.output, args.benchmark)


if __name__ == "__main__":
    main()
