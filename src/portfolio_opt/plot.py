"""Plot backtest equity curves against a benchmark (default SPY).

Usage:
    # Pipe backtest JSON directly into the plotter:
    uv run portfolio-opt ... --offline | uv run python -m portfolio_opt.plot

    # Or pass a saved JSON file:
    uv run python -m portfolio_opt.plot backtest_result.json

    # Or a backtest log file (JSON lines from --log-file):
    uv run python -m portfolio_opt.py --log-file mylog.jsonl
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

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


def _plot_from_log(log_path: str, save_path: str, benchmark: str = "SPY") -> None:
    # For a log file, we just plot the equity values recorded over time
    equities = []
    for line in Path(log_path).read_text().splitlines():
        entry = json.loads(line)
        if "equity" in entry:
            equities.append(entry["equity"])
    if not equities:
        print("No equity data found in log file.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equities, marker="o", markersize=3, linewidth=1.5)
    ax.set_title("Portfolio Equity Over Time")
    ax.set_ylabel("Account Equity ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved chart to {save_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot portfolio performance")
    parser.add_argument("input", nargs="?", help="JSON result file or .jsonl log file")
    parser.add_argument("--output", default="performance.png", help="Output image path")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker")
    args = parser.parse_args()

    if args.input and args.input.endswith(".jsonl"):
        _plot_from_log(args.input, args.output)
        return

    if args.input:
        result = json.loads(Path(args.input).read_text())
    else:
        result = json.loads(sys.stdin.read())

    _plot_from_result(result, args.output, args.benchmark)


if __name__ == "__main__":
    main()
