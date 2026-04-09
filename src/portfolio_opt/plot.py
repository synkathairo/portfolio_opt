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

from .alpaca_interface import AlpacaClient
from .config import AlpacaConfig

matplotlib.use("Agg")


def _plot_from_result(result: dict, save_path: str, benchmark: str = "SPY") -> None:
    daily_values = result.get("backtest", {}).get("daily_values", [])
    if not daily_values:
        print("No daily_values found in result.", file=sys.stderr)
        return

    n = len(daily_values)
    # Generate a business day range ending today so the x-axis shows dates
    end_date = pd.Timestamp(datetime.now()).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    dates = pd.bdate_range(end=end_date, periods=n)

    # Fetch benchmark for the same date range
    ticker = yf.Ticker(benchmark)
    hist = ticker.history(
        start=dates[0] - timedelta(days=5), end=end_date + timedelta(days=1)
    )
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
    payload = alpaca.get_portfolio_history(period=period, timeframe=timeframe)

    timestamps = payload.get("timestamp", [])
    equity_curve = payload.get("equity", [])
    if not timestamps or not equity_curve:
        print("No portfolio history found. Is your account funded?", file=sys.stderr)
        return

    # Auto-detect timestamp unit (Alpaca usually returns ms, but can vary)
    first_ts = timestamps[0]
    if first_ts > 1e11:
        dates = pd.to_datetime(timestamps, unit="ms", utc=True)
    else:
        dates = pd.to_datetime(timestamps, unit="s", utc=True)

    # Normalize to start at 1.0
    equity_values = [
        e / equity_curve[0] if equity_curve[0] > 0 else 1.0 for e in equity_curve
    ]

    # Fetch SPY data from Alpaca for the same range
    spy_bars = []
    try:
        start = dates.min().to_pydatetime().replace(tzinfo=None)
        end = dates.max().to_pydatetime().replace(tzinfo=None)
        spy_bars = alpaca.get_stock_bars_raw(benchmark, start, end)
    except Exception:
        pass

    # Fallback to yfinance if Alpaca didn't return data (e.g. range too long)
    if not spy_bars:
        try:
            ticker = yf.Ticker(benchmark)
            hist = ticker.history(start=dates.min(), end=dates.max())
            if not hist.empty:
                spy_bars = [
                    {"timestamp": t, "close": c}
                    for t, c in zip(hist.index, hist["Close"])
                ]
        except Exception:
            print(f"Warning: Could not fetch {benchmark} data.", file=sys.stderr)

    if not spy_bars:
        print(
            f"Plotting portfolio history only (no {benchmark} data).", file=sys.stderr
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, equity_values, label="Portfolio", linewidth=2)
    else:
        # Create a DataFrame for easy merging
        spy_df = pd.DataFrame(spy_bars)
        spy_df["timestamp"] = pd.to_datetime(spy_df["timestamp"], utc=True)
        spy_df = spy_df.set_index("timestamp").sort_index()

        # Reindex to match portfolio history dates
        bench_series = spy_df["close"].reindex(dates).ffill()
        bench_values = bench_series.values / bench_series.values[0]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, equity_values, label="Portfolio", linewidth=2)
        ax.plot(dates, bench_values, label=benchmark, linestyle="--", alpha=0.7)
        ax.legend()

    ax.set_title("Live Portfolio vs Benchmark")
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
    parser.add_argument(
        "--period", default="1M", help="Alpaca history period (e.g. 1M, 3M, 1Y)."
    )
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
