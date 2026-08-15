import json
import re
import subprocess
from datetime import date

import matplotlib.pyplot as plt
import yfinance as yf

today = str(date.today())

# MODEL_NAME = "Nasdaq 100"
# MODEL_NAME = "Nasdaq100+SP500+sectors"
MODEL_NAME = "nasdaq100_sp500_sector_universe_b2016filtered"
# MODEL_NAME = "yfiua_hsi_202307_current_valid_universe"
# MODEL_NAME = "yfiua_ftse100_202307_current_valid_universe"
# MODEL_NAME = "yfiua_csi300_202307_current_valid_universe"
# MODEL_NAME = "yfiua_csi500_202402_current_valid_universe"
# MODEL_NAME = "yfiua_csi1000_202402_current_valid_universe"
# MODEL_NAME = "yfiua_csi_combined_202402_current_valid_with_benchmarks_universe"
# MODEL_NAME = "yfiua_sse_202307_current_valid_universe"
# MODEL_NAME = "nikkei225_current_backtest_valid_universe"
# MODEL_NAME = "sector_universe_pre2016_nomax"
# MODEL_NAME = "sector_universe"
# MODEL_NAME = "nasdaq100_universe"
# MODEL_NAME = "nasdaq100_sp500_sector_universe"
MODEL = f"examples/{MODEL_NAME}.json"

LOOKBACK_PERIODS = [21, 60, 126, 252]
BACKTEST_DAYS = 2722
INDEX_PERIOD = "20y"
DATASOURCE = "yfinance"
TOP_K = "2"

# Keep the strategy fixed so the plot isolates lookback-period behavior.
STRATEGY_LABEL = f"Dual Momentum Top-{TOP_K} daily rebalance"
STRATEGY_ARGS = [
    "--strategy",
    "dual-momentum",
    "--rebalance-every",
    "1",
    "--top-k",
    TOP_K,
    "--use-cache",
    # "--offline",
]

BENCHMARKS = [
    ("SPY", "SPY (S&P 500)"),
    ("QQQ", "QQQ (Nasdaq 100)"),
    ("IWM", "IWM (Russell 2000)"),
    ("TLT", "TLT (20+Yr Treasury)"),
]

MODEL_FILENAME = f"{MODEL_NAME}_{BACKTEST_DAYS}_{TOP_K}_lookbacks_{today}"


def with_backtest_days(args, backtest_days: int):
    updated = list(args)
    updated[updated.index("--backtest-days") + 1] = str(backtest_days)
    return updated


def supported_backtest_days(stderr: str):
    match = re.search(r"supports at most (\d+) backtest days", stderr)
    if not match:
        return None
    return int(match.group(1))


def run_backtest(label: str, args, *, retry_supported_backtest_days: bool = True):
    print(f"Running: {label} ...")
    result = subprocess.run(
        ["uv", "run", "portfolio-opt"] + args, capture_output=True, text=True
    )
    if result.returncode != 0:
        supported_days = supported_backtest_days(result.stderr)
        requested_days = int(args[args.index("--backtest-days") + 1])
        if (
            retry_supported_backtest_days
            and supported_days is not None
            and supported_days > 0
            and supported_days < requested_days
        ):
            retry_label = f"{label} ({supported_days} backtest days)"
            print(
                f"Requested {requested_days} backtest days is too long for {label}; "
                f"retrying with {supported_days}."
            )
            return run_backtest(
                retry_label,
                with_backtest_days(args, supported_days),
                retry_supported_backtest_days=False,
            )
        print(f"Error running backtest: {result.stderr}")
        return None
    try:
        data = json.loads(result.stdout)
        data["label"] = label
        return data
    except json.JSONDecodeError:
        print(f"Error decoding backtest output: {result.stderr}")
        return None


lookback_results = []
for lookback_days in LOOKBACK_PERIODS:
    lookback_results.append(
        run_backtest(
            f"{STRATEGY_LABEL}, lookback {lookback_days}d",
            [
                "--model",
                MODEL,
                "--data-source",
                DATASOURCE,
                "--lookback-days",
                str(lookback_days),
                "--backtest-days",
                str(BACKTEST_DAYS),
                *STRATEGY_ARGS,
            ],
        )
    )

if not all(lookback_results):
    print("Failed to get backtest data.")
    exit()

strategy_curves = [
    (result["label"], result["backtest"]["daily_values"]) for result in lookback_results
]

print("Fetching Benchmarks...")
benchmark_curves = {}
for ticker, label in BENCHMARKS:
    history = yf.Ticker(ticker).history(period=INDEX_PERIOD)
    if history.empty or "Close" not in history:
        print(f"Skipping benchmark with no close history: {ticker}")
        continue
    benchmark_curves[label] = history["Close"]

if not benchmark_curves:
    print("Failed to fetch benchmark data.")
    exit()

# Align all curves to the same end date, then re-normalize each curve to 1.0.
target_len = min(
    *[len(curve) for curve in benchmark_curves.values()],
    *[len(curve) for _, curve in strategy_curves],
)


def normalize_tail(curve):
    tail = curve[-target_len:]
    return [v / tail[0] for v in tail]


strategy_norms = [(label, normalize_tail(curve)) for label, curve in strategy_curves]
benchmark_norms = {
    label: [v / curve.iloc[-target_len] for v in curve.iloc[-target_len:].tolist()]
    for label, curve in benchmark_curves.items()
}

days = range(target_len)
color_map = plt.get_cmap("tab10")
strategy_colors = {
    label: color_map(index % color_map.N)
    for index, (label, _) in enumerate(strategy_norms)
}
benchmark_colors = {
    label: color_map((index + len(strategy_norms)) % color_map.N)
    for index, label in enumerate(benchmark_norms)
}

plt.figure(figsize=(10, 6))
for label, curve in strategy_norms:
    plt.plot(days, curve, label=label, linewidth=2.0, color=strategy_colors[label])
for label, curve in benchmark_norms.items():
    plt.plot(
        days,
        curve,
        label=label,
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        color=benchmark_colors[label],
    )

plt.yscale("log")
plt.title(f"{MODEL_NAME} Lookback Comparison {today}", fontsize=14)
plt.xlabel("Trading Days", fontsize=12)
plt.ylabel("Growth of $1", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"plots/lookback_comparison_{MODEL_FILENAME}.png", dpi=150)
print(f"Saved plots/lookback_comparison_{MODEL_FILENAME}.png")
