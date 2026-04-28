import json
import subprocess
from datetime import date

import matplotlib.pyplot as plt
import yfinance as yf

today = str(date.today())

# MODEL_NAME = "Nasdaq 100"
# MODEL_NAME = "Nasdaq100+SP500+sectors"
# MODEL_NAME = "nasdaq100_sp500_sector_universe_b2016filtered"
# MODEL_NAME = "yfiua_hsi_202307_current_valid_universe"
# MODEL_NAME = "yfiua_ftse100_202307_current_valid_universe"
# MODEL_NAME = "yfiua_csi300_202307_current_valid_universe"
# MODEL_NAME = "yfiua_csi500_202402_current_valid_universe"
# MODEL_NAME = "yfiua_csi1000_202402_current_valid_universe"
# MODEL_NAME = "yfiua_csi_combined_202402_current_valid_with_benchmarks_universe"
# MODEL_NAME = "yfiua_sse_202307_current_valid_universe"
# MODEL_NAME = "nikkei225_current_backtest_valid_universe"
MODEL_NAME = "sector_universe_pre2016_nomax"
# MODEL_NAME = "sector_universe"
# MODEL_NAME = "nasdaq100_universe"
# MODEL = "examples/nasdaq100_universe.json"
# MODEL_NAME = "nasdaq100_sp500_sector_universe"
MODEL = f"examples/{MODEL_NAME}.json"
# LOOKBACK_DAYS = 60
LOOKBACK_DAYS = 252
# BACKTEST_DAYS = 252*9
# BACKTEST_DAYS = 252*15
BACKTEST_DAYS = 4454
# BACKTEST_DAYS = 680
# BACKTEST_DAYS = 675
# BACKTEST_DAYS = 470
# BACKTEST_DAYS = 408
# BACKTEST_DAYS = 402
# BACKTEST_DAYS = 604
# BACKTEST_DAYS = 1220
MOMENTUM_WINDOW = 40
LIMIT_VOL = "0.3"
MODEL_FILENAME = f"{MODEL_NAME}_{BACKTEST_DAYS}_{LOOKBACK_DAYS}_{today}"
# INDEX_PERIOD = "10y"
INDEX_PERIOD = "20y"
DATASOURCE = "yfinance"
BENCHMARKS = [
    ("SPY", "SPY (S&P 500)"),
    ("QQQ", "QQQ (Nasdaq 100)"),
    ("IWM", "IWM (Russell 2000)"),
    ("TLT", "TLT (20+Yr Treasury)"),
]
# BENCHMARKS = [
#     # ("^HSI", "Hang Seng Index"),
#     ("2800.HK", "Tracker Fund of Hong Kong"),
#     ("2819.HK", "ABF Hong Kong Bond"),
# ]
# BENCHMARKS = [
#     # ("^FTSE", "FTSE 100"),
#     ("ISF.L", "iShares Core FTSE 100"),
#     ("IGLT.L", "iShares UK Gilts"),
#     ("IGLS.L", "iShares UK Gilts 0-5yr"),
# ]
# BENCHMARKS = [
#     ("510300.SS", "Huatai-PB CSI 300 ETF"),
#     ("510500.SS", "China CSI 500 ETF"),
#     # ("512100.SS", "China Southern CSI 1000 ETF"),
#     # ("000852.SS", "CSI 1000 Index"),
#     ("511010.SS", "Guotai 5Y China Treasury ETF"),
#     ("000001.SS", "SSE Composite"),
#     ("399001.SZ", "SZSE Component"),
#     ("510050.SS", "SSE 50 ETF"),
#     ("159915.SZ", "ChiNext ETF")
# ]
# BENCHMARKS = [
#     # ("^N225", "Nikkei 225"),
#     ("1321.T", "NEXT FUNDS Nikkei 225 ETF"),
#     ("1306.T", "NEXT FUNDS TOPIX ETF"),
#     ("2561.T", "iShares Japan Government Bond ETF"),
# ]


def run_backtest(label: str, args):
    print(f"Running: {args[1]} ...")
    result = subprocess.run(
        ["uv", "run", "portfolio-opt"] + args, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error running backtest: {result.stderr}")
        return None
    try:
        data = json.loads(result.stdout)
        data["label"] = label
        return data
    except json.JSONDecodeError:
        print(f"Error running backtest: {result.stderr}")
        return None


# 1. Run Mean-Variance Backtest
mv_data = run_backtest(
    "Mean-Variance (Momentum)",
    [
        "--model",
        f"{MODEL}",
        "--data-source",
        f"{DATASOURCE}",
        "--strategy",
        "mean-variance",
        "--lookback-days",
        f"{LOOKBACK_DAYS}",
        "--backtest-days",
        f"{BACKTEST_DAYS}",
        "--rebalance-every",
        "5",
        "--estimate-from-history",
        "--return-model",
        "momentum",
        "--momentum-window",
        f"{MOMENTUM_WINDOW}",
        "--mean-shrinkage",
        "0.5",
        "--risk-aversion",
        "2.0",
        "--use-cache",
        # , "--offline"
        # "--use-cache", "--refresh-cache"
        # , "--rebalance-threshold 0.1"
    ],
)

# 2. Run Dual Momentum Backtest (Top-5)
dm_data = run_backtest(
    "Dual Momentum (Top-5)",
    [
        "--model",
        f"{MODEL}",
        "--data-source",
        f"{DATASOURCE}",
        "--strategy",
        "dual-momentum",
        "--lookback-days",
        f"{LOOKBACK_DAYS}",
        "--backtest-days",
        f"{BACKTEST_DAYS}",
        "--rebalance-every",
        "5",
        "--top-k",
        "5",
        "--use-cache",
        # , "--offline"
    ],
)

# 3. Run Dual Momentum Backtest (Top-2)
dm_data2 = run_backtest(
    "Dual Momentum (Top-2)",
    [
        "--model",
        f"{MODEL}",
        "--data-source",
        f"{DATASOURCE}",
        "--strategy",
        "dual-momentum",
        "--lookback-days",
        f"{LOOKBACK_DAYS}",
        "--backtest-days",
        f"{BACKTEST_DAYS}",
        "--rebalance-every",
        "5",
        "--top-k",
        "2",
        "--use-cache",
        # , "--offline"
    ],
)

# Run Dual Momentum Backtest (Top-3, rebalance daily), trail stop 0.15
dm_data2b = run_backtest(
    "Dual Momentum (Top-3 daily rebalance, trail-stop 0.15)",
    [
        "--model",
        f"{MODEL}",
        "--data-source",
        f"{DATASOURCE}",
        "--strategy",
        "dual-momentum",
        "--lookback-days",
        f"{LOOKBACK_DAYS}",
        "--backtest-days",
        f"{BACKTEST_DAYS}",
        "--rebalance-every",
        "1",
        "--top-k",
        "3",
        "--trailing-stop",
        "0.15",
        "--use-cache",
        # , "--offline"
    ],
)

# Run Dual Momentum Backtest (Top-3, rebalance daily)
dm_data2c = run_backtest(
    "Dual Momentum (Top-3 daily rebalance)",
    [
        "--model",
        f"{MODEL}",
        "--data-source",
        f"{DATASOURCE}",
        "--strategy",
        "dual-momentum",
        "--lookback-days",
        f"{LOOKBACK_DAYS}",
        "--backtest-days",
        f"{BACKTEST_DAYS}",
        "--rebalance-every",
        "1",
        "--top-k",
        "3",
        "--use-cache",
        # , "--offline"
    ],
)

# dm_data2cb = run_backtest(
#     f"Dual Momentum (Top-3 daily rebalance, vol {LIMIT_VOL})",
#     [
#         "--model",
#         f"{MODEL}",
#         "--data-source",
#         f"{DATASOURCE}",
#         "--strategy",
#         "dual-momentum",
#         "--lookback-days",
#         f"{LOOKBACK_DAYS}",
#         "--backtest-days",
#         f"{BACKTEST_DAYS}",
#         "--rebalance-every",
#         "1",
#         "--top-k",
#         "3",
#         "--use-cache",
#         # , "--offline"
#         "--target-vol",
#         f"{LIMIT_VOL}",
#         "--vol-window",
#         "252"
#     ]
# )

# 3. Run Dual Momentum Backtest (Top-2, rebalance daily)
dm_data3 = run_backtest(
    "Dual Momentum (Top-2 daily rebalance)",
    [
        "--model",
        f"{MODEL}",
        "--data-source",
        f"{DATASOURCE}",
        "--strategy",
        "dual-momentum",
        "--lookback-days",
        f"{LOOKBACK_DAYS}",
        "--backtest-days",
        f"{BACKTEST_DAYS}",
        "--rebalance-every",
        "1",
        "--top-k",
        "2",
        "--use-cache",
        # , "--offline"
    ],
)

# Run Dual Momentum Backtest (Top-2, rebalance daily), trail stop 0.15
dm_data3a = run_backtest(
    "Dual Momentum (Top-2 daily rebalance, trail-stop 0.15)",
    [
        "--model",
        f"{MODEL}",
        "--data-source",
        f"{DATASOURCE}",
        "--strategy",
        "dual-momentum",
        "--lookback-days",
        f"{LOOKBACK_DAYS}",
        "--backtest-days",
        f"{BACKTEST_DAYS}",
        "--rebalance-every",
        "1",
        "--top-k",
        "2",
        "--trailing-stop",
        "0.15",
        "--use-cache",
        # , "--offline"
    ],
)

# 4. Run Dual Momentum Backtest (Top-1, rebalance daily)
dm_data4 = run_backtest(
    "Dual Momentum (Top-1 daily rebalance)",
    [
        "--model",
        f"{MODEL}",
        "--data-source",
        f"{DATASOURCE}",
        "--strategy",
        "dual-momentum",
        "--lookback-days",
        f"{LOOKBACK_DAYS}",
        "--backtest-days",
        f"{BACKTEST_DAYS}",
        "--rebalance-every",
        "1",
        "--top-k",
        "1",
        "--use-cache",
        # , "--offline"
    ],
)

dm_data4b = run_backtest(
    "Dual Momentum (Top-1 daily rebalance, trailstop0.15)",
    [
        "--model",
        f"{MODEL}",
        "--data-source",
        f"{DATASOURCE}",
        "--strategy",
        "dual-momentum",
        "--lookback-days",
        f"{LOOKBACK_DAYS}",
        "--backtest-days",
        f"{BACKTEST_DAYS}",
        "--rebalance-every",
        "1",
        "--top-k",
        "1",
        "--trailing-stop",
        "0.15",
        "--use-cache",
        # , "--offline"
    ],
)

# # Limit volatility
dm_data_limit_volatility = run_backtest(
    f"Dual Momentum (Top-2, vol {LIMIT_VOL})",
    [
        "--model",
        f"{MODEL}",
        "--data-source",
        f"{DATASOURCE}",
        "--strategy",
        "dual-momentum",
        "--lookback-days",
        f"{LOOKBACK_DAYS}",
        "--backtest-days",
        f"{BACKTEST_DAYS}",
        "--rebalance-every",
        "5",
        "--top-k",
        "2",
        "--use-cache",
        # "--offline",
        # "--target-vol", "0.15"
        "--target-vol",
        f"{LIMIT_VOL}",
        "--vol-window",
        "252",
    ],
)

strategy_results = [
    mv_data,
    dm_data,
    dm_data2,
    dm_data2b,
    dm_data2c,
    # dm_data2cb,
    dm_data3,
    dm_data3a,
    dm_data4,
    dm_data4b,
    dm_data_limit_volatility,
]

if not all(strategy_results):
    print("Failed to get backtest data.")
    exit()

# 3. Extract Daily Values
strategy_curves = [
    (result["label"], result["backtest"]["daily_values"]) for result in strategy_results
]

# 4. Fetch Benchmarks
print("Fetching Benchmarks...")
benchmark_curves = {}
for ticker, label in BENCHMARKS:
    history = yf.Ticker(ticker).history(period=f"{INDEX_PERIOD}")
    if history.empty or "Close" not in history:
        print(f"Skipping benchmark with no close history: {ticker}")
        continue
    benchmark_curves[label] = history["Close"]

if not benchmark_curves:
    print("Failed to fetch benchmark data.")
    exit()

# Normalize to 1.0
# spy_norm = (spy / spy.iloc[0]).tolist()
# qqq_norm = (qqq / qqq.iloc[0]).tolist()

# 5. Align Lengths
# Backtest curves usually have 253 points (start + 252 days).
# Yfinance usually has ~252 points. We'll match the shortest length.
# Align all curves to the same end date, then re-normalize each backtest
# curve to 1.0 at the start of the trimmed window so they share a baseline.
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
color_map = plt.get_cmap("tab20")
strategy_colors = {
    label: color_map(index % color_map.N)
    for index, (label, _) in enumerate(strategy_norms)
}
benchmark_colors = {
    label: color_map((index + len(strategy_norms)) % color_map.N)
    for index, label in enumerate(benchmark_norms)
}


def line_width_for_curve_count(count):
    if count <= 8:
        return 2.0
    if count <= 12:
        return 1.6
    if count <= 18:
        return 1.25
    return 1.0


curve_count = len(strategy_norms) + len(benchmark_norms)
strategy_line_width = line_width_for_curve_count(curve_count)
benchmark_line_width = max(0.8, strategy_line_width - 0.2)

# 6. Plot
plt.figure(figsize=(10, 6))
for label, curve in strategy_norms:
    plt.plot(
        days,
        curve,
        label=label,
        linewidth=strategy_line_width,
        color=strategy_colors[label],
    )
for label, curve in benchmark_norms.items():
    plt.plot(
        days,
        curve,
        label=label,
        linewidth=benchmark_line_width,
        linestyle="--",
        alpha=0.7,
        color=benchmark_colors[label],
    )

plt.title(f"{MODEL_NAME} Strategy Comparison {today}", fontsize=14)
plt.xlabel("Trading Days", fontsize=12)
plt.ylabel("Growth of $1", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"plots/strategy_comparison_{MODEL_FILENAME}.png", dpi=150)
print(f"Saved plots/strategy_comparison_{MODEL_FILENAME}.png")
