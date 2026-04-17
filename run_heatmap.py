import subprocess
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import date

today = str(date.today())

# Configuration
# MODEL = "examples/nasdaq100_universe.json"
# MODEL = "examples/nasdaq100_historical_universe.json"
# MODEL = "examples/sector_universe_pre2020.json"
# MODEL = "examples/nasdaq100_sp500_sector_universe.json"
MODEL = "examples/nasdaq100_sp500_sector_universe_b2016filtered.json"
# TITLE_NAME_VAR = "Nasdaq_100_historical"
# TITLE_NAME_VAR = "sector_universe_pre2020"
TITLE_NAME_VAR = "nasdaq100_sp500_sector_universe_b2016filtered"
# TITLE_NAME_VAR = "nasdaq100_sp500_sector_universe_b2016filtered-targetvol0.3-252"
# TITLE_NAME_VAR = "nasdaq100_sp500_sector_universe_b2016filtered-basketoptrisk2.0"
# LOOKBACK = 60
LOOKBACK = 252
BACKTEST_DAYS = 252 * 9
# BACKTEST_DAYS = 252*4
# BACKTEST_DAYS = 6000
# BACKTEST_DAYS = 2520
# TARGET_VOL = 0.3
# VOL_WINDOW = 252
REBALANCE_DAYS = [1, 5, 10, 21, 42, 63]
TOP_KS = [1, 2, 3, 5, 8]
FIGURE_NAME = f"heatmap_comparison_{TITLE_NAME_VAR}_{BACKTEST_DAYS}_{LOOKBACK}_{today}"
# TIMEOUT = 120
# TIMEOUT_LEN = 600
TIMEOUT_LEN = 2000
PRIME_CACHE = True


def build_backtest_cmd(rebal_days, k, *, refresh_cache=False, offline=False):
    cmd = [
        "uv",
        "run",
        "portfolio-opt",
        "--model",
        MODEL,
        "--strategy",
        "dual-momentum",
        "--lookback-days",
        str(LOOKBACK),
        "--backtest-days",
        str(BACKTEST_DAYS),
        "--rebalance-every",
        str(rebal_days),
        "--top-k",
        str(k),
        # "--basket-opt", "mean-variance", "--basket-risk-aversion", "2.0",
        # "--target-vol", str(TARGET_VOL), "--vol-window", str(VOL_WINDOW),
        "--data-source",
        "yfinance",
        "--use-cache",
    ]
    if refresh_cache:
        cmd.append("--refresh-cache")
    if offline:
        cmd.append("--offline")
    return cmd


def prime_cache():
    if not PRIME_CACHE:
        return
    print("Priming cache...")
    cmd = build_backtest_cmd(
        REBALANCE_DAYS[0],
        TOP_KS[0],
        refresh_cache=True,
        offline=False,
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_LEN)
    if result.returncode != 0:
        raise RuntimeError(f"Cache prime failed: {result.stderr.strip()}")


def run_backtest(rebal_days, k):
    cmd = build_backtest_cmd(rebal_days, k, offline=PRIME_CACHE)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_LEN
        )
        if result.returncode != 0:
            print(f"Error for rebal={rebal_days}, k={k}: {result.stderr.strip()}")
            return None
        data = json.loads(result.stdout)
        return data["backtest"]
    except Exception as e:
        print(f"Error for rebal={rebal_days}, k={k}: {e}")
        return None


# Run Grid Search
print("Running grid search...")
prime_cache()
rows = []
for rebal in REBALANCE_DAYS:
    for k in TOP_KS:
        print(f"Running rebal={rebal}, k={k}...")
        res = run_backtest(rebal, k)
        if res:
            ann_ret = res.get("annualized_return", 0)
            ann_vol = res.get("annualized_volatility", 0)
            # ann_vol = res.get("annualized_volatility", 1)
            max_dd = abs(res.get("max_drawdown", 0))

            # Calculate Sharpe Ratio: Return / Vol
            # note TODO: it's technically not minus risk-free return
            # (see https://en.wikipedia.org/wiki/Sharpe_ratio) but good enough for purposes of comparison
            sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0.0

            # Calculate Calmar Ratio: Return / Max Drawdown
            # note TODO: again technically involves risk-free return but this is close enough for purposes here
            # (see https://www.investopedia.com/terms/c/calmarratio.asp https://en.wikipedia.org/wiki/Calmar_ratio)
            calmar = (ann_ret / max_dd) if max_dd > 0 else 0.0

            rows.append(
                {
                    "Rebalance Days": rebal,
                    "Top K": k,
                    "Annualized Return": ann_ret,
                    "Annualized Volatility": ann_vol,
                    "Sharpe Ratio": sharpe,
                    "Calmar Ratio": calmar,
                    "Max Drawdown": res.get("max_drawdown", 0),
                    "Average Turnover": res.get("average_turnover", 0),
                }
            )

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No successful backtests; no heatmap generated.")

# Pivot for heatmaps
df_return = df.pivot(
    index="Rebalance Days", columns="Top K", values="Annualized Return"
)
df_vol = df.pivot(
    index="Rebalance Days", columns="Top K", values="Annualized Volatility"
)
df_dd = df.pivot(index="Rebalance Days", columns="Top K", values="Max Drawdown")
df_turn = df.pivot(index="Rebalance Days", columns="Top K", values="Average Turnover")
df_sharpe = df.pivot(index="Rebalance Days", columns="Top K", values="Sharpe Ratio")
df_calmar = df.pivot(index="Rebalance Days", columns="Top K", values="Calmar Ratio")

# Plotting
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle(
    f"{TITLE_NAME_VAR} Dual Momentum Grid Search (Lookback={LOOKBACK}, Days={BACKTEST_DAYS})",
    fontsize=16,
)

# 1. Return
sns.heatmap(
    df_return,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    ax=axes[0, 0],
    cbar_kws={"label": "Return"},
)
axes[0, 0].set_title("Annualized Return")

# 2. Volatility
sns.heatmap(
    df_vol, annot=True, fmt=".2f", cmap="Reds", ax=axes[0, 1], cbar_kws={"label": "Vol"}
)
axes[0, 1].set_title("Annualized Volatility")

# 3. Drawdown
sns.heatmap(
    df_dd,
    annot=True,
    fmt=".2f",
    cmap="Oranges",
    ax=axes[1, 0],
    cbar_kws={"label": "Drawdown"},
)
axes[1, 0].set_title("Max Drawdown")

# 4. Turnover
sns.heatmap(
    df_turn,
    annot=True,
    fmt=".2f",
    cmap="Purples",
    ax=axes[1, 1],
    cbar_kws={"label": "Turnover"},
)
axes[1, 1].set_title("Average Turnover")

# 5. Sharpe Ratio
sns.heatmap(
    df_sharpe,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    ax=axes[2, 0],
    cbar_kws={"label": "Sharpe"},
)
axes[2, 0].set_title("Sharpe Ratio")

# 6. Calmar Ratio
sns.heatmap(
    df_calmar,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    ax=axes[2, 1],
    cbar_kws={"label": "Calmar"},
)
axes[2, 1].set_title("Calmar Ratio")
# axes[2, 1].set_title("Calmar Ratio (Return / |MaxDD|)")

plt.tight_layout()
plt.savefig(f"{FIGURE_NAME}.png", dpi=150)
print(f"Saved {FIGURE_NAME}.png")
