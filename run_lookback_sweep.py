import subprocess
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import date

today = str(date.today())

# Configuration
TITLE_NAME_VAR = "sector_universe_pre2016_nomax"
# TITLE_NAME_VAR = "nasdaq100_sector_universe_b2016filtered"
# TITLE_NAME_VAR = "nasdaq100_sp500_sector_universe"
MODEL = f"examples/{TITLE_NAME_VAR}.json"
# Set BACKTEST_DAYS to a value that fits the longest lookback in your universe.
# Rule of thumb: min_symbol_history - max(LOOKBACK_DAYS) - 1
# BACKTEST_DAYS = 2520
BACKTEST_DAYS = 252 * 17
REBALANCE_EVERY = 1
LOOKBACK_DAYS = [1, 3, 7, 14, 21, 42, 60, 90, 126, 252, 504]
TOP_KS = [1, 2, 3, 5, 8]
TRADING_DAYS_PER_YEAR = 252
FIGURE_NAME = (
    f"lookback_sweep_{TITLE_NAME_VAR}_{BACKTEST_DAYS}_{REBALANCE_EVERY}_{today}"
)
TIMEOUT_LEN = 2000


def build_backtest_cmd(lookback, k, *, refresh_cache=False):
    cmd = [
        "uv",
        "run",
        "portfolio-opt",
        "--model",
        MODEL,
        "--strategy",
        "dual-momentum",
        "--lookback-days",
        str(lookback),
        "--backtest-days",
        str(BACKTEST_DAYS),
        "--rebalance-every",
        str(REBALANCE_EVERY),
        "--top-k",
        str(k),
        "--dual-momentum-weighting",
        "equal",
        "--estimate-from-history",
        "--trading-days-per-year",
        str(TRADING_DAYS_PER_YEAR),
        "--data-source",
        "yfinance",
        "--use-cache",
    ]
    if refresh_cache:
        cmd.append("--refresh-cache")
    return cmd


def run_backtest(lookback, k):
    cmd = build_backtest_cmd(lookback, k)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_LEN
        )
        if result.stderr.strip():
            print(result.stderr.strip())
        if result.returncode != 0:
            print(f"Error for lookback={lookback}, k={k}: see above")
            return None
        data = json.loads(result.stdout)
        return data["backtest"]
    except Exception as e:
        print(f"Error for lookback={lookback}, k={k}: {e}")
        return None


def fallback_sortino_ratio(res):
    daily_values = res.get("daily_values", [])
    if len(daily_values) < 2:
        return 0.0
    values = np.array(daily_values, dtype=float)
    returns = values[1:] / values[:-1] - 1.0
    downside_returns = np.minimum(returns, 0.0)
    downside_deviation = np.sqrt(float(np.mean(downside_returns**2))) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    if downside_deviation <= 0:
        return 0.0
    return float(res.get("annualized_return", 0.0)) / downside_deviation


# Run Grid Search
print("Running grid search...")
rows = []
for lookback in LOOKBACK_DAYS:
    for k in TOP_KS:
        print(f"Running lookback={lookback}, k={k}...")
        res = run_backtest(lookback, k)
        if res:
            ann_ret = res.get("annualized_return", 0)
            ann_vol = res.get("annualized_volatility", 0)
            max_dd = abs(res.get("max_drawdown", 0))
            sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0.0
            calmar = (ann_ret / max_dd) if max_dd > 0 else 0.0
            sortino = float(res.get("sortino_ratio", fallback_sortino_ratio(res)))
            rows.append(
                {
                    "Lookback Days": lookback,
                    "Top K": k,
                    "Annualized Return": ann_ret,
                    "Annualized Volatility": ann_vol,
                    "Sharpe Ratio": sharpe,
                    "Calmar Ratio": calmar,
                    "Sortino Ratio": sortino,
                    "Max Drawdown": res.get("max_drawdown", 0),
                    "Average Turnover": res.get("average_turnover", 0),
                }
            )

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No successful backtests; no heatmap generated.")

# Pivot for heatmaps
df_return = df.pivot(index="Lookback Days", columns="Top K", values="Annualized Return")
df_vol = df.pivot(
    index="Lookback Days", columns="Top K", values="Annualized Volatility"
)
df_dd = df.pivot(index="Lookback Days", columns="Top K", values="Max Drawdown")
df_turn = df.pivot(index="Lookback Days", columns="Top K", values="Average Turnover")
df_sharpe = df.pivot(index="Lookback Days", columns="Top K", values="Sharpe Ratio")
df_sortino = df.pivot(index="Lookback Days", columns="Top K", values="Sortino Ratio")

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle(
    f"{TITLE_NAME_VAR} Dual Momentum Lookback Sweep (Rebalance={REBALANCE_EVERY}d, Days={BACKTEST_DAYS})",
    fontsize=16,
)

sns.heatmap(
    df_return,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    ax=axes[0, 0],
    cbar_kws={"label": "Return"},
)
axes[0, 0].set_title("Annualized Return")

sns.heatmap(
    df_vol,
    annot=True,
    fmt=".2f",
    cmap="Reds",
    ax=axes[0, 1],
    cbar_kws={"label": "Vol"},
)
axes[0, 1].set_title("Annualized Volatility")

sns.heatmap(
    df_dd,
    annot=True,
    fmt=".2f",
    cmap="Oranges",
    ax=axes[1, 0],
    cbar_kws={"label": "Drawdown"},
)
axes[1, 0].set_title("Max Drawdown")

sns.heatmap(
    df_turn,
    annot=True,
    fmt=".2f",
    cmap="Purples",
    ax=axes[1, 1],
    cbar_kws={"label": "Turnover"},
)
axes[1, 1].set_title("Average Turnover")

sns.heatmap(
    df_sharpe,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    ax=axes[2, 0],
    cbar_kws={"label": "Sharpe"},
)
axes[2, 0].set_title("Sharpe Ratio")

sns.heatmap(
    df_sortino,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    ax=axes[2, 1],
    cbar_kws={"label": "Sortino"},
)
axes[2, 1].set_title("Sortino Ratio")

plt.tight_layout()
plt.savefig(f"plots/{FIGURE_NAME}.png", dpi=150)
print(f"Saved plots/{FIGURE_NAME}.png")
