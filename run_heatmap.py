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
MODEL = "examples/sector_universe_pre2020.json"
# TITLE_NAME_VAR = "Nasdaq_100_historical"
TITLE_NAME_VAR = "sector_universe_pre2020"
LOOKBACK = 60
# LOOKBACK = 252
BACKTEST_DAYS = 252
# BACKTEST_DAYS = 6000
BACKTEST_DAYS = 2520
REBALANCE_DAYS = [1, 5, 10, 21, 42, 63]
TOP_KS = [1, 2, 3, 5, 8]
FIGURE_NAME = f"heatmap_comparison_{TITLE_NAME_VAR}_{BACKTEST_DAYS}_{LOOKBACK}_{today}"

def run_backtest(rebal_days, k):
    cmd = [
        "uv", "run", "portfolio-opt",
        "--model", MODEL,
        "--strategy", "dual-momentum",
        "--lookback-days", str(LOOKBACK),
        "--backtest-days", str(BACKTEST_DAYS),
        "--rebalance-every", str(rebal_days),
        "--top-k", str(k),
        "--use-cache",
        # "--offline"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        data = json.loads(result.stdout)
        return data['backtest']
    except Exception as e:
        print(f"Error for rebal={rebal_days}, k={k}: {e}")
        return None

# Run Grid Search
print("Running grid search...")
rows = []
for rebal in REBALANCE_DAYS:
    for k in TOP_KS:
        print(f"Running rebal={rebal}, k={k}...")
        res = run_backtest(rebal, k)
        if res:
            rows.append({
                "Rebalance Days": rebal,
                "Top K": k,
                "Annualized Return": res.get("annualized_return", 0),
                "Annualized Volatility": res.get("annualized_volatility", 0),
                "Max Drawdown": res.get("max_drawdown", 0),
                "Average Turnover": res.get("average_turnover", 0)
            })

df = pd.DataFrame(rows)

# Pivot for heatmaps
df_return = df.pivot(index="Rebalance Days", columns="Top K", values="Annualized Return")
df_vol = df.pivot(index="Rebalance Days", columns="Top K", values="Annualized Volatility")
df_dd = df.pivot(index="Rebalance Days", columns="Top K", values="Max Drawdown")
df_turn = df.pivot(index="Rebalance Days", columns="Top K", values="Average Turnover")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'{TITLE_NAME_VAR} Dual Momentum Grid Search (Lookback={LOOKBACK}, Days={BACKTEST_DAYS})', fontsize=16)

# 1. Return
sns.heatmap(df_return, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[0, 0], cbar_kws={'label': 'Return'})
axes[0, 0].set_title("Annualized Return")

# 2. Volatility
sns.heatmap(df_vol, annot=True, fmt=".2f", cmap="Reds", ax=axes[0, 1], cbar_kws={'label': 'Vol'})
axes[0, 1].set_title("Annualized Volatility")

# 3. Drawdown
sns.heatmap(df_dd, annot=True, fmt=".2f", cmap="Oranges", ax=axes[1, 0], cbar_kws={'label': 'Drawdown'})
axes[1, 0].set_title("Max Drawdown")

# 4. Turnover
sns.heatmap(df_turn, annot=True, fmt=".2f", cmap="Purples", ax=axes[1, 1], cbar_kws={'label': 'Turnover'})
axes[1, 1].set_title("Average Turnover")

plt.tight_layout()
plt.savefig(f"{FIGURE_NAME}.png", dpi=150)
print(f"Saved {FIGURE_NAME}.png")
