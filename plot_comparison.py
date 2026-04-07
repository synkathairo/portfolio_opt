import subprocess
import json
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date

today = str(date.today())

MODEL = "examples/nasdaq100_universe.json"
MODEL_NAME = "Nasdaq 100"
BACKTEST_DAYS = 252
MODEL_FILENAME = f"{MODEL_NAME}_{BACKTEST_DAYS}_{today}"

def run_backtest(args):
    print(f"Running: {args[1]} ...")
    result = subprocess.run(
        ["uv", "run", "portfolio-opt"] + args,
        capture_output=True, text=True
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Error running backtest: {result.stderr[:100]}")
        return None

# 1. Run Mean-Variance Backtest
mv_data = run_backtest([
    "--model", f"{MODEL}",
    "--strategy", "mean-variance",
    "--lookback-days", "60",
    "--backtest-days", f"{BACKTEST_DAYS}",
    "--rebalance-every", "5",
    "--estimate-from-history",
    "--return-model", "momentum",
    "--momentum-window", "40",
    "--mean-shrinkage", "0.5",
    "--risk-aversion", "2.0",
    "--use-cache", "--offline"
    # , "--rebalance-threshold 0.1"
])

# 2. Run Dual Momentum Backtest (Top-5)
dm_data = run_backtest([
    "--model", f"{MODEL}",
    "--strategy", "dual-momentum",
    "--lookback-days", "60",
    "--backtest-days", f"{BACKTEST_DAYS}",
    "--rebalance-every", "5",
    "--top-k", "5",
    "--use-cache", "--offline"
])

# 3. Run Dual Momentum Backtest (Top-2)
dm_data2 = run_backtest([
    "--model", f"{MODEL}",
    "--strategy", "dual-momentum",
    "--lookback-days", "60",
    "--backtest-days", f"{BACKTEST_DAYS}",
    "--rebalance-every", "5",
    "--top-k", "2",
    "--use-cache", "--offline"
])

# 3. Run Dual Momentum Backtest (Top-2, rebalance daily)
dm_data3 = run_backtest([
    "--model", f"{MODEL}",
    "--strategy", "dual-momentum",
    "--lookback-days", "60",
    "--backtest-days", f"{BACKTEST_DAYS}",
    "--rebalance-every", "1",
    "--top-k", "2",
    "--use-cache", "--offline"
])

# 4. Run Dual Momentum Backtest (Top-1, rebalance daily)
dm_data4 = run_backtest([
    "--model", f"{MODEL}",
    "--strategy", "dual-momentum",
    "--lookback-days", "60",
    "--backtest-days", f"{BACKTEST_DAYS}",
    "--rebalance-every", "1",
    "--top-k", "1",
    "--use-cache", "--offline"
])

# # Limit volatility to
# dm_data_limit_volatility = run_backtest([
#     "--model", "examples/nasdaq100_universe.json",
#     "--strategy", "dual-momentum",
#     "--lookback-days", "60",
#     "--backtest-days", f"{BACKTEST_DAYS}",
#     "--rebalance-every", "5",
#     "--top-k", "5",
#     "--use-cache", "--offline", "--target-vol 0.25"
# ])

if not mv_data or not dm_data:
    print("Failed to get backtest data.")
    exit()

# 3. Extract Daily Values
mv_curve = mv_data['backtest']['daily_values']
dm_curve = dm_data['backtest']['daily_values']
dm_curve2 = dm_data2['backtest']['daily_values']
dm_curve3 = dm_data3['backtest']['daily_values']
dm_curve4 = dm_data3['backtest']['daily_values']

# 4. Fetch Benchmarks
print("Fetching Benchmarks...")
spy = yf.Ticker("SPY").history(period="1y")['Close']
qqq = yf.Ticker("QQQ").history(period="1y")['Close']

# Normalize to 1.0
spy_norm = (spy / spy.iloc[0]).tolist()
qqq_norm = (qqq / qqq.iloc[0]).tolist()

# 5. Align Lengths
# Backtest curves usually have 253 points (start + 252 days).
# Yfinance usually has ~252 points. We'll match the shortest length.
target_len = min(len(spy_norm), len(mv_curve), len(dm_curve))
mv_curve = mv_curve[-target_len:]
dm_curve = dm_curve[-target_len:]
dm_curve2 = dm_curve2[-target_len:]
dm_curve3 = dm_curve3[-target_len:]
dm_curve4 = dm_curve4[-target_len:]
spy_norm = spy_norm[-target_len:]
qqq_norm = qqq_norm[-target_len:]

days = range(target_len)

# 6. Plot
plt.figure(figsize=(10, 6))
plt.plot(days, mv_curve, label='Mean-Variance (Momentum)', linewidth=2, color='#1f77b4')
plt.plot(days, dm_curve, label='Dual Momentum (Top-5)', linewidth=2, color='#c15a00')
plt.plot(days, dm_curve2, label='Dual Momentum (Top-2)', linewidth=2, color='#ff7f0e')
plt.plot(days, dm_curve3, label='Dual Momentum (Top-2 daily rebalance)', linewidth=2, color='#ff9a41')
plt.plot(days, dm_curve4, label='Dual Momentum (Top-1 daily rebalance)', linewidth=2, color='#ffa85b')
plt.plot(days, qqq_norm, label='QQQ (Nasdaq 100)', linestyle='--', alpha=0.7, color='#2ca02c')
plt.plot(days, spy_norm, label='SPY (S&P 500)', linestyle='--', alpha=0.7, color='#d62728')

plt.title(f"{MODEL_NAME} Strategy Comparison (1 Year) {today}", fontsize=14)
plt.xlabel('Trading Days', fontsize=12)
plt.ylabel('Growth of $1', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'strategy_comparison_{MODEL_FILENAME}.png', dpi=150)
print(f"Saved strategy_comparison_{MODEL_FILENAME}.png")
