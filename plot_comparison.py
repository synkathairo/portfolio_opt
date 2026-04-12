import subprocess
import json
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date

today = str(date.today())

# MODEL_NAME = "Nasdaq 100"
# MODEL_NAME = "Nasdaq100+SP500+sectors"
MODEL_NAME = "nasdaq100_sp500_sector_universe_b2016filtered"
# MODEL_NAME = "sector_universe"
# MODEL_NAME = "nasdaq100_universe"
# MODEL = "examples/nasdaq100_universe.json"
# MODEL = "examples/nasdaq100_sp500_sector_universe.json"
MODEL = f"examples/{MODEL_NAME}.json"
# LOOKBACK_DAYS = 60
LOOKBACK_DAYS = 252
# BACKTEST_DAYS = 252
BACKTEST_DAYS = 252*9
MOMENTUM_WINDOW = 40
MODEL_FILENAME = f"{MODEL_NAME}_{BACKTEST_DAYS}_{LOOKBACK_DAYS}_{today}"
INDEX_PERIOD = "10y"
DATASOURCE = "yfinance"

def run_backtest(args):
    print(f"Running: {args[1]} ...")
    result = subprocess.run(
        ["uv", "run", "portfolio-opt"] + args,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error running backtest: {result.stderr}")
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Error running backtest: {result.stderr}")
        return None

# 1. Run Mean-Variance Backtest
mv_data = run_backtest([
    "--model", f"{MODEL}",
    "--data-source", f"{DATASOURCE}",
    "--strategy", "mean-variance",
    "--lookback-days", f"{LOOKBACK_DAYS}",
    "--backtest-days", f"{BACKTEST_DAYS}",
    "--rebalance-every", "5",
    "--estimate-from-history",
    "--return-model", "momentum",
    "--momentum-window", f"{MOMENTUM_WINDOW}",
    "--mean-shrinkage", "0.5",
    "--risk-aversion", "2.0",
    "--use-cache"
    # , "--offline"
    # "--use-cache", "--refresh-cache"
    # , "--rebalance-threshold 0.1"
])

# 2. Run Dual Momentum Backtest (Top-5)
dm_data = run_backtest([
    "--model", f"{MODEL}",
    "--data-source", f"{DATASOURCE}",
    "--strategy", "dual-momentum",
    "--lookback-days", f"{LOOKBACK_DAYS}",
    "--backtest-days", f"{BACKTEST_DAYS}",
    "--rebalance-every", "5",
    "--top-k", "5",
    "--use-cache"
    # , "--offline"
])

# 3. Run Dual Momentum Backtest (Top-2)
dm_data2 = run_backtest([
    "--model", f"{MODEL}",
    "--data-source", f"{DATASOURCE}",
    "--strategy", "dual-momentum",
    "--lookback-days", f"{LOOKBACK_DAYS}",
    "--backtest-days", f"{BACKTEST_DAYS}",
    "--rebalance-every", "5",
    "--top-k", "2",
    "--use-cache"
    # , "--offline"
])

# 3. Run Dual Momentum Backtest (Top-2, rebalance daily)
dm_data3 = run_backtest([
    "--model", f"{MODEL}",
    "--data-source", f"{DATASOURCE}",
    "--strategy", "dual-momentum",
    "--lookback-days", f"{LOOKBACK_DAYS}",
    "--backtest-days", f"{BACKTEST_DAYS}",
    "--rebalance-every", "1",
    "--top-k", "2",
    "--use-cache"
    # , "--offline"
])

# 4. Run Dual Momentum Backtest (Top-1, rebalance daily)
dm_data4 = run_backtest([
    "--model", f"{MODEL}",
    "--data-source", f"{DATASOURCE}",
    "--strategy", "dual-momentum",
    "--lookback-days", f"{LOOKBACK_DAYS}",
    "--backtest-days", f"{BACKTEST_DAYS}",
    "--rebalance-every", "1",
    "--top-k", "1",
    "--use-cache"
    # , "--offline"
])

LIMIT_VOL = "0.3"
# # Limit volatility
dm_data_limit_volatility = run_backtest([
    "--model", f"{MODEL}",
    "--data-source", f"{DATASOURCE}",
    "--strategy", "dual-momentum",
    "--lookback-days", f"{LOOKBACK_DAYS}",
    "--backtest-days", f"{BACKTEST_DAYS}",
    "--rebalance-every", "5",
    "--top-k", "2",
    "--use-cache",
    # "--offline",
    # "--target-vol", "0.15"
    "--target-vol", f"{LIMIT_VOL}", "--vol-window", "252",
])

if not all([mv_data, dm_data, dm_data2, dm_data3, dm_data4, dm_data_limit_volatility]):
    print("Failed to get backtest data.")
    exit()

# 3. Extract Daily Values
mv_curve = mv_data['backtest']['daily_values']
dm_curve = dm_data['backtest']['daily_values']
dm_curve2 = dm_data2['backtest']['daily_values']
dm_curve3 = dm_data3['backtest']['daily_values']
dm_curve4 = dm_data4['backtest']['daily_values']
dmvol_curve = dm_data_limit_volatility['backtest']['daily_values']

# 4. Fetch Benchmarks
print("Fetching Benchmarks...")
spy = yf.Ticker("SPY").history(period=f"{INDEX_PERIOD}")['Close']
qqq = yf.Ticker("QQQ").history(period=f"{INDEX_PERIOD}")['Close']
tlt = yf.Ticker("TLT").history(period=f"{INDEX_PERIOD}")['Close'] # treasuries
iwm = yf.Ticker("IWM").history(period=f"{INDEX_PERIOD}")['Close']  # Russell 2000

# Normalize to 1.0
# spy_norm = (spy / spy.iloc[0]).tolist()
# qqq_norm = (qqq / qqq.iloc[0]).tolist()

# 5. Align Lengths
# Backtest curves usually have 253 points (start + 252 days).
# Yfinance usually has ~252 points. We'll match the shortest length.
# Align all curves to the same end date, then re-normalize each backtest
# curve to 1.0 at the start of the trimmed window so they share a baseline.
target_len = min(len(spy), len(qqq), len(tlt), len(iwm), len(mv_curve), len(dm_curve), len(dm_curve2), len(dm_curve3), len(dm_curve4), len(dmvol_curve))
mv_curve  = [v / mv_curve[0]  for v in mv_curve[-target_len:]]
dm_curve  = [v / dm_curve[0]  for v in dm_curve[-target_len:]]
dm_curve2 = [v / dm_curve2[0] for v in dm_curve2[-target_len:]]
dm_curve3 = [v / dm_curve3[0] for v in dm_curve3[-target_len:]]
dm_curve4 = [v / dm_curve4[0] for v in dm_curve4[-target_len:]]
dmvol_curve= [v / dmvol_curve[0] for v in dmvol_curve[-target_len:]]
spy_norm  = [v / spy.iloc[-target_len]  for v in spy.iloc[-target_len:].tolist()]
qqq_norm  = [v / qqq.iloc[-target_len]  for v in qqq.iloc[-target_len:].tolist()]
tlt_norm  = [v / tlt.iloc[-target_len]  for v in tlt.iloc[-target_len:].tolist()]
iwm_norm  = [v / iwm.iloc[-target_len]  for v in iwm.iloc[-target_len:].tolist()]

days = range(target_len)

# 6. Plot
plt.figure(figsize=(10, 6))
plt.plot(days, mv_curve, label='Mean-Variance (Momentum)', linewidth=2, color='#1f77b4')
plt.plot(days, dm_curve, label='Dual Momentum (Top-5)', linewidth=2, color='#c15a00')
plt.plot(days, dm_curve2, label='Dual Momentum (Top-2)', linewidth=2, color='#ff7f0e')
plt.plot(days, dm_curve3, label='Dual Momentum (Top-2 daily rebalance)', linewidth=2, color='#ff9a41')
plt.plot(days, dm_curve4, label='Dual Momentum (Top-1 daily rebalance)', linewidth=2, color='#ffa85b')
plt.plot(days, dmvol_curve, label=f'Dual Momentum (Top-2, vol {LIMIT_VOL})', linewidth=2, color='#FA5BFF')
plt.plot(days, qqq_norm, label='QQQ (Nasdaq 100)', linestyle='--', alpha=0.7, color='#2ca02c')
plt.plot(days, spy_norm, label='SPY (S&P 500)', linestyle='--', alpha=0.7, color='#d62728')
plt.plot(days, iwm_norm, label='IWM (Russell 2000)', linestyle='--', alpha=0.7, color='#9467bd')
plt.plot(days, tlt_norm, label='TLT (20+Yr Treasury)', linestyle='--', alpha=0.7, color='#8c564b')

plt.title(f"{MODEL_NAME} Strategy Comparison {today}", fontsize=14)
plt.xlabel('Trading Days', fontsize=12)
plt.ylabel('Growth of $1', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'strategy_comparison_{MODEL_FILENAME}.png', dpi=150)
print(f"Saved strategy_comparison_{MODEL_FILENAME}.png")
