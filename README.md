# Portfolio Optimizer

A tactical portfolio optimizer with various strategy paths that can rebalance an Alpaca account:

1. `mean-variance` — ([cvxpy](https://github.com/cvxpy/cvxpy/)) optimization of expected return vs. covariance with asset-class constraints and turnover penalties.
2. `dual-momentum` — ranks assets by trailing return, holds the top-k, exits positions that fall more than a trailing stop threshold.
3. `factor-momentum` — ranks asset-class or sleeve groups first, then selects top assets inside the strongest groups.
4. `protective-momentum` — scales risky exposure based on market breadth and moves the rest into defensive assets.

It can also backtest against historical data (primarily [yfinance](https://pypi.org/project/yfinance/) data, but other data sources can be manually uploaded via csv), and output various metrics of performance.

(**AI Disclosure**: some of the code in this repo was generated using the aid of coding tools such as Qwen Code and Codex)

## Usage
Run commands through `uv run`:

```bash
uv run portfolio-opt --help
```

Common workflows are shown below. The CLI help is the source of truth for the full option list.

`portfolio-opt --backtest-days ...` uses the native backtest engine by default. Use `--backtest-engine cvxportfolio` to run the experimental `cvxportfolio` framework path from the same CLI.


## Strategy

**Dual momentum** uses a simple but empirically strong approach:
- Rank all risky assets by trailing N-day return
- Keep only those that beat the cash-like asset's return (absolute momentum filter)
- Hold the top-k, equal-weighted
- Exit any position that drops more than the trailing stop threshold from its peak
- Otherwise fall back to defensive assets (bonds, cash)

**Mean-variance** solves:

```text
maximize    mu^T w - risk_aversion * w^T Sigma w - turnover_penalty * ||w - w_prev||_1
subject to  sum(w) = 1
            min_weight <= w <= max_weight
            asset-class bounds from model file
```

Baseline assumptions:

- Fixed asset universe
- Long-only portfolio
- Maximum per-asset weight cap
- Minimum rebalance threshold to avoid tiny orders
- Optional trailing stop-loss for drawdown protection

For a walkthrough of the original mean-variance implementation, see `docs/IMPLEMENTATION.md`.

## Install

```bash
uv sync
source .venv/bin/activate
```

`uv.lock` is checked in and should be updated with `uv lock` whenever dependencies change.

For lightweight static type checking:

```bash
uvx ty check
```

## Configure Alpaca

Set environment variables directly, or place them in a local `.env` file:

```bash
APCA_API_KEY_ID=your_key
APCA_API_SECRET_KEY=your_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
APCA_API_DATA_URL=https://data.alpaca.markets
```

`.env` is gitignored in this repository. Paper trading is the default safe target.

## Background & Research

This project implements two complementary approaches to portfolio construction:

**Dual momentum** [3] draws from the time-series and cross-sectional momentum literature [1, 2]:
- Rank all risky assets by trailing return
- Keep only those that beat the cash-like asset's return (absolute momentum filter)
- Hold the top-k, equal-weighted
- Exit any position that drops more than the trailing stop threshold from its peak
- Otherwise fall back to defensive assets (bonds, cash)

This approach avoids the estimation error that DeMiguel, Garlappi, and Uppal (2009) [4] showed makes mean-variance underperform naive diversification out-of-sample.

**Mean-variance optimization** uses [`cvxpy`](https://www.cvxpy.org/) directly — a convex optimization library solving a single-period problem: maximize expected return penalized by risk (covariance) and turnover costs. Four return-estimation methods are available:
- `sample-mean` — historical average returns (shrunken toward zero)
- `momentum` — trailing return as the expected-return signal
- `black-litterman` — market equilibrium returns blended with momentum views [5]
- `risk-parity` — equal risk-contribution weights via covariance only [6]

In our tests across 20 years of data, mean-variance underperforms dual momentum regardless of the return model (6-8% vs 13.5% annualized) because the covariance matrix pushes the optimizer toward minimum-variance positions.

**`cvxportfolio`** (Boyd et al. [7]) provides a broader multi-period framework that models transaction costs, holding costs, and planning across future steps. The `src/cvxportfolio_impl/` directory contains a side-by-side implementation. In our tests it was significantly more conservative than dual momentum (3-4% annualized vs 13-28%) because the optimizer's covariance structure pushes toward minimum-variance positions even with low risk aversion.

### References

[1] Jegadeesh, N., and S. Titman. 1993. "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." *Journal of Finance* 48 (1): 65-91. https://doi.org/10.1111/j.1540-6261.1993.tb04702.x

[2] Moskowitz, T. J., Y. H. Ooi, and L. H. Pedersen. 2012. "Time Series Momentum." *Journal of Financial Economics* 104 (2): 228-250. https://doi.org/10.1016/j.jfineco.2011.03.023

[3] Antonacci, G. 2014. *Dual Momentum Investing: An Innovative Strategy for Higher Returns with Lower Risk*. New York: McGraw-Hill. ISBN 978-0071835893.

[4] DeMiguel, V., L. Garlappi, and R. Uppal. 2009. "Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy?" *Review of Financial Studies* 22 (5): 1915-1953. https://doi.org/10.1093/rfs/hhm075

[5] Black, F., and R. Litterman. 1992. "Global Portfolio Optimization." *Financial Analysts Journal* 48 (5): 28-43. https://doi.org/10.2469/faj.v48.n5.28

[6] Maillard, S., T. Roncalli, and J. Teïletche. 2010. "The Properties of Equally Weighted Risk Contribution Portfolios." *Journal of Portfolio Management* 36 (4): 60-70. https://doi.org/10.3905/jpm.2010.36.4.060

[7] Boyd, S., E. Busseti, S. Diamond, R. Kahn, K. Koh, P. Lundgren, et al. 2017. "Multi-Period Trading via Convex Optimization." arXiv:1705.00109. https://arxiv.org/abs/1705.00109

## Run A Dry Rebalance

```bash
uv run portfolio-opt --model examples/sample_model.json --dry-run
```

This uses the current Alpaca account equity and positions, computes target weights with `cvxpy`, and prints the order plan without submitting trades.

To estimate inputs from Alpaca history instead of a static model file:

```bash
uv run portfolio-opt \
  --model examples/sample_universe.json \
  --estimate-from-history \
  --return-model momentum \
  --lookback-days 126 \
  --momentum-window 63 \
  --mean-shrinkage 0.75 \
  --turnover-penalty 0.05 \
  --dry-run
```

To allow cash and cap rebalance aggressiveness:

```bash
uv run portfolio-opt \
  --model examples/sample_universe.json \
  --estimate-from-history \
  --return-model momentum \
  --lookback-days 126 \
  --momentum-window 63 \
  --mean-shrinkage 0.75 \
  --allow-cash \
  --min-cash-weight 0.10 \
  --min-invested-weight 0.30 \
  --max-turnover 0.30 \
  --turnover-penalty 0.05 \
  --dry-run
```

To run a simple offline backtest with the same policy constraints:

```bash
uv run portfolio-opt \
  --model examples/sample_universe.json \
  --estimate-from-history \
  --return-model momentum \
  --lookback-days 126 \
  --backtest-days 252 \
  --rebalance-every 21 \
  --allow-cash \
  --min-cash-weight 0.10 \
  --min-invested-weight 0.30
```

To cache Alpaca data locally for repeatable testing:

```bash
uv run portfolio-opt \
  --model examples/sample_universe.json \
  --backtest-days 252 \
  --estimate-from-history \
  --use-cache \
  --refresh-cache
```

Then rerun without hitting Alpaca:

```bash
uv run portfolio-opt \
  --model examples/sample_universe.json \
  --backtest-days 252 \
  --estimate-from-history \
  --offline
```

The CLIs now default `XDG_CACHE_HOME` and `MPLCONFIGDIR` into the repo-local `.cache/` directory so offline runs do not try to write user-level Matplotlib or Fontconfig caches outside the workspace.

To backtest with full historical data from Yahoo Finance (going back to each ETF's inception):

```bash
uv run portfolio-opt \
  --model examples/sector_universe_pre2020.json \
  --strategy dual-momentum \
  --lookback-days 252 \
  --backtest-days 4500 \
  --rebalance-every 5 \
  --top-k 2 \
  --data-source yfinance
```

This reaches back to mid-2007 when all symbols in the universe are available,
covering the 2008 financial crisis, the 2020 pandemic crash, and everything since.
Alpaca data is capped at ~7 years; yfinance provides the full history.

To run a simple parameter sweep in backtest mode:

```bash
uv run portfolio-opt \
  --model examples/sample_universe.json \
  --estimate-from-history \
  --return-model momentum \
  --lookback-days 126 \
  --backtest-days 252 \
  --rebalance-every 21 \
  --max-turnover 0.30 \
  --sweep \
  --top-n 5
```

To run the custom dual-momentum research path instead of the mean-variance optimizer:

```bash
uv run portfolio-opt \
  --model examples/broad_universe.json \
  --strategy dual-momentum \
  --lookback-days 252 \
  --backtest-days 252 \
  --rebalance-every 21 \
  --top-k 1 \
  --offline
```

To see whether a custom strategy beats `SPY` across rolling subwindows:

```bash
uv run portfolio-opt \
  --model examples/broad_universe.json \
  --strategy dual-momentum \
  --lookback-days 252 \
  --backtest-days 252 \
  --rebalance-every 21 \
  --top-k 1 \
  --rolling-window-days 126 \
  --rolling-step-days 21 \
  --offline
```

For a wider ETF research universe with sector sleeves in addition to the broad asset-class sleeves, see `examples/sector_universe.json`. This is useful when you want to test:

- broad-beta rotation: `SPY`, `QQQ`, `IWM`, `VEA`, `VWO`
- sector rotation: `XLK`, `XLF`, `XLE`, `XLV`, `XLI`, `XLP`, `XLU`
- defensive sleeves: `BIL`, `IEF`, `TLT`, `TIP`
- real assets: `GLD`, `DBC`, `VNQ`

### Dual-Momentum Strategy (Best Known Configuration)

Backtested on `examples/sector_universe_pre2020.json` using yfinance data (mid-2007 to April 2026 — ~18 years, through the 2008 crisis):

| Configuration | Annualized Return | Volatility | Max Drawdown | Final Value |
|---|---|---|---|---|
| **Top-2 weekly + 15% trailing stop** | **13.53%** | 18.20% | 29.40% | 9.39x |
| Top-2 weekly | 13.39% | 18.59% | 30.95% | 9.19x |
| SPY benchmark | 11.87% | 19.87% | 47.17% | 7.25x |

The strategy survived the 2008 financial crisis with a 29% drawdown vs SPY's 47%,
delivering 1.7% more annualized return over the full 18-year period.

Earlier results on the 2020–2026 window only (Alpaca data):

| Configuration | Annualized Return | Volatility | Max Drawdown |
|---|---|---|---|
| Top-2 weekly (2020–2026) | 28.40% | 20.34% | 23.25% |
| Top-2 weekly + 15% stop (2020–2026) | 29.29% | 19.95% | 20.64% |

The longer yfinance window includes the 2008 crash and early-market conditions,
bringing returns closer to long-run historical averages.

**Recommended deployment:**

```bash
# Daily Mon-Fri 3:30pm ET (stop-loss checks fire daily)
uv run portfolio-opt \
  --model examples/sector_universe.json \
  --strategy dual-momentum \
  --lookback-days 252 \
  --estimate-from-history \
  --top-k 2 \
  --trailing-stop 0.15 \
  --submit
```

**Alpaca native stop-loss alternative:** Instead of running daily, you can use Alpaca's OCO (one-cancels-other) or trailing stop orders on each position. Submit a `trailing_stop` order with `trail_percent=15` alongside each entry order, so Alpaca handles the stop execution server-side. This lets you run the strategy weekly while still having daily stop protection.

If you switch to that universe, prime the Alpaca cache once before offline runs because it uses a different symbol set.

For a more global macro-style research universe with explicit Japan and international-bond sleeves, see `examples/global_universe.json`. This adds:

- country-specific developed exposure: `EWJ`
- international developed sovereign bonds: `BWX`
- emerging market bonds: `EMB`

Those sleeves are optional research tools rather than defaults. They help if you want to test whether international equity and bond rotation improves the signal, but they also add FX and country-risk noise, so they should be compared against the simpler broad and sector universes before becoming a default path.

## Submit Orders

```bash
uv run portfolio-opt --model examples/sample_model.json --submit
```

## cvxportfolio Experiment

The `cvxportfolio` implementation is an experimental framework comparison path under `src/cvxportfolio_impl/`. It is available through the main CLI with `--backtest-engine cvxportfolio`; the old `cvxportfolio-backtest` command remains as a compatibility entrypoint.

The `cvxportfolio` engine supports the same historical data-source and cache flags as native backtests, including `--data-source yfinance`, `csv`, `csv+yfinance`, `stockanalysis`, or `alpaca`.

```bash
uv run portfolio-opt \
  --model examples/sample_universe.json \
  --backtest-engine cvxportfolio \
  --lookback-days 126 \
  --backtest-days 252
```

It can also compute current target weights for order planning through the shared dry-run/submit flow:

```bash
uv run portfolio-opt \
  --model examples/sample_universe.json \
  --backtest-engine cvxportfolio \
  --lookback-days 126 \
  --data-source yfinance \
  --dry-run
```

For repeatable offline comparison runs:

```bash
uv run portfolio-opt \
  --model examples/sample_universe.json \
  --backtest-engine cvxportfolio \
  --lookback-days 126 \
  --backtest-days 252 \
  --data-source yfinance \
  --use-cache \
  --refresh-cache

uv run portfolio-opt \
  --model examples/sample_universe.json \
  --backtest-engine cvxportfolio \
  --lookback-days 126 \
  --backtest-days 252 \
  --data-source yfinance \
  --offline
```

The current best-known preset is captured in `examples/cvxportfolio_best_preset.json`.
It is the current cost-aware single-period candidate:

```bash
uv run portfolio-opt \
  --model examples/sample_universe.json \
  --backtest-engine cvxportfolio \
  --lookback-days 126 \
  --backtest-days 252 \
  --risk-aversion 0.5 \
  --mean-shrinkage 0.5 \
  --momentum-window 84 \
  --min-cash-weight 0.05 \
  --min-invested-weight 0.4 \
  --linear-trade-cost 0.001
```

For a less defensive broad-universe starting point, see `examples/cvxportfolio_broad_high_equity_preset.json`.
For a benchmark-aware starting point that keeps `SPY` as a core holding, see `examples/cvxportfolio_spy_core_preset.json`.
For a benchmark-relative starting point that measures active risk against `SPY`, see `examples/cvxportfolio_spy_benchmark_preset.json`.
For a volatility-targeted broad-universe starting point, see `examples/cvxportfolio_vol_target_preset.json`.

To compare that `cvxportfolio` configuration against the repo's best-known custom baseline on the same cached data and benchmarks:

```bash
uv run portfolio-opt \
  --model examples/sample_universe.json \
  --backtest-engine cvxportfolio \
  --lookback-days 126 \
  --backtest-days 252 \
  --risk-aversion 0.5 \
  --mean-shrinkage 0.5 \
  --momentum-window 84 \
  --min-cash-weight 0.05 \
  --min-invested-weight 0.4 \
  --linear-trade-cost 0.001 \
  --compare-custom \
  --data-source yfinance \
  --offline
```

To see how often a broad-universe policy beats `SPY` across rolling subwindows of the aligned backtest period:

```bash
uv run portfolio-opt \
  --model examples/broad_universe.json \
  --backtest-engine cvxportfolio \
  --lookback-days 126 \
  --backtest-days 252 \
  --risk-aversion 0.5 \
  --mean-shrinkage 0.5 \
  --momentum-window 84 \
  --min-cash-weight 0.0 \
  --min-invested-weight 0.7 \
  --linear-trade-cost 0.001 \
  --rolling-window-days 63 \
  --rolling-step-days 21 \
  --data-source yfinance \
  --offline
```

To keep `SPY` as a required core holding and let the optimizer allocate active tilts around it:

```bash
uv run portfolio-opt \
  --model examples/broad_universe.json \
  --backtest-engine cvxportfolio \
  --lookback-days 126 \
  --backtest-days 252 \
  --risk-aversion 0.5 \
  --mean-shrinkage 0.5 \
  --momentum-window 84 \
  --min-cash-weight 0.0 \
  --min-invested-weight 0.8 \
  --core-symbol SPY \
  --core-weight 0.35 \
  --linear-trade-cost 0.001 \
  --data-source yfinance \
  --offline
```

## Model File Format

```json
{
  "symbols": ["SPY", "QQQ", "TLT", "GLD"],
  "expected_returns": {
    "SPY": 0.09,
    "QQQ": 0.11,
    "TLT": 0.04,
    "GLD": 0.05
  },
  "covariance": [
    [0.032, 0.029, 0.006, 0.008],
    [0.029, 0.041, 0.005, 0.007],
    [0.006, 0.005, 0.018, 0.004],
    [0.008, 0.007, 0.004, 0.021]
  ]
}
```

Returns and covariance should be on the same annualized basis.

For universe-only files, you can also provide asset-class metadata and policy bounds:

```json
{
  "symbols": ["SPY", "QQQ", "TLT", "GLD"],
  "asset_classes": {
    "SPY": "equity",
    "QQQ": "equity",
    "TLT": "bond",
    "GLD": "commodity"
  },
  "class_min_weights": {
    "equity": 0.2
  },
  "class_max_weights": {
    "equity": 0.6,
    "bond": 0.4,
    "commodity": 0.3
  }
}
```

There is also a broader starting universe in `examples/broad_universe.json` with US equities, international equities, short/intermediate/long Treasuries, TIPS, gold, broad commodities, and REITs. Use that file if you want the optimizer to express more than just `SPY` versus `QQQ` plus two hedges.

The built-in benchmark block is still anchored to `SPY`, `TLT`, and equal-weight logic. That keeps comparisons stable, but on broader universes you should treat those benchmarks as reference points, not a full policy match.

## Notes

- Fractional quantity handling uses notional order sizing — Alpaca supports fractional shares for all ETFs.
- Historical data can come from **Alpaca** (~7 year limit) or **yfinance** (full ETF history, `--data-source yfinance`).
- Dual-momentum backtests survive the 2008 crisis with a 29% drawdown vs SPY's 47% (~18 year history).
- The `--trailing-stop 0.15` parameter adds per-asset stop-loss protection (15% from peak).
- Asset-class bounds can be defined in the model file to keep allocations within a portfolio policy.
- Native backtest mode supports `mean-variance`, `dual-momentum`, `factor-momentum`, and `protective-momentum`, with built-in benchmark comparisons against SPY, QQQ, 60/40 SPY/TLT, equal-weight, and half-SPY/half-cash portfolios.
- Backtest output includes final value, total and annualized return, annualized volatility, max drawdown, Sortino ratio, rebalance count, average turnover, latest target weights, cash weight, asset-class exposures, daily equity values, and benchmark summaries.
- Rolling-window comparison currently measures strategy results against SPY.
- Sweep mode runs a backtest grid over core policy parameters and returns the top results.
- `--backtest-engine cvxportfolio` runs a parallel `cvxportfolio` experiment for comparison; it is not expected to match native results exactly because it uses a different simulator, warmup convention, and transaction-cost handling.
- `uvx ty check` is the intended static type-checking entrypoint for this repo.
- The project is managed with `uv`; keep `pyproject.toml` and `uv.lock` in sync.
