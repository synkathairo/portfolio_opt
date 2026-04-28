# Portfolio Optimizer

A tactical portfolio optimizer with various strategy paths that can rebalance an Alpaca account:

1. `mean-variance` — ([cvxpy](https://github.com/cvxpy/cvxpy/)) optimization of expected return vs. covariance with asset-class constraints and turnover penalties.
2. `dual-momentum` — ranks assets by trailing return, holds the top-k, exits positions that fall more than a trailing stop threshold.
3. `factor-momentum` — ranks asset-class or sleeve groups first, then selects top assets inside the strongest groups.
4. `protective-momentum` — scales risky exposure based on market breadth and moves the rest into defensive assets.

It can also backtest against historical data (primarily [yfinance](https://pypi.org/project/yfinance/) data, but other data sources can be manually uploaded via csv), and output various metrics of performance.

(**AI Disclosure**: some of the code in this repo was generated using the aid of coding tools such as Qwen Code and Codex)

## Usage
It is recommended to run the below commands with `uv run`

`portfolio-opt`: Run a mean-variance rebalance against Alpaca.

```
options:
  -h, --help            show this help message and exit
  --model MODEL         Path to model input JSON file.
  --dynamic-universe    Fetch current index constituents dynamically instead
                        of using a model file.
  --filter-before FILTER_BEFORE
                        Only include tickers that started trading before this
                        ISO date (e.g. 2020-01-01).
  --ticker-basket [TICKER_BASKET ...]
                        Universe components for --dynamic-universe (uses
                        fetch_ticker_dict defaults if empty).
  --dynamic-universe-cache-dir DYNAMIC_UNIVERSE_CACHE_DIR
                        Directory for latest-known-good generated dynamic
                        universe model caches.
  --allow-stale-dynamic-universe
                        Use the previous cached dynamic universe if a fresh
                        fetch fails.
  --max-stale-dynamic-universe-days MAX_STALE_DYNAMIC_UNIVERSE_DAYS
                        Maximum age in days for --allow-stale-dynamic-universe
                        fallback.
  --risk-aversion RISK_AVERSION
  --min-weight MIN_WEIGHT
  --max-weight MAX_WEIGHT
  --rebalance-threshold REBALANCE_THRESHOLD
  --turnover-penalty TURNOVER_PENALTY
  --allow-cash          Allow the optimizer to leave part of the portfolio in
                        cash.
  --min-cash-weight MIN_CASH_WEIGHT
                        Minimum cash weight to hold when --allow-cash is
                        enabled.
  --max-turnover MAX_TURNOVER
                        Hard cap on one-step turnover, measured as
                        sum(abs(target-current)).
  --min-invested-weight MIN_INVESTED_WEIGHT
                        Minimum total risky-asset weight when cash is allowed.
  --estimate-from-history
                        Estimate expected returns and covariance from Alpaca
                        daily bars.
  --lookback-days LOOKBACK_DAYS
  --mean-shrinkage MEAN_SHRINKAGE
                        Shrink sample mean returns toward zero to reduce
                        estimation noise.
  --return-model {sample-mean,momentum,black-litterman,risk-parity}
                        How to estimate expected returns when using
                        --estimate-from-history.
  --strategy {mean-variance,dual-momentum,factor-momentum,protective-momentum}
                        Strategy for live or backtest rebalancing. Momentum
                        strategies use live prices when --estimate-from-
                        history is set.
  --momentum-window MOMENTUM_WINDOW
                        Trailing trading-day window used by the momentum
                        return model.
  --top-k TOP_K         Number of assets to hold in dual momentum mode.
  --factor-top-k FACTOR_TOP_K
                        Number of top factor/sleeve groups to search in factor
                        momentum mode.
  --dual-momentum-weighting {equal,score,inverse-vol,softmax}
                        How to weight the selected basket in dual momentum
                        mode.
  --softmax-temperature SOFTMAX_TEMPERATURE
                        Temperature for softmax weighting in dual momentum
                        mode.
  --absolute-momentum-threshold ABSOLUTE_MOMENTUM_THRESHOLD
                        Minimum trailing return required for dual momentum if
                        no cash proxy is present.
  --target-vol TARGET_VOL
                        Target annualized portfolio volatility for the risky
                        basket (vol targeting).
  --vol-window VOL_WINDOW
                        Trailing trading-day window used to estimate
                        volatility for --target-vol.
  --max-single-weight MAX_SINGLE_WEIGHT
                        Maximum weight for any single asset in the dual
                        momentum basket.
  --trailing-stop TRAILING_STOP
                        Trailing stop-loss threshold per asset (e.g. 0.08 to
                        exit an 8% drawdown from peak).
  --basket-opt {mean-variance}
                        How to size the momentum-selected basket (overrides
                        --dual-momentum-weighting).
  --basket-risk-aversion BASKET_RISK_AVERSION
                        Risk aversion for basket mean-variance optimization.
  --breadth-min-risky BREADTH_MIN_RISKY
                        Minimum total risky exposure for protective momentum.
  --breadth-max-risky BREADTH_MAX_RISKY
                        Maximum total risky exposure for protective momentum.
  --defensive-weighting {equal}
                        How protective momentum allocates capital not assigned
                        to risky assets.
  --data-source {alpaca,yfinance,csv,csv+yfinance,stockanalysis}
                        Source for historical price data in backtest mode.
  --csv-dir CSV_DIR     Directory of local OHLCV CSV files when --data-source
                        csv is used. Rows must be
                        symbol,date,open,high,low,close,volume.
  --csv-write-json-cache
                        Write provider-neutral JSON close caches from --csv-
                        dir before running.
  --stockanalysis-start STOCKANALYSIS_START
                        Start date for --data-source stockanalysis chart JSON.
  --stockanalysis-end STOCKANALYSIS_END
                        End date for --data-source stockanalysis chart JSON.
                        Defaults to today.
  --yfinance-max-workers YFINANCE_MAX_WORKERS
                        Maximum concurrent yfinance symbol downloads when
                        --data-source yfinance is used.
  --yfinance-retry-delay YFINANCE_RETRY_DELAY
                        Seconds to wait between yfinance retry attempts.
  --yfinance-symbol-delay YFINANCE_SYMBOL_DELAY
                        Seconds to wait between yfinance symbol downloads when
                        --yfinance-max-workers is 1.
  --benchmark BENCHMARK
                        Additional benchmark ticker to compare in backtest
                        mode. Can be repeated, e.g. --benchmark ^HSI.
  --backtest-days BACKTEST_DAYS
                        Run a simple offline backtest over this many trading
                        days instead of a live rebalance.
  --rebalance-every REBALANCE_EVERY
                        Trading-day interval between rebalances in backtest
                        mode.
  --trading-days-per-year TRADING_DAYS_PER_YEAR
                        Trading sessions per year used for annualized metrics.
  --rolling-window-days ROLLING_WINDOW_DAYS
                        If set, compare the strategy to SPY over rolling
                        windows of this many trading days.
  --rolling-step-days ROLLING_STEP_DAYS
                        Trading-day step between rolling comparison windows.
  --sweep               Run a simple parameter sweep in backtest mode.
  --top-n TOP_N         Number of top parameter combinations to show in sweep
                        mode.
  --submit              Submit market orders to Alpaca. Default behavior is
                        dry-run output only.
  --use-cache           Use cached Alpaca data when available.
  --refresh-cache       Refresh cached Alpaca data from the API.
  --offline             Use cached data only and never call Alpaca.
  --dry-run             Explicit dry-run mode.
```


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

There is now a separate comparison path under `src/cvxportfolio_impl/`.

```bash
uv run cvxportfolio-backtest \
  --model examples/sample_universe.json \
  --lookback-days 126 \
  --backtest-days 252
```

For repeatable offline comparison runs:

```bash
uv run cvxportfolio-backtest \
  --model examples/sample_universe.json \
  --lookback-days 126 \
  --backtest-days 252 \
  --use-cache \
  --refresh-cache

uv run cvxportfolio-backtest \
  --model examples/sample_universe.json \
  --lookback-days 126 \
  --backtest-days 252 \
  --offline
```

The current best-known preset is captured in `examples/cvxportfolio_best_preset.json`.
It is the current cost-aware single-period candidate:

```bash
uv run cvxportfolio-backtest \
  --model examples/sample_universe.json \
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
uv run cvxportfolio-backtest \
  --model examples/sample_universe.json \
  --lookback-days 126 \
  --backtest-days 252 \
  --risk-aversion 0.5 \
  --mean-shrinkage 0.5 \
  --momentum-window 84 \
  --min-cash-weight 0.05 \
  --min-invested-weight 0.4 \
  --linear-trade-cost 0.001 \
  --compare-custom \
  --offline
```

To see how often a broad-universe policy beats `SPY` across rolling subwindows of the aligned backtest period:

```bash
uv run cvxportfolio-backtest \
  --model examples/broad_universe.json \
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
  --offline
```

To keep `SPY` as a required core holding and let the optimizer allocate active tilts around it:

```bash
uv run cvxportfolio-backtest \
  --model examples/broad_universe.json \
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
- Backtest mode supports `mean-variance`, `dual-momentum`, and `dual-momentum` with rolling-window comparison vs SPY.
- Sweep mode runs a backtest grid over core policy parameters and returns the top results.
- `src/cvxportfolio_impl/` contains a parallel `cvxportfolio` experiment for comparison.
- `uvx ty check` is the intended static type-checking entrypoint for this repo.
- The project is managed with `uv`; keep `pyproject.toml` and `uv.lock` in sync.
