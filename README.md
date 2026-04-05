# Portfolio Optimizer Starter

This is a minimal Python starter for a long-only mean-variance portfolio optimizer that can rebalance an Alpaca account.

## Strategy

The optimizer solves:

```text
maximize    mu^T w - risk_aversion * w^T Sigma w
subject to  sum(w) = 1
            min_weight <= w <= max_weight
```

Baseline assumptions:

- Fixed asset universe
- Long-only portfolio
- Fully invested
- Maximum per-asset weight cap
- Minimum rebalance threshold to avoid tiny orders

This is an allocation engine, not a signal-generation system. The strategy quality depends on how you estimate expected returns and covariance.

For a walkthrough of the current implementation and model assumptions, see `docs/IMPLEMENTATION.md`.
For the planned parallel `cvxportfolio` experiment, see `docs/CVXPORTFOLIO_PLAN.md`.

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

## Why This Is Not `cvxportfolio`

The linked paper, "Multi-Period Trading via Convex Optimization" (Boyd et al., arXiv:1705.00109), describes a broader framework that balances expected return, risk, transaction cost, and holding cost across multiple periods, executing only the first trade in a planned sequence.

This starter does not implement that framework. It uses a simpler single-period mean-variance optimization with basic weight constraints, then translates the resulting target weights into Alpaca orders. It uses `cvxpy` directly, not `cvxportfolio`.

That means the current code does not yet model:

- Multi-period planning
- Transaction cost penalties in the objective
- Holding or borrow costs
- Forecast evolution across future steps
- Model predictive control style replanning

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
- defensive sleeves: `SGOV`, `IEF`, `TLT`, `TIP`
- real assets: `GLD`, `DBC`, `VNQ`

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

- Fractional quantity handling is intentionally conservative and uses notional order sizing logic derived from current prices.
- The current code pulls latest trade prices from Alpaca for order sizing.
- Historical estimation can use either annualized sample mean returns or a simpler trailing momentum signal.
- The optimizer can optionally hold cash instead of forcing every dollar into risky assets.
- A hard `max_turnover` constraint can be used alongside the soft turnover penalty.
- The soft turnover penalty is scaled down automatically when the account is mostly in cash.
- Asset-class bounds can be defined in the model file to keep allocations within a portfolio policy.
- Backtest mode reuses the same optimizer with rolling historical estimates and periodic rebalancing.
- Backtest output includes simple fixed-weight benchmarks for comparison.
- Sweep mode runs a small backtest grid over core policy parameters and returns the top results.
- `src/cvxportfolio_impl/` is reserved for a side-by-side `cvxportfolio` experiment.
- `uvx ty check` is the intended static type-checking entrypoint for this repo.
- The project is managed with `uv`; keep `pyproject.toml` and `uv.lock` in sync.
