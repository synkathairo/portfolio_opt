# cvxportfolio Implementation Scaffold

This package is a placeholder for a parallel `cvxportfolio`-based implementation.
It now includes a minimal backtest-only entrypoint intended for side-by-side comparison with the custom optimizer.

## Goal

Keep the current custom optimizer in `src/portfolio_opt/` as the baseline and build a separate implementation here so both approaches can be compared against:

- the same universe
- the same historical close data sources and caches
- the same backtest window
- the same benchmarks

## Intended Responsibilities

- define `cvxportfolio` policies and constraints
- translate the current portfolio policy into `cvxportfolio` terms
- run comparable backtests
- optionally emit target weights or trades for the same Alpaca execution layer

## Suggested Modules

- `policy.py`: portfolio objective and constraints
- `data.py`: adapters for historical data and forecasts
- `backtest.py`: comparison harness
- `execution.py`: adapter that converts outputs into the repo's existing order flow

## Current Entry Point

```bash
uv run portfolio-opt \
  --model examples/sample_universe.json \
  --backtest-engine cvxportfolio \
  --lookback-days 126 \
  --backtest-days 252
```

The old `cvxportfolio-backtest` command remains available as a compatibility wrapper.

This first version is intentionally narrow:

- backtest only
- momentum-style returns forecast
- user-provided market data from the configured historical close source
- cash and asset-class policy bounds where possible

The default sample universe is intentionally small. For a broader allocation experiment, try `examples/broad_universe.json`.

Sweep mode is also available:

```bash
uv run cvxportfolio-backtest \
  --model examples/sample_universe.json \
  --lookback-days 126 \
  --backtest-days 252 \
  --sweep \
  --top-n 5
```

You can also experiment with:

- `--linear-trade-cost` for a simple proportional transaction-cost term
- `--planning-horizon` to switch from single-period to multi-period optimization

The strongest current practical candidate is still the cost-aware single-period policy:

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

For an apples-to-apples framework comparison against the repo's custom baseline preset:

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

Rolling-window comparison against `SPY` is also available:

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

For a `SPY` core plus active-tilt policy:

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

## Comparison Rule

Do not replace `src/portfolio_opt/` yet. The point of this package is side-by-side evaluation, not migration by assumption.
