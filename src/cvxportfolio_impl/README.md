# cvxportfolio Implementation Scaffold

This package is a placeholder for a parallel `cvxportfolio`-based implementation.
It now includes a minimal backtest-only entrypoint intended for side-by-side comparison with the custom optimizer.

## Goal

Keep the current custom optimizer in `src/portfolio_opt/` as the baseline and build a separate implementation here so both approaches can be compared against:

- the same universe
- the same Alpaca data source
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
uv run cvxportfolio-backtest \
  --model examples/sample_universe.json \
  --lookback-days 126 \
  --backtest-days 252
```

This first version is intentionally narrow:

- backtest only
- momentum-style returns forecast
- user-provided market data built from Alpaca closes
- cash and asset-class policy bounds where possible

## Comparison Rule

Do not replace `src/portfolio_opt/` yet. The point of this package is side-by-side evaluation, not migration by assumption.
