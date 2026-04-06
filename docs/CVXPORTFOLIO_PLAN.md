# cvxportfolio Plan

## Why Add A Second Implementation

The current codebase already answers a useful question: a custom `cvxpy` allocator with Alpaca integration is easy to understand and can be tuned. What it does not answer is whether a higher-level framework like `cvxportfolio` improves modeling speed, constraint expressiveness, or backtest realism.

## Scope

Build a second implementation under `src/cvxportfolio_impl/` with the same outer assumptions:

- symbol universe from `examples/sample_universe.json`
- Alpaca historical data
- the same benchmark set already used in backtests
- the same reporting focus: return, volatility, drawdown, turnover

## First Comparison Target

Keep the first experiment narrow:

1. Use the same ETF universe: `SPY`, `QQQ`, `TLT`, `GLD`
2. Use the same rebalance interval
3. Use a momentum-style expected return input
4. Use similar cash and exposure constraints where possible
5. Compare results directly against the custom implementation

The first scaffold now exists as a minimal backtest-only path exposed through `cvxportfolio-backtest`.

## Success Criteria

The `cvxportfolio` path is worth continuing if it clearly improves one or more of:

- modeling clarity
- backtest realism
- ease of expressing costs and constraints
- performance relative to the same benchmarks

If it does not, keep the custom implementation as the primary path.
