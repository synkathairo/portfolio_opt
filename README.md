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

## Install

```bash
uv sync
source .venv/bin/activate
```

`uv.lock` is checked in and should be updated with `uv lock` whenever dependencies change.

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

## Submit Orders

```bash
uv run portfolio-opt --model examples/sample_model.json --submit
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
- The project is managed with `uv`; keep `pyproject.toml` and `uv.lock` in sync.
