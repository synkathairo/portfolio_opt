# Implementation Notes

## What The App Does

This project is a single-period portfolio allocator with Alpaca integration. It does not generate trades from price patterns directly. Instead, it:

1. defines a universe of symbols
2. estimates expected returns and covariance, or reads them from a file
3. solves for target portfolio weights with `cvxpy`
4. compares target weights to the current Alpaca account
5. emits notional buy/sell orders

## Current Model

The optimizer solves a constrained mean-variance problem with a turnover penalty:

```text
maximize
  mu^T w
  - risk_aversion * w^T Sigma w
  - turnover_penalty * ||w - w_current||_1
```

Subject to:

- `min_weight <= w_i <= max_weight`
- `sum(w) = 1` by default

Interpretation:

- `mu` is the expected return estimate
- `Sigma` is the covariance matrix
- `risk_aversion` penalizes volatility
- `turnover_penalty` discourages large changes from current holdings

This is still a single-step rebalance model. It is not a multi-period planner.

## Return Models

When `--estimate-from-history` is used, the code fetches daily closes from Alpaca and always estimates covariance from close-to-close daily returns.

Expected returns can come from:

- `sample-mean`: annualized average daily return, shrunk toward zero
- `momentum`: trailing cumulative return over `--momentum-window`, also shrunk toward zero

The momentum mode is usually easier to interpret than raw sample means.

## Execution Flow

- `src/portfolio_opt/cli.py`: orchestration and JSON output
- `src/portfolio_opt/alpaca.py`: account, positions, market data, and order submission
- `src/portfolio_opt/estimation.py`: historical input estimation
- `src/portfolio_opt/optimizer.py`: convex optimization
- `src/portfolio_opt/rebalance.py`: current weights and order-plan generation

`--dry-run` does not submit orders, but it still queries Alpaca for account state and prices because the rebalance plan depends on live holdings and equity.

## Current Limitations

- no explicit cash sleeve unless the optimizer is changed to allow partial investment
- no hard turnover cap, only a penalty
- no transaction-cost model beyond turnover discouragement
- no tax logic, borrow costs, or multi-period planning
