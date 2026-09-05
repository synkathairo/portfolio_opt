# Strategy Research Protocol

The research objective is positive, repeatable excess return over an appropriate
regional benchmark after trading, financing, and borrow costs. A higher backtest
terminal value on one hand-picked interval is not sufficient evidence.

## Comparison set

Every candidate must be compared with:

- the regional buy-and-hold benchmark;
- a low-cost fixed allocation;
- native mean-variance, dual-momentum, factor-momentum, and protective-momentum;
- the regime-adaptive research strategy;
- the strongest previously accepted candidate.

Fixed SPY/QQQ allocations are benchmark baselines, not alpha strategies.

## Regions and benchmarks

| Region | Primary benchmark | Research universe |
|---|---|---|
| United States | SPY | point-in-time US large-cap or liquid sector universe |
| Developed ex-US | EFA | liquid developed-country ETFs or point-in-time equities |
| Emerging markets | EEM | liquid emerging-country ETFs or point-in-time equities |

Country ETF results are USD returns and therefore include currency exposure.
Local-currency claims require local price series and an explicit FX hedge policy.

## Regimes

Report results separately for at least:

- global financial crisis: October 2007 through June 2009;
- post-crisis expansion: July 2009 through December 2019;
- pandemic shock and rebound: January 2020 through December 2021;
- inflation and rate shock: January 2022 through December 2023;
- recent holdout: January 2024 onward.

Where instrument inception dates prevent a regime test, mark the result missing;
do not replace it with a more favorable interval.

## Validation rules

1. Fix the hypothesis, features, parameter grid, and costs before opening holdout
   results.
2. Use expanding or rolling walk-forward fitting. Signals at day `t` may only use
   observations available before the return earned after day `t`.
3. Use point-in-time constituents and delisted assets for stock-level studies.
4. Report CAGR, volatility, maximum drawdown, Sortino ratio, information ratio,
   turnover, exposure, and the fraction of rolling windows beating the benchmark.
5. Charge bid/ask and market-impact costs, financing on leverage, and borrow fees
   on shorts. Stress all costs at 1x, 2x, and 3x the base assumption.
6. Correct for multiple testing. Retain every attempted configuration, including
   failures, and report bootstrap confidence intervals for excess return.

## Acceptance gate

A candidate remains research-only unless it:

- has positive holdout excess return after base and 2x costs;
- beats its benchmark in a majority of three-year rolling windows;
- is not dependent on a single named regime;
- preserves the result under small parameter perturbations;
- has operationally feasible turnover and Alpaca order requirements;
- passes paper trading with conservative local fill simulation.

## Current findings

- A 60% SPY / 40% QQQ allocation beat SPY in the sampled US history, but this is
  explained by a persistent US growth tilt and is retained only as a baseline.
- Beta-neutral sector residual mean reversion failed after 10 bps trading costs
  and a 3% annual short-borrow assumption. Its turnover was too high.
- EWMA volatility management reduced crisis drawdowns across SPY, EFA, and EEM,
  but did not consistently improve raw return after financing costs.
- Leveraged global inverse-volatility allocation reduced full-period drawdown but
  underperformed SPY in raw return.

The next primary hypothesis should be a point-in-time cross-sectional equity
model combining non-price information (quality, value, earnings revisions, and
event surprise) with beta/sector neutralization. Alpaca can provide market data
and execution, but historical point-in-time fundamental or estimates data must
come from a suitable data source or a carefully constructed SEC dataset.
