# Literature review: reversal, momentum, and implementation risk

This note records the evidence reviewed before extending the research code. It
is a research protocol, not an investment recommendation.

For the later dual-momentum parameter experiment and its additional primary
sources, see [Momentum strategy and parameter comparison](MOMENTUM_VARIANTS_2026-09-29.md)
and its [provenance record](MOMENTUM_VARIANTS_2026-09-29.provenance.md).

## Primary findings

- [Short-term residual reversal](https://www.sciencedirect.com/science/article/abs/pii/S1386418112000468)
  reports stronger risk-adjusted reversal after removing common-factor returns
  and finds results in large-cap samples after estimated costs.
- [Da, Liu, and Schaumburg (New York Fed Staff Report 513)](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr513.pdf)
  decomposes reversal and finds the useful component is largely within-industry
  and related to non-cash-flow price shocks.
- [Frazzini, Israel, and Moskowitz, Trading Costs of Asset Pricing Anomalies](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2294498)
  finds short-term reversal is the most trading-cost-constrained of the major
  anomalies. High turnover and market impact must therefore be first-class
  constraints, not an afterthought.
- [Korajczyk and Sadka, Are Momentum Profits Robust to Trading Costs?](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00656.x)
  shows that portfolio weighting and liquidity materially change the size at
  which an apparent anomaly becomes unprofitable.
- [Daniel and Moskowitz, Momentum Crashes](https://www.kentdaniel.net/papers/published/jfe_16.pdf)
  documents momentum crashes after market declines, when volatility is high and
  the market rebounds. Any momentum sleeve needs explicit stress/regime tests.
- [Blitz, Huij, and Martens, Residual Momentum](https://doi.org/10.1016/j.jempfin.2011.01.003)
  finds lower common-factor exposure and better risk-adjusted behavior when
  momentum is ranked on residual rather than total returns.
- [Asness, Moskowitz, and Pedersen, Value and Momentum Everywhere](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12021)
  reports value and momentum premia across multiple markets and asset classes,
  supporting diversified signal and region tests rather than a single US-only
  backtest.
- [Jegadeesh, Luo, Subrahmanyam, and Titman (2025)](https://doi.org/10.1093/rfs/hhaf057)
  provides recent international evidence that short-horizon reversal and
  longer-horizon momentum vary with market conditions and investor behavior.

## What our tests imply

The SEC quality model was a fundamental strategy, not a technical one, and it
underperformed SPY in the preliminary sample. The first residual-reversal
implementation slightly beat SPY over 2016--2026 but had approximately 166%
average turnover and a drawdown above 57%. Its apparent strength was concentrated
in 2020--2021; it lagged SPY in 2016--2019 and 2023--present. This is not enough
evidence of a durable edge.

The current implementation is intentionally incomplete: its repository sample
does not contain true industry classifications, so the market/industry
neutralization is only a prototype. Results also use a survivorship-biased
historical ticker snapshot and Yahoo price coverage. They must not be treated as
live performance estimates.

## Pre-registered next experiment

Before looking at results, the next implementation will:

1. Use rolling market and industry regressions to form residual returns.
2. Rank losers within industries, with minimum liquidity and maximum position
   limits.
3. Sweep rebalance frequency and 10/25/50 bps linear costs, while reporting
   turnover and a separate market-impact sensitivity.
4. Evaluate 2016--2019, 2020--2021, 2022, and 2023--present separately, plus
   an international universe where reliable prices and classifications exist.
5. Compare against SPY, equal weight, existing momentum, and the SEC quality
   model. No candidate is accepted from one favorable regime.
