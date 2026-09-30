# Momentum strategy and parameter comparison (2026-09-29)

This note connects a bounded local benchmark to primary research on momentum,
diversification, trading costs, and backtest selection. It is exploratory
evidence, not a production strategy selection. The [earlier multi-strategy
benchmark](BENCHMARK_2026-09-04.md) ended on 2026-09-04; this separate cached
price panel ends on 2026-05-15, so their full-period CAGRs should not be
compared directly.

A later [Nasdaq-100-style universe sensitivity](NASDAQ100_SENSITIVITY_2026-09-29.md)
shows how strongly the candidate ranking changes with the symbol set.

## Literature findings and how they informed the test

| Primary study | Finding in the study | Connection to this experiment |
|---|---|---|
| [Moskowitz, Ooi, and Pedersen (2012), *Time Series Momentum*](https://doi.org/10.1016/j.jfineco.2011.11.003) | Past returns persisted over roughly one to twelve months across the futures markets they studied, with partial reversal at longer horizons. | Motivates testing both 126- and 252-trading-day lookbacks. Their futures portfolios differ from our long-only symbol basket, so the paper does not predict our returns. |
| [Antonacci (2017), *Risk Premia Harvesting Through Dual Momentum*](https://doi.org/10.2139/SSRN.2042750) | Tests relative momentum together with an absolute momentum filter and reports lower drawdowns from the absolute filter in its multi-asset setup. | Our baseline ranks positive-momentum symbols and may hold cash when none passes the threshold. It is not Antonacci's specific allocation rule or universe. |
| [Daniel and Moskowitz (2016), *Momentum Crashes*](https://www.nber.org/papers/w20439) | Documents severe momentum losses in high-volatility rebound states. | Motivates reporting maximum drawdown and regime periods, and testing broader baskets and slower rebalancing. Their momentum portfolios are not identical to ours. |
| [Korajczyk and Sadka (2004), *Are Momentum Profits Robust to Trading Costs?*](https://doi.org/10.1111/j.1540-6261.2004.00656.x) | Momentum capacity and profitability depend materially on trading costs and portfolio construction. | Motivates 10- and 30-basis-point turnover-cost runs. Our linear cost model omits nonlinear market impact, spread variation, and taxes. |
| [DeMiguel, Garlappi, and Uppal (2009), *Optimal Versus Naive Diversification*](https://doi.org/10.1093/rfs/hhm075) | In their datasets, estimated optimal allocations did not consistently beat equal weighting out of sample. | Motivates equal weighting as a simple basket-sizing and universe benchmark. This paper does not test our score-weighted momentum rule. |
| [Bailey et al. (2015), *The Probability of Backtest Overfitting*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) | Selecting a backtest winner from many tried rules can produce misleading apparent performance. | The 13 configurations below are inspected on the same history. Their ranking is hypothesis-generating; the paired interval does not correct for this selection. |

These papers support the *questions* tested here. None establishes that a
five-stock, score-weighted portfolio or a 63-day rebalance is optimal for this
repository's universe.

## Experiment

- Input: the same 49-symbol, current-survivor sample as the earlier SEC-quality
  comparison, plus SPY. Prices came from local Yahoo adjusted-close caches.
  All candidates use the 2,353 return days from 2017-01-03 through 2026-05-15
  after a common 252-trading-day warmup.
- Original dual-momentum baseline: 252-day lookback, top two, equal weighting,
  21-day rebalance, 15% simulated trailing stop, and zero absolute-momentum
  threshold.
- Proposed command's research analogue: 252-day lookback, top five, score
  weighting, 50% single-position cap, 21-day rebalance, and the same 15% stop.
  The CLI command's model file and live dynamic universe were **not** tested.
- Alternatives: top-five equal and inverse-volatility weighting, no stop,
  126-day lookback, 63-day rebalance, 20% volatility target, sector-group
  factor momentum, the existing regime-adaptive rule, SPY buy-and-hold, and
  monthly equal weight across all 49 symbols.
- Costs: 10 and 30 basis points per unit of turnover, applied by the existing
  backtest functions. The 15% stop uses the backtest's daily close logic; it
  does not reproduce broker stop fills or intraday gaps.

The [runner](../scripts/benchmark_momentum_variants.py), [input
manifest](data/momentum_variant_input_2026-09-29.json), and [compact
machine-readable results](data/momentum_variant_benchmark_2026-09-29.json) are
tracked. The full daily curves and price inputs remain under ignored `.cache/`;
the tracked result records each price-cache filename and SHA-256 digest. Exact
reproduction requires those local price caches; the earlier ignored raw report
is optional and only supplies a baseline cross-check.

## Results

| Candidate | CAGR, 10 bps | CAGR, 30 bps | Volatility, 10 bps | Max drawdown, 10 bps | Sortino, 10 bps |
|---|---:|---:|---:|---:|---:|
| Top 5 score, no stop | 27.3% | 25.5% | 28.2% | 40.8% | 1.34 |
| Top 5 score, 50% cap | 26.4% | 24.3% | 25.0% | 33.6% | 1.45 |
| Top 5 score, no cap | 26.4% | 24.3% | 25.0% | 33.6% | 1.45 |
| Top 5 equal, 50% cap | 26.2% | 24.3% | 23.1% | 30.4% | 1.57 |
| Original top 2 equal | 25.6% | 23.1% | 31.0% | 39.2% | 1.16 |
| Top 5 score, 63-day rebalance | 25.6% | 24.2% | 20.8% | 21.1% | 1.75 |
| Top 5 inverse-volatility, 50% cap | 24.6% | 22.5% | 22.4% | 29.5% | 1.50 |
| Top 5 score, 20% volatility target | 19.9% | 18.3% | 19.0% | 27.1% | 1.46 |
| Top 5 score, 126-day lookback | 17.4% | 14.8% | 24.5% | 37.6% | 0.99 |
| Factor momentum, 3 sectors | 15.9% | 13.7% | 21.8% | 28.1% | 1.00 |
| SPY buy-and-hold | 15.4% | 15.3% | 18.3% | 33.7% | 1.18 |
| Equal weight, 49 symbols | 13.6% | 13.4% | 18.1% | 37.8% | 1.05 |
| Regime adaptive | 10.7% | 7.7% | 18.6% | 31.1% | 0.81 |

The proposed top-five score rule had higher observed CAGR and lower volatility
and drawdown than the original top-two rule. Its annualized paired log-return
advantage was only **0.6%**, with a 95% interval of **-7.8% to +9.0%** from
2,000 paired circular 21-day block resamples. That interval crosses zero and
does not account for trying multiple configurations. The 50% cap did not bind
in this sample: capped and uncapped equity curves matched to floating-point
precision.

The 63-day rebalance kept roughly the original CAGR while reducing observed
volatility and drawdown. It is the clearest risk-focused follow-up candidate,
not an established improvement. Its paired annual log-return interval versus
the original is **-10.7% to +10.6%**. The no-stop variant had the highest
sample CAGR but a 40.8% drawdown. Removing the stop is therefore not a clear
improvement when drawdown matters.

| Candidate, 10 bps | 2017-2019 CAGR | 2020-2021 | 2022-2023 | 2024-May 2026 |
|---|---:|---:|---:|---:|
| Original top 2 equal | 27.7% | 38.5% | 0.1% | 35.6% |
| Top 5 score | 27.5% | 32.5% | 10.9% | 32.9% |
| Top 5 equal | 29.0% | 30.9% | 12.4% | 30.1% |
| Top 5 score, 63-day rebalance | 26.1% | 35.9% | 14.0% | 27.4% |
| SPY buy-and-hold | 14.8% | 22.8% | 1.3% | 22.3% |

The baseline rerun on these cached prices agrees closely with the previous
research runner over the *same* dates: CAGR differs by only 0.04 percentage
points. The slight difference is consistent with the two Yahoo cache vintages.

## Interpretation and next test

The observed advantage belongs to this selected, current-survivor sample. The
live dynamic-universe model has a different membership; historical point-in-time
constituents and delisted names are absent. These periods were inspected during
strategy development, so none is an untouched holdout. Linear costs do not
capture market impact, taxes, broker stop execution, or operational failures.

The next comparison should predeclare a small shortlist: original top-two
equal, top-five score, top-five equal, and top-five score with 63-day rebalance.
Run them on point-in-time constituents with the exact intended model-building
rules, inspect later untouched data, and paper-trade order plans and stop
behavior. Do not promote the historical winner directly to live submission.
