# Portfolio research log

This is a compact record of the decisions and findings from the strategy
research conversation. Numbers below are research outputs, not forecasts or
investment advice.

The numerical results in this log were generated before the residual-score,
between-rebalance weight-drift, and adjusted-price valuation fixes. They are
retained as historical experiment notes and must be rerun before drawing
comparisons from the current implementation.

## Objective and standards

The objective is to investigate whether a diversified, long-only strategy can
beat SPY after realistic costs. No result is accepted from one favorable period.
Every candidate should be compared with SPY, equal weight, existing momentum,
and the SEC-quality model across multiple market regimes, with turnover and
survivorship limitations disclosed.

HFT was deprioritized: Alpaca access and the available historical data do not
provide a credible latency or market-microstructure edge for this project.

## Data decisions

- Alpha Vantage works with the configured key. Its earnings endpoint provides
  reported date, reported/estimated EPS, surprise, surprise percentage, and
  report time, but the standard allowance is only 25 requests/day. Its current
  estimate-revision endpoint is not a historical point-in-time archive, so it
  is not used as if it were one.
- SEC EDGAR is the primary fundamentals source: no API key, filing dates, and
  cacheable companyfacts/submissions endpoints. The client requires a
  descriptive `SEC_USER_AGENT`, rate-limits requests to at most 10/sec, and
  caches raw responses.
- Prices currently come from Yahoo Finance for research only. Historical
  ticker files and current SEC ticker mappings are not a complete delisted or
  point-in-time constituent database, so survivorship bias remains a limitation.

## Strategies tested

1. Dual/factor/protective momentum and regime-adaptive variants: useful
   baselines, but not evidence of persistent alpha.
2. Fixed allocations and volatility/risk-parity variants: useful drawdown and
   benchmark controls, generally not SPY-beating alpha.
3. SEC quality: annual point-in-time profitability, cash-flow, growth, and
   leverage rankings. The preliminary version did not establish robust alpha.
4. Residual short-term reversal: buys recent market/sector-relative losers with
   inverse-volatility sizing. This candidate is awaiting revalidation.

## Historical empirical results (invalidated by implementation fixes)

The first 20-symbol alphabetical run was rejected as a smoke test, not a valid
study. The runner now samples a sector-labeled 2016-filtered Nasdaq/S&P basket,
filters insufficient price histories, and reports regimes separately.

In the latest 59-symbol run (2016-01-04 through 2026-09-04, 10 bps linear
turnover cost):

- SEC quality: 16.10% CAGR, 36.68% drawdown, 1.39 Sortino.
- Residual reversal: 16.23% CAGR, 50.85% drawdown, 1.05 Sortino, 167.5%
  average turnover.
- SPY: 15.28% CAGR, 33.72% drawdown, 1.21 Sortino.
- Equal weight: 14.79% CAGR, 38.01% drawdown, 1.20 Sortino.

Residual reversal by regime:

| Period | Residual reversal | SPY |
|---|---:|---:|
| 2016–2019 | 26.5% CAGR | 14.7% |
| 2020–2021 | 83.6% | 22.8% |
| 2022 | 2.8% | -18.9% |
| 2023–present | 17.7% | 22.7% |

The aggregate result is dominated by 2020–2021 and is not robust evidence.

The rolling-regression residual signal was then constrained to 50% turnover per
rebalance. In the same 59-symbol sample, its results were:

| Linear cost | Residual CAGR | SPY CAGR | Residual drawdown | Residual Sortino |
|---:|---:|---:|---:|---:|
| 10 bps | 13.5% | 15.3% | 40.6% | 1.03 |
| 25 bps | 12.8% | 15.3% | 39.7% | 0.99 |
| 50 bps | 10.4% | 15.2% | 37.8% | 0.81 |

At every tested cost the capped signal lagged SPY over the full sample and in
2023–present. It led SPY in 2016–2019 and 2020–2021, but that concentration is
exactly why it is not accepted as robust alpha. The capped reports and plots
are retained under `.cache/sector_residual_capped.*`,
`.cache/sector_residual_25bps.*`, and `.cache/sector_residual_50bps.*`.

## Literature conclusions

The literature supports testing residual and within-industry reversal, but also
warns that it is highly turnover- and liquidity-constrained. Momentum returns
can crash after market declines and sharp rebounds. Value and momentum premia
appear across regions and asset classes, suggesting signal and geographic
diversification rather than a single US signal. Sources and links are collected
in [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md).

## Historical strategy bake-off (invalidated by implementation fixes)

The research runner now includes the repository's dual-momentum baseline and a
custom online expert selector (`regime_adaptive`) in addition to SEC quality,
residual reversal, a quality-plus-momentum blend, SPY, and equal weight. The comparison uses the same
59-symbol stock sample, 2016-01-04 through 2026-09-04, 21-day rebalances, and
10 bps cost. Dual momentum is configured as a 252-day lookback, top two
holdings, equal weights, and a 15% trailing stop; the adaptive selector uses a
126-day trend sleeve, 5-day reversal sleeve, and 63-day sleeve-selection
window.

| Strategy | CAGR | Drawdown | Sortino | Avg. turnover |
|---|---:|---:|---:|---:|
| Dual momentum | 19.8% | 44.8% | 0.86 | 75.4% |
| Quality + momentum (50/50) | 17.2% | 33.7% | 1.32 | 57.9% |
| SEC quality | 16.1% | 36.7% | 1.39 | 7.7% |
| SPY | 15.3% | 33.7% | 1.21 | 0.8% |
| Equal weight | 14.8% | 38.0% | 1.20 | 6.2% |
| Residual reversal (50% cap) | 13.3% | 38.3% | 1.04 | 50.0% |
| Custom regime-adaptive selector | 9.2% | 30.6% | 0.75 | 107.8% |

At 50 bps, dual momentum fell to 14.6% CAGR and quality-plus-momentum fell to
14.0%, both below SPY at 15.2%; the custom adaptive selector fell to 6.0%.
Dual momentum led in 2016–2019 and
2020–2021 but lagged SPY in 2023–present at 50 bps. The apparent 10 bps edge is
therefore highly turnover- and sample-sensitive. On the intended 19-ETF
diversified universe, the cached canonical comparison produced 13.3% CAGR and
32.1% drawdown for top-two equal-weight dual momentum versus 12.4% and 47.2%
for SPY before transaction costs.

The quality-plus-momentum blend improved the quality-only risk profile at 10
bps, but its higher turnover erased the return advantage at 50 bps. It is a
reasonable follow-up candidate only with a better execution model, sector and
region diversification, and a held-out evaluation.

### Value + quality + trend result

A predeclared low-turnover composite was tested: 40% filing-date-safe quality,
40% earnings yield (annual net income divided by filed shares outstanding and
the current price), and 20% six-month trend. It selected at most two stocks per
sector, held up to ten names, and rebalanced quarterly. On the same 59-symbol
2016–2026 sample at 10 bps it returned 12.9% CAGR, had a 40.9% drawdown, 0.99
Sortino, and 51.2% average turnover. SPY returned 15.3% and dual momentum
19.8%. The candidate beat SPY in 2016–2019 and 2022, but lagged substantially
in 2020–2021 and 2023–present. It is rejected in this form rather than tuned
against the same sample.

The custom adaptive selector is the intentionally nonstandard candidate: it
allocates to whichever of a trend sleeve and a trend-filtered reversal sleeve
has performed better recently. It reduced drawdown in this stock-only test but
did not improve returns and incurred excessive turnover. It should not be
promoted without a defensive multi-asset universe and strict out-of-sample
validation.

## Automatic multi-asset test

The new `run_etf_automatic_research.py` runner models a scheduled process:
download/cache adjusted closes, compute signals only at each rebalance, apply
turnover costs, and emit a report. On the cached 19-ETF universe over the
available common history (10 bps cost), ordinary dual momentum produced 11.6%
CAGR with a 28.9% drawdown; inverse-volatility trend targeting produced 9.4%
CAGR with a 25.8% drawdown; SPY produced 12.4% CAGR with a 47.2% drawdown; and
60/40 SPY/IEF produced 9.0% CAGR with a 28.4% drawdown. At 50 bps, dual
momentum fell to 8.3% CAGR and the inverse-volatility variant to 6.4%, while
SPY remained at 12.4%. The volatility-targeted version is therefore a risk
control, not an established return enhancer.

## Proposed next strategies and implementation feasibility

These proposals are research candidates, not claims of expected profit.

| Candidate | Intended exposure | Shorting required? | Alpaca fit |
|---|---|---:|---|
| Value + quality + trend | Long-only stocks, sector-aware | No | Good for equity/ETF orders; valuation data must be point-in-time |
| Filing/earnings events | Usually long-only, optionally hedged | No for first version | Good for stocks; SEC data is accessible, while Alpha Vantage is request-limited |
| Market-neutral statistical arbitrage | Long one leg, short another | Yes | Partial; borrow, margin, recalls, and synchronized execution are required |
| Multi-asset carry | Long/short asset sleeves | Often | ETF proxies are practical; true futures/FX carry is outside current scope |
| Options volatility | Options or spreads | Often | Possible only with account approval; historical chains and slippage are major gaps |
| Regime-conditioned allocation | Equities, bonds, commodities, or cash | No | Good fit for scheduled ETF rebalancing |

The first implementation should be the long-only value/quality/trend strategy
and a regime-conditioned ETF allocation. Event-driven signals are a second
low-frequency candidate.

### Shorting and pair-trading risks

Shorting is not required for most proposals. It adds theoretically unlimited
loss, gap risk, margin calls, borrow fees, recalls, forced buy-ins, dividend
obligations, and liquidity risk. A pair trade is a matched long and short
position intended to reduce market exposure; it does not remove those risks.
The relationship can break, the short leg can become unborrowable, and the two
orders can fill at different prices. A credible market-neutral backtest must
model borrow availability and fees, margin rules, legging risk, and forced
exits.

### Alpaca scope

Alpaca is suitable for scheduled long equity/ETF automation and may support
margin, short-equity, and options workflows depending on account eligibility
and API permissions. Availability, borrow, and option approval must be checked
at order time. Alpaca should not be assumed to provide the futures/FX access,
historical options surfaces, colocated infrastructure, or borrow guarantees
needed for institutional carry, options-volatility, or HFT strategies. Shorting
and derivatives research will remain backtest-only until those execution inputs
are explicitly available.

## Decision and next steps

The rolling market/industry regressions, turnover cap, and 10/25/50 bps cost
sensitivities are now implemented. The current decision is to reject this
residual-reversal configuration as a deployable outperformer. Future work
should use point-in-time/delisted universes and pre-registered, genuinely
out-of-sample tests (including other regions or assets) before adding more
signals or connecting execution.

## Conversation findings (succinct)

- Momentum is only one family of signals. Trend/momentum, value, quality,
  carry, defensive allocation, mean reversion, residual reversal, and
  statistical/market-neutral approaches have different return drivers; they
  should not be treated as interchangeable momentum variants.
- A strategy that beats SPY in one recent US bull market is not sufficient:
  tests must include bear, rebound, high-volatility, and sideways regimes,
  multiple start/end dates, and—when data permits—non-US regions. A US-only
  result may be beta, growth/style exposure, or survivorship bias rather than
  skill.
- Fixed SPY/QQQ or leveraged/risk-parity mixes can improve a particular
  return/drawdown trade-off, but are allocations or factor tilts, not proof of
  stock-selection alpha. Fundamental quality is genuinely fundamental rather
  than technical, but the preliminary SEC test did not prove robust
  outperformance.
- Pure technical strategies may have an edge, but realistic costs, turnover,
  liquidity, borrow, slippage, and capacity are decisive. The residual
  reversal result is therefore a hypothesis requiring cost sensitivity and
  out-of-sample/regime validation, not a claim of a deployable edge.
- HFT is not a credible route in this project: Alpaca and the available daily
  data do not provide colocated latency, order-book, or execution advantages.
  Quant-fund headline multiples are not directly reproducible without their
  leverage, universe, data, infrastructure, and risk controls.
- Alpha Vantage is useful for selected event data but its roughly 25
  requests/day limit requires caching and careful request budgeting. SEC EDGAR
  is better for point-in-time filing fundamentals; Yahoo prices are convenient
  research data but do not solve delisted-security or historical-membership
  bias. No live-trading conclusion follows from these preliminary studies.
