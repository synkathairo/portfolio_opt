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

## ETF-universe and risk-control comparison (2026-10-09)

An offline, matched comparison was run across `examples/sector_universe.json` and
seven `sector_alternatives_universe*` variants. It used daily Top-2 equal-weight
dual momentum and a mean-variance momentum configuration, with no transaction
costs. Results and curves are under `.cache/universe_comparison/`; the four
comparison charts are `plots/universe_comparison_{long,recent}_{dm2_daily,mean_variance}_2026-10-09.png`.

- Long window: 2,929 backtest days, 252-day lookback, 2014-09-22 to
  2026-05-15. Short-history assets were automatically dropped (including
  WTLS, AVUV, FTLS, DBMF, HEQT/HEGD, and the equity-income ETFs). Thus this
  is not a full-basket test. Original sector and v3 Top-2 dual momentum both
  returned 16.44% CAGR, 22.21% volatility, and 30.90% maximum drawdown;
  adding BTAL lowered CAGR to 10.85% and volatility to 21.35%, with the same
  maximum drawdown.
- Recent common window: 850 backtest days, 63-day lookback,
  2022-12-22 to 2026-05-15. WTLS was dropped; other v3 additions had enough
  history. Top-2 dual momentum returned 23.11% CAGR for the original sector
  universe, 20.86% for v3, and 18.78% for v3-beta. Maximum drawdown was
  24.28% for all three. SPY returned 23.27% CAGR with 15.17% volatility and
  18.76% drawdown over this same window. The cache endpoint is May 15 despite
  newer file timestamps, so these runs omit later 2026 market data.
- In the recent v3 dual-momentum run, AVUV was selected on 57 of 850 daily
  decisions, FTLS on 9, and DBMF on 23; JEPI, DIVO, SPYI, HEQT, HEGD, and QAI
  were not selected. BTAL was selected on 98 dates in v3-beta. These are
  selection counts, not additional allocations beyond the Top-2 holdings.
- Class min/max weights do not constrain the native dual-momentum path; they
  do affect mean-variance. The recent mean-variance run had 10.50% CAGR and
  9.35% drawdown for sector, 10.03% and 9.12% for v3, and 9.23% and 9.09%
  for v3-beta. The risk/return trade-off is inconclusive, and this is not
  evidence that any alternative reliably beats SPY.
- A separate 49-stock experiment with a 15% simulated trailing stop reported
  25.6% CAGR and 39.2% drawdown for Top-2 versus 15.4% and 33.7% for SPY
  (10 bps costs). The stop did not remove the higher drawdown in that sample;
  it is not a hard portfolio loss limit. Universe membership was a static
  current-survivor sample, so the finding is exploratory, not causal.

Current research direction: broaden the ETF baseline carefully (the compact
`examples/sector_alternatives_universe_v4.json` copies v3, adds XLB/XLC/XLY,
USMV, and QUAL, and removes the inherited class minimums). Evaluate it against
the original and v3 on matched post-inception dates. Do not add leverage as a
substitute for risk controls. For stock universes, compare Top-5/10 with
sector/position limits and point-in-time constituents; more names alone do not
guarantee lower drawdown. Also test trailing stops and volatility targeting
separately, with turnover costs, rolling windows, and a held-out period. The
v4 configuration was subsequently backtested against v3 on matched offline
windows (no costs; SPY cache ended 2026-05-15). Dual-momentum Top-2 daily:
long window CAGR 16.44% for v3 vs. 15.50% for v4, with both at 30.90% max
drawdown; recent window 20.86% vs. 22.68% CAGR, but drawdown rose from
24.28% to 25.79%. Thus v4 was not uniformly worse. Mean-variance CAGR was
lower for v4 (long 5.46% vs. 4.76%; recent 10.03% vs. 7.73%), with slightly
lower volatility / similar or marginally lower drawdown. That comparison is
confounded by v4 removing the inherited class minimums, which mean-variance
respects. In the long window XLC was excluded for insufficient history; recent
window dropped WTLS. All these results are exploratory and cost-free. The v2 comparison was also
checked: on the recent window, dual-momentum Top-2 CAGR was 20.86% for v2,
22.19% for alternatives v1, and 23.11% for the sector baseline; mean-variance
CAGR was 9.96%, 9.86%, and 10.50%, respectively. On the long window v2
matched v1 because its newer additions were filtered out. These tests had no
transaction costs.

## Proposed follow-up test routes (2026-10-09)

These are hypotheses to test, not accepted strategies. Prioritize the following
as separate experiments; do not add every signal or ETF to one basket and tune
against the same history.

1. **Establish whether dual momentum needs better breadth/risk controls.** On
   identical symbols and dates, compare Top-2, Top-5, and Top-10 equal-weight
   portfolios; monthly (21-day) and slower (63-day) rebalance; then test
   trailing stops and volatility targets one at a time. Include no-stop and
   no-target baselines, turnover, and 10/30/50 bps cost sensitivities. Stops
   are not hard loss limits; volatility targeting is backward-looking. The
   current native dual-momentum path does not enforce model class min/max
   weights.
2. **Compare universe sizes fairly.** Use the 20-ETF sector baseline, compact
   v4 ETF universe, an approximately 100–200-name liquid stock universe, and
   the 535-name Nasdaq/S&P/sector file as a broad exploratory endpoint. Apply
   the same strategy, dates, warm-up, transaction costs, and rebalancing; use
   point-in-time membership and delisted names where available. The 198-name
   `nasdaq100_sector_universe_b2016filtered.json` is an existing intermediate
   prototype, but its static membership and large historical drawdown limit
   conclusions.
3. **Test a low-drawdown ETF hypothesis separately.** Compare the completed
   sector sleeve against sector + USMV, sector + QUAL, and sector + both; add
   AVUV only as a separate small-value hypothesis. Keep the existing
   alternatives (managed futures, anti-beta, option-income/hedged equity) as
   separate sleeves/ablations, since they can reduce risk, dilute return, or
   simply fail to enter a momentum Top-K basket.
4. **Explore different strategy families only after the baselines.** Candidate
   approaches are sector-neutral quality + momentum in a broader stock set;
   cross-asset time-series trend with volatility-scaled positions across
   equities/bonds/commodities; and downside-aware or robust allocation (e.g.,
   CVaR / uncertainty-aware constraints). First rerun existing factor,
   protective, regime-adaptive, and risk-parity implementations consistently.
5. **Defer leverage.** Daily-reset leveraged funds add path-dependent exposure
   and nominal position limits do not cap their effective beta. Test them only
   in a separate, leverage-aware design after an unleveraged strategy has a
   repeatable, out-of-sample edge.

Each study should state whether its objective is to beat SPY on raw return or
improve risk-adjusted return/drawdown. Report CAGR, volatility, max drawdown,
turnover, cost sensitivity, rolling/regime results, and a held-out period. Do
not select a winner from a single full-sample CAGR or visual plot.

A v5 ETF-universe test file, `examples/sector_alternatives_universe_v5.json`,
was created as a copy of v4 with SPMO and QQQA added. SPMO is classified with
large-cap U.S. equity; QQQA with U.S. growth, so the existing class caps also
cover these holdings in mean-variance. The configuration is not yet
backtested. Official inception dates are SPMO 2015-10-09 and QQQA 2021-05-18.
Relative to the cached 2026-05-15 endpoint, neither can support the 2,929-day
backtest plus 252-day warm-up used in the long comparison; QQQA can support the
850-day recent test plus 63-day warm-up.

## Hierarchical category momentum and within-category pairs (2026-10-25)

Clarified proposal: pairs are formed *inside* each category, while a
higher-level momentum rule sets the proportional capital allocation to each
category. This is a two-layer structure, not a basket of category-vs-category
spreads.

Assessment:

- The structure is coherent, but the layers answer different questions. A
dollar-neutral within-category pair book has zero net category exposure.
Category momentum therefore does not mechanically improve the pair book; it
only scales the capital assigned to it. The overlay must be justified as a
timing/meta-allocation signal, not assumed to follow from pair neutrality.
- A cleaner decomposition is a category beta sleeve plus a pair alpha overlay.
Category momentum sets net category exposure, while within-category pairs run
as a separate gross-neutral sleeve with its own risk budget. Returns can then
be attributed to category beta and pair alpha separately.
- If all category capital is deployed in dollar-neutral pairs, the portfolio
has no net category exposure. The outer momentum layer is then a pair-capital
allocator, which is a different strategy from "momentum on categories."
- Proportional allocation needs a defined score and floor. Use a category
return series (equal-weight or cap-weight), rank categories, map scores to
nonnegative weights, cap each category, and leave a cash residual when
absolute momentum is negative. Do not feed raw negative category returns
directly into weights.
- The outer and inner layers also have different natural horizons. Category
momentum is usually monthly or quarterly; pairs often need weekly or daily
signals. The implementation needs nested rebalance schedules and separate
turnover accounting, otherwise the category layer will be charged for pair
trades or vice versa.

Empirical caution. A quick probe on 10 sector ETFs over 2008-01 to 2026-05
(monthly rebalance, 252-day lookback, long top-k / short bottom-k,
dollar-neutral, gross 1.0, 5 bps turnover cost, 3% annual borrow) found:

| Variant | CAGR | Volatility | Max drawdown |
|---|---:|---:|---:|
| SPY | 11.32% | 19.85% | 51.48% |
| Long-only sector momentum, top 2 | 10.06% | 19.75% | 48.02% |
| Dollar-neutral category momentum, k=2 | -1.92% | 11.43% | 43.84% |
| Dollar-neutral category momentum, k=3 | -2.47% | 9.28% | 45.38% |

The k=2 variant was -0.44% CAGR with no borrow cost and -0.11% with no costs
at all, so the result is not primarily a cost artifact. This tests
category-vs-category spreads, not within-category pairs, but it warns that an
outer momentum layer is not automatically additive. It also shows that a
dollar-neutral pair book can retain substantial residual volatility and
drawdown.

Feasibility:

- Research-only: medium. The repo has daily prices, category metadata, and a
  backtest skeleton, but not pair formation, spread z-scores, half-life
  filters, nested rebalancing, or borrow accounting.
- Live: low. The long-only assumptions documented in the earlier audit still
  apply, and a hierarchy multiplies the number of short legs, locate checks,
  and legging-risk cases.
- Data: the binding constraint. Many sector ETF universes have only one
  instrument per category, so a "within-category pair" needs a broader
  universe. Stock-level pairs require point-in-time membership and borrow
  availability; current-survivor samples are not sufficient.

Next test, pre-registered:

1. Build within-category pair books with equal risk per pair and a cap on the
   number of pairs per category. De-duplicate shared legs.
2. Compare equal-weight category allocation, momentum-proportional allocation,
   and momentum-with-cash-floor on identical pair books and dates.
3. Add a separate category-beta sleeve to distinguish beta timing from pair
   alpha.
4. Charge 5/10/25 bps turnover and a 3% annual borrow assumption; report gross
   and net exposure, long/short leg contribution, and pair-level attribution.
5. Use nested walk-forward windows and the regimes in
   `RESEARCH_PROTOCOL.md`. The null hypothesis is that momentum allocation does
   not beat equal-weight category allocation after costs; if it does not, drop
   the overlay rather than tuning it.

Decision: research-only hypothesis. Do not wire live shorting or a new CLI
strategy until the pair layer and the allocation overlay clear the existing
acceptance gate.

### Pair-as-asset backtest result (2026-10-25)

The pair layer was implemented as `src/portfolio_opt/pair_momentum.py` and run
with `scripts/backtest_pair_momentum.py`. This is research-only; the live
strategy path is unchanged.

Method:

- Universe: `examples/nasdaq100_sp500_sector_stocks_universe.json`, a
  stock-only sector universe derived from the broader sector file. It contains
  516 sector stocks; 466 have complete cached yfinance history from 2015-01-02
  to 2026-05-15. The broader universe that includes sector ETF proxies was
  also run: the signs are unchanged, but the magnitudes differ.
- Categories: the existing `sector_*` asset classes. Pairs are formed only
  inside a sector.
- Pair screen: daily-return correlation at least 0.2, positive OLS beta, then
  lowest residual volatility. Up to 8 pairs per sector, with at most 2 pairs
  per leg.
- Pair return: beta-neutral, `(r_long - beta * r_short) / (1 + beta)`, gross
  exposure 1.0. A 3% annual borrow cost is charged on the short weight.
- Pair formation uses data through `f`; the pair signal at `t` uses pair
  returns only from `f` to `t`, with `t - f >= 63`. This keeps the formation
  window out of the momentum score. An earlier prototype that overlapped the
  two windows produced optimistic pair-momentum results and was discarded.
- Rebalance every 21 trading days; pair formation every 126 days. Pair cost is
  10 bps per unit of pair-allocation turnover, intended to cover both legs.

Results, stock-only universe, 10 bps pair cost and 3% borrow:

| Strategy | CAGR | Volatility | Max drawdown | Sortino |
|---|---:|---:|---:|---:|
| SPY | 15.45% | 17.87% | 33.72% | 1.22 |
| Equal-weight all pairs | 0.18% | 3.61% | 9.80% | 0.07 |
| Pair momentum, top 10 | -1.98% | 6.13% | 28.10% | -0.44 |
| Pair reversal, top 10 | 3.05% | 7.73% | 13.76% | 0.56 |
| Category index momentum, top 3 categories x 3 pairs | -3.31% | 5.21% | 29.32% | -0.86 |
| Category pair-book momentum, top 3 x 3 | -1.74% | 6.00% | 30.42% | -0.39 |
| Category index momentum + pair reversal | -3.01% | 5.76% | 29.61% | -0.71 |
| Category pair-book momentum + pair reversal | -2.20% | 5.14% | 21.88% | -0.59 |

With sector ETFs included, pair reversal is 2.02% and pair momentum -0.97%;
the category overlays remain negative. The pair-momentum direction has no edge
in either variant. The sign-flipped pair-reversal book is the only positive
result: about 3.1% net CAGR with 7.7% volatility and a 13.8% drawdown. It
remains positive at 25 bps pair cost (1.53%), at a 5% borrow assumption
(2.20%), and at 5 bps pair cost (3.57%), but this is a modest market-neutral
result, not a SPY-beating strategy. The category momentum overlay makes results
worse at every tested setting; equal weighting or direct pair ranking is
better.

Interpretation: the outer category layer is not supported by this test. If the
pair idea is pursued, it should be framed as within-sector mean reversion, not
pair momentum. The pair NAV is a strategy return series, not a tradable price,
and a live implementation still needs borrow availability, locate checks,
buy-to-cover, short stops, leg-level risk, and legging controls.

Caveats: current-survivor universe, no delisted names, no point-in-time sector
membership, no borrow availability or recall modeling, no short rebate, no
dividend/financing detail, no market impact, and one sample with multiple
tested variants. Treat the reversal result as exploratory.

### Curated pair universe result (2026-10-25)

The broad sector screen may be too noisy for pair selection. A second test used
`examples/pair_universe_curated.json`, a manually curated set of 34
same-sub-industry pairs across 67 liquid large-cap stocks. The list was fixed
before the run and was not edited after seeing results. Selection was based on
economic similarity, not historical performance. Pair betas are still estimated
point-in-time on the 252-day formation window.

Representative pairs include `KO/PEP`, `XOM/CVX`, `JPM/BAC`, `V/MA`,
`JNJ/PFE`, `NVDA/AMD`, `UPS/FDX`, `DUK/SO`, and `HD/LOW`.

Results, 10 bps pair cost and 3% borrow:

| Strategy | CAGR | Volatility | Max drawdown | Sortino |
|---|---:|---:|---:|---:|
| SPY | 15.45% | 17.87% | 33.72% | 1.22 |
| Equal-weight all curated pairs | 3.22% | 4.25% | 6.82% | 1.10 |
| Pair momentum, top 10 | 3.00% | 6.84% | 12.02% | 0.63 |
| Pair reversal, top 10 | 4.33% | 5.93% | 10.75% | 1.07 |
| Category index momentum, 3x3 | 0.84% | 6.28% | 19.93% | 0.19 |
| Category pair-book momentum, 3x3 | 4.02% | 7.37% | 15.12% | 0.79 |
| Category index momentum + reversal | -0.83% | 5.82% | 19.49% | -0.19 |
| Category pair-book momentum + reversal | 2.59% | 6.70% | 13.25% | 0.55 |

Unlike the broad systematic screen, pair momentum is positive on the curated
list. But equal weighting and short-term pair reversal are competitive:
equal-weighting all 34 pairs produces 3.22% with 4.25% volatility and a 6.82%
drawdown, while pair reversal produces 4.33% with a 1.07 Sortino. The
category-index overlay still does not help; category-pair-book momentum raises
raw return but also volatility and drawdown.

Cost and borrow sensitivity for the direct pair ranking:

| Variant | 5 bps | 10 bps | 25 bps | 0% borrow | 3% borrow | 5% borrow |
|---|---:|---:|---:|---:|---:|---:|
| Pair momentum | 3.43% | 3.00% | 1.75% | 3.72% | 3.00% | 2.24% |
| Pair reversal | 4.79% | 4.33% | 2.93% | 5.67% | 4.33% | 3.70% |

Interpretation: universe curation materially changes the result. The earlier
broad-screen conclusion was universe-dependent, not a general statement about
pair momentum. On this curated set, both momentum and reversal are positive,
but the simpler equal-weight pair book has the best drawdown and a comparable
Sortino. This is not enough to claim a persistent edge.

Caveats: the manual list is economically motivated but still assembled by us;
it is not an externally pre-registered list. The universe is current-survivor,
does not use point-in-time sub-industry membership, and contains no delisted
names. Borrow availability, recalls, rebates, dividends, financing, and market
impact are not modeled. The list was fixed before this run, but a genuinely
out-of-sample test needs an independent pair list or a pre-specified selection
procedure applied to a held-out period. Treat these results as exploratory.

### Pair layer plus native dual momentum (2026-10-25)

This is the first test of the intended two-layer design: fixed curated pairs
produce point-in-time pair NAVs, then the repository's native dual momentum
selector runs on those NAVs.

Implementation: `scripts/backtest_pair_dual_momentum.py`.

Method:

- Pair list: `examples/pair_universe_curated.json`, 34 same-industry pairs.
- Pair NAV: beta-neutral, `(r_long - beta * r_short) / (1 + beta)`, with beta
  re-estimated every 126 days on a 252-day formation window using data through
  the formation date.
- Pair NAV is flat before the first formation and includes a 3% annual borrow
  cost on the short weight.
- Second layer: the existing `run_dual_momentum_backtest` with `BIL` as the
  cash-like asset, an absolute filter versus BIL, relative top-k ranking, and a
  cash fallback.
- Rebalance every 21 days. Pair allocation cost is 10 bps per unit of
  turnover. The sample ends 2026-05-01 because that is the end of the cached
  BIL history.

Results:

| Strategy | CAGR | Volatility | Max drawdown | Sortino |
|---|---:|---:|---:|---:|
| SPY | 15.06% | 17.89% | 33.72% | 1.19 |
| Equal-weight all pairs | 3.43% | 4.32% | 7.92% | 1.15 |
| Pair dual momentum, 126d top 10 | 3.22% | 6.81% | 11.70% | 0.69 |
| Pair dual momentum, 252d top 10 | 4.67% | 7.31% | 11.45% | 0.93 |
| Pair dual momentum, 126d top 5 | 7.98% | 9.97% | 16.06% | 1.21 |
| Pair dual momentum, 252d top 5 | 6.13% | 10.70% | 17.95% | 0.83 |
| Stock dual momentum, 126d top 10 | 22.82% | 22.92% | 36.41% | 1.42 |
| Stock dual momentum, 252d top 10 | 19.65% | 24.03% | 33.28% | 1.15 |

Cost and borrow sensitivity for pair dual momentum:

| Variant | 10 bps / 3% borrow | 25 bps / 3% borrow | 10 bps / 5% borrow |
|---|---:|---:|---:|
| 126d top 10 | 3.22% | 2.18% | 2.24% |
| 252d top 10 | 4.67% | 3.93% | 4.13% |

Interpretation:

- This is actual dual momentum, not the earlier simple top-k pair ranking. The
  absolute filter and cash fallback are active.
- The pair layer is positive, especially with top-5 selection: 7.98% CAGR at
  9.97% volatility and a 1.21 Sortino for the 126-day lookback.
- It still does not dominate stock dual momentum on the same names. The stock
  version earns much more absolute return, with higher volatility and
  drawdown. The stock baseline also benefits from the 2015–2026 equity bull
  market and the current-survivor universe.
- Equal-weighting all pairs has a better Sortino than the top-10 pair dual
  momentum variants, though lower raw return. Pair dual momentum top-5
  improves raw return but raises volatility and drawdown.
- The pair layer is best interpreted as a lower-volatility diversifier, not a
  replacement for stock momentum.

Caveats: the curated pair list is ours, not an external pre-registered list;
the universe is current-survivor; hedge-ratio rebalance costs, borrow
availability, locate failures, short rebates, dividends, financing, and market
impact are not modeled; BIL history ends earlier than the other caches; and
multiple variants were inspected on one sample.

### Top-2 and alternative pair universes (2026-10-25)

Top-2 pair dual momentum on the curated 34-pair universe changes the picture:

| Variant | CAGR | Volatility | Max drawdown | Sortino |
|---|---:|---:|---:|---:|
| SPY | 15.06% | 17.89% | 33.72% | 1.19 |
| Pair DM, 126d top 2 | 15.31% | 16.61% | 16.70% | 1.43 |
| Pair DM, 252d top 2 | 15.39% | 17.38% | 24.40% | 1.41 |
| Stock DM, 126d top 2 | 29.09% | 40.12% | 53.80% | 1.11 |
| Stock DM, 252d top 2 | 53.93% | 39.89% | 39.08% | 2.06 |

The pair top-2 variants roughly match SPY's CAGR with lower drawdown and a
higher Sortino. Cost sensitivity is still acceptable: at 25 bps pair cost the
126d top-2 variant earns 13.70%, and at a 5% borrow assumption it earns 15.02%.
The 252d top-2 variant earns 14.38% at 25 bps and 14.26% at 5% borrow.

This is a concentrated result: only two of 34 pairs are held, so it can be
driven by a small number of spreads and should not be treated as robust without
pair-level attribution and a held-out test.

Two systematic pair universes were derived from the b2016-filtered files. Pairs
were selected on the first 252 trading days (2015-2016) and then fixed:

- `examples/nasdaq100_sp500_sector_b2016filtered_pairs.json`: 88 pairs from
  113 symbols with complete history.
- `examples/nasdaq100_sector_b2016filtered_pairs.json`: 55 pairs from 64
  symbols with complete history.

Results:

| Universe / strategy | CAGR | Volatility | Max drawdown | Sortino |
|---|---:|---:|---:|---:|
| SP500 b2016 pairs, equal weight | 0.71% | 4.04% | 10.74% | 0.25 |
| SP500 b2016 pairs, DM 126d top 2 | -3.79% | 13.22% | 51.57% | -0.38 |
| SP500 b2016 pairs, DM 252d top 2 | -1.73% | 13.86% | 45.87% | -0.17 |
| SP500 b2016 stocks, DM 126d top 2 | 8.54% | 31.30% | 49.74% | 0.39 |
| SP500 b2016 stocks, DM 252d top 2 | 5.48% | 31.62% | 55.84% | 0.24 |
| NDX b2016 pairs, equal weight | 2.85% | 5.44% | 9.83% | 0.74 |
| NDX b2016 pairs, DM 126d top 2 | -2.02% | 15.36% | 55.54% | -0.18 |
| NDX b2016 pairs, DM 252d top 2 | 0.40% | 15.86% | 49.67% | 0.03 |
| NDX b2016 stocks, DM 126d top 2 | 16.09% | 33.28% | 58.01% | 0.69 |
| NDX b2016 stocks, DM 252d top 2 | 18.98% | 35.19% | 55.30% | 0.77 |

The naked-stock k=2 outperformance does not transfer to these systematic pair
lists. The curated same-industry list is much better, which again points to pair
selection quality as the dominant issue. This is not a clean replication of the
earlier stock result: the pair list is fixed from 2015-2016, the complete-history
filter shrinks the universe, and the sample ends at the BIL cache date.

A pair is two legs by definition. More than two legs is a basket spread or
relative-value basket, not a pair: long one asset against a basket of peers, or
long one basket against another. That can reduce single-leg noise and improve
robustness, but it adds hedge ratios, borrow, costs, and execution complexity.
The current implementation only supports two-leg pairs.

### Pair research evaluation windows and summary (2026-10-25)

Effective windows used in the pair experiments:

| Experiment | Data window | Effective evaluation window | Warmup / selection |
|---|---|---|---|
| Simple pair momentum/reversal, broad and curated | 2015-01-02 to 2026-05-15 | 2016-04-05 to 2026-05-15 | 252-day formation plus 63 days of post-formation pair history |
| Pair dual momentum, curated 34 pairs | 2015-01-02 to 2026-05-01 | 2016-01-04 to 2026-05-01 | 252-day formation; BIL cash proxy |
| b2016 systematic pair universes | 2015-01-02 to 2026-05-01 | 2016-01-04 to 2026-05-01 | pair list selected on the first 252 sessions of 2015 and then fixed |

Key findings:

- The broad systematic sector screen did not produce a robust pair-momentum
  edge. Pair momentum was negative; pair reversal was positive but modest.
- The curated same-industry pairs were materially better. Pair dual momentum
  with top 10 earned about 3.2-4.7% net CAGR with 6.8-7.3% volatility. With
  top 2 it earned 15.3-15.4% CAGR with 16.7-24.4% maximum drawdown and a
  1.41-1.43 Sortino, close to SPY's return with lower drawdown. The top-2 result
  survived 25 bps pair cost and a 5% borrow assumption, but it holds only two
  pairs and is therefore concentrated.
- The b2016 systematic pair universes did not reproduce the naked-stock
  k=2 outperformance. Pair dual momentum top 2 was negative or near flat on
  both b2016 pair files. The pair list was fixed from 2015, the complete-history
  filter reduced the universes, and the sample ended at the BIL cache date, so
  this is not a clean replication of the earlier stock result.

  The full k=2 comparison table is in
  `### Top-2 and alternative pair universes` above.

- A pair is two legs by definition. More than two legs is a basket spread or
  relative-value basket, not a pair. The current implementation supports only
  two-leg pairs.

### Precedent and literature (2026-10-25)

The pair research is a composition of established ideas, not a canonical named
strategy. The exact construction — pair NAVs as synthetic assets, dual momentum
on those NAVs, and an optional category overlay — is our composition. Each
component has prior literature.

Pairs trading and relative value:

- Gatev, Goetzmann, and Rouwenhorst (2006), *Pairs Trading: Performance of a
  Relative-Value Arbitrage Rule*, Review of Financial Studies 19(3), 797-827.
  The standard distance-based pairs-trading reference.
- Engle and Granger (1987), *Co-integration and Error Correction:
  Representation, Estimation, and Testing*, Econometrica 55(2), 251-276.
  The basis for cointegration tests of spread stationarity.
- Avellaneda and Lee (2010), *Statistical Arbitrage in the US Equities Market*,
  Quantitative Finance 10(7), 761-782. Residual/relative-value stat arb with
  factor neutralization.
- Do and Faff (2010), *Does Simple Pairs Trading Still Work?*, Financial
  Analysts Journal 66(4), 83-95; and Do and Faff (2012), *Are Pairs Trading
  Profits Robust to Trading Costs?*, Journal of Financial Research 35(2),
  261-287. Evidence that pairs profitability is cost- and period-sensitive.

Momentum and dual momentum:

- Jegadeesh and Titman (1993), *Returns to Buying Winners and Selling Losers*,
  Journal of Finance 48(1), 65-91. Cross-sectional momentum.
- Moskowitz, Ooi, and Pedersen (2012), *Time Series Momentum*, Journal of
  Financial Economics 104(2), 228-250. Absolute/time-series momentum.
- Antonacci (2014), *Dual Momentum Investing*. The relative-plus-absolute
  momentum framework used by the native strategy.

Synthetic assets, strategy allocation, and hierarchy:

- Treating a spread as a synthetic return series is standard relative-value and
  statistical-arbitrage practice; the strategy layer does not need to know the
  legs.
- Allocating across a set of strategy return streams is commonly described as
  fund-of-strategies, portable alpha, or strategy allocation. Applying momentum
  to those returns is a form of strategy/meta-momentum. These are established
  industry concepts, but they are not a single canonical academic model.
- Lopez de Prado (2016), *Building Diversified Portfolios that Outperform Out
  of Sample*, Journal of Portfolio Management 42(4), 59-69, is related through
  hierarchical allocation, but it is not the same design as this pair layer.

Multiple testing and backtest overfitting:

- Bailey, Borwein, Lopez de Prado, and Zhu (2015), *The Probability of Backtest
  Overfitting*, SSRN 2326253.
- White (2000), *A Reality Check for Data Snooping*, Econometrica 68(5),
  1097-1126.
- Harvey, Liu, and Zhu (2016), *... and the Cross-Section of Expected Returns*,
  Review of Financial Studies 29(1), 5-68.

Implication: the components are known, but the combination is not a proven
anomaly. The pair-selection screen, the second-layer momentum rule, and the
category overlay each add degrees of freedom. Results should be treated as
exploratory until they survive out-of-sample tests, realistic borrow and
two-leg costs, and multiple-testing controls. The broader repository literature
review remains in `docs/LITERATURE_REVIEW.md`.

### Instrument-layer implementation, phase 1 (2026-10-25)

The first lower-level implementation is in the package:

- `src/portfolio_opt/instruments.py`: signed legs, spread specs, instruments,
  and an instrument universe.
- `src/portfolio_opt/pair_layer.py`: point-in-time beta estimation and spread
  NAV construction.
- `src/portfolio_opt/universe_builder.py`: builds naked stock and spread
  instruments from a model plus aligned prices.
- `src/portfolio_opt/strategy_data.py` and `strategy_registry.py`: normalize
  the universe into strategy inputs and dispatch the existing engines.

The model schema now accepts optional `pairs` with signed `legs`; the legacy
`long`/`short` form is still accepted for the existing research files. Pair-only
example universes now use `"symbols": []` and list pair legs only in `pairs`.
Overlap between a naked symbol and a pair leg is allowed explicitly.

This is research plumbing only. The CLI and live order path are not wired to the
pair universe yet. Existing models without `pairs` keep the old behavior.

### Modular strategy data and registry (2026-10-25)

The strategy layer is split into generic data and a registry:

- `src/portfolio_opt/strategy_data.py` builds a strategy-agnostic
  `StrategyData` panel: instrument IDs, NAVs, returns, closes, asset classes,
  categories, kinds, and cash.
- `src/portfolio_opt/strategy_registry.py` contains thin wrappers for the
  existing strategies and `run_strategy(name, data, **kwargs)`.

The wrappers cover dual momentum, mean variance, factor momentum, protective
momentum, regime-adaptive, and fixed allocation. They contain no pair-specific
logic; the pair layer is already embedded in the instrument NAVs. Tests are in
`tests/test_strategy_data.py` and `tests/test_strategy_registry.py`. Current
suite: 178 passed.
