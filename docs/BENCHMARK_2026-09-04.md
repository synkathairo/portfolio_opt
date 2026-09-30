# Aligned research benchmark through 2026-09-04

This is an exploratory comparison of the repository's SEC-quality, residual
reversal, dual-momentum, regime-adaptive, quality-plus-momentum, and
value-quality-trend research strategies against SPY and equal weight. It is not
evidence of an investable edge.

## Setup

- Source period: 2016-01-04 through 2026-09-04, using Yahoo adjusted closing
  prices for returns and unadjusted closes for the value denominator.
- Candidate selection: the runner's deterministic 50-symbol sample from
  `examples/nasdaq100_sp500_sector_universe_b2016filtered.json`. One ETF, QQQ,
  lacked an SEC companyfacts response, leaving 49 common symbols.
- Common evaluation period: 2017-01-03 through 2026-09-04, after 252 trading
  days of warmup. Each displayed strategy metric uses the same 2,430 returns.
- Rebalance interval: 21 trading days except value-quality-trend at 63. Dual
  momentum holds up to two assets. Residual reversal has a 50% per-rebalance
  turnover cap. All runs used the same symbols, dates, and parameters.
- Linear trading costs: 10, 20, and 30 basis points per unit of turnover.
  Taxes, market impact beyond this linear cost, and live execution were not
  modeled.

The 49 symbols were: FISV, DBC, GLD, SLV, ALB, CHTR, AMCR, ADM, APA, ACGL, A,
ALLE, AMT, AAPL, AEE, APD, CMCSA, AMZN, BG, BKR, AFL, ABBV, AME, ARE, ACN,
AEP, CF, DIS, APTV, CAG, COP, AIG, ABT, AOS, BXP, ADBE, AES, CRH, GOOG, AVY,
CCEP, CVX, AIZ, ALGN, AXON, CBRE, ADI, ATO, and DD.

## Full-period comparison at 10 basis points

| Strategy | CAGR | CAGR minus SPY | Volatility | Max drawdown | Sortino |
|---|---:|---:|---:|---:|---:|
| SEC quality | 14.7% | -0.6 pp | 17.4% | 36.4% | 1.18 |
| Residual reversal | 14.2% | -1.2 pp | 17.8% | 36.5% | 1.12 |
| Dual momentum | 22.5% | +7.2 pp | 30.9% | 40.1% | 1.02 |
| Regime adaptive | 9.0% | -6.4 pp | 17.4% | 32.1% | 0.71 |
| Quality + momentum | 14.7% | -0.7 pp | 18.3% | 33.5% | 1.12 |
| Value + quality + trend | 12.1% | -3.2 pp | 18.1% | 37.5% | 0.94 |
| SPY | 15.4% | 0.0 pp | 18.2% | 33.7% | 1.19 |
| Equal weight | 13.9% | -1.5 pp | 17.9% | 37.8% | 1.09 |

Dual momentum's higher CAGR came with higher volatility and drawdown than SPY.
The other five research strategies did not exceed SPY's full-period CAGR at
10 basis points.

## Trading-cost sensitivity

The table reports net CAGR. All three reports had identical symbols and dates.

| Strategy | 10 bps | 20 bps | 30 bps |
|---|---:|---:|---:|
| SEC quality | 14.75% | 14.58% | 14.41% |
| Residual reversal | 14.17% | 13.49% | 12.80% |
| Dual momentum | 22.54% | 21.30% | 20.07% |
| Regime adaptive | 8.98% | 7.59% | 5.97% |
| Quality + momentum | 14.72% | 13.94% | 13.17% |
| Value + quality + trend | 12.13% | 11.88% | 11.62% |
| SPY | 15.37% | 15.37% | 15.37% |
| Equal weight | 13.86% | 13.79% | 13.71% |

## Periods and uncertainty

Dual momentum exceeded SPY's CAGR by 12.9 percentage points in 2017–2019 and
15.6 in 2020–2021. It trailed by 1.3 points in 2022–2023 and led by 1.3
points in 2024–2026. SEC quality led by 10.2 points in 2022–2023 but trailed
by 10.6 points in 2024–2026. The full per-period matrix is in the generated
aligned Markdown report.

Dual momentum beat SPY in 77.6% of overlapping 756-trading-day windows.
Its annualized paired log-return excess was 6.0%, with a 95% interval of
-10.8% to +22.9% from 2,000 circular 21-day block resamples. This interval
crosses zero. The intervals measure sampling variation conditional on the
tested strategy; they do not correct for choosing among strategies after
inspecting their performance.

The current-symbol universe and current SEC ticker map omit delisted and
historical membership changes. Every displayed period was available during
strategy development, so none is an untouched holdout. A point-in-time
constituent dataset and a genuinely future evaluation period are required
before treating any apparent excess return as validated.

## Reproduce

Set `SEC_USER_AGENT` to an identifying application name and contact address.
With network access, run:

```bash
uv run python scripts/run_sec_quality_research.py --max-symbols 50 --start 2016-01-01 --end 2026-09-05 --max-turnover 0.5 --linear-trade-cost 0.001 --output .cache/sec_quality_benchmark_raw_2016_2026.json --plot .cache/sec_quality_benchmark_raw_2016_2026.png
uv run python scripts/compare_research_benchmark.py .cache/sec_quality_benchmark_raw_2016_2026.json --output .cache/sec_quality_benchmark_aligned_2016_2026.json --markdown .cache/sec_quality_benchmark_aligned_2016_2026.md --plot .cache/sec_quality_benchmark_aligned_2016_2026.png
```

Repeat the first command with `--linear-trade-cost 0.002` and `0.003`, using
distinct output names, then compare each raw report. The raw and aligned
reports and plots from this run are under `.cache/` and are deliberately not
versioned.
