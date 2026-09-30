# Nasdaq-100-style universe sensitivity (2026-09-29)

This experiment reruns the leading [momentum variants](MOMENTUM_VARIANTS_2026-09-29.md)
on two repository universes. It tests whether the earlier 49-symbol finding
persists when the eligible symbols change. Neither file supplies historical
membership dates, so these are static-universe sensitivity tests, not
point-in-time Nasdaq-100 simulations.

## Inputs and method

- `examples/nasdaq100_universe.json`: 114 symbols, including individual stocks,
  SPY, QQQ, and defensive ETFs. All 114 had local adjusted-close prices after
  fetching the three missing ETF histories (EWJ, BWX, EMB). Their common price
  window was 2023-09-14 to 2026-05-01. After 252 trading days of common warmup,
  only **407 return days** remained, from 2024-09-16 to 2026-05-01.
- `examples/nasdaq100_sector_universe_b2016filtered.json`: 198 symbols. The
  common price window after a 2016-01-01 cutoff was 2016-01-04 to 2026-04-17;
  the evaluation used **2,334 return days** from 2017-01-03 to 2026-04-17.
  This file is larger than the actual Nasdaq-100 and is a longer-window
  Nasdaq-like stock and ETF set.
- Both runs use the existing native backtest implementation, the model's
  `asset_classes`, a 252-day common warmup, 15% simulated daily-close trailing
  stops for the unqualified dual-momentum rows, and 10 or 30 basis points of
  linear trading cost per unit of turnover. Momentum rules rebalance every
  21 trading days except the named 63-day variant. QQQ and SPY are buy-and-hold
  comparisons; “equal universe” rebalances every 21 trading days.
- The [reproducer](../scripts/benchmark_nasdaq100_sensitivity.py) and [compact
  results](data/nasdaq100_sensitivity_2026-09-29.json) are outside `.cache/`.
  The result JSON records each source model and local price-cache SHA-256.
  Exact reruns require those ignored price caches. Full daily curves remain
  locally at `.cache/nasdaq100_sensitivity_full.json`.

## Full available periods

The figures below are net of 10-basis-point turnover costs. CAGR at 30 basis
points is shown to expose cost sensitivity. The periods differ between
universes, so compare candidates **within** each block.

| Current 114-symbol file, 2024-09 to 2026-05 | CAGR 10 bps | CAGR 30 bps | Volatility | Max drawdown |
|---|---:|---:|---:|---:|
| Dual momentum, top 2 equal | 239.8% | 230.3% | 57.4% | 33.3% |
| Dual momentum, top 5 score | 153.5% | 147.2% | 42.0% | 27.6% |
| Dual momentum, top 5 equal | 115.4% | 110.4% | 37.1% | 24.4% |
| Top 5 score, 63-day rebalance | 88.8% | 86.0% | 33.6% | 20.5% |
| QQQ buy-and-hold | 25.1% | 25.0% | 21.4% | 22.8% |
| SPY buy-and-hold | 18.0% | 17.8% | 17.3% | 18.8% |

| Pre-2016-filtered 198-symbol file, 2017-01 to 2026-04 | CAGR 10 bps | CAGR 30 bps | Volatility | Max drawdown |
|---|---:|---:|---:|---:|
| Dual momentum, top 2 equal | 33.9% | 29.8% | 51.4% | 63.2% |
| Dual momentum, top 5 score | 37.2% | 33.3% | 40.7% | 63.4% |
| Dual momentum, top 5 equal | 35.6% | 31.9% | 35.9% | 59.1% |
| Top 5 score, 63-day rebalance | 20.7% | 19.0% | 27.4% | 44.9% |
| QQQ buy-and-hold | 20.9% | 20.8% | 22.8% | 35.1% |
| SPY buy-and-hold | 15.0% | 15.0% | 18.4% | 33.7% |

The current 114-symbol sample heavily favors top two over its short window.
At scheduled rebalances, the top-two selections included APP on 13 dates,
PLTR on 10, MSTR on five, and WDC on six. Those counts explain the
concentration of the result; they are not an attribution of exact P&L. The
239.8% figure annualizes only 407 daily returns and should not be projected
forward. The top-two rule had 57.4% annualized volatility even during that
strong period.

In the longer 198-symbol set, top-five score had the highest full-period CAGR
among these candidates, but its 63.4% maximum drawdown was severe. Its paired
annual log-return advantage over top two was 2.5%, with a 95% circular
21-day-block interval of -10.3% to +15.7%. This does not establish a return
advantage. Top-five equal lowered volatility and drawdown modestly, at similar
growth.

## Same-date comparison

The two universes share 397 return days from 2024-09-16 to 2026-04-17. QQQ
returns are nearly identical across panels; the small CAGR difference comes
from charging its initial trade at different backtest start dates.

| Universe and candidate | CAGR, 10 bps | Max drawdown |
|---|---:|---:|
| Current 114: top 2 equal | 208.4% | 33.3% |
| Current 114: top 5 score | 130.4% | 27.6% |
| Current 114: QQQ | 22.9% | 22.8% |
| Pre-2016-filtered 198: top 2 equal | 64.8% | 43.7% |
| Pre-2016-filtered 198: top 5 score | 68.5% | 38.0% |
| Pre-2016-filtered 198: QQQ | 23.0% | 22.8% |

The large strategy difference on the same dates is evidence that **universe
choice dominates this particular comparison**. It is not evidence that the
current 114-symbol snapshot would have been available throughout the period.
Nasdaq [reconstitutes the actual index annually and rebalances it quarterly](https://www.nasdaq.com/articles/global-indexes/2025-nasdaq-100-reconstitution-and-performance-highlights),
while both repository files are fixed lists. Inferring past tradable membership
from either list would introduce look-ahead bias. The apparent winners also
come from an inspected strategy search, and live trailing-stop fills, taxes,
spreads, and market impact are absent.

## Decision

This sensitivity test reverses the simple conclusion from the 49-symbol run:
top two leads on the recent current-file slice, while top five score leads in
the longer, broader sample. **It does not justify changing live orders or
declaring top two best.** The next meaningful test requires dated index
membership or the exact dynamic-universe construction replayed at each date,
followed by an untouched period and paper-order verification.
