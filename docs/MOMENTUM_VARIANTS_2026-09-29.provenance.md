# Provenance for momentum variant comparison

Reviewed 2026-09-29. The companion [report](MOMENTUM_VARIANTS_2026-09-29.md)
separates published findings from this repository's own experiment.

## Primary literature

| Claim area | Primary source | Scope used in report |
|---|---|---|
| Momentum lookback | Moskowitz, Ooi, and Pedersen, *Journal of Financial Economics* 104 (2012), [doi:10.1016/j.jfineco.2011.11.003](https://doi.org/10.1016/j.jfineco.2011.11.003) | Evidence on time-series momentum in liquid futures, not our individual-symbol portfolio. |
| Absolute plus relative momentum | Antonacci, *Risk Premia Harvesting Through Dual Momentum* (2017), [doi:10.2139/SSRN.2042750](https://doi.org/10.2139/SSRN.2042750) | Research motivation; its universe and allocation rule differ from ours. |
| Crash regimes | Daniel and Moskowitz, *Momentum Crashes*, [NBER w20439](https://www.nber.org/papers/w20439) and *Journal of Financial Economics* 122 (2016) | Motivates risk and regime diagnostics; not a forecast for our long-only strategy. |
| Costs | Korajczyk and Sadka, *Journal of Finance* 59 (2004), [doi:10.1111/j.1540-6261.2004.00656.x](https://doi.org/10.1111/j.1540-6261.2004.00656.x) | Motivates cost sensitivity; does not validate the chosen 10/30 bps rates. |
| Equal-weight baseline | DeMiguel, Garlappi, and Uppal, *Review of Financial Studies* 22 (2009), [doi:10.1093/rfs/hhm075](https://doi.org/10.1093/rfs/hhm075) | Motivates simple allocation controls; did not test this momentum implementation. |
| Multiple testing | Bailey et al., *The Probability of Backtest Overfitting* (2015), [SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) | Motivates treating the 13-rule ranking as exploratory. |

## Local experimental inputs

- Tracked input manifest: `docs/data/momentum_variant_input_2026-09-29.json`,
  containing the 49 selected symbols, source universe, price window, and the
  earlier raw report's hash. Its SHA-256 is
  `e7b8ae509ad23894a1687b86ed27398d2e414148ea8f17ea7825af6f5267c744`.
- Tracked reproducer: `scripts/benchmark_momentum_variants.py`, SHA-256
  `72ce3202f2e3d7583bc4d7e8e8935c028820470128fc392e21848bd923d7c3f6`.
- Tracked compact results: `docs/data/momentum_variant_benchmark_2026-09-29.json`.
  It contains configurations, metrics, intervals, and each input price-cache
  filename and SHA-256 digest. Its SHA-256 is
  `3cea0115b6564c60a8c58a8ae29fd157bcd987e8cde1ddf870c223d7df8cf5f6`.
  The ignored full-curve file is `.cache/momentum_variant_benchmark.json`,
  SHA-256 `bc2f72a12241cfb0fe4bad4d1b163264d599d4921387b0e96df253ca70daef72`.
- Symbol selection record: `.cache/sec_quality_benchmark_raw_2016_2026.json`,
  SHA-256 `c558a5a7ccac981100e50e525eeff44e13403abc5d20367229b22612e723eb34`.
- Price caches and full equity curves are ignored local artifacts. The report
  and compact JSON preserve the main evidence in Git; reproducing exact curves
  requires the recorded cache files.
