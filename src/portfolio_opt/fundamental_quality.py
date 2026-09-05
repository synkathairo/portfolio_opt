"""Point-in-time quality signals derived from SEC companyfacts data.

This module intentionally contains no market-price lookups or trading code.  A
caller supplies an ``as_of`` date, and every observation is filtered by its SEC
filing date before ratios are calculated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import Any

from .sec_edgar import FactObservation, iter_facts, latest_facts_as_of

_TAG_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
    ),
    "net_income": ("NetIncomeLoss",),
    "assets": ("Assets",),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "liabilities": ("Liabilities",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
}


@dataclass(frozen=True)
class FundamentalSnapshot:
    """Annual fundamentals available no later than ``as_of``."""

    symbol: str
    as_of: date
    period_end: date
    filed: date
    revenue: float | None
    prior_revenue: float | None
    net_income: float | None
    assets: float | None
    equity: float | None
    liabilities: float | None
    operating_cash_flow: float | None

    def metrics(self) -> dict[str, float]:
        """Return scale-free quality metrics with invalid ratios omitted."""
        values: dict[str, float] = {}
        if self.assets and self.net_income is not None:
            values["return_on_assets"] = self.net_income / abs(self.assets)
        if self.assets and self.operating_cash_flow is not None:
            values["cash_return_on_assets"] = self.operating_cash_flow / abs(
                self.assets
            )
        if self.revenue and self.net_income is not None:
            values["net_margin"] = self.net_income / abs(self.revenue)
        if self.assets and self.liabilities is not None:
            values["liability_ratio"] = self.liabilities / abs(self.assets)
        if self.prior_revenue and self.revenue is not None:
            values["revenue_growth"] = self.revenue / abs(self.prior_revenue) - 1.0
        return {key: value for key, value in values.items() if isfinite(value)}


def _annual_observation(
    payload: dict[str, Any],
    *,
    tag: str,
    as_of: date,
) -> FactObservation | None:
    observations = iter_facts(payload, tag=tag, forms=("10-K",))
    available = latest_facts_as_of(observations, as_of)
    eligible = [
        observation for observation in available.values() if observation.end <= as_of
    ]
    return max(eligible, key=lambda observation: observation.end) if eligible else None


def _latest_tag_value(
    payload: dict[str, Any],
    *,
    aliases: tuple[str, ...],
    as_of: date,
) -> tuple[float | None, date | None, date | None]:
    candidates = [
        observation
        for tag in aliases
        for observation in [_annual_observation(payload, tag=tag, as_of=as_of)]
        if observation is not None
    ]
    if not candidates:
        return None, None, None
    selected = max(candidates, key=lambda observation: observation.end)
    return selected.value, selected.end, selected.filed


def build_snapshot(
    symbol: str,
    payload: dict[str, Any],
    *,
    as_of: date,
) -> FundamentalSnapshot | None:
    """Build one annual, filing-date-safe snapshot for ``symbol``."""
    values: dict[str, float | None] = {}
    period_ends: list[date] = []
    filed_dates: list[date] = []
    for name, aliases in _TAG_ALIASES.items():
        value, period_end, filed = _latest_tag_value(
            payload, aliases=aliases, as_of=as_of
        )
        values[name] = value
        if period_end is not None:
            period_ends.append(period_end)
        if filed is not None:
            filed_dates.append(filed)
    if not period_ends or not filed_dates:
        return None

    revenue_history = [
        observation
        for tag in _TAG_ALIASES["revenue"]
        for observation in latest_facts_as_of(
            iter_facts(payload, tag=tag, forms=("10-K",)), as_of
        ).values()
    ]
    revenue_history.sort(key=lambda observation: observation.end, reverse=True)
    latest_revenue = values["revenue"]
    prior_revenue = next(
        (
            observation.value
            for observation in revenue_history
            if latest_revenue is not None
            and observation.value != latest_revenue
            and observation.end < max(period_ends)
        ),
        None,
    )
    return FundamentalSnapshot(
        symbol=symbol,
        as_of=as_of,
        period_end=max(period_ends),
        filed=max(filed_dates),
        revenue=values["revenue"],
        prior_revenue=prior_revenue,
        net_income=values["net_income"],
        assets=values["assets"],
        equity=values["equity"],
        liabilities=values["liabilities"],
        operating_cash_flow=values["operating_cash_flow"],
    )


def rank_quality_scores(
    snapshots: list[FundamentalSnapshot],
    *,
    groups: dict[str, str] | None = None,
    minimum_metrics: int = 3,
) -> dict[str, float]:
    """Rank quality metrics cross-sectionally, optionally within sectors.

    The result is a 0--1 score.  Higher return, cash-flow, margin, and growth
    are rewarded; lower liabilities/assets are rewarded.  A symbol must have
    enough valid metrics to be scored, preventing missing SEC tags from being
    silently treated as zero quality.
    """
    if minimum_metrics < 1:
        raise ValueError("minimum_metrics must be positive.")
    eligible = [
        snapshot for snapshot in snapshots if len(snapshot.metrics()) >= minimum_metrics
    ]
    if not eligible:
        return {}
    grouped: dict[str, list[FundamentalSnapshot]] = {}
    for snapshot in eligible:
        grouped.setdefault((groups or {}).get(snapshot.symbol, "__all__"), []).append(
            snapshot
        )

    lower_is_better = {"liability_ratio"}
    metric_scores: dict[str, list[float]] = {
        snapshot.symbol: [] for snapshot in eligible
    }
    for members in grouped.values():
        metric_names = sorted({name for member in members for name in member.metrics()})
        for metric_name in metric_names:
            ranked = sorted(
                (member for member in members if metric_name in member.metrics()),
                key=lambda member: member.metrics()[metric_name],
            )
            count = len(ranked)
            for index, member in enumerate(ranked):
                percentile = (index + 1) / count
                if metric_name in lower_is_better:
                    percentile = 1.0 - percentile + 1.0 / count
                metric_scores[member.symbol].append(percentile)
    return {
        symbol: sum(scores) / len(scores)
        for symbol, scores in metric_scores.items()
        if scores
    }
