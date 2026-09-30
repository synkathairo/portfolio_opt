"""Compare research backtest curves over identical calendar windows."""

from __future__ import annotations

import argparse
import json
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from portfolio_opt.backtest import summarize_return_series

STRATEGIES = {
    "strategy": "SEC quality",
    "residual_reversion": "Residual reversal",
    "dual_momentum": "Dual momentum",
    "regime_adaptive": "Regime adaptive",
    "quality_momentum": "Quality + momentum",
    "value_quality_trend": "Value + quality + trend",
    "spy": "SPY",
    "equal_weight": "Equal weight",
}
PERIODS = {
    "2017–2019": (date(2017, 1, 1), date(2019, 12, 31)),
    "2020–2021": (date(2020, 1, 1), date(2021, 12, 31)),
    "2022–2023": (date(2022, 1, 1), date(2023, 12, 31)),
    "2024–present": (date(2024, 1, 1), date.max),
}


def _minimum_warmup(report: dict[str, Any]) -> int:
    return max(
        63,
        int(report["dual_momentum_config"]["lookback_days"]),
        int(report["regime_adaptive_config"]["lookback_days"]),
        int(report["quality_momentum_config"]["lookback_days"]),
        int(report["value_quality_trend_config"]["trend_window"]),
    )


def _metrics(values: np.ndarray) -> dict[str, float]:
    returns = values[1:] / values[:-1] - 1.0
    summary = summarize_return_series(returns)
    return {
        "total_return": summary.total_return,
        "cagr": summary.annualized_return,
        "volatility": summary.annualized_volatility,
        "max_drawdown": summary.max_drawdown,
        "sortino": summary.sortino_ratio,
    }


def _diagnostics(curves: dict[str, np.ndarray], start: int) -> dict[str, Any]:
    spy = curves["spy"]
    spy_log_returns = np.log(spy[start + 1 :] / spy[start:-1])
    count = len(spy_log_returns)
    block_days = min(21, count)
    draws = 2000
    rng = np.random.default_rng(20260929)
    block_starts = rng.integers(
        0, count, size=(draws, (count + block_days - 1) // block_days)
    )
    indices = (block_starts[:, :, None] + np.arange(block_days)) % count
    indices = indices.reshape(draws, -1)[:, :count]
    rolling_days = 756
    strategies: dict[str, dict[str, Any]] = {}
    for key, values in curves.items():
        excess = np.log(values[start + 1 :] / values[start:-1]) - spy_log_returns
        simulated = 252.0 * excess[indices].mean(axis=1)
        if count >= rolling_days:
            relative = (
                values[start + rolling_days :] / values[start:-rolling_days]
                > spy[start + rolling_days :] / spy[start:-rolling_days]
            )
            rolling_win_fraction = float(relative.mean())
        else:
            rolling_win_fraction = None
        strategies[key] = {
            "annual_log_excess": float(252.0 * excess.mean()),
            "block_bootstrap_95_ci": [
                float(value) for value in np.percentile(simulated, [2.5, 97.5])
            ],
            "rolling_3y_win_fraction": rolling_win_fraction,
        }
    return {
        "block_days": block_days,
        "bootstrap_draws": draws,
        "rolling_window_days": rolling_days,
        "strategies": strategies,
    }


def compare_report(report: dict[str, Any]) -> dict[str, Any]:
    """Rebase each curve at the common warmup date and compare matching days."""
    dates = [date.fromisoformat(value) for value in report["dates"]]
    if any(left >= right for left, right in pairwise(dates)):
        raise ValueError("Report trading dates must increase strictly.")
    warmup = _minimum_warmup(report)
    if len(dates) < warmup + 2:
        raise ValueError("The report is too short for the common warmup.")
    curves: dict[str, np.ndarray] = {}
    for key in STRATEGIES:
        values = np.asarray(report[f"{key}_values"], dtype=float)
        if (
            len(values) != len(dates)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(f"Invalid or unaligned value series: {key}.")
        curves[key] = values

    windows = {"full": (warmup, len(dates) - 1)}
    for name, (start, end) in PERIODS.items():
        matching = [
            i for i, day in enumerate(dates) if i >= warmup and start <= day <= end
        ]
        if len(matching) >= 2:
            windows[name] = (matching[0], matching[-1])

    results = {
        name: {
            "start": dates[first].isoformat(),
            "end": dates[last].isoformat(),
            "trading_days": last - first,
            "strategies": {
                key: _metrics(values[first : last + 1])
                for key, values in curves.items()
            },
        }
        for name, (first, last) in windows.items()
    }
    return {
        "kind": "aligned_research_benchmark",
        "source": report.get("kind"),
        "symbols": report["symbols"],
        "survivorship_warning": report["survivorship_warning"],
        "linear_trade_cost": report["linear_trade_cost"],
        "max_turnover": report["max_turnover"],
        "warmup_days": warmup,
        "windows": results,
        "diagnostics": _diagnostics(curves, warmup),
    }


def markdown_summary(comparison: dict[str, Any]) -> str:
    full = comparison["windows"]["full"]
    spy_cagr = full["strategies"]["spy"]["cagr"]
    lines = [
        "# Aligned research benchmark",
        "",
        (
            f"Sample: {len(comparison['symbols'])} current-survivor symbols; "
            f"evaluation {full['start']} to {full['end']} "
            f"({full['trading_days']} return days)."
        ),
        (
            f"Linear trading cost: {comparison['linear_trade_cost']:.1%} of turnover. "
            f"Residual turnover cap: {comparison['max_turnover']}."
        ),
        (
            f"Warmup: {comparison['warmup_days']} trading days. "
            "All metrics below use the same evaluation dates."
        ),
        "",
        "| Strategy | CAGR | CAGR minus SPY | Volatility | Max drawdown | Sortino |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in STRATEGIES.items():
        item = full["strategies"][key]
        lines.append(
            f"| {label} | {item['cagr']:.1%} | {item['cagr'] - spy_cagr:+.1%} | "
            f"{item['volatility']:.1%} | {item['max_drawdown']:.1%} | "
            f"{item['sortino']:.2f} |"
        )
    for name, window in comparison["windows"].items():
        if name == "full":
            continue
        spy = window["strategies"]["spy"]["cagr"]
        lines.extend(
            [
                "",
                f"## {name} ({window['start']} to {window['end']})",
                "",
                "| Strategy | CAGR | CAGR minus SPY | Max drawdown |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, label in STRATEGIES.items():
            item = window["strategies"][key]
            lines.append(
                f"| {label} | {item['cagr']:.1%} | {item['cagr'] - spy:+.1%} | "
                f"{item['max_drawdown']:.1%} |"
            )
    lines.extend(
        [
            "",
            "## Paired performance diagnostics",
            "",
            "| Strategy | 3-year windows beating SPY | Annual log excess, 95% block interval |",
            "|---|---:|---:|",
        ]
    )
    for key, label in STRATEGIES.items():
        item = comparison["diagnostics"]["strategies"][key]
        fraction = item["rolling_3y_win_fraction"]
        fraction_text = f"{fraction:.1%}" if fraction is not None else "n/a"
        lower, upper = item["block_bootstrap_95_ci"]
        lines.append(
            f"| {label} | {fraction_text} | "
            f"{item['annual_log_excess']:.1%} [{lower:.1%}, {upper:.1%}] |"
        )
    lines.extend(
        [
            "",
            (
                "Intervals use 2,000 paired circular block resamples of 21 trading days. "
                "Overlapping three-year windows are descriptive, not independent tests. "
                "Intervals do not correct for selecting strategies after inspecting history."
            ),
            "",
            (
                "This is exploratory: the universe uses current survivors and current SEC "
                "ticker mappings. These periods have been inspected during strategy "
                "development, so none is an untouched holdout. The table does not "
                "establish an investable edge."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def plot_aligned_curves(
    report: dict[str, Any], comparison: dict[str, Any], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    dates = [date.fromisoformat(value) for value in report["dates"]]
    start = comparison["warmup_days"]
    figure, axis = plt.subplots(figsize=(11, 6))
    for key, label in STRATEGIES.items():
        values = np.asarray(report[f"{key}_values"], dtype=float)
        axis.plot(
            dates[start:],
            values[start:] / values[start],
            label=label,
            linestyle="--" if key == "spy" else "-",
        )
    axis.set_title("Aligned research benchmark (exploratory)")
    axis.set_ylabel("Growth of $1 from common start")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small", ncol=2)
    figure.autofmt_xdate()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(".cache/aligned_benchmark.json")
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path(".cache/aligned_benchmark.md")
    )
    parser.add_argument(
        "--plot", type=Path, default=Path(".cache/aligned_benchmark.png")
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    comparison = compare_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    args.markdown.write_text(markdown_summary(comparison))
    plot_aligned_curves(report, comparison, args.plot)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "summary": str(args.markdown),
                "plot": str(args.plot),
            }
        )
    )


if __name__ == "__main__":
    main()
