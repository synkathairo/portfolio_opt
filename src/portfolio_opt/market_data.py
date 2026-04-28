from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .alpaca_interface import AlpacaClient
from .config import AlpacaConfig
from .csv_data import (
    fetch_closes as csv_fetch_closes,
    write_json_caches as csv_write_json_caches,
    write_yfinance_compatible_caches as csv_write_yfinance_compatible_caches,
)
from .stockanalysis_data import fetch_closes as stockanalysis_fetch_closes
from .yfinance_data import fetch_closes as yf_fetch_closes

DataSource = Literal["alpaca", "yfinance", "csv", "csv+yfinance", "stockanalysis"]


@dataclass(frozen=True)
class CloseHistory:
    closes_by_symbol: dict[str, list[float]]
    benchmark_closes_by_symbol: dict[str, list[float]]
    benchmark_symbols_universe: list[str]


def load_close_history(
    *,
    symbols: list[str],
    total_days: int,
    data_source: DataSource,
    benchmark_symbols: list[str] | None = None,
    alpaca: AlpacaClient | None = None,
    csv_dir: str = ".cache/csv",
    csv_write_json_cache: bool = False,
    stockanalysis_start: str = "1980-01-01",
    stockanalysis_end: str | None = None,
    yfinance_max_workers: int = 10,
    yfinance_retry_delay: float = 1.0,
    yfinance_symbol_delay: float = 0.02,
    use_cache: bool = False,
    refresh_cache: bool = False,
    offline: bool = False,
) -> CloseHistory:
    benchmark_symbols = [symbol for symbol in benchmark_symbols or [] if symbol]

    if data_source in {"yfinance", "csv", "csv+yfinance", "stockanalysis"}:
        requested_symbols = list(dict.fromkeys([*symbols, *benchmark_symbols]))
        if data_source == "csv+yfinance":
            csv_write_yfinance_compatible_caches(
                csv_dir=csv_dir,
                symbols=requested_symbols,
            )
        if data_source in {"yfinance", "csv+yfinance"}:
            all_closes = yf_fetch_closes(
                requested_symbols,
                period="max",
                use_cache=use_cache or data_source == "csv+yfinance",
                refresh_cache=refresh_cache and data_source != "csv+yfinance",
                offline=offline,
                max_workers=yfinance_max_workers,
                retry_delay=yfinance_retry_delay,
                symbol_delay=yfinance_symbol_delay,
            )
        elif data_source == "stockanalysis":
            all_closes = stockanalysis_fetch_closes(
                requested_symbols,
                start=stockanalysis_start,
                end=stockanalysis_end,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                offline=offline,
            )
        else:
            if csv_write_json_cache:
                csv_write_json_caches(csv_dir=csv_dir, symbols=requested_symbols)
            all_closes = csv_fetch_closes(requested_symbols, csv_dir=csv_dir)

        trimmed_closes = {
            symbol: closes[-total_days:] for symbol, closes in all_closes.items()
        }
        return CloseHistory(
            closes_by_symbol={symbol: trimmed_closes[symbol] for symbol in symbols},
            benchmark_closes_by_symbol=trimmed_closes,
            benchmark_symbols_universe=requested_symbols,
        )

    if benchmark_symbols:
        raise ValueError(
            "--benchmark is only supported with --data-source yfinance, csv, "
            "csv+yfinance, or stockanalysis."
        )

    client = alpaca if alpaca is not None else AlpacaClient(AlpacaConfig.from_env())
    closes_by_symbol = client.get_daily_closes_for_period(
        symbols,
        total_days,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        offline=offline,
    )
    return CloseHistory(
        closes_by_symbol=closes_by_symbol,
        benchmark_closes_by_symbol={},
        benchmark_symbols_universe=symbols,
    )
