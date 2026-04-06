from __future__ import annotations

import pandas as pd


def closes_to_market_data(
    closes_by_symbol: dict[str, list[float]],
    cash_return: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = list(closes_by_symbol)
    price_frame = pd.DataFrame({symbol: closes_by_symbol[symbol] for symbol in symbols})
    if price_frame.isna().any().any():
        raise ValueError("Price history contains missing values.")

    # Alpaca closes are used as a stand-in for open-to-open prices so the first
    # cvxportfolio experiment can share the same data source as the custom path.
    returns = price_frame.iloc[1:].to_numpy() / price_frame.iloc[:-1].to_numpy() - 1.0
    # Returns have one fewer row than prices, so size the shared index from the
    # computed return matrix rather than the original close history length.
    index = pd.date_range(start="2000-01-03", periods=returns.shape[0], freq="B", tz="UTC")
    returns_frame = pd.DataFrame(returns, index=index, columns=symbols)
    returns_frame["USDOLLAR"] = cash_return

    prices_frame = pd.DataFrame(
        price_frame.iloc[:-1].to_numpy(),
        index=index,
        columns=symbols,
    )
    return returns_frame, prices_frame


def momentum_forecast(
    returns_frame: pd.DataFrame,
    momentum_window: int,
    mean_shrinkage: float,
) -> pd.DataFrame:
    asset_returns = returns_frame.drop(columns=["USDOLLAR"], errors="ignore")
    forecasts = pd.DataFrame(0.0, index=asset_returns.index, columns=asset_returns.columns)

    for row_index in range(momentum_window, len(asset_returns)):
        window = asset_returns.iloc[row_index - momentum_window : row_index]
        cumulative = (1.0 + window).prod(axis=0) - 1.0
        forecasts.iloc[row_index] = cumulative * (1.0 - mean_shrinkage)

    return forecasts
