"""Build an instrument universe from a model and aligned price history.

Naked symbols and spread instruments live in the same universe. The strategy
layer sees only instrument NAVs and metadata.
"""

from __future__ import annotations

import numpy as np

from .backtest import align_close_history
from .instruments import Instrument, InstrumentUniverse
from .model import ModelInputs
from .pair_layer import PairLayerConfig, build_spread_nav


def build_instrument_universe(
    *,
    model: ModelInputs,
    closes_by_symbol: dict[str, list[float]],
    pair_config: PairLayerConfig | None = None,
) -> InstrumentUniverse:
    """Build stock and spread instruments with point-in-time spread NAVs."""
    config = pair_config or PairLayerConfig()
    required_symbols = list(model.symbols)
    for pair in model.pairs:
        for leg in pair.legs:
            if leg.symbol not in required_symbols:
                required_symbols.append(leg.symbol)
    missing = [symbol for symbol in required_symbols if symbol not in closes_by_symbol]
    if missing:
        raise ValueError("Missing price history for: " + ", ".join(sorted(missing)))
    if not required_symbols:
        raise ValueError("Model must contain at least one symbol or pair.")

    aligned = align_close_history(required_symbols, closes_by_symbol)
    prices_by_symbol = {
        symbol: np.asarray(aligned[symbol], dtype=float) for symbol in required_symbols
    }
    lengths = {len(values) for values in prices_by_symbol.values()}
    if len(lengths) != 1:
        raise ValueError("Aligned price history must have a single length.")
    periods = next(iter(lengths)) - 1
    returns_by_symbol = {
        symbol: values[1:] / values[:-1] - 1.0
        for symbol, values in prices_by_symbol.items()
    }

    instruments: dict[str, Instrument] = {}
    for symbol in model.symbols:
        prices = prices_by_symbol[symbol]
        nav = prices / prices[0]
        asset_class = model.asset_classes.get(symbol, "unknown")
        instruments[symbol] = Instrument(
            id=symbol,
            kind="stock",
            nav=tuple(float(value) for value in nav),
            asset_class=asset_class,
            category=asset_class,
            available=tuple(True for _ in range(len(nav))),
            shortable=False,
        )

    for pair in model.pairs:
        nav = build_spread_nav(pair, returns_by_symbol, config)
        if len(nav) != periods + 1:
            raise RuntimeError(f"Pair NAV length mismatch for {pair.id}.")
        if pair.id in instruments:
            raise ValueError(f"Pair id collides with an instrument: {pair.id}")
        instruments[pair.id] = Instrument(
            id=pair.id,
            kind="spread",
            nav=tuple(float(value) for value in nav),
            asset_class=pair.asset_class,
            category=pair.category,
            available=tuple(True for _ in range(len(nav))),
            legs=pair.legs,
            shortable=pair.shortable,
        )

    return InstrumentUniverse(instruments=instruments)
