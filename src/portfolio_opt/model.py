from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelInputs:
    symbols: list[str]
    expected_returns: np.ndarray | None
    covariance: np.ndarray | None
    asset_classes: dict[str, str]
    class_min_weights: dict[str, float]
    class_max_weights: dict[str, float]


def load_model_inputs(path: str | Path) -> ModelInputs:
    raw = json.loads(Path(path).read_text())
    symbols = raw["symbols"]
    expected_returns = None
    covariance = None
    asset_classes = raw.get("asset_classes", {})
    class_min_weights = raw.get("class_min_weights", {})
    class_max_weights = raw.get("class_max_weights", {})

    # The file can either contain a complete static model or just a symbol
    # universe when inputs will be estimated from Alpaca history at runtime.
    if "expected_returns" in raw:
        expected_returns_map = raw["expected_returns"]
        expected_returns = np.array([float(expected_returns_map[s]) for s in symbols], dtype=float)

    if "covariance" in raw:
        covariance = np.array(raw["covariance"], dtype=float)
        if covariance.shape != (len(symbols), len(symbols)):
            raise ValueError("Covariance matrix must match the number of symbols.")

    if (expected_returns is None) != (covariance is None):
        raise ValueError("Model inputs must provide both expected_returns and covariance, or neither.")

    unknown_class_symbols = [symbol for symbol in asset_classes if symbol not in symbols]
    if unknown_class_symbols:
        raise ValueError(f"Asset classes provided for unknown symbols: {unknown_class_symbols}")

    return ModelInputs(
        symbols=symbols,
        expected_returns=expected_returns,
        covariance=covariance,
        asset_classes=asset_classes,
        class_min_weights={name: float(value) for name, value in class_min_weights.items()},
        class_max_weights={name: float(value) for name, value in class_max_weights.items()},
    )
