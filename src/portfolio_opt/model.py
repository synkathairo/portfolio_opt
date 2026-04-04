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


def load_model_inputs(path: str | Path) -> ModelInputs:
    raw = json.loads(Path(path).read_text())
    symbols = raw["symbols"]
    expected_returns = None
    covariance = None

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

    return ModelInputs(
        symbols=symbols,
        expected_returns=expected_returns,
        covariance=covariance,
    )
