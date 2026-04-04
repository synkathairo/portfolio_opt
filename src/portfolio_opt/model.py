from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelInputs:
    symbols: list[str]
    expected_returns: np.ndarray
    covariance: np.ndarray


def load_model_inputs(path: str | Path) -> ModelInputs:
    raw = json.loads(Path(path).read_text())
    symbols = raw["symbols"]
    expected_returns_map = raw["expected_returns"]
    expected_returns = np.array([float(expected_returns_map[s]) for s in symbols], dtype=float)
    covariance = np.array(raw["covariance"], dtype=float)

    if covariance.shape != (len(symbols), len(symbols)):
        raise ValueError("Covariance matrix must match the number of symbols.")

    return ModelInputs(
        symbols=symbols,
        expected_returns=expected_returns,
        covariance=covariance,
    )
