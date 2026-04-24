from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelInputs:
    symbols: list[str]
    expected_returns: np.ndarray | None
    covariance: np.ndarray | None
    asset_classes: dict[str, str]
    class_min_weights: dict[str, float]
    class_max_weights: dict[str, float]


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return value


def _float_mapping(value: Any, name: str) -> dict[str, float]:
    raw = _require_mapping(value, name)
    parsed = {str(key): float(item) for key, item in raw.items()}
    if not all(np.isfinite(list(parsed.values()))):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return parsed


def load_model_inputs(path: str | Path) -> ModelInputs:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Model file must contain a JSON object.")
    symbols = raw["symbols"]
    if not isinstance(symbols, list):
        raise ValueError("Model symbols must be a list.")
    if not symbols:
        raise ValueError("Model must contain at least one symbol.")
    if any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        raise ValueError("Model symbols must be non-empty strings.")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Model symbols must be unique.")
    expected_returns = None
    covariance = None
    asset_classes_raw = _require_mapping(raw.get("asset_classes", {}), "asset_classes")
    asset_classes = {
        str(symbol): str(asset_class)
        for symbol, asset_class in asset_classes_raw.items()
    }
    class_min_weights = _float_mapping(raw.get("class_min_weights", {}), "class_min_weights")
    class_max_weights = _float_mapping(raw.get("class_max_weights", {}), "class_max_weights")

    # The file can either contain a complete static model or just a symbol
    # universe when inputs will be estimated from Alpaca history at runtime.
    if "expected_returns" in raw:
        expected_returns_map = _require_mapping(
            raw["expected_returns"],
            "expected_returns",
        )
        unknown_expected_returns = sorted(set(expected_returns_map) - set(symbols))
        if unknown_expected_returns:
            raise ValueError(
                "Expected returns provided for unknown symbols: "
                f"{unknown_expected_returns}"
            )
        missing_expected_returns = [
            symbol for symbol in symbols if symbol not in expected_returns_map
        ]
        if missing_expected_returns:
            raise ValueError(
                "Missing expected_returns entries for symbols: "
                f"{missing_expected_returns}"
            )
        expected_returns = np.array(
            [float(expected_returns_map[s]) for s in symbols], dtype=float
        )
        if not np.all(np.isfinite(expected_returns)):
            raise ValueError("Expected returns must contain only finite values.")

    if "covariance" in raw:
        covariance = np.array(raw["covariance"], dtype=float)
        if covariance.shape != (len(symbols), len(symbols)):
            raise ValueError("Covariance matrix must match the number of symbols.")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("Covariance matrix must contain only finite values.")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("Covariance matrix must be symmetric.")

    if (expected_returns is None) != (covariance is None):
        raise ValueError(
            "Model inputs must provide both expected_returns and covariance, or neither."
        )

    unknown_class_symbols = [
        symbol for symbol in asset_classes if symbol not in symbols
    ]
    if unknown_class_symbols:
        raise ValueError(
            f"Asset classes provided for unknown symbols: {unknown_class_symbols}"
        )

    declared_classes = set(asset_classes.values())
    constrained_classes = set(class_min_weights) | set(class_max_weights)
    unknown_constraint_classes = sorted(constrained_classes - declared_classes)
    if unknown_constraint_classes:
        raise ValueError(
            "Class constraints reference unknown asset classes: "
            f"{unknown_constraint_classes}"
        )
    overlapping_classes = set(class_min_weights) & set(class_max_weights)
    invalid_ranges = sorted(
        class_name
        for class_name in overlapping_classes
        if float(class_min_weights[class_name]) > float(class_max_weights[class_name])
    )
    if invalid_ranges:
        raise ValueError(
            f"Class minimums exceed maximums for asset classes: {invalid_ranges}"
        )
    min_total = sum(class_min_weights.values())
    if min_total > 1.0 + 1e-12:
        raise ValueError("Class minimum weights cannot sum above 1.")
    invalid_class_bounds = sorted(
        class_name
        for class_name, value in (class_min_weights | class_max_weights).items()
        if value < 0.0 or value > 1.0
    )
    if invalid_class_bounds:
        raise ValueError(
            "Class constraints must be between 0 and 1 for asset classes: "
            f"{invalid_class_bounds}"
        )

    return ModelInputs(
        symbols=symbols,
        expected_returns=expected_returns,
        covariance=covariance,
        asset_classes=asset_classes,
        class_min_weights={
            name: float(value) for name, value in class_min_weights.items()
        },
        class_max_weights={
            name: float(value) for name, value in class_max_weights.items()
        },
    )
