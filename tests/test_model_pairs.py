from __future__ import annotations

import json

import pytest

from portfolio_opt.model import load_model_inputs


def _write(tmp_path, payload) -> str:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_load_model_inputs_parses_signed_pair_legs(tmp_path) -> None:
    path = _write(
        tmp_path,
        {
            "symbols": ["AAA"],
            "asset_classes": {"AAA": "sector_test", "BBB": "sector_test"},
            "pairs": [
                {
                    "id": "AAA/BBB",
                    "legs": [
                        {"symbol": "AAA", "weight": 1.0},
                        {"symbol": "BBB", "weight": -1.5},
                    ],
                    "category": "test_pair",
                    "asset_class": "sector_test",
                }
            ],
        },
    )

    model = load_model_inputs(path)

    assert len(model.pairs) == 1
    pair = model.pairs[0]
    assert pair.id == "AAA/BBB"
    assert pair.legs[0].weight == 1.0
    assert pair.legs[1].weight == -1.5
    assert pair.category == "test_pair"


def test_load_model_inputs_accepts_legacy_long_short_pairs(tmp_path) -> None:
    path = _write(
        tmp_path,
        {
            "symbols": ["AAA", "BBB"],
            "asset_classes": {"AAA": "sector_test", "BBB": "sector_test"},
            "pairs": [{"long": "AAA", "short": "BBB", "category": "test_pair"}],
        },
    )

    model = load_model_inputs(path)

    assert model.pairs[0].legs[0].symbol == "AAA"
    assert model.pairs[0].legs[1].weight == -1.0


def test_load_model_inputs_allows_pair_only_universe(tmp_path) -> None:
    path = _write(
        tmp_path,
        {
            "symbols": [],
            "pairs": [
                {
                    "id": "AAA/BBB",
                    "legs": [
                        {"symbol": "AAA", "weight": 1.0},
                        {"symbol": "BBB", "weight": -1.0},
                    ],
                    "category": "test_pair",
                    "asset_class": "sector_test",
                }
            ],
        },
    )

    model = load_model_inputs(path)

    assert model.symbols == []
    assert len(model.pairs) == 1


def test_load_model_inputs_rejects_pair_id_collision(tmp_path) -> None:
    path = _write(
        tmp_path,
        {
            "symbols": ["AAA"],
            "pairs": [
                {
                    "id": "AAA",
                    "legs": [
                        {"symbol": "AAA", "weight": 1.0},
                        {"symbol": "BBB", "weight": -1.0},
                    ],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="collides"):
        load_model_inputs(path)


def test_load_model_inputs_rejects_duplicate_pair_ids(tmp_path) -> None:
    path = _write(
        tmp_path,
        {
            "symbols": ["AAA", "BBB"],
            "pairs": [
                {
                    "id": "AAA/BBB",
                    "legs": [
                        {"symbol": "AAA", "weight": 1.0},
                        {"symbol": "BBB", "weight": -1.0},
                    ],
                },
                {
                    "id": "AAA/BBB",
                    "legs": [
                        {"symbol": "AAA", "weight": 1.0},
                        {"symbol": "BBB", "weight": -1.0},
                    ],
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="Duplicate pair id"):
        load_model_inputs(path)
