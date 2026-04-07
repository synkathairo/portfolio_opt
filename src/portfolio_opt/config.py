from __future__ import annotations

from dataclasses import dataclass, field
from os import getenv

from dotenv import load_dotenv


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str
    api_secret: str
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        load_dotenv()
        api_key = getenv("APCA_API_KEY_ID")
        api_secret = getenv("APCA_API_SECRET_KEY")
        base_url = getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        data_url = getenv("APCA_API_DATA_URL", "https://data.alpaca.markets")
        if not api_key or not api_secret:
            raise ValueError(
                "Missing Alpaca credentials. Set APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY in your environment or a local .env file."
            )
        return cls(
            api_key=api_key, api_secret=api_secret, base_url=base_url, data_url=data_url
        )


@dataclass(frozen=True)
class OptimizationConfig:
    risk_aversion: float = 4.0
    min_weight: float = 0.0
    max_weight: float = 0.35
    rebalance_threshold: float = 0.02
    turnover_penalty: float = 0.02
    force_full_investment: bool = True
    min_cash_weight: float = 0.0
    max_turnover: float | None = None
    min_invested_weight: float = 0.0
    class_min_weights: dict[str, float] = field(default_factory=dict)
    class_max_weights: dict[str, float] = field(default_factory=dict)
