from __future__ import annotations

import json
from typing import Any, Dict

from nge_trader.services.schema import PROFILE_WRAPPER_SCHEMA, migrate_profile
from jsonschema import validate


def default_profiles() -> Dict[str, Any]:
    """Devuelve un wrapper de perfiles por defecto con `schema_version`.

    Contiene perfiles de slicing y riesgo por símbolo.
    """
    return {
        "schema_version": 1,
        "profiles": {
            "slicing": {
                "BTCUSDT": {"slicing_tranches": 4, "slicing_bps": 5.0},
                "ETHUSDT": {"slicing_tranches": 4, "slicing_bps": 6.0},
            },
            "risk": {
                "BTCUSDT": {"risk_pct_per_trade": 0.01},
                "ETHUSDT": {"risk_pct_per_trade": 0.01},
            },
        },
    }


def migrate_profiles_payload(payload: str) -> str:
    """Normaliza/migra payloads de perfiles al wrapper versión 1 y valida contra esquema."""
    wrapped = migrate_profile(payload)
    validate(instance=json.loads(wrapped), schema=PROFILE_WRAPPER_SCHEMA)
    return wrapped

STRATEGY_TEMPLATES: dict[str, dict] = {
    "ma_cross": {"fast": 10, "slow": 20},
    "rsi": {"window": 14, "low": 30.0, "high": 70.0},
    "macd": {"fast": 12, "slow": 26, "signal_win": 9},
    "bollinger": {"window": 20, "num_std": 2.0},
    "momentum": {"window": 20},
}

PIPELINE_TEMPLATES: dict[str, dict] = {
    "agent_basic": {
        "lookback": 30,
        "features": ["log_returns", "rolling_mean", "rolling_std"],
        "target": "next_return_sign",
        "trainer": "ewma_threshold",
    }
}


