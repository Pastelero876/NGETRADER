from __future__ import annotations

from typing import Dict


STATE: Dict[str, float] = {"risk_per_trade": 0.0025, "used_R": 0.0, "daily_budget_left": 1.0}


def get() -> Dict[str, float]:
	return dict(STATE)


def set_used_R(value: float) -> None:
	STATE["used_R"] = float(value)


def set_daily_budget_left(value: float) -> None:
	STATE["daily_budget_left"] = float(value)



