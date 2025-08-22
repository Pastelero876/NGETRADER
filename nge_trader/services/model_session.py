from __future__ import annotations

from typing import Any, Dict

_current: Dict[str, Any] = {"pinned": False, "model_id": None, "scaler_id": None, "features_hash": None}


def pin(model_id: str, scaler_id: str | None = None, features_hash: str | None = None) -> None:
	_current.update({"pinned": True, "model_id": model_id, "scaler_id": scaler_id, "features_hash": features_hash})


def unpin() -> None:
	_current.update({"pinned": False, "model_id": None, "scaler_id": None, "features_hash": None})


def get() -> Dict[str, Any]:
	return dict(_current)
