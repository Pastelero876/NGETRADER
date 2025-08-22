from __future__ import annotations

from typing import Dict
from nge_trader.services.metrics import set_metric_labeled

SLO_STORE: Dict[str, Dict[str, float | str]] = {}


def set_symbol_slo(symbol: str, slippage_bps_p95: float, p95_ms: float, source: str | None = None) -> None:
	data: Dict[str, float | str] = {"slip": float(slippage_bps_p95), "p95": float(p95_ms), "source": str(source or "unknown")}
	SLO_STORE[str(symbol).upper()] = data
	# Export thresholds como gauges etiquetados
	try:
		set_metric_labeled("slo_slippage_bps_threshold", float(slippage_bps_p95), {"symbol": str(symbol).upper()})
		set_metric_labeled("slo_place_ms_threshold", float(p95_ms), {"symbol": str(symbol).upper()})
	except Exception:
		pass


def get_symbol_slo(symbol: str) -> Dict[str, float | str]:
	return dict(SLO_STORE.get(str(symbol).upper(), {"slip": 8.0, "p95": 300.0, "source": "unknown"}))



