from __future__ import annotations

import os
import json
import time
from datetime import datetime, timedelta

UNIVERSE_FILE = os.getenv("UNIVERSE_FILE", "./data/universe/symbols.txt")
TCA_SOURCE = (os.getenv("TCA_SOURCE", "db") or "db").strip().lower()  # db | prom
DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL", "")
TCA_TABLE = os.getenv("TCA_TABLE", "tca_events")
PROM_URL = os.getenv("PROM_URL", "http://localhost:9090")
PROM_SLIP = os.getenv("PROM_SLIPPAGE_METRIC", "slippage_bps")
PROM_LAT = os.getenv("PROM_PLACE_MS_METRIC", "order_place_latency_ms")


def load_universe() -> list[str]:
	if os.path.exists(UNIVERSE_FILE):
		with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
			return [ln.strip() for ln in f if ln.strip()]
	return ["BTCUSDT", "ETHUSDT"]


def get_tca_window(symbol: str, days: int = 7) -> list[dict]:  # noqa: ANN001
	# TODO: sustituir por lectura real (DB/metrics). Formato esperado:
	# [{"slippage_bps": float, "place_ms": int}, ...]
	return []


def p95(xs: list[float] | list[int] | None):  # noqa: ANN001
	if not xs:
		return None
	xs = sorted(xs)
	k = max(0, int(0.95 * (len(xs) - 1)))
	return float(xs[k])


def compute_p95_db(symbol: str, days: int = 7):  # noqa: ANN001
	try:
		from sqlalchemy import create_engine, text  # type: ignore
	except Exception as e:  # noqa: BLE001
		raise RuntimeError("sqlalchemy requerido para TCA_SOURCE=db") from e
	eng = create_engine(DB_URL, pool_pre_ping=True)
	with eng.connect() as cx:  # type: ignore[attr-defined]
		# Intento Postgres (percentile_cont)
		try:
			q = text(f"""
SELECT
percentile_cont(0.95) WITHIN GROUP (ORDER BY slippage_bps) AS slip_p95,
percentile_cont(0.95) WITHIN GROUP (ORDER BY order_place_latency_ms) AS place_p95
FROM {TCA_TABLE}
WHERE symbol=:sym AND ts >= NOW() - INTERVAL '{days} days'
""")
			r = cx.execute(q, {"sym": symbol}).mappings().first()
			if r and r["slip_p95"] is not None and r["place_p95"] is not None:
				return float(r["slip_p95"]), float(r["place_p95"])
		except Exception:
			pass
		# Fallback genérico (cargar filas y calcular en Python)
		q2 = text(f"""
SELECT slippage_bps, order_place_latency_ms
FROM {TCA_TABLE}
WHERE symbol=:sym AND ts >= :since
""")
		since = datetime.utcnow() - timedelta(days=days)
		rows = cx.execute(q2, {"sym": symbol, "since": since}).fetchall()
		slips = [float(r[0]) for r in rows if r[0] is not None]
		lats = [float(r[1]) for r in rows if r[1] is not None]
		return (p95(slips), p95(lats))


def _prom_instant(query: str):  # noqa: ANN001
	import requests
	u = f"{PROM_URL}/api/v1/query"
	try:
		r = requests.get(u, params={"query": query}, timeout=5)
		j = r.json()
		if j.get("status") != "success" or not j.get("data", {}).get("result"):
			return None
		v = j["data"]["result"][0]["value"][1]
		try:
			return float(v)
		except Exception:
			return None
	except Exception:
		return None


def compute_p95_prom(symbol: str, days: int = 7):  # noqa: ANN001
	slip_q = f'quantile_over_time(0.95, {PROM_SLIP}{{symbol="{symbol}"}}[{days}d])'
	lat_q = f'quantile_over_time(0.95, {PROM_LAT}{{symbol="{symbol}"}}[{days}d])'
	return _prom_instant(slip_q), _prom_instant(lat_q)


def main() -> None:
	from nge_trader.services import slo
	universe = load_universe()
	report: dict[str, dict[str, float]] = {}
	for sym in universe:
		if TCA_SOURCE == "db":
			slip, lat = compute_p95_db(sym, 7)
		else:
			slip, lat = compute_p95_prom(sym, 7)
		# defaults si no hay datos
		slip = slip if slip is not None else 8.0
		lat = lat if lat is not None else 300.0
		slo.set_symbol_slo(sym, slippage_bps_p95=slip, p95_ms=lat, source=TCA_SOURCE)
		report[sym] = {"slippage_p95": slip, "place_ms_p95": lat, "source": TCA_SOURCE}
	out = {"ts": int(time.time()), "summary": report}
	print(json.dumps(out, indent=2))


if __name__ == "__main__":
	main()


