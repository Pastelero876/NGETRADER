from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from threading import Thread, Event
from queue import Queue, Empty
from nge_trader.services.metrics import inc_metric, set_metric

ENABLED = str(os.getenv("TCA_ENABLED", "true")).lower() == "true"
SINK = (os.getenv("TCA_SINK", "db") or "db").strip().lower()  # db | prom | both
DB_URL = os.getenv("TCA_DB_URL", "")
TABLE = os.getenv("TCA_TABLE", "tca_events")
PROM_PUSH_URL = os.getenv("PROM_PUSH_URL", "http://localhost:9091").rstrip("/")
PROM_JOB = os.getenv("PROM_JOB", "tca_writer")
PROM_INSTANCE = os.getenv("PROM_INSTANCE", os.getenv("HOSTNAME") or os.getenv("COMPUTERNAME") or "local")
TCA_BATCH_MAX = int(os.getenv("TCA_BATCH_MAX", "200"))
TCA_FLUSH_MS = int(os.getenv("TCA_FLUSH_MS", "500"))
TCA_MAX_QUEUE = int(os.getenv("TCA_MAX_QUEUE", "50000"))


@dataclass
class TCAEvent:
	symbol: str
	ts: float  # epoch seconds
	slippage_bps: float
	order_place_latency_ms: Optional[float]
	maker_taker: str  # "maker"|"taker"|"unknown"
	route_algo: str  # e.g., "TWAP"|"POV"|"BRACKET"|"DIRECT"|"maker_post_only"
	order_id: Optional[str] = None
	fill_id: Optional[str] = None
	side: Optional[str] = None
	price: Optional[float] = None
	ref_px: Optional[float] = None
	qty: Optional[float] = None
	fees_ccy: Optional[float] = None
	fees_bps: Optional[float] = None
	meta: Optional[Dict[str, Any]] = None


_engine = None
if ENABLED and DB_URL and (SINK in ("db", "both")):
	try:
		from sqlalchemy import create_engine  # type: ignore
		_engine = create_engine(DB_URL, pool_pre_ping=True)
	except Exception:
		_engine = None

_q: "Queue[TCAEvent]" = Queue(maxsize=max(TCA_MAX_QUEUE, 1000))
_stop = Event()
_thr: Optional[Thread] = None


def _push_prometrics(e: TCAEvent) -> None:
	if not ENABLED or (SINK not in ("prom", "both")):
		return
	try:
		import requests
		labels = f'symbol="{e.symbol}",route_algo="{e.route_algo}",maker_taker="{e.maker_taker}",side="{e.side or ''}"'
		lines: list[str] = []
		if e.slippage_bps is not None:
			lines.append(f"tca_slippage_bps{{{labels}}} {float(e.slippage_bps):.6f}")
		if e.order_place_latency_ms is not None:
			lines.append(f"tca_place_latency_ms{{{labels}}} {float(e.order_place_latency_ms):.6f}")
		if not lines:
			return
		payload = "\n".join(lines) + "\n"
		# pushgateway: job grouping key
		url = f"{PROM_PUSH_URL}/metrics/job/{PROM_JOB}/instance/{PROM_INSTANCE}"
		requests.put(url, data=payload.encode("utf-8"), timeout=2)
	except Exception:
		return


def _worker() -> None:
	from sqlalchemy import text  # type: ignore
	while not _stop.is_set():
		batch: list[TCAEvent] = []
		try:
			ev = _q.get(timeout=max(TCA_FLUSH_MS, 100) / 1000.0)
			batch.append(ev)
			while len(batch) < max(TCA_BATCH_MAX, 50):
				try:
					batch.append(_q.get_nowait())
				except Empty:
					break
		except Empty:
			continue
		# DB sink
		if _engine and (SINK in ("db", "both")):
			try:
				with _engine.begin() as cx:  # type: ignore[attr-defined]
					for e in batch:
						cx.execute(
							text(
								f"""
INSERT INTO {TABLE}
(symbol, ts, slippage_bps, order_place_latency_ms, maker_taker, route_algo,
order_id, fill_id, side, price, ref_px, qty, fees_ccy, fees_bps, meta)
VALUES
(:symbol, to_timestamp(:ts), :slippage_bps, :lat, :mt, :algo,
:oid, :fid, :side, :price, :ref_px, :qty, :fees_ccy, :fees_bps, CAST(:meta AS JSONB))
ON CONFLICT (order_id, fill_id) DO NOTHING
						"""
						),
						{
							"symbol": e.symbol,
							"ts": e.ts,
							"slippage_bps": e.slippage_bps,
							"lat": e.order_place_latency_ms,
							"mt": e.maker_taker,
							"algo": e.route_algo,
							"oid": e.order_id,
							"fid": e.fill_id,
							"side": e.side,
							"price": e.price,
							"ref_px": e.ref_px,
							"qty": e.qty,
							"fees_ccy": e.fees_ccy,
							"fees_bps": e.fees_bps,
							"meta": (e.meta or {}),
						},
					)
			except Exception:
				inc_metric("tca_writer_db_errors_total", 1.0)
				# swallow errors to avoid blocking producer
				pass
		# Prom sink
		if SINK in ("prom", "both"):
			for e in batch:
				try:
					_push_prometrics(e)
				except Exception:
					inc_metric("tca_writer_push_failures_total", 1.0)
					continue
		# After flush
		inc_metric("tca_writer_batches_total", 1.0)
		try:
			set_metric("tca_writer_queue_len", float(_q.qsize()))
		except Exception:
			pass


def start() -> None:
	global _thr
	if not ENABLED or not _engine or _thr:
		return
	_thr = Thread(target=_worker, name="tca-writer", daemon=True)
	_thr.start()


def stop() -> None:
	_stop.set()


def submit(ev: TCAEvent) -> None:
	if not ENABLED:
		return
	try:
		_q.put_nowait(ev)
		inc_metric("tca_writer_events_total", 1.0)
		try:
			set_metric("tca_writer_queue_len", float(_q.qsize()))
		except Exception:
			pass
	except Exception:
		# Cola llena: no bloquear OMS
		inc_metric("tca_writer_dropped_events_total", 1.0)
		return


