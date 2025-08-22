from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid
import json as _json
import datetime as _dt

from nge_trader.repository.db import Database
from nge_trader.config.settings import Settings
from nge_trader.services.time_sync import estimate_skew_ms
from nge_trader.services.market_clock import is_session_open
from nge_trader.services.metrics import inc_metric, set_metric
from nge_trader.services.metrics import inc_metric_labeled, set_metric_labeled
from nge_trader.services.execution import SLOGate
from nge_trader.services.rate_limiter import GlobalRateLimiter
from decimal import Decimal, getcontext, ROUND_DOWN
import os


def normalize_order(raw: Dict[str, Any]) -> Dict[str, Any]:
	"""Normaliza una orden de broker a campos comunes usados por la app.

	Compatibilidad:
	- Alpaca: id, submitted_at, qty, limit_price
	- Binance: orderId, origQty/qty, price, time
	- Coinbase: id, created_at, size, price
	- Paper: id, created_at, qty, price
	- IBKR: orderId, qty, price
	"""
	order_id = raw.get("id") or raw.get("order_id") or raw.get("orderId")
	qty_val = (
		raw.get("qty")
		or raw.get("quantity")
		or raw.get("origQty")
		or raw.get("size")
		or raw.get("orderQty")
		or 0
	)
	price_val = raw.get("price") or raw.get("limit_price") or raw.get("avg_price") or raw.get("stopLimitPrice") or 0
	ts_val = raw.get("ts") or raw.get("created_at") or raw.get("submitted_at") or raw.get("time")
	symbol_val = raw.get("symbol") or raw.get("product_id") or raw.get("instrument")
	return {
		"ts": ts_val,
		"symbol": symbol_val,
		"side": (raw.get("side") or "").lower(),
		"qty": float(qty_val or 0),
		"price": float(price_val or 0),
		"status": raw.get("status") or raw.get("orderStatus") or "sent",
		"order_id": order_id,
	}


# Helper: cálculo de ejecutabilidad/semáforo
def compute_executability(symbol: str, side: str, expected_edge_bps: float | None = None) -> Dict[str, Any]:
    """Calcula si una operación sería ejecutable (verde/amarillo/rojo) basado en edge vs costes.

    Retorna: { level: green|yellow|red, edge_bps, threshold_bps, spread_bps, fees_bps }
    """
    s = Settings()
    try:
        from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
        q = _LL().get_quote_metrics(symbol)
        bid = float(q.get("bid") or 0.0)
        ask = float(q.get("ask") or 0.0)
        mid = 0.5 * (bid + ask) if (bid > 0 and ask > 0) else 0.0
        spread_bps = ((ask - bid) / max(mid, 1e-9)) * 10000.0 if mid > 0 else 0.0
    except Exception:
        mid = 0.0
        spread_bps = 0.0
    fees_bps = 5.0
    buf_bps = float(getattr(s, "edge_buffer_bps", 5.0) or 5.0)
    thr_bps = fees_bps + 0.5 * spread_bps + buf_bps
    edge_bps = float(expected_edge_bps or 0.0)
    diff = edge_bps - thr_bps
    if diff >= 5.0:
        level = "green"
    elif diff >= 0.0:
        level = "yellow"
    else:
        level = "red"
    return {
        "level": level,
        "edge_bps": edge_bps,
        "threshold_bps": thr_bps,
        "spread_bps": spread_bps,
        "half_spread_bps": 0.5 * spread_bps,
        "fees_bps": fees_bps,
    }

def normalize_fill(raw: Dict[str, Any]) -> Dict[str, Any]:
	"""Normaliza una ejecución/fill a campos comunes.

	Compatibilidad:
	- Alpaca: filled_qty, fill_price, transaction_time, order_id
	- Binance: qty, price, time, orderId
	- Coinbase: size, price, trade_id/order_id, created_at
	"""
	qty_val = raw.get("qty") or raw.get("quantity") or raw.get("filled_qty") or raw.get("executedQty") or raw.get("size") or 0
	price_val = raw.get("price") or raw.get("fill_price") or raw.get("price") or 0
	ts_val = raw.get("ts") or raw.get("transaction_time") or raw.get("timestamp") or raw.get("time") or raw.get("created_at")
	symbol_val = raw.get("symbol") or raw.get("product_id")
	order_id = raw.get("order_id") or raw.get("id") or raw.get("orderId") or raw.get("trade_id")
	# Fees: distintos brokers
	fees_val = (
		raw.get("fees")
		or raw.get("fee")
		or raw.get("commission")
		or 0
	)
	try:
		fees_val = float(fees_val or 0)
	except Exception:
		fees_val = 0.0
	return {
		"ts": ts_val,
		"symbol": symbol_val,
		"side": (raw.get("side") or "").lower(),
		"qty": float(qty_val or 0),
		"price": float(price_val or 0),
		"order_id": order_id,
		"fees": float(fees_val or 0.0),
	}


# SDK de órdenes unificado (envoltorios de conveniencia)

# Precisión numérica: helpers
def _quantize_step(value: float, step: float) -> float:
    try:
        if not step or step <= 0:
            return float(value)
        getcontext().prec = 28
        v = Decimal(str(value))
        st = Decimal(str(step))
        q = (v / st).to_integral_value(rounding=ROUND_DOWN) * st
        return float(q)
    except Exception:
        return float(value)


def quantize_quantity_price(symbol: str, quantity: float, price: float | None) -> tuple[float, float | None]:
    s = Settings()
    # qty step por símbolo o global
    try:
        import json as __json
        lot_map = {}
        if getattr(s, "lot_size_per_symbol", None):
            lot_map = __json.loads(str(s.lot_size_per_symbol))
    except Exception:
        lot_map = {}
    qty_step = float(lot_map.get(symbol.upper()) or (s.qty_step or 0.0) or 0.0)
    q_qty = _quantize_step(float(quantity), qty_step) if qty_step else float(quantity)
    # price tick global (o per-symbol futuro)
    tick = float(s.price_tick or 0.0)
    q_px = _quantize_step(float(price), tick) if (price is not None and tick) else (float(price) if price is not None else None)
    return q_qty, q_px

def place_market_order(broker: Any, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
	if hasattr(broker, "place_order"):
		# Gating básico por perfil/estado/tiempo
		s = Settings()
		# Allowlist LIVE_SYMBOLS
		try:
			live_syms = {x.strip().upper() for x in str(os.getenv("LIVE_SYMBOLS", "")).split(",") if x.strip()}
			if live_syms and symbol.upper() not in live_syms and str(s.profile or "").lower() == "live":
				raise RuntimeError(f"Símbolo {symbol} no permitido en LIVE_SYMBOLS")
		except Exception:
			pass
		# Cumplimiento: bloquear si se requiere KYC/AML y no está marcado como completado
		if bool(getattr(s, "require_kyc_aml", False)) and not bool(getattr(s, "kyc_aml_completed", False)):
			raise RuntimeError("Cumplimiento: KYC/AML requerido no está completado")
		if s.kill_switch_armed:
			raise RuntimeError("Kill-switch armado: bloqueando envíos")
		paused = {x.strip().upper() for x in (s.paused_symbols or "").split(",") if x}
		if symbol.upper() in paused:
			raise RuntimeError(f"Símbolo pausado: {symbol}")
		if s.enable_ws_backfill and not is_session_open(s.market_clock_exchange):
			raise RuntimeError("Mercado fuera de ventana de trading")
		if s.enable_retry_policy:
			skew = abs(estimate_skew_ms(server=s.ntp_server))
			if skew > float(s.max_time_skew_ms):
				raise RuntimeError(f"Skew de reloj excesivo: {skew} ms")
		# SLO gate (error-rate)
		gate = SLOGate()
		if not gate.can_send():
			raise RuntimeError("SLO gate activo: error-rate reciente excede umbral")
		# Best execution: breaker por slippage promedio reciente
		try:
			vals = Database().recent_metric_values("slippage_bps", 50)
			last_n = [v for _, v in vals[-20:]]
			if last_n:
				avg = float(sum(last_n) / len(last_n))
				if abs(avg) > float(Settings().max_slippage_bps or 50.0):
					raise RuntimeError("Breaker por slippage: umbral excedido")
		except Exception:
			pass
		# Fill ratio reciente: desviar a limit post-only si es bajo
		try:
			from nge_trader.services import metrics as _MM
			fills_total = float(_MM._METRICS_STORE.get("fills_total", 0.0))
			orders_total = float(_MM._METRICS_STORE.get("orders_placed_total", 0.0))
			fill_ratio_recent = (fills_total / orders_total) if orders_total > 0 else 1.0
			if fill_ratio_recent < float(Settings().min_fill_ratio_recent or 0.5):
				from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
				mp = _LL().get_microprice(symbol) or 0.0
				px = float(mp)
				if px > 0:
					return place_limit_order_post_only(broker, symbol, side, float(quantity), float(px))
		except Exception:
			pass
		# Umbral de ejecutabilidad por trade: edge esperado >= fees + 0.5*spread + buffer
		try:
			from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
			q = _LL().get_quote_metrics(symbol)
			bid = float(q.get("bid") or 0.0)
			ask = float(q.get("ask") or 0.0)
			mid = 0.5 * (bid + ask) if (bid > 0 and ask > 0) else 0.0
			spread_half = 0.5 * (ask - bid) if (bid > 0 and ask > 0) else 0.0
			# estimación simple de fees (bps), configurable vía fees_schedule en producción
			fees_bps = 5.0
			buf_bps = float(getattr(s, "edge_buffer_bps", 5.0) or 5.0)
			# edge_esperado: si disponemos de intención (por ejemplo, from strategy), aquí simulamos 0
			edge_bps = 0.0
			threshold_bps = fees_bps + (spread_half / max(mid, 1e-9)) * 10000.0 + buf_bps if mid > 0 else (fees_bps + buf_bps)
			if edge_bps < threshold_bps:
				raise RuntimeError(f"No-trade: edge {edge_bps:.1f}bps < umbral {threshold_bps:.1f}bps")
		except Exception:
			pass
		# Validación fraccional/min_qty
		try:
			import json as __json
			min_qty_map = {}
			if getattr(s, "min_qty_per_symbol", None):
				try:
					min_qty_map = __json.loads(str(s.min_qty_per_symbol))
				except Exception:
					min_qty_map = {}
			per_sym = float(min_qty_map.get(symbol.upper()) or 0.0)
			if per_sym and float(quantity) < per_sym:
				raise RuntimeError(f"Cantidad {quantity} < MIN_QTY(symbol) {per_sym}")
			if getattr(s, "min_qty", None) is not None and float(quantity) < float(s.min_qty):
				raise RuntimeError(f"Cantidad {quantity} < MIN_QTY {s.min_qty}")
			if not bool(getattr(s, "allow_fractional", True)):
				if abs(float(quantity) - round(float(quantity))) > 1e-9:
					raise RuntimeError("Fraccionamiento no permitido por configuración")
		except Exception:
			pass
		client_order_id = generate_idempotency_key(symbol, side)
		db = Database()
		correlation_id = f"corr_{uuid.uuid4().hex[:12]}"
		# Budget por estrategia (diario)
		sent_today = 0
		try:
			sent_today = db.get_strategy_orders_sent_today(Settings().strategy_id)
		except Exception:
			pass
		if sent_today >= int(Settings().max_trades_per_day_per_strategy or 50):
			raise RuntimeError("Presupuesto de órdenes diario por estrategia agotado")
		# Rate limiting por broker, estrategia y símbolo
		rl = GlobalRateLimiter.get()
		key_broker = f"orders_{s.broker}"
		key_strategy = f"orders_{s.broker}_strat_{s.strategy_id}"
		key_symbol = f"orders_{s.broker}_{symbol.upper()}"
		rl.configure(key_broker, capacity=int(s.broker_orders_per_minute), refill_per_sec=max(1.0, int(s.broker_orders_per_minute)) / 60.0)
		rl.configure(key_strategy, capacity=int(s.strategy_orders_per_minute), refill_per_sec=max(1.0, int(s.strategy_orders_per_minute)) / 60.0)
		rl.configure(key_symbol, capacity=int(s.symbol_orders_per_minute), refill_per_sec=max(1.0, int(s.symbol_orders_per_minute)) / 60.0)
		if not (rl.acquire(key_broker) and rl.acquire(key_strategy) and rl.acquire(key_symbol)):
			raise RuntimeError("Rate limit alcanzado (broker/estrategia/símbolo)")
		# Budgets por símbolo y cuenta
		try:
			sym_sent = db.get_symbol_orders_sent_today(symbol)
			if sym_sent >= int(Settings().max_trades_per_day_per_symbol or 20):
				raise RuntimeError("Presupuesto de órdenes diario por símbolo agotado")
		except Exception:
			pass
		try:
			acc_sent = db.get_account_orders_sent_today(Settings().account_id)
			if acc_sent >= int(Settings().max_trades_per_day_per_account or 100):
				raise RuntimeError("Presupuesto de órdenes diario por cuenta agotado")
		except Exception:
			pass
		# Validaciones instrumentales previas: notional mínimo y pasos
		try:
			mp = None
			from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
			mp = _LL().get_microprice(symbol)
		except Exception:
			mp = None
		# ES/CVaR intradía: recortar tamaño si excede presupuesto
		try:
			from nge_trader.services.risk import intraday_es_cvar
			# serie intradía desde low-lat, fallback a diario
			price_series = None
			try:
				df_ll = _LL().get_stable_agg(symbol)
				if df_ll is not None and not df_ll.empty:
					price_series = df_ll["price"].astype(float)
			except Exception:
				price_series = None
			if price_series is None or price_series.empty:
				try:
					from nge_trader.services.app_service import AppService as _AS
					df_d = _AS().data_provider.get_daily_adjusted(symbol)
					if df_d is not None and not df_d.empty:
						price_series = df_d["close"].astype(float).tail(200)
				except Exception:
					price_series = None
			_, es_i = intraday_es_cvar(price_series if price_series is not None else __import__("pandas").Series(dtype=float), window_minutes=int(Settings().es_window_minutes or 60), alpha=0.025)
			budget = float(Settings().es_budget_pct or 0.03)
			if float(abs(es_i)) > float(budget):
				scale = max(0.2, float(budget) / max(abs(es_i), 1e-9))
				quantity = float(quantity) * float(scale)
		except Exception:
			pass
		px_est = float(mp) if mp is not None else None
		# Pre-trade: tamaño máximo de orden
		if Settings().max_order_qty and float(quantity) > float(Settings().max_order_qty):
			raise RuntimeError("Tamaño de orden excede el máximo permitido")
		# Pre-trade: posición neta máxima por símbolo (consulta básica por fills)
		try:
			pos = 0.0
			for f in db.recent_fills(1000):
				if (f.get("symbol") or "").upper() == symbol.upper():
					qty = float(f.get("qty") or 0.0)
					pos += qty if (f.get("side") or "").lower() == "buy" else -qty
			if Settings().max_position_qty_per_symbol and abs(pos + float(quantity) * (1 if side.lower()=="buy" else -1)) > float(Settings().max_position_qty_per_symbol):
				raise RuntimeError("Posición neta máxima por símbolo excedida")
		except Exception:
			pass
		# Notional mínimo por símbolo o global
		try:
			import json as __json
			min_notional_map = {}
			if getattr(Settings(), "min_notional_usd_per_symbol", None):
				min_notional_map = __json.loads(str(Settings().min_notional_usd_per_symbol))
		except Exception:
			min_notional_map = {}
		if px_est:
			notional = float(px_est) * float(quantity)
			per_sym_notional = float(min_notional_map.get(symbol.upper()) or 0.0)
			global_notional = float(Settings().min_notional_usd or 0.0)
			min_req = per_sym_notional or global_notional
			if min_req and notional < min_req:
				raise RuntimeError(f"Notional mínimo {min_req} USD no alcanzado")
		if Settings().qty_step and float(Settings().qty_step) > 0:
			step = float(Settings().qty_step)
			if abs((float(quantity) / step) - round(float(quantity) / step)) > 1e-8:
				raise RuntimeError("Cantidad no múltiplo de qty_step")
		# Pre-trade: price collars para órdenes market (validación aproximada con px_est)
		if Settings().price_collar_pct and px_est and float(px_est) > 0:
			collar = float(Settings().price_collar_pct)
			# en market, sólo verificamos que microprice no esté anómalo (placeholder)
			if collar > 0.0 and (px_est <= 0):
				raise RuntimeError("Precio de referencia no disponible para aplicar collar")
		# Quantize cantidad antes de enviar
		quantity = quantize_quantity_price(symbol, quantity, None)[0]
		payload = {
			"symbol": symbol,
			"side": side,
			"quantity": float(quantity),
			"type_": "market",
			"client_order_id": client_order_id,
			"ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds"),
			"decision_id": f"did_{uuid.uuid4().hex[:16]}",
		}
		outbox_id = db.put_order_outbox(_json.dumps(payload, ensure_ascii=False), client_order_id, correlation_id)
		try:
			gate.register_attempt()
			t0 = _dt.datetime.now(_dt.UTC).timestamp()
			try:
				res = broker.place_order(symbol=symbol, side=side, quantity=quantity, type_="market", client_order_id=client_order_id)
			except TypeError:
				res = broker.place_order(symbol=symbol, side=side, quantity=quantity, type_="market")
			broker_order_id = res.get("id") or res.get("order_id") or res.get("orderId")
			db.mark_order_outbox(outbox_id, status="sent", broker_order_id=str(broker_order_id) if broker_order_id else None)
			t1 = _dt.datetime.now(_dt.UTC).timestamp()
			from nge_trader.services.metrics import observe_latency
			set_metric("order_place_latency_ms", (t1 - t0) * 1000.0)
			observe_latency("order_place_latency_ms", (t1 - t0) * 1000.0)
			inc_metric("orders_placed_total", 1.0)
			# Registrar orden con IDs de MiFID (decision/execution)
			try:
				db.record_order({
					"ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds"),
					"symbol": symbol,
					"side": side,
					"qty": float(quantity),
					"price": None,
					"status": "sent",
					"order_id": str(broker_order_id) if broker_order_id else None,
					"decision_id": payload.get("decision_id"),
					"execution_id": f"eid_{uuid.uuid4().hex[:16]}",
					"dea": 1 if bool(Settings().dea_indicator) else 0,
				})
			except Exception:
				pass
			# Medir slippage vs microprice si posible
			try:
				from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
				mp_now = _LL().get_microprice(symbol)
				fill_px = float(res.get("price") or 0.0)
				if mp_now and float(mp_now) > 0 and fill_px > 0:
					bps = (fill_px - float(mp_now)) / float(mp_now) * 10000.0
					if str(side).lower() == "sell":
						bps = -bps
					Database().record_metric_value("slippage_bps", float(bps))
					set_metric("slippage_last_bps", float(bps))
			except Exception:
				pass
			try:
				set_metric_labeled(
					"order_place_latency_ms",
					(t1 - t0) * 1000.0,
					{"broker": s.broker, "strategy": s.strategy_id, "symbol": symbol.upper()},
				)
				inc_metric_labeled(
					"orders_placed_total",
					1.0,
					{"broker": s.broker, "strategy": s.strategy_id, "symbol": symbol.upper()},
				)
			except Exception:
				pass
			try:
				db.inc_strategy_orders_sent_today(Settings().strategy_id, 1)
				db.inc_symbol_orders_sent_today(symbol, 1)
				db.inc_account_orders_sent_today(Settings().account_id, 1)
			except Exception:
				pass
			return res
		except Exception as exc:  # noqa: BLE001
			db.mark_order_outbox(outbox_id, status="error", error=str(exc))
			try:
				gate.register_error()
			except Exception:
				pass
			inc_metric("orders_error_total", 1.0)
			try:
				inc_metric_labeled(
					"orders_error_total",
					1.0,
					{"broker": s.broker, "strategy": s.strategy_id, "symbol": symbol.upper(), "error_class": type(exc).__name__},
				)
			except Exception:
				pass
			try:
				inc_metric("breaker_activations_total", 1.0)
			except Exception:
				pass
			# Registrar orden rechazada en DB
			try:
				self_ts = _dt.datetime.now(_dt.UTC).isoformat()
				db.record_order({
					"ts": self_ts,
					"symbol": symbol,
					"side": side,
					"qty": float(quantity),
					"price": None,
					"status": "rejected",
					"order_id": None,
				})
				try:
					inc_metric_labeled(
						"orders_rejected_total",
						1.0,
						{"broker": s.broker, "strategy": s.strategy_id, "symbol": symbol.upper()},
					)
				except Exception:
					pass
			except Exception:
				pass
			raise
	return NotImplementedError("Broker no soporta place_order")


def place_limit_order_post_only(
	broker: Any,
	symbol: str,
	side: str,
	quantity: float,
	price: float,
	tif: str = "GTC",
) -> Dict[str, Any]:
	if hasattr(broker, "place_order"):
		s = Settings()
		if s.kill_switch_armed:
			raise RuntimeError("Kill-switch armado: bloqueando envíos")
		paused = {x.strip().upper() for x in (s.paused_symbols or "").split(",") if x}
		if symbol.upper() in paused:
			raise RuntimeError(f"Símbolo pausado: {symbol}")
		if s.enable_ws_backfill and not is_session_open(s.market_clock_exchange):
			raise RuntimeError("Mercado fuera de ventana de trading")
		if s.enable_retry_policy:
			skew = abs(estimate_skew_ms(server=s.ntp_server))
			if skew > float(s.max_time_skew_ms):
				raise RuntimeError(f"Skew de reloj excesivo: {skew} ms")
		gate = SLOGate()
		if not gate.can_send():
			raise RuntimeError("SLO gate activo: error-rate reciente excede umbral")
		client_order_id = generate_idempotency_key(symbol, side)
		db = Database()
		correlation_id = f"corr_{uuid.uuid4().hex[:12]}"
		# Señal de precio L1: microprice si está disponible
		try:
			from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
			mp = _LL().get_microprice(symbol)
			if mp is not None:
				from nge_trader.services.metrics import set_metric
				set_metric("intended_microprice", float(mp))
		except Exception:
			pass
		# Validaciones instrumentales previas: notional mínimo y pasos/ticks
		try:
			mp = None
			from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
			mp = _LL().get_microprice(symbol)
		except Exception:
			mp = None
		# Quantize cantidad y precio
		quantity, price = quantize_quantity_price(symbol, quantity, price)
		px_est = float(price or (mp if mp is not None else 0.0))
		if Settings().min_notional_usd and px_est:
			notional = float(px_est) * float(quantity)
			if notional < float(Settings().min_notional_usd):
				raise RuntimeError(f"Notional mínimo {Settings().min_notional_usd} USD no alcanzado")
		if Settings().qty_step and float(Settings().qty_step) > 0:
			step = float(Settings().qty_step)
			if abs((float(quantity) / step) - round(float(quantity) / step)) > 1e-8:
				raise RuntimeError("Cantidad no múltiplo de qty_step")
		if Settings().price_tick and float(Settings().price_tick) > 0:
			tick = float(Settings().price_tick)
			if abs((float(price) / tick) - round(float(price) / tick)) > 1e-8:
				raise RuntimeError("Precio no alineado a price_tick")
		# Price collars para límites
		if Settings().price_collar_pct:
			try:
				from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
				ref = _LL().get_microprice(symbol)
				if ref and float(ref) > 0:
					pct = abs(float(price) / float(ref) - 1.0)
					if pct > float(Settings().price_collar_pct):
						raise RuntimeError("Collar de precio excedido para limit")
			except Exception:
				pass
		payload = {
			"symbol": symbol,
			"side": side,
			"quantity": float(quantity),
			"type_": "limit",
			"price": float(price),
			"tif": tif,
			"post_only": True,
			"client_order_id": client_order_id,
			"ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds"),
			"decision_id": f"did_{uuid.uuid4().hex[:16]}",
		}
		outbox_id = db.put_order_outbox(_json.dumps(payload, ensure_ascii=False), client_order_id, correlation_id)
		# Rate limiting por broker, estrategia y símbolo
		from nge_trader.services.rate_limiter import GlobalRateLimiter
		rl = GlobalRateLimiter.get()
		key_broker = f"orders_{s.broker}"
		key_strategy = f"orders_{s.broker}_strat_{s.strategy_id}"
		key_symbol = f"orders_{s.broker}_{symbol.upper()}"
		rl.configure(key_broker, capacity=int(s.broker_orders_per_minute), refill_per_sec=max(1.0, int(s.broker_orders_per_minute)) / 60.0)
		rl.configure(key_strategy, capacity=int(s.strategy_orders_per_minute), refill_per_sec=max(1.0, int(s.strategy_orders_per_minute)) / 60.0)
		rl.configure(key_symbol, capacity=int(s.symbol_orders_per_minute), refill_per_sec=max(1.0, int(s.symbol_orders_per_minute)) / 60.0)
		if not (rl.acquire(key_broker) and rl.acquire(key_strategy) and rl.acquire(key_symbol)):
			raise RuntimeError("Rate limit alcanzado (broker/estrategia/símbolo)")
		# Budget por estrategia (diario)
		try:
			sent_today = db.get_strategy_orders_sent_today(Settings().strategy_id)
		except Exception:
			sent_today = 0
		if sent_today >= int(Settings().max_trades_per_day_per_strategy or 50):
			raise RuntimeError("Presupuesto de órdenes diario por estrategia agotado")
		# Budgets por símbolo y cuenta
		try:
			sym_sent = db.get_symbol_orders_sent_today(symbol)
			if sym_sent >= int(Settings().max_trades_per_day_per_symbol or 20):
				raise RuntimeError("Presupuesto de órdenes diario por símbolo agotado")
		except Exception:
			pass
		try:
			acc_sent = db.get_account_orders_sent_today(Settings().account_id)
			if acc_sent >= int(Settings().max_trades_per_day_per_account or 100):
				raise RuntimeError("Presupuesto de órdenes diario por cuenta agotado")
		except Exception:
			pass
		# Gating por slippage promedio reciente
		try:
			from nge_trader.repository.db import Database as _DB
			vals = _DB().recent_metric_values("slippage_bps", 50)
			last_n = [v for _, v in vals[-20:]]
			if last_n:
				avg = float(sum(last_n) / len(last_n))
				if abs(avg) > float(Settings().max_slippage_bps or 50.0):
					raise RuntimeError("Breaker por slippage: umbral excedido")
		except Exception:
			pass
		try:
			gate.register_attempt()
			t0 = _dt.datetime.now(_dt.UTC).timestamp()
			try:
				res = broker.place_order(symbol=symbol, side=side, quantity=quantity, type_="limit", tif=tif, price=price, post_only=True, client_order_id=client_order_id)
			except TypeError:
				res = broker.place_order(symbol=symbol, side=side, quantity=quantity, type_="limit", tif=tif, price=price, post_only=True)
			broker_order_id = res.get("id") or res.get("order_id") or res.get("orderId")
			db.mark_order_outbox(outbox_id, status="sent", broker_order_id=str(broker_order_id) if broker_order_id else None)
			t1 = _dt.datetime.now(_dt.UTC).timestamp()
			set_metric("order_place_latency_ms", (t1 - t0) * 1000.0)
			inc_metric("orders_placed_total", 1.0)
			# Registrar MiFID IDs
			try:
				db.record_order({
					"ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds"),
					"symbol": symbol,
					"side": side,
					"qty": float(quantity),
					"price": float(price),
					"status": "sent",
					"order_id": str(broker_order_id) if broker_order_id else None,
					"decision_id": payload.get("decision_id"),
					"execution_id": f"eid_{uuid.uuid4().hex[:16]}",
					"dea": 1 if bool(Settings().dea_indicator) else 0,
				})
			except Exception:
				pass
			try:
				db.inc_strategy_orders_sent_today(Settings().strategy_id, 1)
				db.inc_symbol_orders_sent_today(symbol, 1)
				db.inc_account_orders_sent_today(Settings().account_id, 1)
			except Exception:
				pass
			return res
		except Exception as exc:  # noqa: BLE001
			db.mark_order_outbox(outbox_id, status="error", error=str(exc))
			try:
				gate.register_error()
			except Exception:
				pass
			inc_metric("orders_error_total", 1.0)
			try:
				inc_metric("breaker_activations_total", 1.0)
			except Exception:
				pass
			# Registrar orden rechazada en DB
			try:
				self_ts = _dt.datetime.now(_dt.UTC).isoformat()
				db.record_order({
					"ts": self_ts,
					"symbol": symbol,
					"side": side,
					"qty": float(quantity),
					"price": float(price),
					"status": "rejected",
					"order_id": None,
				})
			except Exception:
				pass
			raise
	return NotImplementedError("Broker no soporta place_order post-only")


def place_limit_order_marketable(
	broker: Any,
	symbol: str,
	side: str,
	quantity: float,
	price: float,
	tif: str = "GTC",
) -> Dict[str, Any]:
	"""Orden limit marketable (sin post_only) reutilizando validaciones de limit.

	Similar a place_limit_order_post_only pero permite ejecución inmediata.
	"""
	if hasattr(broker, "place_order"):
		s = Settings()
		if s.kill_switch_armed:
			raise RuntimeError("Kill-switch armado: bloqueando envíos")
		paused = {x.strip().upper() for x in (s.paused_symbols or "").split(",") if x}
		if symbol.upper() in paused:
			raise RuntimeError(f"Símbolo pausado: {symbol}")
		if s.enable_ws_backfill and not is_session_open(s.market_clock_exchange):
			raise RuntimeError("Mercado fuera de ventana de trading")
		if s.enable_retry_policy:
			skew = abs(estimate_skew_ms(server=s.ntp_server))
			if skew > float(s.max_time_skew_ms):
				raise RuntimeError(f"Skew de reloj excesivo: {skew} ms")
		gate = SLOGate()
		if not gate.can_send():
			raise RuntimeError("SLO gate activo: error-rate reciente excede umbral")
		client_order_id = generate_idempotency_key(symbol, side)
		db = Database()
		correlation_id = f"corr_{uuid.uuid4().hex[:12]}"
		# Señal microprice
		try:
			from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
			mp = _LL().get_microprice(symbol)
			if mp is not None:
				from nge_trader.services.metrics import set_metric
				set_metric("intended_microprice", float(mp))
		except Exception:
			pass
		# Validaciones instrumentales
		try:
			mp = None
			from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
			mp = _LL().get_microprice(symbol)
		except Exception:
			mp = None
		quantity, price = quantize_quantity_price(symbol, quantity, price)
		px_est = float(price or (mp if mp is not None else 0.0))
		if Settings().min_notional_usd and px_est:
			notional = float(px_est) * float(quantity)
			if notional < float(Settings().min_notional_usd):
				raise RuntimeError(f"Notional mínimo {Settings().min_notional_usd} USD no alcanzado")
		if Settings().qty_step and float(Settings().qty_step) > 0:
			step = float(Settings().qty_step)
			if abs((float(quantity) / step) - round(float(quantity) / step)) > 1e-8:
				raise RuntimeError("Cantidad no múltiplo de qty_step")
		# price_tick validación si aplica
		if Settings().price_tick and float(Settings().price_tick) > 0:
			tick = float(Settings().price_tick)
			if abs((float(price) / tick) - round(float(price) / tick)) > 1e-8:
				raise RuntimeError("Precio no alineado a price_tick")
		payload = {
			"symbol": symbol,
			"side": side,
			"quantity": float(quantity),
			"type_": "limit",
			"price": float(price),
			"tif": tif,
			"post_only": False,
			"client_order_id": client_order_id,
			"ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds"),
			"decision_id": f"did_{uuid.uuid4().hex[:16]}",
		}
		outbox_id = db.put_order_outbox(_json.dumps(payload, ensure_ascii=False), client_order_id, correlation_id)
		from nge_trader.services.rate_limiter import GlobalRateLimiter
		rl = GlobalRateLimiter.get()
		key_broker = f"orders_{s.broker}"
		key_strategy = f"orders_{s.broker}_strat_{s.strategy_id}"
		key_symbol = f"orders_{s.broker}_{symbol.upper()}"
		rl.configure(key_broker, capacity=int(s.broker_orders_per_minute), refill_per_sec=max(1.0, int(s.broker_orders_per_minute)) / 60.0)
		rl.configure(key_strategy, capacity=int(s.strategy_orders_per_minute), refill_per_sec=max(1.0, int(s.strategy_orders_per_minute)) / 60.0)
		rl.configure(key_symbol, capacity=int(s.symbol_orders_per_minute), refill_per_sec=max(1.0, int(s.symbol_orders_per_minute)) / 60.0)
		if not (rl.acquire(key_broker) and rl.acquire(key_strategy) and rl.acquire(key_symbol)):
			raise RuntimeError("Rate limit alcanzado (broker/estrategia/símbolo)")
		try:
			gate.register_attempt()
			t0 = _dt.datetime.now(_dt.UTC).timestamp()
			try:
				res = broker.place_order(symbol=symbol, side=side, quantity=quantity, type_="limit", tif=tif, price=price, post_only=False, client_order_id=client_order_id)
			except TypeError:
				res = broker.place_order(symbol=symbol, side=side, quantity=quantity, type_="limit", tif=tif, price=price, post_only=False)
			broker_order_id = res.get("id") or res.get("order_id") or res.get("orderId")
			db.mark_order_outbox(outbox_id, status="sent", broker_order_id=str(broker_order_id) if broker_order_id else None)
			t1 = _dt.datetime.now(_dt.UTC).timestamp()
			set_metric("order_place_latency_ms", (t1 - t0) * 1000.0)
			inc_metric("orders_placed_total", 1.0)
			return res
		except Exception as exc:  # noqa: BLE001
			db.mark_order_outbox(outbox_id, status="error", error=str(exc))
			raise
	return NotImplementedError("Broker no soporta place_order limit")

def place_oco_order_generic(
	broker: Any,
	symbol: str,
	side: str,
	quantity: float,
	limit_price: float,
	stop_price: float,
	stop_limit_price: Optional[float] = None,
	tif: str = "GTC",
) -> Dict[str, Any]:
	"""Intenta usar OCO nativo; si no existe, crea dos órdenes y devuelve ids.

	Nota: En modo genérico no se cancela automáticamente el par; debe usarse
	`cancel_remaining_of_oco` al detectar un fill por fuera (por WS o polling).
	"""
	if hasattr(broker, "place_oco_order"):
		return broker.place_oco_order(
			symbol=symbol,
			side=side,
			quantity=quantity,
			price=limit_price,
			stop_price=stop_price,
			stop_limit_price=stop_limit_price,
			tif=tif,
		)
	# Fallback genérico: limit + stop-limit (se linkean como OCO)
	placed: List[Dict[str, Any]] = []
	# Orden límite
	o1 = broker.place_order(symbol=symbol, side=side, quantity=quantity, type_="limit", price=limit_price, tif=tif)
	placed.append(o1)
	# Orden stop-limit (si broker no soporta, usar stop como limit de peor precio)
	sl_price = stop_limit_price if stop_limit_price is not None else stop_price
	o2 = broker.place_order(symbol=symbol, side=side, quantity=quantity, type_="limit", price=sl_price, tif=tif)
	placed.append(o2)
	# Linkear parent/child en DB si hay ids (OCO)
	try:
		db = Database()
		o1_id = o1.get("id") or o1.get("order_id") or o1.get("orderId")
		o2_id = o2.get("id") or o2.get("order_id") or o2.get("orderId")
		if o1_id and o2_id:
			db.link_orders(str(o1_id), str(o2_id), "OCO")
	except Exception:
		pass
	return {"status": "created_pair", "orders": placed}


def cancel_all_orders_by_symbol(broker: Any, symbol: str) -> Dict[str, Any]:
	"""Cancela todas las órdenes abiertas de un símbolo usando capacidades nativas.

	- Binance: requiere `symbol` obligatorio
	- Coinbase: acepta `product_id`
	- IBKR: itera openOrders y cancela las que coincidan
	- Paper: filtra por `symbol`
	"""
	if hasattr(broker, "cancel_all_orders"):
		try:
			# Binance
			return broker.cancel_all_orders(symbol=symbol)
		except TypeError:
			try:
				# Coinbase
				return broker.cancel_all_orders(product_id=symbol)
			except TypeError:
				# Paper/otros sin parámetro de símbolo
				return broker.cancel_all_orders()
	# IBKR u otros sin cancel_all_orders; intentar manualmente
	if hasattr(broker, "list_orders") and hasattr(broker, "cancel_order"):
		canceled = 0
		for o in broker.list_orders(status="open", limit=500):
			if (o.get("symbol") or o.get("product_id")) == symbol:
				oid = o.get("id") or o.get("order_id") or o.get("orderId")
				if oid is not None:
					try:
						broker.cancel_order(oid)
						canceled += 1
					except Exception:
						pass
		return {"status": "ok", "canceled": canceled}
	raise NotImplementedError("Broker no soporta cancelaciones por símbolo")


def generate_idempotency_key(symbol: str, side: str) -> str:
	"""Genera una key idempotente por envío.

	Algunos brokers aceptan `client_order_id` para evitar duplicados. Usamos un UUID4 con prefijo simbólico.
	"""
	base = f"{symbol.upper()}_{side.lower()}"
	return f"{base}_{uuid.uuid4().hex[:12]}"


def place_market_order_sor(brokers: List[Any], symbol: str, side: str, quantity: float) -> Dict[str, Any]:
	"""SOR básico: intenta en orden la lista de brokers hasta completar un envío satisfactorio.

	- Usa `place_market_order` existente para aplicar gating/validaciones.
	- Devuelve el primer resultado válido; si todos fallan, propaga la última excepción.
	"""
	last_exc: Exception | None = None
	for br in brokers:
		try:
			return place_market_order(br, symbol, side, quantity)
		except Exception as exc:  # noqa: BLE001
			last_exc = exc
			continue
	if last_exc:
		raise last_exc
	return NotImplementedError("SOR: no se pudo ejecutar en ningún broker")


def place_market_order_sor_advanced(
    candidates: List[Dict[str, Any]],
    symbol: str,
    side: str,
    quantity: float,
) -> Dict[str, Any]:
    """SOR avanzado: selecciona broker por score (latencia estimada) y hace failover.

    candidates: lista de { broker, name, latency_ms (opcional) }
    - Ordena por menor latency_ms (fallback: +inf) y prueba en ese orden.
    - Registra métricas de selección y fallos por broker.
    """
    # Orden por latencia estimada
    ordered = sorted(
        list(candidates), key=lambda c: float(c.get("latency_ms") or float("inf"))
    )
    last_exc: Exception | None = None
    for c in ordered:
        br = c.get("broker")
        name = str(c.get("name") or getattr(br, "__class__", type("B", (), {})).__name__)
        try:
            res = place_market_order(br, symbol, side, quantity)
            try:
                set_metric_labeled(
                    "sor_selection_total",
                    1.0,
                    {"broker": name, "symbol": symbol.upper(), "strategy": Settings().strategy_id},
                )
            except Exception:
                pass
            return res
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            try:
                inc_metric_labeled(
                    "sor_failover_total",
                    1.0,
                    {"broker": name, "symbol": symbol.upper(), "strategy": Settings().strategy_id, "error_class": type(exc).__name__},
                )
            except Exception:
                pass
            continue
    if last_exc:
        raise last_exc
    return NotImplementedError("SOR-advanced: no se pudo ejecutar en ningún broker")

def cancel_order_idempotent(broker: Any, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
	"""Cancela una orden con registro en outbox e idempotencia básica."""
	db = Database()
	payload = {
		"action": "cancel",
		"order_id": order_id,
		"symbol": symbol,
		"ts": _dt.datetime.now(_dt.UTC).isoformat(),
	}
	idk = f"cancel_{order_id}"
	outbox_id = db.put_order_outbox(_json.dumps(payload, ensure_ascii=False), idk, f"corr_{uuid.uuid4().hex[:12]}")
	try:
		if hasattr(broker, "cancel_order"):
			res = broker.cancel_order(order_id)
		else:
			raise NotImplementedError("Broker no soporta cancel_order")
		db.mark_order_outbox(outbox_id, status="sent")
		try:
			db.update_order_status_price(order_id, status="canceled")
		except Exception:
			pass
		return res if isinstance(res, dict) else {"status": "ok", "order_id": order_id}
	except Exception as exc:  # noqa: BLE001
		db.mark_order_outbox(outbox_id, status="error", error=str(exc))
		raise


def replace_order_limit_price_idempotent(broker: Any, order_id: str, new_price: float, symbol: Optional[str] = None) -> Dict[str, Any]:
	"""Reemplaza el precio límite de una orden con registro en outbox e idempotencia básica.

	Si el broker no soporta replace, intenta cancel + nueva limit con post-only usando el símbolo.
	"""
	db = Database()
	payload = {
		"action": "replace",
		"order_id": order_id,
		"new_price": float(new_price),
		"symbol": symbol,
		"ts": _dt.datetime.now(_dt.UTC).isoformat(),
	}
	idk = f"replace_{order_id}_{int(float(new_price)*100)}"
	outbox_id = db.put_order_outbox(_json.dumps(payload, ensure_ascii=False), idk, f"corr_{uuid.uuid4().hex[:12]}")
	try:
		if hasattr(broker, "replace_order"):
			res = broker.replace_order(order_id=order_id, new_price=float(new_price))  # type: ignore[call-arg]
		else:
			if not symbol:
				raise NotImplementedError("Broker sin replace; se requiere symbol para fallback cancel+recreate")
			_ = broker.cancel_order(order_id)
			res = broker.place_order(symbol=symbol, side="buy", quantity=0.0, type_="limit", price=float(new_price), tif="GTC", post_only=True)  # type: ignore[call-arg]
		db.mark_order_outbox(outbox_id, status="sent")
		try:
			db.update_order_status_price(order_id, status="replaced", price=float(new_price))
		except Exception:
			pass
		return res if isinstance(res, dict) else {"status": "ok", "order_id": order_id, "new_price": float(new_price)}
	except Exception as exc:  # noqa: BLE001
		db.mark_order_outbox(outbox_id, status="error", error=str(exc))
		raise


