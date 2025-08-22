from __future__ import annotations

import threading
import time
from dataclasses import dataclass
import logging
from collections import deque
import json
from typing import List

import pandas as pd

from nge_trader.services.app_service import AppService
from nge_trader.services.notifier import Notifier
from nge_trader.domain.strategies.moving_average import MovingAverageCrossStrategy
from nge_trader.domain.strategies.rsi import RSIStrategy
from nge_trader.domain.risk.atr import compute_atr, position_size_by_risk
from nge_trader.repository.db import Database
from nge_trader.config.settings import Settings
from nge_trader.adapters.alpaca_ws import AlpacaTradeWS
from nge_trader.adapters.binance_market_ws import BinanceMarketWS
try:
    from nge_trader.adapters.coinbase_ws import CoinbaseUserWS
except Exception:  # noqa: BLE001
    CoinbaseUserWS = None  # type: ignore[assignment]
from nge_trader.ai.policy import AgentPolicy, PolicyConfig  # noqa: F401
from nge_trader.ai.online import OnlineLinearPolicy, OnlineConfig
from nge_trader.ai.replay import ReplayBuffer
from nge_trader.repository.db import Database  # noqa: F811
from nge_trader.services.risk import volatility_target_position, compute_var_es
import numpy as np
from nge_trader.ai.regime import RegimeDetector
from nge_trader.adapters.lowlat_provider import LowLatencyProvider
from nge_trader.services.metrics import set_metric
from nge_trader.services.infra_monitor import InfraMonitor
from nge_trader.services.data_redundancy import RedundantDataProvider
from nge_trader.services.time_sync import estimate_skew_ms
from nge_trader.services.metrics import set_metric
from uuid import uuid4
from nge_trader.services.model_session import get as get_model_session
from nge_trader.services.risk_store import get as risk_get

logger = logging.getLogger(__name__)


def _load_active_model(registry, champion_id: str):  # noqa: ANN001
	ms = get_model_session()
	model_id = ms.get("model_id") if ms.get("pinned") else champion_id
	# Carga mínima: si no hay registry.load disponible, usa política online como modelo
	try:
		model = registry.load(model_id)  # type: ignore[attr-defined]
	except Exception:
		# Fallback: el modelo activo será provisto por LiveEngine.online_policy
		model = None
	return model, model_id, bool(ms.get("pinned"))


@dataclass
class LiveConfig:
    symbols: List[str]
    strategy: str
    capital_per_trade: float = 1000.0
    poll_seconds: int = 30


class LiveEngine:
    def __init__(self, cfg: LiveConfig) -> None:
        self.cfg = cfg
        self._stop = threading.Event()
        self.service = AppService()
        self.notifier = Notifier()
        self.db = Database()
        self.settings = Settings()
        self._equity_open = None
        self._last_sent_ts: dict[str, float] = {}
        self._trades_today: int = 0
        # Online policy y replay buffer
        self.online_policy = OnlineLinearPolicy(OnlineConfig(
            lookback=int(self.settings.online_lookback or 30),
            learning_rate=float(self.settings.online_learning_rate or 0.01),
            l2_reg=float(self.settings.online_l2_reg or 1e-4),
        ))
        self.replay = ReplayBuffer()
        self._baseline_states: list[list[float]] = []
        self._live_states_window: list[list[float]] = []
        # Métricas de robustez
        self._err_events: deque[float] = deque(maxlen=1000)
        self._attempt_events: deque[float] = deque(maxlen=1000)
        self._slippages_bps: deque[float] = deque(maxlen=1000)
        self._last_intended_price: dict[str, float] = {}
        self._last_mkt_price: dict[str, float] = {}
        self._mkt_events: deque[tuple[float, float, float]] = deque(maxlen=2000)  # (ts, price, qty)
        self._regime = RegimeDetector(lookback=60, k=2)
        self.lowlat = LowLatencyProvider(max_events=5000)
        # Métricas de entrenamiento online
        self._train_updates_window: list[float] = []
        self._train_losses_window: list[float] = []
        self._train_action_hist: list[float] = []
        self._online_pause_until_ts: float = 0.0
        # Baseline estático para validación anti-forgetting (PSI)
        self._baseline_static_states = None
        try:
            import numpy as _np
            from pathlib import Path as _Path
            baseline_path = _Path("data/baseline_states.npy")
            if baseline_path.exists():
                self._baseline_static_states = _np.load(str(baseline_path))
            else:
                # Generar baseline a partir del primer símbolo si disponible
                if self.cfg.symbols:
                    df_base = self.service.data_provider.get_daily_adjusted(self.cfg.symbols[0])
                    X, _ = self.online_policy._features(df_base)  # type: ignore[attr-defined]
                    if X.size > 0:
                        baseline_path.parent.mkdir(parents=True, exist_ok=True)
                        _np.save(str(baseline_path), X)
                        self._baseline_static_states = X
        except Exception:
            self._baseline_static_states = None
        # Warm start (transfer learning) si está configurado
        try:
            tf = (self.settings.transfer_from_symbols or "").strip()
            if tf:
                symbols_src = [s.strip() for s in tf.split(",") if s.strip()]
                if symbols_src:
                    dfs = []
                    for sym in symbols_src:
                        try:
                            dfi = self.service.data_provider.get_daily_adjusted(sym)
                            if not dfi.empty:
                                dfs.append(dfi)
                        except Exception:
                            continue
                    if dfs:
                        self.online_policy.warm_start_from_dfs(dfs, l2=float(self.settings.online_l2_reg or 1e-4))
        except Exception:
            pass
        # Iniciar escritor TCA (no bloqueante)
        try:
            from nge_trader.services import tca_writer as _tca
            _tca.start()
        except Exception:
            pass
        # Proveedor redundante de precios: low-lat microprice + daily close como fallback
        class _LLPProvider:
            def __init__(self, llp: LowLatencyProvider) -> None:
                self.llp = llp
            def get_price(self, symbol: str) -> float | None:  # noqa: ANN001
                try:
                    v = self.llp.get_microprice(symbol)
                    return float(v) if v else None
                except Exception:
                    return None
        class _DailyCloseProvider:
            def __init__(self, service: AppService) -> None:
                self.service = service
            def get_price(self, symbol: str) -> float | None:  # noqa: ANN001
                try:
                    df = self.service.data_provider.get_daily_adjusted(symbol)
                    if df.empty:
                        return None
                    return float(df["close"].astype(float).iloc[-1])
                except Exception:
                    return None
        self.price_provider = RedundantDataProvider([
            _LLPProvider(self.lowlat),
            _DailyCloseProvider(self.service),
        ], divergence_threshold_pct=0.01)
        # Cargar asignaciones si hay
        try:
            from nge_trader.services.strategy_store import StrategyStore

            self.assignments = {a.key: a.params.get("strategy") for a in StrategyStore().load_assignments()}
        except Exception:
            self.assignments = {}
        # WS opcional
        self._ws: AlpacaTradeWS | None = None
        self._mws: BinanceMarketWS | None = None
        # Hilo de backfill periódico
        self._bf_thread: threading.Thread | None = None
        self._bf_stop = threading.Event()
        # Monitor infra
        try:
            if bool(self.settings.infra_monitor_enabled):
                self._infra = InfraMonitor(self.settings.infra_probe_url or None, interval_sec=15)
                self._infra.start()
        except Exception:
            self._infra = None

    def stop(self) -> None:
        self._stop.set()
        try:
            self._bf_stop.set()
            if self._bf_thread and self._bf_thread.is_alive():
                self._bf_thread.join(timeout=5)
        except Exception:
            pass

    def _build_strategy(self):
        if self.cfg.strategy == "ma_cross":
            return MovingAverageCrossStrategy(10, 20)
        if self.cfg.strategy == "rsi":
            return RSIStrategy(14, 30.0, 70.0)
        if self.cfg.strategy == "agent":
            # Usar política online por defecto; AgentPolicy se mantiene para batch training/manual
            return self.online_policy
        return None

    def run(self) -> None:
        strat = self._build_strategy()
        if strat is None:
            self.notifier.send("Estrategia no válida para live trading")
            return
        self.notifier.send(f"Live Trading iniciado: {self.cfg.symbols} con {self.cfg.strategy}")
        # Rehydrate inicial: órdenes, posiciones y balances (medir duración)
        import time as _time
        from nge_trader.services.metrics import set_rehydrate_seconds
        t0 = _time.perf_counter()
        try:
            rec_o = self.service.reconcile_state(resolve=True)
            rec_pb = self.service.reconcile_positions_balances(resolve=True)
            set_metric("reconcile_missing_in_db", float(len(rec_o.get("missing_in_db", []))))
            set_metric("reconcile_missing_in_broker", float(len(rec_o.get("missing_in_broker", []))))
            set_metric("positions_snapshot_count", float(len(rec_pb.get("positions", []))))
            set_metric("balances_snapshot_count", float(len(rec_pb.get("balances", []))))
        except Exception:
            pass
        finally:
            try:
                set_rehydrate_seconds(max(_time.perf_counter() - t0, 0.0))
            except Exception:
                pass
        # Monitor de skew NTP periódico básico
        try:
            skew = float(abs(estimate_skew_ms(server=self.settings.ntp_server)))
            set_metric("time_skew_ms", skew)
            if skew > float(self.settings.max_time_skew_ms):
                self.notifier.send_alert_if_threshold("time_skew_ms", skew, float(self.settings.max_time_skew_ms), "WARNING", f"Skew reloj alto: {skew}ms > {self.settings.max_time_skew_ms}")
        except Exception:
            pass
        # Iniciar WS si aplica
        try:
            if hasattr(self.service.broker, "get_account") and self.settings.alpaca_api_key and self.settings.alpaca_api_secret:
                def on_ws_message(msg: dict) -> None:
                    # Manejo básico: registrar fills si vienen por WS
                    try:
                        if isinstance(msg, dict) and msg.get("stream") == "trade_updates":
                            data = msg.get("data", {})
                            if data.get("event") == "fill":
                                self.db.record_fill({
                                    "ts": data.get("timestamp") or pd.Timestamp.now(tz=pd.Timestamp.utcnow().tz).isoformat(),
                                    "symbol": data.get("order", {}).get("symbol"),
                                    "side": (data.get("order", {}).get("side") or "").lower(),
                                    "qty": float(data.get("order", {}).get("filled_qty") or 0),
                                    "price": float(data.get("price") or 0),
                                    "order_id": data.get("order", {}).get("id"),
                                })
                    except Exception:
                        pass
                from nge_trader.services.metrics import update_ws_state
                self._ws = AlpacaTradeWS(self.settings.alpaca_api_key, self.settings.alpaca_api_secret, on_ws_message)
                def _alp_run():
                    update_ws_state("alpaca", True)
                    try:
                self._ws.start()
                    finally:
                        update_ws_state("alpaca", False)
                threading.Thread(target=_alp_run, daemon=True).start()
        except Exception:
            self._ws = None
        # Coinbase user WS si broker es coinbase
        try:
            if self.settings.broker == "coinbase" and CoinbaseUserWS and (self.settings.coinbase_api_key and self.settings.coinbase_api_secret and self.settings.coinbase_passphrase):
                def _cb_cb(msg: dict) -> None:
                    try:
                        if not isinstance(msg, dict):
                            return
                        mtype = str(msg.get("type") or "").lower()
                        ts_now = pd.Timestamp.now(tz=pd.Timestamp.utcnow().tz).isoformat()
                        self.db.append_audit(ts_now, "coinbase_ws", str(msg.get("order_id") or ""), str(msg.get("product_id") or ""), json.dumps(msg))
                        # received → registrar orden
                        if mtype == "received":
                            rec = {
                                "ts": ts_now,
                                "symbol": msg.get("product_id"),
                                "side": msg.get("side"),
                                "qty": float(msg.get("size") or 0),
                                "price": float(msg.get("price") or 0) if msg.get("price") else None,
                                "status": "new",
                                "order_id": msg.get("order_id"),
                            }
                            self.db.record_order(rec)
                        # match (partial/full fill)
                        elif mtype == "match":
                            nf = {
                                "ts": ts_now,
                                "symbol": msg.get("product_id"),
                                "side": msg.get("side"),
                                "qty": float(msg.get("size") or 0),
                                "price": float(msg.get("price") or 0),
                                "order_id": msg.get("order_id"),
                                "liquidity": msg.get("liquidity"),
                                "exchange": "COINBASE",
                            }
                            self.db.record_fill(nf)
                            # Partial fill → status partial; si tamaño coincide con qty total, status filled (simplificado)
                            self.db.update_order_status_price(str(msg.get("order_id") or ""), status="partial")
                        # done (canceled/filled)
                        elif mtype == "done":
                            reason = str(msg.get("reason") or "").lower()
                            new_status = "canceled" if reason == "canceled" else "filled"
                            self.db.update_order_status_price(str(msg.get("order_id") or ""), status=new_status)
                        # change (replace)
                        elif mtype == "change":
                            try:
                                new_px = float(msg.get("new_price")) if msg.get("new_price") is not None else None
                            except Exception:
                                new_px = None
                            self.db.update_order_status_price(str(msg.get("order_id") or ""), status="replaced", price=new_px)
                    except Exception:
                        pass
                def _on_reconnect() -> None:
                    try:
                        for sym in (self.cfg.symbols or []):
                            self.service.backfill_for_symbol(sym, limit=50)
                    except Exception:
                        pass
                self._cws = CoinbaseUserWS(self.settings.coinbase_api_key, self.settings.coinbase_api_secret, self.settings.coinbase_passphrase)  # type: ignore[call-arg]
                self._cws.start(_cb_cb, on_reconnect=_on_reconnect)  # type: ignore[union-attr]
        except Exception:
            try:
                self._cws.stop()  # type: ignore[union-attr]
            except Exception:
                pass
        # WS de mercado para primer símbolo si Binance
        try:
            if self.settings.broker == "binance" and self.cfg.symbols:
                sym0 = self.cfg.symbols[0]
                from nge_trader.services.metrics import update_ws_state
                self._mws = BinanceMarketWS(sym0, testnet=(self.settings.profile.lower()=="dev"), stream="aggTrade")
                def on_mkt(evt: dict) -> None:
                    # medir latencia y clock skew
                    try:
                        et = int(evt.get("event_time") or 0)
                        now_ms = int(pd.Timestamp.now(tz=pd.Timestamp.utcnow().tz).timestamp() * 1000)
                        skew_ms = now_ms - et
                        self.db.append_log_json("INFO", {"metric": "market_ws_skew_ms", "value": skew_ms}, pd.Timestamp.now(tz=pd.Timestamp.utcnow().tz).isoformat())
                        try:
                            from nge_trader.services.metrics import set_metric
                            set_metric("ws_skew_ms", float(skew_ms))
                        except Exception:
                            pass
                        sym = (evt.get("symbol") or sym0).upper()
                        if evt.get("type") == "book":
                            bid = float(evt.get("bid") or 0.0)
                            ask = float(evt.get("ask") or 0.0)
                            bs = evt.get("bid_size")
                            asz = evt.get("ask_size")
                            from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
                            self.lowlat.push_quote(sym, bid=bid, ask=ask, bid_size=bs, ask_size=asz, ts=float(now_ms) / 1000.0)
                        else:
                        px = float(evt.get("price") or 0.0)
                        if px > 0:
                            self._last_mkt_price[sym] = px
                            qty = float(evt.get("qty") or 0.0)
                            self._mkt_events.append((float(now_ms) / 1000.0, px, qty))
                                self.lowlat.push_event(sym, price=px, qty=qty, sequence_id=int(evt.get("sequence_id") or 0) or None, ts=float(now_ms) / 1000.0)
                    except Exception:
                        pass
                def _bm_run():
                    update_ws_state("binance_market", True)
                    try:
                self._mws.start(on_mkt)
                    finally:
                        update_ws_state("binance_market", False)
                threading.Thread(target=_bm_run, daemon=True).start()
        except Exception:
            self._mws = None

        # L2 externo opcional (Kaiko)
        try:
            if str(self.settings.l2_provider).lower() == "kaiko" and self.cfg.symbols:
                from nge_trader.adapters.kaiko_l2 import KaikoL2WS
                from nge_trader.services.metrics import update_ws_state
                self._kws = KaikoL2WS(self.settings.kaiko_api_key)
                sym0 = self.cfg.symbols[0]
                def _on_depth(evt: dict) -> None:
                    try:
                        if not isinstance(evt, dict):
                            return
                        sym = (evt.get("symbol") or sym0).upper()
                        bids = evt.get("bids") or []
                        asks = evt.get("asks") or []
                        ts = float(evt.get("event_time") or 0.0)
                        self.lowlat.push_depth(sym, bids=bids, asks=asks, ts=ts)
                    except Exception:
                        pass
                def _run_k():
                    update_ws_state("kaiko_l2", True)
                    try:
                        self._kws.start(sym0, _on_depth)
                    finally:
                        update_ws_state("kaiko_l2", False)
                threading.Thread(target=_run_k, daemon=True).start()
        except Exception:
            try:
                if getattr(self, "_kws", None):
                    self._kws.stop()
            except Exception:
                pass

        # Backfill en reconexiones del broker subyacente si soporta
        try:
            underlying = getattr(self.service.broker, "_primary", self.service.broker)
            def _rehydrate_symbol(sym: str) -> None:
                try:
                    if hasattr(underlying, "backfill_recent"):
                        out = underlying.backfill_recent(sym, limit=100) if self.settings.broker != "ibkr" else underlying.backfill_recent(100)
                        from nge_trader.repository.db import Database as _DB
                        from nge_trader.services.oms import normalize_order, normalize_fill
                        dbi = _DB()
                        for o in out.get("orders", [])[:100]:
                            try:
                                rec = normalize_order(o)
                                if not dbi.order_exists(rec.get("order_id")):
                                    rec.setdefault("ts", pd.Timestamp.now(tz=pd.Timestamp.utcnow().tz).isoformat())
                                    dbi.record_order(rec)
                            except Exception:
                                continue
                        for f in out.get("fills", [])[:100]:
                            try:
                                nf = normalize_fill(f)
                                if not dbi.fill_exists(nf.get("order_id"), nf.get("price"), nf.get("qty")):
                                    nf.setdefault("ts", pd.Timestamp.now(tz=pd.Timestamp.utcnow().tz).isoformat())
                                    dbi.record_fill(nf)
                            except Exception:
                                continue
                except Exception:
                    pass
            # Ejecutar al inicio para primer símbolo
            if self.cfg.symbols:
                _rehydrate_symbol(self.cfg.symbols[0])
            # Lanzar backfill periódico para todos los símbolos si el adaptador lo soporta
            def _backfill_worker() -> None:
                import time as _time
                while not self._bf_stop.is_set():
                    try:
                        for sym in list(self.cfg.symbols):
                            _rehydrate_symbol(sym)
                    except Exception:
                        pass
                    _time.sleep(max(int(self.settings.cache_ttl_seconds or 60) // 6, 10))
            if hasattr(underlying, "backfill_recent") and not self._bf_thread:
                self._bf_stop.clear()
                self._bf_thread = threading.Thread(target=_backfill_worker, daemon=True)
                self._bf_thread.start()
        except Exception:
            pass

        while not self._stop.is_set():
            try:
                # HA lease: si está habilitado, solo el primario ejecuta loop
                try:
                    if bool(self.settings.ha_enabled):
                        import json as _json
                        from pathlib import Path as _P
                        lease_path = _P(str(self.settings.ha_lease_file))
                        now = pd.Timestamp.utcnow().timestamp()
                        d = {}
                        if lease_path.exists():
                            try:
                                d = _json.loads(lease_path.read_text(encoding="utf-8") or "{}")
                            except Exception:
                                d = {}
                        holder = str(d.get("holder") or "")
                        exp = float(d.get("expires_at") or 0.0)
                        if holder and holder != str(self.settings.cluster_node_id) and now < exp:
                            time.sleep(self.cfg.poll_seconds)
                            continue
                        # Renovar lease
                        new = {"holder": str(self.settings.cluster_node_id), "expires_at": now + float(self.settings.ha_lease_ttl_seconds or 30)}
                        lease_path.parent.mkdir(parents=True, exist_ok=True)
                        lease_path.write_text(_json.dumps(new), encoding="utf-8")
                except Exception:
                    pass
                # reset diario simple de conteo de trades
                try:
                    now = pd.Timestamp.utcnow()
                    if now.hour == 0 and now.minute < 2:
                        self._trades_today = 0
                except Exception:
                    pass
                # Circuit breaker diario
                try:
                    acc = self.service.broker.get_account() if hasattr(self.service.broker, "get_account") else None
                    if acc and self._equity_open is None:
                        self._equity_open = float(acc.get("last_equity") or acc.get("equity") or 0.0)
                    if acc and self._equity_open is not None:
                        current = float(acc.get("equity") or self._equity_open)
                        dd = (current - self._equity_open) / max(self._equity_open, 1e-9)
                        if dd <= -abs(self.settings.max_daily_drawdown_pct):
                            self.notifier.send(f"Circuit breaker: DD diario {dd:.2%} <= {self.settings.max_daily_drawdown_pct:.2%}. Parando.")
                            break
                        # semanal: usar last_equity_week si existe, si no aproximar con last_equity
                        last_week = float(acc.get("last_equity") or self._equity_open)
                        ddw = (current - last_week) / max(last_week, 1e-9)
                        if ddw <= -abs(self.settings.max_weekly_drawdown_pct):
                            self.notifier.send(f"Circuit breaker: DD semanal {ddw:.2%} <= {self.settings.max_weekly_drawdown_pct:.2%}. Parando.")
                            break
                except Exception:
                    pass
                for symbol in self.cfg.symbols:
                    # Pausa por símbolo
                    try:
                        if self.settings.paused_symbols and symbol.upper() in [s.strip().upper() for s in str(self.settings.paused_symbols).split(',') if s.strip()]:
                            continue
                    except Exception:
                        pass
                    df = self.service.data_provider.get_daily_adjusted(symbol)
                    # Kill-switch manual
                    if self.settings.kill_switch_armed:
                        self.notifier.send("Kill-switch activo. Omitiendo envíos de órdenes.")
                        continue
                    # Compliance: bloquear si requiere validación de algoritmo y no está validado
                    try:
                        if bool(self.settings.algo_validation_required):
                            if not Database().is_algo_validated(self.settings.algo_name, self.settings.algo_version):
                                self.notifier.send("Compliance: algoritmo no validado. Bloqueando envíos.")
                                continue
                    except Exception:
                        pass
                    # Kill-switch automático por anomalías (error-rate/slippage/latencia)
                    try:
                        from nge_trader.services import metrics as _MM
                        err_rate = float(_MM._METRICS_STORE.get("slo_error_rate_recent", 0.0))
                        slip_last = float(_MM._METRICS_STORE.get("slippage_last_bps", 0.0))
                        lat_p95 = float(_MM._METRICS_STORE.get("order_place_latency_ms_p95", 0.0))
                        if err_rate > float(self.settings.max_error_rate) or abs(slip_last) > float(self.settings.max_slippage_bps or 50.0):
                            self.settings.kill_switch_armed = True  # type: ignore[attr-defined]
                            self.notifier.send(f"Kill-switch AUTO activado. err_rate={err_rate:.2f}, slippage_bps={slip_last:.2f}, p95={lat_p95:.1f}ms")
                            continue
                    except Exception:
                        pass
                    # Ventana de trading (UTC)
                    try:
                        hhmm_s = self.settings.trading_hours_start
                        hhmm_e = self.settings.trading_hours_end
                        now = pd.Timestamp.utcnow()
                        start_t = pd.Timestamp(now.date().isoformat() + "T" + hhmm_s + ":00Z")
                        end_t = pd.Timestamp(now.date().isoformat() + "T" + hhmm_e + ":00Z")
                        if not (start_t <= now <= end_t):
                            continue
                    except Exception:
                        pass
                    # Estrategia por asignación si existe
                    local_strat = strat
                    try:
                        if symbol in self.assignments:
                            key = self.assignments[symbol]
                            if key == "ma_cross":
                                local_strat = MovingAverageCrossStrategy(10, 20)
                            elif key == "rsi":
                                local_strat = RSIStrategy(14, 30.0, 70.0)
                    except Exception:
                        pass
                    # Detección de régimen simple y router: momentum vs mean-reversion
                    try:
                        close = df["close"].astype(float)
                        rets = close.pct_change().dropna()
                        vol = float(rets.rolling(30).std(ddof=0).iloc[-1]) if len(rets) > 30 else 0.0
                        mom = float((close.iloc[-1] / close.iloc[-20] - 1.0)) if len(close) > 20 else 0.0
                        # Regla simple: si momentum fuerte y volatilidad moderada -> momentum; si bajo mom y baja vol -> mean-reversion (RSI)
                        if abs(mom) > 0.03 and vol < 0.05:
                            from nge_trader.domain.strategies.momentum import MomentumStrategy
                            local_strat = MomentumStrategy()
                        elif abs(mom) < 0.01 and vol < 0.03:
                            from nge_trader.domain.strategies.rsi import RSIStrategy as _RSI
                            local_strat = _RSI(14, 30.0, 70.0)
                    except Exception:
                        pass
                    # Vigilancia anti-abusos básica: posible layering/spoofing -> muchas órdenes limit canceladas rápidamente
                    try:
                        recent = self.db.recent_orders(200)
                        cancels = [o for o in recent if (o.get("symbol") or "").upper() == symbol.upper() and str(o.get("status") or "").lower() == "canceled"]
                        sent = [o for o in recent if (o.get("symbol") or "").upper() == symbol.upper() and str(o.get("status") or "").lower() in ("sent","new","open")]
                        ratio = (len(cancels) / max(len(sent), 1)) if sent else 0.0
                        if ratio > 2.0 and len(cancels) > 10:
                            # Alertar y activar kill-switch preventivo
                            self.settings.kill_switch_armed = True  # type: ignore[attr-defined]
                            self.notifier.send(f"Alerta compliance: patrón cancelaciones excesivas (ratio {ratio:.2f}). Activado kill-switch.")
                            continue
                    except Exception:
                        pass
                    # Precio redundante para métricas/decisiones auxiliares
                    try:
                        bp = self.price_provider.get_price(symbol)
                        if bp is not None:
                            set_metric("best_price", float(bp))
                    except Exception:
                        pass
                    if hasattr(local_strat, "generate_signals"):
                        signal = local_strat.generate_signals(df)
                        last_sig = float(signal.iloc[-1]) if not signal.empty else 0.0
                    else:
                        # Política de IA con soporte de modelo pinned
                        state_vec = self.online_policy.state_from_df(df)
                        # Intento de cargar modelo activo
                        try:
                            from nge_trader.services.model_registry import ModelRegistry as _MR
                            registry = _MR()
                            champion_id = getattr(self.settings, "champion_model_id", "champion")
                            model, model_id_used, is_pinned = _load_active_model(registry, champion_id)
                            if model and hasattr(model, "predict"):
                                ret_pred = float(model.predict(state_vec))  # type: ignore[arg-type]
                                last_sig = float(np.sign(ret_pred))
                            else:
                        last_sig = float(self.online_policy.predict_action(state_vec))
                        except Exception:
                            model, model_id_used, is_pinned = None, getattr(self.settings, "champion_model_id", "champion"), False
                            last_sig = float(self.online_policy.predict_action(state_vec))
                        try:
                            # Corte de entrenamiento: si pinned o disable global
                            if ("is_pinned" in locals() and is_pinned) or (not bool(getattr(self.settings, "enable_online_learning", True))):
                                pass
                            elif bool(self.settings.online_training_enabled):
                                import time as _t
                                now_ts = _t.time()
                                if not (self._online_pause_until_ts and now_ts < self._online_pause_until_ts):
                                    from nge_trader.services.rate_limiter import GlobalRateLimiter as _GRL
                                    rl = _GRL.get()
                                    rl_key = "online_updates"
                                    rl.configure(rl_key, capacity=int(self.settings.online_updates_per_minute), refill_per_sec=max(1.0, int(self.settings.online_updates_per_minute))/60.0)
                                    if rl.acquire(rl_key):
                                        batch = self.replay.sample(limit=64)
                                        for r in batch:
                                            try:
                                                loss = self.online_policy.update(
                                                    state=self.online_policy.state_from_df(pd.DataFrame(r.get("state") or {})) if isinstance(r.get("state"), dict) else r.get("state"),
                                                    reward=float(r.get("reward") or 0.0),
                                                )
                                                try:
                                                    self._train_updates_window.append(1.0)
                                                    if isinstance(loss, (float, int)):
                                                        self._train_losses_window.append(float(loss))
                                                except Exception:
                                                    pass
                                            except Exception:
                                                continue
                                        maxw = int(self.settings.training_metrics_window or 500)
                                        if len(self._train_updates_window) > maxw:
                                            self._train_updates_window = self._train_updates_window[-maxw:]
                                        if len(self._train_losses_window) > maxw:
                                            self._train_losses_window = self._train_losses_window[-maxw:]
                        except Exception:
                            pass
                        # PSI/KL/MMD: exportar drift sobre ventana y activar pausa si excede umbral
                        try:
                            import numpy as _np
                            from nge_trader.services.drift import kl_divergence, maximum_mean_discrepancy
                            self._live_states_window.append(state_vec.tolist())
                            base_arr = None
                            if self._baseline_static_states is not None and getattr(self._baseline_static_states, "size", 0) > 0:
                                base_arr = _np.array(self._baseline_static_states)
                            elif len(self._baseline_states) > 50:
                                base_arr = _np.array(self._baseline_states[-500:])
                            live_arr = _np.array(self._live_states_window[-500:]) if len(self._live_states_window) > 50 else None
                            if base_arr is not None and live_arr is not None and base_arr.size > 0 and live_arr.size > 0:
                                psi_val = self.online_policy.psi(base_arr, live_arr)
                                set_metric("model_drift_psi", float(psi_val))
                                if psi_val > float(self.settings.drift_psi_threshold or 0.2):
                                    import time as _t
                                    self._online_pause_until_ts = _t.time() + 60.0 * float(self.settings.online_training_pause_minutes or 10)
                                # KL/MMD sobre primeras dimensiones
                                try:
                                    p = (_np.mean(base_arr, axis=0)[:10]).tolist()
                                    q = (_np.mean(live_arr, axis=0)[:10]).tolist()
                                    # normalizar a distribución (softmax-like)
                                    def _norm(v: list[float]) -> list[float]:
                                        ex = _np.exp(v - _np.max(v))
                                        s = _np.sum(ex) or 1.0
                                        return (ex / s).tolist()
                                    pk = _norm(p)
                                    qk = _norm(q)
                                    kl = kl_divergence(pk, qk)
                                    mmd = maximum_mean_discrepancy(p, q)
                                    set_metric("model_drift_kl", float(kl))
                                    set_metric("model_drift_mmd", float(mmd))
                                    if kl > float(self.settings.drift_kl_threshold or 0.1) or mmd > float(self.settings.drift_mmd_threshold or 0.05):
                                        import time as _t
                                        self._online_pause_until_ts = _t.time() + 60.0 * float(self.settings.online_training_pause_minutes or 10)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    # Régimen y política por régimen (simple: invertir señal si régimen adverso)
                    try:
                        regime = self._regime.predict(df)
                        if regime == 1:
                            last_sig = -last_sig
                    except Exception:
                        pass
                    # Sesgo HF opcional (series temporales)
                    try:
                        from nge_trader.services.hf_timeseries import HFTimeseriesForecaster
                        if self.settings.huggingface_api_token:
                            hf = HFTimeseriesForecaster()
                            ret_pred = hf.predict_next_return(df)
                            if abs(ret_pred) > 0.0:
                                last_sig = float(np.sign(ret_pred))
                    except Exception:
                        pass
                    # Incorporar señal de sentimiento agregado como sesgo
                    sent_bias = 0.0
                    if self.settings.use_sentiment_bias:
                        try:
                            sent_bias = Database().aggregated_sentiment(
                                symbol=symbol, minutes=self.settings.sentiment_window_minutes
                            )
                        except Exception:
                            sent_bias = 0.0
                    if last_sig == 0.0 and (not self.settings.use_sentiment_bias or abs(sent_bias) < 0.2):
                        continue
                    # Kill-switch por sentimiento negativo extremo
                    try:
                        if self.settings.use_sentiment_bias and sent_bias < -abs(self.settings.nlp_killswitch_threshold):
                            self.notifier.send(f"Kill-switch por sentimiento negativo: {sent_bias:+.2f}. Saltando envíos.")
                            continue
                    except Exception:
                        pass
                    price_series = df["close"].astype(float).reset_index(drop=True)
                    # Gating de horizonte temporal (evitar scalping < 5m; favorecer 15m-4h)
                    try:
                        if bool(self.settings.enforce_horizon_gating):
                            # Heurística: si estrategia produce señales muy frecuentes, bloquear
                            # Medimos cambios de señal en últimas N barras
                            N = 40
                            sig_series = None
                            if 'signal' in df.columns:
                                sig_series = df['signal'].astype(float).tail(N)
                            elif 'close' in df.columns and hasattr(local_strat, 'generate_signals'):
                                sig_series = local_strat.generate_signals(df).tail(N)
                            if sig_series is not None and len(sig_series) >= 5:
                                flips = (sig_series.diff().fillna(0.0) != 0).sum()
                                # si demasiados flips por ventana → horizonte demasiado corto
                                # mapeamos flips a minutos aprox: flips alto => horizonte bajo
                                est_min = max(1, int((N / max(float(flips), 1.0)) * self.cfg.poll_seconds / 60.0))
                                if est_min < int(self.settings.min_horizon_minutes):
                                    # bloquear entradas por considerarse scalping
                                    continue
                    except Exception:
                        pass
                    price = float(price_series.iloc[-1])
                    # Quality gates: outliers de retorno
                    try:
                        r1 = float(price_series.pct_change().iloc[-1]) if len(price_series) > 1 else 0.0
                        if abs(r1) > float(self.settings.outlier_ret_threshold or 0.2):
                            self.notifier.send(f"Gate datos: outlier de retorno {r1:.1%} > umbral {self.settings.outlier_ret_threshold:.1%}")
                            continue
                    except Exception:
                        pass
                    # Gate por spread bajo
                    try:
                        from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
                        q = _LL().get_quote_metrics(symbol)
                        bid = float(q.get("bid") or 0.0)
                        ask = float(q.get("ask") or 0.0)
                        if bid > 0 and ask > 0 and self.settings.max_spread_pct is not None:
                            mid = 0.5 * (bid + ask)
                            spread_pct = (ask - bid) / max(mid, 1e-9)
                            if spread_pct > float(self.settings.max_spread_pct):
                                self.notifier.send(f"Gate spread: {spread_pct:.3%} > {self.settings.max_spread_pct:.3%}. Saltando {symbol}")
                                continue
                    except Exception:
                        pass
                    # Si existe VWAP de lowlat para este símbolo, preferirlo como precio de referencia
                    try:
                        vwap = self.lowlat.get_vwap(symbol)
                        if vwap is not None and vwap > 0:
                            price = float(vwap)
                    except Exception:
                        pass
                    # VaR/ES intradía y límites y breaker por performance reciente
                    try:
                        ret = price_series.pct_change().dropna().iloc[-252:]
                        var, es = compute_var_es(ret, alpha=0.05)
                        if var < -abs(self.settings.var_limit_pct) or es < -abs(self.settings.es_limit_pct):
                            self.notifier.send(f"Bloqueo por riesgo: VaR {var:.2%}, ES {es:.2%} supera límites")
                            continue
                        # Rollout: restringir según etapa
                        try:
                            stage = str(self.settings.rollout_stage).lower()
                            if stage in ("paper", "canary") and symbol.upper() not in (self.cfg.symbols[:1] if self.cfg.symbols else []):
                                continue
                        except Exception:
                            pass
                        # Breaker por Sharpe reciente
                        lb = int(getattr(self.settings, "recent_sharpe_lookback", 20) or 20)
                        rets_recent = price_series.pct_change().dropna().iloc[-max(lb, 5):].values.tolist()
                        if rets_recent:
                            from nge_trader.services.metrics import compute_sharpe as _comp_sharpe
                            sharpe_recent = _comp_sharpe(rets_recent)
                            from math import isnan as _isnan
                            if (self.settings.min_recent_sharpe is not None) and (not _isnan(sharpe_recent)) and (sharpe_recent < float(self.settings.min_recent_sharpe)):
                                self.notifier.send(f"Breaker performance: Sharpe reciente {sharpe_recent:.2f} < {self.settings.min_recent_sharpe}")
                            continue
                    except Exception:
                        pass
                    # Validaciones de riesgo/exposición
                    try:
                        # Reconciliación periódica (órdenes/posiciones)
                        try:
                            now_ts = pd.Timestamp.utcnow().timestamp()
                            last_rec = float(getattr(self, "_last_reconcile_ts", 0.0))
                            if (now_ts - last_rec) >= float(self.settings.reconcile_interval_minutes or 10) * 60.0:
                                self.service.reconcile_state(resolve=True)
                                self.service.reconcile_positions_balances(resolve=True)
                                self._last_reconcile_ts = now_ts
                        except Exception:
                            pass
                        # Fill ratio gating
                        try:
                            from nge_trader.services import metrics as _MM
                            fills_total = float(_MM._METRICS_STORE.get("fills_total", 0.0))
                            orders_total = float(_MM._METRICS_STORE.get("orders_placed_total", 0.0))
                            fr = (fills_total / orders_total) if orders_total > 0 else 1.0
                            if fr < float(self.settings.min_fill_ratio_required or 0.7):
                                continue
                        except Exception:
                            pass
                        # Data dislocations A/B y staleness
                        try:
                            # Si tenemos métricas de staleness y superan SLO, pausar
                            staleness_ms = float(_MM._METRICS_STORE.get("market_ws_staleness_ms", 0.0))
                            if staleness_ms > float(self.settings.data_staleness_slo_ms or 1000):
                                continue
                            # Dislocation entre fuentes: si se expone métrica, comparar con umbral
                            disloc_bps = float(_MM._METRICS_STORE.get("market_dislocation_bps", 0.0))
                            if abs(disloc_bps) > float(self.settings.dislocation_bps_threshold or 20.0):
                                continue
                            # Flash-move y spread-widening
                            flash_bps = float(_MM._METRICS_STORE.get("flash_move_bps", 0.0))
                            if abs(flash_bps) > float(self.settings.flash_move_bps_threshold or 50.0):
                                continue
                            widen_bps = float(_MM._METRICS_STORE.get("spread_widen_bps", 0.0))
                            if abs(widen_bps) > float(self.settings.spread_widen_bps_threshold or 30.0):
                                continue
                        except Exception:
                            pass
                        # cooldown por símbolo
                        import time as _time
                        last_ts = self._last_sent_ts.get(symbol, 0.0)
                        if (_time.time() - last_ts) < (self.settings.cooldown_minutes * 60):
                            continue
                        # cool-down tras racha de pérdidas
                        try:
                            symu = symbol.upper()
                            until = float(self._loss_cooldown_until.get(symu) or 0.0)
                            if until and _time.time() < until:
                                continue
                        except Exception:
                            pass
                        # límite diario de órdenes
                        if self._trades_today >= self.settings.max_trades_per_day:
                            continue
                        # límite de correlación: si símbolo está muy correlacionado con posiciones existentes, saltar
                        try:
                            if positions:
                                # construir matriz de retornos reciente de símbolo + existentes
                                import numpy as _np
                                close_s = df["close"].astype(float).reset_index(drop=True)
                                series_map = {symbol: close_s}
                                for p in positions:
                                    sym_p = p.get("symbol")
                                    if not sym_p or sym_p == symbol:
                                        continue
                                    dpf = self.service.data_provider.get_daily_adjusted(sym_p)
                                    if dpf.empty:
                                        continue
                                    series_map[sym_p] = dpf["close"].astype(float).reset_index(drop=True)
                                if len(series_map) > 1:
                                    # alinear y calcular correlación máx con otros
                                    lens = [len(s) for s in series_map.values()]
                                    n = min(lens)
                                    rets = {}
                                    for k, s in series_map.items():
                                        r = s.iloc[-n:].pct_change().dropna()
                                        rets[k] = r.reset_index(drop=True)
                                    dfret = pd.DataFrame(rets).dropna()
                                    if symbol in dfret.columns and dfret.shape[1] > 1:
                                        corr_s = dfret.corr()[symbol].drop(index=[symbol])
                                        if not corr_s.empty and float(_np.nanmax(_np.abs(corr_s.values))) > self.settings.correlation_limit:
                                            continue
                        except Exception:
                            pass
                        positions = self.service.get_portfolio_positions()
                        if len(positions) >= self.settings.max_open_positions:
                            continue
                        # Exposición por sector
                        try:
                            sector_map = {}
                            if self.settings.sector_map:
                                for kv in str(self.settings.sector_map).split(","):
                                    if "=" in kv:
                                        k, v = kv.split("=", 1)
                                        sector_map[k.strip().upper()] = v.strip().upper()
                            def _sector(sym: str) -> str:
                                return sector_map.get(sym.upper(), "UNKNOWN")
                            sector_cap: dict[str, float] = {}
                            for p in positions:
                                sym = (p.get("symbol") or "").upper()
                                if not sym:
                                    continue
                                mv = float(p.get("market_value") or p.get("value") or 0.0)
                                sector_cap[_sector(sym)] = sector_cap.get(_sector(sym), 0.0) + mv
                            # proyectar esta orden
                            sector = _sector(symbol)
                            sector_cap[sector] = sector_cap.get(sector, 0.0) + float(self.cfg.capital_per_trade)
                            acc = self.service.broker.get_account() if hasattr(self.service.broker, "get_account") else {"equity": self.cfg.capital_per_trade}
                            equity = float(acc.get("equity") or self.cfg.capital_per_trade)
                            if (sector_cap.get(sector, 0.0) / max(equity, 1e-9)) > self.settings.max_sector_exposure_pct:
                                continue
                        except Exception:
                            pass
                        acc = self.service.broker.get_account() if hasattr(self.service.broker, "get_account") else None
                        equity = float(acc.get("equity")) if acc and acc.get("equity") else self.cfg.capital_per_trade
                        # Exposición total estimada tras esta orden
                        total_mv = 0.0
                        for p in positions:
                            mv = p.get("market_value") or p.get("value")
                            total_mv += float(mv) if mv is not None else 0.0
                        prospective = total_mv + (price * max(self.cfg.capital_per_trade / max(price, 1e-9), 0.0))
                        if prospective / max(equity, 1e-9) > self.settings.max_exposure_pct:
                            continue
                    except Exception:
                        pass
                    # Tamaño por riesgo: stop a 1*ATR
                    atr = compute_atr(df).iloc[-1]
                    stop_distance = float(atr) if pd.notna(atr) and atr > 0 else price * 0.02
                    # Usa riesgo del settings
                    qty = position_size_by_risk(self.cfg.capital_per_trade, self.settings.risk_pct_per_trade, stop_distance, price)
                    # Gate por min_qty
                    try:
                        if self.settings.min_qty is not None and float(qty) < float(self.settings.min_qty):
                            self.notifier.send(f"Gate min_qty: qty {qty:.6f} < {self.settings.min_qty}. Saltando {symbol}")
                            continue
                    except Exception:
                        pass
                    # Rollout scaling por etapa
                    try:
                        stage = str(self.settings.rollout_stage).lower()
                        if stage == "ramp":
                            qty *= float(self.settings.rollout_ramp_pct or 0.25)
                    except Exception:
                        pass
                    # targeting de volatilidad (diaria) sobre retornos recientes
                    try:
                        recent = price_series.pct_change().dropna().iloc[-60:]
                        cur_vol = float(recent.std(ddof=0))
                        tgt = float(self.settings.target_daily_volatility)
                        scale = volatility_target_position(cur_vol, tgt)
                        qty *= max(scale, 0.0)
                    except Exception:
                        pass
                # Overrides por símbolo
                try:
                    import json as _json
                    symu = symbol.upper()
                    if self.settings.risk_profile_map:
                        rp = _json.loads(str(self.settings.risk_profile_map))
                        if symu in rp and isinstance(rp[symu], dict):
                            rpc = float(rp[symu].get("risk_pct_per_trade", self.settings.risk_pct_per_trade))
                            qty = position_size_by_risk(self.cfg.capital_per_trade, rpc, stop_distance, price)
                    if self.settings.slicing_profile_map:
                        sp = _json.loads(str(self.settings.slicing_profile_map))
                        if symu in sp and isinstance(sp[symu], dict):
                            # actualizar parámetros de slicing en caliente
                            tr = sp[symu].get("slicing_tranches")
                            bps = sp[symu].get("slicing_bps")
                            if tr is not None:
                                self.settings.slicing_tranches = int(tr)
                            if bps is not None:
                                self.settings.slicing_bps = float(bps)
                except Exception:
                    pass
                    side = "buy" if last_sig > 0 else "sell"
                    if qty <= 0:
                        continue
                    try:
                        # Gating duro por presupuesto diario (simple endpoint risk)
                        try:
                            import requests as _req
                            base = "http://localhost:8000"
                            risk = _req.get(f"{base}/api/kit/risk", timeout=0.5).json()
                            if float(risk.get("daily_budget_left", 0.0)) <= 0.0:
                                self.notifier.send("budget gate: no-trade")
                                continue
                        except Exception:
                            pass
                        # Gating duro por viabilidad: edge vs costes
                        try:
                            from nge_trader.services.viability import get_live_costs, estimate_edge_bps
                            from nge_trader.services.gates import is_trade_viable
                            costs = get_live_costs(symbol)
                            edge_bps = float(estimate_edge_bps(symbol))
                            if not is_trade_viable(edge_bps, float(costs.get("fees_bps", 0.0)), float(costs.get("half_spread_bps", 0.0))):
                                # Bloquear envío si no hay edge neto suficiente
                                self.notifier.send(f"No-trade por edge<cost: {symbol} edge={edge_bps:.1f}bps costes={float(costs.get('fees_bps',0.0))+float(costs.get('half_spread_bps',0.0))+2.0:.1f}bps")
                                continue
                        except Exception:
                            pass
                        # SLO gate: slippage, error-rate, p95 latencia (global)
                        try:
                            from nge_trader.services.metrics import get_series_percentile
                            p95 = float(get_series_percentile("order_place_latency_ms", 95))
                        except Exception:
                            p95 = float('nan')
                        try:
                            from nge_trader.repository.db import Database as _DB
                            rows = _DB().recent_metric_values("slippage_bps", 500)
                            vals = [float(v) for _, v in rows[-100:]]
                            slip_avg = float(sum(vals)/len(vals)) if vals else 0.0
                        except Exception:
                            slip_avg = 0.0
                        try:
                            now_ts = pd.Timestamp.utcnow().timestamp()
                            window_s = 600.0
                            self._err_events = deque([t for t in self._err_events if (now_ts - t) <= window_s], maxlen=1000)
                            self._attempt_events = deque([t for t in self._attempt_events if (now_ts - t) <= window_s], maxlen=1000)
                            attempts = max(len(self._attempt_events), 1)
                            err_rate = len(self._err_events) / attempts
                        except Exception:
                            err_rate = 0.0
                        thr_slip = float(self.settings.slo_slippage_bps or 8.0)
                        thr_err = float(self.settings.slo_error_rate or 0.01)
                        thr_p95 = float(self.settings.slo_p95_ms or 300.0)
                        if (abs(slip_avg) > thr_slip) or (err_rate > thr_err) or (p95 > thr_p95):
                            # Pausar nuevas entradas (permitir salidas/SL)
                            self.notifier.send(f"SLO gate activo: slip={slip_avg:.1f}bps err={err_rate:.2%} p95={p95:.0f}ms")
                            continue
                        # SLO gate por símbolo
                        try:
                            from nge_trader.repository.db import Database as _DB
                            rows = _DB().recent_metric_values(f"slippage_bps_{symbol.upper()}", 2000)
                            import time as _time
                            now_ts = float(_time.time())
                            vals_7d = [float(v) for ts, v in rows if (now_ts - float(ts)) <= 86400*7]
                            slip_sym = float(sum(vals_7d)/len(vals_7d)) if vals_7d else slip_avg
                        except Exception:
                            slip_sym = slip_avg
                        try:
                            from nge_trader.services.metrics import get_series_percentile as _gp
                            p95_sym = float(_gp(f"order_place_latency_ms_{symbol.upper()}", 95))
                        except Exception:
                            p95_sym = p95
                        if (abs(slip_sym) > thr_slip) or (p95_sym > thr_p95):
                            self.notifier.send(f"SLO gate por símbolo activo: {symbol} slip={slip_sym:.1f}bps p95={p95_sym:.0f}ms")
                            continue
                        # Drift gate (PSI): reduce tamaño o no-trade
                        try:
                            psi = float(self.settings.drift_psi_max)  # placeholder: leer de métricas si existe
                            # Si tenemos métrica en memoria:
                            try:
                                from nge_trader.services import metrics as _MM
                                psi = float(_MM._METRICS_STORE.get("model_drift_psi", psi))
                            except Exception:
                                pass
                            if psi > 0.3:
                                self.notifier.send("Drift gate: PSI>0.3 no-trade")
                                continue
                            if 0.2 < psi <= 0.3:
                                qty *= 0.5
                                self.notifier.send("Drift gate: size halved by drift gate")
                        except Exception:
                            pass
                        # SL/TP basados en ATR con break-even y trailing
                        tp = price + (2 * stop_distance if side == "buy" else -2 * stop_distance)
                        # Stop inicial a 1R (stop_distance)
                        sl = price - (1.0 * stop_distance if side == "buy" else -1.0 * stop_distance)
                        order_method = getattr(self.service.broker, "place_bracket_order", None)
                        if order_method:
                            order = order_method(symbol, side, qty, take_profit_price=tp, stop_loss_price=sl)
                        else:
                            # breaker por slippage/error-rate (previo al envío)
                            try:
                                now_ts = pd.Timestamp.utcnow().timestamp()
                                window_s = 600.0
                                self._err_events = deque([t for t in self._err_events if (now_ts - t) <= window_s], maxlen=1000)
                                self._attempt_events = deque([t for t in self._attempt_events if (now_ts - t) <= window_s], maxlen=1000)
                                attempts = max(len(self._attempt_events), 1)
                                err_rate = len(self._err_events) / attempts
                                if err_rate > float(self.settings.max_error_rate or 0.2):
                                    self.notifier.send(f"Breaker: error-rate {err_rate:.2%} > {self.settings.max_error_rate:.2%}. Bloqueando envíos.")
                                    continue
                            except Exception:
                                pass
                            # registrar intento
                            self._attempt_events.append(pd.Timestamp.utcnow().timestamp())
                            self._last_intended_price[symbol.upper()] = price
                            # Slicing adaptativo: escoger TWAP/VWAP/POV según volatilidad/volumen reciente
                            try:
                                from nge_trader.services.oms import place_limit_order_post_only
                                last_px = float(self._last_mkt_price.get(symbol.upper(), price))
                                # Volatilidad y volumen recientes (ventana configurable)
                                now_s = pd.Timestamp.utcnow().timestamp()
                                recent = [(t, px, q) for (t, px, q) in self._mkt_events if (now_s - t) <= float(self.settings.vwap_window_sec or 60)]
                                volp = 0.0
                                if len(recent) > 2:
                                    import numpy as _np
                                    rets = _np.diff([r[1] for r in recent]) / max(recent[0][1], 1e-9)
                                    volp = float(_np.std(rets))
                                total_vol = float(sum(r[2] for r in recent)) or 1.0
                                vwap = float(sum(r[1] * r[2] for r in recent) / total_vol) if recent else last_px
                                # Estrategia
                                if volp > (float(self.settings.target_daily_volatility) * 2.0):
                                    mode = "TWAP"
                                else:
                                    mode = "POV"
                                def _exec(mode: str) -> None:
                                    if mode == "TWAP":
                                        n = int(self.settings.slicing_tranches or 6)
                                        for i in range(n):
                                            part = float(qty) / n
                                            # precio límite más conservador cuando vol es alta
                                            bps = float(self.settings.slicing_bps or 5.0) / 10000.0
                                            try:
                                                from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
                                                mp = _LL().get_microprice(symbol) or last_px
                                                ofi, _ = _LL().get_ofi_obi(symbol)
                                                adj = 1.0 + max(min(ofi, 0.2), -0.2)
                                                delta = float(mp) * (bps + min(volp, 0.005)) * adj
                                                limit_px = (mp - delta) if side == "buy" else (mp + delta)
                                            except Exception:
                                            delta = last_px * (bps + min(volp, 0.005))
                                            limit_px = (last_px - delta) if side == "buy" else (last_px + delta)
                                            try:
                                                # Router maker-first
                                                try:
                                                    from nge_trader.services.execution_router import route_order as _route
                                                    _route(self.service.broker, symbol, side, float(part))
                                                except Exception:
                                                place_limit_order_post_only(self.service.broker, symbol, side, part, limit_px, "GTC")
                                            except Exception:
                                                try:
                                                    self.service.broker.place_order(symbol, side, part, type_="market")
                                                except Exception:
                                                    self._err_events.append(pd.Timestamp.utcnow().timestamp())
                                            time.sleep(max(self.cfg.poll_seconds // n, 1))
                                    else:  # POV
                                        n = int(self.settings.slicing_tranches or 5)
                                        pov_ratio = float(self.settings.pov_ratio or 0.1)
                                        est_part = max(float(total_vol) * pov_ratio / n, float(qty) / (n * 2))
                                        remaining = float(qty)
                                        for i in range(n):
                                            part = min(est_part, remaining)
                                            bps = float(self.settings.slicing_bps or 5.0) / 10000.0
                                            delta = vwap * bps
                                            # Ajustar por OFI/OBI (order flow imbalance) del L1
                                            try:
                                                from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
                                                ofi, obi = _LL().get_ofi_obi(symbol)
                                                # tamaño estimado adaptado por desequilibrio (más pequeño si OBI en contra)
                                                est_part = max(est_part * (1.0 + max(min(ofi, 0.2), -0.2)), float(qty) / (n * 2))
                                                # precio límite sugerido por microprice/OFI
                                                limit_px = _LL().suggest_limit_price(symbol, side, base_price=vwap, bps=bps)
                                            except Exception:
                                            limit_px = (vwap - delta) if side == "buy" else (vwap + delta)
                                            try:
                                                try:
                                                    from nge_trader.services.execution_router import route_order as _route
                                                    _route(self.service.broker, symbol, side, float(part))
                                                except Exception:
                                                place_limit_order_post_only(self.service.broker, symbol, side, part, limit_px, "GTC")
                                                remaining -= part
                                            except Exception:
                                                try:
                                                    self.service.broker.place_order(symbol, side, part, type_="market")
                                                    remaining -= part
                                                except Exception:
                                                    self._err_events.append(pd.Timestamp.utcnow().timestamp())
                                            time.sleep(max(self.cfg.poll_seconds // n, 1))
                                # Auditoría enriquecida antes de enviar
                                is_canary = bool(getattr(self.settings, "enable_canary", False)) and (float(getattr(self.settings, "challenger_share", 0.2)) > 0)
                                try:
                                    corr_id = getattr(self, "correlation_id", str(uuid4()))
                                except Exception:
                                    corr_id = str(uuid4())
                                try:
                                    route_algo = str(mode)
                                except Exception:
                                    route_algo = "TWAP"
                                try:
                                    from nge_trader.services.viability import get_live_costs, estimate_edge_bps
                                    costs = get_live_costs(symbol)
                                    edge_bps = float(estimate_edge_bps(symbol))
                                    fees_bps = float(costs.get("fees_bps", 0.0))
                                    half_spread_bps = float(costs.get("half_spread_bps", 0.0))
                                except Exception:
                                    edge_bps = 0.0
                                    fees_bps = 0.0
                                    half_spread_bps = 0.0
                                audit_extra = {
                                    "correlation_id": corr_id,
                                    "model_used": "pinned" if ("is_pinned" in locals() and is_pinned) else ("challenger" if is_canary else "champion"),
                                    "model_id": (locals().get("model_id_used") or getattr(self.settings, "champion_model_id", "champion")),
                                    "canary": bool(is_canary),
                                    "edge_bps": float(edge_bps),
                                    "costs_bps": float(fees_bps + half_spread_bps + 2.0),
                                    "route_algo": route_algo,
                                }
                                threading.Thread(target=_exec, args=(mode,), daemon=True).start()
                                from nge_trader.services.metrics import inc_metric_labeled
                                inc_metric_labeled("orders_submitted_total", 1.0, {"broker": self.settings.broker, "strategy": getattr(self.cfg, "strategy", "unknown"), "symbol": symbol.upper()})
                                order = {"status": f"{mode.lower()}_submitted", "meta": {}}
                                try:
                                    order["meta"].update(audit_extra)
                                    logger.info("order_submit", extra=audit_extra)
                                except Exception:
                                    pass
                            except Exception:
                                order = self.service.broker.place_order(symbol, side, qty)
                        # Registrar experiencia s, a (acción = sign), reward placeholder 0, next_state = estado tras tick
                        try:
                            next_state = self.online_policy.state_from_df(df)
                            self.replay.append(
                                ts_iso=pd.Timestamp.utcnow().isoformat(),
                                symbol=symbol,
                                state={"state": state_vec.tolist()},
                                action=1.0 if side == "buy" else -1.0,
                                reward=0.0,
                                next_state={"state": next_state.tolist()},
                                done=False,
                            )
                        except Exception:
                            pass
                        self.db.record_order({
                            "ts": pd.Timestamp.utcnow().isoformat(),
                            "symbol": symbol,
                            "side": side,
                            "qty": float(qty),
                            "price": price,
                            "status": order.get("status") if isinstance(order, dict) else "sent",
                            "order_id": order.get("id") if isinstance(order, dict) else None,
                            "model_used": "challenger" if bool(getattr(self.settings, "enable_canary", False)) and (float(getattr(self.settings, "challenger_share", 0.2)) > 0) else "champion",
                        })
                        try:
                            self.db.append_audit(
                                pd.Timestamp.utcnow().isoformat(),
                                "order_sent",
                                order.get("id") if isinstance(order, dict) else None,
                                symbol,
                                json.dumps({"order": order, "intended_price": price}),
                            )
                        except Exception:
                            self.db.append_audit(pd.Timestamp.utcnow().isoformat(), "order_sent", order.get("id") if isinstance(order, dict) else None, symbol, str(order))
                        self.notifier.send(f"Orden enviada: {symbol} {side} {qty:.4f} @ ~{price:.2f}")
                        # TCA: registrar latencia de envío (si aplica)
                        try:
                            from nge_trader.services import tca_writer as _tca
                            place_ms = None
                            try:
                                t0 = float(locals().get("t0_submit") or 0.0)
                                if t0:
                                    import time as _tt
                                    place_ms = (float(_tt.perf_counter()) - t0) * 1000.0
                            except Exception:
                                place_ms = None
                            # Guardar referencia de mid para cálculo posterior (por fill)
                            try:
                                from nge_trader.adapters.lowlat_provider import LowLatencyProvider as _LL
                                bidask = self.lowlat.get_best_bid_ask(symbol) if hasattr(self.lowlat, "get_best_bid_ask") else None
                                if bidask and isinstance(bidask, (list, tuple)) and len(bidask) >= 2:
                                    ref_mid = (float(bidask[0]) + float(bidask[1])) / 2.0
                                else:
                                    ref_mid = float(price)
                            except Exception:
                                ref_mid = float(price)
                            _tca.submit(_tca.TCAEvent(
                                symbol=symbol,
                                ts=pd.Timestamp.utcnow().timestamp(),
                                slippage_bps=0.0,
                                order_place_latency_ms=float(place_ms) if place_ms is not None else None,
                                maker_taker="unknown",
                                route_algo=str(audit_extra.get("route_algo", "UNKNOWN")),
                                order_id=str(order.get("id")) if isinstance(order, dict) and order.get("id") else None,
                                side=side,
                                ref_px=float(ref_mid),
                                price=float(price),
                                qty=float(qty),
                                meta={"event": "submit"},
                            ))
                        except Exception:
                            pass
                        # actualizar cooldown y conteo
                        import time as _time
                        self._last_sent_ts[symbol] = _time.time()
                        self._trades_today += 1
                        # Si estaba en cool-down, al enviar nueva orden solo si expiró; reset si se envía
                        try:
                            symu = symbol.upper()
                            if symu in self._loss_cooldown_until and _time.time() >= float(self._loss_cooldown_until.get(symu) or 0.0):
                                self._loss_cooldown_until.pop(symu, None)
                        except Exception:
                            pass
                    except Exception as exc:  # noqa: BLE001
                        self.notifier.send(f"Error al enviar orden para {symbol}: {exc}")
                        try:
                            self._err_events.append(pd.Timestamp.utcnow().timestamp())
                        except Exception:
                            pass
                time.sleep(self.cfg.poll_seconds)
                # Ingesta de fills recientes y actualización de reward
                try:
                        if hasattr(self.service.broker, "list_fills"):
                            from nge_trader.services.oms import normalize_fill
                            fills = self.service.broker.list_fills()
                            for f in fills[:50]:
                                nf = normalize_fill(f)
                                if not nf.get("ts"):
                                    nf["ts"] = pd.Timestamp.utcnow().isoformat()
                                self.db.record_fill(nf)
                                # Si la orden pertenece a un par OCO, intentar cancelar la contraparte
                                try:
                                    oid = str(nf.get("order_id")) if nf.get("order_id") else None
                                    if oid:
                                        links = self.db.get_linked_orders(oid)
                                        for linked_id, ltype in links:
                                            if ltype == "OCO" and hasattr(self.service.broker, "cancel_order"):
                                                try:
                                                    self.service.broker.cancel_order(linked_id)
                                                    self.db.append_audit(pd.Timestamp.utcnow().isoformat(), "oco_cancel", linked_id, nf.get("symbol"), "auto-cancel after fill")
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
                                self.db.append_audit(pd.Timestamp.utcnow().isoformat(), "fill", nf.get("order_id"), nf.get("symbol"), str(f))
                                # TCA: evento por fill
                                try:
                                    from nge_trader.services import tca_writer as _tca
                                    sym = str(nf.get("symbol") or "").upper()
                                    side_fill = str(nf.get("side") or "").lower()
                                    fill_px = float(nf.get("price") or 0.0)
                                    fill_qty = float(nf.get("qty") or 0.0)
                                    oid = str(nf.get("order_id")) if nf.get("order_id") else None
                                    fid = str(nf.get("id")) if nf.get("id") else None
                                    # Referencia mid/side
                                    try:
                                        bidask = self.lowlat.get_best_bid_ask(sym) if hasattr(self.lowlat, "get_best_bid_ask") else None
                                        if bidask and isinstance(bidask, (list, tuple)) and len(bidask) >= 2:
                                            best_bid = float(bidask[0]); best_ask = float(bidask[1])
                                            ref_mid = (best_bid + best_ask) / 2.0
                                            ref_side = best_ask if side_fill == "buy" else best_bid
                                        else:
                                            ref_mid = float(self._last_intended_price.get(sym) or fill_px)
                                            ref_side = ref_mid
                                    except Exception:
                                        ref_mid = float(self._last_intended_price.get(sym) or fill_px)
                                        ref_side = ref_mid
                                    # Slippage bps
                                    if side_fill == "buy":
                                        slip_mid_bps = ((fill_px - ref_mid) / max(ref_mid, 1e-9)) * 10000.0
                                        slip_side_bps = ((fill_px - ref_side) / max(ref_side, 1e-9)) * 10000.0
                                    else:
                                        slip_mid_bps = ((ref_mid - fill_px) / max(ref_mid, 1e-9)) * 10000.0
                                        slip_side_bps = ((ref_side - fill_px) / max(ref_side, 1e-9)) * 10000.0
                                    # maker/taker real si el broker lo provee; fallback por ruta
                                    maker_taker = str(nf.get("liquidity") or nf.get("maker_taker") or ("maker" if str(audit_extra.get("route_algo", "")).lower().startswith("maker") else "unknown"))
                                    _tca.submit(_tca.TCAEvent(
                                        symbol=sym,
                                        ts=pd.Timestamp.utcnow().timestamp(),
                                        slippage_bps=float(slip_mid_bps),
                                        order_place_latency_ms=None,
                                        maker_taker=maker_taker,
                                        route_algo=str(audit_extra.get("route_algo", "UNKNOWN")),
                                        order_id=oid,
                                        fill_id=fid,
                                        side=side_fill,
                                        price=float(fill_px),
                                        ref_px=float(ref_mid),
                                        qty=float(fill_qty),
                                        fees_ccy=float(nf.get("fee_amount") or 0.0),
                                        fees_bps=float(nf.get("fee_bps") or 0.0),
                                        meta={"slip_side_bps": float(slip_side_bps)},
                                    ))
                                except Exception:
                                    pass
                        # Reward incremental: pnl - fees - dd_penalty - vol_penalty; log slippage y errores
                                try:
                                    pnl = float(nf.get("qty") or 0.0) * float(nf.get("price") or 0.0) * (1.0 if (nf.get("side") or "").lower()=="sell" else -1.0)
                                    fees = 0.0
                                    # drawdown/vol aproximados con ventana corta
                                    df_sym = self.service.data_provider.get_daily_adjusted(nf.get("symbol"))
                                    rets = df_sym["close"].astype(float).pct_change().dropna().iloc[-20:]
                                    vol = float(rets.std(ddof=0)) if not rets.empty else 0.0
                                    dd = float((rets.cumsum().min() or 0.0)) if not rets.empty else 0.0
                                    reward = pnl - 0.1 * abs(dd) - 0.01 * vol - fees
                                    # Actualizar policy con estado reciente y derivar métricas live
                                    st_vec = self.online_policy.state_from_df(df_sym)
                                    try:
                                        import numpy as _np
                                        # Construir baseline inicial con primeras N observaciones
                                        if len(self._baseline_states) < max(200, int(self.settings.online_lookback or 30) * 5):
                                            self._baseline_states.append(list(st_vec))
                                        # Ventana live para PSI
                                        self._live_states_window.append(list(st_vec))
                                        if len(self._live_states_window) > 500:
                                            self._live_states_window = self._live_states_window[-500:]
                                        # Calcular PSI si hay baseline suficiente
                                        if len(self._baseline_states) >= 100 and len(self._live_states_window) >= 50:
                                            base_arr = _np.array(self._baseline_states[-1000:], dtype=float)
                                            live_arr = _np.array(self._live_states_window[-200:], dtype=float)
                                            psi_val = float(self.online_policy.psi(base_arr, live_arr))
                                            from nge_trader.services.metrics import set_metric
                                            set_metric("model_drift_psi", psi_val)
                                            if psi_val > float(self.settings.drift_psi_threshold or 0.2):
                                                self.notifier.send(f"Alerta: drift PSI={psi_val:.3f} > umbral {self.settings.drift_psi_threshold}")
                                    except Exception:
                                        pass
                                    loss = self.online_policy.update(st_vec, reward)
                                    try:
                                        self._train_updates_window.append(1.0)
                                        if isinstance(loss, (float, int)):
                                            self._train_losses_window.append(float(loss))
                                        self._train_action_hist.append(1.0 if last_sig > 0 else (-1.0 if last_sig < 0 else 0.0))
                                        from nge_trader.services.metrics import set_metric
                                        if self._train_losses_window:
                                            set_metric("training_loss_mean", float(sum(self._train_losses_window) / max(len(self._train_losses_window), 1)))
                                        if self._train_action_hist:
                                            pos = [a for a in self._train_action_hist if a > 0]
                                            neg = [a for a in self._train_action_hist if a < 0]
                                            zero = [a for a in self._train_action_hist if a == 0]
                                            set_metric("training_action_buy_ratio", float(len(pos) / max(len(self._train_action_hist), 1)))
                                            set_metric("training_action_sell_ratio", float(len(neg) / max(len(self._train_action_hist), 1)))
                                            set_metric("training_action_hold_ratio", float(len(zero) / max(len(self._train_action_hist), 1)))
                                    except Exception:
                                        pass
                                    # Registrar en replay con prioridad
                                    try:
                                        self.replay.append(pd.Timestamp.utcnow().isoformat(), nf.get("symbol"), {"state": list(st_vec)}, 0.0, float(reward), {"state": list(st_vec)}, False, priority=abs(float(reward)))
                                    except Exception:
                                        pass
                                    # slippage aproximado con precio teórico guardado y breaker por media
                                    try:
                                        symu = (nf.get("symbol") or "").upper()
                                        intended = float(self._last_intended_price.get(symu) or 0.0)
                                        fill_px = float(nf.get("price") or 0.0)
                                        if intended > 0 and fill_px > 0:
                                            side_mult = -1.0 if (nf.get("side") or "").lower()=="buy" else 1.0
                                            slippage_bps = ((fill_px - intended) / intended) * 10000.0 * side_mult
                                            self._slippages_bps.append(slippage_bps)
                                            self.db.append_log_json("INFO", {"metric": "slippage_bps", "value": slippage_bps}, pd.Timestamp.utcnow().isoformat())
                                            try:
                                                from nge_trader.services.metrics import observe_series
                                                observe_series("slippage_bps", float(slippage_bps))
                                            except Exception:
                                                pass
                                            if len(self._slippages_bps) >= 5:
                                                last_n = list(self._slippages_bps)[-20:]
                                                avg_slip = float(sum(last_n)) / max(len(last_n),1)
                                                if abs(avg_slip) > float(self.settings.max_slippage_bps or 50.0):
                                                    self.notifier.send(f"Breaker: slippage medio {avg_slip:.1f} bps > {self.settings.max_slippage_bps} bps. Activando kill-switch.")
                                                    self.settings.kill_switch_armed = True
                                    except Exception:
                                        pass
                                    # métricas rolling: sharpe/hitrate/profit factor (entrenamiento)
                                    try:
                                        from nge_trader.repository.db import Database as _DB
                                        dbi = _DB()
                                        rr = [v for _, v in dbi.recent_rewards(100)]
                                        if rr:
                                            import numpy as _np
                                            arr = _np.array(rr, dtype=float)
                                            hr = float((arr > 0).mean())
                                            sh = float(arr.mean() / (arr.std(ddof=0) or 1e-9)) * (_np.sqrt(len(arr)) if len(arr) > 1 else 1.0)
                                            dbi.put_metric("hit_rate", hr)
                                            dbi.put_metric("sharpe_live", sh)
                                            # profit factor sobre recompensas (aprox)
                                            gains = float(arr[arr > 0].sum())
                                            losses = float(-arr[arr < 0].sum())
                                            pf = float("nan") if losses <= 0 else float(gains / losses)
                                            # Publicar en Prometheus store
                                            try:
                                                from nge_trader.services.metrics import set_metric as _setm
                                                _setm("training_hit_rate", hr)
                                                _setm("training_sharpe", sh)
                                                _setm("training_profit_factor", pf)
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                except Exception:
                    pass
                except Exception:
                    pass
                # Trailing & break-even automáticos: si hay órdenes abiertas, ajustar SL
                try:
                    if hasattr(self.service.broker, "list_orders") and hasattr(self.service.broker, "replace_order"):
                        open_orders = self.service.broker.list_orders(status="open", limit=50)
                        for o in open_orders:
                            # trabajamos con brackets o con órdenes con stop asociado
                            sym = o.get("symbol")
                            if not sym:
                                continue
                            df = self.service.data_provider.get_daily_adjusted(sym)
                            atr = compute_atr(df).iloc[-1]
                            price = float(df["close"].astype(float).iloc[-1])
                            if pd.isna(atr) or atr <= 0:
                                continue
                            # Trailing solo mejora SL en la dirección del trade
                            side = (o.get("side") or "").lower()
                            r_dist = float(atr)
                            # Break-even cuando el precio avanza >= breakeven_r_multiple * R
                            be_mult = float(self.settings.breakeven_r_multiple or 1.0)
                            if side == "buy":
                                be_price = float(o.get("avg_entry_price") or o.get("price") or price)
                                if price >= be_price + be_mult * r_dist:
                                    be_sl = be_price
                                else:
                                    be_sl = None
                                trail_sl = price - float(self.settings.trailing_atr_mult) * r_dist
                                new_sl = max([v for v in [be_sl, trail_sl] if v is not None]) if [v for v in [be_sl, trail_sl] if v is not None] else trail_sl
                            else:
                                be_price = float(o.get("avg_entry_price") or o.get("price") or price)
                                if price <= be_price - be_mult * r_dist:
                                    be_sl = be_price
                                else:
                                    be_sl = None
                                trail_sl = price + float(self.settings.trailing_atr_mult) * r_dist
                                new_sl = min([v for v in [be_sl, trail_sl] if v is not None]) if [v for v in [be_sl, trail_sl] if v is not None] else trail_sl
                            # Reemplazo básico: depende de subórdenes; aquí intentamos directamente sobre la principal si acepta stop_price
                            try:
                                self.service.broker.replace_order(o.get("id"), {"stop_price": round(new_sl, 4)})
                                self.db.append_audit(pd.Timestamp.utcnow().isoformat(), "replace_sl", o.get("id"), sym, json.dumps({"new_sl": new_sl}))
                            except Exception:
                                pass
                except Exception:
                    pass
            except Exception as loop_exc:  # noqa: BLE001
                self.notifier.send(f"Error del loop: {loop_exc}")
                time.sleep(self.cfg.poll_seconds)
        # Detener WS
        try:
            if self._ws:
                self._ws.stop()
        except Exception:
            pass
        try:
            if self._mws:
                self._mws.stop()
        except Exception:
            pass
        try:
            if getattr(self, "_cws", None):
                self._cws.stop()  # type: ignore[union-attr]
        except Exception:
            pass


