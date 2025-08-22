from __future__ import annotations
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from dotenv import dotenv_values

from nge_trader.config.settings import Settings
from nge_trader.services.app_service import AppService
from nge_trader.services.metrics import (
    export_metrics_text,
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_win_rate,
    compute_profit_factor,
    compute_skewness,
    compute_kurtosis,
    get_ws_states_snapshot,
)
from nge_trader.services import metrics as metrics_mod
from nge_trader.repository.db import Database
from nge_trader.services.rate_limiter import GlobalRateLimiter
from nge_trader.services.secret_store import SecretStore
from nge_trader.services.market_clock import is_session_open_cached, session_details
from nge_trader.services.oms import (
    place_market_order,
    place_limit_order_post_only,
    place_oco_order_generic,
    cancel_all_orders_by_symbol,
    cancel_order_idempotent,
    replace_order_limit_price_idempotent,
)
from nge_trader.services.model_registry import ModelRegistry
from nge_trader.adapters.lowlat_provider import LowLatencyProvider
from nge_trader.repository.db import Database
from nge_trader.services.notifier import Notifier
from datetime import datetime, UTC, date
from nge_trader.services.vendor_selector import recommend_vendors
from nge_trader.services import oms as _OMS
from nge_trader.services.control import disarm_and_cancel_if_config
from nge_trader.services.reports import ensure_daily_report
import os
from nge_trader.services.risk_store import get as risk_get
from nge_trader.services import risk_budget as _RB
from nge_trader.services.growth import compute_growth_plan, apply_growth_to_env
from nge_trader.services.risk import intraday_es_cvar
from nge_trader.services.accounting import recompute_lot_accounting
from fastapi import APIRouter, Body, Request
import json
import pandas as pd
from nge_trader.services.gates import is_trade_viable
from nge_trader.services.viability import compute_viability, get_live_costs
from nge_trader.services import health as _HEALTH
from nge_trader.ai.orchestrator import handle_task
from nge_trader.services import model_session
from scripts.model_canary_control import run_once as canary_control_run
from nge_trader.services.metrics import set_metric_labeled


class EnvPayload(BaseModel):
    data_provider: str | None = None
    broker: str | None = None
    alpha_vantage_api_key: str | None = None


app = FastAPI(title="NGEtrader API", version="0.1.0")
router = APIRouter()


@app.get("/api/config", response_model=EnvPayload)
def get_config() -> EnvPayload:
    settings = Settings()
    return EnvPayload(
        data_provider=settings.data_provider,
        broker=settings.broker,
        alpha_vantage_api_key=settings.alpha_vantage_api_key,
    )


def _write_env(updated: dict[str, str | None]) -> None:
    env_path = Path(".env")
    current = {}
    if env_path.exists():
        current = dotenv_values(str(env_path))
    # Normaliza claves a mayúsculas para consistencia
    normalized = {k.upper(): v for k, v in current.items() if v is not None}
    for k, v in updated.items():
        if v is None:
            continue
        normalized[k.upper()] = v
    lines = [f"{k}={v}" for k, v in normalized.items()]
    env_path.write_text("\n".join(lines), encoding="utf-8")


@app.post("/api/config", response_model=EnvPayload)
def set_config(payload: EnvPayload) -> EnvPayload:
    try:
        _write_env(
            {
                "DATA_PROVIDER": payload.data_provider,
                "BROKER": payload.broker,
                "ALPHA_VANTAGE_API_KEY": payload.alpha_vantage_api_key,
            }
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"No se pudo escribir .env: {exc}")
    return get_config()
@app.get("/api/compliance/status")
def api_compliance_status() -> dict[str, Any]:
    s = Settings()
    return {
        "region": s.compliance_region,
        "require_kyc_aml": bool(s.require_kyc_aml),
        "kyc_aml_completed": bool(s.kyc_aml_completed),
        "algo_validation_required": bool(s.algo_validation_required),
        "algo": {"name": s.algo_name, "version": s.algo_version},
    }


@app.post("/api/compliance/ack")
def api_compliance_ack(kind: str = "kyc_aml", completed: bool = True) -> dict[str, Any]:
    try:
        if kind.lower() == "kyc_aml":
            # persist simple via .env
            from nge_trader.config.env_utils import write_env
            write_env({"KYC_AML_COMPLETED": "True" if completed else "False"})
        return {"status": "ok", "kind": kind, "completed": bool(completed)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


class AlgoValidationPayload(BaseModel):
    algo_name: str
    algo_version: str
    approved_by: str | None = None
    notes: str | None = None


@app.post("/api/compliance/validate_algo")
def api_compliance_validate_algo(p: AlgoValidationPayload) -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    try:
        Database().add_algo_validation(p.algo_name, p.algo_version, p.approved_by, p.notes)
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/compliance/audit_chain/verify")
def api_compliance_audit_chain_verify(limit: int = 1000) -> dict[str, Any]:
    try:
        ok = Database().verify_audit_log(limit=limit)
        return {"status": "ok", "valid": bool(ok)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/backtest/{symbol}")
def api_backtest(symbol: str) -> dict[str, Any]:
    service = AppService()
    df = service.data_provider.get_daily_adjusted(symbol)
    head = df.head().to_dict(orient="records")
    return {"rows": len(df), "head": head}


class PlaceOrderPayload(BaseModel):
    symbol: str
    side: str
    quantity: float
    type: str = "market"  # market|limit|limit_post_only
    price: float | None = None
    tif: str = "GTC"


@app.post("/api/broker/order")
def api_place_order(p: PlaceOrderPayload) -> dict[str, Any]:
    # Enforcement de rol operador
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    service = AppService()
    try:
        if p.type == "market":
            res = place_market_order(service.broker, p.symbol, p.side, p.quantity)
        elif p.type == "limit_post_only":
            if p.price is None:
                raise HTTPException(status_code=400, detail="price requerido para limit_post_only")
            res = place_limit_order_post_only(service.broker, p.symbol, p.side, p.quantity, p.price, p.tif)
        elif p.type == "limit":
            if p.price is None:
                raise HTTPException(status_code=400, detail="price requerido para limit")
            # route directa si broker soporta
            res = service.broker.place_order(symbol=p.symbol, side=p.side, quantity=p.quantity, type_="limit", price=p.price, tif=p.tif)
        else:
            raise HTTPException(status_code=400, detail="type no soportado")
        return {"status": "ok", "order": res}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


class OcoPayload(BaseModel):
    symbol: str
    side: str
    quantity: float
    limit_price: float
    stop_price: float
    stop_limit_price: float | None = None
    tif: str = "GTC"


@app.post("/api/broker/oco")
def api_place_oco(p: OcoPayload) -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    service = AppService()
    try:
        res = place_oco_order_generic(
            service.broker,
            p.symbol,
            p.side,
            p.quantity,
            p.limit_price,
            p.stop_price,
            p.stop_limit_price,
            p.tif,
        )
        return {"status": "ok", "result": res}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/broker/orders/{symbol}")
def api_cancel_all(symbol: str) -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    service = AppService()
    try:
        res = cancel_all_orders_by_symbol(service.broker, symbol)
        return {"status": "ok", "result": res}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/broker/cancel_order")
def api_cancel_order(order_id: str, symbol: str | None = None) -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    service = AppService()
    try:
        res = cancel_order_idempotent(service.broker, order_id, symbol)
        return {"status": "ok", "result": res}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/broker/replace_order")
def api_replace_order(order_id: str, new_price: float, symbol: str | None = None) -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    service = AppService()
    try:
        res = replace_order_limit_price_idempotent(service.broker, order_id, new_price, symbol)
        return {"status": "ok", "result": res}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
def index() -> FileResponse:
    index_path = Path("web/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI no encontrada")
    return FileResponse(str(index_path))


@app.get("/health")
def health() -> dict[str, Any]:
    svc = AppService()
    st = svc.get_connectivity_status()
    return {"ok": any(st.values()), "status": st}

@app.get("/api/health")
def api_health() -> dict[str, Any]:
    svc = AppService()
    status = svc.get_connectivity_status()
    ok = any(bool(v) for v in status.values())
    return {"ok": ok, "status": status}


@app.get("/api/vendors/recommend")
def api_vendors_recommend(kind: str = "data", market: str = "CRYPTO", region: str | None = None, top_k: int = 3) -> dict[str, Any]:
    try:
        recs = recommend_vendors(kind=kind, market=market, region=region, top_k=top_k)
        return {"status": "ok", "recommendations": recs}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/universe/select")
def api_universe_select(payload: dict) -> dict[str, Any]:
    """Selecciona símbolos por coste/edge. Espera lista en payload['candidates']."""
    try:
        cands = list(payload.get("candidates") or [])
        s = Settings()
        from nge_trader.services.vendor_selector import select_universe_by_cost
        sel = select_universe_by_cost(
            candidates=cands,
            expected_edge_bps=float(s.universe_edge_expected_bps or 20.0),
            max_spread_edge_ratio=float(s.universe_max_spread_edge_ratio or 0.5),
            top_k=int(payload.get("top_k") or 10),
        )
        return {"status": "ok", "symbols": [sc.__dict__ for sc in sel]}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics", response_class=PlainTextResponse)
def metrics_text() -> PlainTextResponse:
    # Adjuntar métricas de outbox por estado
    try:
        db = Database()
        counts = db.count_outbox_by_status()
        ages = db.oldest_outbox_age_seconds_by_status()
        prefix = "\n".join([f"idempotency_outbox_size{{status=\"{k}\"}} {v}" for k, v in counts.items()])
        prefix2 = "\n".join([f"idempotency_outbox_oldest_age_seconds{{status=\"{k}\"}} {v}" for k, v in ages.items()])
        body = prefix + ("\n" + prefix2 if prefix2 else "")
        return PlainTextResponse(body + "\n" + export_metrics_text(), media_type="text/plain; version=0.0.4")
    except Exception:
        return PlainTextResponse(export_metrics_text(), media_type="text/plain; version=0.0.4")


@app.post("/api/kill_switch")
def api_kill_switch(armed: bool = True, correlation_id: str | None = None) -> dict[str, Any]:
    from nge_trader.config.settings import Settings  # ensure local binding before first use
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    # Nota: en una app real, esto debería persistirse y respetarse en el enrutamiento de órdenes
    s = Settings()
    # Enforce 2FA si está habilitado
    if bool(s.enforce_2fa_critical):
        raise HTTPException(status_code=428, detail="Se requiere 2FA: use /api/kill_switch_secure")
    s.kill_switch_armed = bool(armed)  # type: ignore[attr-defined]
    try:
        Database().append_audit(datetime.now(UTC).isoformat(), "kill_switch", None, None, f"armed={bool(armed)} cid={correlation_id or ''}")
    except Exception:
        pass
    return {"status": "ok", "armed": bool(armed)}


class TotpPayload(BaseModel):
    code: str


@app.post("/api/2fa/setup")
def api_2fa_setup() -> dict[str, Any]:
    s = Settings()
    ss = SecretStore()
    # Generar secreto si no existe y devolver URI provisioning
    uri = ss.provisioning_uri(account_name=s.account_id, issuer=s.totp_issuer)
    return {"status": "ok", "otpauth_uri": uri}


@app.post("/api/kill_switch_secure")
def api_kill_switch_secure(armed: bool = True, payload: TotpPayload | None = None, correlation_id: str | None = None) -> dict[str, Any]:
    # Rol
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    s = Settings()
    if not bool(s.enforce_2fa_critical):
        raise HTTPException(status_code=400, detail="2FA no está habilitado (ENFORCE_2FA_CRITICAL=false)")
    if payload is None or not str(payload.code).strip():
        raise HTTPException(status_code=400, detail="Debe enviar 'code' TOTP")
    ss = SecretStore()
    if not ss.verify_totp(payload.code):
        raise HTTPException(status_code=401, detail="Código TOTP inválido")
    s.kill_switch_armed = bool(armed)  # type: ignore[attr-defined]
    try:
        Database().append_audit(datetime.now(UTC).isoformat(), "kill_switch_secure", None, None, f"armed={bool(armed)} cid={correlation_id or ''}")
    except Exception:
        pass
    return {"status": "ok", "armed": bool(armed)}


class SecretsPayload(BaseModel):
    secrets: dict[str, str]


@app.post("/api/secrets/rotate")
def api_secrets_rotate(payload: TotpPayload | None = None) -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    s = Settings()
    if bool(s.enforce_2fa_critical):
        if payload is None or not SecretStore().verify_totp(payload.code if payload else ""):
            raise HTTPException(status_code=401, detail="Código TOTP inválido para rotación")
    try:
        SecretStore().rotate_key()
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


# ===== Federated Learning API (sencilla) =====
class FedPayload(BaseModel):
    clients: list[dict]


@app.post("/api/federated/aggregate")
def api_federated_aggregate(p: FedPayload) -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    try:
        agg = ModelRegistry().aggregate_federated(p.clients)
        if not agg:
            raise HTTPException(status_code=400, detail="Entrada inválida")
        return {"status": "ok", "model": agg}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/secrets")
def api_secrets_save(p: SecretsPayload, payload: TotpPayload | None = None) -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    s = Settings()
    if bool(s.enforce_2fa_critical):
        if payload is None or not SecretStore().verify_totp(payload.code if payload else ""):
            raise HTTPException(status_code=401, detail="Código TOTP inválido para guardar")
    try:
        SecretStore().save({str(k): str(v) for k, v in (p.secrets or {}).items()})
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


# ===== Operaciones: procedimientos (reinicio, notas manuales) =====
class ManualNotePayload(BaseModel):
    message: str
    severity: str | None = "INFO"


@app.post("/api/ops/manual_note")
def api_ops_manual_note(p: ManualNotePayload) -> dict[str, Any]:
    try:
        ts = datetime.now(UTC).isoformat()
        Database().append_audit(ts, "manual_note", None, None, p.message)
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ops/restart")
def api_ops_restart_request() -> dict[str, Any]:
    try:
        # Marca un archivo que un supervisor/servicio puede observar para reiniciar
        from pathlib import Path as _Path
        p = _Path("data/restart.request")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
        Database().append_audit(datetime.now(UTC).isoformat(), "restart_requested")
        return {"status": "ok", "requested": True}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/pause/{symbol}")
def api_pause_symbol(symbol: str) -> dict[str, Any]:
    from nge_trader.config.settings import Settings  # ensure local binding
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    s = Settings()
    current = (s.paused_symbols or "").split(",") if s.paused_symbols else []
    up = {x.strip().upper() for x in current if x}
    up.add(symbol.upper())
    Path(".env").write_text(f"PAUSED_SYMBOLS={','.join(sorted(up))}\n", encoding="utf-8")
    return {"status": "ok", "paused": sorted(up)}


@app.post("/api/resume/{symbol}")
def api_resume_symbol(symbol: str) -> dict[str, Any]:
    from nge_trader.config.settings import Settings  # ensure local binding
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    s = Settings()
    current = (s.paused_symbols or "").split(",") if s.paused_symbols else []
    up = {x.strip().upper() for x in current if x}
    up.discard(symbol.upper())
    Path(".env").write_text(f"PAUSED_SYMBOLS={','.join(sorted(up))}\n", encoding="utf-8")
    return {"status": "ok", "paused": sorted(up)}


@app.get("/api/risk_summary")
def api_risk_summary() -> dict[str, Any]:
    svc = AppService()
    acc = svc.get_account_summary()
    return {"balance": acc.get("balance"), "pl_today": acc.get("pl_today"), "pl_open": acc.get("pl_open")}


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    s = Settings()
    paused = (s.paused_symbols or "").split(",") if s.paused_symbols else []
    # Estado del bot: DETENIDO / ARMADO / EN PAPER / EN LIVE
    profile = str(s.profile or "dev").lower()
    if bool(s.kill_switch_armed):
        bot_state = "ARMADO"
    elif profile == "paper":
        bot_state = "EN PAPER"
    elif profile == "live":
        bot_state = "EN LIVE"
    else:
        bot_state = "DETENIDO"
    return {
        "role": s.ui_role,
        "kill_switch_armed": bool(s.kill_switch_armed),
        "paused_symbols": [x.strip().upper() for x in paused if x.strip()],
        "bot_state": bot_state,
        "profile": s.profile,
        "broker": s.broker,
        "data_provider": s.data_provider,
    }


@app.get("/api/summary")
def api_summary() -> dict[str, Any]:
    svc = AppService()
    db = Database()
    st = svc.get_connectivity_status()
    outbox = db.count_outbox_by_status()
    # Métricas recientes desde logs/series (si existen)
    slippage_vals = db.recent_metric_values("slippage_bps", 200)
    last_slip = [v for _, v in slippage_vals[-50:]]
    slippage_avg = float(sum(last_slip) / max(len(last_slip), 1)) if last_slip else 0.0
    # Tomar métricas en memoria cuando apliquen
    slo_err = float(metrics_mod._METRICS_STORE.get("slo_error_rate_recent", 0.0))
    order_lat_last = float(metrics_mod._METRICS_STORE.get("order_place_latency_ms", 0.0))
    # Percentiles de series (si registrados)
    try:
        from nge_trader.services.metrics import get_series_percentile
        order_lat_p95 = get_series_percentile("order_place_latency_ms", 95)
        slip_p95 = float("nan")
    except Exception:
        order_lat_p95 = float("nan")
        slip_p95 = float("nan")
    # Rate limit restante por broker/estrategia
    s = Settings()
    rl_key = f"orders_{s.broker}"
    rl_key_strat = f"orders_{s.broker}_strat_{s.strategy_id}"
    rl = GlobalRateLimiter.get()
    rl_remaining = rl.remaining(rl_key)
    rl_capacity = rl.capacity(rl_key)
    rl_remaining_strat = rl.remaining(rl_key_strat)
    rl_capacity_strat = rl.capacity(rl_key_strat)
    # Presupuesto de órdenes (hoy)
    try:
        sent_today = db.get_strategy_orders_sent_today(s.strategy_id)
    except Exception:
        sent_today = 0
    # Estado de mercado según exchange configurado
    market_open = False
    try:
        market_open = bool(is_session_open_cached(Settings().market_clock_exchange))
    except Exception:
        market_open = False
    # Fill ratio (derivado de contadores en memoria)
    fills_total = float(metrics_mod._METRICS_STORE.get("fills_total", 0.0))
    orders_placed_total = float(metrics_mod._METRICS_STORE.get("orders_placed_total", 0.0))
    fill_ratio = (fills_total / orders_placed_total) if orders_placed_total > 0 else 0.0
    # Deducir símbolo reciente para métricas por símbolo si no hay input explícito
    try:
        last_by_sym = db.last_order_ts_by_symbol()
        top_symbol = max(last_by_sym.items(), key=lambda kv: kv[1])[0] if last_by_sym else None
    except Exception:
        top_symbol = None
    # Métricas L2/VPIN desde LowLatencyProvider
    vpin = 0.0
    l2_ofi = 0.0
    l2_mp = 0.0
    try:
        if top_symbol:
            ll = LowLatencyProvider()
            vp = ll.get_vpin(top_symbol)
            ofi2 = ll.get_l2_ofi(top_symbol)
            mp2 = ll.get_l2_microprice(top_symbol)
            vpin = float(vp) if vp is not None else 0.0
            l2_ofi = float(ofi2 or 0.0)
            l2_mp = float(mp2 or 0.0)
    except Exception:
        pass
    # KPIs y riesgo a partir de equity_curve (si disponible)
    sharpe = float("nan")
    sortino = float("nan")
    mdd = float("nan")
    dd_current = float("nan")
    win_rate = float("nan")
    es_97_5 = float("nan")
    cvar_97_5 = float("nan")
    try:
        eq = db.load_equity_curve()
        if not eq.empty:
            rets = eq.pct_change().dropna().values.tolist()
            sharpe = compute_sharpe(rets)
            sortino = compute_sortino(rets)
            mdd = compute_max_drawdown(eq.values.tolist())
            try:
                from nge_trader.services.metrics import compute_drawdown_series
                dd_series = compute_drawdown_series(eq.values.tolist())
                dd_current = float(dd_series.iloc[-1]) if not dd_series.empty else float("nan")
            except Exception:
                dd_current = float("nan")
            win_rate = compute_win_rate(rets)
            profit_factor = compute_profit_factor(rets)
            skew = compute_skewness(rets)
            kurt = compute_kurtosis(rets)
            try:
                # Usar retornos intradía aproximados desde equity para ES/CVaR (placeholder)
                import pandas as _pd
                ser = _pd.Series(eq.values.tolist()).pct_change().dropna().tail(200)
                _, es_est = intraday_es_cvar(ser, window_minutes=int(getattr(Settings(), "es_window_minutes", 60) or 60), alpha=0.025)
                es_97_5 = float(es_est)
                cvar_97_5 = float(es_est)
            except Exception:
                es_97_5 = float("nan")
                cvar_97_5 = float("nan")
    except Exception:
        pass
    # Budgets de riesgo y R usados/pendientes (aprox)
    try:
        acc = AppService().get_account_summary()
        balance = float(acc.get("balance") or 0.0)
        pl_today = float(acc.get("pl_today") or 0.0)
    except Exception:
        balance = 0.0
        pl_today = 0.0
    risk_per_trade_pct = float(Settings().risk_pct_per_trade)
    daily_loss_budget_pct = float(Settings().max_daily_drawdown_pct)
    r_size_abs = float(risk_per_trade_pct * max(balance, 1e-9))
    risk_used_R = float(abs(pl_today) / max(r_size_abs, 1e-9)) if r_size_abs > 0 else float("nan")
    risk_pending_R = float(max((daily_loss_budget_pct * balance) - abs(pl_today), 0.0) / max(r_size_abs, 1e-9)) if r_size_abs > 0 else float("nan")
    # WS/NTP/Heartbeat expuestos para barra de estado
    ws_states = {}
    try:
        ws_states = get_ws_states_snapshot()
    except Exception:
        ws_states = {}
    ntp_skew_ms = float(metrics_mod._METRICS_STORE.get("time_skew_ms", 0.0))
    hb_latency_ms = float(metrics_mod._METRICS_STORE.get("heartbeat_latency_ms", 0.0))
    # TCA rolling 1d/7d (aprox por timestamp)
    slippage_1d = float("nan")
    slippage_7d = float("nan")
    try:
        import time as _time
        rows = db.recent_metric_values("slippage_bps", 2000)
        now_ts = float(_time.time())
        vals_1d = [float(v) for ts, v in rows if (now_ts - float(ts)) <= 86400]
        vals_7d = [float(v) for ts, v in rows if (now_ts - float(ts)) <= 86400 * 7]
        if vals_1d:
            slippage_1d = float(sum(vals_1d) / len(vals_1d))
        if vals_7d:
            slippage_7d = float(sum(vals_7d) / len(vals_7d))
    except Exception:
        pass
    maker_ratio = float(metrics_mod._METRICS_STORE.get("maker_ratio", float("nan")))
    # Color SLO TCA
    slo_color = "green"
    try:
        if (not (slippage_1d != slippage_1d)) and abs(slippage_1d) > float(Settings().max_slippage_bps):
            slo_color = "red"
        elif order_lat_p95 and order_lat_p95 > 150.0:
            slo_color = "yellow"
        if float(metrics_mod._METRICS_STORE.get("slo_error_rate_recent", 0.0)) > float(Settings().max_error_rate):
            slo_color = "red"
    except Exception:
        pass
    slo_alert = (slo_color == "red")
    # Estado del modelo (champion/canary, PSI y dist. acciones)
    model_version = None
    try:
        models = Database().list_agent_models(limit=1)
        if models:
            model_version = models[0].get("version")
    except Exception:
        model_version = None
    canary_share_pct = float(getattr(Settings(), "canary_traffic_pct", 0.0) or 0.0)
    psi_features = float(metrics_mod._METRICS_STORE.get("model_drift_psi", float("nan")))
    action_dist = {
        "buy": float(metrics_mod._METRICS_STORE.get("action_dist_buy", float("nan"))),
        "sell": float(metrics_mod._METRICS_STORE.get("action_dist_sell", float("nan"))),
        "none": float(metrics_mod._METRICS_STORE.get("action_dist_none", float("nan"))),
    }
    # Series mini-gráfico (últimos 30 puntos)
    try:
        from nge_trader.services.metrics import _SERIES as _S
        train_loss = list(_S.get("train_loss", []))[-30:]
        train_entropy = list(_S.get("train_entropy", []))[-30:]
    except Exception:
        train_loss = []
        train_entropy = []
    # Adjuntar métricas de WS uptime/reconnects al summary en un bloque compacto
    try:
        ws_states_export = {}
        for ws_name, st in ws_states.items():
            # uptime aproximado desde export text ya manejado, aquí exportamos el snapshot
            ws_states_export[ws_name] = {
                "connected": bool(st.get("connected")),
                "reconnects": int(st.get("reconnects", 0)),
            }
    except Exception:
        ws_states_export = {}
    # Lectura de pausa global de entradas (flag de archivo)
    try:
        from pathlib import Path as _Path
        entries_paused = _Path("data/pause_entries.flag").exists()
    except Exception:
        entries_paused = False
    return {
        "connectivity": st,
        "outbox": outbox,
        "market_open": market_open,
        "session": session_details(Settings().market_clock_exchange),
        "ws": ws_states,
        "ws_basic": ws_states_export,
        "ntp_skew_ms": ntp_skew_ms,
        "heartbeat_latency_ms": hb_latency_ms,
        "entries_paused": bool(entries_paused),
        "strategy_id": s.strategy_id,
        "thresholds": {
            "max_error_rate": float(s.max_error_rate),
            "max_slippage_bps": float(s.max_slippage_bps),
            "drift_psi_threshold": float(s.drift_psi_threshold),
        },
        "metrics": {
            "slippage_bps_avg": float(slippage_avg),
            "order_place_latency_ms_last": float(order_lat_last),
            "order_place_latency_ms_p95": float(order_lat_p95),
            "slo_error_rate_recent": float(slo_err),
            "model_drift_psi": float(metrics_mod._METRICS_STORE.get("model_drift_psi", 0.0)),
            "fills_total": float(metrics_mod._METRICS_STORE.get("fills_total", 0.0)),
            "breaker_activations_total": float(metrics_mod._METRICS_STORE.get("breaker_activations_total", 0.0)),
            "fill_ratio": float(fill_ratio),
            "slippage_bps_p95": float(slip_p95),
            "rate_limit_broker_remaining": float(rl_remaining),
            "rate_limit_broker_capacity": float(rl_capacity),
            "rate_limit_strategy_remaining": float(rl_remaining_strat),
            "rate_limit_strategy_capacity": float(rl_capacity_strat),
            "rate_limit_symbol_capacity_per_minute": float(s.symbol_orders_per_minute),
            # Compatibilidad hacia atrás
            "rate_limit_remaining": float(rl_remaining),
            "rate_limit_capacity": float(rl_capacity),
            "orders_sent_today_strategy": float(sent_today),
            # Budgets agregados
            "orders_sent_today_symbol": float(db.get_symbol_orders_sent_today(top_symbol or "")),
            "orders_sent_today_account": float(db.get_account_orders_sent_today(s.account_id)),
            # L2/VPIN (si hay datos)
            "vpin": float(vpin),
            "l2_ofi": float(l2_ofi),
            "l2_microprice": float(l2_mp),
        },
        "kpis": {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": mdd,
            "dd_current": dd_current,
            "win_rate": win_rate,
            "profit_factor": profit_factor if 'profit_factor' in locals() else float('nan'),
            "skewness": skew if 'skew' in locals() else float('nan'),
            "kurtosis": kurt if 'kurt' in locals() else float('nan'),
        },
        "risk": {
            "dd_intraday_pct": dd_current,
            "maxdd_rolling": mdd,
            "risk_per_trade_pct": risk_per_trade_pct,
            "daily_loss_budget_pct": daily_loss_budget_pct,
            "risk_used_R": risk_used_R,
            "risk_pending_R": risk_pending_R,
            "es_97_5": es_97_5,
            "cvar_97_5": cvar_97_5,
        },
        "tca": {
            "slippage_bps_1d": slippage_1d,
            "slippage_bps_7d": slippage_7d,
            "fill_ratio": float(fill_ratio),
            "maker_ratio": maker_ratio,
            "order_place_latency_ms_p95": float(order_lat_p95),
            "slo_color": slo_color,
            "slo_alert": bool(slo_alert),
        },
        "model": {
            "model_version": model_version,
            "canary_share_pct": canary_share_pct,
            "psi_features": psi_features,
            "action_dist_buy_sell_none": action_dist,
            "loss_series": train_loss,
            "entropy_series": train_entropy,
        },
    }


@app.post("/api/pause_entries")
def api_pause_entries(correlation_id: str | None = None) -> dict[str, Any]:
    try:
        from pathlib import Path as _Path
        p = _Path("data/pause_entries.flag")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
        Database().append_audit(datetime.now(UTC).isoformat(), "pause_entries", None, None, f"cid={correlation_id or ''}")
        return {"status": "ok", "paused": True}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/resume_entries")
def api_resume_entries(correlation_id: str | None = None) -> dict[str, Any]:
    try:
        from pathlib import Path as _Path
        p = _Path("data/pause_entries.flag")
        if p.exists():
            p.unlink()
        Database().append_audit(datetime.now(UTC).isoformat(), "resume_entries", None, None, f"cid={correlation_id or ''}")
        return {"status": "ok", "paused": False}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/history_7d")
def api_history_7d() -> dict[str, Any]:
    """Series de últimos 7 días: slippage_bps (avg por día), fill_ratio por día, Sharpe y MaxDD 7d."""
    try:
        import time as _time
        from collections import defaultdict
        db = Database()
        now_ts = float(_time.time())
        # Slippage por día
        rows = db.recent_metric_values("slippage_bps", 20000)
        by_day_vals: dict[str, list[float]] = defaultdict(list)
        for ts, v in rows:
            if (now_ts - float(ts)) <= 86400 * 7:
                day = datetime.fromtimestamp(float(ts), UTC).strftime("%Y-%m-%d")
                by_day_vals[day].append(float(v))
        slippage_avg_by_day = [{"day": k, "avg_slippage_bps": (sum(vs) / len(vs) if vs else 0.0)} for k, vs in sorted(by_day_vals.items())]
        # Fill ratio por día (órdenes vs fills del día)
        fills = [r for r in db.recent_fills(10000) if r.get("ts")]
        try:
            orders = [r for r in db.recent_orders(10000) if r.get("ts")]
        except Exception:
            orders = []
        day_to_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"fills": 0, "orders": 0})
        for f in fills:
            try:
                d = str(f.get("ts")).split("T")[0]
                day_to_counts[d]["fills"] += 1
            except Exception:
                continue
        for o in orders:
            try:
                d = str(o.get("ts")).split("T")[0]
                day_to_counts[d]["orders"] += 1
            except Exception:
                continue
        days_sorted = sorted([d for d in day_to_counts.keys() if d >= datetime.now(UTC).strftime("%Y-%m-%d")[:0]])
        fill_ratio_by_day = []
        for d in sorted(day_to_counts.keys()):
            if len(fill_ratio_by_day) >= 7:
                break
            c = day_to_counts[d]
            fr = (c["fills"] / c["orders"]) if c["orders"] > 0 else None
            fill_ratio_by_day.append({"day": d, "fill_ratio": fr})
        # KPIs 7d desde equity
        sharpe_7d = float("nan")
        maxdd_7d = float("nan")
        try:
            eq = db.load_equity_curve()
            if not eq.empty:
                vals = [float(x) for x in eq.values.tolist()][-7:]
                if len(vals) >= 3:
                    import pandas as _pd
                    rets = _pd.Series(vals).pct_change().dropna().values.tolist()
                    sharpe_7d = compute_sharpe(rets)
                    maxdd_7d = compute_max_drawdown(vals)
        except Exception:
            pass
        return {
            "status": "ok",
            "slippage_bps_avg_by_day": slippage_avg_by_day,
            "fill_ratio_by_day": fill_ratio_by_day,
            "sharpe_7d": sharpe_7d,
            "maxdd_7d": maxdd_7d,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/equity_curve")
def api_equity_curve(benchmark_symbol: str | None = "SPY") -> dict[str, Any]:
    """Devuelve curva de capital y, si es posible, benchmark y banda de MaxDD.

    benchmark_symbol: símbolo de referencia; si no disponible, se omite.
    """
    try:
        db = Database()
        eq = db.load_equity_curve()
        if eq.empty:
            return {"status": "empty"}
        eq_vals = [float(x) for x in eq.values.tolist()]
        dd_series = []
        try:
            from nge_trader.services.metrics import compute_drawdown_series
            dd = compute_drawdown_series(eq_vals)
            dd_series = [float(x) for x in dd.values.tolist()]
        except Exception:
            dd_series = []
        bench = []
        try:
            if benchmark_symbol:
                svc = AppService()
                dfb = svc.data_provider.get_daily_adjusted(benchmark_symbol)
                if dfb is not None and not dfb.empty:
                    bench = [float(x) for x in dfb["close"].astype(float).tail(len(eq_vals)).values.tolist()]
        except Exception:
            bench = []
        return {"status": "ok", "equity": eq_vals, "maxdd_band": dd_series, "benchmark": bench, "symbol": benchmark_symbol}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/oms/executability")
def api_oms_executability(symbol: str, side: str, edge_bps: float | None = None) -> dict[str, Any]:
    try:
        res = _OMS.compute_executability(symbol, side, edge_bps)
        edge = float(res.get("edge_bps", 0.0))
        thr = float(res.get("threshold_bps", 0.0))
        trade_viable = bool(edge >= thr)
        # Placeholder de min_notional_ok (verdadero si no hay información)
        min_notional_ok = True
        return {"ok": True, **res, "trade_viable": trade_viable, "edge_neto_bps": float(edge - thr), "min_notional_ok": bool(min_notional_ok)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/oms/executability/global")
def api_oms_executability_global(symbols: str, side: str = "buy", buffer_bps: float | None = None, top_k: int = 5) -> dict[str, Any]:
    """Agrega semáforo global: calcula score neto por símbolo y promedio de viables.

    Params:
      - symbols: CSV de símbolos candidatos
      - side: buy/sell
      - buffer_bps: opcional, sobreescribe Settings.edge_buffer_bps
      - top_k: top-K por edge estimado (si no hay edge, usa 0)
    """
    try:
        s = Settings()
        if buffer_bps is not None:
            setattr(s, "edge_buffer_bps", float(buffer_bps))  # type: ignore[attr-defined]
        sym_list = [x.strip().upper() for x in (symbols or "").split(",") if x.strip()]
        results: list[dict] = []
        for sym in sym_list:
            try:
                res = _OMS.compute_executability(sym, side, None)
                net = float(res.get("edge_bps", 0.0)) - float(res.get("threshold_bps", 0.0))
                results.append({"symbol": sym, "level": res.get("level"), "edge_bps": float(res.get("edge_bps", 0.0)), "threshold_bps": float(res.get("threshold_bps", 0.0)), "net_bps": net})
            except Exception:
                results.append({"symbol": sym, "level": "red", "edge_bps": 0.0, "threshold_bps": 0.0, "net_bps": -999.0})
        # top-K por net_bps
        results_sorted = sorted(results, key=lambda x: float(x.get("net_bps", 0.0)), reverse=True)[: max(int(top_k), 1)]
        viables = [r for r in results_sorted if float(r.get("net_bps", -1.0)) >= 0.0]
        pct_viables = (len(viables) / max(len(results_sorted), 1)) if results_sorted else 0.0
        avg_net = float(sum([float(r.get("net_bps", 0.0)) for r in viables]) / max(len(viables), 1)) if viables else float(min([float(r.get("net_bps", -999.0)) for r in results_sorted], default=-999.0))
        # Colores: verde (≥0), ámbar (−2..0), rojo (<−2)
        if avg_net >= 0.0 and pct_viables >= 0.5:
            color = "green"
        elif avg_net >= -2.0 and pct_viables >= 0.2:
            color = "yellow"
        else:
            color = "red"
        return {"ok": True, "global_color": color, "avg_net_bps": avg_net, "pct_viables": pct_viables, "symbols": results_sorted}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/can_start")
def api_can_start(symbols: str, side: str = "buy", ntp_skew_limit_ms: float = 100.0) -> dict[str, Any]:
    """Gating para botón "Iniciar": requiere mercado OK, NTP OK, WS OK y semáforo global no rojo.

    symbols: CSV con top-K símbolos a evaluar para el semáforo global.
    """
    try:
        s = Settings()
        # Market clock
        if not is_session_open_cached(s.market_clock_exchange):
            return {"can_start": False, "reason": "market-clock=closed"}
        # NTP skew
        skew_ms = float(metrics_mod._METRICS_STORE.get("time_skew_ms", 0.0))
        if skew_ms > float(ntp_skew_limit_ms):
            return {"can_start": False, "reason": "time_skew_ms>limit"}
        # WS health (al menos un WS conectado)
        try:
            ws_states = get_ws_states_snapshot()
            any_ws = any(bool(st.get("connected")) for st in ws_states.values())
        except Exception:
            any_ws = True
        if not any_ws:
            return {"can_start": False, "reason": "ws_down"}
        # Semáforo global
        sym_list = [x.strip().upper() for x in (symbols or "").split(",") if x.strip()]
        if not sym_list:
            return {"can_start": False, "reason": "no_symbols"}
        res = api_oms_executability_global(",".join(sym_list), side)
        if str(res.get("global_color")) == "red":
            return {"can_start": False, "reason": f"semaforo_global=rojo ({res.get('bottleneck_msg')})"}
        return {"can_start": True}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/growth/plan")
def api_growth_plan() -> dict[str, Any]:
    try:
        plan = compute_growth_plan()
        return {"status": "ok", "plan": plan}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/growth/apply")
def api_growth_apply() -> dict[str, Any]:
    try:
        plan = compute_growth_plan()
        res = apply_growth_to_env(plan)
        return {"status": "ok", "applied": res}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/alert/test")
def api_alert_test(msg: str = "Prueba de alerta NGEtrader") -> dict[str, Any]:
    try:
        Notifier().send(str(msg))
        return {"status": "sent"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/models/registry")
def api_models_registry(limit: int = 50) -> dict[str, Any]:
    try:
        db = Database()
        rows = db.list_agent_models(limit=limit)
        return {"status": "ok", "models": rows}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/config/snapshot")
def api_config_snapshot() -> dict[str, Any]:
    try:
        s = Settings()
        snapshot = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": {
                "strategy_id": s.strategy_id,
                "broker": s.broker,
                "data_provider": s.data_provider,
                "online_training_enabled": bool(getattr(s, "online_training_enabled", True)),
                "online_lookback": int(getattr(s, "online_lookback", 30)),
                "online_learning_rate": float(getattr(s, "online_learning_rate", 0.01)),
                "online_l2_reg": float(getattr(s, "online_l2_reg", 1e-4)),
                "min_recent_sharpe": getattr(s, "min_recent_sharpe", None),
                "recent_sharpe_lookback": int(getattr(s, "recent_sharpe_lookback", 20)),
                "drift_psi_threshold": float(getattr(s, "drift_psi_threshold", 0.2)),
                "canary_traffic_pct": float(getattr(s, "canary_traffic_pct", 0.1)),
            },
            "models": Database().list_agent_models(limit=20),
        }
        from pathlib import Path as _Path
        p = _Path("reports/config_snapshot.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(__import__("json").dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "ok", "snapshot": snapshot}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


class FeePayload(BaseModel):
    exchange: str
    symbol: str
    maker_bps: float
    taker_bps: float
    tier: str | None = "default"
    effective_at: str | None = None


@app.post("/api/fees_schedule")
def api_set_fee_schedule(p: FeePayload) -> dict[str, Any]:
    try:
        db = Database()
        db.set_fee_schedule(p.exchange, p.symbol, p.maker_bps, p.taker_bps, tier=p.tier or "default", effective_at_iso=p.effective_at)
        row = db.get_fee_schedule_any(p.exchange, p.symbol)
        return {"status": "ok", "row": row}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/fees_schedule/{exchange}/{symbol}")
def api_get_fee_schedule(exchange: str, symbol: str) -> dict[str, Any]:
    try:
        db = Database()
        row = db.get_fee_schedule_any(exchange, symbol)
        if not row:
            raise HTTPException(status_code=404, detail="No encontrado")
        return {"status": "ok", "row": row}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


class TwoFAPayload(BaseModel):
    code: str


@app.post("/api/2fa/setup")
def api_2fa_setup(secret_b32: str | None = None, account: str = "operator", issuer: str = "NGEtrader") -> dict[str, Any]:
    ss = SecretStore()
    if secret_b32:
        ss.set_totp_secret(secret_b32)
        secret = secret_b32
    else:
        secret = ss.generate_totp_secret()
    uri = ss.provisioning_uri(account_name=account, issuer=issuer)
    return {"status": "ok", "secret": secret, "otpauth_uri": uri}


@app.post("/api/2fa/verify")
def api_2fa_verify(p: TwoFAPayload) -> dict[str, Any]:
    ss = SecretStore()
    ok = ss.verify_totp(p.code)
    if not ok:
        raise HTTPException(status_code=401, detail="Código 2FA inválido")
    return {"status": "ok"}


# ===== Minimal kit: analysis-task, intent-preview, ui-summary =====
@router.post("/analysis-task")
def analysis_task(task: dict = Body(...)) -> dict[str, Any]:
    return handle_task(task)


@router.post("/intent/preview")
def intent_preview(request: Request, state: dict = Body(...)) -> dict[str, Any]:
    edge = float(state.get("edge_bps", 0.0) or 0.0)
    fees = state.get("fees_bps")
    hs = state.get("half_spread_bps")
    if fees is None or hs is None:
        c = get_live_costs(str(state.get("symbol", "")))
        fees = c.get("fees_bps")
        hs = c.get("half_spread_bps")
    ok = is_trade_viable(edge, float(fees), float(hs))
    resp = {
        "symbol": state.get("symbol"),
        "decision": "trade" if ok else "no-trade",
        "edge": edge,
        "costs_bps": float(fees) + float(hs) + 2.0,
    }
    # Auditoría
    try:
        corr = request.headers.get("X-Correlation-ID")
        Database().append_audit(pd.Timestamp.utcnow().isoformat(), "kit_intent_preview", None, state.get("symbol"), json.dumps({"req": state, "resp": resp, "corr": corr}))
    except Exception:
        pass
    return resp


SYMBOLS_UNIVERSE = ["BTCUSDT", "ETHUSDT"]  # será override desde UNIVERSE_FILE si existe


@router.get("/ui/summary")
def ui_summary(request: Request) -> dict[str, Any]:
    # Cargar universo desde archivo si está configurado
    try:
        import os
        s = Settings()
        path = os.getenv("UNIVERSE_FILE", None) or getattr(s, "universe_file", None)
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                syms = [ln.strip().upper() for ln in fh.readlines() if ln.strip() and not ln.strip().startswith("#")]
                if syms:
                    universe = syms
                else:
                    universe = SYMBOLS_UNIVERSE
        else:
            universe = SYMBOLS_UNIVERSE
    except Exception:
        universe = SYMBOLS_UNIVERSE
    vs = compute_viability(universe, buffer_bps=2.0)
    color = "green" if any(v.get("trade_viable") for v in vs.values()) else "red"
    out = {
        "market_open": _HEALTH.market_clock_open(),
        "time_skew_ms": _HEALTH.time_skew_ms(),
        "ws_uptime_seconds": 36_000,
        "ws_reconnects_total": 1,
        "viability": vs,
        "viability_global": color,
    }
    try:
        corr = request.headers.get("X-Correlation-ID")
        Database().append_audit(pd.Timestamp.utcnow().isoformat(), "kit_ui_summary", None, ",".join(list(vs.keys())), json.dumps({"resp": out, "corr": corr}))
    except Exception:
        pass
    return out


@router.get("/preflight")
def preflight(request: Request) -> dict[str, Any]:
    checks = {
        "market_open": _HEALTH.market_clock_open(),
        "time_skew_ms": _HEALTH.time_skew_ms(),
        "ws_ok": _HEALTH.ws_ok(),
        "idempotency_ok": _HEALTH.idempotency_ok(),
        "budgets_loaded": _HEALTH.budgets_loaded(),
    }
    checks["ok"] = (
        bool(checks["market_open"]) and bool(checks["ws_ok"]) and bool(checks["idempotency_ok"]) and bool(checks["budgets_loaded"]) and int(checks["time_skew_ms"]) <= 100
    )
    try:
        corr = request.headers.get("X-Correlation-ID")
        Database().append_audit(pd.Timestamp.utcnow().isoformat(), "kit_preflight", None, None, json.dumps({"checks": checks, "corr": corr}))
    except Exception:
        pass
    return checks


@router.get("/tca-summary")
def tca_summary() -> dict[str, Any]:
    """Resumen básico de TCA desde métricas en memoria/DB.

    Retorna por símbolo y global: slippage_bps, fill_ratio, maker_ratio, p95_place_ms
    """
    db = Database()
    from nge_trader.services import metrics as _MM
    # Globales
    try:
        from nge_trader.services.metrics import get_series_percentile
        p95 = float(get_series_percentile("order_place_latency_ms", 95))
    except Exception:
        p95 = float("nan")
    fills_total = float(_MM._METRICS_STORE.get("fills_total", 0.0))
    orders_placed_total = float(_MM._METRICS_STORE.get("orders_placed_total", 0.0))
    fill_ratio_global = (fills_total / orders_placed_total) if orders_placed_total > 0 else float("nan")
    maker_ratio_global = float(_MM._METRICS_STORE.get("maker_ratio", float("nan")))
    # Slippage global promedio reciente
    try:
        rows = db.recent_metric_values("slippage_bps", 500)
        slip_vals = [float(v) for _, v in rows[-100:]]
        slip_global = float(sum(slip_vals) / len(slip_vals)) if slip_vals else float("nan")
    except Exception:
        slip_global = float("nan")
    # Por símbolo (usar fills recientes)
    per_symbol: dict[str, dict[str, float]] = {}
    try:
        fills = db.recent_fills(limit=200)
        from collections import defaultdict
        agg = defaultdict(lambda: {"sum_bps": 0.0, "n": 0})
        for f in fills:
            sym = (f.get("symbol") or "").upper()
            # no tenemos slippage por fill en todas, se usa métrica global si no
            # aquí solo contamos fills para fill_ratio por símbolo (si quisiéramos)
            agg[sym]["n"] += 1
        for sym, a in agg.items():
            per_symbol[sym] = {
                "slippage_bps": float("nan"),
                "fill_ratio": float("nan"),
                "maker_ratio": float("nan"),
                "p95_place_ms": p95,
            }
    except Exception:
        pass
    return {
        "global": {
            "slippage_bps": slip_global,
            "fill_ratio": fill_ratio_global,
            "maker_ratio": maker_ratio_global,
            "p95_place_ms": p95,
        },
        "per_symbol": per_symbol,
    }

app.include_router(router, prefix="/api/kit")
@router.post("/disarm")
def kit_disarm(request: Request) -> dict[str, Any]:
    # Desarmar kill switch
    try:
        _write_env({"KILL_SWITCH_ARMED": "1"})
    except Exception:
        pass
    out = disarm_and_cancel_if_config()
    try:
        corr = request.headers.get("X-Correlation-ID")
        Database().append_audit(pd.Timestamp.utcnow().isoformat(), "kit_disarm", None, None, json.dumps({"auto_cancel": out, "corr": corr}))
    except Exception:
        pass
    import time as _t
    return {"ok": True, "ts": int(_t.time()), "kill": True, "auto_cancel": out}


@router.get("/report/today")
def report_today() -> Any:
    ymd = date.today().strftime("%Y-%m-%d")
    path = ensure_daily_report(ymd)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "reason": "report_not_found"}, status_code=404)
    return FileResponse(path, media_type="application/zip", filename=f"report_{ymd}.zip")


@router.post("/model/pin")
def model_pin(body: dict = Body(...)) -> dict[str, Any]:
    model_session.pin(body["model_id"], body.get("scaler_id"), body.get("features_hash"))
    return {"ok": True, **model_session.get()}


@router.post("/model/unpin")
def model_unpin() -> dict[str, Any]:
    model_session.unpin()
    return {"ok": True}


@router.get("/risk")
def kit_risk() -> dict[str, Any]:
    data = risk_get()
    try:
        used, left = _RB.get_today()
        data.update({"used_R": float(used), "daily_budget_left": float(left)})
    except Exception:
        pass
    return data


@router.post("/risk/reset")
def kit_risk_reset() -> dict[str, Any]:
    try:
        _RB.reset_today()
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/ready")
def ready() -> dict[str, Any]:
    try:
        ok = True
        # rehydrate_seconds expuesta
        from nge_trader.services.metrics import _rehydrate_gauge  # type: ignore
        rh = True if _rehydrate_gauge is not None else False
        # budgets_loaded vía health
        bud = _HEALTH.budgets_loaded()
        ok = bool(rh and bud)
        return {"ok": ok, "rehydrate_done": rh, "budgets_loaded": bud}
    except Exception:
        return {"ok": False, "rehydrate_done": False, "budgets_loaded": False}


@router.get("/state")
def state() -> dict[str, Any]:
    from nge_trader.services.model_session import get as ms_get
    return {"pinned": ms_get(), "risk": risk_get(), "kill_switch": bool(Settings().kill_switch_armed), "mode": Settings().profile}


@router.get("/model/candidates")
def model_candidates(limit: int = 20) -> dict[str, Any]:
    try:
        rows = Database().list_agent_models(limit=limit)
        return {"status": "ok", "candidates": rows}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/model/auto-control")
def model_auto_control(enable: bool | None = None) -> dict[str, Any]:
    # Simple: si enable es None, ejecuta una vez; si es True, responde stub (cron externo debería ejecutarlo periódicamente)
    try:
        if enable is None:
            res = canary_control_run()
            return {"status": "ok", "run": res}
        # Persistencia del flag puede agregarse en settings/env si se requiere
        return {"status": "ok", "enabled": bool(enable)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/model/canary_share")
def model_canary_share(pct: float) -> dict[str, Any]:
    try:
        s = Settings()
        setattr(s, "canary_traffic_pct", float(pct))  # type: ignore[attr-defined]
        Database().append_audit(datetime.now(UTC).isoformat(), "canary_share_updated", None, None, f"pct={pct}")
        return {"status": "ok", "pct": float(pct)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
@router.get("/slo")
def kit_slo(symbol: str | None = None) -> dict[str, Any]:
    from nge_trader.services.metrics import get_series_percentile
    s = Settings()
    db = Database()
    # Global
    try:
        p95_global = float(get_series_percentile("order_place_latency_ms", 95))
    except Exception:
        p95_global = float('nan')
    try:
        rows = db.recent_metric_values("slippage_bps", 2000)
        import time as _time
        now_ts = float(_time.time())
        vals_7d = [float(v) for ts, v in rows if (now_ts - float(ts)) <= 86400*7]
        slip_global = float(sum(vals_7d)/len(vals_7d)) if vals_7d else float('nan')
    except Exception:
        slip_global = float('nan')
    err_rate_global = 0.0
    gate_global = (abs(slip_global) > float(s.slo_slippage_bps)) or (p95_global > float(s.slo_p95_ms)) or (err_rate_global > float(s.slo_error_rate))
    try:
        set_metric_labeled("gate_slo_active", 1.0 if gate_global else 0.0, {"symbol": "_GLOBAL"})
    except Exception:
        pass
    # Por símbolo (si se solicita o todo el universo usado en summary)
    syms = []
    if symbol:
        syms = [symbol.upper()]
    else:
        try:
            import os
            path = os.getenv("UNIVERSE_FILE", None) or getattr(s, "universe_file", None)
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    syms = [ln.strip().upper() for ln in fh if ln.strip() and not ln.strip().startswith('#')]
        except Exception:
            syms = []
    per_symbol: dict[str, dict[str, float | bool | str]] = {}
    for sym in syms:
        try:
            rows = db.recent_metric_values(f"slippage_bps_{sym}", 2000)
            import time as _time
            now_ts = float(_time.time())
            vals_7d = [float(v) for ts, v in rows if (now_ts - float(ts)) <= 86400*7]
            slip_sym = float(sum(vals_7d)/len(vals_7d)) if vals_7d else slip_global
        except Exception:
            slip_sym = slip_global
        try:
            from nge_trader.services.metrics import get_series_percentile
            p95_sym = float(get_series_percentile(f"order_place_latency_ms_{sym}", 95))
        except Exception:
            p95_sym = p95_global
        err_rate_sym = err_rate_global
        gate = (abs(slip_sym) > float(s.slo_slippage_bps)) or (p95_sym > float(s.slo_p95_ms)) or (err_rate_sym > float(s.slo_error_rate))
        try:
            set_metric_labeled("gate_slo_active", 1.0 if gate else 0.0, {"symbol": sym})
        except Exception:
            pass
        try:
            from nge_trader.services.slo import get_symbol_slo as _get_slo
            slo = _get_slo(sym)
            src = str(slo.get("source", "unknown"))
        except Exception:
            src = "unknown"
        per_symbol[sym] = {"slippage_bps": slip_sym, "error_rate": err_rate_sym, "p95_ms": p95_sym, "gate": bool(gate), "source": src}
    return {"global": {"slippage_bps": slip_global, "error_rate": err_rate_global, "p95_ms": p95_global, "gate": bool(gate_global)}, "per_symbol": per_symbol}


@router.post("/arm")
def kit_arm() -> dict[str, Any]:
    # Marca estado ARMADO y emite gate_budget_active según presupuesto
    try:
        used, left = _RB.get_today()
        set_metric_labeled("gate_budget_active", 1.0 if float(left) <= 0.0 else 0.0, {"symbol": "_GLOBAL"})
    except Exception:
        pass
    from nge_trader.config.settings import Settings as _S
    s = _S()
    s.kill_switch_armed = True  # type: ignore[attr-defined]
    return {"ok": True, "armed": True}


@router.post("/slo/refresh")
def kit_slo_refresh() -> dict[str, Any]:
    try:
        from scripts.update_slo_from_tca import main as _refresh_main
        _refresh_main()
        return {"ok": True, "refreshed": True}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"No se pudo refrescar SLO: {exc}")


@app.post("/api/kill_switch_secure")
def api_kill_switch_secure(armed: bool, code: str) -> dict[str, Any]:
    ss = SecretStore()
    if not ss.verify_totp(code):
        raise HTTPException(status_code=401, detail="Código 2FA inválido")
    from nge_trader.config.settings import Settings
    s = Settings()
    s.kill_switch_armed = bool(armed)  # type: ignore[attr-defined]
    return {"status": "ok", "armed": bool(armed)}


@app.post("/api/broker/cancel_all_secure")
def api_cancel_all_secure(symbol: str, code: str, correlation_id: str | None = None) -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    ss = SecretStore()
    if bool(Settings().enforce_2fa_critical) and not ss.verify_totp(code):
        raise HTTPException(status_code=401, detail="Código 2FA inválido")
    service = AppService()
    try:
        res = cancel_all_orders_by_symbol(service.broker, symbol)
        try:
            Database().append_audit(datetime.now(UTC).isoformat(), "cancel_all", None, symbol, f"cid={correlation_id or ''}")
        except Exception:
            pass
        return {"status": "ok", "result": res}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


# ===== Canary / Model Registry =====
@app.post("/api/model/publish")
def api_model_publish(version: str, path: str, config_json: str, metrics_json: str | None = None) -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    ModelRegistry().publish(version, path, config_json, metrics_json)
    return {"status": "ok"}


@app.post("/api/model/promote_canary")
def api_model_promote_canary() -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    # Ventana segura de despliegue
    try:
        s = Settings()
        now = datetime.now(UTC)
        hhmm = now.strftime("%H:%M")
        if not (s.deploy_window_start <= hhmm <= s.deploy_window_end):
            raise HTTPException(status_code=423, detail="Fuera de ventana segura de despliegue")
    except HTTPException:
        raise
    except Exception:
        pass
    ModelRegistry().promote_canary()
    return {"status": "ok"}


@app.post("/api/model/rollback")
def api_model_rollback() -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    ModelRegistry().rollback()
    return {"status": "ok"}


@app.post("/api/secrets/rotate")
def api_rotate_secrets() -> dict[str, Any]:
    if str(Settings().ui_role).lower() != "operator":
        raise HTTPException(status_code=403, detail="Operación no permitida para este rol")
    try:
        SecretStore().rotate_key()
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))

