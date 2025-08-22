# Runbooks Go-Live Canario

## Congelación y arranque (T-1)
- Fijar `requirements.txt` y tag git de modelos.
- Backup DB:
```
python scripts/backup_db.py
```
- Verificar auditoría:
```
python scripts/verify_audit_chain.py
```
- Entorno live (copiar `.env.live.example` a `.env` y ajustar): `PROFILE=live`, `ENABLE_LIVE_TRADING=true`, `ENABLE_ONLINE_LEARNING=false`.
- Preflight obligatorio:
```
curl http://localhost:8000/api/kit/preflight
```

## Seeds
- Reset budget:
```
curl -X POST http://localhost:8000/api/kit/risk/reset
```
- Pin de modelo (opcional):
```
curl -X POST http://localhost:8000/api/kit/model/pin -H "Content-Type: application/json" -d '{"model_id":"champion_YYYYMMDD"}'
```

## Ramp Canario (T0→T+48h)
- 0–8h: `pct=0.05`, si OK → 0.10 (T+8h) → 0.20 (T+24h).
```
curl -X PATCH "http://localhost:8000/api/kit/model/canary_share?pct=0.05"
```
- KPIs: `/api/kit/slo`, `/api/kit/slo?symbol=SYM`, `/api/kit/risk`, `/api/kit/slo/refresh`.
- Gates: SLO, drift PSI, budget, preflight, kill-switch.

## Límites Live iniciales
- `RISK_PER_TRADE_PCT=0.0015–0.0020`, `MAX_OPEN_POSITIONS=1–2`, `MAX_TRADES_PER_DAY=5–8`.
- Daily loss cap: `MAX_DAILY_DRAWDOWN_PCT=0.01`.
- Canary hard cap: notional canario ≤ 25% diario.

## Alertas
- BudgetDepleted: `gate_budget_active==1 OR daily_budget_left<=0`.
- SLOGateStuck: `sum_over_time(gate_slo_active[30m]) > 25`.
- CanaryImbalance: `abs((canary_trades_total/total_trades) - CHALLENGER_SHARE) > 0.05`.
- Data-quality: `features_missing_ratio>1%`, `feed_dislocation_bps>5bps`, `data_stale_seconds>1s`.

## UI mínima live
- Panel Estado Live: `model_id_used`, `pinned`, `canary_share`, `budget_left`, gates activos.
- Botones con doble confirmación: Arm/Disarm, Promote/Rollback, Risk Reset.

## Reporte diario
- `/api/kit/report/today` devuelve ZIP con `orders.csv`, `fills.csv`, `pnl.csv`, `tca.csv`, `metadata.json`.
- Verificador de audit log: `scripts/verify_audit_chain.py`.

## Smoke final
- `/api/kit/preflight` → ok:true; `/api/kit/arm` → ok:true.
- `/api/kit/model/canary_share?pct=0.05` → 200; `/api/kit/slo?symbol=BTCUSDT` → gate:false.
- `/api/kit/risk` → budget ~1.0 al inicio; decrece tras trades.
- PSI>0.3 → no-trade; 0.2<PSI≤0.3 → size halved (log).
- `/api/kit/report/today` → ZIP real.
- Grafana TCA: sin alertas críticas activas.

## Go/No-Go (T+48h)
- error_rate<1%, p95_place_ms<300, slippage_bps ≤ SLO (global y símbolo).
- Sharpe_rolling_20d≥1.0 (paper previo), DD_día≤1%.
- rehydrate≤5s, huérfanas=0, NoTCAEvents5m=0, BudgetDepleted=0 (última hora).



