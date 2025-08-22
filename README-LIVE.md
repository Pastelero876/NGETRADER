# Live — Checklist y Guardrails

## Preflight (T-1)
- `python scripts/backup_db.py`
- `python scripts/verify_audit_chain.py`
- `/api/kit/preflight` → ok:true
- `/api/kit/state` y `/metrics` visibles

## Armado y Ramp Canario
- `/api/kit/arm` (PROFILE=live requiere OTP)
- PATCH `/api/kit/model/canary_share?pct=0.05 → 0.10 → 0.20`

## Gates activos
- SLO, Drift PSI, Budget, Loss Cap, Canary Hard-Cap, Allowlist LIVE_SYMBOLS

## Runbooks
- Promote/Rollback (auto y manual)
- Disarm + Cancel-All
- Reconcile hourly: `python scripts/reconcile_and_alert.py`

## Observabilidad
- Prometheus/Grafana
- Alertas: BudgetDepleted, SLOGateStuck, CanaryImbalance, ReconciliationMismatches, CanaryCapReached

## Seguridad
- 2FA en `/api/kit/arm` (PROFILE=live)
- Gestión de secretos y rotación

## Parámetros iniciales (1000 €)
- `RISK_PER_TRADE_PCT=0.0015–0.0020`
- `MAX_TRADES_PER_DAY=5–8`
- `LOSS_CAP_PCT=0.010`
- `MAX_CANARY_NOTIONAL_PCT_DAILY=0.25`
- `LIVE_SYMBOLS=BTCUSDT,ETHUSDT`

## EOD
- Flatten opcional 5–0 min antes del cierre
