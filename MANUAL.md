# MANUAL de Operaciones y Runbooks

Este documento resume procedimientos ante incidentes y tareas operativas.

## WS caído / reconexiones frecuentes
- Ver métricas: `ws_uptime_seconds{ws=...}`, `ws_reconnects_total{ws=...}` en Grafana.
- Acciones:
  1) Revisar conexión/red, activar simulador con `NET_SIM_*` solo para reproducir.
  2) Reiniciar con backoff (ya automático). Si persiste, ejecutar `scripts/ws_network_harness.py` para aislar.
  3) Ejecutar `scripts/reconcile_and_alert.py` para rehidratar órdenes/fills.

## Skew NTP alto
- Ver `time_skew_ms` y alertas.
- Acciones: Cambiar `NTP_SERVER` o sincronizar SO. OMS bloquea si supera `MAX_TIME_SKEW_MS`.

## Duplicados de órdenes / idempotencia
- Revisar `/metrics`: `idempotency_outbox_*` y `orders_error_total{...}`.
- Acciones: Confirmar `ENABLE_IDEMPOTENCY=true`. Revisar outbox en DB (`order_outbox`).

## Desajuste de posiciones/saldos
- Ejecutar `scripts/reconcile_and_alert.py` (resuelve y registra métricas `reconciliation_mismatches_total{type,symbol}`).
- Ver panel de reconciliación (missing_in_db / missing_in_broker / status_mismatch).

## Breakers: error-rate o slippage
- Ver `slo_error_rate_recent`, `slippage_bps_p95`.
- Acciones: Armar `kill-switch` desde UI. Ajustar `MAX_ERROR_RATE` o `MAX_SLIPPAGE_BPS` si corresponde.

## Presupuestos (budgets) agotados
- UI muestra envíos hoy por estrategia/símbolo/cuenta. Ajustar límites en `config/settings.py`.

## Model Drift (PSI)
- Ver `model_drift_psi`. Si supera umbral, considerar rollback de canary desde UI o endpoints de modelo.

## Reportes diarios
- `python scripts/export_daily_report.py` genera CSV de órdenes y fills por día en `reports/`.

## Salud general / Healthcheck
- `python scripts/healthcheck.py` devuelve conectividad, outbox y skew con exit code.

---

## Seguridad/Compliance
- Secretos en Vault o variables de entorno cifradas. Rotación periódica de claves (Fernet) documentada.
- 2FA TOTP para operaciones críticas. Roles operator/observer en API/UI.
- Auditoría inmutable: cadena hash en logs y registros de órdenes (IDs de decisión/ejecución, timestamps µs).
- Hardening de servidores (Windows/Linux):
  - Mantener SO actualizado (Windows Update / apt/yum). Deshabilitar servicios innecesarios.
  - Firewall mínimo: solo puertos estrictamente necesarios (API 8000/tcp). SSH con claves (Linux) y restricción por IP.
  - IDS/AV recomendados: Windows Defender/ATP, Falco/OSSEC. Alertas ante cambios en binarios y reglas.
  - Cifrado de configuración y secretos en reposo; discos cifrados (BitLocker/LUKS) en producción.
  - Usuarios/roles mínimos, auditoría de acceso, MFA en proveedores cloud.
- Exportes regulatorios: `scripts/export_regulatory_report.py` genera CSV con `decision_id`, `execution_id`, `dea` y timestamps µs.
- Reporte y revisión de compliance: checklist en `implementaciones.txt`.

### Checklist de salida a producción
- KPIs en verde (latencia p95, error-rate, slippage bps, fill ratio).
- Reconciliación sin discrepancias críticas.
- NTP y breakers verificados.
- Secretos y 2FA configurados.
- Backups actuales y `restore_db.py` probado.

## Staging (entorno espejo)
- Levantar entorno staging (puerto 8080) sin tocar producción:
  - `docker-compose -f docker-compose.staging.yml up -d`
  - Usa `PROFILE=staging`, `UI_ROLE=observer`, `KILL_SWITCH_ARMED=true`, entrenamiento online deshabilitado.
  - Healthchecks activos en `http://localhost:8080/health`.



