# Observability stack (Prometheus + Pushgateway + Grafana)

## Prerrequisitos
- La API expone `/metrics` (Prometheus client activo).
- El TCA writer puede pushear al Pushgateway si `TCA_SINK` incluye `prom`.
- Regla `tca.rules.yml` copiada en `observability/prometheus/rules`.

## Arranque
```
cd observability
docker compose up -d
```

- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Pushgateway: http://localhost:9091

## Verificación
- En Prometheus: consulta `tca_slippage_bps` y `tca_place_latency_ms`.
- Forzar refresh SLO: `curl -X POST http://<tu-api>:8000/api/kit/slo/refresh`.
- Ver `/api/kit/slo` en tu API para confirmar p95 y "Fuente".

## Producción (sugerencias)
- Cambia credenciales de Grafana; restringe puertos por firewall.
- En `prometheus.yml` usa discovery o targets por DNS.
- Retención (`--storage.tsdb.retention.time`) según disco.
- Particiona `tca_events` por día/mes en Postgres si crece rápido.

## Notas de red
- Si tu API corre en el mismo compose, añade su servicio y usa su nombre en `targets`.
- Si corre fuera, usa `host.docker.internal:8000` (Win/Mac) o IP del host (Linux).

## Comandos útiles
- Recargar reglas sin reiniciar: `curl -X POST http://localhost:9090/-/reload`
- Logs Prometheus: `docker logs -f prometheus`
- Reset admin Grafana: `docker exec -it grafana grafana-cli admin reset-admin-password <nuevo>`
