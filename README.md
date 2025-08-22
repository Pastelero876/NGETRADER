NGEtrader - Bot de Trading (Desktop + Live + IA + NLP)

Requisitos rápidos

- Python 3.11+
- Conexión a Internet para proveedor de datos

Instalación

1) Crear entorno virtual e instalar dependencias

   Windows PowerShell:

   - python -m venv .venv
   - .\.venv\Scripts\Activate.ps1
   - pip install -r requirements.txt

2) Configurar variables de entorno

   - Copia `env.sample` a `.env` y rellena `ALPHA_VANTAGE_API_KEY`.

Uso

- Ayuda del CLI:

  python -m nge_trader --help

- Backtest simple (descarga datos y muestra un resumen):

  python -m nge_trader backtest AAPL

- Estado de configuración:

  python -m nge_trader status

- Lanzar la app de escritorio (Tkinter):

  python -m nge_trader.entrypoints.desktop

Windows (inicio sin bloquear terminal):

  start_desktop.bat

Funciones destacadas

- Live trading (paper/real) con gestión de riesgo (ATR, límites, circuit breakers) y OMS (cancel/replace/cancel all)
- Backtesting por señales con simulación de ejecución (slippage + comisiones)
- IA “agent” entrenable desde UI/CLI y estrategias `ma_cross`, `rsi`, `macd`
- NLP con Hugging Face Inference API: sentimiento, resumen y RSS→Señales persistidas


Desarrollo rápido

- Instalar herramientas dev y hooks:

  pip install -r requirements-dev.txt && pre-commit install

- Ejecutar tests en watch:

  ptw -q

- Levantar API local:

  uvicorn nge_trader.entrypoints.api:app --host 0.0.0.0 --port 8000
 - Ejemplo reproducible (paper broker):

   DATA_PROVIDER=alpha_vantage BROKER=paper python scripts/example_paper_trade.py

 - Generar dashboard base para Grafana:

   python scripts/generate_grafana_dashboards.py > dashboard.json

 - Healthcheck/Watchdog:

   python scripts/healthcheck.py


- Optimización de hiperparámetros:

  python scripts/hparam_opt.py

  python scripts/hparam_opt_optuna.py

CI/CD

- Lint/format automáticos en CI (ruff/black/isort)
- Cobertura mínima 80% en módulos críticos
- Artefactos: `coverage.xml` y reportes diarios


KPIs mínimos y salud

- KPIs objetivo (paper):
  - p95 order placement < 300ms
  - error-rate < 1%
  - slippage medio dentro del presupuesto de la estrategia
  - skew NTP absoluto < 50ms sostenido
- Comandos útiles:
  - Healthcheck: `python scripts/healthcheck.py`
  - Métricas Prometheus: `GET /metrics`
  - Resumen operativo: `GET /api/summary`
  - Dashboard base Grafana: `python scripts/generate_grafana_dashboards.py > dashboard.json`
  - Benchmark throughput/latencia (paper): `BROKER=paper python scripts/throughput_benchmark.py --symbol BTCUSDT --n 200`
  - Validación ETL: `python scripts/validate_historical_data.py --symbols "BTCUSDT,ETHUSDT"`
  - Drift de performance: `python scripts/validate_performance_drift.py --lookback 20 --threshold 0.5`
  - Tracking error: `python scripts/compute_tracking_error.py --symbol BTCUSDT --strategy ma_cross`
  - Impacto datos alternativos: `python scripts/evaluate_data_impact.py --symbols "BTCUSDT,ETHUSDT" --strategy ma_cross`

Despliegue (Docker/Compose)

- Producción básica:
  - `docker build . -t ngetrader`
  - `docker-compose up -d`
- Staging espejo:
  - `docker-compose -f docker-compose.staging.yml up -d`


