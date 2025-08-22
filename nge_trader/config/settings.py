from typing import List  # noqa: F401

from pydantic_settings import BaseSettings
from pydantic import Field
from pydantic import AliasChoices


class Settings(BaseSettings):
    """Configuración de la aplicación (se carga desde variables de entorno)."""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "",
        "case_sensitive": False,
    }

    data_provider: str = Field(
        default="alpha_vantage",
        description="Proveedor de datos de mercado",
        validation_alias=AliasChoices("DATA_PROVIDER", "data_provider"),
    )
    broker: str = Field(
        default="paper",
        description="Broker a utilizar",
        validation_alias=AliasChoices("BROKER", "broker"),
    )
    enable_live_trading: bool = Field(
        default=False,
        description="Permite envíos a broker real (usar true solo en .env.live)",
        validation_alias=AliasChoices("ENABLE_LIVE_TRADING", "enable_live_trading"),
    )
    profile: str = Field(
        default="dev",
        description="Perfil de ejecución (dev/paper/live)",
        validation_alias=AliasChoices("PROFILE", "profile"),
    )
    alpha_vantage_api_key: str | None = Field(
        default=None,
        description="API key para Alpha Vantage",
        validation_alias=AliasChoices("ALPHA_VANTAGE_API_KEY", "alpha_vantage_api_key"),
    )
    alpaca_api_key: str | None = Field(
        default=None,
        description="API key para Alpaca",
        validation_alias=AliasChoices("ALPACA_API_KEY", "alpaca_api_key"),
    )
    alpaca_api_secret: str | None = Field(
        default=None,
        description="API secret para Alpaca",
        validation_alias=AliasChoices("ALPACA_API_SECRET", "alpaca_api_secret"),
    )
    binance_api_key: str | None = Field(
        default=None,
        description="API key para Binance",
        validation_alias=AliasChoices("BINANCE_API_KEY", "binance_api_key"),
    )
    binance_api_secret: str | None = Field(
        default=None,
        description="API secret para Binance",
        validation_alias=AliasChoices("BINANCE_API_SECRET", "binance_api_secret"),
    )
    coinbase_api_key: str | None = Field(
        default=None,
        description="API key para Coinbase Exchange",
        validation_alias=AliasChoices("COINBASE_API_KEY", "coinbase_api_key"),
    )
    coinbase_api_secret: str | None = Field(
        default=None,
        description="API secret (base64) para Coinbase Exchange",
        validation_alias=AliasChoices("COINBASE_API_SECRET", "coinbase_api_secret"),
    )
    coinbase_passphrase: str | None = Field(
        default=None,
        description="Passphrase de la API de Coinbase Exchange",
        validation_alias=AliasChoices("COINBASE_PASSPHRASE", "coinbase_passphrase"),
    )
    ibkr_host: str | None = Field(
        default="127.0.0.1",
        description="Host de TWS/Gateway de Interactive Brokers",
        validation_alias=AliasChoices("IBKR_HOST", "ibkr_host"),
    )
    ibkr_port: int | None = Field(
        default=7497,
        description="Puerto de TWS/Gateway (7497 paper, 7496 live por defecto)",
        validation_alias=AliasChoices("IBKR_PORT", "ibkr_port"),
    )
    ibkr_client_id: int | None = Field(
        default=1,
        description="Client ID para conexión a IBKR",
        validation_alias=AliasChoices("IBKR_CLIENT_ID", "ibkr_client_id"),
    )
    huggingface_api_token: str | None = Field(
        default=None,
        description="Token de API de Hugging Face",
        validation_alias=AliasChoices("HUGGINGFACE_API_TOKEN", "huggingface_api_token"),
    )
    huggingface_sentiment_model: str = Field(
        default="ProsusAI/finbert",
        description="Modelo de sentimiento en Hugging Face",
        validation_alias=AliasChoices("HUGGINGFACE_SENTIMENT_MODEL", "huggingface_sentiment_model"),
    )
    huggingface_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Modelo de embeddings en Hugging Face",
        validation_alias=AliasChoices("HUGGINGFACE_EMBEDDING_MODEL", "huggingface_embedding_model"),
    )
    huggingface_summarization_model: str = Field(
        default="facebook/bart-large-cnn",
        description="Modelo de resumen en Hugging Face",
        validation_alias=AliasChoices("HUGGINGFACE_SUMMARIZATION_MODEL", "huggingface_summarization_model"),
    )
    max_daily_drawdown_pct: float = Field(
        default=0.05,
        description="Máximo drawdown diario permitido antes de parar trading",
        validation_alias=AliasChoices("MAX_DAILY_DRAWDOWN_PCT", "max_daily_drawdown_pct"),
    )
    risk_pct_per_trade: float = Field(
        default=0.01,
        description="Riesgo por operación (porcentaje del equity por trade)",
        validation_alias=AliasChoices("RISK_PCT_PER_TRADE", "risk_pct_per_trade"),
    )
    trailing_atr_mult: float = Field(
        default=1.5,
        description="Multiplicador ATR para trailing stop virtual",
        validation_alias=AliasChoices("TRAILING_ATR_MULT", "trailing_atr_mult"),
    )
    breakeven_r_multiple: float = Field(
        default=1.0,
        description="Múltiplos de R a partir de los cuales subir SL a break-even",
        validation_alias=AliasChoices("BREAKEVEN_R_MULTIPLE", "breakeven_r_multiple"),
    )
    cache_ttl_seconds: int = Field(
        default=43200,
        description="TTL de caché de datos de mercado en segundos",
        validation_alias=AliasChoices("CACHE_TTL_SECONDS", "cache_ttl_seconds"),
    )
    max_open_positions: int = Field(
        default=2,
        description="Máximo número de posiciones abiertas permitidas",
        validation_alias=AliasChoices("MAX_OPEN_POSITIONS", "max_open_positions"),
    )
    max_exposure_pct: float = Field(
        default=1.0,
        description="Exposición máxima total sobre el equity (1.0 = 100%)",
        validation_alias=AliasChoices("MAX_EXPOSURE_PCT", "max_exposure_pct"),
    )
    max_symbol_exposure_pct: float = Field(
        default=0.2,
        description="Exposición máxima por símbolo sobre el equity (0.2 = 20%)",
        validation_alias=AliasChoices("MAX_SYMBOL_EXPOSURE_PCT", "max_symbol_exposure_pct"),
    )
    reconcile_interval_minutes: int = Field(
        default=10,
        description="Cada cuántos minutos ejecutar reconciliación periódica (órdenes/posiciones)",
        validation_alias=AliasChoices("RECONCILE_INTERVAL_MINUTES", "reconcile_interval_minutes"),
    )
    fill_prob_threshold: float = Field(
        default=0.6,
        description="Umbral mínimo de probabilidad de fill para permitir marketable-limit",
        validation_alias=AliasChoices("FILL_PROB_THRESHOLD", "fill_prob_threshold"),
    )
    cancel_on_reconnect: bool = Field(
        default=False,
        description="Si está activado, cancelar órdenes abiertas del símbolo al reconectar WS",
        validation_alias=AliasChoices("CANCEL_ON_RECONNECT", "cancel_on_reconnect"),
    )
    es_budget_pct: float = Field(
        default=0.03,
        description="Presupuesto de ES (97.5%) intradía permitido por símbolo (porcentaje)",
        validation_alias=AliasChoices("ES_BUDGET_PCT", "es_budget_pct"),
    )
    es_window_minutes: int = Field(
        default=60,
        description="Ventana en minutos para cálculo intradía de ES/CVaR",
        validation_alias=AliasChoices("ES_WINDOW_MINUTES", "es_window_minutes"),
    )
    dislocation_bps_threshold: float = Field(
        default=20.0,
        description="Umbral de dislocation entre fuentes (en bps) para pausar decisiones",
        validation_alias=AliasChoices("DISLOCATION_BPS_THRESHOLD", "dislocation_bps_threshold"),
    )
    data_staleness_slo_ms: int = Field(
        default=1000,
        description="SLO de frescura de datos de cotización (ms) para bloqueo preventivo",
        validation_alias=AliasChoices("DATA_STALENESS_SLO_MS", "data_staleness_slo_ms"),
    )
    flash_move_bps_threshold: float = Field(
        default=50.0,
        description="Umbral de flash-move en bps para activar bloqueo preventivo",
        validation_alias=AliasChoices("FLASH_MOVE_BPS_THRESHOLD", "flash_move_bps_threshold"),
    )
    spread_widen_bps_threshold: float = Field(
        default=30.0,
        description="Umbral de aumento de spread en bps para bloqueo preventivo",
        validation_alias=AliasChoices("SPREAD_WIDEN_BPS_THRESHOLD", "spread_widen_bps_threshold"),
    )
    vol_targeting_enabled: bool = Field(
        default=False,
        description="Habilita volatility targeting para dimensionamiento de órdenes",
        validation_alias=AliasChoices("VOL_TARGETING_ENABLED", "vol_targeting_enabled"),
    )
    target_vol_daily: float = Field(
        default=0.02,
        description="Volatilidad objetivo diaria para volatility targeting",
        validation_alias=AliasChoices("TARGET_VOL_DAILY", "target_vol_daily"),
    )
    exposure_reduce_hours_utc: str | None = Field(
        default=None,
        description="Rangos horarios UTC para reducir exposición (formato 'HH-HH,HH-HH')",
        validation_alias=AliasChoices("EXPOSURE_REDUCE_HOURS_UTC", "exposure_reduce_hours_utc"),
    )
    exposure_increase_hours_utc: str | None = Field(
        default=None,
        description="Rangos horarios UTC para incrementar exposición (formato 'HH-HH,HH-HH')",
        validation_alias=AliasChoices("EXPOSURE_INCREASE_HOURS_UTC", "exposure_increase_hours_utc"),
    )
    max_sector_exposure_pct: float = Field(
        default=0.35,
        description="Exposición máxima por sector (0.35 = 35%)",
        validation_alias=AliasChoices("MAX_SECTOR_EXPOSURE_PCT", "max_sector_exposure_pct"),
    )
    sector_map: str | None = Field(
        default=None,
        description="Mapa de sectores por símbolo (ej: AAPL=TECH,MSFT=TECH,BTCUSDT=CRYPTO)",
        validation_alias=AliasChoices("SECTOR_MAP", "sector_map"),
    )
    correlation_limit: float = Field(
        default=0.9,
        description="Límite de correlación permitido para abrir nuevas posiciones altamente correlacionadas",
        validation_alias=AliasChoices("CORRELATION_LIMIT", "correlation_limit"),
    )
    use_sentiment_bias: bool = Field(
        default=True,
        description="Usar sesgo de sentimiento agregado en decisiones de live trading",
        validation_alias=AliasChoices("USE_SENTIMENT_BIAS", "use_sentiment_bias"),
    )
    sentiment_window_minutes: int = Field(
        default=180,
        description="Ventana (minutos) para calcular sentimiento agregado",
        validation_alias=AliasChoices("SENTIMENT_WINDOW_MINUTES", "sentiment_window_minutes"),
    )
    max_weekly_drawdown_pct: float = Field(
        default=0.10,
        description="Máximo drawdown semanal permitido antes de parar trading",
        validation_alias=AliasChoices("MAX_WEEKLY_DRAWDOWN_PCT", "max_weekly_drawdown_pct"),
    )
    # Gating de horizontes
    enforce_horizon_gating: bool = Field(
        default=True,
        description="Aplicar gating de horizonte temporal (min/max minutos)",
        validation_alias=AliasChoices("ENFORCE_HORIZON_GATING", "enforce_horizon_gating"),
    )
    min_horizon_minutes: int = Field(
        default=15,
        description="Horizonte mínimo permitido en minutos (favorecer >= 15m)",
        validation_alias=AliasChoices("MIN_HORIZON_MINUTES", "min_horizon_minutes"),
    )
    max_horizon_minutes: int = Field(
        default=240,
        description="Horizonte máximo recomendado en minutos (<= 4h)",
        validation_alias=AliasChoices("MAX_HORIZON_MINUTES", "max_horizon_minutes"),
    )
    telegram_bot_token: str | None = Field(
        default=None,
        description="Token del bot de Telegram para notificaciones",
        validation_alias=AliasChoices("TELEGRAM_BOT_TOKEN", "telegram_bot_token"),
    )
    telegram_chat_id: str | None = Field(
        default=None,
        description="Chat ID de Telegram para notificaciones",
        validation_alias=AliasChoices("TELEGRAM_CHAT_ID", "telegram_chat_id"),
    )

    # Riesgo avanzado / IA
    target_daily_volatility: float = Field(
        default=0.01,
        description="Volatilidad diaria objetivo para position sizing (e.g., 0.01 = 1%)",
        validation_alias=AliasChoices("TARGET_DAILY_VOLATILITY", "target_daily_volatility"),
    )
    kelly_cap_pct: float = Field(
        default=0.25,
        description="Cap de Kelly fracción para dimensionamiento (0.25 = 25%)",
        validation_alias=AliasChoices("KELLY_CAP_PCT", "kelly_cap_pct"),
    )
    max_error_rate: float = Field(
        default=0.2,
        description="Umbral de ratio de errores recientes para activar breaker (0.2=20%)",
        validation_alias=AliasChoices("MAX_ERROR_RATE", "max_error_rate"),
    )
    max_slippage_bps: float = Field(
        default=50.0,
        description="Umbral de slippage medio (bps) para alerta/breaker",
        validation_alias=AliasChoices("MAX_SLIPPAGE_BPS", "max_slippage_bps"),
    )
    min_fill_ratio_recent: float = Field(
        default=0.5,
        description="Umbral mínimo de fill ratio reciente (0..1) para reconfigurar entrada/bloquear",
        validation_alias=AliasChoices("MIN_FILL_RATIO_RECENT", "min_fill_ratio_recent"),
    )
    kill_switch_armed: bool = Field(
        default=False,
        description="Si está activo, bloquea el envío de nuevas órdenes",
        validation_alias=AliasChoices("KILL_SWITCH_ARMED", "kill_switch_armed"),
    )
    nlp_killswitch_threshold: float = Field(
        default=0.6,
        description="Umbral de sentimiento negativo extremo para kill-switch (0..1)",
        validation_alias=AliasChoices("NLP_KILLSWITCH_THRESHOLD", "nlp_killswitch_threshold"),
    )
    var_limit_pct: float = Field(
        default=0.10,
        description="Límite de VaR (alpha=5%) como porcentaje negativo permitido",
        validation_alias=AliasChoices("VAR_LIMIT_PCT", "var_limit_pct"),
    )
    es_limit_pct: float = Field(
        default=0.15,
        description="Límite de ES (alpha=5%) como porcentaje negativo permitido",
        validation_alias=AliasChoices("ES_LIMIT_PCT", "es_limit_pct"),
    )
    # Hiperparámetros de política online
    online_lookback: int = Field(
        default=30,
        description="Ventana de retornos para la política online",
        validation_alias=AliasChoices("ONLINE_LOOKBACK", "online_lookback"),
    )
    online_learning_rate: float = Field(
        default=0.01,
        description="Tasa de aprendizaje (SGD) de la política online",
        validation_alias=AliasChoices("ONLINE_LEARNING_RATE", "online_learning_rate"),
    )
    online_l2_reg: float = Field(
        default=1e-4,
        description="Regularización L2 de la política online",
        validation_alias=AliasChoices("ONLINE_L2_REG", "online_l2_reg"),
    )
    online_training_enabled: bool = Field(
        default=True,
        description="Habilita entrenamiento online durante el live loop",
        validation_alias=AliasChoices("ONLINE_TRAINING_ENABLED", "online_training_enabled"),
    )
    online_updates_per_minute: int = Field(
        default=120,
        description="Límite de actualizaciones online por minuto",
        validation_alias=AliasChoices("ONLINE_UPDATES_PER_MINUTE", "online_updates_per_minute"),
    )
    online_training_pause_minutes: int = Field(
        default=10,
        description="Minutos de pausa del entrenamiento online si se activa breaker (PSI/SLI)",
        validation_alias=AliasChoices("ONLINE_TRAINING_PAUSE_MINUTES", "online_training_pause_minutes"),
    )
    training_metrics_window: int = Field(
        default=500,
        description="Ventana de observabilidad para métricas de entrenamiento",
        validation_alias=AliasChoices("TRAINING_METRICS_WINDOW", "training_metrics_window"),
    )
    drift_psi_threshold: float = Field(
        default=0.2,
        description="Umbral de PSI para detectar drift de distribución",
        validation_alias=AliasChoices("DRIFT_PSI_THRESHOLD", "drift_psi_threshold"),
    )
    drift_kl_threshold: float = Field(
        default=0.1,
        description="Umbral de KL divergence para drift",
        validation_alias=AliasChoices("DRIFT_KL_THRESHOLD", "drift_kl_threshold"),
    )
    drift_mmd_threshold: float = Field(
        default=0.05,
        description="Umbral de MMD para drift",
        validation_alias=AliasChoices("DRIFT_MMD_THRESHOLD", "drift_mmd_threshold"),
    )
    recent_sharpe_lookback: int = Field(
        default=20,
        description="Ventana de días para cálculo de Sharpe reciente en breaker",
        validation_alias=AliasChoices("RECENT_SHARPE_LOOKBACK", "recent_sharpe_lookback"),
    )
    min_recent_sharpe: float | None = Field(
        default=None,
        description="Umbral mínimo de Sharpe reciente para permitir trading (None desactiva)",
        validation_alias=AliasChoices("MIN_RECENT_SHARPE", "min_recent_sharpe"),
    )
    canary_traffic_pct: float = Field(
        default=0.1,
        description="Porcentaje de tráfico asignado al modelo canario (0..1)",
        validation_alias=AliasChoices("CANARY_TRAFFIC_PCT", "canary_traffic_pct"),
    )
    # Cumplimiento/región
    compliance_region: str | None = Field(
        default=None,
        description="Región de cumplimiento (EU/US/CRYPTO)",
        validation_alias=AliasChoices("COMPLIANCE_REGION", "compliance_region"),
    )
    require_kyc_aml: bool = Field(
        default=False,
        description="Exigir KYC/AML antes de permitir operaciones",
        validation_alias=AliasChoices("REQUIRE_KYC_AML", "require_kyc_aml"),
    )
    kyc_aml_completed: bool = Field(
        default=False,
        description="Marcador de KYC/AML completado",
        validation_alias=AliasChoices("KYC_AML_COMPLETED", "kyc_aml_completed"),
    )
    smtp_host: str | None = Field(
        default=None,
        description="Servidor SMTP para alertas email",
        validation_alias=AliasChoices("SMTP_HOST", "smtp_host"),
    )
    smtp_port: int | None = Field(
        default=587,
        description="Puerto SMTP",
        validation_alias=AliasChoices("SMTP_PORT", "smtp_port"),
    )
    smtp_user: str | None = Field(
        default=None,
        description="Usuario SMTP",
        validation_alias=AliasChoices("SMTP_USER", "smtp_user"),
    )
    smtp_password: str | None = Field(
        default=None,
        description="Password SMTP",
        validation_alias=AliasChoices("SMTP_PASSWORD", "smtp_password"),
    )
    smtp_to: str | None = Field(
        default=None,
        description="Email destino para alertas",
        validation_alias=AliasChoices("SMTP_TO", "smtp_to"),
    )
    fred_api_key: str | None = Field(
        default=None,
        description="API key de FRED para calendario económico",
        validation_alias=AliasChoices("FRED_API_KEY", "fred_api_key"),
    )
    twitter_bearer_token: str | None = Field(
        default=None,
        description="Bearer token de Twitter API v2 para búsquedas recientes",
        validation_alias=AliasChoices("TWITTER_BEARER_TOKEN", "twitter_bearer_token"),
    )
    tiingo_api_key: str | None = Field(
        default=None,
        description="API key de Tiingo para datos multi-activo",
        validation_alias=AliasChoices("TIINGO_API_KEY", "tiingo_api_key"),
    )
    kaiko_api_key: str | None = Field(
        default=None,
        description="API key de Kaiko para L2",
        validation_alias=AliasChoices("KAIKO_API_KEY", "kaiko_api_key"),
    )

    # Secret management / 2FA
    secret_backend: str = Field(
        default="file",
        description="Backend de secretos: file|env|vault",
        validation_alias=AliasChoices("SECRET_BACKEND", "secret_backend"),
    )
    vault_addr: str | None = Field(
        default=None,
        description="Dirección de Vault (http[s]://host:8200)",
        validation_alias=AliasChoices("VAULT_ADDR", "vault_addr"),
    )
    vault_token: str | None = Field(
        default=None,
        description="Token de autenticación de Vault",
        validation_alias=AliasChoices("VAULT_TOKEN", "vault_token"),
    )
    vault_kv_path: str | None = Field(
        default=None,
        description="Ruta de KV secrets en Vault (ej: secret/data/nge)",
        validation_alias=AliasChoices("VAULT_KV_PATH", "vault_kv_path"),
    )
    enforce_2fa_critical: bool = Field(
        default=False,
        description="Exigir 2FA (TOTP) para operaciones críticas (kill-switch, rotación secreta)",
        validation_alias=AliasChoices("ENFORCE_2FA_CRITICAL", "enforce_2fa_critical"),
    )
    totp_issuer: str = Field(
        default="NGEtrader",
        description="Issuer mostrado en la app TOTP",
        validation_alias=AliasChoices("TOTP_ISSUER", "totp_issuer"),
    )

    # Rollout gradual
    rollout_stage: str = Field(
        default="paper",
        description="Etapa de despliegue: paper|canary|ramp|full",
        validation_alias=AliasChoices("ROLLOUT_STAGE", "rollout_stage"),
    )
    rollout_ramp_pct: float = Field(
        default=0.25,
        description="Porcentaje de tamaño de orden en etapa ramp (0..1)",
        validation_alias=AliasChoices("ROLLOUT_RAMP_PCT", "rollout_ramp_pct"),
    )
    l2_provider: str = Field(
        default="none",
        description="Proveedor de L2: none|kaiko",
        validation_alias=AliasChoices("L2_PROVIDER", "l2_provider"),
    )
    # Universo por coste/edge
    universe_edge_expected_bps: float = Field(
        default=20.0,
        description="Edge esperado (bps) usado para filtrar símbolos en el selector de universo",
        validation_alias=AliasChoices("UNIVERSE_EDGE_EXPECTED_BPS", "universe_edge_expected_bps"),
    )
    universe_max_spread_edge_ratio: float = Field(
        default=0.5,
        description="Si spread_pct*10000 > ratio*edge_esperado → excluir del universo",
        validation_alias=AliasChoices("UNIVERSE_MAX_SPREAD_EDGE_RATIO", "universe_max_spread_edge_ratio"),
    )
    transfer_from_symbols: str | None = Field(
        default=None,
        description="Lista de símbolos origen para transfer learning (coma-separada)",
        validation_alias=AliasChoices("TRANSFER_FROM_SYMBOLS", "transfer_from_symbols"),
    )

    # HA / Failover
    ha_enabled: bool = Field(
        default=False,
        description="Alta disponibilidad activada (lease/primario)",
        validation_alias=AliasChoices("HA_ENABLED", "ha_enabled"),
    )
    cluster_node_id: str = Field(
        default="node-1",
        description="Identificador de nodo en el clúster",
        validation_alias=AliasChoices("CLUSTER_NODE_ID", "cluster_node_id"),
    )
    ha_lease_file: str = Field(
        default="data/ha_lease.json",
        description="Ruta del archivo de lease para primario",
        validation_alias=AliasChoices("HA_LEASE_FILE", "ha_lease_file"),
    )
    ha_lease_ttl_seconds: int = Field(
        default=30,
        description="TTL del lease en segundos",
        validation_alias=AliasChoices("HA_LEASE_TTL_SECONDS", "ha_lease_ttl_seconds"),
    )

    # Ventanas de despliegue seguras
    deploy_window_start: str = Field(
        default="00:00",
        description="Inicio de ventana segura de despliegue (UTC, HH:MM)",
        validation_alias=AliasChoices("DEPLOY_WINDOW_START", "deploy_window_start"),
    )
    deploy_window_end: str = Field(
        default="23:59",
        description="Fin de ventana segura de despliegue (UTC, HH:MM)",
        validation_alias=AliasChoices("DEPLOY_WINDOW_END", "deploy_window_end"),
    )

    # Límites operativos
    cooldown_minutes: int = Field(
        default=5,
        description="Minutos de enfriamiento por símbolo tras enviar una orden",
        validation_alias=AliasChoices("COOLDOWN_MINUTES", "cooldown_minutes"),
    )
    max_trades_per_day: int = Field(
        default=8,
        description="Número máximo de órdenes enviadas por día",
        validation_alias=AliasChoices("MAX_TRADES_PER_DAY", "max_trades_per_day"),
    )
    slo_slippage_bps: float = Field(
        default=8.0,
        description="Umbral SLO de slippage medio reciente (bps)",
        validation_alias=AliasChoices("SLO_SLIPPAGE_BPS", "slo_slippage_bps"),
    )
    slo_error_rate: float = Field(
        default=0.01,
        description="Umbral SLO de tasa de error reciente (0..1)",
        validation_alias=AliasChoices("SLO_ERROR_RATE", "slo_error_rate"),
    )
    slo_p95_ms: float = Field(
        default=300.0,
        description="Umbral SLO de p95 de latencia de envío (ms)",
        validation_alias=AliasChoices("SLO_P95_MS", "slo_p95_ms"),
    )
    enable_canary: bool = Field(
        default=False,
        description="Habilitar canary/challenger en inferencia",
        validation_alias=AliasChoices("ENABLE_CANARY", "enable_canary"),
    )
    challenger_share: float = Field(
        default=0.2,
        description="Fracción de decisiones enviadas al challenger (0..1)",
        validation_alias=AliasChoices("CHALLENGER_SHARE", "challenger_share"),
    )
    drift_psi_max: float = Field(
        default=0.2,
        description="Umbral máximo de PSI para gating por drift",
        validation_alias=AliasChoices("DRIFT_PSI_MAX", "drift_psi_max"),
    )
    max_trades_per_day_per_strategy: int = Field(
        default=50,
        description="Número máximo de órdenes por día por estrategia",
        validation_alias=AliasChoices("MAX_TRADES_PER_DAY_PER_STRATEGY", "max_trades_per_day_per_strategy"),
    )
    enable_idempotency: bool = Field(
        default=True,
        description="Activa el uso de idempotency keys y outbox en OMS",
        validation_alias=AliasChoices("ENABLE_IDEMPOTENCY", "enable_idempotency"),
    )
    enable_retry_policy: bool = Field(
        default=True,
        description="Activa la política de reintentos con backoff+jitter en broker resiliente",
        validation_alias=AliasChoices("ENABLE_RETRY_POLICY", "enable_retry_policy"),
    )
    enable_ws_backfill: bool = Field(
        default=True,
        description="Activa backfill REST tras reconexión de WS",
        validation_alias=AliasChoices("ENABLE_WS_BACKFILL", "enable_ws_backfill"),
    )
    enable_market_clock_enforcement: bool = Field(
        default=False,
        description="Enforzar ventanas de mercado (no enviar órdenes fuera de sesión)",
        validation_alias=AliasChoices("ENABLE_MARKET_CLOCK_ENFORCEMENT", "enable_market_clock_enforcement"),
    )
    enable_time_skew_breaker: bool = Field(
        default=True,
        description="Activar breaker por skew NTP",
        validation_alias=AliasChoices("ENABLE_TIME_SKEW_BREAKER", "enable_time_skew_breaker"),
    )
    trading_hours_start: str = Field(
        default="00:00",
        description="Hora inicio de trading (UTC, HH:MM)",
        validation_alias=AliasChoices("TRADING_HOURS_START", "trading_hours_start"),
    )
    trading_hours_end: str = Field(
        default="23:59",
        description="Hora fin de trading (UTC, HH:MM)",
        validation_alias=AliasChoices("TRADING_HOURS_END", "trading_hours_end"),
    )
    ui_role: str = Field(
        default="operator",
        description="Rol de UI (operator/observer)",
        validation_alias=AliasChoices("UI_ROLE", "ui_role"),
    )
    # Validación de algoritmos (MiFID/Compliance)
    algo_validation_required: bool = Field(
        default=False,
        description="Exigir validación previa del algoritmo para permitir trading",
        validation_alias=AliasChoices("ALGO_VALIDATION_REQUIRED", "algo_validation_required"),
    )
    algo_name: str = Field(
        default="default_strategy",
        description="Nombre del algoritmo/estrategia para propósitos de compliance",
        validation_alias=AliasChoices("ALGO_NAME", "algo_name"),
    )
    algo_version: str = Field(
        default="v1",
        description="Versión del algoritmo/estrategia",
        validation_alias=AliasChoices("ALGO_VERSION", "algo_version"),
    )
    account_id: str = Field(
        default="default",
        description="Identificador de cuenta para budgets y reporting",
        validation_alias=AliasChoices("ACCOUNT_ID", "account_id"),
    )
    strategy_id: str = Field(
        default="default",
        description="Identificador de estrategia para rate limits por estrategia",
        validation_alias=AliasChoices("STRATEGY_ID", "strategy_id"),
    )
    paused_symbols: str | None = Field(
        default=None,
        description="Lista separada por comas de símbolos pausados (ej: BTCUSDT,ETHUSDT)",
        validation_alias=AliasChoices("PAUSED_SYMBOLS", "paused_symbols"),
    )

    # Parámetros de ejecución (slicing)
    slicing_tranches: int = Field(
        default=4,
        description="Número base de tramos para slicing (TWAP/POV)",
        validation_alias=AliasChoices("SLICING_TRANCHES", "slicing_tranches"),
    )
    slicing_bps: float = Field(
        default=5.0,
        description="Margen límite en puntos básicos para órdenes post-only (por tramo)",
        validation_alias=AliasChoices("SLICING_BPS", "slicing_bps"),
    )
    vwap_window_sec: int = Field(
        default=60,
        description="Ventana en segundos para VWAP/volumen reciente",
        validation_alias=AliasChoices("VWAP_WINDOW_SEC", "vwap_window_sec"),
    )
    pov_ratio: float = Field(
        default=0.10,
        description="Proporción objetivo de volumen (POV) por ventana",
        validation_alias=AliasChoices("POV_RATIO", "pov_ratio"),
    )
    edge_buffer_bps: float = Field(
        default=5.0,
        description="Buffer de bps que se suma a fees+half-spread para el umbral de ejecutabilidad",
        validation_alias=AliasChoices("EDGE_BUFFER_BPS", "edge_buffer_bps"),
    )
    # Coeficientes de coste en reward
    lambda_slippage: float = Field(
        default=0.1,
        description="Penalización por slippage en la función de recompensa",
        validation_alias=AliasChoices("LAMBDA_SLIPPAGE", "lambda_slippage"),
    )
    lambda_fees: float = Field(
        default=1.0,
        description="Penalización por fees en la función de recompensa",
        validation_alias=AliasChoices("LAMBDA_FEES", "lambda_fees"),
    )
    lambda_drawdown: float = Field(
        default=0.2,
        description="Penalización por drawdown local en la función de recompensa",
        validation_alias=AliasChoices("LAMBDA_DRAWDOWN", "lambda_drawdown"),
    )
    maker_fill_prob_threshold: float = Field(
        default=0.6,
        description="Umbral mínimo de probabilidad de fill para abandonar maker y usar limit marketable/market",
        validation_alias=AliasChoices("MAKER_FILL_PROB_THRESHOLD", "maker_fill_prob_threshold"),
    )
    maker_ratio_target: float = Field(
        default=0.7,
        description="Objetivo de ratio de órdenes maker en ejecución",
        validation_alias=AliasChoices("MAKER_RATIO_TARGET", "maker_ratio_target"),
    )
    stale_tick_max_seconds: int = Field(
        default=10,
        description="Edad máxima del último tick/quote antes de considerarlo stale (segundos)",
        validation_alias=AliasChoices("STALE_TICK_MAX_SECONDS", "stale_tick_max_seconds"),
    )
    gap_events_max_seconds: int = Field(
        default=30,
        description="Máximo tiempo sin eventos low-lat antes de considerar gap (segundos)",
        validation_alias=AliasChoices("GAP_EVENTS_MAX_SECONDS", "gap_events_max_seconds"),
    )
    outlier_ret_threshold: float = Field(
        default=0.2,
        description="Umbral absoluto de retorno por barra para considerar outlier y bloquear (e.g. 0.2=20%)",
        validation_alias=AliasChoices("OUTLIER_RET_THRESHOLD", "outlier_ret_threshold"),
    )
    loss_streak_threshold: int = Field(
        default=2,
        description="Número de pérdidas consecutivas que activan cool-down por símbolo",
        validation_alias=AliasChoices("LOSS_STREAK_THRESHOLD", "loss_streak_threshold"),
    )
    cooldown_after_losses_minutes: int = Field(
        default=30,
        description="Minutos de cool-down tras alcanzar el umbral de pérdidas consecutivas",
        validation_alias=AliasChoices("COOLDOWN_AFTER_LOSSES_MINUTES", "cooldown_after_losses_minutes"),
    )
    slicing_profile_map: str | None = Field(
        default=None,
        description="Overrides por símbolo en JSON. Ej: {'BTCUSDT': {'slicing_tranches':6,'slicing_bps':7}}",
        validation_alias=AliasChoices("SLICING_PROFILE_MAP", "slicing_profile_map"),
    )
    risk_profile_map: str | None = Field(
        default=None,
        description="Overrides de riesgo por símbolo en JSON. Ej: {'BTCUSDT': {'risk_pct_per_trade':0.02}}",
        validation_alias=AliasChoices("RISK_PROFILE_MAP", "risk_profile_map"),
    )
    ntp_server: str = Field(
        default="pool.ntp.org",
        description="Servidor NTP para sincronización de reloj",
        validation_alias=AliasChoices("NTP_SERVER", "ntp_server"),
    )
    max_time_skew_ms: int = Field(
        default=100,
        description="Umbral máximo de skew de reloj permitido (ms)",
        validation_alias=AliasChoices("MAX_TIME_SKEW_MS", "max_time_skew_ms"),
    )
    market_clock_exchange: str = Field(
        default="BINANCE",
        description="Nombre de exchange/base para validación de ventanas (ej: BINANCE/COINBASE/NYSE)",
        validation_alias=AliasChoices("MARKET_CLOCK_EXCHANGE", "market_clock_exchange"),
    )

    # Rate limits
    broker_orders_per_minute: int = Field(
        default=60,
        description="Límite de órdenes por minuto a nivel broker",
        validation_alias=AliasChoices("BROKER_ORDERS_PER_MINUTE", "broker_orders_per_minute"),
    )
    symbol_orders_per_minute: int = Field(
        default=10,
        description="Límite de órdenes por minuto por símbolo",
        validation_alias=AliasChoices("SYMBOL_ORDERS_PER_MINUTE", "symbol_orders_per_minute"),
    )
    strategy_orders_per_minute: int = Field(
        default=30,
        description="Límite de órdenes por minuto por estrategia",
        validation_alias=AliasChoices("STRATEGY_ORDERS_PER_MINUTE", "strategy_orders_per_minute"),
    )
    min_fill_ratio_required: float = Field(
        default=0.7,
        description="Fill ratio mínimo reciente requerido para permitir nuevas entradas",
        validation_alias=AliasChoices("MIN_FILL_RATIO_REQUIRED", "min_fill_ratio_required"),
    )
    # Validaciones de instrumento
    min_notional_usd: float | None = Field(
        default=None,
        description="Notional mínimo por orden en USD (validación previa)",
        validation_alias=AliasChoices("MIN_NOTIONAL_USD", "min_notional_usd"),
    )
    min_notional_usd_per_symbol: str | None = Field(
        default=None,
        description="Mapa JSON de notional mínimo por símbolo en USD",
        validation_alias=AliasChoices("MIN_NOTIONAL_USD_PER_SYMBOL", "min_notional_usd_per_symbol"),
    )
    qty_step: float | None = Field(
        default=None,
        description="Paso mínimo de cantidad (lote)",
        validation_alias=AliasChoices("QTY_STEP", "qty_step"),
    )
    price_tick: float | None = Field(
        default=None,
        description="Incremento mínimo de precio (tick)",
        validation_alias=AliasChoices("PRICE_TICK", "price_tick"),
    )
    min_qty: float | None = Field(
        default=None,
        description="Cantidad mínima por orden (validación previa)",
        validation_alias=AliasChoices("MIN_QTY", "min_qty"),
    )
    allow_fractional: bool = Field(
        default=True,
        description="Permitir fracciones (si el venue/broker lo soporta)",
        validation_alias=AliasChoices("ALLOW_FRACTIONAL", "allow_fractional"),
    )
    min_qty_per_symbol: str | None = Field(
        default=None,
        description="Mapa JSON de min_qty por símbolo, ej: {\"BTCUSDT\":0.0001,\"AAPL\":1}",
        validation_alias=AliasChoices("MIN_QTY_PER_SYMBOL", "min_qty_per_symbol"),
    )
    lot_size_per_symbol: str | None = Field(
        default=None,
        description="Mapa JSON de lot_size por símbolo (paso de cantidad)",
        validation_alias=AliasChoices("LOT_SIZE_PER_SYMBOL", "lot_size_per_symbol"),
    )
    max_spread_pct: float | None = Field(
        default=0.005,
        description="Umbral máximo de spread relativo (p. ej., 0.005 = 0.5%) para permitir envíos",
        validation_alias=AliasChoices("MAX_SPREAD_PCT", "max_spread_pct"),
    )
    # Controles pre-trade adicionales
    max_order_qty: float | None = Field(
        default=None,
        description="Cantidad máxima permitida por orden (pre-trade)",
        validation_alias=AliasChoices("MAX_ORDER_QTY", "max_order_qty"),
    )
    max_position_qty_per_symbol: float | None = Field(
        default=None,
        description="Posición neta máxima permitida por símbolo (pre-trade)",
        validation_alias=AliasChoices("MAX_POSITION_QTY_PER_SYMBOL", "max_position_qty_per_symbol"),
    )
    price_collar_pct: float | None = Field(
        default=None,
        description="Collar de precio relativo al microprice para límites (ej. 0.02 = ±2%)",
        validation_alias=AliasChoices("PRICE_COLLAR_PCT", "price_collar_pct"),
    )
    secret_backend: str = Field(
        default="file",
        description="Backend de secretos (file|env)",
        validation_alias=AliasChoices("SECRET_BACKEND", "secret_backend"),
    )
    enforce_2fa_critical: bool = Field(
        default=False,
        description="Exigir 2FA en endpoints críticos (kill-switch, cancel-all, OCO)",
        validation_alias=AliasChoices("ENFORCE_2FA_CRITICAL", "enforce_2fa_critical"),
    )
    dea_indicator: bool = Field(
        default=True,
        description="DEA indicator para reportes regulatorios (Direct Electronic Access)",
        validation_alias=AliasChoices("DEA_INDICATOR", "dea_indicator"),
    )
    vault_addr: str | None = Field(
        default=None,
        description="URL de Vault (opcional)",
        validation_alias=AliasChoices("VAULT_ADDR", "vault_addr"),
    )
    vault_token: str | None = Field(
        default=None,
        description="Token de Vault (opcional)",
        validation_alias=AliasChoices("VAULT_TOKEN", "vault_token"),
    )
    infra_monitor_enabled: bool = Field(
        default=True,
        description="Habilita monitorización de CPU/memoria/RTT de red",
        validation_alias=AliasChoices("INFRA_MONITOR_ENABLED", "infra_monitor_enabled"),
    )
    infra_probe_url: str | None = Field(
        default=None,
        description="URL para medir RTT de red (HEAD)",
        validation_alias=AliasChoices("INFRA_PROBE_URL", "infra_probe_url"),
    )
    vault_kv_path: str | None = Field(
        default=None,
        description="Ruta KV en Vault (p.ej. secret/data/ngetrader)",
        validation_alias=AliasChoices("VAULT_KV_PATH", "vault_kv_path"),
    )
    max_trades_per_day_per_symbol: int = Field(
        default=20,
        description="Número máximo de órdenes por día por símbolo",
        validation_alias=AliasChoices("MAX_TRADES_PER_DAY_PER_SYMBOL", "max_trades_per_day_per_symbol"),
    )
    max_trades_per_day_per_account: int = Field(
        default=100,
        description="Número máximo de órdenes por día por cuenta",
        validation_alias=AliasChoices("MAX_TRADES_PER_DAY_PER_ACCOUNT", "max_trades_per_day_per_account"),
    )
    # Plan de crecimiento (equity steps)
    growth_equity_step: float = Field(
        default=250.0,
        description="Incremento de equity por paso del plan (EUR/USD)",
        validation_alias=AliasChoices("GROWTH_EQUITY_STEP", "growth_equity_step"),
    )
    growth_min_equity: float = Field(
        default=0.0,
        description="Equity mínimo a partir del cual empezar a aplicar el plan",
        validation_alias=AliasChoices("GROWTH_MIN_EQUITY", "growth_min_equity"),
    )
    growth_max_equity: float = Field(
        default=0.0,
        description="Equity máximo considerado para escalar pasos (0 = ilimitado)",
        validation_alias=AliasChoices("GROWTH_MAX_EQUITY", "growth_max_equity"),
    )
    growth_risk_pct_base: float = Field(
        default=0.01,
        description="Riesgo base por trade desde el que se parte (fracción)",
        validation_alias=AliasChoices("GROWTH_RISK_PCT_BASE", "growth_risk_pct_base"),
    )
    growth_risk_pct_increment: float = Field(
        default=0.001,
        description="Incremento de riesgo por paso",
        validation_alias=AliasChoices("GROWTH_RISK_PCT_INCREMENT", "growth_risk_pct_increment"),
    )
    growth_top_k_base: int = Field(
        default=1,
        description="Top-K base por ciclo",
        validation_alias=AliasChoices("GROWTH_TOP_K_BASE", "growth_top_k_base"),
    )
    growth_top_k_increment: int = Field(
        default=1,
        description="Incremento de K por paso",
        validation_alias=AliasChoices("GROWTH_TOP_K_INCREMENT", "growth_top_k_increment"),
    )
    growth_top_k_max: int = Field(
        default=3,
        description="Máximo Top-K permitido por ciclo",
        validation_alias=AliasChoices("GROWTH_TOP_K_MAX", "growth_top_k_max"),
    )

    def missing_required(self) -> list[str]:
        """Devuelve una lista de variables de entorno faltantes requeridas.

        Nota: Los nombres se devuelven en mayúsculas para claridad.
        """

        missing: list[str] = []
        if self.data_provider == "alpha_vantage" and not self.alpha_vantage_api_key:
            missing.append("ALPHA_VANTAGE_API_KEY")
        return missing


