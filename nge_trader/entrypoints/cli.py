import click

from nge_trader.services.app_service import AppService
from nge_trader.config.settings import Settings
from nge_trader.ai.policy import AgentPolicy, PolicyConfig
from nge_trader.services.oms import (
    place_market_order,
    place_limit_order_post_only,
    place_oco_order_generic,
    cancel_all_orders_by_symbol,
)
from pathlib import Path
from dotenv import dotenv_values


@click.group(help="Bot de trading autónomo. Usa los subcomandos.")
def cli() -> None:
    pass


@cli.command("status", help="Muestra el estado de configuración.")
def status() -> None:
    settings = Settings()
    click.echo(
        f"Entorno cargado. Proveedor de datos: {settings.data_provider}. Broker: {settings.broker}"
    )
    missing = settings.missing_required()
    if missing:
        click.echo(f"Variables faltantes: {', '.join(missing)}")
    else:
        click.echo("Configuración completa.")


@cli.command("backtest", help="Ejecuta un backtest simple usando el proveedor de datos configurado.")
@click.argument("symbol")
def backtest(symbol: str) -> None:
    service = AppService()
    service.start_backtest(symbol)


@cli.command("run-live", help="Inicia el trading en vivo (modo demostración/paper por defecto).")
@click.argument("symbol")
def run_live(symbol: str) -> None:
    service = AppService()
    service.start_live(symbol)


@cli.command("train-agent", help="Entrena la política de IA con datos históricos del símbolo.")
@click.argument("symbol")
def train_agent(symbol: str) -> None:
    service = AppService()
    df = service.data_provider.get_daily_adjusted(symbol)
    policy = AgentPolicy(PolicyConfig())
    policy.fit(df)
    policy.save()
    click.echo("Política entrenada y guardada.")


@cli.command("order", help="Envía una orden al broker configurado.")
@click.option("--symbol", required=True)
@click.option("--side", required=True, type=click.Choice(["buy", "sell"], case_sensitive=False))
@click.option("--quantity", required=True, type=float)
@click.option("--type", "type_", default="market", type=click.Choice(["market", "limit", "limit_post_only"], case_sensitive=False))
@click.option("--price", type=float)
@click.option("--tif", default="GTC")
def order(symbol: str, side: str, quantity: float, type_: str, price: float | None, tif: str) -> None:
    service = AppService()
    if type_.lower() == "market":
        res = place_market_order(service.broker, symbol, side, quantity)
    elif type_.lower() == "limit_post_only":
        if price is None:
            raise click.ClickException("--price requerido para limit_post_only")
        res = place_limit_order_post_only(service.broker, symbol, side, quantity, price, tif)
    else:
        if price is None:
            raise click.ClickException("--price requerido para limit")
        res = service.broker.place_order(symbol=symbol, side=side, quantity=quantity, type_="limit", price=price, tif=tif)
    click.echo(res)


@cli.command("oco", help="Crea una orden OCO en el broker (nativo o genérico).")
@click.option("--symbol", required=True)
@click.option("--side", required=True, type=click.Choice(["buy", "sell"], case_sensitive=False))
@click.option("--quantity", required=True, type=float)
@click.option("--limit", "limit_price", required=True, type=float)
@click.option("--stop", "stop_price", required=True, type=float)
@click.option("--stop-limit", "stop_limit_price", type=float)
@click.option("--tif", default="GTC")
def oco(symbol: str, side: str, quantity: float, limit_price: float, stop_price: float, stop_limit_price: float | None, tif: str) -> None:
    service = AppService()
    res = place_oco_order_generic(service.broker, symbol, side, quantity, limit_price, stop_price, stop_limit_price, tif)
    click.echo(res)


@cli.command("cancel-all", help="Cancela todas las órdenes abiertas de un símbolo.")
@click.option("--symbol", required=True)
def cancel_all(symbol: str) -> None:
    service = AppService()
    res = cancel_all_orders_by_symbol(service.broker, symbol)
    click.echo(res)


# ===== Utilidades locales =====
def _write_env(updated: dict[str, str | None]) -> None:
    env_path = Path('.env')
    current: dict[str, str] = {}
    if env_path.exists():
        current_raw = dotenv_values(str(env_path))
        current = {k: v for k, v in current_raw.items() if v is not None}  # type: ignore[assignment]
    normalized = {k.upper(): v for k, v in current.items() if v is not None}
    for k, v in updated.items():
        if v is None:
            continue
        normalized[k.upper()] = v
    lines = [f"{k}={v}" for k, v in normalized.items()]
    env_path.write_text("\n".join(lines), encoding="utf-8")


@cli.command("kill-switch", help="Activa o desactiva el kill-switch y lo persiste en .env.")
@click.option("--armed/--disarmed", default=True, help="Activa (--armed) o desactiva (--disarmed) el kill-switch")
def kill_switch(armed: bool) -> None:
    _write_env({"KILL_SWITCH_ARMED": "1" if armed else "0"})
    click.echo({"status": "ok", "armed": armed})


@cli.command("pause", help="Pausa un símbolo (se persiste en .env)")
@click.option("--symbol", required=True)
def pause(symbol: str) -> None:
    s = Settings()
    current = (s.paused_symbols or "").split(",") if s.paused_symbols else []
    up = {x.strip().upper() for x in current if x}
    up.add(symbol.upper())
    _write_env({"PAUSED_SYMBOLS": ",".join(sorted(up))})
    click.echo({"status": "ok", "paused": sorted(up)})


@cli.command("resume", help="Reanuda un símbolo pausado (se persiste en .env)")
@click.option("--symbol", required=True)
def resume(symbol: str) -> None:
    s = Settings()
    current = (s.paused_symbols or "").split(",") if s.paused_symbols else []
    up = {x.strip().upper() for x in current if x}
    up.discard(symbol.upper())
    _write_env({"PAUSED_SYMBOLS": ",".join(sorted(up))})
    click.echo({"status": "ok", "paused": sorted(up)})


@cli.command("risk-summary", help="Muestra resumen de riesgo/saldo de la cuenta")
def risk_summary() -> None:
    service = AppService()
    acc = service.get_account_summary()
    click.echo(acc)


@cli.command("reconcile", help="Reconciliación básica de órdenes/posiciones/balances con opción de resolver")
@click.option("--resolve", is_flag=True, default=False, help="Aplica resolución mínima (registrar faltantes en DB y auditar)")
def reconcile(resolve: bool) -> None:
    service = AppService()
    diffs = service.reconcile_state(resolve=resolve)
    pb = service.reconcile_positions_balances(resolve=resolve)
    click.echo({"orders": diffs, "portfolio": pb})

