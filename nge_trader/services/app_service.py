import click

from nge_trader.config.settings import Settings
from nge_trader.adapters.alpha_vantage import AlphaVantageDataProvider
from nge_trader.adapters.binance import BinanceDataProvider
from nge_trader.adapters.tiingo import TiingoDataProvider
from nge_trader.adapters.paper_broker import PaperBroker
from nge_trader.adapters.alpaca import AlpacaBroker
from nge_trader.adapters.binance_trade import BinanceBroker
from nge_trader.adapters.coinbase_broker import CoinbaseBroker
from nge_trader.adapters.cache import DataCache
from nge_trader.adapters.resilient import ResilientBroker
from nge_trader.adapters.ibkr_broker import IBKRBroker
from nge_trader.adapters.binance_ws import BinanceUserDataStream
from nge_trader.services.oms import cancel_all_orders_by_symbol


class AppService:
    """Orquesta la lógica de alto nivel de la aplicación."""

    def __init__(self) -> None:
        self.settings = Settings()
        self.cache = DataCache()
        self.data_provider = self._create_data_provider()
        self.broker = self._create_broker()
        # Sincronizar órdenes abiertas del broker hacia DB al iniciar
        try:
            self._sync_open_orders()
        except Exception:
            pass

    def _create_data_provider(self):
        if self.settings.data_provider == "alpha_vantage":
            return AlphaVantageDataProvider(api_key=self.settings.alpha_vantage_api_key)
        if self.settings.data_provider == "binance":
            return BinanceDataProvider()
        if self.settings.data_provider == "tiingo":
            return TiingoDataProvider(api_key=self.settings.tiingo_api_key)
        raise ValueError("Proveedor de datos no soportado")

    def _create_broker(self):
        if self.settings.broker == "paper":
            return PaperBroker()
        if self.settings.broker == "alpaca":
            if not (self.settings.alpaca_api_key and self.settings.alpaca_api_secret):
                # fallback a paper si faltan credenciales
                return PaperBroker()
            try:
                primary = AlpacaBroker(
                    api_key=self.settings.alpaca_api_key,
                    api_secret=self.settings.alpaca_api_secret,
                )
                # Cadena de fallbacks: paper como último recurso
                return ResilientBroker(primary, fallback_brokers=[PaperBroker()])
            except Exception:
                # fallback a paper si falla creación
                return PaperBroker()
        if self.settings.broker == "binance":
            if not (self.settings.binance_api_key and self.settings.binance_api_secret):
                return PaperBroker()
            try:
                # usar testnet si perfil es dev
                use_testnet = (self.settings.profile.lower() == "dev")
                primary = BinanceBroker(
                    api_key=self.settings.binance_api_key,
                    api_secret=self.settings.binance_api_secret,
                    testnet=use_testnet,
                )
                # Si hay Alpaca credenciales, incluirlo como segundo fallback
                fbs = [PaperBroker()]
                try:
                    if self.settings.alpaca_api_key and self.settings.alpaca_api_secret:
                        fbs.insert(0, AlpacaBroker(api_key=self.settings.alpaca_api_key, api_secret=self.settings.alpaca_api_secret))
                except Exception:
                    pass
                return ResilientBroker(primary, fallback_brokers=fbs)
            except Exception:
                return PaperBroker()
        if self.settings.broker == "coinbase":
            if not (self.settings.coinbase_api_key and self.settings.coinbase_api_secret and self.settings.coinbase_passphrase):
                return PaperBroker()
            try:
                primary = CoinbaseBroker(
                    api_key=self.settings.coinbase_api_key,
                    api_secret=self.settings.coinbase_api_secret,
                    passphrase=self.settings.coinbase_passphrase,
                )
                return ResilientBroker(primary, PaperBroker())
            except Exception:
                return PaperBroker()
        if self.settings.broker == "ibkr":
            try:
                primary = IBKRBroker(
                    host=self.settings.ibkr_host or "127.0.0.1",
                    port=int(self.settings.ibkr_port or 7497),
                    client_id=int(self.settings.ibkr_client_id or 1),
                )
                return ResilientBroker(primary, PaperBroker())
            except Exception:
                return PaperBroker()
        raise ValueError("Broker no soportado")

    def _sync_open_orders(self) -> None:
        try:
            if hasattr(self.broker, "list_orders"):
                from nge_trader.repository.db import Database
                import datetime as _dt
                from nge_trader.services.oms import normalize_order
                db = Database()
                orders = self.broker.list_orders(status="open", limit=200)
                for o in orders:
                    rec = normalize_order(o)
                    if not rec.get("ts"):
                        rec["ts"] = _dt.datetime.now(_dt.UTC).isoformat()
                    db.record_order(rec)
        except Exception:
            pass

    def reconcile_state(self, resolve: bool = False) -> dict:
        """Reconciliación básica: compara DB con broker y devuelve diffs.

        Si resolve=True: registra en DB lo que falte (missing_in_db) y crea auditorías
        para lo que falte en broker.
        """
        from nge_trader.repository.db import Database
        db = Database()
        diffs: dict[str, list[dict]] = {"missing_in_db": [], "missing_in_broker": [], "status_mismatches": []}
        try:
            broker_orders = []
            if hasattr(self.broker, "list_orders"):
                broker_orders = list(self.broker.list_orders(status="open", limit=500))
            db_orders = db.recent_orders(500) if hasattr(db, "recent_orders") else []
            bo = {str(o.get("id") or o.get("order_id") or o.get("orderId")): o for o in broker_orders if (o.get("id") or o.get("order_id") or o.get("orderId"))}
            do = {str(o.get("order_id")): o for o in db_orders if o.get("order_id")}
            for k in bo.keys():
                if k not in do:
                    diffs["missing_in_db"].append(bo[k])
            for k in do.keys():
                if k not in bo:
                    diffs["missing_in_broker"].append(do[k])
            # Mismatches de estado
            for k in set(bo.keys()).intersection(set(do.keys())):
                b_st = str(bo[k].get("status") or "").lower()
                d_st = str(do[k].get("status") or "").lower()
                if b_st and d_st and b_st != d_st:
                    diffs["status_mismatches"].append({"order_id": k, "broker": b_st, "db": d_st})
                    if resolve:
                        try:
                            db.update_order_status_price(k, status=b_st)
                        except Exception:
                            pass
            # Métricas de reconciliación
            try:
                from nge_trader.services.metrics import inc_metric_labeled
                for o in diffs["missing_in_db"]:
                    sym = str(o.get("symbol") or "").upper()
                    inc_metric_labeled("reconciliation_mismatches_total", 1.0, {"type": "missing_in_db", "symbol": sym})
                for o in diffs["missing_in_broker"]:
                    sym = str(o.get("symbol") or "").upper()
                    inc_metric_labeled("reconciliation_mismatches_total", 1.0, {"type": "missing_in_broker", "symbol": sym})
                for o in diffs["status_mismatches"]:
                    inc_metric_labeled("reconciliation_mismatches_total", 1.0, {"type": "status_mismatch", "symbol": ""})
            except Exception:
                pass
            if resolve:
                import datetime as _dt
                from nge_trader.services.oms import normalize_order
                # Registrar lo que falta en DB
                for o in diffs["missing_in_db"]:
                    try:
                        rec = normalize_order(o)
                        if not rec.get("ts"):
                            rec["ts"] = _dt.datetime.now(_dt.UTC).isoformat()
                        db.record_order(rec)
                        db.append_audit(rec["ts"], "reconcile_add_order", rec.get("order_id"), rec.get("symbol"), "added to DB")
                    except Exception:
                        pass
                # Auditar lo que falta en broker
                for o in diffs["missing_in_broker"]:
                    try:
                        ts = _dt.datetime.now(_dt.UTC).isoformat()
                        db.append_audit(ts, "reconcile_missing_in_broker", o.get("order_id"), o.get("symbol"), "order absent in broker")
                    except Exception:
                        pass
        except Exception:
            pass
        return diffs

    def reconcile_positions_balances(self, resolve: bool = False) -> dict:
        """Reconciliación de posiciones y balances entre broker y DB.

        Si resolve=True, guarda snapshots y audita diferencias.
        """
        from nge_trader.repository.db import Database
        db = Database()
        out: dict[str, list[dict]] = {"positions": [], "balances": []}
        # Posiciones
        try:
            current = self.get_portfolio_positions()
            last = db.recent_positions(50)
            out["positions"] = current
            if resolve:
                # Guardado ya sucede en get_portfolio_positions; solo auditar tamaño
                import datetime as _dt
                db.append_audit(
                    ts=_dt.datetime.now(_dt.UTC).isoformat(),
                    event="reconcile_positions",
                    order_id=None,
                    symbol=None,
                    details=f"pos_count={len(current)} last_snapshots={len(last)}",
                )
        except Exception:
            pass
        # Balances
        try:
            acc = self.get_account_summary()
            snaps = db.recent_balances(10)
            out["balances"] = snaps
            if resolve:
                import datetime as _dt
                db.append_audit(
                    ts=_dt.datetime.now(_dt.UTC).isoformat(),
                    event="reconcile_balances",
                    order_id=None,
                    symbol=None,
                    details=f"balance={acc.get('balance')} snaps={len(snaps)}",
                )
        except Exception:
            pass
        return out

    def backfill_for_symbol(self, symbol: str, limit: int = 100) -> dict:
        """Backfill ligero: trae órdenes abiertas y fills recientes y los registra en DB evitando duplicados.

        Devuelve conteos de elementos nuevos registrados.
        """
        from nge_trader.repository.db import Database
        from nge_trader.services.oms import normalize_order, normalize_fill
        import datetime as _dt
        db = Database()
        new_orders = 0
        new_fills = 0
        # Helper para introspección segura de parámetros
        def _has_param(fn, name: str) -> bool:  # type: ignore[no-redef]
            try:
                import inspect as _inspect  # noqa: WPS433
                return name in _inspect.signature(fn).parameters
            except Exception:
                return False

        # Órdenes
        try:
            if hasattr(self.broker, "list_orders"):
                lo = getattr(self.broker, "list_orders")
                if _has_param(lo, "symbol"):
                    orders = lo(status="open", limit=limit, symbol=symbol)
                else:
                    orders = lo(status="open", limit=limit)
                for o in orders:
                    try:
                        rec = normalize_order(o)
                        oid = rec.get("order_id")
                        if not db.order_exists(oid):
                            if not rec.get("ts"):
                                rec["ts"] = _dt.datetime.now(_dt.UTC).isoformat()
                            db.record_order(rec)
                            new_orders += 1
                        else:
                            # Actualizar estado/precio si el broker reporta cambios (filled/canceled/replaced/rejected)
                            st = (rec.get("status") or "").lower()
                            if st in {"filled", "canceled", "replaced", "rejected", "partial"}:
                                try:
                                    db.update_order_status_price(str(oid), status=st, price=rec.get("price"))
                                except Exception:
                                    pass
                    except Exception:
                        continue
        except Exception:
            pass
        # Fills
        try:
            if hasattr(self.broker, "list_fills"):
                lf = getattr(self.broker, "list_fills")
                if _has_param(lf, "symbol"):
                    fills = lf(symbol=symbol, limit=limit)
                else:
                    fills = lf()
                for f in fills:
                    try:
                        nf = normalize_fill(f)
                        if not nf.get("ts"):
                            nf["ts"] = _dt.datetime.now(_dt.UTC).isoformat()
                        if not db.fill_exists(nf.get("order_id"), nf.get("price"), nf.get("qty")):
                            db.record_fill(nf)
                            new_fills += 1
                    except Exception:
                        continue
        except Exception:
            pass
        return {"orders": new_orders, "fills": new_fills}

    def start_backtest(self, symbol: str) -> None:
        click.echo(f"Iniciando backtest para {symbol}...")
        cache_key = f"daily_{symbol}"
        df = self.cache.load_df(cache_key) or self.data_provider.get_daily_adjusted(symbol)
        if df is not None:
            self.cache.save_df(cache_key, df)
        click.echo(f"Datos recibidos: {len(df)} velas. Mostrando 5 primeras filas:")
        click.echo(df.head().to_string())

    def start_live(self, symbol: str) -> None:
        click.echo(f"Iniciando trading en vivo para {symbol}...")
        # Si broker es Binance, iniciar user data stream
        if isinstance(self.broker, ResilientBroker):  # unwrap si aplica
            underlying = getattr(self.broker, "_primary", self.broker)
        else:
            underlying = self.broker
        if isinstance(underlying, BinanceBroker) and (self.settings.binance_api_key and self.settings.binance_api_secret):
            uds = BinanceUserDataStream(
                api_key=self.settings.binance_api_key,
                api_secret=self.settings.binance_api_secret,
                testnet=(self.settings.profile.lower() == "dev"),
            )
            def _cb(evt: dict) -> None:
                try:
                    et = evt.get('e') or evt.get('eventType')
                    if et:
                        click.echo(f"Evento WS: {et}")
                except Exception:
                    pass
            def _rehydrate() -> None:
                # Backfill ligero al reconectar: listar órdenes abiertas y registrar
                try:
                    from nge_trader.repository.db import Database
                    from nge_trader.services.oms import normalize_order
                    db = Database()
                    orders = underlying.list_orders(status="open", limit=200)
                    for o in orders:
                        try:
                            rec = normalize_order(o)
                            if not rec.get("ts"):
                                rec["ts"] = __import__("datetime").datetime.datetime.now(__import__("datetime").datetime.UTC).isoformat()
                            db.record_order(rec)
                        except Exception:
                            pass
                except Exception:
                    pass
            uds.start(_cb, on_reconnect=_rehydrate)
            click.echo("WebSocket de usuario (Binance) iniciado.")
            # Iniciar WS de mercado básico para el símbolo
            try:
                from nge_trader.adapters.binance_market_ws import BinanceMarketWS
                mws = BinanceMarketWS(symbol, testnet=(self.settings.profile.lower()=="dev"), stream="aggTrade")
                mws.start(lambda e: None)
                click.echo("WebSocket de mercado (Binance) iniciado.")
            except Exception:
                pass
            # Cancel-all por símbolo opcional tras reconexión
            try:
                if bool(getattr(self.settings, "cancel_on_reconnect", False)):
                    cancel_all_orders_by_symbol(underlying, symbol)
            except Exception:
                pass
        else:
            click.echo("Live básico iniciado (sin WS específico de broker).")

    def get_connectivity_status(self) -> dict:
        status = {
            "alpha_vantage": bool(self.settings.alpha_vantage_api_key),
            "alpaca": False,
            "binance": False,
            "coinbase": False,
            "ibkr": False,
        }
        broker_obj = self.broker
        if isinstance(self.broker, ResilientBroker):
            broker_obj = getattr(self.broker, "_primary", self.broker)
        try:
            if isinstance(broker_obj, AlpacaBroker):
                status["alpaca"] = broker_obj.connectivity_ok()
            if isinstance(broker_obj, BinanceBroker):
                status["binance"] = broker_obj.connectivity_ok()
            if isinstance(broker_obj, CoinbaseBroker):
                status["coinbase"] = broker_obj.connectivity_ok()
            if isinstance(broker_obj, IBKRBroker):
                status["ibkr"] = broker_obj.connectivity_ok()
        except Exception:
            pass
        return status

    def get_portfolio_positions(self) -> list[dict]:
        if hasattr(self.broker, "list_positions"):
            try:
                pos = self.broker.list_positions()
                # snapshot en DB
                try:
                    from nge_trader.repository.db import Database
                    Database().save_positions_snapshot(pos)
                except Exception:
                    pass
                return pos
            except Exception:
                return []
        return []

    def close_all_positions(self) -> dict:
        if hasattr(self.broker, "close_all_positions"):
            return self.broker.close_all_positions()
        return {"status": "noop"}

    def close_position(self, symbol: str) -> dict:
        if hasattr(self.broker, "close_symbol_position"):
            return self.broker.close_symbol_position(symbol)
        return {"status": "noop"}

    def get_account_summary(self) -> dict:
        summary = {"balance": None, "pl_today": None, "pl_open": None}
        try:
            if hasattr(self.broker, "get_account"):
                acc = self.broker.get_account()
                summary["balance"] = float(acc.get("equity")) if acc.get("equity") is not None else None
                summary["pl_today"] = float(acc.get("equity") or 0) - float(acc.get("last_equity") or 0)
                summary["pl_open"] = float(acc.get("unrealized_pl") or 0)
                # snapshot de saldos
                try:
                    from nge_trader.repository.db import Database
                    Database().save_balance_snapshot(currency=None, cash=None, equity=summary["balance"])
                except Exception:
                    pass
        except Exception:
            pass
        return summary


