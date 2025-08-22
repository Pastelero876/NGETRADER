from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import pandas as pd
import json as _json


DB_PATH = Path("data/app.db")


def _ensure_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS equity_curve (
                ts TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                in_time TEXT,
                out_time TEXT,
                fees REAL DEFAULT 0,
                realized REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL,
                status TEXT,
                order_id TEXT,
                decision_id TEXT,
                execution_id TEXT,
                dea INTEGER DEFAULT 1
            )
            """
        )
        # Migración: añadir columnas si faltan
        try:
            cur.execute("PRAGMA table_info(orders)")
            cols = [r[1] for r in cur.fetchall()]
            if "order_id" not in cols:
                cur.execute("ALTER TABLE orders ADD COLUMN order_id TEXT")
            if "decision_id" not in cols:
                cur.execute("ALTER TABLE orders ADD COLUMN decision_id TEXT")
            if "execution_id" not in cols:
                cur.execute("ALTER TABLE orders ADD COLUMN execution_id TEXT")
            if "dea" not in cols:
                cur.execute("ALTER TABLE orders ADD COLUMN dea INTEGER DEFAULT 1")
        except Exception:
            pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                order_id TEXT,
                fees REAL DEFAULT 0,
                liquidity TEXT
            )
            """
        )
        # Migración: añadir columna fees si falta
        try:
            cur.execute("PRAGMA table_info(fills)")
            cols = [r[1] for r in cur.fetchall()]
            if "fees" not in cols:
                cur.execute("ALTER TABLE fills ADD COLUMN fees REAL DEFAULT 0")
            if "liquidity" not in cols:
                cur.execute("ALTER TABLE fills ADD COLUMN liquidity TEXT")
        except Exception:
            pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                order_id TEXT,
                symbol TEXT,
                event TEXT NOT NULL,
                details TEXT
            )
            """
        )
        # Tabla de validaciones de algoritmo (compliance)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS algo_validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                algo_name TEXT NOT NULL,
                algo_version TEXT NOT NULL,
                approved_by TEXT,
                notes TEXT
            )
            """
        )
        # Audit log encadenado
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL DEFAULT (datetime('now')),
                prev_hash TEXT,
                payload_hash TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        # Snapshot de posiciones
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                qty REAL NOT NULL,
                avg_price REAL,
                market_value REAL,
                upl REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS alt_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source TEXT NOT NULL,
                symbol TEXT,
                title TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                summary TEXT,
                link TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS nlp_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                model TEXT NOT NULL,
                hash TEXT NOT NULL,
                result TEXT NOT NULL
            )
            """
        )
        # Cache de calendario de mercado
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS market_calendar_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange TEXT NOT NULL,
                date TEXT NOT NULL,
                session TEXT NOT NULL,
                is_open INTEGER NOT NULL,
                UNIQUE(exchange, date, session)
            )
            """
        )
        # Métricas (clave-valor temporal)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL
            )
            """
        )
        # Feature store (tabla larga: una fila por (ts, symbol, feature))
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                feature TEXT NOT NULL,
                value REAL NOT NULL
            )
            """
        )
        # Fees schedule por exchange/símbolo
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fees_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                tier TEXT DEFAULT 'default',
                maker_bps REAL NOT NULL,
                taker_bps REAL NOT NULL,
                effective_at TEXT NOT NULL,
                UNIQUE(exchange, symbol, tier, effective_at)
            )
            """
        )
        # Idempotencia y outbox de órdenes
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS order_outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL DEFAULT (datetime('now')),
                correlation_id TEXT,
                idempotency_key TEXT,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                broker_order_id TEXT,
                error TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS order_idempotency (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL UNIQUE,
                first_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
                status TEXT NOT NULL DEFAULT 'seen'
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS order_link (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_order_id TEXT NOT NULL,
                child_order_id TEXT NOT NULL,
                link_type TEXT NOT NULL
            )
            """
        )
        # Autoaprendizaje / IA
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT,
                state TEXT NOT NULL,
                action REAL NOT NULL,
                reward REAL NOT NULL,
                next_state TEXT NOT NULL,
                done INTEGER NOT NULL,
                priority REAL DEFAULT 1.0
            )
            """
        )
        # Migración: añadir columna priority si falta
        try:
            cur.execute("PRAGMA table_info(agent_experiences)")
            cols = [r[1] for r in cur.fetchall()]
            if "priority" not in cols:
                cur.execute("ALTER TABLE agent_experiences ADD COLUMN priority REAL DEFAULT 1.0")
        except Exception:
            pass
        # Balances/saldos
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS balances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                currency TEXT,
                cash REAL,
                equity REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                version TEXT NOT NULL,
                config TEXT NOT NULL,
                metrics TEXT,
                path TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT,
                lookback INTEGER,
                train_start TEXT,
                train_end TEXT,
                metrics TEXT,
                model_path TEXT
            )
            """
        )
        # Presupuestos por estrategia (órdenes por día)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_budgets (
                date TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                orders_sent INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(date, strategy_id)
            )
            """
        )
        # Presupuestos por símbolo
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS symbol_budgets (
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                orders_sent INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(date, symbol)
            )
            """
        )
        # Presupuestos por cuenta
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS account_budgets (
                date TEXT NOT NULL,
                account_id TEXT NOT NULL,
                orders_sent INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(date, account_id)
            )
            """
        )
        conn.commit()


class Database:
    def __init__(self) -> None:
        _ensure_db()

    def save_equity_curve(self, series: pd.Series) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM equity_curve")
            rows: List[Tuple[str, float]] = []
            if series.index.name == "date":
                index_iter: Iterable[str] = [
                    str(v) for v in series.index.astype(str).tolist()
                ]
            else:
                index_iter = [str(i) for i in range(len(series))]
            for idx, val in zip(index_iter, series.astype(float).tolist(), strict=False):
                rows.append((idx, float(val)))
            cur.executemany("INSERT INTO equity_curve(ts, value) VALUES(?, ?)", rows)
            conn.commit()

    def load_equity_curve(self) -> pd.Series:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("SELECT ts, value FROM equity_curve ORDER BY ts", conn)
        if df.empty:
            return pd.Series(dtype=float)
        s = pd.Series(df["value"].values, index=pd.Index(df["ts"], name="ts"))
        return s

    def record_trade(self, trade: dict) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO trades(symbol, side, qty, price, in_time, out_time, fees, realized)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.get("symbol"),
                    trade.get("side"),
                    float(trade.get("qty", 0)),
                    float(trade.get("price", 0)),
                    trade.get("in_time"),
                    trade.get("out_time"),
                    float(trade.get("fees", 0)),
                    trade.get("realized"),
                ),
            )
            conn.commit()

    def recent_trades(self, limit: int = 5) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def append_log(self, level: str, message: str, ts: str) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO logs(ts, level, message) VALUES(?, ?, ?)", (ts, level, message)
            )
            conn.commit()

    def append_log_json(self, level: str, payload: dict, ts: str) -> None:
        self.append_log(level, _json.dumps(payload, ensure_ascii=False), ts)

    # ====== Métricas (clave-valor temporal) ======
    def put_metric(self, key: str, value: float, ts: Optional[str] = None) -> None:
        if ts is None:
            ts = "datetime('now')"
            use_fn = True
        else:
            use_fn = False
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            if use_fn:
                cur.execute("INSERT INTO metrics(ts, key, value) VALUES(datetime('now'), ?, ?)", (key, float(value)))
            else:
                cur.execute("INSERT INTO metrics(ts, key, value) VALUES(?, ?, ?)", (ts, key, float(value)))
            conn.commit()

    def recent_metric_series(self, key: str, limit: int = 200) -> list[tuple[str, float]]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT ts, value FROM metrics WHERE key=? ORDER BY id DESC LIMIT ?", (key, limit))
            rows = cur.fetchall()
        out: list[tuple[str, float]] = [(r["ts"], float(r["value"])) for r in rows]
        return list(reversed(out))

    def tail_logs(self, limit: int = 200, level: str | None = None) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            if level and level.lower() != "all":
                cur.execute(
                    "SELECT * FROM logs WHERE level = ? ORDER BY id DESC LIMIT ?",
                    (level.upper(), limit),
                )
            else:
                cur.execute("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()][::-1]

    def record_order(self, order: dict) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO orders(ts, symbol, side, qty, price, status, order_id)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.get("ts"),
                    order.get("symbol"),
                    order.get("side"),
                    float(order.get("qty") or 0),
                    order.get("price"),
                    order.get("status"),
                    order.get("order_id"),
                ),
            )
            conn.commit()

    def update_order_status_price(self, order_id: str, status: str | None = None, price: float | None = None) -> None:
        if not order_id:
            return
        sets: list[str] = []
        params: list[object] = []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if price is not None:
            sets.append("price = ?")
            params.append(float(price))
        if not sets:
            return
        params.append(order_id)
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(f"UPDATE orders SET {', '.join(sets)} WHERE order_id = ?", params)
            conn.commit()

    def recent_orders(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def order_exists(self, order_id: str | None) -> bool:
        if order_id is None:
            return False
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM orders WHERE order_id = ? LIMIT 1", (order_id,))
            return cur.fetchone() is not None

    def count_orders_since(self, since_ts_iso: str) -> int:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(1) FROM orders WHERE ts >= ?", (since_ts_iso,))
            row = cur.fetchone()
            return int(row[0] if row and row[0] is not None else 0)

    def last_order_ts_by_symbol(self) -> dict[str, float]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT symbol, MAX(ts) AS last_ts
                FROM orders
                WHERE symbol IS NOT NULL
                GROUP BY symbol
                """
            )
            rows = cur.fetchall()
        out: dict[str, float] = {}
        for r in rows:
            try:
                ts = pd.Timestamp(r["last_ts"]).timestamp()
                out[str(r["symbol"])] = float(ts)
            except Exception:
                continue
        return out

    def record_fill(self, fill: dict) -> None:
        # Calcular fees si no vienen y hay schedule disponible (maker/taker si hay 'liquidity')
        try:
            fee_in = fill.get("fees") if fill.get("fees") is not None else fill.get("fee")
            if (fee_in is None or float(fee_in or 0.0) == 0.0) and fill.get("symbol") and fill.get("exchange"):
                exch = str(fill.get("exchange")).upper()
                sym = str(fill.get("symbol")).upper()
                row = None
                try:
                    row = self.get_fee_schedule(exch, sym)  # type: ignore[attr-defined]
                except Exception:
                    pass
                if row is None:
                    try:
                        row = self.get_fee_schedule_any(exch, sym)  # type: ignore[attr-defined]
                    except Exception:
                        row = None
                if row is not None:
                    maker_bps = float(row.get("maker_bps") or 0.0)
                    taker_bps = float(row.get("taker_bps") or 0.0)
                    qty_val = float(fill.get("qty") or 0.0)
                    px_val = float(fill.get("price") or 0.0)
                    liq = str(fill.get("liquidity") or "").lower()
                    is_maker = liq.startswith("m") or liq == "maker" or liq == "add"
                    bps = maker_bps if is_maker else taker_bps
                    computed_fee = (qty_val * px_val) * (bps / 10000.0)
                    fill = {**fill, "fees": computed_fee}
        except Exception:
            pass
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO fills(ts, symbol, side, qty, price, order_id, fees, liquidity)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.get("ts"),
                    fill.get("symbol"),
                    fill.get("side"),
                    float(fill.get("qty") or 0),
                    float(fill.get("price") or 0),
                    fill.get("order_id"),
                    float(fill.get("fees") or fill.get("fee") or 0.0),
                    (fill.get("liquidity") if fill.get("liquidity") is not None else None),
                ),
            )
            conn.commit()
        try:
            # Contador simple de fills para fill ratio
            from nge_trader.services.metrics import inc_metric
            inc_metric("fills_total", 1.0)
        except Exception:
            pass
        # Derivar realized P&L y registrar trade simple (stub, por símbolo sin inventario avanzado)
        try:
            side = (fill.get("side") or "").lower()
            qty = float(fill.get("qty") or 0.0)
            price = float(fill.get("price") or 0.0)
            if qty > 0 and price > 0:
                fee_val = float(fill.get("fees") or fill.get("fee") or 0.0)
                pnl = qty * price * (1.0 if side == "sell" else -1.0) - (fee_val if side == "sell" else 0.0)
                self.record_trade({
                    "symbol": fill.get("symbol"),
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "in_time": fill.get("ts") if side == "buy" else None,
                    "out_time": fill.get("ts") if side == "sell" else None,
                    "fees": fee_val,
                    "realized": pnl if side == "sell" else None,
                })
        except Exception:
            pass

    def recent_fills(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM fills ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def fill_exists(self, order_id: str | None, price: float | None, qty: float | None) -> bool:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 1 FROM fills
                WHERE COALESCE(order_id, '') = COALESCE(?, '')
                  AND ABS(COALESCE(price, 0) - COALESCE(?, 0)) < 1e-9
                  AND ABS(COALESCE(qty, 0) - COALESCE(?, 0)) < 1e-9
                LIMIT 1
                """,
                (order_id, float(price or 0.0), float(qty or 0.0)),
            )
            return cur.fetchone() is not None

    def append_audit(self, ts: str, event: str, order_id: str | None = None, symbol: str | None = None, details: str | None = None) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO audits(ts, order_id, symbol, event, details) VALUES(?, ?, ?, ?, ?)",
                (ts, order_id, symbol, event, details),
            )
            conn.commit()

    def recent_audits(self, limit: int = 200) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM audits ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()][::-1]

    # ===== Compliance: validación de algoritmo =====
    def add_algo_validation(self, algo_name: str, algo_version: str, approved_by: str | None, notes: str | None) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO algo_validations(ts, algo_name, algo_version, approved_by, notes) VALUES(datetime('now'), ?, ?, ?, ?)",
                (algo_name, algo_version, approved_by, notes),
            )
            conn.commit()

    def is_algo_validated(self, algo_name: str, algo_version: str) -> bool:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM algo_validations WHERE algo_name=? AND algo_version=? ORDER BY id DESC LIMIT 1",
                (algo_name, algo_version),
            )
            return cur.fetchone() is not None

    def save_alt_signals(self, rows: list[dict]) -> None:
        if not rows:
            return
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO alt_signals(ts, source, symbol, title, sentiment_label, sentiment_score, summary, link)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.get("ts"),
                        r.get("source", "rss"),
                        r.get("symbol"),
                        r.get("title"),
                        r.get("sentiment_label"),
                        float(r.get("sentiment_score") or 0.0),
                        r.get("summary"),
                        r.get("link"),
                    )
                    for r in rows
                ],
            )
            conn.commit()

    def recent_alt_signals(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM alt_signals ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def aggregated_sentiment(self, symbol: Optional[str] = None, minutes: int = 24 * 60) -> float:
        """Devuelve score medio ponderado [-1,1] en ventana reciente.

        Si symbol es None, calcula sobre todas las señales.
        """
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            if symbol:
                cur.execute(
                    """
                    SELECT sentiment_label AS label, sentiment_score AS score
                    FROM alt_signals
                    WHERE (symbol = ? OR ? IS NULL)
                      AND ts >= datetime('now', ?)
                    """,
                    (symbol, symbol, f'-{minutes} minutes'),
                )
            else:
                cur.execute(
                    """
                    SELECT sentiment_label AS label, sentiment_score AS score
                    FROM alt_signals
                    WHERE ts >= datetime('now', ?)
                    """,
                    (f'-{minutes} minutes',),
                )
            rows = cur.fetchall()
        total = 0.0
        weight = 0.0
        for r in rows:
            label = (r["label"] or "").lower()
            score = float(r["score"] or 0.0)
            pol = 1.0 if "pos" in label else (-1.0 if "neg" in label else 0.0)
            total += pol * score
            weight += score
        if weight <= 0:
            return 0.0
        # normaliza a [-1,1]
        return max(-1.0, min(1.0, total / weight))

    def get_nlp_cache(self, kind: str, model: str, key_hash: str, max_age_minutes: int = 1440) -> str | None:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT result FROM nlp_cache
                WHERE kind=? AND model=? AND hash=?
                  AND ts >= datetime('now', ?)
                ORDER BY id DESC LIMIT 1
                """,
                (kind, model, key_hash, f"-{max_age_minutes} minutes"),
            )
            row = cur.fetchone()
            return row["result"] if row else None

    def put_nlp_cache(self, kind: str, model: str, key_hash: str, result: str) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO nlp_cache(ts, kind, model, hash, result) VALUES(datetime('now'), ?, ?, ?, ?)",
                (kind, model, key_hash, result),
            )
            conn.commit()

    # ====== Autoaprendizaje / IA ======
    def append_experiences(self, rows: list[dict]) -> None:
        if not rows:
            return
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO agent_experiences(ts, symbol, state, action, reward, next_state, done, priority)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.get("ts"),
                        r.get("symbol"),
                        r.get("state"),
                        float(r.get("action", 0.0)),
                        float(r.get("reward", 0.0)),
                        r.get("next_state"),
                        1 if r.get("done") else 0,
                        float(r.get("priority", abs(float(r.get("reward", 0.0)))))
                    )
                    for r in rows
                ],
            )
            conn.commit()

    def sample_experiences(self, limit: int = 1000) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM agent_experiences ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def sample_experiences_prioritized(self, limit: int = 1000, top_ratio: float = 0.7) -> list[dict]:
        top_n = int(max(min(limit, 100000), 1) * max(min(top_ratio, 0.99), 0.0))
        rest_n = max(limit - top_n, 0)
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            rows: list[dict] = []
            if top_n > 0:
                cur.execute("SELECT * FROM agent_experiences ORDER BY priority DESC, id DESC LIMIT ?", (top_n,))
                rows.extend([dict(r) for r in cur.fetchall()])
            if rest_n > 0:
                cur.execute("SELECT * FROM agent_experiences ORDER BY RANDOM() LIMIT ?", (rest_n,))
                rows.extend([dict(r) for r in cur.fetchall()])
            return rows

    def update_experience_priorities(self, items: list[tuple[int, float]]) -> None:
        if not items:
            return
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.executemany("UPDATE agent_experiences SET priority=? WHERE id=?", [(float(p), int(i)) for (i, p) in items])
            conn.commit()

    def save_agent_model(self, version: str, config_json: str, metrics_json: str | None, path: str) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO agent_models(ts, version, config, metrics, path) VALUES(datetime('now'), ?, ?, ?, ?)",
                (version, config_json, metrics_json, path),
            )
            conn.commit()

    def list_agent_models(self, limit: int = 20) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM agent_models ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def record_training_run(
        self,
        symbol: str | None,
        lookback: int,
        train_start: str | None,
        train_end: str | None,
        metrics_json: str | None,
        model_path: str | None,
    ) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO agent_runs(ts, symbol, lookback, train_start, train_end, metrics, model_path)
                VALUES(datetime('now'), ?, ?, ?, ?, ?, ?)
                """,
                (symbol, lookback, train_start, train_end, metrics_json, model_path),
            )
            conn.commit()

    def recent_training_runs(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM agent_runs ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    # ====== Presupuestos por estrategia ======
    def get_strategy_orders_sent_today(self, strategy_id: str) -> int:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT orders_sent FROM strategy_budgets WHERE date=date('now','localtime') AND strategy_id=?",
                (strategy_id,),
            )
            row = cur.fetchone()
            return int(row["orders_sent"]) if row else 0

    def inc_strategy_orders_sent_today(self, strategy_id: str, inc: int = 1) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            # upsert simple
            cur.execute(
                "INSERT INTO strategy_budgets(date, strategy_id, orders_sent) VALUES(date('now','localtime'), ?, ?)\n                 ON CONFLICT(date, strategy_id) DO UPDATE SET orders_sent = orders_sent + excluded.orders_sent",
                (strategy_id, int(inc)),
            )
            conn.commit()

    def get_symbol_orders_sent_today(self, symbol: str) -> int:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT orders_sent FROM symbol_budgets WHERE date=date('now','localtime') AND symbol=?",
                (symbol.upper(),),
            )
            row = cur.fetchone()
            return int(row["orders_sent"]) if row else 0

    def inc_symbol_orders_sent_today(self, symbol: str, inc: int = 1) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO symbol_budgets(date, symbol, orders_sent) VALUES(date('now','localtime'), ?, ?)\n                 ON CONFLICT(date, symbol) DO UPDATE SET orders_sent = orders_sent + excluded.orders_sent",
                (symbol.upper(), int(inc)),
            )
            conn.commit()

    def get_account_orders_sent_today(self, account_id: str) -> int:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT orders_sent FROM account_budgets WHERE date=date('now','localtime') AND account_id=?",
                (account_id,),
            )
            row = cur.fetchone()
            return int(row["orders_sent"]) if row else 0

    def inc_account_orders_sent_today(self, account_id: str, inc: int = 1) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO account_budgets(date, account_id, orders_sent) VALUES(date('now','localtime'), ?, ?)\n                 ON CONFLICT(date, account_id) DO UPDATE SET orders_sent = orders_sent + excluded.orders_sent",
                (account_id, int(inc)),
            )
            conn.commit()

    # ====== Métricas derivadas de logs/experiencias (legacy) ======
    def recent_metric_values(self, metric_key: str, limit: int = 200) -> list[tuple[str, float]]:
        """Parquea los últimos valores de un metric guardado como JSON en logs.message."""
        out: list[tuple[str, float]] = []
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT ts, message FROM logs ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        for r in rows:
            msg = r["message"]
            try:
                payload = _json.loads(msg)
                if isinstance(payload, dict) and payload.get("metric") == metric_key:
                    val = float(payload.get("value") or 0.0)
                    out.append((r["ts"], val))
            except Exception:
                continue
        return list(reversed(out))

    def recent_rewards(self, limit: int = 200) -> list[tuple[str, float]]:
        """Recupera recompensas recientes del buffer de experiencias."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT ts, reward FROM agent_experiences ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        out: list[tuple[str, float]] = [(r["ts"], float(r["reward"]) if r["reward"] is not None else 0.0) for r in rows]
        return list(reversed(out))

    # ====== Posiciones (snapshot) ======
    def save_positions_snapshot(self, rows: list[dict]) -> None:
        if not rows:
            return
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            ts = pd.Timestamp.now(tz=pd.Timestamp.utcnow().tz).isoformat()
            cur.executemany(
                """
                INSERT INTO positions(ts, symbol, qty, avg_price, market_value, upl)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        ts,
                        (r.get("symbol") or r.get("Symbol") or "").upper(),
                        float(r.get("qty") or r.get("quantity") or r.get("Qty") or 0.0),
                        float(r.get("avg_price") or r.get("avg_entry_price") or 0.0),
                        float(r.get("market_value") or r.get("value") or 0.0),
                        float(r.get("upl") or r.get("unrealized_pl") or 0.0),
                    )
                    for r in rows
                ],
            )
            conn.commit()

    def recent_positions(self, limit: int = 200) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM positions ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def save_balance_snapshot(self, currency: Optional[str], cash: Optional[float], equity: Optional[float]) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO balances(ts, currency, cash, equity) VALUES(datetime('now'), ?, ?, ?)",
                (currency, float(cash or 0.0), float(equity or 0.0)),
            )
            conn.commit()

    def recent_balances(self, limit: int = 200) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM balances ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    # ====== Feature store ======
    def save_features(self, rows: list[dict]) -> None:
        if not rows:
            return
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO features(ts, symbol, feature, value)
                VALUES(?, ?, ?, ?)
                """,
                [
                    (
                        r.get("ts"),
                        r.get("symbol"),
                        r.get("feature"),
                        float(r.get("value") or 0.0),
                    )
                    for r in rows
                ],
            )
            conn.commit()

    def recent_features(self, symbol: Optional[str] = None, limit: int = 200) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            if symbol:
                cur.execute(
                    "SELECT * FROM features WHERE symbol=? ORDER BY id DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                cur.execute("SELECT * FROM features ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    # ====== Idempotencia / Outbox ======
    def put_order_outbox(self, payload_json: str, idempotency_key: Optional[str], correlation_id: Optional[str]) -> int:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO order_outbox(correlation_id, idempotency_key, payload)
                VALUES(?, ?, ?)
                """,
                (correlation_id, idempotency_key, payload_json),
            )
            conn.commit()
            return int(cur.lastrowid)

    def mark_order_outbox(self, outbox_id: int, status: str, broker_order_id: Optional[str] = None, error: Optional[str] = None) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE order_outbox
                SET status = ?, broker_order_id = COALESCE(?, broker_order_id), error = COALESCE(?, error)
                WHERE id = ?
                """,
                (status, broker_order_id, error, outbox_id),
            )
            conn.commit()

    def get_outbox_row(self, outbox_id: int) -> dict | None:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM order_outbox WHERE id=?", (outbox_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def record_idempotency_seen(self, idempotency_key: str, status: str = "seen") -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO order_idempotency(idempotency_key, status)
                VALUES(?, ?)
                ON CONFLICT(idempotency_key) DO UPDATE SET last_seen_at = datetime('now'), status = excluded.status
                """,
                (idempotency_key, status),
            )
            conn.commit()
        # Métrica de duplicados (si ya existía y status vuelve a 'seen')
        try:
            from nge_trader.services.metrics import inc_metric
            if status == "seen":
                inc_metric("idempotency_duplicates_total", 1.0)
        except Exception:
            pass

    def idempotency_exists(self, idempotency_key: str) -> bool:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM order_idempotency WHERE idempotency_key = ? LIMIT 1", (idempotency_key,))
            return cur.fetchone() is not None

    def link_orders(self, parent_order_id: str, child_order_id: str, link_type: str) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO order_link(parent_order_id, child_order_id, link_type) VALUES(?, ?, ?)",
                (parent_order_id, child_order_id, link_type),
            )
            conn.commit()

    def get_linked_orders(self, order_id: str) -> list[tuple[str, str]]:
        """Devuelve lista de (linked_id, link_type) para un order_id como padre o hijo."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT child_order_id AS linked_id, link_type
                FROM order_link WHERE parent_order_id = ?
                UNION ALL
                SELECT parent_order_id AS linked_id, link_type
                FROM order_link WHERE child_order_id = ?
                """,
                (order_id, order_id),
            )
            rows = cur.fetchall()
            return [(str(r["linked_id"]), str(r["link_type"])) for r in rows]

    def count_outbox_by_status(self) -> dict[str, int]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT status, COUNT(1) AS n FROM order_outbox GROUP BY status")
            rows = cur.fetchall()
        return {str(r["status"]): int(r["n"]) for r in rows}

    def oldest_outbox_age_seconds_by_status(self) -> dict[str, float]:
        """Devuelve, por estado, la edad en segundos del registro más antiguo."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT status,
                       MAX(0, (julianday('now') - julianday(MIN(ts))) * 86400.0) AS age_seconds
                FROM order_outbox
                GROUP BY status
                """
            )
            rows = cur.fetchall()
        return {str(r["status"]): float(r["age_seconds"]) if r["age_seconds"] is not None else 0.0 for r in rows}

    def gc_order_outbox(self, max_age_days: int = 7, statuses: Optional[Iterable[str]] = None) -> int:
        """Elimina filas antiguas del outbox para estados dados y devuelve el número de filas borradas."""
        status_tuple = tuple((s.lower() for s in (statuses or ("sent", "error"))))
        if not status_tuple:
            return 0
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            # SQLite no acepta directly tuple param para IN con placeholders variables; construir sql seguro
            placeholders = ",".join(["?"] * len(status_tuple))
            sql = (
                f"DELETE FROM order_outbox WHERE LOWER(status) IN ({placeholders}) "
                "AND ts < datetime('now', ?)"
            )
            params = list(status_tuple) + [f"-{int(max_age_days)} days"]
            cur.execute(sql, params)
            deleted = cur.rowcount or 0
            conn.commit()
            return int(deleted)

    def update_outbox_ts(self, outbox_id: int, ts_iso: str) -> None:
        """Actualiza el timestamp de una fila de outbox (útil para pruebas)."""
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("UPDATE order_outbox SET ts=? WHERE id=?", (ts_iso, int(outbox_id)))
            conn.commit()

    # ====== Fees schedule helpers ======
    def set_fee_schedule(self, exchange: str, symbol: str, maker_bps: float, taker_bps: float, tier: str = "default", effective_at_iso: str | None = None) -> None:
        import datetime as _dt
        eff = effective_at_iso or _dt.datetime.now(_dt.UTC).isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO fees_schedule(exchange, symbol, tier, maker_bps, taker_bps, effective_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(exchange, symbol, tier, effective_at) DO UPDATE SET
                    maker_bps=excluded.maker_bps,
                    taker_bps=excluded.taker_bps
                """,
                (exchange.upper(), symbol.upper(), tier, float(maker_bps), float(taker_bps), eff),
            )
            conn.commit()

    def get_fee_schedule(self, exchange: str, symbol: str, tier: str = "default") -> dict | None:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT exchange, symbol, tier, maker_bps, taker_bps, effective_at
                FROM fees_schedule
                WHERE exchange=? AND symbol=? AND tier=?
                ORDER BY effective_at DESC
                LIMIT 1
                """,
                (exchange.upper(), symbol.upper(), tier),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_fee_schedule_any(self, exchange: str, symbol: str) -> dict | None:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT exchange, symbol, tier, maker_bps, taker_bps, effective_at
                FROM fees_schedule
                WHERE exchange=? AND symbol=?
                ORDER BY effective_at DESC
                LIMIT 1
                """,
                (exchange.upper(), symbol.upper()),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # ====== Market calendar cache ======
    def set_market_calendar(self, exchange: str, date_iso: str, session: str, is_open: bool) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO market_calendar_cache(exchange, date, session, is_open)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(exchange, date, session) DO UPDATE SET is_open=excluded.is_open
                """,
                (exchange.upper(), date_iso, session.lower(), 1 if is_open else 0),
            )
            conn.commit()

    def get_market_calendar(self, exchange: str, date_iso: str, session: str) -> Optional[bool]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT is_open FROM market_calendar_cache WHERE exchange=? AND date=? AND session=?",
                (exchange.upper(), date_iso, session.lower()),
            )
            row = cur.fetchone()
            if not row:
                return None
            return bool(int(row["is_open"]))

    # ====== Audit log encadenado ======
    def append_audit_log(self, payload_json: str) -> None:
        import hashlib
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT payload_hash FROM audit_log ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            prev_hash = row[0] if row else None
            payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            cur.execute(
                "INSERT INTO audit_log(prev_hash, payload_hash, payload) VALUES(?, ?, ?)",
                (prev_hash, payload_hash, payload_json),
            )
            conn.commit()

    def verify_audit_log(self, limit: int = 1000) -> bool:
        import hashlib
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT id, prev_hash, payload_hash, payload FROM audit_log ORDER BY id ASC LIMIT ?", (limit,))
            rows = cur.fetchall()
        last_hash: str | None = None
        for r in rows:
            calc = hashlib.sha256((r["payload"] or "").encode("utf-8")).hexdigest()
            if calc != r["payload_hash"]:
                return False
            if r["prev_hash"] != last_hash:
                # primera fila permite prev_hash None
                if r["id"] != rows[0]["id"]:
                    return False
            last_hash = r["payload_hash"]
        return True


