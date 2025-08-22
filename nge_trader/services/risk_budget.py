from __future__ import annotations

import sqlite3
from typing import Tuple

from nge_trader.repository.db import DB_PATH


def _ensure_table() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_budget (
                date TEXT PRIMARY KEY,
                used_R REAL NOT NULL DEFAULT 0.0,
                left_R REAL NOT NULL DEFAULT 1.0
            )
            """
        )
        # Budgets por estrategia
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_budget_strategy (
                date TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                used_R REAL NOT NULL DEFAULT 0.0,
                left_R REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY(date, strategy_id)
            )
            """
        )
        # Budgets por símbolo
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_budget_symbol (
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                used_R REAL NOT NULL DEFAULT 0.0,
                left_R REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY(date, symbol)
            )
            """
        )
        # Parámetros de riesgo (loss cap diario)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_params (
                date TEXT PRIMARY KEY,
                loss_cap_pct REAL NOT NULL DEFAULT 0.01
            )
            """
        )
        conn.commit()


def get_today() -> Tuple[float, float]:
    _ensure_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT used_R, left_R FROM risk_budget WHERE date=date('now','localtime')")
        row = cur.fetchone()
        if not row:
            return (0.0, 1.0)
        return (float(row["used_R"] or 0.0), float(row["left_R"] or 1.0))


def set_today(used_R: float | None = None, left_R: float | None = None) -> None:
    _ensure_table()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO risk_budget(date, used_R, left_R) VALUES(date('now','localtime'), 0.0, 1.0)
            ON CONFLICT(date) DO NOTHING
            """
        )
        sets = []
        params: list[object] = []
        if used_R is not None:
            sets.append("used_R = ?")
            params.append(float(used_R))
        if left_R is not None:
            sets.append("left_R = ?")
            params.append(float(left_R))
        if sets:
            sql = f"UPDATE risk_budget SET {', '.join(sets)} WHERE date=date('now','localtime')"
            cur.execute(sql, params)
        conn.commit()


def reset_today() -> None:
    _ensure_table()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO risk_budget(date, used_R, left_R) VALUES(date('now','localtime'), 0.0, 1.0)
            ON CONFLICT(date) DO UPDATE SET used_R=0.0, left_R=1.0
            """
        )
        conn.commit()


# ===== Per-strategy/symbol helpers =====
def set_today_strategy(strategy_id: str, used_R: float | None = None, left_R: float | None = None) -> None:
    _ensure_table()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO risk_budget_strategy(date, strategy_id, used_R, left_R)
            VALUES(date('now','localtime'), ?, 0.0, 1.0)
            ON CONFLICT(date, strategy_id) DO NOTHING
            """,
            (strategy_id,),
        )
        sets, params = [], []
        if used_R is not None:
            sets.append("used_R=?")
            params.append(float(used_R))
        if left_R is not None:
            sets.append("left_R=?")
            params.append(float(left_R))
        if sets:
            cur.execute(f"UPDATE risk_budget_strategy SET {', '.join(sets)} WHERE date=date('now','localtime') AND strategy_id=?", params + [strategy_id])
        conn.commit()


def get_today_strategy(strategy_id: str) -> tuple[float, float]:
    _ensure_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT used_R,left_R FROM risk_budget_strategy WHERE date=date('now','localtime') AND strategy_id=?", (strategy_id,))
        row = cur.fetchone()
        if not row:
            return (0.0, 1.0)
        return (float(row["used_R"] or 0.0), float(row["left_R"] or 1.0))


def set_today_symbol(symbol: str, used_R: float | None = None, left_R: float | None = None) -> None:
    _ensure_table()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        sym = symbol.upper()
        cur.execute(
            """
            INSERT INTO risk_budget_symbol(date, symbol, used_R, left_R)
            VALUES(date('now','localtime'), ?, 0.0, 1.0)
            ON CONFLICT(date, symbol) DO NOTHING
            """,
            (sym,),
        )
        sets, params = [], []
        if used_R is not None:
            sets.append("used_R=?")
            params.append(float(used_R))
        if left_R is not None:
            sets.append("left_R=?")
            params.append(float(left_R))
        if sets:
            cur.execute(f"UPDATE risk_budget_symbol SET {', '.join(sets)} WHERE date=date('now','localtime') AND symbol=?", params + [sym])
        conn.commit()


def get_today_symbol(symbol: str) -> tuple[float, float]:
    _ensure_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT used_R,left_R FROM risk_budget_symbol WHERE date=date('now','localtime') AND symbol=?", (symbol.upper(),))
        row = cur.fetchone()
        if not row:
            return (0.0, 1.0)
        return (float(row["used_R"] or 0.0), float(row["left_R"] or 1.0))


# ===== Loss cap params =====
def get_today_loss_cap_pct(default_pct: float = 0.01) -> float:
    _ensure_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT loss_cap_pct FROM risk_params WHERE date=date('now','localtime')")
        row = cur.fetchone()
        if not row:
            return float(default_pct)
        return float(row["loss_cap_pct"] or default_pct)


def set_today_loss_cap_pct(pct: float) -> None:
    _ensure_table()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO risk_params(date, loss_cap_pct) VALUES(date('now','localtime'), ?)
            ON CONFLICT(date) DO UPDATE SET loss_cap_pct=excluded.loss_cap_pct
            """,
            (float(pct),),
        )
        conn.commit()


