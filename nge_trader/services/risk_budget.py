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


