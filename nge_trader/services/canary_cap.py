from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import os

try:
    from prometheus_client import Gauge  # type: ignore
except Exception:  # pragma: no cover
    Gauge = None  # type: ignore

canary_notional_used_pct = Gauge("canary_notional_used_pct", "% notional canario usado (día)") if Gauge else None  # type: ignore


def _today_utc():
    return datetime.now(timezone.utc).date()


def compute_used_pct(db) -> float:
    """Calcula notional_canary/total del día a partir de tca_events (DB Postgres).

    Ignora eventos con meta.shadow=true.
    Acepta `db` como engine/conexión SQLAlchemy; si es None usa TCA_DB_URL.
    """
    try:
        from sqlalchemy import create_engine, text  # type: ignore
    except Exception:
        return 0.0
    engine = None
    try:
        if db is None:
            url = os.getenv("TCA_DB_URL", "")
            if not url:
                return 0.0
            engine = create_engine(url, pool_pre_ping=True)
            cx = engine.begin()
        else:
            # db puede ser engine o connection con .execute
            cx = db.begin() if hasattr(db, "begin") else db
        q = text(
            """
WITH today AS (
  SELECT date_trunc('day', now() AT TIME ZONE 'UTC') AS d0,
         date_trunc('day', now() AT TIME ZONE 'UTC') + interval '1 day' AS d1
)
SELECT
  COALESCE(SUM(CASE WHEN (meta->>'canary')::boolean IS TRUE THEN ABS(price*qty) ELSE 0 END),0) AS canary_notional,
  COALESCE(SUM(ABS(price*qty)),0) AS total_notional
FROM tca_events, today
WHERE ts >= (SELECT d0 FROM today) AND ts < (SELECT d1 FROM today)
  AND price IS NOT NULL AND qty IS NOT NULL
  AND (meta->>'shadow')::boolean IS DISTINCT FROM TRUE
            """
        )
        r = cx.execute(q).mappings().first()  # type: ignore[attr-defined]
        canary = float((r or {}).get("canary_notional", 0.0) or 0.0)
        total = float((r or {}).get("total_notional", 0.0) or 0.0)
        if not (total > 0.0):
            return 0.0
        return float(canary / total)
    except Exception:
        return 0.0
    finally:
        try:
            if engine is not None:
                engine.dispose()
        except Exception:
            pass


def gate(db, max_pct: float) -> bool:
    used = float(compute_used_pct(db))
    try:
        if canary_notional_used_pct is not None:  # type: ignore
            canary_notional_used_pct.set(used)  # type: ignore[attr-defined]
    except Exception:
        pass
    return bool(used >= float(max_pct))


