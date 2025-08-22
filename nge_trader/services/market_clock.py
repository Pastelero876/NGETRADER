from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from nge_trader.repository.db import Database


@dataclass
class Session:
    open_time: _dt.time
    close_time: _dt.time
    premarket_open: _dt.time | None = None
    after_hours_close: _dt.time | None = None


EXCHANGE_SESSIONS: dict[str, Session] = {
    "BINANCE": Session(open_time=_dt.time(0, 0), close_time=_dt.time(23, 59)),
    "COINBASE": Session(open_time=_dt.time(0, 0), close_time=_dt.time(23, 59)),
    "NYSE": Session(open_time=_dt.time(9, 30), close_time=_dt.time(16, 0), premarket_open=_dt.time(4, 0), after_hours_close=_dt.time(20, 0)),
}

# Feriados simples (ejemplo) por exchange (YYYY-MM-DD)
_HOLIDAYS: dict[str, set[str]] = {
    "NYSE": {
        "2025-01-01",  # New Year's Day
        "2025-01-20",  # MLK Day
        "2025-02-17",  # Presidents' Day
        "2025-04-18",  # Good Friday
        "2025-05-26",  # Memorial Day
        "2025-06-19",  # Juneteenth
        "2025-07-04",  # Independence Day
        "2025-09-01",  # Labor Day
        "2025-11-27",  # Thanksgiving
        "2025-12-25",  # Christmas Day
    }
}


def is_holiday(exchange: str, date: _dt.date) -> bool:
    days = _HOLIDAYS.get(exchange.upper()) or set()
    return date.isoformat() in days


def is_session_open(exchange: str, now_utc: _dt.datetime | None = None, include_premarket: bool = True, include_after_hours: bool = True) -> bool:
    now = now_utc or _dt.datetime.now(_dt.UTC)
    sess = EXCHANGE_SESSIONS.get(exchange.upper())
    if not sess:
        return True
    # Feriado
    if is_holiday(exchange, now.date()):
        return False
    t = now.time()
    # Mercado 24/7
    if sess.open_time <= _dt.time(0, 0) and sess.close_time >= _dt.time(23, 59):
        return True
    # Premarket
    if include_premarket and sess.premarket_open and sess.premarket_open <= t < sess.open_time:
        return True
    # Regular
    if sess.open_time <= t <= sess.close_time:
        return True
    # After-hours
    if include_after_hours and sess.after_hours_close and sess.close_time < t <= sess.after_hours_close:
        return True
    # Cache resultado del día
    try:
        db = Database()
        date_iso = now.date().isoformat()
        is_open = (
            (sess.open_time <= _dt.time(0, 0) and sess.close_time >= _dt.time(23, 59))
            or (include_premarket and sess.premarket_open and sess.premarket_open <= t < sess.open_time)
            or (sess.open_time <= t <= sess.close_time)
            or (include_after_hours and sess.after_hours_close and sess.close_time < t <= sess.after_hours_close)
        )
        db.set_market_calendar(exchange.upper(), date_iso, "regular", bool(is_open))
    except Exception:
        pass
    return False


def is_session_open_cached(exchange: str, now_utc: _dt.datetime | None = None) -> bool:
    """Consulta la caché por día y, si no existe, delega a is_session_open y cachea el resultado.

    Cache por (exchange, date, session='regular').
    """
    now = now_utc or _dt.datetime.now(_dt.UTC)
    date_iso = now.date().isoformat()
    try:
        db = Database()
        val = db.get_market_calendar(exchange.upper(), date_iso, "regular")
        if val is not None:
            return bool(val)
    except Exception:
        val = None
    # calcular y cachear
    return is_session_open(exchange, now)


def session_details(exchange: str, now_utc: _dt.datetime | None = None) -> dict:
    """Devuelve detalles de sesión para UI/API: label y horarios.

    Labels: 'open', 'premarket', 'after_hours', 'closed', '24x7'
    """
    now = now_utc or _dt.datetime.now(_dt.UTC)
    sess = EXCHANGE_SESSIONS.get(exchange.upper())
    if not sess:
        return {"session": "24x7", "regular_open": None, "regular_close": None, "premarket_open": None, "after_hours_close": None, "countdown_seconds": None}
    if is_holiday(exchange, now.date()):
        label = "closed"
    else:
        t = now.time()
        if sess.open_time <= _dt.time(0, 0) and sess.close_time >= _dt.time(23, 59):
            label = "24x7"
        elif sess.premarket_open and sess.premarket_open <= t < sess.open_time:
            label = "premarket"
        elif sess.open_time <= t <= sess.close_time:
            label = "open"
        elif sess.after_hours_close and sess.close_time < t <= sess.after_hours_close:
            label = "after_hours"
        else:
            label = "closed"
    # Calcular cuenta atrás al siguiente hito
    def _to_dt(time_obj: _dt.time, base: _dt.datetime) -> _dt.datetime:
        return _dt.datetime.combine(base.date(), time_obj, tzinfo=_dt.UTC)
    countdown: int | None = None
    try:
        if label == "24x7":
            countdown = None
        elif label == "premarket" and sess.open_time:
            target = _to_dt(sess.open_time, now)
            countdown = int(max((target - now).total_seconds(), 0))
        elif label == "open" and sess.close_time:
            target = _to_dt(sess.close_time, now)
            countdown = int(max((target - now).total_seconds(), 0))
        elif label == "after_hours" and sess.after_hours_close:
            target = _to_dt(sess.after_hours_close, now)
            countdown = int(max((target - now).total_seconds(), 0))
        else:
            # closed: próximo premarket si existe, si no próximo open (día siguiente si ya pasó)
            if sess.premarket_open:
                tgt = _to_dt(sess.premarket_open, now)
            else:
                tgt = _to_dt(sess.open_time, now)
            if tgt <= now:
                tgt = tgt + _dt.timedelta(days=1)
            countdown = int(max((tgt - now).total_seconds(), 0))
    except Exception:
        countdown = None
    return {
        "session": label,
        "regular_open": sess.open_time.isoformat() if sess.open_time else None,
        "regular_close": sess.close_time.isoformat() if sess.close_time else None,
        "premarket_open": sess.premarket_open.isoformat() if sess.premarket_open else None,
        "after_hours_close": sess.after_hours_close.isoformat() if sess.after_hours_close else None,
        "countdown_seconds": countdown,
    }

