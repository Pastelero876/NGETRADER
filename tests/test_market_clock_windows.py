from __future__ import annotations

import datetime as dt

from nge_trader.services.market_clock import is_session_open, is_holiday


def test_nyse_premarket_and_regular():
    # Premarket 04:00-09:30 (usar UTC sin DST exacto; probamos premarket/regular)
    t_premarket = dt.datetime(2025, 1, 2, 8, 0)
    assert is_session_open("NYSE", now_utc=t_premarket, include_premarket=True)
    # Regular 09:30-16:00
    t_regular = dt.datetime(2025, 1, 2, 10, 0)
    assert is_session_open("NYSE", now_utc=t_regular)


def test_nyse_after_hours_and_closed():
    # After-hours hasta 20:00
    t_after = dt.datetime(2025, 1, 2, 19, 30)
    assert is_session_open("NYSE", now_utc=t_after, include_after_hours=True)
    # Fuera de horario
    t_closed = dt.datetime(2025, 1, 2, 3, 59)
    assert not is_session_open("NYSE", now_utc=t_closed, include_premarket=True)
    # Holiday (New Year's Day)
    t_holiday = dt.datetime(2025, 1, 1, 12, 0)
    assert is_holiday("NYSE", t_holiday.date())
    assert not is_session_open("NYSE", now_utc=t_holiday)


