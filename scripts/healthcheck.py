from __future__ import annotations

from nge_trader.services.app_service import AppService
from nge_trader.repository.db import Database
from nge_trader.services.time_sync import estimate_skew_ms
import os


def main() -> None:
    svc = AppService()
    st = svc.get_connectivity_status()
    ok = any(st.values())
    # Métricas: tamaño del outbox por estado
    db = Database()
    outbox = db.count_outbox_by_status()
    # Skew de reloj
    skew_ms = estimate_skew_ms(server=svc.settings.ntp_server)
    # Umbrales
    max_skew = float(os.getenv("HC_MAX_SKEW_MS", svc.settings.max_time_skew_ms or 100))
    skew_ok = abs(skew_ms) <= max_skew
    # Umbral de outbox por estado (errores viejos)
    ages = db.oldest_outbox_age_seconds_by_status()
    max_outbox_age = float(os.getenv("HC_MAX_OUTBOX_AGE_SEC", "600"))
    outbox_ok = all((age <= max_outbox_age) for age in ages.values()) if ages else True
    # Estado compuesto
    payload = {
        "connectivity": st,
        "outbox": outbox,
        "skew_ms": skew_ms,
        "outbox_ages": ages,
        "ok": bool(ok and skew_ok and outbox_ok),
    }
    print(payload)
    # Códigos: 0 OK; 2 fallo general; 3 skew; 4 outbox
    if not ok:
        raise SystemExit(2)
    if not skew_ok:
        raise SystemExit(3)
    if not outbox_ok:
        raise SystemExit(4)
    raise SystemExit(0)


if __name__ == "__main__":
    main()


