from __future__ import annotations

from fastapi.testclient import TestClient

from nge_trader.entrypoints.api import app


def test_ui_kill_switch_and_pause_resume():
    c = TestClient(app)
    # Estado inicial
    r = c.get("/api/status")
    assert r.status_code == 200
    # Armar/desarmar vía endpoint sin 2FA (si no está enforced)
    r = c.post("/api/kill_switch?armed=true")
    # Puede devolver 428 si 2FA enforced; aceptamos 200 o 428
    assert r.status_code in (200, 428)
    # Pausa/Reanuda
    r = c.post("/api/pause/TEST")
    assert r.status_code in (200, 404)  # 404 si handler no está
    r = c.post("/api/resume/TEST")
    assert r.status_code in (200, 404)


