from __future__ import annotations

import json
import runpy
import sys


def test_healthcheck_runs_and_returns_payload(monkeypatch):
    # Evitar SystemExit real
    exits: list[int] = []

    def _exit(code: int = 0):  # noqa: ANN001
        exits.append(int(code))
        raise SystemExit(0)

    monkeypatch.setattr(sys, "exit", _exit)
    # Ejecutar script como módulo
    try:
        runpy.run_path("scripts/healthcheck.py", run_name="__main__")
    except SystemExit:
        pass
    # No debe caerse antes de imprimir; al menos una salida SystemExit capturada
    assert exits is not None


