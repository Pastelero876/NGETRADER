from __future__ import annotations

from pathlib import Path
from typing import Dict

from dotenv import dotenv_values


def read_env() -> dict[str, str]:
    """Lee variables desde `.env` y devuelve un dict con claves en mayúsculas."""
    env_path = Path(".env")
    if not env_path.exists():
        return {}
    data = dotenv_values(str(env_path))
    return {k.upper(): v for k, v in data.items() if v is not None}


def write_env(updated: Dict[str, str | None]) -> None:
    """Escribe/actualiza variables en `.env` manteniendo otras existentes."""
    env_path = Path(".env")
    current = read_env()
    for k, v in updated.items():
        if v is None:
            continue
        current[k.upper()] = v
    lines = [f"{k}={v}" for k, v in current.items()]
    env_path.write_text("\n".join(lines), encoding="utf-8")


