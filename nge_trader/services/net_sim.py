from __future__ import annotations

import os
import random
import time
from typing import Tuple


def _read_bool(env_key: str, default: bool = False) -> bool:
    val = os.getenv(env_key)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def get_network_profile() -> Tuple[bool, float, float]:
    """Lee configuración de simulación de red desde variables de entorno.

    Devuelve (enabled, drop_prob, jitter_ms).
    """
    enabled = _read_bool("NET_SIM_ENABLED", False)
    try:
        drop_prob = float(os.getenv("NET_SIM_DROP_PROB", "0.0"))
    except Exception:
        drop_prob = 0.0
    try:
        jitter_ms = float(os.getenv("NET_SIM_JITTER_MS", "0.0"))
    except Exception:
        jitter_ms = 0.0
    return enabled, max(0.0, min(1.0, drop_prob)), max(0.0, jitter_ms)


def maybe_jitter_and_drop() -> bool:
    """Aplica jitter opcional y devuelve True si se debe dropear el evento.

    - Lee NET_SIM_ENABLED, NET_SIM_DROP_PROB, NET_SIM_JITTER_MS.
    - Si enabled y jitter_ms > 0: duerme un tiempo aleatorio en [0, jitter_ms].
    - Si enabled y rand < drop_prob: retorna True para indicar drop.
    """
    enabled, drop_prob, jitter_ms = get_network_profile()
    if not enabled:
        return False
    if jitter_ms > 0:
        time.sleep(random.uniform(0.0, jitter_ms) / 1000.0)
    if drop_prob > 0.0 and random.random() < drop_prob:
        return True
    return False


