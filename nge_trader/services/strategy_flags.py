from __future__ import annotations

import os
from typing import Dict

_SHADOW: Dict[str, bool] = {}


def is_shadow(name: str) -> bool:
    if name in _SHADOW:
        return bool(_SHADOW[name])
    default = str(os.getenv("SHADOW_DEFAULT", "false")).strip().lower() == "true"
    return bool(default)


def set_shadow(name: str, enabled: bool) -> None:
    _SHADOW[str(name)] = bool(enabled)


def list_shadow() -> Dict[str, bool]:
    return dict(_SHADOW)


