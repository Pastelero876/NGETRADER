from __future__ import annotations

import json
from typing import Any

from nge_trader.repository.db import Database


def append_event(event: str, details: dict[str, Any]) -> None:
    payload = {"event": event, "details": details}
    Database().append_audit_log(json.dumps(payload, ensure_ascii=False))


def verify_chain(limit: int = 1000) -> bool:
    return Database().verify_audit_log(limit=limit)


