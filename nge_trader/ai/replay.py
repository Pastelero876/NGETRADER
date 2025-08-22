from __future__ import annotations

import json
from typing import List, Optional

from nge_trader.repository.db import Database


class ReplayBuffer:
    """Buffer de experiencias persistente sobre SQLite.

    Guarda y samplea experiencias para entrenamiento online.
    """

    def __init__(self) -> None:
        self.db = Database()

    def append(self, ts_iso: str, symbol: Optional[str], state: dict, action: float, reward: float, next_state: dict, done: bool, priority: float | None = None) -> None:
        row = {
            "ts": ts_iso,
            "symbol": symbol,
            "state": json.dumps(state, ensure_ascii=False),
            "action": float(action),
            "reward": float(reward),
            "next_state": json.dumps(next_state, ensure_ascii=False),
            "done": bool(done),
            "priority": float(priority if priority is not None else abs(float(reward))),
        }
        self.db.append_experiences([row])

    def sample(self, limit: int = 1000) -> List[dict]:
        rows = self.db.sample_experiences_prioritized(limit=limit, top_ratio=0.7)
        for r in rows:
            try:
                r["state"] = json.loads(r.get("state") or "{}")
                r["next_state"] = json.loads(r.get("next_state") or "{}")
            except Exception:
                pass
        return rows


