from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


STORE_PATH = Path("data/strategies.json")
ASSIGN_PATH = Path("data/assignments.json")


@dataclass
class StrategyConfig:
    key: str
    params: Dict[str, Any]


class StrategyStore:
    def __init__(self) -> None:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> List[StrategyConfig]:
        if not STORE_PATH.exists():
            return []
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        return [StrategyConfig(**item) for item in data]

    def save_all(self, configs: List[StrategyConfig]) -> None:
        data = [asdict(c) for c in configs]
        STORE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Asignaciones símbolo->estrategia
    def load_assignments(self) -> List[StrategyConfig]:
        if not ASSIGN_PATH.exists():
            return []
        data = json.loads(ASSIGN_PATH.read_text(encoding="utf-8"))
        return [StrategyConfig(**item) for item in data]

    def save_assignments(self, configs: List[StrategyConfig]) -> None:
        ASSIGN_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(c) for c in configs]
        ASSIGN_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


