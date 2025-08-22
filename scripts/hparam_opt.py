from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from nge_trader.ai.online import OnlineLinearPolicy, OnlineConfig
from nge_trader.services.app_service import AppService
from nge_trader.repository.db import Database


@dataclass
class SearchSpace:
    lookbacks: Tuple[int, ...] = (20, 30, 40)
    lrs: Tuple[float, ...] = (0.005, 0.01, 0.02)
    regs: Tuple[float, ...] = (1e-5, 1e-4, 1e-3)


def score_config(symbol: str, cfg: OnlineConfig) -> float:
    svc = AppService()
    df = svc.data_provider.get_daily_adjusted(symbol)
    if df.empty or len(df) < cfg.lookback + 10:
        return -1e9
    pol = OnlineLinearPolicy(cfg)
    X, y = pol._features(df)  # type: ignore[attr-defined]
    if len(y) == 0:
        return -1e9
    equity = 1.0
    for i in range(len(y)):
        state = X[i]
        ret = y[i]
        action = np.sign(pol.predict_score(state))
        equity *= (1.0 + float(action) * float(ret))
        pol.update(state, float(action) * float(ret))
    return float(equity - 1.0)


def main() -> None:
    space = SearchSpace()
    best = (None, -1e18)
    results = []
    symbol = "AAPL"
    for L in space.lookbacks:
        for lr in space.lrs:
            for reg in space.regs:
                cfg = OnlineConfig(lookback=L, learning_rate=lr, l2_reg=reg)
                sc = score_config(symbol, cfg)
                results.append({"lookback": L, "lr": lr, "l2": reg, "score": sc})
                if sc > best[1]:
                    best = (cfg, sc)
    print(json.dumps({"best": best[1], "config": best[0].__dict__ if best[0] else None, "results": results}, ensure_ascii=False))
    Database().record_training_run(symbol, best[0].lookback if best[0] else 0, None, None, json.dumps({"best": best[1]}), None)


if __name__ == "__main__":
    main()


