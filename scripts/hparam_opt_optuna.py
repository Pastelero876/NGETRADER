from __future__ import annotations

import json
import optuna
import numpy as np

from nge_trader.ai.online import OnlineLinearPolicy, OnlineConfig
from nge_trader.services.app_service import AppService
from nge_trader.repository.db import Database


def objective(trial: optuna.trial.Trial) -> float:
    symbol = "AAPL"
    svc = AppService()
    df = svc.data_provider.get_daily_adjusted(symbol)
    L = trial.suggest_int("lookback", 10, 60)
    lr = trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True)
    l2 = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True)
    cfg = OnlineConfig(lookback=L, learning_rate=lr, l2_reg=l2)
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


def main(n_trials: int = 30) -> None:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params | {"value": study.best_value}
    print(json.dumps({"best": best}, ensure_ascii=False))
    Database().record_training_run("AAPL", int(best.get("lookback", 0)), None, None, json.dumps(best), None)


if __name__ == "__main__":
    main()


