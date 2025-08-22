from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import json


@dataclass
class PolicyConfig:
    lookback: int = 30
    model_path: str = "data/models/policy_rf.joblib"


class AgentPolicy:
    """Política de IA simple (clasificador) que emite señales {-1,0,1}."""

    def __init__(self, cfg: PolicyConfig | None = None) -> None:
        self.cfg = cfg or PolicyConfig()
        # Modelo ligero: promedio ponderado de retornos recientes
        self.model: dict[str, float] = {"threshold": 0.0}

    def _features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        close = df["close"].astype(float).values
        # Retornos log y ventana
        rets = np.diff(np.log(close), prepend=np.log(close[0]))
        X = []
        y = []
        L = self.cfg.lookback
        for i in range(L, len(close) - 1):
            window = rets[i - L : i]
            X.append(window)
            # etiqueta: signo del retorno próximo
            y.append(1 if rets[i + 1] > 0 else 0)
        return np.array(X), np.array(y)

    def fit(self, df: pd.DataFrame) -> None:
        X, y = self._features(df)
        if len(X) == 0:
            return
        # Umbral = media de retornos próximos sobre el histórico
        close = df["close"].astype(float).values
        rets = np.diff(np.log(close), prepend=np.log(close[0]))
        future = rets[self.cfg.lookback + 1 :]
        thr = float(np.nanmean(future)) if future.size else 0.0
        self.model = {"threshold": thr}

    def predict_signal(self, df: pd.DataFrame) -> float:
        # Usa última ventana
        close = df["close"].astype(float).values
        if len(close) <= self.cfg.lookback + 1:
            return 0.0
        rets = np.diff(np.log(close), prepend=np.log(close[0]))
        window = rets[-self.cfg.lookback :]
        score = float(window.mean() - self.model.get("threshold", 0.0))
        if score > 0:
            return 1.0
        if score < 0:
            return -1.0
        return 0.0

    def save(self) -> None:
        path = self.cfg.model_path
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model, f)

    def load(self) -> None:
        with open(self.cfg.model_path, "r", encoding="utf-8") as f:
            self.model = json.load(f)

    def generate_signals_series(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        """Genera serie de señales aplicando la política sobre una ventana deslizante."""
        signals = []
        for i in range(len(df)):
            sub = df.iloc[: i + 1]
            sig = self.predict_signal(sub)
            signals.append(sig)
        s = pd.Series(signals, index=df.index, name="signal")
        return s

    def walk_forward(self, df: pd.DataFrame, folds: int = 5) -> dict:
        """Entrenamiento walk-forward simple: divide en pliegues secuenciales.

        Devuelve métricas out-of-sample y guarda el mejor modelo.
        """
        n = len(df)
        if n < (self.cfg.lookback * 4):
            return {"error": "dataset too small"}
        fold_size = max((n - self.cfg.lookback) // max(folds, 1), self.cfg.lookback + 1)
        results = []
        best = None
        best_metric = -1e9
        for k in range(folds):
            end_train = min(self.cfg.lookback + fold_size * (k + 1), n - 2)
            start_train = max(0, end_train - fold_size)
            train_df = df.iloc[start_train:end_train]
            test_df = df.iloc[end_train - self.cfg.lookback : end_train + fold_size]
            if len(test_df) <= self.cfg.lookback + 2:
                break
            # fit en train
            self.fit(train_df)
            # evaluar en test
            sig = self.generate_signals_series(test_df)
            close = test_df["close"].astype(float).reset_index(drop=True)
            ret = close.pct_change().fillna(0.0)
            strat_ret = ret * sig.shift(1).fillna(0.0)
            # métrica: media/vol (Sharpe aproximado sin rf)
            vol = float(strat_ret.std(ddof=0)) or 1e-9
            metric = float(strat_ret.mean() / vol)
            results.append({"fold": k + 1, "metric": metric})
            if metric > best_metric:
                best_metric = metric
                best = dict(self.model)
        # restaurar mejor
        if best is not None:
            self.model = best
            try:
                from nge_trader.repository.db import Database
                import json as _json
                Database().record_training_run(
                    symbol=None,
                    lookback=self.cfg.lookback,
                    train_start=None,
                    train_end=None,
                    metrics_json=_json.dumps({"folds": results, "best": best_metric}),
                    model_path=self.cfg.model_path,
                )
            except Exception:
                pass
        return {"folds": results, "best_metric": best_metric}


