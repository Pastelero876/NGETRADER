from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


class RegimeDetector:
    """Detector de régimen simple basado en clustering K-Means sobre features de retornos.

    Features: media y desviación estándar de retornos en ventana.
    """

    def __init__(self, lookback: int = 60, k: int = 2, seed: int = 42) -> None:
        self.lookback = lookback
        self.k = k
        self.seed = seed
        self.centroids: np.ndarray | None = None

    @staticmethod
    def _kmeans(X: np.ndarray, k: int, seed: int = 42, iters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        rs = np.random.RandomState(seed)
        idx = rs.choice(len(X), size=k, replace=False)
        C = X[idx].copy()
        for _ in range(iters):
            dists = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
            labels = dists.argmin(axis=1)
            newC = np.vstack([X[labels == j].mean(axis=0) if np.any(labels == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        return C, labels

    def fit(self, df: pd.DataFrame) -> None:
        close = df["close"].astype(float).values
        rets = np.diff(np.log(close), prepend=np.log(close[0]))
        if len(rets) < self.lookback + 2:
            self.centroids = None
            return
        feats = []
        for i in range(self.lookback, len(rets)):
            w = rets[i - self.lookback : i]
            feats.append([float(w.mean()), float(w.std(ddof=0))])
        X = np.array(feats)
        C, _ = self._kmeans(X, self.k, self.seed)
        self.centroids = C

    def predict(self, df: pd.DataFrame) -> int:
        if self.centroids is None:
            self.fit(df)
            if self.centroids is None:
                return 0
        close = df["close"].astype(float).values
        rets = np.diff(np.log(close), prepend=np.log(close[0]))
        w = rets[-self.lookback :]
        x = np.array([float(w.mean()), float(w.std(ddof=0))])
        d = np.linalg.norm(self.centroids - x[None, :], axis=1)
        return int(d.argmin())


