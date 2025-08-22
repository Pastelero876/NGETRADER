from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class AllocationResult:
    weights: Dict[str, float]
    risk: float


class PortfolioEngine:
    """Motor de carteras con asignaciones simples.

    - risk_parity: pesos proporcionales a 1/vol
    - min_variance: solución cerrada w ~ inv(C) * 1 / sum(inv(C) * 1)
    """

    def risk_parity(self, returns: pd.DataFrame) -> AllocationResult:
        if returns.empty:
            return AllocationResult({}, 0.0)
        vol = returns.std(ddof=0).replace(0.0, 1e-9)
        inv_vol = 1.0 / vol
        w = inv_vol / inv_vol.sum()
        weights = {c: float(w.loc[c]) for c in returns.columns}
        port_var = float((returns.cov(ddof=0).values @ w.values).T @ w.values)
        return AllocationResult(weights=weights, risk=port_var**0.5)

    def min_variance(self, returns: pd.DataFrame, ridge: float = 1e-6) -> AllocationResult:
        if returns.empty:
            return AllocationResult({}, 0.0)
        cov = returns.cov(ddof=0).values
        n = cov.shape[0]
        cov_r = cov + ridge * np.eye(n)
        inv = np.linalg.pinv(cov_r)
        ones = np.ones((n, 1))
        w = inv @ ones / float(ones.T @ inv @ ones)
        w = w.flatten()
        cols = list(returns.columns)
        weights = {cols[i]: float(w[i]) for i in range(n)}
        port_var = float(w.T @ cov @ w)
        return AllocationResult(weights=weights, risk=port_var**0.5)

    def black_litterman_basic(
        self,
        returns: pd.DataFrame,
        delta: float = 2.5,
        tau: float = 0.05,
        ridge: float = 1e-6,
    ) -> AllocationResult:
        """Implementación básica de Black-Litterman sin vistas explícitas.

        - Asume pesos de mercado iguales como prior
        - Prior de rendimientos esperados: pi = delta * cov * w_mkt
        - Posterior mean = pi (sin vistas)
        - Solución tipo mean-variance: w ∝ inv(cov) * mu
        """
        if returns.empty or returns.shape[1] < 2:
            return self.min_variance(returns)
        cov = returns.cov(ddof=0).values
        n = cov.shape[0]
        cov_r = cov + ridge * np.eye(n)
        inv = np.linalg.pinv(cov_r)
        w_mkt = np.ones((n, 1)) / n
        pi = delta * cov_r @ w_mkt  # prior expected returns
        mu = pi  # sin vistas -> posterior = prior
        w = inv @ mu
        w = w / float(np.sum(w))
        w = w.flatten()
        cols = list(returns.columns)
        weights = {cols[i]: float(w[i]) for i in range(n)}
        port_var = float(w.T @ cov @ w)
        return AllocationResult(weights=weights, risk=port_var**0.5)


class RiskAllocator:
    def volatility_target_weights(self, symbols_to_series: Dict[str, pd.Series], target_vol_daily: float) -> Dict[str, float]:
        vols: Dict[str, float] = {}
        for sym, s in symbols_to_series.items():
            rets = pd.to_numeric(s, errors="coerce").astype(float).pct_change().dropna()
            v = float(rets.std(ddof=0)) if not rets.empty else 0.0
            vols[sym] = max(v, 1e-9)
        inv_vol = {k: 1.0 / v for k, v in vols.items()}
        total = float(sum(inv_vol.values())) or 1.0
        base_w = {k: v / total for k, v in inv_vol.items()}
        scale = float(target_vol_daily)
        weights = {k: float(min(max(w * scale, 0.0), 1.0)) for k, w in base_w.items()}
        tw = float(sum(weights.values())) or 1.0
        return {k: v / tw for k, v in weights.items()}


