from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_target_position(current_vol: float, target_vol: float) -> float:
    if current_vol <= 0:
        return 1.0
    w = target_vol / current_vol
    return float(max(min(w, 3.0), 0.0))


def compute_var_es(returns: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    if returns.empty:
        return 0.0, 0.0
    sorted_ret = np.sort(returns.dropna().values)
    n = len(sorted_ret)
    k = max(int(alpha * n), 1)
    var = float(sorted_ret[k - 1])
    es = float(sorted_ret[:k].mean())
    return var, es


def compute_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """CVaR/ES a nivel alpha (convención: retorno; valor negativo es pérdida esperada en cola)."""
    if returns.empty:
        return 0.0
    _, es = compute_var_es(returns, alpha=alpha)
    return float(es)


def concentration_limits_breached(weights: pd.Series, max_per_symbol: float = 0.2, corr_matrix: pd.DataFrame | None = None, max_avg_corr: float = 0.7) -> bool:
    """Chequea límites de concentración y correlación promedio.

    - weights: ponderaciones por símbolo.
    - max_per_symbol: límite de concentración por activo.
    - corr_matrix: matriz de correlación opcional.
    - max_avg_corr: límite promedio de correlación entre activos.
    """
    if not isinstance(weights, pd.Series) or weights.empty:
        return False
    if float(weights.abs().max()) > float(max_per_symbol):
        return True
    if corr_matrix is not None and not corr_matrix.empty:
        try:
            # correlación promedio (off-diagonal)
            m = corr_matrix.values
            n = m.shape[0]
            if n > 1:
                avg = (m.sum() - np.trace(m)) / (n * (n - 1))
                if float(abs(avg)) > float(max_avg_corr):
                    return True
        except Exception:
            return False
    return False


def intraday_es_cvar(series_prices: pd.Series, window_minutes: int = 60, alpha: float = 0.025) -> tuple[float, float]:
    """Calcula VaR/ES intradía (percentiles de retornos) en ventana reciente.

    Retorna (VaR, ES) como valores negativos si pérdidas.
    """
    if series_prices is None or series_prices.empty:
        return 0.0, 0.0
    s = series_prices.astype(float).reset_index(drop=True)
    rets = s.pct_change().dropna()
    if rets.empty:
        return 0.0, 0.0
    var, es = compute_var_es(rets, alpha=alpha)
    return float(var), float(es)

