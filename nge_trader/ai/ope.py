from __future__ import annotations

import numpy as np
import pandas as pd


def off_policy_evaluation(returns: pd.Series, behavior_signals: pd.Series, candidate_signals: pd.Series) -> dict:
    """OPE simplificado por importancia: estima rendimiento de candidate con pesos por propensión.

    returns: serie de retornos del activo
    behavior_signals: señales {-1,0,1} ejecutadas históricamente
    candidate_signals: señales {-1,0,1} de la política candidata
    """
    r = returns.fillna(0.0).values
    b = behavior_signals.fillna(0.0).values
    c = candidate_signals.fillna(0.0).values
    # Propensión simple: p(a) ~ frecuencia de acciones en behavior
    unique, counts = np.unique(b, return_counts=True)
    freq = {u: counts[i] / float(len(b)) for i, u in enumerate(unique)}
    p_b = np.array([freq.get(a, 1e-6) for a in b])
    p_c = np.array([freq.get(a, 1e-6) for a in c])  # aproximación si no conocemos propensión real
    w = (p_c / np.maximum(p_b, 1e-9))
    # Clipping para estabilidad
    w = np.clip(w, 0.1, 10.0)
    # Métricas: retorno medio ponderado, Sharpe, y varianza de IS
    strat_ret = r * c  # payoff simple
    wr = strat_ret * w
    est = np.mean(wr)
    vol = np.std(wr) or 1e-9
    sharpe = est / vol
    is_var = float(np.var(w))
    return {"weighted_mean": float(est), "weighted_sharpe": float(sharpe), "importance_variance": is_var}


def off_policy_evaluation_ci(
    returns: pd.Series,
    behavior_signals: pd.Series,
    candidate_signals: pd.Series,
    n_boot: int = 500,
    alpha: float = 0.05,
) -> dict:
    """OPE con bandas (bootstrap) para media ponderada y Sharpe.

    Devuelve estimadores puntuales y bandas [low, high].
    """
    base = off_policy_evaluation(returns, behavior_signals, candidate_signals)
    r = returns.fillna(0.0).values
    b = behavior_signals.fillna(0.0).values
    c = candidate_signals.fillna(0.0).values
    import numpy as np
    unique, counts = np.unique(b, return_counts=True)
    freq = {u: counts[i] / float(len(b)) for i, u in enumerate(unique)}
    p_b = np.array([freq.get(a, 1e-6) for a in b])
    p_c = np.array([freq.get(a, 1e-6) for a in c])
    w = (p_c / np.maximum(p_b, 1e-9))
    w = np.clip(w, 0.1, 10.0)
    strat_ret = r * c
    wr = strat_ret * w
    means = []
    sharpes = []
    rs = np.random.RandomState(42)
    n = len(wr)
    for _ in range(max(10, int(n_boot))):
        idx = rs.choice(n, size=n, replace=True)
        wr_b = wr[idx]
        m = float(np.mean(wr_b))
        v = float(np.std(wr_b)) or 1e-9
        means.append(m)
        sharpes.append(m / v)
    low = int(alpha / 2 * len(means))
    high = int((1 - alpha / 2) * len(means)) - 1
    means_sorted = sorted(means)
    sharpe_sorted = sorted(sharpes)
    return {
        **base,
        "mean_ci": [float(means_sorted[low]), float(means_sorted[high])],
        "sharpe_ci": [float(sharpe_sorted[low]), float(sharpe_sorted[high])],
    }

