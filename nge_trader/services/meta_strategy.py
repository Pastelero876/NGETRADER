from __future__ import annotations

import random
from typing import Callable

from nge_trader.config.settings import Settings


class CanaryRouter:
    """Enrutador simple que divide tráfico entre primary y canary por porcentaje.

    Uso:
      router = CanaryRouter(Settings().canary_traffic_pct)
      fn = router.choose(primary_fn, canary_fn)
      fn()
    """

    def __init__(self, canary_pct: float) -> None:
        self.canary_pct = max(0.0, min(1.0, float(canary_pct)))

    def choose(self, primary: Callable, canary: Callable | None) -> Callable:
        if canary is None or self.canary_pct <= 0.0:
            return primary
        if random.random() < self.canary_pct:
            return canary
        return primary

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class EnsembleConfig:
    decay: float = 0.94  # EWMA para performance reciente


class MetaStrategy:
    """Ensemble de señales {-1,0,1} con ponderación por performance reciente.

    - Calcula retorno de cada señal (pos desplazada * retorno del activo)
    - EWMA del rendimiento; pesos ~ performance_pos
    - Devuelve señal combinada en [-1,1]
    """

    def __init__(self, cfg: EnsembleConfig | None = None) -> None:
        self.cfg = cfg or EnsembleConfig()

    def combine(self, price_df: pd.DataFrame, signals: Dict[str, pd.Series]) -> pd.Series:
        if not signals:
            return pd.Series(dtype=float)
        close = price_df["close"].astype(float).reset_index(drop=True)
        ret = close.pct_change().fillna(0.0)
        aligned = {k: s.reset_index(drop=True).fillna(0.0).clip(-1.0, 1.0) for k, s in signals.items()}
        perf = {}
        alpha = 1.0 - self.cfg.decay
        for key, s in aligned.items():
            strat_ret = (s.shift(1).fillna(0.0) * ret)
            ew = strat_ret.ewm(alpha=alpha, adjust=False).mean()
            perf[key] = ew
        # pesos normalizados positivos
        perf_df = pd.DataFrame(perf).clip(lower=0.0)
        weight_sum = perf_df.sum(axis=1).replace(0.0, 1e-9)
        weights = perf_df.div(weight_sum, axis=0).fillna(0.0)
        # señal final
        sig_df = pd.DataFrame(aligned)
        final_sig = (weights * sig_df).sum(axis=1).clip(-1.0, 1.0)
        final_sig.name = "signal"
        return final_sig


class RegimeWeightedMeta:
    """Combina señales momentum y mean-reversion ponderadas por régimen detectado.

    - Espera keys 'momentum' y 'mean_reversion' en `signals`.
    - Usa `RegimeDetector` (k-means) para clasificar el régimen sobre `price_df`.
    - Asigna pesos por régimen: tendencia → momentum 0.7, MR 0.3; lateral → MR 0.7, momentum 0.3.
    """

    def __init__(self, lookback: int = 60) -> None:
        self.lookback = lookback

    def combine(self, price_df: pd.DataFrame, signals: Dict[str, pd.Series]) -> pd.Series:
        if not signals:
            return pd.Series(dtype=float)
        from nge_trader.ai.regime import RegimeDetector
        rd = RegimeDetector(lookback=self.lookback, k=2, seed=42)
        try:
            rd.fit(price_df)
            regime = int(rd.predict(price_df))
        except Exception:
            regime = 0
        mom = signals.get("momentum")
        mr = signals.get("mean_reversion")
        if mom is None or mr is None:
            # fallback a ensemble clásico
            return MetaStrategy().combine(price_df, signals)
        mom = mom.reset_index(drop=True).fillna(0.0).clip(-1.0, 1.0)
        mr = mr.reset_index(drop=True).fillna(0.0).clip(-1.0, 1.0)
        if regime == 1:
            w_mom, w_mr = 0.7, 0.3
        else:
            w_mom, w_mr = 0.3, 0.7
        n = min(len(mom), len(mr))
        final = (w_mom * mom.iloc[-n:].values + w_mr * mr.iloc[-n:].values)
        s = pd.Series(final, name="signal")
        return s


class RegimeOPEPromoter:
    """Selecciona mezcla por régimen y valida con OPE en ventana rodante.

    - Usa `RegimeDetector` para clasificar.
    - Evalúa cada sub-política con OPE (WIS simple) y el mix; expone métricas para validación continua.
    """

    def __init__(self, lookback: int = 60) -> None:
        self.lookback = lookback

    def evaluate(self, price_df: pd.DataFrame, behavior: pd.Series, candidates: Dict[str, pd.Series]) -> dict:
        from nge_trader.ai.ope import off_policy_evaluation
        from nge_trader.ai.regime import RegimeDetector
        close = price_df["close"].astype(float).reset_index(drop=True)
        ret = close.pct_change().fillna(0.0)
        rd = RegimeDetector(lookback=self.lookback, k=2, seed=42)
        try:
            rd.fit(price_df)
            regime = int(rd.predict(price_df))
        except Exception:
            regime = 0
        # Mezcla simple según régimen
        mom = candidates.get("momentum", pd.Series(dtype=float)).reset_index(drop=True).fillna(0.0)
        mr = candidates.get("mean_reversion", pd.Series(dtype=float)).reset_index(drop=True).fillna(0.0)
        n = min(len(mom), len(mr), len(ret), len(behavior))
        if n <= 3:
            return {"error": "insufficient data"}
        mom = mom.iloc[-n:]
        mr = mr.iloc[-n:]
        ret = ret.iloc[-n:]
        behavior = behavior.reset_index(drop=True).fillna(0.0).iloc[-n:]
        if regime == 1:
            mix = 0.7 * mom + 0.3 * mr
        else:
            mix = 0.3 * mom + 0.7 * mr
        out = {"regime": regime}
        out["mix"] = off_policy_evaluation(ret, behavior, mix)
        if len(mom) == len(behavior):
            out["momentum"] = off_policy_evaluation(ret, behavior, mom)
        if len(mr) == len(behavior):
            out["mean_reversion"] = off_policy_evaluation(ret, behavior, mr)
        return out

