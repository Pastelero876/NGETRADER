from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class OnlineConfig:
    lookback: int = 30
    learning_rate: float = 0.01
    l2_reg: float = 1e-4
    reward_vol_penalty: float = 0.0  # penalización por volatilidad del estado


class OnlineLinearPolicy:
    """Política online tipo contextual bandit con regresión lineal (SGD).

    - Estado: ventana de retornos log de tamaño `lookback`.
    - Acción: {-1, 0, +1} según el signo del score lineal.
    - Actualización: minimiza (r_hat - reward)^2 con regularización L2.
    """

    def __init__(self, cfg: Optional[OnlineConfig] = None) -> None:
        self.cfg = cfg or OnlineConfig()
        self._w = np.zeros(self.cfg.lookback, dtype=float)
        self._b = 0.0

    def _features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        close = df["close"].astype(float).values
        rets = np.diff(np.log(close), prepend=np.log(close[0]))
        L = self.cfg.lookback
        if len(rets) < L + 2:
            return np.empty((0, L)), np.empty((0,))
        X = np.stack([rets[i - L : i] for i in range(L, len(rets))])
        y = rets[L:]
        return X, y

    def state_from_df(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._features(df)
        return X[-1] if len(X) else np.zeros(self.cfg.lookback, dtype=float)

    def predict_score(self, state: np.ndarray) -> float:
        return float(self._w.dot(state) + self._b)

    def predict_action(self, state: np.ndarray) -> float:
        score = self.predict_score(state)
        if score > 0:
            return 1.0
        if score < 0:
            return -1.0
        return 0.0

    def update(self, state: np.ndarray, reward: float) -> None:
        # Reward robusta: penalización por volatilidad + Huber
        if self.cfg.reward_vol_penalty and state.size > 0:
            vol = float(np.std(state))
            reward = float(reward) - float(self.cfg.reward_vol_penalty) * vol
        # Huber loss aproximado (delta=1.0) en el error (pred - reward)
        pred = self._w.dot(state) + self._b
        err_raw = (pred - float(reward))
        delta = 1.0
        if abs(err_raw) <= delta:
            err = err_raw
            huber_grad = err
        else:
            err = delta * (1 if err_raw > 0 else -1)
            huber_grad = err
        # SGD sobre MSE/Huber: grad = 2*grad_err*x + 2*lambda*w
        grad_w = 2.0 * err * state + 2.0 * self.cfg.l2_reg * self._w
        grad_b = 2.0 * err
        self._w -= self.cfg.learning_rate * grad_w
        self._b -= self.cfg.learning_rate * grad_b

    def update_with_costs(self, state: np.ndarray, pnl: float, slippage_bps: float, fees: float, drawdown_local: float, lambdas: dict | None = None) -> None:
        """Actualiza usando recompensa con costes: r = pnl - λ_slip*slip - λ_fee*fees - λ_dd*DD.

        lambdas: { 'lambda_slippage': ..., 'lambda_fees': ..., 'lambda_drawdown': ... }
        """
        ls = float((lambdas or {}).get("lambda_slippage", 0.1))
        lf = float((lambdas or {}).get("lambda_fees", 1.0))
        ld = float((lambdas or {}).get("lambda_drawdown", 0.2))
        reward = float(pnl) - ls * float(slippage_bps) / 10000.0 - lf * float(fees) - ld * float(drawdown_local)
        self.update(state, reward)

    # ====== Warm start / Transfer learning ======
    def warm_start_from_arrays(self, X: np.ndarray, y: np.ndarray, l2: float = 1e-4) -> None:
        """Inicializa pesos con solución ridge cerrada a partir de arrays acumulados.

        Resuelve w,b minimizando ||y - (Xw + b)||^2 + l2||w||^2.
        """
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            return
        try:
            # Centrado para estimar intercepto
            yv = y.astype(float)
            Xv = X.astype(float)
            y_mean = float(yv.mean())
            X_mean = Xv.mean(axis=0)
            Xc = Xv - X_mean
            yc = yv - y_mean
            # Ridge: (Xc^T Xc + l2 I)^{-1} Xc^T yc
            XtX = Xc.T @ Xc
            regI = np.eye(XtX.shape[0]) * float(l2)
            w = np.linalg.pinv(XtX + regI) @ (Xc.T @ yc)
            b = y_mean - float(X_mean @ w)
            # Ajustar dimensiones a lookback actual
            if w.shape[0] != self._w.shape[0]:
                # recortar o pad con ceros
                L = self._w.shape[0]
                if w.shape[0] > L:
                    w = w[-L:]
                else:
                    pad = np.zeros(L)
                    pad[-w.shape[0]:] = w
                    w = pad
            self._w = w.astype(float)
            self._b = float(b)
        except Exception:
            # fallback: ignorar si fallo numérico
            return

    def warm_start_from_dfs(self, dfs: list[pd.DataFrame], l2: float = 1e-4) -> None:
        """Construye X,y de varios DataFrames y hace warm start."""
        try:
            Xs: list[np.ndarray] = []
            ys: list[np.ndarray] = []
            for df in (dfs or []):
                X, y = self._features(df)
                if len(X) and len(y):
                    Xs.append(X)
                    ys.append(y)
            if not Xs:
                return
            Xall = np.vstack(Xs)
            yall = np.concatenate(ys)
            self.warm_start_from_arrays(Xall, yall, l2=l2)
        except Exception:
            return

    # ====== Export/Import para federated learning ======
    def to_dict(self) -> dict:
        return {"w": self._w.tolist(), "b": float(self._b), "lookback": int(self.cfg.lookback)}

    def load_from(self, w: list[float], b: float) -> None:
        try:
            arr = np.array(list(w), dtype=float)
            L = self._w.shape[0]
            if arr.shape[0] != L:
                if arr.shape[0] > L:
                    arr = arr[-L:]
                else:
                    pad = np.zeros(L)
                    pad[-arr.shape[0]:] = arr
                    arr = pad
            self._w = arr
            self._b = float(b)
        except Exception:
            pass

    # ====== Drift detection (PSI simple) ======
    def psi(self, baseline_states: np.ndarray, live_states: np.ndarray, num_bins: int = 10) -> float:
        """Population Stability Index entre baseline y live."""
        if baseline_states.size == 0 or live_states.size == 0:
            return 0.0
        # proyectar a score 1D para comparar distribuciones
        base_scores = baseline_states @ self._w + self._b
        live_scores = live_states @ self._w + self._b
        min_v = float(min(base_scores.min(), live_scores.min()))
        max_v = float(max(base_scores.max(), live_scores.max()))
        if max_v - min_v <= 1e-12:
            return 0.0
        bins = np.linspace(min_v, max_v, num_bins + 1)
        base_hist, _ = np.histogram(base_scores, bins=bins)
        live_hist, _ = np.histogram(live_scores, bins=bins)
        base_p = np.maximum(base_hist / max(base_hist.sum(), 1), 1e-6)
        live_p = np.maximum(live_hist / max(live_hist.sum(), 1), 1e-6)
        psi_val = float(np.sum((live_p - base_p) * np.log(live_p / base_p)))
        return psi_val


class ConstrainedOnlinePolicy:
    """Wrapper con restricciones tipo Lagrangiano sobre `OnlineLinearPolicy`.

    Soporta penalizar violaciones de constraints y bloqueo duro si exceden límites.
    """

    def __init__(self, base: OnlineLinearPolicy, lr_lambda: float = 0.01) -> None:
        self.base = base
        self.lr_lambda = float(lr_lambda)
        # Multiplicadores para constraints
        self.lmb: dict[str, float] = {
            "drawdown": 0.0,
            "risk": 0.0,
            "orders": 0.0,
        }
        # Límites por defecto
        self.limits: dict[str, float] = {
            "max_drawdown": 0.05,
            "max_orders_per_day": 20.0,
            "max_risk_per_trade": 0.005,
        }
        # Estado contadores
        self._orders_today: float = 0.0

    def set_limits(self, max_drawdown: float | None = None, max_orders_per_day: float | None = None, max_risk_per_trade: float | None = None) -> None:
        if max_drawdown is not None:
            self.limits["max_drawdown"] = float(max_drawdown)
        if max_orders_per_day is not None:
            self.limits["max_orders_per_day"] = float(max_orders_per_day)
        if max_risk_per_trade is not None:
            self.limits["max_risk_per_trade"] = float(max_risk_per_trade)

    def should_block(self, est_risk_per_trade: float, current_drawdown: float) -> bool:
        if float(est_risk_per_trade) > float(self.limits["max_risk_per_trade"]):
            return True
        if float(current_drawdown) > float(self.limits["max_drawdown"]):
            return True
        if float(self._orders_today) >= float(self.limits["max_orders_per_day"]):
            return True
        return False

    def update_lambdas(self, est_risk_per_trade: float, current_drawdown: float) -> None:
        # g_i(x) <= 0 formato
        g_dd = float(current_drawdown) - float(self.limits["max_drawdown"])
        g_risk = float(est_risk_per_trade) - float(self.limits["max_risk_per_trade"])
        g_orders = float(self._orders_today) - float(self.limits["max_orders_per_day"])
        self.lmb["drawdown"] = max(0.0, self.lmb["drawdown"] + self.lr_lambda * g_dd)
        self.lmb["risk"] = max(0.0, self.lmb["risk"] + self.lr_lambda * g_risk)
        self.lmb["orders"] = max(0.0, self.lmb["orders"] + self.lr_lambda * g_orders)

    def update_with_constraints(self, state: np.ndarray, pnl: float, slippage_bps: float, fees: float, drawdown_local: float, est_risk_per_trade: float) -> None:
        # Recompensa penalizada por Lagrangiano
        ls = 0.1
        lf = 1.0
        ld = 0.2
        reward = float(pnl) - ls * float(slippage_bps) / 10000.0 - lf * float(fees) - ld * float(drawdown_local)
        penalty = (
            self.lmb["drawdown"] * max(0.0, float(drawdown_local) - float(self.limits["max_drawdown"]))
            + self.lmb["risk"] * max(0.0, float(est_risk_per_trade) - float(self.limits["max_risk_per_trade"]))
            + self.lmb["orders"] * max(0.0, float(self._orders_today) - float(self.limits["max_orders_per_day"]))
        )
        self.base.update(state, float(reward) - float(penalty))
        # Actualizar lambdas
        self.update_lambdas(est_risk_per_trade, drawdown_local)
        # Incrementar contador de órdenes si recompensa sugiere acción (proxy)
        if self.base.predict_action(state) != 0.0:
            self._orders_today += 1.0


class AdaptiveExplorationController:
    """Controla la tasa de exploración (epsilon) en función de drawdown y drift.

    - Epsilon disminuye cuando el drawdown o el drift aumentan (exploración segura).
    """

    def __init__(self, base_epsilon: float = 0.10, min_epsilon: float = 0.01, max_epsilon: float = 0.30, k_dd: float = 3.0, k_drift: float = 2.0) -> None:
        self.base = float(base_epsilon)
        self.min_eps = float(min_epsilon)
        self.max_eps = float(max_epsilon)
        self.k_dd = float(k_dd)
        self.k_drift = float(k_drift)

    def compute_epsilon(self, drawdown_abs: float, drift_metric: float) -> float:
        # drawdown_abs: valor positivo [0,1]; drift_metric: PSI/KL/MMD ya normalizado
        import math
        eps = self.base * math.exp(-self.k_dd * max(0.0, float(drawdown_abs))) * math.exp(-self.k_drift * max(0.0, float(drift_metric)))
        return max(self.min_eps, min(self.max_eps, float(eps)))


def predict_action_with_epsilon(policy: OnlineLinearPolicy, state: np.ndarray, epsilon: float) -> float:
    """Elige acción con prob. epsilon de explorar aleatoriamente {-1,0,1}."""
    try:
        if float(epsilon) > 0.0 and np.random.rand() < float(epsilon):
            return float(np.random.choice([-1.0, 0.0, 1.0]))
    except Exception:
        pass
    return float(policy.predict_action(state))

