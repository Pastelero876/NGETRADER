from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from nge_trader.repository.db import Database


@dataclass
class ModelVersion:
    version: str
    path: str
    metrics_json: str | None


class ModelRegistry:
    """Registro simple de modelos con soporte canary/rollback.

    Persistido en SQLite vía `agent_models` y `agent_runs`.
    """

    def __init__(self) -> None:
        self.db = Database()
        self._primary: Optional[ModelVersion] = None
        self._canary: Optional[ModelVersion] = None

    def publish(self, version: str, path: str, config_json: str, metrics_json: str | None) -> None:
        self.db.save_agent_model(version, config_json, metrics_json, path)
        # Por simplicidad, nuevo publish como canary
        self._canary = ModelVersion(version=version, path=path, metrics_json=metrics_json)

    def promote_canary(self) -> None:
        if self._canary is None:
            return
        self._primary = self._canary
        self._canary = None

    def rollback(self) -> None:
        # Simple: desactiva canary, mantiene primary previo
        self._canary = None

    def auto_promote_if_kpis(self, sharpe: float, drawdown_pct: float, min_sharpe: float = 0.8, max_dd: float = 0.05) -> bool:
        """Promueve el canario a primary si cumple KPIs mínimos."""
        if self._canary is None:
            return False
        if (sharpe >= min_sharpe) and (drawdown_pct >= -abs(max_dd)):
            self._primary = self._canary
            self._canary = None
            return True
        return False

    def canary_guard(self, slo: dict) -> bool:
        """Evalúa SLOs duros para canario (Sharpe/Slippage/Fill) y señaliza rollback necesario.

        slo: { 'min_sharpe': ..., 'max_slippage_bps': ..., 'min_fill_ratio': ..., 'max_hours': ... }
        """
        try:
            min_sharpe = float(slo.get("min_sharpe", 0.5))
            max_slip = float(slo.get("max_slippage_bps", 50.0))
            min_fill = float(slo.get("min_fill_ratio", 0.6))
        except Exception:
            return False
        # Placeholder: métricas recientes deberían venir del store/DB
        from nge_trader.repository.db import Database
        db = Database()
        try:
            # slippage reciente
            slips = [v for _, v in db.recent_metric_values("slippage_bps", 200)[-50:]]
            slip_ok = (abs(float(sum(slips) / max(len(slips), 1))) <= max_slip) if slips else True
        except Exception:
            slip_ok = True
        try:
            from nge_trader.services import metrics as _MM
            fills_total = float(_MM._METRICS_STORE.get("fills_total", 0.0))
            orders_total = float(_MM._METRICS_STORE.get("orders_placed_total", 0.0))
            fill_ratio = (fills_total / orders_total) if orders_total > 0 else 1.0
            fill_ok = (fill_ratio >= min_fill)
        except Exception:
            fill_ok = True
        # Sharpe canario (placeholder: usar métrica almacenada)
        try:
            sharpe_live = [v for _, v in db.recent_metric_series("sharpe_live", 200)[-20:]]
            sh_ok = (float(sum(v for _, v in sharpe_live) / max(len(sharpe_live), 1)) >= min_sharpe) if sharpe_live else True
        except Exception:
            sh_ok = True
        return not (slip_ok and fill_ok and sh_ok)

    # ===== Federated Aggregation (sencilla) =====
    def aggregate_federated(self, client_models: list[dict]) -> dict | None:
        """Agrega modelos client (w,b) por promedio ponderado simple.

        client_models: lista de { 'w': [...], 'b': float, 'weight': float }
        """
        if not client_models:
            return None
        try:
            import numpy as _np
            # normalizar pesos
            ws = []
            bs = []
            weights = []
            for cm in client_models:
                w = _np.array(list(cm.get('w') or []), dtype=float)
                b = float(cm.get('b') or 0.0)
                wt = float(cm.get('weight') or 1.0)
                ws.append(w)
                bs.append(b)
                weights.append(wt)
            W = _np.array(weights, dtype=float)
            W = W / (W.sum() if W.sum() != 0 else 1.0)
            # alinear longitudes por padding a la longitud máxima
            L = max((len(w) for w in ws), default=0)
            if L == 0:
                return None
            ws_pad = []
            for w in ws:
                if len(w) < L:
                    pad = _np.zeros(L)
                    pad[-len(w):] = w
                    ws_pad.append(pad)
                else:
                    ws_pad.append(w[-L:])
            Wmat = _np.stack(ws_pad)
            w_avg = (W.reshape(-1,1) * Wmat).sum(axis=0)
            b_avg = float((W * _np.array(bs, dtype=float)).sum())
            return {"w": w_avg.tolist(), "b": b_avg}
        except Exception:
            return None

    def primary(self) -> Optional[ModelVersion]:
        return self._primary

    def canary(self) -> Optional[ModelVersion]:
        return self._canary


