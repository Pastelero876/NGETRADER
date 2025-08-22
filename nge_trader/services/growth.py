from __future__ import annotations

from typing import Any, Dict

from nge_trader.config.settings import Settings


def compute_growth_plan(equity: float | None = None) -> Dict[str, Any]:
    s = Settings()
    step_eur = float(getattr(s, "growth_equity_step", 250.0) or 250.0)
    min_eq = float(getattr(s, "growth_min_equity", 0.0) or 0.0)
    max_eq = float(getattr(s, "growth_max_equity", 0.0) or 0.0)
    base_r = float(getattr(s, "growth_risk_pct_base", s.risk_pct_per_trade) or s.risk_pct_per_trade)
    inc_r = float(getattr(s, "growth_risk_pct_increment", 0.001) or 0.001)
    base_k = int(getattr(s, "growth_top_k_base", 1) or 1)
    inc_k = int(getattr(s, "growth_top_k_increment", 1) or 1)
    max_k = int(getattr(s, "growth_top_k_max", 3) or 3)

    if equity is None:
        try:
            from nge_trader.services.app_service import AppService
            acc = AppService().broker.get_account() if hasattr(AppService().broker, "get_account") else None
            equity = float(acc.get("equity") or acc.get("last_equity") or 0.0) if acc else 0.0
        except Exception:
            equity = 0.0

    eq = float(equity or 0.0)
    if max_eq and eq > max_eq:
        eq = max_eq
    if eq < min_eq:
        step_idx = 0
    else:
        step_idx = int((eq - min_eq) // max(step_eur, 1.0))

    target_r = max(0.0, base_r + inc_r * step_idx)
    target_k = min(max_k, max(1, base_k + inc_k * step_idx))
    next_equity = min_eq + (step_idx + 1) * step_eur

    return {
        "equity": equity,
        "step_index": step_idx,
        "next_step_equity": next_equity,
        "target_risk_pct_per_trade": target_r,
        "target_top_k_per_cycle": int(target_k),
        "params": {
            "growth_equity_step": step_eur,
            "growth_min_equity": min_eq,
            "growth_max_equity": max_eq,
            "growth_risk_pct_base": base_r,
            "growth_risk_pct_increment": inc_r,
            "growth_top_k_base": base_k,
            "growth_top_k_increment": inc_k,
            "growth_top_k_max": max_k,
        },
    }


def apply_growth_to_env(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Guarda los parámetros objetivo (riesgo y K) en .env para próximas ejecuciones."""
    try:
        from nge_trader.config.env_utils import write_env
        r = float(plan.get("target_risk_pct_per_trade") or 0.0)
        k = int(plan.get("target_top_k_per_cycle") or 1)
        write_env({
            "RISK_PCT_PER_TRADE": str(r),
            "TOP_K_PER_CYCLE": str(k),
        })
        return {"status": "ok", "risk_pct_per_trade": r, "top_k_per_cycle": k}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


