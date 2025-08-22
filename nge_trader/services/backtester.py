from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import random
import pandas as pd


@dataclass
class BacktestSpec:
    dataset_root: str
    version: str
    symbols: List[str]
    tf: str = "1h"
    slippage_bps: float = 10.0
    fee_bps: float = 5.0
    max_orders_per_day: int = 8
    stress_latency_ms: int = 50
    stress_partial_fill_prob: float = 0.2


def _load_features(dataset_root: str, version: str, tf: str, symbol: str) -> pd.DataFrame:
    from pathlib import Path
    base = Path(dataset_root) / version / "features" / tf / f"symbol={symbol.upper()}"
    parts = list(base.rglob("features.parquet"))
    if not parts:
        return pd.DataFrame()
    dfs = [pd.read_parquet(str(p)) for p in sorted(parts)]
    return pd.concat(dfs, ignore_index=True)


def _apply_costs(px: float, side: str, slip_bps: float, fee_bps: float) -> float:
    m = 1.0 + (slip_bps + fee_bps) / 10000.0
    return px * (m if side == "buy" else (2.0 - m))


def run_backtest(spec: BacktestSpec) -> Dict[str, Any]:
    eq_curve: Dict[str, List[float]] = {}
    metrics: Dict[str, Any] = {"symbols": {}}
    for sym in spec.symbols:
        df = _load_features(spec.dataset_root, spec.version, spec.tf, sym)
        if df.empty:
            continue
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        # Simple policy: trade when reward>0; respect budgets
        eq = 1.0
        daily_orders = 0
        last_day = None
        curve = []
        for idx, row in df.iterrows():
            ts = pd.to_datetime(row.get("timestamp") or pd.Timestamp.utcnow())
            day = ts.date()
            if last_day != day:
                daily_orders = 0
                last_day = day
            if daily_orders >= spec.max_orders_per_day:
                curve.append(eq)
                continue
            r = float(row.get("reward") or 0.0)
            side = "buy" if r > 0 else "hold"
            if side == "hold":
                curve.append(eq)
                continue
            # stress: latency noop, partial fill reduces PnL proportionally
            part_mult = 1.0 - (spec.stress_partial_fill_prob * 0.5 if random.random() < spec.stress_partial_fill_prob else 0.0)
            # apply costs
            px = float(row.get("close") or 0.0)
            px_eff = _apply_costs(px, side, spec.slippage_bps, spec.fee_bps)
            pnl = (px_eff / max(px, 1e-9) - 1.0) * part_mult
            eq *= (1.0 + pnl)
            curve.append(eq)
            daily_orders += 1
        eq_curve[sym] = curve
        # summary metrics per symbol
        ser = pd.Series(curve)
        rets = ser.pct_change().dropna()
        sharpe = float(rets.mean() / (rets.std(ddof=0) or 1e-9)) * (len(rets) ** 0.5)
        metrics["symbols"][sym] = {"final": float(ser.iloc[-1]), "sharpe": sharpe}
    # Aggregate
    metrics["final_equity_mean"] = float(pd.Series([v[-1] for v in eq_curve.values() if v]).mean()) if eq_curve else 1.0
    return {"equity_curves": eq_curve, "metrics": metrics}


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[dict]
    stats: Dict[str, float]


class SimpleBuyHoldBacktester:
    """Backtest placeholder: buy & hold desde la primera vela disponible."""

    def run(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10_000.0,
        date_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
    ) -> BacktestResult:
        if data.empty:
            return BacktestResult(pd.Series(dtype=float), [], {})
        df = data.copy()
        if date_range and "date" in df.columns:
            start, end = date_range
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            if df.empty:
                return BacktestResult(pd.Series(dtype=float), [], {})
        prices = df["close"].astype(float).reset_index(drop=True)
        qty = initial_capital / prices.iloc[0]
        equity = qty * prices
        trades = [
            {
                "symbol": "SYMBOL",
                "side": "buy",
                "qty": float(qty),
                "price": float(prices.iloc[0]),
                "time": str(df.iloc[0].get("date", "")),
            }
        ]
        returns = equity.pct_change().fillna(0.0)
        stats = {
            "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
            "volatility": float(returns.std(ddof=0) * (252 ** 0.5)),
        }
        return BacktestResult(equity, trades, stats)


class SignalBacktester:
    """Backtester basado en señales de una estrategia discreta {-1,0,1}."""

    def run(
        self,
        data: pd.DataFrame,
        signal: pd.Series,
        initial_capital: float = 10_000.0,
        date_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
    ) -> BacktestResult:
        df = data.copy()
        df["signal"] = signal
        if date_range and "date" in df.columns:
            start, end = date_range
            df = df[(df["date"] >= start) & (df["date"] <= end)]
        if df.empty:
            return BacktestResult(pd.Series(dtype=float), [], {})
        close = df["close"].astype(float).reset_index(drop=True)
        sig = df["signal"].fillna(0).astype(float).reset_index(drop=True)
        # PnL diario como señal desplazada * retorno
        ret = close.pct_change().fillna(0.0)
        pos = sig.shift(1).fillna(0.0)
        strategy_ret = ret * pos
        equity = (1 + strategy_ret).cumprod() * initial_capital
        trades: List[dict] = []
        stats = {
            "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
            "volatility": float(strategy_ret.std(ddof=0) * (252 ** 0.5)),
        }
        return BacktestResult(equity, trades, stats)


class ExecutedSignalBacktester:
    """Backtester con simulación de órdenes de mercado, comisiones y slippage.

    - Ejecuta entradas/salidas cuando cambia la señal (0->1, 0->-1, 1->0, -1->0, 1->-1, -1->1)
    - Precio de ejecución = close * (1 +/- slippage)
    - Comisión aplicada sobre el valor nominal de la transacción
    - Mantiene posición unitaria (usa capital para dimensionar 1x equity inicial)
    """

    def run(
        self,
        data: pd.DataFrame,
        signal: pd.Series,
        initial_capital: float = 10_000.0,
        commission_pct: float = 0.0005,
        slippage_bps: float = 1.0,
        exec_algo: Optional[str] = None,
        date_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
        spread_bps: float = 0.0,
        borrow_cost_annual: float = 0.0,
        impact_bps_per_turnover: float = 0.0,
        partial_fill_slices: int = 1,
        dynamic_spread_atr_mult: float = 0.0,
        funding_rate_daily: float = 0.0,
    ) -> BacktestResult:
        df = data.copy()
        df["signal"] = signal
        if date_range and "date" in df.columns:
            s, e = date_range
            df = df[(df["date"] >= s) & (df["date"] <= e)]
        if df.empty:
            return BacktestResult(pd.Series(dtype=float), [], {})

        close = df["close"].astype(float).reset_index(drop=True)
        # ATR para spread dinámico (si hay high/low)
        atr_series = None
        try:
            if dynamic_spread_atr_mult > 0 and {"high", "low", "close"}.issubset(set(df.columns)):
                from nge_trader.domain.risk.atr import compute_atr
                atr_series = compute_atr(df).reset_index(drop=True)
        except Exception:
            atr_series = None
        sig = df["signal"].fillna(0.0).astype(float).clip(-1, 1).reset_index(drop=True)

        slippage = slippage_bps / 10_000.0
        equity = initial_capital
        qty = 0.0
        position = 0.0
        equity_curve = []
        trades: List[dict] = []
        # carga perezosa para evitar dependencia circular
        try:
            from .execution import ExecutionAlgos  # type: ignore
        except Exception:
            ExecutionAlgos = None  # type: ignore

        daily_borrow = borrow_cost_annual / 252.0 if borrow_cost_annual > 0 else 0.0
        for i in range(len(close)):
            price = float(close.iloc[i])
            desired = float(sig.iloc[i])
            # Si hay cambio de posición, cerramos la actual y abrimos la nueva
            if desired != position:
                # cerrar si hay
                if position != 0.0 and qty != 0.0:
                    # simulación de fills parciales
                    slices = max(int(partial_fill_slices), 1)
                    for sidx in range(slices):
                        if ExecutionAlgos and exec_algo:
                            exit_price = ExecutionAlgos.apply(price, position, df.iloc[i], exec_algo, slippage)
                        else:
                            exit_price = price * (1 - slippage if position > 0 else 1 + slippage)
                        # spread/impact dinámicos
                        dyn_spread = 0.0
                        if atr_series is not None and pd.notna(atr_series.iloc[i]):
                            dyn_spread = dynamic_spread_atr_mult * float(atr_series.iloc[i]) / max(price, 1e-9) * 10_000.0
                        eff_spread_bps = spread_bps + dyn_spread
                        exit_price *= (1 - eff_spread_bps / 10_000.0)
                        exit_price -= (impact_bps_per_turnover / 10_000.0) * exit_price
                        part_qty = qty / slices
                        proceeds = part_qty * exit_price
                        commission = abs(proceeds) * commission_pct
                        pnl = proceeds - commission
                        equity += pnl
                    trades.append(
                        {
                            "symbol": "SYMBOL",
                            "side": "sell" if position > 0 else "buy",
                            "qty": float(qty),
                            "price": float(exit_price),
                            "out_time": str(df.iloc[i].get("date", "")),
                        }
                    )
                    qty = 0.0
                    position = 0.0
                # abrir nueva si desired != 0
                if desired != 0.0:
                    # tamaño: todo el equity a una posición unitaria
                    qty = (equity / max(price, 1e-9)) * abs(desired)
                    # ejecutar en rebanadas si se desea
                    slices = max(int(partial_fill_slices), 1)
                    cum_cost = 0.0
                    total_comm = 0.0
                    for sidx in range(slices):
                        if ExecutionAlgos and exec_algo:
                            entry_price = ExecutionAlgos.apply(price, desired, df.iloc[i], exec_algo, slippage)
                        else:
                            entry_price = price * (1 + slippage if desired > 0 else 1 - slippage)
                        dyn_spread = 0.0
                        if atr_series is not None and pd.notna(atr_series.iloc[i]):
                            dyn_spread = dynamic_spread_atr_mult * float(atr_series.iloc[i]) / max(price, 1e-9) * 10_000.0
                        eff_spread_bps = spread_bps + dyn_spread
                        entry_price *= (1 + eff_spread_bps / 10_000.0)
                        entry_price += (impact_bps_per_turnover / 10_000.0) * entry_price
                        part_qty = qty / slices
                        cost = part_qty * entry_price
                        commission = abs(cost) * commission_pct
                        cum_cost += cost
                        total_comm += commission
                    equity -= total_comm
                    position = desired
                    trades.append(
                        {
                            "symbol": "SYMBOL",
                            "side": "buy" if desired > 0 else "sell",
                            "qty": float(qty),
                            "price": float(cum_cost / max(qty, 1e-9)),
                            "in_time": str(df.iloc[i].get("date", "")),
                        }
                    )

            # marca a mercado
            mtm = equity
            if position != 0.0 and qty != 0.0:
                mtm = equity + qty * price * (1 if position > 0 else -1)
                # coste de borrow si short
                if position < 0 and daily_borrow > 0:
                    mtm *= (1.0 - daily_borrow)
                # funding (cripto)
                if funding_rate_daily != 0.0:
                    mtm *= (1.0 - funding_rate_daily)
            equity_curve.append(mtm)

        eq_series = pd.Series(equity_curve)
        returns = eq_series.pct_change().fillna(0.0)
        stats = {
            "total_return": float(eq_series.iloc[-1] / eq_series.iloc[0] - 1.0) if not eq_series.empty else 0.0,
            "volatility": float(returns.std(ddof=0) * (252 ** 0.5)) if not returns.empty else 0.0,
        }
        return BacktestResult(eq_series, trades, stats)


