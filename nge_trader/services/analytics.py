from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import zipfile
from matplotlib.figure import Figure


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_equity_csv(target_dir: Path, equity_curve: Iterable[float]) -> Path:
    ensure_dir(target_dir)
    s = pd.Series(list(equity_curve))
    out = target_dir / "equity.csv"
    s.to_csv(out, index=True, header=["equity"])  # index as row number
    return out


def export_trades_csv(target_dir: Path, trades: List[dict]) -> Path:
    ensure_dir(target_dir)
    df = pd.DataFrame(trades)
    out = target_dir / "trades.csv"
    df.to_csv(out, index=False)
    return out


def export_report_html(target_dir: Path, metrics: Dict[str, float]) -> Path:
    ensure_dir(target_dir)
    out = target_dir / "report.html"
    lines = [
        "<html><head><meta charset='utf-8'><title>Backtest Report</title></head><body>",
        "<h1>Informe de Backtest</h1>",
        "<h2>Métricas</h2>",
        "<ul>",
    ]
    for k, v in metrics.items():
        if k in {"sharpe", "sortino", "win_rate"}:
            lines.append(f"<li>{k}: {v:.2f}</li>")
        elif k in {"max_drawdown", "total_return"}:
            lines.append(f"<li>{k}: {v:.2%}</li>")
        else:
            lines.append(f"<li>{k}: {v}</li>")
    lines += [
        "</ul>",
        "<p>Archivos: equity.csv y trades.csv generados en este directorio.</p>",
        "<h2>Gráficos</h2>",
        "<p><img src='equity.png' alt='Equity' style='max-width:100%'></p>",
        "<p><img src='drawdown.png' alt='Drawdown' style='max-width:100%'></p>",
        "<p><img src='rolling_sharpe.png' alt='Rolling Sharpe' style='max-width:100%'></p>",
        "<h2>Métricas Live</h2>",
        "<p>Consulte metrics.csv para series de hit_rate, sharpe_live, slippage_bps y market_ws_skew_ms.</p>",
        "</body></html>",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def export_metrics_csv(target_dir: Path, rows: List[Dict[str, float]]) -> Path:
    ensure_dir(target_dir)
    df = pd.DataFrame(rows)
    out = target_dir / "metrics.csv"
    df.to_csv(out, index=False)
    return out


def make_report_dir() -> Path:
    ts = datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    return Path("data/reports") / ts


def export_report_zip(target_dir: Path) -> Path:
    """Crea un ZIP con los artefactos del reporte en target_dir."""
    ensure_dir(target_dir)
    zip_path = target_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in target_dir.glob("*"):
            zf.write(p, arcname=p.name)
    return zip_path


def export_tearsheet(target_dir: Path, equity: pd.Series) -> None:
    ensure_dir(target_dir)
    if equity.empty:
        return
    # Equity
    fig1 = Figure(figsize=(6, 3), dpi=120)
    ax1 = fig1.add_subplot(111)
    (equity / float(equity.iloc[0])).plot(ax=ax1, title="Equity (normalizado)")
    fig1.savefig(target_dir / "equity.png")
    # Drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0.0, 1e-9)
    fig2 = Figure(figsize=(6, 2.5), dpi=120)
    ax2 = fig2.add_subplot(111)
    dd.plot(ax=ax2, color="tomato", title="Drawdown")
    fig2.savefig(target_dir / "drawdown.png")
    # Rolling Sharpe (252d)
    ret = equity.pct_change().fillna(0.0)
    roll = ret.rolling(252).mean() / ret.rolling(252).std(ddof=0).replace(0.0, 1e-9)
    fig3 = Figure(figsize=(6, 2.5), dpi=120)
    ax3 = fig3.add_subplot(111)
    roll.plot(ax=ax3, color="#22d3ee", title="Sharpe rodante (252d)")
    fig3.savefig(target_dir / "rolling_sharpe.png")
    # Exportar métricas live si existen en DB
    try:
        from nge_trader.repository.db import Database
        db = Database()
        coll: list[dict] = []
        for key in ("hit_rate", "sharpe_live"):
            for ts, val in db.recent_metric_series(key, 200):
                coll.append({"ts": ts, "key": key, "value": float(val)})
        for key in ("slippage_bps", "market_ws_skew_ms"):
            for ts, val in db.recent_metric_values(key, 200):
                coll.append({"ts": ts, "key": key, "value": float(val)})
        if coll:
            export_metrics_csv(target_dir, coll)
    except Exception:
        pass

