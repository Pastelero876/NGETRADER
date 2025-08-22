from __future__ import annotations

import os
import json
import zipfile
import glob


REPORT_DIR = "reports"


def folder_for(day: str) -> str:
    d = os.path.join(REPORT_DIR, day)
    os.makedirs(d, exist_ok=True)
    return d


def zip_path(day: str) -> str:
    return os.path.join(folder_for(day), f"report_{day}.zip")


def ensure_daily_report(day: str) -> str:
    zpath = zip_path(day)
    if os.path.exists(zpath):
        return zpath
    folder = folder_for(day)
    # TODO: generar CSVs reales; empaquetar si existen
    csvs: list[str] = []
    for pat in ["orders*.csv", "fills*.csv", "tca_*.csv", "pnl_*.csv"]:
        csvs.extend(glob.glob(os.path.join(folder, pat)))
    if not csvs:
        p = os.path.join(folder, "readme.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"Reporte {day}: añade generadores CSV")
        csvs = [p]
    # metadata.json
    meta_path = os.path.join(folder, "metadata.json")
    if not os.path.exists(meta_path):
        try:
            from nge_trader.services import model_session
            from nge_trader.config.settings import Settings
            meta = {
                "model_id": model_session.get().get("model_id"),
                "canary_share": float(getattr(Settings(), "challenger_share", 0.0)),
                "pinned": bool(model_session.get().get("pinned")),
                "generated_at": str(day),
            }
        except Exception:
            meta = {"generated_at": str(day)}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    # Zip
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in csvs:
            z.write(f, arcname=os.path.basename(f))
        # incluir metadata
        z.write(meta_path, arcname=os.path.basename(meta_path))
    return zpath


