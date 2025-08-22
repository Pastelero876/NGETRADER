from __future__ import annotations

import sys
from nge_trader.services.analytics import make_report_dir, export_equity_csv, export_trades_csv, export_report_html, export_tearsheet, export_report_zip
from nge_trader.repository.db import Database
import pandas as pd
from nge_trader.services.notifier import Notifier


def main() -> None:
    db = Database()
    d = make_report_dir()
    eq = db.load_equity_curve()
    export_equity_csv(d, eq.values if not eq.empty else [1, 1])
    export_trades_csv(d, db.recent_trades(200))
    export_tearsheet(d, eq if not eq.empty else pd.Series([1, 1]))
    export_report_html(d, {"sharpe": 0, "sortino": 0, "win_rate": 0})
    zip_path = export_report_zip(d)
    n = Notifier()
    n.send(f"Reporte diario generado: {zip_path}")
    try:
        n.send_document(str(zip_path), caption="Reporte diario")
        n.send_email_with_attachment("Reporte diario", "Adjunto reporte.", str(zip_path))
    except Exception:
        pass
    print(zip_path)
    sys.exit(0)


if __name__ == "__main__":
    main()


