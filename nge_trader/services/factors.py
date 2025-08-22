from __future__ import annotations

import pandas as pd


class FactorSignals:
    """Cálculo simple de factores: value, momentum, low-vol.

    Devuelve un DataFrame con columnas: value, momentum, low_vol en escala [-1,1].
    """

    def compute(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows: list[dict] = []
        for symbol, df in prices.items():
            if df.empty:
                continue
            p = df["close"].astype(float).reset_index(drop=True)
            vol = p.pct_change().rolling(20).std(ddof=0).iloc[-1]
            mom = (p.iloc[-1] / p.iloc[max(len(p) - 21, 0)] - 1.0) if len(p) > 21 else 0.0
            # Proxy value: retorno de 1 año inverso al precio (placeholder sin fundamentals)
            val = -float(p.iloc[-1])
            rows.append({"symbol": symbol, "value": val, "momentum": float(mom), "low_vol": float(-vol)})
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        # normalizar a [-1,1]
        for col in ["value", "momentum", "low_vol"]:
            s = out[col]
            rng = (s.max() - s.min()) or 1e-9
            out[col] = (2.0 * (s - s.min()) / rng) - 1.0
        return out.set_index("symbol")


