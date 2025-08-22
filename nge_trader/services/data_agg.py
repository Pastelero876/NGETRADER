from __future__ import annotations

import pandas as pd


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    if "date" in df.columns:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])  # type: ignore[assignment]
        d = d.set_index("date").sort_index()
    else:
        d = df.copy()
        d.index = pd.to_datetime(d.index)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = d.resample(rule).agg(agg).dropna(how="all")
    out = out.reset_index().rename(columns={"index": "date"})
    return out


def deduplicate_and_order_events(df: pd.DataFrame, sequence_col: str = "sequence_id", ts_col: str = "ts") -> pd.DataFrame:
    """Ordena por `sequence_id` (o timestamp) y elimina duplicados.

    Si no existe la columna, ordena por timestamp si está disponible.
    """
    d = df.copy()
    if sequence_col in d.columns:
        # Mantener la última ocurrencia por sequence_id respetando el orden de llegada
        d = d.drop_duplicates(subset=[sequence_col], keep="last")
        d = d.sort_values(by=[sequence_col]).reset_index(drop=True)
        return d
    if ts_col in d.columns:
        d = d.sort_values(by=[ts_col])
        d = d.drop_duplicates(subset=[ts_col], keep="last")
        return d.reset_index(drop=True)
    return d.reset_index(drop=True)


def stable_agg_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Limpia, ordena por `sequence_id`/`ts`, deduplica y calcula un VWAP/volumen agregados de ventana corta.

    Espera columnas: ['price','qty'] y opcionalmente ['sequence_id','ts'].
    """
    if trades.empty:
        return trades
    d = deduplicate_and_order_events(trades)
    d = d.copy()
    d["price"] = d["price"].astype(float)
    d["qty"] = d["qty"].astype(float)
    # Ventana móvil simple de 20 eventos
    d["vwap20"] = (d["price"] * d["qty"]).cumsum() / d["qty"].cumsum()
    d["vol20"] = d["qty"].cumsum()
    return d
