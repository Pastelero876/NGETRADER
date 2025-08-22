from __future__ import annotations

import re
from typing import List, Set


CASHTAG_RE = re.compile(r"\$[A-Z]{1,5}\b")
TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")


def extract_tickers(text: str, known_universe: List[str] | None = None) -> List[str]:
    """Extrae posibles tickers de un texto (cashtags $AAPL y tokens en mayúsculas).

    Si se pasa un universo conocido, filtra por ese universo.
    """
    found: Set[str] = set()
    for m in CASHTAG_RE.findall(text or ""):
        found.add(m[1:])
    for m in TICKER_RE.findall(text or ""):
        found.add(m)
    if known_universe:
        uni = set(s.upper() for s in known_universe)
        found = {s for s in found if s in uni}
    # filtra palabras cortas obvias
    found = {s for s in found if 1 <= len(s) <= 5}
    return sorted(found)


