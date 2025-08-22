from __future__ import annotations

from typing import Protocol, Any


class PriceProvider(Protocol):
    def get_price(self, symbol: str) -> float | None:  # pragma: no cover - interface
        ...


class RedundantDataProvider:
    """Agrega múltiples proveedores y verifica divergencias.

    - Usa el primer proveedor disponible como principal.
    - Si hay divergencia > threshold_pct entre proveedores disponibles, levanta alerta (ValueError) o marca bandera.
    """

    def __init__(self, providers: list[PriceProvider], divergence_threshold_pct: float = 0.005) -> None:
        self.providers = list(providers)
        self.divergence_threshold_pct = float(divergence_threshold_pct)

    def get_price(self, symbol: str) -> float | None:
        prices: list[float] = []
        for p in self.providers:
            try:
                v = p.get_price(symbol)
                if v is not None and float(v) > 0:
                    prices.append(float(v))
            except Exception:
                continue
        if not prices:
            return None
        ref = prices[0]
        for v in prices[1:]:
            if abs(v - ref) / ref > self.divergence_threshold_pct:
                # Señalar divergencia
                raise ValueError(f"Divergencia entre proveedores: ref={ref} vs {v}")
        return ref


