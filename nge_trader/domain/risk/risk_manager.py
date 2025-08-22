class RiskManager:
    """Gestión de riesgo simple basada en porcentaje de capital por operación."""

    def __init__(self, max_risk_per_trade: float = 0.01) -> None:
        self.max_risk_per_trade = max_risk_per_trade

    def position_size(self, equity: float, stop_loss_pct: float) -> float:
        """Calcula tamaño de posición en función del riesgo permitido y el stop.

        equity: capital total
        stop_loss_pct: porcentaje de stop (ej. 0.02 para 2%)
        """

        if stop_loss_pct <= 0:
            return 0.0
        return (equity * self.max_risk_per_trade) / stop_loss_pct


