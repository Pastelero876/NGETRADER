from __future__ import annotations


class PageHinkley:
    """Detector sencillo de drift basado en Page-Hinkley.

    Parámetros:
    - delta: tolerancia para desviaciones pequeñas
    - lambda_: umbral de alarma
    - alpha: factor de olvido para la media acumulada
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0, alpha: float = 0.999) -> None:
        self.delta = float(delta)
        self.lambda_ = float(lambda_)
        self.alpha = float(alpha)
        self._mean = 0.0
        self._m_t = 0.0
        self._min_m_t = 0.0

    @property
    def score(self) -> float:
        return float(self._m_t - self._min_m_t)

    def update(self, x: float) -> bool:
        # media exponencialmente ponderada
        self._mean = self.alpha * self._mean + (1.0 - self.alpha) * float(x)
        self._m_t = self._m_t + (float(x) - self._mean - self.delta)
        self._min_m_t = min(self._min_m_t, self._m_t)
        return (self._m_t - self._min_m_t) > self.lambda_



def kl_divergence(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    import math
    s = 0.0
    for pi, qi in zip(p, q):
        s += (pi + eps) * math.log((pi + eps) / (qi + eps))
    return float(s)


def maximum_mean_discrepancy(x: list[float], y: list[float], gamma: float = 1.0) -> float:
    # RBF-kernel MMD^2
    import math
    def k(a: float, b: float) -> float:
        return math.exp(-gamma * (a - b) * (a - b))
    xx = sum(k(a, b) for a in x for b in x) / (len(x) ** 2 if x else 1)
    yy = sum(k(a, b) for a in y for b in y) / (len(y) ** 2 if y else 1)
    xy = sum(k(a, b) for a in x for b in y) / ((len(x) * len(y)) if (x and y) else 1)
    return float(xx + yy - 2 * xy)

