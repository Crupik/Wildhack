import numpy as np
from app.types import CutoffCache


def predict_lag_formula_for_cut(
    values: np.ndarray,
    slots: np.ndarray,
    cut: int,
    cache: CutoffCache,
    h: int,
    a: float = 0.6876323378629036,
    b: float = 0.15585209397332267,
    c: float = 0.0632273470621092,
    d: float = 0.12919920398783646,
) -> np.ndarray:
    # Recursive prediction to horizon h for a single cutoff.
    last = cache.lag1.copy()
    for step in range(h + 1):
        ti = cut + step
        s = int(slots[ti])
        p = a * last + b * values[ti - 48] + c * values[ti - 336] + d * cache.slot_mean[s]
        p = np.clip(p, 0, None)
        last = p
    return last