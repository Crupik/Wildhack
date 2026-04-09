import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class CutoffCache:
    lag1: np.ndarray
    lag2: np.ndarray
    lag3: np.ndarray
    lag4: np.ndarray
    mean4: np.ndarray
    mean8: np.ndarray
    mean16: np.ndarray
    std8: np.ndarray
    std16: np.ndarray
    route_mean: np.ndarray
    route_mean_share: np.ndarray
    slot_mean: np.ndarray
    slot_mean_share: np.ndarray
    total_slot_mean: np.ndarray

@dataclass
class StackCalibration:
    mode: str
    global_scale: float = 1.0
    horizon_scale: Optional[np.ndarray] = None
    horizon_affine_a: Optional[np.ndarray] = None
    horizon_affine_b: Optional[np.ndarray] = None