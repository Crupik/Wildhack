import numpy as np


def build_route_groups(routes: np.ndarray, n_groups: int = 4) -> np.ndarray:
    order = np.argsort(routes)
    groups_sorted = np.arange(len(routes)) % n_groups
    groups = np.empty(len(routes), dtype=int)
    groups[order] = groups_sorted
    return groups