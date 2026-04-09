import numpy as np
import pandas as pd

from app.data_loader import load_train_data
from app.panel import build_target_panel, build_total_series
from app.grouping import build_route_groups


def prepare_training_matrices(n_groups: int = 4):
    train_df = load_train_data()

    panel = build_target_panel(train_df)
    total_series = build_total_series(panel)

    values = panel.to_numpy(dtype=float)
    times = panel.index
    routes = panel.columns.to_numpy()
    slots = (times.hour * 2 + (times.minute // 30)).to_numpy(dtype=int)
    totals = total_series.to_numpy(dtype=float)

    route_groups = build_route_groups(routes, n_groups=n_groups)

    return {
        "train_df": train_df,
        "panel": panel,
        "total_series": total_series,
        "values": values,
        "times": times,
        "routes": routes,
        "slots": slots,
        "totals": totals,
        "route_groups": route_groups,
    }