import pandas as pd


def build_target_panel(train_df: pd.DataFrame) -> pd.DataFrame:
    panel = (
        train_df
        .pivot(index="timestamp", columns="route_id", values="target_2h")
        .sort_index()
        .astype(float)
    )
    return panel


def build_total_series(panel: pd.DataFrame) -> pd.Series:
    return panel.sum(axis=1)


def describe_panel(panel: pd.DataFrame, total_series: pd.Series) -> None:
    print("Panel shape:", panel.shape)
    print("Total mean:", round(float(total_series.mean()), 3))
    print("Total std:", round(float(total_series.std()), 3))