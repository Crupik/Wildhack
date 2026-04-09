import pandas as pd


class HistoryStore:
    def __init__(self, path: str):
        self.df = pd.read_parquet(
            path,
            columns=["route_id", "timestamp", "target_2h"]
        )
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

    def get_history(self) -> pd.DataFrame:
        return self.df.copy()

    def get_last_timestamp(self):
        return self.df["timestamp"].max()