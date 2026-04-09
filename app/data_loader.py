import pandas as pd
from app.config import TRAIN_PATH, TEST_PATH

def load_train_data():
    train_df = pd.read_parquet(TRAIN_PATH, columns=['route_id', 'timestamp', 'target_2h'])
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    return train_df

def load_test_data():
    test_df = pd.read_parquet(TEST_PATH, columns=['id', 'route_id', 'timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    return test_df

def describe_loaded_data(train_df, test_df):
    print("Train rows:", f"{len(train_df):,}")
    print("Train routes:", train_df['route_id'].nunique())
    print("Train range:", train_df["timestamp"].min(), "->", train_df["timestamp"].max())
    print("Test rows:", f"{len(test_df):,}")
    print("Test routes:", test_df["route_id"].nunique())
    print("Test range:", test_df["timestamp"].min(), "->", test_df["timestamp"].max())