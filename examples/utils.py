import pandas as pd


def load_data():
    return pd.read_parquet("data/subsample_lol_data")
