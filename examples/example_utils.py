from pathlib import Path
from typing import Union

import pandas as pd
import polars as pl
import os

def get_sub_sample_lol_data(as_pandas: bool = True,as_polars: bool = False) -> Union[pd.DataFrame]:
    file_path = os.path.join(Path.cwd(),"examples","lol", "data", "subsample_lol_data")
    if as_polars:
        return pl.read_parquet(file_path)
    elif as_pandas:
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Must specify either as_pandas or as_polars")


def get_sub_sample_nba_data(as_pandas: bool = True, as_polars: bool = False) -> Union[pd.DataFrame]:
    file_path = os.path.join(Path.cwd(),"examples","nba", "data", "game_player_subsample.parquet")
    if as_polars:
        return pl.read_parquet(file_path)
    elif as_pandas:
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Must specify either as_pandas or as_polars")
