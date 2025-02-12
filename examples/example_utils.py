from typing import Union

import pandas as pd
import polars as pl

def get_sub_sample_lol_data(as_pandas: bool = True,as_polars: bool = False) -> Union[pd.DataFrame]:
    if as_polars:
        return pl.read_parquet("lol/data/subsample_lol_data")
    elif as_pandas:
        return pd.read_parquet("lol/data/subsample_lol_data")
    else:
        raise ValueError("Must specify either as_pandas or as_polars")


def get_sub_sample_nba_data(as_pandas: bool = True, as_polars: bool = False) -> Union[pd.DataFrame]:
    if as_polars:
        return pl.read_parquet("nba/data/game_player_subsample.parquet")
    elif as_pandas:
        return pd.read_parquet("nba/data/game_player_subsample.parquet")
    else:
        raise ValueError("Must specify either as_pandas or as_polars")