import os
from pathlib import Path
from typing import Union

import polars as pl
import pandas as pd


def get_sub_sample_nba_data(
    as_pandas: bool = True, as_polars: bool = False
) -> Union[pd.DataFrame, pl.DataFrame]:
    script_dir = Path(__file__).parent
    file_path = os.path.join(script_dir, "game_player_subsample.parquet")
    if as_polars:
        return pl.read_parquet(file_path)
    elif as_pandas:
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Must specify either as_pandas or as_polars")
