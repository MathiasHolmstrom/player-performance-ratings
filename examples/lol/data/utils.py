import os
from pathlib import Path

import pandas as pd
import polars as pl


def get_sub_sample_lol_data(as_pandas: bool = True, as_polars: bool = False) -> pd.DataFrame:
    script_dir = Path(__file__).parent
    file_path = os.path.join(script_dir, "subsample_lol_data.parquet")

    if as_polars:
        return pl.read_parquet(file_path)
    elif as_pandas:
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Must specify either as_pandas or as_polars")
