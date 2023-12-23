import os.path

import pandas as pd


def load_lol_subsampled_data():
    return pd.read_parquet("lol/data/subsample_lol_data")


def load_nba_subsampled_game_player_data() -> pd.DataFrame:
    return pd.read_pickle(os.path.join(os.path.dirname(__file__),"nba" ,"data","game_player_subsample.pickle"))
