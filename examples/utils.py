import pandas as pd


def load_data():
    return pd.read_parquet("data/subsample_lol_data")

def load_nba_game_player_data() -> pd.DataFrame:
    game_player = pd.read_pickle("data/game_player.pickle")
    game_team = pd.read_pickle("data/game_team.pickle")
    game = pd.read_pickle("data/game.pickle")

    game_player = game_player.merge(game_team[['GAME_ID','TEAM_ID','SCORE', 'SCORE_OPPONENT', 'WON']], on=['GAME_ID', 'TEAM_ID'], how='left')
    game_player.loc[game_player['WON'] == True, 'WON'] = 1
    game_player.loc[game_player['WON'] == False, 'WON'] = 0
    return game_player.merge(game, on=['GAME_ID'], how='left')
