import pandas as pd


def load_lol_data():
    return pd.read_parquet("data/subsample_lol_data")


def load_nba_game_player_data() -> pd.DataFrame:
    game_player = pd.read_pickle("data/game_player.pickle")
    game_team = pd.read_pickle("data/game_team.pickle")
    game = pd.read_pickle("data/game.pickle")
    game = game.rename(columns={'minutes': 'game_minutes'})
    game_player = game_player.merge(game_team[['game_id', 'team_id', 'score', 'score_opponent', 'won']],
                                    on=['game_id', 'team_id'], how='inner')
    game_player.loc[game_player['won'] == True, 'won'] = 1
    game_player.loc[game_player['won'] == False, 'won'] = 0
    return game_player.merge(game, on=['game_id'], how='inner')


def load_nba_game_matchup_data() -> pd.DataFrame:
    game = pd.read_pickle("data/game.pickle")
    lineup_matchup = pd.read_pickle("data/lineup_matchup.pickle")
    # lineup_matchup['unique_combination'] = lineup_matchup['game_id'].astype(str) + '-' + lineup_matchup[
    #     'lineup_id_offense'].astype('str') + '-' + lineup_matchup['lineup_id_defense'].astype('str')

    # Assigning a unique integer to each unique combination
    # lineup_matchup['matchup_game_id'] = pd.factorize(lineup_matchup['unique_combination'])[0]
    # lineup_matchup.drop(columns=['unique_combination'], inplace=True)

    lineup_matchup = lineup_matchup[
        ['game_id', 'team_id_offense', 'lineup_id_offense', 'lineup_id_defense', 'points', 'minutes_lineup_matchup',
         'lineup_offense', 'lineup_defense']]
    temp_df = lineup_matchup.copy()
    temp_df['temp_id'] = temp_df['lineup_id_offense']
    temp_df['temp_lineup'] = temp_df['lineup_offense']
    temp_df['lineup_id_offense'] = temp_df['lineup_id_defense']
    temp_df['lineup_offense'] = temp_df['lineup_defense']
    temp_df['lineup_id_defense'] = temp_df['temp_id']
    temp_df['lineup_defense'] = temp_df['temp_lineup']
    temp_df = temp_df.drop('temp_id', axis=1)
    temp_df = temp_df.rename(
        columns={'points': 'points_opponent', 'minutes_lineup_matchup': 'minutes_lineup_matchup_opponent'})
    temp_df = temp_df[
        ['game_id', 'lineup_id_offense', 'lineup_id_defense', 'points_opponent', 'minutes_lineup_matchup_opponent',
         'lineup_defense', 'lineup_offense']]
    # Merging the original DataFrame with the temporary one
    lineup_matchup = lineup_matchup.merge(temp_df,
                                          on=['game_id', 'lineup_id_offense', 'lineup_id_defense', 'lineup_offense',
                                              'lineup_defense'], how='outer')

    lineup_matchup['points'] = lineup_matchup['points'].fillna(0)
    lineup_matchup['points_opponent'] = lineup_matchup['points_opponent'].fillna(0)
    lineup_matchup['minutes_lineup_matchup'] = lineup_matchup['minutes_lineup_matchup']
    lineup_matchup['minutes_lineup_matchup_opponent'] = lineup_matchup['minutes_lineup_matchup_opponent'].fillna(0)

    lineup_matchup['minutes_lineup_matchup'] = lineup_matchup['minutes_lineup_matchup'] + lineup_matchup[
        'minutes_lineup_matchup_opponent']
    lineup_matchup.drop(columns=['minutes_lineup_matchup_opponent'], inplace=True)

    lineup_match_players = lineup_matchup.explode('lineup_offense')
    lineup_match_players = lineup_match_players.rename(columns={'lineup_offense': 'player_id'})
    lineup_match_players['team_minutes'] = lineup_match_players.groupby(['game_id', 'team_id_offense'])[
                                               'minutes_lineup_matchup'].transform('sum') / 5
    lineup_match_players.rename(columns={'team_id_offense': 'team_id', 'lineup_id_offense': 'lineup_id',
                                         'lineup_id_defense': 'lineup_id_opponent'}, inplace=True)

  #  game_player = pd.read_pickle("data/game_player.pickle")

    game_team = pd.read_pickle("data/game_team.pickle")

    game = game.rename(columns={'minutes': 'game_minutes'})

    lineup_match_players = lineup_match_players.merge(
        game_team[['game_id', 'team_id', 'score', 'score_opponent', 'won']], on=['game_id', 'team_id'])

    lineup_match_players.loc[lineup_match_players['won'] == True, 'won'] = 1
    lineup_match_players.loc[lineup_match_players['won'] == False, 'won'] = 0
    lineup_match_players['matchup_id'] = lineup_match_players['game_id'].astype(str) + '-' + lineup_match_players[
        'lineup_id'].astype('str') + '-' + lineup_match_players['lineup_id_opponent'].astype('str')
    lineup_match_players['participation_weight'] = lineup_match_players['minutes_lineup_matchup'] / lineup_match_players[
        'team_minutes']
    lineup_match_players.drop(columns=['team_minutes'], inplace=True)
    return lineup_match_players.merge(game, on=['game_id'], how='inner')
