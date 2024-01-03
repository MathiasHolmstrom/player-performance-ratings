import pandas as pd

df = pd.read_pickle("game_player.pickle")
gt = pd.read_pickle("game_team.pickle")
game = pd.read_pickle("game.pickle")
game = game.rename(columns={'minutes': 'game_minutes'})
df = df.merge(game[['game_id', 'game_minutes', 'start_date']], on='game_id', how='left')
df = df.merge(gt[['game_id', 'team_id', "team_id_opponent", 'score', 'score_opponent', 'won', "location"]],
              on=['game_id', 'team_id'], how='inner')
df.loc[df['won'] == True, 'won'] = 1
df.loc[df['won'] == False, 'won'] = 0

# randomly select 10% of game_ids

df = df[["team_id", "start_date", "game_id", "player_id", 'player_name', 'start_position', "team_id_opponent", "points",
         "game_minutes",
         "minutes", "won", "plus_minus", "location", "score", "score_opponent"]]
#df = df[df['start_date'].between('2022-10-17', '2023-02-01')]

df.to_pickle("game_player_full.pickle")
