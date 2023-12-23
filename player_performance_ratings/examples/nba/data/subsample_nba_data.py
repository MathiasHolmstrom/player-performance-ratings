import pandas as pd

df = pd.read_pickle("game_player.pickle")
gt = pd.read_pickle("game_team.pickle")

df = df.merge(gt[['game_id', 'team_id', 'team_id_opponent']], on=['game_id', 'team_id'], how='left')

#randomly select 10% of game_ids
game_ids = df['game_id'].unique().tolist()
import random
random.shuffle(game_ids)
game_ids = game_ids[:int(len(game_ids)*0.1)]
df = df[df['game_id'].isin(game_ids)]
df.to_pickle("game_player_subsample.pickle")
