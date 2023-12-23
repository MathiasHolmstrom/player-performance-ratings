import pandas as pd

df = pd.read_pickle("game_player.pickle")
gt = pd.read_pickle("game_team.pickle")
game = pd.read_pickle("game.pickle")
game = game.rename(columns={'minutes': 'game_minutes'})
df = df.merge(game[['game_id', 'game_minutes', 'start_date']], on='game_id', how='left')
df = df.merge(gt[['game_id', 'team_id', 'score', 'score_opponent', 'won']],
                                on=['game_id', 'team_id'], how='inner')
df.loc[df['won'] == True, 'won'] = 1
df.loc[df['won'] == False, 'won'] = 0


#randomly select 10% of game_ids
game_ids = df['game_id'].unique().tolist()
import random
random.shuffle(game_ids)
game_ids = game_ids[:int(len(game_ids)*0.1)]
df = df[df['game_id'].isin(game_ids)]
df.to_pickle("game_player_subsample.pickle")
