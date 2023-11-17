import pandas as pd


class LineupMatchupPerformance():

    def __init__(self, lineup_matchup: pd.DataFrame):
        self.lineup_matchup = lineup_matchup

    def generate(self, df: pd.DataFrame):
        q = self.lineup_matchup.explode('LINEUP_OFFENSE')
        u = 2


if __name__ == '__main__':
    game = pd.read_pickle("data/game.pickle")
    game = game.rename(columns={'MINUTES': 'GAME_MINUTES'})
    game_team = pd.read_pickle("data/game_team.pickle")
    game_player = pd.read_pickle("data/game_player.pickle")

    df = game_player.merge(game_team, on=['GAME_ID', 'TEAM_ID'], how='left')
    df = df.merge(game, on=['GAME_ID'], how='left')

    df.loc[df['WON']==True, 'WON'] = 1
    df.loc[df['WON'] == False, 'WON'] = 0

    lineup_matchup = pd.read_pickle("data/lineup_matchup.pickle")
    l = LineupMatchupPerformance(lineup_matchup)
    l.generate(df)
