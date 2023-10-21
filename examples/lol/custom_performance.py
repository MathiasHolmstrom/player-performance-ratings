import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src import MatchPredictor
from src import ColumnNames
from src import BaseTransformer


class LolPlayerPerformanceGenerator(BaseTransformer):

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df['dpm'] = df['damagetochampions'] / df['duration']
        df['kills_assists'] = df['kills'] + df['assists']

        df = df.assign(dpm_z_score=(df['dpm'] - df['dpm'].mean()) - df['dpm'].std())

        champion_position_grouped = df.groupby(['champion', 'position'])

        df = df.assign(
            champion_dpm=champion_position_grouped['dpm'].transform('mean'),
            champion_deaths=champion_position_grouped['deaths'].transform('mean'),
            champion_kills_assists=champion_position_grouped['kills_assists'].transform('mean'),
            champion_kills=champion_position_grouped['kills'].transform('mean'),
        )
        team_grouped = df.groupby(['gameid', 'teamname'])
        df = df.assign(
            team_sum_champion_dpm=team_grouped['champion_dpm'].transform('sum'),
            team_sum_dpm=team_grouped['dpm'].transform('sum'),
            team_sum_champion_kills=team_grouped['champion_kills'].transform('sum'),
            team_sum_kills=team_grouped['kills'].transform('sum'),
            team_sum_champion_deaths=team_grouped['champion_deaths'].transform('sum'),
            team_sum_deaths=team_grouped['deaths'].transform('sum'),
        )
        df['total_kills'] = df.groupby('gameid')['kills'].transform('sum')

        df['teamdeaths'] = df['total_kills'] - df['teamkills']
        df['ka_weight'] = df['teamkills'] / 35
        df.loc[df['ka_weight'] > 1, 'ka_weight'] = 1
        df.loc[df['teamkills'] == 0, 'ka_weight'] = 0
        df['deaths_weight'] = df['teamdeaths'] / 35
        df.loc[df['deaths_weight'] > 1, 'deaths_weight'] = 1
        df.loc[df['teamdeaths'] == 0, 'deaths_weight'] = 0

        df['predicted_damage_percentage'] = df['champion_dpm'] / df['team_sum_champion_dpm']
        df['actual_damage_percentage'] = df['dpm'] / df['team_sum_dpm']
        df['net_damage_percentage'] = df['actual_damage_percentage'] - df['predicted_damage_percentage']

        df['predicted_deaths_percentage'] = df['champion_deaths'] / df['team_sum_champion_deaths']
        df['actual_deaths_percentage'] = df['deaths'] / df['team_sum_deaths']
        df['net_deaths_percentage'] = 1 - (df['actual_deaths_percentage'] - df['predicted_deaths_percentage']) * df[
            'deaths_weight']

        df['predicted_kills_assists_percentage'] = df['champion_kills_assists'] / df['team_sum_champion_kills']
        df['actual_kills_assists_percentage'] = df['kills_assists'] / df['team_sum_kills']
        df['net_kills_assists_percentage'] = (df['actual_kills_assists_percentage'] - df[
            'predicted_kills_assists_percentage']) * df['ka_weight']

        df['net_damage_percentage'] = df['net_damage_percentage'].fillna(df['net_damage_percentage'].mean())
        df['net_deaths_percentage'] = df['net_deaths_percentage'].fillna(df['net_deaths_percentage'].mean())
        df['net_kills_assists_percentage'] = df['net_kills_assists_percentage'].fillna(
            df['net_kills_assists_percentage'].mean())

        return df


class DurationPerformanceGenerator(BaseTransformer):

    def __init__(self, max_duration: int = 43, min_duration: int = 28):
        self.max_duration = max_duration
        self.min_duration = min_duration

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df['duration'] = df['gamelength'] / 60

        df = df.assign(
            team_duration_performance=np.where(
                (df['result'] == 1) & (df['duration'] > self.max_duration),
                0.6,
                np.where(
                    (df['result'] == 1),
                    ((self.max_duration - df['duration']) / (self.max_duration - self.min_duration)) * 0.4 + 0.6,
                    np.where(
                        (df['result'] == 0) & (df['duration'] <= self.max_duration),
                        1 - ((self.max_duration - df['duration']) / (
                                self.max_duration - self.min_duration) * 0.4 + 0.6),
                        0.4,
                    )
                )
            )
            .clip(0, 1)
        )
        return df


class FinalLolTransformer(BaseTransformer):

    def __init__(self, column_names: ColumnNames):
        self.column_names = column_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.column_names.performance] *= 0.5 / df[self.column_names.performance].mean()
        df.sort_values(by=['date', 'gameid', 'teamname'], ascending=True)
        return df


if __name__ == '__main__':
    column_names = ColumnNames(
        team_id='teamname',
        match_id='gameid',
        start_date="date",
        player_id="playername",
        performance='performance',
        league='league'
    )
    performance_generator = CustomLolPerformanceGenerators(column_names=column_names)
    df = pd.read_csv("data/2023_LoL.csv")
    df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])

    df = (
        df.loc[lambda x: x.position != 'team']
        .assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
        .loc[lambda x: x.team_count == 2]
    )

    match_predictor = MatchPredictor(
        column_names=column_names,
        pre_rating_transformers=[performance_generator],
        target='result'

    )
    df = match_predictor.generate(df)

    print(log_loss(df['result'], df[match_predictor.predictor.pred_column]))
