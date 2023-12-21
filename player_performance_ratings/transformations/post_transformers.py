import logging
from typing import Optional

import pandas as pd

from player_performance_ratings import BaseTransformer


class MovingAverageBayesianTransformation(BaseTransformer):

    def __init__(self,
                 feature_names: list[str],
                 game_id: str,
                 granularity: str,
                 date_time_column_name: str,
                 weight: float = 0.95,
                 max_days_ago: int = 600,
                 max_prior_weight_sum: int = 100,
                 prior_weight: float = 0.95,
                 prior_granualarity_mean: Optional[list[str]] = None,
                 prefix: str = 'moving_average_bayesian_'
                 ):
        self.feature_names = feature_names
        self.game_id = game_id
        self.granularity = granularity
        self.date_column_name = date_time_column_name
        self.weight = weight
        self.max_days_ago = max_days_ago
        self.max_prior_weight_sum = max_prior_weight_sum
        self.prior_weight = prior_weight
        self.prior_granularity_mean = prior_granualarity_mean
        self.prefix = prefix
        self.df = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df['id'] = df[self.granularity].astype(str) + df[self.game_id].astype(str)

        if self.df is not None:
            df = pd.concat([self.df, df])
            df = df.drop_duplicates(subset=['id', self.date_column_name], keep='last')

        df.assign(day_number=(df[self.date_column_name] - pd.Timestamp("1970-01-01").tz_localize(
            'UTC')) // pd.Timedelta('1d'))

        df_adj = pd.DataFrame()


        for player in df[self.granularity].unique():
            df_player = df[df[self.granularity] == player]
            for current_day in df_player["day_number"].unique():
                # Creating a row for the current day
                current_day_row = df_player[df_player['day_number'] == current_day].iloc[0].copy()

                # Subset the DataFrame up to but not including the current day
                df_day_prior = df_player[df_player['day_number'] < current_day]

                # Calculate the normalized weighted performance if there is prior data
                if not df_day_prior.empty:
                    df_day_prior['weight'] = df_day_prior['day_number'].apply(
                        lambda x: self.weight ** (current_day - x))

                    df_day_prior['likelihood_weight'] = df_day_prior['day_number'].apply(
                        lambda x: self.prior_weight ** (current_day - x))

                    cumulative_sum_likelihood_weights = df_day_prior['likelihood_weight'].sum()

                    for feature_name in self.feature_names:
                        df_day_prior[f'weighted_feature_value'] = df_day_prior[feature_name] * df_day_prior['weight']
                        cumulative_sum_of_weights = df_day_prior['weight'].sum()
                        cumulative_weighted_performance = df_day_prior['weighted_feature_value'].sum()
                        normalized_weighted_performance = cumulative_weighted_performance / cumulative_sum_of_weights
                        current_day_row[f"{feature_name}_normalized_weighted_performance"] = normalized_weighted_performance

                        current_day_row['cumulative_sum_likelihood_weights']

                else:
                    for feature_name in self.feature_names:
                        current_day_row[f"{feature_name}_normalized_weighted_performance"] = 0
                        cumulative_sum_likelihood_weights = 0


                df_adj = df_adj.append(current_day_row, ignore_index=True)

        df_adj["cumsum_normalized_weighted_performance"] = \
            df_adj.groupby(["player_id"])[
                "normalized_weighted_performance"].cumsum()
        df_adj['games_played'] = df_adj.groupby(["player_id"]).cumcount() + 1

        return df_adj


class GameLagTransformation(BaseTransformer):

    def __init__(self, feature_names: list[str], game_id: str, granularity: list[str], lag: int, prefix: str = 'lag'):
        self.feature_names = feature_names
        self.game_id = game_id
        self.granularity = granularity
        self.lag = lag
        self.prefix = prefix

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature_name in self.feature_names:
            output_column_name = f'{self.prefix}_{self.lag}_{feature_name}'
            if output_column_name in df.columns:
                output_column_name += '_1'
                logging.warning(f'Column {output_column_name} already exists, renaming to {output_column_name}')

            df.assign(**{output_column_name: df.groupby(self.granularity)[feature_name].shift(self.lag)})

        return df


class RollingMeanTransformations(BaseTransformer):

    def __init__(self, feature_names: list[str], granularity: list[str], window: int, prefix: str = 'rolling_mean'):
        self.feature_names = feature_names
        self.granularity = granularity
        self.window = window
        self.prefix = prefix

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature_name in self.feature_names:
            output_column_name = f'{self.prefix}_{self.window}_{feature_name}'
            if output_column_name in df.columns:
                output_column_name += '_1'
                logging.warning(f'Column {output_column_name} already exists, renaming to {output_column_name}')

            df.assign(**{output_column_name: df.groupby(self.granularity)[feature_name].shift(self.window).rolling(
                self.window).mean()})

        return df
