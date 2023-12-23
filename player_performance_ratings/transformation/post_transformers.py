import logging
from typing import Optional, Union

import pandas as pd

from player_performance_ratings.transformation.base_transformer import BaseTransformer


def create_output_column_by_game_group(data: pd.DataFrame, feature_name: str,
                                       weight_column: str, game_id: str, granularity: list[str]) -> pd.DataFrame:
    if weight_column:
        data = data.assign(**{f'weighted_{feature_name}': data[feature_name] * data[weight_column]})
        data = data.assign(
            **{"sum_weighted": data.groupby(granularity + [game_id])[weight_column].transform("sum")})
        data = data.assign(**{feature_name: data[f'weighted_{feature_name}'] / data['sum_weighted']})
        data = data.groupby(granularity + [game_id])[feature_name].sum().reset_index()

    else:
        data = data.groupby(granularity + [game_id])[feature_name].mean().reset_index()
    return data


class LagTransformation(BaseTransformer):

    def __init__(self,
                 feature_names: list[str],
                 lag_length: int,
                 granularity: Union[list[str], str],
                 game_id: Optional[str] = None,
                 weight_column: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None,
                 prefix: str = 'lag_'
                 ):

        """
        :param
            feature_names: Which features to lag

        :param lag_length:
            Number of lags

        :param granularity:
            Columns to group by before lagging. E.g. player_id or [player_id, position].
             In the latter case it will get the lag for each player_id and position combination.

        :param game_id:
            Column name of game_id.
            If there are more multiple rows per granularity per game_id and you want to add the lag from the prior games, set game_id
            This will calculate the mean of the features per game and the lag will be the prior means

        :param weight_column:
            Only used if game_id is set.
            Will calculate weighted mean of the features per game and the lag will be the prior weighted means.
            This is useful when working with partial game-data of different lengths.
            In that case it can beneficial to set the game-length as weight_colum.

        :param df: Optional parameter to pass in a dataframe to calculate the lag on.
            If not passed in, it will use the dataframe passed in the transform method.
            This is useful if you want to calculate the lag on a different dataframe than the one you want to transform.
            Will merge the two dataframes on game_id and granularity.

        :param prefix:
            Prefix for the new lag columns
        """

        self.feature_names = feature_names
        self.game_id = game_id
        self.granularity = granularity or []
        if isinstance(self.granularity, str):
            self.granularity = [self.granularity]
        self.lag_length = lag_length
        self.weight_column = weight_column
        self.df = df
        self.prefix = prefix
        self._features_created = []
        for feature_name in self.feature_names:
            for lag in range(1, self.lag_length + 1):
                self._features_created.append(f'{self.prefix}{lag}_{feature_name}')


        if self.df is not None and self.game_id is None:
            raise ValueError('If passing in a dataframe to calculate the lag on, you need to set game_id')

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.df is not None:
            data = self.df
        else:
            data = df

        for feature_name in self.feature_names:
            for lag in range(1, self.lag_length + 1):
                output_column_name = f'{self.prefix}{lag}_{feature_name}'
                if output_column_name in data.columns:
                    raise ValueError(
                        f'Column {output_column_name} already exists. Choose different prefix or ensure no duplication was performed')

                if self.game_id:
                    data = create_output_column_by_game_group(data=data, feature_name=feature_name,
                                                              weight_column=self.weight_column, game_id=self.game_id,
                                                              granularity=self.granularity)

                data = data.assign(**{output_column_name: data.groupby(self.granularity)[feature_name].shift(lag)})

        if self.game_id is not None:
            df = df.merge(data[self._features_created + self.granularity + [self.game_id]],
                          on=self.granularity + [self.game_id], how='left')
        else:
            df = data

        return df

    @property
    def features_created(self) -> list[str]:
        return self._features_created


class RollingMeanTransformation(BaseTransformer):

    def __init__(self,
                 feature_names: list[str],
                 window: int,
                 granularity: Union[list[str], str],
                 min_periods: int = 1,
                 game_id: Optional[str] = None,
                 weight_column: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None,
                 prefix: str = 'rolling_mean_'):
        """

        :param feature_names:
            Features to create rolling mean for

        :param granularity:
            Columns to group by before rolling mean. E.g. player_id or [player_id, position].
             In the latter case it will get the rolling mean for each player_id and position combination.

        :param window:
            Window size for rolling mean, if 10 will calculate rolling mean over the prior 10 observations

        :param game_id:
            Column name of game_id.
            If there are more multiple rows per granularity per game_id and you want to add the rolling mean per game_id, set game_id
            This will calculate the mean of the features per game and the rolling mean will be calculated on that.

        :param weight_column:
            Only used if game_id is set.
            Will calculate weighted mean of the features per game and the rolling mean will be the prior weighted means.
            This is useful when working with partial game-data of different lengths.
            In that case it can beneficial to set the game-length as weight_colum.

        :param df: Optional parameter to pass in a dataframe to calculate the rolling mean on.
            If not passed in, it will use the dataframe passed in the transform method.
            This is useful if you want to calculate the rolling mean on a different dataframe than the one you want to transform.
            Will merge the two dataframes on game_id and granularity.

        :param prefix:
            Prefix for the new rolling mean columns
        """

        self.feature_names = feature_names
        self.granularity = granularity
        if isinstance(self.granularity, str):
            self.granularity = [self.granularity]
        self.window = window
        self.min_periods = min_periods
        self.weight_column = weight_column
        self.game_id = game_id
        self.df = df
        self.prefix = prefix
        self._features_created = [f'{self.prefix}{self.window}_{c}' for c in self.feature_names]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.df is not None:
            data = self.df
        else:
            data = df

        for feature_name in self.feature_names:
            output_column_name = f'{self.prefix}{self.window}_{feature_name}'
            if output_column_name in df.columns:
                raise ValueError(
                    f'Column {output_column_name} already exists. Choose different prefix or ensure no duplication was performed')

            if self.game_id:
                data = create_output_column_by_game_group(data=data, feature_name=feature_name,
                                                          weight_column=self.weight_column, game_id=self.game_id,
                                                          granularity=self.granularity)

            data = data.assign(**{output_column_name: data.groupby(self.granularity)[feature_name].apply(
                lambda x: x.shift().rolling(self.window, min_periods=self.min_periods).mean())})

            if self.game_id is not None:
                df = df.merge(data[self._features_created + self.granularity + [self.game_id]],
                              on=self.granularity + [self.game_id], how='left')
            else:
                df = data

        return df

    @property
    def features_created(self) -> list[str]:
        return self._features_created
