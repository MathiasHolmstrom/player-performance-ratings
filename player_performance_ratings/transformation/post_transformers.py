import logging
from typing import Optional, Union

import pandas as pd

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformation.base_transformer import BaseTransformer, \
    DifferentGranularityTransformer
from player_performance_ratings.utils import validate_sorting


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


class LagTransformer(BaseTransformer):

    def __init__(self,
                 feature_names: list[str],
                 lag_length: int,
                 column_names: ColumnNames,
                 granularity: Optional[list[str]] = None,
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
            Defaults to player_id

        :param column_names:


        :param group_to_game_level
            If there are more multiple rows per granularity per game_id and you want to add the lag from the prior games, set to True.
            This will calculate the mean of the features per game and the lag will be the prior means


        :param prefix:
            Prefix for the new lag columns
        """

        self.feature_names = feature_names
        self.column_names = column_names
        self.granularity = granularity or [self.column_names.player_id]
        self.lag_length = lag_length
        self.prefix = prefix
        self._features_created = []
        self._df = None
        for feature_name in self.feature_names:
            for lag in range(1, self.lag_length + 1):
                self._features_created.append(f'{self.prefix}{lag}_{feature_name}')

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_sorting(df=df, column_names=self.column_names)
        if self._df is None:
            self._df = df
        else:
            self._df = pd.concat([self._df, df], axis=0)

        self._df = self._df.assign(
            __id=self._df[self.column_names.match_id].astype('str') + "__" + self._df[
                self.column_names.player_id].astype(
                'str'))
        self._df = self._df.drop_duplicates(subset=['__id'], keep='last')

        transformed_df = self.transform(pd.DataFrame(df))
        return transformed_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        ori_cols = df.columns.tolist()
        ori_index_values = df.index.tolist()

        all_df = pd.concat([self._df, df], axis=0)
        all_df = all_df.assign(
            __id=all_df[self.column_names.match_id].astype('str') + "__" + all_df[self.column_names.player_id].astype(
                'str'))
        all_df = all_df.drop_duplicates(subset=['__id'], keep='last')

        validate_sorting(df=all_df, column_names=self.column_names)

        for feature_name in self.feature_names:
            for lag in range(1, self.lag_length + 1):
                output_column_name = f'{self.prefix}{lag}_{feature_name}'
                if output_column_name in all_df.columns:
                    raise ValueError(
                        f'Column {output_column_name} already exists. Choose different prefix or ensure no duplication was performed')

        for feature_name in self.feature_names:
            for lag in range(1, self.lag_length + 1):
                output_column_name = f'{self.prefix}{lag}_{feature_name}'

                all_df = all_df.assign(
                    **{output_column_name: all_df.groupby(self.granularity)[feature_name].shift(lag)})

        df = df.assign(
            __id=df[self.column_names.match_id].astype('str') + "__" + df[
                self.column_names.player_id].astype(
                'str'))

        transformed_df = all_df[all_df['__id'].isin(df['__id'].unique().tolist())][ori_cols + self._features_created]
        transformed_df.index = ori_index_values
        return transformed_df[list(set(ori_cols + self._features_created))]

    @property
    def features_created(self) -> list[str]:
        return self._features_created


class LagLowerGranularityTransformer(DifferentGranularityTransformer):

    def __init__(self,
                 feature_names: list[str],
                 lag_length: int,
                 column_names: ColumnNames,
                 granularity: Union[list[str], str] = None,
                 weight_column: Optional[str] = None,
                 prefix: str = 'lag_lower_'
                 ):
        """
        :param
            feature_names: Which features to lag

        :param lag_length:
            Number of lags

        :param granularity:
            Columns to group by before lagging. E.g. player_id or [player_id, position].
            In the latter case it will get the lag for each player_id and position combination.
            Defaults to player_id

        :param column_names:
        """
        self.feature_names = feature_names
        self.column_names = column_names
        self.granularity = granularity or [self.column_names.player_id]
        self.lag_length = lag_length
        self.prefix = prefix
        self.weight_column = weight_column
        self._features_created = []
        self._diff_granularity_df = None
        for feature_name in self.feature_names:
            for lag in range(1, self.lag_length + 1):
                self._features_created.append(f'{self.prefix}{lag}_{feature_name}')

    def fit_transform(self, diff_granularity_df: pd.DataFrame, game_player_df: pd.DataFrame) -> pd.DataFrame:

        validate_sorting(df=game_player_df, column_names=self.column_names)

        if self._diff_granularity_df is None:
            self._diff_granularity_df = diff_granularity_df
        else:
            self._diff_granularity_df = pd.concat([self._diff_granularity_df, diff_granularity_df], axis=0)

        if len(self._diff_granularity_df.drop_duplicates()) != len(self._diff_granularity_df):
            raise ValueError(
                "Duplicated rows in diff_granularity_df. Please ensure there are no duplicates in the data")

        empty_diff_granularity_df = pd.DataFrame(columns=diff_granularity_df.columns)

        transformed_game_player_df = self.transform(diff_granularity_df=empty_diff_granularity_df,
                                                    game_player_df=game_player_df)

        return transformed_game_player_df

    def transform(self, diff_granularity_df: pd.DataFrame, game_player_df: pd.DataFrame) -> pd.DataFrame:
        validate_sorting(df=game_player_df, column_names=self.column_names)
        if self._diff_granularity_df is None:
            raise ValueError("fit_transform needs to be called before transform")

        diff_granularity_df = pd.concat([self._diff_granularity_df, diff_granularity_df], axis=0)

        diff_granularity_df = diff_granularity_df.sort_values(
            by=[self.column_names.start_date, self.column_names.match_id, self.column_names.team_id,
                self.column_names.player_id])

        for feature_name in self.feature_names:
            for lag in range(1, self.lag_length + 1):
                output_column_name = f'{self.prefix}{lag}_{feature_name}'
                if output_column_name in diff_granularity_df.columns:
                    raise ValueError(
                        f'Column {output_column_name} already exists. Choose different prefix or ensure no duplication was performed')

        for feature_name in self.feature_names:
            for lag in range(1, self.lag_length + 1):
                grouped_data = create_output_column_by_game_group(data=diff_granularity_df, feature_name=feature_name,
                                                                  weight_column=self.weight_column,
                                                                  game_id=self.column_names.match_id,
                                                                  granularity=self.granularity)

                output_column_name = f'{self.prefix}{lag}_{feature_name}'

                grouped_data = grouped_data.assign(
                    **{output_column_name: grouped_data.groupby(self.granularity)[feature_name].shift(lag)})
                game_player_df = game_player_df[[c for c in game_player_df.columns if c != output_column_name]].merge(
                    grouped_data[[output_column_name, self.column_names.match_id] + self.granularity],
                    on=self.granularity + [self.column_names.match_id], how='left')

        return game_player_df

    @property
    def features_created(self) -> list[str]:
        return self._features_created


class RollingMeanTransformer(BaseTransformer):

    def __init__(self,
                 feature_names: list[str],
                 window: int,
                 column_names: ColumnNames,
                 granularity: Union[list[str], str] = None,
                 min_periods: int = 1,

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
        self.column_names = column_names
        self._df = None
        self.prefix = prefix
        self._features_created = [f'{self.prefix}{self.window}_{c}' for c in self.feature_names]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        ori_cols = df.columns.tolist()
        ori_index_values = df.index.tolist()

        all_df = pd.concat([self._df, df], axis=0).reset_index()
        all_df = all_df.assign(
            __id=all_df[self.column_names.match_id].astype('str') + "__" + all_df[self.column_names.player_id].astype(
                'str'))
        all_df = all_df.drop_duplicates(subset=['__id'], keep='last')

        validate_sorting(df=all_df, column_names=self.column_names)

        for feature_name in self.feature_names:

            output_column_name = f'{self.prefix}{self.window}_{feature_name}'
            if output_column_name in all_df.columns:
                raise ValueError(
                    f'Column {output_column_name} already exists. Choose different prefix or ensure no duplication was performed')

            all_df = all_df.assign(**{output_column_name: all_df.groupby(self.granularity)[feature_name].apply(
                lambda x: x.shift().rolling(self.window, min_periods=self.min_periods).mean())})

        df = df.assign(
            __id=df[self.column_names.match_id].astype('str') + "__" + df[
                self.column_names.player_id].astype(
                'str'))

        transformed_df = all_df[all_df['__id'].isin(df['__id'].unique().tolist())][ori_cols + self._features_created]
        transformed_df.index = ori_index_values
        return transformed_df[list(set(ori_cols + self._features_created))]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        validate_sorting(df=df, column_names=self.column_names)
        if self._df is None:
            self._df = df
        else:
            self._df = pd.concat([self._df, df], axis=0)

        self._df = self._df.assign(
            __id=self._df[self.column_names.match_id].astype('str') + "__" + self._df[
                self.column_names.player_id].astype(
                'str'))
        self._df = self._df.drop_duplicates(subset=['__id'], keep='last')

        transformed_df = self.transform(pd.DataFrame(df))
        return transformed_df

    @property
    def features_created(self) -> list[str]:
        return self._features_created
