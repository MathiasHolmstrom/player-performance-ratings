import logging
from typing import Optional, Union

import pandas as pd

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformation.base_transformer import BaseTransformer, \
    DifferentGranularityTransformer, BasePostTransformer
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


class SklearnPredictorTransformer(BasePostTransformer):

    def __init__(self, estimator, features: list[str], target: str, train_date: str, date_column_name: str,
                 feature_out: Optional[str] = None):
        super().__init__(features=features)
        self.estimator = estimator
        self.target = target
        self.train_date = train_date
        self.date_column_name = date_column_name
        self.feature_out = feature_out or f'{self.target}_transform_prediction'

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        train_df = df[df[self.date_column_name] < self.train_date]
        self.estimator.fit(train_df[self.features], train_df[self.target])
        return self.transform(df=df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        prediction = self.estimator.predict(df[self.features])
        df = df.assign(**{self.feature_out: prediction})
        return df

    @property
    def features_out(self) -> list[str]:
        return [f'{self.feature_out}']


class GameTeamMembersColumnsTransformer(BasePostTransformer):

    def __init__(self, column_names: ColumnNames, features: list[str], players_per_match_per_team_count: int,
                 sort_by: Optional[list[str]] = None):
        super().__init__(features=features)
        self.column_names = column_names
        self.players_per_match_per_team_count = players_per_match_per_team_count
        self.sort_by = sort_by
        self._features_out = []
        for number in range(1, self.players_per_match_per_team_count + 1):
            for feature in self.features:
                feature_name = f'team_player{number}_{feature}'
                self._features_out.append(feature_name)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df=df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        def add_teammate_minutes(row, grouped_df):
            game_team_group = grouped_df.get_group(
                (row[self.column_names.rating_update_id], row[self.column_names.team_id]))
            number = 0
            if self.sort_by:
                game_team_group = game_team_group.sort_values(by=self.sort_by, ascending=False)

            for index, teammate in game_team_group.iterrows():
                if teammate[self.column_names.player_id] != row[self.column_names.player_id]:
                    number += 1
                    for feature in self.features:
                        feature_name = f'team_player{number}_{feature}'
                        row[feature_name] = teammate[feature]

            return row

        grouped = df.groupby([self.column_names.rating_update_id, self.column_names.team_id])

        df = df.apply(lambda row: add_teammate_minutes(row, grouped), axis=1)
        return df.sort_values(
            by=[self.column_names.start_date, self.column_names.rating_update_id, self.column_names.team_id,
                self.column_names.player_id])

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class NormalizerTransformer(BasePostTransformer):

    def __init__(self, features: list[str], granularity, target_mean: Optional[float] = None,
                 create_target_as_mean: bool = False):
        super().__init__(features=features)
        self.granularity = granularity
        self.target_mean = target_mean
        self.create_target_as_mean = create_target_as_mean
        self._features_to_normalization_target = {}

        if self.target_mean is None and not self.create_target_as_mean:
            raise ValueError("Either target_sum or create_target_as_mean must be set")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._features_to_normalization_target = {f: self.target_mean for f in self.features} if self.target_mean else {
            f: df[f].mean() for f in self.features}
        return self.transform(df=df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(**{f'__mean_{feature}': df.groupby(self.granularity)[feature].transform('mean') for feature in
                          self.features})
        for feature, target_sum in self._features_to_normalization_target.items():
            df = df.assign(**{feature: df[feature] / df[f'__mean_{feature}'] * target_sum})
        return df.drop(columns=[f'__mean_{feature}' for feature in self.features])

    @property
    def features_out(self) -> list[str]:
        return self.features


class LagTransformer(BasePostTransformer):

    def __init__(self,
                 features: list[str],
                 lag_length: int,
                 column_names: ColumnNames,
                 granularity: Optional[list[str]] = None,
                 days_between_lags: Optional[list[int]] = None,
                 prefix: str = 'lag_',
                 future_lag: bool = False,
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

        super().__init__(features=features)
        self.column_names = column_names
        self.granularity = granularity or [self.column_names.player_id]
        self.lag_length = lag_length
        self.days_between_lags = days_between_lags or []
        self.prefix = prefix
        self.future_lag = future_lag
        self._features_out = []
        self._df = None
        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                self._features_out.append(f'{self.prefix}{lag}_{feature_name}')

        for days_lag in self.days_between_lags:
            self._features_out.append(f'{self.prefix}{days_lag}_days_ago')

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_sorting(df=df, column_names=self.column_names)
        for feature_out in self._features_out:
            if feature_out in df.columns:
                raise ValueError(
                    f'Column {feature_out} already exists. Choose different prefix or ensure no duplication was performed')
        if self._df is None:
            self._df = df
        else:
            self._df = pd.concat([self._df, df], axis=0)

        self._df = self._df.assign(
            __id=self._df[self.column_names.rating_update_id].astype('str') + "__" + self._df[
                self.column_names.player_id].astype(
                'str'))
        self._df = self._df.drop_duplicates(subset=['__id'], keep='last')

        transformed_df = self.transform(pd.DataFrame(df))
        return transformed_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        if len(df.drop_duplicates(subset=[self.column_names.player_id, self.column_names.rating_update_id])) != len(df):
            raise ValueError(
                f"Duplicated rows in df. Df must be a unique combination of {self.column_names.player_id} and {self.column_names.rating_update_id}")

        ori_cols = df.columns.tolist()
        ori_index_values = df.index.tolist()

        all_df = pd.concat([self._df, df], axis=0)
        all_df = all_df.assign(
            __id=all_df[self.column_names.rating_update_id].astype('str') + "__" + all_df[
                self.column_names.player_id].astype(
                'str'))
        all_df = all_df.drop_duplicates(subset=['__id'], keep='last')

        validate_sorting(df=all_df, column_names=self.column_names)

        grouped = all_df.groupby(self.granularity + [self.column_names.rating_update_id, self.column_names.start_date])[self.features].mean().reset_index()

        for days_lag in self.days_between_lags:
            if self.future_lag:
                grouped["shifted_days"] = grouped.groupby(self.granularity)[self.column_names.start_date].shift(-days_lag)
                grouped[f'{self.prefix}{days_lag}_days_ago'] = (
                        pd.to_datetime(grouped["shifted_days"]) - pd.to_datetime(
                    grouped[self.column_names.start_date])).dt.days
            else:
                grouped["shifted_days"] = grouped.groupby(self.granularity)[self.column_names.start_date].shift(days_lag)
                grouped[f'{self.prefix}{days_lag}_days_ago'] = (pd.to_datetime(
                    grouped[self.column_names.start_date]) - pd.to_datetime(grouped["shifted_days"])).dt.days

            grouped = grouped.drop(columns=["shifted_days"])

        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                output_column_name = f'{self.prefix}{lag}_{feature_name}'
                if output_column_name in all_df.columns:
                    raise ValueError(
                        f'Column {output_column_name} already exists. Choose different prefix or ensure no duplication was performed')

        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                output_column_name = f'{self.prefix}{lag}_{feature_name}'
                if self.future_lag:
                    grouped = grouped.assign(
                        **{output_column_name: grouped.groupby(self.granularity)[feature_name].shift(-lag)})
                else:
                    grouped = grouped.assign(
                        **{output_column_name: grouped.groupby(self.granularity)[feature_name].shift(lag)})

        all_df = all_df.merge(grouped[self.granularity + [self.column_names.rating_update_id, *self.features_out]],
                              on=self.granularity + [self.column_names.rating_update_id], how='left')

        df = df.assign(
            __id=df[self.column_names.rating_update_id].astype('str') + "__" + df[
                self.column_names.player_id].astype(
                'str'))

        transformed_df = all_df[all_df['__id'].isin(df['__id'].unique().tolist())][ori_cols + self._features_out]
        transformed_df.index = ori_index_values
        return transformed_df[list(set(ori_cols + self._features_out))]

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class LagLowerGranularityTransformer(DifferentGranularityTransformer):

    def __init__(self,
                 features: list[str],
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
        super().__init__(features)
        self.column_names = column_names
        self.granularity = granularity or [self.column_names.player_id]
        self.lag_length = lag_length
        self.prefix = prefix
        self.weight_column = weight_column
        self._features_out = []
        self._diff_granularity_df = None
        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                self._features_out.append(f'{self.prefix}{lag}_{feature_name}')

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
            by=[self.column_names.start_date, self.column_names.rating_update_id, self.column_names.team_id,
                self.column_names.player_id])

        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                output_column_name = f'{self.prefix}{lag}_{feature_name}'
                if output_column_name in diff_granularity_df.columns:
                    raise ValueError(
                        f'Column {output_column_name} already exists. Choose different prefix or ensure no duplication was performed')

        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                grouped_data = create_output_column_by_game_group(data=diff_granularity_df, feature_name=feature_name,
                                                                  weight_column=self.weight_column,
                                                                  game_id=self.column_names.rating_update_id,
                                                                  granularity=self.granularity)

                output_column_name = f'{self.prefix}{lag}_{feature_name}'

                grouped_data = grouped_data.assign(
                    **{output_column_name: grouped_data.groupby(self.granularity)[feature_name].shift(lag)})
                game_player_df = game_player_df[[c for c in game_player_df.columns if c != output_column_name]].merge(
                    grouped_data[[output_column_name, self.column_names.rating_update_id] + self.granularity],
                    on=self.granularity + [self.column_names.rating_update_id], how='left')

        return game_player_df

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class RollingMeanTransformer(BasePostTransformer):

    def __init__(self,
                 features: list[str],
                 window: int,
                 column_names: ColumnNames,
                 granularity: Union[list[str], str] = None,
                 min_periods: int = 1,
                 prefix: str = 'rolling_mean_'):
        """

        :param features:
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

        super().__init__(features=features)
        self.granularity = granularity
        if isinstance(self.granularity, str):
            self.granularity = [self.granularity]
        self.window = window
        self.min_periods = min_periods
        self.column_names = column_names
        self._df = None
        self.prefix = prefix
        self._features_out = [f'{self.prefix}{self.window}_{c}' for c in self.features]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        ori_cols = df.columns.tolist()
        ori_index_values = df.index.tolist()

        all_df = pd.concat([self._df, df], axis=0).reset_index()
        all_df = all_df.assign(
            __id=all_df[self.column_names.rating_update_id].astype('str') + "__" + all_df[
                self.column_names.player_id].astype(
                'str'))
        all_df = all_df.drop_duplicates(subset=['__id'], keep='last')

        validate_sorting(df=all_df, column_names=self.column_names)

        for feature_name in self.features:

            output_column_name = f'{self.prefix}{self.window}_{feature_name}'
            if output_column_name in all_df.columns:
                raise ValueError(
                    f'Column {output_column_name} already exists. Choose different prefix or ensure no duplication was performed')

            grp = all_df.groupby(self.granularity + [self.column_names.rating_update_id])[feature_name].mean().reset_index()

            grp = grp.assign(**{output_column_name: grp.groupby(self.granularity)[feature_name].apply(
                lambda x: x.shift().rolling(self.window, min_periods=self.min_periods).mean())})

            all_df = all_df.merge(grp[self.granularity + [self.column_names.rating_update_id, output_column_name]],
                                  on=self.granularity + [self.column_names.rating_update_id], how='left')
            all_df = all_df.sort_values(by=[self.column_names.start_date, self.column_names.match_id,
                                        self.column_names.team_id, self.column_names.player_id])
        df = df.assign(
            __id=df[self.column_names.rating_update_id].astype('str') + "__" + df[
                self.column_names.player_id].astype(
                'str'))

        transformed_df = all_df[all_df['__id'].isin(df['__id'].unique().tolist())][ori_cols + self._features_out]
        transformed_df.index = ori_index_values
        return transformed_df[list(set(ori_cols + self._features_out))]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        validate_sorting(df=df, column_names=self.column_names)
        if self._df is None:
            self._df = df
        else:
            self._df = pd.concat([self._df, df], axis=0)

        self._df = self._df.assign(
            __id=self._df[self.column_names.rating_update_id].astype('str') + "__" + self._df[
                self.column_names.player_id].astype(
                'str'))
        self._df = self._df.drop_duplicates(subset=['__id'], keep='last')

        transformed_df = self.transform(pd.DataFrame(df))
        return transformed_df

    @property
    def features_out(self) -> list[str]:
        return self._features_out
