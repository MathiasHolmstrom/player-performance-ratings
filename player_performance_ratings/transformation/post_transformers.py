from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd

from player_performance_ratings import ColumnNames
from player_performance_ratings.predictor import Predictor, BasePredictor
from player_performance_ratings.transformation.base_transformer import DifferentGranularityTransformer, \
    BasePostTransformer
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


def add_opponent_features(df: pd.DataFrame, column_names: ColumnNames, features: list[str]) -> pd.DataFrame:
    team_features = df.groupby([column_names.team_id, column_names.rating_update_match_id])[
        features].mean().reset_index()
    df_opponent_feature = team_features.rename(
        columns={**{column_names.team_id: 'opponent_team_id'},
                 **{f: f"{f}_opponent" for f in features}}
    )
    new_df = df.merge(df_opponent_feature, on=[column_names.match_id], suffixes=('', '_team_sum'))
    new_df = new_df[new_df[column_names.team_id] != new_df['opponent_team_id']].drop(
        columns=['opponent_team_id'])

    new_feats = [f"{f}_opponent" for f in features]
    return df.merge(new_df[[column_names.match_id, column_names.team_id, column_names.player_id, *new_feats]],
                    on=[column_names.match_id, column_names.team_id, column_names.player_id], how='left')


class PredictorTransformer(BasePostTransformer):

    def __init__(self, predictor: BasePredictor, features: list[str] = None):
        self.predictor = predictor
        super().__init__(features=features)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.predictor.train(df=df, estimator_features=self.features)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.predictor.add_prediction(df=df)
        return df

    @property
    def features_out(self) -> list[str]:
        return [f'{self.predictor.pred_column}']


class RatioTeamPredictorTransformer(BasePostTransformer):
    def __init__(self,
                 features: list[str],
                 predictor: BasePredictor,
                 game_id: str,
                 team_id: str,
                 team_total_prediction_column: Optional[str] = None,
                 prefix: str = "_ratio_team"
                 ):
        super().__init__(features=features)
        self.predictor = predictor
        self.game_id = game_id
        self.team_id = team_id
        self.team_total_prediction_column = team_total_prediction_column
        self.prefix = prefix
        self.predictor._pred_column = f"__prediction__{self.predictor.target}"
        self._features_out = [self.predictor.target + prefix]
        if self.team_total_prediction_column:
            self._features_out.append(self.predictor.target + prefix + "_team_total_multiplied")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.predictor.train(df=df, estimator_features=self.features)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.predictor.add_prediction(df=df)
        df[self.predictor.pred_column + "_sum"] = df.groupby([self.game_id, self.team_id])[
            self.predictor.pred_column].transform('sum')
        df[self._features_out[0]] = df[self.predictor.pred_column] / df[self.predictor.pred_column + "_sum"]
        if self.team_total_prediction_column:
            df = df.assign(**{self.predictor.target + self.prefix + "_team_total_multiplied": df[self._features_out[
                0]] * df[
                                                                                                  self.team_total_prediction_column]})
        return df.drop(columns=[self.predictor.pred_column + "_sum", self.predictor.pred_column])

    @property
    def features_out(self) -> list[str]:
        return self._features_out


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


class NormalizerTargetColumnTransformer(BasePostTransformer):

    def __init__(self, features: list[str], granularity, target_sum_column_name: str, prefix: str = "__normalized_"):
        super().__init__(features=features)
        self.granularity = granularity
        self.prefix = prefix
        self.target_sum_column_name = target_sum_column_name
        self._features_to_normalization_target = {}
        self._features_out = []
        for feature in self.features:
            self._features_out.append(f'{self.prefix}{feature}')

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df=df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            df[f"{feature}_sum"] = df.groupby(self.granularity)[feature].transform('sum')
            df = df.assign(
                **{self.prefix + feature: df[feature] / df[f"{feature}_sum"] * df[self.target_sum_column_name]})
            df = df.drop(columns=[f"{feature}_sum"])
        return df

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
                 add_opponent: bool = False,
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
        self.add_opponent = add_opponent
        self._features_out = []
        self._df = None

        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                self._features_out.append(f'{self.prefix}{lag}_{feature_name}')
                if self.add_opponent:
                    self._features_out.append(f'{self.prefix}{lag}_{feature_name}_opponent')

        for days_lag in self.days_between_lags:
            self._features_out.append(f'{self.prefix}{days_lag}_days_ago')
            if self.add_opponent:
                self._features_out.append(f'{self.prefix}{days_lag}_days_ago_opponent')

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(**{self.column_names.player_id: lambda x: x[self.column_names.player_id].astype('str')})
        df = df.assign(**{
            self.column_names.rating_update_match_id: lambda x: x[self.column_names.rating_update_match_id].astype(
                'str')})
        df = df.assign(**{
            self.column_names.parent_team_id: lambda x: x[self.column_names.parent_team_id].astype(
                'str')})

        for feature_out in self._features_out:
            if feature_out in df.columns:
                raise ValueError(
                    f'Column {feature_out} already exists. Choose different prefix or ensure no duplication was performed')
        if self._df is None:
            self._df = df
        else:
            self._df = pd.concat([self._df, df], axis=0)

        self._df = self._df.assign(
            __id=self._df[[self.column_names.rating_update_match_id, self.column_names.parent_team_id,
                           self.column_names.player_id]].agg('__'.join, axis=1))
        self._df = self._df.drop_duplicates(subset=['__id'], keep='last')

        transformed_df = self.transform(pd.DataFrame(df))
        return transformed_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(**{self.column_names.player_id: lambda x: x[self.column_names.player_id].astype('str')})
        df = df.assign(**{
            self.column_names.rating_update_match_id: lambda x: x[self.column_names.rating_update_match_id].astype(
                'str')})
        df = df.assign(**{
            self.column_names.parent_team_id: lambda x: x[self.column_names.parent_team_id].astype(
                'str')})
        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        if len(df.drop_duplicates(
                subset=[self.column_names.player_id, self.column_names.parent_team_id,
                        self.column_names.rating_update_match_id])) != len(df):
            raise ValueError(
                f"Duplicated rows in df. Df must be a unique combination of {self.column_names.player_id} and {self.column_names.rating_update_match_id}")

        ori_cols = df.columns.tolist()
        ori_index_values = df.index.tolist()

        all_df = pd.concat([self._df, df], axis=0)
        all_df = all_df.assign(
            __id=all_df[[self.column_names.rating_update_match_id, self.column_names.parent_team_id,
                         self.column_names.player_id]].agg('__'.join, axis=1))
        all_df = all_df.drop_duplicates(subset=['__id'], keep='last')
        if self.column_names.participation_weight:
            for feature in self.features:
                all_df = all_df.assign(**{feature: all_df[feature] * all_df[self.column_names.participation_weight]})

        grouped = \
            all_df.groupby(self.granularity + [self.column_names.rating_update_match_id, self.column_names.start_date])[
                self.features].mean().reset_index()

        grouped = grouped.sort_values([self.column_names.start_date, self.column_names.rating_update_match_id])

        for days_lag in self.days_between_lags:
            if self.future_lag:
                grouped["shifted_days"] = grouped.groupby(self.granularity)[self.column_names.start_date].shift(
                    -days_lag)
                grouped[f'{self.prefix}{days_lag}_days_ago'] = (
                        pd.to_datetime(grouped["shifted_days"]) - pd.to_datetime(
                    grouped[self.column_names.start_date])).dt.days
            else:
                grouped["shifted_days"] = grouped.groupby(self.granularity)[self.column_names.start_date].shift(
                    days_lag)
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

        feats_out = []
        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                feats_out.append(f'{self.prefix}{lag}_{feature_name}')

        for days_lag in self.days_between_lags:
            feats_out.append(f'{self.prefix}{days_lag}_days_ago')

        all_df = all_df.merge(
            grouped[self.granularity + [self.column_names.rating_update_match_id, *feats_out]],
            on=self.granularity + [self.column_names.rating_update_match_id], how='left')

        if self.add_opponent:
            all_df = add_opponent_features(df=all_df, column_names=self.column_names, features=feats_out)

        df = df.assign(
            __id=df[[self.column_names.rating_update_match_id, self.column_names.parent_team_id,
                     self.column_names.player_id]].agg('__'.join, axis=1))

        transformed_df = all_df[all_df['__id'].isin(df['__id'].unique().tolist())][ori_cols + self._features_out]
        transformed_df.index = ori_index_values
        return transformed_df[list(set(ori_cols + self._features_out))]

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class LagLowerGranularityTransformer(DifferentGranularityTransformer):
    """
    TODO: Make work
    """

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
            by=[self.column_names.start_date, self.column_names.rating_update_match_id, self.column_names.team_id,
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
                                                                  game_id=self.column_names.rating_update_match_id,
                                                                  granularity=self.granularity)

                output_column_name = f'{self.prefix}{lag}_{feature_name}'

                grouped_data = grouped_data.assign(
                    **{output_column_name: grouped_data.groupby(self.granularity)[feature_name].shift(lag)})
                game_player_df = game_player_df[[c for c in game_player_df.columns if c != output_column_name]].merge(
                    grouped_data[[output_column_name, self.column_names.rating_update_match_id] + self.granularity],
                    on=self.granularity + [self.column_names.rating_update_match_id], how='left')

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
        self.granularity = granularity or [column_names.player_id]
        if isinstance(self.granularity, str):
            self.granularity = [self.granularity]
        self.window = window
        self.min_periods = min_periods
        self.column_names = column_names
        self._df = None
        self.prefix = prefix
        self._features_out = [f'{self.prefix}{self.window}_{c}' for c in self.features]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(**{self.column_names.player_id: lambda x: x[self.column_names.player_id].astype('str')})
        df = df.assign(**{
            self.column_names.rating_update_match_id: lambda x: x[self.column_names.rating_update_match_id].astype(
                'str')})
        df = df.assign(**{
            self.column_names.parent_team_id: lambda x: x[self.column_names.parent_team_id].astype(
                'str')})
        validate_sorting(df=df, column_names=self.column_names)
        if self._df is None:
            self._df = df
        else:
            self._df = pd.concat([self._df, df], axis=0)

        self._df = self._df.assign(
            __id=self._df[[self.column_names.rating_update_match_id, self.column_names.parent_team_id,
                           self.column_names.player_id]].agg('__'.join, axis=1))
        self._df = self._df.drop_duplicates(subset=['__id'], keep='last')

        transformed_df = self.transform(pd.DataFrame(df))
        return transformed_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(**{self.column_names.player_id: lambda x: x[self.column_names.player_id].astype('str')})
        df = df.assign(**{
            self.column_names.rating_update_match_id: lambda x: x[self.column_names.rating_update_match_id].astype(
                'str')})
        df = df.assign(**{
            self.column_names.parent_team_id: lambda x: x[self.column_names.parent_team_id].astype(
                'str')})
        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        ori_cols = df.columns.tolist()
        ori_index_values = df.index.tolist()

        all_df = pd.concat([self._df, df], axis=0).reset_index()
        all_df = all_df.assign(
            __id=all_df[[self.column_names.rating_update_match_id, self.column_names.parent_team_id,
                         self.column_names.player_id]].agg('__'.join, axis=1))
        all_df = all_df.drop_duplicates(subset=['__id'], keep='last')

        for feature_name in self.features:

            if self.column_names.participation_weight:
                all_df = all_df.assign(
                    **{feature_name: all_df[feature_name] * all_df[self.column_names.participation_weight]})

            output_column_name = f'{self.prefix}{self.window}_{feature_name}'
            if output_column_name in all_df.columns:
                raise ValueError(
                    f'Column {output_column_name} already exists. Choose different prefix or ensure no duplication was performed')

            agg_dict = {feature_name: 'mean', self.column_names.start_date: 'first'}
            grp = all_df.groupby(self.granularity + [self.column_names.rating_update_match_id]).agg(
                agg_dict).reset_index()
            grp.sort_values(by=[self.column_names.start_date, self.column_names.rating_update_match_id], inplace=True)

            grp = grp.assign(**{output_column_name: grp.groupby(self.granularity)[feature_name].apply(
                lambda x: x.shift().rolling(self.window, min_periods=self.min_periods).mean())})

            all_df = all_df.merge(
                grp[self.granularity + [self.column_names.rating_update_match_id, output_column_name]],
                on=self.granularity + [self.column_names.rating_update_match_id], how='left')
            all_df = all_df.sort_values(by=[self.column_names.start_date, self.column_names.match_id,
                                            self.column_names.team_id, self.column_names.player_id])
        df = df.assign(
            __id=df[[self.column_names.rating_update_match_id, self.column_names.parent_team_id,
                     self.column_names.player_id]].agg('__'.join, axis=1))
        transformed_df = all_df[all_df['__id'].isin(df['__id'].unique().tolist())][ori_cols + self._features_out]
        transformed_df.index = ori_index_values
        return transformed_df[list(set(ori_cols + self._features_out))]

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class RollingMeanDaysTransformer(BasePostTransformer):

    def __init__(self,
                 features: list[str],
                 days: Union[int, list[int]],
                 column_names: ColumnNames,
                 granularity: Union[list[str], str] = None,
                 add_count: bool = False,
                 add_opponent: bool = False,
                 prefix: str = 'rolling_mean_days_'):
        super().__init__(features=features)
        self.column_names = column_names
        self.features = features
        self.days = days
        if isinstance(self.days, int):
            self.days = [self.days]
        self.granularity = granularity or [self.column_names.player_id]
        self.add_count = add_count
        self.add_opponent = add_opponent
        self.prefix = prefix
        self._features_out = []
        self._df = None

        for day in self.days:
            for feature_name in self.features:
                self._features_out.append(f'{self.prefix}{day}_{feature_name}')
                if self.add_opponent:
                    self._features_out.append(f'{self.prefix}{day}_{feature_name}_opponent')

            if self.add_count:
                self._features_out.append(f'{self.prefix}{day}_count')

                if self.add_opponent:
                    self._features_out.append(f'{self.prefix}{day}_count_opponent')

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.assign(**{self.column_names.player_id: lambda x: x[self.column_names.player_id].astype('str')})
        df = df.assign(**{
            self.column_names.rating_update_match_id: lambda x: x[self.column_names.rating_update_match_id].astype(
                'str')})
        df = df.assign(**{
            self.column_names.parent_team_id: lambda x: x[self.column_names.parent_team_id].astype(
                'str')})

        validate_sorting(df=df, column_names=self.column_names)
        if self._df is None:
            self._df = df
        else:
            self._df = pd.concat([self._df, df], axis=0)

        self._df = self._df.assign(
            __id=self._df[[self.column_names.rating_update_match_id, self.column_names.parent_team_id,
                           self.column_names.player_id]].agg('__'.join, axis=1))
        self._df = self._df.drop_duplicates(subset=['__id'], keep='last')

        transformed_df = self.transform(df)
        return transformed_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        df = df.assign(**{self.column_names.player_id: lambda x: x[self.column_names.player_id].astype('str')})
        df = df.assign(**{
            self.column_names.rating_update_match_id: lambda x: x[self.column_names.rating_update_match_id].astype(
                'str')})
        df = df.assign(**{
            self.column_names.parent_team_id: lambda x: x[self.column_names.parent_team_id].astype(
                'str')})
        df = df.assign(
            __id=df[[self.column_names.rating_update_match_id, self.column_names.parent_team_id,
                     self.column_names.player_id]].agg('__'.join, axis=1))

        all_df = pd.concat([self._df, df], axis=0).reset_index()
        all_df = all_df.drop_duplicates(subset=['__id'], keep='last')

        ori_cols = df.columns.tolist()
        ori_index_values = df.index.tolist()

        for feature_name in self.features:

            if self.column_names.participation_weight:
                all_df = all_df.assign(
                    **{feature_name: all_df[feature_name] * all_df[self.column_names.participation_weight]})

        all_df[self.column_names.start_date] = pd.to_datetime(all_df[self.column_names.start_date]).dt.date

        for day in self.days:
            prefix_day = f'{self.prefix}{day}'
            all_df = self._add_rolling_feature(all_df=all_df, day=day, granularity=self.granularity,
                                               prefix_day=prefix_day)

        if self.add_opponent:
            feats = []
            for day in self.days:
                for feature_name in self.features:
                    feats.append(f'{self.prefix}{day}_{feature_name}')
                if self.add_count:
                    feats.append(f'{self.prefix}{day}_count')

            all_df = add_opponent_features(df=all_df, column_names=self.column_names, features=feats)

        all_df = all_df.sort_values(by=[self.column_names.start_date, self.column_names.match_id,
                                        self.column_names.team_id, self.column_names.player_id])

        df = df.assign(
            __id=df[[self.column_names.rating_update_match_id, self.column_names.parent_team_id,
                     self.column_names.player_id]].agg('__'.join, axis=1))
        transformed_df = all_df[all_df['__id'].isin(df['__id'].unique().tolist())][ori_cols + self._features_out]
        transformed_df.index = ori_index_values
        return transformed_df[list(set(ori_cols + self._features_out))].drop(columns=['__id'])

    def _add_rolling_feature(self, all_df: pd.DataFrame, day: int, granularity: list[str], prefix_day: str):
        df1 = (all_df
               .groupby([self.column_names.start_date, *granularity])[self.features]
               .agg(['sum', 'size'])
               .unstack()
               .asfreq('d', fill_value=np.nan)
               .rolling(window=day, min_periods=1)
               .sum()
               .shift()
               .stack()
               )
        feats = []
        for feature_name in self.features:
            feats.append(f'{prefix_day}_{feature_name}_sum')
            feats.append(f'{prefix_day}_{feature_name}_count')

        df1.columns = feats
        for feature_name in self.features:
            df1[f'{prefix_day}_{feature_name}'] = df1[f'{prefix_day}_{feature_name}_sum'] / df1[
                f'{prefix_day}_{feature_name}_count']

        if self.add_count:
            df1[f'{prefix_day}_count'] = df1[f'{prefix_day}_{self.features[0]}_count']
            df1 = df1.drop(columns=[f'{prefix_day}_{feature_name}_count' for feature_name in self.features])

        all_df[self.column_names.start_date] = pd.to_datetime(all_df[self.column_names.start_date])
        all_df = all_df.join(df1[[c for c in df1.columns if c in self.features_out]],
                             on=[self.column_names.start_date, *granularity])
        if self.add_count:
            all_df[f'{prefix_day}_count'] = all_df[f'{prefix_day}_count'].fillna(0)

        return all_df

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class Operation(Enum):
    SUBTRACT = "subtract"


@dataclass
class ModifyOperation:
    feature1: str
    operation: Operation
    feature2: str
    new_column_name: Optional[str] = None

    def __post_init__(self):
        if self.operation == Operation.SUBTRACT and not self.new_column_name:
            self.new_column_name = f"{self.feature1}_minus_{self.feature2}"


class ModifierTransformer(BasePostTransformer):

    def __init__(self,
                 modify_operations: list[ModifyOperation],
                 features: list[str] = None,
                 are_estimator_features: bool = True,
                 ):
        super().__init__(features=features, are_estimator_features=are_estimator_features)
        self.modify_operations = modify_operations
        self._features_out = [operation.new_column_name for operation in self.modify_operations]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for operation in self.modify_operations:
            if operation.operation == Operation.SUBTRACT:
                df[operation.new_column_name] = df[operation.feature1] - df[operation.feature2]

        return df
