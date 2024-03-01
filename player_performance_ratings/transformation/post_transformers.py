from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd
from player_performance_ratings.predictor import GameTeamPredictor

from player_performance_ratings import ColumnNames
from player_performance_ratings.predictor._base import BasePredictor
from player_performance_ratings.transformation.base_transformer import BasePostTransformer, BaseLagTransformer
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


class NetOverPredictedPostTransformer(BasePostTransformer):

    def __init__(self,
                 predictor: GameTeamPredictor,
                 features: list[str] = None,
                 prefix: str = "net_over_predicted_",
                 are_estimator_features: bool = False,
                 ):
        super().__init__(features=features, are_estimator_features=are_estimator_features)
        self.prefix = prefix
        self._predictor = predictor
        self._features_out = []
        self.column_names = None
        new_feature_name = self.prefix + self._predictor.pred_column
        self._features_out.append(new_feature_name)
        if self.prefix is "":
            raise ValueError("Prefix must not be empty")

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[ColumnNames] = None) -> pd.DataFrame:
        self.column_names = column_names
        self._predictor.train(df, estimator_features=self.features)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._predictor.add_prediction(df)
        new_feature_name = self.prefix + self._predictor.pred_column
        if self._predictor.target not in df.columns:
            df = df.assign(**{new_feature_name: np.nan})
        else:
            df = df.assign(**{new_feature_name: df[self._predictor.target] - df[self._predictor.pred_column]})
        df = df.drop(columns=[self._predictor.pred_column])

        return df

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class PredictorTransformer(BasePostTransformer):

    def __init__(self, predictor: BasePredictor, features: list[str] = None):
        self.predictor = predictor
        super().__init__(features=features)

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[None] = None) -> pd.DataFrame:
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
                 team_total_prediction_column: Optional[str] = None,
                 prefix: str = "_ratio_team"
                 ):
        super().__init__(features=features)
        self.predictor = predictor
        self.team_total_prediction_column = team_total_prediction_column
        self.prefix = prefix
        self.predictor._pred_column = f"__prediction__{self.predictor.target}"
        self._features_out = [self.predictor.target + prefix]
        if self.team_total_prediction_column:
            self._features_out.append(self.predictor.target + prefix + "_team_total_multiplied")

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[ColumnNames] = None) -> pd.DataFrame:
        self.column_names = column_names
        self.predictor.train(df=df, estimator_features=self.features)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.predictor.add_prediction(df=df)
        df[self.predictor.pred_column + "_sum"] = df.groupby([self.column_names.match_id, self.column_names.team_id])[
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

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[ColumnNames] = None) -> pd.DataFrame:
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

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[ColumnNames] = None) -> pd.DataFrame:
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


class LagTransformer(BaseLagTransformer):

    def __init__(self,
                 features: list[str],
                 lag_length: int,
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

        super().__init__(features=features, add_opponent=add_opponent, prefix=prefix,
                         iterations=[i for i in range(1, lag_length + 1)])
        self.days_between_lags = days_between_lags or []
        for days_lag in self.days_between_lags:
            self._features_out.append(f'{prefix}{days_lag}_days_ago')

        self.granularity = granularity
        self.lag_length = lag_length
        self.future_lag = future_lag
        self._df = None

    def fit_transform(self, df: pd.DataFrame, column_names: ColumnNames) -> pd.DataFrame:
        self.column_names = column_names
        self.granularity = self.granularity or [self.column_names.player_id]

        for feature_out in self._features_out:
            if feature_out in df.columns:
                raise ValueError(
                    f'Column {feature_out} already exists. Choose different prefix or ensure no duplication was performed')

        return self._fit_transform(df=df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        if len(df.drop_duplicates(
                subset=[self.column_names.player_id, self.column_names.parent_team_id,
                        self.column_names.rating_update_match_id])) != len(df):
            raise ValueError(
                f"Duplicated rows in df. Df must be a unique combination of {self.column_names.player_id} and {self.column_names.rating_update_match_id}")

        concat_df = self._concat_df(df=df)

        if self.column_names.participation_weight:
            for feature in self.features:
                concat_df = concat_df.assign(
                    **{feature: concat_df[feature] * concat_df[self.column_names.participation_weight]})

        grouped = \
            concat_df.groupby(
                self.granularity + [self.column_names.rating_update_match_id, self.column_names.start_date])[
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
                if output_column_name in concat_df.columns:
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

        concat_df = concat_df.merge(
            grouped[self.granularity + [self.column_names.rating_update_match_id, *feats_out]],
            on=self.granularity + [self.column_names.rating_update_match_id], how='left')

        return self._create_transformed_df(df=df, concat_df=concat_df)

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class RollingMeanTransformer(BaseLagTransformer):

    def __init__(self,
                 features: list[str],
                 window: int,
                 granularity: Union[list[str], str] = None,
                 add_opponent: bool = False,
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
        super().__init__(features=features, add_opponent=add_opponent, iterations=[window],
                         prefix=prefix)
        self.granularity = granularity
        if isinstance(self.granularity, str):
            self.granularity = [self.granularity]
        self.window = window
        self.min_periods = min_periods

    def fit_transform(self, df: pd.DataFrame, column_names: ColumnNames) -> pd.DataFrame:
        self.column_names = column_names
        self.granularity = self.granularity or [column_names.player_id]
        return self._fit_transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        concat_df = self._concat_df(df)

        for feature_name in self.features:

            if self.column_names.participation_weight:
                concat_df = concat_df.assign(
                    **{feature_name: concat_df[feature_name] * concat_df[self.column_names.participation_weight]})

            output_column_name = f'{self.prefix}{self.window}_{feature_name}'
            if output_column_name in concat_df.columns:
                raise ValueError(
                    f'Column {output_column_name} already exists. Choose different prefix or ensure no duplication was performed')

            agg_dict = {feature_name: 'mean', self.column_names.start_date: 'first'}
            grp = concat_df.groupby(self.granularity + [self.column_names.rating_update_match_id]).agg(
                agg_dict).reset_index()
            grp.sort_values(by=[self.column_names.start_date, self.column_names.rating_update_match_id], inplace=True)

            grp = grp.assign(**{output_column_name: grp.groupby(self.granularity)[feature_name].apply(
                lambda x: x.shift().rolling(self.window, min_periods=self.min_periods).mean())})

            concat_df = concat_df.merge(
                grp[self.granularity + [self.column_names.rating_update_match_id, output_column_name]],
                on=self.granularity + [self.column_names.rating_update_match_id], how='left')
            concat_df = concat_df.sort_values(by=[self.column_names.start_date, self.column_names.match_id,
                                                  self.column_names.team_id, self.column_names.player_id])

        return self._create_transformed_df(df=df, concat_df=concat_df)

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class RollingMeanDaysTransformer(BaseLagTransformer):

    def __init__(self,
                 features: list[str],
                 days: Union[int, list[int]],
                 granularity: Union[list[str], str] = None,
                 add_count: bool = False,
                 add_opponent: bool = False,
                 prefix: str = 'rolling_mean_days_'):
        self.days = days
        if isinstance(self.days, int):
            self.days = [self.days]
        super().__init__(features=features, iterations=[i for i in self.days], prefix=prefix,
                         add_opponent=add_opponent)

        self.granularity = granularity
        self.add_count = add_count

        for day in self.days:
            if self.add_count:
                feature = f'{self.prefix}{day}_count'
                self._features_out.append(feature)
                self._entity_features.append(feature)

                if self.add_opponent:
                    self._features_out.append(f'{self.prefix}{day}_count_opponent')

    def fit_transform(self, df: pd.DataFrame, column_names: ColumnNames) -> pd.DataFrame:
        self.granularity = self.granularity or [self.column_names.player_id]
        self.column_names = column_names
        validate_sorting(df=df, column_names=self.column_names)
        return self._fit_transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        concat_df = self._concat_df(df)

        for feature_name in self.features:

            if self.column_names.participation_weight:
                concat_df = concat_df.assign(
                    **{feature_name: concat_df[feature_name] * concat_df[self.column_names.participation_weight]})

        concat_df = concat_df.assign(
            **{self.column_names.start_date: lambda x: x[self.column_names.start_date].dt.date})

        for day in self.days:
            prefix_day = f'{self.prefix}{day}'
            concat_df = self._add_rolling_feature(concat_df=concat_df, day=day, granularity=self.granularity,
                                                  prefix_day=prefix_day)

        return self._create_transformed_df(df=df, concat_df=concat_df)

    def _add_rolling_feature(self, concat_df: pd.DataFrame, day: int, granularity: list[str], prefix_day: str):

        if len(granularity) > 1:
            granularity_concat = '__'.join(granularity)
            temporary_str_df = concat_df[granularity].astype(str)
            concat_df[granularity_concat] = temporary_str_df.agg('__'.join, axis=1)
        else:
            granularity_concat = granularity[0]

        df1 = (concat_df
               .groupby([self.column_names.start_date, granularity_concat])[self.features]
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

        concat_df[self.column_names.start_date] = pd.to_datetime(concat_df[self.column_names.start_date])
        concat_df = concat_df.join(df1[[c for c in df1.columns if c in self.features_out]],
                                   on=[self.column_names.start_date, granularity_concat])
        if self.add_count:
            concat_df[f'{prefix_day}_count'] = concat_df[f'{prefix_day}_count'].fillna(0)

        return concat_df

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

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[ColumnNames]) -> pd.DataFrame:
        self.column_names = column_names
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for operation in self.modify_operations:
            if operation.operation == Operation.SUBTRACT:
                if operation.feature1 not in df.columns or operation.feature2 not in df.columns:
                    df = df.assign(**{operation.new_column_name: np.nan})

                else:
                    df = df.assign(**{operation.new_column_name: df[operation.feature1] - df[operation.feature2]})

        return df


class BinaryOutcomeRollingMeanTransformer(BaseLagTransformer):

    def __init__(self,
                 features: list[str],
                 window: int,
                 binary_column: str,
                 granularity: list[str] = None,
                 prob_column: Optional[str] = None,
                 min_periods: int = 1,
                 add_opponent: bool = False,
                 prefix: str = 'rolling_mean_binary_'):
        super().__init__(features=features, add_opponent=add_opponent, prefix=prefix,
                         iterations=[])
        self.granularity = granularity
        self.window = window
        self.min_periods = min_periods
        self.binary_column = binary_column
        self.prob_column = prob_column
        for feature_name in self.features:
            feature1 = f'{self.prefix}{self.window}_{feature_name}_1'
            feature2 = f'{self.prefix}{self.window}_{feature_name}_0'
            self._features_out.append(f'{self.prefix}{self.window}_{feature_name}_1')
            self._features_out.append(f'{self.prefix}{self.window}_{feature_name}_0')
            self._entity_features.append(feature1)
            self._entity_features.append(feature2)

            if self.add_opponent:
                self._features_out.append(f'{self.prefix}{self.window}_{feature_name}_1_opponent')
                self._features_out.append(f'{self.prefix}{self.window}_{feature_name}_0_opponent')

        if self.prob_column:
            for feature_name in self.features:
                prob_feature = f'{self.prefix}{self.window}_{self.prob_column}_{feature_name}'
                self._features_out.append(prob_feature)
                self._entity_features.append(prob_feature)

                if self.add_opponent:
                    self._features_out.append(f'{prob_feature}_opponent')

    def fit_transform(self, df: pd.DataFrame, column_names: ColumnNames) -> pd.DataFrame:
        self.column_names = column_names
        self.granularity = self.granularity or [column_names.player_id]
        validate_sorting(df=df, column_names=self.column_names)
        return self._fit_transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        concat_df = self._concat_df(df)

        for feature in self.features:
            mask_result_1 = concat_df[self.binary_column] == 1
            mask_result_0 = concat_df[self.binary_column] == 0

            concat_df['value_result_1'] = concat_df[feature].where(mask_result_1)
            concat_df['value_result_0'] = concat_df[feature].where(mask_result_0)

            concat_df[f'{self.prefix}{self.window}_{feature}_1'] = concat_df.groupby(self.granularity)[
                'value_result_1'].transform(
                lambda x: x.shift().rolling(window=self.window, min_periods=self.min_periods).mean())
            concat_df[f'{self.prefix}{self.window}_{feature}_0'] = concat_df.groupby(self.granularity)[
                'value_result_0'].transform(
                lambda x: x.shift().rolling(window=self.window, min_periods=self.min_periods).mean())

            concat_df.drop(['value_result_1', 'value_result_0'], axis=1, inplace=True)

        if self.prob_column:
            for idx, feature_name in enumerate(self.features):
                concat_df[f'{self.prefix}{self.window}_{self.prob_column}_{feature_name}'] = concat_df[
                                                                                                 f'{self.prefix}{self.window}_{feature_name}_1'] * \
                                                                                             concat_df[
                                                                                                 self.prob_column] + \
                                                                                             concat_df[
                                                                                                 f'{self.prefix}{self.window}_{feature_name}_0'] * (
                                                                                                     1 -
                                                                                                     concat_df[
                                                                                                         self.prob_column])

        return self._create_transformed_df(df=df, concat_df=concat_df)
