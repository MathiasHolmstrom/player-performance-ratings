from abc import abstractmethod, ABC
from typing import Optional

import numpy as np
import pandas as pd
from player_performance_ratings import ColumnNames


class BasePerformancesTransformer(ABC):

    def __init__(self, features: list[str]):
        self.features = features

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def features_out(self) -> list[str]:
        pass


class BaseTransformer(ABC):

    def __init__(self, features: list[str], features_out: list[str], are_estimator_features: bool = True):
        self._features_out = features_out
        self.features = features
        self._are_estimator_features = are_estimator_features
        self.column_names = None
        self._estimator_features_out = self._features_out if self._are_estimator_features else []

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame, column_names: ColumnNames) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    @property
    def estimator_features_out(self) -> list[str]:
        return self._estimator_features_out

    def reset(self) -> "BaseTransformer":
        return self


class BaseLagGenerator():

    def __init__(self,
                 granularity: list[str],
                 features: list[str],
                 add_opponent: bool,
                 iterations: list[int],
                 prefix: str,
                 are_estimator_features: bool = True,
                 ):

        self.features = features
        self.iterations = iterations
        self._features_out = []
        self._are_estimator_features = are_estimator_features
        self.granularity = granularity
        if isinstance(self.granularity, str):
            self.granularity = [self.granularity]
        self._entity_features = []
        self.add_opponent = add_opponent
        self.prefix = prefix
        self._df = None
        self._entity_features = []
        self.column_names: Optional[ColumnNames] = None

        for feature_name in self.features:
            for lag in iterations:
                self._features_out.append(f'{prefix}{lag}_{feature_name}')
                self._entity_features.append(f'{prefix}{lag}_{feature_name}')
                if self.add_opponent:
                    self._features_out.append(f'{prefix}{lag}_{feature_name}_opponent')

    @abstractmethod
    def generate_historical(self, df: pd.DataFrame, column_names: ColumnNames) -> pd.DataFrame:
        pass

    @abstractmethod
    def generate_future(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    def estimator_features_out(self) -> list[str]:
        if self._are_estimator_features:
            return self.features_out
        return []

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    def _concat_df(self, df: pd.DataFrame, additional_cols_to_use: Optional[list[str]] = None) -> pd.DataFrame:
        df = self._string_convert(df=df)
        for feature in self.features:
            if feature in df.columns:
                df = df.assign(**{feature: lambda x: x[feature].astype('float')})

        df = df[[c for c in df.columns if c not in self.features_out]]

        cols = [f for f in list(
            set([*self.features, *self.granularity, self.column_names.match_id, self.column_names.team_id, 'is_future',
                 self.column_names.player_id,
                 self.column_names.parent_team_id, self.column_names.update_match_id,
                 self.column_names.start_date])) if f in df.columns]

        if self.column_names.participation_weight in df.columns:
            cols += [self.column_names.participation_weight]
        if self.column_names.projected_participation_weight in df.columns:
            cols += [self.column_names.projected_participation_weight]

        if additional_cols_to_use:
            cols += [f for f in additional_cols_to_use if f in df.columns]

        concat_df = pd.concat([self._df, df[cols]], axis=0).reset_index()

        if 'index' in concat_df.columns:
            concat_df = concat_df.drop(columns=['index'])
        if concat_df[self.column_names.start_date].dtype in ('str', 'object'):
            concat_df[self.column_names.start_date] = pd.to_datetime(concat_df[self.column_names.start_date])
        return concat_df.drop_duplicates(
            subset=[self.column_names.match_id, self.column_names.team_id, self.column_names.player_id], keep='last')

    def _store_df(self, df: pd.DataFrame, additional_cols_to_use: Optional[list[str]] = None):
        df = self._string_convert(df)

        cols = list(
            set([*self.features, *self.granularity, self.column_names.match_id, self.column_names.team_id,
                 self.column_names.player_id,
                 self.column_names.parent_team_id, self.column_names.update_match_id,
                 self.column_names.start_date, 'is_future']))
        if self.column_names.participation_weight in df.columns:
            cols += [self.column_names.participation_weight]
        if self.column_names.projected_participation_weight in df.columns:
            cols += [self.column_names.projected_participation_weight]

        if additional_cols_to_use:
            cols += additional_cols_to_use

        if self._df is None:
            self._df = df[cols]
        else:
            self._df = pd.concat([self._df, df[cols]], axis=0)

        self._df = self._df.drop_duplicates(
            subset=[self.column_names.match_id, self.column_names.team_id, self.column_names.player_id], keep='last')

    def _string_convert(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in [self.column_names.match_id, self.column_names.parent_team_id, self.column_names.player_id,
                       self.column_names.update_match_id]:
            df = df.assign(**{column: lambda x: x[column].astype('str')})
        return df

    def _create_transformed_df(self, df: pd.DataFrame, concat_df: pd.DataFrame) -> pd.DataFrame:

        cn = self.column_names

        if self.add_opponent:
            concat_df = self._add_opponent_features(df=concat_df)

        ori_cols = [c for c in df.columns if c not in concat_df.columns] + [cn.match_id, cn.player_id, cn.team_id]
        ori_index_values = df.index.tolist()

        df = self._string_convert(df)

        transformed_df = concat_df.merge(df[ori_cols],
                                         on=[cn.match_id, cn.player_id, cn.team_id], how='inner')

        transformed_df.index = ori_index_values
        transformed_df = transformed_df.sort_values(by=[cn.start_date, cn.match_id,
                                                        cn.team_id, cn.player_id])

        return transformed_df[list(set(df.columns.tolist() + self.features_out))]

    def _add_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        team_features = df.groupby([self.column_names.team_id, self.column_names.match_id])[
            self._entity_features].mean().reset_index()
        df_opponent_feature = team_features.rename(
            columns={**{self.column_names.team_id: '__opponent_team_id'},
                     **{f: f"{f}_opponent" for f in self._entity_features}}
        )
        new_df = df.merge(df_opponent_feature, on=[self.column_names.match_id], suffixes=('', '_team_sum'))
        new_df = new_df[new_df[self.column_names.team_id] != new_df['__opponent_team_id']].drop(
            columns=['__opponent_team_id'])

        new_feats = [f"{f}_opponent" for f in self._entity_features]
        return df.merge(
            new_df[[self.column_names.match_id, self.column_names.team_id, self.column_names.player_id,
                    *new_feats]],
            on=[self.column_names.match_id, self.column_names.team_id, self.column_names.player_id], how='left')

    def _generate_future_feats(self, transformed_df: pd.DataFrame, ori_df: pd.DataFrame,
                               known_future_features: Optional[list[str]] = None) -> pd.DataFrame:
        known_future_features = known_future_features or []
        ori_cols = ori_df.columns.tolist()
        ori_index_values = ori_df.index.tolist()
        cn = self.column_names

        transformed_df[self._entity_features] = transformed_df[self._entity_features].fillna(-999.21345)
        first_grp = transformed_df.groupby(self.granularity)[self._entity_features].first().reset_index()
        transformed_df = transformed_df[
            [c for c in transformed_df.columns if c not in self._entity_features]].merge(first_grp,
                                                                                         on=self.granularity,
                                                                                         how='left')
        for f in self._entity_features:
            transformed_df[f].replace(-999.21345, np.nan, inplace=True)
        if not transformed_df[self._entity_features].isnull().all().all():
            transformed_df[self._entity_features] = transformed_df.groupby(self.granularity)[self._entity_features].transform(lambda x: x.fillna(method='ffill'))

        team_features = transformed_df.groupby([self.column_names.team_id, self.column_names.match_id])[
            self._entity_features].mean().reset_index()
        df_opponent_feature = team_features.rename(
            columns={**{self.column_names.team_id: '__opponent_team_id'},
                     **{f: f"{f}_opponent" for f in self._entity_features}}
        )
        opponent_feat_names = [f"{f}_opponent" for f in self._entity_features]
        new_df = transformed_df.merge(df_opponent_feature, on=[self.column_names.match_id],
                                      suffixes=('', '_team_sum'))
        new_df = new_df[new_df[self.column_names.team_id] != new_df['__opponent_team_id']]
        new_df[opponent_feat_names] = new_df[
            opponent_feat_names].fillna(-999.21345)
        first_grp = new_df.groupby('__opponent_team_id')[
            opponent_feat_names].first().reset_index()
        new_df = new_df[
            [c for c in new_df.columns if c not in opponent_feat_names]].merge(first_grp,
                                                                               on='__opponent_team_id',
                                                                               how='left')
        for f in opponent_feat_names:
            new_df[f].replace(-999.21345, np.nan, inplace=True)

        new_df = new_df.sort_values(by=[cn.start_date, cn.match_id, '__opponent_team_id'])
        new_df.groupby('__opponent_team_id')[opponent_feat_names].fillna(method='ffill',
                                                                         inplace=True)

        transformed_df = transformed_df.merge(
            new_df[[self.column_names.match_id, self.column_names.team_id, self.column_names.player_id,
                    *opponent_feat_names]],
            on=[self.column_names.match_id, self.column_names.team_id, self.column_names.player_id], how='left')

        for col in [cn.match_id, cn.player_id, cn.team_id]:
            ori_df = ori_df.assign(**{col: lambda x: x[col].astype('str')})

        ori_feats_to_use = [f for f in ori_cols if f not in self.features_out]
        transformed_feats_to_use = [cn.match_id, cn.team_id, cn.player_id,
                                    *[f for f in
                                      self.features_out if
                                      f not in known_future_features]]

        transformed_df = ori_df[ori_feats_to_use].merge(transformed_df[transformed_feats_to_use],
                                                        on=[cn.match_id, cn.team_id,
                                                            cn.player_id])

        transformed_df = transformed_df.sort_values(by=[cn.start_date, cn.match_id,
                                                        cn.team_id, cn.player_id])
        transformed_df.index = ori_index_values
        return transformed_df[list(set(ori_cols + [f for f in self.features_out if f not in known_future_features]))]

    def reset(self) -> "BaseLagGenerator":
        self._df = None
        return self
