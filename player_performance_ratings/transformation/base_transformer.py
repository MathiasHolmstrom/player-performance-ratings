from abc import abstractmethod, ABC
from typing import Optional

import pandas as pd
from player_performance_ratings import ColumnNames


class BaseTransformer(ABC):

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


class BasePostTransformer(ABC):

    def __init__(self, features: list[str], are_estimator_features: bool = True):
        self.features = features
        self._are_estimator_features = are_estimator_features
        self._features_out = []
        self.column_names = None

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
        if self._are_estimator_features:
            return self.features_out
        return []


class BaseLagTransformer(BasePostTransformer):

    def __init__(self,
                 features: list[str],
                 add_opponent: bool,
                 iterations: list[int],
                 prefix: str,
                 are_estimator_features: bool = True,
                 ):
        super().__init__(features, are_estimator_features)
        self._entity_features = []
        self.add_opponent = add_opponent
        self.prefix = prefix
        self._df = None
        self._entity_features = []

        for feature_name in self.features:
            for lag in iterations:
                self._features_out.append(f'{prefix}{lag}_{feature_name}')
                self._entity_features.append(f'{prefix}{lag}_{feature_name}')
                if self.add_opponent:
                    self._features_out.append(f'{prefix}{lag}_{feature_name}_opponent')

    def _concat_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._string_convert(df=df)
        df = df.assign(
            __id=df[[self.column_names.rating_update_match_id, self.column_names.parent_team_id,
                     self.column_names.player_id]].agg('__'.join, axis=1))

        concat_df = pd.concat([self._df, df], axis=0).reset_index()
        if concat_df[self.column_names.start_date].dtype in('str', 'object'):
            concat_df[self.column_names.start_date] = pd.to_datetime(concat_df[self.column_names.start_date])
        return concat_df.drop_duplicates(subset=['__id'], keep='last')

    def _fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._string_convert(df)
        if self._df is None:
            self._df = df
        else:
            self._df = pd.concat([self._df, df], axis=0)

        self._df = self._df.assign(
            __id=self._df[[self.column_names.rating_update_match_id, self.column_names.parent_team_id,
                           self.column_names.player_id]].agg('__'.join, axis=1))
        self._df = self._df.drop_duplicates(subset=['__id'], keep='last')

        transformed_df = self.transform(df=df)
        return transformed_df

    def _string_convert(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(**{self.column_names.player_id: lambda x: x[self.column_names.player_id].astype('str')})
        df = df.assign(**{
            self.column_names.rating_update_match_id: lambda x: x[self.column_names.rating_update_match_id].astype(
                'str')})
        df = df.assign(**{
            self.column_names.parent_team_id: lambda x: x[self.column_names.parent_team_id].astype(
                'str')})

        return df

    def _create_transformed_df(self, df: pd.DataFrame, concat_df: pd.DataFrame) -> pd.DataFrame:

        if self.add_opponent:
            concat_df = self._add_opponent_features(df=concat_df)

        ori_cols = df.columns.tolist()
        ori_index_values = df.index.tolist()

        df = self._string_convert(df)
        df = df.assign(
            __id=df[[self.column_names.rating_update_match_id, self.column_names.parent_team_id,
                     self.column_names.player_id]].agg('__'.join, axis=1))

        transformed_df = concat_df[concat_df['__id'].isin(df['__id'].unique().tolist())][ori_cols + self._features_out]
        transformed_df.index = ori_index_values
        transformed_df = transformed_df.sort_values(by=[self.column_names.start_date, self.column_names.match_id,
                                                        self.column_names.team_id, self.column_names.player_id])
        return transformed_df[list(set(ori_cols + self._features_out))]

    def _add_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        team_features = df.groupby([self.column_names.team_id, self.column_names.rating_update_match_id])[
            self._entity_features].mean().reset_index()
        df_opponent_feature = team_features.rename(
            columns={**{self.column_names.team_id: 'opponent_team_id'},
                     **{f: f"{f}_opponent" for f in self._entity_features}}
        )
        new_df = df.merge(df_opponent_feature, on=[self.column_names.match_id], suffixes=('', '_team_sum'))
        new_df = new_df[new_df[self.column_names.team_id] != new_df['opponent_team_id']].drop(
            columns=['opponent_team_id'])

        new_feats = [f"{f}_opponent" for f in self._entity_features]
        return df.merge(
            new_df[[self.column_names.match_id, self.column_names.team_id, self.column_names.player_id, *new_feats]],
            on=[self.column_names.match_id, self.column_names.team_id, self.column_names.player_id], how='left')
