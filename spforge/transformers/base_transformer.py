import logging
from abc import abstractmethod, ABC
from functools import wraps
from typing import Optional

import numpy as np
import pandas as pd
from narwhals.typing import FrameT, IntoFrameT
import narwhals as nw

from spforge import ColumnNames


def future_validator(method):
    @wraps(method)
    def wrapper(self, df: FrameT, *args, **kwargs):
        assert (
            self.column_names is not None
        ), "column names must be passed when calling transform_future"
        return method(self, df, *args, **kwargs)

    return wrapper


def row_count_validator(method):
    @wraps(method)
    def wrapper(self, df: FrameT, *args, **kwargs):
        input_row_count = len(df)
        result = method(self, df, *args, **kwargs)
        output_row_count = len(result)
        assert (
            input_row_count == output_row_count
        ), f"Row count mismatch: input had {input_row_count} rows, output had {output_row_count} rows"
        return result

    return wrapper


def required_lag_column_names(method):
    @wraps(method)
    def wrapper(
        self, df: FrameT, column_names: Optional[ColumnNames] = None, *args, **kwargs
    ):
        self.column_names = column_names or self.column_names

        if not self.column_names:
            if "__row_index" not in df.columns:
                df = df.with_row_index(name="__row_index")

            if hasattr(self, "days_between_lags") and self.days_between_lags:
                raise ValueError(
                    "column names must be passed if days_between_lags is set"
                )

            assert (
                self.match_id_update_column is not None
            ), "if column names is not passed. match_id_update_column must be passed"

            if self.add_opponent:
                logging.warning(
                    "add_opponent is set but column names must be passed for opponent feats to be created"
                )
        else:
            self.match_id_update_column = self.column_names.update_match_id
        return method(self, df, self.column_names, *args, **kwargs)

    return wrapper


class BaseTransformer(ABC):

    def __init__(
        self,
        features: list[str],
        features_out: list[str],
        are_estimator_features: bool = True,
    ):
        self._features_out = features_out
        self.features = features
        self._are_estimator_features = are_estimator_features
        self.column_names = None
        self._predictor_features_out = (
            self._features_out if self._are_estimator_features else []
        )

    @abstractmethod
    def fit_transform(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        pass

    @abstractmethod
    def transform(self, df: FrameT) -> IntoFrameT:
        pass

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    @property
    def predictor_features_out(self) -> list[str]:
        return self._predictor_features_out

    def reset(self) -> "BaseTransformer":
        return self


class BaseLagGenerator:

    def __init__(
        self,
        granularity: list[str],
        features: list[str],
        add_opponent: bool,
        iterations: list[int],
        prefix: str,
        match_id_update_column: Optional[str],
        column_names: Optional[ColumnNames] = None,
        are_estimator_features: bool = True,
        unique_constraint: Optional[list[str]] = None,
    ):
        self.match_id_update_column = match_id_update_column
        self.column_names = column_names
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
        self.unique_constraint = unique_constraint

        for feature_name in self.features:
            for lag in iterations:
                self._features_out.append(f"{prefix}_{feature_name}{lag}")
                self._entity_features.append(f"{prefix}_{feature_name}{lag}")
                if self.add_opponent:
                    self._features_out.append(f"{prefix}_{feature_name}{lag}_opponent")

    @abstractmethod
    def transform_historical(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        pass

    @abstractmethod
    def transform_future(self, df: FrameT) -> IntoFrameT:
        pass

    @property
    def predictor_features_out(self) -> list[str]:
        if self._are_estimator_features:
            return self.features_out
        return []

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    def _concat_with_stored(self, df: FrameT) -> FrameT:

        sort_cols = [self.column_names.start_date, self.column_names.update_match_id]

        df = df.with_columns(
            [
                nw.col(feature).cast(nw.Float64).alias(feature)
                for feature in self.features
                if feature in df.columns
            ]
        )

        df = df.select([c for c in df.columns if c not in self.features_out])
        cols = [c for c in self._df.columns if c in df.columns]

        stored_df = nw.from_native(self._df)
        for col in stored_df.columns:
            if col in df.columns and stored_df.schema[col] != df.schema[col]:
                df = df.with_columns(df[col].cast(stored_df.schema[col]))

        concat_df = nw.concat(
            [stored_df, df.select(cols)],
            how="diagonal",
        ).sort(sort_cols)
        if self.unique_constraint:
            unique_cols = self.unique_constraint
        else:
            unique_cols = (
                [
                    self.column_names.match_id,
                    self.column_names.team_id,
                    self.column_names.player_id,
                ]
                if self.column_names.player_id
                else [
                    self.column_names.match_id,
                    self.column_names.team_id,
                ]
            )
        return concat_df.unique(
            subset=unique_cols,
            maintain_order=True,
        )

    def _store_df(
        self, df: nw.DataFrame, additional_cols_to_use: Optional[list[str]] = None
    ):
        df = df.with_columns(
            [
                nw.col(feature).cast(nw.Float64).alias(feature)
                for feature in self.features
                if feature in df.columns
            ]
        )

        cols = list(
            {
                *self.features,
                *self.granularity,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
                "is_future",
                self.column_names.update_match_id,
                self.column_names.start_date,
            }
        )
        if self.column_names.player_id not in df.columns:
            cols.remove(self.column_names.player_id)
        if self.column_names.participation_weight in df.columns:
            cols += [self.column_names.participation_weight]
        if self.column_names.projected_participation_weight in df.columns:
            cols += [self.column_names.projected_participation_weight]

        if additional_cols_to_use:
            cols += additional_cols_to_use

        if self._df is None:
            self._df = df.select(cols)
        else:
            self._df = nw.concat([nw.from_native(self._df), df.select(cols)])

        sort_cols = (
            [
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
            if self.column_names.player_id in self._df.columns
            else [
                self.column_names.match_id,
                self.column_names.team_id,
            ]
        )

        unique_constraint = self.unique_constraint or sort_cols

        self._df = (
            self._df.sort(sort_cols)
            .unique(
                subset=unique_constraint,
                maintain_order=True,
            )
            .to_native()
        )

    def _create_transformed_df(
        self, df: FrameT, concat_df: FrameT, match_id_join_on: Optional[str] = None
    ) -> IntoFrameT:

        if self.add_opponent:
            concat_df = self._add_opponent_features(df=concat_df)
        on_cols = (
            [match_id_join_on, self.column_names.team_id, self.column_names.player_id]
            if self.column_names.player_id
            else [
                match_id_join_on,
                self.column_names.team_id,
            ]
        )
        ori_cols = [c for c in df.columns if c not in concat_df.columns] + on_cols
        unique_cols = (
            [
                self.column_names.player_id,
                self.column_names.team_id,
                self.column_names.match_id,
            ]
            if self.column_names.player_id
            else [self.column_names.team_id, self.column_names.match_id]
        )

        sort_cols = (
            [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
            if self.column_names.player_id
            else [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
            ]
        )

        transformed_df = (
            concat_df.join(df.select(ori_cols), on=on_cols, how="inner")
            .unique(unique_cols)
            .sort(sort_cols)
        )
        return transformed_df.select(list(set(df.columns + self.features_out)))

    def _add_opponent_features(self, df: FrameT) -> FrameT:
        team_features = df.group_by(
            [self.column_names.team_id, self.column_names.update_match_id]
        ).agg(**{col: nw.mean(col) for col in self._entity_features})

        df_opponent_feature = team_features.with_columns(
            [
                nw.col(self.column_names.team_id).alias("__opponent_team_id"),
                *[nw.col(f).alias(f"{f}_opponent") for f in self._entity_features],
            ]
        )

        new_df = df.join(
            df_opponent_feature,
            on=self.column_names.update_match_id,
            suffix="_team_sum",
        )

        new_df = new_df.filter(
            nw.col(self.column_names.team_id) != nw.col("__opponent_team_id")
        ).drop(["__opponent_team_id"])

        new_feats = [f"{f}_opponent" for f in self._entity_features]
        on_cols = (
            [
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
            if self.column_names.player_id in df.columns
            else [
                self.column_names.match_id,
                self.column_names.team_id,
            ]
        )
        return df.join(
            new_df.select(
                [
                    *on_cols,
                    *new_feats,
                ]
            ),
            on=on_cols,
            how="left",
        )

    def _generate_future_feats(
        self,
        transformed_df: FrameT,
        ori_df: FrameT,
        known_future_features: Optional[list[str]] = None,
    ) -> FrameT:
        known_future_features = known_future_features or []
        ori_cols = ori_df.columns
        cn = self.column_names

        transformed_df = transformed_df.with_columns(
            [nw.col(f).fill_null(-999.21345).alias(f) for f in self._entity_features]
        )
        transformed_df = transformed_df.sort([cn.start_date, cn.match_id, cn.team_id])
        first_grp = (
            transformed_df.with_columns(
                nw.col(self.column_names.match_id)
                .cum_count()
                .over(self.granularity)
                .alias("_row_index")
            )
            .filter(nw.col("_row_index") == 1)
            .drop("_row_index")
        ).select([*self.granularity, *self._entity_features])

        transformed_df = transformed_df.select(
            [c for c in transformed_df.columns if c not in self._entity_features]
        ).join(first_grp, on=self.granularity, how="left")

        transformed_df = transformed_df.sort([cn.start_date, cn.match_id, cn.team_id])
        for f in self._entity_features:
            transformed_df = transformed_df.with_columns(
                nw.when(nw.col(f) == -999.21345)
                .then(nw.lit(None))
                .otherwise(nw.col(f))
                .alias(f)
            )
            if transformed_df[f].is_null().sum() == len(transformed_df):
                transformed_df = transformed_df.with_columns(
                    nw.col(f)
                    .fill_null(strategy="forward")
                    .over(self.granularity)
                    .alias(f)
                    for f in self._entity_features
                )

        team_features = transformed_df.group_by(
            [self.column_names.team_id, self.column_names.match_id]
        ).agg([nw.col(f).mean().alias(f) for f in self._entity_features])

        rename_mapping = {
            self.column_names.team_id: "__opponent_team_id",
            **{f: f"{f}_opponent" for f in self._entity_features},
        }

        df_opponent_feature = team_features.select(
            [
                nw.col(name).alias(rename_mapping.get(name, name))
                for name in team_features.columns
            ]
        )
        opponent_feat_names = [f"{f}_opponent" for f in self._entity_features]
        new_df = transformed_df.join(
            df_opponent_feature, on=[self.column_names.match_id], suffix="_team_sum"
        )
        new_df = new_df.filter(
            nw.col(self.column_names.team_id) != nw.col("__opponent_team_id")
        )
        new_df = new_df.with_columns(
            [
                nw.col(column).fill_null(-999.21345).alias(column)
                for column in opponent_feat_names
            ]
        )
        new_df = new_df.sort([cn.start_date, cn.match_id, "__opponent_team_id"])
        first_grp = (
            new_df.with_columns(
                nw.col(self.column_names.match_id)
                .cum_count()
                .over("__opponent_team_id")
                .alias("_row_index")
            )
            .filter(nw.col("_row_index") == 1)
            .drop("_row_index")
        ).select(["__opponent_team_id", *opponent_feat_names])

        new_df = new_df.select(
            [c for c in new_df.columns if c not in opponent_feat_names]
        ).join(first_grp, on="__opponent_team_id", how="left")
        new_df = new_df.with_columns(
            [
                nw.when(nw.col(f) == -999.21345)
                .then(nw.lit(None))
                .otherwise(nw.col(f))
                .alias(f)
                for f in opponent_feat_names
            ]
        )

        new_df = new_df.sort([cn.start_date, cn.match_id, "__opponent_team_id"])
        for f in opponent_feat_names:
            new_df = new_df.with_columns(
                nw.col(f)
                .fill_null(strategy="forward")
                .over("__opponent_team_id")
                .alias(f)
            )
        join_cols = self.unique_constraint or [
            self.column_names.match_id,
            self.column_names.team_id,
            self.column_names.player_id,
        ]
        transformed_df = transformed_df.join(
            new_df.select(
                [
                    *join_cols,
                    *opponent_feat_names,
                ]
            ),
            on=join_cols,
            how="left",
        )

        ori_feats_to_use = [f for f in ori_cols if f not in self.features_out]
        transformed_feats_to_use = [
            *join_cols,
            *[f for f in self.features_out if f not in known_future_features],
        ]

        transformed_df = ori_df.select(ori_feats_to_use).join(
            transformed_df.select(transformed_feats_to_use),
            on=join_cols,
        )

        return transformed_df.select(
            list(
                set(
                    ori_cols
                    + [f for f in self.features_out if f not in known_future_features]
                )
            )
        )

    def reset(self) -> "BaseLagGenerator":
        self._df = None
        return self

    @property
    def historical_df(self) -> FrameT:
        return self._df
