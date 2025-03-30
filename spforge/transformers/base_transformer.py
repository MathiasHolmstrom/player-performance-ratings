import logging
from abc import abstractmethod, ABC
from functools import wraps
from typing import Optional

import polars as pl
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


def future_lag_transformations_wrapper(method):
    @wraps(method)
    def wrapper(self, df: FrameT,*args, **kwargs):
        df = df.drop([f for f in self.features_out if f in df.columns])
        input_cols = df.columns
        if '__row_index' not in df.columns:
            df = df.with_row_index('__row_index')

        if isinstance(nw.to_native(df), pd.DataFrame):
            ori_native = "pd"
            df = nw.from_native(pl.DataFrame(nw.to_native(df)))
        else:
            ori_native = "pl"

        df = df.with_columns(nw.lit(1).alias("is_future"))
        if self.unique_constraint:
            assert len(df.select(self.unique_constraint)) == len(
                df), f"Specified unique constraint {self.unique_constraint} is not unique on the input dataframe"


        result = method(self=self, df=df, *args, **kwargs).sort("__row_index")

        if "is_future" in result.columns:
            result = result.drop("is_future")
        input_cols = [c for c in input_cols if c not in ('is_future')]
        if ori_native == "pd":
            return result.select(list(set(input_cols + self.features_out))).to_pandas()
        return result.select(list(set(input_cols + self.features_out)))

    return wrapper


def historical_lag_transformations_wrapper(method):
    @wraps(method)
    def wrapper(self, df: FrameT, column_names: Optional[ColumnNames] = None, *args, **kwargs):
        input_cols = df.columns
        if "__row_index" not in df.columns:
            df = df.with_row_index(name="__row_index")
        self.column_names = column_names or self.column_names
        if self.column_names and not self.unique_constraint:
            self.unique_constraint = [self.column_names.match_id, self.column_names.player_id,
                                      self.column_names.team_id] if self.column_names.player_id else [
                self.column_names.match_id, self.column_names.team_id]

        if self.unique_constraint:
            assert len(df.select(self.unique_constraint)) == len(
                df), f"Specified unique constraint {self.unique_constraint} is not unique on the input dataframe"
        if (
                self.scale_by_participation_weight
                and not self.column_names
                or self.scale_by_participation_weight
                and not self.column_names.participation_weight
        ):
            raise ValueError(
                "scale_by_participation_weight requires column_names to be provided"
            )
        df = df.with_columns(nw.lit(0).alias("is_future"))
        native = nw.to_native(df)
        if isinstance(native, pd.DataFrame):
            df = nw.from_native(pl.DataFrame(native))
            ori_native = "pd"
        else:
            ori_native = "pl"
        result = method(self=self, df=df, *args, **kwargs).sort("__row_index")

        if "is_future" in result.columns:
            result = result.drop("is_future")
        input_cols = [c for c in input_cols if c not in ('is_future')]
        if ori_native == "pd":
            return result.select(list(set(input_cols + self.features_out))).to_pandas()
        return result.select(list(set(input_cols + self.features_out)))

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
            scale_by_participation_weight: bool = False
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
        self._entity_features_out = []
        self.add_opponent = add_opponent
        self.prefix = prefix
        self._df = None
        self._entity_features_out = []
        cn = self.column_names
        self.scale_by_participation_weight = scale_by_participation_weight
        self.unique_constraint = unique_constraint if unique_constraint else [cn.player_id, cn.match_id,
                                                                              cn.team_id] if cn and cn.player_id else [
            cn.match_id, cn.team_id] if cn else None

        for feature_name in self.features:
            for lag in iterations:
                self._features_out.append(f"{prefix}_{feature_name}{lag}")
                self._entity_features_out.append(f"{prefix}_{feature_name}{lag}")
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

        return concat_df.unique(
            subset=self.unique_constraint,
            maintain_order=True,
        )

    def _store_df(
            self, df: nw.DataFrame, additional_cols: Optional[list[str]] = None
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

        if additional_cols:
            cols += additional_cols

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

        self._df = (
            self._df.sort(sort_cols)
            .unique(
                subset=self.unique_constraint,
                maintain_order=True,
            )
            .to_native()
        )

    def _merge_into_input_df(
            self, df: FrameT, concat_df: FrameT, match_id_join_on: Optional[str] = None
    ) -> IntoFrameT:

        ori_cols = [c for c in df.columns if c not in self.features_out]

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
        join_cols = [*self.granularity, self.column_names.update_match_id]
        for column in join_cols:
            if concat_df[column].dtype != df[column].dtype:
                concat_df = concat_df.with_columns(concat_df[column].cast(df[column].dtype))

        transformed_df = df.select(ori_cols).join(concat_df.select([*join_cols, *self._entity_features_out]),
                                                  on=join_cols, how='left').unique(self.unique_constraint).sort(
            sort_cols)
        return transformed_df.select(list(set(df.columns + self._entity_features_out)))

    def _add_opponent_features(self, df: FrameT) -> FrameT:
        team_features = df.group_by(
            [self.column_names.team_id, self.column_names.update_match_id]
        ).agg(nw.col(self._entity_features_out).mean())

        df_opponent_feature = team_features.with_columns(
            [
                nw.col(self.column_names.team_id).alias("__opponent_team_id"),
                *[nw.col(f).alias(f"{f}_opponent") for f in self._entity_features_out],
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

        new_feats = [f"{f}_opponent" for f in self._entity_features_out]
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

    def _forward_fill_future_features(
            self,
            df: FrameT,
            known_future_features: Optional[list[str]] = None,
    ) -> FrameT:
        cn = self.column_names

        df = df.with_columns(
            [nw.col(f).fill_null(-999.21345).alias(f) for f in self._entity_features_out]
        )
        df = df.sort([cn.start_date])
        first_grp = (
            df.with_columns(
                nw.col(self.column_names.match_id)
                .cum_count()
                .over(self.granularity)
                .alias("__entity_row_index")
            )
            .filter(nw.col("__entity_row_index") == 1)
            .drop("__entity_row_index")
        ).select([*self.granularity, *self._entity_features_out])

        df = df.select(
            [c for c in df.columns if c not in self._entity_features_out]
        ).join(first_grp, on=self.granularity, how="left")

        df = df.sort([cn.start_date, cn.match_id, cn.team_id])
        for f in self._entity_features_out:
            df = df.with_columns(
                nw.when(nw.col(f) == -999.21345)
                .then(nw.lit(None))
                .otherwise(nw.col(f))
                .alias(f)
            )
            if df[f].is_null().sum() == len(df):
                df = df.with_columns(
                    nw.col(f)
                    .fill_null(strategy="forward")
                    .over(self.granularity)
                    .alias(f)
                    for f in self._entity_features_out
                )


        team_features = df.group_by(
            [self.column_names.team_id, self.column_names.match_id]
        ).agg([nw.col(f).mean().alias(f) for f in self._entity_features_out])

        rename_mapping = {
            self.column_names.team_id: "__opponent_team_id",
            **{f: f"{f}_opponent" for f in self._entity_features_out},
        }

        df_opponent_feature = team_features.select(
            [
                nw.col(name).alias(rename_mapping.get(name, name))
                for name in team_features.columns
            ]
        )
        opponent_feat_names = [f"{f}_opponent" for f in self._entity_features_out]
        new_df = df.join(
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
                .alias("__opp_row_index")
            )
            .filter(nw.col("__opp_row_index") == 1)
            .drop("__opp_row_index")
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
        cols_to_drop = [c for c in opponent_feat_names if c in df.columns]
        return  df.drop(cols_to_drop).join(
            new_df.select(
                [
                    *join_cols,
                    *opponent_feat_names,
                ]
            ),
            on=join_cols,
            how="left",
        )

    def reset(self) -> "BaseLagGenerator":
        self._df = None
        return self

    @property
    def historical_df(self) -> FrameT:
        return self._df
