from abc import abstractmethod, ABC
from typing import Optional

import numpy as np
import pandas as pd
from narwhals.typing import FrameT, IntoFrameT
import narwhals as nw

from player_performance_ratings import ColumnNames


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
        self._estimator_features_out = (
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
    def estimator_features_out(self) -> list[str]:
        return self._estimator_features_out

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
                self._features_out.append(f"{prefix}{lag}_{feature_name}")
                self._entity_features.append(f"{prefix}{lag}_{feature_name}")
                if self.add_opponent:
                    self._features_out.append(f"{prefix}{lag}_{feature_name}_opponent")

    @abstractmethod
    def generate_historical(self, df: FrameT, column_names: ColumnNames) -> IntoFrameT:
        pass

    @abstractmethod
    def generate_future(self, df: FrameT) -> IntoFrameT:
        pass

    @property
    def estimator_features_out(self) -> list[str]:
        if self._are_estimator_features:
            return self.features_out
        return []

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    def _concat_df(self, df: FrameT) -> FrameT:
        df = self._string_convert(df=df)
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

        sort_cols = [self.column_names.start_date, self.column_names.match_id, self.column_names.team_id,
                     self.column_names.player_id] if self.column_names.player_id in df.columns else [
            self.column_names.start_date, self.column_names.match_id, self.column_names.team_id]
        concat_df = nw.concat(
            [stored_df, df.select(cols)],
            how="diagonal",
            # how="diagonal_relaxed"
        ).sort(sort_cols).unique(subset=sort_cols, maintain_order=True)
        if concat_df[self.column_names.start_date].dtype in ("str", "object"):
            concat_df[self.column_names.start_date] = pd.to_datetime(
                concat_df[self.column_names.start_date]
            )

        unique_cols = [self.column_names.match_id,
                       self.column_names.team_id,
                       self.column_names.player_id] if self.column_names.player_id in concat_df.columns else [
            self.column_names.match_id,
            self.column_names.team_id,
        ]
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

        df = self._string_convert(df)

        cols = list(
            {
                *self.features,
                *self.granularity,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
                "is_future",
                self.column_names.parent_team_id,
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

        sort_cols = [
            self.column_names.match_id,
            self.column_names.team_id,
            self.column_names.player_id,
        ] if self.column_names.player_id in self._df.columns else [
            self.column_names.match_id,
            self.column_names.team_id,
        ]

        self._df = (
            self._df.sort(sort_cols
                          #   descending=True
                          )
            .unique(
                subset=sort_cols,
                maintain_order=True,
            )
            .to_native()
        )

    def _string_convert(self, df: FrameT) -> FrameT:

        if df.schema[self.column_names.start_date] == nw.Datetime("ns"):
            df = df.with_columns(
                df[self.column_names.start_date].cast(nw.Datetime("us"))
            )
        return df

    def _create_transformed_df(self, df: FrameT, concat_df: FrameT) -> IntoFrameT:

        cn = self.column_names

        if self.add_opponent:
            concat_df = self._add_opponent_features(df=concat_df)
        on_cols = [cn.match_id, cn.team_id, cn.player_id] if cn.player_id in df.columns else [
            cn.match_id,
            cn.team_id,
        ]
        ori_cols = [c for c in df.columns if c not in concat_df.columns] + on_cols

        df = self._string_convert(df)

        transformed_df = concat_df.join(
            df.select(ori_cols), on=on_cols, how="inner"
        )
        return transformed_df.select(list(set(df.columns + self.features_out)))

    def _add_opponent_features(self, df: FrameT) -> FrameT:
        team_features = df.group_by(
            [self.column_names.team_id, self.column_names.match_id]
        ).agg(**{col: nw.mean(col) for col in self._entity_features})

        df_opponent_feature = team_features.with_columns(
            [
                nw.col(self.column_names.team_id).alias("__opponent_team_id"),
                *[nw.col(f).alias(f"{f}_opponent") for f in self._entity_features],
            ]
        )

        new_df = df.join(
            df_opponent_feature, on=self.column_names.match_id, suffix="_team_sum"
        )

        new_df = new_df.filter(
            nw.col(self.column_names.team_id) != nw.col("__opponent_team_id")
        ).drop(["__opponent_team_id"])

        new_feats = [f"{f}_opponent" for f in self._entity_features]
        on_cols = [self.column_names.match_id,
                   self.column_names.team_id,
                   self.column_names.player_id] if self.column_names.player_id in df.columns else [
            self.column_names.match_id,
            self.column_names.team_id,
        ]
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

        transformed_df = transformed_df.join(
            new_df.select(
                [
                    self.column_names.match_id,
                    self.column_names.team_id,
                    self.column_names.player_id,
                    *opponent_feat_names,
                ]
            ),
            on=[
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ],
            how="left",
        )

        #        ori_df = ori_df.with_columns(
        #         [
        #             nw.col(col).cast(nw.String).alias(col)
        #               for col in [cn.match_id, cn.player_id, cn.team_id]
        #         ]
        #     )

        ori_feats_to_use = [f for f in ori_cols if f not in self.features_out]
        transformed_feats_to_use = [
            cn.match_id,
            cn.team_id,
            cn.player_id,
            *[f for f in self.features_out if f not in known_future_features],
        ]

        transformed_df = ori_df.select(ori_feats_to_use).join(
            transformed_df.select(transformed_feats_to_use),
            on=[cn.match_id, cn.team_id, cn.player_id],
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
